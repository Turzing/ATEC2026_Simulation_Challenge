"""
数据采集 — 在 Isaac Sim 机器上跑仿真，保存真实 RGB + YOLO 标签

用法:
    conda activate isaaclab
    cd ATEC2026_Simulation_Challenge

    # 推荐: 手动控制，采 1000 张后自动 800 训练 / 200 验证
    python collect_real_data.py --mode manual

    # 自定义数量与划分
    python collect_real_data.py --num_images 1000 --train_count 800 --val_count 200

手动模式按键 (先点一下 Isaac Sim 窗口):
    W / S     前进 / 后退
    A / D     左转 / 右转
    SPACE     立刻拍一张 (仅当视野里有物体时)
    松开键    站立

输出 (采满后自动划分):
    datasets/real/images/train/   800 张
    datasets/real/images/val/     200 张
    datasets/real/labels/train/
    datasets/real/labels/val/
    datasets/real/dataset.yaml
"""

import argparse
import os
import random
import shutil
import sys
import numpy as np
import cv2
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect real rendering data for YOLO training")
parser.add_argument("--num_images", type=int, default=1000,
                    help="Total images to collect (default 1000)")
parser.add_argument("--train_count", type=int, default=800,
                    help="Train split size after collection (default 800)")
parser.add_argument("--val_count", type=int, default=200,
                    help="Val/test split size after collection (default 200)")
parser.add_argument("--split_seed", type=int, default=42,
                    help="Random seed for train/val shuffle split")
parser.add_argument("--output", type=str, default="datasets/real")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--save_every", type=int, default=8,
                    help="Auto-save every N steps when visible>=min_visible (manual/auto)")
parser.add_argument("--min_visible", type=int, default=1,
                    help="Only save when >=N objects projected in head camera (default 1)")
parser.add_argument("--mode", type=str, default="manual", choices=["manual", "auto", "rl"],
                    help="manual=keyboard, auto=scripted walk, rl=RL policy walk")
parser.add_argument("--policy", type=str, default="",
                    help="RL policy.pt path (default: demo/policy.pt if exists)")
parser.add_argument("--forward_speed", type=float, default=0.5,
                    help="RL forward cmd when W held in manual+rl mode")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.tasks
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
try:
    from rl_utils import camera_follow
except ImportError:
    camera_follow = None

import importlib.util

_perc_dir = os.path.join(os.path.dirname(__file__), "taskb_perception")
_spec = importlib.util.spec_from_file_location(
    "taskb_config", os.path.join(_perc_dir, "config.py"))
_taskb_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_taskb_config)

HEAD_CAM_MATRIX = _taskb_config.HEAD_CAM_MATRIX
IMG_W = _taskb_config.IMG_W
IMG_H = _taskb_config.IMG_H
HEAD_CAM_POS_ROBOT = _taskb_config.HEAD_CAM_POS_ROBOT
HEAD_CAM_ROT_MATRIX_INV = _taskb_config.HEAD_CAM_ROT_MATRIX_INV

_spec_proj = importlib.util.spec_from_file_location(
    "projection_utils", os.path.join(_perc_dir, "projection_utils.py"))
_proj = importlib.util.module_from_spec(_spec_proj)
_spec_proj.loader.exec_module(_proj)
compute_bbox_3d_cam_pose = _proj.compute_bbox_3d_cam_pose
bbox_to_yolo_line = _proj.bbox_to_yolo_line
obj_index_to_class_id = _proj.obj_index_to_class_id
yaw_from_quat_wxyz = _proj.yaw_from_quat_wxyz
class_id_to_size = _proj.class_id_to_size


def get_head_camera_pose(env):
    """从仿真 head_camera 读取世界位姿。

    必须用 quat_w_ros（+Z 朝前），不能用 quat_w_world（+X 朝前），
    否则针孔投影会全部落在画面外 → visible=0。
    """
    cam = env.unwrapped.scene["head_camera"]
    cam_pos_w = cam.data.pos_w[0].cpu().numpy()
    if hasattr(cam.data, "quat_w_ros"):
        cam_quat = cam.data.quat_w_ros[0].cpu().numpy()
    elif hasattr(cam.data, "quat_w_world"):
        cam_quat = cam.data.quat_w_world[0].cpu().numpy()
        print("  [WARN] quat_w_ros unavailable, projection may be wrong", flush=True)
    else:
        cam_quat = cam.data.quat_w[0].cpu().numpy()

    cam_matrix = HEAD_CAM_MATRIX
    if getattr(cam.data, "intrinsic_matrices", None) is not None:
        cam_matrix = cam.data.intrinsic_matrices[0].cpu().numpy()
    return cam_pos_w, cam_quat, cam_matrix


def obj_to_class(obj_index: int) -> int:
    return obj_index_to_class_id(obj_index)


def print_manual_help():
    print("\n  --- Manual controls (click Isaac Sim window first) ---", flush=True)
    print("  W / S     forward / backward", flush=True)
    print("  A / D     turn left / right", flush=True)
    print("  SPACE     snapshot now (only if objects visible)", flush=True)
    print("  release   stand still", flush=True)
    print("  Auto-save every N steps when visible >= min_visible", flush=True)
    print("  ----------------------------------------------------\n", flush=True)


class SimpleWalkController:
    """自动走路: RL policy 或内置步态"""

    LEG_IDX = list(range(12))

    def __init__(self, device: str, policy_path: str = "", forward_speed: float = 0.5,
                 force_rl: bool = False):
        self.device = device
        self.mode = "scripted"
        self.policy = None
        self.forward_speed = forward_speed

        self.train_to_env = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=device, dtype=torch.float32
        ).view(1, -1)
        self.env_to_train = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=device, dtype=torch.float32
        ).view(1, -1)
        self.velocity_cmd = torch.tensor(
            [forward_speed, 0.0, 0.0], device=device, dtype=torch.float32
        ).view(1, 3)

        if not force_rl:
            return

        candidates = [
            policy_path,
            os.path.join(os.path.dirname(__file__), "demo", "policy.pt"),
            os.path.join(os.path.dirname(__file__), "policy.pt"),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                self.policy = torch.jit.load(p, map_location=device)
                self.policy.eval()
                self.mode = "rl"
                print(f"[WalkController] RL — {p}", flush=True)
                break
        if self.mode == "scripted":
            print("[WalkController] Scripted trot (no policy.pt)", flush=True)

    def set_velocity_cmd(self, vx: float, vy: float, wz: float):
        self.velocity_cmd = torch.tensor(
            [vx, vy, wz], device=self.device, dtype=torch.float32
        ).view(1, 3)

    def _rl_action(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        idx = 12
        base_ang_vel = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]
        joint_pos = proprio[:, idx:idx + action_dim]
        joint_vel = proprio[:, idx + action_dim:idx + 2 * action_dim]
        actions_all = proprio[:, idx + 2 * action_dim:idx + 3 * action_dim]

        leg_pos = joint_pos[:, self.LEG_IDX]
        leg_vel = joint_vel[:, self.LEG_IDX]
        leg_act = actions_all[:, self.LEG_IDX]
        leg_act_train = leg_act * self.env_to_train.to(dtype=proprio.dtype)
        cmd = self.velocity_cmd.to(dtype=proprio.dtype)

        policy_obs = torch.cat([
            base_ang_vel * 0.25, projected_gravity, cmd,
            leg_pos, leg_vel * 0.05, leg_act_train,
        ], dim=-1)

        with torch.inference_mode():
            act_train = self.policy(policy_obs)
        if not isinstance(act_train, torch.Tensor):
            act_train = torch.as_tensor(act_train, device=self.device, dtype=torch.float32)
        if act_train.ndim == 1:
            act_train = act_train.unsqueeze(0)

        full = torch.zeros(1, action_dim, device=self.device, dtype=torch.float32)
        full[:, self.LEG_IDX] = act_train * self.train_to_env
        return full

    def _scripted_action(self, step_count: int, turn_bias: float = 0.0) -> torch.Tensor:
        a = torch.zeros(1, 20, dtype=torch.float32, device=self.device)
        t = step_count * 0.02
        phase = t * 3.0
        s1, s2 = np.sin(phase), np.sin(phase + np.pi)

        def leg(i_hip, i_thigh, i_calf, s):
            a[0, i_hip] = 0.25 * s
            a[0, i_thigh] = 0.55 + 0.45 * s
            a[0, i_calf] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1)
        leg(3, 4, 5, s2)
        leg(6, 7, 8, s2)
        leg(9, 10, 11, s1)

        if turn_bias != 0.0:
            a[0, 0] += turn_bias
            a[0, 6] += turn_bias
            a[0, 3] -= turn_bias
            a[0, 9] -= turn_bias
        return a

    def get_action(self, obs, step_count: int, turn_bias: float = 0.0) -> torch.Tensor:
        if self.mode == "rl":
            return self._rl_action(obs)
        return self._scripted_action(step_count, turn_bias)


class ManualKeyboardController:
    """键盘手动控制 + SPACE 触发拍照"""

    def __init__(self, device: str, policy_path: str = "", forward_speed: float = 0.5):
        self.device = device
        self._keyboard = None
        self._input = None
        self._ki = None
        self._space_was_down = False
        self.snap_now = False

        self._walk = SimpleWalkController(device, policy_path, forward_speed, force_rl=True)
        if self._walk.mode != "rl":
            print("[Manual] No policy.pt — using keyed scripted gait", flush=True)

        self._init_keyboard()

    def _init_keyboard(self):
        try:
            import carb.input
            import omni.appwindow
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._ki = carb.input.KeyboardInput
            print("[Manual] Keyboard ready", flush=True)
        except Exception as e:
            print(f"[Manual] WARN: keyboard unavailable ({e})", flush=True)
            print("  Use --mode auto if headless", flush=True)

    def _key_down(self, key) -> bool:
        if self._input is None or self._keyboard is None:
            return False
        return self._input.get_keyboard_value(self._keyboard, key) > 0

    def poll(self):
        """每帧调用: 更新 snap_now (SPACE 边沿触发)"""
        self.snap_now = False
        if self._ki is None:
            return

        space = self._key_down(self._ki.SPACE)
        if space and not self._space_was_down:
            self.snap_now = True
        self._space_was_down = space

    def get_motion(self):
        """返回 (moving, turn_bias, vx, wz)"""
        if self._ki is None:
            return False, 0.0, 0.0, 0.0

        w = self._key_down(self._ki.W)
        s = self._key_down(self._ki.S)
        a = self._key_down(self._ki.A)
        d = self._key_down(self._ki.D)

        vx = 0.5 if w else (-0.25 if s else 0.0)
        wz = 0.45 if a else (-0.45 if d else 0.0)
        if w and (a or d):
            vx = 0.35
        turn_bias = 0.5 if a else (-0.5 if d else 0.0)
        moving = w or s or a or d
        return moving, turn_bias, vx, wz

    def get_action(self, obs, step_count: int) -> torch.Tensor:
        self.poll()
        moving, turn_bias, vx, wz = self.get_motion()

        if not moving:
            return torch.zeros(1, 20, dtype=torch.float32, device=self.device)

        if self._walk.mode == "rl":
            self._walk.set_velocity_cmd(vx, 0.0, wz)
            return self._walk.get_action(obs, step_count)
        return self._walk.get_action(obs, step_count, turn_bias=turn_bias)


def _project_one_object(obj_pos, obj_quat, size_3d, use_cam_pose, cam_pos_w, cam_quat_w,
                        cam_matrix, robot_pos, robot_yaw):
    if use_cam_pose:
        return compute_bbox_3d_cam_pose(
            obj_pos, obj_quat, size_3d, cam_pos_w, cam_quat_w, cam_matrix,
        )
    return _proj.compute_bbox_3d(
        obj_pos, size_3d, robot_pos, robot_yaw,
        cam_matrix, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX_INV,
        obj_quat_wxyz=obj_quat,
    )


def project_labels(env, robot_pos, robot_yaw):
    """用仿真 head_camera 真实位姿 + 物体姿态投影 YOLO 框"""
    objects = []
    for obj_idx in range(1, 19):
        try:
            obj = env.unwrapped.scene[f"object_{obj_idx}"]
            objects.append((
                obj_idx,
                obj.data.root_pos_w[0].cpu().numpy(),
                obj.data.root_quat_w[0].cpu().numpy(),
            ))
        except (AttributeError, KeyError):
            continue

    found = len(objects)
    if found == 0:
        return [], 0, 0

    use_cam_pose = False
    cam_pos_w, cam_quat_w, cam_matrix = None, None, HEAD_CAM_MATRIX
    try:
        cam_pos_w, cam_quat_w, cam_matrix = get_head_camera_pose(env)
        use_cam_pose = True
    except (AttributeError, KeyError, TypeError):
        pass

    def _run(use_cam):
        labels, visible = [], 0
        for obj_idx, obj_pos, obj_quat in objects:
            cid = obj_to_class(obj_idx)
            size_3d = class_id_to_size(cid)
            bbox = _project_one_object(
                obj_pos, obj_quat, size_3d, use_cam,
                cam_pos_w, cam_quat_w, cam_matrix, robot_pos, robot_yaw,
            )
            if bbox is None:
                continue
            visible += 1
            labels.append(bbox_to_yolo_line(cid, bbox))
        return labels, visible

    labels, visible = _run(use_cam_pose)

    # 相机四元数用错约定时 visible=0；回退到机器人基座近似外参
    if use_cam_pose and visible == 0:
        print("  [WARN] cam_pose projection visible=0, fallback to robot-base", flush=True)
        labels, visible = _run(False)

    return labels, visible, found


def split_train_val(output_dir: str, train_count: int, val_count: int, seed: int = 42):
    """将 _staging 中的图片随机划分为 train / val"""
    root = Path(output_dir)
    staging_img = root / "images" / "_staging"
    staging_lbl = root / "labels" / "_staging"
    dirs = {
        "train_img": root / "images" / "train",
        "val_img": root / "images" / "val",
        "train_lbl": root / "labels" / "train",
        "val_lbl": root / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.is_file():
                old.unlink()

    pairs = []
    for img in sorted(staging_img.glob("*.png")):
        lbl = staging_lbl / f"{img.stem}.txt"
        if lbl.is_file():
            pairs.append((img, lbl))

    total = len(pairs)
    if total == 0:
        print("  WARN: no images in _staging to split", flush=True)
        return 0, 0

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = min(val_count, total)
    n_train = min(train_count, total - n_val)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:n_val + n_train]

    for img, lbl in train_pairs:
        shutil.move(str(img), str(dirs["train_img"] / img.name))
        shutil.move(str(lbl), str(dirs["train_lbl"] / lbl.name))
    for img, lbl in val_pairs:
        shutil.move(str(img), str(dirs["val_img"] / img.name))
        shutil.move(str(lbl), str(dirs["val_lbl"] / lbl.name))

    for staging in (staging_img, staging_lbl):
        try:
            staging.rmdir()
        except OSError:
            pass

    print(f"  Split: train={len(train_pairs)}  val={len(val_pairs)}  "
          f"(total collected={total}, seed={seed})", flush=True)
    return len(train_pairs), len(val_pairs)


def write_dataset_yaml(output_dir: str):
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# Auto-generated by collect_real_data.py
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
""")
    return yaml_path


def try_save_frame(img_dir, lbl_dir, rgb_np, labels, visible_count, collected, num_images):
    if visible_count < 1:
        return collected, False

    fname = f"render_{collected:06d}"
    cv2.imwrite(
        os.path.join(img_dir, f"{fname}.png"),
        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
    )
    with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")

    collected += 1
    print(f"  [{collected}/{num_images}] {fname}.png — {visible_count} labels", flush=True)
    return collected, True


def main():
    num_images = args_cli.num_images
    output_dir = args_cli.output
    save_every = args_cli.save_every
    min_visible = args_cli.min_visible
    mode = args_cli.mode
    sim_device = getattr(args_cli, "device", None) or (
        "cuda:0" if torch.cuda.is_available() else "cpu")
    use_gui = not getattr(args_cli, "headless", False)

    if mode == "manual" and not use_gui:
        print("ERROR: --mode manual requires GUI (do not use --headless)", flush=True)
        sys.exit(1)

    img_dir = os.path.join(output_dir, "images", "_staging")
    lbl_dir = os.path.join(output_dir, "labels", "_staging")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    train_n = args_cli.train_count
    val_n = args_cli.val_count
    if train_n + val_n != num_images:
        print(f"  NOTE: train({train_n})+val({val_n}) != num_images({num_images}), "
              f"will split best-effort after collection", flush=True)

    print("=" * 70, flush=True)
    print("  ATEC Task B — Real Data Collector", flush=True)
    print(f"  Mode: {mode}  |  target={num_images}  ->  train={train_n}  val={val_n}",
          flush=True)
    print(f"  min_visible={min_visible}  (only save frames with objects)", flush=True)
    print(f"  Output: {os.path.abspath(output_dir)}", flush=True)
    if mode == "manual":
        print_manual_help()

    env_cfg = parse_env_cfg(
        args_cli.task, device=sim_device, num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if mode == "manual":
        controller = ManualKeyboardController(
            sim_device, args_cli.policy, args_cli.forward_speed)
        walk = None
    else:
        controller = None
        walk = SimpleWalkController(
            sim_device, args_cli.policy, args_cli.forward_speed,
            force_rl=(mode == "rl"))

    collected = 0
    step_count = 0
    skipped_empty = 0
    episode = 0

    while collected < num_images:
        episode += 1
        print(f"\n[Episode {episode}] reset...  (collected {collected}/{num_images})",
              flush=True)
        obs, _ = env.reset()

        for _ in range(20):
            obs, _, _, _, _ = env.step(
                torch.zeros(1, 20, dtype=torch.float32, device=sim_device))
            step_count += 1
            if use_gui and camera_follow:
                camera_follow(env)

        while collected < num_images:
            if mode == "manual":
                action = controller.get_action(obs, step_count)
            else:
                action = walk.get_action(obs, step_count)

            obs, _, terminated, truncated, _ = env.step(action)
            step_count += 1

            if use_gui and camera_follow:
                camera_follow(env)

            # 检查是否该尝试保存
            periodic = (step_count % save_every == 0)
            space_snap = mode == "manual" and controller.snap_now
            if not (periodic or space_snap):
                if terminated or truncated:
                    break
                continue

            try:
                rgb = obs["image"]["head_rgb"].squeeze(0)
            except (KeyError, TypeError):
                continue

            rgb_np = (rgb.cpu() if rgb.device.type == "cuda" else rgb).numpy().astype(np.uint8)

            try:
                robot = env.unwrapped.scene["robot"]
                robot_pos = robot.data.root_pos_w[0].cpu().numpy()
                robot_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0].cpu().numpy())
            except Exception:
                robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
                robot_yaw = 0.0

            labels, visible_count, objects_found = project_labels(
                env, robot_pos, robot_yaw)

            if visible_count < min_visible:
                skipped_empty += 1
                if space_snap:
                    print(f"  [SPACE] no objects in view (visible=0), not saved",
                          flush=True)
                elif step_count % 100 == 0:
                    print(f"  [Step {step_count}] visible=0, move to find objects "
                          f"(skipped {skipped_empty} empty checks)", flush=True)
                if terminated or truncated:
                    break
                continue

            collected, saved = try_save_frame(
                img_dir, lbl_dir, rgb_np, labels, visible_count, collected, num_images)
            if not saved:
                if terminated or truncated:
                    break
                continue

            if terminated or truncated:
                break

    env.close()

    print("\n  Splitting into train / val ...", flush=True)
    n_train, n_val = split_train_val(
        output_dir, train_n, val_n, seed=args_cli.split_seed)
    yaml_path = write_dataset_yaml(output_dir)

    print("\n" + "=" * 70, flush=True)
    print(f"  Done: collected {collected}  ->  train {n_train}  val {n_val}", flush=True)
    print(f"  Skipped empty checks: {skipped_empty}", flush=True)
    print(f"  Config: {yaml_path}", flush=True)
    print(f"  Preview train: python taskb_perception/preview_labels.py "
          f"--data {output_dir} --split train", flush=True)
    print(f"  Preview val:   python taskb_perception/preview_labels.py "
          f"--data {output_dir} --split val", flush=True)
    print(f"  Train YOLO:    cd taskb_perception && "
          f"python train_yolo.py --data ../{output_dir}/dataset.yaml", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
