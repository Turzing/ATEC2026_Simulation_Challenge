"""
数据采集 — 在 Isaac Sim 机器上跑仿真，保存真实 RGB + YOLO 标签

用法:
    conda activate isaaclab
    cd ATEC2026_Simulation_Challenge
    python collect_real_data.py --num_images 500 --output datasets/real

    有 GUI 时镜头会跟随机器人；若有 demo/policy.pt 会自动用 RL 行走，否则用内置简易步态。

输出:
    datasets/real/images/train/*.png
    datasets/real/labels/train/*.txt
    datasets/real/dataset.yaml
"""

import argparse
import os
import sys
import numpy as np
import cv2

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect real rendering data for YOLO training")
parser.add_argument("--num_images", type=int, default=500)
parser.add_argument("--output", type=str, default="datasets/real")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--save_every", type=int, default=4,
                    help="Save every N sim steps (default 4 ≈ 0.08s)")
parser.add_argument("--min_visible", type=int, default=0,
                    help="0=always save RGB; 1=only save frames with >=1 projected label")
parser.add_argument("--policy", type=str, default="",
                    help="RL policy.pt path (default: demo/policy.pt if exists)")
parser.add_argument("--forward_speed", type=float, default=0.5,
                    help="RL forward cmd [vx, vy, wz] when using policy.pt")
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
compute_bbox_3d = _proj.compute_bbox_3d
bbox_to_yolo_line = _proj.bbox_to_yolo_line
obj_index_to_class_id = _proj.obj_index_to_class_id
yaw_from_quat_wxyz = _proj.yaw_from_quat_wxyz
class_id_to_size = _proj.class_id_to_size


def obj_to_class(obj_index: int) -> int:
    return obj_index_to_class_id(obj_index)


class SimpleWalkController:
    """
    数据采集专用移动控制器（不依赖师姐的 motion_controller）:
      1. 若存在 policy.pt → 用与 demo/solution_rl.py 相同的 RL 行走
      2. 否则 → 内置对角步态 + 周期性转向（幅度足够大，能明显移动）
    """

    LEG_DIM = 12
    ARM_DIM = 8
    LEG_IDX = list(range(12))
    ARM_IDX = list(range(12, 20))

    def __init__(self, device: str, policy_path: str = "", forward_speed: float = 0.5):
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
                print(f"[WalkController] RL mode — loaded {p}", flush=True)
                break
        if self.mode == "scripted":
            print("[WalkController] Scripted trot mode (place demo/policy.pt for RL walk)",
                  flush=True)

    def _rl_action(self, obs) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        idx = 0
        idx += 3  # base_lin_vel
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3  # velocity_commands from env
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        leg_pos = joint_pos[:, self.LEG_IDX]
        leg_vel = joint_vel[:, self.LEG_IDX]
        leg_act = actions_all[:, self.LEG_IDX]
        leg_act_train = leg_act * self.env_to_train.to(dtype=proprio.dtype)

        cmd = self.velocity_cmd.to(dtype=proprio.dtype)
        policy_obs = torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            cmd,
            leg_pos,
            leg_vel * 0.05,
            leg_act_train,
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

    def _scripted_action(self, step_count: int) -> torch.Tensor:
        """对角小跑步态 + 每 ~8s 换向，保证扫到更多物体"""
        a = torch.zeros(1, 20, dtype=torch.float32, device=self.device)
        t = step_count * 0.02
        phase = t * 3.0

        # 对角腿对: (FR,RL) vs (FL,RR)
        s1 = np.sin(phase)
        s2 = np.sin(phase + np.pi)

        def leg(i_hip, i_thigh, i_calf, s):
            a[0, i_hip] = 0.25 * s
            a[0, i_thigh] = 0.55 + 0.45 * s
            a[0, i_calf] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1)   # FR
        leg(3, 4, 5, s2)   # FL
        leg(6, 7, 8, s2)   # RR
        leg(9, 10, 11, s1)  # RL

        # 周期性转向: 4s 直走 + 4s 左转
        cycle = int(step_count // 200) % 2
        if cycle == 1:
            turn = 0.55
            a[0, 0] += turn
            a[0, 6] += turn
            a[0, 3] -= turn
            a[0, 9] -= turn

        return a

    def get_action(self, obs, step_count: int) -> torch.Tensor:
        if self.mode == "rl":
            return self._rl_action(obs)
        return self._scripted_action(step_count)


def project_labels(env, robot_pos, robot_yaw):
    labels = []
    visible = 0
    found = 0
    for obj_idx in range(1, 19):
        try:
            obj_pos = env.unwrapped.scene[f"object_{obj_idx}"].data.root_pos_w[0].cpu().numpy()
            found += 1
        except (AttributeError, KeyError):
            continue

        cid = obj_to_class(obj_idx)
        size_3d = class_id_to_size(cid)
        bbox = compute_bbox_3d(
            obj_pos, size_3d, robot_pos, robot_yaw,
            HEAD_CAM_MATRIX, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX_INV,
        )
        if bbox is None:
            continue
        visible += 1
        labels.append(bbox_to_yolo_line(cid, bbox))
    return labels, visible, found


def main():
    num_images = args_cli.num_images
    output_dir = args_cli.output
    save_every = args_cli.save_every
    min_visible = args_cli.min_visible
    sim_device = getattr(args_cli, "device", None) or (
        "cuda:0" if torch.cuda.is_available() else "cpu")
    use_gui = not getattr(args_cli, "headless", False)

    img_dir = os.path.join(output_dir, "images", "train")
    lbl_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("  ATEC Task B — Real Data Collector (walk + save)", flush=True)
    print(f"  Target: {num_images} frames, save every {save_every} steps", flush=True)
    print(f"  min_visible={min_visible}  (0=save all RGB, 1+=need labels)", flush=True)
    print(f"  Output: {os.path.abspath(output_dir)}", flush=True)
    print(f"  GUI follow cam: {'yes' if use_gui and camera_follow else 'no (headless)'}",
          flush=True)
    print("=" * 70, flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task, device=sim_device, num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    walk = SimpleWalkController(sim_device, args_cli.policy, args_cli.forward_speed)

    collected = 0
    labeled = 0
    step_count = 0
    episode = 0
    max_steps_per_ep = 3000

    while collected < num_images:
        episode += 1
        print(f"\n[Episode {episode}] reset...", flush=True)
        obs, _ = env.reset()

        for _ in range(20):
            obs, _, _, _, _ = env.step(
                torch.zeros(1, 20, dtype=torch.float32, device=sim_device))
            step_count += 1
            if use_gui and camera_follow:
                camera_follow(env)

        diag_once = True
        for _ in range(max_steps_per_ep):
            if collected >= num_images:
                break

            action = walk.get_action(obs, step_count)
            obs, _, terminated, truncated, _ = env.step(action)
            step_count += 1

            if use_gui and camera_follow:
                camera_follow(env)

            if step_count % save_every != 0:
                if terminated or truncated:
                    break
                continue

            try:
                rgb = obs["image"]["head_rgb"].squeeze(0)
            except (KeyError, TypeError):
                if step_count % 100 == 0:
                    print("  WARN: head_rgb missing — need --enable_cameras", flush=True)
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

            if diag_once:
                print(f"  [DIAG] objects={objects_found}/18  visible={visible_count}  "
                      f"pos={robot_pos[:2]}  yaw={robot_yaw:.2f}  walk={walk.mode}",
                      flush=True)
                diag_once = False

            if visible_count < min_visible:
                if step_count % 100 == 0:
                    print(f"  [Step {step_count}] visible={visible_count} "
                          f"(need>={min_visible}), robot moving...", flush=True)
                if terminated or truncated:
                    break
                continue

            fname = f"render_{collected:06d}"
            cv2.imwrite(
                os.path.join(img_dir, f"{fname}.png"),
                cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            )
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                if labels:
                    f.write("\n".join(labels) + "\n")

            collected += 1
            if visible_count > 0:
                labeled += 1
            tag = f"{visible_count} labels" if visible_count else "RGB only"
            print(f"  [{collected}/{num_images}] {fname}.png — {tag}", flush=True)

            if terminated or truncated:
                break

    env.close()

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# Auto-generated by collect_real_data.py
path: {os.path.abspath(output_dir)}
train: images/train
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
""")

    print("\n" + "=" * 70, flush=True)
    print(f"  Done: {collected} images ({labeled} with labels)", flush=True)
    print(f"  Images: {img_dir}", flush=True)
    print(f"  Labels: {lbl_dir}", flush=True)
    print(f"  Config: {yaml_path}", flush=True)
    if labeled == 0 and collected > 0:
        print("\n  TIP: got RGB but no labels — try:", flush=True)
        print("    1. Copy demo/policy.pt here for RL walk", flush=True)
        print("    2. Run without --headless to verify robot moves", flush=True)
        print("    3. Train on labeled subset after more collection", flush=True)
    elif collected > 0:
        print("\n  Next:", flush=True)
        print("    1. Copy datasets/real back to your PC", flush=True)
        print("    2. Check labels:  cd taskb_perception && python preview_labels.py --data ../datasets/real",
              flush=True)
        print("    3. Train YOLO:     python train_yolo.py", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
