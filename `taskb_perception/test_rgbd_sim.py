"""
仿真内手动测试 RGB-D 感知 — 你自己键盘控机器人，实时看检测框

用法 (师姐电脑 / 任何有 Isaac Sim 的机器):
    conda activate isaaclab
    cd ATEC2026_Simulation_Challenge/taskb_perception
    python test_rgbd_sim.py

操作 (先点一下 Isaac Sim 窗口):
    W / S     前进 / 后退
    A / D     左转 / 右转
    P         保存一张标注图 (不要用空格! Isaac Sim 会抢空格切远景)
    Q         退出
    松开键    站立

实时预览:
    默认开 OpenCV 窗口显示检测框 (按 Q 关闭)

依赖文件 (只拷这 3 个即可):
    rgbd_perception_pipeline.py
    config.py
    test_rgbd_sim.py

可选: 把 demo/policy.pt 拷到 ../demo/policy.pt 可走 RL 步态; 没有则用简易步态。
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual RGB-D perception test in Isaac Sim")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--mode", type=str, default="manual", choices=["manual", "auto"])
parser.add_argument("--policy", type=str, default="", help="RL policy.pt (default: ../demo/policy.pt)")
parser.add_argument("--out", type=str, default="../datasets/rgbd_debug")
parser.add_argument("--live", action="store_true", default=True, help="OpenCV live overlay")
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview_every", type=int, default=4, help="Update live window every N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.tasks
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts = os.path.join(_root, "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
try:
    from rl_utils import camera_follow
except ImportError:
    camera_follow = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgbd_perception_pipeline import RgbdPerceptionPipeline

COLORS_BGR = {
    "sugar_box": (0, 255, 255),
    "mustard_bottle": (255, 0, 255),
    "banana": (0, 255, 0),
}


# ─────────────────────────────────────────────────────────────
# 简易行走 (无 policy.pt 时)
# ─────────────────────────────────────────────────────────────
class SimpleWalk:
    LEG = list(range(12))

    def __init__(self, device: str, policy_path: str = ""):
        self.device = device
        self.policy = None
        self.mode = "scripted"
        self.vx, self.vy, self.wz = 0.5, 0.0, 0.0
        if policy_path and os.path.isfile(policy_path):
            self.policy = torch.jit.load(policy_path, map_location=device)
            self.policy.eval()
            self.mode = "rl"
            self.env_to_train = torch.tensor(
                [4.0, 2.0, 2.0] * 4, device=device, dtype=torch.float32).view(1, -1)
            self.train_to_env = torch.tensor(
                [0.25, 0.5, 0.5] * 4, device=device, dtype=torch.float32).view(1, -1)
            print(f"[walk] RL policy: {policy_path}", flush=True)
        else:
            print("[walk] scripted gait (no policy.pt)", flush=True)

    def set_cmd(self, vx, wz):
        self.vx, self.wz = vx, wz

    def get_action(self, obs, step: int, turn_bias: float = 0.0):
        if self.mode == "rl":
            return self._rl(obs)
        a = torch.zeros(1, 20, dtype=torch.float32, device=self.device)
        t = step * 0.02
        s1, s2 = np.sin(t * 3), np.sin(t * 3 + np.pi)

        def leg(ih, it, ic, s):
            a[0, ih] = 0.25 * s
            a[0, it] = 0.55 + 0.45 * s
            a[0, ic] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1); leg(3, 4, 5, s2)
        leg(6, 7, 8, s2); leg(9, 10, 11, s1)
        if turn_bias:
            a[0, 0] += turn_bias; a[0, 6] += turn_bias
            a[0, 3] -= turn_bias; a[0, 9] -= turn_bias
        return a

    def _rl(self, obs):
        p = obs["proprio"].to(self.device).float()
        leg_pos = p[:, 12:24]
        leg_vel = p[:, 32:44]
        leg_act = p[:, 52:64] * self.env_to_train
        cmd = torch.tensor([[self.vx, self.vy, self.wz]], device=self.device, dtype=p.dtype)
        pol_in = torch.cat([p[:, 3:6] * 0.25, p[:, 9:12], cmd, leg_pos, leg_vel * 0.05, leg_act], dim=-1)
        with torch.inference_mode():
            leg = self.policy(pol_in)
        full = torch.zeros(1, 20, device=self.device, dtype=p.dtype)
        full[:, :12] = leg * self.train_to_env
        return full


class ManualKeyboard:
    def __init__(self, device: str, policy_path: str):
        self.walk = SimpleWalk(device, policy_path)
        self._input = self._kb = self._ki = None
        self._p_was = False
        self.snap = False
        self.quit = False
        try:
            import carb.input
            import omni.appwindow
            self._input = carb.input.acquire_input_interface()
            self._kb = omni.appwindow.get_default_app_window().get_keyboard()
            self._ki = carb.input.KeyboardInput
            print("[keyboard] OK — click Isaac Sim window first", flush=True)
        except Exception as e:
            print(f"[keyboard] unavailable: {e}", flush=True)

    def _down(self, key):
        if self._input is None:
            return False
        return self._input.get_keyboard_value(self._kb, key) > 0

    def poll(self):
        self.snap = False
        if self._ki is None:
            return
        # 用 P 存图; 不要用 SPACE — Isaac Sim 默认占空格(暂停/切镜头)
        p_key = self._down(self._ki.P)
        if p_key and not self._p_was:
            self.snap = True
        self._p_was = p_key
        if self._down(self._ki.Q):
            self.quit = True

    def get_action(self, obs, step: int):
        self.poll()
        if self._ki is None:
            return torch.zeros(1, 20, dtype=torch.float32, device=self.walk.device)

        w = self._down(self._ki.W)
        s = self._down(self._ki.S)
        a = self._down(self._ki.A)
        d = self._down(self._ki.D)
        if not (w or s or a or d):
            return torch.zeros(1, 20, dtype=torch.float32, device=self.walk.device)

        vx = 0.5 if w else (-0.25 if s else 0.0)
        wz = 0.45 if a else (-0.45 if d else 0.0)
        if w and (a or d):
            vx = 0.35
        tb = 0.5 if a else (-0.5 if d else 0.0)
        if self.walk.mode == "rl":
            self.walk.set_cmd(vx, wz)
            return self.walk.get_action(obs, step)
        return self.walk.get_action(obs, step, turn_bias=tb)


def draw_vis(rgb, out_dict):
    vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    for obj in out_dict.get("objects_detailed", []):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        c = COLORS_BGR.get(obj["class"], (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        label = f"ID{obj['id']} {obj['class']} {obj['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(12, y1 - 4)), 0, 0.45, c, 1)
        pw = obj["pos_world"]
        gp = obj.get("grasp_pos_world", pw)
        cv2.putText(vis, f"d={obj['dist_to_robot']:.2f} z={pw[2]:.2f}",
                    (x1, min(h - 4, y2 + 14)), 0, 0.4, c, 1)
        cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 0, 255), -1)
    tgt = out_dict.get("target")
    if tgt:
        cv2.putText(vis,
                    f"TARGET {tgt['class']} d={tgt['dist_to_robot']:.2f} "
                    f"grasp_z={tgt['grasp_pos_world'][2]:.2f}",
                    (8, h - 8), 0, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(vis, "TARGET None — walk to objects (WASD)", (8, h - 8),
                    0, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, f"objects={len(out_dict.get('objects_detailed', []))}",
                (8, 22), 0, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, "P=save  Q=quit (no SPACE)", (8, 42), 0, 0.45, (200, 200, 200), 1)
    return vis


def print_help():
    print("\n=== RGB-D perception manual test ===", flush=True)
    print("  Click Isaac Sim window, then:", flush=True)
    print("  W/S/A/D   move    P   save png    Q   quit", flush=True)
    print("  Do NOT use SPACE — Isaac Sim steals it (zoom/pause)", flush=True)
    print("  If camera drifts: click Isaac window, WASD still works", flush=True)
    print("  OpenCV window: P=save  Q=quit", flush=True)
    print("==================================\n", flush=True)


def resolve_policy():
    if args_cli.policy:
        return args_cli.policy
    for p in [
        os.path.join(_root, "demo", "policy.pt"),
        os.path.join(os.path.dirname(__file__), "policy.pt"),
    ]:
        if os.path.isfile(p):
            return p
    return ""


def run_perception(pipeline, obs):
    return pipeline.process(obs, dt=0.02)


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args_cli.out))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")

    use_gui = not getattr(args_cli, "headless", False)
    policy = resolve_policy()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    apply_safe_action_spec(env)

    pipeline = RgbdPerceptionPipeline()
    device = env.unwrapped.device
    manual = ManualKeyboard(device, policy) if args_cli.mode == "manual" else None

    print_help()
    print(f"Output folder: {out_dir}", flush=True)

    obs, _ = env.reset()
    for _ in range(15):
        obs, _, _, _, _ = env.step(torch.zeros(1, 20, dtype=torch.float32, device=device))
        if use_gui and camera_follow:
            camera_follow(env)

    saved = 0
    step = 0
    win = "RGBD Perception (Q to quit)"

    with open(log_path, "w", encoding="utf-8") as logf:
        while True:
            if args_cli.mode == "manual":
                if manual.quit:
                    break
                action = manual.get_action(obs, step)
            else:
                action = torch.zeros(1, 20, dtype=torch.float32, device=device)
                action[0, 0] = 0.3

            obs, _, term, trunc, _ = env.step(action)
            step += 1
            if use_gui and camera_follow:
                camera_follow(env)

            do_preview = args_cli.live and (step % args_cli.preview_every == 0)
            opencv_key = -1
            do_save = manual.snap if manual else (step % 20 == 0)

            if do_preview or do_save:
                try:
                    out = run_perception(pipeline, obs)
                except Exception as e:
                    print(f"perception error: {e}", flush=True)
                    continue

                rgb = obs["image"]["head_rgb"].squeeze(0)
                rgb_np = (rgb.cpu() if rgb.device.type == "cuda" else rgb).numpy().astype(np.uint8)

                vis = draw_vis(rgb_np, out)
                if do_save:
                    path = os.path.join(out_dir, f"vis_{saved:04d}.png")
                    cv2.imwrite(path, vis)
                    n = len(out.get("objects_detailed", []))
                    tag = "[P] " if manual and manual.snap else ""
                    msg = f"{tag}saved {path} objects={n}"
                    if out.get("target"):
                        t = out["target"]
                        msg += f" target={t['class']} dist={t['dist_to_robot']:.2f}"
                    print(msg, flush=True)
                    logf.write(msg + "\n")
                    saved += 1

                if args_cli.live and do_preview:
                    cv2.imshow(win, vis)
                    opencv_key = cv2.waitKey(1) & 0xFF
                    if opencv_key == ord("q"):
                        break
                    if opencv_key == ord("p") and manual:
                        manual.snap = True

            # 存图或 OpenCV 抢焦点后，把视口拉回跟随机器人
            if (do_save or opencv_key == ord("p")) and use_gui and camera_follow:
                camera_follow(env)

            if term or trunc:
                obs, _ = env.reset()
                pipeline.reset()

    if args_cli.live:
        cv2.destroyAllWindows()
    print(f"Done. {saved} images in {out_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
