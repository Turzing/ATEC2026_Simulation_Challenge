"""
Task B 识别测试 — 黄颜色 HSV 分割 (第 1 阶段: 只验证框住物体)

用法:
    conda activate isaaclab
    cd taskb_perception
    python test_yellow_detect.py

操作 (先点 Isaac Sim 窗口):
    W/S/A/D  移动   P  存图   Q  退出
    不要用空格!

成功标准:
    走近黄物体 1~2m → 绿框在物体上, objects>=1
    只看灰地面     → objects=0

依赖:
    yellow_detect_pipeline.py
    config.py
    test_yellow_detect.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Yellow HSV detection test in Isaac Sim")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--mode", type=str, default="manual", choices=["manual", "auto"])
parser.add_argument("--policy", type=str, default="")
parser.add_argument("--out", type=str, default="../datasets/yellow_debug")
parser.add_argument("--live", action="store_true", default=True)
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview_every", type=int, default=3)
parser.add_argument("--no-depth-gate", action="store_true", help="Disable depth protrusion filter")
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
import yellow_detect_pipeline as ydp
from yellow_detect_pipeline import YellowDetectPipeline

if args_cli.no_depth_gate:
    ydp.USE_DEPTH_GATE = False


class SimpleWalk:
    def __init__(self, device: str, policy_path: str = ""):
        self.device = device
        self.policy = None
        self.mode = "scripted"
        self.vx, self.wz = 0.5, 0.0
        if policy_path and os.path.isfile(policy_path):
            self.policy = torch.jit.load(policy_path, map_location=device)
            self.policy.eval()
            self.mode = "rl"
            self.env_to_train = torch.tensor([4.0, 2.0, 2.0] * 4, device=device, dtype=torch.float32).view(1, -1)
            self.train_to_env = torch.tensor([0.25, 0.5, 0.5] * 4, device=device, dtype=torch.float32).view(1, -1)

    def set_cmd(self, vx, wz):
        self.vx, self.wz = vx, wz

    def get_action(self, obs, step: int, turn_bias: float = 0.0):
        if self.mode == "rl":
            p = obs["proprio"].to(self.device).float()
            leg_pos, leg_vel = p[:, 12:24], p[:, 32:44]
            leg_act = p[:, 52:64] * self.env_to_train
            cmd = torch.tensor([[self.vx, 0.0, self.wz]], device=self.device, dtype=p.dtype)
            pol_in = torch.cat([p[:, 3:6] * 0.25, p[:, 9:12], cmd, leg_pos, leg_vel * 0.05, leg_act], dim=-1)
            with torch.inference_mode():
                leg = self.policy(pol_in)
            full = torch.zeros(1, 20, device=self.device, dtype=p.dtype)
            full[:, :12] = leg * self.train_to_env
            return full
        a = torch.zeros(1, 20, dtype=torch.float32, device=self.device)
        t, s1, s2 = step * 0.02, np.sin(step * 0.02 * 3), np.sin(step * 0.02 * 3 + np.pi)

        def leg(ih, it, ic, s):
            a[0, ih], a[0, it], a[0, ic] = 0.25 * s, 0.55 + 0.45 * s, -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1); leg(3, 4, 5, s2)
        leg(6, 7, 8, s2); leg(9, 10, 11, s1)
        if turn_bias:
            a[0, 0] += turn_bias; a[0, 6] += turn_bias
            a[0, 3] -= turn_bias; a[0, 9] -= turn_bias
        return a


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
            print("[keyboard] OK", flush=True)
        except Exception as e:
            print(f"[keyboard] unavailable: {e}", flush=True)

    def _down(self, key):
        return self._input is not None and self._input.get_keyboard_value(self._kb, key) > 0

    def poll(self):
        self.snap = False
        if self._ki is None:
            return
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
        w, s = self._down(self._ki.W), self._down(self._ki.S)
        a, d = self._down(self._ki.A), self._down(self._ki.D)
        if not (w or s or a or d):
            return torch.zeros(1, 20, dtype=torch.float32, device=self.walk.device)
        vx = 0.5 if w else (-0.25 if s else 0.0)
        wz = 0.45 if a else (-0.45 if d else 0.0)
        tb = 0.5 if a else (-0.5 if d else 0.0)
        if self.walk.mode == "rl":
            self.walk.set_cmd(vx, wz)
            return self.walk.get_action(obs, step)
        return self.walk.get_action(obs, step, turn_bias=tb)


def draw_vis(rgb, out_dict, pipeline: YellowDetectPipeline):
    vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    n_obj = len(out_dict.get("objects_detailed", []))

    ymask = pipeline.get_debug_yellow_mask()
    if ymask is not None:
        tint = cv2.cvtColor(ymask, cv2.COLOR_GRAY2BGR)
        tint[:, :, 1] = ymask
        small = cv2.resize(tint, (w // 3, h // 3))
        vis[0:small.shape[0], 0:small.shape[1]] = cv2.addWeighted(
            vis[0:small.shape[0], 0:small.shape[1]], 0.55, small, 0.45, 0,
        )
        cv2.putText(vis, "yellow mask", (4, small.shape[0] - 4), 0, 0.35, (0, 255, 255), 1)

    for obj in out_dict.get("objects_detailed", []):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID{obj['id']} conf={obj['conf']:.2f}"
        if obj.get("depth_m") is not None:
            label += f" d={obj['depth_m']:.2f}m"
        cv2.putText(vis, label, (x1, max(14, y1 - 4)), 0, 0.45, (0, 255, 0), 1)
        cx, cy = obj.get("centroid_uv", [(x1 + x2) / 2, (y1 + y2) / 2])
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    cv2.putText(vis, f"objects={n_obj}", (8, 24), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "WASD move | P save | Q quit", (8, 48), 0, 0.45, (200, 200, 200), 1)
    if n_obj == 0:
        cv2.putText(vis, "Walk closer to YELLOW objects (1-2m)", (8, h - 10), 0, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "OK: box on object?", (8, h - 10), 0, 0.5, (0, 255, 0), 2)
    return vis


def resolve_policy():
    if args_cli.policy:
        return args_cli.policy
    for p in [os.path.join(_root, "demo", "policy.pt"), os.path.join(os.path.dirname(__file__), "policy.pt")]:
        if os.path.isfile(p):
            return p
    return ""


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args_cli.out))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    use_gui = not getattr(args_cli, "headless", False)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg = apply_safe_action_spec(env_cfg, "{}")

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        pipeline = YellowDetectPipeline()
        device = env.unwrapped.device
        manual = ManualKeyboard(device, resolve_policy()) if args_cli.mode == "manual" else None

        print("\n=== Yellow detect test (phase 1: 2D boxes only) ===", flush=True)
        print(f"  depth gate: {ydp.USE_DEPTH_GATE}", flush=True)
        print(f"  output: {out_dir}\n", flush=True)

        obs, _ = env.reset()
        for _ in range(15):
            obs, _, _, _, _ = env.step(torch.zeros(1, 20, dtype=torch.float32, device=device))
            if use_gui and camera_follow:
                camera_follow(env)

        saved, step = 0, 0
        win = "Yellow Detect (Q quit)"
        with open(log_path, "w", encoding="utf-8") as logf:
            while True:
                if manual and manual.quit:
                    break
                if manual:
                    action = manual.get_action(obs, step)
                else:
                    action = torch.zeros(1, 20, dtype=torch.float32, device=device)
                    action[0, 0] = 0.3

                obs, _, term, trunc, _ = env.step(action)
                step += 1
                if use_gui and camera_follow:
                    camera_follow(env)

                do_preview = args_cli.live and (step % args_cli.preview_every == 0)
                do_save = manual.snap if manual else False

                if do_preview or do_save:
                    out = pipeline.process(obs)
                    rgb = obs["image"]["head_rgb"].squeeze(0)
                    rgb_np = (rgb.cpu() if rgb.device.type == "cuda" else rgb).numpy().astype(np.uint8)
                    vis = draw_vis(rgb_np, out, pipeline)

                    if do_save:
                        path = os.path.join(out_dir, f"vis_{saved:04d}.png")
                        cv2.imwrite(path, vis)
                        n = len(out.get("objects_detailed", []))
                        msg = f"saved {path} objects={n}"
                        print(msg, flush=True)
                        logf.write(msg + "\n")
                        saved += 1

                    if do_preview:
                        cv2.imshow(win, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        if key == ord("p") and manual:
                            manual.snap = True

                if term or trunc:
                    obs, _ = env.reset()
                    pipeline.reset()

        if args_cli.live:
            cv2.destroyAllWindows()
        print(f"Done. {saved} images in {out_dir}", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
