"""
官方 RGB-D 融合识别测试 (head_rgb + head_depth)

    python verify_rgbd.py          # 先确认 depth 有数据
    python test_rgbd_detect.py     # 再跑识别

WASD 移动 | P 存图 | Q 退出
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Official RGB-D fusion detection")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="../datasets/rgbd_detect_debug")
parser.add_argument("--live", action="store_true", default=True)
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview_every", type=int, default=3)
parser.add_argument("--fast", action="store_true", help="MIN_TRACK_HITS=1 for quicker boxes")
parser.add_argument("--policy", type=str, default="", help="RL policy.pt (default: ../demo/policy.pt)")
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
sys.path.insert(0, os.path.join(_root, "scripts"))
try:
    from rl_utils import camera_follow
except ImportError:
    camera_follow = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rgbd_detect_pipeline as rdp
from rgbd_detect_pipeline import RgbdDetectPipeline
from rgbd_utils import depth_to_vis, format_stats_line

if args_cli.fast:
    rdp.MIN_TRACK_HITS = 1


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


class SimpleWalk:
    """与 test_rgbd_sim.py 相同 — W 前进用 RL 策略, 无 policy 时原地踏步+转向"""

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
            print("[walk] scripted gait — copy demo/policy.pt for smooth forward walk", flush=True)

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
        self.snap = self.quit = False
        self._p_was = False
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


def draw_panel(vis, img, x, y, label):
    h, w = img.shape[:2]
    vis[y:y + h, x:x + w] = img
    cv2.putText(vis, label, (x + 4, y + 16), 0, 0.4, (255, 255, 255), 1)


def draw_vis(rgb, out, pipeline, depth):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    n = len(out.get("objects_detailed", []))

    pw, ph = w // 4, h // 4
    mini = np.zeros((ph, pw * 4, 3), dtype=np.uint8)
    dvis = cv2.resize(depth_to_vis(depth), (pw, ph))
    draw_panel(mini, dvis, 0, 0, "depth")
    for i, key in enumerate(["valid_depth", "color", "relief", "rgbd"]):
        m = pipeline.get_debug(key)
        if m is not None:
            c = cv2.cvtColor(cv2.resize(m, (pw, ph)), cv2.COLOR_GRAY2BGR)
            draw_panel(mini, c, i * pw, 0, key)

    vis = np.vstack([vis, mini])

    for obj in out.get("objects_detailed", []):
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dm = obj.get("depth_m")
        lab = f"ID{obj['id']}"
        if dm:
            lab += f" {dm:.2f}m"
        cv2.putText(vis, lab, (x1, max(14, y1 - 4)), 0, 0.45, (0, 255, 0), 1)

    st = out.get("depth_stats", {})
    cv2.putText(vis, f"objects={n}  {format_stats_line(st)}", (8, 22), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, f"max_relief={out.get('max_relief', 0):.3f}", (8, 44), 0, 0.45, (0, 255, 255), 1)
    cv2.putText(vis, "RGB+D fusion | WASD P Q", (8, 66), 0, 0.4, (200, 200, 200), 1)
    return vis


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args_cli.out))
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    device = env.unwrapped.device
    pipeline = RgbdDetectPipeline()
    kb = ManualKeyboard(device, resolve_policy())

    print("\n=== RGB-D fusion detect (official head_rgb + head_depth) ===", flush=True)
    print("  Step 1: python verify_rgbd.py  if depth valid_ratio=0", flush=True)
    print("  Step 2: walk to objects 1-3m, watch bottom row 'rgbd' panel\n", flush=True)

    obs, _ = env.reset()
    for _ in range(20):
        obs, _, _, _, _ = env.step(torch.zeros(1, 20, dtype=torch.float32, device=device))
        if camera_follow:
            camera_follow(env)

    step = saved = 0
    rgb_np, depth_np = None, None
    try:
        while simulation_app.is_running():
            if kb.quit:
                break
            act = kb.get_action(obs, step)
            obs, _, term, trunc, _ = env.step(act)
            step += 1
            if camera_follow:
                camera_follow(env)

            if step % args_cli.preview_every != 0 and not kb.snap:
                if term or trunc:
                    obs, _ = env.reset()
                    pipeline.reset()
                continue

            out = pipeline.process(obs)
            from rgbd_utils import parse_head_rgbd
            rgb_np, depth_np = parse_head_rgbd(obs)
            vis = draw_vis(rgb_np, out, pipeline, depth_np)

            if kb.snap:
                p = os.path.join(out_dir, f"vis_{saved:04d}.png")
                cv2.imwrite(p, vis)
                print(f"saved {p} objects={len(out['objects_detailed'])}", flush=True)
                saved += 1
                kb.snap = False

            if args_cli.live:
                cv2.imshow("RGBD Detect", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if term or trunc:
                obs, _ = env.reset()
                pipeline.reset()
    finally:
        cv2.destroyAllWindows()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
