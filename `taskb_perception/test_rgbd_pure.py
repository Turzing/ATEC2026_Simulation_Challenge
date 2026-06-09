"""
老师版 RGB-D 自测 — depth 凸起 + RGB 黄色融合

    python test_rgbd_pure.py

WASD 移动 | 停住再看更稳 | P 存图 | Q 退出
底栏: depth | relief | rgb_fg | fusion
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RGB-D fusion (depth relief + RGB)")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="../datasets/rgbd_pure_debug")
parser.add_argument("--live", action="store_true", default=True)
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview_every", type=int, default=4)
parser.add_argument("--policy", type=str, default="")
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
from rgbd_pure_pipeline import RgbdPurePipeline
from rgbd_utils import depth_to_vis, format_stats_line, parse_head_rgbd
from sim_test_common import CLASS_COLORS_BGR, ManualKeyboard, draw_panel, resolve_policy


def draw_vis(rgb, out, pipeline, depth):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    objs = out.get("objects_detailed", [])

    pw, ph = w // 4, h // 4
    mini = np.zeros((ph, pw * 4, 3), dtype=np.uint8)
    draw_panel(mini, cv2.resize(depth_to_vis(depth), (pw, ph)), 0, 0, "depth")
    for i, key in enumerate(["relief", "rgb_fg", "fusion"], start=1):
        m = pipeline.get_debug(key)
        if m is not None:
            if key == "relief":
                c = cv2.applyColorMap(cv2.resize(m, (pw, ph)), cv2.COLORMAP_TURBO)
            else:
                c = cv2.cvtColor(cv2.resize(m, (pw, ph)), cv2.COLOR_GRAY2BGR)
            draw_panel(mini, c, i * pw, 0, key)
    vis = np.vstack([vis, mini])

    for obj in objs:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cls = obj.get("class", "?")
        c = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        dm = obj.get("depth_m")
        lab = f"ID{obj['id']} {cls}"
        if dm:
            lab += f" {dm:.2f}m"
        cv2.putText(vis, lab, (x1, max(14, y1 - 4)), 0, 0.42, c, 1)
        cu, cv = obj.get("centroid_uv", [0, 0])
        cv2.circle(vis, (int(cu), int(cv)), 4, (0, 0, 255), -1)

    st = out.get("depth_stats", {})
    cv2.putText(vis, f"RGBD-fusion  objects={len(objs)}  mask={out.get('mask_components', 0)}",
                (8, 22), 0, 0.5, (0, 255, 255), 2)
    cv2.putText(vis, format_stats_line(st), (8, 44), 0, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, "depth凸起+RGB黄 | 停稳再看 | WASD P Q", (8, 66), 0, 0.4, (180, 180, 180), 1)
    tgt = out.get("target")
    if tgt and tgt.get("grasp_pos_world"):
        g = tgt["grasp_pos_world"]
        cv2.putText(vis, f"grasp ({g[0]:.2f},{g[1]:.2f},{g[2]:.2f})", (8, 86), 0, 0.38, (0, 220, 255), 1)
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
    pipeline = RgbdPurePipeline()
    kb = ManualKeyboard(device, resolve_policy(_root, args_cli.policy))

    print("\n=== RGB-D 融合版 (depth relief + RGB yellow) ===", flush=True)
    print("  底栏 fusion 有白块才算 RGBD 对上\n", flush=True)

    obs, _ = env.reset()
    for _ in range(20):
        obs, _, _, _, _ = env.step(torch.zeros(1, 20, dtype=torch.float32, device=device))
        if camera_follow:
            camera_follow(env)

    step = saved = 0
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
            rgb, depth = parse_head_rgbd(obs)
            vis = draw_vis(rgb, out, pipeline, depth)

            if kb.snap:
                p = os.path.join(out_dir, f"pure_{saved:04d}.png")
                cv2.imwrite(p, vis)
                print(f"saved objects={len(out['objects_detailed'])}", flush=True)
                saved += 1
                kb.snap = False

            if args_cli.live:
                cv2.imshow("RGBD-Pure", vis)
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
