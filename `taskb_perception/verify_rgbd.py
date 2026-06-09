"""
一键检查官方 RGB-D 是否正常 — 先跑这个再跑检测

    conda activate isaaclab
    cd taskb_perception
    python verify_rgbd.py

输出:
    终端打印 depth 统计
    ../datasets/rgbd_verify/verify.png  (RGB | 深度 | 饱和度 | relief)
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify head RGB-D from official obs")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgbd_utils import depth_stats, depth_to_vis, format_stats_line, parse_head_rgbd

OUT = os.path.join(os.path.dirname(__file__), "..", "datasets", "rgbd_verify")


def main():
    os.makedirs(OUT, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    dev = env.unwrapped.device
    for _ in range(40):
        obs, _, _, _, _ = env.step(torch.zeros(1, 20, dtype=torch.float32, device=dev))

    rgb, depth = parse_head_rgbd(obs)
    stats = depth_stats(depth)

    print("\n=== Official head RGB-D check ===")
    print(f"  rgb shape:   {rgb.shape} dtype={rgb.dtype}")
    print(f"  depth shape: {depth.shape} dtype={depth.dtype}")
    print(f"  depth stats: {format_stats_line(stats)}")

    if stats["valid_ratio"] < 0.05:
        print("\n  FAIL: depth almost empty — use --enable_cameras, check Isaac camera")
    elif stats["median"] < 0.1 or stats["median"] > 40:
        print("\n  WARN: depth median looks wrong — check units (expect meters ~1-8)")
    else:
        print("\n  OK: depth has valid pixels — RGB-D data is usable")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    ground = depth.copy()
    ground[ground <= 0] = 8.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    g = cv2.morphologyEx(ground, cv2.MORPH_OPEN, k)
    relief = np.clip((g - depth) / 0.15 * 255, 0, 255).astype(np.uint8)

    panel1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel2 = depth_to_vis(depth)
    panel3 = cv2.applyColorMap(sat, cv2.COLORMAP_JET)
    panel4 = cv2.applyColorMap(relief, cv2.COLORMAP_HOT)

    for p, t in zip([panel1, panel2, panel3, panel4], ["RGB", "Depth(m)", "Saturation", "Relief"]):
        cv2.putText(p, t, (8, 22), 0, 0.6, (255, 255, 255), 2)

    top = np.hstack([panel1, panel2])
    bot = np.hstack([panel3, panel4])
    vis = np.vstack([top, bot])
    cv2.putText(vis, format_stats_line(stats), (8, vis.shape[0] - 8), 0, 0.5, (0, 255, 255), 1)

    path = os.path.join(OUT, "verify.png")
    cv2.imwrite(path, vis)
    print(f"  saved: {os.path.abspath(path)}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
