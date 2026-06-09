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
parser.add_argument("--fast", action="store_true", help="MIN_TRACK_HITS=1 (default pipeline already uses 1)")
parser.add_argument("--sat-min", type=int, default=None, help="Override SAT_MIN_ABSOLUTE (e.g. 40)")
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
from sim_test_common import CLASS_COLORS_BGR, ManualKeyboard, draw_panel, resolve_policy

if args_cli.sat_min is not None:
    rdp.SAT_MIN_ABSOLUTE = args_cli.sat_min


def draw_vis(rgb, out, pipeline, depth):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    n = len(out.get("objects_detailed", []))

    pw, ph = w // 4, h // 4
    mini = np.zeros((ph, pw * 4, 3), dtype=np.uint8)
    draw_panel(mini, cv2.resize(depth_to_vis(depth), (pw, ph)), 0, 0, "depth")
    sat = pipeline.get_debug("saturation")
    if sat is not None:
        draw_panel(mini, cv2.applyColorMap(cv2.resize(sat, (pw, ph)), cv2.COLORMAP_JET), pw, 0, "S")
    for i, key in enumerate(["sat_mask", "rgbd"], start=2):
        m = pipeline.get_debug(key)
        if m is not None:
            c = cv2.cvtColor(cv2.resize(m, (pw, ph)), cv2.COLOR_GRAY2BGR)
            draw_panel(mini, c, i * pw, 0, key)

    vis = np.vstack([vis, mini])

    for obj in out.get("objects_detailed", []):
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cls = obj.get("class", "object")
        c = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        dm = obj.get("depth_m")
        cc = obj.get("class_conf", 0.0)
        lab = f"ID{obj['id']} {cls}"
        if cls == "unknown":
            lab += " ?"
        lab += f" {cc:.2f}"
        if dm:
            lab += f" {dm:.2f}m"
        cv2.putText(vis, lab, (x1, max(14, y1 - 4)), 0, 0.42, c, 1)
        ext = obj.get("geom_extents")
        if ext:
            cv2.putText(vis, f"3D {ext[0]:.2f}/{ext[1]:.2f}/{ext[2]:.2f}",
                        (x1, min(vis.shape[0] - 6, y2 + 14)), 0, 0.38, c, 1)

    st = out.get("depth_stats", {})
    cv2.putText(vis, f"objects={n}  mask={out.get('mask_components', 0)}  {format_stats_line(st)}",
                (8, 22), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(vis,
                f"sat={out.get('sat_thresh', 0):.0f}/{out.get('sat_thresh_far', 0):.0f} (near/far)",
                (8, 44), 0, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, "SATURATION detect | WASD P Q", (8, 66), 0, 0.4, (200, 200, 200), 1)
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
    kb = ManualKeyboard(device, resolve_policy(_root, args_cli.policy))

    print("\n=== Saturation detect (S channel + depth distance) ===", flush=True)
    print("  Bottom: depth | S heatmap | sat_mask | detect", flush=True)
    print("  Walk to objects 1-3m — sat_mask should match verify.png Saturation\n", flush=True)

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
