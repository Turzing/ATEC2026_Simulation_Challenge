"""
双相机分工测试 — head 导航 / EE 抓取

    python test_rgbd_dual_detect.py

主画面: head (NAV)  右上: EE (GRASP)  各管各的, 不融合
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="head=nav ee=grasp")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--out", type=str, default="../datasets/rgbd_dual_debug")
parser.add_argument("--live", action="store_true", default=True)
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview_every", type=int, default=3)
parser.add_argument("--sat-min", type=int, default=None)
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
import rgbd_detect_pipeline as rdp
from rgbd_dual_pipeline import GRASP_PHASE_DEPTH_M, RgbdDualPipeline
from rgbd_utils import depth_to_vis, format_stats_line, parse_ee_rgbd, parse_head_rgbd
from test_rgbd_detect import CLASS_COLORS_BGR, ManualKeyboard, draw_panel, resolve_policy

if args_cli.sat_min is not None:
    rdp.SAT_MIN_ABSOLUTE = args_cli.sat_min


def draw_boxes(vis, objects, prefix="", thickness=2):
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cls = obj.get("class", "?")
        c = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, thickness)
        dm = obj.get("depth_m")
        lab = f"{prefix}ID{obj['id']} {cls}"
        if dm:
            lab += f" {dm:.2f}m"
        cv2.putText(vis, lab, (x1, max(14, y1 - 4)), 0, 0.4, c, 1)


def draw_vis(head_rgb, ee_rgb, out, pipeline, head_depth):
    vis = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    nav = out.get("objects_nav", [])
    grasp = out.get("objects_grasp", [])
    phase = out.get("phase", "approach")

    draw_boxes(vis, nav, prefix="N", thickness=3 if phase == "approach" else 1)

    pw, ph = w // 4, h // 4
    mini = np.zeros((ph, pw * 4, 3), dtype=np.uint8)
    draw_panel(mini, cv2.resize(depth_to_vis(head_depth), (pw, ph)), 0, 0, "head_d")
    sat = pipeline.get_debug("head", "saturation")
    if sat is not None:
        draw_panel(mini, cv2.applyColorMap(cv2.resize(sat, (pw, ph)), cv2.COLORMAP_JET), pw, 0, "S")
    for i, key in enumerate(["sat_mask", "rgbd"], start=2):
        m = pipeline.get_debug("head", key)
        if m is not None:
            draw_panel(mini, cv2.cvtColor(cv2.resize(m, (pw, ph)), cv2.COLOR_GRAY2BGR), i * pw, 0, key)
    vis = np.vstack([vis, mini])

    if ee_rgb is not None:
        ew, eh = w // 3, h // 3
        ee_vis = cv2.resize(cv2.cvtColor(ee_rgb, cv2.COLOR_RGB2BGR), (ew, eh))
        sx, sy = ew / ee_rgb.shape[1], eh / ee_rgb.shape[0]
        for obj in grasp:
            x1, y1, x2, y2 = obj["bbox"]
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cls = obj.get("class", "?")
            c = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
            t = 3 if phase == "grasp" else 1
            cv2.rectangle(ee_vis, (x1, y1), (x2, y2), c, t)
            cv2.putText(ee_vis, f"G{obj['id']}", (x1, max(12, y1 - 2)), 0, 0.35, c, 1)
        cv2.rectangle(ee_vis, (0, 0), (ew - 1, eh - 1), (0, 255, 255), 2)
        cv2.putText(ee_vis, "GRASP(EE)", (4, 14), 0, 0.38, (0, 255, 255), 1)
        vis[8:8 + eh, w - ew - 8:w - 8] = ee_vis

    st = out.get("depth_stats", {})
    cv2.putText(vis, f"phase={phase}  NAV(head)={len(nav)}  GRASP(ee)={len(grasp)}",
                (8, 22), 0, 0.5, (0, 255, 255), 2)
    cv2.putText(vis, format_stats_line(st), (8, 44), 0, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, f"grasp when nav d<{GRASP_PHASE_DEPTH_M:.1f}m | WASD P Q",
                (8, 64), 0, 0.38, (180, 180, 180), 1)
    tn, tg = out.get("target_nav"), out.get("target_grasp")
    if tn:
        cv2.putText(vis, f"NAV  {tn.get('class')} d={tn.get('depth_m', 0):.2f}m",
                    (8, 84), 0, 0.4, (0, 255, 0), 1)
    if tg:
        cv2.putText(vis, f"GRASP {tg.get('class')} d={tg.get('depth_m', 0):.2f}m",
                    (8, 104), 0, 0.4, (0, 255, 255), 1)
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
    pipeline = RgbdDualPipeline()
    kb = ManualKeyboard(device, resolve_policy())

    print("\n=== head=NAV  ee=GRASP  (parallel, no fusion) ===\n", flush=True)

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
            head_rgb, head_depth = parse_head_rgbd(obs)
            ee_rgb, _ = parse_ee_rgbd(obs)
            vis = draw_vis(head_rgb, ee_rgb, out, pipeline, head_depth)

            if kb.snap:
                p = os.path.join(out_dir, f"dual_{saved:04d}.png")
                cv2.imwrite(p, vis)
                print(f"saved nav={len(out['objects_nav'])} grasp={len(out['objects_grasp'])} "
                      f"phase={out['phase']}", flush=True)
                saved += 1
                kb.snap = False

            if args_cli.live:
                cv2.imshow("Nav+Grasp", vis)
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
