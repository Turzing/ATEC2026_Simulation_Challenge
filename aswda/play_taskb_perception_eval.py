#!/usr/bin/env python3
"""
Task B 感知层单独验证 — 不跑 solution_rl / 操作层

默认: 手动模式 (WASD 移动 + P 拍照 + OpenCV 预览)

    cd ATEC2026_Simulation_Challenge_RIL-myq
    conda activate isaaclab
    python scripts/play_taskb_perception_eval.py --task ATEC-TaskB-B2Piper --enable_cameras

与 solution_rl 同条件 (无 GT 位姿):
    python scripts/play_taskb_perception_eval.py --task ATEC-TaskB-B2Piper --enable_cameras --no-gt-pose

两个窗口:
    Isaac Sim  = 3D 场景 (默认机器人 head 视角, 在此按 WASD)
    OpenCV     = 感知预览 (head+检测框, 不要在此按 WASD)

操作 (先点 Isaac Sim 窗口获焦):
    W/S     前进 / 后退
    A/D     左转 / 右转
    P       拍照存图 + 终端打印检测/GT 误差
    Q       退出

必须带 policy.pt 才能稳定站立行走:
    python scripts/play_taskb_perception_eval.py --task ATEC-TaskB-B2Piper --enable_cameras \\
        --policy demo/policy.pt

无 OpenCV 预览 (远程桌面易闪退):
    ... --no-live

自动批量评测 (旧模式):
    ... --auto --steps 300

输出: logs/perception_eval/<时间戳>/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from datetime import datetime
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
PERCEPTION_DIR = os.path.join(REPO_ROOT, "taskb_perception")
if os.path.isdir(PERCEPTION_DIR):
    sys.path.insert(0, PERCEPTION_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task B perception eval: manual WASD + snapshot (default).")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--policy", type=str, default="", help="demo/policy.pt for stable stand/walk")
parser.add_argument("--live", action="store_true", default=True, help="OpenCV live preview")
parser.add_argument("--no-live", action="store_false", dest="live")
parser.add_argument("--preview-every", type=int, default=2, help="Refresh preview every N sim steps")
parser.add_argument("--auto", action="store_true", default=False, help="Batch auto eval (no keyboard)")
parser.add_argument("--steps", type=int, default=300, help="Steps for --auto mode")
parser.add_argument("--warmup", type=int, default=15)
parser.add_argument("--save-every", type=int, default=25, help="Auto mode save interval")
parser.add_argument("--match-radius", type=float, default=0.55)
parser.add_argument("--max-gt-dist", type=float, default=6.0)
parser.add_argument(
    "--no-gt-pose",
    action="store_true",
    default=False,
    help="感知位姿用里程计 (与 solution_rl GT guidance=OFF 一致); 默认传 GT 便于看检测误差",
)
parser.add_argument(
    "--simulate-fuse",
    action="store_true",
    default=True,
    help="预览 solution_rl 融合层是否会拒 target_nav (默认开)",
)
parser.add_argument("--no-simulate-fuse", action="store_false", dest="simulate_fuse")
parser.add_argument(
    "--view",
    type=str,
    default="follow",
    choices=("follow", "head", "ee"),
    help="Isaac 主视口: follow=机器人后方第三人称 head/ee=传感器第一人称",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402

from config import BIN_CENTER, BIN_RADIUS  # noqa: E402
from rgbd_utils import depth_to_vis, parse_ee_rgbd, parse_head_rgbd  # noqa: E402
from sim_test_common import ManualKeyboard, resolve_policy  # noqa: E402
from taskb_perception import PERCEPTION_BUILD, TaskBPerception  # RgbdPureDualPipeline

try:
    from rl_utils import camera_follow, camera_follow_behind, camera_robot_sensor_view
except ImportError:
    camera_follow = None
    camera_follow_behind = None
    camera_robot_sensor_view = None


def _object_index(name: str) -> int | None:
    digits = "".join(c for c in str(name) if c.isdigit())
    if not digits:
        return None
    idx = int(digits)
    return idx if 1 <= idx <= 18 else None


def _object_class(idx: int) -> str:
    if idx <= 6:
        return "sugar_box"
    if idx <= 12:
        return "mustard_bottle"
    return "banana"


def _get_scene(env):
    env_u = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(env_u, "scene"):
        return env_u.scene
    if hasattr(env_u, "_env") and hasattr(env_u._env, "scene"):
        return env_u._env.scene
    return None


def get_robot_pose(env) -> tuple[np.ndarray, float]:
    scene = _get_scene(env)
    if scene is None:
        raise RuntimeError("Cannot access scene")
    robot = scene["robot"]
    if isinstance(robot, (list, tuple)):
        robot = robot[0]
    pos = robot.data.root_pos_w.cpu().numpy()[0].astype(np.float32)
    pos[2] = 0.68
    quat = robot.data.root_quat_w.cpu().numpy()[0]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pos, float(yaw)


def get_gt_objects(env, robot_pos: np.ndarray) -> list[dict[str, Any]]:
    scene = _get_scene(env)
    if scene is None:
        return []
    objects: list[dict[str, Any]] = []
    containers = []
    if hasattr(scene, "rigid_objects"):
        containers.append(scene.rigid_objects)
    if hasattr(scene, "articulations"):
        containers.append(scene.articulations)

    for container in containers:
        for name, body in container.items():
            idx = _object_index(name)
            if idx is None:
                continue
            try:
                if hasattr(body, "data") and hasattr(body.data, "root_pos_w"):
                    pw = body.data.root_pos_w.cpu().numpy()[0].astype(np.float32)
                elif hasattr(body, "root_pos_w"):
                    pw = body.root_pos_w.cpu().numpy()[0].astype(np.float32)
                else:
                    continue
            except Exception:
                continue
            dist_bin = float(np.linalg.norm(pw[:2] - BIN_CENTER[:2]))
            dist_robot = float(np.linalg.norm(pw[:2] - robot_pos[:2]))
            objects.append(
                {
                    "id": name,
                    "idx": idx,
                    "class": _object_class(idx),
                    "pos_world": pw,
                    "in_bin": dist_bin < BIN_RADIUS,
                    "dist_to_robot": dist_robot,
                }
            )
    return objects


def match_gt(
    pos_world: list | np.ndarray,
    gt_objects: list[dict],
    *,
    max_dist: float,
) -> tuple[dict | None, float]:
    pw = np.asarray(pos_world, dtype=np.float32)
    best, best_d = None, max_dist
    for g in gt_objects:
        if g["in_bin"]:
            continue
        d = float(np.linalg.norm(pw[:2] - g["pos_world"][:2]))
        if d < best_d:
            best_d, best = d, g
    return best, best_d


def _draw_cam(
    rgb: np.ndarray,
    objects: list[dict],
    gt_objects: list[dict],
    *,
    match_radius: float,
    prefix: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
    for obj in objects:
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pw = obj.get("pos_world")
        gt, err = (None, 999.0)
        if pw is not None:
            gt, err = match_gt(pw, gt_objects, max_dist=match_radius)
        color = (0, 220, 0) if gt is not None else (0, 80, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cls = str(obj.get("class") or "?")
        depth = float(obj.get("depth_m") or 0.0)
        dist = float(obj.get("dist_to_robot") or 0.0)
        oid = obj.get("id", "?")
        lab = f"{prefix}{oid} {cls} z={depth:.2f} r={dist:.2f}"
        if gt is not None:
            lab += f" err={err:.2f}m"
        else:
            lab += " NO_GT"
        cv2.putText(bgr, lab, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return bgr


def _compute_errors(
    ee_objs: list[dict],
    head_objs: list[dict],
    gt_vis: list[dict],
    match_radius: float,
) -> tuple[list[float], list[float], set[str]]:
    ee_errs, head_errs = [], []
    matched = set()
    for obj in ee_objs:
        pw = obj.get("pos_world")
        if pw is None:
            continue
        gt, err = match_gt(pw, gt_vis, max_dist=match_radius)
        if gt is not None:
            ee_errs.append(err)
            matched.add(gt["id"])
    for obj in head_objs:
        pw = obj.get("pos_world")
        if pw is None:
            continue
        gt, err = match_gt(pw, gt_vis, max_dist=match_radius)
        if gt is not None:
            head_errs.append(err)
            matched.add(gt["id"])
    return ee_errs, head_errs, matched


def build_vis(
    obs,
    out: dict,
    gt_vis: list[dict],
    perception,
    *,
    step: int,
    match_radius: float,
    fuse_msg: str = "",
    layer_verdict: str = "",
) -> np.ndarray:
    head_rgb, head_depth = parse_head_rgbd(obs)
    ee_rgb, _ = parse_ee_rgbd(obs)
    ee_objs = list(out.get("ee_objects") or [])
    head_objs = list(out.get("head_objects") or [])
    target_nav = out.get("target_nav") or {}
    nav_dist = float(target_nav.get("dist_to_robot") or target_nav.get("depth_m") or 0.0)
    ee_errs, head_errs, matched = _compute_errors(ee_objs, head_objs, gt_vis, match_radius)
    recall = len(matched) / max(len(gt_vis), 1)
    ee_mean = float(np.mean(ee_errs)) if ee_errs else float("nan")
    head_mean = float(np.mean(head_errs)) if head_errs else float("nan")
    phase = out.get("phase", "approach")
    nav_cam = out.get("nav_authority") or (out.get("navigation") or {}).get("camera") or "?"

    head_bgr = _draw_cam(head_rgb, head_objs, gt_vis, match_radius=match_radius, prefix="H")
    h, w = head_bgr.shape[:2]

    if ee_rgb is not None:
        ew, eh = w // 3, h // 3
        ee_bgr = cv2.resize(_draw_cam(ee_rgb, ee_objs, gt_vis, match_radius=match_radius, prefix="E"), (ew, eh))
        cv2.putText(ee_bgr, f"EE nav ({len(ee_objs)})", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        head_bgr[8 : 8 + eh, w - ew - 8 : w - 8] = ee_bgr

    pw, ph = w // 4, h // 4
    mini = np.zeros((ph, pw * 4, 3), dtype=np.uint8)
    mini[:, :pw] = depth_to_vis(head_depth)[:ph, :pw]
    if hasattr(perception, "get_debug"):
        for i, key in enumerate(["relief", "rgb_fg", "fusion"], start=1):
            m = perception.get_debug("head", key)
            if m is not None:
                if key == "relief":
                    c = cv2.applyColorMap(cv2.resize(m, (pw, ph)), cv2.COLORMAP_TURBO)
                else:
                    c = cv2.cvtColor(cv2.resize(m, (pw, ph)), cv2.COLOR_GRAY2BGR)
                mini[:, i * pw : (i + 1) * pw] = c
    head_bgr = np.vstack([head_bgr, mini])

    lines = [
        f"RGBD-dual step={step} phase={phase} NAV({nav_cam}) build={PERCEPTION_BUILD}",
        f"ee={len(ee_objs)} head={len(head_objs)} gt={len(gt_vis)} recall={recall:.0%}",
        f"ee_err={ee_mean:.2f}m head_err={head_mean:.2f}m | Isaac: WASD P=拍照 Q=退出",
    ]
    if layer_verdict:
        lines.append(layer_verdict[:90])
    elif fuse_msg:
        lines.append(fuse_msg[:90])
    y = 22
    for line in lines:
        cv2.putText(head_bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18
    if target_nav:
        cv2.putText(
            head_bgr,
            f"NAV {target_nav.get('class')} z={nav_dist:.2f}m [{nav_cam}]",
            (8, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
        )
    tg = out.get("target_grasp")
    if tg and tg.get("grasp_pos_world"):
        gp = tg["grasp_pos_world"]
        cv2.putText(
            head_bgr,
            f"GRASP ({gp[0]:.2f},{gp[1]:.2f},{gp[2]:.2f})",
            (8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 220, 255),
            1,
        )
    return head_bgr


def _print_snapshot(
    step: int,
    out: dict,
    gt_vis: list[dict],
    ee_errs: list[float],
    head_errs: list[float],
    matched: set[str],
    *,
    fuse_msg: str = "",
    layer_verdict: str = "",
    pose_mode: str = "gt",
) -> None:
    ee_objs = out.get("ee_objects") or []
    head_objs = out.get("head_objects") or []
    tn = out.get("target_nav") or {}
    print(
        f"\n[SNAP step={step} pose={pose_mode}] ee={len(ee_objs)} head={len(head_objs)} "
        f"auth={out.get('nav_authority')} nav_d={tn.get('dist_to_robot', 0):.2f} "
        f"phase={out.get('phase')} grasp_rel={out.get('grasp_reliable')}",
        flush=True,
    )
    if layer_verdict:
        print(f"  >>> {layer_verdict}", flush=True)
    elif fuse_msg:
        print(f"  >>> {fuse_msg}", flush=True)
    print(f"  GT in range={len(gt_vis)} matched={len(matched)} recall={len(matched)/max(len(gt_vis),1):.0%}", flush=True)
    if ee_errs:
        print(f"  EE errors: min={min(ee_errs):.2f} mean={np.mean(ee_errs):.2f} max={max(ee_errs):.2f} m", flush=True)
    if head_errs:
        print(f"  HEAD errors: min={min(head_errs):.2f} mean={np.mean(head_errs):.2f} max={max(head_errs):.2f} m", flush=True)
    for cam, objs in (("EE", ee_objs), ("HEAD", head_objs)):
        for o in objs[:4]:
            pw = o.get("pos_world")
            pw_s = f"({pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f})" if pw else "?"
            print(
                f"  {cam} id={o.get('id')} {o.get('class')} depth={o.get('depth_m', 0):.2f} world={pw_s}",
                flush=True,
            )


def _obs_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    raise TypeError(f"Unexpected obs type: {type(obs)}")


def _episode_done(terminated, truncated) -> bool:
    for flag in (terminated, truncated):
        if hasattr(flag, "any"):
            if bool(flag.any()):
                return True
        elif bool(flag):
            return True
    return False


def simulate_fuse_nav(target_nav: dict | None) -> str:
    """复现 solution_rl._fuse_perception_target 的 Z 门控 (含 EE 平面兜底)."""
    if not isinstance(target_nav, dict) or target_nav.get("pos_world") is None:
        return "FUSE: no target_nav"
    nav_cam = str(target_nav.get("source_camera") or "ee")
    pr = np.asarray(target_nav.get("pos_robot") or [0, 0, 0], dtype=np.float32)
    pw = np.asarray(target_nav.get("pos_world") or [0, 0, 0], dtype=np.float32)
    if float(pr[2]) < -0.30 or float(pw[2]) < 0.02 or float(pw[2]) > 0.45:
        depth_m = float(target_nav.get("nav_depth_m") or target_nav.get("depth_m") or 0.0)
        yaw_r = target_nav.get("nav_yaw_rel")
        if yaw_r is None:
            yaw_r = target_nav.get("yaw_rel")
        if nav_cam == "ee" and depth_m > 0.15 and yaw_r is not None:
            return "FUSE: OK (EE planar fallback)"
        return f"FUSE: REJECT bad_z robot_z={pr[2]:.2f} world_z={pw[2]:.2f}"
    return "FUSE: OK"


def _layer_verdict(out: dict, fuse_msg: str) -> str:
    ee_n = len(out.get("ee_objects") or [])
    head_n = len(out.get("head_objects") or [])
    tn = out.get("target_nav")
    has_nav = isinstance(tn, dict) and tn.get("pos_world") is not None
    if ee_n == 0 and head_n == 0:
        return "感知层: 未检出 (ee=0 head=0)"
    if not has_nav:
        return f"感知层: 有框但未出 target_nav (ee={ee_n} head={head_n})"
    if fuse_msg.startswith("FUSE: REJECT"):
        return f"操作层: 感知有目标但融合拒掉 — {fuse_msg}"
    if ee_n > 0 or head_n > 0:
        return f"感知层 OK (ee={ee_n} head={head_n}) | {fuse_msg}"
    return fuse_msg


def _update_viewport(env) -> bool:
    view = str(getattr(args_cli, "view", "follow") or "follow")
    ok = False
    if view == "follow":
        if camera_follow_behind:
            ok = bool(camera_follow_behind(env))
        elif camera_follow:
            camera_follow(env)
            ok = True
    else:
        cam_name = "ee_camera" if view == "ee" else "head_camera"
        if camera_robot_sensor_view:
            ok = bool(camera_robot_sensor_view(env, cam_name))
        if not ok:
            if camera_follow_behind:
                ok = bool(camera_follow_behind(env))
            elif camera_follow:
                camera_follow(env)
                ok = True
    if not getattr(_update_viewport, "_logged", False):
        _update_viewport._logged = True
        print(
            f"[viewport] mode={view} applied={'OK' if ok else 'FAILED (仍可能是 Isaac 默认俯视)'}",
            flush=True,
        )
    return ok


def _make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def run_manual() -> int:
    out_dir = os.path.join(REPO_ROOT, "logs", "perception_eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    policy_path = resolve_policy(REPO_ROOT, args_cli.policy)

    print(f"[PERC-EVAL] build={PERCEPTION_BUILD}  mode=MANUAL")
    print(f"[PERC-EVAL] pose={'odometry (--no-gt-pose)' if args_cli.no_gt_pose else 'GT (debug only, not control)'}")
    print(f"[PERC-EVAL] fuse_sim={'ON' if args_cli.simulate_fuse else 'OFF'}")
    print(f"[PERC-EVAL] output -> {out_dir}")
    if policy_path:
        print(f"[PERC-EVAL] policy -> {policy_path}")
    else:
        print("[PERC-EVAL] *** 未找到 demo/policy.pt — 狗站不稳, WASD 可能无效 ***")
        print("[PERC-EVAL] 请加: --policy demo/policy.pt")

    env = _make_env()
    device = env.unwrapped.device
    action_dim = int(np.prod(env.action_space.shape))
    perception = TaskBPerception()
    kb = ManualKeyboard(str(device), policy_path, action_dim=action_dim)
    print(f"[PERC-EVAL] action_dim={action_dim} keyboard={'OK' if kb.keyboard_ok else 'FAILED'}")

    live = args_cli.live
    print(f"[PERC-EVAL] Isaac viewport --view={args_cli.view} (follow=后方第三人称, head/ee=传感器视角)")

    print(
        "\n=== 感知验证 (绕过 solution_rl) ===\n"
        "  Isaac Sim = 默认机器人后方视角 + WASD\n"
        "  OpenCV 窗 = head 相机 + 检测框 (TaskB-Perception-Eval)\n"
        "  绿框=检测与 GT 对齐  橙框=检测但 GT 对不上\n"
        "  1. 点击 Isaac Sim 窗口 (不是 OpenCV)\n"
        "  2. WASD 移动找垃圾\n"
        "  3. P 拍照 (存图 + 打印 感知/融合 诊断)\n"
        "  4. Q 退出\n"
        "\n  判责: ee/head>0 且 target_nav 有 → 感知 OK\n"
        "        显示 FUSE: REJECT → 操作层 solution_rl 拒目标\n",
        flush=True,
    )

    obs, _ = env.reset()
    obs = _obs_dict(obs)
    for warm in range(20):
        act = kb.get_action(obs, warm)
        obs, _, term, trunc, _ = env.step(act)
        obs = _obs_dict(obs)
        _update_viewport(env)
        if _episode_done(term, trunc):
            obs, _ = env.reset()
            obs = _obs_dict(obs)

    step = saved = 0
    gt_printed = False
    frame_errors = 0

    try:
        while simulation_app.is_running() and not kb.quit:
            try:
                act = kb.get_action(obs, step)
                obs, _, term, trunc, _ = env.step(act)
                obs = _obs_dict(obs)
                step += 1
                _update_viewport(env)

                if step % 120 == 0 and kb.keyboard_ok:
                    vx, wz = kb.walk.last_cmd
                    if vx != 0.0 or wz != 0.0:
                        print(f"[PERC-EVAL] cmd vx={vx:.2f} wz={wz:.2f} (Isaac 窗 WASD)", flush=True)

                need_perceive = live or kb.snap or (step % max(1, args_cli.preview_every) == 0)
                if not need_perceive:
                    if _episode_done(term, trunc):
                        obs, _ = env.reset()
                        obs = _obs_dict(obs)
                        perception.reset()
                    continue

                robot_pos, robot_yaw = get_robot_pose(env)
                gt_all = get_gt_objects(env, robot_pos)
                gt_vis = [g for g in gt_all if (not g["in_bin"]) and g["dist_to_robot"] <= args_cli.max_gt_dist]

                if not gt_printed and gt_vis:
                    gt_printed = True
                    print("\n[PERC-EVAL] ===== GT (in range, 仅误差对比) =====", flush=True)
                    for g in sorted(gt_vis, key=lambda x: x["dist_to_robot"])[:12]:
                        pw = g["pos_world"]
                        print(
                            f"  {g['id']:10s} {g['class']:14s} dist={g['dist_to_robot']:.2f}m "
                            f"({pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f})",
                            flush=True,
                        )
                    print("[PERC-EVAL] ========================\n", flush=True)

                if args_cli.no_gt_pose:
                    out = perception.process(obs)
                    pose_mode = "odom"
                else:
                    out = perception.process(obs, gt_robot_pos=robot_pos, gt_robot_yaw=robot_yaw)
                    pose_mode = "gt"

                fuse_msg = simulate_fuse_nav(out.get("target_nav")) if args_cli.simulate_fuse else ""
                layer_verdict = _layer_verdict(out, fuse_msg) if args_cli.simulate_fuse else ""

                vis = build_vis(
                    obs, out, gt_vis, perception,
                    step=step, match_radius=args_cli.match_radius,
                    fuse_msg=fuse_msg, layer_verdict=layer_verdict,
                )
                ee_errs, head_errs, matched = _compute_errors(
                    list(out.get("ee_objects") or []),
                    list(out.get("head_objects") or []),
                    gt_vis,
                    args_cli.match_radius,
                )

                if kb.snap:
                    path = os.path.join(out_dir, f"snap_{saved:04d}.jpg")
                    cv2.imwrite(path, vis)
                    _print_snapshot(
                        step, out, gt_vis, ee_errs, head_errs, matched,
                        fuse_msg=fuse_msg, layer_verdict=layer_verdict, pose_mode=pose_mode,
                    )
                    print(f"  -> saved {path}\n", flush=True)
                    saved += 1
                    kb.snap = False

                if live:
                    cv2.imshow("TaskB-Perception-Eval", vis)
                    cv2.waitKey(1)

                if _episode_done(term, trunc):
                    obs, _ = env.reset()
                    obs = _obs_dict(obs)
                    perception.reset()

            except Exception:
                frame_errors += 1
                print(f"[PERC-EVAL] frame error #{frame_errors}:", flush=True)
                traceback.print_exc()
                if frame_errors >= 15:
                    break

    except KeyboardInterrupt:
        print("[PERC-EVAL] KeyboardInterrupt", flush=True)
    finally:
        cv2.destroyAllWindows()
        env.close()
        print(f"[PERC-EVAL] exit steps={step} saved={saved} -> {out_dir}", flush=True)
    return 0


def run_auto() -> int:
    out_dir = os.path.join(REPO_ROOT, "logs", "perception_eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    print(f"[PERC-EVAL] build={PERCEPTION_BUILD}  mode=AUTO")
    print(f"[PERC-EVAL] output -> {out_dir}")

    env = _make_env()
    perception = TaskBPerception()
    device = env.unwrapped.device
    action_dim = int(np.prod(env.action_space.shape))
    zero_action = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=device)

    obs, _ = env.reset()
    obs = _obs_dict(obs)
    for _ in range(max(0, args_cli.warmup)):
        obs, _, term, trunc, _ = env.step(zero_action)
        obs = _obs_dict(obs)
        if _episode_done(term, trunc):
            obs, _ = env.reset()
            obs = _obs_dict(obs)

    stats = {
        "frames": 0,
        "ee_det_total": 0,
        "head_det_total": 0,
        "ee_matched": 0,
        "head_matched": 0,
        "ee_err_sum": 0.0,
        "head_err_sum": 0.0,
        "ee_fp": 0,
        "head_fp": 0,
        "gt_recall_hits": 0,
        "gt_recall_total": 0,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "ee_n", "head_n", "nav_auth", "nav_dist", "grasp_rel", "ee_mean_err", "head_mean_err", "gt_visible", "gt_recall"])

        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            robot_pos, robot_yaw = get_robot_pose(env)
            gt_vis = [
                g for g in get_gt_objects(env, robot_pos)
                if (not g["in_bin"]) and g["dist_to_robot"] <= args_cli.max_gt_dist
            ]
            out = perception.process(obs, gt_robot_pos=robot_pos, gt_robot_yaw=robot_yaw)
            ee_objs = list(out.get("ee_objects") or [])
            head_objs = list(out.get("head_objects") or [])
            ee_errs, head_errs, matched = _compute_errors(ee_objs, head_objs, gt_vis, args_cli.match_radius)

            stats["frames"] += 1
            stats["ee_det_total"] += len(ee_objs)
            stats["head_det_total"] += len(head_objs)
            stats["ee_matched"] += len(ee_errs)
            stats["head_err_sum"] += sum(head_errs)
            stats["ee_err_sum"] += sum(ee_errs)
            stats["head_matched"] += len(head_errs)
            stats["gt_recall_total"] += len(gt_vis)
            stats["gt_recall_hits"] += len(matched)

            nav_dist = float((out.get("target_nav") or {}).get("dist_to_robot") or 0.0)
            writer.writerow([
                step, len(ee_objs), len(head_objs), out.get("nav_authority"), f"{nav_dist:.2f}",
                int(bool(out.get("grasp_reliable"))),
                f"{np.mean(ee_errs):.3f}" if ee_errs else "",
                f"{np.mean(head_errs):.3f}" if head_errs else "",
                len(gt_vis), f"{len(matched)/max(len(gt_vis),1):.2f}",
            ])

            if step % max(1, args_cli.save_every) == 0:
                vis = build_vis(obs, out, gt_vis, perception, step=step, match_radius=args_cli.match_radius)
                cv2.imwrite(os.path.join(out_dir, f"frame_{step:06d}.jpg"), vis)

            obs, _, term, trunc, _ = env.step(zero_action)
            obs = _obs_dict(obs)
            if _episode_done(term, trunc):
                obs, _ = env.reset()
                obs = _obs_dict(obs)
                perception.reset()

    env.close()
    n = max(stats["frames"], 1)
    summary = (
        f"frames={stats['frames']}\n"
        f"avg ee/head det: {stats['ee_det_total']/n:.2f} / {stats['head_det_total']/n:.2f}\n"
        f"GT recall: {stats['gt_recall_hits']}/{stats['gt_recall_total']}\n"
        f"-> {out_dir}\n"
    )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    try:
        code = run_auto() if args_cli.auto else run_manual()
    finally:
        simulation_app.close()
    raise SystemExit(code)
