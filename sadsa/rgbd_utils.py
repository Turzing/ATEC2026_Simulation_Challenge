"""
官方 obs['image'] 里 head_rgb / head_depth 的解析与诊断

Isaac Lab Camera depth:
    - float32, 单位米, 沿光轴距离 (与 perception_pipeline 一致)
    - 无效像素常为 inf 或 clipping 上限 (~50)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_ARM_JOINTS,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
)

CAMERA_MODELS = {
    "head": (HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX),
    "ee": (EE_CAM, EE_CAM_POS_ROBOT, EE_CAM_ROT_MATRIX),
}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "device") and getattr(x, "device", None) is not None:
        if x.device.type == "cuda":
            x = x.cpu()
    return np.asarray(x)


def align_rgb_to_depth(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """rgb/depth 尺寸不一致时对齐，避免 relief/HSV broadcast 崩溃"""
    h, w = depth.shape[:2]
    rgb = np.asarray(rgb)
    if rgb.ndim < 3:
        return rgb
    rgb = rgb[..., :3].astype(np.uint8)
    rh, rw = rgb.shape[:2]
    if rh == h and rw == w:
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def parse_head_rgbd(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    obs['image']['head_rgb']   → (H,W,3) uint8 RGB
    obs['image']['head_depth'] → (H,W)   float32 米
    """
    rgb = _to_numpy(obs["image"]["head_rgb"]).squeeze()
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb.astype(np.uint8)[..., :3]

    dep = _to_numpy(obs["image"]["head_depth"]).squeeze()
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep = dep.astype(np.float32)
    return rgb, sanitize_depth(dep)


def parse_ee_rgbd(obs: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        rgb = _to_numpy(obs["image"]["ee_rgb"]).squeeze()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = rgb.astype(np.uint8)[..., :3]
        dep = _to_numpy(obs["image"]["ee_depth"]).squeeze()
        if dep.ndim == 3:
            dep = dep[..., 0]
        return rgb, sanitize_depth(dep.astype(np.float32))
    except (KeyError, TypeError, AttributeError):
        return None, None


def sanitize_depth(depth: np.ndarray, max_m: float = 49.5) -> np.ndarray:
    d = depth.copy()
    bad = ~np.isfinite(d) | (d <= 0.01) | (d > max_m)
    d[bad] = 0.0
    return d


def depth_stats(depth: np.ndarray) -> Dict[str, float]:
    valid = depth[(depth > 0.05) & (depth < 49.0)]
    if valid.size == 0:
        return {"valid_ratio": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "valid_ratio": float(valid.size / depth.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
    }


def depth_to_vis(depth: np.ndarray, d_min: float = 0.5, d_max: float = 8.0) -> np.ndarray:
    """深度 → 伪彩色 BGR (调试用)"""
    d = depth.copy()
    d[d <= 0] = d_max
    d = np.clip(d, d_min, d_max)
    norm = ((d_max - d) / max(d_max - d_min, 1e-3) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def median_depth_in_mask(depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
    vals = depth[mask > 0]
    vals = vals[(vals > 0.05) & (vals < 49.0)]
    if len(vals) < 5:
        return None
    return float(np.median(vals))


def estimate_pos_robot(
    centroid_uv, depth_m: Optional[float], camera: str,
) -> Optional[np.ndarray]:
    """检测框中心 + depth → 机器人基座系 3D 点 (双相机融合用)"""
    if depth_m is None or depth_m <= 0.05 or camera not in CAMERA_MODELS:
        return None
    cam, pos, rot = CAMERA_MODELS[camera]
    u, v = float(centroid_uv[0]), float(centroid_uv[1])
    p_cam = pixel_depth_to_cam(u, v, float(depth_m), cam)
    return (pos + rot @ p_cam).astype(np.float32)


def horizontal_dist_robot(pos_robot: Optional[np.ndarray]) -> Optional[float]:
    if pos_robot is None:
        return None
    return float(np.linalg.norm(pos_robot[:2]))


def annotate_pos_robot(obj: dict, camera: str) -> dict:
    """给单相机 object 补上 pos_robot / dist_to_robot (掩码反投影优先于框中心)"""
    out = dict(obj)
    pos = None
    if obj.get("pos_robot") is not None:
        pos = np.asarray(obj["pos_robot"], dtype=np.float32)
    if pos is None:
        pos = estimate_pos_robot(obj.get("centroid_uv"), obj.get("depth_m"), camera)
        if pos is not None:
            out["pos_robot"] = pos.tolist()
    dist = horizontal_dist_robot(pos)
    if dist is not None:
        out["dist_to_robot"] = dist
    return out


def robot_to_world(
    p_robot, robot_pos: np.ndarray, robot_yaw: float,
) -> np.ndarray:
    """机器人基座系 → 世界系"""
    p = np.asarray(p_robot, dtype=np.float32).reshape(3)
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (robot_pos.astype(np.float32) + rot @ p).astype(np.float32)


def world_to_robot_frame(
    p_world, robot_pos: np.ndarray, robot_yaw: float,
) -> np.ndarray:
    """世界系 → 机器人基座系"""
    pw = np.asarray(p_world, dtype=np.float32).reshape(3)
    rp = np.asarray(robot_pos, dtype=np.float32).reshape(3)
    dx, dy, dz = pw[0] - rp[0], pw[1] - rp[1], pw[2] - rp[2]
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    return np.array([c * dx + s * dy, -s * dx + c * dy, dz], dtype=np.float32)


GRASP_RELIABLE_DEPTH_M = 0.80
WORLD_RELIABLE_DEPTH_M = 2.00

# 地面物体在 robot 系下的合理 z 范围 (蹲下后略放宽)
ROBOT_Z_MIN = -0.78
ROBOT_Z_MAX = 0.28


def filter_plausible_objects(
    objects: list,
    camera: str,
    *,
    ee_near_m: float = 999.0,
) -> list:
    """剔除机器人本体误检、地下/悬空坐标"""
    out = []
    for o in objects:
        pr = o.get("pos_robot")
        if pr is None:
            continue
        try:
            px, py, pz = float(pr[0]), float(pr[1]), float(pr[2])
        except (TypeError, ValueError, IndexError):
            continue
        if pz < ROBOT_Z_MIN or pz > ROBOT_Z_MAX:
            continue
        if float(np.hypot(px, py)) < 0.12 and abs(py) < 0.18:
            continue
        depth = float(o.get("depth_m") or 99.0)
        sm = float(o.get("blob_sat_mean") or 0.0)
        vm = float(o.get("blob_val_mean") or 0.0)
        bbox = o.get("bbox")
        if camera == "head":
            if depth < 1.05 and sm < 46:
                continue
            if depth < 0.80 and sm < 54:
                continue
            if bbox and len(bbox) == 4:
                cx = 0.5 * (bbox[0] + bbox[2])
                cy = 0.5 * (bbox[1] + bbox[3])
                if depth < 0.85 and cy > 200 and 160 < cx < 480 and sm < 58:
                    continue
                if depth < 0.55 and cy > 140 and sm < 62 and 55 < vm < 155:
                    continue
            if ee_near_m < 1.15 and depth < 1.20:
                continue
        if camera == "ee" and depth < 0.35 and sm < 40:
            continue
        out.append(o)
    return out


def compute_dynamic_ee_cam_pos(arm_joints) -> np.ndarray:
    """臂关节相对默认姿 → EE 相机在 robot 系下的偏移 (无 GT, 蹲下/伸臂时修正)"""
    q = np.asarray(arm_joints, dtype=np.float32).reshape(-1)[:6]
    dq = q - DEFAULT_ARM_JOINTS
    delta = np.array([
        0.16 * np.sin(dq[0]) + 0.12 * np.sin(dq[2]) + 0.08 * dq[3] + 0.04 * dq[5],
        0.08 * np.sin(dq[0]) + 0.05 * np.cos(dq[1]) + 0.03 * dq[4],
        0.12 * (np.cos(dq[1]) - 1.0) + 0.09 * np.sin(dq[3]) + 0.06 * dq[4] + 0.05 * dq[2],
    ], dtype=np.float32)
    return (EE_CAM_POS_ROBOT + delta).astype(np.float32)


def compute_dynamic_head_cam_pos(projected_gravity) -> np.ndarray:
    """蹲下/倾身时修正 head 外参 (projected_gravity 偏离 [0,0,-1])"""
    pos = HEAD_CAM_POS_ROBOT.copy()
    if projected_gravity is None:
        return pos
    g = np.asarray(projected_gravity, dtype=np.float32).reshape(3)
    gn = float(np.linalg.norm(g))
    if gn < 1e-4:
        return pos
    g = g / gn
    tilt_xy = float(np.hypot(g[0], g[1]))
    pos[0] += 0.10 * float(g[0]) + 0.04 * float(g[1])
    pos[2] += 0.14 * tilt_xy
    return pos.astype(np.float32)


def pixel_to_robot(
    u: float,
    v: float,
    depth_m: float,
    cam_cfg: dict,
    cam_pos_robot: np.ndarray,
    cam_rot_matrix: np.ndarray,
) -> np.ndarray:
    p_cam = pixel_depth_to_cam(u, v, float(depth_m), cam_cfg)
    return (np.asarray(cam_pos_robot, dtype=np.float32) + cam_rot_matrix @ p_cam).astype(np.float32)


def refresh_ee_object_pose(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_pos_robot: np.ndarray,
) -> dict:
    """每帧用 centroid(导航) + anchor(抓取) + 当前 EE 外参重算 world"""
    out = dict(obj)
    nav_uv = out.get("centroid_uv")
    nav_depth = out.get("depth_m")
    anchor_uv = out.get("grasp_anchor_uv") or nav_uv
    anchor_depth = out.get("grasp_anchor_depth") or nav_depth
    if nav_uv is None or nav_depth is None:
        return out
    try:
        nu, nv = float(nav_uv[0]), float(nav_uv[1])
        nav_d = float(nav_depth)
        au, av = float(anchor_uv[0]), float(anchor_uv[1])
        anchor_d = float(anchor_depth)
    except (TypeError, ValueError, IndexError):
        return out
    if nav_d <= 0.05:
        return out

    pos_r = pixel_to_robot(nu, nv, nav_d, EE_CAM, cam_pos_robot, EE_CAM_ROT_MATRIX)
    offset = np.asarray(
        out.get("grasp_offset_robot") or [0.0, 0.0, -float(GRASP_DEPTH_OFFSET)],
        dtype=np.float32,
    )
    if anchor_d > 0.05:
        anchor_r = pixel_to_robot(au, av, anchor_d, EE_CAM, cam_pos_robot, EE_CAM_ROT_MATRIX)
        grasp_r = anchor_r + offset
    else:
        grasp_r = pos_r + offset
    out["pos_robot"] = pos_r.tolist()
    out["pos_world"] = robot_to_world(pos_r, robot_pos, robot_yaw).tolist()
    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pos_r[:2]))
    out["yaw_rel"] = float(np.arctan2(pos_r[1], pos_r[0]))
    out["nav_depth_m"] = nav_d
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["grasp_reliable"] = nav_d < GRASP_RELIABLE_DEPTH_M
    out["world_reliable"] = nav_d < WORLD_RELIABLE_DEPTH_M
    return out


def refresh_head_object_pose(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_pos_robot: np.ndarray,
) -> dict:
    """head: centroid+depth + 动态外参重算 world"""
    out = dict(obj)
    uv = out.get("centroid_uv")
    depth = out.get("depth_m")
    if uv is None or depth is None:
        return out
    try:
        u, v = float(uv[0]), float(uv[1])
        depth_f = float(depth)
    except (TypeError, ValueError, IndexError):
        return out
    if depth_f <= 0.05:
        return out
    pos_r = pixel_to_robot(u, v, depth_f, HEAD_CAM, cam_pos_robot, HEAD_CAM_ROT_MATRIX)
    out["pos_robot"] = pos_r.tolist()
    out["pos_world"] = robot_to_world(pos_r, robot_pos, robot_yaw).tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pos_r[:2]))
    out["yaw_rel"] = float(np.arctan2(pos_r[1], pos_r[0]))
    out["nav_depth_m"] = depth_f
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["world_reliable"] = depth_f < WORLD_RELIABLE_DEPTH_M
    return out


def refresh_locked_grasp(
    frozen: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """蹲下后冻结 robot 系抓取点，只随 robot pose 更新 world"""
    out = dict(frozen)
    out["grasp_locked"] = True
    pr = out.get("pos_robot")
    gr = out.get("grasp_pos_robot")
    if pr is not None:
        pr_np = np.asarray(pr, dtype=np.float32)
        out["pos_world"] = robot_to_world(pr_np, robot_pos, robot_yaw).tolist()
        out["dist_to_robot"] = float(np.linalg.norm(pr_np[:2]))
        out["yaw_rel"] = float(np.arctan2(pr_np[1], pr_np[0]))
        out["nav_yaw_rel"] = out["yaw_rel"]
    if gr is not None:
        out["grasp_pos_world"] = robot_to_world(
            np.asarray(gr, dtype=np.float32), robot_pos, robot_yaw,
        ).tolist()
    out["grasp_reliable"] = True
    out["world_reliable"] = True
    return out


def apply_gt_camera_pose(
    obj: dict,
    cam_pos_w: np.ndarray,
    cam_rot_w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """
    与 solution_rl._correct_object_with_camera_pose 同公式:
    centroid_uv + depth_m + GT 相机位姿 → pos_world;
    grasp = pos_robot + grasp_offset_robot (robot 系偏移, 与 pos 同步校正).
    """
    from config import GRASP_DEPTH_OFFSET

    out = dict(obj)
    depth_m = out.get("depth_m")
    centroid_uv = out.get("centroid_uv") or out.get("centroid")
    if depth_m is None or centroid_uv is None:
        return out
    try:
        u, v = float(centroid_uv[0]), float(centroid_uv[1])
        depth_m_f = float(depth_m)
    except (TypeError, ValueError, IndexError):
        return out
    if depth_m_f <= 0.01 or depth_m_f > 100.0:
        return out

    x = (u - cx) / fx * depth_m_f
    y = (v - cy) / fy * depth_m_f
    p_cam = np.array([x, y, depth_m_f], dtype=np.float32)
    p_world = np.asarray(cam_pos_w, dtype=np.float32) + np.asarray(cam_rot_w, dtype=np.float32) @ p_cam
    p_robot = world_to_robot_frame(p_world, robot_pos, robot_yaw)

    out["pos_world"] = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
    out["pos_robot"] = [float(p_robot[0]), float(p_robot[1]), float(p_robot[2])]
    out["dist_to_robot"] = float(np.linalg.norm(p_robot[:2]))
    out["yaw_rel"] = float(np.arctan2(p_robot[1], p_robot[0]))
    out["nav_depth_m"] = depth_m_f
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["world_reliable"] = True

    offset = out.get("grasp_offset_robot")
    if offset is None:
        offset = [0.0, 0.0, -float(GRASP_DEPTH_OFFSET)]
    grasp_robot = p_robot + np.asarray(offset, dtype=np.float32)
    out["grasp_offset_robot"] = [float(offset[0]), float(offset[1]), float(offset[2])]
    out["grasp_pos_robot"] = [float(grasp_robot[0]), float(grasp_robot[1]), float(grasp_robot[2])]
    out["grasp_pos_world"] = robot_to_world(grasp_robot, robot_pos, robot_yaw).tolist()
    out["grasp_reliable"] = depth_m_f < GRASP_RELIABLE_DEPTH_M
    out["pose_source"] = "gt_camera"
    return out


def annotate_world_coords(
    obj: dict, robot_pos: np.ndarray, robot_yaw: float,
) -> dict:
    """pos_robot / grasp_pos_robot → 世界坐标 (操作层用 grasp_pos_world)"""
    out = dict(obj)
    pr = obj.get("pos_robot")
    if pr is not None:
        pw = robot_to_world(pr, robot_pos, robot_yaw)
        out["pos_world"] = pw.tolist()
    gr = obj.get("grasp_pos_robot")
    if gr is not None:
        out["grasp_pos_world"] = robot_to_world(gr, robot_pos, robot_yaw).tolist()
    elif out.get("pos_world") is not None:
        gp = np.asarray(out["pos_world"], dtype=np.float32).copy()
        from config import GRASP_DEPTH_OFFSET
        gp[2] -= GRASP_DEPTH_OFFSET
        out["grasp_pos_world"] = gp.tolist()
    return out


def pixel_depth_to_cam(
    u: float, v: float, z: float, cam: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    c = cam or HEAD_CAM
    x = (u - c["cx"]) / c["fx"] * z
    y = (v - c["cy"]) / c["fy"] * z
    return np.array([x, y, z], dtype=np.float32)


def bbox_center_depth(depth: np.ndarray, bbox, pad: int = 0) -> Optional[float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth.shape
    x1, y1 = max(0, x1 + pad), max(0, y1 + pad)
    x2, y2 = min(w - 1, x2 - pad), min(h - 1, y2 - pad)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth[y1:y2, x1:x2]
    valid = patch[(patch > 0.05) & (patch < 49.0)]
    if len(valid) < 3:
        return None
    return float(np.median(valid))


def format_stats_line(stats: Dict[str, float]) -> str:
    return (
        f"valid={stats['valid_ratio']*100:.1f}% "
        f"med={stats['median']:.2f}m "
        f"range=[{stats['min']:.2f},{stats['max']:.2f}]"
    )
