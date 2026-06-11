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
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
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
