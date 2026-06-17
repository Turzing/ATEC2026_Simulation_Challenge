"""
Task B 黄物检测 — 重新设计，无遗留 phantom/ROI/类名限制.

流程: 全画面 HSV 黄 → depth 邻域补洞 → 连通域 → 3D 反投影 → 输出 nav 目标.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage

from config import CLASS_NAME_TO_ID, HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX
from rgbd_utils import classify_taskb_simple, pixel_depth_to_cam

HUE_LO, HUE_HI = 8, 55
SAT_MIN, VAL_MIN = 5, 22
MIN_BLOB_PX = 10
MIN_SIDE = 3
DEPTH_MIN, DEPTH_MAX = 0.08, 9.5
MAX_BLOB_SIDE = 280


def _robot_to_world(p_robot: np.ndarray, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    x, y = float(p_robot[0]), float(p_robot[1])
    wx = float(robot_pos[0]) + c * x - s * y
    wy = float(robot_pos[1]) + s * x + c * y
    wz = float(robot_pos[2]) + float(p_robot[2])
    return np.array([wx, wy, wz], dtype=np.float32)


def _uv_to_robot(u: float, v: float, z: float) -> np.ndarray:
    p_cam = pixel_depth_to_cam(u, v, z, HEAD_CAM)
    return (HEAD_CAM_POS_ROBOT + HEAD_CAM_ROT_MATRIX @ p_cam).astype(np.float32)


def _robust_depth(ys: np.ndarray, xs: np.ndarray, depth: np.ndarray) -> Optional[float]:
    d = depth[ys, xs]
    d = d[(d > DEPTH_MIN) & (d < DEPTH_MAX) & np.isfinite(d)]
    if d.size < 2:
        return None
    return float(np.percentile(d, 15))


def _pos_from_blob(
    ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, depth_m: float,
) -> Optional[np.ndarray]:
    pts = []
    step = 1 if len(ys) < 200 else 2
    for y, x in zip(ys[::step], xs[::step]):
        z = float(depth[y, x])
        if z <= DEPTH_MIN or z >= DEPTH_MAX or not np.isfinite(z):
            continue
        pts.append(_uv_to_robot(float(x), float(y), z))
    if len(pts) >= 3:
        return np.median(np.stack(pts, axis=0), axis=0).astype(np.float32)
    y2, x2 = int(ys.max()), int(np.median(xs))
    z = float(depth[y2, x2])
    if z <= DEPTH_MIN or z >= DEPTH_MAX:
        z = depth_m
    pr = _uv_to_robot(float(x2), float(y2), z)
    return pr if pr is not None else None


def _build_yellow_mask(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape[:2]
    roi = np.ones((h, w), dtype=bool)
    roi[int(h * 0.97) :, :] = False

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (
        roi
        & (hue >= HUE_LO) & (hue <= HUE_HI)
        & (sat >= SAT_MIN) & (val >= VAL_MIN)
    )

    valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
    if int(np.sum(valid)) < 12:
        valid = (depth > 0.01) & (depth < DEPTH_MAX) & np.isfinite(depth)
    vk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valid_near = cv2.dilate(valid.astype(np.uint8), vk, iterations=2).astype(bool)
    mask = (yellow & valid_near).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _blob_to_det(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> Optional[dict]:
    if len(ys) < MIN_BLOB_PX:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if min(bw, bh) < MIN_SIDE or max(bw, bh) > MAX_BLOB_SIDE:
        return None

    depth_m = _robust_depth(ys, xs, depth)
    if depth_m is None:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sm = float(np.mean(hsv[ys, xs, 1]))
    vm = float(np.mean(hsv[ys, xs, 2]))
    hm = float(np.mean(hsv[ys, xs, 0]))

    pos_r = _pos_from_blob(ys, xs, depth, depth_m)
    if pos_r is None:
        return None
    if float(pos_r[0]) < 0.04:
        return None

    bbox = [x1, y1, x2, y2]
    aspect = float(bw) / float(max(bh, 1))
    cls, cls_conf = classify_taskb_simple(hm, sm, vm, aspect, 0.08)
    pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
    cx, cy = float(np.median(xs)), float(np.median(ys))

    return {
        "class": cls,
        "class_id": CLASS_NAME_TO_ID.get(cls, 1),
        "conf": float(min(0.92, 0.55 + cls_conf * 0.35)),
        "class_conf": cls_conf,
        "bbox": bbox,
        "centroid": (cx, cy),
        "centroid_uv": [cx, cy],
        "nav_anchor_uv": [cx, y2],
        "depth_m": depth_m,
        "nav_depth_m": depth_m,
        "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
        "yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "nav_yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "pos_robot": pos_r.tolist(),
        "pos_world": pos_w.tolist(),
        "blob_sat_mean": sm,
        "blob_val_mean": vm,
        "blob_hue_mean": hm,
        "source": "yellow_detect",
        "camera": "head",
        "role": "nav",
        "head_far_fallback": True,
        "pipeline_tier": 1,
        "gt_correctable": True,
        "class_agnostic": True,
        "world_reliable": depth_m < 6.0,
    }


def detect_head_yellow(
    rgb: np.ndarray,
    depth: np.ndarray,
    robot_pos,
    robot_yaw: float,
) -> List[dict]:
    """全画面黄物检测，返回 0..N 个目标 (按距离排序)."""
    rp = np.asarray(robot_pos, dtype=np.float32)
    ry = float(robot_yaw)
    mask = _build_yellow_mask(rgb, depth)
    labeled, n = ndimage.label(mask > 0)
    if n <= 0:
        return []

    dets: List[dict] = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        det = _blob_to_det(ys, xs, depth, rgb, rp, ry)
        if det is not None:
            dets.append(det)
    dets.sort(key=lambda d: float(d.get("depth_m") or 99.0))
    return dets
