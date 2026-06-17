"""
Task B 黄物检测 v16 — 两步感知:
  1) ee:   站立远距黄物 3D 导航 (head 远距看不见)
  2) ee:   趴下后近距黄物 + 顶边抓取点
  head 仅作近距补充，不作远距主导
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from config import (
    CLASS_NAME_TO_ID,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
)
from rgbd_utils import classify_taskb_simple, is_head_sky_phantom, pixel_depth_to_cam

HUE_LO, HUE_HI = 6, 58
SAT_MIN, VAL_MIN = 4, 20
MIN_BLOB_PX_HEAD = 10
MIN_BLOB_PX_EE = 24
MIN_SIDE = 3
DEPTH_MIN, DEPTH_MAX = 0.08, 9.5
EE_DEPTH_MIN, EE_DEPTH_MAX = 0.12, 1.65
EE_NAV_DEPTH_MIN, EE_NAV_DEPTH_MAX = 0.85, 5.5
MIN_BLOB_PX_EE_NAV = 8
MAX_BLOB_SIDE_HEAD = 240
MAX_BLOB_SIDE_EE = 320
MAX_BLOB_SIDE_EE_NAV = 420
HEAD_MAX_ROBOT_Z = -0.05
EE_Z_LO, EE_Z_HI = -0.92, 0.18
EE_NAV_Z_LO, EE_NAV_Z_HI = -1.15, 0.45


def _robot_to_world(p_robot: np.ndarray, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    x, y = float(p_robot[0]), float(p_robot[1])
    wx = float(robot_pos[0]) + c * x - s * y
    wy = float(robot_pos[1]) + s * x + c * y
    wz = float(robot_pos[2]) + float(p_robot[2])
    return np.array([wx, wy, wz], dtype=np.float32)


def _uv_to_robot(
    u: float, v: float, z: float,
    cam,
    cam_pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    p_cam = pixel_depth_to_cam(u, v, z, cam)
    return (cam_pos + rot @ p_cam).astype(np.float32)


def _yellow_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    roi: np.ndarray,
    *,
    d_near: float = DEPTH_MIN,
    d_far: float = DEPTH_MAX,
) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = (
        roi
        & (hue >= HUE_LO) & (hue <= HUE_HI)
        & (sat >= SAT_MIN) & (val >= VAL_MIN)
    )
    valid = (depth > d_near) & (depth < d_far) & np.isfinite(depth)
    vk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    valid_near = cv2.dilate(valid.astype(np.uint8), vk, iterations=2).astype(bool)
    mask = (yellow & valid_near).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _robust_depth(ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, *, bottom_frac: float = 0.72) -> Optional[float]:
    y_cut = float(np.percentile(ys.astype(np.float32), bottom_frac * 100.0))
    bot = ys >= y_cut
    d = depth[ys[bot], xs[bot]] if int(np.sum(bot)) >= 3 else depth[ys, xs]
    d = d[(d > DEPTH_MIN) & (d < DEPTH_MAX) & np.isfinite(d)]
    if d.size < 2:
        return None
    return float(np.percentile(d, 18))


def _points_from_blob(
    ys: np.ndarray, xs: np.ndarray, depth: np.ndarray,
    cam, cam_pos: np.ndarray, rot: np.ndarray,
    *, bottom_frac: float = 0.75,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    y_cut = float(np.percentile(ys.astype(np.float32), bottom_frac * 100.0))
    bot_y, bot_x = ys[ys >= y_cut], xs[ys >= y_cut]
    if bot_y.size < 2:
        bot_y, bot_x = ys, xs
    pts = []
    step = 1 if len(bot_y) < 200 else 2
    for y, x in zip(bot_y[::step], bot_x[::step]):
        z = float(depth[y, x])
        if z <= DEPTH_MIN or z >= DEPTH_MAX or not np.isfinite(z):
            continue
        pts.append(_uv_to_robot(float(x), float(y), z, cam, cam_pos, rot))
    if len(pts) >= 3:
        stack = np.stack(pts, axis=0).astype(np.float32)
        return np.median(stack, axis=0), stack, len(pts)
    y2, x2 = int(bot_y.max()), int(np.median(bot_x))
    z = float(depth[y2, x2])
    if z <= DEPTH_MIN or z >= DEPTH_MAX:
        return None, None, 0
    pr = _uv_to_robot(float(x2), float(y2), z, cam, cam_pos, rot)
    return pr, None, 1


def _head_rois(h: int, w: int) -> List[np.ndarray]:
    """近距地面 + 中远距地面 (不含天空带)."""
    ground = np.zeros((h, w), dtype=bool)
    ground[int(h * 0.38) : int(h * 0.97), int(w * 0.01) : int(w * 0.99)] = True
    mid = np.zeros((h, w), dtype=bool)
    mid[int(h * 0.28) : int(h * 0.72), int(w * 0.01) : int(w * 0.99)] = True
    return [ground, mid]


def _head_blob_to_det(
    ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, rgb: np.ndarray,
    robot_pos: np.ndarray, robot_yaw: float, img_h: int, img_w: int,
    *, min_y2_frac: float,
) -> Optional[dict]:
    if len(ys) < MIN_BLOB_PX_HEAD:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if min(bw, bh) < MIN_SIDE or max(bw, bh) > MAX_BLOB_SIDE_HEAD:
        return None
    if y2 < img_h * min_y2_frac:
        return None

    depth_m = _robust_depth(ys, xs, depth)
    if depth_m is None:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sm = float(np.mean(hsv[ys, xs, 1]))
    vm = float(np.mean(hsv[ys, xs, 2]))
    hm = float(np.mean(hsv[ys, xs, 0]))

    pos_r, _, npts = _points_from_blob(
        ys, xs, depth, HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX,
    )
    if pos_r is None:
        return None
    if float(pos_r[0]) < 0.05 or float(pos_r[2]) > HEAD_MAX_ROBOT_Z:
        return None

    bbox = [x1, y1, x2, y2]
    aspect = float(bw) / float(max(bh, 1))
    cls, cls_conf = classify_taskb_simple(hm, sm, vm, aspect, 0.08)
    pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
    cx, cy = float(np.median(xs)), float(np.median(ys))

    det = {
        "class": cls,
        "class_id": CLASS_NAME_TO_ID.get(cls, 1),
        "conf": float(min(0.92, 0.55 + cls_conf * 0.35)),
        "class_conf": cls_conf,
        "bbox": bbox,
        "centroid": (cx, cy),
        "centroid_uv": [cx, cy],
        "nav_anchor_uv": [cx, y2],
        "nav_anchor_depth": depth_m,
        "depth_m": depth_m,
        "nav_depth_m": depth_m,
        "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
        "yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "nav_yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "pos_robot": pos_r.tolist(),
        "pos_world": pos_w.tolist(),
        "pos_from_pointcloud": npts >= 6,
        "nav_point_count": npts,
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
    if is_head_sky_phantom(det, img_h=img_h, img_w=img_w):
        return None
    return det


def _ee_blob_to_det(
    ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, rgb: np.ndarray,
    robot_pos: np.ndarray, robot_yaw: float,
    cam_pos: np.ndarray, img_h: int, img_w: int,
) -> Optional[dict]:
    if len(ys) < MIN_BLOB_PX_EE:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if min(bw, bh) < 5 or max(bw, bh) > MAX_BLOB_SIDE_EE:
        return None
    if y2 < img_h * 0.22:
        return None

    depth_m = _robust_depth(ys, xs, depth, bottom_frac=0.68)
    if depth_m is None or depth_m > EE_DEPTH_MAX or depth_m < EE_DEPTH_MIN:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sm = float(np.mean(hsv[ys, xs, 1]))
    vm = float(np.mean(hsv[ys, xs, 2]))
    hm = float(np.mean(hsv[ys, xs, 0]))

    pos_r, pts_stack, npts = _points_from_blob(
        ys, xs, depth, EE_CAM, cam_pos, EE_CAM_ROT_MATRIX, bottom_frac=0.70,
    )
    if pos_r is None:
        return None
    pz = float(pos_r[2])
    if float(pos_r[0]) < -0.18 or pz < EE_Z_LO or pz > EE_Z_HI:
        return None

    top_y = int(ys.min())
    top_x = int(np.median(xs[ys <= top_y + 2])) if int(np.sum(ys <= top_y + 2)) > 0 else int(np.median(xs))
    top_z = float(depth[top_y, top_x])
    if top_z <= EE_DEPTH_MIN or top_z >= EE_DEPTH_MAX:
        top_z = depth_m

    if pts_stack is not None and len(pts_stack) >= 4:
        z_top = float(np.percentile(pts_stack[:, 2], 82))
        grasp_r = pos_r.copy()
        grasp_r[2] = z_top - float(GRASP_DEPTH_OFFSET)
    else:
        top_r = _uv_to_robot(float(top_x), float(top_y), top_z, EE_CAM, cam_pos, EE_CAM_ROT_MATRIX)
        grasp_r = top_r.copy()
        grasp_r[2] -= float(GRASP_DEPTH_OFFSET)

    pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
    grasp_w = _robot_to_world(grasp_r, robot_pos, robot_yaw)
    cx, cy = float(np.median(xs)), float(np.median(ys))
    aspect = float(bw) / float(max(bh, 1))
    cls, cls_conf = classify_taskb_simple(hm, sm, vm, aspect, 0.12)

    return {
        "class": cls,
        "class_id": CLASS_NAME_TO_ID.get(cls, 1),
        "conf": float(min(0.94, 0.60 + cls_conf * 0.30)),
        "class_conf": cls_conf,
        "bbox": [x1, y1, x2, y2],
        "centroid": (cx, cy),
        "centroid_uv": [cx, cy],
        "nav_anchor_uv": [cx, y2],
        "nav_anchor_depth": depth_m,
        "grasp_anchor_uv": [float(top_x), float(top_y)],
        "grasp_anchor_depth": top_z,
        "grasp_offset_robot": [0.0, 0.0, float(grasp_r[2] - pos_r[2])],
        "depth_m": depth_m,
        "nav_depth_m": depth_m,
        "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
        "yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "pos_robot": pos_r.tolist(),
        "pos_world": pos_w.tolist(),
        "grasp_pos_robot": grasp_r.tolist(),
        "grasp_pos_world": grasp_w.tolist(),
        "blob_sat_mean": sm,
        "blob_val_mean": vm,
        "blob_hue_mean": hm,
        "source": "ee_yellow_detect",
        "camera": "ee",
        "role": "nav_grasp",
        "static_snapshot": True,
        "grasp_reliable": True,
        "world_reliable": True,
        "class_agnostic": True,
        "nav_point_count": npts,
    }


def _dedupe_dets(dets: List[dict]) -> List[dict]:
    if len(dets) < 2:
        return dets
    kept: List[dict] = []
    for d in sorted(dets, key=lambda o: -float((o.get("bbox") or [0, 0, 0, 0])[3])):
        bb = d.get("bbox")
        if not bb:
            kept.append(d)
            continue
        dup = False
        for k in kept:
            kb = k.get("bbox")
            if not kb:
                continue
            ix1 = max(bb[0], kb[0])
            iy1 = max(bb[1], kb[1])
            ix2 = min(bb[2], kb[2])
            iy2 = min(bb[3], kb[3])
            inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
            if inter <= 0:
                continue
            area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
            karea = (kb[2] - kb[0] + 1) * (kb[3] - kb[1] + 1)
            if inter / max(min(area, karea), 1) > 0.45:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def detect_head_yellow(
    rgb: np.ndarray,
    depth: np.ndarray,
    robot_pos,
    robot_yaw: float,
) -> List[dict]:
    """head 粗导航: 地面黄物, 优先画面下方."""
    rp = np.asarray(robot_pos, dtype=np.float32)
    ry = float(robot_yaw)
    h, w = depth.shape[:2]
    rois = _head_rois(h, w)
    min_y2 = [0.36, 0.28]
    dets: List[dict] = []
    for roi, y2f in zip(rois, min_y2):
        mask = _yellow_mask(rgb, depth, roi)
        labeled, n = ndimage.label(mask > 0)
        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            det = _head_blob_to_det(ys, xs, depth, rgb, rp, ry, h, w, min_y2_frac=y2f)
            if det is not None:
                dets.append(det)
    dets = _dedupe_dets(dets)
    dets.sort(
        key=lambda d: (
            -float((d.get("bbox") or [0, 0, 0, 0])[3]),
            float(d.get("pos_robot", [0, 0, 0])[2]),
            float(d.get("depth_m") or 99.0),
        ),
    )
    return dets


def _ee_blob_to_nav_det(
    ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, rgb: np.ndarray,
    robot_pos: np.ndarray, robot_yaw: float,
    cam_pos: np.ndarray, img_h: int, img_w: int,
) -> Optional[dict]:
    """EE 站立远距: 黄物 3D 导航点 (无 grasp)."""
    if len(ys) < MIN_BLOB_PX_EE_NAV:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if min(bw, bh) < 4 or max(bw, bh) > MAX_BLOB_SIDE_EE_NAV:
        return None
    if y2 < img_h * 0.14:
        return None

    depth_m = _robust_depth(ys, xs, depth, bottom_frac=0.70)
    if depth_m is None or depth_m > EE_NAV_DEPTH_MAX or depth_m < EE_NAV_DEPTH_MIN:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sm = float(np.mean(hsv[ys, xs, 1]))
    vm = float(np.mean(hsv[ys, xs, 2]))
    hm = float(np.mean(hsv[ys, xs, 0]))

    pos_r, _, npts = _points_from_blob(
        ys, xs, depth, EE_CAM, cam_pos, EE_CAM_ROT_MATRIX, bottom_frac=0.72,
    )
    if pos_r is None:
        return None
    pz = float(pos_r[2])
    if float(pos_r[0]) < -0.22 or pz < EE_NAV_Z_LO or pz > EE_NAV_Z_HI:
        return None

    pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
    cx, cy = float(np.median(xs)), float(np.median(ys))
    aspect = float(bw) / float(max(bh, 1))
    cls, cls_conf = classify_taskb_simple(hm, sm, vm, aspect, 0.12)

    return {
        "class": cls,
        "class_id": CLASS_NAME_TO_ID.get(cls, 1),
        "conf": float(min(0.90, 0.52 + cls_conf * 0.32)),
        "class_conf": cls_conf,
        "bbox": [x1, y1, x2, y2],
        "centroid": (cx, cy),
        "centroid_uv": [cx, cy],
        "nav_anchor_uv": [cx, y2],
        "nav_anchor_depth": depth_m,
        "depth_m": depth_m,
        "nav_depth_m": depth_m,
        "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
        "yaw_rel": float(np.arctan2(pos_r[1], pos_r[0])),
        "pos_robot": pos_r.tolist(),
        "pos_world": pos_w.tolist(),
        "pos_from_pointcloud": npts >= 5,
        "nav_point_count": npts,
        "blob_sat_mean": sm,
        "blob_val_mean": vm,
        "blob_hue_mean": hm,
        "source": "ee_yellow_nav",
        "camera": "ee",
        "role": "nav",
        "pipeline_tier": 1,
        "class_agnostic": True,
        "world_reliable": depth_m < 4.8,
    }


def detect_ee_yellow_nav(
    rgb: np.ndarray,
    depth: np.ndarray,
    robot_pos,
    robot_yaw: float,
    cam_pos: Optional[np.ndarray] = None,
) -> List[dict]:
    """EE 站立远距: 黄物 3D 导航 (head 远距不可见时用)."""
    rp = np.asarray(robot_pos, dtype=np.float32)
    ry = float(robot_yaw)
    cp = np.asarray(cam_pos if cam_pos is not None else EE_CAM_POS_ROBOT, dtype=np.float32)
    h, w = depth.shape[:2]
    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * 0.08) : int(h * 0.97), int(w * 0.03) : int(w * 0.97)] = True
    mask = _yellow_mask(rgb, depth, roi, d_near=EE_NAV_DEPTH_MIN, d_far=EE_NAV_DEPTH_MAX)
    labeled, n = ndimage.label(mask > 0)
    dets: List[dict] = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        det = _ee_blob_to_nav_det(ys, xs, depth, rgb, rp, ry, cp, h, w)
        if det is not None:
            dets.append(det)
    dets = _dedupe_dets(dets)
    dets.sort(key=lambda d: float(d.get("depth_m") or 99.0))
    return dets


def detect_ee_yellow(
    rgb: np.ndarray,
    depth: np.ndarray,
    robot_pos,
    robot_yaw: float,
    cam_pos: Optional[np.ndarray] = None,
) -> List[dict]:
    """EE 趴下后精抓取: 近距黄物 + 顶边 grasp 点."""
    rp = np.asarray(robot_pos, dtype=np.float32)
    ry = float(robot_yaw)
    cp = np.asarray(cam_pos if cam_pos is not None else EE_CAM_POS_ROBOT, dtype=np.float32)
    h, w = depth.shape[:2]
    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * 0.06) : int(h * 0.96), int(w * 0.04) : int(w * 0.96)] = True
    mask = _yellow_mask(rgb, depth, roi, d_near=EE_DEPTH_MIN, d_far=EE_DEPTH_MAX)
    labeled, n = ndimage.label(mask > 0)
    dets: List[dict] = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        det = _ee_blob_to_det(ys, xs, depth, rgb, rp, ry, cp, h, w)
        if det is not None:
            dets.append(det)
    dets = _dedupe_dets(dets)
    dets.sort(key=lambda d: float(d.get("depth_m") or 99.0))
    return dets
