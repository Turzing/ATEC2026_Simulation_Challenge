"""
Task B 感知层

  EE   → 3D 导航坐标 (点云反投影, 与当前臂姿外参一致)
  head → 3D 抓取坐标 (点云顶面)

只改本文件 + rgbd_utils 外参; 不依赖 rgbd_pure_* 旧 pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from config import (
    BIN_CENTER,
    BIN_RADIUS,
    DEFAULT_ARM_JOINTS,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    IMG_H,
    IMG_W,
    PROPRIO_ARM_LEN,
    PROPRIO_ARM_START,
    PROPRIO_BASE_ANG_VEL,
    PROPRIO_BASE_LIN_VEL,
    PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    TOTAL_OBJECTS,
)
from rgbd_utils import (
    _to_numpy,
    bbox_lateral_consistent,
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    compute_ee_cam_rot_matrix,
    depth_stats,
    is_head_edge_phantom,
    is_sky_phantom_bbox,
    parse_ee_rgbd,
    parse_head_rgbd,
    pixel_depth_to_cam,
    pixel_to_robot,
    robot_to_world,
    world_to_robot_frame,
)

PERCEPTION_BUILD = "20260618-3d-dual"

# ── 分工距离 ────────────────────────────────────────────────────────────────
FAR_EE_M = 1.30
GRASP_HEAD_M = 1.20
LOCK_MATCH_M = 0.50
COAST_MAX = 12
HEAD_STABLE_FRAMES = 4
HEAD_GRASP_JUMP_M = 0.08
HEAD_POS_EMA = 0.22

WORLD_Z_LO, WORLD_Z_HI = 0.02, 0.42
CLOSER_THAN_BG_M = 0.014
RELIEF_MAX = 0.35

EE = {
    "relief_min": 0.013,
    "relief_min_near": 0.009,
    "depth_lo": 0.35,
    "depth_hi": 6.5,
    "v_lo": 0.22,
    "v_hi": 0.94,
    "min_area": 12,
    "min_side": 4,
    "min_points": 6,
    "robot_z_max": 0.26,
    "max_depth_std": 0.09,
}
HEAD = {
    "relief_min": 0.008,
    "relief_min_near": 0.006,
    "depth_lo": 0.10,
    "depth_hi": 2.6,
    "v_lo": 0.14,
    "v_hi": 0.96,
    "y2_min_frac": 0.44,
    "min_area": 10,
    "min_side": 4,
    "min_points": 8,
    "robot_z_lo": -0.38,
    "robot_z_max": 0.14,
    "max_depth_std": 0.065,
}


def _classify_shape(bw: int, bh: int, depth_span: float) -> str:
    """三类都是黄色 → 用 2D 形状 + 深度厚度粗分, 不依赖 HSV."""
    asp = max(bw, bh) / max(min(bw, bh), 1)
    if bh >= bw * 1.10 and asp >= 1.12:
        return "mustard_bottle"
    if bw >= bh * 1.22 and asp >= 1.20:
        return "banana"
    if depth_span < 0.06 and asp < 1.55:
        return "sugar_box"
    return "sugar_box"


def _ground_plane(depth: np.ndarray, valid: np.ndarray, k: int = 13) -> np.ndarray:
    d = depth.copy()
    d[~valid] = 0.0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(d.astype(np.float32), cv2.MORPH_OPEN, ker)


def _ground_relief(depth: np.ndarray, ground: np.ndarray) -> np.ndarray:
    return np.clip(ground - depth, 0.0, None)


def _rgb_not_floor_shadow(rgb: np.ndarray) -> np.ndarray:
    """弱 RGB 门控: 去掉阴影/过曝, 不用 HSV 分黄物."""
    gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a, b)
    return (gray > 42) & (gray < 248) & (chroma > 6.0)


def _depth_foreground_mask(
    depth: np.ndarray,
    rgb: np.ndarray,
    roi: np.ndarray,
    valid: np.ndarray,
    *,
    relief_min: float,
    relief_min_near: float,
) -> np.ndarray:
    """Depth relief + 比背景更近 → 前景; RGB 仅滤阴影."""
    ground = _ground_plane(depth, valid)
    relief = _ground_relief(depth, ground)
    near = depth < 1.15
    rmin = relief_min_near if np.any(near & valid) else relief_min

    roi_vals = depth[roi & valid]
    closer = np.zeros_like(depth, dtype=bool)
    if roi_vals.size > 60:
        bg = float(np.percentile(roi_vals, 56))
        closer = valid & (depth < bg - CLOSER_THAN_BG_M)

    depth_fg = valid & (((relief >= rmin) & (relief <= RELIEF_MAX)) | closer)
    rgb_ok = _rgb_not_floor_shadow(rgb)
    return depth_fg & roi & rgb_ok


def _sample_blob_points(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    cam_cfg: dict,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> Optional[np.ndarray]:
    step = 1 if ys.size < 220 else 2
    pts: List[np.ndarray] = []
    for y, x in zip(ys[::step], xs[::step]):
        z = float(depth[int(y), int(x)])
        if z <= 0.05 or z > 49.0 or not np.isfinite(z):
            continue
        pr = pixel_to_robot(float(x), float(y), z, cam_cfg, cam_pos, cam_rot)
        pts.append(pr)
    if len(pts) < 4:
        return None
    return np.stack(pts, axis=0).astype(np.float32)


def _nav_from_points(pts: np.ndarray, *, head: bool) -> Tuple[np.ndarray, float]:
    """点云中位数定位; head 用较浅分位(靠近相机=物体上表面)."""
    if head:
        z_key = float(np.percentile(pts[:, 2], 38))
        near = pts[pts[:, 2] <= z_key + 0.025]
        use = near if near.shape[0] >= 4 else pts
        nav = np.median(use, axis=0).astype(np.float32)
    else:
        y_cut = float(np.percentile(pts[:, 1], 72) if pts.shape[0] > 6 else np.max(pts[:, 1]))
        bot = pts[pts[:, 1] >= y_cut - 0.04] if pts.shape[0] > 6 else pts
        nav = np.median(bot, axis=0).astype(np.float32)
    depth_span = float(np.percentile(pts[:, 2], 88) - np.percentile(pts[:, 2], 12))
    return nav, depth_span


def _world_ok(pw: List[float]) -> bool:
    try:
        return WORLD_Z_LO <= float(pw[2]) <= WORLD_Z_HI
    except (TypeError, ValueError, IndexError):
        return False


def _detect_objects(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera: str,
    cam_cfg: dict,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> List[dict]:
    """Depth relief 主检测 + 点云 3D; RGB 仅滤阴影."""
    cfg = EE if camera == "ee" else HEAD
    is_head = camera == "head"
    h, w = depth.shape[:2]
    rgb = rgb[..., :3].astype(np.uint8)
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h))

    v0, v1 = int(h * cfg["v_lo"]), int(h * cfg["v_hi"])
    valid = (depth >= cfg["depth_lo"]) & (depth <= cfg["depth_hi"])
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[v0:v1, :] = 1

    fg = _depth_foreground_mask(
        depth, rgb, roi.astype(bool), valid,
        relief_min=cfg["relief_min"],
        relief_min_near=cfg["relief_min_near"],
    ).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = 7 if is_head else 5
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    labeled, n = ndimage.label(fg)
    if n <= 0:
        return []

    ground = _ground_plane(depth, valid, k=9 if is_head else 11)
    relief = _ground_relief(depth, ground)

    out: List[dict] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labeled == lab)
        if ys.size < cfg["min_area"]:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if min(bw, bh) < cfg["min_side"]:
            continue
        if is_head and y2 < h * cfg["y2_min_frac"]:
            continue

        dvals = depth[ys, xs]
        dvals = dvals[(dvals > 0.05) & (dvals < 49.0)]
        if dvals.size < cfg["min_points"]:
            continue
        if float(np.std(dvals)) > cfg["max_depth_std"]:
            continue

        pts = _sample_blob_points(ys, xs, depth, cam_cfg, cam_pos, cam_rot)
        if pts is None or pts.shape[0] < cfg["min_points"]:
            continue

        pr, depth_span = _nav_from_points(pts, head=is_head)
        depth_m = float(np.median(dvals))
        pw = robot_to_world(pr, robot_pos, robot_yaw).tolist()
        if not _world_ok(pw):
            continue

        pz = float(pr[2])
        rm = float(np.median(relief[ys, xs]))
        if not is_head:
            if pz > cfg["robot_z_max"]:
                continue
            if rm < cfg["relief_min"] * 0.75 and depth_m < 2.0:
                continue
            if y2 < h * 0.28 and depth_m > 1.0:
                continue
        else:
            if pz < cfg["robot_z_lo"] or pz > cfg["robot_z_max"]:
                continue

        u = float(0.5 * (x1 + x2))
        v = float(y1 + 0.62 * bh)
        cls = _classify_shape(bw, bh, depth_span)
        obj = {
            "bbox": [x1, y1, x2, y2],
            "centroid_uv": [u, 0.5 * (y1 + y2)],
            "nav_anchor_uv": [u, v],
            "nav_anchor_depth": depth_m,
            "depth_m": depth_m,
            "nav_depth_m": depth_m,
            "pos_robot": pr.tolist(),
            "pos_world": pw,
            "dist_to_robot": float(np.linalg.norm(pr[:2])),
            "yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "nav_yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "class": cls,
            "conf": min(0.92, 0.62 + rm * 8.0),
            "relief_med": rm,
            "blob_depth_std": float(np.std(dvals)),
            "nav_point_count": int(pts.shape[0]),
            "pos_from_pointcloud": True,
            "detect_mode": "depth_primary",
            "source_camera": camera,
            "camera": camera,
            "visible": True,
            "world_reliable": depth_m < 2.2,
        }

        if is_head:
            obj["skip_camera_correction"] = True
            obj["role"] = "grasp"
            obj["pointcloud"] = pts
            if is_sky_phantom_bbox(obj) or is_head_edge_phantom(obj):
                continue
            if depth_m < 2.0 and not bbox_lateral_consistent(obj):
                continue
        else:
            obj["skip_camera_correction"] = False
            obj["role"] = "nav"
            obj["nav_detect_priority"] = True
            obj["pos_tolerance_m"] = 0.35

        out.append(obj)

    out.sort(key=lambda o: float(o.get("depth_m") or 999.0))
    return out


def _match_tracks(
    dets: List[dict],
    tracks: Dict[int, dict],
    next_id: int,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> Tuple[List[dict], Dict[int, dict], int]:
    used: set[int] = set()
    matched: List[dict] = []
    new_tracks: Dict[int, dict] = {}

    for det in dets:
        pw = np.asarray(det["pos_world"], dtype=np.float32)
        best_id, best_d = None, LOCK_MATCH_M
        for tid, tr in tracks.items():
            if tid in used:
                continue
            tw = np.asarray(tr["pos_world"], dtype=np.float32)
            d = float(np.linalg.norm(pw[:2] - tw[:2]))
            if d < best_d:
                best_d, best_id = d, tid
        if best_id is None:
            best_id = next_id
            next_id += 1
        used.add(best_id)
        prev = tracks.get(best_id)
        obj = dict(det)
        obj["id"] = int(best_id)
        if prev is not None:
            alpha = HEAD_POS_EMA if det.get("camera") == "head" else 0.35
            sw_prev = np.asarray(prev.get("grasp_smooth_world") or prev["pos_world"], dtype=np.float32)
            pw_now = np.asarray(det["pos_world"], dtype=np.float32)
            jump = float(np.linalg.norm(pw_now[:2] - sw_prev[:2]))
            if det.get("camera") == "head" and jump > HEAD_GRASP_JUMP_M:
                sw = sw_prev.copy()
                obj["pos_jump_rejected"] = True
            else:
                sw = (1.0 - alpha) * sw_prev + alpha * pw_now
            obj["pos_world"] = sw.tolist()
            obj["grasp_smooth_world"] = sw.tolist()
            pr = world_to_robot_frame(sw, robot_pos, robot_yaw)
            obj["pos_robot"] = pr.tolist()
            obj["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
            obj["yaw_rel"] = float(np.arctan2(pr[1], pr[0]))
            obj["nav_yaw_rel"] = obj["yaw_rel"]
            obj["head_stable_count"] = (
                int(prev.get("head_stable_count") or 0) + (0 if obj.get("pos_jump_rejected") else 1)
            )
        else:
            obj["head_stable_count"] = 1
        obj["_miss"] = 0
        new_tracks[best_id] = obj
        matched.append(obj)

    for tid, tr in tracks.items():
        if tid in used:
            continue
        miss = int(tr.get("_miss", 0)) + 1
        if miss > COAST_MAX:
            continue
        coast = dict(tr)
        coast["_miss"] = miss
        coast["track_coast"] = True
        coast["visible"] = False
        sw = np.asarray(coast["pos_world"], dtype=np.float32)
        pr = world_to_robot_frame(sw, robot_pos, robot_yaw)
        coast["pos_robot"] = pr.tolist()
        coast["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
        coast["yaw_rel"] = float(np.arctan2(pr[1], pr[0]))
        new_tracks[tid] = coast
        matched.append(coast)

    matched.sort(key=lambda o: (int(o.get("_miss", 0) > 0), float(o.get("depth_m") or 999.0)))
    return matched, new_tracks, next_id


def _read_pose(obs, dt: float, gt_pos, gt_yaw, state: dict) -> Tuple[np.ndarray, float]:
    if gt_pos is not None and gt_yaw is not None:
        state["pos"] = np.asarray(gt_pos, dtype=np.float32).copy()
        state["yaw"] = float(gt_yaw)
        return state["pos"], state["yaw"]
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        lin = p[PROPRIO_BASE_LIN_VEL]
        ang = p[PROPRIO_BASE_ANG_VEL]
        grav = p[PROPRIO_PROJECTED_GRAVITY]
        c, s = float(np.cos(state["yaw"])), float(np.sin(state["yaw"]))
        state["pos"][0] += (c * lin[0] - s * lin[1]) * dt
        state["pos"][1] += (s * lin[0] + c * lin[1]) * dt
        gy = float(np.arctan2(-grav[0], -grav[1]))
        state["yaw"] = PROPRIO_YAW_FUSION_ALPHA * gy + (1.0 - PROPRIO_YAW_FUSION_ALPHA) * (
            state["yaw"] + ang[2] * dt
        )
    except (KeyError, TypeError, ValueError, IndexError):
        pass
    return state["pos"], state["yaw"]


def _read_arm(obs) -> np.ndarray:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        return p[PROPRIO_ARM_START : PROPRIO_ARM_START + PROPRIO_ARM_LEN].copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return DEFAULT_ARM_JOINTS.copy()


def _read_gravity(obs) -> Optional[np.ndarray]:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        return p[PROPRIO_PROJECTED_GRAVITY].copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _grasp_from_head(obj: dict, robot_pos: np.ndarray, robot_yaw: float) -> dict:
    """head 抓取: 点云最浅 15% 像素 → 顶面 grasp, 比单点 depth 更准."""
    out = dict(obj)
    pts = obj.get("pointcloud")
    grasp_r: np.ndarray

    if pts is not None and isinstance(pts, np.ndarray) and pts.shape[0] >= 6:
        z_hi = float(np.percentile(pts[:, 2], 82))
        top = pts[pts[:, 2] >= z_hi - 0.015]
        use = top if top.shape[0] >= 4 else pts
        grasp_r = np.median(use, axis=0).astype(np.float32)
        grasp_r[2] -= float(GRASP_DEPTH_OFFSET)
    else:
        bbox = out.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        gu = float(0.5 * (x1 + x2))
        gv = float(y1 + 0.52 * (y2 - y1 + 1))
        z = float(out.get("nav_anchor_depth") or out.get("depth_m") or 0.0)
        p_cam = pixel_depth_to_cam(gu, gv, z, HEAD_CAM)
        grasp_r = (HEAD_CAM_POS_ROBOT + HEAD_CAM_ROT_MATRIX @ p_cam).astype(np.float32)
        grasp_r[2] -= float(GRASP_DEPTH_OFFSET)
        out["grasp_anchor_uv"] = [gu, gv]

    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["skip_camera_correction"] = True
    out["role"] = "grasp"
    out["grasp_precision"] = "pointcloud_top"
    stable = int(out.get("head_stable_count") or 0)
    depth = float(out.get("depth_m") or 99.0)
    npts = int(out.get("nav_point_count") or 0)
    out["grasp_reliable"] = (
        depth < 1.05
        and stable >= HEAD_STABLE_FRAMES
        and npts >= HEAD["min_points"]
        and not out.get("track_coast")
        and not out.get("pos_jump_rejected")
    )
    return out


def _dist(obj: Optional[dict]) -> float:
    if not obj:
        return 999.0
    return float(obj.get("dist_to_robot") or obj.get("depth_m") or 999.0)


def _find_by_lock(pool: List[dict], lock_id: Optional[int], lock_world: Optional[List[float]]) -> Optional[dict]:
    if lock_id is not None:
        for o in pool:
            if int(o.get("id", -1)) == int(lock_id) and not o.get("track_coast"):
                return o
    if lock_world is not None:
        lw = np.asarray(lock_world, dtype=np.float32)
        best, bd = None, LOCK_MATCH_M
        for o in pool:
            if o.get("track_coast"):
                continue
            ow = np.asarray(o.get("pos_world"), dtype=np.float32)
            d = float(np.linalg.norm(ow[:2] - lw[:2]))
            if d < bd:
                bd, best = d, o
        return best
    return None


class TaskBPerception:
    """EE 导航 + head 抓取 — 单文件实现, 无旧 pipeline 依赖."""

    def __init__(self) -> None:
        self.frame_count = 0
        self._pose = {"pos": ROBOT_INIT_POS.copy().astype(np.float32), "yaw": float(ROBOT_INIT_YAW)}
        self._ee_tracks: Dict[int, dict] = {}
        self._head_tracks: Dict[int, dict] = {}
        self._next_id = 0
        self._lock_id: Optional[int] = None
        self._lock_world: Optional[List[float]] = None
        self._lock_miss = 0
        print(f"[TaskBPerception] build={PERCEPTION_BUILD} | EE=3d-nav | head=3d-grasp")

    def reset(self) -> None:
        self.frame_count = 0
        self._pose = {"pos": ROBOT_INIT_POS.copy().astype(np.float32), "yaw": float(ROBOT_INIT_YAW)}
        self._ee_tracks.clear()
        self._head_tracks.clear()
        self._next_id = 0
        self._lock_id = None
        self._lock_world = None
        self._lock_miss = 0

    def get_debug(self, camera: str, name: str):
        return None

    def process(
        self,
        obs,
        dt: float = 0.02,
        gt_robot_pos=None,
        gt_robot_yaw=None,
        **_: Any,
    ) -> dict:
        self.frame_count += 1
        rp, ry = _read_pose(obs, dt, gt_robot_pos, gt_robot_yaw, self._pose)
        arm_q = _read_arm(obs)
        grav = _read_gravity(obs)

        h_rgb, h_depth = parse_head_rgbd(obs)
        ee_rgb, ee_depth = parse_ee_rgbd(obs)

        head_cam_pos = compute_dynamic_head_cam_pos(grav)
        # EE 3D 必须与当前帧实际臂姿一致 (画面怎么拍就怎么反投影)
        ee_cam_pos = compute_dynamic_ee_cam_pos(arm_q)
        ee_cam_rot = compute_ee_cam_rot_matrix(arm_q)

        ee_raw: List[dict] = []
        if ee_rgb is not None and ee_depth is not None:
            ee_raw = _detect_objects(
                ee_rgb, ee_depth,
                camera="ee",
                cam_cfg=EE_CAM,
                cam_pos=ee_cam_pos,
                cam_rot=ee_cam_rot,
                robot_pos=rp,
                robot_yaw=ry,
            )

        head_raw = _detect_objects(
            h_rgb, h_depth,
            camera="head",
            cam_cfg=HEAD_CAM,
            cam_pos=head_cam_pos,
            cam_rot=HEAD_CAM_ROT_MATRIX,
            robot_pos=rp,
            robot_yaw=ry,
        )

        ee_objs, self._ee_tracks, self._next_id = _match_tracks(
            ee_raw, self._ee_tracks, self._next_id, rp, ry,
        )
        head_objs, self._head_tracks, self._next_id = _match_tracks(
            head_raw, self._head_tracks, self._next_id, rp, ry,
        )

        ee_live = [o for o in ee_objs if not o.get("track_coast")]
        head_live = [o for o in head_objs if not o.get("track_coast")]

        # ── 锁目标: 优先 EE 最近 live ──
        if self._lock_id is None and ee_live:
            seed = min(ee_live, key=_dist)
            self._lock_id = int(seed["id"])
            self._lock_world = list(seed["pos_world"])
            self._lock_miss = 0
        elif self._lock_id is not None:
            hit = _find_by_lock(ee_live + head_live, self._lock_id, self._lock_world)
            if hit is not None:
                self._lock_world = list(hit["pos_world"])
                self._lock_miss = 0
            else:
                self._lock_miss += 1
                if self._lock_miss > COAST_MAX:
                    self._lock_id = None
                    self._lock_world = None
                    if ee_live:
                        seed = min(ee_live, key=_dist)
                        self._lock_id = int(seed["id"])
                        self._lock_world = list(seed["pos_world"])
                        self._lock_miss = 0

        # ── 导航: 只用 EE ──
        target_nav: Optional[dict] = None
        if self._lock_id is not None:
            target_nav = _find_by_lock(ee_live, self._lock_id, self._lock_world)
        if target_nav is None and ee_live:
            target_nav = min(ee_live, key=_dist)
        elif target_nav is None and self._lock_world is not None and self._lock_miss <= COAST_MAX:
            pr = world_to_robot_frame(np.asarray(self._lock_world, dtype=np.float32), rp, ry)
            target_nav = {
                "id": self._lock_id,
                "pos_world": list(self._lock_world),
                "pos_robot": pr.tolist(),
                "dist_to_robot": float(np.linalg.norm(pr[:2])),
                "depth_m": float(np.linalg.norm(pr[:2])),
                "yaw_rel": float(np.arctan2(pr[1], pr[0])),
                "source_camera": "ee",
                "camera": "ee",
                "role": "nav",
                "nav_coast": True,
                "skip_camera_correction": False,
                "visible": False,
            }

        nav_d = _dist(target_nav)

        # ── 抓取: 只用 head, 近距 + 稳定 ──
        target_grasp: Optional[dict] = None
        head_hit = _find_by_lock(head_live, self._lock_id, self._lock_world)
        want_grasp = (
            nav_d < GRASP_HEAD_M
            and head_hit is not None
            and int(head_hit.get("head_stable_count") or 0) >= HEAD_STABLE_FRAMES
        )
        if want_grasp:
            target_grasp = _grasp_from_head(head_hit, rp, ry)

        phase = "grasp" if want_grasp and target_grasp else "approach"
        nav_stage = "grasp" if want_grasp else ("far_ee" if nav_d > FAR_EE_M else "near_head")

        ee_hint = None
        if target_nav and not target_nav.get("nav_coast"):
            ee_hint = {
                "yaw_rel": float(target_nav.get("yaw_rel") or 0.0),
                "class": target_nav.get("class"),
                "id": target_nav.get("id"),
                "bearing_only": nav_d > FAR_EE_M,
                "depth_m": target_nav.get("depth_m"),
            }

        if os.getenv("ATEC_TASKB_PERC_DEBUG", "0").lower() in ("1", "true", "yes"):
            every = max(1, int(os.getenv("ATEC_TASKB_PERC_DEBUG_EVERY", "25")))
            if self.frame_count % every == 0:
                print(
                    f"[PERC] f={self.frame_count} ee={len(ee_live)}/{len(ee_raw)} "
                    f"head={len(head_live)}/{len(head_raw)} lock={self._lock_id} "
                    f"phase={phase} nav_d={nav_d:.2f} grasp={bool(target_grasp)}"
                )

        return {
            "roles": {"ee": "nav_only", "head": "grasp_only"},
            "nav_stage": nav_stage,
            "nav_authority": "ee",
            "nav_authority_mode": "ee_only",
            "nav_lock_id": self._lock_id,
            "nav_lock_class": None if self._lock_id is None else (target_nav or {}).get("class"),
            "nav_lock_ee_only": True,
            "nav_lock_stable": self._lock_id is not None and self._lock_miss == 0,
            "nav_pos_confidence": None if target_nav is None else 0.75,
            "ee_search_hint": ee_hint,
            "navigation": {"camera": "ee", "target": target_nav, "objects_detailed": ee_live or ee_objs},
            "target_nav": target_nav,
            "objects_nav": ee_live or ee_objs,
            "ee_objects": ee_live or ee_objs,
            "ee_objects_list": ee_objs,
            "grasp": {"camera": "head", "target": target_grasp, "objects_detailed": [target_grasp] if target_grasp else head_live},
            "target_grasp": target_grasp,
            "objects_grasp": [target_grasp] if target_grasp else head_live,
            "head_objects": head_live or head_objs,
            "head_objects_list": head_objs,
            "target": target_grasp if want_grasp and target_grasp else target_nav,
            "objects_remaining": ee_objs + head_objs,
            "active_camera": "head" if want_grasp else "ee",
            "phase": phase,
            "grasp_reliable": bool(target_grasp and target_grasp.get("grasp_reliable")),
            "grasp_locked": bool(target_grasp),
            "head_dist_m": _dist(head_live[0] if head_live else None),
            "ee_dist_m": _dist(ee_live[0] if ee_live else None),
            "head_count_raw": len(head_raw),
            "ee_count_raw": len(ee_raw),
            "depth_stats": depth_stats(h_depth),
            "ee_depth_stats": depth_stats(ee_depth) if ee_depth is not None else {},
            "bin": {
                "center_world": BIN_CENTER.tolist(),
                "radius_m": float(BIN_RADIUS),
                "dist_to_robot": float(np.linalg.norm(rp[:2] - BIN_CENTER[:2])),
            },
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": rp.tolist(), "yaw": float(ry)},
            "perception_build": PERCEPTION_BUILD,
        }
