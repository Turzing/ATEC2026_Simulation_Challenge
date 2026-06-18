"""
Task B RGBD 感知 v5

相机分工 (与官方一致):
  EE   = eye-in-hand 广角 → 远距导航 (>1.4m), 输出 anchor 供 GT 相机校正
  head = eye-to-hand 俯视 → 近距导航 + 抓取 (<1.4m)

检测: 自适应 HSV + depth, 参考 rgbd_pure_pipeline 阈值
稳定: track coast (丢检仍输出) + EMA 平滑
精度: head 目标 skip_camera_correction, 避免 solution_rl HEAD CORR 二次拉偏
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
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
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
    compute_dynamic_ee_cam_pos,
    depth_stats,
    is_ee_floor_gripper_phantom,
    is_ee_sky_blob,
    parse_ee_rgbd,
    parse_head_rgbd,
    pixel_depth_to_cam,
    pixel_to_robot,
    refresh_ee_object_pose,
    robot_to_world,
    stabilize_ee_nav_pose,
    world_to_robot_frame,
    _to_numpy,
)

PERCEPTION_BUILD = "20260618-taskb-rgbd-v5"
CLASS_NAME = "mustard_bottle"

FAR_NAV_M = 1.40
GRASP_DIST_M = 1.25
POS_EMA_ALPHA = 0.38
JUMP_REJECT_FAR_M = 0.42
JUMP_REJECT_NEAR_M = 0.16
NEAR_M = 1.35
COAST_MAX_MISS = 36
MATCH_RADIUS_M = 0.62
LOCK_MATCH_M = 0.72
ROBOT_Z_MIN, ROBOT_Z_MAX = -0.82, 0.22

CAM_DETECT = {
    "head": {
        "hue_lo": 12, "hue_hi": 50,
        "sat_min": 16, "sat_relax": 16, "val_min": 26, "val_relax": 14,
        "v_min": 0.05, "v_max": 0.96,
        "depth_min": 0.10, "depth_max": 4.8,
        "min_blob": 8, "min_side": 3,
    },
    "ee": {
        "hue_lo": 12, "hue_hi": 50,
        "sat_min": 12, "sat_relax": 24, "val_min": 22, "val_relax": 20,
        "v_min": 0.02, "v_max": 0.98,
        "depth_min": 0.22, "depth_max": 8.5,
        "min_blob": 6, "min_side": 3,
    },
}


def _read_arm_joints(obs) -> Optional[np.ndarray]:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        return p[PROPRIO_ARM_START : PROPRIO_ARM_START + PROPRIO_ARM_LEN].copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _cam_static(camera: str):
    if camera == "head":
        return HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX
    return EE_CAM, EE_CAM_POS_ROBOT, EE_CAM_ROT_MATRIX


def _robust_depth(depth, ys, xs, *, anchor_v: float, band_frac: float = 0.20) -> Optional[float]:
    y1, y2 = int(ys.min()), int(ys.max())
    band = max(3, int((y2 - y1 + 1) * band_frac))
    v_lo = max(y1, int(anchor_v) - band)
    sel = (ys >= v_lo) & (ys <= int(anchor_v))
    if not np.any(sel):
        sel = ys >= max(y1, y2 - band)
    dvals = depth[ys[sel], xs[sel]]
    dvals = dvals[(dvals > 0.05) & (dvals < 9.5)]
    if dvals.size < 3:
        dvals = depth[ys, xs]
        dvals = dvals[(dvals > 0.05) & (dvals < 9.5)]
    if dvals.size < 3:
        return None
    return float(np.median(dvals))


def _head_nav_anchor_v(y1: int, y2: int) -> float:
    """俯视 head: 略低于 bbox 中心取深度, 避免 y2 贴地采到地板."""
    return float(y1 + 0.68 * (y2 - y1 + 1))


def _apply_correction_hints(obj: dict) -> dict:
    """head 外参已与仿真一致时, 跳过 solution_rl 的 HEAD GT 重投影."""
    out = dict(obj)
    cam = str(out.get("source_camera") or out.get("camera") or "")
    depth = float(out.get("depth_m") or out.get("nav_depth_m") or 99.0)
    if "head" in cam and out.get("world_reliable") and depth < 2.5:
        out["skip_camera_correction"] = True
        out["correction_policy"] = "perception_trusted"
    return out


def _plausible_obj(obj: dict, camera: str) -> bool:
    pr = obj.get("pos_robot")
    bbox = obj.get("bbox")
    if pr is None or not bbox:
        return False
    pz = float(pr[2])
    if pz < ROBOT_Z_MIN or pz > ROBOT_Z_MAX:
        return False
    horiz = float(np.hypot(float(pr[0]), float(pr[1])))
    if horiz < 0.07 and abs(float(pr[1])) < 0.10:
        return False
    x1, y1, x2, y2 = bbox
    cy = 0.5 * (y1 + y2)
    depth = float(obj.get("depth_m") or 99.0)
    if camera == "head":
        # 俯视: 远处物体在画面上方，不能按 y2 下限过滤
        if cy < IMG_H * 0.06:
            return False
    else:
        if is_ee_floor_gripper_phantom(obj):
            return False
        if is_ee_sky_blob(obj):
            return False
    return True


def _detect_yellow_rgbd(rgb: np.ndarray, depth: np.ndarray, camera: str) -> List[dict]:
    cfg = CAM_DETECT[camera]
    h, w = depth.shape[:2]
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb[..., :3], (w, h), interpolation=cv2.INTER_LINEAR)

    v0 = max(0, int(h * cfg["v_min"]))
    v1 = min(h, max(v0 + 2, int(h * cfg["v_max"])))
    dmin, dmax = float(cfg["depth_min"]), float(cfg["depth_max"])

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    valid = depth[(depth > dmin) & (depth < dmax)]
    g_sat = float(np.median(hsv[:, :, 1][(depth > dmin) & (depth < dmax)])) if valid.size else 40.0
    g_val = float(np.median(hsv[:, :, 2][(depth > dmin) & (depth < dmax)])) if valid.size else 80.0
    sat_thr = max(cfg["sat_min"], g_sat - cfg["sat_relax"])
    val_thr = max(cfg["val_min"], g_val - cfg["val_relax"])

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[v0:v1, :] = 1
    mask = (
        roi
        & (hsv[:, :, 0] >= cfg["hue_lo"]) & (hsv[:, :, 0] <= cfg["hue_hi"])
        & (hsv[:, :, 1] >= sat_thr)
        & (hsv[:, :, 2] >= val_thr)
        & (depth >= dmin) & (depth <= dmax)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = 7 if camera == "head" else 5
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    labeled, n = ndimage.label(mask)
    if n <= 0:
        return []

    cam_cfg, cam_pos, cam_rot = _cam_static(camera)
    out: List[dict] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labeled == lab)
        if ys.size < cfg["min_blob"]:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if (x2 - x1 + 1) < cfg["min_side"] or (y2 - y1 + 1) < cfg["min_side"]:
            continue

        nav_u = float(0.5 * (x1 + x2))
        nav_v = _head_nav_anchor_v(y1, y2) if camera == "head" else float(y2)
        nav_depth = _robust_depth(depth, ys, xs, anchor_v=nav_v)
        if nav_depth is None:
            continue

        pr = pixel_to_robot(nav_u, nav_v, nav_depth, cam_cfg, cam_pos, cam_rot)

        obj = {
            "bbox": [x1, y1, x2, y2],
            "centroid_uv": [nav_u, 0.5 * (y1 + y2)],
            "nav_anchor_uv": [nav_u, nav_v],
            "nav_anchor_depth": nav_depth,
            "grasp_anchor_uv": [nav_u, nav_v],
            "grasp_anchor_depth": nav_depth,
            "depth_m": nav_depth,
            "nav_depth_m": nav_depth,
            "pos_robot": pr.tolist(),
            "dist_to_robot": float(np.linalg.norm(pr[:2])),
            "horiz_dist_m": float(np.linalg.norm(pr[:2])),
            "yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "nav_yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "source_camera": camera,
            "camera": camera,
            "source": "rgbd_hsv",
            "class": CLASS_NAME,
            "conf": 0.80,
            "world_reliable": nav_depth < 2.5,
            "pos_confidence": 0.84 if nav_depth < 1.6 else 0.70,
            "visible": True,
            "role": "nav_grasp" if camera == "head" else "nav",
            "blob_sat_mean": float(np.median(hsv[:, :, 1][ys, xs])),
            "blob_val_mean": float(np.median(hsv[:, :, 2][ys, xs])),
        }
        if _plausible_obj(obj, camera):
            out.append(_apply_correction_hints(obj))

    out.sort(key=lambda o: float(o.get("depth_m") or 999.0))
    return out


def _enrich_world(obj: dict, robot_pos: np.ndarray, robot_yaw: float) -> dict:
    out = dict(obj)
    pr = np.asarray(out["pos_robot"], dtype=np.float32)
    out["pos_world"] = robot_to_world(pr, robot_pos, robot_yaw).tolist()
    return out


def _smooth_with_prev(obj: dict, prev: Optional[dict], robot_pos: np.ndarray, robot_yaw: float) -> dict:
    out = dict(obj)
    pw = np.asarray(out["pos_world"], dtype=np.float32)
    if prev is None:
        out["pos_smooth_world"] = pw.tolist()
        out["_miss"] = 0
        return out

    prev_sw = np.asarray(prev.get("pos_smooth_world") or prev["pos_world"], dtype=np.float32)
    jump = float(np.linalg.norm(pw[:2] - prev_sw[:2]))
    dist = float(out.get("dist_to_robot") or 99.0)
    limit = JUMP_REJECT_NEAR_M if dist < NEAR_M else JUMP_REJECT_FAR_M
    if jump > limit and not out.get("track_coast"):
        sw = prev_sw.copy()
        out["pos_jump_rejected"] = True
    else:
        a = POS_EMA_ALPHA
        sw = (1.0 - a) * prev_sw + a * pw
        sw[2] = (1.0 - a) * prev_sw[2] + a * pw[2]

    out["pos_smooth_world"] = sw.tolist()
    out["pos_world"] = sw.tolist()
    pr = world_to_robot_frame(sw, robot_pos, robot_yaw)
    out["pos_robot"] = pr.tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
    out["yaw_rel"] = float(np.arctan2(float(pr[1]), float(pr[0])))
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["_miss"] = 0
    return out


def _match_tracks(
    dets: List[dict],
    tracks: Dict[int, dict],
    robot_pos: np.ndarray,
    robot_yaw: float,
    next_id: int,
) -> Tuple[List[dict], Dict[int, dict], int]:
    enriched = [_enrich_world(d, robot_pos, robot_yaw) for d in dets]
    used: set[int] = set()
    matched: List[dict] = []
    new_tracks: Dict[int, dict] = {}

    for obj in enriched:
        pw = np.asarray(obj["pos_world"], dtype=np.float32)
        best_id, best_d = None, LOCK_MATCH_M
        for tid, tr in tracks.items():
            if tid in used:
                continue
            tw = np.asarray(tr.get("pos_smooth_world") or tr["pos_world"], dtype=np.float32)
            if float(np.linalg.norm(pw[:2] - tw[:2])) < best_d:
                best_d = float(np.linalg.norm(pw[:2] - tw[:2]))
                best_id = tid
        if best_id is None:
            best_id = next_id
            next_id += 1
        used.add(best_id)
        prev = tracks.get(best_id)
        obj["id"] = int(best_id)
        smoothed = _apply_correction_hints(_smooth_with_prev(obj, prev, robot_pos, robot_yaw))
        if prev is not None:
            for k in ("nav_anchor_uv", "nav_anchor_depth", "grasp_anchor_uv", "grasp_anchor_depth"):
                if k in prev and smoothed.get(k) is None:
                    smoothed[k] = prev[k]
        new_tracks[best_id] = smoothed
        matched.append(smoothed)

    for tid, tr in tracks.items():
        if tid in used:
            continue
        miss = int(tr.get("_miss", 0)) + 1
        if miss > COAST_MAX_MISS:
            continue
        coast = dict(tr)
        coast["_miss"] = miss
        coast["track_coast"] = True
        coast["visible"] = False
        sw = np.asarray(coast.get("pos_smooth_world") or coast["pos_world"], dtype=np.float32)
        pr = world_to_robot_frame(sw, robot_pos, robot_yaw)
        coast["pos_robot"] = pr.tolist()
        coast["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
        coast["yaw_rel"] = float(np.arctan2(float(pr[1]), float(pr[0])))
        coast["nav_yaw_rel"] = coast["yaw_rel"]
        new_tracks[tid] = coast
        matched.append(coast)

    matched.sort(key=lambda o: (int(o.get("_miss", 0) > 0), float(o.get("depth_m") or 999.0)))
    return matched, new_tracks, next_id


def _finalize_ee(obj: dict, robot_pos: np.ndarray, robot_yaw: float, arm_joints) -> dict:
    out = stabilize_ee_nav_pose(dict(obj))
    cam_pos = compute_dynamic_ee_cam_pos(arm_joints if arm_joints is not None else DEFAULT_ARM_JOINTS)
    out = refresh_ee_object_pose(out, robot_pos, robot_yaw, cam_pos)
    out["source_camera"] = "ee"
    out["camera"] = "ee"
    out["role"] = "nav"
    return out


def _grasp_from_head(head_obj: dict, robot_pos: np.ndarray, robot_yaw: float) -> dict:
    out = dict(head_obj)
    bbox = out.get("bbox") or [0, 0, 0, 0]
    x1, _, x2, y2 = bbox
    gu = float(out.get("grasp_anchor_uv", [0.5 * (x1 + x2), y2])[0])
    gv = float(out.get("grasp_anchor_uv", [gu, y2])[1])
    gdepth = float(out.get("grasp_anchor_depth") or out.get("nav_anchor_depth") or out.get("depth_m") or 0.0)
    p_cam = pixel_depth_to_cam(gu, gv, gdepth, HEAD_CAM)
    grasp_r = (HEAD_CAM_POS_ROBOT + HEAD_CAM_ROT_MATRIX @ p_cam).astype(np.float32)
    grasp_r[2] -= float(GRASP_DEPTH_OFFSET)
    out["grasp_anchor_uv"] = [gu, gv]
    out["grasp_anchor_depth"] = gdepth
    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["source_camera"] = "head"
    out["camera"] = "head"
    out["grasp_reliable"] = gdepth < 1.20
    out["role"] = "grasp"
    return out


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


def _pick_nav(ee_objs: List[dict], head_objs: List[dict]) -> Tuple[Optional[dict], Optional[dict], str]:
    ee_vis = [o for o in ee_objs if not o.get("track_coast")]
    head_vis = [o for o in head_objs if not o.get("track_coast")]
    ee_nav = ee_vis[0] if ee_vis else (ee_objs[0] if ee_objs else None)
    head_nav = head_vis[0] if head_vis else (head_objs[0] if head_objs else None)

    if head_nav is not None and float(head_nav.get("dist_to_robot") or 999.0) <= FAR_NAV_M:
        return head_nav, ee_nav, "head"
    if ee_nav is not None:
        return ee_nav, head_nav, "ee"
    if head_nav is not None:
        return head_nav, ee_nav, "head"
    return None, ee_nav, "none"


class TaskBPerceptionClean:
    def __init__(self) -> None:
        self.frame_count = 0
        self._pose = {"pos": ROBOT_INIT_POS.copy().astype(np.float32), "yaw": float(ROBOT_INIT_YAW)}
        self._ee_tracks: Dict[int, dict] = {}
        self._head_tracks: Dict[int, dict] = {}
        self._next_id = 0
        self._lock_id: Optional[int] = None
        self._lock_world: Optional[List[float]] = None
        self._lock_miss = 0
        self._nav_authority = "ee"
        print(f"[TaskBPerceptionClean] build={PERCEPTION_BUILD} ee-far-nav | head-near-grasp")

    def reset(self) -> None:
        self.frame_count = 0
        self._pose = {"pos": ROBOT_INIT_POS.copy().astype(np.float32), "yaw": float(ROBOT_INIT_YAW)}
        self._ee_tracks.clear()
        self._head_tracks.clear()
        self._next_id = 0
        self._lock_id = None
        self._lock_world = None
        self._lock_miss = 0
        self._nav_authority = "ee"

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
        arm_q = _read_arm_joints(obs)

        h_rgb, h_depth = parse_head_rgbd(obs)
        ee_rgb, ee_depth = parse_ee_rgbd(obs)

        ee_raw = _detect_yellow_rgbd(ee_rgb, ee_depth, "ee") if ee_rgb is not None and ee_depth is not None else []
        head_raw = _detect_yellow_rgbd(h_rgb, h_depth, "head")

        ee_objs, self._ee_tracks, self._next_id = _match_tracks(
            ee_raw, self._ee_tracks, rp, ry, self._next_id,
        )
        head_objs, self._head_tracks, self._next_id = _match_tracks(
            head_raw, self._head_tracks, rp, ry, self._next_id,
        )
        ee_objs = [_finalize_ee(o, rp, ry, arm_q) for o in ee_objs if _plausible_obj(o, "ee")]

        primary_nav, secondary_nav, authority = _pick_nav(ee_objs, head_objs)
        self._nav_authority = authority

        if self._lock_id is None and primary_nav is not None:
            self._lock_id = int(primary_nav["id"])
            self._lock_world = list(primary_nav.get("pos_smooth_world") or primary_nav["pos_world"])
            self._lock_miss = 0
        elif self._lock_id is not None:
            pool = ee_objs + head_objs
            hit = next((o for o in pool if int(o["id"]) == int(self._lock_id)), None)
            if hit is None and self._lock_world is not None:
                lw = np.asarray(self._lock_world, dtype=np.float32)
                for o in pool:
                    ow = np.asarray(o.get("pos_smooth_world") or o["pos_world"], dtype=np.float32)
                    if float(np.linalg.norm(ow[:2] - lw[:2])) < MATCH_RADIUS_M:
                        hit = o
                        break
            if hit is not None:
                self._lock_id = int(hit["id"])
                self._lock_world = list(hit.get("pos_smooth_world") or hit["pos_world"])
                self._lock_miss = 0
            else:
                self._lock_miss += 1
                if self._lock_miss > COAST_MAX_MISS:
                    self._lock_id = None
                    self._lock_world = None
                    if primary_nav is not None:
                        self._lock_id = int(primary_nav["id"])
                        self._lock_world = list(primary_nav.get("pos_smooth_world") or primary_nav["pos_world"])
                        self._lock_miss = 0

        target_nav: Optional[dict] = None
        if self._lock_id is not None and self._lock_world is not None:
            pool = ee_objs + head_objs
            hit = next((o for o in pool if int(o["id"]) == int(self._lock_id)), None)
            if hit is not None:
                target_nav = dict(hit)
                src = str(hit.get("source_camera") or authority)
                target_nav["source_camera"] = src
                target_nav["camera"] = src
            elif self._lock_miss <= COAST_MAX_MISS:
                pr = world_to_robot_frame(np.asarray(self._lock_world, dtype=np.float32), rp, ry)
                target_nav = {
                    "id": int(self._lock_id),
                    "class": CLASS_NAME,
                    "pos_world": list(self._lock_world),
                    "pos_smooth_world": list(self._lock_world),
                    "pos_robot": pr.tolist(),
                    "dist_to_robot": float(np.linalg.norm(pr[:2])),
                    "source_camera": "lock_coast",
                    "nav_coast": True,
                    "camera": self._nav_authority,
                    "world_reliable": True,
                    "yaw_rel": float(np.arctan2(pr[1], pr[0])),
                    "nav_yaw_rel": float(np.arctan2(pr[1], pr[0])),
                }
        if target_nav is None and primary_nav is not None:
            target_nav = dict(primary_nav)
        if target_nav is not None:
            target_nav = _apply_correction_hints(target_nav)

        lock_dist = float(target_nav.get("dist_to_robot") or 999.0) if target_nav else 999.0
        head_hit = None
        if self._lock_id is not None:
            head_hit = next((o for o in head_objs if int(o["id"]) == int(self._lock_id)), None)
            if head_hit is None and self._lock_world is not None:
                lw = np.asarray(self._lock_world, dtype=np.float32)
                for o in head_objs:
                    ow = np.asarray(o.get("pos_smooth_world") or o["pos_world"], dtype=np.float32)
                    if float(np.linalg.norm(ow[:2] - lw[:2])) <= MATCH_RADIUS_M:
                        head_hit = o
                        break

        want_grasp = (
            self._lock_id is not None
            and lock_dist < GRASP_DIST_M
            and head_hit is not None
            and self._lock_miss == 0
            and not head_hit.get("track_coast")
        )
        phase = "grasp" if want_grasp else "approach"
        target_grasp = (
            _apply_correction_hints(_grasp_from_head(head_hit, rp, ry))
            if want_grasp and head_hit
            else None
        )

        ee_export = [_apply_correction_hints(o) for o in ee_objs if not o.get("track_coast")][:3]
        head_export = [_apply_correction_hints(o) for o in head_objs if not o.get("track_coast")][:3]
        nav_cam = str((target_nav or {}).get("source_camera") or self._nav_authority)
        active_cam = "head" if (want_grasp or lock_dist <= FAR_NAV_M) else "ee"

        ee_hint = None
        if target_nav and target_nav.get("yaw_rel") is not None and not target_nav.get("nav_coast"):
            ee_hint = {
                "yaw_rel": float(target_nav["yaw_rel"]),
                "class": target_nav.get("class"),
                "id": target_nav.get("id"),
                "bearing_only": nav_cam == "ee" and lock_dist > FAR_NAV_M,
                "depth_m": target_nav.get("depth_m"),
            }
        elif ee_export:
            b = ee_export[0]
            ee_hint = {
                "yaw_rel": float(b.get("yaw_rel") or 0.0),
                "class": b.get("class"),
                "id": b.get("id"),
                "bearing_only": True,
                "depth_m": b.get("depth_m"),
            }

        if os.getenv("ATEC_TASKB_PERC_DEBUG", "0").lower() in ("1", "true", "yes"):
            every = max(1, int(os.getenv("ATEC_TASKB_PERC_DEBUG_EVERY", "20")))
            if self.frame_count % every == 0:
                print(
                    f"[PERC-RGBD] f={self.frame_count} raw ee={len(ee_raw)} head={len(head_raw)} "
                    f"out ee={len(ee_objs)} head={len(head_objs)} auth={self._nav_authority} "
                    f"lock={self._lock_id} miss={self._lock_miss} phase={phase} dist={lock_dist:.2f}"
                )

        nav_stage = "grasp" if want_grasp else ("far_ee" if lock_dist > FAR_NAV_M else "near_head")
        ee_for_nav = [target_nav] if target_nav and nav_cam == "ee" else ee_export

        return {
            "roles": {"ee": "nav_far", "head": "nav_near_grasp"},
            "nav_stage": nav_stage,
            "nav_authority": self._nav_authority,
            "nav_authority_mode": "primary",
            "nav_lock_id": self._lock_id,
            "nav_lock_class": CLASS_NAME if self._lock_id is not None else None,
            "nav_lock_ee_only": bool(self._lock_id is not None and not head_objs),
            "nav_lock_stable": self._lock_id is not None and self._lock_miss == 0,
            "nav_pos_confidence": None if target_nav is None else target_nav.get("pos_confidence"),
            "ee_search_hint": ee_hint,
            "navigation": {"camera": nav_cam, "target": target_nav, "objects_detailed": ee_for_nav or ee_export},
            "target_nav": target_nav,
            "objects_nav": ee_for_nav if nav_cam == "ee" else (head_export or head_objs),
            "ee_objects": ee_for_nav or ee_export,
            "ee_objects_list": ee_objs,
            "grasp": {"camera": "head", "target": target_grasp, "objects_detailed": [target_grasp] if target_grasp else head_export},
            "target_grasp": target_grasp,
            "objects_grasp": [target_grasp] if target_grasp else head_export,
            "head_objects": head_export or head_objs,
            "head_objects_list": head_objs,
            "target": target_grasp if want_grasp else target_nav,
            "objects_remaining": head_objs + ee_objs,
            "active_camera": active_cam,
            "phase": phase,
            "grasp_reliable": bool(target_grasp and target_grasp.get("grasp_reliable")),
            "grasp_locked": bool(target_grasp),
            "head_dist_m": float((head_objs[0] if head_objs else {}).get("depth_m") or 999.0),
            "ee_dist_m": float((ee_objs[0] if ee_objs else {}).get("depth_m") or 999.0),
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


RgbdPureDualPipeline = TaskBPerceptionClean
