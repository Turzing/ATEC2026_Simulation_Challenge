"""
Task B 感知 — RGBD 精简版

官方传感器 (readme):
  head = eye-to-hand  → 稳定俯视，负责导航 + 抓取 3D
  ee   = eye-in-hand  → 随臂转动，不做固定外参 3D (低头时看地面/夹爪)

EE 仅输出 bearing 提示; 3D 坐标一律由 head RGBD 反投影 + 时序平滑。
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
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    IMG_H,
    IMG_W,
    PROPRIO_BASE_ANG_VEL,
    PROPRIO_BASE_LIN_VEL,
    PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    TOTAL_OBJECTS,
)
from rgbd_utils import (
    depth_stats,
    is_ee_floor_gripper_phantom,
    is_ee_sky_blob,
    parse_ee_rgbd,
    parse_head_rgbd,
    pixel_depth_to_cam,
    robot_to_world,
    world_to_robot_frame,
    _to_numpy,
)

PERCEPTION_BUILD = "20260618-taskb-rgbd-v3"

# --- HSV 黄瓶 ---
HUE_LO, HUE_HI = 6, 58
SAT_MIN, VAL_MIN = 4, 18
MIN_BLOB_PX = 10
MIN_SIDE = 4

# --- 时序稳定 ---
POS_EMA_ALPHA = 0.42          # 新检测权重; 越小越稳
JUMP_REJECT_FAR_M = 0.38
JUMP_REJECT_NEAR_M = 0.14
NEAR_M = 1.35
COAST_MAX_MISS = 22
MATCH_RADIUS_M = 0.58
LOCK_MATCH_M = 0.68

GRASP_DIST_M = 1.25
ROBOT_Z_MIN, ROBOT_Z_MAX = -0.85, 0.18
CLASS_NAME = "mustard_bottle"


def _filter_ee_objects(objs: List[dict]) -> List[dict]:
    """EE 不做 3D 导航; 剔除低头看地/夹爪假检."""
    out: List[dict] = []
    for o in objs:
        if is_ee_floor_gripper_phantom(o):
            continue
        if is_ee_sky_blob(o):
            continue
        pr = o.get("pos_robot")
        if pr is not None and float(pr[2]) < -0.10:
            continue
        out.append(o)
    return out


def _cam_params(camera: str):
    if camera == "head":
        return HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX
    return EE_CAM, EE_CAM_POS_ROBOT, EE_CAM_ROT_MATRIX


def _uv_depth_to_robot(u: float, v: float, depth_m: float, camera: str) -> Optional[np.ndarray]:
    if depth_m <= 0.05 or depth_m > 9.5:
        return None
    cam, pos, rot = _cam_params(camera)
    p_cam = pixel_depth_to_cam(u, v, depth_m, cam)
    return (pos + rot @ p_cam).astype(np.float32)


def _robust_depth(
    depth: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    anchor_v: float,
    band_frac: float = 0.22,
) -> Optional[float]:
    """取 blob 底边附近 depth 的中位数, 比整 blob 中值更贴地."""
    y1, y2 = int(ys.min()), int(ys.max())
    band = max(3, int((y2 - y1 + 1) * band_frac))
    v_lo = max(y1, int(anchor_v) - band)
    sel = (ys >= v_lo) & (ys <= int(anchor_v))
    if not np.any(sel):
        sel = ys >= max(y1, y2 - band)
    dvals = depth[ys[sel], xs[sel]]
    dvals = dvals[(dvals > 0.05) & (dvals < 9.0)]
    if dvals.size < 4:
        dvals = depth[ys, xs]
        dvals = dvals[(dvals > 0.05) & (dvals < 9.0)]
    if dvals.size < 4:
        return None
    return float(np.median(dvals))


def _plausible_robot_z(pr: np.ndarray, camera: str) -> bool:
    pz = float(pr[2])
    if pz < ROBOT_Z_MIN or pz > ROBOT_Z_MAX:
        return False
    horiz = float(np.hypot(float(pr[0]), float(pr[1])))
    if horiz < 0.08 and abs(float(pr[1])) < 0.12:
        return False
    return True


def _plausible_bbox(bbox: List[int], pr: np.ndarray, camera: str, depth_m: float) -> bool:
    x1, y1, x2, y2 = bbox
    cy = 0.5 * (y1 + y2)
    if camera == "head":
        if y2 < IMG_H * 0.22:
            return False
    else:
        if is_ee_floor_gripper_phantom({"bbox": bbox, "depth_m": depth_m, "pos_robot": pr}):
            return False
        if cy < IMG_H * 0.12 and depth_m < 2.5:
            return False
    return True


def _detect_yellow_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: str,
    *,
    v_min: float,
    v_max: float,
    depth_min: float,
    depth_max: float,
) -> List[dict]:
    h, w = depth.shape[:2]
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb[..., :3], (w, h), interpolation=cv2.INTER_LINEAR)

    v0, v1 = int(h * v_min), int(h * v_max)
    v0 = max(0, min(v0, h - 2))
    v1 = max(v0 + 2, min(v1, h))

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[v0:v1, :] = 1
    mask = (
        roi
        & ((hsv[:, :, 0] >= HUE_LO) & (hsv[:, :, 0] <= HUE_HI))
        & (hsv[:, :, 1] >= SAT_MIN)
        & (hsv[:, :, 2] >= VAL_MIN)
        & (depth >= depth_min)
        & (depth <= depth_max)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    labeled, n = ndimage.label(mask)
    if n <= 0:
        return []

    out: List[dict] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labeled == lab)
        if ys.size < MIN_BLOB_PX:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if (x2 - x1 + 1) < MIN_SIDE or (y2 - y1 + 1) < MIN_SIDE:
            continue

        nav_u = float(0.5 * (x1 + x2))
        nav_v = float(y2)
        nav_depth = _robust_depth(depth, ys, xs, anchor_v=nav_v)
        if nav_depth is None:
            continue

        pr = _uv_depth_to_robot(nav_u, nav_v, nav_depth, camera)
        if pr is None or not _plausible_robot_z(pr, camera):
            continue
        bbox = [x1, y1, x2, y2]
        if not _plausible_bbox(bbox, pr, camera, nav_depth):
            continue

        dist = float(np.linalg.norm(pr[:2]))
        obj = {
            "bbox": bbox,
            "centroid_uv": [nav_u, 0.5 * (y1 + y2)],
            "nav_anchor_uv": [nav_u, nav_v],
            "nav_anchor_depth": nav_depth,
            "depth_m": nav_depth,
            "nav_depth_m": nav_depth,
            "pos_robot": pr.tolist(),
            "dist_to_robot": dist,
            "yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "nav_yaw_rel": float(np.arctan2(float(pr[1]), float(pr[0]))),
            "source_camera": camera,
            "camera": camera,
            "source": "rgbd_hsv",
            "class": CLASS_NAME,
            "conf": 0.78,
            "world_reliable": nav_depth < 2.4,
            "pos_confidence": 0.82 if nav_depth < 1.8 else 0.68,
            "visible": True,
            "role": "nav",
        }
        out.append(obj)

    out.sort(key=lambda o: float(o.get("depth_m") or 999.0))
    return out


def _enrich_world(obj: dict, robot_pos: np.ndarray, robot_yaw: float) -> dict:
    out = dict(obj)
    pr = np.asarray(out["pos_robot"], dtype=np.float32)
    pw = robot_to_world(pr, robot_pos, robot_yaw)
    out["pos_world"] = pw.tolist()
    return out


def _smooth_with_prev(obj: dict, prev: Optional[dict], robot_pos: np.ndarray, robot_yaw: float) -> dict:
    """EMA 平滑 world 坐标, 抑制单帧 depth 抖动."""
    out = dict(obj)
    pw = np.asarray(out["pos_world"], dtype=np.float32)
    if prev is None:
        out["pos_smooth_world"] = pw.tolist()
        return out

    prev_sw = np.asarray(prev.get("pos_smooth_world") or prev["pos_world"], dtype=np.float32)
    jump = float(np.linalg.norm(pw[:2] - prev_sw[:2]))
    dist = float(out.get("dist_to_robot") or 99.0)
    limit = JUMP_REJECT_NEAR_M if dist < NEAR_M else JUMP_REJECT_FAR_M
    if jump > limit:
        # 突变: 沿用上一帧平滑值, 保留新 bbox/depth 供 debug
        out["pos_jump_rejected"] = True
        sw = prev_sw.copy()
    else:
        a = POS_EMA_ALPHA
        sw = (1.0 - a) * prev_sw + a * pw
        sw[2] = (1.0 - a) * prev_sw[2] + a * pw[2]

    out["pos_smooth_world"] = sw.tolist()
    pr = world_to_robot_frame(sw, robot_pos, robot_yaw)
    out["pos_world"] = sw.tolist()
    out["pos_robot"] = pr.tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
    out["yaw_rel"] = float(np.arctan2(float(pr[1]), float(pr[0])))
    out["nav_yaw_rel"] = out["yaw_rel"]
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
            dxy = float(np.linalg.norm(pw[:2] - tw[:2]))
            if dxy < best_d:
                best_d, best_id = dxy, tid
        if best_id is None:
            best_id = next_id
            next_id += 1
        used.add(best_id)
        prev = tracks.get(best_id)
        obj["id"] = int(best_id)
        smoothed = _smooth_with_prev(obj, prev, robot_pos, robot_yaw)
        if prev is not None:
            for k in ("nav_anchor_uv", "nav_anchor_depth", "grasp_anchor_uv", "grasp_anchor_depth"):
                if k in prev and k not in smoothed:
                    smoothed[k] = prev[k]
        new_tracks[best_id] = smoothed
        matched.append(smoothed)

    return matched, new_tracks, next_id


def _grasp_from_head(head_obj: dict, robot_pos: np.ndarray, robot_yaw: float) -> dict:
    """head RGBD: 底边 anchor → 3D 抓取点."""
    out = dict(head_obj)
    bbox = out.get("bbox") or [0, 0, 0, 0]
    x1, _, x2, y2 = bbox
    gu = float(out.get("grasp_anchor_uv", [0.5 * (x1 + x2), y2])[0])
    gv = float(out.get("grasp_anchor_uv", [gu, y2])[1])
    gdepth = float(out.get("grasp_anchor_depth") or out.get("nav_anchor_depth") or out.get("depth_m") or 0.0)

    grasp_r = _uv_depth_to_robot(gu, gv, gdepth, "head")
    if grasp_r is None:
        pr = np.asarray(out["pos_robot"], dtype=np.float32)
        grasp_r = pr.copy()
        grasp_r[2] -= float(GRASP_DEPTH_OFFSET)
    else:
        grasp_r[2] = float(grasp_r[2]) - float(GRASP_DEPTH_OFFSET)

    out["grasp_anchor_uv"] = [gu, gv]
    out["grasp_anchor_depth"] = gdepth
    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["grasp_offset_robot"] = [
        float(grasp_r[0] - np.asarray(out["pos_robot"])[0]),
        float(grasp_r[1] - np.asarray(out["pos_robot"])[1]),
        float(grasp_r[2] - np.asarray(out["pos_robot"])[2]),
    ]
    out["source_camera"] = "head"
    out["camera"] = "head"
    out["nav_from_head"] = True
    out["grasp_reliable"] = float(gdepth) < 1.15
    out["role"] = "grasp"
    out["skip_camera_correction"] = False
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


class TaskBPerceptionClean:
    """RGBD 感知: 稳定坐标 + 准确抓取点."""

    def __init__(self) -> None:
        self.frame_count = 0
        self._pose = {"pos": ROBOT_INIT_POS.copy().astype(np.float32), "yaw": float(ROBOT_INIT_YAW)}
        self._ee_tracks: Dict[int, dict] = {}
        self._head_tracks: Dict[int, dict] = {}
        self._next_id = 0
        self._lock_id: Optional[int] = None
        self._lock_world: Optional[List[float]] = None
        self._lock_miss = 0
        print(f"[TaskBPerceptionClean] build={PERCEPTION_BUILD} head-RGBD nav+grasp | ee=bearing-only")

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

        h_rgb, h_depth = parse_head_rgbd(obs)
        ee_rgb, ee_depth = parse_ee_rgbd(obs)

        ee_raw = (
            _detect_yellow_rgbd(ee_rgb, ee_depth, "ee", v_min=0.04, v_max=0.96, depth_min=0.30, depth_max=7.0)
            if ee_rgb is not None and ee_depth is not None
            else []
        )
        head_raw = _detect_yellow_rgbd(
            h_rgb, h_depth, "head", v_min=0.10, v_max=0.98, depth_min=0.10, depth_max=4.2,
        )

        ee_objs, self._ee_tracks, self._next_id = _match_tracks(
            ee_raw, self._ee_tracks, rp, ry, self._next_id,
        )
        head_objs, self._head_tracks, self._next_id = _match_tracks(
            head_raw, self._head_tracks, rp, ry, self._next_id,
        )
        ee_objs = _filter_ee_objects(ee_objs)

        head_nav = head_objs[0] if head_objs else None
        ee_nav = ee_objs[0] if ee_objs else None

        # --- nav lock: head 3D 为主 (eye-to-hand 稳定) ---
        if self._lock_id is None and head_nav is not None:
            self._lock_id = int(head_nav["id"])
            self._lock_world = list(head_nav.get("pos_smooth_world") or head_nav["pos_world"])
            self._lock_miss = 0
        elif self._lock_id is None and ee_nav is not None:
            self._lock_id = int(ee_nav["id"])
            self._lock_world = list(ee_nav.get("pos_smooth_world") or ee_nav["pos_world"])
            self._lock_miss = 0
        elif self._lock_id is not None:
            hit = next((o for o in head_objs if int(o["id"]) == int(self._lock_id)), None)
            if hit is None:
                hit = next((o for o in ee_objs if int(o["id"]) == int(self._lock_id)), None)
            if hit is None and self._lock_world is not None:
                lw = np.asarray(self._lock_world, dtype=np.float32)
                for o in head_objs + ee_objs:
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
                    if head_nav is not None:
                        self._lock_id = int(head_nav["id"])
                        self._lock_world = list(head_nav.get("pos_smooth_world") or head_nav["pos_world"])
                        self._lock_miss = 0
                    elif ee_nav is not None:
                        self._lock_id = int(ee_nav["id"])
                        self._lock_world = list(ee_nav.get("pos_smooth_world") or ee_nav["pos_world"])
                        self._lock_miss = 0

        target_nav: Optional[dict] = None
        if self._lock_id is not None and self._lock_world is not None:
            hit = next((o for o in head_objs if int(o["id"]) == int(self._lock_id)), None)
            if hit is None:
                hit = next((o for o in ee_objs if int(o["id"]) == int(self._lock_id)), None)
            if hit is not None:
                target_nav = dict(hit)
                target_nav["source_camera"] = hit.get("source_camera", "head")
                target_nav["camera"] = target_nav["source_camera"]
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
                    "camera": "head",
                    "world_reliable": True,
                    "depth_m": float(np.linalg.norm(pr[:2])),
                    "yaw_rel": float(np.arctan2(pr[1], pr[0])),
                    "nav_yaw_rel": float(np.arctan2(pr[1], pr[0])),
                }
        if target_nav is None and head_nav is not None:
            target_nav = dict(head_nav)
            target_nav["source_camera"] = "head"
        elif target_nav is None and ee_nav is not None:
            target_nav = dict(ee_nav)
            target_nav["source_camera"] = "ee"
            target_nav["bearing_only"] = True

        lock_dist = float(target_nav.get("dist_to_robot") or 999.0) if target_nav else 999.0
        head_hit: Optional[dict] = None
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
        )
        phase = "grasp" if want_grasp else "approach"
        target_grasp = _grasp_from_head(head_hit, rp, ry) if want_grasp and head_hit else None

        ee_export: List[dict] = []
        nav_export: List[dict] = []
        if target_nav is not None:
            exp = dict(target_nav)
            src = str(exp.get("source_camera") or "head")
            exp["camera"] = src
            nav_export = [exp]
            if src == "head":
                ee_export = []
            else:
                ee_export = [exp]
        elif head_nav is not None:
            nav_export = [dict(head_nav)]
        elif ee_nav is not None:
            hint = dict(ee_nav)
            hint["bearing_only"] = True
            ee_export = [hint]

        ee_hint = None
        if target_nav and target_nav.get("yaw_rel") is not None and not target_nav.get("bearing_only"):
            src = target_nav
            ee_hint = {
                "yaw_rel": float(src["yaw_rel"]),
                "class": src.get("class"),
                "id": src.get("id"),
                "bearing_only": False,
                "depth_m": src.get("depth_m"),
            }
        elif ee_objs:
            b = ee_objs[0]
            ee_hint = {
                "yaw_rel": float(b.get("yaw_rel") or 0.0),
                "class": b.get("class"),
                "id": b.get("id"),
                "bearing_only": True,
                "depth_m": b.get("depth_m"),
            }

        if os.getenv("ATEC_TASKB_PERC_DEBUG", "0").lower() in ("1", "true", "yes"):
            every = max(1, int(os.getenv("ATEC_TASKB_PERC_DEBUG_EVERY", "25")))
            if self.frame_count % every == 0:
                nav_pw = None
                if target_nav:
                    nav_pw = np.asarray(target_nav.get("pos_smooth_world") or target_nav["pos_world"]).round(3).tolist()
                grasp_pw = None
                if target_grasp:
                    grasp_pw = np.asarray(target_grasp.get("grasp_pos_world")).round(3).tolist()
                print(
                    f"[PERC-RGBD] f={self.frame_count} raw ee={len(ee_raw)} head={len(head_raw)} "
                    f"lock={self._lock_id} miss={self._lock_miss} phase={phase} dist={lock_dist:.2f} "
                    f"nav_w={nav_pw} grasp_w={grasp_pw}"
                )

        return {
            "roles": {"ee": "bearing", "head": "nav_grasp"},
            "nav_stage": "grasp" if want_grasp else "near_head",
            "nav_authority": "head",
            "nav_authority_mode": "primary",
            "nav_lock_id": self._lock_id,
            "nav_lock_class": CLASS_NAME if self._lock_id is not None else None,
            "nav_lock_ee_only": False,
            "nav_lock_stable": self._lock_id is not None and self._lock_miss == 0,
            "nav_pos_confidence": None if target_nav is None else target_nav.get("pos_confidence"),
            "ee_search_hint": ee_hint,
            "navigation": {"camera": "head", "target": target_nav, "objects_detailed": nav_export or head_objs},
            "target_nav": target_nav,
            "objects_nav": nav_export or head_objs,
            "ee_objects": ee_export,
            "ee_objects_list": ee_objs,
            "grasp": {"camera": "head", "target": target_grasp, "objects_detailed": [target_grasp] if target_grasp else head_objs},
            "target_grasp": target_grasp,
            "objects_grasp": [target_grasp] if target_grasp else head_objs,
            "head_objects": head_objs if head_objs else nav_export,
            "head_objects_list": head_objs,
            "target": target_grasp if want_grasp else target_nav,
            "objects_remaining": head_objs + ee_objs,
            "active_camera": "head",
            "phase": phase,
            "grasp_reliable": bool(target_grasp and target_grasp.get("grasp_reliable")),
            "grasp_locked": bool(target_grasp),
            "head_dist_m": float(head_nav.get("depth_m") or 999.0) if head_nav else 999.0,
            "ee_dist_m": float(ee_nav.get("depth_m") or 999.0) if ee_nav else 999.0,
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
