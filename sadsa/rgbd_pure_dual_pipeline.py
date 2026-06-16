"""
RGB-D 双摄像头感知

分工 (固定):
    ee   → 导航 + 抓取  (nav_depth_m / nav_yaw_rel / grasp_pos_world)
    head → 仅导航       (无 grasp 字段)

精度 (无 GT):
    - rgb/depth 对齐, 修 blob broadcast 崩溃
    - head/EE 动态外参 (倾身/臂关节)
    - 近距冻结 EE grasp (蹲下不漂)
    - head 近距/蹲下禁用, 本体假检过滤
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BIN_CENTER,
    BIN_RADIUS,
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
from rgbd_pure_pipeline import RgbdPureCamera, _robot_to_world, _yaw_from_gravity
from rgbd_utils import (
    GRASP_RELIABLE_DEPTH_M,
    _to_numpy,
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    depth_stats,
    filter_plausible_objects,
    parse_ee_rgbd,
    parse_head_rgbd,
    refresh_ee_object_pose,
    refresh_head_object_pose,
    refresh_locked_grasp,
    stabilize_ee_nav_pose,
)

GRASP_PHASE_DIST_M = 1.10
GRASP_LOCK_DIST_M = 1.22
GRASP_UNLOCK_DIST_M = 1.50
HEAD_DISABLE_DIST_M = 1.05
EE_NAV_PREFER_DIST_M = 1.35
HEAD_MIRROR_EE_MIN_M = 1.25
TEMPORAL_MEDIAN_N = 6
GRASP_TEMPORAL_N = 10
MOTION_FREEZE_THRESH = 0.35


def _obj_dist(obj: Optional[dict]) -> float:
    if not obj:
        return 999.0
    d = obj.get("depth_m")
    if d is not None and d > 0.05:
        return float(d)
    return float(obj.get("dist_to_robot") or 999.0)


def _motion_level(obs) -> float:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        lin = float(np.linalg.norm(p[PROPRIO_BASE_LIN_VEL]))
        ang = float(np.linalg.norm(p[PROPRIO_BASE_ANG_VEL]))
        return lin * 0.6 + ang * 0.4
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0


def _read_arm_joints(obs) -> Optional[np.ndarray]:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        j0 = PROPRIO_ARM_START
        return p[j0 : j0 + PROPRIO_ARM_LEN].copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _read_projected_gravity(obs) -> Optional[np.ndarray]:
    try:
        p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        return p[PROPRIO_PROJECTED_GRAVITY].copy()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _enrich_nav(obj: Optional[dict]) -> Optional[dict]:
    if obj is None:
        return None
    o = dict(obj)
    if o.get("nav_depth_m") is None and o.get("depth_m") is not None:
        o["nav_depth_m"] = float(o["depth_m"])
    if o.get("nav_yaw_rel") is None and o.get("yaw_rel") is not None:
        o["nav_yaw_rel"] = float(o["yaw_rel"])
    if "world_reliable" not in o:
        o["world_reliable"] = float(o.get("depth_m") or 99.0) < 2.0
    return o


def _as_head_nav(o: dict) -> dict:
    out = _enrich_nav(o) or {}
    out["camera"] = "head"
    out["role"] = "nav"
    out["grasp_reliable"] = False
    for k in (
        "grasp_pos_world", "grasp_quat_world", "grasp_pos_robot",
        "grasp_offset_robot", "grasp_anchor_uv", "grasp_anchor_depth",
    ):
        out.pop(k, None)
    return out


def _finalize_ee(o: dict, robot_pos, robot_yaw, arm_joints) -> dict:
    cam_pos = compute_dynamic_ee_cam_pos(arm_joints) if arm_joints is not None else None
    if cam_pos is None:
        from config import EE_CAM_POS_ROBOT
        cam_pos = EE_CAM_POS_ROBOT
    out = refresh_ee_object_pose(o, robot_pos, robot_yaw, cam_pos)
    out["camera"] = "ee"
    out["role"] = "nav_grasp"
    return _enrich_nav(out) or out


def _finalize_head(o: dict, robot_pos, robot_yaw, grav) -> dict:
    cam_pos = compute_dynamic_head_cam_pos(grav)
    out = refresh_head_object_pose(o, robot_pos, robot_yaw, cam_pos)
    out["camera"] = "head"
    out["role"] = "nav"
    return _enrich_nav(out) or out


def _object_summary(o: dict, cam: str) -> dict:
    s = {
        "id": int(o["id"]),
        "camera": cam,
        "role": o.get("role", "nav" if cam == "head" else "nav_grasp"),
        "class": o.get("class"),
        "conf": float(o.get("conf", 0)),
        "depth_m": o.get("depth_m"),
        "nav_depth_m": o.get("nav_depth_m"),
        "nav_yaw_rel": o.get("nav_yaw_rel"),
        "dist_to_robot": o.get("dist_to_robot"),
        "pos_world": o.get("pos_world"),
        "pos_robot": o.get("pos_robot"),
        "yaw_rel": o.get("yaw_rel"),
        "world_reliable": o.get("world_reliable"),
        "grasp_reliable": o.get("grasp_reliable", False),
        "bbox": o.get("bbox"),
    }
    if cam == "ee":
        for k in ("grasp_pos_world", "grasp_quat_world", "grasp_offset_robot", "grasp_locked"):
            if o.get(k) is not None:
                s[k] = o[k]
    return s


class _TemporalMedian:
    def __init__(self, n: int = TEMPORAL_MEDIAN_N, grasp_n: int = GRASP_TEMPORAL_N):
        self.n = n
        self.grasp_n = grasp_n
        self._hist: Dict[Tuple[str, int], List[dict]] = {}
        self._grasp_hist: Dict[Tuple[str, int], List[dict]] = {}

    def reset(self):
        self._hist.clear()
        self._grasp_hist.clear()

    def apply(
        self,
        objects: List[dict],
        cam: str,
        robot_pos,
        robot_yaw,
        motion: float = 0.0,
    ) -> List[dict]:
        out = []
        shaky = motion > MOTION_FREEZE_THRESH
        for o in objects:
            key = (cam, int(o["id"]))
            self._hist.setdefault(key, [])
            self._hist[key].append(o)
            self._hist[key] = self._hist[key][-self.n :]
            if cam == "ee":
                self._grasp_hist.setdefault(key, [])
                self._grasp_hist[key].append(o)
                self._grasp_hist[key] = self._grasp_hist[key][-self.grasp_n :]

            h = self._hist[key]
            m = dict(o)

            depths = [x["depth_m"] for x in h if x.get("depth_m") is not None]
            if depths:
                m["depth_m"] = float(np.median(depths))
                m["nav_depth_m"] = m["depth_m"]

            prs = [x["pos_robot"] for x in h if x.get("pos_robot") is not None]
            if prs:
                med = np.median(np.stack([np.asarray(p, dtype=np.float32) for p in prs]), axis=0)
                m["pos_robot"] = med.tolist()
                m["pos_world"] = _robot_to_world(med, robot_pos, robot_yaw).tolist()
                m["dist_to_robot"] = float(np.linalg.norm(med[:2]))
                m["yaw_rel"] = float(np.arctan2(med[1], med[0]))
                m["nav_yaw_rel"] = m["yaw_rel"]

            if cam == "ee":
                gh = self._grasp_hist.get(key, h)
                if shaky and gh:
                    prev = gh[-2] if len(gh) >= 2 else gh[-1]
                    if prev.get("grasp_offset_robot") is not None:
                        m["grasp_offset_robot"] = prev["grasp_offset_robot"]
                    if prev.get("grasp_anchor_uv") is not None:
                        m["grasp_anchor_uv"] = prev["grasp_anchor_uv"]
                    if prev.get("grasp_anchor_depth") is not None:
                        m["grasp_anchor_depth"] = prev["grasp_anchor_depth"]
                    m["grasp_reliable"] = False
                else:
                    gos = [x.get("grasp_offset_robot") for x in gh if x.get("grasp_offset_robot")]
                    if gos:
                        go = np.median(np.stack([np.asarray(g, dtype=np.float32) for g in gos]), axis=0)
                        m["grasp_offset_robot"] = go.tolist()
                    m["grasp_reliable"] = float(m.get("depth_m") or 99.0) < GRASP_RELIABLE_DEPTH_M

                gqs = [x.get("grasp_quat_world") for x in gh if x.get("grasp_quat_world")]
                if gqs:
                    m["grasp_quat_world"] = np.median(
                        np.stack([np.asarray(q, dtype=np.float32) for q in gqs]), axis=0,
                    ).tolist()

            m["world_reliable"] = float(m.get("depth_m") or 99.0) < 2.0
            out.append(m)
        return out


def _nav_quality(obj: dict) -> float:
    sm = float(obj.get("blob_sat_mean", 50))
    vm = float(obj.get("blob_val_mean", 90))
    area = int((obj["bbox"][2] - obj["bbox"][0] + 1) * (obj["bbox"][3] - obj["bbox"][1] + 1))
    q = sm * 0.5 + vm * 0.2 - min(area, 3500) * 0.003
    if area > 1200 and sm < 50:
        q -= 80.0
    if obj.get("world_reliable"):
        q += 30.0
    return q


def _best_nav_target(objs: List[dict]) -> Optional[dict]:
    if not objs:
        return None
    ranked = sorted(objs, key=_obj_dist)
    best = ranked[0]
    bd, bq = _obj_dist(best), _nav_quality(best)
    for o in ranked[1:]:
        if _obj_dist(o) - bd > 0.40:
            break
        if _nav_quality(o) > bq + 18.0 and _obj_dist(o) < bd + 0.65:
            best = o
    return _enrich_nav(best)


def _grasp_quality(obj: dict) -> float:
    q = _nav_quality(obj)
    if obj.get("grasp_reliable"):
        q += 120.0
    if obj.get("grasp_quat_world"):
        q += 40.0
    return q


def _best_ee_grasp(objs: List[dict]) -> Optional[dict]:
    if not objs:
        return None
    pool = [o for o in objs if o.get("grasp_reliable")] or objs
    scored = [(_grasp_quality(o), _obj_dist(o), o) for o in pool]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def _pick_nav_target(
    ee_tgt: Optional[dict],
    head_tgt: Optional[dict],
    ee_objs: List[dict],
    head_objs: List[dict],
    ee_near: float,
) -> Tuple[str, List[dict], Optional[dict]]:
    if ee_near < EE_NAV_PREFER_DIST_M and ee_tgt:
        return "ee", ee_objs, ee_tgt
    if head_tgt and (not ee_tgt or _obj_dist(head_tgt) + 0.15 < _obj_dist(ee_tgt)):
        return "head", head_objs, head_tgt
    if ee_tgt and not head_tgt:
        return "ee", ee_objs, ee_tgt
    if head_tgt and not ee_tgt:
        return "head", head_objs, head_tgt
    if not ee_tgt and not head_tgt:
        return "ee", ee_objs, None
    if _nav_quality(head_tgt) > _nav_quality(ee_tgt) + 8.0:
        return "head", head_objs, head_tgt
    return "ee", ee_objs, ee_tgt


def _inject_head_nav_into_ee(
    ee_objs: List[dict],
    head_nav: Optional[dict],
    phase: str,
    ee_near: float,
) -> List[dict]:
    """solution_rl approach 固定 preferred=ee；远距把 head 导航位姿注入 ee_objects."""
    if phase != "approach" or head_nav is None or ee_near < HEAD_MIRROR_EE_MIN_M:
        return ee_objs
    hd = _obj_dist(head_nav)
    if hd > 3.0:
        return ee_objs
    mirror = dict(head_nav)
    for k in (
        "grasp_pos_robot", "grasp_pos_world", "grasp_quat_world",
        "grasp_reliable", "grasp_locked", "grasp_offset_robot",
    ):
        mirror.pop(k, None)
    mirror["grasp_reliable"] = False
    mirror["nav_from_head"] = True
    cls = mirror.get("class")
    kept = [
        o for o in ee_objs
        if not (cls is not None and o.get("class") == cls and _obj_dist(o) > hd * 0.80)
    ]
    return [mirror] + kept


class RgbdPureDualPipeline:
    def __init__(self):
        self.ee = RgbdPureCamera("ee")
        self.head = RgbdPureCamera("head")
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)
        self._temporal = _TemporalMedian()
        self._frozen_grasp: Optional[dict] = None
        print("[RgbdPureDual] ee=nav+grasp | head=nav-only | grasp-lock | head-mirror-ee")

    def reset(self):
        self.ee.reset()
        self.head.reset()
        self._temporal.reset()
        self._frozen_grasp = None
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)

    def _update_robot_pose(self, obs, dt: float = 0.02) -> None:
        try:
            p = _to_numpy(obs["proprio"]).astype(np.float32).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return
        if p.size < 12:
            return
        lin, ang, grav = p[PROPRIO_BASE_LIN_VEL], p[PROPRIO_BASE_ANG_VEL], p[PROPRIO_PROJECTED_GRAVITY]
        c, s = np.cos(self.robot_yaw), np.sin(self.robot_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        dxy = rot @ lin[:2] * dt
        self.robot_pos[0] += dxy[0]
        self.robot_pos[1] += dxy[1]
        self.robot_pos[2] = ROBOT_INIT_POS[2]
        yaw_g = _yaw_from_gravity(grav)
        yaw_i = self.robot_yaw + ang[2] * dt
        a = PROPRIO_YAW_FUSION_ALPHA
        self.robot_yaw = float((a * yaw_g + (1 - a) * yaw_i + np.pi) % (2 * np.pi) - np.pi)

    def _update_grasp_lock(
        self,
        ee_grasp: Optional[dict],
        ee_d: float,
        rp,
        ry,
        arm_q,
    ) -> Optional[dict]:
        if ee_d > GRASP_UNLOCK_DIST_M:
            self._frozen_grasp = None
            return None
        if ee_grasp is None:
            if self._frozen_grasp is not None:
                return refresh_locked_grasp(self._frozen_grasp, rp, ry)
            return None
        cand = _finalize_ee(ee_grasp, rp, ry, arm_q)
        if ee_d < GRASP_LOCK_DIST_M and (
            cand.get("grasp_reliable") or ee_d < GRASP_PHASE_DIST_M
        ):
            if self._frozen_grasp is None:
                self._frozen_grasp = dict(cand)
            else:
                old_id = self._frozen_grasp.get("id")
                if old_id is not None and cand.get("id") == old_id:
                    self._frozen_grasp = dict(cand)
            return refresh_locked_grasp(self._frozen_grasp, rp, ry)
        if self._frozen_grasp is not None:
            return refresh_locked_grasp(self._frozen_grasp, rp, ry)
        return cand

    def process(self, obs, dt: float = 0.02, gt_robot_pos=None, gt_robot_yaw=None, **_) -> dict:
        self.frame_count += 1
        if gt_robot_pos is not None and gt_robot_yaw is not None:
            self.robot_pos = np.asarray(gt_robot_pos, dtype=np.float32).copy()
            self.robot_yaw = float(gt_robot_yaw)
        else:
            self._update_robot_pose(obs, dt)

        rp, ry = self.robot_pos, self.robot_yaw
        motion = _motion_level(obs)
        arm_q = _read_arm_joints(obs)
        grav = _read_projected_gravity(obs)
        self.ee.set_arm_joints(arm_q)
        self.head.set_projected_gravity(grav)
        self.ee.set_projected_gravity(grav)

        h_rgb, h_depth = parse_head_rgbd(obs)
        head_objs, _, head_meta = self.head.process_frame(h_rgb, h_depth, rp, ry)

        ee_objs: List[dict] = []
        ee_meta: dict = {}
        ee_stats: dict = {}
        e_rgb, e_depth = parse_ee_rgbd(obs)
        if e_rgb is not None and e_depth is not None:
            ee_objs, _, ee_meta = self.ee.process_frame(e_rgb, e_depth, rp, ry)
            ee_stats = ee_meta.get("depth_stats") or depth_stats(e_depth)

        head_objs = [_as_head_nav(o) for o in head_objs]
        head_objs = self._temporal.apply(head_objs, "head", rp, ry, motion)
        ee_objs = self._temporal.apply(ee_objs, "ee", rp, ry, motion)
        head_objs = [_finalize_head(o, rp, ry, grav) for o in head_objs]
        ee_objs = [_finalize_ee(o, rp, ry, arm_q) for o in ee_objs]

        ee_objs = filter_plausible_objects(ee_objs, "ee")
        ee_near = _obj_dist(_best_nav_target(ee_objs) or _best_ee_grasp(ee_objs))
        head_objs = filter_plausible_objects(head_objs, "head", ee_near_m=ee_near)

        head_objs.sort(key=_obj_dist)
        ee_objs.sort(key=_obj_dist)

        ee_nav = _best_nav_target(ee_objs)
        head_nav = _best_nav_target(head_objs)
        ee_grasp = _best_ee_grasp(ee_objs)
        ee_d = _obj_dist(ee_grasp or ee_nav)

        grasp_tgt = self._update_grasp_lock(ee_grasp or ee_nav, ee_d, rp, ry, arm_q)
        phase = "grasp" if (grasp_tgt and ee_d < GRASP_PHASE_DIST_M) else "approach"

        if phase == "grasp" and ee_near < HEAD_DISABLE_DIST_M:
            head_objs = []

        ee_objs = [stabilize_ee_nav_pose(o) for o in ee_objs]
        ee_objs = _inject_head_nav_into_ee(ee_objs, head_nav, phase, ee_near)

        nav_cam, nav_objs, nav_tgt = _pick_nav_target(
            ee_nav, head_nav, ee_objs, head_objs, ee_near,
        )
        if phase == "grasp":
            nav_cam, nav_objs, nav_tgt = "ee", ee_objs, ee_nav or grasp_tgt

        use_grasp = phase == "grasp" and grasp_tgt is not None
        ee_list = [_object_summary(o, "ee") for o in ee_objs]
        head_list = [_object_summary(o, "head") for o in head_objs]

        return {
            "roles": {"ee": "nav_grasp", "head": "nav"},
            "navigation": {"camera": nav_cam, "target": nav_tgt, "objects_detailed": nav_objs},
            "target_nav": nav_tgt,
            "objects_nav": nav_objs,
            "ee_objects": ee_objs,
            "ee_objects_list": ee_list,
            "grasp": {"camera": "ee", "target": grasp_tgt, "objects_detailed": ee_objs},
            "target_grasp": grasp_tgt,
            "objects_grasp": ee_objs,
            "head_objects": head_objs,
            "head_objects_list": head_list,
            "target": grasp_tgt if use_grasp else nav_tgt,
            "objects_remaining": ee_list + head_list,
            "active_camera": "ee" if use_grasp else nav_cam,
            "phase": phase,
            "grasp_reliable": bool(grasp_tgt and grasp_tgt.get("grasp_reliable")),
            "grasp_locked": bool(grasp_tgt and grasp_tgt.get("grasp_locked")),
            "head_dist_m": _obj_dist(head_nav),
            "ee_dist_m": ee_d,
            "nav_depth_m": None if nav_tgt is None else nav_tgt.get("nav_depth_m"),
            "nav_yaw_rel": None if nav_tgt is None else nav_tgt.get("nav_yaw_rel"),
            "world_reliable": bool(nav_tgt and nav_tgt.get("world_reliable")),
            "motion_level": motion,
            "depth_stats": head_meta.get("depth_stats") or depth_stats(h_depth),
            "ee_depth_stats": ee_stats,
            "bin": {
                "center_world": BIN_CENTER.tolist(),
                "radius_m": float(BIN_RADIUS),
                "dist_to_robot": float(np.linalg.norm(self.robot_pos[:2] - BIN_CENTER[:2])),
            },
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": self.robot_pos.tolist(), "yaw": self.robot_yaw},
        }

    def get_debug(self, camera: str, name: str):
        pipe = self.head if camera == "head" else self.ee
        return pipe.get_debug(name)
