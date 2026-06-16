"""
RGB-D 双摄像头感知 — 三段式 + 优先权

阶段 (距离迟滞):
    far_ee   (>= ~1.35m) → EE 主导导航
    near_head(< ~1.28m)  → head 主导导航
    grasp    (< 1.10m)    → EE 独占抓取 (每帧重算)

优先权 (非平权):
    NAV_AUTHORITY: 每阶段唯一主导相机
    NAV_FALLBACK:  仅 primary 丢失时降级
    nav_lock_id:   全程锁定同一目标，防 EE/head 抢目标
    ee_objects:    每帧只导出 1 条 (给 solution_rl)
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
    MIN_NAV_POS_CONF,
    _to_numpy,
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    depth_stats,
    filter_plausible_objects,
    head_nav_pos_confidence,
    parse_ee_rgbd,
    parse_head_rgbd,
    refresh_ee_object_pose,
    refresh_head_object_pose,
    refresh_locked_grasp,
    reject_pos_world_jump,
    stabilize_ee_nav_pose,
    world_to_robot_frame,
)

GRASP_PHASE_DIST_M = 1.10
GRASP_LOCK_DIST_M = 1.22
GRASP_UNLOCK_DIST_M = 1.50
HEAD_DISABLE_DIST_M = 1.05
NAV_EE_FAR_MIN_M = 1.35          # >= 远距 EE 导航
NAV_EE_TO_HEAD_M = 1.28          # 迟滞: 远→近
NAV_HEAD_TO_EE_M = 1.42          # 迟滞: 近→远
NAV_LOCK_MISS_MAX = 18           # 丢失多少帧后解锁重选
TARGET_MATCH_RADIUS = 0.55       # 判定同一导航目标
HEAD_MIRROR_EE_MIN_M = 0.85
TEMPORAL_MEDIAN_N = 6
GRASP_TEMPORAL_N = 10
MOTION_FREEZE_THRESH = 0.35

# 各阶段唯一主导相机 (非平权); 仅 primary 不可用时走 fallback
NAV_AUTHORITY = {"far_ee": "ee", "near_head": "head", "grasp": "ee"}
NAV_FALLBACK = {"far_ee": "head", "near_head": "ee", "grasp": None}


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


class _PosWorldGate:
    """跨帧 world 位置门控: 拒跳变、维持稳定输出."""

    def __init__(self):
        self._last: Dict[Tuple[str, int, str], np.ndarray] = {}

    def reset(self):
        self._last.clear()

    def apply(self, obj: dict, cam: str, robot_pos, robot_yaw) -> dict:
        cls = str(obj.get("class") or "")
        key = (cam, int(obj["id"]), cls)
        prev = self._last.get(key)
        out = reject_pos_world_jump(obj, prev, robot_pos, robot_yaw)
        pw = out.get("pos_world")
        if pw is None:
            return out
        pw_np = np.asarray(pw, dtype=np.float32)
        if not out.get("pos_jump_rejected"):
            self._last[key] = pw_np.copy()
        elif prev is not None:
            self._last[key] = prev.copy()
        out["pos_confidence"] = head_nav_pos_confidence(out) if cam == "head" else out.get("pos_confidence")
        return out


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

            uvs = [x.get("centroid_uv") for x in h if x.get("centroid_uv") is not None]
            if uvs:
                uv_med = np.median(np.stack([np.asarray(u, dtype=np.float32) for u in uvs]), axis=0)
                m["centroid_uv"] = [float(uv_med[0]), float(uv_med[1])]

            # 世界系滤波：转圈时 robot 系坐标会变，不能对 pos_robot 做 median
            worlds = [x.get("pos_world") for x in h if x.get("pos_world") is not None]
            if worlds:
                stack_w = np.stack([np.asarray(w, dtype=np.float32) for w in worlds], axis=0)
                if len(stack_w) >= 3:
                    rough = np.median(stack_w, axis=0)
                    dist_m = float(m.get("depth_m") or m.get("nav_depth_m") or 2.0)
                    lim = 0.32 if dist_m < 1.35 else 0.48
                    keep = [
                        float(np.linalg.norm(stack_w[i, :2] - rough[:2])) <= lim
                        for i in range(len(stack_w))
                    ]
                    if any(keep):
                        stack_w = stack_w[np.asarray(keep, dtype=bool)]
                med_w = np.median(stack_w, axis=0)
                m["pos_world"] = med_w.tolist()
                med_r = world_to_robot_frame(med_w, robot_pos, robot_yaw)
                m["pos_robot"] = med_r.tolist()
                m["dist_to_robot"] = float(np.linalg.norm(med_r[:2]))
                m["yaw_rel"] = float(np.arctan2(med_r[1], med_r[0]))
                m["nav_yaw_rel"] = m["yaw_rel"]
                m["nav_depth_m"] = float(m.get("nav_depth_m") or m.get("depth_m") or m["dist_to_robot"])

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
            if cam == "head":
                m["pos_confidence"] = head_nav_pos_confidence(m)
                m["world_reliable"] = (
                    m["world_reliable"]
                    and m["pos_confidence"] >= MIN_NAV_POS_CONF
                    and bool(m.get("pos_from_pointcloud"))
                )
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
    q += float(obj.get("pos_confidence") or head_nav_pos_confidence(obj)) * 120.0
    if obj.get("pos_jump_rejected"):
        q -= 200.0
    return q


def _best_nav_target(
    objs: List[dict],
    lock_id: Optional[int] = None,
    lock_class: Optional[str] = None,
) -> Optional[dict]:
    if not objs:
        return None
    locked = _find_in_pool(objs, lock_id, lock_class)
    if locked is not None:
        return _enrich_nav(locked)
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


def _best_ee_grasp(
    objs: List[dict],
    ref: Optional[dict] = None,
) -> Optional[dict]:
    if not objs:
        return None
    if ref is not None:
        locked = _find_in_pool(objs, ref.get("id"), ref.get("class"))
        if locked is not None:
            return locked
    pool = [o for o in objs if o.get("grasp_reliable")] or objs
    scored = [(_grasp_quality(o), _obj_dist(o), o) for o in pool]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def _world_xy_dist(a: dict, b: dict) -> float:
    pa, pb = a.get("pos_world"), b.get("pos_world")
    if pa is None or pb is None:
        return 999.0
    try:
        ax, ay = float(pa[0]), float(pa[1])
        bx, by = float(pb[0]), float(pb[1])
    except (TypeError, ValueError, IndexError):
        return 999.0
    return float(np.hypot(ax - bx, ay - by))


def _same_nav_target(a: Optional[dict], b: Optional[dict], radius: float = TARGET_MATCH_RADIUS) -> bool:
    if a is None or b is None:
        return False
    aid, bid = a.get("id"), b.get("id")
    if aid is not None and bid is not None and int(aid) == int(bid):
        return True
    if a.get("class") and a.get("class") == b.get("class"):
        return _world_xy_dist(a, b) < radius
    return _world_xy_dist(a, b) < radius * 0.75


def _find_in_pool(
    pool: List[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
    lock_world: Optional[List[float]] = None,
) -> Optional[dict]:
    if lock_id is not None:
        for o in pool:
            if int(o.get("id", -1)) != int(lock_id):
                continue
            if lock_class and o.get("class") != lock_class:
                continue
            return o
    if lock_class and lock_world is not None:
        cands = [o for o in pool if o.get("class") == lock_class]
        if cands:
            ref = {"pos_world": lock_world}
            return min(cands, key=lambda o: _world_xy_dist(o, ref))
    if lock_class:
        cands = [o for o in pool if o.get("class") == lock_class]
        if cands:
            return min(cands, key=_obj_dist)
    return None


def _find_locked_target(
    head_objs: List[dict],
    ee_objs: List[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
    lock_world: Optional[List[float]] = None,
) -> Optional[dict]:
    return (
        _find_in_pool(head_objs, lock_id, lock_class, lock_world)
        or _find_in_pool(ee_objs, lock_id, lock_class, lock_world)
    )


def _acquire_nav_lock(
    head_nav: Optional[dict],
    ee_nav: Optional[dict],
    head_d: float,
    ee_d: float,
) -> Optional[dict]:
    """无锁时单次只选一个目标，近距优先 head，且必须 pos_confidence 达标."""
    cands: List[Tuple[float, dict]] = []
    if head_nav and head_d < NAV_EE_TO_HEAD_M:
        cands.append((head_d, head_nav))
    if ee_nav and (not head_nav or ee_d <= head_d + 0.20):
        cands.append((ee_d, ee_nav))
    if head_nav and not cands:
        cands.append((head_d, head_nav))
    if ee_nav and not cands:
        cands.append((ee_d, ee_nav))
    cands.sort(key=lambda x: x[0])
    for _, c in cands:
        conf = float(c.get("pos_confidence") or head_nav_pos_confidence(c))
        if conf >= MIN_NAV_POS_CONF or c.get("pos_from_pointcloud"):
            out = dict(c)
            out["pos_confidence"] = conf
            return out
    return cands[0][1] if cands else None


def _resolve_nav_stage(nav_dist: float, want_grasp: bool, prev: str) -> str:
    if want_grasp:
        return "grasp"
    stage = prev if prev in ("far_ee", "near_head") else "far_ee"
    if stage == "far_ee":
        if nav_dist < NAV_EE_TO_HEAD_M:
            stage = "near_head"
    else:
        if nav_dist > NAV_HEAD_TO_EE_M:
            stage = "far_ee"
    return stage


def _resolve_authoritative_target(
    nav_stage: str,
    head_objs: List[dict],
    ee_objs: List[dict],
    head_nav: Optional[dict],
    ee_nav: Optional[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
    lock_world: Optional[List[float]] = None,
) -> Tuple[Optional[dict], str, str]:
    """
    按阶段唯一主导相机选目标; primary 缺失才降级 fallback，二者不同时生效。
    返回 (target, authority_camera, mode)  mode=primary|fallback|none
    """
    primary = NAV_AUTHORITY[nav_stage]
    fallback = NAV_FALLBACK[nav_stage]
    pools = {"head": head_objs, "ee": ee_objs}
    navs = {"head": head_nav, "ee": ee_nav}

    tgt = _find_in_pool(pools[primary], lock_id, lock_class, lock_world)
    if tgt is None:
        tgt = navs[primary]
    if tgt is not None and float(tgt.get("pos_confidence") or 0) >= MIN_NAV_POS_CONF * 0.7:
        return tgt, primary, "primary"
    if tgt is not None and lock_id is not None:
        return tgt, primary, "primary"

    if fallback:
        fb = _find_in_pool(pools[fallback], lock_id, lock_class, lock_world)
        if fb is None:
            fb = navs[fallback]
        if fb is not None:
            return fb, fallback, "fallback"

    return None, primary, "none"


def _navigation_for_stage(
    nav_stage: str,
    auth_tgt: Optional[dict],
    auth_cam: str,
    ee_motion: List[dict],
    ee_raw: List[dict],
    head_objs: List[dict],
    grasp_tgt: Optional[dict],
) -> Tuple[str, List[dict], Optional[dict]]:
    if nav_stage == "grasp":
        objs = ee_raw if grasp_tgt else ee_motion
        return "ee", objs, grasp_tgt or auth_tgt
    if auth_cam == "head":
        return "head", head_objs, auth_tgt
    return "ee", ee_motion, auth_tgt


def _is_far_ee_nav_unreliable(obj: dict) -> bool:
    """侧视 EE 远距假检：depth>2m 且横向过大，易锁错目标导致只转不走."""
    if obj.get("nav_from_head") or obj.get("grasp_reliable"):
        return False
    depth = float(obj.get("nav_depth_m") or obj.get("depth_m") or 0.0)
    if depth < 2.05:
        return False
    pr = obj.get("pos_robot")
    if pr is None:
        return depth > 2.8
    try:
        px, py = abs(float(pr[0])), abs(float(pr[1]))
    except (TypeError, ValueError, IndexError):
        return True
    if py > 0.85 and py > px * 0.55:
        return True
    if depth > 2.8 and not obj.get("world_reliable"):
        return True
    return False


def _drop_ee_class_conflict(ee_objs: List[dict], head_objs: List[dict], radius: float = 0.55) -> List[dict]:
    """同位置 head/EE 类别不一致 → 丢 EE（EE 侧视误分类率高）."""
    if not head_objs:
        return ee_objs
    kept: List[dict] = []
    for eo in ee_objs:
        drop = False
        for ho in head_objs:
            if _world_xy_dist(eo, ho) > radius:
                continue
            if eo.get("class") == ho.get("class"):
                continue
            hc = float(ho.get("conf") or 0.0)
            ec = float(eo.get("conf") or 0.0)
            if hc + 0.03 >= ec:
                drop = True
                break
        if not drop:
            kept.append(eo)
    return kept


def _make_head_mirror(head_nav: dict) -> dict:
    mirror = dict(head_nav)
    for k in (
        "grasp_pos_robot", "grasp_pos_world", "grasp_quat_world",
        "grasp_reliable", "grasp_locked", "grasp_offset_robot",
    ):
        mirror.pop(k, None)
    mirror["grasp_reliable"] = False
    mirror["nav_from_head"] = True
    return mirror


def _export_ee_for_motion(
    ee_objs: List[dict],
    head_objs: List[dict],
    auth_tgt: Optional[dict],
    auth_cam: str,
    nav_stage: str,
    authority_mode: str,
) -> List[dict]:
    """
    solution_rl 只读 ee_objects: 每阶段仅导出「主导相机」的单一目标，非平权多目标。
    """
    ee_objs = _drop_ee_class_conflict(ee_objs, head_objs)
    if auth_tgt is None:
        return []

    if nav_stage == "grasp":
        ee_tgt = _find_in_pool(ee_objs, auth_tgt.get("id"), auth_tgt.get("class"))
        if ee_tgt is None and auth_cam == "ee":
            ee_tgt = auth_tgt
        if ee_tgt is None:
            return []
        return [stabilize_ee_nav_pose(dict(ee_tgt))]

    if nav_stage == "near_head":
        if auth_cam == "head":
            return [_make_head_mirror(auth_tgt)]
        ee_tgt = _find_in_pool(ee_objs, auth_tgt.get("id"), auth_tgt.get("class")) or auth_tgt
        if ee_tgt and not _is_far_ee_nav_unreliable(ee_tgt):
            return [stabilize_ee_nav_pose(dict(ee_tgt))]
        return []

    # far_ee: EE 主导，仅 head fallback
    if auth_cam == "ee":
        ee_tgt = _find_in_pool(ee_objs, auth_tgt.get("id"), auth_tgt.get("class")) or auth_tgt
        if ee_tgt and not _is_far_ee_nav_unreliable(ee_tgt):
            return [stabilize_ee_nav_pose(dict(ee_tgt))]
    if authority_mode == "fallback" and auth_cam == "head":
        return [_make_head_mirror(auth_tgt)]
    return []


def _ensure_ee_motion_export(
    ee_motion: List[dict],
    auth_tgt: Optional[dict],
    auth_cam: str,
    head_nav: Optional[dict],
    ee_nav: Optional[dict],
    head_objs: List[dict],
    ee_objs: List[dict],
    nav_stage: str,
) -> List[dict]:
    """solution_rl preferred=ee: 永不让 ee_objects 为空 (head 有目标时必须 mirror)."""
    if ee_motion:
        return ee_motion
    if auth_tgt is not None:
        if auth_cam == "head" or nav_stage == "near_head":
            return [_make_head_mirror(auth_tgt)]
        if not _is_far_ee_nav_unreliable(auth_tgt):
            return [stabilize_ee_nav_pose(dict(auth_tgt))]
    if head_nav is not None and float(head_nav.get("pos_confidence") or 0) >= MIN_NAV_POS_CONF * 0.85:
        return [_make_head_mirror(head_nav)]
    if ee_nav is not None and not _is_far_ee_nav_unreliable(ee_nav):
        return [stabilize_ee_nav_pose(dict(ee_nav))]
    if head_objs:
        best = max(head_objs, key=lambda o: float(o.get("pos_confidence") or head_nav_pos_confidence(o)))
        if float(best.get("pos_confidence") or 0) >= MIN_NAV_POS_CONF * 0.75:
            return [_make_head_mirror(best)]
    if ee_objs:
        best = min(ee_objs, key=_obj_dist)
        if not _is_far_ee_nav_unreliable(best):
            return [stabilize_ee_nav_pose(dict(best))]
    return []


class RgbdPureDualPipeline:
    def __init__(self):
        self.ee = RgbdPureCamera("ee")
        self.head = RgbdPureCamera("head")
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)
        self._temporal = _TemporalMedian()
        self._pos_gate = _PosWorldGate()
        self._frozen_grasp: Optional[dict] = None
        self._nav_stage = "far_ee"
        self._nav_lock_id: Optional[int] = None
        self._nav_lock_class: Optional[str] = None
        self._nav_lock_world: Optional[List[float]] = None
        self._nav_lock_miss = 0
        print(
            "[RgbdPureDual] 3-stage + pos_gate: far=ee>head | near=head>ee | grasp=ee-only "
            f"(min_conf={MIN_NAV_POS_CONF}, ee_objects never empty if head sees target)"
        )

    def reset(self):
        self.ee.reset()
        self.head.reset()
        self._temporal.reset()
        self._pos_gate.reset()
        self._frozen_grasp = None
        self._nav_stage = "far_ee"
        self._nav_lock_id = None
        self._nav_lock_class = None
        self._nav_lock_world = None
        self._nav_lock_miss = 0
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
        *,
        force_recalc: bool = False,
    ) -> Optional[dict]:
        if ee_d > GRASP_UNLOCK_DIST_M:
            self._frozen_grasp = None
            return None
        if force_recalc:
            if ee_grasp is None:
                return None
            return _finalize_ee(ee_grasp, rp, ry, arm_q)
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
        head_objs = [self._pos_gate.apply(o, "head", rp, ry) for o in head_objs]
        ee_objs = [self._pos_gate.apply(o, "ee", rp, ry) for o in ee_objs]

        ee_objs = filter_plausible_objects(ee_objs, "ee")
        ee_near = _obj_dist(_best_nav_target(ee_objs) or _best_ee_grasp(ee_objs))
        head_objs = filter_plausible_objects(head_objs, "head", ee_near_m=ee_near)
        ee_objs = _drop_ee_class_conflict(ee_objs, head_objs)

        head_objs.sort(key=_obj_dist)
        ee_objs.sort(key=_obj_dist)

        ee_nav = _best_nav_target(ee_objs, self._nav_lock_id, self._nav_lock_class)
        head_nav = _best_nav_target(head_objs, self._nav_lock_id, self._nav_lock_class)
        lock_ref = _find_locked_target(
            head_objs, ee_objs,
            self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
        )
        ee_grasp = _best_ee_grasp(ee_objs, lock_ref or head_nav or ee_nav)
        ee_d = _obj_dist(ee_grasp or ee_nav)
        head_d = _obj_dist(head_nav)

        locked = lock_ref
        if locked is None and self._nav_lock_id is None:
            seed = _acquire_nav_lock(head_nav, ee_nav, head_d, ee_d)
            if seed is not None:
                self._nav_lock_id = int(seed["id"])
                self._nav_lock_class = seed.get("class")
                pw = seed.get("pos_world")
                self._nav_lock_world = list(pw) if pw is not None else None
                locked = seed

        nav_dist = _obj_dist(locked or head_nav or ee_nav)
        ee_grasp_nav = ee_grasp if _same_nav_target(ee_grasp, locked or head_nav or ee_nav) else None
        want_grasp = (
            nav_dist < GRASP_PHASE_DIST_M
            and ee_grasp_nav is not None
            and (locked is not None or ee_nav is not None or head_nav is not None)
        )

        nav_stage = _resolve_nav_stage(nav_dist, want_grasp, self._nav_stage)
        self._nav_stage = nav_stage
        phase = "grasp" if nav_stage == "grasp" else "approach"

        auth_tgt, auth_cam, auth_mode = _resolve_authoritative_target(
            nav_stage, head_objs, ee_objs, head_nav, ee_nav,
            self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
        )
        if auth_tgt is not None:
            nav_dist = _obj_dist(auth_tgt)
            if auth_mode == "primary":
                self._nav_lock_id = int(auth_tgt["id"])
                self._nav_lock_class = auth_tgt.get("class")
                pw = auth_tgt.get("pos_world")
                self._nav_lock_world = list(pw) if pw is not None else self._nav_lock_world
                self._nav_lock_miss = 0
            else:
                self._nav_lock_miss += 1
        else:
            self._nav_lock_miss += 1

        if self._nav_lock_miss >= NAV_LOCK_MISS_MAX:
            self._nav_lock_id = None
            self._nav_lock_class = None
            self._nav_lock_world = None
            self._nav_lock_miss = 0
            seed = _acquire_nav_lock(head_nav, ee_nav, head_d, ee_d)
            if seed is not None:
                self._nav_lock_id = int(seed["id"])
                self._nav_lock_class = seed.get("class")
                pw = seed.get("pos_world")
                self._nav_lock_world = list(pw) if pw is not None else None

        grasp_src = ee_grasp_nav if _same_nav_target(ee_grasp_nav, auth_tgt) else None
        if grasp_src is None and auth_tgt is not None:
            grasp_src = _find_in_pool(ee_objs, auth_tgt.get("id"), auth_tgt.get("class"))
        grasp_tgt = self._update_grasp_lock(
            grasp_src,
            _obj_dist(grasp_src or auth_tgt),
            rp,
            ry,
            arm_q,
            force_recalc=(nav_stage == "grasp"),
        )

        if nav_stage == "grasp" and ee_near < HEAD_DISABLE_DIST_M:
            head_objs = []

        ee_objs_raw = list(ee_objs)
        ee_motion = _export_ee_for_motion(
            ee_objs, head_objs, auth_tgt, auth_cam, nav_stage, auth_mode,
        )
        ee_motion = _ensure_ee_motion_export(
            ee_motion, auth_tgt, auth_cam, head_nav, ee_nav,
            head_objs, ee_objs_raw, nav_stage,
        )
        ee_objs = ee_motion

        nav_cam, nav_objs, nav_tgt = _navigation_for_stage(
            nav_stage, auth_tgt, auth_cam, ee_motion, ee_objs_raw, head_objs, grasp_tgt,
        )

        use_grasp = nav_stage == "grasp" and grasp_tgt is not None
        grasp_objs = ee_objs_raw
        if auth_tgt is not None:
            g0 = _find_in_pool(ee_objs_raw, auth_tgt.get("id"), auth_tgt.get("class"))
            grasp_objs = [g0] if g0 is not None else []
        ee_list = [_object_summary(o, "ee") for o in ee_objs]
        head_list = [_object_summary(o, "head") for o in head_objs]

        return {
            "roles": {"ee": "nav_far+grasp", "head": "nav_near"},
            "nav_stage": nav_stage,
            "nav_authority": auth_cam,
            "nav_authority_mode": auth_mode,
            "nav_lock_id": self._nav_lock_id,
            "nav_pos_confidence": None if nav_tgt is None else nav_tgt.get("pos_confidence"),
            "navigation": {"camera": nav_cam, "target": nav_tgt, "objects_detailed": nav_objs},
            "target_nav": nav_tgt,
            "objects_nav": nav_objs,
            "ee_objects": ee_objs,
            "ee_objects_list": ee_list,
            "grasp": {"camera": "ee", "target": grasp_tgt, "objects_detailed": grasp_objs},
            "target_grasp": grasp_tgt,
            "objects_grasp": grasp_objs,
            "head_objects": head_objs,
            "head_objects_list": head_list,
            "target": grasp_tgt if use_grasp else nav_tgt,
            "objects_remaining": ee_list + head_list,
            "active_camera": "ee" if use_grasp else auth_cam,
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
