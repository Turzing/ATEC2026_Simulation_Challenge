"""
RGB-D 双摄像头感知 — 三段式 + 优先权

阶段 (距离迟滞):
    far_ee   (>= ~1.28m) → head 主导导航
    near_head(< ~1.28m)  → head 主导导航
    grasp    (< 1.10m)    → EE 独占抓取

ee_objects: 给 solution_rl (approach 固定读 ee); head 目标 mirror 写入
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BIN_CENTER,
    BIN_RADIUS,
    CLASS_FLIP_CONF,
    DETECTION_COAST_FRAMES,
    EE_PHANTOM_HEAD_GAP_M,
    EE_PHANTOM_NEAR_M,
    MIN_LOCK_POINT_COUNT,
    MIN_NAV_LOCK_CONF,
    POS_JUMP_REJECT_FAR_M,
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
    bbox_lateral_consistent,
    compensate_grasp_for_gripper_base,
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    depth_stats,
    filter_plausible_objects,
    head_nav_pos_confidence,
    is_ee_sky_blob,
    is_head_edge_phantom,
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
GRASP_APPROACH_DIST_M = 1.22   # head/lock 近距即请求 grasp 阶段
GRASP_LOCK_DIST_M = 1.22
GRASP_UNLOCK_DIST_M = 1.50
HEAD_DISABLE_DIST_M = 1.05
NAV_EE_FAR_MIN_M = 1.35          # >= 远距 EE 导航
NAV_EE_TO_HEAD_M = 1.28          # 迟滞: 远→近
NAV_HEAD_TO_EE_M = 1.42          # 迟滞: 近→远
NAV_LOCK_MISS_MAX = 45           # 丢失多少帧后解锁重选 (~0.9s)
EE_NAV_LOCK_MAX_DEPTH_M = 2.05   # EE 独锁最大深度 (log: 2.36m 偏 1.55m)
EE_ONLY_HEAD_CONFIRM_MAX = 30    # EE-only 锁无 head 确认则解锁 (~0.6s)
NAV_RELOCK_MAX_XY_M = 1.25       # 重锁不得离上一 lock_world 超过此距
TARGET_MATCH_RADIUS = 0.55       # 判定同一导航目标
HEAD_MIRROR_EE_MIN_M = 0.85
TEMPORAL_MEDIAN_N = 6
GRASP_TEMPORAL_N = 10
MOTION_FREEZE_THRESH = 0.35

# 远/近距导航均 head 主导 (EE 侧视远距假检太多); grasp 才 EE
NAV_AUTHORITY = {"far_ee": "head", "near_head": "head", "grasp": "ee"}
NAV_FALLBACK = {"far_ee": "ee", "near_head": "ee", "grasp": None}


def _obj_dist(obj: Optional[dict]) -> float:
    if not obj:
        return 999.0
    d = obj.get("depth_m")
    if d is not None and d > 0.05:
        return float(d)
    return float(obj.get("dist_to_robot") or 999.0)


def _nav_dist_conservative(obj: Optional[dict]) -> float:
    """motion 用 dist_to_robot 滤波; grasp 门控取 depth 与 dist 较大值防假近."""
    if not obj:
        return 999.0
    d_depth = _obj_dist(obj)
    d_robot = float(obj.get("dist_to_robot") or 0.0)
    if d_robot > 0.08:
        return max(d_depth, d_robot)
    return d_depth


def _synthesize_grasp_from_head(obj: dict, robot_pos, robot_yaw, arm_q) -> dict:
    """近距无 EE grasp 时, 直接沿用 head 3D 点生成 grasp (不再走 EE 外参重投影)."""
    from config import GRASP_DEPTH_OFFSET

    out = dict(obj)
    pr = out.get("pos_robot")
    if pr is None:
        return out
    pr_np = np.asarray(pr, dtype=np.float32).reshape(3)
    grasp_r = pr_np.copy()
    grasp_r[2] = float(pr_np[2]) - float(GRASP_DEPTH_OFFSET)
    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = _robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["grasp_offset_robot"] = [0.0, 0.0, -float(GRASP_DEPTH_OFFSET)]
    out["camera"] = "ee"
    out["source_camera"] = "head"
    out["nav_from_head"] = True
    out["grasp_reliable"] = True
    out["role"] = "nav_grasp"
    return compensate_grasp_for_gripper_base(out, robot_pos, robot_yaw)


def _close_dist(obj: Optional[dict]) -> float:
    """grasp 门控用保守距离: depth 与 dist 取较大值, 防假近 depth 提前蹲下."""
    return _nav_dist_conservative(obj)


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
        self._miss: Dict[Tuple[str, int], int] = {}

    def reset(self):
        self._hist.clear()
        self._grasp_hist.clear()
        self._miss.clear()

    def _blend_track(
        self,
        o: dict,
        key: Tuple[str, int],
        cam: str,
        robot_pos,
        robot_yaw,
        motion: float,
    ) -> dict:
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
            # depth 历史中值若明显小于 pos 水平距 → 同步, 避免假近 depth 触发 grasp
            d_r = float(m["dist_to_robot"])
            d_m = float(m.get("depth_m") or d_r)
            if d_r - d_m > 0.42:
                m["depth_m"] = d_r
                m["nav_depth_m"] = d_r

        shaky = motion > MOTION_FREEZE_THRESH
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
                and m["pos_confidence"] >= MIN_NAV_POS_CONF * 0.40
            )
        m["coast_frame"] = False
        return m

    def apply(
        self,
        objects: List[dict],
        cam: str,
        robot_pos,
        robot_yaw,
        motion: float = 0.0,
    ) -> List[dict]:
        out = []
        seen: set = set()
        for o in objects:
            key = (cam, int(o["id"]))
            seen.add(key)
            self._miss[key] = 0
            out.append(self._blend_track(o, key, cam, robot_pos, robot_yaw, motion))

        for key, h in list(self._hist.items()):
            if key[0] != cam or key in seen or not h:
                continue
            self._miss[key] = self._miss.get(key, 0) + 1
            if self._miss[key] > DETECTION_COAST_FRAMES:
                self._hist.pop(key, None)
                self._grasp_hist.pop(key, None)
                self._miss.pop(key, None)
                continue
            coast = dict(h[-1])
            coast["coast_frame"] = True
            pr = coast.get("pos_robot")
            pw = coast.get("pos_world")
            if pw is not None:
                pw_np = np.asarray(pw, dtype=np.float32)
                pr_new = world_to_robot_frame(pw_np, robot_pos, robot_yaw)
                coast["pos_robot"] = pr_new.tolist()
                coast["dist_to_robot"] = float(np.linalg.norm(pr_new[:2]))
                coast["yaw_rel"] = float(np.arctan2(pr_new[1], pr_new[0]))
                coast["nav_yaw_rel"] = coast["yaw_rel"]
            elif pr is not None:
                pr_np = np.asarray(pr, dtype=np.float32)
                coast["dist_to_robot"] = float(np.linalg.norm(pr_np[:2]))
                coast["yaw_rel"] = float(np.arctan2(pr_np[1], pr_np[0]))
                coast["nav_yaw_rel"] = coast["yaw_rel"]
            out.append(coast)
        return out


class _TrackClassStable:
    """ByteTrack id 跨类复用时保持上一帧 class, 防 banana↔mustard 乱跳."""

    def __init__(self):
        self._class: Dict[Tuple[str, int], str] = {}

    def reset(self):
        self._class.clear()

    def apply(self, objects: List[dict], cam: str) -> List[dict]:
        out: List[dict] = []
        for o in objects:
            m = dict(o)
            oid = int(m["id"])
            key = (cam, oid)
            cls = m.get("class")
            prev = self._class.get(key)
            if prev and cls and cls != prev:
                conf = float(m.get("pos_confidence") or head_nav_pos_confidence(m))
                if conf < CLASS_FLIP_CONF:
                    m["class"] = prev
                    m["class_stabilized"] = True
                else:
                    self._class[key] = cls
            elif cls:
                self._class[key] = cls
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


def _is_ee_phantom_near(ee_o: dict, head_objs: List[dict]) -> bool:
    """EE 侧视假近距：仅天空框 / 同 id 异类 / head 明显更远."""
    if is_ee_sky_blob(ee_o):
        return True
    ee_d = _obj_dist(ee_o)
    if ee_d > EE_PHANTOM_NEAR_M:
        return False
    ee_cls = ee_o.get("class")
    ee_id = ee_o.get("id")
    for ho in head_objs:
        hid = ho.get("id")
        if ee_id is not None and hid is not None and int(ee_id) == int(hid):
            if ho.get("class") != ee_cls:
                return True
        if _world_xy_dist(ee_o, ho) >= 0.55:
            continue
        if ho.get("class") == ee_cls:
            return False
        hc = float(ho.get("conf") or 0.0)
        ec = float(ee_o.get("conf") or 0.0)
        return hc + 0.03 >= ec
    if head_objs:
        min_head_d = min(_obj_dist(h) for h in head_objs)
        if min_head_d > ee_d + EE_PHANTOM_HEAD_GAP_M:
            return True
        if ee_d < 2.0 and min_head_d > ee_d + 0.20:
            return True
    return ee_d < 1.50 and not bool(ee_o.get("world_reliable"))


def _filter_phantom_ee(ee_objs: List[dict], head_objs: List[dict]) -> List[dict]:
    kept = []
    for eo in ee_objs:
        if _is_ee_phantom_near(eo, head_objs):
            continue
        kept.append(eo)
    return kept


def _nav_lock_rank(o: dict, prefer_head: bool = True) -> tuple:
    d = _obj_dist(o)
    is_head = o.get("camera") == "head"
    cam_pen = 0 if (prefer_head and is_head) else 1
    ee_pen = 1.2 if (not is_head and d < 2.0) else 0.0
    conf = float(o.get("pos_confidence") or head_nav_pos_confidence(o))
    return (cam_pen, d + ee_pen, -conf)


def _best_nav_target(
    objs: List[dict],
    lock_id: Optional[int] = None,
    lock_class: Optional[str] = None,
    lock_world: Optional[List[float]] = None,
) -> Optional[dict]:
    if not objs:
        return None
    locked = _find_in_pool(objs, lock_id, lock_class, lock_world)
    if locked is not None:
        return _enrich_nav(locked)
    return _enrich_nav(min(objs, key=lambda o: _nav_lock_rank(o)))


def _grasp_quality(obj: dict) -> float:
    q = _nav_quality(obj)
    if obj.get("grasp_reliable"):
        q += 120.0
    if obj.get("grasp_quat_world"):
        q += 40.0
    return q


def _bbox_center_penalty(obj: dict, img_w: float = 640.0, img_h: float = 480.0) -> float:
    """bbox 中心离画面中心越远惩罚越大 (EE 近距抓取优先居中目标)."""
    uv = obj.get("centroid_uv") or obj.get("centroid")
    if uv is None:
        return 1.0
    try:
        u, v = float(uv[0]), float(uv[1])
    except (TypeError, ValueError, IndexError):
        return 1.0
    return float(np.hypot(u - img_w * 0.5, v - img_h * 0.55) / max(img_w, img_h))


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
        same = [o for o in objs if o.get("class") == ref.get("class")]
        if same:
            near = [o for o in same if _obj_dist(o) < 1.35]
            pool = near if near else same
            return min(pool, key=lambda o: (_bbox_center_penalty(o), _obj_dist(o)))
    pool = [o for o in objs if o.get("grasp_reliable")] or objs
    scored = [(_grasp_quality(o), _bbox_center_penalty(o), _obj_dist(o), o) for o in pool]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored[0][3]


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


def _strict_lock_match(a: Optional[dict], b: Optional[dict]) -> bool:
    if a is None or b is None:
        return False
    if a.get("class") and b.get("class") and a.get("class") != b.get("class"):
        return False
    aid, bid = a.get("id"), b.get("id")
    if aid is not None and bid is not None:
        return int(aid) == int(bid)
    return _same_nav_target(a, b, radius=TARGET_MATCH_RADIUS * 0.65)


def _is_head_nav_unreliable(obj: dict) -> bool:
    """拒 coast/跳变/边缘假检/bbox-3D 横向不一致."""
    if obj.get("coast_frame"):
        return False
    if obj.get("pos_jump_rejected"):
        return True
    if is_head_edge_phantom(obj):
        return True
    depth = float(obj.get("depth_m") or obj.get("nav_depth_m") or 99.0)
    if depth < 2.5 and not bbox_lateral_consistent(obj):
        return True
    return False


def _find_in_pool_by_id(pool: List[dict], lock_id: Optional[int]) -> Optional[dict]:
    if lock_id is None:
        return None
    for o in pool:
        if int(o.get("id", -1)) == int(lock_id):
            return o
    return None


def _find_in_pool(
    pool: List[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
    lock_world: Optional[List[float]] = None,
) -> Optional[dict]:
    if lock_id is not None:
        cands = [o for o in pool if int(o.get("id", -1)) == int(lock_id)]
        if lock_class:
            cands = [o for o in cands if o.get("class") == lock_class]
        if cands:
            if lock_world is not None:
                ref = {"pos_world": lock_world}
                cands.sort(key=lambda o: _world_xy_dist(o, ref))
                best = cands[0]
                if _world_xy_dist(best, ref) > POS_JUMP_REJECT_FAR_M:
                    return None
                return best
            return cands[0]
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


def _head_confirms_lock(
    head_objs: List[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
) -> bool:
    if lock_id is None or not head_objs:
        return False
    return _find_in_pool(head_objs, lock_id, lock_class, None) is not None


def _find_locked_target(
    head_objs: List[dict],
    ee_objs: List[dict],
    lock_id: Optional[int],
    lock_class: Optional[str],
    lock_world: Optional[List[float]] = None,
    ee_only_lock: bool = False,
) -> Optional[dict]:
    hit = _find_in_pool(head_objs, lock_id, lock_class, lock_world)
    if hit is not None:
        return hit
    if lock_world is not None and not ee_only_lock:
        return None
    return _find_in_pool(
        ee_objs, lock_id, lock_class, None if ee_only_lock else lock_world,
    )


def _coast_nav_from_lock(
    lock_id: int,
    lock_class: Optional[str],
    lock_world: List[float],
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    pw = np.asarray(lock_world, dtype=np.float32)
    pr = world_to_robot_frame(pw, robot_pos, robot_yaw)
    dist = float(np.linalg.norm(pr[:2]))
    out = {
        "id": int(lock_id),
        "class": lock_class,
        "pos_world": pw.tolist(),
        "pos_robot": pr.tolist(),
        "dist_to_robot": dist,
        "nav_depth_m": dist,
        "depth_m": dist,
        "source_camera": "lock_coast",
        "nav_coast": True,
        "world_reliable": True,
        "pos_confidence": 0.55,
    }
    return _enrich_nav(out) or out


def _acquire_nav_lock(
    head_objs: List[dict],
    ee_objs: List[dict],
    head_nav: Optional[dict],
    ee_nav: Optional[dict],
    head_d: float,
    ee_d: float,
    prefer_class: Optional[str] = None,
    prefer_world: Optional[List[float]] = None,
) -> Optional[dict]:
    """新锁：head 优先；剔除 EE 假近与同 id 异类."""

    def _near_preferred(o: dict) -> bool:
        if prefer_world is None:
            return True
        return _world_xy_dist(o, {"pos_world": prefer_world}) <= NAV_RELOCK_MAX_XY_M

    def _eligible(o: dict, from_ee: bool) -> bool:
        if from_ee:
            if _is_ee_phantom_near(o, head_objs):
                return False
            if _is_far_ee_nav_unreliable(o):
                return False
            depth = float(o.get("depth_m") or o.get("nav_depth_m") or 99.0)
            if depth > EE_NAV_LOCK_MAX_DEPTH_M or not bool(o.get("world_reliable")):
                return False
        else:
            if _is_head_nav_unreliable(o):
                return False
            conf = float(o.get("pos_confidence") or head_nav_pos_confidence(o))
            if conf < MIN_NAV_LOCK_CONF * 0.35:
                return False
        return True

    head_cands = [o for o in head_objs if _eligible(o, False) and _near_preferred(o)]
    if prefer_class and head_cands:
        same_cls = [o for o in head_cands if o.get("class") == prefer_class]
        if same_cls:
            head_cands = same_cls
    if head_cands:
        best = min(head_cands, key=lambda o: _nav_lock_rank(o))
        out = dict(best)
        out["pos_confidence"] = float(best.get("pos_confidence") or head_nav_pos_confidence(best))
        return out

    ee_cands = [o for o in ee_objs if _eligible(o, True) and _near_preferred(o)]
    if prefer_class and ee_cands:
        same_cls = [o for o in ee_cands if o.get("class") == prefer_class]
        if same_cls:
            ee_cands = same_cls
    if ee_cands:
        best = min(ee_cands, key=lambda o: _nav_lock_rank(o, prefer_head=False))
        out = dict(best)
        out["pos_confidence"] = float(best.get("pos_confidence") or head_nav_pos_confidence(best))
        return out

    if prefer_world is not None:
        return None

    if head_nav is not None and _eligible(head_nav, False):
        return head_nav
    if ee_nav is not None and not _is_ee_phantom_near(ee_nav, head_objs):
        if not _is_far_ee_nav_unreliable(ee_nav):
            ed = float(ee_nav.get("depth_m") or ee_nav.get("nav_depth_m") or 99.0)
            if ed <= EE_NAV_LOCK_MAX_DEPTH_M and bool(ee_nav.get("world_reliable")):
                return ee_nav
    return None


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
    has_lock = lock_id is not None

    tgt = _find_in_pool(pools[primary], lock_id, lock_class, lock_world)
    if tgt is None and not has_lock:
        tgt = navs[primary]
    if tgt is not None:
        if primary == "ee" and _is_ee_phantom_near(tgt, head_objs):
            fb = _find_in_pool(pools["head"], lock_id, lock_class, lock_world)
            if fb is None and not has_lock:
                fb = head_nav
            if fb is not None:
                return fb, "head", "fallback"
        return tgt, primary, "primary"

    if fallback:
        fb = _find_in_pool(pools[fallback], lock_id, lock_class, lock_world)
        if fb is None and not has_lock:
            fb = navs[fallback]
        if fb is not None:
            if fallback == "ee" and _is_ee_phantom_near(fb, head_objs):
                fb_head = _find_in_pool(pools["head"], lock_id, lock_class, lock_world)
                if fb_head is None and not has_lock:
                    fb_head = head_nav
                if fb_head is not None:
                    return fb_head, "head", "fallback"
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
    if depth > 3.20:
        return True
    if depth < 1.85:
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
    """同 id 异类 / 同位置 head 类别不一致 → 丢 EE."""
    if not head_objs:
        return ee_objs
    kept: List[dict] = []
    for eo in ee_objs:
        drop = False
        eid = eo.get("id")
        for ho in head_objs:
            hid = ho.get("id")
            if (
                eid is not None and hid is not None
                and int(eid) == int(hid)
                and eo.get("class") != ho.get("class")
            ):
                drop = True
                break
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
    mirror["camera"] = "ee"
    mirror["source_camera"] = "head"
    mirror["grasp_reliable"] = False
    mirror["nav_from_head"] = True
    return mirror


def _safe_ee_export(
    tgt: dict,
    head_objs: List[dict],
    ee_objs: List[dict],
) -> Optional[dict]:
    """导出给 motion 的 EE 条目：拒绝 phantom / 不可靠 EE."""
    out = dict(tgt)
    if out.get("nav_from_head"):
        return _make_head_mirror(out)
    if _is_ee_phantom_near(out, head_objs):
        fb = _best_nav_target(head_objs)
        return _make_head_mirror(fb) if fb is not None else None
    if _is_far_ee_nav_unreliable(out):
        pool_hit = _find_in_pool(ee_objs, out.get("id"), out.get("class"))
        if pool_hit is None:
            fb = _best_nav_target(head_objs)
            return _make_head_mirror(fb) if fb is not None else None
        out = dict(pool_hit)
    return stabilize_ee_nav_pose(out)


def _export_ee_for_motion(
    ee_objs: List[dict],
    head_objs: List[dict],
    auth_tgt: Optional[dict],
    auth_cam: str,
    nav_stage: str,
    authority_mode: str,
) -> List[dict]:
    """
    motion approach 读 ee_objects / head_objects:
    - 有 head → mirror 最近/锁定目标到 ee_objects
    - 无 head → export 最佳 EE (仅 sky 过滤)
    - grasp → EE 真检测 (grasp_reliable)
    """
    ee_objs = _drop_ee_class_conflict(ee_objs, head_objs)

    if nav_stage == "grasp":
        if auth_tgt is None:
            return []
        ee_tgt = _find_in_pool(ee_objs, auth_tgt.get("id"), auth_tgt.get("class"))
        if ee_tgt is None:
            same = [o for o in ee_objs if o.get("class") == auth_tgt.get("class")]
            if same:
                ee_tgt = min(same, key=lambda o: (_bbox_center_penalty(o), _obj_dist(o)))
        if ee_tgt is None:
            ee_tgt = auth_tgt if auth_cam == "ee" else None
        if ee_tgt is None:
            return []
        near_ok = _obj_dist(ee_tgt) < 1.35 and _bbox_center_penalty(ee_tgt) < 0.42
        if not (
            ee_tgt.get("grasp_reliable")
            or ee_tgt.get("nav_from_head")
            or near_ok
        ):
            return []
        exp = _safe_ee_export(ee_tgt, head_objs, ee_objs)
        return [exp] if exp is not None else []

    if head_objs:
        best = auth_tgt if auth_tgt is not None else min(head_objs, key=lambda o: _nav_lock_rank(o))
        if best is not None:
            return [_make_head_mirror(best)]

    if auth_tgt is not None and auth_cam == "head":
        return [_make_head_mirror(auth_tgt)]

    if ee_objs:
        pool = [o for o in ee_objs if not _is_ee_phantom_near(o, head_objs)]
        if pool:
            best_ee = min(pool, key=_obj_dist)
            exp = _safe_ee_export(best_ee, head_objs, ee_objs)
            if exp is not None:
                return [exp]
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
        if auth_cam == "head" or nav_stage in ("near_head", "far_ee"):
            return [_make_head_mirror(auth_tgt)]
        exp = _safe_ee_export(auth_tgt, head_objs, ee_objs)
        if exp is not None:
            return [exp]
    if head_nav is not None:
        return [_make_head_mirror(head_nav)]
    if head_objs:
        best = min(head_objs, key=lambda o: _nav_lock_rank(o))
        return [_make_head_mirror(best)]
    if ee_nav is not None:
        exp = _safe_ee_export(ee_nav, head_objs, ee_objs)
        if exp is not None:
            return [exp]
    if ee_objs:
        pool = [o for o in ee_objs if not _is_ee_phantom_near(o, head_objs)]
        if pool:
            best = min(pool, key=_obj_dist)
            exp = _safe_ee_export(best, head_objs, ee_objs)
            if exp is not None:
                return [exp]
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
        self._class_stable = _TrackClassStable()
        self._frozen_grasp: Optional[dict] = None
        self._nav_stage = "far_ee"
        self._nav_lock_id: Optional[int] = None
        self._nav_lock_class: Optional[str] = None
        self._nav_lock_world: Optional[List[float]] = None
        self._nav_lock_miss = 0
        self._nav_lock_ee_only = False
        self._nav_lock_ee_only_frames = 0
        self._last_ee_motion: List[dict] = []
        self._last_head_objs: List[dict] = []
        print(
            "[RgbdPureDual] simple-dual | head-nav | ee-sky-only | "
            f"coast={DETECTION_COAST_FRAMES}f"
        )

    def reset(self):
        self.ee.reset()
        self.head.reset()
        self._temporal.reset()
        self._pos_gate.reset()
        self._class_stable.reset()
        self._frozen_grasp = None
        self._nav_stage = "far_ee"
        self._nav_lock_id = None
        self._nav_lock_class = None
        self._nav_lock_world = None
        self._nav_lock_miss = 0
        self._nav_lock_ee_only = False
        self._nav_lock_ee_only_frames = 0
        self._last_ee_motion = []
        self._last_head_objs = []
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)

    def _refresh_coast_obj(self, obj: dict, robot_pos, robot_yaw) -> dict:
        out = dict(obj)
        pw = out.get("pos_world")
        if pw is not None:
            pw_np = np.asarray(pw, dtype=np.float32)
            pr = world_to_robot_frame(pw_np, robot_pos, robot_yaw)
            out["pos_robot"] = pr.tolist()
            out["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
            out["yaw_rel"] = float(np.arctan2(pr[1], pr[0]))
            out["nav_yaw_rel"] = out["yaw_rel"]
        return out

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
        ee_objs = self._class_stable.apply(ee_objs, "ee")
        head_objs = self._class_stable.apply(head_objs, "head")
        head_objs = self._temporal.apply(head_objs, "head", rp, ry, motion)
        ee_objs = self._temporal.apply(ee_objs, "ee", rp, ry, motion)
        head_objs = [_finalize_head(o, rp, ry, grav) for o in head_objs]
        ee_objs = [_finalize_ee(o, rp, ry, arm_q) for o in ee_objs]
        head_objs = [self._pos_gate.apply(o, "head", rp, ry) for o in head_objs]
        ee_objs = [self._pos_gate.apply(o, "ee", rp, ry) for o in ee_objs]

        ee_objs = filter_plausible_objects(ee_objs, "ee")
        ee_objs = _filter_phantom_ee(ee_objs, head_objs)
        ee_near = _obj_dist(_best_nav_target(ee_objs) or _best_ee_grasp(ee_objs))
        head_objs = filter_plausible_objects(head_objs, "head", ee_near_m=ee_near)
        ee_objs = _drop_ee_class_conflict(ee_objs, head_objs)

        if not head_objs and self._last_head_objs:
            head_objs = [
                self._refresh_coast_obj(o, rp, ry) for o in self._last_head_objs
            ]
        if not ee_objs and self._last_ee_motion:
            ee_objs = [
                self._refresh_coast_obj(o, rp, ry) for o in self._last_ee_motion
            ]

        head_objs.sort(key=_obj_dist)
        ee_objs.sort(key=_obj_dist)

        ee_nav = _best_nav_target(
            ee_objs, self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
        )
        head_nav = _best_nav_target(
            head_objs, self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
        )

        lock_ref = _find_locked_target(
            head_objs, ee_objs,
            self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
            ee_only_lock=self._nav_lock_ee_only,
        )
        ee_grasp = _best_ee_grasp(ee_objs, lock_ref or head_nav or ee_nav)
        ee_d = _obj_dist(ee_grasp or ee_nav)
        head_d = _obj_dist(head_nav)

        locked = lock_ref
        if locked is None and self._nav_lock_id is None:
            seed = _acquire_nav_lock(head_objs, ee_objs, head_nav, ee_nav, head_d, ee_d)
            if seed is not None and seed.get("source_camera") == "ee":
                sd = _obj_dist(seed)
                if (
                    not head_objs
                    or sd > EE_NAV_LOCK_MAX_DEPTH_M
                    or _is_far_ee_nav_unreliable(seed)
                    or not bool(seed.get("world_reliable"))
                ):
                    seed = None
                elif head_objs:
                    hc = [o for o in head_objs if o.get("class") == seed.get("class")]
                    if hc and _obj_dist(min(hc, key=_obj_dist)) < _obj_dist(seed) + 0.35:
                        seed = None
            if seed is not None:
                self._nav_lock_id = int(seed["id"])
                self._nav_lock_class = seed.get("class")
                pw = seed.get("pos_world")
                self._nav_lock_world = list(pw) if pw is not None else None
                self._nav_lock_ee_only = (
                    seed.get("source_camera") == "ee"
                    and not _head_confirms_lock(head_objs, self._nav_lock_id, self._nav_lock_class)
                )
                self._nav_lock_ee_only_frames = 0
                locked = seed

        if self._nav_lock_id is not None:
            if _head_confirms_lock(head_objs, self._nav_lock_id, self._nav_lock_class):
                self._nav_lock_ee_only = False
                self._nav_lock_ee_only_frames = 0
            elif self._nav_lock_ee_only:
                self._nav_lock_ee_only_frames += 1
            elif locked is not None and locked.get("source_camera") == "ee" and not head_objs:
                self._nav_lock_ee_only = True
                self._nav_lock_ee_only_frames = 0

        if self._nav_lock_ee_only_frames >= EE_ONLY_HEAD_CONFIRM_MAX:
            self._nav_lock_id = None
            self._nav_lock_class = None
            self._nav_lock_world = None
            self._nav_lock_miss = 0
            self._nav_lock_ee_only = False
            self._nav_lock_ee_only_frames = 0
            locked = None
            lock_ref = None

        nav_dist = _nav_dist_conservative(locked or head_nav or ee_nav)
        lock_ref_obj = locked or head_nav or ee_nav
        close_d = min(
            _close_dist(locked),
            _close_dist(head_nav),
            _close_dist(ee_grasp),
            _close_dist(ee_nav),
        )
        ee_grasp_nav = ee_grasp if _strict_lock_match(ee_grasp, lock_ref_obj) else None
        ee_grasp_ok = ee_grasp_nav is not None and bool(ee_grasp_nav.get("grasp_reliable"))
        head_close = _nav_dist_conservative(head_nav) < GRASP_APPROACH_DIST_M
        lock_dist = _nav_dist_conservative(locked or lock_ref_obj)
        want_grasp = (
            close_d < GRASP_APPROACH_DIST_M
            and lock_dist < GRASP_APPROACH_DIST_M + 0.08
            and lock_ref_obj is not None
            and (ee_grasp_ok or head_close)
        )

        nav_stage = _resolve_nav_stage(nav_dist, want_grasp, self._nav_stage)
        self._nav_stage = nav_stage
        phase = "grasp" if nav_stage == "grasp" else "approach"

        auth_tgt, auth_cam, auth_mode = _resolve_authoritative_target(
            nav_stage, head_objs, ee_objs, head_nav, ee_nav,
            self._nav_lock_id, self._nav_lock_class, self._nav_lock_world,
        )
        if auth_tgt is not None and self._nav_lock_id is not None:
            if int(auth_tgt.get("id", -1)) != int(self._nav_lock_id):
                auth_tgt = None
            elif self._nav_lock_class and auth_tgt.get("class") != self._nav_lock_class:
                auth_tgt = None

        if auth_tgt is not None:
            nav_dist = _nav_dist_conservative(auth_tgt)
            pw = auth_tgt.get("pos_world")
            if pw is not None and self._nav_lock_world is not None:
                jump = float(
                    np.linalg.norm(
                        np.asarray(pw, dtype=np.float32)[:2] - np.asarray(self._nav_lock_world, dtype=np.float32)[:2]
                    )
                )
                if jump > POS_JUMP_REJECT_FAR_M:
                    auth_tgt = dict(auth_tgt)
                    auth_tgt["pos_world"] = list(self._nav_lock_world)
                    auth_tgt["pos_jump_rejected"] = True
            if self._nav_lock_id is None:
                self._nav_lock_id = int(auth_tgt["id"])
                self._nav_lock_class = auth_tgt.get("class")
            pw = auth_tgt.get("pos_world")
            if pw is not None and not auth_tgt.get("pos_jump_rejected"):
                self._nav_lock_world = list(pw)
            self._nav_lock_miss = 0
        else:
            self._nav_lock_miss += 1

        if self._nav_lock_miss >= NAV_LOCK_MISS_MAX:
            prev_class = self._nav_lock_class
            prev_world = list(self._nav_lock_world) if self._nav_lock_world is not None else None
            prev_id = self._nav_lock_id
            self._nav_lock_id = None
            self._nav_lock_class = None
            self._nav_lock_world = None
            self._nav_lock_miss = 0
            seed = _acquire_nav_lock(
                head_objs, ee_objs, head_nav, ee_nav, head_d, ee_d,
                prefer_class=prev_class, prefer_world=prev_world,
            )
            if seed is not None:
                self._nav_lock_id = int(seed["id"])
                self._nav_lock_class = seed.get("class")
                pw = seed.get("pos_world")
                self._nav_lock_world = list(pw) if pw is not None else None
            elif prev_id is not None and prev_world is not None:
                self._nav_lock_id = int(prev_id)
                self._nav_lock_class = prev_class
                self._nav_lock_world = prev_world
                self._nav_lock_miss = NAV_LOCK_MISS_MAX // 2

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
        if nav_stage == "grasp" and grasp_tgt is None and auth_tgt is not None:
            grasp_tgt = _synthesize_grasp_from_head(auth_tgt, rp, ry, arm_q)

        if nav_stage == "grasp" and ee_near < HEAD_DISABLE_DIST_M:
            pass  # 保留 head_objects 供 motion fallback

        ee_objs_raw = list(ee_objs)
        ee_motion = _export_ee_for_motion(
            ee_objs, head_objs, auth_tgt, auth_cam, nav_stage, auth_mode,
        )
        ee_motion = _ensure_ee_motion_export(
            ee_motion, auth_tgt, auth_cam, head_nav, ee_nav,
            head_objs, ee_objs_raw, nav_stage,
        )

        if head_objs:
            self._last_head_objs = [dict(o) for o in head_objs]
        if ee_motion:
            self._last_ee_motion = [dict(o) for o in ee_motion]

        ee_objs = ee_motion

        if (
            auth_tgt is None
            and self._nav_lock_id is not None
            and self._nav_lock_world is not None
            and self._nav_lock_miss < NAV_LOCK_MISS_MAX
        ):
            auth_tgt = _coast_nav_from_lock(
                self._nav_lock_id,
                self._nav_lock_class,
                self._nav_lock_world,
                rp,
                ry,
            )
            auth_cam = "lock_coast"
            auth_mode = "coast"

        nav_cam, nav_objs, nav_tgt = _navigation_for_stage(
            nav_stage, auth_tgt, auth_cam, ee_motion, ee_objs_raw, head_objs, grasp_tgt,
        )
        if nav_tgt is None and auth_tgt is not None:
            nav_tgt = auth_tgt
        if nav_tgt is None and locked is not None:
            nav_tgt = locked
        if (
            nav_tgt is not None
            and self._nav_lock_world is not None
            and nav_tgt.get("pos_world") is not None
        ):
            jump = float(
                np.linalg.norm(
                    np.asarray(nav_tgt["pos_world"], dtype=np.float32)[:2]
                    - np.asarray(self._nav_lock_world, dtype=np.float32)[:2]
                )
            )
            if jump > POS_JUMP_REJECT_FAR_M:
                nav_tgt = _coast_nav_from_lock(
                    int(self._nav_lock_id),
                    self._nav_lock_class,
                    self._nav_lock_world,
                    rp,
                    ry,
                )
                nav_tgt["pos_jump_rejected"] = True
        if (
            nav_tgt is None
            and self._nav_lock_id is not None
            and self._nav_lock_world is not None
        ):
            nav_tgt = _coast_nav_from_lock(
                self._nav_lock_id,
                self._nav_lock_class,
                self._nav_lock_world,
                rp,
                ry,
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
            "nav_lock_class": self._nav_lock_class,
            "nav_lock_ee_only": self._nav_lock_ee_only,
            "nav_lock_stable": self._nav_lock_id is not None and self._nav_lock_miss == 0,
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
            "active_camera": "ee" if use_grasp else "head",
            "phase": phase,
            "grasp_reliable": bool(grasp_tgt and grasp_tgt.get("grasp_reliable")),
            "grasp_locked": bool(grasp_tgt and grasp_tgt.get("grasp_locked")),
            "head_dist_m": _obj_dist(head_nav),
            "ee_dist_m": ee_d,
            "head_count_raw": len(head_objs),
            "ee_count_raw": len(ee_objs_raw),
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
