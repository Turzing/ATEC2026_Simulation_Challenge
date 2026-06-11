"""
RGB-D 双摄像头感知 — 部署接口

分工:
    ee   → 远距: 列出可见黄物体 + depth_m / dist_to_robot (导航)
    head → 近距: 高精度 3D 点云 → pos_world / grasp_pos_world / grasp_quat_world

    out = RgbdPureDualPipeline().process(obs)

运动层建议:
    approach: 读 out["target_nav"] / out["ee_objects"]
    grasp:    读 out["target_grasp"]["grasp_pos_world"], ["grasp_quat_world"]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BIN_CENTER,
    BIN_RADIUS,
    PROPRIO_BASE_ANG_VEL,
    PROPRIO_BASE_LIN_VEL,
    PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    TOTAL_OBJECTS,
)
from rgbd_pure_pipeline import RgbdPureCamera, _robot_to_world, _yaw_from_gravity
from rgbd_utils import _to_numpy, depth_stats, parse_ee_rgbd, parse_head_rgbd

GRASP_PHASE_DIST_M = 1.10
TEMPORAL_MEDIAN_N = 6


def _obj_dist(obj: Optional[dict]) -> float:
    if not obj:
        return 999.0
    d = obj.get("depth_m")
    if d is not None and d > 0.05:
        return float(d)
    dr = obj.get("dist_to_robot")
    return float(dr) if dr is not None else 999.0


def _object_summary(o: dict, cam: str) -> dict:
    s = {
        "id": int(o["id"]),
        "camera": cam,
        "class": o.get("class"),
        "conf": float(o.get("conf", 0)),
        "depth_m": o.get("depth_m"),
        "dist_to_robot": o.get("dist_to_robot"),
        "pos_world": o.get("pos_world"),
        "pos_robot": o.get("pos_robot"),
        "bbox": o.get("bbox"),
    }
    if cam == "head":
        s["grasp_pos_world"] = o.get("grasp_pos_world")
        s["grasp_quat_world"] = o.get("grasp_quat_world")
        s["geom_extents"] = o.get("geom_extents")
    return s


class _TemporalMedian:
    """多帧中值平滑 depth / 3D 位姿 (移动中不必停)"""

    def __init__(self, n: int = TEMPORAL_MEDIAN_N):
        self.n = n
        self._hist: Dict[Tuple[str, int], List[dict]] = {}

    def reset(self):
        self._hist.clear()

    def apply(self, objects: List[dict], cam: str, robot_pos, robot_yaw) -> List[dict]:
        out = []
        for o in objects:
            key = (cam, int(o["id"]))
            self._hist.setdefault(key, [])
            self._hist[key].append(o)
            self._hist[key] = self._hist[key][-self.n :]
            h = self._hist[key]
            m = dict(o)

            depths = [x["depth_m"] for x in h if x.get("depth_m") is not None]
            if depths:
                m["depth_m"] = float(np.median(depths))

            prs = [x["pos_robot"] for x in h if x.get("pos_robot") is not None]
            if prs:
                med = np.median(np.stack([np.asarray(p, dtype=np.float32) for p in prs]), axis=0)
                m["pos_robot"] = med.tolist()
                m["pos_world"] = _robot_to_world(med, robot_pos, robot_yaw).tolist()
                m["dist_to_robot"] = float(np.linalg.norm(med[:2]))

            if cam == "head":
                gps = [x.get("grasp_pos_world") for x in h if x.get("grasp_pos_world")]
                if gps:
                    m["grasp_pos_world"] = np.median(
                        np.stack([np.asarray(g, dtype=np.float32) for g in gps]), axis=0,
                    ).tolist()
                gqs = [x.get("grasp_quat_world") for x in h if x.get("grasp_quat_world")]
                if gqs:
                    m["grasp_quat_world"] = np.median(
                        np.stack([np.asarray(q, dtype=np.float32) for q in gqs]), axis=0,
                    ).tolist()

            out.append(m)
        return out


def _best_head_grasp_target(objs: List[dict]) -> Optional[dict]:
    if not objs:
        return None
    scored = []
    for o in objs:
        sm = float(o.get("blob_sat_mean", 0))
        vm = float(o.get("blob_val_mean", 0))
        area = int((o["bbox"][2] - o["bbox"][0] + 1) * (o["bbox"][3] - o["bbox"][1] + 1))
        if sm < 40 and vm < 65:
            continue
        if area > 900 and sm < 46 and vm < 80:
            continue
        q = sm * 0.55 + vm * 0.25 - min(area, 4000) * 0.003 + (80.0 if o.get("grasp_quat_world") else 0)
        scored.append((q, o))
    if not scored:
        return min(objs, key=_obj_dist)
    scored.sort(key=lambda x: (-x[0], _obj_dist(x[1])))
    return scored[0][1]


def _nav_quality(obj: dict) -> float:
    sm = float(obj.get("blob_sat_mean", 50))
    vm = float(obj.get("blob_val_mean", 90))
    area = int((obj["bbox"][2] - obj["bbox"][0] + 1) * (obj["bbox"][3] - obj["bbox"][1] + 1))
    q = sm * 0.5 + vm * 0.2 - min(area, 3500) * 0.003
    if area > 1200 and sm < 50:
        q -= 80.0
    return q


def _best_nav_target(objs: List[dict]) -> Optional[dict]:
    if not objs:
        return None
    ranked = sorted(objs, key=_obj_dist)
    best = ranked[0]
    bd, bq = _obj_dist(best), _nav_quality(best)
    area0 = int((best["bbox"][2] - best["bbox"][0] + 1) * (best["bbox"][3] - best["bbox"][1] + 1))
    if area0 > 1100 and float(best.get("blob_sat_mean", 50)) < 46:
        for o in ranked[1:6]:
            if _nav_quality(o) > bq + 20:
                return o
    for o in ranked[1:]:
        if _obj_dist(o) - bd > 0.40:
            break
        area = int((o["bbox"][2] - o["bbox"][0] + 1) * (o["bbox"][3] - o["bbox"][1] + 1))
        if area > 1200 and float(o.get("blob_sat_mean", 0)) < 46:
            continue
        if _nav_quality(o) > bq + 18.0 and _obj_dist(o) < bd + 0.65:
            best, bq = o, _nav_quality(o)
    return best


class RgbdPureDualPipeline:
    def __init__(self):
        self.ee = RgbdPureCamera("ee")
        self.head = RgbdPureCamera("head")
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)
        self._temporal = _TemporalMedian()
        print("[RgbdPureDual] ee=nav(list+distance)  head=grasp(3D pose+quat)")

    def reset(self):
        self.ee.reset()
        self.head.reset()
        self._temporal.reset()
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

    def process(self, obs, dt: float = 0.02) -> dict:
        self.frame_count += 1
        self._update_robot_pose(obs, dt)
        rp, ry = self.robot_pos, self.robot_yaw

        h_rgb, h_depth = parse_head_rgbd(obs)
        head_objs, _, head_meta = self.head.process_frame(h_rgb, h_depth, rp, ry)
        head_stats = head_meta.get("depth_stats") or depth_stats(h_depth)

        ee_objs: List[dict] = []
        ee_meta: dict = {}
        ee_stats: dict = {}
        e_rgb, e_depth = parse_ee_rgbd(obs)
        if e_rgb is not None and e_depth is not None:
            ee_objs, _, ee_meta = self.ee.process_frame(e_rgb, e_depth, rp, ry)
            ee_stats = ee_meta.get("depth_stats") or depth_stats(e_depth)

        head_objs = self._temporal.apply(head_objs, "head", rp, ry)
        ee_objs = self._temporal.apply(ee_objs, "ee", rp, ry)
        head_objs.sort(key=_obj_dist)
        ee_objs.sort(key=_obj_dist)

        head_tgt = _best_head_grasp_target(head_objs)
        ee_tgt = _best_nav_target(ee_objs)
        head_d = _obj_dist(head_tgt)
        phase = "grasp" if (head_tgt is not None and head_d < GRASP_PHASE_DIST_M) else "approach"

        if phase == "approach":
            nav_cam, nav_objs, nav_tgt = "ee", ee_objs, ee_tgt
            grasp_cam, grasp_objs, grasp_tgt = "ee", ee_objs, ee_tgt
        else:
            nav_cam, nav_objs, nav_tgt = "head", head_objs, head_tgt
            grasp_cam, grasp_objs, grasp_tgt = "head", head_objs, head_tgt

        use_grasp = phase == "grasp" and grasp_tgt is not None
        target = grasp_tgt if use_grasp else nav_tgt

        ee_list = [_object_summary(o, "ee") for o in ee_objs]
        head_list = [_object_summary(o, "head") for o in head_objs]
        bin_xy = BIN_CENTER[:2]
        bin_dist = float(np.linalg.norm(self.robot_pos[:2] - bin_xy))

        return {
            "navigation": {
                "camera": nav_cam,
                "objects_detailed": nav_objs,
                "target": nav_tgt,
                "objects_count": len(nav_objs),
            },
            "target_nav": nav_tgt,
            "objects_nav": nav_objs,
            "ee_objects": ee_objs,
            "ee_objects_list": ee_list,

            "grasp": {
                "camera": grasp_cam,
                "objects_detailed": grasp_objs,
                "target": grasp_tgt,
                "objects_count": len(grasp_objs),
            },
            "target_grasp": grasp_tgt,
            "objects_grasp": grasp_objs,
            "head_objects": head_objs,
            "head_objects_list": head_list,

            "target": target,
            "objects_detailed": grasp_objs if use_grasp else nav_objs,
            "objects_remaining": ee_list if phase == "approach" else head_list,
            "active_camera": grasp_cam if use_grasp else nav_cam,
            "phase": phase,
            "head_dist_m": head_d,
            "ee_dist_m": _obj_dist(ee_tgt),

            "depth_stats": head_stats,
            "ee_depth_stats": ee_stats,
            "head_mask_components": head_meta.get("mask_components", 0),
            "ee_mask_components": ee_meta.get("mask_components", 0),
            "bin": {
                "center_world": BIN_CENTER.tolist(),
                "radius_m": float(BIN_RADIUS),
                "dist_to_robot": bin_dist,
            },
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": self.robot_pos.tolist(), "yaw": self.robot_yaw},
        }

    def get_debug(self, camera: str, name: str):
        pipe = self.head if camera == "head" else self.ee
        return pipe.get_debug(name)
