"""
老师版 RGB-D 双摄像头

    ee   → 远距导航 (视野广, 找远处目标)
    head → 近距抓取 (身边/脚下物品)

每路独立 RGB-D 融合 (depth 凸起 + RGB 黄), 不合并 track_id.

    out = RgbdPureDualPipeline().process(obs)
    out["target_nav"]     # approach: 优先 ee
    out["target_grasp"]   # grasp: head
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
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

# 仅当 head 确认近距目标时才切 grasp; 远距导航只用 ee
GRASP_PHASE_DIST_M = 1.10
TEMPORAL_MEDIAN_N = 6


class _TemporalMedian:
    """移动中多帧中值 — 不必停住再测"""

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
                gps = [x.get("grasp_pos_world") for x in h if x.get("grasp_pos_world")]
                if gps:
                    m["grasp_pos_world"] = np.median(
                        np.stack([np.asarray(g, dtype=np.float32) for g in gps]), axis=0,
                    ).tolist()
            out.append(m)
        return out


def _obj_dist(obj: Optional[dict]) -> float:
    """测距优先 depth_m (光轴), 比 dist_to_robot 稳"""
    if not obj:
        return 999.0
    d = obj.get("depth_m")
    if d is not None and d > 0.05:
        return float(d)
    dr = obj.get("dist_to_robot")
    return float(dr) if dr is not None else 999.0


def _best_nav_target(objs: List[dict]) -> Optional[dict]:
    """EE 导航: 最近目标; 深度接近时取 conf 更高"""
    if not objs:
        return None
    ranked = sorted(objs, key=_obj_dist)
    best = ranked[0]
    bd = _obj_dist(best)
    for o in ranked[1:]:
        if _obj_dist(o) - bd > 0.30:
            break
        if float(o.get("conf", 0)) > float(best.get("conf", 0)) + 0.08:
            best = o
    return best


class RgbdPureDualPipeline:
    def __init__(self):
        self.ee = RgbdPureCamera("ee")
        self.head = RgbdPureCamera("head")
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)
        self._temporal = _TemporalMedian()
        print("[RgbdPureDual] ee=nav(far)  head=grasp(near)  RGBD fusion x2")

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
        head_objs, head_tgt, head_meta = self.head.process_frame(h_rgb, h_depth, rp, ry)
        head_stats = head_meta.get("depth_stats") or depth_stats(h_depth)

        ee_objs: List[dict] = []
        ee_tgt: Optional[dict] = None
        ee_meta: dict = {}
        ee_stats: dict = {}
        e_rgb, e_depth = parse_ee_rgbd(obs)
        if e_rgb is not None and e_depth is not None:
            ee_objs, ee_tgt, ee_meta = self.ee.process_frame(e_rgb, e_depth, rp, ry)
            ee_stats = ee_meta.get("depth_stats") or depth_stats(e_depth)

        head_objs = self._temporal.apply(head_objs, "head", rp, ry)
        ee_objs = self._temporal.apply(ee_objs, "ee", rp, ry)
        head_tgt = min(head_objs, key=_obj_dist) if head_objs else None
        ee_tgt = min(ee_objs, key=_obj_dist) if ee_objs else None

        head_d = _obj_dist(head_tgt)
        # grasp: 必须 head 有近目标; 远距一律 approach + 只信 ee 导航
        phase = "grasp" if (head_tgt is not None and head_d < GRASP_PHASE_DIST_M) else "approach"

        if phase == "approach":
            nav_cam, nav_objs = "ee", ee_objs
            nav_tgt = _best_nav_target(ee_objs)
            grasp_cam, grasp_objs, grasp_tgt = "ee", ee_objs, ee_tgt
        else:
            nav_cam, nav_objs, nav_tgt = "head", head_objs, head_tgt
            grasp_cam, grasp_objs, grasp_tgt = "head", head_objs, head_tgt

        use_grasp = phase == "grasp" and grasp_tgt is not None
        target = grasp_tgt if use_grasp else nav_tgt

        return {
            "navigation": {
                "camera": nav_cam,
                "objects_detailed": nav_objs,
                "target": nav_tgt,
                "objects_count": len(nav_objs),
            },
            "target_nav": nav_tgt,
            "objects_nav": nav_objs,
            "head_objects": head_objs,
            "ee_objects": ee_objs,

            "grasp": {
                "camera": grasp_cam,
                "objects_detailed": grasp_objs,
                "target": grasp_tgt,
                "objects_count": len(grasp_objs),
            },
            "target_grasp": grasp_tgt,
            "objects_grasp": grasp_objs,

            "target": target,
            "objects_detailed": grasp_objs if use_grasp else nav_objs,
            "active_camera": grasp_cam if use_grasp else nav_cam,
            "phase": phase,
            "head_dist_m": head_d,
            "ee_dist_m": _obj_dist(ee_tgt),

            "depth_stats": head_stats,
            "ee_depth_stats": ee_stats,
            "head_mask_components": head_meta.get("mask_components", 0),
            "ee_mask_components": ee_meta.get("mask_components", 0),
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": self.robot_pos.tolist(), "yaw": self.robot_yaw},
        }

    def get_debug(self, camera: str, name: str):
        pipe = self.head if camera == "head" else self.ee
        return pipe.get_debug(name)
