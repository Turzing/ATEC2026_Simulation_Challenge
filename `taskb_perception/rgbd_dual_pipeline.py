"""
双相机 RGB-D — 分工明确, 不做融合

    head  → 导航: 远距搜目标、走路跟谁 (objects_nav / target_nav)
    ee    → 抓取: 近距对准、下爪    (objects_grasp / target_grasp)

两路每帧同时跑, 各用各的 track_id, 不强行合并成一个 G-id.

用法:
    out = RgbdDualPipeline().process(obs)
    out["target_nav"]    # head 最近目标
    out["target_grasp"]  # ee 最近目标
"""

from __future__ import annotations

from typing import List, Optional

from config import ROBOT_INIT_POS, ROBOT_INIT_YAW, TOTAL_OBJECTS
from rgbd_detect_pipeline import RgbdDetectPipeline
from rgbd_utils import parse_ee_rgbd, parse_head_rgbd

# 小于此距离 → 建议运动层用 grasp(EE), 否则用 nav(head)
GRASP_PHASE_DEPTH_M = 1.05


class RgbdDualPipeline:
    def __init__(self):
        self.head = RgbdDetectPipeline(camera="head")
        self.ee = RgbdDetectPipeline(camera="ee")
        self.frame_count = 0
        print("[RgbdDualPipeline] head=nav  ee=grasp  (no fusion)")

    def reset(self):
        self.head.reset()
        self.ee.reset()
        self.frame_count = 0

    def process(self, obs, dt: float = 0.02) -> dict:
        self.frame_count += 1

        h_rgb, h_depth = parse_head_rgbd(obs)
        nav_objs, nav_tgt, nav_meta = self.head.process_frame(h_rgb, h_depth)

        grasp_objs: List[dict] = []
        grasp_tgt: Optional[dict] = None
        grasp_meta: dict = {}
        e_rgb, e_depth = parse_ee_rgbd(obs)
        if e_rgb is not None and e_depth is not None:
            grasp_objs, grasp_tgt, grasp_meta = self.ee.process_frame(e_rgb, e_depth)

        nav_d = (nav_tgt.get("depth_m") if nav_tgt else None) or 999.0
        phase = "grasp" if nav_d < GRASP_PHASE_DEPTH_M else "approach"
        use_grasp = phase == "grasp" and grasp_tgt is not None

        return {
            # 导航层读这些
            "navigation": {
                "camera": "head",
                "objects_detailed": nav_objs,
                "target": nav_tgt,
                "objects_count": len(nav_objs),
            },
            "target_nav": nav_tgt,
            "objects_nav": nav_objs,

            # 抓取层读这些
            "grasp": {
                "camera": "ee",
                "objects_detailed": grasp_objs,
                "target": grasp_tgt,
                "objects_count": len(grasp_objs),
            },
            "target_grasp": grasp_tgt,
            "objects_grasp": grasp_objs,

            # 兼容旧字段: 走路阶段=nav, 抓取阶段=grasp (若 ee 有目标)
            "target": grasp_tgt if use_grasp else nav_tgt,
            "objects_detailed": grasp_objs if use_grasp else nav_objs,
            "active_camera": "ee" if use_grasp else "head",
            "phase": phase,

            "depth_stats": nav_meta.get("depth_stats", {}),
            "sat_thresh": nav_meta.get("sat_thresh", 0.0),
            "head_mask_components": nav_meta.get("mask_components", 0),
            "ee_mask_components": grasp_meta.get("mask_components", 0),
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": ROBOT_INIT_POS.tolist(), "yaw": float(ROBOT_INIT_YAW)},
        }

    def get_debug(self, camera: str, name: str):
        pipe = self.head if camera == "head" else self.ee
        return pipe.get_debug(name)
