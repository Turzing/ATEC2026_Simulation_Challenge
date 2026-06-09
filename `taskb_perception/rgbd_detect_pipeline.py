"""
Task B 识别 — 官方 RGB-D 融合 (head_rgb + head_depth)

官网给的观测:
    obs['image']['head_rgb']    uint8  RGB
    obs['image']['head_depth']  float32 米

检测逻辑 (不用 YOLO):
    1. depth → 局部地面 + relief (比地面近的凸起)
    2. rgb   → 比灰地面更饱和 (Task B 物体有颜色、地面灰)
    3. 两者 AND → 连通域 → bbox
    4. bbox 内 depth 中值 → 距离

仿真测试:
    python test_rgbd_detect.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HEAD_CAM, ROBOT_INIT_POS, ROBOT_INIT_YAW, TOTAL_OBJECTS
from rgbd_utils import (
    bbox_center_depth,
    depth_stats,
    median_depth_in_mask,
    parse_head_rgbd,
    sanitize_depth,
)

# =============================================================================
# 可调参数
# =============================================================================
DEPTH_MIN = 0.35
DEPTH_MAX = 12.0
GROUND_OPEN_K = 25
RELIEF_MIN = 0.022          # 比周围地面近 2.2cm+

ROI_V_MIN, ROI_V_MAX = 0.18, 0.86
ROI_U_MARGIN = 0.05

# RGB: 地面灰、物体有颜色 — 饱和度高于地面参考
SAT_MIN_ABSOLUTE = 22
SAT_ABOVE_GROUND = 12       # 比 ROI 地面 sat 分位数高多少

MIN_BLOB_AREA = 120
MAX_BLOB_AREA = 35000
MIN_BLOB_SIDE = 6
MAX_BLOB_SIDE = 300
MAX_DETECTIONS = 12

TRACK_MATCH_PX = 90
TRACK_MAX_AGE = 15
MIN_TRACK_HITS = 2          # 识别阶段可改为 1


class Tracker2D:
    def __init__(self):
        self.tracks: Dict[int, dict] = {}
        self._next = 0

    def reset(self):
        self.tracks.clear()
        self._next = 0

    def update(self, dets: List[dict]) -> List[dict]:
        for t in self.tracks.values():
            t["age"] += 1
        assigned = set()
        out = []
        for d in dets:
            cx, cy = d["centroid"]
            best_id, best_d = None, 1e9
            for tid, t in self.tracks.items():
                if tid in assigned:
                    continue
                tcx, tcy = t["centroid"]
                dist = float(np.hypot(cx - tcx, cy - tcy))
                if dist < TRACK_MATCH_PX and dist < best_d:
                    best_d, best_id = dist, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {"centroid": (cx, cy), "age": 0, "hits": 0}
            tr = self.tracks[best_id]
            assigned.add(best_id)
            ocx, ocy = tr["centroid"]
            tr["centroid"] = (0.6 * ocx + 0.4 * cx, 0.6 * ocy + 0.4 * cy)
            tr["age"] = 0
            tr["hits"] = tr.get("hits", 0) + 1
            out.append({**d, "track_id": best_id})
        for tid in [k for k, v in self.tracks.items() if v["age"] > TRACK_MAX_AGE]:
            del self.tracks[tid]
        return out


class RgbdDetectPipeline:
    """官方 RGB + D 融合 2D 检测"""

    def __init__(self):
        self.tracker = Tracker2D()
        self.frame_count = 0
        self._debug: Dict[str, np.ndarray] = {}
        self._depth_stats: Dict[str, float] = {}
        print("[RgbdDetectPipeline] head_rgb + head_depth fusion")

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0
        self._debug.clear()

    def _roi(self, h: int, w: int) -> np.ndarray:
        u0, u1 = int(w * ROI_U_MARGIN), int(w * (1 - ROI_U_MARGIN))
        v0, v1 = int(h * ROI_V_MIN), int(h * ROI_V_MAX)
        m = np.zeros((h, w), dtype=np.uint8)
        m[v0:v1, u0:u1] = 255
        return m

    def _ground_depth(self, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        d = depth.copy()
        d[~valid] = DEPTH_MAX
        k = GROUND_OPEN_K
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        g = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel)
        g[~valid] = 0
        return g

    def build_masks(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = depth.shape
        valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        roi = self._roi(h, w) > 0

        ground = self._ground_depth(depth, valid)
        relief = np.zeros_like(depth, dtype=np.float32)
        ok_g = ground > 0
        relief[ok_g] = ground[ok_g] - depth[ok_g]
        relief_mask = (relief >= RELIEF_MIN) & valid & roi

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        roi_sat = sat[roi & valid]
        if roi_sat.size > 100:
            ground_sat = float(np.percentile(roi_sat, 35))
        else:
            ground_sat = 30.0
        sat_thresh = max(SAT_MIN_ABSOLUTE, ground_sat + SAT_ABOVE_GROUND)
        color_mask = (sat >= sat_thresh) & roi

        # 核心: depth 凸起 AND (有颜色 OR 凸起很强)
        strong_relief = relief >= (RELIEF_MIN * 2.2)
        rgbd_mask = (relief_mask & (color_mask | strong_relief)).astype(np.uint8)
        rgbd_mask = cv2.morphologyEx(rgbd_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        rgbd_mask = cv2.morphologyEx(rgbd_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        self._debug = {
            "relief": np.clip(relief / 0.15 * 255, 0, 255).astype(np.uint8),
            "color": color_mask.astype(np.uint8) * 255,
            "rgbd": rgbd_mask * 255,
            "valid_depth": (valid & roi).astype(np.uint8) * 255,
        }
        self._depth_stats = depth_stats(depth)
        self._max_relief = float(np.max(relief[roi & valid])) if np.any(roi & valid) else 0.0
        return rgbd_mask, relief_mask.astype(np.uint8), color_mask.astype(np.uint8)

    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> List[dict]:
        depth = sanitize_depth(depth)
        mask, _, _ = self.build_masks(rgb, depth)
        labeled, n = ndimage.label(mask > 0)
        if n == 0:
            return []

        dets = []
        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            area = len(ys)
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            if min(bw, bh) < MIN_BLOB_SIDE or max(bw, bh) > MAX_BLOB_SIDE:
                continue

            bbox = [x1, y1, x2, y2]
            d_m = bbox_center_depth(depth, bbox, pad=1)
            if d_m is None:
                d_m = median_depth_in_mask(depth, (labeled == cid).astype(np.uint8))

            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            conf = float(min(0.95, 0.5 + area / 3000.0 + (0.1 if d_m else 0)))

            dets.append({
                "class": "object",
                "conf": conf,
                "bbox": bbox,
                "centroid": (cx, cy),
                "depth_m": d_m,
                "area": area,
                "source": "rgbd_fusion",
            })

        dets.sort(key=lambda x: (x["depth_m"] if x["depth_m"] is not None else 999, -x["area"]))
        return dets[:MAX_DETECTIONS]

    def get_debug(self, name: str) -> Optional[np.ndarray]:
        return self._debug.get(name)

    def get_depth_stats(self) -> Dict[str, float]:
        return dict(self._depth_stats)

    def get_max_relief(self) -> float:
        return getattr(self, "_max_relief", 0.0)

    def process(self, obs, dt: float = 0.02) -> dict:
        self.frame_count += 1
        rgb, depth = parse_head_rgbd(obs)
        raw = self.detect(rgb, depth)
        tracks = self.tracker.update(raw)
        confirmed = [
            t for t in tracks
            if self.tracker.tracks.get(t["track_id"], {}).get("hits", 0) >= MIN_TRACK_HITS
        ]

        objects = []
        target = None
        for t in confirmed:
            obj = {
                "id": int(t["track_id"]),
                "class": "object",
                "conf": float(t["conf"]),
                "bbox": t["bbox"],
                "centroid_uv": list(t["centroid"]),
                "depth_m": t.get("depth_m"),
                "dist_to_robot": t.get("depth_m"),
                "pos_world": None,
                "grasp_pos_world": None,
                "source": "rgbd_fusion",
            }
            objects.append(obj)
            if target is None or (obj["depth_m"] or 999) < (target.get("depth_m") or 999):
                target = obj

        return {
            "target": target,
            "objects_detailed": objects,
            "objects_remaining": [{"id": o["id"], "class": o["class"], "dist": o["depth_m"]} for o in objects],
            "depth_stats": self.get_depth_stats(),
            "max_relief": self.get_max_relief(),
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": ROBOT_INIT_POS.tolist(), "yaw": float(ROBOT_INIT_YAW)},
        }
