"""
Task B 识别 — 饱和度 (Saturation) 为主 + depth 测距

Task B 特点: 灰地面(低饱和) vs 黄物体(高饱和) → verify.png 里 Saturation 一眼能分清

流程:
    head_rgb  → HSV 的 S 通道 → 比地面饱和度高 → 连通域 → bbox
    head_depth → bbox 内中值 → 距离 d (米)

depth 不参与「有没有物体」, 只参与距离.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROBOT_INIT_POS, ROBOT_INIT_YAW, TOTAL_OBJECTS
from rgbd_utils import (
    bbox_center_depth,
    depth_stats,
    median_depth_in_mask,
    parse_head_rgbd,
    sanitize_depth,
)

# =============================================================================
# 可调参数 — 不对就改这里
# =============================================================================
DEPTH_MIN = 0.40
DEPTH_MAX = 12.0

ROI_V_MIN, ROI_V_MAX = 0.12, 0.90
ROI_U_MARGIN = 0.04

# 饱和度: 比「地面参考」高多少才算物体 (灰地 S~20~40, 黄物体 S~80~180)
SAT_MIN_ABSOLUTE = 50          # 绝对下限
SAT_ABOVE_GROUND = 25          # 比地面参考高多少
VAL_MIN = 45                   # 太暗的像素不要 (阴影)

MIN_BLOB_AREA = 60
MAX_BLOB_AREA = 45000
MIN_BLOB_SIDE = 5
MAX_BLOB_SIDE = 340
MAX_DETECTIONS = 12

TRACK_MATCH_PX = 90
TRACK_MAX_AGE = 15
MIN_TRACK_HITS = 1


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
    """饱和度检测 + depth 距离"""

    def __init__(self):
        self.tracker = Tracker2D()
        self.frame_count = 0
        self._debug: Dict[str, np.ndarray] = {}
        self._depth_stats: Dict[str, float] = {}
        self._sat_thresh = 0.0
        self._sat_ground_ref = 0.0
        print("[RgbdDetectPipeline] SATURATION-primary + depth distance")

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0
        self._debug.clear()

    def _roi(self, h: int, w: int) -> np.ndarray:
        u0, u1 = int(w * ROI_U_MARGIN), int(w * (1 - ROI_U_MARGIN))
        v0, v1 = int(h * ROI_V_MIN), int(h * ROI_V_MAX)
        m = np.zeros((h, w), dtype=bool)
        m[v0:v1, u0:u1] = True
        return m

    def _ground_sat_reference(self, sat: np.ndarray, roi: np.ndarray, h: int) -> float:
        """用画面下半部估计灰地面饱和度"""
        v_mid = int(h * 0.48)
        ground_region = roi.copy()
        ground_region[:v_mid, :] = False
        vals = sat[ground_region]
        if vals.size > 80:
            return float(np.percentile(vals, 55))
        vals = sat[roi]
        return float(np.percentile(vals, 40)) if vals.size else 30.0

    def build_saturation_mask(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        roi = self._roi(h, w)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        self._sat_ground_ref = self._ground_sat_reference(sat, roi, h)
        self._sat_thresh = max(SAT_MIN_ABSOLUTE, self._sat_ground_ref + SAT_ABOVE_GROUND)

        sat_mask = (sat >= self._sat_thresh) & (val >= VAL_MIN) & roi & valid

        detect = sat_mask.astype(np.uint8)
        detect = cv2.morphologyEx(detect, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        detect = cv2.morphologyEx(detect, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        self._debug = {
            "saturation": sat,
            "sat_mask": sat_mask.astype(np.uint8) * 255,
            "rgbd": detect * 255,
            "valid_depth": (valid & roi).astype(np.uint8) * 255,
        }
        self._depth_stats = depth_stats(depth)
        return detect

    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> List[dict]:
        depth = sanitize_depth(depth)
        mask = self.build_saturation_mask(rgb, depth)
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
            patch_sat = float(np.mean(self._debug["saturation"][ys, xs]))
            conf = float(min(0.97, 0.5 + (patch_sat - self._sat_thresh) / 128.0 + area / 3000.0))

            dets.append({
                "class": "object",
                "conf": conf,
                "bbox": bbox,
                "centroid": (cx, cy),
                "depth_m": d_m,
                "area": area,
                "mean_sat": patch_sat,
                "source": "saturation",
            })

        dets.sort(key=lambda x: (x["depth_m"] if x["depth_m"] is not None else 999, -x["area"]))
        return dets[:MAX_DETECTIONS]

    def get_debug(self, name: str) -> Optional[np.ndarray]:
        if name == "saturation":
            sat = self._debug.get("saturation")
            if sat is not None:
                return sat
        return self._debug.get(name)

    def get_depth_stats(self) -> Dict[str, float]:
        return dict(self._depth_stats)

    def get_sat_info(self) -> Dict[str, float]:
        return {"thresh": self._sat_thresh, "ground_ref": self._sat_ground_ref}

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
                "source": "saturation",
            }
            objects.append(obj)
            if target is None or (obj["depth_m"] or 999) < (target.get("depth_m") or 999):
                target = obj

        sat_info = self.get_sat_info()
        return {
            "target": target,
            "objects_detailed": objects,
            "objects_remaining": [{"id": o["id"], "class": o["class"], "dist": o["depth_m"]} for o in objects],
            "depth_stats": self.get_depth_stats(),
            "sat_thresh": sat_info["thresh"],
            "sat_ground_ref": sat_info["ground_ref"],
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": ROBOT_INIT_POS.tolist(), "yaw": float(ROBOT_INIT_YAW)},
        }
