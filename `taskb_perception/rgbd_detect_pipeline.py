"""
Task B — 饱和度检测 + depth 测距

改进:
    - 裁掉画面下方 (机器人腿/自身影子)
    - 亮度过滤 (地面影子偏暗)
    - 框扩大 + 掩码略膨胀
    - 分水岭拆开挨在一起的两个物体
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

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

DEPTH_MIN = 0.40
DEPTH_MAX = 12.0

ROI_V_MIN = 0.12
ROI_V_MAX = 0.76          # 裁掉最下 24% (机身/自身影)
ROI_U_MARGIN = 0.05

SAT_MIN_ABSOLUTE = 50
SAT_ABOVE_GROUND = 25
VAL_MIN = 58                # 像素亮度下限 (影子更暗)
VAL_ABOVE_GROUND = 18       # 比地面亮多少

BLOB_VAL_MEAN_MIN = 74      # 整块平均亮度太低 → 当影子丢掉
BLOB_VAL_REFINE_P = 42      # 簇内保留亮度前 58% 像素 (去掉拖影)

BBOX_PAD_RATIO = 0.14
BBOX_PAD_PX = 12
BBOX_DILATE_K = 5           # 算框前膨胀掩码

MIN_BLOB_AREA = 80
MAX_BLOB_AREA = 45000
MIN_BLOB_SIDE = 8
MAX_BLOB_SIDE = 340
MAX_DETECTIONS = 14

# 分水岭: 宽度明显大于高度且够大 → 尝试拆成两个
SPLIT_MIN_WIDTH = 55
SPLIT_MIN_AREA = 500
SPLIT_WIDTH_OVER_HEIGHT = 1.35

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
                dxy = float(np.hypot(cx - t["centroid"][0], cy - t["centroid"][1]))
                if dxy < TRACK_MATCH_PX and dxy < best_d:
                    best_d, best_id = dxy, tid
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
    def __init__(self):
        self.tracker = Tracker2D()
        self.frame_count = 0
        self._debug: Dict[str, np.ndarray] = {}
        self._depth_stats: Dict[str, float] = {}
        self._sat_thresh = 0.0
        self._sat_ground_ref = 0.0
        self._val_ground_ref = 0.0
        print("[RgbdDetectPipeline] saturation + shadow filter + split + bbox expand")

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

    def _ground_refs(self, sat: np.ndarray, val: np.ndarray, roi: np.ndarray, h: int) -> Tuple[float, float]:
        v_mid = int(h * 0.50)
        ground = roi.copy()
        ground[:v_mid, :] = False
        s_vals = sat[ground]
        v_vals = val[ground]
        if s_vals.size > 80:
            return float(np.percentile(s_vals, 55)), float(np.percentile(v_vals, 50))
        s_vals = sat[roi]
        v_vals = val[roi]
        return (
            float(np.percentile(s_vals, 40)) if s_vals.size else 30.0,
            float(np.percentile(v_vals, 45)) if v_vals.size else 80.0,
        )

    def build_saturation_mask(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        roi = self._roi(h, w)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]

        self._sat_ground_ref, self._val_ground_ref = self._ground_refs(sat, val, roi, h)
        self._sat_thresh = max(SAT_MIN_ABSOLUTE, self._sat_ground_ref + SAT_ABOVE_GROUND)
        val_thresh = max(VAL_MIN, self._val_ground_ref + VAL_ABOVE_GROUND)

        sat_mask = (sat >= self._sat_thresh) & (val >= val_thresh) & roi & valid

        detect = sat_mask.astype(np.uint8)
        detect = cv2.morphologyEx(detect, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # 用小 close, 避免两个物体粘成一个
        detect = cv2.morphologyEx(detect, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

        self._debug = {
            "saturation": sat,
            "sat_mask": sat_mask.astype(np.uint8) * 255,
            "rgbd": detect * 255,
        }
        self._depth_stats = depth_stats(depth)
        self._hsv_val = val
        return detect

    def _is_shadow_blob(self, ys: np.ndarray, xs: np.ndarray, val: np.ndarray, h: int) -> bool:
        v = val[ys, xs]
        mean_v = float(np.mean(v))
        if mean_v < BLOB_VAL_MEAN_MIN:
            return True
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        cy = float(np.mean(ys))
        aspect = bw / max(bh, 1)
        # 画面下方细长暗块 = 自身/地面影
        if cy > h * 0.68 and aspect > 2.2 and mean_v < 88:
            return True
        if aspect > 3.8 and mean_v < 92:
            return True
        return False

    def _refine_blob(self, ys: np.ndarray, xs: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """去掉簇里偏暗的拖影像素"""
        v = val[ys, xs]
        v_cut = max(VAL_MIN, float(np.percentile(v, BLOB_VAL_REFINE_P)))
        keep = v >= v_cut
        if int(np.sum(keep)) < MIN_BLOB_AREA // 2:
            return ys, xs
        return ys[keep], xs[keep]

    def _expand_bbox(self, x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> List[int]:
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        px = max(BBOX_PAD_PX, int(bw * BBOX_PAD_RATIO))
        py = max(BBOX_PAD_PX, int(bh * BBOX_PAD_RATIO))
        return [
            max(0, x1 - px), max(0, y1 - py),
            min(w - 1, x2 + px), min(h - 1, y2 + py),
        ]

    def _bbox_from_mask(self, ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> List[int]:
        sub = np.zeros((h, w), dtype=np.uint8)
        sub[ys, xs] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BBOX_DILATE_K, BBOX_DILATE_K))
        sub = cv2.dilate(sub, k, iterations=1)
        ys2, xs2 = np.where(sub > 0)
        if len(ys2) < 5:
            ys2, xs2 = ys, xs
        x1, x2 = int(xs2.min()), int(xs2.max())
        y1, y2 = int(ys2.min()), int(ys2.max())
        return self._expand_bbox(x1, y1, x2, y2, h, w)

    def _watershed_split(
        self, ys: np.ndarray, xs: np.ndarray, rgb: np.ndarray, h: int, w: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        area = len(ys)
        if bw < SPLIT_MIN_WIDTH or area < SPLIT_MIN_AREA or bw < bh * SPLIT_WIDTH_OVER_HEIGHT:
            return [(ys, xs)]

        sub = np.zeros((h, w), dtype=np.uint8)
        sub[ys, xs] = 255
        roi = sub[y1:y2 + 1, x1:x2 + 1]
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        if dist.max() < 6:
            return [(ys, xs)]

        _, sure_fg = cv2.threshold(dist, 0.42 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        n_fg, _ = cv2.connectedComponents(sure_fg)
        if n_fg - 1 < 2:
            return [(ys, xs)]

        unknown = cv2.subtract(roi, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        color_roi = cv2.cvtColor(rgb[y1:y2 + 1, x1:x2 + 1], cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(color_roi, markers)

        parts = []
        for mid in range(2, int(markers.max()) + 1):
            py, px = np.where(markers == mid)
            if len(py) < MIN_BLOB_AREA:
                continue
            parts.append((py + y1, px + x1))
        return parts if len(parts) >= 2 else [(ys, xs)]

    def _blob_to_det(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray,
        sat: np.ndarray, val: np.ndarray, h: int, w: int,
    ) -> Optional[dict]:
        if self._is_shadow_blob(ys, xs, val, h):
            return None
        ys, xs = self._refine_blob(ys, xs, val)
        if len(ys) < MIN_BLOB_AREA:
            return None

        bbox = self._bbox_from_mask(ys, xs, h, w)
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if min(bw, bh) < MIN_BLOB_SIDE or max(bw, bh) > MAX_BLOB_SIDE:
            return None

        d_m = bbox_center_depth(depth, bbox, pad=2)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        patch_sat = float(np.mean(sat[ys, xs]))
        conf = float(min(0.97, 0.5 + (patch_sat - self._sat_thresh) / 120.0))

        return {
            "class": "object",
            "conf": conf,
            "bbox": bbox,
            "centroid": (cx, cy),
            "depth_m": d_m,
            "area": int(len(ys)),
            "source": "saturation",
        }

    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> List[dict]:
        depth = sanitize_depth(depth)
        h, w = depth.shape
        mask = self.build_saturation_mask(rgb, depth)
        sat = self._debug["saturation"]
        val = self._hsv_val

        labeled, n = ndimage.label(mask > 0)
        if n == 0:
            return []

        dets: List[dict] = []
        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            if len(ys) > MAX_BLOB_AREA:
                continue

            parts = self._watershed_split(ys, xs, rgb, h, w)
            for pys, pxs in parts:
                det = self._blob_to_det(pys, pxs, depth, sat, val, h, w)
                if det is not None:
                    dets.append(det)

        dets.sort(key=lambda x: (x["depth_m"] if x["depth_m"] is not None else 999, -x["area"]))
        return dets[:MAX_DETECTIONS]

    def get_debug(self, name: str) -> Optional[np.ndarray]:
        if name == "saturation":
            return self._debug.get("saturation")
        return self._debug.get(name)

    def get_depth_stats(self) -> Dict[str, float]:
        return dict(self._depth_stats)

    def get_sat_info(self) -> Dict[str, float]:
        return {
            "thresh": self._sat_thresh,
            "ground_ref": self._sat_ground_ref,
            "val_ground": self._val_ground_ref,
        }

    def process(self, obs, dt: float = 0.02) -> dict:
        self.frame_count += 1
        rgb, depth = parse_head_rgbd(obs)
        raw = self.detect(rgb, depth)
        tracks = self.tracker.update(raw)
        confirmed = [
            t for t in tracks
            if self.tracker.tracks.get(t["track_id"], {}).get("hits", 0) >= MIN_TRACK_HITS
        ]

        objects, target = [], None
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

        info = self.get_sat_info()
        return {
            "target": target,
            "objects_detailed": objects,
            "objects_remaining": [{"id": o["id"], "class": o["class"], "dist": o["depth_m"]} for o in objects],
            "depth_stats": self.get_depth_stats(),
            "sat_thresh": info["thresh"],
            "sat_ground_ref": info["ground_ref"],
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": ROBOT_INIT_POS.tolist(), "yaw": float(ROBOT_INIT_YAW)},
        }
