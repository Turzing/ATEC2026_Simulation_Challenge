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
from config import (
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    TOTAL_OBJECTS,
    CLASS_NAME_TO_ID,
    OBJECT_SIZES,
    DEFAULT_OBJECT_SIZE,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
)
from rgbd_utils import (
    bbox_center_depth,
    depth_stats,
    median_depth_in_mask,
    parse_head_rgbd,
    pixel_depth_to_cam,
    sanitize_depth,
)

DEPTH_MIN = 0.40
DEPTH_MAX = 12.0

ROI_V_MIN = 0.12
ROI_V_MAX = 0.76          # 裁掉最下 24% (机身/自身影)
ROI_U_MARGIN = 0.05

SAT_MIN_ABSOLUTE = 48
SAT_ABOVE_GROUND = 22
VAL_MIN = 55
VAL_ABOVE_GROUND = 15

# 远距离 (2m+) 物体更小、饱和度略低 → 单独放宽
FAR_DEPTH_M = 2.0
SAT_RELAX_FAR = 16          # 远距 sat 阈值降低
VAL_RELAX_FAR = 10
FAR_MASK_DILATE = 3         # 远距掩码略膨胀, 补小目标

BLOB_VAL_MEAN_MIN_NEAR = 74
BLOB_VAL_MEAN_MIN_FAR = 66
BLOB_VAL_REFINE_P_NEAR = 42
BLOB_VAL_REFINE_P_FAR = 28  # 远距少裁暗边, 避免整块没了

BBOX_PAD_RATIO = 0.14
BBOX_PAD_RATIO_FAR = 0.22   # 远距框多扩一点
BBOX_PAD_PX = 12
BBOX_PAD_PX_FAR = 18
BBOX_DILATE_K = 5

MIN_BLOB_AREA_NEAR = 80
MIN_BLOB_AREA_FAR = 32      # 3~5m 可能只有几十个像素
MAX_BLOB_AREA = 45000
MIN_BLOB_SIDE_NEAR = 8
MIN_BLOB_SIDE_FAR = 5
MAX_BLOB_SIDE = 340
MAX_DETECTIONS = 14

# 分水岭: 宽度明显大于高度且够大 → 尝试拆成两个
SPLIT_MIN_WIDTH = 55
SPLIT_MIN_AREA = 500
SPLIT_WIDTH_OVER_HEIGHT = 1.35

TRACK_MATCH_PX = 90
TRACK_MAX_AGE = 15
MIN_TRACK_HITS = 1

MIN_CLASS_MARGIN = 0.10
GEOM_MATCH_SIGMA = 0.065     # 部分点云 + 立/倒 容差 (米)
RATIO_MATCH_SIGMA = 0.35     # 三边比例匹配 (与姿态无关)
MIN_3D_POINTS = 8

# 立起/倒下后仍不变: 三边排序 + 比例 (e1/e0, e2/e1)
# 糖盒 0.039/0.098/0.175  瓶 0.057/0.082/0.189  香蕉 0.04/0.04/0.195
_CLASS_RATIO_TEMPLATES: Dict[str, Tuple[float, float]] = {}


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


def _build_ratio_templates() -> Dict[str, Tuple[float, float]]:
    out = {}
    for name in OBJECT_SIZES:
        e = np.sort([OBJECT_SIZES[name]["lx"], OBJECT_SIZES[name]["ly"], OBJECT_SIZES[name]["lz"]])
        out[name] = (float(e[1] / max(e[0], 1e-4)), float(e[2] / max(e[1], 1e-4)))
    return out


_CLASS_RATIO_TEMPLATES.update(_build_ratio_templates())


class RgbdDetectPipeline:
    def __init__(self):
        self.tracker = Tracker2D()
        self.frame_count = 0
        self._debug: Dict[str, np.ndarray] = {}
        self._depth_stats: Dict[str, float] = {}
        self._sat_thresh = 0.0
        self._sat_ground_ref = 0.0
        self._val_ground_ref = 0.0
        print("[RgbdDetectPipeline] sat detect + 3D geom classify")

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
        sat_far = max(40, self._sat_thresh - SAT_RELAX_FAR)
        val_far = max(48, val_thresh - VAL_RELAX_FAR)

        near = depth < FAR_DEPTH_M
        far = depth >= FAR_DEPTH_M

        sat_near = (sat >= self._sat_thresh) & (val >= val_thresh) & near
        sat_far_m = (sat >= sat_far) & (val >= val_far) & far
        sat_mask = (sat_near | sat_far_m) & roi & valid

        detect = sat_mask.astype(np.uint8)
        if FAR_MASK_DILATE > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FAR_MASK_DILATE, FAR_MASK_DILATE))
            far_u8 = (sat_far_m & roi & valid).astype(np.uint8)
            far_u8 = cv2.dilate(far_u8, k, iterations=1)
            detect = cv2.bitwise_or(detect, far_u8)

        detect = cv2.morphologyEx(detect, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        detect = cv2.morphologyEx(detect, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        self._sat_thresh_far = sat_far

        self._debug = {
            "saturation": sat,
            "sat_mask": sat_mask.astype(np.uint8) * 255,
            "rgbd": detect * 255,
        }
        self._depth_stats = depth_stats(depth)
        self._hsv_val = val
        return detect

    def _blob_depth_m(self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray) -> Optional[float]:
        d = depth[ys, xs]
        d = d[(d > DEPTH_MIN) & (d < DEPTH_MAX)]
        return float(np.median(d)) if len(d) >= 3 else None

    def _adaptive_limits(self, depth_m: Optional[float]) -> dict:
        if depth_m is None or depth_m < FAR_DEPTH_M:
            return {
                "min_area": MIN_BLOB_AREA_NEAR,
                "min_side": MIN_BLOB_SIDE_NEAR,
                "val_mean_min": BLOB_VAL_MEAN_MIN_NEAR,
                "val_refine_p": BLOB_VAL_REFINE_P_NEAR,
                "pad_ratio": BBOX_PAD_RATIO,
                "pad_px": BBOX_PAD_PX,
            }
        t = min(1.0, max(0.0, (depth_m - FAR_DEPTH_M) / 3.0))
        return {
            "min_area": int(MIN_BLOB_AREA_NEAR + t * (MIN_BLOB_AREA_FAR - MIN_BLOB_AREA_NEAR)),
            "min_side": MIN_BLOB_SIDE_FAR,
            "val_mean_min": BLOB_VAL_MEAN_MIN_NEAR + t * (BLOB_VAL_MEAN_MIN_FAR - BLOB_VAL_MEAN_MIN_NEAR),
            "val_refine_p": BLOB_VAL_REFINE_P_NEAR + t * (BLOB_VAL_REFINE_P_FAR - BLOB_VAL_REFINE_P_NEAR),
            "pad_ratio": BBOX_PAD_RATIO + t * (BBOX_PAD_RATIO_FAR - BBOX_PAD_RATIO),
            "pad_px": int(BBOX_PAD_PX + t * (BBOX_PAD_PX_FAR - BBOX_PAD_PX)),
        }

    def _is_shadow_blob(
        self, ys: np.ndarray, xs: np.ndarray, val: np.ndarray, h: int, lim: dict,
    ) -> bool:
        v = val[ys, xs]
        mean_v = float(np.mean(v))
        if mean_v < lim["val_mean_min"]:
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

    def _refine_blob(
        self, ys: np.ndarray, xs: np.ndarray, val: np.ndarray, lim: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v = val[ys, xs]
        v_cut = max(VAL_MIN, float(np.percentile(v, lim["val_refine_p"])))
        keep = v >= v_cut
        if int(np.sum(keep)) < max(20, lim["min_area"] // 2):
            return ys, xs
        return ys[keep], xs[keep]

    def _expand_bbox(
        self, x1: int, y1: int, x2: int, y2: int, h: int, w: int, lim: dict,
    ) -> List[int]:
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        px = max(lim["pad_px"], int(bw * lim["pad_ratio"]))
        py = max(lim["pad_px"], int(bh * lim["pad_ratio"]))
        return [
            max(0, x1 - px), max(0, y1 - py),
            min(w - 1, x2 + px), min(h - 1, y2 + py),
        ]

    def _bbox_from_mask(
        self, ys: np.ndarray, xs: np.ndarray, h: int, w: int, lim: dict,
    ) -> List[int]:
        sub = np.zeros((h, w), dtype=np.uint8)
        sub[ys, xs] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BBOX_DILATE_K, BBOX_DILATE_K))
        sub = cv2.dilate(sub, k, iterations=1)
        ys2, xs2 = np.where(sub > 0)
        if len(ys2) < 5:
            ys2, xs2 = ys, xs
        x1, x2 = int(xs2.min()), int(xs2.max())
        y1, y2 = int(ys2.min()), int(ys2.max())
        return self._expand_bbox(x1, y1, x2, y2, h, w, lim)

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
            if len(py) < MIN_BLOB_AREA_FAR:
                continue
            parts.append((py + y1, px + x1))
        return parts if len(parts) >= 2 else [(ys, xs)]

    def _blob_points_robot(self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        """掩码像素 + depth → 机器人系 3D 点云"""
        pts = []
        step = 1 if len(ys) < 400 else 2
        for y, x in zip(ys[::step], xs[::step]):
            z = float(depth[y, x])
            if z <= DEPTH_MIN or z >= DEPTH_MAX:
                continue
            p_cam = pixel_depth_to_cam(float(x), float(y), z)
            p_robot = HEAD_CAM_POS_ROBOT + HEAD_CAM_ROT_MATRIX @ p_cam
            pts.append(p_robot)
        if len(pts) < MIN_3D_POINTS:
            return None
        return np.stack(pts, axis=0).astype(np.float32)

    @staticmethod
    def _template_extents(name: str) -> np.ndarray:
        s = OBJECT_SIZES[name]
        return np.sort([s["lx"], s["ly"], s["lz"]]).astype(np.float32)

    def _pca_sorted_extents(self, pts: np.ndarray) -> np.ndarray:
        """
        在点云主轴上量三边 (立/倒/侧放都不变).
        机器人系 AABB 会随姿态变; PCA 范围与物体真实长宽高一致.
        """
        centered = pts - np.mean(pts, axis=0)
        if len(centered) < MIN_3D_POINTS:
            return np.sort(np.ptp(pts, axis=0)).astype(np.float32)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt.T
            ext = np.ptp(proj, axis=0)
        except np.linalg.LinAlgError:
            ext = np.ptp(pts, axis=0)
        ext = np.maximum(ext, 1e-4)
        return np.sort(ext).astype(np.float32)

    @staticmethod
    def _extent_ratios(ext: np.ndarray) -> Tuple[float, float, float]:
        e0, e1, e2 = [float(v) for v in ext]
        return e1 / max(e0, 1e-4), e2 / max(e1, 1e-4), e0 / max(e2, 1e-4)

    def _score_3d_shape(self, ext: np.ndarray) -> Dict[str, float]:
        """
        立起/倒下: 用排序尺寸 + 三边比例 (不假设哪边是高度).
        香蕉: e1/e0≈1 (截面两维接近); 糖盒: 三边比 1:2.5:4.5; 瓶: 中间
        """
        r10, r21, flat = self._extent_ratios(ext)
        scores: Dict[str, float] = {}

        for name in OBJECT_SIZES:
            t_ext = self._template_extents(name)
            d_ext = float(np.linalg.norm(ext - t_ext))
            s_ext = float(np.exp(-(d_ext / GEOM_MATCH_SIGMA) ** 2))

            tr10, tr21 = _CLASS_RATIO_TEMPLATES[name]
            d_ratio = float(np.hypot(r10 - tr10, r21 - tr21))
            s_ratio = float(np.exp(-(d_ratio / RATIO_MATCH_SIGMA) ** 2))

            # 比例更稳 (姿态无关), 绝对尺寸辅助 (近距)
            scores[name] = 0.38 * s_ext + 0.62 * s_ratio

        # 香蕉: 最短两边很接近 (躺/立都一样)
        if r10 < 1.18:
            scores["banana"] *= 1.55
            scores["sugar_box"] *= 0.75
        # 糖盒: 扁 (最小/最大 小) 且 不是香蕉那种 e1≈e0
        if flat < 0.26 and r10 > 1.35:
            scores["sugar_box"] *= 1.45
        # 瓶: 三边比例 1:1.4:2.3 附近, 且不是香蕉
        if 1.25 < r10 < 1.65 and 1.9 < r21 < 2.8:
            scores["mustard_bottle"] *= 1.40

        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}

    def _classify_blob_2d_fallback(self, bbox: List[int], depth: np.ndarray, ys, xs) -> Tuple[str, float]:
        """3D 点不够时的备用"""
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        aspect = max(bw, bh) / min(bw, bh)
        if aspect >= 2.0:
            return "banana", 0.45
        if aspect < 1.2:
            return "mustard_bottle", 0.40
        return "sugar_box", 0.40

    def _classify_blob(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        bbox: List[int],
        depth: np.ndarray,
        rgb: np.ndarray,
    ) -> Tuple[str, float, Optional[List[float]]]:
        """
        3D 分类: 点云 PCA 三边 + YCB 模板 (立/倒用比例, 不假设竖着)
        """
        pts = self._blob_points_robot(ys, xs, depth)
        if pts is None:
            name, conf = self._classify_blob_2d_fallback(bbox, depth, ys, xs)
            return name, conf * 0.7, None

        ext = self._pca_sorted_extents(pts)
        if float(ext[2]) < 0.025:
            name, conf = self._classify_blob_2d_fallback(bbox, depth, ys, xs)
            return name, conf * 0.65, ext.tolist()

        scores = self._score_3d_shape(ext)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best, s1 = ranked[0]
        s2 = ranked[1][1]
        margin = s1 - s2
        cls_conf = float(min(0.94, 0.5 + margin * 1.2 + min(0.15, len(pts) / 200.0)))

        if margin < MIN_CLASS_MARGIN:
            return "unknown", cls_conf * 0.55, ext.tolist()
        return best, cls_conf, ext.tolist()

    def _blob_to_det(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray,
        sat: np.ndarray, val: np.ndarray, rgb: np.ndarray, h: int, w: int,
    ) -> Optional[dict]:
        depth_m0 = self._blob_depth_m(ys, xs, depth)
        lim = self._adaptive_limits(depth_m0)
        if self._is_shadow_blob(ys, xs, val, h, lim):
            return None
        ys, xs = self._refine_blob(ys, xs, val, lim)
        if len(ys) < lim["min_area"]:
            return None

        bbox = self._bbox_from_mask(ys, xs, h, w, lim)
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if min(bw, bh) < lim["min_side"] or max(bw, bh) > MAX_BLOB_SIDE:
            return None

        d_m = bbox_center_depth(depth, bbox, pad=2)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        patch_sat = float(np.mean(sat[ys, xs]))
        det_conf = float(min(0.97, 0.5 + (patch_sat - self._sat_thresh) / 120.0))

        cls_name, cls_conf, geom_ext = self._classify_blob(ys, xs, bbox, depth, rgb)
        conf = float(min(0.97, det_conf * 0.45 + cls_conf * 0.55))
        size = OBJECT_SIZES.get(cls_name, DEFAULT_OBJECT_SIZE)

        return {
            "class": cls_name,
            "class_id": CLASS_NAME_TO_ID.get(cls_name, -1),
            "conf": conf,
            "class_conf": cls_conf,
            "bbox": bbox,
            "centroid": (cx, cy),
            "depth_m": d_m,
            "area": int(len(ys)),
            "geom_extents": geom_ext,
            "size_world": [size["lx"], size["ly"], size["lz"]],
            "source": "saturation+geom3d",
        }

    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> List[dict]:
        depth = sanitize_depth(depth)
        h, w = depth.shape
        mask = self.build_saturation_mask(rgb, depth)
        sat = self._debug["saturation"]
        val = self._hsv_val
        self._last_rgb = rgb

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
                det = self._blob_to_det(pys, pxs, depth, sat, val, rgb, h, w)
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
            "thresh_far": getattr(self, "_sat_thresh_far", self._sat_thresh),
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
            cls_name = t.get("class", "unknown")
            size = OBJECT_SIZES.get(cls_name, DEFAULT_OBJECT_SIZE)
            obj = {
                "id": int(t["track_id"]),
                "class": cls_name,
                "class_id": t.get("class_id", -1),
                "conf": float(t["conf"]),
                "class_conf": float(t.get("class_conf", 0.0)),
                "bbox": t["bbox"],
                "centroid_uv": list(t["centroid"]),
                "depth_m": t.get("depth_m"),
                "dist_to_robot": t.get("depth_m"),
                "size_world": t.get("size_world", [size["lx"], size["ly"], size["lz"]]),
                "geom_extents": t.get("geom_extents"),
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
            "sat_thresh_far": info.get("thresh_far", info["thresh"]),
            "sat_ground_ref": info["ground_ref"],
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": ROBOT_INIT_POS.tolist(), "yaw": float(ROBOT_INIT_YAW)},
        }
