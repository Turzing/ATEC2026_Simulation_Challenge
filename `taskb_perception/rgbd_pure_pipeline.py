"""
ATEC Task B — 老师版 RGB-D（depth 凸起 + RGB 黄色融合）

单摄:
    from rgbd_pure_pipeline import RgbdPurePipeline
    out = RgbdPurePipeline().process(obs)

双摄 (老师要求: 爪远 head 近):
    from rgbd_pure_dual_pipeline import RgbdPureDualPipeline
    out = RgbdPureDualPipeline().process(obs)
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
    CLASS_NAME_TO_ID,
    DEFAULT_OBJECT_SIZE,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    OBJECT_SIZES,
    PROPRIO_BASE_ANG_VEL,
    PROPRIO_BASE_LIN_VEL,
    PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    TOTAL_OBJECTS,
)
from rgbd_utils import (
    _to_numpy,
    depth_stats,
    parse_head_rgbd,
    pixel_depth_to_cam,
    sanitize_depth,
)

CAMERA_CFG = {
    "head": {
        "cam": HEAD_CAM,
        "pos": HEAD_CAM_POS_ROBOT,
        "rot": HEAD_CAM_ROT_MATRIX,
        "roi_v_min": 0.08,
        "roi_v_max": 0.78,
        "roi_v_max_near": 0.96,
        "roi_u_margin": 0.04,
        "bottom_strip_v0": 0.66,
        "relief_min": 0.018,
        "relief_min_near": 0.006,
        "ground_k": 13,
        "sat_extra": 10,
        "sat_relax": 10,
        "sat_relax_very_near": 16,
        "val_relax_very_near": 18,
        "min_area": 24,
        "min_side": 5,
        "mask_close_k": 7,
        "fusion_mode": "head_grasp",
        "depth_min": 0.14,
        "depth_max_near": 2.25,
        "val_refine_p": 32,
        "min_blob_sat": 42,
        "min_blob_val": 62,
        "min_relief_med": 0.007,
        "max_shadow_area": 1500,
        "max_depth_std": 0.088,
        "min_track_hits": 2,
        "near_strong_sat": 42,
        "near_strong_val": 60,
        "near_fb_depth_m": 1.55,
    },
    "ee": {
        "cam": EE_CAM,
        "pos": EE_CAM_POS_ROBOT,
        "rot": EE_CAM_ROT_MATRIX,
        "roi_v_min": 0.04,
        "roi_v_max": 0.96,
        "roi_v_max_near": 0.96,
        "roi_u_margin": 0.04,
        "relief_min": 0.010,
        "relief_min_near": 0.008,
        "ground_k": 11,
        "sat_extra": 4,
        "sat_relax": 22,
        "min_area": 14,
        "min_side": 3,
        "min_track_hits": 2,
        "far_mask_dilate": 3,
        "rgb_dilate_far": 2,
        "mask_close_k": 7,
        "fusion_mode": "ee_nav",
        "val_refine_p": 38,
        "min_blob_sat": 42,
        "max_depth_std": 0.058,
        "max_ee_blob_area": 1900,
        "min_relief_med": 0.006,
    },
}

# ── 深度 ─────────────────────────────────────────────────────────
DEPTH_MIN = 0.32
DEPTH_MAX = 11.0
GROUND_OPEN_K = 19
RELIEF_MIN = 0.026
RELIEF_MIN_NEAR = 0.018
RELIEF_MAX = 0.28
NEAR_DEPTH_M = 1.15
CLOSER_THAN_BG_M = 0.012

# ── 画面 ROI ─────────────────────────────────────────────────────
ROI_V_MIN = 0.10
ROI_V_MAX = 0.78
ROI_V_MAX_NEAR = 0.93
ROI_U_MARGIN = 0.05
NEAR_SCENE_P10 = 1.08
NEAR_SCENE_MIN = 0.95
VERY_NEAR_DEPTH_M = 1.15
HEAD_HUE_LO, HEAD_HUE_HI = 12, 50

# ── RGB ──────────────────────────────────────────────────────────
HUE_LO, HUE_HI = 10, 52
SAT_MIN = 26
VAL_MIN = 42

# ── 检出 / 跟踪 ──────────────────────────────────────────────────
MIN_AREA = 36
MIN_SIDE = 6
MAX_SIDE = 360
TRACK_MATCH_PX = 88
TRACK_MAX_AGE = 12
MIN_TRACK_HITS = 2
DEPTH_SMOOTH = 0.72
POS_SMOOTH = 0.65
DEPTH_POINT_TOL = 0.07
DEPTH_POINT_TOL_NEAR = 0.05
MAX_BLOB_DEPTH_STD = 0.065
MIN_BLOB_SAT_MEAN = 38
SHADOW_VAL_MAX = 72


def _yaw_from_gravity(g) -> float:
    return float(np.arctan2(-g[0], -g[1]))


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
                dist = float(np.hypot(cx - t["centroid"][0], cy - t["centroid"][1]))
                if dist < TRACK_MATCH_PX and dist < best_d:
                    best_d, best_id = dist, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {"centroid": (cx, cy), "age": 0, "hits": 0, "depth_m": d.get("depth_m")}
            tr = self.tracks[best_id]
            assigned.add(best_id)
            ocx, ocy = tr["centroid"]
            tr["centroid"] = (0.65 * ocx + 0.35 * cx, 0.65 * ocy + 0.35 * cy)
            tr["age"] = 0
            tr["hits"] = tr.get("hits", 0) + 1
            if d.get("depth_m") is not None:
                old = tr.get("depth_m")
                nd = d["depth_m"]
                tr["depth_m"] = nd if old is None else DEPTH_SMOOTH * old + (1 - DEPTH_SMOOTH) * nd
                d = {**d, "depth_m": tr["depth_m"]}
            if d.get("pos_robot") is not None:
                op = tr.get("pos_robot")
                np_ = np.asarray(d["pos_robot"], dtype=np.float32)
                if op is None:
                    tr["pos_robot"] = np_.copy()
                else:
                    tr["pos_robot"] = POS_SMOOTH * op + (1 - POS_SMOOTH) * np_
                pr = tr["pos_robot"]
                d = {
                    **d,
                    "pos_robot": pr.tolist(),
                    "dist_to_robot": float(np.linalg.norm(pr[:2])),
                }
            out.append({**d, "track_id": best_id})
        for tid in [k for k, v in self.tracks.items() if v["age"] > TRACK_MAX_AGE]:
            del self.tracks[tid]
        return out


class RgbdPureCamera:
    """单路 RGB-D 融合 (head 或 ee)"""

    def __init__(self, camera: str = "head"):
        if camera not in CAMERA_CFG:
            raise ValueError(f"camera must be one of {list(CAMERA_CFG)}")
        self.camera_name = camera
        self._cfg = CAMERA_CFG[camera]
        self.tracker = Tracker2D()
        self.frame_count = 0
        self._debug: Dict[str, np.ndarray] = {}

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0
        self._debug.clear()

    def get_debug(self, name: str) -> Optional[np.ndarray]:
        return self._debug.get(name)

    def _depth_min(self) -> float:
        return float(self._cfg.get("depth_min", DEPTH_MIN))

    def _depth_max(self, near: bool = False) -> float:
        if self.camera_name == "head" and near:
            return float(self._cfg.get("depth_max_near", DEPTH_MAX))
        return DEPTH_MAX

    def _valid_depth(self, depth: np.ndarray, near: bool = False) -> np.ndarray:
        return (depth > self._depth_min()) & (depth < self._depth_max(near))

    def _bottom_strip_roi(self, h: int, w: int) -> np.ndarray:
        c = self._cfg
        u0, u1 = int(w * c["roi_u_margin"]), int(w * (1 - c["roi_u_margin"]))
        v0 = int(h * float(c.get("bottom_strip_v0", 0.66)))
        m = np.zeros((h, w), dtype=bool)
        if h - 2 > v0:
            m[v0 : h - 2, u0:u1] = True
        return m

    def _roi(self, h: int, w: int, near: bool) -> np.ndarray:
        c = self._cfg
        vmax = c["roi_v_max_near"] if near else c["roi_v_max"]
        u0, u1 = int(w * c["roi_u_margin"]), int(w * (1 - c["roi_u_margin"]))
        v0, v1 = int(h * c["roi_v_min"]), int(h * vmax)
        m = np.zeros((h, w), dtype=bool)
        m[v0:v1, u0:u1] = True
        return m

    def _ground_depth(self, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        d = depth.copy()
        d[~valid] = DEPTH_MAX
        k = self._cfg["ground_k"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        g = cv2.morphologyEx(d.astype(np.float32), cv2.MORPH_OPEN, kernel)
        g[~valid] = 0.0
        return g

    def _rgb_foreground(
        self,
        hue,
        sat,
        val,
        roi: np.ndarray,
        valid: np.ndarray,
        near: bool,
        depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        c = self._cfg
        g_sat = float(np.percentile(sat[roi & valid], 50)) if np.any(roi & valid) else 28.0
        relax = int(c.get("sat_relax", 12))
        sat_thr = max(SAT_MIN - 6, g_sat + c["sat_extra"])
        if self.camera_name == "head":
            sat_lo = max(26, sat_thr - relax)
            val_lo = max(52, VAL_MIN + 2)
            yellow = (
                (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
                & (sat >= sat_lo) & (val >= val_lo)
            )
            high_sat = sat >= max(sat_lo + 4, sat_thr - 6)
            fg = (yellow | high_sat) & roi
            if depth is not None:
                vn = valid & (depth < VERY_NEAR_DEPTH_M)
                rs = int(c.get("sat_relax_very_near", 24))
                rv = int(c.get("val_relax_very_near", 28))
                sat_vn = max(30, sat_thr - rs)
                val_vn = max(50, val_lo - rv)
                vn_y = (
                    vn & roi
                    & (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
                    & (sat >= sat_vn) & (val >= val_vn)
                )
                fg = fg | vn_y
            shadow = (val < 72) & (sat < 42)
            fg = fg & ~shadow
        else:
            sat_lo = max(16, sat_thr - relax)
            val_lo = VAL_MIN - 10
            yellow = (
                (hue >= HUE_LO) & (hue <= HUE_HI)
                & (sat >= sat_lo) & (val >= val_lo)
            )
            high_sat = sat >= max(sat_lo + 6, sat_thr - 4)
            fg = (yellow | high_sat) & roi
        return fg

    def _depth_gate(
        self, depth: np.ndarray, relief: np.ndarray, valid: np.ndarray,
        roi: np.ndarray, near: bool,
    ) -> np.ndarray:
        """Depth 校验: relief 凸起 或 比场景地面更近 (仿真里 relief 常很弱)"""
        c = self._cfg
        mode = c.get("fusion_mode", "soft")
        rmin = c["relief_min_near"] if near else c["relief_min"]
        depth_fg = (relief >= rmin) & (relief <= RELIEF_MAX) & valid

        roi_d = depth[roi & valid]
        closer = np.zeros_like(depth, dtype=bool)
        if roi_d.size > 80:
            d_bg = float(np.percentile(roi_d, 58))
            closer = valid & (depth < d_bg - CLOSER_THAN_BG_M)

        if mode == "rgb_depth":
            return valid & (depth > 0.38) & (depth < 10.5)

        if mode == "ee_nav":
            # EE 远距: 必须有一点 depth 线索, 不能纯 RGB (垃圾箱影子)
            depth_cue = depth_fg | closer | (valid & (relief >= rmin * 0.50))
            if not near:
                depth_cue = depth_cue | (
                    valid & (depth > 1.0) & (depth < 9.5) & (relief >= rmin * 0.38)
                )
            return valid & depth_cue & (depth > 0.38) & (depth < 10.5)

        if mode == "head_grasp":
            dmin = float(c.get("depth_min", DEPTH_MIN))
            dmax_near = float(c.get("depth_max_near", 2.25))
            has_cue = depth_fg | closer | (valid & (relief >= rmin * 0.38))
            if near:
                # 近距: 有 depth 线索的像素优先; 纯 RGB 兜底在 _build_fusion_mask 里单独加
                return valid & (depth > dmin) & (depth < dmax_near) & has_cue
            return valid & has_cue & (depth > dmin) & (depth < DEPTH_MAX)

        if mode in ("head_strict", "head_balanced"):
            soft = 0.65 if mode == "head_strict" else 0.42
            obj_depth = depth_fg | (closer & (relief >= rmin * soft))
            if near:
                near_soft = 0.45 if mode == "head_strict" else 0.28
                obj_depth = obj_depth | (
                    valid & (depth < NEAR_DEPTH_M) & (relief >= rmin * near_soft)
                )
                obj_depth = obj_depth | (closer & valid & (depth < 1.35))
            return valid & obj_depth

        if near:
            return valid & (depth_fg | closer | (depth < NEAR_DEPTH_M))
        return valid & (depth_fg | closer)

    @staticmethod
    def _close_components(mask: np.ndarray, ksize: int) -> np.ndarray:
        labeled, n = ndimage.label(mask > 0)
        if n == 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        out = np.zeros_like(mask)
        for cid in range(1, n + 1):
            comp = (labeled == cid).astype(np.uint8) * 255
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k)
            out = cv2.bitwise_or(out, comp)
        return out

    def _build_fusion_mask(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        near = self._scene_near(depth)
        valid = self._valid_depth(depth, near)
        roi = self._roi(h, w, near)
        c = self._cfg

        ground = self._ground_depth(depth, valid)
        relief = ground - depth

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        rgb_fg = self._rgb_foreground(hue, sat, val, roi, valid, near, depth=depth)

        rd = int(c.get("rgb_dilate_far", 0))
        if self.camera_name == "ee" and not near and rd > 0:
            rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rd, rd))
            rgb_fg = cv2.dilate(rgb_fg.astype(np.uint8), rk, 1).astype(bool)

        rmin = c["relief_min_near"] if near else c["relief_min"]
        depth_fg = (relief >= rmin) & (relief <= RELIEF_MAX) & valid
        roi_d = depth[roi & valid]
        closer = np.zeros_like(depth, dtype=bool)
        if roi_d.size > 80:
            d_bg = float(np.percentile(roi_d, 58))
            closer = valid & (depth < d_bg - CLOSER_THAN_BG_M)

        if self.camera_name == "head" and near:
            # 近距双通道: (1)有 relief/closer (2)中近距高饱和黄 — 修 debug(3) rgb_fg有/fusion无
            has_cue = depth_fg | closer | (relief >= rmin * 0.32)
            fused_main = (rgb_fg & valid & has_cue).astype(np.uint8)
            ss = int(c.get("near_strong_sat", 42))
            sv = int(c.get("near_strong_val", 60))
            fb_d = float(c.get("near_fb_depth_m", 1.55))
            strong = (
                (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
                & (sat >= ss) & (val >= sv)
            )
            mid_near = valid & (depth > self._depth_min()) & (depth < fb_d)
            fused_fb = (rgb_fg & mid_near & strong).astype(np.uint8)
            fused = cv2.bitwise_or(fused_main, fused_fb)
            depth_ok = has_cue | (mid_near & strong)
        else:
            depth_ok = self._depth_gate(depth, relief, valid, roi, near)
            fused = (rgb_fg & depth_ok).astype(np.uint8)

        ck = int(c.get("mask_close_k", 5))
        fused = self._close_components(fused, ck)

        fd = int(c.get("far_mask_dilate", 0))
        if self.camera_name == "ee" and not near and fd > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fd, fd))
            fused = cv2.dilate(fused, k, iterations=1)

        self._debug = {
            "relief": np.clip(relief / RELIEF_MAX * 255, 0, 255).astype(np.uint8),
            "depth_fg": (depth_ok & roi).astype(np.uint8) * 255,
            "rgb_fg": (rgb_fg & roi).astype(np.uint8) * 255,
            "fusion": fused.astype(np.uint8) * 255,
        }
        return fused

    def _uv_depth_to_robot(self, u: float, v: float, z: float) -> np.ndarray:
        cam = self._cfg["cam"]
        p_cam = pixel_depth_to_cam(u, v, z, cam)
        return (self._cfg["pos"] + self._cfg["rot"] @ p_cam).astype(np.float32)

    def _filter_depth_outliers(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, depth_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = depth[ys, xs]
        tol = DEPTH_POINT_TOL_NEAR if depth_m < 1.2 else DEPTH_POINT_TOL
        dmin, dmax = self._depth_min(), DEPTH_MAX
        keep = (d > dmin) & (d < dmax) & (d <= depth_m + tol) & (d >= depth_m - tol * 0.7)
        if int(np.sum(keep)) < 6:
            return ys, xs
        return ys[keep], xs[keep]

    def _refine_blob(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray,
        val: Optional[np.ndarray], h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dmin, dmax = self._depth_min(), DEPTH_MAX
        d0 = depth[ys, xs]
        ok = (d0 > dmin) & (d0 < dmax)
        if int(np.sum(ok)) < 5:
            return ys, xs
        d_med = float(np.median(d0[ok]))
        ys, xs = self._filter_depth_outliers(ys, xs, depth, d_med)
        if val is not None and len(ys) > 8 and self.camera_name != "head":
            v = val[ys, xs]
            v_cut = float(np.percentile(v, self._cfg.get("val_refine_p", 35)))
            keep = v >= max(SHADOW_VAL_MAX - 18, v_cut)
            if int(np.sum(keep)) >= max(6, len(ys) // 3):
                ys, xs = ys[keep], xs[keep]
        if self.camera_name == "head" and val is not None and len(ys) > 8:
            v = val[ys, xs]
            v_cut = float(np.percentile(v, self._cfg.get("val_refine_p", 32)))
            keep = v >= max(52, v_cut)
            if int(np.sum(keep)) >= max(5, len(ys) // 4):
                ys, xs = ys[keep], xs[keep]
            # 横长块: 影子在侧面/下方, 保留更亮一侧
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            if bw > bh * 1.35 and len(ys) > 60:
                cx = float(np.median(xs))
                bright = v >= float(np.percentile(v, 55))
                left, right = xs < cx, xs >= cx
                n_l = int(np.sum(bright & left))
                n_r = int(np.sum(bright & right))
                if n_l > 12 and n_r > 12:
                    keep_side = left if n_l >= n_r else right
                    keep2 = bright & keep_side
                    if int(np.sum(keep2)) >= max(8, len(ys) // 5):
                        ys, xs = ys[keep2], xs[keep2]
        return ys, xs

    def _is_shadow_shape(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        val: np.ndarray,
        sat: np.ndarray,
        h: int,
        w: int,
        relief: Optional[np.ndarray] = None,
    ) -> bool:
        if len(ys) < 5:
            return False
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        area = len(ys)
        mean_v = float(np.mean(val[ys, xs]))
        mean_s = float(np.mean(sat[ys, xs]))
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        cy = float(np.mean(ys))
        mean_r = float(np.median(relief[ys, xs])) if relief is not None else 0.02
        img_frac = area / max(h * w, 1)

        if self.camera_name == "ee":
            return self._is_ee_false_positive(
                area, bw, bh, aspect, cy, mean_v, mean_s, mean_r, img_frac, h, w,
            )

        # head: sat+val+relief 联合判影子 (不靠面积一刀切)
        obj_score = mean_s * 0.55 + mean_v * 0.25 + mean_r * 1200.0
        if obj_score >= 62.0 and mean_s >= 46 and mean_v >= 68:
            return False
        if area < 160 and mean_s >= 40 and mean_v >= 62 and mean_r >= 0.006:
            return False

        if mean_r < 0.007 and mean_v < 82:
            return True
        if mean_r < 0.009 and mean_s < 44:
            return True
        if mean_r < 0.010 and area > 200 and aspect > 1.35 and mean_v < 88:
            return True
        if cy > h * 0.44 and mean_r < 0.011 and aspect > 1.45 and area > 160:
            return True
        if area > 320 and aspect > 1.55 and mean_v < 86 and mean_s < 46:
            return True
        if area > int(self._cfg.get("max_shadow_area", 1500)) and mean_r < 0.011:
            return True
        if area > 900 and aspect > 2.0 and mean_r < 0.012 and mean_s < 48:
            return True
        return False

    def _is_ee_false_positive(
        self,
        area: int,
        bw: int,
        bh: int,
        aspect: float,
        cy: float,
        mean_v: float,
        mean_s: float,
        mean_r: float,
        img_frac: float,
        h: int,
        w: int,
    ) -> bool:
        """EE 远距: 垃圾箱/地面大块扁影"""
        c = self._cfg
        if area > int(c.get("max_ee_blob_area", 1900)):
            return True
        if img_frac > 0.075 and mean_r < 0.011:
            return True
        if area > 620 and mean_r < 0.008:
            return True
        if area > 480 and mean_r < 0.010 and mean_v < 98:
            return True
        if bw > w * 0.32 and bh < h * 0.14 and mean_r < 0.011:
            return True
        if aspect > 2.1 and area > 700 and mean_r < 0.010 and mean_v < 102:
            return True
        if cy > h * 0.58 and area > 550 and mean_r < 0.009 and mean_s < 52:
            return True
        return False

    def _robust_depth(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, bbox: List[int],
    ) -> Optional[float]:
        dmin, dmax = self._depth_min(), DEPTH_MAX
        d = depth[ys, xs]
        d = d[(d > dmin) & (d < dmax)]
        if len(d) < 4:
            return None
        d_mask = float(np.percentile(d, 38))
        x1, y1, x2, y2 = bbox
        iw, ih = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        inner = [
            x1 + iw // 4, y1 + ih // 4,
            x2 - iw // 4, y2 - ih // 4,
        ]
        ix1, iy1, ix2, iy2 = inner
        inner_d = depth[iy1:iy2 + 1, ix1:ix2 + 1]
        inner_d = inner_d[(inner_d > dmin) & (inner_d < dmax)]
        d_inner = float(np.median(inner_d)) if len(inner_d) >= 3 else d_mask
        return float(np.median([d_mask, d_inner]))

    def _pos_from_mask(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray,
    ) -> Optional[np.ndarray]:
        pts = []
        step = 1 if len(ys) < 120 else 2
        for y, x in zip(ys[::step], xs[::step]):
            z = float(depth[y, x])
            if z <= self._depth_min() or z >= DEPTH_MAX:
                continue
            pts.append(self._uv_depth_to_robot(float(x), float(y), z))
        if len(pts) < 5:
            return None
        return np.median(np.stack(pts, axis=0), axis=0).astype(np.float32)

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1 + 1) * (ay2 - ay1 + 1) + (bx2 - bx1 + 1) * (by2 - by1 + 1) - inter
        return float(inter) / max(ua, 1)

    def _merge_dets(self, dets: List[dict]) -> List[dict]:
        if len(dets) < 2:
            return dets
        dets = sorted(
            dets,
            key=lambda x: (
                -(float(x.get("blob_sat_mean", 0)) * 0.5 + float(x.get("blob_val_mean", 0)) * 0.3),
                x.get("depth_m") or 999.0,
            ),
        )
        kept: List[dict] = []
        for d in dets:
            dup = False
            for k in kept:
                if abs((d.get("depth_m") or 0) - (k.get("depth_m") or 0)) > 0.42:
                    continue
                iou = self._bbox_iou(d["bbox"], k["bbox"])
                if iou > 0.12:
                    dup = True
                    break
                ax1, ay1, ax2, ay2 = d["bbox"]
                bx1, by1, bx2, by2 = k["bbox"]
                cx_d = (ax1 + ax2) * 0.5
                cy_d = (ay1 + ay2) * 0.5
                cx_k = (bx1 + bx2) * 0.5
                cy_k = (by1 + by2) * 0.5
                if float(np.hypot(cx_d - cx_k, cy_d - cy_k)) < 72 and iou > 0.04:
                    dup = True
                    break
            if not dup:
                kept.append(d)
        return kept

    def _blob_det(
        self, ys, xs, depth, h, w, robot_pos, robot_yaw,
        val: Optional[np.ndarray] = None, sat: Optional[np.ndarray] = None,
        relief: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        if self._is_shadow_shape(ys, xs, val, sat, h, w, relief):
            return None
        ys, xs = self._refine_blob(ys, xs, depth, val, h)
        if len(ys) < 5:
            return None
        if self._is_shadow_shape(ys, xs, val, sat, h, w, relief):
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        min_a = int(self._cfg.get("min_area", MIN_AREA))
        min_s = int(self._cfg.get("min_side", MIN_SIDE))
        if min(bw, bh) < min_s or max(bw, bh) > MAX_SIDE or len(ys) < min_a:
            return None
        bbox = [x1, y1, x2, y2]
        depth_m = self._robust_depth(ys, xs, depth, bbox)
        if depth_m is None:
            return None
        if self.camera_name == "ee" and relief is not None:
            rm = float(np.median(relief[ys, xs]))
            if depth_m > 2.2 and len(ys) > 750 and rm < 0.011:
                return None
            if depth_m > 3.5 and len(ys) > 450 and rm < 0.013:
                return None
        d_vals = depth[ys, xs]
        d_vals = d_vals[(d_vals > self._depth_min()) & (d_vals < DEPTH_MAX)]
        if float(np.std(d_vals)) > self._cfg.get("max_depth_std", MAX_BLOB_DEPTH_STD):
            return None
        vm = float(np.mean(val[ys, xs])) if val is not None else 128.0
        sm = float(np.mean(sat[ys, xs])) if sat is not None else 64.0
        if sm < self._cfg.get("min_blob_sat", MIN_BLOB_SAT_MEAN):
            return None
        if self.camera_name == "head" and vm < self._cfg.get("min_blob_val", 62):
            return None
        if relief is not None and self.camera_name == "head":
            rm = float(np.median(relief[ys, xs]))
            rmin_med = float(self._cfg.get("min_relief_med", 0.007))
            if rm < rmin_med and vm < 86 and sm < 50:
                return None
            if rm < rmin_med * 1.4 and len(ys) > 380 and vm < 82:
                return None
        if relief is not None and self.camera_name == "ee":
            rm = float(np.median(relief[ys, xs]))
            rmin_med = float(self._cfg.get("min_relief_med", 0.006))
            if rm < rmin_med and len(ys) > 520:
                return None
        cx, cy = float(np.median(xs)), float(np.median(ys))
        cls, cls_conf = RgbdPureCamera._classify_2d(bbox, len(ys), depth_m)
        pos_r = self._pos_from_mask(ys, xs, depth)
        if pos_r is None:
            pos_r = self._uv_depth_to_robot(cx, cy, depth_m)
        pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
        grasp_r = pos_r.copy()
        z_pts = []
        for y, x in zip(ys[::2], xs[::2]):
            z = float(depth[y, x])
            if self._depth_min() < z < DEPTH_MAX:
                p = self._uv_depth_to_robot(float(x), float(y), z)
                z_pts.append(float(p[2]))
        if z_pts:
            grasp_r[2] = float(np.percentile(z_pts, 82)) - GRASP_DEPTH_OFFSET
        grasp_w = _robot_to_world(grasp_r, robot_pos, robot_yaw)
        size = OBJECT_SIZES.get(cls, DEFAULT_OBJECT_SIZE)
        conf = float(min(0.94, 0.45 + cls_conf * 0.55))
        return {
            "class": cls,
            "class_id": CLASS_NAME_TO_ID.get(cls, -1),
            "conf": conf,
            "class_conf": cls_conf,
            "bbox": bbox,
            "centroid": (cx, cy),
            "centroid_uv": [cx, cy],
            "depth_m": depth_m,
            "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
            "pos_robot": pos_r.tolist(),
            "pos_world": pos_w.tolist(),
            "grasp_pos_world": grasp_w.tolist(),
            "size_world": [size["lx"], size["ly"], size["lz"]],
            "blob_sat_mean": sm,
            "blob_val_mean": vm,
            "source": f"rgbd_fusion_{self.camera_name}",
        }

    def _dets_from_mask(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        robot_pos,
        robot_yaw,
    ) -> List[dict]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        val, sat = hsv[:, :, 2], hsv[:, :, 1]
        near = self._scene_near(depth)
        valid = self._valid_depth(depth, near)
        ground = self._ground_depth(depth, valid)
        relief = ground - depth
        labeled, n = ndimage.label(mask > 0)
        self._mask_n = int(n)
        if n == 0:
            return []
        h, w = depth.shape
        dets = []
        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            det = self._blob_det(
                ys, xs, depth, h, w, robot_pos, robot_yaw,
                val=val, sat=sat, relief=relief,
            )
            if det is not None:
                dets.append(det)
        return dets

    def _detect_bottom_strip(
        self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw,
    ) -> List[dict]:
        """近距物体贴画面底边、主融合仍空时用 hue 黄条带补检"""
        h, w = depth.shape
        near = self._scene_near(depth)
        valid = self._valid_depth(depth, near)
        strip = self._bottom_strip_roi(h, w)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (
            strip & valid
            & (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
            & (sat >= 32) & (val >= 52)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self._debug["bottom_strip"] = mask * 255
        return self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)

    def detect(self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw) -> List[dict]:
        depth = sanitize_depth(depth)
        mask = self._build_fusion_mask(rgb, depth)
        dets = self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)

        if self.camera_name == "head" and len(dets) == 0 and self._scene_near(depth):
            strip_dets = self._detect_bottom_strip(rgb, depth, robot_pos, robot_yaw)
            if strip_dets:
                dets = strip_dets

        dets = self._merge_dets(dets)
        dets.sort(key=lambda x: x.get("depth_m") or 999.0)
        return dets

    def process_frame(
        self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw,
    ) -> Tuple[List[dict], Optional[dict], dict]:
        self.frame_count += 1
        st = depth_stats(depth)
        raw = self.detect(rgb, depth, robot_pos, robot_yaw)
        tracks = self.tracker.update(raw)
        objects = []
        min_hits = int(self._cfg.get("min_track_hits", MIN_TRACK_HITS))
        for t in tracks:
            if self.tracker.tracks.get(t["track_id"], {}).get("hits", 0) < min_hits:
                continue
            o = {**t, "id": int(t["track_id"]), "camera": self.camera_name}
            if o.get("pos_robot") is not None:
                pr = np.asarray(o["pos_robot"], dtype=np.float32)
                o["pos_world"] = _robot_to_world(pr, robot_pos, robot_yaw).tolist()
                o["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
            objects.append(o)
        target = min(objects, key=lambda o: o.get("depth_m") or 999.0) if objects else None
        meta = {"depth_stats": st, "mask_components": getattr(self, "_mask_n", 0)}
        return objects, target, meta

    def _scene_near(self, depth: np.ndarray) -> bool:
        st = depth_stats(depth)
        return st.get("p10", 99) < NEAR_SCENE_P10 or st.get("min", 99) < NEAR_SCENE_MIN

    @staticmethod
    def _classify_2d(bbox: List[int], area: int, depth_m: Optional[float] = None) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        asp = max(bw, bh) / min(bw, bh)
        fill = area / max(bw * bh, 1)
        vert, horiz = bh >= bw * 1.12, bw >= bh * 1.12
        if vert and asp >= 1.20:
            return "mustard_bottle", min(0.84, 0.58 + 0.08 * (asp - 1.2))
        if horiz and asp >= 1.18 and fill < 0.55:
            return "banana", min(0.84, 0.54 + 0.10 * (asp - 1.18))
        if fill > 0.36 and asp < 1.48:
            conf = 0.68 if (depth_m is not None and depth_m < 1.25) else 0.60
            return "sugar_box", conf
        if asp < 1.25:
            return "sugar_box", 0.58
        if horiz:
            return "banana", 0.56
        if vert:
            return "mustard_bottle", 0.57
        return "sugar_box", 0.52


def _robot_to_world(p_robot: np.ndarray, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return robot_pos + rot @ p_robot


class RgbdPurePipeline:
    """单 head 兼容入口"""

    def __init__(self):
        self._cam = RgbdPureCamera("head")
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)
        print("[RgbdPurePipeline] head only — dual cam: RgbdPureDualPipeline")

    def reset(self):
        self._cam.reset()
        self.frame_count = 0
        self.robot_pos = ROBOT_INIT_POS.copy().astype(np.float32)
        self.robot_yaw = float(ROBOT_INIT_YAW)

    def get_debug(self, name: str) -> Optional[np.ndarray]:
        return self._cam.get_debug(name)

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
        rgb, depth = parse_head_rgbd(obs)
        objects, target, meta = self._cam.process_frame(
            rgb, depth, self.robot_pos, self.robot_yaw,
        )
        return {
            "target": target,
            "objects_detailed": objects,
            "objects_remaining": [
                {"id": o["id"], "class": o["class"], "dist": o.get("depth_m"), "pos_world": o.get("pos_world")}
                for o in objects
            ],
            "depth_stats": meta["depth_stats"],
            "mask_components": meta["mask_components"],
            "active_camera": "head",
            "phase": "approach",
            "gripper": {"is_holding": False, "width": 0.04},
            "progress": {"total": TOTAL_OBJECTS, "inside_bin": 0, "remaining": TOTAL_OBJECTS},
            "robot": {"pos_world": self.robot_pos.tolist(), "yaw": self.robot_yaw},
        }
