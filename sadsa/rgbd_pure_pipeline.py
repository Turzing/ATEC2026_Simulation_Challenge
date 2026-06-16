"""
ATEC Task B — RGB-D 双摄感知

RGB: HSV 黄 / 分深度段饱和度
Depth: relief 融合 + 掩码点云反投影 + PCA 3D 形状分类 (head)
EE:   导航 + 近距抓取 (pos_world / grasp_pos_world / grasp_quat_world)
Head: 导航补充 (站立远距 fallback)

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
    DEFAULT_GRASP_FIXED_QUAT,
    DEFAULT_OBJECT_SIZE,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    GRASP_FIXED_QUAT,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    HEAD_NAV_BOTTOM_FRAC,
    HEAD_NAV_Z_PERCENTILE,
    MIN_NAV_POINT_COUNT,
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
        "val_refine_p": 30,
        "min_blob_sat": 32,
        "min_blob_val": 42,
        "mid_pale_sat_min": 16,
        "mid_pale_val_min": 48,
        "min_relief_med": 0.007,
        "max_shadow_area": 1500,
        "max_depth_std": 0.088,
        "min_track_hits": 1,
        "near_strong_sat": 30,
        "near_strong_val": 50,
        "near_fb_depth_m": 1.65,
        "sat_min": 40,
        "sat_relax_far": 12,
        "val_relax_far": 10,
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
        "min_area": 12,
        "min_side": 3,
        "min_track_hits": 1,
        "far_mask_dilate": 7,
        "rgb_dilate_far": 3,
        "mask_close_k": 5,
        "fusion_mode": "ee_sat",
        "val_refine_p": 28,
        "min_blob_sat": 28,
        "max_depth_std": 0.072,
        "max_ee_blob_area": 2400,
        "sat_min": 32,
        "sat_relax_far": 30,
        "val_relax_far": 18,
        "hue_sat_relax": 30,
        "far_pale_sat_min": 12,
        "far_pale_val_min": 44,
        "mid_pale_sat_min": 12,
        "mid_pale_val_min": 44,
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
FAR_DEPTH_M = 2.0
HEAD_HUE_LO, HEAD_HUE_HI = 12, 50
EE_HUE_LO, EE_HUE_HI = 12, 50

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

# ── 3D 形状分类 (head 精定位 / ee 辅助) ───────────────────────────
MIN_3D_POINTS = 8
MIN_CLASS_MARGIN = 0.10
GEOM_MATCH_SIGMA = 0.065
RATIO_MATCH_SIGMA = 0.35
FLAT_TOP_E0_M = 0.045
FLAT_TOP_E1_M = 0.085
SHALLOW_TOP_E0_M = 0.055
HEAD_DEPTH_POINT_TOL = 0.06
EE_DEPTH_POINT_TOL_NEAR = 0.10
EE_DEPTH_POINT_TOL_FAR = 0.14
EE_GRASP_NEAR_M = 1.00
EE_GRASP_MIN_POINTS = 6
GRASP_RELIABLE_DEPTH_M = 0.80
WORLD_RELIABLE_DEPTH_M = 2.00


def _build_ratio_templates() -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for name in OBJECT_SIZES:
        e = np.sort([OBJECT_SIZES[name]["lx"], OBJECT_SIZES[name]["ly"], OBJECT_SIZES[name]["lz"]])
        out[name] = (float(e[1] / max(e[0], 1e-4)), float(e[2] / max(e[1], 1e-4)))
    return out


_CLASS_RATIO_TEMPLATES = _build_ratio_templates()


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def _quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float32)


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
        self._arm_joints: Optional[np.ndarray] = None
        self._projected_gravity: Optional[np.ndarray] = None

    def reset(self):
        self.tracker.reset()
        self.frame_count = 0
        self._debug.clear()
        self._arm_joints = None
        self._projected_gravity = None

    def set_arm_joints(self, arm_joints) -> None:
        if arm_joints is None:
            self._arm_joints = None
        else:
            self._arm_joints = np.asarray(arm_joints, dtype=np.float32).reshape(-1)[:6]

    def set_projected_gravity(self, grav) -> None:
        if grav is None:
            self._projected_gravity = None
        else:
            self._projected_gravity = np.asarray(grav, dtype=np.float32).reshape(3)

    def _cam_pos_robot(self) -> np.ndarray:
        if self.camera_name == "ee" and self._arm_joints is not None:
            from rgbd_utils import compute_dynamic_ee_cam_pos
            return compute_dynamic_ee_cam_pos(self._arm_joints)
        if self.camera_name == "head" and self._projected_gravity is not None:
            from rgbd_utils import compute_dynamic_head_cam_pos
            return compute_dynamic_head_cam_pos(self._projected_gravity)
        return self._cfg["pos"]

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

    def _ground_refs(
        self, sat: np.ndarray, val: np.ndarray, roi: np.ndarray, h: int,
    ) -> Tuple[float, float]:
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

    def _build_sat_mask(
        self,
        hue: np.ndarray,
        sat: np.ndarray,
        val: np.ndarray,
        depth: np.ndarray,
        roi: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """分深度段饱和度掩码 — EE 远距检出主路径 (对齐 rgbd_detect)"""
        c = self._cfg
        h = depth.shape[0]
        g_sat, g_val = self._ground_refs(sat, val, roi, h)
        sat_min = float(c.get("sat_min", 40))
        sat_thr = max(sat_min, g_sat + (18 if self.camera_name == "ee" else 14))
        val_thr = max(48 if self.camera_name == "ee" else 52, g_val + (12 if self.camera_name == "ee" else 10))

        relax_far = int(c.get("sat_relax_far", 16))
        relax_far_v = int(c.get("val_relax_far", 10))
        sat_far = max(22 if self.camera_name == "ee" else 36, sat_thr - relax_far)
        val_far = max(38 if self.camera_name == "ee" else 46, val_thr - relax_far_v)

        very_near = depth < VERY_NEAR_DEPTH_M
        near = (depth >= VERY_NEAR_DEPTH_M) & (depth < FAR_DEPTH_M)
        far = depth >= FAR_DEPTH_M

        relax_vn = int(c.get("sat_relax_very_near", 18))
        relax_vn_v = int(c.get("val_relax_very_near", 20))
        sat_vn = max(26 if self.camera_name == "ee" else 28, sat_thr - relax_vn)
        val_vn = max(38 if self.camera_name == "ee" else 42, val_thr - relax_vn_v)

        hue_lo = EE_HUE_LO if self.camera_name == "ee" else HEAD_HUE_LO
        hue_hi = EE_HUE_HI if self.camera_name == "ee" else HEAD_HUE_HI

        m_vn = (sat >= sat_vn) & (val >= val_vn) & very_near
        m_near = (sat >= sat_thr) & (val >= val_thr) & near
        m_far = (sat >= sat_far) & (val >= val_far) & far
        sat_mask = (m_vn | m_near | m_far) & roi & valid

        hue_relax = int(c.get("hue_sat_relax", 22))
        sat_hue = max(20, sat_thr - hue_relax)
        yellow = (
            (hue >= hue_lo) & (hue <= hue_hi)
            & (sat >= sat_hue) & (val >= val_thr - 18)
            & roi & valid
        )
        detect = (sat_mask | yellow).astype(np.uint8)

        cam = self.camera_name
        if cam in ("ee", "head"):
            # 中近距淡黄/白黄糖盒: 1.5~2m 最常见漏检带 (上次 far_pale 只管 >=2m)
            mid_pale_s = int(c.get("mid_pale_sat_min", 14))
            mid_pale_v = int(c.get("mid_pale_val_min", 46))
            mid_band = (depth >= 0.85) & (depth < (FAR_DEPTH_M + 0.55))
            hue_lo_p = EE_HUE_LO if cam == "ee" else HEAD_HUE_LO
            hue_hi_p = EE_HUE_HI if cam == "ee" else HEAD_HUE_HI
            mid_pale = (
                mid_band & roi & valid
                & (hue >= hue_lo_p) & (hue <= hue_hi_p)
                & (sat >= mid_pale_s) & (val >= mid_pale_v)
            )
            detect = (detect.astype(bool) | mid_pale).astype(np.uint8)

        if self.camera_name == "ee":
            pale_s = int(c.get("far_pale_sat_min", 14))
            pale_v = int(c.get("far_pale_val_min", 48))
            far_pale = (
                far & roi & valid
                & (hue >= EE_HUE_LO) & (hue <= EE_HUE_HI)
                & (sat >= pale_s) & (val >= pale_v)
            )
            detect = (detect.astype(bool) | far_pale).astype(np.uint8)

        if self.camera_name == "head":
            shadow = (val < 58) & (sat < 36)
            detect = (detect.astype(bool) & ~shadow).astype(np.uint8)

        near_u8 = cv2.bitwise_and(detect, (very_near | near).astype(np.uint8))
        far_u8 = cv2.bitwise_and(detect, far.astype(np.uint8))
        near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if self.camera_name == "ee":
            # 远距 blob 仅几个像素; OPEN 会把糖盒抹掉, 只做 CLOSE 连通
            far_u8 = cv2.morphologyEx(far_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        else:
            far_u8 = cv2.morphologyEx(far_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            far_u8 = cv2.morphologyEx(far_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.bitwise_or(near_u8, far_u8)

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

        if mode in ("ee_nav", "ee_sat"):
            return valid & (depth > 0.35) & (depth < 10.8)

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
        sat_mask = self._build_sat_mask(hue, sat, val, depth, roi, valid)

        rmin = c["relief_min_near"] if near else c["relief_min"]
        depth_fg = (relief >= rmin) & (relief <= RELIEF_MAX) & valid
        roi_d = depth[roi & valid]
        closer = np.zeros_like(depth, dtype=bool)
        if roi_d.size > 80:
            d_bg = float(np.percentile(roi_d, 58))
            closer = valid & (depth < d_bg - CLOSER_THAN_BG_M)

        if self.camera_name == "ee":
            fused = sat_mask.copy()
            fd = int(c.get("far_mask_dilate", 5))
            if fd > 0:
                far = (depth >= FAR_DEPTH_M) & valid & roi
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fd, fd))
                far_u8 = cv2.dilate((sat_mask > 0).astype(np.uint8) & far.astype(np.uint8), k, 1)
                fused = cv2.bitwise_or(fused, far_u8)
            depth_ok = valid & (depth > 0.35) & (depth < 10.8)
            rgb_fg = (sat_mask > 0) | rgb_fg
        elif near:
            has_cue = depth_fg | closer | (relief >= rmin * 0.28)
            relief_f = (rgb_fg & valid & has_cue).astype(np.uint8)
            ss = int(c.get("near_strong_sat", 40))
            sv = int(c.get("near_strong_val", 58))
            fb_d = float(c.get("near_fb_depth_m", 1.65))
            strong = (
                (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
                & (sat >= ss) & (val >= sv)
            )
            mid_near = valid & (depth > self._depth_min()) & (depth < fb_d)
            fused_fb = (rgb_fg & mid_near & strong).astype(np.uint8)
            fused = cv2.bitwise_or(sat_mask, relief_f)
            fused = cv2.bitwise_or(fused, fused_fb)
            depth_ok = (sat_mask > 0) | has_cue | (mid_near & strong)
        else:
            depth_ok = self._depth_gate(depth, relief, valid, roi, near)
            fused = cv2.bitwise_or(sat_mask, (rgb_fg & depth_ok).astype(np.uint8))

        ck = int(c.get("mask_close_k", 5))
        fused = self._close_components(fused, ck)

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
        return (self._cam_pos_robot() + self._cfg["rot"] @ p_cam).astype(np.float32)

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
            v = val[ys, xs].astype(np.float32)
            v_cut = float(np.percentile(v, self._cfg.get("val_refine_p", 35)))
            v_floor = 42 if self.camera_name == "ee" else (SHADOW_VAL_MAX - 18)
            keep = v >= max(v_floor, v_cut)
            if int(np.sum(keep)) >= max(5, len(ys) // 4):
                ys, xs = ys[keep], xs[keep]
        if self.camera_name == "head" and val is not None and len(ys) > 8:
            v = val[ys, xs].astype(np.float32)
            v_cut = float(np.percentile(v, self._cfg.get("val_refine_p", 32)))
            keep = v >= max(52, v_cut)
            if int(np.sum(keep)) >= max(5, len(ys) // 4):
                ys, xs = ys[keep], xs[keep]
                v = val[ys, xs].astype(np.float32)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            if bw > bh * 1.35 and len(ys) > 60 and v.size == xs.size:
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
        val: Optional[np.ndarray],
        sat: Optional[np.ndarray],
        h: int,
        w: int,
        relief: Optional[np.ndarray] = None,
    ) -> bool:
        if len(ys) < 5 or val is None or sat is None:
            return False
        if val.shape[:2] != (h, w) or sat.shape[:2] != (h, w):
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
        # 中近距淡黄糖盒: 地面 relief 弱 + 饱和度低, 勿当影子
        if area < 280 and mean_s >= 26 and mean_v >= 52 and mean_r >= 0.003:
            return False
        if area < 120 and mean_v >= 58 and mean_s >= 22:
            return False

        if mean_r < 0.007 and mean_v < 82:
            return True
        if mean_r < 0.009 and mean_s < 36:
            return True
        # 蹲下: 灰色臂/躯干贴脸 — 低饱和+近距+画面中下
        if self.camera_name == "head" and cy > h * 0.26 and mean_s < 54 and mean_v > 58:
            if mean_r < 0.018 and area < 900:
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
        """EE: 只剔垃圾箱级大扁影, 不杀远处小目标"""
        c = self._cfg
        if area < 420:
            return False
        if mean_s >= 44 and mean_v >= 72:
            return False
        if area > int(c.get("max_ee_blob_area", 2400)):
            return True
        if img_frac > 0.09 and mean_r < 0.009 and mean_s < 46:
            return True
        if area > 1100 and mean_r < 0.007 and mean_v < 95:
            return True
        if bw > w * 0.34 and bh < h * 0.12 and mean_r < 0.009 and area > 800:
            return True
        if aspect > 2.3 and area > 1000 and mean_r < 0.008 and mean_s < 48:
            return True
        if cy > h * 0.60 and area > 900 and mean_r < 0.007 and mean_s < 46:
            return True
        if cy < h * 0.38 and mean_r < 0.012 and area < 1800:
            return True
        if cy < h * 0.30 and mean_s >= 40 and mean_v >= 70:
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

    def _filter_blob_pixels_by_depth(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, depth_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = depth[ys, xs]
        dmin = self._depth_min()
        ok = (d > dmin) & (d < DEPTH_MAX)
        if int(np.sum(ok)) < MIN_3D_POINTS:
            return ys, xs
        if self.camera_name == "head":
            tol = HEAD_DEPTH_POINT_TOL
        else:
            tol = EE_DEPTH_POINT_TOL_NEAR if depth_m < FAR_DEPTH_M else EE_DEPTH_POINT_TOL_FAR
        d_med = float(np.median(d[ok]))
        keep = ok & (d <= d_med + tol) & (d >= d_med - tol * 0.55)
        if int(np.sum(keep)) >= max(6, MIN_3D_POINTS // 2):
            return ys[keep], xs[keep]
        return ys, xs

    def _blob_points_robot(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, depth_m: float,
    ) -> Optional[np.ndarray]:
        ys, xs = self._filter_blob_pixels_by_depth(ys, xs, depth, depth_m)
        pts = []
        step = 1 if len(ys) < 400 else 2
        dmin = self._depth_min()
        for y, x in zip(ys[::step], xs[::step]):
            z = float(depth[y, x])
            if z <= dmin or z >= DEPTH_MAX:
                continue
            pts.append(self._uv_depth_to_robot(float(x), float(y), z))
        if len(pts) < MIN_3D_POINTS:
            return None
        return np.stack(pts, axis=0).astype(np.float32)

    @staticmethod
    def _template_extents(name: str) -> np.ndarray:
        s = OBJECT_SIZES[name]
        return np.sort([s["lx"], s["ly"], s["lz"]]).astype(np.float32)

    def _pca_sorted_extents(self, pts: np.ndarray) -> np.ndarray:
        centered = pts - np.mean(pts, axis=0)
        if len(centered) < MIN_3D_POINTS:
            return np.sort(np.ptp(pts, axis=0)).astype(np.float32)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            ext = np.ptp(centered @ vt.T, axis=0)
        except np.linalg.LinAlgError:
            ext = np.ptp(pts, axis=0)
        return np.sort(np.maximum(ext, 1e-4)).astype(np.float32)

    @staticmethod
    def _extent_ratios(ext: np.ndarray) -> Tuple[float, float, float]:
        e0, e1, e2 = float(ext[0]), float(ext[1]), float(ext[2])
        return e1 / max(e0, 1e-4), e2 / max(e1, 1e-4), e0 / max(e2, 1e-4)

    def _score_3d_shape(self, ext: np.ndarray) -> Dict[str, float]:
        r10, r21, flat = self._extent_ratios(ext)
        scores: Dict[str, float] = {}
        for name in OBJECT_SIZES:
            t_ext = self._template_extents(name)
            d_ext = float(np.linalg.norm(ext - t_ext))
            s_ext = float(np.exp(-(d_ext / GEOM_MATCH_SIGMA) ** 2))
            tr10, tr21 = _CLASS_RATIO_TEMPLATES[name]
            d_ratio = float(np.hypot(r10 - tr10, r21 - tr21))
            s_ratio = float(np.exp(-(d_ratio / RATIO_MATCH_SIGMA) ** 2))
            scores[name] = 0.38 * s_ext + 0.62 * s_ratio
        if r10 < 1.18:
            scores["banana"] = scores.get("banana", 0) * 1.55
            scores["sugar_box"] = scores.get("sugar_box", 0) * 0.75
        if flat < 0.26 and r10 > 1.35:
            scores["sugar_box"] = scores.get("sugar_box", 0) * 1.45
        if 1.25 < r10 < 1.65 and 1.9 < r21 < 2.8:
            scores["mustard_bottle"] = scores.get("mustard_bottle", 0) * 1.40
        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _classify_2d_shape(
        bbox: List[int], area: int, sat_mean: float, depth_m: Optional[float] = None,
    ) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        asp = max(bw, bh) / min(bw, bh)
        fill = area / max(bw * bh, 1)
        vert, horiz = bh >= bw * 1.20, bw >= bh * 1.20
        if fill > 0.36 and asp < 1.50:
            conf = 0.72 if (depth_m is not None and depth_m < 1.30) else 0.65
            return "sugar_box", conf
        if vert and asp >= 1.32:
            return "mustard_bottle", min(0.86, 0.56 + 0.11 * (asp - 1.32))
        if horiz and asp >= 1.22:
            return "banana", min(0.84, 0.54 + 0.13 * (asp - 1.22))
        if asp >= 1.72:
            return ("mustard_bottle", 0.64) if vert else ("banana", 0.66)
        if asp < 1.16:
            return "sugar_box", 0.60
        return "sugar_box", 0.55

    def _classify_object(
        self,
        bbox: List[int],
        area: int,
        ys: np.ndarray,
        xs: np.ndarray,
        depth: np.ndarray,
        sat_mean: float,
        depth_m: float,
        use_3d: bool,
    ) -> Tuple[str, float, Optional[List[float]]]:
        name_2d, conf_2d = self._classify_2d_shape(bbox, area, sat_mean, depth_m)
        if not use_3d:
            return name_2d, conf_2d, None
        pts = self._blob_points_robot(ys, xs, depth, depth_m)
        if pts is None:
            return name_2d, conf_2d * 0.85, None
        ext = self._pca_sorted_extents(pts)
        e0, e1, e2 = float(ext[0]), float(ext[1]), float(ext[2])
        if (
            (e0 < FLAT_TOP_E0_M and e1 < FLAT_TOP_E1_M)
            or (e0 < SHALLOW_TOP_E0_M and e2 > 0.11)
            or e2 < 0.025
        ):
            return name_2d, conf_2d, ext.tolist()
        scores = self._score_3d_shape(ext)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = s1 - s2
        cls_conf = float(min(0.94, 0.5 + margin * 1.2 + min(0.15, len(pts) / 200.0)))
        if margin < MIN_CLASS_MARGIN:
            if conf_2d >= 0.50:
                return name_2d, conf_2d * 0.88, ext.tolist()
            return name_2d, cls_conf * 0.7, ext.tolist()
        return best, cls_conf, ext.tolist()

    @staticmethod
    def _grasp_from_robot_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.median(pts, axis=0).astype(np.float32)
        z_top = float(np.percentile(pts[:, 2], 88))
        grasp = pos.copy()
        grasp[2] = z_top - GRASP_DEPTH_OFFSET
        return pos, grasp

    def _compute_grasp_quat(self, class_name: str, pos_world: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        fixed = GRASP_FIXED_QUAT.get(str(class_name), DEFAULT_GRASP_FIXED_QUAT).copy()
        dx = float(pos_world[0] - robot_pos[0])
        dy = float(pos_world[1] - robot_pos[1])
        yaw = float(np.arctan2(dy, dx))
        return _quat_multiply_wxyz(_quat_from_yaw_wxyz(yaw), fixed)

    def _pos_from_mask(
        self, ys: np.ndarray, xs: np.ndarray, depth: np.ndarray, depth_m: float,
    ) -> Optional[np.ndarray]:
        pts = self._blob_points_robot(ys, xs, depth, depth_m)
        if pts is not None:
            return np.median(pts, axis=0).astype(np.float32)
        raw = []
        step = 1 if len(ys) < 120 else 2
        dmin = self._depth_min()
        for y, x in zip(ys[::step], xs[::step]):
            z = float(depth[y, x])
            if z <= dmin or z >= DEPTH_MAX:
                continue
            raw.append(self._uv_depth_to_robot(float(x), float(y), z))
        if len(raw) < 5:
            return None
        return np.median(np.stack(raw, axis=0), axis=0).astype(np.float32)

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
        dmin_gate = self._depth_min()
        if self.camera_name == "ee" and depth_m < 1.45:
            blob_d = depth[ys, xs]
            blob_d = blob_d[(blob_d > dmin_gate) & (blob_d < DEPTH_MAX)]
            if blob_d.size >= 4:
                depth_m = float(np.percentile(blob_d, 12))
        if self.camera_name == "ee" and relief is not None:
            rm = float(np.median(relief[ys, xs]))
            if len(ys) > 1200 and rm < 0.007 and float(np.mean(val[ys, xs])) < 90:
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
        if self.camera_name == "head" and depth_m < 1.05 and sm < 46:
            return None
        if self.camera_name == "head" and depth_m < 0.80 and sm < 54:
            return None
        if relief is not None and self.camera_name == "head":
            rm = float(np.median(relief[ys, xs]))
            asp = max(bw, bh) / max(min(bw, bh), 1)
            if rm < 0.006 and vm < 78 and sm < 44 and len(ys) > 500:
                return None
            if rm < 0.007 and len(ys) > 700 and asp > 1.8 and vm < 84:
                return None
        cx, cy = float(np.median(xs)), float(np.median(ys))
        is_head = self.camera_name == "head"
        is_ee = self.camera_name == "ee"
        x1, y1, x2, y2 = bbox

        # ── head: 点云 + 底边 anchor 导航 (近端 depth, 避免 bbox 底边打到地面) ──
        if is_head:
            nav_u = float(cx)
            nav_v = float(y2)
            dmin, dmax = self._depth_min(), DEPTH_MAX
            blob_d = depth[ys, xs]
            blob_d = blob_d[(blob_d > dmin) & (blob_d < dmax)]
            if blob_d.size >= 6:
                az = float(np.percentile(blob_d, 10))
                near_idx = int(np.argmin(np.abs(blob_d - az)))
                near_y, near_x = int(ys[near_idx]), int(xs[near_idx])
                nav_u = float(near_x)
                nav_v = float(near_y)
            else:
                au = int(np.clip(nav_u, 0, w - 1))
                av = int(np.clip(nav_v, 0, h - 1))
                az = float(depth[av, au])
                if az <= dmin or az >= dmax:
                    az = float(depth_m)

            pts = self._blob_points_robot(ys, xs, depth, depth_m)
            pos_from_pc = False
            nav_point_count = 0
            if pts is not None and len(pts) >= MIN_NAV_POINT_COUNT:
                y_cut = float(np.quantile(ys.astype(np.float32), HEAD_NAV_BOTTOM_FRAC))
                bot = ys >= y_cut
                if int(np.sum(bot)) >= 6:
                    pts_bot = []
                    step = 1 if int(np.sum(bot)) < 400 else 2
                    dmin = self._depth_min()
                    for y, x in zip(ys[bot][::step], xs[bot][::step]):
                        z = float(depth[y, x])
                        if z <= dmin or z >= DEPTH_MAX:
                            continue
                        pts_bot.append(self._uv_depth_to_robot(float(x), float(y), z))
                    if len(pts_bot) >= 6:
                        pts_bot = np.stack(pts_bot, axis=0).astype(np.float32)
                        pos_r = np.median(pts_bot, axis=0).astype(np.float32)
                        pos_r[2] = float(np.percentile(pts[:, 2], HEAD_NAV_Z_PERCENTILE))
                    else:
                        pos_r = np.median(pts, axis=0).astype(np.float32)
                else:
                    pos_r = np.median(pts, axis=0).astype(np.float32)
                pos_from_pc = True
                nav_point_count = int(len(pts))
            else:
                pos_r = self._uv_depth_to_robot(nav_u, nav_v, az)
                if pos_r is None:
                    pos_r = self._pos_from_mask(ys, xs, depth, depth_m)
                if pos_r is None:
                    return None
                nav_point_count = int(len(ys))
            if float(pos_r[2]) < -0.78 or float(pos_r[2]) > 0.28:
                return None
            if depth_m < 0.85 and sm < 58 and cy > h * 0.30 and abs(cx - w * 0.5) < w * 0.34:
                return None
            pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
            cls, cls_conf, geom_ext = self._classify_object(
                bbox, len(ys), ys, xs, depth, sm, depth_m, use_3d=True,
            )
            conf = float(min(0.94, 0.45 + cls_conf * 0.55))
            yaw_rel = float(np.arctan2(pos_r[1], pos_r[0]))
            depth_f = float(az if az > self._depth_min() else depth_m)
            return {
                "class": cls,
                "class_id": CLASS_NAME_TO_ID.get(cls, -1),
                "conf": conf,
                "class_conf": cls_conf,
                "bbox": bbox,
                "centroid": (cx, cy),
                "centroid_uv": [cx, cy],
                "nav_anchor_uv": [nav_u, nav_v],
                "nav_anchor_depth": az,
                "depth_m": depth_f,
                "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
                "yaw_rel": yaw_rel,
                "nav_depth_m": depth_f,
                "nav_yaw_rel": yaw_rel,
                "pos_robot": pos_r.tolist(),
                "pos_world": pos_w.tolist(),
                "pos_from_pointcloud": pos_from_pc,
                "nav_point_count": nav_point_count,
                "blob_sat_mean": sm,
                "blob_val_mean": vm,
                "source": "rgbd_nav_head",
                "camera": "head",
                "role": "nav",
                "world_reliable": depth_f < WORLD_RELIABLE_DEPTH_M and pos_from_pc,
                "grasp_reliable": False,
            }

        use_3d = is_ee and (len(ys) >= 24 or depth_m < EE_GRASP_NEAR_M)
        cls, cls_conf, geom_ext = self._classify_object(
            bbox, len(ys), ys, xs, depth, sm, depth_m, use_3d=use_3d,
        )
        pts = self._blob_points_robot(ys, xs, depth, depth_m)
        use_grasp_pc = pts is not None and len(pts) >= EE_GRASP_MIN_POINTS and depth_m < EE_GRASP_NEAR_M

        # EE 导航 pos: centroid + depth
        pos_r = self._uv_depth_to_robot(cx, cy, depth_m)
        if pos_r is None:
            pos_r = self._pos_from_mask(ys, xs, depth, depth_m)
        if pos_r is None:
            return None
        if float(pos_r[2]) < -0.78 or float(pos_r[2]) > 0.28:
            return None

        # EE 抓取 anchor: bbox 底边中心 (贴地物体更准)
        anchor_u = cx
        anchor_v = float(y2)
        au, av = int(np.clip(anchor_u, 0, w - 1)), int(np.clip(anchor_v, 0, h - 1))
        az = float(depth[av, au])
        if az <= self._depth_min() or az >= DEPTH_MAX:
            az = float(depth_m)
        grasp_offset = np.array([0.0, 0.0, -GRASP_DEPTH_OFFSET], dtype=np.float32)
        if use_grasp_pc:
            z_grasp = float(np.percentile(pts[:, 2], 88)) - GRASP_DEPTH_OFFSET
            grasp_offset[2] = z_grasp - float(pos_r[2])
        grasp_r = pos_r + grasp_offset

        pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
        grasp_w = _robot_to_world(grasp_r, robot_pos, robot_yaw)
        grasp_q = self._compute_grasp_quat(cls, pos_w, robot_pos).tolist()
        grasp_reliable = float(depth_m) < GRASP_RELIABLE_DEPTH_M
        world_reliable = float(depth_m) < WORLD_RELIABLE_DEPTH_M
        size = OBJECT_SIZES.get(cls, DEFAULT_OBJECT_SIZE)
        conf = float(min(0.94, 0.45 + cls_conf * 0.55))
        yaw_rel = float(np.arctan2(pos_r[1], pos_r[0]))
        out = {
            "class": cls,
            "class_id": CLASS_NAME_TO_ID.get(cls, -1),
            "conf": conf,
            "class_conf": cls_conf,
            "bbox": bbox,
            "centroid": (cx, cy),
            "centroid_uv": [cx, cy],
            "depth_m": depth_m,
            "dist_to_robot": float(np.linalg.norm(pos_r[:2])),
            "yaw_rel": yaw_rel,
            "pos_robot": pos_r.tolist(),
            "pos_world": pos_w.tolist(),
            "grasp_pos_robot": grasp_r.tolist(),
            "grasp_offset_robot": grasp_offset.tolist(),
            "grasp_pos_world": grasp_w.tolist(),
            "size_world": [size["lx"], size["ly"], size["lz"]],
            "blob_sat_mean": sm,
            "blob_val_mean": vm,
            "source": f"rgbd_fusion_{self.camera_name}",
            "camera": "ee",
            "role": "nav_grasp",
            "grasp_anchor_uv": [anchor_u, anchor_v],
            "grasp_anchor_depth": az,
            "nav_depth_m": float(depth_m),
            "nav_yaw_rel": yaw_rel,
            "grasp_reliable": grasp_reliable,
            "world_reliable": world_reliable,
        }
        if geom_ext is not None:
            out["geom_extents"] = geom_ext
        if grasp_q is not None:
            out["grasp_quat_world"] = grasp_q
        return out

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

    def _detect_head_far_yellow(
        self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw,
    ) -> List[dict]:
        """head 1.2~6m 远距黄物 (fusion relief 漏检时 head=0 的根因)."""
        h, w = depth.shape
        valid = self._valid_depth(depth, near=False) & (depth >= 1.0) & (depth <= 6.5)
        roi = self._roi(h, w, near=False)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (
            roi & valid
            & (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
            & (sat >= 14) & (val >= 36)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        dets = self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)
        for d in dets or []:
            d["head_far_fallback"] = True
            d["pos_confidence"] = min(float(d.get("pos_confidence") or 0.5), 0.62)
        return dets

    def _detect_midfield_yellow(
        self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw,
    ) -> List[dict]:
        """head 2~4m 画面中部香蕉/黄物补检 (log: head=0 但 EE 见 banana@2.3m)."""
        h, w = depth.shape
        st = depth_stats(depth)
        p10 = float(st.get("p10", 99.0))
        if p10 < 1.0 or p10 > 5.5:
            return []
        roi = np.zeros((h, w), dtype=bool)
        roi[int(h * 0.08) : int(h * 0.58), int(w * 0.10) : int(w * 0.90)] = True
        valid = (
            self._valid_depth(depth, near=False)
            & (depth >= 1.0)
            & (depth <= 5.0)
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (
            roi & valid
            & (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
            & (sat >= 20) & (val >= 42)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)

    def _detect_edge_yellow(
        self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw,
    ) -> List[dict]:
        """画面左右边缘黄物 (截图: banana 在 FOV 左缘 head YOLO 漏检)."""
        h, w = depth.shape
        st = depth_stats(depth)
        p10 = float(st.get("p10", 99.0))
        if p10 < 1.0 or p10 > 5.5:
            return []
        valid = (
            self._valid_depth(depth, near=False)
            & (depth >= 1.0)
            & (depth <= 5.0)
        )
        roi = np.zeros((h, w), dtype=bool)
        roi[int(h * 0.05) : int(h * 0.62), : int(w * 0.18)] = True
        roi[int(h * 0.05) : int(h * 0.62), int(w * 0.82) :] = True
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask = (
            roi & valid
            & (hue >= HEAD_HUE_LO) & (hue <= HEAD_HUE_HI)
            & (sat >= 18) & (val >= 40)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)

    def detect(self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw) -> List[dict]:
        from rgbd_utils import align_rgb_to_depth
        depth = sanitize_depth(depth)
        rgb = align_rgb_to_depth(rgb, depth)
        mask = self._build_fusion_mask(rgb, depth)
        dets = self._dets_from_mask(mask, rgb, depth, robot_pos, robot_yaw)

        if self.camera_name == "head" and len(dets) == 0:
            st = depth_stats(depth)
            p10 = float(st.get("p10", 99.0))
            far_dets = self._detect_head_far_yellow(rgb, depth, robot_pos, robot_yaw)
            if far_dets:
                dets = far_dets
            elif self._scene_near(depth):
                strip_dets = self._detect_bottom_strip(rgb, depth, robot_pos, robot_yaw)
                if strip_dets:
                    dets = strip_dets
            elif 1.0 <= p10 <= 5.5:
                mid_dets = self._detect_midfield_yellow(rgb, depth, robot_pos, robot_yaw)
                if mid_dets:
                    dets = mid_dets
                else:
                    edge_dets = self._detect_edge_yellow(rgb, depth, robot_pos, robot_yaw)
                    if edge_dets:
                        dets = edge_dets

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
        min_hits = 1 if self.camera_name == "head" else int(self._cfg.get("min_track_hits", MIN_TRACK_HITS))
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
