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
        "roi_v_min": 0.10,
        "roi_v_max": 0.78,
        "roi_v_max_near": 0.93,
        "roi_u_margin": 0.05,
        "relief_min": 0.018,
        "relief_min_near": 0.012,
        "ground_k": 17,
        "sat_extra": 10,
        "sat_relax": 14,
        "min_area": 28,
        "min_side": 5,
        "min_track_hits": 1,
        "mask_close_k": 9,
        "fusion_mode": "soft",  # relief 弱时 RGB+depth 兜底
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
        "min_track_hits": 1,
        "far_mask_dilate": 7,
        "rgb_dilate_far": 5,
        "mask_close_k": 7,
        "fusion_mode": "rgb_depth",  # 远距: 黄+有效深度 (relief 常失效)
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
DEPTH_SMOOTH = 0.55


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
        self, hue, sat, val, roi: np.ndarray, valid: np.ndarray, near: bool,
    ) -> np.ndarray:
        c = self._cfg
        g_sat = float(np.percentile(sat[roi & valid], 50)) if np.any(roi & valid) else 28.0
        relax = int(c.get("sat_relax", 12))
        sat_thr = max(SAT_MIN - 6, g_sat + c["sat_extra"])
        sat_lo = max(16, sat_thr - relax)
        val_lo = VAL_MIN - (10 if self.camera_name == "ee" else 6)
        yellow = (
            (hue >= HUE_LO) & (hue <= HUE_HI)
            & (sat >= sat_lo) & (val >= val_lo)
        )
        high_sat = sat >= max(sat_lo + 4, sat_thr - 6)
        return (yellow | high_sat) & roi

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
        valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
        roi = self._roi(h, w, near)
        c = self._cfg

        ground = self._ground_depth(depth, valid)
        relief = ground - depth

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        rgb_fg = self._rgb_foreground(hue, sat, val, roi, valid, near)

        rd = int(c.get("rgb_dilate_far", 0))
        if self.camera_name == "ee" and not near and rd > 0:
            rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rd, rd))
            rgb_fg = cv2.dilate(rgb_fg.astype(np.uint8), rk, 1).astype(bool)

        depth_ok = self._depth_gate(depth, relief, valid, roi, near)
        fused = (rgb_fg & depth_ok).astype(np.uint8)

        min_a = int(c.get("min_area", MIN_AREA))
        if int(np.sum(fused)) < min_a and int(np.sum(rgb_fg & valid)) >= min_a // 2:
            fused = (rgb_fg & valid & roi).astype(np.uint8)

        if self.camera_name == "head" and near:
            um = c["roi_u_margin"]
            strip = np.zeros((h, w), dtype=bool)
            strip[int(h * 0.60) : h - 2, int(w * um) : int(w * (1 - um))] = True
            fused = cv2.bitwise_or(fused, (rgb_fg & valid & strip).astype(np.uint8))

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

    def _blob_det(
        self, ys, xs, depth, h, w, robot_pos, robot_yaw,
    ) -> Optional[dict]:
        d = depth[ys, xs]
        d = d[(d > DEPTH_MIN) & (d < DEPTH_MAX)]
        if len(d) < 5:
            return None
        depth_m = float(np.percentile(d, 42 if len(d) >= 12 else 50))
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        min_a = int(self._cfg.get("min_area", MIN_AREA))
        min_s = int(self._cfg.get("min_side", MIN_SIDE))
        if min(bw, bh) < min_s or max(bw, bh) > MAX_SIDE or len(ys) < min_a:
            return None
        bbox = [x1, y1, x2, y2]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        cls, cls_conf = RgbdPureCamera._classify_2d(bbox, len(ys))
        pos_r = self._uv_depth_to_robot(cx, cy, depth_m)
        pos_w = _robot_to_world(pos_r, robot_pos, robot_yaw)
        grasp_w = pos_w.copy()
        grasp_w[2] = float(pos_w[2] + 0.06) - GRASP_DEPTH_OFFSET
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
            "source": f"rgbd_fusion_{self.camera_name}",
        }

    def detect(self, rgb: np.ndarray, depth: np.ndarray, robot_pos, robot_yaw) -> List[dict]:
        depth = sanitize_depth(depth)
        mask = self._build_fusion_mask(rgb, depth)
        labeled, n = ndimage.label(mask > 0)
        self._mask_n = int(n)
        if n == 0:
            return []
        h, w = depth.shape
        dets = []
        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            det = self._blob_det(ys, xs, depth, h, w, robot_pos, robot_yaw)
            if det is not None:
                dets.append(det)
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
            objects.append({**t, "id": int(t["track_id"]), "camera": self.camera_name})
        target = min(objects, key=lambda o: o.get("depth_m") or 999.0) if objects else None
        meta = {"depth_stats": st, "mask_components": getattr(self, "_mask_n", 0)}
        return objects, target, meta

    def _scene_near(self, depth: np.ndarray) -> bool:
        st = depth_stats(depth)
        return st.get("p10", 99) < NEAR_SCENE_P10 or st.get("min", 99) < NEAR_SCENE_MIN

    @staticmethod
    def _classify_2d(bbox: List[int], area: int) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        asp = max(bw, bh) / min(bw, bh)
        fill = area / max(bw * bh, 1)
        if bh >= bw * 1.22 and asp >= 1.28:
            return "mustard_bottle", min(0.82, 0.55 + 0.1 * asp)
        if bw >= bh * 1.18 and asp >= 1.20:
            return "banana", min(0.84, 0.52 + 0.12 * asp)
        if asp < 1.22 or (fill > 0.38 and asp < 1.45):
            return "sugar_box", 0.62 if fill > 0.35 else 0.55
        if asp >= 1.55:
            return ("banana", 0.58) if bw >= bh else ("mustard_bottle", 0.56)
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
