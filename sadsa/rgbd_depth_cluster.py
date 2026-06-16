"""
Head/EE 纯 depth 检测: 点云 → RANSAC 剔桌面 → 欧氏聚类 → 3D + RGB 分类

不再用 relief/Gaussian 假凸起 mask (log: head=0 / 位置偏 1m+).
"""

from __future__ import annotations

PERCEPTION_BUILD_ID = "20260617-ransac-head"

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CLASS_NAME_TO_ID,
    DEFAULT_GRASP_FIXED_QUAT,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    GRASP_FIXED_QUAT,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
)
from depth_ransac_cluster import _euclidean_clusters, _ransac_plane
from rgbd_utils import (
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    pixel_to_robot,
    sanitize_depth,
)

# --- head 粗导航 (远距、俯视) ---
HEAD_DEPTH_MIN = 0.22
HEAD_DEPTH_MAX = 6.0
HEAD_PIXEL_STEP = 2
HEAD_ROI_V_MIN = 0.08
HEAD_ROI_V_MAX = 0.72
HEAD_ROI_U_MIN = 0.04
HEAD_ROI_U_MAX = 0.96
HEAD_RANSAC_ITERS = 220
HEAD_RANSAC_THRESH = 0.018
HEAD_RANSAC_MIN_INLIERS = 55
HEAD_CLUSTER_EPS = 0.048
HEAD_CLUSTER_MIN_PTS = 12
HEAD_ROBOT_Z_MIN = -0.52
HEAD_ROBOT_Z_MAX = 0.58
HEAD_MIN_XY_DIST = 0.35
HEAD_MAX_XY_DIST = 5.5

# --- EE ---
EE_DEPTH_MIN = 0.15
EE_DEPTH_MAX = 2.8
EE_PIXEL_STEP = 2
EE_ROI_V_MIN = 0.05
EE_ROI_V_MAX = 0.94
EE_RANSAC_MIN_INLIERS = 100
EE_CLUSTER_EPS = 0.028
EE_CLUSTER_MIN_PTS = 30
EE_GRASP_NEAR_M = 1.15

MAX_CLUSTERS = 16
MIN_SAMPLE_PTS = 4


def _robot_to_world(p_robot: np.ndarray, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return robot_pos + rot @ p_robot


def _quat_multiply_wxyz(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def _quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    h = 0.5 * yaw
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float32)


def _classify_cluster_rgb(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    bbox: List[int],
    extent: np.ndarray,
) -> Tuple[str, float]:
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    aspect = bw / max(bh, 1)
    z_extent = float(np.max(extent))

    patch = rgb[ys, xs] if rgb is not None and len(ys) > 0 else None
    if patch is None or patch.size == 0:
        return _classify_geometry_only(extent)

    bgr = patch[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = float(np.median(hsv[:, 0]))
    sat = float(np.median(hsv[:, 1]))
    yellow = 14 <= hue <= 46 and sat >= 22

    scores = {"sugar_box": 0.15, "mustard_bottle": 0.12, "banana": 0.12}
    if yellow:
        scores["mustard_bottle"] += 0.30
        scores["banana"] += 0.24
    if bw >= bh * 1.12 and aspect > 1.22:
        scores["banana"] += 0.38
    elif bh >= bw * 1.10 and aspect > 1.18:
        scores["mustard_bottle"] += 0.34
    elif aspect < 1.22:
        scores["sugar_box"] += 0.36
    if z_extent < 0.09 and aspect < 1.35:
        scores["sugar_box"] += 0.28
    if z_extent > 0.10 and yellow:
        scores["mustard_bottle"] += 0.22
    name = max(scores, key=scores.get)
    return name, float(min(0.90, scores[name]))


def _classify_geometry_only(extent: np.ndarray) -> Tuple[str, float]:
    ex = np.sort(np.maximum(extent, 1e-3))
    short, mid, tall = float(ex[0]), float(ex[1]), float(ex[2])
    aspect = tall / max(short, 1e-3)
    if tall >= 0.07 and aspect >= 1.30:
        return "mustard_bottle", 0.68
    if mid >= short * 1.20 and tall < 0.10:
        return "banana", 0.66
    return "sugar_box", 0.64


class DepthClusterDetector:
    """RANSAC 剔桌 + 欧氏聚类 → 3D 目标 (head 粗导航 / ee 辅助)."""

    _CAM = {
        "head": (HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX),
        "ee": (EE_CAM, EE_CAM_POS_ROBOT, EE_CAM_ROT_MATRIX),
    }

    def __init__(self, camera: str = "head"):
        if camera not in self._CAM:
            raise ValueError(f"camera must be head|ee, got {camera!r}")
        self.camera_name = camera
        self._arm_joints: Optional[np.ndarray] = None
        self._projected_gravity: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None

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

    def _cam_pose(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        cam_cfg, pos_def, rot = self._CAM[self.camera_name]
        if self.camera_name == "ee" and self._arm_joints is not None:
            pos = compute_dynamic_ee_cam_pos(self._arm_joints)
        elif self.camera_name == "head":
            pos = compute_dynamic_head_cam_pos(self._projected_gravity)
        else:
            pos = pos_def.copy()
        return cam_cfg, pos.astype(np.float32), rot.astype(np.float32)

    def _cloud_params(self) -> dict:
        if self.camera_name == "head":
            return {
                "depth_min": HEAD_DEPTH_MIN,
                "depth_max": HEAD_DEPTH_MAX,
                "step": HEAD_PIXEL_STEP,
                "v0": HEAD_ROI_V_MIN,
                "v1": HEAD_ROI_V_MAX,
                "u0": HEAD_ROI_U_MIN,
                "u1": HEAD_ROI_U_MAX,
                "z_min": HEAD_ROBOT_Z_MIN,
                "z_max": HEAD_ROBOT_Z_MAX,
                "ransac_min_inl": HEAD_RANSAC_MIN_INLIERS,
                "cluster_eps": HEAD_CLUSTER_EPS,
                "cluster_min_pts": HEAD_CLUSTER_MIN_PTS,
            }
        return {
            "depth_min": EE_DEPTH_MIN,
            "depth_max": EE_DEPTH_MAX,
            "step": EE_PIXEL_STEP,
            "v0": EE_ROI_V_MIN,
            "v1": EE_ROI_V_MAX,
            "u0": 0.04,
            "u1": 0.96,
            "z_min": -0.65,
            "z_max": 0.55,
            "ransac_min_inl": EE_RANSAC_MIN_INLIERS,
            "cluster_eps": EE_CLUSTER_EPS,
            "cluster_min_pts": EE_CLUSTER_MIN_PTS,
        }

    def _depth_to_cloud(
        self,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """points_robot (N,3), depths (N,), us (N,), vs (N,)."""
        depth = sanitize_depth(depth)
        h, w = depth.shape
        cam_cfg, cam_pos, cam_rot = self._cam_pose()
        p = self._cloud_params()
        u0, u1 = int(w * p["u0"]), int(w * p["u1"])
        v0, v1 = int(h * p["v0"]), int(h * p["v1"])
        step = int(p["step"])
        pts: List[np.ndarray] = []
        ds: List[float] = []
        us: List[float] = []
        vs: List[float] = []
        for v in range(v0, v1, step):
            for u in range(u0, u1, step):
                z = float(depth[v, u])
                if z <= p["depth_min"] or z >= p["depth_max"] or not np.isfinite(z):
                    continue
                pr = pixel_to_robot(float(u), float(v), z, cam_cfg, cam_pos, cam_rot)
                if not np.all(np.isfinite(pr)):
                    continue
                if pr[2] < p["z_min"] or pr[2] > p["z_max"]:
                    continue
                pts.append(pr)
                ds.append(z)
                us.append(float(u))
                vs.append(float(v))
        if not pts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.stack(pts, axis=0).astype(np.float32),
            np.asarray(ds, dtype=np.float32),
            np.asarray(us, dtype=np.float32),
            np.asarray(vs, dtype=np.float32),
        )

    def _cluster_passes(self, cpts: np.ndarray, us: np.ndarray, vs: np.ndarray, img_h: int, img_w: int) -> bool:
        if len(cpts) < MIN_SAMPLE_PTS:
            return False
        pos = np.median(cpts, axis=0)
        dist_xy = float(np.linalg.norm(pos[:2]))
        extent = cpts.max(axis=0) - cpts.min(axis=0)
        z_ext = float(extent[2])
        if self.camera_name == "head":
            if dist_xy < HEAD_MIN_XY_DIST or dist_xy > HEAD_MAX_XY_DIST:
                return False
            if float(pos[0]) < 0.12:
                return False
            if z_ext < 0.008 or z_ext > 0.62:
                return False
            cy = float(np.median(vs))
            if cy > img_h * 0.70:
                return False
            area_px = (float(us.max()) - float(us.min()) + 1) * (float(vs.max()) - float(vs.min()) + 1)
            if area_px > img_h * img_w * 0.20:
                return False
        else:
            if dist_xy < 0.08 or dist_xy > 2.6:
                return False
        return True

    def _grasp_quat(self, class_name: str, pos_world: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        fixed = GRASP_FIXED_QUAT.get(str(class_name), DEFAULT_GRASP_FIXED_QUAT).copy()
        dx = float(pos_world[0] - robot_pos[0])
        dy = float(pos_world[1] - robot_pos[1])
        yaw = float(np.arctan2(dy, dx))
        return _quat_multiply_wxyz(_quat_from_yaw_wxyz(yaw), fixed)

    def _cluster_to_det(
        self,
        rgb: np.ndarray,
        cpts: np.ndarray,
        ds: np.ndarray,
        us: np.ndarray,
        vs: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> Optional[dict]:
        pos_robot = np.median(cpts, axis=0).astype(np.float32)
        extent = (cpts.max(axis=0) - cpts.min(axis=0)).astype(np.float32)
        pos_world = _robot_to_world(pos_robot, robot_pos, robot_yaw)
        dist_xy = float(np.linalg.norm(pos_robot[:2]))
        depth_m = dist_xy

        ys_idx = vs.astype(np.int32)
        xs_idx = us.astype(np.int32)
        h, w = rgb.shape[:2] if rgb is not None else (480, 640)
        ys_idx = np.clip(ys_idx, 0, h - 1)
        xs_idx = np.clip(xs_idx, 0, w - 1)
        x1, x2 = int(xs_idx.min()), int(xs_idx.max())
        y1, y2 = int(ys_idx.min()), int(ys_idx.max())
        bbox = [x1, y1, x2, y2]
        cx, cy = float(np.median(us)), float(np.median(vs))
        anchor_depth = float(np.median(ds))

        cls, cls_conf = _classify_cluster_rgb(rgb, ys_idx, xs_idx, bbox, extent)
        is_head = self.camera_name == "head"
        is_ee = self.camera_name == "ee"

        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        z_top = float(pos_world[2]) + max(0.04, float(extent[2]) * 0.45)
        grasp_world = pos_world.copy()
        grasp_world[2] = z_top - GRASP_DEPTH_OFFSET
        grasp_robot = rot_inv @ (grasp_world - robot_pos)

        yaw_rel = float(np.arctan2(pos_robot[1], pos_robot[0]))
        near_grasp = is_ee and depth_m < EE_GRASP_NEAR_M

        det = {
            "class": cls,
            "class_id": CLASS_NAME_TO_ID.get(cls, -1),
            "conf": float(min(0.92, 0.44 + cls_conf * 0.48)),
            "class_conf": cls_conf,
            "bbox": bbox,
            "centroid": (cx, cy),
            "centroid_uv": [cx, cy],
            "depth_m": depth_m,
            "nav_depth_m": depth_m,
            "dist_to_robot": dist_xy,
            "yaw_rel": yaw_rel,
            "nav_yaw_rel": yaw_rel,
            "pos_robot": pos_robot.tolist(),
            "pos_world": pos_world.tolist(),
            "pos_from_pointcloud": len(cpts) >= MIN_SAMPLE_PTS + 2,
            "nav_point_count": int(len(cpts)),
            "nav_anchor_uv": [cx, float(y2)],
            "nav_anchor_depth": anchor_depth,
            "source": "depth_ransac_cluster",
            "camera": self.camera_name,
            "role": "nav" if is_head else "nav_grasp",
            "world_reliable": depth_m < 2.4,
            "grasp_reliable": near_grasp and is_ee,
            "cluster_pixels": int(len(cpts)),
            "geom_z_extent": float(extent[2]),
            "geom_extent": extent.tolist(),
        }
        if is_ee:
            det["grasp_pos_world"] = grasp_world.tolist()
            det["grasp_pos_robot"] = grasp_robot.tolist()
            det["grasp_quat_world"] = self._grasp_quat(cls, grasp_world, robot_pos).tolist()
            det["grasp_anchor_uv"] = [cx, float(y2)]
            det["grasp_anchor_depth"] = anchor_depth
        return det

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> List[dict]:
        robot_pos = np.asarray(robot_pos, dtype=np.float32)
        img_h, img_w = int(depth.shape[0]), int(depth.shape[1])
        p = self._cloud_params()

        points, depths, us, vs = self._depth_to_cloud(depth)
        if len(points) < p["ransac_min_inl"] + p["cluster_min_pts"]:
            self._last_mask = None
            return []

        _, _, table_inl = _ransac_plane(
            points,
            n_iter=HEAD_RANSAC_ITERS if self.camera_name == "head" else 180,
            thresh=HEAD_RANSAC_THRESH if self.camera_name == "head" else 0.014,
        )
        if int(table_inl.sum()) < p["ransac_min_inl"]:
            self._last_mask = None
            return []

        obj_mask = ~table_inl
        obj_pts = points[obj_mask]
        obj_ds = depths[obj_mask]
        obj_us = us[obj_mask]
        obj_vs = vs[obj_mask]
        if len(obj_pts) < p["cluster_min_pts"]:
            self._last_mask = None
            return []

        # debug mask: 投影 object 点到图像
        dbg = np.zeros((img_h, img_w), dtype=np.uint8)
        for u, v in zip(obj_us.astype(int), obj_vs.astype(int)):
            if 0 <= v < img_h and 0 <= u < img_w:
                dbg[v, u] = 255
        self._last_mask = dbg

        clusters = _euclidean_clusters(
            obj_pts,
            eps=p["cluster_eps"],
            min_pts=p["cluster_min_pts"],
        )
        if not clusters:
            return []

        dets: List[dict] = []
        for members in clusters[:MAX_CLUSTERS]:
            idx = members
            cpts = obj_pts[idx]
            if not self._cluster_passes(cpts, obj_us[idx], obj_vs[idx], img_h, img_w):
                continue
            det = self._cluster_to_det(
                rgb, cpts, obj_ds[idx], obj_us[idx], obj_vs[idx], robot_pos, robot_yaw,
            )
            if det is not None:
                dets.append(det)

        dets.sort(key=lambda d: d.get("depth_m") or 999.0)
        return dets[:MAX_CLUSTERS]

    def get_debug_mask(self) -> Optional[np.ndarray]:
        if self._last_mask is None:
            return None
        return self._last_mask.astype(np.uint8)
