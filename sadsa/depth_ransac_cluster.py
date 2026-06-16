"""
纯几何 EE 静态感知: depth → 点云 → RANSAC 剔桌面 → 欧氏聚类 → 3D 目标

完全忽略 RGB，光照无关。用于机械狗停稳/趴下后的精定位。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from config import (
    CLASS_NAME_TO_ID,
    DEFAULT_GRASP_FIXED_QUAT,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    GRASP_DEPTH_OFFSET,
    GRASP_FIXED_QUAT,
)
from rgbd_utils import compute_dynamic_ee_cam_pos, pixel_to_robot, sanitize_depth

DEPTH_MIN = 0.15
DEPTH_MAX = 2.8
PIXEL_STEP = 3
RANSAC_ITERS = 180
RANSAC_THRESH_M = 0.014
RANSAC_MIN_INLIERS = 120
CLUSTER_EPS_M = 0.028
CLUSTER_MIN_PTS = 35
MAX_CLUSTERS = 12
ROI_V_MIN = 0.05
ROI_V_MAX = 0.94
ROI_U_MIN = 0.04
ROI_U_MAX = 0.96


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


def _classify_geometry(extent: np.ndarray) -> Tuple[str, float]:
    """纯 3D 形状粗分类 (无 RGB)."""
    ex = np.sort(np.maximum(extent, 1e-3))
    short, mid, tall = float(ex[0]), float(ex[1]), float(ex[2])
    aspect = tall / max(short, 1e-3)
    flat = mid / max(tall, 1e-3)
    if tall >= 0.07 and aspect >= 1.35 and flat < 0.82:
        return "mustard_bottle", min(0.86, 0.62 + 0.08 * (aspect - 1.2))
    if mid >= short * 1.25 and tall < 0.09:
        return "banana", min(0.84, 0.58 + 0.06 * (mid / short - 1.0))
    return "sugar_box", 0.72


def _ransac_plane(
    points: np.ndarray,
    n_iter: int = RANSAC_ITERS,
    thresh: float = RANSAC_THRESH_M,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """返回 (normal, d) 满足 n·p + d = 0, inlier mask."""
    n_pts = len(points)
    if n_pts < 3:
        return None, 0.0, np.zeros(n_pts, dtype=bool)
    best_inl = np.zeros(n_pts, dtype=bool)
    best_n: Optional[np.ndarray] = None
    best_d = 0.0
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-5:
            continue
        n = n / ln
        d = -float(np.dot(n, p0))
        dist = np.abs(points @ n + d)
        inl = dist < thresh
        if int(inl.sum()) > int(best_inl.sum()):
            best_inl = inl
            best_n = n
            best_d = d
    return best_n, best_d, best_inl


def _euclidean_clusters(
    points: np.ndarray,
    eps: float = CLUSTER_EPS_M,
    min_pts: int = CLUSTER_MIN_PTS,
) -> List[np.ndarray]:
    """简单 BFS 欧氏聚类，返回各簇 index 数组列表."""
    n = len(points)
    if n == 0:
        return []
    labels = np.full(n, -1, dtype=np.int32)
    clusters: List[np.ndarray] = []
    cid = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        queue = [i]
        labels[i] = cid
        members = [i]
        while queue:
            j = queue.pop()
            d = np.linalg.norm(points - points[j], axis=1)
            nbr = np.where((d < eps) & (labels < 0))[0]
            for k in nbr:
                labels[k] = cid
                queue.append(k)
                members.append(int(k))
        if len(members) >= min_pts:
            clusters.append(np.asarray(members, dtype=np.int32))
            cid += 1
        else:
            for m in members:
                labels[m] = -1
    return clusters


class RansacClusterDetector:
    """EE 相机静态 depth: RANSAC 剔桌 + 欧氏聚类."""

    def __init__(self):
        self._arm_joints: Optional[np.ndarray] = None
        self._track_id = 0

    def set_arm_joints(self, arm_joints) -> None:
        if arm_joints is None:
            self._arm_joints = None
        else:
            self._arm_joints = np.asarray(arm_joints, dtype=np.float32).reshape(-1)[:6]

    def _cam_pose(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        if self._arm_joints is not None:
            pos = compute_dynamic_ee_cam_pos(self._arm_joints)
        else:
            pos = EE_CAM_POS_ROBOT.copy()
        return EE_CAM, pos.astype(np.float32), EE_CAM_ROT_MATRIX.astype(np.float32)

    def _depth_to_cloud(
        self,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 points_robot (N,3), pixel_u, pixel_v."""
        depth = sanitize_depth(depth)
        h, w = depth.shape
        cam_cfg, cam_pos, cam_rot = self._cam_pose()
        u0, u1 = int(w * ROI_U_MIN), int(w * ROI_U_MAX)
        v0, v1 = int(h * ROI_V_MIN), int(h * ROI_V_MAX)
        pts: List[np.ndarray] = []
        us: List[float] = []
        vs: List[float] = []
        for v in range(v0, v1, PIXEL_STEP):
            for u in range(u0, u1, PIXEL_STEP):
                z = float(depth[v, u])
                if z <= DEPTH_MIN or z >= DEPTH_MAX or not np.isfinite(z):
                    continue
                pr = pixel_to_robot(float(u), float(v), z, cam_cfg, cam_pos, cam_rot)
                if not np.all(np.isfinite(pr)):
                    continue
                if pr[2] < -0.65 or pr[2] > 0.55:
                    continue
                pts.append(pr)
                us.append(float(u))
                vs.append(float(v))
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0), np.zeros(0)
        return (
            np.stack(pts, axis=0).astype(np.float32),
            np.asarray(us, dtype=np.float32),
            np.asarray(vs, dtype=np.float32),
        )

    def _grasp_quat(self, class_name: str, pos_world: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        fixed = GRASP_FIXED_QUAT.get(str(class_name), DEFAULT_GRASP_FIXED_QUAT).copy()
        dx = float(pos_world[0] - robot_pos[0])
        dy = float(pos_world[1] - robot_pos[1])
        yaw = float(np.arctan2(dy, dx))
        return _quat_multiply_wxyz(_quat_from_yaw_wxyz(yaw), fixed)

    def _cluster_to_det(
        self,
        points: np.ndarray,
        idx: np.ndarray,
        us: np.ndarray,
        vs: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        det_id: int,
    ) -> dict:
        cpts = points[idx]
        pos_robot = np.median(cpts, axis=0).astype(np.float32)
        extent = (cpts.max(axis=0) - cpts.min(axis=0)).astype(np.float32)
        pos_world = _robot_to_world(pos_robot, robot_pos, robot_yaw)
        cls, cls_conf = _classify_geometry(extent)
        cu = float(np.median(us[idx]))
        cv = float(np.median(vs[idx]))
        dist_xy = float(np.linalg.norm(pos_robot[:2]))
        depth_m = float(np.linalg.norm(cpts - self._cam_pose()[1], axis=1).min())
        z_top = float(pos_world[2]) + max(0.035, float(extent[2]) * 0.42)
        grasp_world = pos_world.copy()
        grasp_world[2] = z_top - GRASP_DEPTH_OFFSET
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        grasp_robot = rot_inv @ (grasp_world - robot_pos)
        x1, x2 = float(us[idx].min()), float(us[idx].max())
        y1, y2 = float(vs[idx].min()), float(vs[idx].max())
        return {
            "id": int(det_id),
            "class": cls,
            "class_id": CLASS_NAME_TO_ID.get(cls, -1),
            "conf": float(min(0.93, 0.50 + cls_conf * 0.4)),
            "class_conf": cls_conf,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "centroid": (cu, cv),
            "centroid_uv": [cu, cv],
            "depth_m": depth_m,
            "nav_depth_m": depth_m,
            "dist_to_robot": dist_xy,
            "yaw_rel": float(np.arctan2(pos_robot[1], pos_robot[0])),
            "nav_yaw_rel": float(np.arctan2(pos_robot[1], pos_robot[0])),
            "pos_robot": pos_robot.tolist(),
            "pos_world": pos_world.tolist(),
            "pos_from_pointcloud": True,
            "nav_point_count": int(len(idx)),
            "nav_anchor_uv": [cu, float(y2)],
            "nav_anchor_depth": depth_m,
            "source": "ransac_cluster",
            "camera": "ee",
            "role": "nav_grasp",
            "world_reliable": True,
            "grasp_reliable": True,
            "static_snapshot": True,
            "cluster_pixels": int(len(idx)),
            "geom_extent": extent.tolist(),
            "grasp_pos_world": grasp_world.tolist(),
            "grasp_pos_robot": grasp_robot.tolist(),
            "grasp_quat_world": self._grasp_quat(cls, grasp_world, robot_pos).tolist(),
            "grasp_anchor_uv": [cu, float(y2)],
            "grasp_anchor_depth": depth_m,
        }

    def detect(
        self,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        *,
        nav_hint: Optional[dict] = None,
    ) -> List[dict]:
        points, us, vs = self._depth_to_cloud(depth)
        if len(points) < RANSAC_MIN_INLIERS + CLUSTER_MIN_PTS:
            return []

        _, _, table_inl = _ransac_plane(points)
        if int(table_inl.sum()) < RANSAC_MIN_INLIERS:
            return []
        obj_mask = ~table_inl
        obj_pts = points[obj_mask]
        obj_us = us[obj_mask]
        obj_vs = vs[obj_mask]
        if len(obj_pts) < CLUSTER_MIN_PTS:
            return []

        cluster_indices = _euclidean_clusters(obj_pts)
        if not cluster_indices:
            return []

        dets: List[dict] = []
        for members in cluster_indices[:MAX_CLUSTERS]:
            self._track_id += 1
            det = self._cluster_to_det(
                obj_pts, members, obj_us, obj_vs,
                robot_pos, robot_yaw, self._track_id,
            )
            dets.append(det)

        if nav_hint is not None and nav_hint.get("pos_world") is not None:
            hint_w = np.asarray(nav_hint["pos_world"], dtype=np.float32)
            dets.sort(
                key=lambda d: float(
                    np.linalg.norm(
                        np.asarray(d["pos_world"], dtype=np.float32)[:2] - hint_w[:2]
                    )
                )
            )
        else:
            dets.sort(key=lambda d: d.get("dist_to_robot") or 999.0)
        return dets
