"""
纯几何 depth 感知: 点云 → RANSAC 剔桌面 → 欧氏聚类 → 3D 目标

head: 粗导航 (站直/远距)
ee:   静态抓取 (趴下/停稳)
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
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
)
from rgbd_utils import (
    compute_dynamic_ee_cam_pos,
    compute_dynamic_head_cam_pos,
    pixel_to_robot,
    sanitize_depth,
)

PERCEPTION_RANSAC_BUILD = "20260617-taskb-perc-only-v27"
TASKB_PIPELINE_MODE = "blob_gt_coast"
RANSAC_HEAD_SUPPLEMENT_PX = 72

_CAM_CFG = {
    "head": {
        "cam": HEAD_CAM,
        "pos_def": HEAD_CAM_POS_ROBOT,
        "rot": HEAD_CAM_ROT_MATRIX,
        "depth_min": 0.12,
        "depth_max": 8.5,
        "pixel_step": 2,
        "ransac_iters": 220,
        "ransac_thresh": 0.022,
        "ransac_min_inl": 28,
        "cluster_eps": 0.038,
        "cluster_min_pts": 4,
        "max_cluster_pts": 900,
        "max_cluster_extent_m": 1.25,
        # 远距搜索: 物体在画面上半/下半都可能出现; robot_z 须含地面 (~-0.55)
        "roi_u": (0.02, 0.98),
        "roi_v": (0.02, 0.98),
        "robot_x_min": 0.06,
        "robot_z": (-0.80, 0.52),
        "role": "nav",
        "camera_label": "head",
    },
    "ee": {
        "cam": EE_CAM,
        "pos_def": EE_CAM_POS_ROBOT,
        "rot": EE_CAM_ROT_MATRIX,
        "depth_min": 0.15,
        "depth_max": 2.8,
        "pixel_step": 3,
        "ransac_iters": 180,
        "ransac_thresh": 0.014,
        "ransac_min_inl": 75,
        "cluster_eps": 0.028,
        "cluster_min_pts": 35,
        "roi_u": (0.04, 0.96),
        "roi_v": (0.05, 0.94),
        "robot_x_min": 0.05,
        "robot_z": (-0.65, 0.55),
        "role": "nav_grasp",
        "camera_label": "ee",
    },
}

MAX_CLUSTERS = 12


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
    n_iter: int,
    thresh: float,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
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
    eps: float,
    min_pts: int,
) -> List[np.ndarray]:
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


def _cluster_objects_adaptive(
    points: np.ndarray,
    base_eps: float,
    base_min_pts: int,
) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    med_depth = float(np.median(np.linalg.norm(points, axis=1)))
    scale = max(1.0, med_depth / 2.0)
    eps = float(base_eps * scale)
    min_pts = int(base_min_pts)
    for attempt, (eps_mul, min_mul) in enumerate(
        ((1.0, 1.0), (1.55, 0.72), (2.35, 0.55)),
    ):
        trial_eps = eps * eps_mul
        trial_min = max(4, int(round(min_pts * min_mul)))
        clusters = _euclidean_clusters(points, trial_eps, trial_min)
        if clusters:
            return clusters
    return []


class RansacClusterDetector:
    """depth → RANSAC 剔桌 + 欧氏聚类 (head / ee)."""

    def __init__(self, camera: str = "ee"):
        if camera not in _CAM_CFG:
            raise ValueError(f"camera must be head|ee, got {camera!r}")
        self.camera_name = camera
        self._cfg = _CAM_CFG[camera]
        self._arm_joints: Optional[np.ndarray] = None
        self._projected_gravity: Optional[np.ndarray] = None
        self._track_id = 0
        self._last_n_cloud = 0
        self._last_n_clusters = 0
        self._last_n_obj_pts = 0
        self._last_n_table_inl = 0

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
        """head 用 gravity 动态外参 (与 rgbd_pure_pipeline 一致); EE 用臂关节."""
        c = self._cfg
        if self.camera_name == "ee" and self._arm_joints is not None:
            pos = compute_dynamic_ee_cam_pos(self._arm_joints)
        elif self.camera_name == "head" and self._projected_gravity is not None:
            pos = compute_dynamic_head_cam_pos(self._projected_gravity)
        else:
            pos = c["pos_def"].copy()
        return c["cam"], pos.astype(np.float32), c["rot"].astype(np.float32)

    def _depth_to_cloud(
        self,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        depth = sanitize_depth(depth)
        h, w = depth.shape
        cam_cfg, cam_pos, cam_rot = self._cam_pose()
        c = self._cfg
        u0, u1 = int(w * c["roi_u"][0]), int(w * c["roi_u"][1])
        v0, v1 = int(h * c["roi_v"][0]), int(h * c["roi_v"][1])
        step = int(c["pixel_step"])
        dmin, dmax = float(c["depth_min"]), float(c["depth_max"])
        z_lo, z_hi = c["robot_z"]
        x_min = float(c["robot_x_min"])
        pts: List[np.ndarray] = []
        us: List[float] = []
        vs: List[float] = []
        z_cam: List[float] = []
        for v in range(v0, v1, step):
            for u in range(u0, u1, step):
                z = float(depth[v, u])
                if z <= dmin or z >= dmax or not np.isfinite(z):
                    continue
                pr = pixel_to_robot(float(u), float(v), z, cam_cfg, cam_pos, cam_rot)
                if not np.all(np.isfinite(pr)):
                    continue
                if float(pr[0]) < x_min or float(pr[2]) < z_lo or float(pr[2]) > z_hi:
                    continue
                pts.append(pr)
                us.append(float(u))
                vs.append(float(v))
                z_cam.append(z)
        self._last_n_cloud = len(pts)
        if not pts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
            )
        return (
            np.stack(pts, axis=0).astype(np.float32),
            np.asarray(us, dtype=np.float32),
            np.asarray(vs, dtype=np.float32),
            np.asarray(z_cam, dtype=np.float32),
        )

    def _grasp_quat(self, class_name: str, pos_world: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        fixed = GRASP_FIXED_QUAT.get(str(class_name), DEFAULT_GRASP_FIXED_QUAT).copy()
        dx = float(pos_world[0] - robot_pos[0])
        dy = float(pos_world[1] - robot_pos[1])
        yaw = float(np.arctan2(dy, dx))
        return _quat_multiply_wxyz(_quat_from_yaw_wxyz(yaw), fixed)

    def _cluster_bottom_indices(
        self,
        vs: np.ndarray,
        idx: np.ndarray,
        *,
        bottom_frac: float = 0.35,
    ) -> np.ndarray:
        """图像 v 越大越靠下; 取簇内靠下像素用于地面接触定位."""
        v_idx = vs[idx]
        if len(v_idx) < 6:
            return idx
        v_cut = float(np.percentile(v_idx, 100.0 * (1.0 - bottom_frac)))
        bot = v_idx >= v_cut
        if int(np.sum(bot)) >= max(4, len(idx) // 6):
            return idx[bot]
        return idx

    def _cluster_top_indices(
        self,
        vs: np.ndarray,
        idx: np.ndarray,
        *,
        top_frac: float = 0.30,
    ) -> np.ndarray:
        v_idx = vs[idx]
        if len(v_idx) < 6:
            return idx
        v_cut = float(np.percentile(v_idx, 100.0 * top_frac))
        top = v_idx <= v_cut
        if int(np.sum(top)) >= max(4, len(idx) // 8):
            return idx[top]
        return idx

    def _cluster_nav_pos(
        self,
        points: np.ndarray,
        us: np.ndarray,
        vs: np.ndarray,
        z_cam: np.ndarray,
        idx: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """底边加权 3D + nav anchor (贴地接触点)."""
        bot_idx = self._cluster_bottom_indices(vs, idx)
        nav_pts = points[bot_idx]
        if len(nav_pts) < 4:
            nav_pts = points[idx]
        pos_robot = np.median(nav_pts, axis=0).astype(np.float32)
        pos_robot[2] = float(np.percentile(points[idx, 2], 18))
        nav_u = float(np.median(us[bot_idx]))
        nav_v = float(np.max(vs[bot_idx]))
        anchor_depth = float(np.median(z_cam[bot_idx]))
        return pos_robot, nav_u, nav_v, anchor_depth

    def _cluster_to_det(
        self,
        points: np.ndarray,
        idx: np.ndarray,
        us: np.ndarray,
        vs: np.ndarray,
        z_cam: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        det_id: int,
    ) -> dict:
        cpts = points[idx]
        pos_robot, nav_u, nav_v, anchor_depth = self._cluster_nav_pos(
            points, us, vs, z_cam, idx,
        )
        extent = (cpts.max(axis=0) - cpts.min(axis=0)).astype(np.float32)
        pos_world = _robot_to_world(pos_robot, robot_pos, robot_yaw)
        cls, cls_conf = _classify_geometry(extent)
        cu = float(np.median(us[idx]))
        cv = float(np.median(vs[idx]))
        dist_xy = float(np.linalg.norm(pos_robot[:2]))
        depth_m = float(anchor_depth if anchor_depth > 0.05 else np.median(z_cam[idx]))
        yaw_rel = float(np.arctan2(pos_robot[1], pos_robot[0]))
        x1, x2 = int(us[idx].min()), int(us[idx].max())
        y1, y2 = int(vs[idx].min()), int(vs[idx].max())
        is_ee = self.camera_name == "ee"
        near_grasp = is_ee and depth_m < 1.15

        det = {
            "id": int(det_id),
            "class": cls,
            "class_id": CLASS_NAME_TO_ID.get(cls, -1),
            "conf": float(min(0.93, 0.50 + cls_conf * 0.4)),
            "class_conf": cls_conf,
            "bbox": [x1, y1, x2, y2],
            "centroid": (cu, cv),
            "centroid_uv": [cu, cv],
            "depth_m": depth_m,
            "nav_depth_m": depth_m,
            "dist_to_robot": dist_xy,
            "yaw_rel": yaw_rel,
            "nav_yaw_rel": yaw_rel,
            "pos_robot": pos_robot.tolist(),
            "pos_world": pos_world.tolist(),
            "pos_from_pointcloud": True,
            "nav_point_count": int(len(idx)),
            "nav_anchor_uv": [nav_u, nav_v],
            "nav_anchor_depth": anchor_depth,
            "source": "ransac_cluster",
            "camera": self._cfg["camera_label"],
            "role": self._cfg["role"],
            "world_reliable": depth_m < (2.2 if self.camera_name == "head" else 2.0),
            "grasp_reliable": near_grasp,
            "cluster_pixels": int(len(idx)),
            "geom_extent": extent.tolist(),
            "geom_z_extent": float(extent[2]),
        }
        if is_ee:
            top_idx = self._cluster_top_indices(vs, idx)
            top_pts = points[top_idx]
            z_top_r = float(np.percentile(top_pts[:, 2], 82))
            z_top_w = float(pos_world[2]) + max(0.035, float(extent[2]) * 0.38)
            z_top_w = max(z_top_w, float(_robot_to_world(
                np.array([pos_robot[0], pos_robot[1], z_top_r], dtype=np.float32),
                robot_pos, robot_yaw,
            )[2]))
            grasp_world = pos_world.copy()
            grasp_world[2] = z_top_w - GRASP_DEPTH_OFFSET
            c, s = np.cos(robot_yaw), np.sin(robot_yaw)
            rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            grasp_robot = rot_inv @ (grasp_world - robot_pos)
            det["static_snapshot"] = True
            det["grasp_pos_world"] = grasp_world.tolist()
            det["grasp_pos_robot"] = grasp_robot.tolist()
            det["grasp_quat_world"] = self._grasp_quat(cls, grasp_world, robot_pos).tolist()
            det["grasp_anchor_uv"] = [nav_u, nav_v]
            det["grasp_anchor_depth"] = anchor_depth
        return det

    def _cluster_passes_filters(self, points: np.ndarray, idx: np.ndarray) -> bool:
        c = self._cfg
        n = int(len(idx))
        if n < int(c["cluster_min_pts"]) or n > int(c.get("max_cluster_pts", 9999)):
            return False
        cpts = points[idx]
        extent = float(np.max(np.linalg.norm(cpts - cpts.mean(axis=0, keepdims=True), axis=1)))
        if extent > float(c.get("max_cluster_extent_m", 0.75)):
            return False
        pos_robot = np.median(cpts, axis=0)
        z_lo, z_hi = c["robot_z"]
        if float(pos_robot[0]) < float(c["robot_x_min"]):
            return False
        if float(pos_robot[2]) < z_lo or float(pos_robot[2]) > z_hi:
            return False
        return True

    def detect(
        self,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        *,
        nav_hint: Optional[dict] = None,
    ) -> List[dict]:
        c = self._cfg
        points, us, vs, z_cam = self._depth_to_cloud(depth)
        if len(points) < int(c["ransac_min_inl"]) + 4:
            self._last_n_clusters = 0
            self._last_n_obj_pts = 0
            self._last_n_table_inl = 0
            return []

        _, _, table_inl = _ransac_plane(
            points, int(c["ransac_iters"]), float(c["ransac_thresh"]),
        )
        self._last_n_table_inl = int(table_inl.sum())
        if self._last_n_table_inl < int(c["ransac_min_inl"]):
            self._last_n_clusters = 0
            self._last_n_obj_pts = 0
            return []
        obj_mask = ~table_inl
        obj_pts = points[obj_mask]
        obj_us = us[obj_mask]
        obj_vs = vs[obj_mask]
        obj_z = z_cam[obj_mask]
        # 远距小凸起易被宽阈值 RANSAC 吞进桌面 → 用更紧阈值再剔一次
        if len(obj_pts) < 4 and len(points) >= int(c["ransac_min_inl"]) + 16:
            _, _, tight_inl = _ransac_plane(
                points,
                int(c["ransac_iters"]),
                float(c["ransac_thresh"]) * 0.55,
            )
            if int(tight_inl.sum()) >= int(c["ransac_min_inl"]):
                obj_mask = ~tight_inl
                obj_pts = points[obj_mask]
                obj_us = us[obj_mask]
                obj_vs = vs[obj_mask]
                obj_z = z_cam[obj_mask]
        self._last_n_obj_pts = len(obj_pts)
        if len(obj_pts) < 4:
            self._last_n_clusters = 0
            return []

        cluster_indices = _cluster_objects_adaptive(
            obj_pts, float(c["cluster_eps"]), int(c["cluster_min_pts"]),
        )
        if (
            len(cluster_indices) == 1
            and len(cluster_indices[0]) > int(c.get("max_cluster_pts", 320))
        ):
            members = cluster_indices[0]
            sub = _cluster_objects_adaptive(
                obj_pts[members],
                float(c["cluster_eps"]) * 0.38,
                max(5, int(c["cluster_min_pts"])),
            )
            if len(sub) >= 2:
                cluster_indices = [members[s] for s in sub]

        self._last_n_clusters = len(cluster_indices)
        if not cluster_indices:
            return []

        dets: List[dict] = []
        for members in cluster_indices[:MAX_CLUSTERS]:
            if not self._cluster_passes_filters(obj_pts, members):
                continue
            self._track_id += 1
            dets.append(
                self._cluster_to_det(
                    obj_pts, members, obj_us, obj_vs, obj_z,
                    robot_pos, robot_yaw, self._track_id,
                )
            )
        self._last_n_clusters = len(dets)

        if nav_hint is not None and nav_hint.get("pos_world") is not None:
            hint_w = np.asarray(nav_hint["pos_world"], dtype=np.float32)
            dets.sort(
                key=lambda d: float(
                    np.linalg.norm(np.asarray(d["pos_world"], dtype=np.float32)[:2] - hint_w[:2])
                )
            )
        else:
            dets.sort(key=lambda d: d.get("dist_to_robot") or 999.0)
        return dets

    @property
    def last_stats(self) -> dict:
        return {
            "cloud_pts": self._last_n_cloud,
            "clusters": self._last_n_clusters,
            "obj_pts": self._last_n_obj_pts,
            "table_inl": self._last_n_table_inl,
        }
