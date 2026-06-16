"""
纯 Depth 聚类检测 — 不加 HSV/fusion/fallback。

流程: 有效深度 → 去地面(相对深度凸起) → 连通域 → 反投影 3D → 簇内 RGB 仅做分类
"""

from __future__ import annotations

PERCEPTION_BUILD_ID = "20260617-fix-search-osc"

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

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

DEPTH_MIN = 0.18
DEPTH_MAX = 8.0
MIN_CLUSTER_PX_HEAD = 10
MIN_CLUSTER_PX_EE = 8
MAX_CLUSTERS = 20
ROI_V_MIN = 0.06
ROI_V_MAX_HEAD = 0.68   # head 排除画面下部地砖 (假检重灾区)
ROI_V_MAX_EE = 0.96
PROTRUDE_HEAD_M = 0.010
PROTRUDE_EE_M = 0.010
LOCAL_GROUND_K = 15
EE_GRASP_NEAR_M = 1.15
MIN_SAMPLE_PTS = 3


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


def _classify_cluster(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    bbox: List[int],
    z_extent: float,
) -> Tuple[str, float]:
    """簇已定位后，用 RGB 形状 + 3D 高度做粗分类（不参与检测 mask）。"""
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    aspect = bw / bh
    fill = len(ys) / max(bw * bh, 1)

    patch = rgb[ys, xs]
    if patch.size == 0:
        return "sugar_box", 0.35

    bgr = patch[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = float(np.median(hsv[:, 0]))
    sat = float(np.median(hsv[:, 1]))
    val = float(np.median(hsv[:, 2]))
    yellow = 14 <= hue <= 46 and sat >= 24

    if sat < 32 and val > 65 and aspect < 1.55:
        return "sugar_box", min(0.88, 0.70 + 0.06 * (32 - sat) / 32)
    if not yellow and aspect < 1.35 and fill > 0.25:
        return "sugar_box", 0.72

    scores = {"sugar_box": 0.12, "mustard_bottle": 0.12, "banana": 0.12}
    if yellow:
        scores["mustard_bottle"] += 0.28
        scores["banana"] += 0.26
    if bw >= bh * 1.15 and aspect > 1.28:
        scores["banana"] += 0.40
    elif bh >= bw * 1.12 and aspect > 1.22:
        scores["mustard_bottle"] += 0.36
    elif aspect < 1.18:
        scores["sugar_box"] += 0.38
    if z_extent < 0.10 and aspect < 1.32:
        scores["sugar_box"] += 0.32
    if z_extent > 0.11 and aspect < 1.22 and yellow:
        scores["mustard_bottle"] += 0.28
    name = max(scores, key=scores.get)
    return name, float(min(0.90, scores[name]))


class DepthClusterDetector:
    """单相机纯 depth 聚类 → 3D 位置。"""

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

    def _build_mask(self, depth: np.ndarray, *, protrude_scale: float = 1.0) -> np.ndarray:
        h, w = depth.shape
        valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
        vs = np.arange(h, dtype=np.float32)
        vv = np.meshgrid(np.arange(w), vs)[1]
        roi_v_max = ROI_V_MAX_EE if self.camera_name == "ee" else ROI_V_MAX_HEAD
        in_roi = (vv >= int(h * ROI_V_MIN)) & (vv <= int(h * roi_v_max))

        base = PROTRUDE_EE_M if self.camera_name == "ee" else PROTRUDE_HEAD_M
        protrude_m = max(0.004, float(base) * float(protrude_scale))
        fill = np.where(valid, depth, np.median(depth[valid]) if np.any(valid) else 3.0).astype(np.float32)
        k = LOCAL_GROUND_K | 1
        local_ground = cv2.GaussianBlur(fill, (k, k), 0)
        relief = local_ground - depth

        v0 = int(h * ROI_V_MIN)
        roi = depth[v0:int(h * 0.82), :]
        roi_ok = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
        if roi_ok.size > 80:
            ground_d = float(np.percentile(roi_ok, 38))
        else:
            ground_d = float(np.median(depth[valid])) if np.any(valid) else 3.0

        global_protrude = depth < (ground_d - protrude_m)
        local_protrude = relief >= protrude_m
        mask = (valid & in_roi & (global_protrude | local_protrude)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self._last_mask = mask
        return mask

    def _backproject_cluster(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
        depth_m: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """反投影簇 → pos_robot / pos_world；不做 world-Z 硬杀（solution_rl 会用 FK 再校正）。"""
        cam_cfg, cam_pos, cam_rot = self._cam_pose()
        pts_w = []
        step = 1 if len(ys) < 400 else 2
        for y, x in zip(ys[::step], xs[::step]):
            z = float(depth[y, x])
            if z <= DEPTH_MIN or z >= DEPTH_MAX:
                continue
            pr = pixel_to_robot(float(x), float(y), z, cam_cfg, cam_pos, cam_rot)
            if not np.isfinite(pr).all():
                continue
            pts_w.append(_robot_to_world(pr, robot_pos, robot_yaw))

        if len(pts_w) >= MIN_SAMPLE_PTS:
            world_pts = np.stack(pts_w, axis=0).astype(np.float32)
            pos_world = np.median(world_pts, axis=0)
            z_extent = float(world_pts[:, 2].max() - world_pts[:, 2].min())
            c, s = np.cos(robot_yaw), np.sin(robot_yaw)
            rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            pos_robot = rot_inv @ (pos_world - robot_pos)
            return pos_robot.astype(np.float32), pos_world.astype(np.float32), z_extent

        cx, cy = float(np.median(xs)), float(np.median(ys))
        pr = pixel_to_robot(cx, cy, depth_m, cam_cfg, cam_pos, cam_rot)
        if not np.isfinite(pr).all():
            return None, None, 0.08
        pos_world = _robot_to_world(pr, robot_pos, robot_yaw)
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pos_robot = rot_inv @ (pos_world - robot_pos)
        return pos_robot.astype(np.float32), pos_world.astype(np.float32), 0.08

    def _grasp_quat(self, class_name: str, pos_world: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        fixed = GRASP_FIXED_QUAT.get(str(class_name), DEFAULT_GRASP_FIXED_QUAT).copy()
        dx = float(pos_world[0] - robot_pos[0])
        dy = float(pos_world[1] - robot_pos[1])
        yaw = float(np.arctan2(dy, dx))
        return _quat_multiply_wxyz(_quat_from_yaw_wxyz(yaw), fixed)

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> List[dict]:
        depth = sanitize_depth(depth)
        img_h, img_w = int(depth.shape[0]), int(depth.shape[1])
        mask = self._build_mask(depth)
        labeled, n = ndimage.label(mask)
        if n == 0:
            mask = self._build_mask(depth, protrude_scale=0.45)
            labeled, n = ndimage.label(mask)
        if n == 0:
            mask = self._build_mask(depth, protrude_scale=0.28)
            labeled, n = ndimage.label(mask)
        if n == 0:
            return []

        is_head = self.camera_name == "head"
        is_ee = self.camera_name == "ee"
        min_px = MIN_CLUSTER_PX_EE if is_ee else MIN_CLUSTER_PX_HEAD
        dets: List[dict] = []

        for cid in range(1, n + 1):
            ys, xs = np.where(labeled == cid)
            if len(ys) < min_px:
                continue

            dvals = depth[ys, xs]
            dvals = dvals[(dvals > DEPTH_MIN) & (dvals < DEPTH_MAX)]
            if dvals.size < 6:
                continue

            depth_m = float(np.percentile(dvals, 12))
            pos_robot, pos_world, z_extent = self._backproject_cluster(
                ys, xs, depth, robot_pos, robot_yaw, depth_m,
            )
            if pos_robot is None or pos_world is None:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            if is_head and float(np.median(ys)) > img_h * 0.66:
                continue
            bbox = [x1, y1, x2, y2]
            bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
            if is_head and bw * bh > img_h * img_w * 0.18:
                continue
            cx, cy = float(np.median(xs)), float(np.median(ys))
            cls, cls_conf = _classify_cluster(rgb, ys, xs, bbox, z_extent)

            c, s = np.cos(robot_yaw), np.sin(robot_yaw)
            rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            z_top = float(pos_world[2]) + max(0.04, z_extent * 0.45)
            grasp_world = pos_world.copy()
            grasp_world[2] = z_top - GRASP_DEPTH_OFFSET
            grasp_robot = rot_inv @ (grasp_world - robot_pos)

            yaw_rel = float(np.arctan2(pos_robot[1], pos_robot[0]))
            dist = float(np.linalg.norm(pos_robot[:2]))
            near_grasp = is_ee and depth_m < EE_GRASP_NEAR_M

            det = {
                "class": cls,
                "class_id": CLASS_NAME_TO_ID.get(cls, -1),
                "conf": float(min(0.92, 0.42 + cls_conf * 0.5)),
                "class_conf": cls_conf,
                "bbox": bbox,
                "centroid": (cx, cy),
                "centroid_uv": [cx, cy],
                "depth_m": depth_m,
                "nav_depth_m": depth_m,
                "dist_to_robot": dist,
                "yaw_rel": yaw_rel,
                "nav_yaw_rel": yaw_rel,
                "pos_robot": pos_robot.tolist(),
                "pos_world": pos_world.tolist(),
                "pos_from_pointcloud": len(ys) >= MIN_SAMPLE_PTS + 4,
                "nav_point_count": int(len(ys)),
                "nav_anchor_uv": [cx, float(y2)],
                "nav_anchor_depth": depth_m,
                "source": "depth_cluster",
                "camera": self.camera_name,
                "role": "nav" if is_head else "nav_grasp",
                "world_reliable": depth_m < 2.2,
                "grasp_reliable": near_grasp and is_ee,
                "cluster_pixels": int(len(ys)),
                "geom_z_extent": z_extent,
            }
            if is_ee:
                det["grasp_pos_world"] = grasp_world.tolist()
                det["grasp_pos_robot"] = grasp_robot.tolist()
                det["grasp_quat_world"] = self._grasp_quat(cls, grasp_world, robot_pos).tolist()
                det["grasp_anchor_uv"] = [cx, float(y2)]
                det["grasp_anchor_depth"] = depth_m
            dets.append(det)

        dets.sort(key=lambda d: d.get("depth_m") or 999.0)
        return dets[:MAX_CLUSTERS]

    def get_debug_mask(self) -> Optional[np.ndarray]:
        if self._last_mask is None:
            return None
        return (self._last_mask * 255).astype(np.uint8)
