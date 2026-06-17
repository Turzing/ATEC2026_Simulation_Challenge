"""
Head / EE depth 检测 — RANSAC 剔桌面 + 欧氏聚类 + 可选 RGB 分类。

检测核心在 depth_ransac_cluster.py；本模块做 camera 封装 + RGB 几何分类 refine。
"""

from __future__ import annotations

import os
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
from depth_ransac_cluster import PERCEPTION_RANSAC_BUILD, RansacClusterDetector
from rgbd_utils import sanitize_depth

PERCEPTION_BUILD_ID = PERCEPTION_RANSAC_BUILD

# 默认纯几何分类 (检测只用 depth; RGB 不参与)
DEPTH_ONLY_CLASS = os.getenv("ATEC_DEPTH_ONLY", "1").strip().lower() not in ("0", "false", "no")


def _classify_cluster_rgb(
    rgb: np.ndarray,
    bbox: List[int],
    z_extent: float,
) -> Tuple[str, float]:
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    aspect = bw / bh
    patch = rgb[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
    if patch.size == 0:
        return "sugar_box", 0.35

    bgr = patch.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = float(np.median(hsv[:, 0]))
    sat = float(np.median(hsv[:, 1]))
    val = float(np.median(hsv[:, 2]))
    yellow = 14 <= hue <= 46 and sat >= 24

    if sat < 32 and val > 65 and aspect < 1.55:
        return "sugar_box", min(0.88, 0.70 + 0.06 * (32 - sat) / 32)
    if not yellow and aspect < 1.35:
        return "sugar_box", 0.72

    scores = {"sugar_box": 0.12, "mustard_bottle": 0.12, "banana": 0.12}
    if yellow:
        scores["mustard_bottle"] += 0.28
        scores["banana"] += 0.26
    if bh >= bw * 1.12 and aspect > 1.18:
        scores["mustard_bottle"] += 0.42
        scores["banana"] -= 0.12
    elif bw >= bh * 1.15 and aspect > 1.28:
        scores["banana"] += 0.40
    elif aspect < 1.18:
        scores["sugar_box"] += 0.38
    if yellow and bh >= bw * 1.08 and sat >= 30:
        return "mustard_bottle", min(0.90, 0.78 + 0.04 * (aspect - 1.0))
    if z_extent < 0.10 and aspect < 1.32:
        scores["sugar_box"] += 0.32
    if z_extent > 0.11 and aspect < 1.22 and yellow:
        scores["mustard_bottle"] += 0.28
    name = max(scores, key=scores.get)
    return name, float(min(0.90, scores[name]))


class DepthClusterDetector:
    """单相机 depth 检测 (RANSAC + cluster)."""

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
        self._ransac = RansacClusterDetector(camera)
        self._last_mask: Optional[np.ndarray] = None

    def set_arm_joints(self, arm_joints) -> None:
        if arm_joints is None:
            self._arm_joints = None
        else:
            self._arm_joints = np.asarray(arm_joints, dtype=np.float32).reshape(-1)[:6]
        self._ransac.set_arm_joints(arm_joints)

    def set_projected_gravity(self, grav) -> None:
        if grav is None:
            self._projected_gravity = None
        else:
            self._projected_gravity = np.asarray(grav, dtype=np.float32).reshape(3)
        self._ransac.set_projected_gravity(grav)

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> List[dict]:
        depth = sanitize_depth(depth)
        dets = self._ransac.detect(
            depth,
            np.asarray(robot_pos, dtype=np.float32),
            float(robot_yaw),
        )
        out: List[dict] = []
        for det in dets:
            bbox = det.get("bbox") or [0, 0, 0, 0]
            z_ext = float(det.get("geom_z_extent") or 0.08)
            geom_cls = det.get("class")
            rgb_cls, rgb_conf = _classify_cluster_rgb(rgb, bbox, z_ext)
            use_rgb = self.camera_name == "head" or not DEPTH_ONLY_CLASS
            if use_rgb and rgb_conf >= 0.55:
                det["class"] = rgb_cls
                det["class_id"] = CLASS_NAME_TO_ID.get(rgb_cls, -1)
                det["class_conf"] = rgb_conf
            elif not DEPTH_ONLY_CLASS and rgb_conf >= 0.68:
                det["class"] = rgb_cls
                det["class_id"] = CLASS_NAME_TO_ID.get(rgb_cls, -1)
                det["class_conf"] = rgb_conf
            else:
                det["class"] = geom_cls or rgb_cls
                det["class_id"] = CLASS_NAME_TO_ID.get(det["class"], -1)
                det["class_conf"] = max(float(det.get("class_conf") or 0), rgb_conf * 0.85)
            det["conf"] = float(min(0.93, 0.45 + float(det["class_conf"]) * 0.48))
            out.append(det)
        return out

    def get_debug_mask(self) -> Optional[np.ndarray]:
        return self._last_mask

    @property
    def last_stats(self) -> dict:
        return self._ransac.last_stats
