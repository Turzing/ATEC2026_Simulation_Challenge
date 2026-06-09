"""
3D → 2D 投影工具 — 用于仿真真实数据采集的自动 YOLO 标注

与 generate_synthetic_dataset.py 无关（那个是画假图用的合成数据）。
本模块只做几何变换:
    物体世界坐标 + 尺寸 + head 相机模型 → 像素 bbox (YOLO 格式)

用法 (采集脚本 / 离线自检):
    from projection_utils import compute_bbox_3d, bbox_to_yolo_line

    bbox = compute_bbox_3d(obj_pos, (lx, ly, lz), robot_pos, robot_yaw)
    if bbox:
        line = bbox_to_yolo_line(class_id, bbox)

自检:
    cd taskb_perception
    python projection_utils.py
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HEAD_CAM_MATRIX,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX_INV,
    IMG_H,
    IMG_W,
    OBJECT_SIZES,
    CLASS_NAMES,
)

# 光轴深度下限 (米): 点在相机后方时不投影
MIN_CAM_Z = 0.05


def yaw_from_quat_wxyz(quat_wxyz: Sequence[float]) -> float:
    """root 四元数 (w,x,y,z) → 绕世界 Z 的 yaw (rad)"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def world_to_robot(
    point_world: np.ndarray,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> np.ndarray:
    """世界坐标 → 机器人基座坐标 (仅平移 + 绕 Z 旋转)"""
    dx = point_world[0] - robot_pos[0]
    dy = point_world[1] - robot_pos[1]
    dz = point_world[2] - robot_pos[2]
    cy, sy = np.cos(-robot_yaw), np.sin(-robot_yaw)
    return np.array([cy * dx - sy * dy, sy * dx + cy * dy, dz], dtype=np.float32)


def robot_offset_to_camera(
    p_robot: np.ndarray,
    cam_pos_robot: np.ndarray,
    robot2cam: np.ndarray,
) -> np.ndarray:
    """机器人坐标系下的点 → OpenCV 相机坐标 (含 head 30° 俯仰)"""
    p_off = p_robot - cam_pos_robot
    return robot2cam @ p_off.astype(np.float32)


def camera_to_pixel(cam_x: float, cam_y: float, cam_z: float, cam_matrix: np.ndarray) -> Tuple[float, float]:
    """OpenCV 相机坐标 → 像素 (u, v)"""
    u = cam_matrix[0, 0] * cam_x / cam_z + cam_matrix[0, 2]
    v = cam_matrix[1, 1] * cam_y / cam_z + cam_matrix[1, 2]
    return float(u), float(v)


def world_to_camera_pixel(
    point_world: np.ndarray,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_matrix: np.ndarray = HEAD_CAM_MATRIX,
    cam_pos_robot: np.ndarray = HEAD_CAM_POS_ROBOT,
    robot2cam: np.ndarray = HEAD_CAM_ROT_MATRIX_INV,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Optional[Tuple[int, int]]:
    """
    世界坐标 → head 相机像素 (u, v)。
    点在相机后方或出画面时返回 None。
    """
    p_robot = world_to_robot(point_world, robot_pos, robot_yaw)
    p_cam = robot_offset_to_camera(p_robot, cam_pos_robot, robot2cam)
    if p_cam[2] <= MIN_CAM_Z:
        return None

    u, v = camera_to_pixel(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]), cam_matrix)
    ui, vi = int(round(u)), int(round(v))
    if 0 <= ui < img_w and 0 <= vi < img_h:
        return ui, vi
    return None


def compute_bbox_3d(
    point_world: np.ndarray,
    size_3d: Sequence[float],
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_matrix: np.ndarray = HEAD_CAM_MATRIX,
    cam_pos_robot: np.ndarray = HEAD_CAM_POS_ROBOT,
    robot2cam: np.ndarray = HEAD_CAM_ROT_MATRIX_INV,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Optional[Tuple[int, int, int, int]]:
    """
    物体中心 + 轴对齐尺寸 (lx, ly, lz) → 2D 像素 bbox (x1,y1,x2,y2)。

    将 3D 盒 8 角点投影到图像，取可见角点的外接矩形。
    注意: 未建模物体绕 Z 的旋转，与仿真初始姿态有偏差时框可能略偏。
    """
    lx, ly, lz = size_3d
    hx, hy, hz = lx / 2, ly / 2, lz / 2

    corners_local = np.array([
        [-hx, -hx,  hx,  hx, -hx, -hx,  hx,  hx],
        [-hy,  hy,  hy, -hy, -hy,  hy,  hy, -hy],
        [-hz, -hz, -hz, -hz,  hz,  hz,  hz,  hz],
    ], dtype=np.float32)
    corners_world = corners_local + np.asarray(point_world, dtype=np.float32).reshape(3, 1)

    pixels: List[Tuple[int, int]] = []
    for i in range(8):
        pix = world_to_camera_pixel(
            corners_world[:, i], robot_pos, robot_yaw,
            cam_matrix, cam_pos_robot, robot2cam, img_w, img_h,
        )
        if pix is not None:
            pixels.append(pix)

    if len(pixels) < 2:
        return None

    arr = np.array(pixels)
    x1 = max(0, int(arr[:, 0].min()))
    y1 = max(0, int(arr[:, 1].min()))
    x2 = min(img_w - 1, int(arr[:, 0].max()))
    y2 = min(img_h - 1, int(arr[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(
    bbox: Sequence[int],
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Tuple[float, float, float, float]:
    """像素 bbox → YOLO 归一化 (cx, cy, w, h)"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def bbox_to_yolo_line(
    class_id: int,
    bbox: Sequence[int],
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> str:
    cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def obj_index_to_class_id(obj_index: int) -> int:
    """仿真 object_1~18 → YOLO class_id (0/1/2)"""
    if obj_index <= 6:
        return 0
    if obj_index <= 12:
        return 1
    return 2


def class_id_to_size(class_id: int) -> Tuple[float, float, float]:
    name = CLASS_NAMES[class_id]
    d = OBJECT_SIZES.get(name, {"lx": 0.15, "ly": 0.10, "lz": 0.10})
    return d["lx"], d["ly"], d["lz"]


def project_object_labels(
    object_positions: Iterable[Tuple[int, np.ndarray]],
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> Tuple[List[str], int]:
    """
    批量投影多个物体 → YOLO 标签行列表。

    Args:
        object_positions: [(obj_index, world_pos(3,)), ...]
    Returns:
        (label_lines, visible_count)
    """
    labels: List[str] = []
    visible = 0
    for obj_idx, obj_pos in object_positions:
        cid = obj_index_to_class_id(obj_idx)
        size = class_id_to_size(cid)
        bbox = compute_bbox_3d(obj_pos, size, robot_pos, robot_yaw)
        if bbox is None:
            continue
        visible += 1
        labels.append(bbox_to_yolo_line(cid, bbox))
    return labels, visible


def _self_test():
    """简单自检: 机器人前方物体应投影到图像内"""
    robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
    robot_yaw = 0.0
    ok = 0
    cases = [
        (np.array([-8.0, -10.0, 0.15]), (0.21, 0.12, 0.07), True),
        (np.array([-12.0, -12.0, 0.15]), (0.21, 0.12, 0.07), False),
    ]
    print("=== projection_utils self-test ===")
    for pos, size, expect_visible in cases:
        bbox = compute_bbox_3d(pos, size, robot_pos, robot_yaw)
        vis = bbox is not None
        mark = "OK" if vis == expect_visible else "FAIL"
        if mark == "OK":
            ok += 1
        print(f"  pos={pos[:2]} visible={vis} bbox={bbox} [{mark}]")
    print(f"  {ok}/{len(cases)} passed")
    return ok == len(cases)


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
