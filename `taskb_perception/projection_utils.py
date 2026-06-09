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


def quat_wxyz_to_rot(quat_wxyz: Sequence[float]) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转矩阵 (将相机/物体局部坐标旋到世界系)"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def axis_aligned_box_corners(size_3d: Sequence[float]) -> np.ndarray:
    """物体局部坐标系下的 8 个角点 → (3, 8)"""
    lx, ly, lz = size_3d
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    return np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float32).T


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


def world_to_camera_pixel_cam_pose(
    point_world: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w: Sequence[float],
    cam_matrix: np.ndarray = HEAD_CAM_MATRIX,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Optional[Tuple[int, int]]:
    """
    世界坐标 → 像素，使用仿真 head_camera 的真实位姿 (推荐)。
    cam_quat_w: Isaac Lab Camera.data.quat_w_ros，(w,x,y,z)。
    勿用 quat_w_world（+X 朝前），须用 ROS（+Z 朝前，与 OpenCV 针孔一致）。
    """
    p_world = np.asarray(point_world, dtype=np.float32).reshape(3)
    cam_pos = np.asarray(cam_pos_w, dtype=np.float32).reshape(3)
    r_c2w = quat_wxyz_to_rot(cam_quat_w)
    p_cam = r_c2w.T @ (p_world - cam_pos)
    if p_cam[2] <= MIN_CAM_Z:
        return None

    u, v = camera_to_pixel(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]), cam_matrix)
    ui, vi = int(round(u)), int(round(v))
    if 0 <= ui < img_w and 0 <= vi < img_h:
        return ui, vi
    return None


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


def _bbox_from_projected_pixels(
    pixels: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
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


def compute_bbox_3d_cam_pose(
    point_world: np.ndarray,
    obj_quat_wxyz: Sequence[float],
    size_3d: Sequence[float],
    cam_pos_w: np.ndarray,
    cam_quat_w: Sequence[float],
    cam_matrix: np.ndarray = HEAD_CAM_MATRIX,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Optional[Tuple[int, int, int, int]]:
    """
    物体中心 + 姿态 + 尺寸 → 2D bbox。
    使用仿真 head_camera 世界位姿投影 (采集脚本推荐路径)。
    """
    center = np.asarray(point_world, dtype=np.float32).reshape(3, 1)
    corners_local = axis_aligned_box_corners(size_3d)
    corners_world = quat_wxyz_to_rot(obj_quat_wxyz) @ corners_local + center

    pixels: List[Tuple[int, int]] = []
    for i in range(8):
        pix = world_to_camera_pixel_cam_pose(
            corners_world[:, i], cam_pos_w, cam_quat_w, cam_matrix, img_w, img_h,
        )
        if pix is not None:
            pixels.append(pix)
    return _bbox_from_projected_pixels(pixels, img_w, img_h)


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
    obj_quat_wxyz: Optional[Sequence[float]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    物体中心 + 尺寸 → 2D bbox (机器人基座近似外参，离线回退用)。
    优先在采集时使用 compute_bbox_3d_cam_pose。
    """
    center = np.asarray(point_world, dtype=np.float32).reshape(3, 1)
    corners_local = axis_aligned_box_corners(size_3d)
    if obj_quat_wxyz is not None:
        corners_world = quat_wxyz_to_rot(obj_quat_wxyz) @ corners_local + center
    else:
        corners_world = corners_local + center

    pixels: List[Tuple[int, int]] = []
    for i in range(8):
        pix = world_to_camera_pixel(
            corners_world[:, i], robot_pos, robot_yaw,
            cam_matrix, cam_pos_robot, robot2cam, img_w, img_h,
        )
        if pix is not None:
            pixels.append(pix)
    return _bbox_from_projected_pixels(pixels, img_w, img_h)


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
    cam_pos_w: Optional[np.ndarray] = None,
    cam_quat_w: Optional[Sequence[float]] = None,
    object_quats: Optional[dict] = None,
) -> Tuple[List[str], int]:
    """
    批量投影多个物体 → YOLO 标签行列表。

    Args:
        object_positions: [(obj_index, world_pos(3,)), ...]
        cam_pos_w / cam_quat_w: 仿真 head_camera 位姿 (优先)
        object_quats: {obj_index: quat_wxyz} 物体姿态
    Returns:
        (label_lines, visible_count)
    """
    labels: List[str] = []
    visible = 0
    for obj_idx, obj_pos in object_positions:
        cid = obj_index_to_class_id(obj_idx)
        size = class_id_to_size(cid)
        obj_q = None if object_quats is None else object_quats.get(obj_idx)
        if cam_pos_w is not None and cam_quat_w is not None and obj_q is not None:
            bbox = compute_bbox_3d_cam_pose(
                obj_pos, obj_q, size, cam_pos_w, cam_quat_w,
            )
        else:
            bbox = compute_bbox_3d(
                obj_pos, size, robot_pos, robot_yaw, obj_quat_wxyz=obj_q,
            )
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
