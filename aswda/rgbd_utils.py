"""
官方 obs['image'] 里 head_rgb / head_depth 的解析与诊断

Isaac Lab Camera depth:
    - float32, 单位米, 沿光轴距离 (与 perception_pipeline 一致)
    - 无效像素常为 inf 或 clipping 上限 (~50)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# 默认: 纯 depth 聚类 (ATEC_RGBD_SIMPLE=0 才走旧 fusion 管线)
RGBD_SIMPLE = os.getenv("ATEC_RGBD_SIMPLE", "1").strip().lower() not in ("0", "false", "no")
# 两步走: 粗导航(head depth) → 停稳趴下 → 静态 EE RANSAC 抓取
STATIC_TWO_STEP = os.getenv("ATEC_TASKB_STATIC_TWO_STEP", "1").strip().lower() not in ("0", "false", "no")

from config import (
    BBOX_LATERAL_TOL,
    DEFAULT_ARM_JOINTS,
    EE_CAM,
    EE_CAM_POS_ROBOT,
    EE_CAM_ROT_MATRIX,
    EE_NAV_ROBOT_Z_MIN,
    EE_PHANTOM_HEAD_GAP_M,
    EE_PHANTOM_NEAR_M,
    EE_SKY_CY_FRAC,
    GRASP_DEPTH_OFFSET,
    GRIPPER_TIP_OFFSET_ENABLE,
    GRIPPER_TIP_OFFSET_M,
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    HEAD_NAV_BOTTOM_FRAC,
    HEAD_NAV_Z_PERCENTILE,
    IMG_H,
    IMG_W,
    MIN_NAV_LOCK_CONF,
    MIN_NAV_POINT_COUNT,
    MIN_NAV_POS_CONF,
    MOTION_GRASP_HEIGHT_OFFSET,
    POS_JUMP_REJECT_FAR_M,
    POS_JUMP_REJECT_NEAR_M,
)

CAMERA_MODELS = {
    "head": (HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX),
    "ee": (EE_CAM, EE_CAM_POS_ROBOT, EE_CAM_ROT_MATRIX),
}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "device") and getattr(x, "device", None) is not None:
        if x.device.type == "cuda":
            x = x.cpu()
    return np.asarray(x)


def align_rgb_to_depth(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """rgb/depth 尺寸不一致时对齐，避免 relief/HSV broadcast 崩溃"""
    h, w = depth.shape[:2]
    rgb = np.asarray(rgb)
    if rgb.ndim < 3:
        return rgb
    rgb = rgb[..., :3].astype(np.uint8)
    rh, rw = rgb.shape[:2]
    if rh == h and rw == w:
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def parse_head_rgbd(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    obs['image']['head_rgb']   → (H,W,3) uint8 RGB
    obs['image']['head_depth'] → (H,W)   float32 米
    """
    rgb = _to_numpy(obs["image"]["head_rgb"]).squeeze()
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb.astype(np.uint8)[..., :3]

    dep = _to_numpy(obs["image"]["head_depth"]).squeeze()
    if dep.ndim == 3:
        dep = dep[..., 0]
    dep = dep.astype(np.float32)
    return rgb, sanitize_depth(dep)


def parse_ee_rgbd(obs: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        rgb = _to_numpy(obs["image"]["ee_rgb"]).squeeze()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = rgb.astype(np.uint8)[..., :3]
        dep = _to_numpy(obs["image"]["ee_depth"]).squeeze()
        if dep.ndim == 3:
            dep = dep[..., 0]
        return rgb, sanitize_depth(dep.astype(np.float32))
    except (KeyError, TypeError, AttributeError):
        return None, None


def sanitize_depth(depth: np.ndarray, max_m: float = 49.5) -> np.ndarray:
    d = depth.copy()
    bad = ~np.isfinite(d) | (d <= 0.01) | (d > max_m)
    d[bad] = 0.0
    return d


def depth_stats(depth: np.ndarray) -> Dict[str, float]:
    valid = depth[(depth > 0.05) & (depth < 49.0)]
    if valid.size == 0:
        return {"valid_ratio": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "valid_ratio": float(valid.size / depth.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
    }


def depth_to_vis(depth: np.ndarray, d_min: float = 0.5, d_max: float = 8.0) -> np.ndarray:
    """深度 → 伪彩色 BGR (调试用)"""
    d = depth.copy()
    d[d <= 0] = d_max
    d = np.clip(d, d_min, d_max)
    norm = ((d_max - d) / max(d_max - d_min, 1e-3) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def median_depth_in_mask(depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
    vals = depth[mask > 0]
    vals = vals[(vals > 0.05) & (vals < 49.0)]
    if len(vals) < 5:
        return None
    return float(np.median(vals))


def estimate_pos_robot(
    centroid_uv, depth_m: Optional[float], camera: str,
) -> Optional[np.ndarray]:
    """检测框中心 + depth → 机器人基座系 3D 点 (双相机融合用)"""
    if depth_m is None or depth_m <= 0.05 or camera not in CAMERA_MODELS:
        return None
    cam, pos, rot = CAMERA_MODELS[camera]
    u, v = float(centroid_uv[0]), float(centroid_uv[1])
    p_cam = pixel_depth_to_cam(u, v, float(depth_m), cam)
    return (pos + rot @ p_cam).astype(np.float32)


def horizontal_dist_robot(pos_robot: Optional[np.ndarray]) -> Optional[float]:
    if pos_robot is None:
        return None
    return float(np.linalg.norm(pos_robot[:2]))


def annotate_pos_robot(obj: dict, camera: str) -> dict:
    """给单相机 object 补上 pos_robot / dist_to_robot (掩码反投影优先于框中心)"""
    out = dict(obj)
    pos = None
    if obj.get("pos_robot") is not None:
        pos = np.asarray(obj["pos_robot"], dtype=np.float32)
    if pos is None:
        pos = estimate_pos_robot(obj.get("centroid_uv"), obj.get("depth_m"), camera)
        if pos is not None:
            out["pos_robot"] = pos.tolist()
    dist = horizontal_dist_robot(pos)
    if dist is not None:
        out["dist_to_robot"] = dist
    return out


def robot_to_world(
    p_robot, robot_pos: np.ndarray, robot_yaw: float,
) -> np.ndarray:
    """机器人基座系 → 世界系"""
    p = np.asarray(p_robot, dtype=np.float32).reshape(3)
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (robot_pos.astype(np.float32) + rot @ p).astype(np.float32)


def world_to_robot_frame(
    p_world, robot_pos: np.ndarray, robot_yaw: float,
) -> np.ndarray:
    """世界系 → 机器人基座系"""
    pw = np.asarray(p_world, dtype=np.float32).reshape(3)
    rp = np.asarray(robot_pos, dtype=np.float32).reshape(3)
    dx, dy, dz = pw[0] - rp[0], pw[1] - rp[1], pw[2] - rp[2]
    c, s = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
    return np.array([c * dx + s * dy, -s * dx + c * dy, dz], dtype=np.float32)


GRASP_RELIABLE_DEPTH_M = 0.80
WORLD_RELIABLE_DEPTH_M = 2.00

# 地面物体在 robot 系下的合理 z 范围 (蹲下后略放宽)
ROBOT_Z_MIN = -0.78
ROBOT_Z_MAX = 0.28


def bbox_lateral_consistent(
    obj: dict,
    *,
    img_w: int = IMG_W,
    tol: float = BBOX_LATERAL_TOL,
) -> bool:
    """bbox 在图像左/右应与 pos_robot 横向符号一致, 否则 depth 锚点错位."""
    bbox = obj.get("bbox")
    pr = obj.get("pos_robot")
    if not bbox or len(bbox) != 4 or pr is None:
        return True
    try:
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        px, py = float(pr[0]), float(pr[1])
    except (TypeError, ValueError):
        return True
    if abs(px) < 0.22 and abs(py) < 0.22:
        return True
    img_b = (cx - img_w * 0.5) / max(img_w * 0.5, 1.0)
    rob_b = py / max(float(np.hypot(px, py)), 0.18)
    if abs(img_b) < 0.06 or abs(rob_b) < 0.06:
        return True
    return abs(np.clip(img_b, -1.0, 1.0) - np.clip(rob_b, -1.0, 1.0)) <= tol


def is_head_edge_phantom(obj: dict, *, img_w: int = IMG_W) -> bool:
    """head bbox 贴画面边缘 → depth/3D 不可靠 (log: sugar 框在 u≈580 却报 1.05m 正前方)."""
    bbox = obj.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    bw = float(bbox[2]) - float(bbox[0])
    depth = float(obj.get("depth_m") or obj.get("nav_depth_m") or 99.0)
    at_edge = cx > img_w * 0.72 or cx < img_w * 0.28
    if cx > img_w * 0.78 or cx < img_w * 0.22:
        return True
    if at_edge and bw < img_w * 0.24 and depth < 2.2:
        return True
    return False


def is_sky_phantom_bbox(obj: dict, *, img_h: int = IMG_H) -> bool:
    """物体在地面: bbox 底边应在画面下半区; 否则是天空/地平线假检 (截图红框打天)."""
    bbox = obj.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    y1, y2 = float(bbox[1]), float(bbox[3])
    if y2 < img_h * 0.52:
        return True
    cy = 0.5 * (y1 + y2)
    bh = max(y2 - y1, 1.0)
    if cy < img_h * 0.36 and bh > img_h * 0.10:
        return True
    return False


def is_head_floor_phantom(obj: dict, *, img_h: int = IMG_H, img_w: int = IMG_W) -> bool:
    """Head 地砖/近地板大面积假检 (截图: 下半屏巨框 → 假 banana)."""
    bbox = obj.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = map(float, bbox)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    img_area = float(img_h * img_w)
    cy = 0.5 * (y1 + y2)
    depth = float(obj.get("depth_m") or obj.get("nav_depth_m") or 99.0)
    pixels = int(obj.get("cluster_pixels") or 0)
    if cy > img_h * 0.56 and area > img_area * 0.10:
        return True
    if y1 > img_h * 0.40 and area > img_area * 0.16:
        return True
    if depth < 1.20 and cy > img_h * 0.48 and area > img_area * 0.07:
        return True
    if pixels > 1500 and cy > img_h * 0.50:
        return True
    return False


def is_ee_floor_phantom(obj: dict, *, img_h: int = IMG_H, img_w: int = IMG_W) -> bool:
    """EE 地砖/空地板假 mustard (log: 0.94 conf 框内无物, pos_w≈[-9,-9.87])."""
    if obj.get("grasp_reliable"):
        return False
    depth = float(obj.get("depth_m") or obj.get("nav_depth_m") or 99.0)
    if depth > 1.20 or depth < 0.50:
        return False
    sm = float(obj.get("blob_sat_mean") or 0.0)
    vm = float(obj.get("blob_val_mean") or 0.0)
    bbox = obj.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    area = int((x2 - x1 + 1) * (y2 - y1 + 1))
    cy = 0.5 * (float(y1) + float(y2))
    if depth < 1.05 and sm < 34 and vm > 115 and area < 3200:
        return True
    if depth < 0.90 and sm < 42 and cy > img_h * 0.38 and area < 2800:
        return True
    return False


def is_ee_sky_blob(obj: dict, *, img_h: int = IMG_H, img_w: int = IMG_W) -> bool:
    """EE 框在画面上方 + 深度假近 → 地平线 phantom (log 里 conf 0.86 天空 mustard)."""
    if is_sky_phantom_bbox(obj, img_h=img_h):
        return True
    bbox = obj.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
    if cy >= img_h * EE_SKY_CY_FRAC:
        return False
    depth = float(obj.get("depth_m") or obj.get("nav_depth_m") or 99.0)
    pr = obj.get("pos_robot")
    pz = float(pr[2]) if pr is not None else 0.0
    if depth < 2.4 and cy < img_h * 0.32:
        return True
    if pz < EE_NAV_ROBOT_Z_MIN and depth < 2.2:
        return True
    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    if cy < img_h * 0.28 and (cx < img_w * 0.12 or cx > img_w * 0.88):
        return True
    return False


def _is_head_fallback_det(obj: dict) -> bool:
    return bool(
        obj.get("head_far_fallback")
        or obj.get("head_depth_fallback")
        or obj.get("head_neutral_fallback")
    )


def is_valid_taskb_ground_det(obj: dict, *, img_h: int = IMG_H) -> bool:
    """
    Task B 平地垃圾: 用 robot 系 3D 高度判断是否在地面, 不用 bbox 上下位置.
    head 俯拍时远距物体在画面上方 (小 v), 不能用 is_sky_phantom_bbox 误杀.
    """
    pr = obj.get("pos_robot")
    if pr is None:
        return False
    try:
        px, py, pz = float(pr[0]), float(pr[1]), float(pr[2])
    except (TypeError, ValueError, IndexError):
        return False
    if px < 0.22 or px > 8.5:
        return False
    if pz < -0.82 or pz > -0.04:
        return False
    if float(np.hypot(px, py)) < 0.10:
        return False

    if obj.get("head_far_fallback") or obj.get("head_depth_fallback") or obj.get("head_neutral_fallback"):
        return True

    src = str(obj.get("source") or "")
    if src == "ransac_cluster":
        pixels = int(obj.get("cluster_pixels") or 0)
        extent = obj.get("geom_extent")
        max_ext = float(max(extent)) if isinstance(extent, (list, tuple)) and extent else 0.0
        if pz > -0.10:
            return False
        if pixels > 520 or max_ext > 1.05:
            return False
        return True

    if is_sky_phantom_bbox(obj, img_h=img_h) and pz > -0.30:
        return False
    return True


def is_head_ransac_phantom(obj: dict, *, img_h: int = IMG_H, img_w: int = IMG_W) -> bool:
    """RANSAC head 假检: 仅杀悬空/超大簇 (不杀远距地面上方像素)."""
    if str(obj.get("source") or "") != "ransac_cluster":
        return False
    return not is_valid_taskb_ground_det(obj, img_h=img_h)


def _filter_plausible_simple(objects: list, camera: str) -> list:
    """简化后滤: 杀 sky/地板 phantom (含 RANSAC)."""
    out = []
    for o in objects:
        pr = o.get("pos_robot")
        if pr is None:
            continue
        try:
            px, py, pz = float(pr[0]), float(pr[1]), float(pr[2])
        except (TypeError, ValueError, IndexError):
            continue
        if pz < -0.95 or pz > 0.45:
            continue
        if float(np.hypot(px, py)) < 0.08 and abs(py) < 0.14:
            continue
        if float(px) < 0.12:
            continue
        if camera == "head":
            if not is_valid_taskb_ground_det(o):
                continue
            if is_head_floor_phantom(o) and int(o.get("cluster_pixels") or 0) > 2200:
                continue
        if camera == "ee":
            if is_ee_sky_blob(o) or is_ee_floor_phantom(o):
                continue
        out.append(o)
    return out


def filter_plausible_objects(
    objects: list,
    camera: str,
    *,
    ee_near_m: float = 999.0,
) -> list:
    """剔除机器人本体误检、地下/悬空坐标"""
    if RGBD_SIMPLE:
        return _filter_plausible_simple(objects, camera)
    out = []
    for o in objects:
        pr = o.get("pos_robot")
        if pr is None:
            continue
        try:
            px, py, pz = float(pr[0]), float(pr[1]), float(pr[2])
        except (TypeError, ValueError, IndexError):
            continue
        if pz < ROBOT_Z_MIN or pz > ROBOT_Z_MAX:
            continue
        if float(np.hypot(px, py)) < 0.12 and abs(py) < 0.18:
            continue
        depth = float(o.get("depth_m") or 99.0)
        sm = float(o.get("blob_sat_mean") or 0.0)
        vm = float(o.get("blob_val_mean") or 0.0)
        bbox = o.get("bbox")
        if camera == "head":
            if _is_head_fallback_det(o) and 0.75 <= depth <= 6.5:
                bbox = o.get("bbox")
                if bbox and len(bbox) == 4 and depth < 1.15:
                    area = int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
                    if area < 480 and sm < 14 and vm > 145:
                        continue
                if is_sky_phantom_bbox(o):
                    continue
                out.append(o)
                continue
            if is_sky_phantom_bbox(o):
                continue
            if is_head_edge_phantom(o) and not (depth >= 1.35 and _is_head_fallback_det(o)):
                continue
            if depth < 2.5 and not _is_head_fallback_det(o) and not bbox_lateral_consistent(o):
                continue
            if depth < 1.05 and sm < 46:
                continue
            if depth < 0.80 and sm < 54:
                continue
            if bbox and len(bbox) == 4:
                cx = 0.5 * (bbox[0] + bbox[2])
                cy = 0.5 * (bbox[1] + bbox[3])
                if depth < 0.85 and cy > 200 and 160 < cx < 480 and sm < 58:
                    continue
                if depth < 0.55 and cy > 140 and sm < 62 and 55 < vm < 155:
                    continue
        if camera == "ee":
            if depth < 0.35 and sm < 40:
                continue
            if is_ee_sky_blob(o):
                continue
            if is_ee_floor_phantom(o):
                continue
        out.append(o)
    return out


def compute_dynamic_ee_cam_pos(arm_joints) -> np.ndarray:
    """臂关节相对默认姿 → EE 相机在 robot 系下的偏移 (无 GT, 蹲下/伸臂时修正)"""
    q = np.asarray(arm_joints, dtype=np.float32).reshape(-1)[:6]
    dq = q - DEFAULT_ARM_JOINTS
    delta = np.array([
        0.16 * np.sin(dq[0]) + 0.12 * np.sin(dq[2]) + 0.08 * dq[3] + 0.04 * dq[5],
        0.08 * np.sin(dq[0]) + 0.05 * np.cos(dq[1]) + 0.03 * dq[4],
        0.12 * (np.cos(dq[1]) - 1.0) + 0.09 * np.sin(dq[3]) + 0.06 * dq[4] + 0.05 * dq[2],
    ], dtype=np.float32)
    return (EE_CAM_POS_ROBOT + delta).astype(np.float32)


def compute_dynamic_head_cam_pos(projected_gravity) -> np.ndarray:
    """蹲下/倾身时修正 head 外参 (projected_gravity 偏离 [0,0,-1])"""
    pos = HEAD_CAM_POS_ROBOT.copy()
    if projected_gravity is None:
        return pos
    g = np.asarray(projected_gravity, dtype=np.float32).reshape(3)
    gn = float(np.linalg.norm(g))
    if gn < 1e-4:
        return pos
    g = g / gn
    tilt_xy = float(np.hypot(g[0], g[1]))
    pos[0] += 0.10 * float(g[0]) + 0.04 * float(g[1])
    pos[2] += 0.14 * tilt_xy
    return pos.astype(np.float32)


def pixel_to_robot(
    u: float,
    v: float,
    depth_m: float,
    cam_cfg: dict,
    cam_pos_robot: np.ndarray,
    cam_rot_matrix: np.ndarray,
) -> np.ndarray:
    p_cam = pixel_depth_to_cam(u, v, float(depth_m), cam_cfg)
    return (np.asarray(cam_pos_robot, dtype=np.float32) + cam_rot_matrix @ p_cam).astype(np.float32)


def quat_rotate_vec_wxyz(q_wxyz, v) -> np.ndarray:
    """Rotate vector v by unit quaternion [w, x, y, z]."""
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    v = np.asarray(v, dtype=np.float32).reshape(3)
    w, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qv = np.array([qx, qy, qz], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return (v + w * t + np.cross(qv, t)).astype(np.float32)


def compensate_grasp_for_gripper_base(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """
    grasp_pos 先按指尖接触点估计，再回退为 gripper_base IK 目标。

    solution_gt: ee_body=gripper_base, DISABLE_FINGER_COMP 默认开 → 不再做指尖补偿；
    start_grasp 还会对 trash_pos_w 沿 world +Z 加 MOTION_GRASP_HEIGHT_OFFSET。
    """
    if not GRIPPER_TIP_OFFSET_ENABLE:
        return obj
    out = dict(obj)
    contact_w = out.get("grasp_pos_world")
    if contact_w is None:
        return out
    contact_w = np.asarray(contact_w, dtype=np.float32)
    tip = float(GRIPPER_TIP_OFFSET_M)
    gq = out.get("grasp_quat_world")
    if gq is not None:
        z_world = quat_rotate_vec_wxyz(gq, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        zn = float(np.linalg.norm(z_world))
        if zn > 1e-4:
            z_world = z_world / zn
        base_w = contact_w - z_world * tip
    else:
        contact_r = out.get("grasp_pos_robot")
        if contact_r is None:
            return out
        cr = np.asarray(contact_r, dtype=np.float32)
        horiz = float(np.hypot(cr[0], cr[1]))
        if horiz < 0.05:
            return out
        dir_xy = cr[:2] / horiz
        base_r = cr - np.array([dir_xy[0] * tip, dir_xy[1] * tip, 0.0], dtype=np.float32)
        base_w = robot_to_world(base_r, robot_pos, robot_yaw)
    base_w = base_w - np.array([0.0, 0.0, float(MOTION_GRASP_HEIGHT_OFFSET)], dtype=np.float32)
    out["grasp_tip_contact_world"] = contact_w.tolist()
    out["grasp_pos_world"] = base_w.tolist()
    out["grasp_pos_robot"] = world_to_robot_frame(base_w, robot_pos, robot_yaw).tolist()
    pos_r = out.get("pos_robot")
    if pos_r is not None:
        out["grasp_offset_robot"] = (
            np.asarray(out["grasp_pos_robot"], dtype=np.float32) - np.asarray(pos_r, dtype=np.float32)
        ).tolist()
    out["gripper_base_compensated"] = True
    return out


def stabilize_ee_nav_pose(obj: dict) -> dict:
    """侧视 EE 远距时 pos_robot 横向偏差大，用 depth + yaw 拉回导航距离."""
    out = dict(obj)
    pr = out.get("pos_robot")
    depth = out.get("nav_depth_m") or out.get("depth_m")
    if pr is None or depth is None:
        return out
    try:
        px, py, pz = float(pr[0]), float(pr[1]), float(pr[2])
        depth_f = float(depth)
    except (TypeError, ValueError, IndexError):
        return out
    if depth_f < 0.90 or depth_f > 2.60 or out.get("nav_from_head"):
        return out
    xy = float(np.hypot(px, py))
    if xy < 0.15:
        return out
    yaw = float(out.get("nav_yaw_rel") or out.get("yaw_rel") or np.arctan2(py, px))
    blend = 0.55 if depth_f > 1.60 else 0.35
    nav_xy = depth_f * blend + xy * (1.0 - blend)
    out["pos_robot"] = [
        float(nav_xy * np.cos(yaw)),
        float(nav_xy * np.sin(yaw)),
        pz,
    ]
    out["dist_to_robot"] = float(np.hypot(out["pos_robot"][0], out["pos_robot"][1]))
    out["yaw_rel"] = yaw
    out["nav_yaw_rel"] = yaw
    out["nav_depth_m"] = depth_f
    return out


def refresh_ee_object_pose(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_pos_robot: np.ndarray,
) -> dict:
    """每帧用 centroid(导航) + anchor(抓取) + 当前 EE 外参重算 world"""
    out = dict(obj)
    nav_uv = out.get("centroid_uv")
    nav_depth = out.get("depth_m")
    anchor_uv = out.get("grasp_anchor_uv") or nav_uv
    anchor_depth = out.get("grasp_anchor_depth") or nav_depth
    if nav_uv is None or nav_depth is None:
        return out
    try:
        nu, nv = float(nav_uv[0]), float(nav_uv[1])
        nav_d = float(nav_depth)
        au, av = float(anchor_uv[0]), float(anchor_uv[1])
        anchor_d = float(anchor_depth)
    except (TypeError, ValueError, IndexError):
        return out
    if nav_d <= 0.05:
        return out

    pos_r = pixel_to_robot(nu, nv, nav_d, EE_CAM, cam_pos_robot, EE_CAM_ROT_MATRIX)
    offset = np.asarray(
        out.get("grasp_offset_robot") or [0.0, 0.0, -float(GRASP_DEPTH_OFFSET)],
        dtype=np.float32,
    )
    if anchor_d > 0.05:
        anchor_r = pixel_to_robot(au, av, anchor_d, EE_CAM, cam_pos_robot, EE_CAM_ROT_MATRIX)
        grasp_r = anchor_r + offset
    else:
        grasp_r = pos_r + offset
    out["pos_robot"] = pos_r.tolist()
    out["pos_world"] = robot_to_world(pos_r, robot_pos, robot_yaw).tolist()
    out["grasp_pos_robot"] = grasp_r.tolist()
    out["grasp_pos_world"] = robot_to_world(grasp_r, robot_pos, robot_yaw).tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pos_r[:2]))
    out["yaw_rel"] = float(np.arctan2(pos_r[1], pos_r[0]))
    out["nav_depth_m"] = nav_d
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["grasp_reliable"] = nav_d < GRASP_RELIABLE_DEPTH_M
    out["world_reliable"] = nav_d < WORLD_RELIABLE_DEPTH_M
    return compensate_grasp_for_gripper_base(out, robot_pos, robot_yaw)


def head_nav_pos_confidence(obj: dict) -> float:
    n = int(obj.get("nav_point_count") or 0)
    sm = float(obj.get("blob_sat_mean") or 0.0)
    vm = float(obj.get("blob_val_mean") or 0.0)
    depth = float(obj.get("depth_m") or 99.0)
    score = min(1.0, n / max(MIN_NAV_POINT_COUNT, 1)) * 0.45
    if obj.get("pos_from_pointcloud"):
        score += 0.38
    if depth < 1.35:
        score += 0.08
    if sm > 48 and vm > 65:
        score += 0.09
    if obj.get("pos_jump_rejected"):
        score *= 0.55
    return float(np.clip(score, 0.0, 1.0))


def pos_jump_limit_m(dist_m: float) -> float:
    return POS_JUMP_REJECT_NEAR_M if dist_m < 1.35 else POS_JUMP_REJECT_FAR_M


def reject_pos_world_jump(
    obj: dict,
    prev_world: Optional[np.ndarray],
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """单帧 world XY 跳变过大 → 保持上一帧稳定位置."""
    out = dict(obj)
    pw = out.get("pos_world")
    if pw is None or prev_world is None:
        return out
    pw_np = np.asarray(pw, dtype=np.float32)
    prev_np = np.asarray(prev_world, dtype=np.float32).reshape(3)
    dist = float(out.get("dist_to_robot") or out.get("nav_depth_m") or out.get("depth_m") or 2.0)
    lim = pos_jump_limit_m(dist)
    if float(np.linalg.norm(pw_np[:2] - prev_np[:2])) > lim:
        out["pos_world"] = prev_np.tolist()
        pr = world_to_robot_frame(prev_np, robot_pos, robot_yaw)
        out["pos_robot"] = pr.tolist()
        out["dist_to_robot"] = float(np.linalg.norm(pr[:2]))
        out["yaw_rel"] = float(np.arctan2(pr[1], pr[0]))
        out["nav_yaw_rel"] = out["yaw_rel"]
        out["pos_jump_rejected"] = True
        out["world_reliable"] = False
    return out


def refresh_head_object_pose(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_pos_robot: np.ndarray,
) -> dict:
    """head: 底边 anchor + 点云融合 + 动态外参 (不用 centroid 单点 depth 覆盖点云)"""
    out = dict(obj)
    anchor_uv = out.get("nav_anchor_uv") or out.get("centroid_uv")
    depth = out.get("nav_anchor_depth") or out.get("depth_m")
    if anchor_uv is None or depth is None:
        return out
    try:
        au, av = float(anchor_uv[0]), float(anchor_uv[1])
        depth_f = float(depth)
    except (TypeError, ValueError, IndexError):
        return out
    if depth_f <= 0.05:
        return out

    anchor_r = pixel_to_robot(au, av, depth_f, HEAD_CAM, cam_pos_robot, HEAD_CAM_ROT_MATRIX)
    if out.get("pos_from_pointcloud") and out.get("pos_robot") is not None:
        pc_r = np.asarray(out["pos_robot"], dtype=np.float32)
        n = int(out.get("nav_point_count") or 0)
        src = str(out.get("source") or "")
        if src == "rgbd_nav_head":
            w_pc = min(0.90, 0.68 + n / 55.0)
            pos_r = (w_pc * pc_r + (1.0 - w_pc) * anchor_r).astype(np.float32)
            pos_r[2] = float(pc_r[2])
        elif src == "ransac_cluster" and n <= 72:
            w_pc = min(0.72, 0.48 + n / 90.0)
            pos_r = (w_pc * pc_r + (1.0 - w_pc) * anchor_r).astype(np.float32)
            pos_r[2] = float(pc_r[2])
        elif src == "ransac_cluster" or n >= 18:
            pos_r = pc_r.copy()
        else:
            w_pc = min(0.82, 0.55 + n / 80.0)
            pos_r = (w_pc * pc_r + (1.0 - w_pc) * anchor_r).astype(np.float32)
            pos_r[2] = pc_r[2]
    else:
        pos_r = anchor_r.astype(np.float32)

    out["pos_robot"] = pos_r.tolist()
    out["pos_world"] = robot_to_world(pos_r, robot_pos, robot_yaw).tolist()
    out["dist_to_robot"] = float(np.linalg.norm(pos_r[:2]))
    out["yaw_rel"] = float(np.arctan2(pos_r[1], pos_r[0]))
    out["nav_depth_m"] = depth_f
    out["nav_yaw_rel"] = out["yaw_rel"]
    conf = head_nav_pos_confidence(out)
    out["pos_confidence"] = conf
    out["world_reliable"] = (
        depth_f < WORLD_RELIABLE_DEPTH_M
        and conf >= MIN_NAV_POS_CONF * 0.40
    )
    return out


def align_nav_pos_to_bbox_ray(
    obj: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
    cam_pos_robot: np.ndarray,
    cam_cfg: dict,
    cam_rot_matrix: np.ndarray,
) -> dict:
    """bbox 底边像素射线定方位; head 点云时只修横向."""
    out = dict(obj)
    bbox = out.get("bbox")
    pr = out.get("pos_robot")
    if not bbox or len(bbox) != 4 or pr is None:
        return out
    if out.get("pos_from_pointcloud"):
        return out
    try:
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        cy = float(bbox[3]) - 1.0
        anchor_uv = out.get("nav_anchor_uv")
        if anchor_uv is not None and len(anchor_uv) == 2:
            cx = float(anchor_uv[0])
            cy = float(anchor_uv[1])
        depth = float(
            out.get("nav_anchor_depth")
            or out.get("nav_depth_m")
            or out.get("depth_m")
            or np.linalg.norm(np.asarray(pr, dtype=np.float32)[:2])
        )
    except (TypeError, ValueError, IndexError):
        return out
    if depth <= 0.05:
        return out
    ray = pixel_to_robot(cx, cy, depth, cam_cfg, cam_pos_robot, cam_rot_matrix)
    dist = float(np.linalg.norm(np.asarray(pr, dtype=np.float32)[:2]))
    ray_xy = float(np.linalg.norm(ray[:2]))
    if ray_xy < 0.05:
        return out
    scale = dist / ray_xy
    pos_r = np.array(
        [float(ray[0]) * scale, float(ray[1]) * scale, float(pr[2]) if len(pr) > 2 else float(ray[2])],
        dtype=np.float32,
    )
    out["pos_robot"] = pos_r.tolist()
    out["pos_world"] = robot_to_world(pos_r, robot_pos, robot_yaw).tolist()
    out["dist_to_robot"] = dist
    out["yaw_rel"] = float(np.arctan2(pos_r[1], pos_r[0]))
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["nav_from_bbox"] = True
    return out


def refresh_locked_grasp(
    frozen: dict,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """蹲下后冻结 robot 系抓取点，只随 robot pose 更新 world"""
    out = dict(frozen)
    out["grasp_locked"] = True
    pr = out.get("pos_robot")
    gr = out.get("grasp_pos_robot")
    if pr is not None:
        pr_np = np.asarray(pr, dtype=np.float32)
        out["pos_world"] = robot_to_world(pr_np, robot_pos, robot_yaw).tolist()
        out["dist_to_robot"] = float(np.linalg.norm(pr_np[:2]))
        out["yaw_rel"] = float(np.arctan2(pr_np[1], pr_np[0]))
        out["nav_yaw_rel"] = out["yaw_rel"]
    if gr is not None:
        out["grasp_pos_world"] = robot_to_world(
            np.asarray(gr, dtype=np.float32), robot_pos, robot_yaw,
        ).tolist()
    out["grasp_reliable"] = True
    out["world_reliable"] = True
    return out


def pick_reproject_uv_depth(obj: dict) -> Optional[Tuple[float, float, float]]:
    """GT 重投影用: head 优先 nav_anchor + 近端 depth."""
    anchor_uv = obj.get("nav_anchor_uv")
    anchor_depth = obj.get("nav_anchor_depth") or obj.get("nav_depth_m")
    centroid_uv = obj.get("centroid_uv") or obj.get("centroid")
    depth_m = obj.get("depth_m")
    uv = anchor_uv if anchor_uv is not None else centroid_uv
    depth = anchor_depth if anchor_uv is not None and anchor_depth is not None else depth_m
    if uv is None or depth is None:
        return None
    try:
        u = float(uv[0])
        v = float(uv[1])
        depth_f = float(depth)
    except (TypeError, ValueError, IndexError):
        return None
    if depth_f <= 0.01 or depth_f > 100.0:
        return None
    return u, v, depth_f


def apply_gt_camera_pose(
    obj: dict,
    cam_pos_w: np.ndarray,
    cam_rot_w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> dict:
    """
    与 solution_rl._correct_object_with_camera_pose 同公式:
    centroid_uv + depth_m + GT 相机位姿 → pos_world;
    grasp = pos_robot + grasp_offset_robot (robot 系偏移, 与 pos 同步校正).
    """
    from config import GRASP_DEPTH_OFFSET

    out = dict(obj)
    picked = pick_reproject_uv_depth(out)
    if picked is None:
        return out
    u, v, depth_m_f = picked

    x = (u - cx) / fx * depth_m_f
    y = (v - cy) / fy * depth_m_f
    p_cam = np.array([x, y, depth_m_f], dtype=np.float32)
    p_world = np.asarray(cam_pos_w, dtype=np.float32) + np.asarray(cam_rot_w, dtype=np.float32) @ p_cam
    p_robot = world_to_robot_frame(p_world, robot_pos, robot_yaw)

    out["pos_world"] = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
    out["pos_robot"] = [float(p_robot[0]), float(p_robot[1]), float(p_robot[2])]
    out["dist_to_robot"] = float(np.linalg.norm(p_robot[:2]))
    out["yaw_rel"] = float(np.arctan2(p_robot[1], p_robot[0]))
    out["nav_depth_m"] = depth_m_f
    out["nav_yaw_rel"] = out["yaw_rel"]
    out["world_reliable"] = True

    offset = out.get("grasp_offset_robot")
    if offset is None:
        offset = [0.0, 0.0, -float(GRASP_DEPTH_OFFSET)]
    grasp_robot = p_robot + np.asarray(offset, dtype=np.float32)
    out["grasp_offset_robot"] = [float(offset[0]), float(offset[1]), float(offset[2])]
    out["grasp_pos_robot"] = [float(grasp_robot[0]), float(grasp_robot[1]), float(grasp_robot[2])]
    out["grasp_pos_world"] = robot_to_world(grasp_robot, robot_pos, robot_yaw).tolist()
    out["grasp_reliable"] = depth_m_f < GRASP_RELIABLE_DEPTH_M
    out["pose_source"] = "gt_camera"
    return out


def annotate_world_coords(
    obj: dict, robot_pos: np.ndarray, robot_yaw: float,
) -> dict:
    """pos_robot / grasp_pos_robot → 世界坐标 (操作层用 grasp_pos_world)"""
    out = dict(obj)
    pr = obj.get("pos_robot")
    if pr is not None:
        pw = robot_to_world(pr, robot_pos, robot_yaw)
        out["pos_world"] = pw.tolist()
    gr = obj.get("grasp_pos_robot")
    if gr is not None:
        out["grasp_pos_world"] = robot_to_world(gr, robot_pos, robot_yaw).tolist()
    elif out.get("pos_world") is not None:
        gp = np.asarray(out["pos_world"], dtype=np.float32).copy()
        from config import GRASP_DEPTH_OFFSET
        gp[2] -= GRASP_DEPTH_OFFSET
        out["grasp_pos_world"] = gp.tolist()
    return out


def pixel_depth_to_cam(
    u: float, v: float, z: float, cam: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    c = cam or HEAD_CAM
    x = (u - c["cx"]) / c["fx"] * z
    y = (v - c["cy"]) / c["fy"] * z
    return np.array([x, y, z], dtype=np.float32)


def bbox_center_depth(depth: np.ndarray, bbox, pad: int = 0) -> Optional[float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth.shape
    x1, y1 = max(0, x1 + pad), max(0, y1 + pad)
    x2, y2 = min(w - 1, x2 - pad), min(h - 1, y2 - pad)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth[y1:y2, x1:x2]
    valid = patch[(patch > 0.05) & (patch < 49.0)]
    if len(valid) < 3:
        return None
    return float(np.median(valid))


def format_stats_line(stats: Dict[str, float]) -> str:
    return (
        f"valid={stats['valid_ratio']*100:.1f}% "
        f"med={stats['median']:.2f}m "
        f"range=[{stats['min']:.2f},{stats['max']:.2f}]"
    )
