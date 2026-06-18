"""Isaac 视口控制 — 第三人称跟随 / 机器人传感器视角."""

from __future__ import annotations

import math

import numpy as np
import torch
import isaaclab.utils.math as math_utils


def _quat_wxyz_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _yaw_from_quat_wxyz(quat) -> float:
    w, x, y, z = (float(quat[i]) for i in range(4))
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _set_viewport_eye_lookat(env, eye: np.ndarray, lookat: np.ndarray, env_index: int = 0) -> bool:
    """写入 Isaac 主视口 (viewport_controller + sim.set_camera_view 双通道)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)
    unwrapped = env.unwrapped
    ok = False

    if hasattr(unwrapped, "viewport_camera_controller"):
        try:
            ctrl = unwrapped.viewport_camera_controller
            ctrl.set_view_env_index(env_index=env_index)
            ctrl.update_view_location(eye=eye, lookat=lookat)
            ok = True
        except Exception:
            pass

    sim_ctx = getattr(unwrapped, "sim", None)
    if sim_ctx is not None and hasattr(sim_ctx, "set_camera_view"):
        try:
            sim_ctx.set_camera_view(eye.tolist(), lookat.tolist())
            ok = True
        except Exception:
            pass

    return ok


def camera_follow_behind(
    env,
    robot_name: str = "robot",
    env_index: int = 0,
    distance: float = 4.0,
    height: float = 1.4,
    alpha: float = 0.18,
) -> bool:
    """第三人称: 机器人正后方平视 (不是天上俯视)."""
    unwrapped = env.unwrapped
    try:
        robot = unwrapped.scene[robot_name]
    except KeyError:
        return False

    pos = robot.data.root_pos_w[env_index].detach().cpu().numpy().astype(np.float64)
    quat = robot.data.root_quat_w[env_index].detach().cpu().numpy()
    yaw = _yaw_from_quat_wxyz(quat)
    fx, fy = math.cos(yaw), math.sin(yaw)
    target_eye = np.array(
        [pos[0] - fx * distance, pos[1] - fy * distance, pos[2] + height],
        dtype=np.float64,
    )
    target_lookat = np.array(
        [pos[0] + fx * 1.2, pos[1] + fy * 1.2, pos[2] + 0.35],
        dtype=np.float64,
    )

    key = (id(unwrapped), env_index, "follow_behind")
    if not hasattr(camera_follow_behind, "_smooth"):
        camera_follow_behind._smooth = {}
    if key not in camera_follow_behind._smooth:
        camera_follow_behind._smooth[key] = target_eye.copy()
    smooth_eye = (1.0 - alpha) * camera_follow_behind._smooth[key] + alpha * target_eye
    camera_follow_behind._smooth[key] = smooth_eye
    return _set_viewport_eye_lookat(env, smooth_eye, target_lookat, env_index=env_index)


def camera_robot_sensor_view(
    env,
    camera_name: str = "head_camera",
    env_index: int = 0,
    look_ahead: float = 2.0,
) -> bool:
    """第一人称: 传感器 pos + 光轴 (head / ee)."""
    unwrapped = env.unwrapped
    try:
        cam = unwrapped.scene[camera_name]
    except KeyError:
        return False
    cam_data = cam.data
    if not hasattr(cam_data, "pos_w"):
        return False
    pos = cam_data.pos_w[env_index].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(pos).all() or float(pos[2]) > 8.0 or float(pos[2]) < -1.0:
        return False
    if hasattr(cam_data, "quat_w_ros"):
        quat = cam_data.quat_w_ros[env_index].detach().cpu().numpy()
    elif hasattr(cam_data, "quat_w_world"):
        quat = cam_data.quat_w_world[env_index].detach().cpu().numpy()
    elif hasattr(cam_data, "root_quat_w"):
        quat = cam_data.root_quat_w[env_index].detach().cpu().numpy()
    else:
        return False
    w, x, y, z = (float(quat[i]) for i in range(4))
    forward = _quat_wxyz_to_rot(w, x, y, z) @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    lookat = pos + forward.astype(np.float64) * float(look_ahead)
    return _set_viewport_eye_lookat(env, pos, lookat, env_index=env_index)


def camera_follow(env, robot_name: str = "robot", env_index: int = 0, alpha: float = 0.15):
    """兼容旧接口 — 优先用 yaw 后方跟随 (比 body-frame offset 更稳)."""
    if camera_follow_behind(env, robot_name=robot_name, env_index=env_index, alpha=alpha):
        return

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "viewport_camera_controller"):
        return

    try:
        robot = unwrapped.scene[robot_name]
    except KeyError as e:
        raise KeyError(
            f"Robot asset '{robot_name}' not found in env.unwrapped.scene."
        ) from e

    device = unwrapped.device
    robot_pos = robot.data.root_pos_w[env_index]
    robot_quat = robot.data.root_quat_w[env_index]
    camera_offset = torch.tensor([-6.0, 0.0, 0.8], dtype=torch.float32, device=device)
    target_camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0),
        pos=robot_pos.unsqueeze(0),
        quat=robot_quat.unsqueeze(0),
    ).squeeze(0)
    target_camera_pos[2] = torch.clamp(target_camera_pos[2], min=0.2)

    if not hasattr(camera_follow, "_smooth_pos"):
        camera_follow._smooth_pos = {}
    if env_index not in camera_follow._smooth_pos:
        camera_follow._smooth_pos[env_index] = target_camera_pos.clone()
    smooth_camera_pos = camera_follow._smooth_pos[env_index]
    smooth_camera_pos = (1.0 - alpha) * smooth_camera_pos + alpha * target_camera_pos
    camera_follow._smooth_pos[env_index] = smooth_camera_pos
    _set_viewport_eye_lookat(
        env,
        smooth_camera_pos.detach().cpu().numpy(),
        robot_pos.detach().cpu().numpy(),
        env_index=env_index,
    )
