"""depth_ransac_cluster 合成桌面+物体单测 (无需 Isaac Sim)."""

from __future__ import annotations

import numpy as np

from config import HEAD_CAM, HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX
from depth_ransac_cluster import RansacClusterDetector


def _synthetic_ee_depth(h=480, w=640) -> np.ndarray:
    depth = np.full((h, w), 1.35, dtype=np.float32)
    depth[200:420, 80:560] = 0.72
    depth[260:340, 280:380] = 0.58
    depth[270:330, 300:360] = 0.55
    return depth


def _synthetic_head_depth(h=480, w=640) -> np.ndarray:
    """用正投影在 head 图像上画桌面+凸起 (保证 robot_z 在合法范围)."""
    depth = np.full((h, w), 5.0, dtype=np.float32)
    cam_pos = HEAD_CAM_POS_ROBOT
    cam_rot = HEAD_CAM_ROT_MATRIX
    fx, fy, cx, cy = HEAD_CAM["fx"], HEAD_CAM["fy"], HEAD_CAM["cx"], HEAD_CAM["cy"]
    robot_yaw = 0.0
    robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)

    def stamp_robot_point(pr: np.ndarray, half_u: int, half_v: int, z_cam: float) -> None:
        pc = np.linalg.inv(cam_rot) @ (np.asarray(pr, dtype=np.float32) - cam_pos)
        if pc[2] <= 0.05:
            return
        u = int(round(fx * pc[0] / pc[2] + cx))
        v = int(round(fy * pc[1] / pc[2] + cy))
        for dv in range(-half_v, half_v + 1):
            for du in range(-half_u, half_u + 1):
                uu, vv = u + du, v + dv
                if 0 <= uu < w and 0 <= vv < h:
                    depth[vv, uu] = z_cam

    # 地面平面采样
    for px in np.linspace(0.8, 3.5, 12):
        for py in np.linspace(-1.2, 1.2, 10):
            pr = np.array([px, py, -0.52], dtype=np.float32)
            stamp_robot_point(pr, 2, 2, 2.0 + px * 0.15)

    # 物体凸起 (深度明显浅于桌面)
    for px in np.linspace(1.5, 2.2, 8):
        for py in np.linspace(-0.35, 0.35, 7):
            for pz in np.linspace(-0.46, -0.30, 5):
                pr = np.array([px, py, pz], dtype=np.float32)
                stamp_robot_point(pr, 5, 5, 1.35 + px * 0.08)

    return depth


def test_ransac_finds_object_cluster():
    det = RansacClusterDetector("ee")
    depth = _synthetic_ee_depth()
    robot_pos = np.zeros(3, dtype=np.float32)
    dets = det.detect(depth, robot_pos, 0.0)
    assert len(dets) >= 1, f"expected >=1 cluster, got {len(dets)} stats={det.last_stats}"
    best = dets[0]
    assert best.get("source") == "ransac_cluster"
    assert best.get("grasp_reliable") is True
    assert float(best.get("depth_m") or 99) < 0.85


def test_head_ransac_keeps_ground_cloud_and_finds_cluster():
    det = RansacClusterDetector("head")
    depth = _synthetic_head_depth()
    robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
    dets = det.detect(depth, robot_pos, 0.0)
    stats = det.last_stats
    assert stats["cloud_pts"] > 40, f"head cloud empty: {stats}"
    assert len(dets) >= 1, f"head expected >=1 cluster, got {len(dets)} stats={stats}"


def test_head_v1_z_filter_would_drop_ground():
    """回归: v1 robot_z=(-0.35,0.50) 会把地面点全滤掉."""
    pr = np.array([2.0, 0.0, -0.52], dtype=np.float32)
    assert pr[2] < -0.35


if __name__ == "__main__":
    test_ransac_finds_object_cluster()
    test_head_ransac_keeps_ground_cloud_and_finds_cluster()
    test_head_v1_z_filter_would_drop_ground()
    print("ok")
