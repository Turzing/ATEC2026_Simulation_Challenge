"""depth_ransac_cluster 合成桌面+物体单测."""

from __future__ import annotations

import numpy as np

from depth_ransac_cluster import RansacClusterDetector


def _synthetic_ee_depth(h=480, w=640) -> np.ndarray:
    depth = np.full((h, w), 1.35, dtype=np.float32)
    # 桌面区域: 较一致深度
    depth[200:420, 80:560] = 0.72
    # 物体 blob: 比桌面近
    depth[260:340, 280:380] = 0.58
    depth[270:330, 300:360] = 0.55
    return depth


def test_ransac_finds_object_cluster():
    det = RansacClusterDetector("ee")
    depth = _synthetic_ee_depth()
    robot_pos = np.zeros(3, dtype=np.float32)
    dets = det.detect(depth, robot_pos, 0.0)
    assert len(dets) >= 1, f"expected >=1 cluster, got {len(dets)}"
    best = dets[0]
    assert best.get("source") == "ransac_cluster"
    assert best.get("grasp_reliable") is True
    assert float(best.get("depth_m") or 99) < 0.85


if __name__ == "__main__":
    test_ransac_finds_object_cluster()
    print("ok")
