"""纯函数过滤器单测 — 无需 Isaac Sim."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rgbd_pure_dual_pipeline import (  # noqa: E402
    GRASP_PHASE_DIST_M,
    _is_ee_phantom_near,
    _is_head_nav_unreliable,
    _nav_dist_conservative,
    _strict_lock_match,
)
from rgbd_utils import bbox_lateral_consistent, is_ee_sky_blob  # noqa: E402


class TestRgbdDualFilters(unittest.TestCase):
    def test_sky_blob(self):
        sky = {
            "bbox": [10, 20, 50, 80],
            "depth_m": 1.45,
            "pos_robot": [1.4, -0.4, -0.5],
        }
        self.assertTrue(is_ee_sky_blob(sky))

    def test_phantom_sky(self):
        ee = {
            "id": 2,
            "class": "mustard_bottle",
            "bbox": [8, 15, 42, 70],
            "depth_m": 1.45,
            "pos_robot": [1.45, -0.42, -0.5],
            "world_reliable": True,
        }
        self.assertTrue(_is_ee_phantom_near(ee, []))

    def test_bearing_mismatch(self):
        bad = {
            "bbox": [40, 300, 120, 420],
            "pos_robot": [1.5, 0.9, -0.1],
            "depth_m": 1.6,
        }
        self.assertFalse(bbox_lateral_consistent(bad))

    def test_nav_dist_conservative(self):
        o = {"depth_m": 1.05, "dist_to_robot": 2.26}
        self.assertAlmostEqual(_nav_dist_conservative(o), 2.26)
        self.assertGreater(_nav_dist_conservative(o), GRASP_PHASE_DIST_M)

    def test_grasp_lock_match(self):
        lock = {"id": 0, "class": "mustard_bottle"}
        sugar = {"id": 1, "class": "sugar_box", "pos_world": [-7.8, -10.7, 0.1]}
        self.assertFalse(_strict_lock_match(sugar, lock))

    def test_head_lock_requires_points(self):
        weak = {
            "depth_m": 1.7,
            "pos_from_pointcloud": False,
            "nav_point_count": 3,
            "bbox": [200, 280, 280, 400],
            "pos_robot": [1.6, 0.05, -0.2],
        }
        self.assertFalse(_is_head_nav_unreliable(weak))
        jumped = dict(weak, pos_jump_rejected=True)
        self.assertTrue(_is_head_nav_unreliable(jumped))


if __name__ == "__main__":
    unittest.main()
