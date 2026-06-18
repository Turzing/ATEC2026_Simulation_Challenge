"""
Task B 感知入口 — 复用已验证的 rgbd_pure_dual 管线

  EE   → 远/近导航 (HSV黄 + depth relief + 时序滤波)
  Head → 近距抓取 3D + grasp_quat

与 test_rgbd_pure_dual.py / README_师姐使用手册 一致。
"""

from __future__ import annotations

from rgbd_pure_dual_pipeline import RgbdPureDualPipeline

PERCEPTION_BUILD = "rgbd-pure-dual"

# solution_rl.py 固定 import 此名
TaskBPerception = RgbdPureDualPipeline

__all__ = ["TaskBPerception", "RgbdPureDualPipeline", "PERCEPTION_BUILD"]
