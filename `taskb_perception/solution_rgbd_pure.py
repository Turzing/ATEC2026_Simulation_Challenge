"""
部署用 — 老师版 RGB-D 感知 + RL 腿

拷到 demo/ 时:
    solution.py              ← 本文件改名
    rgbd_pure_pipeline.py
    rgbd_pure_dual_pipeline.py
    config.py
    rgbd_utils.py
"""

import os
import numpy as np
import torch
from typing import Any

from rgbd_pure_dual_pipeline import RgbdPureDualPipeline


class AlgSolution:
    def __init__(self):
        print("[AlgSolution-RGBD-Pure] Initializing...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        if os.path.exists(policy_path):
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
        else:
            self.policy = None
        self.perception = RgbdPureDualPipeline()
        self.dt = 0.02
        print("[AlgSolution-RGBD-Pure] Ready (ee+head RGBD fusion)")

    def reset(self):
        self.perception.reset()

    def act(self, obs: dict) -> Any:
        out = self.perception.process(obs, self.dt)
        # 运动/抓取状态机由师姐层接 target / grasp_pos_world
        return out
