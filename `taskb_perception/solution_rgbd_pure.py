"""
部署用 — RGB-D 双摄感知

拷到 demo/ 时:
    solution.py              ← 本文件改名
    rgbd_pure_pipeline.py
    rgbd_pure_dual_pipeline.py
    config.py
    rgbd_utils.py
    policy.pt                ← 可选, 腿站立

运动层接口:
    out = pipeline.process(obs)
    # 远距导航
    out["target_nav"]           # EE 最近/最优目标
    out["ee_objects_list"]      # [{class, depth_m, dist_to_robot, pos_world}, ...]
    # 近距抓取 (phase=="grasp")
    g = out["target_grasp"]
    g["grasp_pos_world"]        # [x,y,z] 世界系下爪点
    g["grasp_quat_world"]       # [w,x,y,z] 抓取姿态
    g["pos_world"]              # 物体中心
"""

import os
import torch
from typing import Any

from rgbd_pure_dual_pipeline import RgbdPureDualPipeline


class AlgSolution:
    def __init__(self):
        print("[AlgSolution-RGBD-Dual] Initializing...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        if os.path.exists(policy_path):
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
        else:
            self.policy = None
        self.perception = RgbdPureDualPipeline()
        self.dt = 0.02
        print("[AlgSolution-RGBD-Dual] Ready")

    def reset(self):
        self.perception.reset()

    def act(self, obs: dict) -> Any:
        return self.perception.process(obs, self.dt)

    def predicts(self, obs, current_score):
        return self.act(obs)
