"""
数据采集脚本 — 在能跑仿真渲染的机器上运行（阿里云/腾讯云/恒源云/本地）
跑仿真 → 自动保存真实渲染图 + YOLO bbox 标签 → 给 AutoDL 训练用

用法:
    conda activate isaaclab   (或 taskb_data)
    cd ATEC2026_Simulation_Challenge
    python taskb_perception/collect_real_data.py --num_images 500 --output datasets/real

输出:
    datasets/real/images/train/   — RGB 图 (PNG)
    datasets/real/labels/train/   — YOLO 标注 (txt)
    datasets/real/dataset.yaml    — YOLO 配置
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

from isaaclab.app import AppLauncher

# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Collect real rendering data for YOLO training")
parser.add_argument("--num_images", type=int, default=500, help="Number of frames to collect")
parser.add_argument("--output", type=str, default="datasets/real", help="Output directory")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper", help="Task name")
parser.add_argument("--save_every", type=int, default=2, help="Save every N steps (avoid repetition)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# 数据采集需要相机渲染
if not getattr(args_cli, 'enable_cameras', False):
    args_cli.enable_cameras = True

# ═══════════════════════════════════════════════════════
# Launch Isaac Sim
# ═══════════════════════════════════════════════════════
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
import atec_rl_lab.tasks
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

# 导入感知层参数 (同目录, 直接用)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HEAD_CAM_MATRIX, IMG_W, IMG_H,
    HEAD_CAM_POS_ROBOT, HEAD_CAM_ROT_MATRIX_INV,
    OBJECT_SIZES, CLASS_NAMES,
)

# ═══════════════════════════════════════════════════════
# 类别映射: Object1-6=sugar_box, 7-12=mustard_bottle, 13-18=banana
# ═══════════════════════════════════════════════════════
def obj_to_class(obj_index: int):
    """物体序号(1-18) → YOLO class_id (0/1/2)"""
    if obj_index <= 6:
        return 0  # sugar_box
    elif obj_index <= 12:
        return 1  # mustard_bottle
    else:
        return 2  # banana


def world_to_bbox(world_pos, robot_pos, robot_yaw, size_3d):
    """物体世界坐标 → 像素 bbox [x1,y1,x2,y2], 含 30° 俯仰角"""
    dx, dy, dz = world_pos[0] - robot_pos[0], world_pos[1] - robot_pos[1], world_pos[2] - robot_pos[2]
    cy, sy = np.cos(-robot_yaw), np.sin(-robot_yaw)
    xr = cy * dx - sy * dy
    yr = sy * dx + cy * dy
    zr = dz

    p_off = np.array([xr - HEAD_CAM_POS_ROBOT[0],
                       yr - HEAD_CAM_POS_ROBOT[1],
                       zr - HEAD_CAM_POS_ROBOT[2]], dtype=np.float32)
    p_cam = HEAD_CAM_ROT_MATRIX_INV @ p_off
    if p_cam[2] <= 0.05:  # 在镜头后面
        return None

    # 针孔投影
    u = HEAD_CAM_MATRIX[0, 0] * p_cam[0] / p_cam[2] + HEAD_CAM_MATRIX[0, 2]
    v = HEAD_CAM_MATRIX[1, 1] * p_cam[1] / p_cam[2] + HEAD_CAM_MATRIX[1, 2]
    ui, vi = int(round(u)), int(round(v))

    if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
        return None

    # 根据距离和物体尺寸估算 bbox 像素大小
    dist = float(np.linalg.norm(world_pos[:2] - robot_pos[:2])) + 0.5
    pixel_w = max(12, int(IMG_W * size_3d[0] / (dist * 0.5)))
    pixel_h = max(12, int(IMG_H * size_3d[2] / (dist * 0.3)))
    half_w, half_h = pixel_w // 2, pixel_h // 2

    x1, y1 = max(0, ui - half_w), max(0, vi - half_h)
    x2, y2 = min(IMG_W - 1, ui + half_w), min(IMG_H - 1, vi + half_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def main():
    num_images = args_cli.num_images
    output_dir = args_cli.output
    save_every = args_cli.save_every

    # ---- 创建目录 ----
    img_dir = os.path.join(output_dir, "images", "train")
    lbl_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print("=" * 70)
    print(f"  ATEC Task B — Real Data Collector")
    print(f"  Target: {num_images} frames, save every {save_every} steps")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print("=" * 70)

    # ---- 创建仿真环境 ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                             use_fabric=not getattr(args_cli, 'disable_fabric', False))
    env_cfg = apply_safe_action_spec(env_cfg, "{}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else 0.02

    # ---- 主循环 ----
    collected = 0
    step_count = 0
    episode = 0

    while collected < num_images:
        episode += 1
        print(f"\n[DEBUG] Calling env.reset() for episode {episode}...", flush=True)
        obs, _ = env.reset()
        print(f"[DEBUG] env.reset() done. Episode {episode} started.", flush=True)

        # 前几步跳过（机器人还未站稳）
        for _ in range(10):
            actions = torch.zeros(1, 20, dtype=torch.float32, device='cuda:0')
            obs, _, _, _, _ = env.step(actions)
            step_count += 1

        for sim_step in range(500):
            if collected >= num_images:
                break

            # 师姐的移动控制器（有则用，没有则回退随机动作）
            try:
                from motion_controller import get_action as controller_get_action
                action = controller_get_action(obs, env)
            except ImportError:
                action = torch.randn(1, 20, dtype=torch.float32, device='cuda:0') * 0.05
            action = torch.tensor(action, dtype=torch.float32, device='cuda:0').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if step_count % 50 == 0:
                print(f"  [Step {step_count}] term={terminated}, trunc={truncated}", flush=True)

            if step_count % save_every != 0:
                if terminated or truncated:
                    break
                continue

            # ---- 取 RGB 图 ----
            try:
                rgb_tensor = obs['image']['head_rgb'].squeeze(0)
            except (KeyError, TypeError):
                continue

            if rgb_tensor.device.type == 'cuda':
                rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
            else:
                rgb_np = rgb_tensor.numpy().astype(np.uint8)

            # ---- 获取机器人位姿 ----
            try:
                robot_pos = env.unwrapped.scene['robot'].data.root_pos_w[0].cpu().numpy()
                # 从投影重力反算 yaw
                proprio = obs['proprio'].squeeze(0)
                if proprio.device.type == 'cuda':
                    proprio_np = proprio.cpu().numpy()
                else:
                    proprio_np = proprio.numpy()
                gx, gy = proprio_np[9], proprio_np[10]
                robot_yaw = float(np.arctan2(-gx, -gy))
            except Exception:
                robot_pos = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
                robot_yaw = 0.0

            # ---- 一次性诊断: 列出场景中所有物体属性 ----
            if episode == 1 and sim_step == 0:
                print("[DIAG] Scene attributes:", sorted([a for a in dir(env.unwrapped.scene) if not a.startswith('_')]))
                # 试读 object_1
                try:
                    p = env.unwrapped.scene.object_1.data.root_pos_w[0].cpu().numpy()
                    print(f"[DIAG] object_1 pos = {p}")
                except Exception as e:
                    print(f"[DIAG] object_1: {e}")
                # 也试试 Object1
                try:
                    p = env.unwrapped.scene.Object1.data.root_pos_w[0].cpu().numpy()
                    print(f"[DIAG] Object1 pos = {p}")
                except Exception as e:
                    print(f"[DIAG] Object1: {e}")

            # ---- 获取物体世界坐标 → 像素 bbox ----
            labels = []
            visible_count = 0
            for obj_idx in range(1, 19):
                obj_name = f"object_{obj_idx}"
                try:
                    scene_obj = getattr(env.unwrapped.scene, obj_name)
                    obj_pos = scene_obj.data.root_pos_w[0].cpu().numpy()
                except (AttributeError, KeyError):
                    continue

                class_id = obj_to_class(obj_idx)
                cls_name = CLASS_NAMES[class_id]
                size_dict = OBJECT_SIZES.get(cls_name, {"lx": 0.15, "ly": 0.10, "lz": 0.10})
                size_3d = (size_dict["lx"], size_dict["ly"], size_dict["lz"])

                bbox = world_to_bbox(obj_pos, robot_pos, robot_yaw, size_3d)
                if bbox is None:
                    continue

                visible_count += 1
                # YOLO 格式: class_id cx cy w h (归一化)
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2.0) / IMG_W
                cy = ((y1 + y2) / 2.0) / IMG_H
                w = (x2 - x1) / IMG_W
                h = (y2 - y1) / IMG_H
                labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # 跳过没有可见物体的帧（至少1个物体可见即可，放宽阈值）
            if visible_count < 1:
                if step_count % 20 == 0:
                    print(f"  [Step {step_count}] visible=0, skipping", flush=True)
                continue

            # ---- 保存 ----
            fname = f"render_{collected:06d}"
            cv2.imwrite(os.path.join(img_dir, f"{fname}.png"),
                        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            with open(os.path.join(lbl_dir, f"{fname}.txt"), 'w') as f:
                f.write('\n'.join(labels) + '\n')

            collected += 1
            print(f"  [{collected:4d}/{num_images}] {fname}.png — {visible_count} objects visible")

            if terminated or truncated:
                break

    env.close()

    # ---- 生成 dataset.yaml ----
    yaml_content = f"""# Auto-generated by collect_real_data.py
path: {os.path.abspath(output_dir)}
train: images/train
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
"""
    with open(os.path.join(output_dir, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print("\n" + "=" * 70)
    print(f"  ✅ 数据采集完成！")
    print(f"  图片: {img_dir} ({collected} 张)")
    print(f"  标签: {lbl_dir}")
    print(f"  配置: {output_dir}/dataset.yaml")
    print("=" * 70)
    print("\n下一步:")
    print(f"  1. 下载 {output_dir}/ 到本地或传到 AutoDL")
    print(f"  2. 在 taskb_perception 运行: python train_yolo.py --data ../{os.path.basename(output_dir)}/{os.path.basename(output_dir)}.yaml")
    print(f"  3. 部署: cp runs/train/*/weights/best.pt taskb_perception/taskb_ycb.pt")


if __name__ == "__main__":
    main()