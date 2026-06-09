"""
ATEC 2026 Task B — 纯 RGB-D 感知（无 YOLO）

思路:
    head_depth → 去地面 → 连通域聚类 → 每簇 3D 质心/顶面 → 抓取点
    head_rgb   → 簇内颜色/形状启发式 → 粗分类 (糖盒/瓶/香蕉)

与 perception_pipeline.py 接口兼容:
    pipeline = RgbdPerceptionPipeline()
    out = pipeline.process(obs, dt=0.02)

不修改、不依赖 YOLO / ultralytics。仅 import config。

仿真内自测 (需 Isaac):
    在 solution 里把 PerceptionPipeline 换成 RgbdPerceptionPipeline 即可对比。

离线示意 (合成图):
    cd taskb_perception
    python rgbd_perception_pipeline.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HEAD_CAM,
    HEAD_CAM_POS_ROBOT,
    HEAD_CAM_ROT_MATRIX,
    HEAD_CAM_ROT_MATRIX_INV,
    ROBOT_INIT_POS,
    ROBOT_INIT_YAW,
    BIN_CENTER,
    BIN_RADIUS,
    BIN_Z_MIN,
    BIN_Z_MAX,
    CLASS_NAMES,
    CLASS_NAME_TO_ID,
    TOTAL_OBJECTS,
    OBJECT_SIZES,
    DEFAULT_OBJECT_SIZE,
    GRASP_FIXED_QUAT,
    DEFAULT_GRASP_FIXED_QUAT,
    GRASP_DEPTH_OFFSET,
    GRIPPER_HOLDING_MAX_WIDTH,
    PROPRIO_JOINT_POS_START,
    PROPRIO_JOINT_POS_LENGTH,
    PROPRIO_FINGER_LEFT,
    PROPRIO_FINGER_RIGHT,
    PROPRIO_BASE_LIN_VEL,
    PROPRIO_BASE_ANG_VEL,
    PROPRIO_PROJECTED_GRAVITY,
    PROPRIO_YAW_FUSION_ALPHA,
)

# =============================================================================
# 可调参数 — 仿真里不对就改这里
# =============================================================================
DEPTH_MIN = 0.25          # 最近有效深度 (m)
DEPTH_MAX = 8.0           # 最远有效深度 (m)
WORLD_Z_MIN = 0.03        # 世界系高度: 地面杂物 (糖盒~0.15m)
WORLD_Z_MAX = 0.45        # 世界系高度上限
MIN_CLUSTER_PIXELS = 25   # 簇最小像素数 (远处物体很小)
MAX_CLUSTERS = 20         # 每帧最多保留簇数
TRACK_MATCH_DIST = 0.35   # 跨帧 3D 关联距离 (m)
TRACK_MAX_AGE = 25        # 丢失多少帧后删 track
# 头顶相机俯视: 物体多在画面中部偏下 (比例 0~1)
ROI_V_MIN = 0.22
ROI_V_MAX = 0.92
DEPTH_ABOVE_GROUND = 0.04  # 比局部地面近多少 (m) 视为前景


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def yaw_from_gravity(projected_gravity):
    gx, gy, _ = projected_gravity
    return float(np.arctan2(-gx, -gy))


class SimpleTracker3D:
    """用 3D 位置做跨帧关联 (无 bbox IoU)"""

    def __init__(self, match_dist: float = TRACK_MATCH_DIST, max_age: int = TRACK_MAX_AGE):
        self.match_dist = match_dist
        self.max_age = max_age
        self.tracks: Dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections: List[dict]) -> List[dict]:
        for t in self.tracks.values():
            t["age"] += 1

        assigned = set()
        out = []
        for det in detections:
            pos = np.asarray(det["pos_world"], dtype=np.float32)
            best_id, best_d = None, float("inf")
            for tid, t in self.tracks.items():
                if tid in assigned:
                    continue
                d = float(np.linalg.norm(pos - t["pos_world"]))
                if d < self.match_dist and d < best_d:
                    best_d, best_id = d, tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {"pos_world": pos.copy(), "age": 0, "hits": 0}

            assigned.add(best_id)
            tr = self.tracks[best_id]
            tr["pos_world"] = 0.7 * tr["pos_world"] + 0.3 * pos
            tr["age"] = 0
            tr["hits"] = tr.get("hits", 0) + 1
            out.append({**det, "track_id": best_id})

        stale = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        return out


class RgbdPerceptionPipeline:
  """
  纯 RGB-D 感知: 深度聚类定位 + RGB 启发式分类。
  输出格式与 PerceptionPipeline.process() 一致。
  """

  def __init__(self):
      self.head_cam2robot = HEAD_CAM_ROT_MATRIX.copy()
      self.head_robot2cam = HEAD_CAM_ROT_MATRIX_INV.copy()
      self.head_pos_robot = HEAD_CAM_POS_ROBOT.copy()
      self.robot_pos = ROBOT_INIT_POS.copy()
      self.robot_yaw = ROBOT_INIT_YAW
      self.tracker = SimpleTracker3D()
      self.in_bin_ids = set()
      self.frame_count = 0
      print("[RgbdPerceptionPipeline] ready (no YOLO)")

  def reset(self):
      self.tracker = SimpleTracker3D()
      self.robot_pos = ROBOT_INIT_POS.copy()
      self.robot_yaw = ROBOT_INIT_YAW
      self.in_bin_ids = set()
      self.frame_count = 0

  # ------------------------------------------------------------------ 坐标变换
  def _cam_to_robot(self, p_cam: np.ndarray) -> np.ndarray:
      return self.head_pos_robot + self.head_cam2robot @ p_cam

  def _robot_to_cam(self, p_robot: np.ndarray) -> np.ndarray:
      return self.head_robot2cam @ (p_robot - self.head_pos_robot)

  def _robot_to_world(self, p_robot: np.ndarray) -> np.ndarray:
      c, s = np.cos(self.robot_yaw), np.sin(self.robot_yaw)
      rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
      return self.robot_pos + rot @ p_robot

  def _world_to_robot(self, p_world: np.ndarray) -> np.ndarray:
      c, s = np.cos(self.robot_yaw), np.sin(self.robot_yaw)
      rot_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
      return rot_inv @ (p_world - self.robot_pos)

  def _pixel_depth_to_cam(self, u: float, v: float, depth: float) -> np.ndarray:
      x = (u - HEAD_CAM["cx"]) / HEAD_CAM["fx"] * depth
      y = (v - HEAD_CAM["cy"]) / HEAD_CAM["fy"] * depth
      return np.array([x, y, depth], dtype=np.float32)

  def _pixel_to_world(self, u: float, v: float, depth: float) -> Optional[np.ndarray]:
      if depth <= 0 or not np.isfinite(depth):
          return None
      if depth < DEPTH_MIN or depth > DEPTH_MAX:
          return None
      p_cam = self._pixel_depth_to_cam(u, v, depth)
      if p_cam[2] <= 0.05:
          return None
      return self._robot_to_world(self._cam_to_robot(p_cam))

  # ------------------------------------------------------------------ RGB-D 检测
  def _world_z_map(self, depth: np.ndarray) -> np.ndarray:
      """深度图 → 每个像素的世界系 Z"""
      h, w = depth.shape
      us = np.arange(w, dtype=np.float32)
      vs = np.arange(h, dtype=np.float32)
      uu, vv = np.meshgrid(us, vs)
      x = (uu - HEAD_CAM["cx"]) / HEAD_CAM["fx"] * depth
      y = (vv - HEAD_CAM["cy"]) / HEAD_CAM["fy"] * depth
      p_cam = np.stack([x, y, depth], axis=-1)
      p_robot = np.einsum("ij,...j->...i", self.head_cam2robot, p_cam) + self.head_pos_robot
      c, s = np.cos(self.robot_yaw), np.sin(self.robot_yaw)
      rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
      p_world = np.einsum("ij,...j->...i", rot, p_robot) + self.robot_pos
      return p_world[..., 2]

  def _build_object_mask(self, depth: np.ndarray) -> np.ndarray:
      """有效深度 + 世界高度 / 相对深度 → 前景 mask"""
      h, w = depth.shape
      valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)

      vs = np.arange(h, dtype=np.float32)
      vv = np.meshgrid(np.arange(w), vs)[1]
      in_roi = (vv >= int(h * ROI_V_MIN)) & (vv <= int(h * ROI_V_MAX))

      world_z = self._world_z_map(depth)
      height_ok = (world_z >= WORLD_Z_MIN) & (world_z <= WORLD_Z_MAX)

      # 相对深度: 比画面下半部地面更近的凸起 (不依赖里程计也很稳)
      v0, v1 = int(h * ROI_V_MIN), int(h * 0.75)
      roi = depth[v0:v1, :]
      roi_valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
      if len(roi_valid) > 100:
          ground_d = float(np.percentile(roi_valid, 35))
      else:
          ground_d = float(np.median(depth[valid])) if np.any(valid) else 3.0
      protrude = depth < (ground_d - DEPTH_ABOVE_GROUND)

      mask = valid & in_roi & (height_ok | protrude)
      mask = mask.astype(np.uint8)
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
      self._last_debug_mask = mask
      return mask

  def get_debug_mask(self, depth: np.ndarray) -> np.ndarray:
      """调试用: 看哪些像素被当成前景 (白=前景)"""
      if getattr(self, "_last_debug_mask", None) is not None:
          return (self._last_debug_mask * 255).astype(np.uint8)
      return (self._build_object_mask(depth) * 255).astype(np.uint8)

  def _cluster_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
      labeled, n = ndimage.label(mask)
      if n == 0:
          return labeled, 0
      counts = np.bincount(labeled.ravel())
      counts[0] = 0
      keep = np.where(counts >= MIN_CLUSTER_PIXELS)[0]
      out = np.zeros_like(labeled)
      new_id = 0
      for old_id in keep:
          new_id += 1
          out[labeled == old_id] = new_id
      return out, new_id

  def _classify_cluster(
      self, rgb: np.ndarray, ys: np.ndarray, xs: np.ndarray, bbox: List[int],
      z_extent: float,
  ) -> Tuple[str, float]:
      """RGB 颜色 + 2D 形状 + 3D 高度 → 粗分类"""
      x1, y1, x2, y2 = bbox
      bw, bh = max(1, x2 - x1), max(1, y2 - y1)
      aspect = bw / bh

      patch = rgb[ys, xs].astype(np.float32)
      if patch.size == 0:
          return "sugar_box", 0.3

      bgr = patch[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
      hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
      hue = float(np.median(hsv[:, 0]))
      sat = float(np.median(hsv[:, 1]))
      val = float(np.median(hsv[:, 2]))

      scores = {
          "sugar_box": 0.2,
          "mustard_bottle": 0.2,
          "banana": 0.2,
      }

      # 黄色系 (仿真里瓶/香蕉偏黄)
      if 15 <= hue <= 45 and sat > 40:
          scores["mustard_bottle"] += 0.35
          scores["banana"] += 0.30
      if val > 120:
          scores["mustard_bottle"] += 0.15
          scores["banana"] += 0.15

      # 细长 → 瓶或香蕉
      if aspect > 1.35:
          scores["banana"] += 0.45
          scores["mustard_bottle"] += 0.25
      elif aspect < 0.95:
          scores["sugar_box"] += 0.45

      # 扁平盒: 3D 高度小、2D 不太细长
      if z_extent < 0.10 and aspect < 1.3:
          scores["sugar_box"] += 0.40

      # 竖立瓶: 3D 高度较大
      if z_extent > 0.11 and aspect < 1.25:
          scores["mustard_bottle"] += 0.35

      name = max(scores, key=scores.get)
      conf = float(min(0.92, scores[name]))
      return name, conf

  def _detect_from_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> List[dict]:
      mask = self._build_object_mask(depth)
      labeled, n = self._cluster_mask(mask)
      if n == 0:
          return []

      detections = []
      for cid in range(1, n + 1):
          ys, xs = np.where(labeled == cid)
          if len(ys) < MIN_CLUSTER_PIXELS:
              continue

          depths = depth[ys, xs]
          depths = depths[(depths > DEPTH_MIN) & (depths < DEPTH_MAX)]
          if len(depths) < 10:
              continue

          # 3D 点云 (世界系)
          world_pts = []
          for y, x in zip(ys[::3], xs[::3]):  # 子采样提速
              pw = self._pixel_to_world(float(x), float(y), float(depth[y, x]))
              if pw is not None:
                  world_pts.append(pw)
          if len(world_pts) < 5:
              continue
          world_pts = np.stack(world_pts, axis=0)

          pos_world = np.median(world_pts, axis=0)
          z_extent = float(world_pts[:, 2].max() - world_pts[:, 2].min())

          # 顶面抓取: 取簇内较高点
          z_top = float(np.percentile(world_pts[:, 2], 92))
          grasp_pos = pos_world.copy()
          grasp_pos[2] = z_top - GRASP_DEPTH_OFFSET

          x1, x2 = int(xs.min()), int(xs.max())
          y1, y2 = int(ys.min()), int(ys.max())
          bbox = [x1, y1, x2, y2]

          cls_name, conf = self._classify_cluster(rgb, ys, xs, bbox, z_extent)

          detections.append({
              "class": cls_name,
              "class_id": CLASS_NAME_TO_ID.get(cls_name, 0),
              "conf": conf,
              "pos_world": pos_world,
              "grasp_pos_world": grasp_pos,
              "bbox": bbox,
              "source": "rgbd_cluster",
              "cluster_pixels": int(len(ys)),
          })

      detections.sort(key=lambda d: float(np.linalg.norm(d["pos_world"][:2] - self.robot_pos[:2])))
      return detections[:MAX_CLUSTERS]

  # ------------------------------------------------------------------ 抓取 / 状态
  def _compute_grasp_quat(self, class_name: str, obj_pos_world: np.ndarray) -> np.ndarray:
      fixed_q = GRASP_FIXED_QUAT.get(class_name, DEFAULT_GRASP_FIXED_QUAT).copy()
      dx = obj_pos_world[0] - self.robot_pos[0]
      dy = obj_pos_world[1] - self.robot_pos[1]
      approach_yaw = float(np.arctan2(dy, dx))
      rot_yaw = R.from_euler("z", approach_yaw).as_quat(scalar_first=True).astype(np.float32)
      return quat_multiply(rot_yaw, fixed_q)

  def _read_gripper(self, proprio: np.ndarray) -> dict:
      joint_pos = proprio[PROPRIO_JOINT_POS_START:PROPRIO_JOINT_POS_START + PROPRIO_JOINT_POS_LENGTH]
      fl = PROPRIO_FINGER_LEFT - PROPRIO_JOINT_POS_START
      fr = PROPRIO_FINGER_RIGHT - PROPRIO_JOINT_POS_START
      if len(joint_pos) <= fr:
          return {"is_holding": False, "width": 0.04, "held_object_id": None}
      width = abs(joint_pos[fl] - joint_pos[fr]) * GRIPPER_HOLDING_MAX_WIDTH * 2.5
      width = float(np.clip(width, 0.0, 0.08))
      return {
          "is_holding": bool(0.002 < width < GRIPPER_HOLDING_MAX_WIDTH),
          "width": width,
          "held_object_id": None,
      }

  def _check_in_bin(self, pos_world: np.ndarray) -> bool:
      dist_xy = np.linalg.norm(pos_world[:2] - BIN_CENTER[:2])
      z = pos_world[2]
      return bool(dist_xy <= BIN_RADIUS and BIN_Z_MIN <= z <= BIN_Z_MAX)

  def _update_robot_pose(self, proprio: np.ndarray, dt: float = 0.02):
      lin_vel = proprio[PROPRIO_BASE_LIN_VEL]
      ang_vel = proprio[PROPRIO_BASE_ANG_VEL]
      grav = proprio[PROPRIO_PROJECTED_GRAVITY]

      c, s = np.cos(self.robot_yaw), np.sin(self.robot_yaw)
      rot = np.array([[c, -s], [s, c]], dtype=np.float32)
      dxy = rot @ lin_vel[:2] * dt
      self.robot_pos[0] += dxy[0]
      self.robot_pos[1] += dxy[1]
      self.robot_pos[2] = ROBOT_INIT_POS[2]

      yaw_grav = yaw_from_gravity(grav)
      yaw_gyro = self.robot_yaw + ang_vel[2] * dt
      alpha = PROPRIO_YAW_FUSION_ALPHA
      delta = alpha * yaw_grav + (1 - alpha) * yaw_gyro
      self.robot_yaw = float((delta + np.pi) % (2 * np.pi) - np.pi)

  # ------------------------------------------------------------------ 主入口
  def process(self, obs, dt: float = 0.02) -> dict:
      self.frame_count += 1

      h_rgb = obs["image"]["head_rgb"].squeeze(0)
      h_dep = obs["image"]["head_depth"].squeeze(0)
      proprio = obs["proprio"].squeeze(0)

      rgb = (h_rgb.cpu() if hasattr(h_rgb, "device") and h_rgb.device.type == "cuda" else h_rgb)
      rgb = np.asarray(rgb, dtype=np.uint8)
      if rgb.ndim == 3 and rgb.shape[-1] == 3:
          pass
      else:
          rgb = rgb[..., :3]

      dep = (h_dep.cpu() if hasattr(h_dep, "device") and h_dep.device.type == "cuda" else h_dep)
      dep = np.asarray(dep, dtype=np.float32).squeeze()
      if dep.ndim == 3:
          dep = dep[..., 0]

      p_np = (proprio.cpu() if hasattr(proprio, "device") and proprio.device.type == "cuda" else proprio)
      p_np = np.asarray(p_np, dtype=np.float32)

      self._update_robot_pose(p_np, dt)

      raw_dets = self._detect_from_rgbd(rgb, dep)
      tracks = self.tracker.update(raw_dets)

      objects_list = []
      target = None
      best_dist = float("inf")

      for t in tracks:
          p_world = np.asarray(t["pos_world"], dtype=np.float32)
          cls_name = t["class"]
          conf = float(t["conf"])
          bbox = t["bbox"]
          track_id = int(t["track_id"])

          grasp_pos = np.asarray(t.get("grasp_pos_world", p_world), dtype=np.float32).copy()
          if "grasp_pos_world" not in t:
              grasp_pos[2] -= GRASP_DEPTH_OFFSET

          grasp_q = self._compute_grasp_quat(cls_name, p_world)
          dist = float(np.linalg.norm(p_world[:2] - self.robot_pos[:2]))
          yaw_rel = float(np.arctan2(p_world[1] - self.robot_pos[1],
                                     p_world[0] - self.robot_pos[0]) - self.robot_yaw)
          in_bin = self._check_in_bin(p_world)
          if in_bin:
              self.in_bin_ids.add(track_id)

          size = OBJECT_SIZES.get(cls_name, DEFAULT_OBJECT_SIZE)
          obj_info = {
              "id": track_id,
              "class": cls_name,
              "conf": conf,
              "pos_world": p_world.tolist(),
              "pos_robot": self._world_to_robot(p_world).tolist(),
              "grasp_pos_world": grasp_pos.tolist(),
              "grasp_quat_world": grasp_q.tolist(),
              "dist_to_robot": dist,
              "yaw_rel": yaw_rel,
              "in_bin": in_bin,
              "size_world": [size["lx"], size["ly"], size["lz"]],
              "bbox": bbox,
              "source": t.get("source", "rgbd_cluster"),
          }
          objects_list.append(obj_info)

          if not in_bin and dist < best_dist:
              best_dist = dist
              target = obj_info

      objects_list.sort(key=lambda x: x["dist_to_robot"])
      gripper = self._read_gripper(p_np)
      if target is not None and gripper["is_holding"]:
          gripper["held_object_id"] = target["id"]

      bin_dist = float(np.linalg.norm(BIN_CENTER[:2] - self.robot_pos[:2]))
      bin_yaw_rel = float(np.arctan2(BIN_CENTER[1] - self.robot_pos[1],
                                     BIN_CENTER[0] - self.robot_pos[0]) - self.robot_yaw)

      inside_bin = len(self.in_bin_ids)
      obstacles = [
          {"id": o["id"], "pos_world": o["pos_world"], "radius": 0.25}
          for o in objects_list
          if target is None or o["id"] != target["id"]
      ]

      return {
          "target": target,
          "objects_remaining": [
              {"id": o["id"], "class": o["class"], "dist": o["dist_to_robot"],
               "pos_world": o["pos_world"], "in_bin": o["in_bin"]}
              for o in objects_list
          ],
          "objects_detailed": objects_list,
          "gripper": gripper,
          "bin": {
              "center_world": BIN_CENTER.tolist(),
              "radius": BIN_RADIUS,
              "drop_height": BIN_Z_MAX,
              "dist_to_robot": bin_dist,
              "yaw_rel": bin_yaw_rel,
          },
          "obstacles": obstacles,
          "progress": {
              "total": TOTAL_OBJECTS,
              "inside_bin": inside_bin,
              "remaining": TOTAL_OBJECTS - inside_bin,
          },
          "robot": {
              "pos_world": self.robot_pos.tolist(),
              "yaw": float(self.robot_yaw),
          },
      }


def _demo_synthetic():
      """无仿真时用假数据跑通接口"""
      print("=== RgbdPerceptionPipeline synthetic demo ===")
      pipeline = RgbdPerceptionPipeline()
      pipeline.robot_pos = ROBOT_INIT_POS.copy()
      pipeline.robot_yaw = 0.0

      h, w = HEAD_CAM["height"], HEAD_CAM["width"]
      rgb = np.full((h, w, 3), 160, dtype=np.uint8)
      rgb[h // 2:, :] = (90, 90, 90)
      depth = np.ones((h, w), dtype=np.float32) * 4.0

      # 假物体: 画面中心一块更近的深度
      cy, cx = h // 2 + 40, w // 2
      for dy in range(-25, 25):
          for dx in range(-20, 20):
              y, x = cy + dy, cx + dx
              if 0 <= y < h and 0 <= x < w:
                  depth[y, x] = 2.2
                  rgb[y, x] = (40, 180, 220)  # 黄偏色

      obs = {
          "image": {
              "head_rgb": np.expand_dims(rgb, 0),
              "head_depth": np.expand_dims(depth[..., None], 0),
          },
          "proprio": np.zeros((1, 72), dtype=np.float32),
      }
      out = pipeline.process(obs)
      print(f"  detected: {len(out['objects_detailed'])}")
      if out["target"]:
          t = out["target"]
          print(f"  target: {t['class']} conf={t['conf']:.2f} pos={t['pos_world']}")
      else:
          print("  target: None (tune DEPTH_/WORLD_Z_/MIN_CLUSTER_PIXELS in sim)")
      return out


if __name__ == "__main__":
      _demo_synthetic()
