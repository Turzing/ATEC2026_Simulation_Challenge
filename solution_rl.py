"""
Task B 操作层 — 感知 taskb_perception + 本文件全自动: 找→走→抓→送→丢→再找

状态机
  SEARCH → APPROACH → GRASP/GRASP_ARM → CARRY → DROP → SEARCH

运行
  python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --enable_cameras
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import torch

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

try:
    from arm_grasp import ArmGraspController
except ImportError:
    from demo.arm_grasp import ArmGraspController

_REPO_ROOT = os.path.dirname(_DEMO_DIR)
_PERCEPTION_DIR = os.path.join(_REPO_ROOT, "taskb_perception")
if os.path.isdir(_PERCEPTION_DIR) and _PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_DIR)

from config import BIN_CENTER, BIN_RADIUS  # noqa: E402
from rgbd_pure_dual_pipeline import RgbdPureDualPipeline  # noqa: E402


class AlgSolution:
    ACTION_SCALE = 0.5
    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
        "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
    ]
    EE_BODY_NAME = "gripper_base"

    GRASP_DEPTH_M = 1.10
    WARMUP_STEPS = 90              # 开局先站稳再动 (~1.8s)
    SEARCH_VX = 0.0                # 搜索原地转，不边转边走
    SEARCH_WZ = 0.22
    YAW_TURN_THRESH = 0.32
    NAV_WZ_GAIN = 0.85
    NAV_WZ_MAX = 0.38
    NAV_VX_MIN = 0.10
    NAV_VX_MAX = 0.42
    BIN_ARRIVE_M = 1.05

    DROP_OVER_Z = 0.48
    DROP_RELEASE_Z = 0.20
    DROP_RETREAT_Z = 0.58
    DROP_OPEN_STEPS = 40

    def __init__(self, env=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_path = self._resolve_policy_path()
        self.policy = self._load_leg_policy(policy_path)
        self._leg_mode = "rl" if self.policy is not None else "scripted"

        self.perception = RgbdPureDualPipeline()
        self.dt = 0.02

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=self.device, dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=self.device, dtype=torch.float32,
        ).view(1, -1)

        self._velocity_commands = torch.tensor(
            [[0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32,
        )
        self._arm_default_action = torch.zeros((1, 8), device=self.device, dtype=torch.float32)

        self._task_state = "SEARCH"
        self._step = 0
        self._arm_grasp: ArmGraspController | None = None
        self._camera_follow_enabled = True

        self._hold_carry_pose = False
        self._carry_arm_jpos: torch.Tensor | None = None
        self._carry_gripper: torch.Tensor | None = None

        self._drop_phase = ""
        self._drop_wait = 0
        self._objects_dropped = 0
        self._post_drop_stand_until = 0

        print(
            f"[TaskB-RL] full loop: search→grasp→carry→drop | bin={BIN_CENTER.tolist()} "
            f"r={BIN_RADIUS} leg={self._leg_mode}",
            flush=True,
        )
        if self._leg_mode == "scripted":
            print(
                "[TaskB-RL] *** leg=scripted 站不稳! 请确保 demo/policy.pt 存在且非 LFS 指针 ***",
                flush=True,
            )

    def _resolve_policy_path(self) -> str:
        candidates = [
            os.path.join(_DEMO_DIR, "policy.pt"),
            os.path.join(_REPO_ROOT, "atec_robot_model", "baseline", "unitree_b2_flat", "policy.pt"),
            os.path.join(_PERCEPTION_DIR, "policy.pt"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _load_leg_policy(self, policy_path: str):
        if not os.path.isfile(policy_path):
            print(
                f"[TaskB-RL] WARN: 无 demo/policy.pt，腿用简易步态（请拷官方 policy.pt）",
                flush=True,
            )
            return None
        size = os.path.getsize(policy_path)
        try:
            with open(policy_path, "rb") as f:
                head = f.read(256)
            if b"git-lfs" in head or head.startswith(b"version https://"):
                print(
                    "[TaskB-RL] ERROR: demo/policy.pt 是 Git LFS 指针，未下载真实文件。\n"
                    "  请在工程根执行: git lfs pull\n"
                    "  或从官方/师姐处拷完整 demo/policy.pt（通常几百 KB 以上）",
                    flush=True,
                )
                return None
        except OSError:
            pass
        try:
            pol = torch.jit.load(policy_path, map_location=self.device)
            pol.eval()
            print(f"[TaskB-RL] loaded policy.pt ({size // 1024} KB)", flush=True)
            return pol
        except Exception as exc:
            hint = "文件可能损坏/拷了一半。请重新拷官方 demo/policy.pt"
            try:
                obj = torch.load(policy_path, map_location=self.device, weights_only=False)
                if isinstance(obj, dict):
                    hint = (
                        "这是训练 checkpoint (torch.save)，不是 JIT 导出的 policy.pt。\n"
                        "  请让师姐用 scripts/rsl_rl/play.py 导出 JIT:\n"
                        "    python scripts/rsl_rl/play.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-v0\n"
                        "  然后拷 exported/policy.pt 到 demo/\n"
                        "  或直接用官方: atec_robot_model/baseline/unitree_b2_flat/policy.pt"
                    )
            except Exception:
                if "constants.pkl" in str(exc):
                    hint = (
                        "文件不是完整 TorchScript JIT (缺 constants.pkl)。\n"
                        "  常见原因: 拷了训练 checkpoint 而非 play.py 导出的 JIT。\n"
                        "  请用官方 baseline 或 play.py → exported/policy.pt"
                    )
            print(
                f"[TaskB-RL] ERROR: policy.pt 无法加载 ({exc})\n"
                f"  路径: {policy_path}  大小: {size} bytes\n"
                f"  {hint}",
                flush=True,
            )
            return None

    def _scripted_leg_action(self, action_dim: int) -> torch.Tensor:
        """policy.pt 不可用: 用简易 trot；静止也不能全零(会倒)."""
        vx = float(self._velocity_commands[0, 0].item())
        wz = float(self._velocity_commands[0, 2].item())
        a = torch.zeros(1, action_dim, device=self.device, dtype=torch.float32)

        t = self._step * self.dt
        amp = 0.35 if vx < 0.05 and abs(wz) < 0.05 else 1.0
        s1, s2 = amp * math.sin(t * 3.0), amp * math.sin(t * 3.0 + math.pi)

        def leg(ih: int, it: int, ic: int, s: float) -> None:
            a[0, ih] = 0.25 * s
            a[0, it] = 0.55 + 0.45 * s
            a[0, ic] = -1.35 - 0.35 * abs(s)

        leg(0, 1, 2, s1)
        leg(3, 4, 5, s2)
        leg(6, 7, 8, s2)
        leg(9, 10, 11, s1)

        if wz > 0.08:
            a[0, 0] += 0.35
            a[0, 6] += 0.35
            a[0, 3] -= 0.35
            a[0, 9] -= 0.35
        elif wz < -0.08:
            a[0, 0] -= 0.35
            a[0, 6] -= 0.35
            a[0, 3] += 0.35
            a[0, 9] += 0.35
        if vx < 0.08:
            a[:, :12] *= 0.5
        a[:, self.arm_joint_indices] = self._arm_default_action
        return a

    @property
    def camera_follow_enabled(self) -> bool:
        return self._camera_follow_enabled

    @camera_follow_enabled.setter
    def camera_follow_enabled(self, value: bool) -> None:
        self._camera_follow_enabled = bool(value)

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def reset(self) -> None:
        self.perception.reset()
        self._task_state = "SEARCH"
        self._step = 0
        self._hold_carry_pose = False
        self._carry_arm_jpos = None
        self._carry_gripper = None
        self._drop_phase = ""
        self._drop_wait = 0
        self._post_drop_stand_until = 0
        if self._arm_grasp is not None:
            self._arm_grasp.reset()

    def _scene(self):
        if self.env is None:
            return None
        env_u = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(env_u, "scene"):
            return env_u.scene
        if hasattr(env_u, "_env") and hasattr(env_u._env, "scene"):
            return env_u._env.scene
        return None

    def _robot(self):
        scene = self._scene()
        if scene is None:
            return None
        try:
            robot = scene["robot"]
        except Exception:
            return None
        if isinstance(robot, (list, tuple)):
            return robot[0]
        return robot

    def _ensure_arm_controller(self) -> ArmGraspController | None:
        if self._arm_grasp is not None:
            return self._arm_grasp
        robot = self._robot()
        if robot is None:
            return None
        try:
            self._arm_grasp = ArmGraspController(
                robot=robot,
                device=self.device,
                arm_joint_names=self.ARM_JOINT_NAMES[:6],
                gripper_joint_names=self.ARM_JOINT_NAMES[6:],
                ee_body_name=self.EE_BODY_NAME,
                action_scale=self.ACTION_SCALE,
            )
        except Exception as exc:
            print(f"[TaskB-RL] ArmGraspController init failed: {exc}", flush=True)
            self._arm_grasp = None
        return self._arm_grasp

    def _set_velocity_commands(self, vx: float, vy: float, wz: float) -> None:
        self._velocity_commands = torch.tensor(
            [[float(vx), float(vy), float(wz)]],
            device=self.device,
            dtype=torch.float32,
        )

    def _extract_policy_obs(self, obs: dict, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx: idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx: idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx: idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx: idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx: idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        num_envs = proprio.shape[0]
        cmd = self._velocity_commands.to(dtype=proprio.dtype)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                cmd,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _leg_action(self, obs: dict, action_dim: int) -> torch.Tensor:
        if self.policy is None:
            return self._scripted_leg_action(action_dim)

        policy_obs = self._extract_policy_obs(obs, action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self._arm_default_action.repeat(num_envs, 1)
        return action_env

    @staticmethod
    def _nav_depth(nav: dict) -> float:
        for key in ("nav_depth_m", "depth_m", "dist_to_robot"):
            val = nav.get(key)
            if val is not None:
                try:
                    d = float(val)
                    if d > 0.05:
                        return d
                except (TypeError, ValueError):
                    pass
        return 99.0

    @staticmethod
    def _nav_yaw(nav: dict) -> float:
        for key in ("yaw_rel", "nav_yaw_rel"):
            val = nav.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _robot_xy_yaw(self, perc: dict) -> tuple[np.ndarray, float] | None:
        robot = perc.get("robot") or {}
        pos_w = robot.get("pos_world")
        if pos_w is None:
            return None
        rp = np.asarray(pos_w, dtype=np.float32)
        yaw = float(robot.get("yaw") or 0.0)
        return rp, yaw

    def _dist_to_bin(self, perc: dict) -> float:
        xy = self._robot_xy_yaw(perc)
        if xy is None:
            return 99.0
        rp, _ = xy
        return float(np.linalg.norm(rp[:2] - BIN_CENTER[:2]))

    def _at_bin(self, perc: dict) -> bool:
        return self._dist_to_bin(perc) < self.BIN_ARRIVE_M

    def _cmd_approach(self, nav: dict) -> tuple[float, float]:
        depth = self._nav_depth(nav)
        yaw = self._nav_yaw(nav)
        wz_cap = self.NAV_WZ_MAX
        if depth < self.GRASP_DEPTH_M:
            return 0.06, float(np.clip(yaw * self.NAV_WZ_GAIN, -wz_cap, wz_cap))
        if abs(yaw) > self.YAW_TURN_THRESH:
            return 0.0, float(np.clip(yaw * self.NAV_WZ_GAIN, -wz_cap, wz_cap))
        vx = self.NAV_VX_MIN + (self.NAV_VX_MAX - self.NAV_VX_MIN) * min(depth, 3.0) / 3.0
        wz = float(np.clip(yaw * self.NAV_WZ_GAIN * 0.7, -wz_cap * 0.6, wz_cap * 0.6))
        return float(vx), wz

    def _cmd_search(self) -> tuple[float, float]:
        return self.SEARCH_VX, self.SEARCH_WZ

    def _cmd_carry(self, perc: dict) -> tuple[float, float]:
        xy = self._robot_xy_yaw(perc)
        if xy is None:
            return 0.15, 0.0
        rp, yaw = xy
        delta = np.asarray(BIN_CENTER[:2], dtype=np.float32) - rp[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.BIN_ARRIVE_M:
            return 0.0, 0.0
        yaw_to_bin = float(math.atan2(delta[1], delta[0]))
        err = (yaw_to_bin - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > 0.35:
            return 0.10, float(np.clip(err * 0.9, -0.45, 0.45))
        vx = float(np.clip(0.12 + 0.25 * min(dist, 4.0) / 4.0, 0.12, 0.45))
        wz = float(np.clip(err * 0.6, -0.25, 0.25))
        return vx, wz

    def _choose_velocity(self, perc: dict) -> tuple[float, float, str]:
        if self._step < self.WARMUP_STEPS or self._step < self._post_drop_stand_until:
            return 0.0, 0.0, "STAND"

        arm = self._arm_grasp
        if arm is not None and arm.state not in ("IDLE", "DONE", "FAILED"):
            return 0.0, 0.0, "GRASP_ARM"

        if self._task_state == "DROP" or self._drop_phase:
            return 0.0, 0.0, "DROP"

        if self._task_state == "CARRY":
            if self._at_bin(perc):
                return 0.0, 0.0, "DROP"
            vx, wz = self._cmd_carry(perc)
            return min(vx, 0.32), wz, "CARRY"

        phase = str(perc.get("phase") or "approach")
        grasp = perc.get("target_grasp")
        nav = perc.get("target_nav")

        if phase == "grasp" and isinstance(grasp, dict) and grasp.get("grasp_pos_world"):
            return 0.0, 0.0, "GRASP"

        if isinstance(nav, dict):
            return *self._cmd_approach(nav), "APPROACH"

        return *self._cmd_search(), "SEARCH"

    def _is_standing_phase(self) -> bool:
        return self._step < self.WARMUP_STEPS or self._step < self._post_drop_stand_until

    def _update_grasp(self, perc: dict) -> None:
        if self._is_standing_phase():
            return
        if str(perc.get("phase") or "") != "grasp":
            return
        if self._hold_carry_pose or self._drop_phase:
            return
        grasp = perc.get("target_grasp")
        if not isinstance(grasp, dict) or grasp.get("grasp_pos_world") is None:
            return

        arm = self._ensure_arm_controller()
        if arm is None or arm.state != "IDLE":
            return

        obj_center = grasp.get("pos_world") or grasp["grasp_pos_world"]
        gq = grasp.get("grasp_quat_world")
        arm.start_grasp(grasp, np.asarray(obj_center, dtype=np.float32), current_ee_quat_w=gq)
        self._task_state = "GRASP"
        print(
            f"[TaskB-RL] start grasp id={grasp.get('id')} class={grasp.get('class', '?')}",
            flush=True,
        )

    def _lock_carry_pose(self, arm: ArmGraspController) -> None:
        """抓取成功后保持抬升姿态 + 夹爪闭合，搬运时不会掉."""
        arm.close_gripper()
        if arm.desired_arm_joint_pos is not None:
            self._carry_arm_jpos = arm.desired_arm_joint_pos.detach().clone()
        else:
            robot = self._robot()
            if robot is not None:
                self._carry_arm_jpos = robot.data.joint_pos[:, arm.arm_joint_ids].clone()
        self._carry_gripper = arm.gripper_close_pos.clone()
        self._hold_carry_pose = True
        arm.state = "IDLE"

    def _step_arm_grasp(self, action_env: torch.Tensor) -> torch.Tensor:
        arm = self._arm_grasp
        robot = self._robot()
        if arm is None or robot is None or arm.state == "IDLE":
            return action_env

        scene = self._scene()
        done, success = arm.step(robot, scene, self.dt)
        action_env = arm.apply_to_action_tensor(action_env, robot)

        if done:
            if success:
                self._lock_carry_pose(arm)
                self._task_state = "CARRY"
                print("[TaskB-RL] grasp OK → CARRY to bin", flush=True)
            else:
                print(f"[TaskB-RL] grasp failed: {arm.failure_reason}", flush=True)
                self._task_state = "SEARCH"
                arm.reset()
                self._hold_carry_pose = False
        return action_env

    def _apply_carry_arm(self, action_env: torch.Tensor) -> torch.Tensor:
        if not self._hold_carry_pose:
            return action_env
        arm = self._ensure_arm_controller()
        robot = self._robot()
        if arm is None or robot is None:
            return action_env
        if self._carry_arm_jpos is not None:
            arm.desired_arm_joint_pos = self._carry_arm_jpos.to(
                device=action_env.device, dtype=action_env.dtype,
            )
        if self._carry_gripper is not None:
            arm.desired_gripper_joint_pos = self._carry_gripper.to(
                device=action_env.device, dtype=action_env.dtype,
            )
        return arm.apply_to_action_tensor(action_env, robot)

    @staticmethod
    def _bin_pos(z: float) -> np.ndarray:
        return np.array([float(BIN_CENTER[0]), float(BIN_CENTER[1]), float(z)], dtype=np.float32)

    def _step_drop(self, action_env: torch.Tensor) -> torch.Tensor:
        arm = self._ensure_arm_controller()
        robot = self._robot()
        if arm is None or robot is None:
            self._finish_drop()
            return action_env

        if not self._drop_phase:
            self._drop_phase = "MOVE_OVER"
            self._drop_wait = 0
            print(f"[TaskB-RL] DROP start @ bin dist={self.BIN_ARRIVE_M}m", flush=True)

        ee_pos, _ = arm.get_ee_pose()

        if self._drop_phase == "MOVE_OVER":
            arm.close_gripper()
            tgt = self._bin_pos(self.DROP_OVER_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "LOWER"
                print("[TaskB-RL] DROP: over bin → lower", flush=True)

        elif self._drop_phase == "LOWER":
            arm.close_gripper()
            tgt = self._bin_pos(self.DROP_RELEASE_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "OPEN"
                self._drop_wait = 0
                print("[TaskB-RL] DROP: release", flush=True)

        elif self._drop_phase == "OPEN":
            arm.open_gripper()
            self._drop_wait += 1
            if self._drop_wait >= self.DROP_OPEN_STEPS:
                self._drop_phase = "RETREAT"

        elif self._drop_phase == "RETREAT":
            arm.open_gripper()
            tgt = self._bin_pos(self.DROP_RETREAT_Z)
            arm.move_ee_to_pose(tgt, None)
            if arm.ee_reached(ee_pos, tgt):
                self._drop_phase = "DONE"

        elif self._drop_phase == "DONE":
            self._finish_drop()
            return action_env

        return arm.apply_to_action_tensor(action_env, robot)

    def _finish_drop(self) -> None:
        self._objects_dropped += 1
        self._drop_phase = ""
        self._drop_wait = 0
        self._hold_carry_pose = False
        self._carry_arm_jpos = None
        self._carry_gripper = None
        self._task_state = "SEARCH"
        if self._arm_grasp is not None:
            self._arm_grasp.reset()
        self._post_drop_stand_until = self._step + self.WARMUP_STEPS // 2
        print(
            f"[TaskB-RL] DROP done (total={self._objects_dropped}) → SEARCH next object",
            flush=True,
        )

    def _safe_perception(self, obs: dict) -> dict:
        try:
            return self.perception.process(obs, self.dt)
        except Exception as exc:
            if self._step % 60 == 0:
                print(f"[TaskB-RL] WARN: perception error (skip frame): {exc}", flush=True)
            return {
                "phase": "approach",
                "ee_objects": [],
                "target_nav": None,
                "target_grasp": None,
                "robot": {},
            }

    def _log_status(self, perc: dict, vx: float, wz: float, state: str) -> None:
        if self._step % 60 != 0:
            return
        nav = perc.get("target_nav") or {}
        nd = self._nav_depth(nav) if nav else 0.0
        print(
            f"[TaskB] step={self._step} state={state} drop={self._drop_phase or '-'} "
            f"perc={perc.get('phase')} ee={len(perc.get('ee_objects') or [])} "
            f"nav_d={nd:.2f} bin_d={self._dist_to_bin(perc):.2f} dropped={self._objects_dropped} "
            f"cmd=({vx:.2f},{wz:.2f})",
            flush=True,
        )

    def predicts(self, obs, current_score):
        if current_score > 1:
            return {"action": [], "giveup": True}

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        perc = self._safe_perception(obs)
        vx, wz, nav_state = self._choose_velocity(perc)
        if nav_state != "STAND":
            self._task_state = nav_state
        self._set_velocity_commands(vx, 0.0, wz)

        self._update_grasp(perc)
        action_env = self._leg_action(obs, action_dim)

        if self._task_state == "DROP" or self._drop_phase:
            action_env = self._step_drop(action_env)
        elif self._task_state == "CARRY":
            action_env = self._apply_carry_arm(action_env)
        elif self._arm_grasp is not None and self._arm_grasp.state not in ("IDLE", "DONE", "FAILED"):
            action_env = self._step_arm_grasp(action_env)

        self._log_status(perc, vx, wz, nav_state if nav_state == "STAND" else self._task_state)
        self._step += 1

        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}
