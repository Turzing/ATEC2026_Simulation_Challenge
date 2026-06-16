import math
import os
import sys
import time
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCEPTION_DIR = os.path.join(REPO_ROOT, "taskb_perception")
if os.path.isdir(PERCEPTION_DIR) and PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, PERCEPTION_DIR)

# 强制导入 taskb_perception 模块，不使用回退
from config import BIN_CENTER, ROBOT_INIT_POS, ROBOT_INIT_YAW
from rgbd_pure_dual_pipeline import RgbdPureDualPipeline
from rgbd_utils import depth_to_vis, parse_ee_rgbd, parse_head_rgbd, depth_stats

try:
    from .solution_gt import ArmGraspController, LegPostureController
except Exception:
    from solution_gt import ArmGraspController, LegPostureController

from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG

B2_PIPER_LEG_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
B2_PIPER_ARM_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.arm_joint_names)
B2_PIPER_TOTAL_JOINT_NAMES = list(UNITREE_B2_PIPER_CFG.joint_names)


class B2PiperActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class AlgSolution:
    def __init__(self, env=None):
        self.env = env
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.camera_follow_enabled = os.getenv("ATEC_TASKB_CAMERA_FOLLOW", "1").lower() in {"1", "true", "yes", "on"}
        self.nav_debug = os.getenv("ATEC_TASKB_NAV_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
        self.nav_debug_every = max(1, int(os.getenv("ATEC_TASKB_NAV_DEBUG_EVERY", "25")))
        self.target_mode = os.getenv("ATEC_TASKB_TARGET_MODE", "object").lower()
        self._nav_debug_step = 0
        self._step_count = 0
        self._perception_error_printed = False
        self._odom_pos = np.asarray(ROBOT_INIT_POS, dtype=np.float32).copy()
        self._odom_yaw = float(ROBOT_INIT_YAW)
        self._last_base_cmd = np.zeros(3, dtype=np.float32)
        self._last_nav_info: dict[str, Any] = {}
        self._last_perception_output: dict[str, Any] | None = None
        self._locked_target: dict[str, Any] | None = None
        self._pending_target: dict[str, Any] | None = None
        self._tracked_target: dict[str, Any] | None = None
        self._last_known_target_pos: list[float] | None = None
        self._target_lost_count = 0
        self._task_state = "APPROACH_OBJECT"
        self._pending_grasp_target: dict[str, Any] | None = None
        self._locked_goal_xy: np.ndarray | None = None
        self._locked_goal_yaw: float | None = None
        self._locked_goal_target_id: Any | None = None
        self._locked_target_world: list[float] | None = None
        self._pregrasp_stall_steps = 0
        self._pregrasp_last_robot_xy: np.ndarray | None = None
        self._nav_heading_error_f: float | None = None
        self._nav_turn_sign: int = 0
        self._nav_turn_sign_hold = 0
        self._fuse_lock_key: tuple[Any, Any] | None = None
        self._fuse_pos_world: np.ndarray | None = None
        self._ee_only_no_head_frames = 0
        self._nav_stall_turn_rad = 0.0
        self._nav_stall_dist_start: float | None = None
        self._nav_ignore_perc_until_head = False
        self.ee_only_unlock_frames = max(12, int(os.getenv("ATEC_TASKB_EE_ONLY_UNLOCK_FRAMES", "18")))
        self.nav_stall_turn_rad = float(os.getenv("ATEC_TASKB_NAV_STALL_TURN_RAD", "1.15"))
        self.nav_stall_min_dist_m = float(os.getenv("ATEC_TASKB_NAV_STALL_MIN_DIST", "2.0"))
        self._release_step_count = 0
        self._arm_grasp_controller = None
        self._arm_controller_init_failed = False
        self._leg_posture_controller = LegPostureController(
            leg_joint_names=list(B2_PIPER_LEG_JOINT_NAMES),
            crouch_drop_height=float(os.getenv("ATEC_TASKB_CROUCH_DROP_HEIGHT", "0.10")),
            crouch_duration=float(os.getenv("ATEC_TASKB_CROUCH_DURATION", "2.0")),
            stand_up_duration=float(os.getenv("ATEC_TASKB_STAND_UP_DURATION", "2.0")),
            foot_pos_tol=float(os.getenv("ATEC_TASKB_CROUCH_FOOT_TOL", "0.1")),
            body_height_tol=float(os.getenv("ATEC_TASKB_CROUCH_HEIGHT_TOL", "0.02")),
            ik_damping=float(os.getenv("ATEC_TASKB_CROUCH_IK_DAMPING", "0.1")),
            max_joint_step=float(os.getenv("ATEC_TASKB_CROUCH_MAX_JOINT_STEP", "0.08")),
        )
        self._pending_grasp_status = None
        self.sit_down_actor = None
        self.sit_down_actor_obs_dim = None
        self.sit_down_min_steps = max(1, int(os.getenv("ATEC_TASKB_SIT_DOWN_MIN_STEPS", "30")))
        self.sit_down_stable_steps_required = max(1, int(os.getenv("ATEC_TASKB_SIT_DOWN_STABLE_STEPS", "15")))
        self.sit_down_roll_pitch_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_RP_THRESH", "0.18"))
        self.sit_down_height_vel_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_ZVEL_THRESH", "0.15"))
        self.sit_down_ang_vel_thresh = float(os.getenv("ATEC_TASKB_SIT_DOWN_ANGVEL_THRESH", "0.4"))
        self._sit_down_step_count = 0
        self._sit_down_stable_count = 0

        # 初始化日志系统
        self._log_file_path = self._init_logging()

        self.checkpoint_path = os.path.join(
            REPO_ROOT,
            "logs",
            "rsl_rl",
            "unitree_b2_piper_flat",
            "2026-06-02_14-40-32",
            "model_4999.pt",
        )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        # 强制初始化感知管道，不使用回退
        self.perception = RgbdPureDualPipeline()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        self.leg_joint_names = list(B2_PIPER_LEG_JOINT_NAMES)
        self.arm_joint_names = list(B2_PIPER_ARM_JOINT_NAMES)
        self.leg_action_dim = len(self.leg_joint_names)
        self.arm_action_dim = len(self.arm_joint_names)
        self.total_action_dim = len(B2_PIPER_TOTAL_JOINT_NAMES)

        actor_input_dim = state_dict["actor.0.weight"].shape[1]
        actor_output_dim = state_dict["actor.6.bias"].shape[0]
        self.actor = B2PiperActor(actor_input_dim, actor_output_dim).to(self.device)
        actor_state = {key: value for key, value in state_dict.items() if key.startswith("actor.")}
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()

        self.leg_action_scale = torch.tensor(
            [0.25 if "hip_joint" in name else 0.5 for name in self.leg_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.leg_action_scale_inv = torch.reciprocal(self.leg_action_scale)

        arm_default_pos = {
            "arm_joint1": 0.0,
            "arm_joint2": 2.13,
            "arm_joint3": -1.20,
            "arm_joint4": 0.0,
            "arm_joint5": -0.8,
            "arm_joint6": 0.0,
            "arm_joint7": 0.0,
            "arm_joint8": 0.0,
        }
        self.arm_default_action = torch.tensor(
            [arm_default_pos.get(name, 0.0) for name in self.arm_joint_names],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        if actor_input_dim != 45 or actor_output_dim != self.leg_action_dim:
            raise ValueError(
                f"Checkpoint shape mismatch: actor input={actor_input_dim}, output={actor_output_dim}, "
                f"expected input=45, output={self.leg_action_dim}"
            )

        self.stand_still_steps = max(0, int(float(os.getenv("ATEC_TASKB_STAND_SECONDS", "0.3")) / 0.02))
        self.stop_distance = float(os.getenv("ATEC_TASKB_STOP_DISTANCE", "0.9"))
        self.object_stop_distance = float(os.getenv("ATEC_TASKB_OBJECT_STOP_DISTANCE", "0.7"))
        self.stop_tolerance = float(os.getenv("ATEC_TASKB_STOP_TOLERANCE", "0.2"))
        self.object_stop_tolerance = float(os.getenv("ATEC_TASKB_OBJECT_STOP_TOLERANCE", "0.15"))
        self.object_lateral_tolerance = float(
            os.getenv("ATEC_TASKB_OBJECT_LATERAL_TOLERANCE", str(self.object_stop_tolerance))
        )
        self.object_forward_tolerance = float(
            os.getenv("ATEC_TASKB_OBJECT_FORWARD_TOLERANCE", str(self.object_stop_tolerance))
        )
        self.object_yaw_tolerance = float(os.getenv("ATEC_TASKB_OBJECT_YAW_TOLERANCE", "0.15"))
        self.slow_down_radius = float(os.getenv("ATEC_TASKB_SLOW_DOWN_RADIUS", "1.5"))
        self.max_lin_vel = float(os.getenv("ATEC_TASKB_MAX_LIN_VEL", "0.8"))
        self.max_lat_vel = float(os.getenv("ATEC_TASKB_MAX_LAT_VEL", "0.4"))
        self.max_ang_vel = float(os.getenv("ATEC_TASKB_MAX_ANG_VEL", "0.6"))
        self.heading_kp = float(os.getenv("ATEC_TASKB_HEADING_KP", "1.2"))
        self.min_heading_lin_scale = float(os.getenv("ATEC_TASKB_MIN_HEADING_LIN_SCALE", "0.25"))
        self.max_target_distance = float(os.getenv("ATEC_TASKB_MAX_TARGET_DISTANCE", "6.0"))
        self.min_target_distance = float(os.getenv("ATEC_TASKB_MIN_TARGET_DISTANCE", "0.15"))
        self.min_target_conf = float(os.getenv("ATEC_TASKB_MIN_TARGET_CONF", "0.0"))
        self.target_reacquire_radius = float(os.getenv("ATEC_TASKB_TARGET_REACQUIRE_RADIUS", "0.45"))
        self.target_lock_timeout_steps = max(1, int(os.getenv("ATEC_TASKB_TARGET_LOCK_TIMEOUT_STEPS", "45")))
        self.target_confirm_steps = max(1, int(os.getenv("ATEC_TASKB_TARGET_CONFIRM_STEPS", "5")))
        self.target_head_confirm_steps = max(1, int(os.getenv("ATEC_TASKB_TARGET_HEAD_CONFIRM_STEPS", "1")))
        self.target_pending_timeout_steps = max(1, int(os.getenv("ATEC_TASKB_TARGET_PENDING_TIMEOUT_STEPS", "12")))
        self.target_far_match_radius = float(os.getenv("ATEC_TASKB_TARGET_FAR_MATCH_RADIUS", "0.60"))
        self.target_near_match_radius = float(os.getenv("ATEC_TASKB_TARGET_NEAR_MATCH_RADIUS", "0.25"))
        self.target_pending_match_radius = float(os.getenv("ATEC_TASKB_TARGET_PENDING_MATCH_RADIUS", "0.35"))
        self.target_ema_alpha_ee = float(os.getenv("ATEC_TASKB_TARGET_EMA_ALPHA_EE", "0.25"))
        self.target_ema_alpha_head = float(os.getenv("ATEC_TASKB_TARGET_EMA_ALPHA_HEAD", "0.45"))
        self.target_near_range = float(os.getenv("ATEC_TASKB_TARGET_NEAR_RANGE", "1.80"))
        self.target_freeze_distance = float(os.getenv("ATEC_TASKB_TARGET_FREEZE_DISTANCE", "0.95"))
        self.target_relock_disagreement = float(os.getenv("ATEC_TASKB_TARGET_RELOCK_DISAGREEMENT", "0.35"))
        self.ee_track_max_distance = float(os.getenv("ATEC_TASKB_EE_TRACK_MAX_DISTANCE", "2.15"))
        self.ee_search_fallback_distance = float(os.getenv("ATEC_TASKB_EE_SEARCH_FALLBACK_DISTANCE", "2.90"))
        self.track_jump_reject_m = float(os.getenv("ATEC_TASKB_TRACK_JUMP_REJECT_M", "0.45"))
        self.nav_heading_ema_alpha = float(os.getenv("ATEC_TASKB_NAV_HEADING_EMA", "0.28"))
        self.nav_turn_sign_hold_steps = max(3, int(os.getenv("ATEC_TASKB_TURN_SIGN_HOLD_STEPS", "8")))
        self.pregrasp_stall_steps = max(10, int(os.getenv("ATEC_TASKB_PREGRASP_STALL_STEPS", "35")))
        self.turn_then_go_yaw_threshold = float(os.getenv("ATEC_TASKB_TURN_THEN_GO_YAW_THRESHOLD", "0.30"))
        self.turn_then_go_heading_hold = float(os.getenv("ATEC_TASKB_TURN_THEN_GO_HEADING_HOLD", "0.18"))
        self.dynamic_stop_distance_far = float(os.getenv("ATEC_TASKB_DYNAMIC_STOP_DISTANCE_FAR", "1.15"))
        self.dynamic_stop_distance_near = float(os.getenv("ATEC_TASKB_DYNAMIC_STOP_DISTANCE_NEAR", "0.82"))
        self.dynamic_lateral_gain = float(os.getenv("ATEC_TASKB_DYNAMIC_LATERAL_GAIN", "0.60"))
        self.search_yaw_rate = float(os.getenv("ATEC_TASKB_SEARCH_YAW_RATE", "0.35"))
        self.grasp_start_depth = float(os.getenv("ATEC_TASKB_GRASP_START_DEPTH", "1.25"))
        self.bin_drop_radius = float(os.getenv("ATEC_TASKB_BIN_DROP_RADIUS", "1.0"))
        self.release_steps = max(1, int(os.getenv("ATEC_TASKB_RELEASE_STEPS", "25")))
        self.default_bin_center = np.asarray(BIN_CENTER, dtype=np.float32)
        self.dt = 0.02
        self.arm_action_scale = 0.5
        self.arm_ik_joint_names = list(self.arm_joint_names[:6])
        self.gripper_joint_names = list(self.arm_joint_names[6:])

        self.show_rgb = os.getenv("ATEC_SHOW_RGB", "1").lower() in {"1", "true", "yes", "on"}
        self.rgb_debug_every = max(1, int(os.getenv("ATEC_SHOW_RGB_EVERY", "50")))
        self._rgb_debug_failed = False
        self._rgb_debug_warned = False

        self.save_rgb = os.getenv("ATEC_SAVE_RGB", "1").lower() in {"1", "true", "yes", "on"}
        self.save_rgb_dir = os.path.join(REPO_ROOT, "logs", "rgb_frames")
        os.makedirs(self.save_rgb_dir, exist_ok=True)
        self._rgb_save_warned = False

        self._load_sit_down_actor_model()

    def _load_sit_down_actor_model(self) -> None:
        """加载 sit_down.pt 策略模型。"""
        checkpoint_path = os.path.join(REPO_ROOT, "demo", "sit_down.pt")
        if not os.path.exists(checkpoint_path):
            self._log(f"[TaskB-SIT] sit-down checkpoint not found: {checkpoint_path}")
            self.sit_down_actor = None
            self.sit_down_actor_obs_dim = None
            return
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            actor_input_dim = state_dict["actor.0.weight"].shape[1]
            actor_output_dim = state_dict["actor.6.bias"].shape[0]
            self.sit_down_actor = B2PiperActor(actor_input_dim, actor_output_dim).to(self.device)
            actor_state = {key: value for key, value in state_dict.items() if key.startswith("actor.")}
            self.sit_down_actor.load_state_dict(actor_state, strict=True)
            self.sit_down_actor.eval()
            self.sit_down_actor_obs_dim = actor_input_dim
            self._log(f"[TaskB-SIT] sit-down actor loaded: input={actor_input_dim}, output={actor_output_dim}")
        except Exception as exc:
            self._log(f"[TaskB-SIT] Failed to load sit-down actor model: {exc}")
            self.sit_down_actor = None
            self.sit_down_actor_obs_dim = None

    def _generate_sit_down_action_tensor(self, obs) -> torch.Tensor:
        zero_cmd = np.zeros(3, dtype=np.float32)
        robot = self._get_robot()
        if self.sit_down_actor is None:
            # 无 sit-down actor 时，回退到 LegPostureController.step() 的 IK 方案
            return self._generate_control_action_tensor(obs, zero_cmd, robot)
        try:
            policy_obs = self._extract_policy_obs(obs, zero_cmd, obs_dim=self.sit_down_actor_obs_dim)
            with torch.inference_mode():
                action_train = self.sit_down_actor(policy_obs)
            if action_train.ndim == 1:
                action_train = action_train.unsqueeze(0)
            return self._map_policy_action_to_env_action(action_train)
        except Exception as exc:
            self._log(f"[TaskB-SIT] sit-down actor inference failed: {exc}")
            return self._generate_control_action_tensor(obs, zero_cmd, robot)

    def _reset_sit_down_tracking(self) -> None:
        self._sit_down_step_count = 0
        self._sit_down_stable_count = 0

    def _is_sit_down_stable(self, robot) -> bool:
        if robot is None or not hasattr(robot, "data"):
            return False
        try:
            quat = robot.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (w * y - z * x)
            pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
            lin_vel_z = torch.abs(robot.data.root_lin_vel_b[:, 2])
            ang_vel_xy = torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)
            stable = (
                (torch.abs(roll) <= self.sit_down_roll_pitch_thresh)
                & (torch.abs(pitch) <= self.sit_down_roll_pitch_thresh)
                & (lin_vel_z <= self.sit_down_height_vel_thresh)
                & (ang_vel_xy <= self.sit_down_ang_vel_thresh)
            )
            return bool(torch.all(stable))
        except Exception:
            return False

    def _init_logging(self) -> str:
        log_dir = os.path.join(REPO_ROOT, "logs", "solution_rl")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(log_dir, f"solution_rl_{timestamp}.log")
        print(f"Log file: {log_file_path}", flush=True)
        return log_file_path

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with open(self._log_file_path, "a") as f:
            f.write(log_line)

    def reset(self, **_: Any) -> None:
        self._nav_debug_step = 0
        self._step_count = 0
        self._perception_error_printed = False
        self._odom_pos = np.asarray(ROBOT_INIT_POS, dtype=np.float32).copy()
        self._odom_yaw = float(ROBOT_INIT_YAW)
        self._last_base_cmd = np.zeros(3, dtype=np.float32)
        self._last_policy_action = np.zeros(3, dtype=np.float32)
        self._last_nav_info = {}
        self._last_perception_output = None
        self._locked_target = None
        self._pending_target = None
        self._tracked_target = None
        self._last_known_target_pos = None
        self._target_lost_count = 0
        self._task_state = "APPROACH_OBJECT"
        self._pending_grasp_target = None
        self._locked_goal_xy = None
        self._locked_goal_yaw = None
        self._locked_goal_target_id = None
        self._locked_target_world = None
        self._release_step_count = 0
        self._pending_grasp_status = None
        self._reset_sit_down_tracking()
        if self._leg_posture_controller is not None:
            self._leg_posture_controller.state = "IDLE"
        if self._arm_grasp_controller is not None:
            self._arm_grasp_controller.reset()
        if self.perception is not None and hasattr(self.perception, "reset"):
            self.perception.reset()

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def _get_scene(self):
        if self.env is None:
            return None
        env_unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(env_unwrapped, "scene"):
            return env_unwrapped.scene
        if hasattr(env_unwrapped, "_env") and hasattr(env_unwrapped._env, "scene"):
            return env_unwrapped._env.scene
        return None

    def _get_robot(self):
        scene = self._get_scene()
        if scene is None:
            return None
        try:
            robot = scene["robot"]
        except Exception:
            return None
        if isinstance(robot, (list, tuple)):
            robot = robot[0] if robot else None
        return robot

    def _ensure_arm_grasp_controller(self):
        if self._arm_grasp_controller is not None or self._arm_controller_init_failed or ArmGraspController is None:
            return self._arm_grasp_controller
        robot = self._get_robot()
        if robot is None:
            return None
        try:
            self._arm_grasp_controller = ArmGraspController(
                robot=robot,
                device=self.device,
                arm_joint_names=self.arm_ik_joint_names,
                gripper_joint_names=self.gripper_joint_names,
                ee_body_name="gripper_base",
                action_scale=self.arm_action_scale,
            )
        except Exception as exc:
            self._arm_controller_init_failed = True
            self._log(f"[TaskB-GRASP] controller init failed: {type(exc).__name__}: {exc}")
            self._arm_grasp_controller = None
        return self._arm_grasp_controller

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _safe_numpy(vector, fallback: np.ndarray) -> np.ndarray:
        base = fallback if vector is None else vector
        arr = np.asarray(base, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        return np.nan_to_num(arr[:3], nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_float(value: Any, fallback: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(scalar):
            return fallback
        return scalar

    @staticmethod
    def _tensor_to_numpy(data: Any) -> np.ndarray | None:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            data = data.detach()
            if data.device.type == "cuda":
                data = data.cpu()
            data = data.numpy()
        return np.asarray(data)

    @staticmethod
    def _get_nested(mapping: Any, *path: str) -> Any:
        current = mapping
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _world_to_robot_frame(
        self,
        point_world: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> np.ndarray:
        delta = self._safe_numpy(point_world, np.zeros(3, dtype=np.float32)) - self._safe_numpy(
            robot_pos_world, np.zeros(3, dtype=np.float32)
        )
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        return np.array(
            [
                cos_yaw * delta[0] + sin_yaw * delta[1],
                -sin_yaw * delta[0] + cos_yaw * delta[1],
                delta[2],
            ],
            dtype=np.float32,
        )

    def _robot_to_world_frame(
        self,
        point_robot: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> np.ndarray:
        point_robot = self._safe_numpy(point_robot, np.zeros(3, dtype=np.float32))
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        return self._safe_numpy(robot_pos_world, np.zeros(3, dtype=np.float32)) + np.array(
            [
                cos_yaw * point_robot[0] - sin_yaw * point_robot[1],
                sin_yaw * point_robot[0] + cos_yaw * point_robot[1],
                point_robot[2],
            ],
            dtype=np.float32,
        )

    def _update_local_odometry(self, obs: dict[str, Any]) -> dict[str, Any]:
        proprio = torch.as_tensor(obs["proprio"], device="cpu", dtype=torch.float32).squeeze(0).numpy()
        lin_vel = np.nan_to_num(proprio[0:3], nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate = float(np.nan_to_num(proprio[5], nan=0.0, posinf=0.0, neginf=0.0))

        self._odom_yaw = self._wrap_angle(self._odom_yaw + yaw_rate * self.dt)
        cos_yaw = math.cos(self._odom_yaw)
        sin_yaw = math.sin(self._odom_yaw)
        vel_world = np.array(
            [
                cos_yaw * lin_vel[0] - sin_yaw * lin_vel[1],
                sin_yaw * lin_vel[0] + cos_yaw * lin_vel[1],
            ],
            dtype=np.float32,
        )
        self._odom_pos[:2] += vel_world * self.dt
        self._odom_pos[2] = float(ROBOT_INIT_POS[2])

        return {
            "robot": {
                "pos_world": self._odom_pos.tolist(),
                "yaw": float(self._odom_yaw),
                "pose_source": "local_odometry",
            },
            "bin": {
                "center_world": self.default_bin_center.tolist(),
            },
        }

    def _resolve_robot_pose_world(
        self,
        obs: dict[str, Any],
        local_nav: dict[str, Any],
        perception_output: dict[str, Any] | None,
    ) -> tuple[np.ndarray, float, str]:
        # 直接使用环境中的 ground truth 机器人位姿，不回退
        gt_pose = self._get_ground_truth_robot_pose()
        if gt_pose is not None:
            return gt_pose
        
        # 如果获取失败，返回初始位置（不应该发生）
        self._log("[WARNING] Failed to get ground truth robot pose, using initial position")
        return np.asarray(ROBOT_INIT_POS, dtype=np.float32).copy(), float(ROBOT_INIT_YAW), "robot_init"

    def _get_ground_truth_robot_pose(self) -> tuple[np.ndarray, float, str] | None:
        """从环境中获取真实的机器人位姿（参考 solution_gt）"""
        if self.env is None:
            return None
        
        try:
            env_unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            scene = None
            
            if hasattr(env_unwrapped, 'scene'):
                scene = env_unwrapped.scene
            elif hasattr(env_unwrapped, '_env') and hasattr(env_unwrapped._env, 'scene'):
                scene = env_unwrapped._env.scene
            
            if scene is None:
                return None
            
            # 直接尝试获取 robot
            try:
                robot = scene['robot']
            except KeyError:
                return None
            
            # 处理 scene['robot'] 返回数组的情况
            if isinstance(robot, (list, tuple)):
                robot = robot[0]
            
            if not (hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w')):
                return None
            
            if not (hasattr(robot, 'data') and hasattr(robot.data, 'root_quat_w')):
                return None
            
            # 获取真实位置
            pos_world = robot.data.root_pos_w.cpu().numpy()[0]
            pos_world = np.array(pos_world, dtype=np.float32)
            pos_world[2] = 0.68  # 保持高度不变
            
            # 从四元数提取 yaw（Isaac Sim 使用 w, x, y, z 格式）
            quat = robot.data.root_quat_w.cpu().numpy()[0]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = float(math.atan2(siny_cosp, cosy_cosp))
            
            return pos_world, yaw, "ground_truth"
        except Exception:
            # 如果获取失败，返回 None 让其他方法处理
            return None

    def _scene_camera_pose(self, cam_obj) -> tuple[np.ndarray, np.ndarray] | None:
        """Isaac Lab Camera: pos_w + quat_w_ros (OpenCV +Z 光轴)."""
        if isinstance(cam_obj, (list, tuple)):
            cam_obj = cam_obj[0] if cam_obj else None
        if cam_obj is None or not hasattr(cam_obj, "data"):
            return None
        cam_data = cam_obj.data
        if hasattr(cam_data, "pos_w"):
            pos_world = np.array(cam_data.pos_w[0].cpu().numpy(), dtype=np.float32)
        elif hasattr(cam_data, "root_pos_w"):
            pos_world = np.array(cam_data.root_pos_w[0].cpu().numpy(), dtype=np.float32)
        else:
            return None
        if hasattr(cam_data, "quat_w_ros"):
            quat = cam_data.quat_w_ros[0].cpu().numpy()
        elif hasattr(cam_data, "root_quat_w"):
            quat = cam_data.root_quat_w[0].cpu().numpy()
        else:
            return None
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return pos_world, self._quat_to_rot_matrix(w, x, y, z)

    def _get_ground_truth_camera_pose(self, camera_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """从 scene 传感器读取相机真实世界位姿 (pos_w / quat_w_ros)."""
        if self.env is None:
            return None

        try:
            env_unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
            scene = None

            if hasattr(env_unwrapped, 'scene'):
                scene = env_unwrapped.scene
            elif hasattr(env_unwrapped, '_env') and hasattr(env_unwrapped._env, 'scene'):
                scene = env_unwrapped._env.scene

            if scene is None:
                return None

            if "head" in camera_name:
                candidate_names = ["head_camera", camera_name, "camera_head"]
            elif "ee" in camera_name:
                candidate_names = ["ee_camera", camera_name, "camera_ee"]
            else:
                candidate_names = [
                    camera_name,
                    f"{camera_name}_camera",
                    "head_camera",
                    "ee_camera",
                ]

            found = None
            for name in candidate_names:
                try:
                    cam_obj = scene[name]
                except (KeyError, IndexError):
                    continue
                pose = self._scene_camera_pose(cam_obj)
                if pose is not None:
                    found = (name, pose)
                    break

            if found is None:
                available = []
                for key in list(scene.keys())[:50]:
                    try:
                        val = scene[key]
                        if self._scene_camera_pose(val) is not None:
                            available.append(key)
                    except (KeyError, TypeError):
                        continue
                if self._step_count % 30 == 0:
                    self._log(f"[CAM] camera '{camera_name}' not found. Available sensors: {available}")
                return None

            name, (pos_world, rot_matrix) = found
            if self._step_count % 60 == 0:
                forward_world = rot_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                self._log(
                    f"[CAM] '{name}' pos=[{pos_world[0]:.2f}, {pos_world[1]:.2f}, {pos_world[2]:.2f}] "
                    f"fwd=[{forward_world[0]:+.3f}, {forward_world[1]:+.3f}, {forward_world[2]:+.3f}]"
                )
            return pos_world, rot_matrix
        except Exception as exc:
            if self._step_count % 30 == 0:
                self._log(f"[CAM] camera '{camera_name}' error: {type(exc).__name__}: {exc}")
            return None

    def _quat_to_rot_matrix(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """将四元数 (w, x, y, z) 转换为旋转矩阵"""
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        
        rot = np.array([
            [1 - 2*yy - 2*zz, 2*xy - 2*wz,   2*xz + 2*wy],
            [2*xy + 2*wz,   1 - 2*xx - 2*zz, 2*yz - 2*wx],
            [2*xz - 2*wy,   2*yz + 2*wx,   1 - 2*xx - 2*yy]
        ], dtype=np.float32)
        
        return rot

    def _prepare_pipeline_object(
        self,
        obj: dict[str, Any] | None,
        source_camera: str,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any] | None:
        if not isinstance(obj, dict):
            return None
        out = dict(obj)
        out["source_camera"] = source_camera
        out["camera"] = source_camera

        pos_robot = out.get("pos_robot")
        pos_world = out.get("pos_world")
        if pos_robot is not None:
            pos_robot_np = self._safe_numpy(pos_robot, np.zeros(3, dtype=np.float32))
        elif pos_world is not None:
            pos_robot_np = self._world_to_robot_frame(pos_world, robot_pos_world, robot_yaw)
            out["pos_robot"] = pos_robot_np.tolist()
        else:
            pos_robot_np = None

        if pos_robot_np is not None:
            pos_world_np = self._robot_to_world_frame(pos_robot_np, robot_pos_world, robot_yaw)
            out["pos_world"] = pos_world_np.tolist()
        elif pos_world is not None:
            pos_world_np = self._safe_numpy(pos_world, np.zeros(3, dtype=np.float32))
        else:
            pos_world_np = None

        if pos_robot_np is not None:
            out["dist_to_robot"] = self._safe_float(out.get("dist_to_robot"), float(np.linalg.norm(pos_robot_np[:2])))
            out["yaw_rel"] = self._safe_float(out.get("yaw_rel"), float(math.atan2(pos_robot_np[1], pos_robot_np[0])))
        return out

    def _collect_pipeline_objects(
        self,
        perception_output: dict[str, Any] | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> list[dict[str, Any]]:
        if not isinstance(perception_output, dict):
            return []
        collected: list[dict[str, Any]] = []
        for camera_key, source_camera in (("ee_objects", "ee"), ("head_objects", "head")):
            for obj in perception_output.get(camera_key, []) or []:
                prepared = self._prepare_pipeline_object(obj, source_camera, robot_pos_world, robot_yaw)
                if prepared is not None:
                    collected.append(prepared)
        return collected

    def _get_perception_output(
        self,
        obs: dict[str, Any],
        local_nav: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, np.ndarray, float, str]:
        perception_output = None
        
        # # 添加数据检查（调试用，已注释）
        # image_obs = obs.get("image", {})
        # ee_rgb_raw = image_obs.get("ee_rgb")
        # ee_depth_raw = image_obs.get("ee_depth")
        # 
        # if self._step_count % 30 == 0:
        #     if ee_rgb_raw is not None:
        #         if hasattr(ee_rgb_raw, 'shape'):
        #             print(f"[SolutionRL] Raw ee_rgb shape: {ee_rgb_raw.shape}", flush=True)
        #         else:
        #             print(f"[SolutionRL] Raw ee_rgb type: {type(ee_rgb_raw)}", flush=True)
        #     if ee_depth_raw is not None:
        #         if hasattr(ee_depth_raw, 'shape'):
        #             print(f"[SolutionRL] Raw ee_depth shape: {ee_depth_raw.shape}", flush=True)
        #         else:
        #             print(f"[SolutionRL] Raw ee_depth type: {type(ee_depth_raw)}", flush=True)
        
        gt_pose = self._get_ground_truth_robot_pose()
        gt_pos = gt_pose[0] if gt_pose is not None else None
        gt_yaw = gt_pose[1] if gt_pose is not None else None
        try:
            perception_output = self.perception.process(obs, dt=self.dt, gt_robot_pos=gt_pos, gt_robot_yaw=gt_yaw)
            
            # 使用相机真实位姿校正 EE 相机物体位置
            self._correct_ee_camera_objects(perception_output, gt_pos, gt_yaw)
            
        except Exception as exc:
            if not self._perception_error_printed:
                self._log(f"[TaskB-PERCEPTION] disabled after error: {type(exc).__name__}: {exc}")
                self._perception_error_printed = True
            perception_output = None
        robot_pos_world, robot_yaw, pose_source = self._resolve_robot_pose_world(obs, local_nav, perception_output)
        return perception_output, robot_pos_world, robot_yaw, pose_source

    def _correct_ee_camera_objects(self, perception_output: dict, robot_pos: np.ndarray, robot_yaw: float) -> None:
        """使用相机真实位姿校正物体位置 (EE + head)"""
        if perception_output is None:
            return

        # Camera intrinsic parameters
        EE_INTR = (458.29, 458.29, 320.0, 240.0)
        HEAD_INTR = (733.27, 733.27, 320.0, 240.0)

        corrected_any = False

        def _is_head_sourced(obj: dict) -> bool:
            if not isinstance(obj, dict):
                return False
            if obj.get("nav_from_head"):
                return True
            src = str(obj.get("source_camera") or obj.get("source") or "")
            return "head" in src.lower()

        # 获取 EE 相机真实位姿并校正 (跳过 head mirror，由 head 相机校正)
        ee_cam_pose = self._get_ground_truth_camera_pose("ee_camera")
        if ee_cam_pose is not None:
            ee_cam_pos_w, ee_cam_rot_w = ee_cam_pose
            ee_objects = perception_output.get("ee_objects", [])
            n_before = len(ee_objects)
            for obj in ee_objects:
                if obj.get("skip_camera_correction") or _is_head_sourced(obj):
                    continue
                self._correct_object_with_camera_pose(
                    obj, ee_cam_pos_w, ee_cam_rot_w, *EE_INTR, robot_pos, robot_yaw,
                )
            target_nav = perception_output.get("target_nav")
            if isinstance(target_nav, dict) and not target_nav.get("skip_camera_correction"):
                if not _is_head_sourced(target_nav) and perception_output.get("nav_lock_id") is None:
                    self._correct_object_with_camera_pose(
                        target_nav, ee_cam_pos_w, ee_cam_rot_w, *EE_INTR, robot_pos, robot_yaw,
                    )
            target_grasp = perception_output.get("target_grasp")
            if isinstance(target_grasp, dict):
                src = str(target_grasp.get("source") or target_grasp.get("camera") or "")
                head_objs = perception_output.get("head_objects", [])
                is_from_head = "head" in src.lower() or target_grasp in head_objs
                if not is_from_head:
                    self._correct_object_with_camera_pose(target_grasp, ee_cam_pos_w, ee_cam_rot_w, *EE_INTR, robot_pos, robot_yaw)
            corrected_any = True
            if self._step_count % 60 == 0:
                pw = ee_cam_pos_w
                self._log(f"[CORR] EE camera correction applied, {n_before} objects at pos=[{pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f}]")

        # 获取 head 相机真实位姿并校正 head 检测 + head mirror (ee_objects)
        head_cam_pose = self._get_ground_truth_camera_pose("head_camera")
        if head_cam_pose is not None:
            head_cam_pos_w, head_cam_rot_w = head_cam_pose
            head_objects = perception_output.get("head_objects", [])
            self._correct_objects_with_camera_pose(
                head_objects, head_cam_pos_w, head_cam_rot_w, HEAD_INTR, robot_pos, robot_yaw,
            )
            for obj in perception_output.get("ee_objects", []):
                if obj.get("skip_camera_correction") or not _is_head_sourced(obj):
                    continue
                self._correct_object_with_camera_pose(
                    obj, head_cam_pos_w, head_cam_rot_w, *HEAD_INTR, robot_pos, robot_yaw,
                )
            target_nav = perception_output.get("target_nav")
            if isinstance(target_nav, dict) and not target_nav.get("skip_camera_correction"):
                if _is_head_sourced(target_nav):
                    self._correct_object_with_camera_pose(
                        target_nav, head_cam_pos_w, head_cam_rot_w, *HEAD_INTR, robot_pos, robot_yaw,
                    )
            target_grasp = perception_output.get("target_grasp")
            if isinstance(target_grasp, dict):
                src = str(target_grasp.get("source") or target_grasp.get("camera") or "")
                head_objs = perception_output.get("head_objects", [])
                is_from_head = _is_head_sourced(target_grasp) or "head" in src.lower() or target_grasp in head_objs
                if is_from_head and not target_grasp.get("skip_camera_correction"):
                    self._correct_object_with_camera_pose(
                        target_grasp, head_cam_pos_w, head_cam_rot_w, *HEAD_INTR, robot_pos, robot_yaw,
                    )
            corrected_any = True
            if self._step_count % 60 == 0:
                pw = head_cam_pos_w
                self._log(f"[CORR] HEAD camera correction applied at pos=[{pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f}]")

        # Log one corrected target's pos_world for debugging
        if corrected_any and self._step_count % 60 == 0:
            target_nav = perception_output.get("target_nav")
            if isinstance(target_nav, dict):
                pw = target_nav.get("pos_world")
                dr = target_nav.get("dist_to_robot")
                if pw is not None:
                    self._log(
                        f"[CORR] target_nav pos_w=[{pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f}] "
                        f"dist_to_robot={dr:.2f}m"
                    )

    def _correct_objects_with_camera_pose(
        self,
        objects: list,
        cam_pos_w: np.ndarray,
        cam_rot_w: np.ndarray,
        fx_fy_cx_cy: tuple,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """批量校正物体位置"""
        fx, fy, cx, cy = fx_fy_cx_cy
        for obj in objects:
            if obj.get("skip_camera_correction"):
                continue
            self._correct_object_with_camera_pose(obj, cam_pos_w, cam_rot_w, fx, fy, cx, cy, robot_pos, robot_yaw)

    def _pick_reproject_uv_depth(
        self, obj: dict, *, prefer_centroid: bool = False,
    ) -> tuple[float, float, float] | None:
        """head 优先 nav_anchor; prefer_centroid 时仅用 bbox 中心 (GT 校正 fallback)."""
        centroid_uv = obj.get("centroid_uv") or obj.get("centroid")
        depth_m = obj.get("depth_m")
        if prefer_centroid:
            uv = centroid_uv
            depth = depth_m
        else:
            anchor_uv = obj.get("nav_anchor_uv") or obj.get("grasp_anchor_uv")
            anchor_depth = obj.get("nav_anchor_depth") or obj.get("grasp_anchor_depth") or obj.get("nav_depth_m")
            uv = anchor_uv if anchor_uv is not None else centroid_uv
            depth = anchor_depth if anchor_uv is not None and anchor_depth is not None else depth_m
        if uv is None or depth is None:
            return None
        try:
            u = float(uv[0])
            v = float(uv[1])
            depth_f = float(depth)
        except (TypeError, ValueError, IndexError):
            return None
        if depth_f <= 0.01 or depth_f > 100.0:
            return None
        return u, v, depth_f

    def _correction_pose_plausible(
        self, p_robot: np.ndarray, p_world: np.ndarray,
    ) -> bool:
        if p_robot[2] < -0.55 or p_robot[2] > 0.42:
            return False
        if p_world[2] < -0.12 or p_world[2] > 1.05:
            return False
        return True

    def _reproject_with_camera(
        self,
        u: float, v: float, depth_m_f: float,
        cam_pos_w: np.ndarray, cam_rot_w: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        robot_pos: np.ndarray, robot_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        x = (u - cx) / fx * depth_m_f
        y = (v - cy) / fy * depth_m_f
        p_cam = np.array([x, y, depth_m_f], dtype=np.float32)
        try:
            p_world = cam_pos_w + cam_rot_w @ p_cam
            p_robot = self._world_to_robot_frame(p_world, robot_pos, robot_yaw)
        except Exception:
            return None
        if not self._correction_pose_plausible(p_robot, p_world):
            return None
        return p_world, p_robot

    def _correct_object_with_camera_pose(
        self,
        obj: dict,
        cam_pos_w: np.ndarray,
        cam_rot_w: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """使用相机真实位姿校正单个物体位置"""
        picked = self._pick_reproject_uv_depth(obj)
        if picked is None:
            return
        u, v, depth_m_f = picked

        old_pos_w = obj.get("pos_world")
        old_gpw = obj.get("grasp_pos_world")
        old_pos_robot = None
        old_grasp_robot = None
        if old_pos_w is not None:
            try:
                old_pos_robot = self._world_to_robot_frame(
                    np.array(old_pos_w, dtype=np.float32), robot_pos, robot_yaw,
                )
            except Exception:
                old_pos_robot = None
        if old_gpw is not None and old_pos_robot is not None:
            try:
                old_grasp_robot = self._world_to_robot_frame(
                    np.array(old_gpw, dtype=np.float32), robot_pos, robot_yaw,
                ) - old_pos_robot
            except Exception:
                old_grasp_robot = None

        reproj = self._reproject_with_camera(
            u, v, depth_m_f, cam_pos_w, cam_rot_w, fx, fy, cx, cy, robot_pos, robot_yaw,
        )
        if reproj is None:
            fb = self._pick_reproject_uv_depth(obj, prefer_centroid=True)
            if fb is not None and fb != picked:
                reproj = self._reproject_with_camera(
                    fb[0], fb[1], fb[2], cam_pos_w, cam_rot_w, fx, fy, cx, cy, robot_pos, robot_yaw,
                )
        if reproj is None:
            return
        p_world, p_robot = reproj

        obj["pos_world"] = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
        obj["pose_source"] = "gt_camera"
        obj["pos_robot"] = [float(p_robot[0]), float(p_robot[1]), float(p_robot[2])]
        obj["dist_to_robot"] = float(np.linalg.norm(p_robot[:2]))
        obj["yaw_rel"] = float(math.atan2(p_robot[1], p_robot[0]))
        obj["nav_yaw_rel"] = obj["yaw_rel"]
        obj["nav_depth_m"] = depth_m_f
        obj["depth_m"] = depth_m_f
        obj["world_reliable"] = True

        if old_grasp_robot is not None:
            try:
                new_grasp_robot = p_robot + old_grasp_robot
                c, s = math.cos(robot_yaw), math.sin(robot_yaw)
                new_grasp_world = robot_pos + np.array([
                    c * new_grasp_robot[0] - s * new_grasp_robot[1],
                    s * new_grasp_robot[0] + c * new_grasp_robot[1],
                    new_grasp_robot[2],
                ], dtype=np.float32)
                obj["grasp_pos_world"] = [
                    float(new_grasp_world[0]), float(new_grasp_world[1]), float(new_grasp_world[2]),
                ]
                obj["grasp_pos_robot"] = [
                    float(new_grasp_robot[0]), float(new_grasp_robot[1]), float(new_grasp_robot[2]),
                ]
            except Exception:
                pass

    def _print_perception_targets(
        self,
        perception_output: dict[str, Any] | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        pose_source: str,
    ) -> None:
        if not self.nav_debug or not isinstance(perception_output, dict):
            return

        target_nav = self._prepare_pipeline_object(
            perception_output.get("target_nav"),
            str((perception_output.get("navigation") or {}).get("camera", "ee")),
            robot_pos_world,
            robot_yaw,
        )
        target_grasp = self._prepare_pipeline_object(
            perception_output.get("target_grasp"),
            str((perception_output.get("grasp") or {}).get("camera", "head")),
            robot_pos_world,
            robot_yaw,
        )
        all_objects = self._collect_pipeline_objects(perception_output, robot_pos_world, robot_yaw)
        
        if self._step_count % 10 == 0:
            self._log(
                "[PERC] "
                f"phase={perception_output.get('phase')} "
                f"ee={len(perception_output.get('ee_objects') or [])} "
                f"head={len(perception_output.get('head_objects') or [])} "
                f"pose_source={pose_source}"
            )
        
        if target_nav is not None and self._step_count % 10 == 0:
            pos_w = target_nav.get("pos_world")
            self._log(
                "[TARGET] "
                f"id={target_nav.get('id')} class={target_nav.get('class')} "
                f"dist={self._safe_float(target_nav.get('depth_m'), 0.0):.2f}m "
                f"pos_w=[{pos_w[0]:.2f},{pos_w[1]:.2f}]" if pos_w is not None else ""
            )

    def _compute_pregrasp_pose(
        self,
        robot_pos_world: np.ndarray,
        target_pos_world: np.ndarray,
        stand_off: float = 0.6,
    ) -> tuple[np.ndarray, float]:
        robot_xy = self._safe_numpy(robot_pos_world, np.zeros(3, dtype=np.float32))[:2]
        target_xy = self._safe_numpy(target_pos_world, np.zeros(3, dtype=np.float32))[:2]
        direction = robot_xy - target_xy
        norm = float(np.linalg.norm(direction))
        if norm < 0.01:
            direction = np.array([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction_normalized = direction / norm
        goal_xy = target_xy + direction_normalized * stand_off
        dx = target_xy[0] - goal_xy[0]
        dy = target_xy[1] - goal_xy[1]
        goal_yaw = float(math.atan2(dy, dx))
        return goal_xy, goal_yaw

    def _freeze_pregrasp_for_target(
        self,
        target_nav: dict[str, Any],
        robot_pos_world: np.ndarray,
    ) -> None:
        target_pos_world = self._safe_numpy(target_nav.get("pos_world"), np.zeros(3, dtype=np.float32))
        self._locked_goal_xy, self._locked_goal_yaw = self._compute_pregrasp_pose(
            robot_pos_world,
            target_pos_world,
            stand_off=0.6,
        )
        self._locked_goal_target_id = target_nav.get("id")
        self._locked_target_world = target_pos_world.tolist()
        self._log(
            f"[NAV] Locked pregrasp: id={target_nav.get('id')} "
            f"goal_xy={self._locked_goal_xy.round(2).tolist()} goal_yaw={self._locked_goal_yaw:.2f}rad"
        )

    def _dynamic_stop_distance_for_target(self, target_nav: dict[str, Any]) -> float:
        target_dist = self._conservative_target_dist(target_nav)
        if target_dist <= self.target_near_range:
            return self.dynamic_stop_distance_near
        return self.dynamic_stop_distance_far

    def _conservative_target_dist(
        self,
        target_nav: dict[str, Any],
        robot_pos_world: np.ndarray | None = None,
    ) -> float:
        """depth / dist_to_robot / world XY 取最大, 防假近 depth 提前蹲下."""
        if robot_pos_world is None:
            robot_pos_world, _, _ = self._resolve_robot_pose_world({}, {}, None)
        robot_pos_world = self._safe_numpy(robot_pos_world, np.zeros(3, dtype=np.float32))
        parts: list[float] = []
        dr = self._safe_float(target_nav.get("dist_to_robot"), 0.0)
        if dr > 0.05:
            parts.append(dr)
        dd = self._safe_float(target_nav.get("depth_m"), 0.0)
        if dd > 0.05:
            parts.append(dd)
        nd = self._safe_float(target_nav.get("nav_depth_m"), 0.0)
        if nd > 0.05:
            parts.append(nd)
        pw = self._safe_numpy(target_nav.get("pos_world"), np.zeros(3, dtype=np.float32))
        if np.all(np.isfinite(pw[:2])):
            hw = float(np.linalg.norm(pw[:2] - robot_pos_world[:2]))
            if hw > 0.05:
                parts.append(hw)
        pr = self._safe_numpy(target_nav.get("pos_robot"), np.zeros(3, dtype=np.float32))
        if np.all(np.isfinite(pr[:2])) and float(np.linalg.norm(pr[:2])) > 0.05:
            parts.append(float(np.linalg.norm(pr[:2])))
        return max(parts) if parts else float("inf")

    def _smooth_nav_heading(self, heading_error: float) -> float:
        alpha = self.nav_heading_ema_alpha
        if self._nav_heading_error_f is None:
            self._nav_heading_error_f = float(heading_error)
        else:
            self._nav_heading_error_f = (1.0 - alpha) * self._nav_heading_error_f + alpha * float(heading_error)
        smoothed = float(self._nav_heading_error_f)
        desired_sign = 0 if abs(smoothed) < 0.06 else (1 if smoothed > 0 else -1)
        if desired_sign == 0:
            self._nav_turn_sign = 0
            self._nav_turn_sign_hold = 0
        elif self._nav_turn_sign == 0 or self._nav_turn_sign == desired_sign:
            self._nav_turn_sign = desired_sign
            self._nav_turn_sign_hold = self.nav_turn_sign_hold_steps
        elif self._nav_turn_sign_hold > 0:
            self._nav_turn_sign_hold -= 1
        else:
            self._nav_turn_sign = desired_sign
            self._nav_turn_sign_hold = self.nav_turn_sign_hold_steps
        if self._nav_turn_sign != 0 and desired_sign != 0 and self._nav_turn_sign != desired_sign:
            return float(self._nav_turn_sign) * min(abs(smoothed), 0.12)
        return smoothed

    def _compute_dynamic_visual_cmd(
        self,
        target_nav: dict[str, Any],
        target_pos_robot: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target_dist = float(np.linalg.norm(target_pos_robot[:2]))
        heading_error = self._smooth_nav_heading(float(math.atan2(target_pos_robot[1], target_pos_robot[0])))
        stop_distance = self._dynamic_stop_distance_for_target(target_nav)
        remaining = max(0.0, target_dist - stop_distance)
        lateral_error = float(target_pos_robot[1])
        goal_dist = remaining

        phase = "near_refine" if target_dist <= self.target_near_range else "far_approach"

        approach_scale = float(np.clip(remaining / self.slow_down_radius, 0.0, 1.0))
        heading_scale = max(self.min_heading_lin_scale, abs(math.cos(heading_error)))

        lin_x = float(np.clip(remaining * approach_scale * heading_scale, 0.0, self.max_lin_vel))
        if remaining > 0.35 and lin_x < 0.45:
            lin_x = 0.45 * approach_scale
        lin_y = 0.0
        ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
        if remaining > 1.0:
            ang_z *= 0.72

        if abs(heading_error) > self.turn_then_go_yaw_threshold and lin_x < 0.20:
            phase = "turn_to_target"
        elif target_dist <= self.target_near_range:
            phase = "near_refine"
        else:
            phase = "far_approach"

        stopped = False
        if goal_dist <= self.object_stop_tolerance and abs(heading_error) <= self.object_yaw_tolerance:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0
            stopped = True
            phase = "refine_hold"

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        return base_cmd, {
            "phase": phase,
            "target": target_nav,
            "target_dist": target_dist,
            "goal_dist": goal_dist,
            "heading_error": heading_error,
            "forward_error": remaining,
            "lateral_error": lateral_error,
            "stopped": stopped,
            "ang_z": float(ang_z),
        }

    def _compute_frozen_pregrasp_cmd(
        self,
        target_nav: dict[str, Any],
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        robot_xy = robot_pos_world[:2]
        goal_xy = np.asarray(self._locked_goal_xy, dtype=np.float32)
        target_world = self._safe_numpy(
            self._locked_target_world if self._locked_target_world is not None else target_nav.get("pos_world"),
            np.zeros(3, dtype=np.float32),
        )
        error_xy_w = goal_xy - robot_xy
        pos_error_norm = float(np.linalg.norm(error_xy_w))

        goal_x_robot = math.cos(robot_yaw) * error_xy_w[0] + math.sin(robot_yaw) * error_xy_w[1]
        goal_y_robot = -math.sin(robot_yaw) * error_xy_w[0] + math.cos(robot_yaw) * error_xy_w[1]
        desired_yaw = float(math.atan2(target_world[1] - robot_pos_world[1], target_world[0] - robot_pos_world[0]))
        yaw_error = self._wrap_angle(desired_yaw - robot_yaw)

        approach_scale = float(np.clip(pos_error_norm / self.slow_down_radius, 0.0, 1.0))
        heading_scale = max(self.min_heading_lin_scale, math.cos(min(abs(yaw_error), math.pi / 2.0)))
        align_threshold = self.turn_then_go_yaw_threshold if pos_error_norm > 0.25 else self.turn_then_go_heading_hold
        if abs(yaw_error) > align_threshold:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = float(np.clip(yaw_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
            phase = "turn_to_pregrasp"
        else:
            lin_x = float(np.clip(goal_x_robot * approach_scale * heading_scale, -self.max_lin_vel, self.max_lin_vel))
            lin_y = 0.0
            ang_z = 0.0
            phase = "locked_pregrasp"
        stopped = False
        target_dist = float(np.linalg.norm(target_world[:2] - robot_pos_world[:2]))
        if pos_error_norm <= 0.32 and abs(yaw_error) <= self.object_yaw_tolerance:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0
            phase = "ready_to_grasp"
            stopped = True
        elif (
            target_dist <= self.grasp_start_depth
            and self._conservative_target_dist(target_nav, robot_pos_world) <= self.grasp_start_depth
            and abs(yaw_error) <= max(self.object_yaw_tolerance * 2.5, 0.35)
        ):
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0
            phase = "ready_to_grasp"
            stopped = True

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        return base_cmd, {
            "phase": phase,
            "target": target_nav,
            "target_dist": target_dist,
            "goal_dist": pos_error_norm,
            "heading_error": yaw_error,
            "forward_error": float(goal_x_robot),
            "lateral_error": float(goal_y_robot),
            "stopped": stopped,
        }

    def _compute_nav_cmd_from_target_nav(self, target_nav: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        robot_pos_world, robot_yaw, _ = self._resolve_robot_pose_world({}, {}, None)
        target_pos_world = self._safe_numpy(target_nav.get("pos_world"), np.zeros(3, dtype=np.float32))
        target_pos_robot = self._safe_numpy(target_nav.get("pos_robot"), np.zeros(3, dtype=np.float32))
        if not np.all(np.isfinite(target_pos_robot)) or float(np.linalg.norm(target_pos_robot[:2])) < 1e-6:
            target_pos_robot = self._world_to_robot_frame(target_pos_world, robot_pos_world, robot_yaw)
        target_nav["pos_robot"] = target_pos_robot.tolist()
        target_nav["dist_to_robot"] = float(np.linalg.norm(target_pos_robot[:2]))
        target_nav["yaw_rel"] = float(math.atan2(target_pos_robot[1], target_pos_robot[0]))

        # Close-range head relocalization can invalidate a stale frozen pregrasp.
        if (
            self._locked_target_world is not None
            and bool(target_nav.get("visible", True))
            and target_nav.get("source_camera") == "head"
            and target_nav["dist_to_robot"] <= self.target_near_range
        ):
            frozen_target = self._safe_numpy(self._locked_target_world, np.zeros(3, dtype=np.float32))
            disagreement = float(np.linalg.norm(target_pos_world[:2] - frozen_target[:2]))
            if disagreement > self.target_relock_disagreement:
                self._log(
                    f"[NAV] head relocalization disagrees with frozen target by {disagreement:.2f}m, "
                    "unlocking pregrasp"
                )
                self._clear_frozen_pregrasp()

        cons_dist = self._conservative_target_dist(target_nav, robot_pos_world)
        if (
            self._locked_goal_xy is None
            and bool(target_nav.get("confirmed", False))
            and bool(target_nav.get("visible", True))
            and target_nav["dist_to_robot"] <= self.target_freeze_distance
            and cons_dist <= self.grasp_start_depth - 0.05
            and target_nav.get("source_camera") == "head"
            and int(target_nav.get("head_stable_count", 0)) >= self.target_head_confirm_steps
        ):
            self._freeze_pregrasp_for_target(target_nav, robot_pos_world)

        if self._locked_goal_xy is not None and self._locked_goal_yaw is not None:
            base_cmd, nav_info = self._compute_frozen_pregrasp_cmd(target_nav, robot_pos_world, robot_yaw)
            goal_dist = float(nav_info.get("goal_dist", 0.0))
            robot_xy = robot_pos_world[:2]
            if self._pregrasp_last_robot_xy is not None:
                moved = float(np.linalg.norm(robot_xy - self._pregrasp_last_robot_xy))
                if moved < 0.015 and goal_dist > 0.28 and abs(float(base_cmd[0])) > 0.05:
                    self._pregrasp_stall_steps += 1
                else:
                    self._pregrasp_stall_steps = 0
            self._pregrasp_last_robot_xy = robot_xy.copy()
            if self._pregrasp_stall_steps >= self.pregrasp_stall_steps:
                self._log(
                    f"[NAV] pregrasp stall ({self._pregrasp_stall_steps} steps, goal={goal_dist:.2f}m), "
                    "unlocking frozen goal"
                )
                self._clear_frozen_pregrasp()
                return self._compute_dynamic_visual_cmd(target_nav, target_pos_robot)
            return base_cmd, nav_info
        self._pregrasp_stall_steps = 0
        self._pregrasp_last_robot_xy = None
        base_cmd, nav_info = self._compute_dynamic_visual_cmd(target_nav, target_pos_robot)
        return base_cmd, nav_info

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _make_pipeline_nav_info(
        self,
        perception_output: dict[str, Any] | None,
        pose_source: str,
        target: dict[str, Any] | None = None,
        phase: str | None = None,
        stopped: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        robot_pos_world = None
        robot_yaw = None
        if isinstance(perception_output, dict):
            robot = perception_output.get("robot") or {}
            robot_pos_world = robot.get("pos_world")
            robot_yaw = robot.get("yaw")
        objects = self._collect_pipeline_objects(
            perception_output,
            self._safe_numpy(robot_pos_world, self._odom_pos),
            self._safe_float(robot_yaw, self._odom_yaw),
        )
        info = {
            "phase": phase or (None if perception_output is None else str(perception_output.get("phase", "approach"))),
            "target": target,
            "stopped": stopped,
            "target_source_camera": None if target is None else target.get("source_camera"),
            "preferred_camera": None if perception_output is None else perception_output.get("active_camera"),
            "pose_source": pose_source,
            "objects_detailed": objects,
            "objects_remaining": objects,
        }
        if extra:
            info.update(extra)
        return info

    def _normalize_object(
        self,
        obj: dict[str, Any],
        source_camera: str,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any] | None:
        if not isinstance(obj, dict):
            return None

        pos_world = obj.get("pos_world")
        pos_robot = obj.get("pos_robot")

        if pos_world is not None:
            pos_world_np = self._safe_numpy(pos_world, np.zeros(3, dtype=np.float32))
        elif pos_robot is not None:
            pos_world_np = self._robot_to_world_frame(pos_robot, robot_pos_world, robot_yaw)
        else:
            return None

        if not np.all(np.isfinite(pos_world_np)):
            return None

        pos_robot_np = self._world_to_robot_frame(pos_world_np, robot_pos_world, robot_yaw)
        if not np.all(np.isfinite(pos_robot_np)):
            return None

        out = dict(obj)
        out["source_camera"] = source_camera
        out["camera"] = source_camera
        out["pos_world"] = pos_world_np.tolist()
        out["pos_robot"] = pos_robot_np.tolist()
        out["dist_to_robot"] = float(np.linalg.norm(pos_robot_np[:2]))
        out["yaw_rel"] = float(math.atan2(pos_robot_np[1], pos_robot_np[0]))
        out["conf"] = self._safe_float(obj.get("conf", 1.0), 1.0)

        if out.get("grasp_pos_world") is None:
            grasp_pos = pos_world_np.copy()
            out["grasp_pos_world"] = grasp_pos.tolist()

        return out

    def _is_valid_target_candidate(self, obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        if obj.get("in_bin", False):
            return False
        if self._safe_float(obj.get("conf", 1.0), 1.0) < self.min_target_conf:
            return False

        pos_robot = self._safe_numpy(obj.get("pos_robot"), np.zeros(3, dtype=np.float32))
        if not np.all(np.isfinite(pos_robot)):
            return False
        if pos_robot[0] <= 0.05:
            return False

        dist = self._safe_float(obj.get("dist_to_robot", np.linalg.norm(pos_robot[:2])), fallback=float("inf"))
        if not np.isfinite(dist) or dist < self.min_target_distance or dist > self.max_target_distance:
            return False

        depth_m = obj.get("depth_m")
        if depth_m is not None:
            depth_value = self._safe_float(depth_m, fallback=float("inf"))
            if not np.isfinite(depth_value) or depth_value <= 0.05:
                return False

        return True

    def _normalize_camera_objects(
        self,
        objects: list[dict[str, Any]] | None,
        source_camera: str,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for obj in objects or []:
            adapted = self._normalize_object(obj, source_camera, robot_pos_world, robot_yaw)
            if adapted is not None and self._is_valid_target_candidate(adapted):
                normalized.append(adapted)
        normalized.sort(key=lambda item: self._safe_float(item.get("dist_to_robot"), fallback=float("inf")))
        return normalized

    def _tracking_alpha(self, source_camera: str) -> float:
        return self.target_ema_alpha_head if source_camera == "head" else self.target_ema_alpha_ee

    def _tracking_match_radius(self, source_camera: str, target_dist: float) -> float:
        if source_camera == "head" or target_dist <= self.target_near_range:
            return self.target_near_match_radius
        return self.target_far_match_radius

    def _set_locked_target_snapshot(self, target: dict[str, Any] | None) -> None:
        if not isinstance(target, dict):
            self._locked_target = None
            return
        self._locked_target = {
            "class": target.get("class"),
            "id": target.get("id"),
            "source_camera": target.get("source_camera"),
            "pos_world": self._safe_numpy(target.get("pos_world"), np.zeros(3, dtype=np.float32)).tolist(),
            "pos_robot": self._safe_numpy(target.get("pos_robot"), np.zeros(3, dtype=np.float32)).tolist(),
            "last_seen_step": self._step_count,
        }

    def _clear_frozen_pregrasp(self) -> None:
        self._locked_goal_xy = None
        self._locked_goal_yaw = None
        self._locked_goal_target_id = None
        self._locked_target_world = None
        self._pregrasp_stall_steps = 0
        self._pregrasp_last_robot_xy = None

    def _clear_locked_target(self) -> None:
        self._locked_target = None
        self._pending_target = None
        self._tracked_target = None
        self._last_known_target_pos = None
        self._target_lost_count = 0
        self._clear_frozen_pregrasp()

    def _make_tracking_state(self, candidate: dict[str, Any], stable_count: int = 1) -> dict[str, Any]:
        pos_world = self._safe_numpy(candidate.get("pos_world"), np.zeros(3, dtype=np.float32))
        pos_robot = self._safe_numpy(candidate.get("pos_robot"), np.zeros(3, dtype=np.float32))
        source_camera = str(candidate.get("source_camera", "ee"))
        return {
            "id": candidate.get("id"),
            "class": candidate.get("class"),
            "source_camera": source_camera,
            "filtered_pos_world": pos_world.tolist(),
            "last_candidate": dict(candidate),
            "last_seen_step": self._step_count,
            "stable_count": stable_count,
            "head_stable_count": 1 if source_camera == "head" else 0,
            "dist_to_robot": float(np.linalg.norm(pos_robot[:2])),
            "conf": self._safe_float(candidate.get("conf", 1.0), 1.0),
        }

    def _target_state_distance(self, state_or_target: dict[str, Any], candidate: dict[str, Any]) -> float:
        state_pos = state_or_target.get("filtered_pos_world")
        if state_pos is None:
            state_pos = state_or_target.get("pos_world")
        candidate_pos = candidate.get("pos_world")
        if state_pos is None or candidate_pos is None:
            return float("inf")
        state_pos_np = self._safe_numpy(state_pos, np.zeros(3, dtype=np.float32))
        candidate_pos_np = self._safe_numpy(candidate_pos, np.zeros(3, dtype=np.float32))
        return float(np.linalg.norm(state_pos_np[:2] - candidate_pos_np[:2]))

    def _targets_compatible(
        self,
        state_or_target: dict[str, Any],
        candidate: dict[str, Any],
        radius: float,
    ) -> bool:
        track_class = state_or_target.get("class")
        cand_class = candidate.get("class")
        if track_class is not None and cand_class is not None and track_class != cand_class:
            return False
        return self._target_state_distance(state_or_target, candidate) <= radius

    def _candidate_rank(
        self,
        candidate: dict[str, Any],
        preferred_camera: str,
        reference: dict[str, Any] | None = None,
    ) -> float:
        pos_robot = self._safe_numpy(candidate.get("pos_robot"), np.zeros(3, dtype=np.float32))
        pos_world = self._safe_numpy(candidate.get("pos_world"), np.zeros(3, dtype=np.float32))
        dist = self._safe_float(candidate.get("dist_to_robot"), float(np.linalg.norm(pos_robot[:2])))
        conf = self._safe_float(candidate.get("conf", 1.0), 1.0)
        yaw_rel = abs(self._safe_float(candidate.get("yaw_rel", math.atan2(pos_robot[1], pos_robot[0]))))
        score = dist
        score += yaw_rel * 0.35
        score += max(abs(float(pos_world[2])) - 0.35, 0.0) * 0.3
        score += (1.0 - conf) * 0.45
        if candidate.get("source_camera") != preferred_camera:
            score += 0.12
        if reference is not None:
            score += self._target_state_distance(reference, candidate) * 2.2
            ref_class = reference.get("class")
            if ref_class is not None and candidate.get("class") is not None and candidate.get("class") != ref_class:
                score += 3.0
        return score

    def _tracking_candidate_allowed(
        self,
        candidate: dict[str, Any],
        *,
        has_head_candidates: bool = True,
    ) -> bool:
        source = str(candidate.get("source_camera", "ee"))
        pos_robot = self._safe_numpy(candidate.get("pos_robot"), np.zeros(3, dtype=np.float32))
        dist = self._safe_float(candidate.get("dist_to_robot"), float(np.linalg.norm(pos_robot[:2])))
        if source == "ee" and dist > self.ee_track_max_distance:
            if not has_head_candidates and dist <= self.ee_search_fallback_distance:
                return True
            return False
        return True

    def _select_best_candidate(
        self,
        candidates: list[dict[str, Any]],
        preferred_camera: str,
        reference: dict[str, Any] | None = None,
        *,
        has_head_candidates: bool = True,
    ) -> dict[str, Any] | None:
        eligible = [
            c for c in candidates
            if self._tracking_candidate_allowed(c, has_head_candidates=has_head_candidates)
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda candidate: self._candidate_rank(candidate, preferred_camera, reference))

    def _match_candidate_to_track(
        self,
        candidates: list[dict[str, Any]],
        track: dict[str, Any],
    ) -> dict[str, Any] | None:
        best_match = None
        best_score = float("inf")
        track_dist = self._safe_float(track.get("dist_to_robot"), float("inf"))
        for candidate in candidates:
            cand_dist = self._safe_float(candidate.get("dist_to_robot"), fallback=float("inf"))
            radius = self._tracking_match_radius(str(candidate.get("source_camera", "ee")), min(track_dist, cand_dist))
            if not self._targets_compatible(track, candidate, radius):
                continue
            score = self._candidate_rank(candidate, str(track.get("source_camera", "ee")), track)
            if score < best_score:
                best_score = score
                best_match = candidate
        return best_match

    def _update_tracking_state_from_candidate(
        self,
        state: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        alpha = self._tracking_alpha(str(candidate.get("source_camera", "ee")))
        prev_pos = self._safe_numpy(state.get("filtered_pos_world"), np.zeros(3, dtype=np.float32))
        obs_pos = self._safe_numpy(candidate.get("pos_world"), np.zeros(3, dtype=np.float32))
        jump_xy = float(np.linalg.norm(obs_pos[:2] - prev_pos[:2]))
        jump_limit = self.track_jump_reject_m
        if str(candidate.get("source_camera", "ee")) == "ee":
            dist = float(np.linalg.norm(self._safe_numpy(candidate.get("pos_robot"), np.zeros(3, dtype=np.float32))[:2]))
            jump_limit = min(jump_limit, 0.35 if dist > 2.2 else 0.45)
        if candidate.get("pos_jump_rejected") or jump_xy > jump_limit:
            filtered_pos = prev_pos
        elif jump_xy > 1.20:
            filtered_pos = prev_pos
            state["stable_count"] = max(int(state.get("stable_count", 1)) - 1, 1)
        else:
            filtered_pos = (1.0 - alpha) * prev_pos + alpha * obs_pos
        pos_robot = self._safe_numpy(candidate.get("pos_robot"), np.zeros(3, dtype=np.float32))
        state["filtered_pos_world"] = filtered_pos.tolist()
        state["last_candidate"] = dict(candidate)
        state["source_camera"] = candidate.get("source_camera")
        state["id"] = candidate.get("id", state.get("id"))
        if candidate.get("class") is not None:
            state["class"] = candidate.get("class")
        state["last_seen_step"] = self._step_count
        state["stable_count"] = min(int(state.get("stable_count", 0)) + 1, self.target_confirm_steps + 4)
        if candidate.get("source_camera") == "head" and float(np.linalg.norm(pos_robot[:2])) <= self.target_near_range:
            state["head_stable_count"] = min(
                int(state.get("head_stable_count", 0)) + 1,
                self.target_head_confirm_steps + 4,
            )
        else:
            state["head_stable_count"] = max(int(state.get("head_stable_count", 0)) - 1, 0)
        state["dist_to_robot"] = float(np.linalg.norm(pos_robot[:2]))
        state["conf"] = self._safe_float(candidate.get("conf", state.get("conf", 1.0)), 1.0)
        return state

    def _build_tracked_nav_target(
        self,
        track: dict[str, Any],
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any]:
        filtered_pos = self._safe_numpy(track.get("filtered_pos_world"), np.zeros(3, dtype=np.float32))
        pos_robot = self._world_to_robot_frame(filtered_pos, robot_pos_world, robot_yaw)
        target = dict(track.get("last_candidate") or {})
        target["id"] = track.get("id")
        target["class"] = track.get("class")
        target["source_camera"] = track.get("source_camera")
        target["camera"] = track.get("source_camera")
        target["pos_world"] = filtered_pos.tolist()
        target["pos_robot"] = pos_robot.tolist()
        target["dist_to_robot"] = float(np.linalg.norm(pos_robot[:2]))
        target["yaw_rel"] = float(math.atan2(pos_robot[1], pos_robot[0]))
        target["stable_count"] = int(track.get("stable_count", 0))
        target["head_stable_count"] = int(track.get("head_stable_count", 0))
        target["confirmed"] = True
        target["visible"] = int(track.get("last_seen_step", -1)) == self._step_count
        target["last_seen_step"] = int(track.get("last_seen_step", self._step_count))
        if target.get("grasp_pos_world") is None:
            target["grasp_pos_world"] = filtered_pos.tolist()
        return target

    def _update_visual_target_tracking(
        self,
        preferred_candidates: list[dict[str, Any]],
        fallback_candidates: list[dict[str, Any]],
        preferred_camera: str,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any] | None:
        all_candidates = preferred_candidates + fallback_candidates
        has_head_candidates = bool(preferred_candidates) if preferred_camera == "head" else bool(fallback_candidates)

        if self._tracked_target is not None:
            matched = self._match_candidate_to_track(all_candidates, self._tracked_target)
            if matched is not None:
                self._tracked_target = self._update_tracking_state_from_candidate(self._tracked_target, matched)
                self._target_lost_count = 0
            else:
                same_class_candidates = [
                    candidate
                    for candidate in all_candidates
                    if self._tracked_target.get("class") in {None, candidate.get("class")}
                ]
                if same_class_candidates:
                    self._target_lost_count += 1
                else:
                    self._target_lost_count += 2
                if self._target_lost_count > self.target_lock_timeout_steps:
                    self._log("[NAV] tracked target lost too long, clearing visual target state")
                    self._clear_locked_target()
                    return None

        if self._tracked_target is None:
            best = self._select_best_candidate(
                all_candidates,
                preferred_camera,
                self._pending_target,
                has_head_candidates=has_head_candidates,
            )
            if best is None:
                if (
                    self._pending_target is not None
                    and self._step_count - int(self._pending_target.get("last_seen_step", self._step_count))
                    > self.target_pending_timeout_steps
                ):
                    self._pending_target = None
                return None

            if self._pending_target is not None and self._targets_compatible(
                self._pending_target,
                best,
                self.target_pending_match_radius,
            ):
                self._pending_target = self._update_tracking_state_from_candidate(self._pending_target, best)
            else:
                self._pending_target = self._make_tracking_state(best, stable_count=1)

            if int(self._pending_target.get("stable_count", 0)) >= self.target_confirm_steps:
                self._tracked_target = dict(self._pending_target)
                self._pending_target = None
                self._target_lost_count = 0
                self._log(
                    "[NAV] Confirmed tracked target: "
                    f"id={self._tracked_target.get('id')} class={self._tracked_target.get('class')} "
                    f"camera={self._tracked_target.get('source_camera')}"
                )

        if self._tracked_target is None:
            return None

        target = self._build_tracked_nav_target(self._tracked_target, robot_pos_world, robot_yaw)
        self._last_known_target_pos = target.get("pos_world")
        self._set_locked_target_snapshot(target)
        return target

    def _get_camera_hint(self, perception_output: dict[str, Any], camera: str) -> dict[str, Any] | None:
        for key in ("navigation", "grasp"):
            section = perception_output.get(key) or {}
            if section.get("camera") == camera and isinstance(section.get("target"), dict):
                return section.get("target")
        return None

    def _compute_search_cmd(self, perception_output: dict[str, Any] | None) -> np.ndarray:
        """head=0 时用 EE 方位定向转，避免盲转 43s (log 02-27-18)."""
        hint = (perception_output or {}).get("ee_search_hint")
        if isinstance(hint, dict) and hint.get("yaw_rel") is not None:
            bearing = float(hint["yaw_rel"])
            if abs(bearing) > 0.12:
                ang = float(np.clip(bearing * self.heading_kp, -0.55, 0.55))
                return np.array([0.0, 0.0, ang], dtype=np.float32)
        return np.array([0.0, 0.0, self.search_yaw_rate], dtype=np.float32)

    def _build_search_nav_info(self, nav_input: dict[str, Any], target: dict[str, Any] | None) -> dict[str, Any]:
        robot = nav_input.get("robot") or {}
        return {
            "phase": str(nav_input.get("phase", "search")),
            "goal_dist": 0.0,
            "bin_dist": 0.0,
            "heading_error": 0.0,
            "target_dist": 0.0,
            "stopped": False,
            "target": target,
            "pose_source": robot.get("pose_source"),
            "preferred_camera": nav_input.get("preferred_camera"),
            "fallback_camera": nav_input.get("fallback_camera"),
            "target_source_camera": nav_input.get("target_source_camera"),
            "objects_detailed": list(nav_input.get("objects_detailed") or []),
            "objects_remaining": list(nav_input.get("objects_remaining") or []),
            "searching": True,
        }

    def _attach_nav_debug_context(
        self,
        nav_info: dict[str, Any],
        nav_input: dict[str, Any],
        target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(nav_info)
        robot = nav_input.get("robot") or {}
        merged["target"] = target
        merged["preferred_camera"] = nav_input.get("preferred_camera")
        merged["fallback_camera"] = nav_input.get("fallback_camera")
        merged["target_source_camera"] = nav_input.get("target_source_camera")
        merged["pose_source"] = robot.get("pose_source")
        merged["objects_detailed"] = list(nav_input.get("objects_detailed") or [])
        merged["objects_remaining"] = list(nav_input.get("objects_remaining") or [])
        return merged

    def _clear_fuse_nav_lock(self, reason: str = "") -> None:
        if reason:
            self._log(f"[NAV] fuse unlock: {reason}")
        self._fuse_lock_key = None
        self._fuse_pos_world = None
        self._ee_only_no_head_frames = 0
        self._nav_stall_turn_rad = 0.0
        self._nav_stall_dist_start = None
        self._nav_heading_error_f = None
        self._nav_turn_sign = 0
        self._nav_turn_sign_hold = 0
        self._nav_ignore_perc_until_head = False

    def _nav_approach_stalled(self, nav_info: dict[str, Any], target_dist: float) -> bool:
        phase = str(nav_info.get("phase", ""))
        if phase != "turn_to_target":
            self._nav_stall_turn_rad = 0.0
            self._nav_stall_dist_start = None
            return False
        if self._nav_stall_dist_start is None:
            self._nav_stall_dist_start = target_dist
        ang_z = abs(float(nav_info.get("ang_z") or nav_info.get("cmd_ang_z") or 0.0))
        self._nav_stall_turn_rad += ang_z * 0.02
        if (
            self._nav_stall_turn_rad >= self.nav_stall_turn_rad
            and target_dist >= self.nav_stall_min_dist_m
            and self._nav_stall_dist_start is not None
            and (self._nav_stall_dist_start - target_dist) < 0.12
        ):
            return True
        return False

    def _fuse_perception_target(
        self,
        perception_output: dict[str, Any],
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any] | None:
        """单一导航出口：只跟感知 target_nav."""
        raw_nav = perception_output.get("target_nav")
        if not isinstance(raw_nav, dict) or raw_nav.get("pos_world") is None:
            return None

        lock_id = perception_output.get("nav_lock_id")
        lock_class = perception_output.get("nav_lock_class")

        active = perception_output.get("active_camera")
        nav_cam = str(raw_nav.get("source_camera") or active or "head")
        if nav_cam in ("lock_coast", "none"):
            nav_cam = "head" if active != "ee" else "ee"

        target = self._prepare_pipeline_object(raw_nav, nav_cam, robot_pos_world, robot_yaw)
        if target is None:
            return None

        lock_key = (lock_id, lock_class)
        pw = self._safe_numpy(target.get("pos_world"), np.zeros(3, dtype=np.float32))
        if lock_key != self._fuse_lock_key or self._fuse_pos_world is None:
            self._fuse_lock_key = lock_key
            self._fuse_pos_world = pw.copy()
            self._nav_heading_error_f = None
        else:
            alpha = 0.35
            self._fuse_pos_world = (1.0 - alpha) * self._fuse_pos_world + alpha * pw
            pw = self._fuse_pos_world.copy()
        self._fuse_pos_world = pw.copy()
        pos_robot = self._world_to_robot_frame(pw, robot_pos_world, robot_yaw)
        target["id"] = lock_id if lock_id is not None else target.get("id")
        target["class"] = lock_class or target.get("class")
        target["pos_world"] = pw.tolist()
        target["pos_robot"] = pos_robot.tolist()
        target["dist_to_robot"] = float(np.linalg.norm(pos_robot[:2]))
        target["yaw_rel"] = float(math.atan2(pos_robot[1], pos_robot[0]))
        target["source_camera"] = nav_cam
        target["camera"] = nav_cam
        target["confirmed"] = True
        target["visible"] = True
        target["nav_lock_id"] = lock_id
        self._last_known_target_pos = target["pos_world"]
        self._set_locked_target_snapshot(target)
        return target

    def _adapt_perception_output(
        self,
        obs: dict[str, Any],
        local_nav: dict[str, Any],
        perception_output: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        robot_pos_world, robot_yaw, pose_source = self._resolve_robot_pose_world(obs, local_nav, perception_output)

        phase = str(perception_output.get("phase", "approach"))
        active = perception_output.get("active_camera")
        if active in ("head", "ee"):
            preferred_camera = str(active)
        else:
            preferred_camera = "ee" if phase == "grasp" else "head"
        fallback_camera = "ee" if preferred_camera == "head" else "head"

        preferred_raw = perception_output.get("head_objects", []) if preferred_camera == "head" else perception_output.get("ee_objects", [])
        fallback_raw = perception_output.get("ee_objects", []) if preferred_camera == "head" else perception_output.get("head_objects", [])

        preferred_candidates = self._normalize_camera_objects(preferred_raw, preferred_camera, robot_pos_world, robot_yaw)
        fallback_candidates = self._normalize_camera_objects(fallback_raw, fallback_camera, robot_pos_world, robot_yaw)

        target = self._fuse_perception_target(perception_output, robot_pos_world, robot_yaw)
        if target is None and perception_output.get("nav_lock_id") is None:
            target = self._update_visual_target_tracking(
                preferred_candidates,
                fallback_candidates,
                preferred_camera,
                robot_pos_world,
                robot_yaw,
            )

        bin_info = perception_output.get("bin") or {}
        bin_center = self._safe_numpy(bin_info.get("center_world"), self.default_bin_center)

        nav_input = {
            "robot": {
                "pos_world": robot_pos_world.tolist(),
                "yaw": float(robot_yaw),
                "pose_source": pose_source,
            },
            "bin": {
                "center_world": bin_center.tolist(),
            },
            "phase": phase,
            "preferred_camera": preferred_camera,
            "fallback_camera": fallback_camera,
            "target_source_camera": None if target is None else target.get("source_camera"),
            "target": target,
            "objects_detailed": preferred_candidates + fallback_candidates,
            "objects_remaining": preferred_candidates + fallback_candidates,
        }
        return nav_input, target

    def _compute_object_cmd(self, target: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        pos_robot = self._safe_numpy(target.get("pos_robot"), np.zeros(3, dtype=np.float32))
        # 检查是否有有效距离
        target_dist = float(np.linalg.norm(pos_robot[:2]))
        if target_dist < 0.05:
            return np.zeros(3, dtype=np.float32), {
                "phase": "search",
                "goal_dist": 0.0,
                "target_dist": 0.0,
                "heading_error": 0.0,
                "forward_error": 0.0,
                "lateral_error": 0.0,
                "stopped": False,
                "target": target,
            }

        heading_error = float(math.atan2(pos_robot[1], pos_robot[0]))
        forward_error = float(pos_robot[0] - self.object_stop_distance)
        lateral_error = float(pos_robot[1])
        goal_dist = float(np.linalg.norm([forward_error, lateral_error]))

        close_forward = abs(forward_error) <= self.object_forward_tolerance
        close_lateral = abs(lateral_error) <= self.object_lateral_tolerance
        close_heading = abs(heading_error) <= self.object_yaw_tolerance

        if close_forward and close_lateral and close_heading:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0
            phase = "stopped"
        elif close_forward and close_lateral:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
            phase = "align"
        else:
            approach_scale = float(np.clip(goal_dist / self.slow_down_radius, 0.0, 1.0))
            heading_scale = max(self.min_heading_lin_scale, math.cos(min(abs(heading_error), math.pi / 2.0)))
            lin_x = float(np.clip(forward_error * approach_scale * heading_scale, -self.max_lin_vel, self.max_lin_vel))
            lin_y = float(np.clip(lateral_error * approach_scale * 0.4, -self.max_lat_vel, self.max_lat_vel))
            ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))
            phase = "approach"

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        base_cmd = np.nan_to_num(base_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        return base_cmd, {
            "phase": phase,
            "goal_dist": goal_dist,
            "target_dist": target_dist,
            "heading_error": heading_error,
            "forward_error": forward_error,
            "lateral_error": lateral_error,
            "stopped": phase == "stopped",
            "target": target,
        }

    def _get_navigation_input(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        local_nav = self._update_local_odometry(obs)

        gt_pose = self._get_ground_truth_robot_pose()
        gt_pos = gt_pose[0] if gt_pose is not None else None
        gt_yaw = gt_pose[1] if gt_pose is not None else None
        try:
            perception_output = self.perception.process(obs, dt=self.dt, gt_robot_pos=gt_pos, gt_robot_yaw=gt_yaw)
            return self._adapt_perception_output(obs, local_nav, perception_output)
        except Exception as exc:
            if not self._perception_error_printed:
                self._log(f"[TaskB-PERCEPTION] disabled after error: {type(exc).__name__}: {exc}")
                self._perception_error_printed = True
            return local_nav, None

    def _compute_base_cmd(self, nav_input: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        robot_info = nav_input.get("robot") or {}
        bin_info = nav_input.get("bin") or {}

        robot_pos = self._safe_numpy(robot_info.get("pos_world"), np.zeros(3, dtype=np.float32))
        robot_yaw = float(robot_info.get("yaw", 0.0))
        if not np.isfinite(robot_yaw):
            robot_yaw = 0.0

        bin_center = self._safe_numpy(bin_info.get("center_world"), self.default_bin_center)
        vector_to_bin = bin_center[:2] - robot_pos[:2]
        dist_to_bin = float(np.linalg.norm(vector_to_bin))
        if not np.isfinite(dist_to_bin) or dist_to_bin < 1e-6:
            return np.zeros(3, dtype=np.float32), {
                "heading": 0.0,
                "goal_dist": 0.0,
                "bin_dist": dist_to_bin,
                "stopped": True,
            }

        stop_point = bin_center[:2] - self.stop_distance * vector_to_bin / dist_to_bin
        vector_to_goal = stop_point - robot_pos[:2]
        goal_dist = float(np.linalg.norm(vector_to_goal))

        if not np.isfinite(goal_dist) or goal_dist <= self.stop_tolerance:
            return np.zeros(3, dtype=np.float32), {
                "heading": 0.0,
                "goal_dist": max(goal_dist, 0.0),
                "bin_dist": dist_to_bin,
                "stopped": True,
            }

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        goal_x_robot = cos_yaw * vector_to_goal[0] + sin_yaw * vector_to_goal[1]
        goal_y_robot = -sin_yaw * vector_to_goal[0] + cos_yaw * vector_to_goal[1]

        desired_heading = math.atan2(vector_to_goal[1], vector_to_goal[0])
        heading_error = self._wrap_angle(desired_heading - robot_yaw)
        approach_scale = float(np.clip(goal_dist / self.slow_down_radius, 0.0, 1.0))
        heading_scale = max(self.min_heading_lin_scale, math.cos(min(abs(heading_error), math.pi / 2.0)))

        lin_x = float(np.clip(goal_x_robot * approach_scale * heading_scale, -self.max_lin_vel, self.max_lin_vel))
        lin_y = float(np.clip(goal_y_robot * approach_scale * 0.3, -self.max_lat_vel, self.max_lat_vel))
        ang_z = float(np.clip(heading_error * self.heading_kp, -self.max_ang_vel, self.max_ang_vel))

        if goal_dist < self.stop_tolerance * 2.0:
            lin_x = 0.0
            lin_y = 0.0
            ang_z = 0.0

        base_cmd = np.array([lin_x, lin_y, ang_z], dtype=np.float32)
        base_cmd = np.nan_to_num(base_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        return base_cmd, {
            "heading": desired_heading,
            "heading_error": heading_error,
            "goal_dist": goal_dist,
            "bin_dist": dist_to_bin,
            "stopped": np.allclose(base_cmd, 0.0),
        }

    def _extract_policy_obs(self, obs: dict[str, Any], base_cmd: np.ndarray, obs_dim: int | None = None) -> torch.Tensor:
        proprio = torch.as_tensor(obs["proprio"], device=self.device, dtype=torch.float32)
        expected_dim = 3 + 3 + 3 + 3 + self.total_action_dim + self.total_action_dim + self.total_action_dim
        if proprio.shape[-1] != expected_dim:
            raise ValueError(f"Unexpected proprio dim: got {proprio.shape[-1]}, expected {expected_dim}")

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + self.total_action_dim]
        idx += self.total_action_dim

        joint_vel_all = proprio[:, idx:idx + self.total_action_dim]
        idx += self.total_action_dim

        actions_all = proprio[:, idx:idx + self.total_action_dim]

        joint_pos_leg = joint_pos_all[:, :self.leg_action_dim]
        joint_vel_leg = joint_vel_all[:, :self.leg_action_dim]
        actions_leg_env = actions_all[:, :self.leg_action_dim]
        actions_leg_train = actions_leg_env * self.leg_action_scale_inv.to(dtype=proprio.dtype)

        # 默认 obs_dim 使用 walking policy (>= 45, 包含 velocity commands)；
        # sit-down policy 使用较小值 (如 42)，不包含 velocity commands。
        if obs_dim is None:
            obs_dim = int(getattr(self.actor.actor[0], "in_features", 45)) if self.actor is not None else 45

        components = [
            base_ang_vel * 0.25,
            projected_gravity,
        ]
        # 仅在 walking policy 维度(>= 45) 时附加 velocity commands
        if obs_dim >= 45:
            velocity_commands = torch.as_tensor(base_cmd, device=self.device, dtype=proprio.dtype).view(1, 3)
            if proprio.shape[0] > 1:
                velocity_commands = velocity_commands.repeat(proprio.shape[0], 1)
            components.append(velocity_commands)
        components.extend([
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_leg_train,
        ])

        policy_obs = torch.cat(components, dim=-1)
        return torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}")

        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, self.total_action_dim), device=self.device, dtype=torch.float32)
        action_env[:, :self.leg_action_dim] = action_train * self.leg_action_scale
        # 机械臂动作空间使用相对默认位置的偏差（use_default_offset=True）
        # 当动作值为 0 时，机械臂会保持在环境配置中设置的默认位置
        action_env[:, self.leg_action_dim:] = 0.0
        return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)

    def _format_rgb_overlay(self, rgb: np.ndarray, nav_info: dict[str, Any], camera_name: str, detected_objects: list = None) -> np.ndarray:
        bgr = np.ascontiguousarray(rgb[..., :3][:, :, ::-1])
        phase = str(nav_info.get("phase", "nav"))
        source_camera = nav_info.get("target_source_camera") or nav_info.get("preferred_camera") or "none"
        stopped = bool(nav_info.get("stopped", False))
        target = nav_info.get("target")
        target_label = "none" if target is None else f"{target.get('class', '?')}#{target.get('id', '?')}"
        target_robot = None if target is None else target.get("pos_robot")
        target_world = None if target is None else target.get("pos_world")
        target_dist = float(nav_info.get("target_dist", 0.0))
        lines = [
            f"{camera_name} phase={phase}",
            f"target_cam={source_camera} stopped={int(stopped)}",
            f"target={target_label}",
        ]
        if target_robot is not None:
            target_robot_np = np.asarray(target_robot, dtype=np.float32).round(2).tolist()
            lines.append(f"target_robot={target_robot_np} d={target_dist:.2f}m")
        elif nav_info.get("searching"):
            lines.append("target_robot=None scanning")
        if target_world is not None:
            target_world_np = np.asarray(target_world, dtype=np.float32).round(2).tolist()
            lines.append(f"target_world={target_world_np}")
        y = 22
        for line in lines:
            cv2.putText(bgr, line, (8, y), 0, 0.5, (0, 255, 255), 2)
            y += 20

        if detected_objects is not None and cv2 is not None:
            for obj in detected_objects:
                bbox = obj.get("bbox")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    obj_class = obj.get("class", "unknown")
                    conf = obj.get("conf", 0.0)
                    is_target = target is not None and obj.get("id") == target.get("id")
                    color = (0, 0, 255) if is_target else (0, 255, 0)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                    pos_robot = obj.get("pos_robot")
                    if pos_robot is not None:
                        pos_robot_np = np.asarray(pos_robot, dtype=np.float32).round(2).tolist()
                        coord_label = f"r={pos_robot_np}"
                    else:
                        coord_label = "r=?"
                    label = f"{obj_class}: {conf:.2f}"
                    cv2.putText(bgr, label, (x1, y1 - 5), 0, 0.4, color, 1)
                    cv2.putText(bgr, coord_label, (x1, min(bgr.shape[0] - 8, y2 + 16)), 0, 0.4, color, 1)

        return bgr

    def _maybe_show_rgb_debug(self, obs: dict[str, Any], nav_info: dict[str, Any]) -> None:
        if not self.show_rgb and not self.save_rgb:
            return
        if self._step_count % self.rgb_debug_every != 0:
            return

        image_obs = obs.get("image") or {}
        detected_objects = nav_info.get("objects_detailed", [])

        try:
            head_rgb = self._tensor_to_numpy(image_obs.get("head_rgb"))
            ee_rgb = self._tensor_to_numpy(image_obs.get("ee_rgb"))
            head_depth = self._tensor_to_numpy(image_obs.get("head_depth"))
            ee_depth = self._tensor_to_numpy(image_obs.get("ee_depth"))

            head_objects = [obj for obj in detected_objects if obj.get("source_camera") == "head"]
            ee_objects = [obj for obj in detected_objects if obj.get("source_camera") == "ee"]

            if head_rgb is not None:
                head_rgb = np.asarray(head_rgb).squeeze()
                if head_rgb.ndim == 3 and head_rgb.shape[-1] >= 3:
                    head_bgr = self._format_rgb_overlay(head_rgb.astype(np.uint8), nav_info, "head", head_objects)
                    if self.show_rgb and not self._rgb_debug_failed:
                        if os.name != "nt" and not os.environ.get("DISPLAY"):
                            if not self._rgb_debug_warned:
                                self._log("[TaskB-RGB] DISPLAY not available, RGB debug disabled.")
                                self._rgb_debug_warned = True
                            self._rgb_debug_failed = True
                        else:
                            cv2.imshow("head_rgb", head_bgr)
                    if self.save_rgb:
                        head_path = os.path.join(self.save_rgb_dir, f"head_{self._step_count:06d}.png")
                        cv2.imwrite(head_path, head_bgr)

            if head_depth is not None and depth_to_vis is not None:
                head_depth = np.asarray(head_depth).squeeze()
                if head_depth.ndim == 2 or (head_depth.ndim == 3 and head_depth.shape[-1] == 1):
                    if head_depth.ndim == 3:
                        head_depth = head_depth[:, :, 0]
                    head_depth_vis = depth_to_vis(head_depth.astype(np.float32))
                    if self.show_rgb and not self._rgb_debug_failed:
                        cv2.imshow("head_depth", head_depth_vis)
                    if self.save_rgb:
                        depth_path = os.path.join(self.save_rgb_dir, f"head_depth_{self._step_count:06d}.png")
                        cv2.imwrite(depth_path, head_depth_vis)

            if ee_rgb is not None:
                ee_rgb = np.asarray(ee_rgb).squeeze()
                if ee_rgb.ndim == 3 and ee_rgb.shape[-1] >= 3:
                    ee_bgr = self._format_rgb_overlay(ee_rgb.astype(np.uint8), nav_info, "ee", ee_objects)
                    if self.show_rgb and not self._rgb_debug_failed:
                        cv2.imshow("ee_rgb", ee_bgr)
                    if self.save_rgb:
                        ee_path = os.path.join(self.save_rgb_dir, f"ee_{self._step_count:06d}.png")
                        cv2.imwrite(ee_path, ee_bgr)

            if ee_depth is not None and depth_to_vis is not None:
                ee_depth = np.asarray(ee_depth).squeeze()
                if ee_depth.ndim == 2 or (ee_depth.ndim == 3 and ee_depth.shape[-1] == 1):
                    if ee_depth.ndim == 3:
                        ee_depth = ee_depth[:, :, 0]
                    ee_depth_vis = depth_to_vis(ee_depth.astype(np.float32))
                    if self.show_rgb and not self._rgb_debug_failed:
                        cv2.imshow("ee_depth", ee_depth_vis)
                    if self.save_rgb:
                        depth_path = os.path.join(self.save_rgb_dir, f"ee_depth_{self._step_count:06d}.png")
                        cv2.imwrite(depth_path, ee_depth_vis)

            if self.show_rgb and not self._rgb_debug_failed:
                cv2.waitKey(1)
        except Exception as exc:
            if not self._rgb_debug_warned:
                self._log(f"[TaskB-RGB] debug disabled after error: {type(exc).__name__}: {exc}")
                self._rgb_debug_warned = True
            self._rgb_debug_failed = True

    @staticmethod
    def _vis_gray_image(image: np.ndarray | None, turbo: bool = False) -> np.ndarray | None:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return np.ascontiguousarray(arr)
        if arr.ndim != 2:
            return None
        vis = np.asarray(arr, dtype=np.uint8)
        if turbo and cv2 is not None:
            return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        if cv2 is not None:
            return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        return np.repeat(vis[..., None], 3, axis=-1)

    def _maybe_print_nav_debug(self, base_cmd: np.ndarray, nav_info: dict[str, Any]) -> None:
        if self._step_count % 5 != 0:
            return

        goal_dist = float(nav_info.get("goal_dist", 0.0))
        heading_error = float(nav_info.get("heading_error", 0.0))
        phase = str(nav_info.get("phase", "nav"))
        target = nav_info.get("target")
        target_id = None if target is None else target.get("id")
        target_class = None if target is None else target.get("class")
        target_dist = float(nav_info.get("target_dist", 0.0))
        stopped = bool(nav_info.get("stopped", False))
        
        # 获取机器人世界坐标
        robot_pos_world, robot_yaw, _ = self._resolve_robot_pose_world({}, {}, None)
        
        self._log(
            "[NAV] "
            f"phase={phase} "
            f"target={target_id}:{target_class} "
            f"dist={target_dist:.2f}m "
            f"err={heading_error:.2f}rad "
            f"goal={goal_dist:.2f}m "
            f"cmd=[{base_cmd[0]:.2f}, {base_cmd[1]:.2f}, {base_cmd[2]:.2f}] "
            f"robot=[{robot_pos_world[0]:.2f}, {robot_pos_world[1]:.2f}] yaw={robot_yaw:.2f}"
        )
        self._maybe_print_locked_target_gt_debug(nav_info)

    def _get_object_index(self, obj_name: str) -> int | None:
        obj_name_lower = str(obj_name).lower()
        if "object" not in obj_name_lower:
            return None
        digits = "".join(ch for ch in obj_name_lower if ch.isdigit())
        if not digits:
            return None
        obj_idx = int(digits)
        if 1 <= obj_idx <= 18:
            return obj_idx
        return None

    def _get_gt_objects(self) -> list[dict[str, Any]]:
        if self.env is None:
            return []

        try:
            scene = self._get_scene()
            if scene is None:
                return []

            robot_pos_world, robot_yaw, _ = self._resolve_robot_pose_world({}, {}, None)
            objects: list[dict[str, Any]] = []
            obj_containers = []
            if hasattr(scene, "rigid_objects"):
                obj_containers.append(scene.rigid_objects)
            if hasattr(scene, "articulations"):
                obj_containers.append(scene.articulations)

            for container in obj_containers:
                for obj_name, obj_handle in container.items():
                    obj_idx = self._get_object_index(obj_name)
                    if obj_idx is None:
                        continue
                    try:
                        if hasattr(obj_handle, "data") and hasattr(obj_handle.data, "root_pos_w"):
                            pos_world = obj_handle.data.root_pos_w.cpu().numpy()[0]
                        elif hasattr(obj_handle, "root_pos_w"):
                            pos_world = obj_handle.root_pos_w.cpu().numpy()[0]
                        else:
                            continue
                    except Exception:
                        continue

                    if obj_idx <= 6:
                        obj_class = "sugar_box"
                    elif obj_idx <= 12:
                        obj_class = "mustard_bottle"
                    else:
                        obj_class = "banana"

                    pos_world_np = np.asarray(pos_world, dtype=np.float32)
                    pos_robot_np = self._world_to_robot_frame(pos_world_np, robot_pos_world, robot_yaw)
                    objects.append({
                        "id": obj_name,
                        "class": obj_class,
                        "pos_world": pos_world_np.tolist(),
                        "pos_robot": pos_robot_np.tolist(),
                        "dist_to_robot": float(np.linalg.norm(pos_robot_np[:2])),
                    })
            return objects
        except Exception:
            return []

    def _match_gt_for_target(self, target: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(target, dict):
            return None
        target_pos = target.get("pos_world")
        if target_pos is None:
            return None

        target_pos_np = self._safe_numpy(target_pos, np.zeros(3, dtype=np.float32))
        target_class = target.get("class")
        gt_objects = self._get_gt_objects()
        if not gt_objects:
            return None

        candidates = [obj for obj in gt_objects if target_class is None or obj.get("class") == target_class]
        if not candidates:
            candidates = gt_objects

        best = min(
            candidates,
            key=lambda obj: float(
                np.linalg.norm(
                    self._safe_numpy(obj.get("pos_world"), np.zeros(3, dtype=np.float32))[:2] - target_pos_np[:2]
                )
            ),
        )
        return best

    def _maybe_print_locked_target_gt_debug(self, nav_info: dict[str, Any]) -> None:
        target = nav_info.get("target")
        if not isinstance(target, dict):
            target = self._locked_target
        if not isinstance(target, dict):
            return

        gt_target = self._match_gt_for_target(target)
        if gt_target is None:
            return

        perc_pos_w = self._safe_numpy(target.get("pos_world"), np.zeros(3, dtype=np.float32))
        gt_pos_w = self._safe_numpy(gt_target.get("pos_world"), np.zeros(3, dtype=np.float32))
        perc_pos_r = self._safe_numpy(target.get("pos_robot"), np.zeros(3, dtype=np.float32))
        gt_pos_r = self._safe_numpy(gt_target.get("pos_robot"), np.zeros(3, dtype=np.float32))

        pos_err_xy = gt_pos_w[:2] - perc_pos_w[:2]
        pos_err_norm = float(np.linalg.norm(pos_err_xy))
        perc_heading = float(math.atan2(perc_pos_r[1], perc_pos_r[0]))
        gt_heading = float(math.atan2(gt_pos_r[1], gt_pos_r[0]))
        heading_err = self._wrap_angle(perc_heading - gt_heading)

        self._log(
            "[GT-CMP] "
            f"lock={target.get('id')}:{target.get('class')} "
            f"perc_w=[{perc_pos_w[0]:.2f}, {perc_pos_w[1]:.2f}] "
            f"gt={gt_target.get('id')}:{gt_target.get('class')} "
            f"gt_w=[{gt_pos_w[0]:.2f}, {gt_pos_w[1]:.2f}] "
            f"pos_err=[{pos_err_xy[0]:.2f}, {pos_err_xy[1]:.2f}] "
            f"|err|={pos_err_norm:.2f}m "
            f"heading perc={perc_heading:.2f} gt={gt_heading:.2f} diff={heading_err:.2f}rad"
        )

    def _select_grasp_target(
        self,
        target_nav: dict[str, Any] | None,
        target_grasp: dict[str, Any] | None,
        perception_output: dict[str, Any] | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
    ) -> dict[str, Any] | None:
        if target_nav is None:
            return None

        tracked_pos = self._safe_numpy(target_nav.get("pos_world"), np.zeros(3, dtype=np.float32))
        tracked_class = target_nav.get("class")
        candidates: list[dict[str, Any]] = []

        if isinstance(perception_output, dict):
            head_candidates = self._normalize_camera_objects(
                perception_output.get("head_objects", []),
                "head",
                robot_pos_world,
                robot_yaw,
            )
            ee_candidates = self._normalize_camera_objects(
                perception_output.get("ee_objects", []),
                "ee",
                robot_pos_world,
                robot_yaw,
            )
            candidates.extend(head_candidates)
            candidates.extend(ee_candidates)

        if target_grasp is not None:
            candidates.insert(0, target_grasp)

        compatible: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if tracked_class is not None and candidate.get("class") not in {None, tracked_class}:
                continue
            candidate_pos = candidate.get("pos_world")
            if candidate_pos is None:
                continue
            candidate_pos_np = self._safe_numpy(candidate_pos, np.zeros(3, dtype=np.float32))
            if float(np.linalg.norm(candidate_pos_np[:2] - tracked_pos[:2])) <= self.target_near_match_radius * 2.0:
                compatible.append(candidate)

        if compatible:
            return min(
                compatible,
                key=lambda candidate: float(
                    np.linalg.norm(
                        self._safe_numpy(candidate.get("pos_world"), np.zeros(3, dtype=np.float32))[:2]
                        - tracked_pos[:2]
                    )
                ),
            )

        fallback = dict(target_nav)
        if fallback.get("grasp_pos_world") is None:
            fallback["grasp_pos_world"] = tracked_pos.tolist()
        return fallback

    def _policy_action_from_base_cmd(self, obs: dict[str, Any], base_cmd: np.ndarray) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(obs, base_cmd)
        with torch.inference_mode():
            action_train = self.actor(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        
        # 记录并裁剪 policy action 到合理范围 [-1, 1]
        raw_policy_action = action_train[0, :3].cpu().numpy()
        clipped_policy_action = np.clip(raw_policy_action, -1.0, 1.0)
        self._last_policy_action = clipped_policy_action
        
        return self._map_policy_action_to_env_action(action_train)

    def _ensure_leg_joint_ids(self, robot) -> list[int] | None:
        """延迟初始化 leg_joint_ids (与 ArmGraspController 中 arm_joint_ids 类似)。"""
        if getattr(self, "_leg_joint_ids_cached", None) is not None:
            return self._leg_joint_ids_cached
        if robot is None:
            return None
        try:
            leg_joint_ids, _ = robot.find_joints(list(self.leg_joint_names))
            self._leg_joint_ids_cached = list(leg_joint_ids)
            return self._leg_joint_ids_cached
        except Exception as exc:
            self._log(f"[TaskB-GRASP] leg joint ids not found: {exc}")
            return None

    def _generate_control_action_tensor(self, obs, base_cmd, robot) -> torch.Tensor:
        """
        生成环境动作张量。当 LegPostureController 不在 IDLE 时，使用其 IK 结果覆盖腿动作。
        否则回退到 walking policy。
        """
        # 先生成默认动作（使用 walking policy）
        action_env = self._policy_action_from_base_cmd(obs, base_cmd)

        # 若 LegPostureController 不在 IDLE，则用其 IK 结果覆盖腿动作
        if self._leg_posture_controller is not None and self._leg_posture_controller.state != "IDLE":
            try:
                _, target_dof_pos = self._leg_posture_controller.step(robot, self.dt)
                if target_dof_pos is None:
                    return action_env

                leg_ids = self._ensure_leg_joint_ids(robot)
                if leg_ids is None:
                    return action_env

                # 计算 leg action: (target_dof_pos - default_joint_pos) / action_scale
                # 这与 ArmGraspController.apply_to_action_tensor 的风格一致
                default_joint_pos = robot.data.default_joint_pos.to(
                    device=action_env.device, dtype=action_env.dtype
                )
                current_joint_pos = robot.data.joint_pos.to(
                    device=action_env.device, dtype=action_env.dtype
                )
                target_dof_pos = target_dof_pos.to(
                    device=action_env.device, dtype=action_env.dtype
                )
                # 使用目标位置与当前位置的差作为动作 (与 solution_gt 保持一致)
                num_envs = action_env.shape[0]
                leg_target_expanded = torch.zeros_like(action_env)
                # 将 target_dof_pos 映射到环境动作空间
                for i, env_idx in enumerate(leg_ids):
                    leg_target_expanded[:, env_idx] = target_dof_pos[:, i]
                # 计算 scaled offset: (target - default) / scale —— 保持与 arm 一致的约定
                leg_scaled = (
                    target_dof_pos - default_joint_pos[:, : self.leg_action_dim]
                ) / self.leg_action_scale.to(dtype=action_env.dtype)
                action_env[:, : self.leg_action_dim] = leg_scaled
                return torch.nan_to_num(action_env, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as exc:
                self._log(f"[TaskB-GRASP] leg posture control failed: {exc}")
                return action_env

        return action_env

    def _start_crouch_then_grasp(self, target_grasp: dict[str, Any] | None) -> bool:
        """到达目标后先蹲下，再开始抓取。"""
        if target_grasp is None:
            return False
        robot = self._get_robot()
        if robot is None:
            self._log("[TaskB-GRASP] robot unavailable, skip crouch start.")
            return False
        controller = self._ensure_arm_grasp_controller()
        if controller is None:
            self._log("[TaskB-GRASP] arm grasp controller unavailable, skip crouch start.")
            return False
        # 启动 LegPostureController: 若 sit_down_actor 存在则仅用 IK 记录目标；
        # 否则完全依赖 LegPostureController.step() 进行 IK 控制
        if self._leg_posture_controller is not None:
            try:
                self._leg_posture_controller.start_crouch(robot)
            except Exception as exc:
                self._log(f"[TaskB-GRASP] leg posture controller start_crouch failed: {exc}")
        else:
            self._log("[TaskB-GRASP] LegPostureController unavailable, will rely on sit_down_actor only.")
        self._reset_sit_down_tracking()
        self._pending_grasp_target = dict(target_grasp)
        self._task_state = "CROUCHING"
        self._log(
            "[TaskB-GRASP] start crouch "
            f"id={target_grasp.get('id')} class={target_grasp.get('class')} "
            f"grasp_pos_world={np.asarray(target_grasp.get('grasp_pos_world') or target_grasp.get('pos_world'), dtype=np.float32).round(3).tolist()}"
        )
        return True

    def _finish_stand_up_and_return_to_navigation(self, success: bool) -> None:
        """站立完成：根据抓取成功与否切换到导航或重新接近。"""
        if success and self._pending_grasp_target is not None:
            self._task_state = "NAV_TO_BIN"
            self._log("[TaskB-GRASP] stand up complete with success, switching to NAV_TO_BIN")
        else:
            self._task_state = "APPROACH_OBJECT"
            self._pending_grasp_target = None
            self._clear_locked_target()
            self._log("[TaskB-GRASP] stand up complete without success, return to APPROACH_OBJECT")

    def predicts(self, obs, current_score):
        del current_score
        try:
            return self._predicts_impl(obs)
        except Exception as exc:
            self._log(f"[TaskB-FATAL] predicts failed: {type(exc).__name__} {exc}")
            try:
                safe_action = self._policy_action_from_base_cmd(obs, np.zeros(3, dtype=np.float32))
                return {"action": safe_action.cpu().numpy().tolist(), "giveup": False}
            except Exception as exc2:
                self._log(f"[TaskB-FATAL] fallback policy also failed: {type(exc2).__name__} {exc2}")
                return {"action": [[0.0] * self.total_action_dim], "giveup": False}

    def _predicts_impl(self, obs):
        self._step_count += 1
        local_nav = self._update_local_odometry(obs)
        perception_output, robot_pos_world, robot_yaw, pose_source = self._get_perception_output(obs, local_nav)
        self._last_perception_output = perception_output
        self._print_perception_targets(perception_output, robot_pos_world, robot_yaw, pose_source)

        nav_input = None
        target_nav = None
        target_grasp = None
        if isinstance(perception_output, dict):
            target_grasp = self._prepare_pipeline_object(
                perception_output.get("target_grasp"),
                str((perception_output.get("grasp") or {}).get("camera", "head")),
                robot_pos_world,
                robot_yaw,
            )
            tracked_dist = float("inf")
            if self._tracked_target is not None:
                tracked_pos = self._safe_numpy(self._tracked_target.get("filtered_pos_world"), np.zeros(3, dtype=np.float32))
                tracked_robot = self._world_to_robot_frame(tracked_pos, robot_pos_world, robot_yaw)
                tracked_dist = float(np.linalg.norm(tracked_robot[:2]))
            if tracked_dist <= self.target_near_range:
                perception_output["phase"] = "grasp"
            nav_input, target_nav = self._adapt_perception_output(obs, local_nav, perception_output)

        base_cmd = np.zeros(3, dtype=np.float32)
        nav_info = self._make_pipeline_nav_info(perception_output, pose_source, target_nav, phase="idle", stopped=True)
        action_env: torch.Tensor | None = None

        if self._step_count <= self.stand_still_steps:
            nav_info["phase"] = "stand"
            if target_nav is not None:
                base_cmd, nav_info = self._compute_nav_cmd_from_target_nav(target_nav)
                nav_info["phase"] = str(nav_info.get("phase", "far_approach"))
                nav_info["stopped"] = bool(nav_info.get("stopped", False))
            elif self._step_count > max(4, self.stand_still_steps // 2):
                base_cmd = self._compute_search_cmd(perception_output)
                nav_info["phase"] = "search"
                nav_info["stopped"] = False
        elif self._task_state == "CROUCHING":
            robot = self._get_robot()
            controller = self._ensure_arm_grasp_controller()
            action_env = self._generate_sit_down_action_tensor(obs)
            nav_info = self._make_pipeline_nav_info(
                perception_output,
                pose_source,
                self._pending_grasp_target,
                phase="crouching",
                stopped=True,
            )
            if robot is None:
                self._log("[TaskB-GRASP] robot missing during crouch, return to approach.")
                self._task_state = "APPROACH_OBJECT"
                self._pending_grasp_target = None
                self._clear_frozen_pregrasp()
            else:
                self._sit_down_step_count += 1
                crouch_ready = False
                if self.sit_down_actor is not None:
                    # 策略驱动：基于姿态稳定判定
                    if self._is_sit_down_stable(robot):
                        self._sit_down_stable_count += 1
                    else:
                        self._sit_down_stable_count = 0
                    crouch_ready = (
                        self._sit_down_step_count >= self.sit_down_min_steps
                        and self._sit_down_stable_count >= self.sit_down_stable_steps_required
                    )
                    if crouch_ready:
                        try:
                            self._leg_posture_controller.state = "HOLDING_CROUCH"
                        except Exception:
                            pass
                else:
                    # IK 驱动：当 LegPostureController 到达 HOLDING_CROUCH 即视为就绪
                    if self._leg_posture_controller is not None:
                        crouch_ready = self._leg_posture_controller.state == "HOLDING_CROUCH"

                # 超时保护：避免永久卡住
                max_crouch_steps = max(1, int(os.getenv("ATEC_TASKB_MAX_CROUCH_STEPS", "400")))
                if self._sit_down_step_count >= max_crouch_steps and not crouch_ready:
                    self._log(f"[TaskB-GRASP] crouch timeout after {self._sit_down_step_count} steps, giving up.")
                    self._pending_grasp_status = "failed"
                    try:
                        self._leg_posture_controller.start_stand_up(robot)
                    except Exception:
                        pass
                    self._reset_sit_down_tracking()
                    self._task_state = "STAND_UP"
                    crouch_ready = False

                if crouch_ready:
                    if self._pending_grasp_target is not None and controller is not None:
                        grasp_pos_world = self._pending_grasp_target.get("grasp_pos_world") or self._pending_grasp_target.get("pos_world")
                        try:
                            current_ee_quat_w = controller.get_ee_pose()[1]
                            controller.start_grasp(
                                self._pending_grasp_target,
                                grasp_pos_world,
                                current_ee_quat_w=current_ee_quat_w,
                            )
                            self._task_state = "GRASPING"
                            self._log(
                                "[TaskB-GRASP] crouch complete, started arm grasp "
                                f"id={self._pending_grasp_target.get('id')} "
                                f"class={self._pending_grasp_target.get('class')}"
                            )
                        except Exception as exc:
                            self._log(f"[TaskB-GRASP] start_grasp failed: {exc}, standing up.")
                            self._pending_grasp_status = "failed"
                            try:
                                self._leg_posture_controller.start_stand_up(robot)
                            except Exception:
                                pass
                            self._reset_sit_down_tracking()
                            self._task_state = "STAND_UP"
                    elif controller is None:
                        self._pending_grasp_status = "failed"
                        try:
                            self._leg_posture_controller.start_stand_up(robot)
                        except Exception:
                            pass
                        self._reset_sit_down_tracking()
                        self._task_state = "STAND_UP"
                        self._log("[TaskB-GRASP] arm grasp controller unavailable after crouch, standing up.")
        elif self._task_state == "GRASPING":
            robot = self._get_robot()
            scene = self._get_scene()
            controller = self._ensure_arm_grasp_controller()
            action_env = self._generate_sit_down_action_tensor(obs)
            nav_info = self._make_pipeline_nav_info(
                perception_output,
                pose_source,
                self._pending_grasp_target,
                phase="grasping",
                stopped=True,
            )
            if controller is None or robot is None:
                self._log("[TaskB-GRASP] robot/controller unavailable during grasp, standing up.")
                if robot is not None:
                    try:
                        self._leg_posture_controller.start_stand_up(robot)
                    except Exception:
                        pass
                    self._pending_grasp_status = "failed"
                    self._reset_sit_down_tracking()
                    self._task_state = "STAND_UP"
                else:
                    self._task_state = "APPROACH_OBJECT"
                    self._pending_grasp_target = None
            else:
                done, success = controller.step(robot, scene, self.dt)
                action_env = controller.apply_to_action_tensor(action_env, robot)
                if done:
                    self._pending_grasp_status = "grasped" if success else "failed"
                    try:
                        self._leg_posture_controller.start_stand_up(robot)
                    except Exception:
                        pass
                    self._reset_sit_down_tracking()
                    self._task_state = "STAND_UP"
                    self._log(
                        f"[TaskB-GRASP] arm grasp finished, success={success}. Starting stand up."
                    )
        elif self._task_state == "STAND_UP":
            robot = self._get_robot()
            action_env = self._generate_sit_down_action_tensor(obs)
            nav_info = self._make_pipeline_nav_info(
                perception_output,
                pose_source,
                self._pending_grasp_target,
                phase="standing_up",
                stopped=True,
            )
            if robot is None:
                self._log("[TaskB-GRASP] robot missing during stand up, return to approach.")
                self._finish_stand_up_and_return_to_navigation(False)
            else:
                self._sit_down_step_count += 1
                stand_up_done = False
                if self.sit_down_actor is not None:
                    if self._is_sit_down_stable(robot):
                        self._sit_down_stable_count += 1
                    else:
                        self._sit_down_stable_count = 0
                    stand_up_done = (
                        self._sit_down_step_count >= self.sit_down_min_steps
                        and self._sit_down_stable_count >= self.sit_down_stable_steps_required
                    )
                else:
                    # IK 驱动：当 LegPostureController 回到 IDLE 视为站立完成
                    if self._leg_posture_controller is not None:
                        stand_up_done = self._leg_posture_controller.state == "IDLE"

                # 超时保护
                max_stand_steps = max(1, int(os.getenv("ATEC_TASKB_MAX_STAND_STEPS", "400")))
                if self._sit_down_step_count >= max_stand_steps and not stand_up_done:
                    self._log(f"[TaskB-GRASP] stand up timeout after {self._sit_down_step_count} steps, forcing done.")
                    stand_up_done = True

                if stand_up_done:
                    self._reset_sit_down_tracking()
                    try:
                        self._leg_posture_controller.state = "IDLE"
                    except Exception:
                        pass
                    grasp_success = self._pending_grasp_status == "grasped"
                    self._finish_stand_up_and_return_to_navigation(grasp_success)
        elif self._task_state == "NAV_TO_BIN":
            bin_nav_input = {
                "robot": {"pos_world": robot_pos_world.tolist(), "yaw": float(robot_yaw)},
                "bin": {"center_world": self.default_bin_center.tolist()},
            }
            base_cmd, nav_info = self._compute_base_cmd(bin_nav_input)
            nav_info = self._make_pipeline_nav_info(
                perception_output,
                pose_source,
                self._pending_grasp_target,
                phase="nav_to_bin",
                stopped=bool(nav_info.get("stopped", False)),
                extra=nav_info,
            )
            if float(nav_info.get("bin_dist", float("inf"))) <= self.bin_drop_radius:
                self._task_state = "RELEASE_OBJECT"
                self._release_step_count = 0
                base_cmd = np.zeros(3, dtype=np.float32)
                nav_info["phase"] = "at_bin"
                nav_info["stopped"] = True
                self._log("[TaskB-BIN] reached bin area, switching to RELEASE_OBJECT")
        elif self._task_state == "RELEASE_OBJECT":
            robot = self._get_robot()
            controller = self._ensure_arm_grasp_controller()
            action_env = self._policy_action_from_base_cmd(obs, np.zeros(3, dtype=np.float32))
            nav_info = self._make_pipeline_nav_info(
                perception_output,
                pose_source,
                self._pending_grasp_target,
                phase="release",
                stopped=True,
            )
            if controller is not None and robot is not None:
                controller.open_gripper()
                action_env = controller.apply_to_action_tensor(action_env, robot)
            self._release_step_count += 1
            if self._release_step_count >= self.release_steps:
                if controller is not None:
                    controller.reset()
                self._task_state = "APPROACH_OBJECT"
                self._pending_grasp_target = None
                self._clear_locked_target()
                self._release_step_count = 0
                self._log("[TaskB-BIN] release finished, switching back to APPROACH_OBJECT")
        else:
            self._task_state = "APPROACH_OBJECT"
            if target_nav is not None:
                base_cmd, nav_info = self._compute_nav_cmd_from_target_nav(target_nav)
                if nav_input is not None:
                    nav_info = self._attach_nav_debug_context(nav_info, nav_input, target_nav)
                else:
                    nav_info = self._make_pipeline_nav_info(
                        perception_output,
                        pose_source,
                        target_nav,
                        phase=str(nav_info.get("phase", "approach")),
                        stopped=bool(nav_info.get("stopped", False)),
                        extra=nav_info,
                    )
                perception_phase = None if perception_output is None else str(perception_output.get("phase", "approach"))
                matched_grasp_target = self._select_grasp_target(
                    target_nav,
                    target_grasp,
                    perception_output,
                    robot_pos_world,
                    robot_yaw,
                )
                cons_dist = self._conservative_target_dist(target_nav, robot_pos_world)
                dist_ok = cons_dist <= self.grasp_start_depth
                heading_ok = abs(float(nav_info.get("heading_error") or 999.0)) <= max(
                    self.object_yaw_tolerance * 3.0, 0.45,
                )
                if (
                    matched_grasp_target is not None
                    and dist_ok
                    and heading_ok
                    and (
                        nav_info.get("phase") == "ready_to_grasp"
                        or perception_phase == "grasp"
                        or cons_dist <= self.grasp_start_depth
                    )
                ):
                    base_cmd = np.zeros(3, dtype=np.float32)
                    nav_info["phase"] = "start_grasp"
                    nav_info["stopped"] = True
                    # 改为先蹲下再抓取，与 solution_gt 对齐
                    started = self._start_crouch_then_grasp(matched_grasp_target)
                    if not started:
                        # 如果无法启动蹲下流程，回退到直接抓取逻辑
                        self._log("[TaskB-GRASP] crouch start failed, fall back to direct grasp (if available).")
                        # 尝试直接启动抓取控制器；若都不可用则保持 approach
                        ctrl = self._ensure_arm_grasp_controller()
                        if ctrl is not None:
                            try:
                                grasp_pos_world = matched_grasp_target.get("grasp_pos_world") or matched_grasp_target.get("pos_world")
                                current_ee_quat_w = ctrl.get_ee_pose()[1]
                                ctrl.start_grasp(matched_grasp_target, grasp_pos_world, current_ee_quat_w=current_ee_quat_w)
                                self._pending_grasp_target = dict(matched_grasp_target)
                                # 无 sit-down actor 时直接进入 GRASPING 阶段
                                self._task_state = "GRASPING"
                                self._log("[TaskB-GRASP] direct grasp started (no sit-down).")
                            except Exception as exc:
                                self._log(f"[TaskB-GRASP] direct grasp also failed: {exc}")
                                self._task_state = "APPROACH_OBJECT"
                                self._pending_grasp_target = None
                                self._clear_locked_target()
                        else:
                            self._task_state = "APPROACH_OBJECT"
                            self._pending_grasp_target = None
                            self._clear_locked_target()
            else:
                base_cmd = self._compute_search_cmd(perception_output)
                search_phase = "search"
                if perception_output.get("nav_lock_id") is not None:
                    search_phase = "search_coast"
                elif self._fuse_lock_key is not None:
                    search_phase = "searching_lost_target"
                nav_info = self._make_pipeline_nav_info(
                    perception_output,
                    pose_source,
                    None,
                    phase=search_phase,
                    stopped=False,
                    extra={"goal_dist": 0.0, "heading_error": 0.0, "target_dist": 0.0},
                )

        if action_env is None:
            action_env = self._policy_action_from_base_cmd(obs, base_cmd)

        self._last_base_cmd = base_cmd
        self._last_nav_info = nav_info
        self._maybe_print_nav_debug(base_cmd, nav_info)
        self._maybe_show_rgb_debug(obs, nav_info)
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}


if __name__ == "__main__":
    solution = AlgSolution()
    dummy_obs = {
        "proprio": torch.zeros(1, 72, dtype=torch.float32),
        "image": {
            "head_rgb": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "head_depth": torch.ones(1, 480, 640, 1, dtype=torch.float32),
            "ee_rgb": torch.zeros(1, 480, 640, 3, dtype=torch.uint8),
            "ee_depth": torch.ones(1, 480, 640, 1, dtype=torch.float32),
        },
    }
    try:
        result = solution.predicts(dummy_obs, current_score=0.0)
        solution._log(f"action_dim {len(result['action'][0])}")
        solution._log(f"giveup {result['giveup']}")
    except Exception as exc:
        solution._log(f"{type(exc).__name__} {exc}")