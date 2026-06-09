"""
ATEC 2026 Task B 感知层 — 配置常量
包含相机参数、场景常量、物体尺寸表等所有硬编码参数

基于 Isaac Lab v2.3.2 中 B2-Piper 的传感器/场景配置精确提取
"""

import numpy as np

# =============================================================================
# 相机内参 (从 Isaac Lab UsdCameraCfg 计算)
# =============================================================================
# 传感器尺寸: horizontal_aperture=20.955mm, vertical=15.716mm (4:3)
# focal_length_px = focal_length_mm * resolution / aperture_mm
#
# head_camera:  focal_length=24.0mm, 640×480
# ee_camera:    focal_length=15.0mm, 640×480

HEAD_CAM = {
    "width": 640,
    "height": 480,
    "fx": 733.27,   # 24.0 * 640 / 20.955
    "fy": 733.27,   # 24.0 * 480 / 15.716
    "cx": 320.0,
    "cy": 240.0,
}

# 便捷别名 (test_offline.py / generate_synthetic_dataset.py 使用)
IMG_W = HEAD_CAM["width"]
IMG_H = HEAD_CAM["height"]
HEAD_CAM_MATRIX = np.array([
    [HEAD_CAM["fx"], 0, HEAD_CAM["cx"]],
    [0, HEAD_CAM["fy"], HEAD_CAM["cy"]],
    [0, 0, 1],
], dtype=np.float32)

EE_CAM = {
    "width": 640,
    "height": 480,
    "fx": 458.29,   # 15.0 * 640 / 20.955
    "fy": 458.29,
    "cx": 320.0,
    "cy": 240.0,
}

# =============================================================================
# 相机外参 — 相对机器人基座的位姿 (从 b2.py: UsdCameraCfg offset + target)
# =============================================================================
# head_camera 装于 B2 头顶:
#   offset=(0.422, 0.025, 0.062), pitch=30° 朝下
# ee_camera 装于 gripper_base:
#   offset=(-0.05, 0, 0.06), yaw=-90°

HEAD_CAM_POS_ROBOT = np.array([0.422, 0.025, 0.062], dtype=np.float32)

# 相机俯仰角 (度): 负值=朝下
# Isaac Lab b2.py 中 UsdCameraCfg 的 pitch=30° 朝下
HEAD_CAM_PITCH_DEG = -30.0  # 离线回退路径；采集时请用仿真 head_camera 真实位姿
HEAD_CAM_PITCH_RAD = np.deg2rad(HEAD_CAM_PITCH_DEG)

EE_CAM_POS_ROBOT = np.array([-0.05, 0.0, 0.06], dtype=np.float32)

# EE 相机旋转矩阵 (预计算, cam ↔ robot)
# ee_camera 装于夹爪, yaw=-90°: 光轴指向机器人右侧
def _build_ee_cam_rotation():
    """EE cam: 绕 robot Z 旋转 -90°, 光轴→robot_Y(右)"""
    # cam → robot: robot_X=-cam_X, robot_Y=cam_Z, robot_Z=-cam_Y
    cam2robot = np.array([
        [-1,  0,  0],
        [ 0,  0,  1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    # robot → cam (逆)
    robot2cam = np.array([
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
    ], dtype=np.float32)
    return cam2robot, robot2cam

EE_CAM_ROT_MATRIX, EE_CAM_ROT_MATRIX_INV = _build_ee_cam_rotation()

# =============================================================================
# 相机旋转矩阵 (预计算，避免重复运算)
# =============================================================================
# 坐标系约定:
#   - 机器人坐标系: X=前, Y=右, Z=上
#   - OpenCV 相机坐标系: X=右, Y=下, Z=前(光轴)
#
# 无反投影变换链:
#   1. 轴交换 (R_no_tilt): cam[X=右,Y=下,Z=前] → robot[X=前,Y=右,Z=上]
#      robot_X = cam_Z, robot_Y = cam_X, robot_Z = -cam_Y
#   2. 俯仰旋转 (R_pitch): 绕 cam X 轴旋转 HEAD_CAM_PITCH_RAD
#      (负角度=朝下俯仰)
#
#   cam → robot:  p_robot_offset = R_no_tilt @ R_pitch @ p_cam
#   robot → cam:  p_cam = R_pitch.T @ R_no_tilt.T @ p_robot_offset
#
# _build_head_cam_rotation() 预计算这两个矩阵

def _build_head_cam_rotation(pitch_rad: float):
    """
    构建 head camera 的旋转矩阵 (cam ↔ robot)

    Args:
        pitch_rad: 俯仰角 (弧度), 负值=朝下

    Returns:
        cam2robot: (3,3) 相机坐标系 → 机器人坐标系
        robot2cam: (3,3) 机器人坐标系 → 相机坐标系 (OpenCV 约定)
    """
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)

    # 轴交换矩阵: OpenCV cam → robot (无俯仰)
    # robot_X=cam_Z, robot_Y=cam_X, robot_Z=-cam_Y
    R_no_tilt = np.array([
        [0,  0,  1],
        [1,  0,  0],
        [0, -1,  0],
    ], dtype=np.float32)

    # 俯仰旋转矩阵 (绕 cam X 轴): R_x(θ)
    # R_x(θ) = [[1, 0, 0], [0, cosθ, -sinθ], [0, sinθ, cosθ]]
    # 注: 俯仰角 θ 为负值时, sinθ 为负, 光轴 cam_Z 朝下旋转
    R_pitch = np.array([
        [1,  0,   0],
        [0, cp, -sp],
        [0, sp,  cp],
    ], dtype=np.float32)

    # cam → robot: 先俯仰(在cam系中旋转), 再轴交换
    cam2robot = (R_no_tilt @ R_pitch).astype(np.float32)

    # robot → cam: 先逆轴交换, 再逆俯仰
    robot2cam = (R_pitch.T @ R_no_tilt.T).astype(np.float32)

    return cam2robot, robot2cam


HEAD_CAM_ROT_MATRIX, HEAD_CAM_ROT_MATRIX_INV = _build_head_cam_rotation(HEAD_CAM_PITCH_RAD)

# =============================================================================
# 场景常量 (从 terrain.py / env_cfg.py)
# =============================================================================
# 地形: 20m×20m 平地
# 机器人初始位姿
ROBOT_INIT_POS = np.array([-10.0, -10.0, 0.68], dtype=np.float32)
ROBOT_INIT_YAW = 0.0

# 评分虚拟圈 (奖励函数 ObjectsInCircle 判分位置)
BIN_CENTER = np.array([-3.0, -10.0, 0.0], dtype=np.float32)  # 世界坐标
BIN_RADIUS = 1.0                                               # 有效半径 (m)
BIN_Z_MIN = 0.0                                                # 物体 z 下限 (在地面)
BIN_Z_MAX = 0.5                                                # 物体 z 上限 (不能悬空)

# 物体初始分布范围
OBJ_SPAWN_X_RANGE = (-15.0, -5.0)
OBJ_SPAWN_Y_RANGE = (-15.0, -5.0)

# =============================================================================
# 物体类别映射 (YOLO class index ↔ 名称)
# =============================================================================
CLASS_NAMES = {
    0: "sugar_box",
    1: "mustard_bottle",
    2: "banana",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = len(CLASS_NAMES)
TOTAL_OBJECTS = 18  # 3 类 × 6 个

# =============================================================================
# 物体物理尺寸 (基于 YCB 实物 + Isaac Lab USD 模型尺度)
# 单位: 米, 用于抓取规划去夹爪开合控制
# =============================================================================
# YCB 实物尺寸 (米), 与 task_b USD 模型 1:1 尺度; 用于自动标注 bbox
# 旧值偏大 ~2x, 会导致框明显套不准
OBJECT_SIZES = {
    "sugar_box":       {"lx": 0.175, "ly": 0.098, "lz": 0.039},   # 004_sugar_box
    "mustard_bottle":  {"lx": 0.189, "ly": 0.082, "lz": 0.057},   # 006_mustard_bottle
    "banana":          {"lx": 0.195, "ly": 0.040, "lz": 0.040},   # 011_banana
}

# 自动标注在投影框外再留一点边 (YOLO 训练更稳)
LABEL_BBOX_MARGIN = 1.12

# 默认尺寸（未知物体回退）
DEFAULT_OBJECT_SIZE = {"lx": 0.15, "ly": 0.10, "lz": 0.10}

# =============================================================================
# 抓取姿态定义 (固定部分 — "从哪个轴抓取")
# 所有物体在地面，从上往下抓 (approach = -Z_world)
# 固定旋转描述的是物体自身轴到夹爪 approach 轴的变换
# 操作层会在此基础上叠加机器人接近方向的 yaw 旋转
# =============================================================================
# grasp_fixed_quat: 物体默认姿态下的抓取四元数 (w,x,y,z) scalar-first
# sugar 初始 pose: rot=(0, 0.707, 0, 0.707) → 绕 Y 90°
# mustard 初始 pose: rot=(0, 0, -0.707, 0.707) → 绕 Z -90°
# banana 初始 pose: rot=(0, 0, -0.707, 0.707) → 绕 Z -90°
# 这些在 object.py init_state 中定义

# 固定抓取四元数（物体坐标系下的抓取姿态）
# sugar: 盒体平放，夹爪水平从侧面抓 → 绕 Y 旋转使夹爪前倾
# mustard/banana: 竖立，从上往下抓 → 无需固定旋转 (identity, 夹爪自然向下)
GRASP_FIXED_QUAT = {
    "sugar_box":       np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float32),   # 绕 Y 90°
    "mustard_bottle":  np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),       # identity
    "banana":          np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),       # identity
}
DEFAULT_GRASP_FIXED_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# 抓取深度: 从物体顶面向下偏移量 (m), 确保夹爪啮合物体
GRASP_DEPTH_OFFSET = 0.03

# =============================================================================
# 夹爪参数 (用于 is_holding 判定)
# =============================================================================
GRIPPER_OPEN_WIDTH = 0.08       # 完全张开宽度 (m)
GRIPPER_CLOSE_WIDTH = 0.0       # 完全闭合宽度
GRIPPER_HOLDING_MAX_WIDTH = 0.04  # 宽度小于此值认为在夹持

# =============================================================================
# ByteTrack 参数
# =============================================================================
TRACK_MAX_AGE = 30              # 多少帧未匹配后丢弃 track
TRACK_MIN_HITS = 3              # 连续匹配多少帧后才确认 track
TRACK_IOU_THRESHOLD = 0.3       # IoU 匹配阈值

# =============================================================================
# YOLO 检测参数
# =============================================================================
YOLO_CONF_THRESHOLD = 0.15      # 置信度阈值 (默认0.25, 降低以减少漏检)

# =============================================================================
# Proprioception 索引 (B2-Piper 20 自由度)
# =============================================================================
# proprio 布局 (72 维):
#   [0:3]   base_lin_vel
#   [3:6]   base_ang_vel
#   [6:9]   velocity_cmd
#   [9:12]  projected_gravity
#   [12:32] joint_pos (20)  — 12 腿 + 8 臂
#   [32:52] joint_vel (20)
#   [52:72] last_action (20)
#
# 关节顺序 (b2.py):
#   12 个腿部关节: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#                  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
#   8 个臂部关节:  arm_joint1..arm_joint6 (臂), arm_joint7 (左指), arm_joint8 (右指)

PROPRIO_JOINT_POS_START = 12   # joint_pos 起始索引
PROPRIO_JOINT_POS_LENGTH = 20  # joint_pos 长度 (12 leg + 8 arm)
PROPRIO_ARM_START = 24         # arm_joint1 在 joint_pos 中的索引 (=12+12)
PROPRIO_FINGER_LEFT = 30       # arm_joint7 (=12+18)
PROPRIO_FINGER_RIGHT = 31      # arm_joint8 (=12+19)

PROPRIO_BASE_LIN_VEL = slice(0, 3)
PROPRIO_BASE_ANG_VEL = slice(3, 6)
PROPRIO_PROJECTED_GRAVITY = slice(9, 12)

# 里程计融合权重: yaw 角度 = alpha*重力反算 + (1-alpha)*角速度积分
PROPRIO_YAW_FUSION_ALPHA = 0.8  # 重力反算权重 (0~1), 越大越信任重力