#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle（开挂）专家数据采集脚本 —— 桌面 Pick-and-Place，为 ACT 训练收集专家轨迹。

架构总览
--------
    main / OracleCollector         episode 主循环：重置 -> 随机摆位 -> 依次抓放 -> 落盘
      ├─ EpisodeRandomizer          每个 episode 把红/黄/绿在安全抓取区内随机摆放（互不干涉）
      ├─ ArmModel                  按名字解析 6 个手臂关节 / 执行器 / TCP site（不写死下标）
      ├─ OracleIKController        阻尼最小二乘 IK（mj_jacSite），输出关节目标
      ├─ WaypointPickPlace         门字形路点状态机（8 阶段），一拍输出一个 action
      ├─ CameraCapture             两路 RGB 取帧（走 OrcaStudio 推流，帧与 action 同帧同步）
      └─ HDF5Recorder              每个 episode 一个独立 hdf5，50Hz 同步落盘 obs/* 与 action

单个方块的门字形阶段（严格按序）：
    移动到方块正上方 -> 垂直下压 -> 闭合夹爪 -> 垂直抬升
    -> 平移至放置点正上方 -> 垂直下压 -> 张开夹爪 -> 垂直抬升复位
抓取顺序恒为 红 -> 黄 -> 绿；放置点固定为桌面右下角区域的三个错开点。

执行约定
--------
    物理跑在本进程（orca_gym 的 OrcaGymLocalEnv），OrcaStudio 只负责可视化。
    所以必须先在 OrcaStudio 里点「启动」（50051 的 gRPC 服务才会起来），再跑本脚本。
    一个控制周期 = FRAME_SKIP 个物理步 = 20 ms -> 50 Hz，录制频率与之对齐。

用法
----
    # 1) 手动跑（先在 OrcaStudio 里进入运行模式，或让 OrcaLab 用「Empty Loop」启动外部程序）
    #    默认按 50Hz 实时播放，编辑器视口能看到机械臂动起来
    /home/user/miniconda3/envs/orcalab/bin/python oracle_collect.py --episodes 5

    # 2) 只跑通抓放、先不录图像（相机没配好 / 只想验证抓放时用这个）
    /home/user/miniconda3/envs/orcalab/bin/python oracle_collect.py --episodes 20 --no-cameras

    # 3) 批量数采想尽快跑完、不用看画面时加 --fast 关掉实时节流
    /home/user/miniconda3/envs/orcalab/bin/python oracle_collect.py --episodes 50 --fast

    # 4) 先探相机：不跑 episode，只打印后端注册的相机与各路推流端口
    /home/user/miniconda3/envs/orcalab/bin/python oracle_collect.py --episodes 0 --diag

    # 5) 作为 OrcaLab 的外部程序自动启动（推荐，点「启动」即开始数采）：
    #    在 .orcalab/config.toml 的 [[external_programs.programs]] 里加一条，
    #    command 指向 orcalab 环境的 python，args 指向本文件，并把 default 改成它的 name。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubes.randomize_cubes import (  # noqa: E402
    CUBE_Z,
    find_cube_free_joints,
    random_quat,
    sample_positions,
    world_offset_from_model,
)

# ============================== 全局配置 ==============================

DEFAULT_ADDR = "localhost:50051"
FRAME_SKIP = 20                      # 一个控制周期 = 20 个物理步
TIME_STEP = 0.001                    # 物理积分步长
CTRL_DT = FRAME_SKIP * TIME_STEP     # 0.02 s -> 50 Hz

TCP_SITE_KEY = "gripper_tcp"         # 抓取中心
GRIP_ACT_KEY = "gripper_act"         # 夹爪执行器
# 2FG7 是位置执行器：ctrl = 两指行程之和（fixed tendon: left_slide + right_slide），
# 指垫净开口随 ctrl 严格线性（实测）：
#     净开口 = GRIP_GAP_CLOSED + ctrl   （ctrl=0 -> 26.8mm 全闭，ctrl=0.038 -> 64.8mm 全开）
GRIP_GAP_CLOSED = 0.0268             # 全闭时的指垫净开口（米）
GRIP_GAP_OPEN = 0.0648               # 全开时的指垫净开口（米）
GRIP_CTRL_MAX = 0.038                # ctrl 上限 = 两指行程之和
# obs / action 的夹爪通道统一用「净开口（米）」表示，同量纲，方便 ACT 直接对齐。
GRIP_OPEN, GRIP_CLOSE = GRIP_GAP_OPEN, GRIP_GAP_CLOSED
# 过盈配合（硬挤压）：合爪目标开度 = 方块宽度 - GRASP_SQUEEZE，故意让位置伺服去够一个
# 比方块更窄的开度。方块挡住手指后，伺服误差被压成法向夹持力：
#     F ≈ kp * GRASP_SQUEEZE      （kp = 夹爪位置伺服的刚度，见 GRIP_ACT_KP_TARGET）
# 取 0.005 时方块不会被挤飞（软接触 solimp 0.95/0.99/0.001 会兜住），夹持力也已足够。
# 离线 16 seeds 对照：过盈 5 / 15 mm 都是 16/16，而把 kp / forcerange 放大 10 倍会掉到
# 12~13/16 —— 破坏成功率的是电机力量，不是过盈量，所以电机保持原厂值不动。
GRASP_SQUEEZE = 0.005
# 夹爪电机的力量：位置伺服刚度 kp 与力矩上限 forcerange 的「保底目标值」。
# 模型里的值低于它就在运行时抬高到目标值（幂等，已够大则不重复放大）。
# 保持 2FG7 原厂值：kp=2000 N/m、±60 N，实测比放大 10 倍更稳。
GRIP_ACT_KP_TARGET = 2000.0      # 2FG7 原厂位置伺服刚度（N/m）
GRIP_ACT_FORCE_TARGET = 60.0     # 2FG7 原厂力矩上限（N）

# 下压前的横向对准：UR5e 的位置伺服在重力下带 5~7 mm 稳态误差，approach 结束时 TCP
# 会偏在方块中心旁边 5~7 mm，而夹爪全开只有 7.4 mm 单边余量 —— 指垫会刮到方块上角把
# 方块掀翻（离线实测 7/16 的 seed 被掀到 19°~39°），合爪时就只能夹住顶角、搬运中滑落。
# 这里用积分把该误差补进目标 XY，让实际 TCP 落到方块正上方。
ALIGN_GAIN = 1.0                 # 横向对准积分增益（1.0 一拍即收敛）
ALIGN_MAX = 0.03                 # 修正量上限（m）
ALIGN_GATE = 0.03                # 距离目标 3 cm 以内才开始积分（避免远距离乱积）


def grip_width_for_ctrl(ctrl: float) -> float:
    """夹爪位置指令 ctrl(0..0.038) -> 指垫净开口（米）。"""
    return GRIP_GAP_CLOSED + float(ctrl)


def grip_ctrl_for_width(width: float) -> float:
    """希望指垫夹到的净开口（米）-> 夹爪位置指令 ctrl；超出行程会被夹到端点。"""
    return float(np.clip(width - GRIP_GAP_CLOSED, 0.0, GRIP_CTRL_MAX))

PICK_ORDER = ("red", "yellow", "green")      # 严格按 红 -> 黄 -> 绿

# 桌面「右下角」放置区（XML 坐标）：三个略微错开的放置点，避免放下时互相碰撞弹飞
PLACE_CENTER = (-1.05, 0.05)
PLACE_OFFSETS = ((-0.09, 0.0), (0.0, 0.0), (0.09, 0.0))

# 放置区白色托盘的底板厚度：方块是落在托盘底板上的，比直接落在桌面高这么多。
# 与 ur5e_table.xml 里 tray_floor 的 2*size_z 保持一致（底板顶面高出桌面 7 mm）。
TRAY_FLOOR_DZ = 0.007

APPROACH_H = 0.12        # 方块正上方接近高度（相对方块中心）
LIFT_H = 0.20            # 抬升 / 撤离高度
GRASP_DZ = 0.010         # 抓取 / 下放时 TCP 相对方块中心的高度偏移（正 = 抓点略高于质心）。
                         # 几何窗口（方块 50 mm、指垫碰撞体 TCP±19 mm、夹爪本体下沿 TCP+25.6 mm）：
                         #   · 指垫下沿 TCP-19 不能低于方块底面（cz-25）      -> dz > -6 mm
                         #   · 夹爪本体下沿 TCP+25.6 不能插进方块顶面（cz+25）-> dz > -0.6 mm
                         #   · 指垫要基本贴住方块侧面（上沿 TCP+19 <= cz+25） -> dz <= +12 mm
                         # 实测 dz=0 时位置伺服重力下沉 1.4 mm，本体下沿正好插进方块顶面 0.8 mm，
                         # 整个搬运过程以 10~27 N 压着方块顶面，张爪瞬间把方块弹飞/翻 90°。
                         # 取 +10 mm 让本体下沿留出 ~10.6 mm 名义间隙，指垫仍贴合 34/50 mm 侧面
                         # （摩擦系数 10，不会打滑）。
                         # 注意本值同时用于 descend（抓取点）和 lower（下放点 = place_z + GRASP_DZ
                         # + LOWER_DZ）：方块被夹住后「块中心 - TCP」的偏移是冻结的（= sag - GRASP_DZ），
                         # 与 GRASP_DZ 相消，所以改这个值不会改变方块最终的落盘高度。
LOWER_DZ = 0.0015        # 放置时在「方块贴盘高度」之上再留一点余量
# 合爪/张开的保持拍数：夹爪是刚度很高的位置伺服，实测一步（20 ms）就能从全开 64.8 mm
# 合到方块宽 50.0 mm；空手合到底也只要 7 拍。原来 60 / 15 拍（1.2 s / 0.3 s）里
# 绝大部分是「手臂完全不动、夹爪也不动」的纯静止帧，属于无信息的观测别名来源。
CLOSE_HOLD_STEPS = 10    # 合爪后保持 10 拍（0.2 s）：覆盖空手合到底的 7 拍余量
OPEN_HOLD_STEPS = 5      # 张开后保持 5 拍（0.1 s）：实测 1 拍就张到 64.8 mm
CARRY_STAGES = ("lift", "transfer", "lower")   # 手上有方块的阶段，必须慢
STAGE_MAX_STEPS = 250    # 单阶段最多走多少拍，超了就带着残差进入下一阶段（防止卡死）
# 直线阶段走完后的 settle 上限：位置伺服在重力下的稳态误差会让 ready()（reached + settled）
# 长期不成立，原来只能空等到 STAGE_MAX_STEPS=250 —— 实测每段白等 ~140 拍（占全集 28%）
# 手臂一动不动、而学界要求的下一步动作又完全不同，制造了大量「观测相同、动作矛盾」的样本。
# 直线本身已把 TCP 精确解算到目标点，这里只留几拍闭环压残差就切阶段。
SETTLE_MAX_STEPS = 5

# ---- 抓偏空抓（Miss Grasp）抢救 ----
# 实测物理摩擦很稳、基本没出现过「手滑脱落」，真正的失败模式是「抓偏」：TCP 没对准方块，
# 夹爪在空气里合拢。这里以 MISS_PROB 的概率故意给抓取目标 XY 叠一个 MISS_OFF_MIN~MAX 的
# 随机偏移，制造空抓，逼专家演示「发现抓空 -> 原地垂直抬起 -> 张爪 -> 重读方块位姿 ->
# 回到 approach 精准二次抓取」的完整纠错动作，供 ACT 学习 retry。
MISS_PROB = 0.10         # 每次抓取注入抓偏噪声的概率（可用 --miss-prob 覆盖）
MISS_OFF_MIN = 0.03      # 抓偏偏移量下限（m）
MISS_OFF_MAX = 0.05      # 抓偏偏移量上限（m）
MISS_GAP_EPS = 0.0025    # 净开口 <= grip_hold + EPS 视为「夹爪合到了指令位置、手里是空气」
MISS_Z_EPS = 0.005       # 方块 Z 相对合爪前变化 <= 5 mm 视为「高度未变」（没被拎起来）
MISS_MAX_RETRIES = 2     # 每个方块最多允许的抢救重试次数，超过则本集作废
KNOCK_TILT_DEG = 20.0    # 方块倾角超过该值视为「被碰倒」-> 本集作废
# 抓偏噪声下的下压深度：只降到方块顶面之上再合爪，保证「抓空但不碰倒方块」。
# 夹爪全开净开口 64.8 mm（单边 32.4 mm），而方块只有 50 mm（半宽 25 mm）——3~5 cm 的
# 横向偏移下，若按正常抓取深度下压到底，靠内那根手指会压到方块顶角，实测直接把方块
# 掀翻 90°。所以抓偏这一次不下压到底，只停在方块上方再合爪，在空气里合拢 -> 干净的抓空。
MISS_DESCEND_DZ = 0.075

# 笛卡尔绝对直线：descend（下压到抓取点）/ lift（垂直抬升）/ transfer（平移搬运）/
# lower（下放到托盘）/ retreat（放完垂直撤离）五个阶段。末端目标不再用「闭环 DLS 单点靠近」
# （那会走出弧线、产生横向超调），而是先解算出一条严格直线的密集关节序列，再逐拍下发，
# 保证 TCP 走绝对直线。
# retreat 必须走直线：它的 xy 目标就是当前 TCP 的 xy（见 _target），是一条纯垂直上升线。
# 若走闭环支路，一是速度快 4 倍（IK_DQ_MAX 0.20 rad/拍 → 实测 TCP 0.65 m/s，手指还挂在
# 方块顶角上就把它抽飞），二是 DLS 满增益会把 y 向目标甩过头 5 mm 以上，张开的手指横向
# 蹭到方块把它掀翻（seed 4 yellow 实测被拖走 34 mm、翻到 63°）。
LINE_STAGES = ("descend", "lift", "transfer", "lower", "retreat")
LINE_STEP = 0.003        # 直线插值间距（m）：越小越贴合直线，但解算和拍数越多
LINE_MIN_STEPS = 5       # 至少插几个路点

IK_ALPHA = 1.0           # IK 每拍步长系数
IK_DQ_MAX = 0.20         # 单拍关节增量上限（rad）——空载移动用
IK_DQ_MAX_CARRY = 0.05   # 夹着方块移动时的单拍关节增量上限（rad）——太快方块会被甩掉
                         # 注意：UR5e 位置伺服是重阻尼的（biasprm 的 -400*qvel），
                         # 关节最大速度 ≈ 5 × 目标误差，所以 DQ_MAX 直接决定运动快慢。
IK_LAMBDA = 1e-2         # DLS 阻尼
IK_POS_TOL = 0.012       # 位置到位阈值（m）
IK_ROT_TOL = 0.06        # 姿态到位阈值（rad）
IK_VEL_TOL = 0.05        # 关节速度阈值（rad/s）：必须基本停稳才能进入下一阶段

IMG_SIZE = 224
CAMERA_TOP, CAMERA_WRIST = "camera_top", "camera_wrist"

# 相机：obs 里的 key -> OrcaStudio 场景里注册的相机名。
# 取帧走官方推流链路（env.get_camera_names -> env.start_streaming -> SingleFrameTask），
# 端口由后端属性给出，不在这里写死。可用 --cameras 覆盖本映射。
CAM_TARGETS = ((CAMERA_TOP, "camera_float"), (CAMERA_WRIST, "gripper_camera"))
# start_streaming 必须显式 capture_rgb=True，否则后端默认不起彩色流，
# get_recorder 会一直返回 None（现象是「连上了但永远没有帧」）。
CAM_STREAM_KWARGS = {"capture_rgb": True}
CAM_MAX_BUFFER_FRAMES = 2048     # 每路相机的滚动缓存帧数（默认 36000 太吃内存）
CAM_FLUSH_TIMEOUT = 30.0         # 一个 episode 结束后等齐所有帧的总超时（秒）
CAM_WAIT_FIRST_FRAME = 20.0      # 开流后等首帧的超时（秒）；超时只告警不中断

# ACT 的语言指令：写进每个 episode 的 attrs，供训练时做语言条件（不用也可留空）
TASK_INSTRUCTION = "pick up the cube and place it into the tray"


# ============================== 模型名字工具 ==============================

_COUNT_ATTR = {
    mujoco.mjtObj.mjOBJ_BODY: "nbody",
    mujoco.mjtObj.mjOBJ_JOINT: "njnt",
    mujoco.mjtObj.mjOBJ_ACTUATOR: "nu",
    mujoco.mjtObj.mjOBJ_SITE: "nsite",
    mujoco.mjtObj.mjOBJ_GEOM: "ngeom",
    mujoco.mjtObj.mjOBJ_CAMERA: "ncam",
}


def names_of(m: mujoco.MjModel, obj_type) -> list[str]:
    """列出某类对象的所有名字（可能带 prefab 前缀）。"""
    n = getattr(m, _COUNT_ATTR[obj_type])
    return [(mujoco.mj_id2name(m, obj_type, i) or "") for i in range(n)]


def find_suffix(m: mujoco.MjModel, obj_type, suffix: str, tag: str) -> int:
    """按名字后缀唯一确定一个对象。"""
    hits = [i for i, n in enumerate(names_of(m, obj_type)) if n.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError("无法唯一确定 %s（后缀 %r 匹配到 %d 个）" % (tag, suffix, len(hits)))
    return hits[0]


def find_contains(m: mujoco.MjModel, obj_type, key: str) -> list[int]:
    """按名字子串匹配（大小写不敏感）。"""
    key = key.lower()
    return [i for i, n in enumerate(names_of(m, obj_type)) if key in n.lower()]


def to_chw_224(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (3,224,224) uint8（最近邻缩放，避免引入额外依赖）。"""
    rgb = np.asarray(rgb)[:, :, :3]
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        ih = np.linspace(0, rgb.shape[0] - 1, IMG_SIZE).astype(np.int32)
        iw = np.linspace(0, rgb.shape[1] - 1, IMG_SIZE).astype(np.int32)
        rgb = rgb[ih][:, iw]
    return np.transpose(rgb, (2, 0, 1)).astype(np.uint8)


# ============================== 机器人模型索引 ==============================

class ArmModel:
    """把「名字 -> 下标」集中解析一次。

    注意：OrcaStudio 导出到 Python 侧的模型里，关节顺序不一定与 MJCF 一致
    （方块 free joint 可能排在手臂关节前面），所以一律按名字查，不写死 0..5。
    """

    ARM_JOINT_SUFFIX = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    ARM_ACT_SUFFIX = ("shoulder_pan", "shoulder_lift", "elbow",
                      "wrist_1", "wrist_2", "wrist_3")

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m, self.d = m, d
        jids = [find_suffix(m, mujoco.mjtObj.mjOBJ_JOINT, s, s) for s in self.ARM_JOINT_SUFFIX]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in jids], dtype=int)
        rng = np.array([m.jnt_range[j] for j in jids])
        self.jnt_lo = rng[:, 0]
        self.jnt_hi = np.where(rng[:, 1] - rng[:, 0] > 1e-6, rng[:, 1], 1e9)

        self.act_ids = [find_suffix(m, mujoco.mjtObj.mjOBJ_ACTUATOR, s, s)
                        for s in self.ARM_ACT_SUFFIX]
        self.grip_act = find_suffix(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACT_KEY, GRIP_ACT_KEY)
        self.tcp_site = find_suffix(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_KEY, TCP_SITE_KEY)

        drv = find_contains(m, mujoco.mjtObj.mjOBJ_JOINT, "slide")
        self.grip_qpos_adr = [int(m.jnt_qposadr[j]) for j in drv]   # 两指 slide 关节
        self.grip_dof_adr = [int(m.jnt_dofadr[j]) for j in drv]

    def arm_qpos(self) -> np.ndarray:
        return self.d.qpos[self.qpos_adr].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.tcp_site].copy()

    def grip_state(self) -> np.ndarray:
        """1 维夹爪状态：指垫当前净开口（米）＝ GRIP_GAP_CLOSED + 两指行程之和。

        与 action 里的夹爪目标开度同量纲，ACT 可直接在 obs/action 间对齐。
        """
        if not self.grip_qpos_adr:
            return np.array([GRIP_GAP_CLOSED], dtype=np.float32)
        travel = float(np.sum(self.d.qpos[self.grip_qpos_adr]))
        # 位置伺服在快速开合瞬间会冲过软限位，读数会短暂越界，压回物理行程内
        width = grip_width_for_ctrl(travel)
        return np.array([np.clip(width, GRIP_GAP_CLOSED, GRIP_GAP_OPEN)], dtype=np.float32)

    def boost_gripper_force(self,
                            kp_target: float = GRIP_ACT_KP_TARGET,
                            force_target: float = GRIP_ACT_FORCE_TARGET) -> tuple[float, float]:
        """把夹爪电机的力量抬到「保底目标值」以上（幂等）。

        2FG7 在模型里是 `<position>` 位置伺服，出力 = kp * (ctrl - 实际行程)：
            · kp 太低  -> 方块一挡就「憋停」，法向力上不去；
            · forcerange 太小 -> 力被硬性截断，同样捏不紧。
        这里直接改 MjModel 的 actuator 数组（MuJoCo 每步都会重新读，运行时生效），
        并且只往上抬、不往下压，所以重复调用不会叠加放大。

        Returns:
            (kp, forcerange 上限) 抬升后的实际值。
        """
        a = self.grip_act
        if self.m.actuator_gainprm[a, 0] < kp_target:
            scale = kp_target / float(self.m.actuator_gainprm[a, 0])
            self.m.actuator_gainprm[a, 0] *= scale
            self.m.actuator_biasprm[a, 1] *= scale    # biasprm[1] = -kp，必须同比放大
        if self.m.actuator_forcerange[a, 1] < force_target:
            self.m.actuator_forcerange[a] = [-force_target, force_target]
        return float(self.m.actuator_gainprm[a, 0]), float(self.m.actuator_forcerange[a, 1])

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.jnt_lo, self.jnt_hi)


# ============================== 抓取位姿（6D：位置 + 姿态） ==============================

def wrap_yaw_symmetry(yaw: float, symmetry: float = math.pi / 2.0) -> float:
    """把偏航角折算到最小阻力区间 [-symmetry/2, +symmetry/2)。

    正方形方块绕 Z 每转 90°（= symmetry）在几何上完全等效，所以手腕真正需要转的
    只是「对 90° 取模后的余量」。把它压进 ±45°，手腕就不会为了一个视觉上等价的
    姿态反向拧 180°，避免 wrist_3_joint 打结或撞上奇异点。
    """
    half = symmetry / 2.0
    return (yaw + half) % symmetry - half


def with_yaw(R: Rotation, yaw: float = 0.0) -> Rotation:
    """把姿态的世界偏航角强制覆写为 yaw（弧度），俯仰/滚转（工具轴指向）保持不变。

    做法是在世界系左乘一个绕 Z 的旋转：yaw(Rz(θ)·R) = yaw(R) + θ，所以取
    θ = yaw − yaw(R) 就能把偏航精确置到目标值；左乘世界 Z 旋转也不改变工具轴与竖直
    方向的夹角，因此「夹爪垂直向下」的姿态不会被破坏（工具轴竖直时，绕 Z 旋转等价于
    绕工具轴自转，只动 wrist_3、不改变 TCP 位置）。
    """
    return Rotation.from_euler("z", yaw - _matrix_yaw(R.as_matrix())) * R


def snap_yaw(R: Rotation, step: float = math.pi / 2.0) -> float:
    """取「离 R 的世界偏航最近的 step 整数倍」角度（弧度）。

    正方形方块绕 Z 每 step(=90°) 视觉上完全等效，所以回正到与桌面 X/Y 轴平行时，
    不必强拧到 0°，而是吸附到最近的 90° 整数倍 —— 手腕只需转 ±45° 以内，
    和抓取时的偏航补偿同量级，不会出现 180° 翻腕或撞关节限位。
    """
    return round(_matrix_yaw(R.as_matrix()) / step) * step


def cube_yaw(d: mujoco.MjData, cube_bid: int) -> float:
    """读取目标方块当前姿态，提取它绕世界 Z 轴的真实偏航角（yaw，弧度）。

    方块是自由刚体，世界姿态可以直接从仿真数据里拿：
        d.xquat[cube_bid]  (w, x, y, z)  方块世界四元数
        d.xmat[cube_bid]   (3, 3)        方块世界旋转矩阵
    这里取「方块局部 x 轴」投影到 XY 平面后的方位角，俯仰/滚转不会干扰 yaw。
    """
    x_axis = d.xmat[cube_bid].reshape(3, 3)[:, 0]
    return math.atan2(x_axis[1], x_axis[0])


def cube_tilt_deg(d: mujoco.MjData, cube_bid: int) -> float:
    """方块局部 +z 轴与竖直方向的夹角（度）——衡量方块被拧/掀了多少。"""
    z_axis = d.xmat[cube_bid].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(float(np.clip(z_axis[2], -1.0, 1.0))))


def _matrix_yaw(R: np.ndarray) -> float:
    """从旋转矩阵里取绕世界 Z 的 yaw（ZYX 约定）。"""
    return math.atan2(R[1, 0], R[0, 0])


def calculate_grasp_pose(d: mujoco.MjData, cube_bid: int,
                         R_home: Rotation, symmetry_deg: float = 90.0) -> Rotation:
    """算出抓取该方块时 TCP 应有的目标姿态（6D 目标里的姿态部分）。

    三步（严格对应需求）：
      1) 提取偏航角：读方块世界姿态，算出绕 Z 的真实 yaw；
      2) 对称性取模过滤：因为正方体 90° 等效，把「相对标准抓取姿态需要转的角度」
         对 90° 取模并偏移，强制落在 [-45°, +45°] 内，避免手腕过度扭转；
      3) 姿态补偿：把过滤后的 yaw 作为世界系绕 Z 的旋转，叠加到「标准向下抓取姿态」
         R_home 上，得到完整的抓取姿态。

    Args:
        d: 本地 MuJoCo 数据（当前帧）。
        cube_bid: 目标方块 body 下标。
        R_home: 末端标准向下抓取姿态（qpos 全 0 时 TCP 的姿态）。
        symmetry_deg: 对称周期，正方体取 90°。

    Returns:
        scipy Rotation：抓取该方块所需的 TCP 目标姿态。
    """
    R_home_mat = R_home.as_matrix()
    yaw_cube = cube_yaw(d, cube_bid)
    yaw_home = _matrix_yaw(R_home_mat)
    dyaw = wrap_yaw_symmetry(yaw_cube - yaw_home, math.radians(symmetry_deg))
    return Rotation.from_euler("z", dyaw) * R_home       # 世界系绕 Z 转 dyaw = 姿态补偿


# ============================== 逆运动学 ==============================

class OracleIKController:
    """阻尼最小二乘（DLS）逆运动学：把 TCP site 驱动到目标位置 + 抓取姿态。"""

    def __init__(self, arm: ArmModel, lam: float = IK_LAMBDA):
        self.arm = arm
        self.lam = lam
        self._R_home: Rotation | None = None     # 标准向下抓取姿态（姿态补偿的基准）
        self._R_goal: Rotation | None = None     # 当前 6D 目标姿态；默认 = R_home

    def capture_home_orientation(self) -> None:
        """记录手臂 qpos 全 0 时 TCP 的姿态作为抓取姿态（该零位 TCP 的 z 轴朝下）。"""
        m, d, arm = self.arm.m, self.arm.d, self.arm
        qpos_bak, qvel_bak = d.qpos.copy(), d.qvel.copy()
        d.qpos[arm.qpos_adr] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._R_home = Rotation.from_matrix(d.site_xmat[arm.tcp_site].reshape(3, 3))
        self._R_goal = self._R_home               # 默认不加偏航补偿
        d.qpos[:] = qpos_bak
        d.qvel[:] = qvel_bak
        mujoco.mj_forward(m, d)

    # ---------- 6D 目标姿态 ----------
    def home_orientation(self) -> Rotation:
        """标准向下抓取姿态（姿态补偿的基准）。"""
        return self._R_home

    def goal_orientation(self) -> Rotation:
        return self._R_home if self._R_goal is None else self._R_goal

    def set_goal_orientation(self, R: Rotation) -> None:
        """设定当前抓取的目标姿态（含方块 yaw 补偿）。Rotation 不可变，直接引用即可。"""
        self._R_goal = R

    def reset_goal_orientation(self) -> None:
        """回到标准向下抓取姿态。"""
        self._R_goal = self._R_home

    def errors(self, goal_pos: np.ndarray) -> tuple[float, float]:
        """返回 (位置误差, 姿态误差)。"""
        m, d, arm = self.arm.m, self.arm.d, self.arm
        mujoco.mj_kinematics(m, d)
        pos_err = float(np.linalg.norm(np.asarray(goal_pos, float) - d.site_xpos[arm.tcp_site]))
        R_cur = Rotation.from_matrix(d.site_xmat[arm.tcp_site].reshape(3, 3))
        rot_err = float(np.linalg.norm((self.goal_orientation() * R_cur.inv()).as_rotvec()))
        return pos_err, rot_err

    def reached(self, goal_pos: np.ndarray) -> bool:
        p, r = self.errors(goal_pos)
        return p < IK_POS_TOL and r < IK_ROT_TOL

    def settled(self) -> bool:
        """手臂是否已基本停稳（关节速度足够小）。"""
        return float(np.max(np.abs(self.arm.d.qvel[self.arm.dof_adr]))) < IK_VEL_TOL

    def ready(self, goal_pos: np.ndarray) -> bool:
        """到位并且停稳——只有这时才允许切换阶段（否则合爪时手臂还在动，方块会被推飞）。"""
        return self.reached(goal_pos) and self.settled()

    def delta_qpos(self, goal_pos: np.ndarray) -> np.ndarray:
        m, d, arm = self.arm.m, self.arm.d, self.arm
        mujoco.mj_kinematics(m, d)
        R_cur = Rotation.from_matrix(d.site_xmat[arm.tcp_site].reshape(3, 3))
        err = np.concatenate([
            np.asarray(goal_pos, float) - d.site_xpos[arm.tcp_site],
            (self.goal_orientation() * R_cur.inv()).as_rotvec(),
        ])
        jac_p = np.zeros((3, m.nv))
        jac_r = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac_p, jac_r, arm.tcp_site)
        J = np.vstack([jac_p[:, arm.dof_adr], jac_r[:, arm.dof_adr]])
        return J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(6), err)

    def action_towards(self, goal_pos: np.ndarray, dq_max: float = IK_DQ_MAX) -> np.ndarray:
        """给出这一拍应该下发的手臂 6 个关节目标角。"""
        dq = np.clip(IK_ALPHA * self.delta_qpos(goal_pos), -dq_max, dq_max)
        return self.arm.clamp(self.arm.arm_qpos() + dq)


# ============================== 笛卡尔绝对直线轨迹 ==============================

class IKSingularityError(RuntimeError):
    """某个笛卡尔路点上 IK 无解（奇异点 / 超出工作空间 / 关节超限）。"""

    def __init__(self, index: int, pos, reason: str):
        pos = np.asarray(pos, float).reshape(3)
        super().__init__(
            "笛卡尔路点 #%d 处 IK 无解（奇异点/不可达）：pos=(%.4f, %.4f, %.4f) —— %s"
            % (index, pos[0], pos[1], pos[2], reason)
        )
        self.index = index
        self.pos = pos
        self.reason = reason


def _as_pose(pose):
    """把 pose 归一化成 (pos(3,) float, Rotation)。约定 pose = (xyz, R)。

    xyz 为 (3,) 笛卡尔位置（m）；R 为 scipy Rotation（四元数/欧拉角/旋转矩阵均可）。
    """
    if not isinstance(pose, (tuple, list)) or len(pose) != 2:
        raise TypeError("pose 必须是 (xyz, Rotation) 二元组")
    pos, R = pose
    pos = np.asarray(pos, float).reshape(3)
    if not isinstance(R, Rotation):
        arr = np.asarray(R, float).ravel()
        if arr.size == 4:
            R = Rotation.from_quat(arr)
        elif arr.size == 3:
            R = Rotation.from_euler("xyz", arr)
        else:
            raise TypeError("无法把 %d 个数的姿态参数解析成 Rotation" % arr.size)
    return pos, R


def solve_ik(goal_pos, goal_R):
    """单个笛卡尔位姿 -> 关节角。

    TODO: 调用 OrcaStudio IK API
        把 TCP 驱动到 (goal_pos, goal_R)，返回 (nq,) 关节角 ndarray。
        工程里默认走 make_dls_ik_solver()（复用已有的阻尼最小二乘控制器）；
        若有 OrcaStudio 原生/解析 IK，可换成它。
        约定：有解 -> (nq,) ndarray；无解 -> 返回 None / 抛异常 / 返回 NaN，
              外层 generate_cartesian_linear_trajectory 会统一转成 IKSingularityError。
    """
    raise NotImplementedError("TODO: 调用 OrcaStudio IK API（goal_pos=%r, goal_R=%r）"
                              % (goal_pos, goal_R))


def make_dls_ik_solver(arm, ik, max_iter: int = 200,
                       pos_tol: float = 2e-3, rot_tol: float = 2e-2):
    """把闭环 DLS 控制器包装成「一次调用解一个静态位姿」的 IK 求解器。

    每次求解都在状态副本上迭代收敛，结束后把 qpos/qvel/目标姿态原样还原，
    因此可以在推进轨迹时被高频调用而不污染正在跑的仿真。
    """
    m, d = arm.m, arm.d

    def solve_goal(goal_pos, goal_R):
        qpos_bak, qvel_bak = d.qpos.copy(), d.qvel.copy()
        R_bak = ik._R_goal
        err_p = err_r = float("inf")
        try:
            ik.set_goal_orientation(goal_R)
            for _ in range(max_iter):
                d.qpos[arm.qpos_adr] = arm.clamp(arm.arm_qpos() + ik.delta_qpos(goal_pos))
                mujoco.mj_forward(m, d)
                err_p, err_r = ik.errors(goal_pos)
                if err_p < pos_tol and err_r < rot_tol:
                    return arm.arm_qpos().copy()
            return None                      # 收敛不了 -> 视为奇异点/不可达
        finally:
            d.qpos[:] = qpos_bak
            d.qvel[:] = qvel_bak
            if R_bak is not None:
                ik.set_goal_orientation(R_bak)
            else:
                ik.reset_goal_orientation()
            mujoco.mj_forward(m, d)

    return solve_goal


def generate_cartesian_linear_trajectory(start_pose, end_pose, num_steps, ik_solver=None):
    """在笛卡尔空间生成 start_pose -> end_pose 的绝对直线轨迹，并逐点求解 IK。

    位置：TCP 在 3D 笛卡尔坐标系里严格沿直线运动（各轴独立 linspace，绝不打弧线）。
    姿态：起终点不同时用球面线性插值（SLERP）平滑过渡，即「边走直线边转手腕」；
          相同时退化为恒定姿态。SLERP 走的是旋转测地线，起终点只差绕世界 Z 的偏航时，
          工具轴全程保持竖直，不会把垂直下压的姿势转歪。

    Args:
        start_pose: 起点位姿 (xyz, R)。
        end_pose:   终点位姿 (xyz, R)。
        num_steps:  路点数量（>= 2）。第 0 个 = start_pose，最后一个 = end_pose。
        ik_solver:  单点位姿 -> 关节角 的求解回调，默认 solve_ik（TODO 占位）。
                    工程里实际传 make_dls_ik_solver(arm, ik)。

    Returns:
        np.ndarray, shape = (num_steps, nq)：每行一个路点的关节角；.ravel() 可拉平。

    Raises:
        ValueError:         num_steps < 2。
        IKSingularityError: 某个路点 IK 无解（奇异点）。
    """
    if num_steps < 2:
        raise ValueError("num_steps 至少为 2，收到 %d" % num_steps)
    if ik_solver is None:
        ik_solver = solve_ik

    p_start, R_start = _as_pose(start_pose)
    p_end, R_end = _as_pose(end_pose)
    rot_lerp = Slerp([0.0, 1.0], Rotation.concatenate([R_start, R_end]))

    # 笛卡尔插值：每个轴各自 linspace，合成 (num_steps, 3) 的严格直线点列
    waypoints = np.stack([
        np.linspace(p_start[0], p_end[0], num_steps),
        np.linspace(p_start[1], p_end[1], num_steps),
        np.linspace(p_start[2], p_end[2], num_steps),
    ], axis=1)

    # 逐点密集 IK
    q_seq: list[np.ndarray] = []
    for i in range(num_steps):
        pos = waypoints[i]
        R = rot_lerp(i / (num_steps - 1.0))      # 位置按弧长比例、姿态按同一比例 SLERP
        try:
            q = ik_solver(pos, R)
        except IKSingularityError:
            raise
        except Exception as e:                       # 求解器自己抛的 -> 统一成奇异点异常
            raise IKSingularityError(i, pos, "%s: %s" % (type(e).__name__, e)) from e
        if q is None:
            raise IKSingularityError(i, pos, "IK 返回无解 (None)")
        q = np.asarray(q, float).ravel()
        if not np.all(np.isfinite(q)):
            raise IKSingularityError(i, pos, "IK 返回了非有限值 (NaN/Inf)")
        # 相邻路点关节跳变接近/超过 180° => 极可能跨过奇异点或翻腕
        if q_seq and np.max(np.abs(q - q_seq[-1])) > math.pi:
            raise IKSingularityError(i, pos, "相邻路点关节跳变超过 180°，疑似奇异点")
        q_seq.append(q)

    return np.stack(q_seq, axis=0)                    # (num_steps, nq)


# ============================== 路点状态机 ==============================

class WaypointPickPlace:
    """
    单个方块的门字形抓放状态机。

    每个 50Hz 控制拍调用一次：
        action, stage, done = machine.step()
        # action: (7,) = 6 个手臂关节目标角 + 1 维夹爪目标开度（净开口，米）
        # 调用方负责下发 action、步进一个控制周期、再调用下一次 step()

    空抓抢救：若是「抓偏」导致合爪时夹到空气（净开口全闭 + 方块没被拎起来），状态机会
    自动插入一段抢救序列（retract 原地垂直抬升 -> release 完全张开），跑完后重读方块
    真实位姿、清掉抓偏偏移，把阶段重置回 approach 做精准二次抓取。重试次数用尽或中途
    把方块碰倒则 failed=True，由调用方把本集作废。
    """

    STAGES = ("approach", "descend", "close", "lift",
              "transfer", "lower", "open", "retreat")

    # 空抓抢救专用阶段：合爪判定空抓后，先原地垂直抬到安全高度（retract），再完全张爪
    # （release）；跑完这两段后重读方块位姿并回到 approach。全程 xy 只在「安全高度之上」
    # 由 approach 重新对准，绝不在方块边上做底部横向平移（那会撞飞方块）。
    RECOVER_STAGES = ("retract", "release")

    # 从 transfer 起，TCP 姿态由「抓取姿态」回正为「放置姿态」（世界偏航 = 0）。
    # transfer 阶段会边走直线边 SLERP 转手腕，把方块原始偏航角抵消掉，
    # 于是 lower / open / retreat 全程保持回正姿态，方块落到托盘里时边缘与桌面 X/Y 轴平行。
    ALIGN_STAGES = ("transfer", "lower", "open", "retreat")

    def __init__(self, ik: OracleIKController, cube_body: str, place_xy, place_z: float):
        self.ik = ik
        self.arm = ik.arm
        self.ik_solver = make_dls_ik_solver(ik.arm, ik)   # 直线轨迹用的静态 IK 求解器
        m = ik.arm.m
        self.cube_body = cube_body
        self.cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube_body)
        if self.cube_bid < 0:                       # 兜底：按子串再找一次
            hits = find_contains(m, mujoco.mjtObj.mjOBJ_BODY, cube_body)
            if not hits:
                raise RuntimeError("找不到方块 body：%s" % cube_body)
            self.cube_bid = hits[0]
        self.place_xy = np.asarray(place_xy, dtype=float)
        self.place_z = float(place_z)
        self.cube_width = self._read_cube_width()      # 目标方块边长（米）
        # 过盈配合：合爪目标开度 = 方块宽度 - 5 mm。位置伺服够不到这个开度，误差被压成
        # 法向夹持力（F ≈ kp * 0.005），把方块真正捏住，而不是「轻轻碰到」。
        self.grip_hold = self.cube_width - GRASP_SQUEEZE
        # 抓偏空抓抢救状态：故意注入的抓偏偏移、已用重试数、是否正在抢救、是否已判废。
        # 这些跨 reset 保留（重试过程中不能被清），由 _restart_pick / 调用方维护。
        self.miss_offset = np.zeros(2)   # 抓取目标 XY 的抓偏噪声（米），[0,0]=不注入
        self.retries = 0                 # 本方块已用掉的抢救重试次数
        self.miss_count = 0              # 本方块累计空抓次数（诊断）
        self.recovering = False          # 当前是否处于抢救序列（retract/release）
        self.failed = False              # 抢救失败/碰倒 -> 本方块作废（本集 success=False）
        self._close_z0 = 0.0             # 进入合爪时方块的 Z，用来判定「高度未变」
        self.reset()

    def _read_cube_width(self) -> float:
        """从目标方块的 box 几何体读出边长（取最大水平尺寸），读不到就兜底 50 mm。"""
        m = self.arm.m
        sizes = [2.0 * float(max(m.geom_size[g][0], m.geom_size[g][1]))
                 for g in range(m.ngeom)
                 if m.geom_bodyid[g] == self.cube_bid
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        return max(sizes) if sizes else 0.05

    # ---------- 生命周期 ----------
    def reset(self):
        self.cube_pos = self.arm.d.xpos[self.cube_bid].copy()
        # 6D 目标：读方块 yaw -> 对 90° 取模压到 ±45° -> 叠加到标准向下抓取姿态上。
        # 整个门字形（含抬升/搬运/下放）都用这个姿态，方块才不会在指间被拧转。
        self.R_grasp = calculate_grasp_pose(self.arm.d, self.cube_bid,
                                            self.ik.home_orientation())
        # 放置姿态：把抓取姿态的偏航回正到最近的 90° 整数倍（Roll/Pitch 原样保留）。
        # 方块夹住后其 yaw 与 TCP 的 yaw 是刚性绑定的（块不会在指间自己转），且抓取时
        #     tcp_yaw − cube_yaw = k·90°（就是上面那个 wrap 出来的 dyaw 的余数）
        # 所以把 TCP 的 yaw 置成 90° 的整数倍，放下时方块的 yaw 也落在 90° 整数倍上 ——
        # 边缘正好与桌面 X/Y 轴平行。取「最近的倍数」而不是死板的 0°，手腕只需转 ±45°。
        self.R_place = with_yaw(self.R_grasp, snap_yaw(self.R_grasp))
        self.ik.set_goal_orientation(self.R_grasp)
        self.stages = list(self.STAGES)   # 当前生效的阶段表（抢救时会被换成 RECOVER_STAGES）
        self.stage_idx = 0
        self.hold = 0
        self.stage_timer = 0
        self.settle_timer = 0        # 直线走完后的 settle 已用拍数（见 SETTLE_MAX_STEPS）
        self.line_qs = None          # 当前直线阶段的预解算关节序列
        self.line_i = 0              # 已下发到第几个路点
        self.xy_fix = np.zeros(2)    # 横向对准修正量（补位置伺服的重力稳态误差）
        self.retreat_xy = np.zeros(2)  # 撤离阶段的冻结 xy 锚点（进入 retreat 时取当时的 TCP xy）
        self.retract_xy = np.zeros(2)  # 抢救抬升阶段的冻结 xy 锚点（进入 retract 时取 TCP xy）
        self.done = False

    @property
    def stage(self) -> str:
        return "done" if self.done else self.stages[self.stage_idx]

    # ---------- 阶段 -> 目标 ----------
    def _stage_R(self) -> Rotation:
        """本阶段的 TCP 目标姿态：搬运前用抓取姿态，搬运后（回正段）用放置姿态。"""
        return self.R_place if self.stage in self.ALIGN_STAGES else self.R_grasp

    def _grasp_xy(self) -> np.ndarray:
        """本次抓取的目标水平位置 = 方块真实 xy + 抓偏噪声偏移（不注入时偏移为 0）。"""
        return self.cube_pos[:2] + self.miss_offset

    def _target(self) -> np.ndarray:
        cz = float(self.cube_pos[2])
        gx, gy = self._grasp_xy()              # 含抓偏噪声的抓取目标 xy
        px, py = self.place_xy
        fx, fy = self.xy_fix                   # 方块系目标的横向对准修正
        s = self.stage
        if s == "retreat":
            # 撤离：只垂直上升，xy 锚定在「进入本阶段时的 TCP xy」（见 _advance），不回中。
            # 托盘处手臂横向有 7~9 mm 的弹性误差（transfer 到位时 TCP 就比指令偏 y -7 mm），
            # 若在此把 TCP 横向拉回 (px, py)，张开的手指会在方块顶角高度处横着蹭过方块，
            # 把它拖走并掀翻（实测方块被拖 34 mm、翻 63°）。纯垂直撤离完全避开这次横向接触。
            # 用冻结值而不是「当前 TCP」：目标若每拍跟着实际走，闭环就没有横向锚点，
            # 直线走完后残余的零空间漂移会无阻尼累积（实测 180 拍漂 52 mm）。
            return np.array([self.retreat_xy[0], self.retreat_xy[1],
                             self.place_z + LIFT_H])
        if s == "retract":
            # 抢救抬升：xy 冻结在「进入本阶段时的 TCP xy」（见 _enter_recover），把 TCP
            # 纯垂直抬到方块上方安全高度。抬高之后才允许横向移动（交给 _restart_pick 后的
            # approach 阶段），避免贴着方块做横向平移把它撞飞。
            return np.array([self.retract_xy[0], self.retract_xy[1], cz + LIFT_H])
        dz = GRASP_DZ
        if float(np.linalg.norm(self.miss_offset)) > 1e-9:
            # 抓偏这一次不下压到正常抓取深度，只停在方块顶面之上（见 MISS_DESCEND_DZ），
            # 夹爪在空气里合拢 -> 干净的抓空，不会把方块顶角压翻。
            dz = MISS_DESCEND_DZ
        return {
            "approach": np.array([gx + fx, gy + fy, cz + APPROACH_H]),
            "descend": np.array([gx + fx, gy + fy, cz + dz]),
            "lift": np.array([gx + fx, gy + fy, cz + LIFT_H]),
            "transfer": np.array([px, py, self.place_z + LIFT_H]),
            "lower": np.array([px, py, self.place_z + GRASP_DZ + LOWER_DZ]),
        }.get(s, self.arm.tcp_pos())          # close / open / release：原地保持

    def _align_xy(self):
        """在下压前把 TCP 横向对准到方块中心（积分补偿位置伺服的稳态误差）。

        UR5e 是位置伺服，重力下关节会垂一点，TCP 稳态位置比指令目标偏 5~7 mm；而夹爪
        全开时单边只有 7.4 mm 余量，偏这么多会让指垫刮到方块上角、把方块掀翻，后面只能
        夹住顶角，搬运时就滑落。这里把误差直接补进目标 XY，让实际 TCP 落到方块正上方。
        """
        cube_xy = self.arm.d.xpos[self.cube_bid][:2]
        tcp = self.arm.tcp_pos()
        if float(np.linalg.norm(self.miss_offset)) > 1e-9:
            return                                   # 抓偏噪声：故意不拉回方块中心，否则噪声被积分抵消
        if float(np.linalg.norm(self._target() - tcp)) > ALIGN_GATE:
            return                                   # 还离得远，先别积分
        self.xy_fix = np.clip(self.xy_fix + ALIGN_GAIN * (cube_xy - tcp[:2]),
                              -ALIGN_MAX, ALIGN_MAX)

    def _grip_cmd(self) -> float:
        """夹爪目标开度（米）：抓取阶段过盈合到「方块宽度 - 5mm」，其余张开到全开。"""
        s = self.stage
        if s in ("close", "lift", "transfer", "lower", "retract"):
            return self.grip_hold             # 过盈配合，硬挤到比方块更窄的开度
        return GRIP_OPEN

    def _line_active(self) -> bool:
        """当前是否正在走「笛卡尔绝对直线」序列（还没走完）。"""
        return (self.stage in LINE_STAGES
                and self.line_qs is not None and self.line_i < len(self.line_qs))

    def _build_line(self):
        """进入直线阶段时，预解算「当前 TCP -> 阶段目标」的笛卡尔绝对直线关节序列。

        起点 = 进入本阶段那一刻 TCP 的**实际**位姿（位置 + 姿态都取实际值，接得最顺），
        终点 = 本阶段目标点 + 本阶段目标姿态，位置逐点线性插值、姿态逐点 SLERP：
            · descend / lift：起终点姿态都是抓取姿态 -> 退化成纯平移直线；
            · transfer      ：起点抓取姿态、终点放置姿态 -> 边走直线边把偏航回正；
            · lower         ：起终点都是放置姿态 -> 纯平移直线，垂直下压。
        """
        goal = np.asarray(self._target(), float)
        start = self.arm.tcp_pos()
        R_goal = self._stage_R()
        self.ik.set_goal_orientation(R_goal)      # 让后续闭环保位/到位判定也用同一姿态
        R_start = Rotation.from_matrix(self.arm.d.site_xmat[self.arm.tcp_site].reshape(3, 3))
        dist = float(np.linalg.norm(goal - start))
        n = max(LINE_MIN_STEPS, int(round(dist / LINE_STEP)) + 1)
        self.line_qs = generate_cartesian_linear_trajectory(
            (start, R_start), (goal, R_goal), n, ik_solver=self.ik_solver)
        self.line_i = 0

    # ---------- 单拍推进 ----------
    def step(self) -> tuple[np.ndarray, str, bool]:
        while not self.done:
            s = self.stage
            if s in ("close", "open", "release"):   # 纯保持阶段，按拍数计时
                if self.hold > 0:
                    self.hold -= 1
                    break
                self._advance()
                continue
            # 直线阶段：序列没走完之前一律不切阶段，否则会半路停下、走不出完整直线
            if self._line_active():
                break
            # 直线走完了：只在「已到位」或「settle 超时」时切阶段。
            # 不能像原来那样死等 ready() —— 重力稳态误差让它长期不成立，每段白等 ~140 拍。
            if self.stage in LINE_STAGES and self.line_qs is not None:
                if self.ik.ready(self._target()) or self.settle_timer >= SETTLE_MAX_STEPS:
                    self._advance()
                    continue
                self.settle_timer += 1
                break
            if self.ik.ready(self._target()) or self.stage_timer >= STAGE_MAX_STEPS:
                self._advance()
                continue
            break

        s = self.stage
        if s in ("close", "open", "release"):
            arm_target = self.arm.arm_qpos()
        elif self._line_active():             # 笛卡尔绝对直线：逐拍下发预解算关节角
            arm_target = self.arm.clamp(self.line_qs[self.line_i])
            self.line_i += 1
            self.stage_timer += 1
        else:
            self.ik.set_goal_orientation(self._stage_R())   # 闭环支路也要用本阶段的姿态
            if s == "approach":
                self._align_xy()             # 下压前先把 TCP 横向对到方块正上方
            if s in CARRY_STAGES:
                dq_max = IK_DQ_MAX_CARRY
            else:
                dq_max = IK_DQ_MAX
            arm_target = self.ik.action_towards(self._target(), dq_max)
            self.stage_timer += 1
        action = np.concatenate([arm_target, [self._grip_cmd()]]).astype(np.float32)
        return action, s, self.done

    def _close_outcome(self) -> str:
        """合爪结束时的判定：

        · grasped：夹爪被方块挡住，没合到指令开度 -> 指间夹住了方块 -> 正常；
        · miss   ：夹爪合到了指令开度（grip_hold）且方块 Z 高度没变 -> 空抓，值得抢救；
        · knock  ：夹爪合到位但方块 Z 变了 -> 下压/合爪把方块碰倒或推起 -> 无可救药。

        注意 2FG7 是位置伺服：close 阶段下发的目标开度是 grip_hold（= 方块宽度 - 过盈量），
        两手空空时手指会直接合到这个指令开度；夹住方块时则被方块挡在方块宽度上，差出的
        过盈量被压成夹持力。所以「空抓」的判据是开度 ≈ grip_hold，而不是全闭的
        GRIP_GAP_CLOSED。
        """
        gap = float(self.arm.grip_state()[0])
        z_now = float(self.arm.d.xpos[self.cube_bid][2])
        if gap > self.grip_hold + MISS_GAP_EPS:
            return "grasped"
        return "miss" if abs(z_now - self._close_z0) <= MISS_Z_EPS else "knock"

    def _enter_recover(self):
        """检测到空抓：进入抢救序列（原地垂直抬升 -> 完全张开）。

        xy 冻结在进入这一刻的 TCP xy（纯垂直抬升，绝不在方块边上横向平移），抬到安全
        高度后再由 _restart_pick 后的 approach 阶段重新对准方块。
        """
        self.recovering = True
        self.retract_xy = self.arm.tcp_pos()[:2].copy()   # 冻结当前 TCP xy 作为抬升锚点
        self.stages = list(self.RECOVER_STAGES)
        self.stage_idx = 0
        self.hold = 0
        self.stage_timer = 0
        self.settle_timer = 0
        self.line_qs, self.line_i = None, 0

    def _restart_pick(self):
        """抢救动作做完：重读方块真实位姿、清掉抓偏偏移，回到 approach 做精准二次抓取。"""
        self.recovering = False
        self.miss_offset = np.zeros(2)                    # 二次抓取不再带噪声，要抓准
        self.cube_pos = self.arm.d.xpos[self.cube_bid].copy()   # 重读方块真实位姿
        self.R_grasp = calculate_grasp_pose(self.arm.d, self.cube_bid,
                                            self.ik.home_orientation())
        self.R_place = with_yaw(self.R_grasp, snap_yaw(self.R_grasp))
        self.ik.set_goal_orientation(self.R_grasp)
        self.stages = list(self.STAGES)                   # 切回正常门字形，从 approach 重来
        self.stage_idx = 0
        self.hold = 0
        self.stage_timer = 0
        self.settle_timer = 0
        self.line_qs, self.line_i = None, 0
        self.xy_fix = np.zeros(2)                         # 对准积分清零，重新对准新位姿
        self.done = False

    def _advance(self):
        """推进到下一阶段；进入 close/open/release 时初始化该阶段的保持拍数。

        离开展开态的「close」时做一次空抓判定：命中空抓且还有重试额度就插入抢救序列
        （retract -> release），否则判废（failed=True）由调用方作废本集。
        """
        s = self.stage
        if s == "close" and not self.recovering:
            outcome = self._close_outcome()
            if outcome == "miss":
                if self.retries < MISS_MAX_RETRIES:
                    self.retries += 1
                    self.miss_count += 1
                    self._enter_recover()
                    return
                self.failed = True                        # 重试额度用尽仍抓空 -> 本集作废
                self.done = True
                return
            if outcome == "knock":                        # 下压/合爪把方块碰倒 -> 本集作废
                self.failed = True
                self.done = True
                return
        if s == "release":                                # 抢救动作（抬升 + 张爪）结束
            self._restart_pick()
            return
        self.stage_idx += 1
        self.stage_timer = 0
        self.settle_timer = 0
        self.line_qs, self.line_i = None, 0
        if self.stage_idx >= len(self.stages):
            self.done = True
            return
        nxt = self.stages[self.stage_idx]
        if nxt == "close":
            self.hold = CLOSE_HOLD_STEPS
            self._close_z0 = float(self.arm.d.xpos[self.cube_bid][2])   # 合爪前的方块 Z 基准
        elif nxt == "release":
            self.hold = OPEN_HOLD_STEPS
        if nxt in LINE_STAGES:                # 进入直线阶段：预解算绝对直线关节序列
            if nxt == "retreat":
                # 冻结进入撤离时的 TCP xy 作为锚点（见 _target 的 retreat 分支）
                self.retreat_xy = self.arm.tcp_pos()[:2].copy()
            self._build_line()


# ============================== 随机摆位 ==============================

class EpisodeRandomizer:
    """每个 episode 把三个方块在安全抓取区内随机摆放，并保证两两不干涉。"""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.joint_names = find_cube_free_joints(m)
        self.dx, self.dy = world_offset_from_model(m, verbose=False)
        self.adr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                    for n in self.joint_names}
        self.dof = {n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                    for n in self.joint_names}

    def reset(self, d: mujoco.MjData, rng) -> dict[str, np.ndarray]:
        placements = sample_positions(len(self.joint_names), rng)
        out = {}
        for name, (x, y) in zip(self.joint_names, placements):
            wx, wy = x + self.dx, y + self.dy
            a = self.adr[name]
            d.qpos[a:a + 3] = [wx, wy, CUBE_Z]
            d.qpos[a + 3:a + 7] = random_quat(rng)
            d.qvel[self.dof[name]:self.dof[name] + 6] = 0.0
            out[name] = np.array([wx, wy, CUBE_Z])
        mujoco.mj_forward(self.m, d)
        return out

    def offset(self) -> np.ndarray:
        return np.array([self.dx, self.dy])


# ============================== 相机 ==============================

_SINGLE_FRAME_TASK = None


def _single_frame_task_cls():
    """惰性导入 SingleFrameTask（只在本进程真的接了相机时才需要 orca_gym.recorder）。"""
    global _SINGLE_FRAME_TASK
    if _SINGLE_FRAME_TASK is None:
        from orca_gym.recorder.recording_task import SingleFrameTask
        _SINGLE_FRAME_TASK = SingleFrameTask
    return _SINGLE_FRAME_TASK


class CameraCapture:
    """两路 RGB 取帧（空中俯视 + 腕部），走 orca_gym 官方推流链路。

    取帧链路（与官方 ``OfficialCameraFeed`` 一致）：
        env.get_camera_names() -> env.start_streaming(name, capture_rgb=True)
        -> mgr.get_recorder(name, "color")   # 端口由后端属性给出，不写死
        -> rec.get_latest_simulate_index()
        -> rec.submit_task(SingleFrameTask(idx, cb)) -> Future.result()

    帧与 action 的同帧同步靠 ``simulate_index``：每拍 ``env.render(simulate_index=idx)``
    把 idx 写进帧头，再按同一个 idx 提交取帧任务，回调里拿到的就是那一帧。

    两个必须绕开的坑（见 orca_gym/recorder/recording_task.py）：
      1. ``SingleFrameTask.should_trigger`` 是 ``current >= idx``，且 submit_task 时若
         ``latest >= idx`` 会**立即派发**，而回调第二参是 ``recorder.last_decoded_frame``
         （不是 idx 那一帧）—— 所以 request() 先守一道 ``latest >= idx`` 就丢弃该帧，
         宁缺勿错。
      2. ``get_frame(idx)`` 返回 None（帧已滚出缓冲区）时 future 直接 set_result(None)，
         计为丢帧。

    离线 ``FakeEnv`` 没有 ``get_camera_names`` / ``start_streaming``，open() 会失败并
    保持 ``active=False``；此时相机完全不参与，主循环走与改动前逐字节一致的路径，
    两个离线回归脚本无需改动。
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData,
                 names=(CAMERA_TOP, CAMERA_WRIST), targets=CAM_TARGETS,
                 enabled: bool = True):
        self.m, self.d = m, d
        self.names = list(names)
        self.targets = dict(targets)
        self.enabled = bool(enabled)
        self.available = names_of(m, mujoco.mjtObj.mjOBJ_CAMERA)
        self.env = None
        self.manager = None
        self.error = ""
        self.streams: dict[str, dict] = {}

    # ---------------------------------------------------------------- 状态
    @property
    def active(self) -> bool:
        """是否真的接上了至少一路相机。"""
        return bool(self.streams)

    @staticmethod
    def _latest(st: dict):
        try:
            return st["rec"].get_latest_simulate_index()
        except Exception as e:                              # noqa: BLE001
            st["err"] = "get_latest_simulate_index 失败: %s" % e
            return None

    @staticmethod
    def _resolve(registered: list[str], want: str) -> str | None:
        """把「想要的相机名」解析成后端实际注册名（可能带 prefab/uuid 后缀）。"""
        if want in registered:
            return want
        low = want.lower()
        hits = [n for n in registered if low in n.lower()]
        if len(hits) > 1:
            print("[cam] %r 模糊匹配到多个相机 %s，取第一个" % (want, hits), flush=True)
        return hits[0] if hits else None

    # ---------------------------------------------------------------- 启动
    def open(self, env) -> bool:
        """按官方流程开流并建 recorder。只在 main 里启动时调一次（WS 握手约 2s/路）。"""
        if not self.enabled:
            return False
        if not hasattr(env, "get_camera_names") or not hasattr(env, "start_streaming"):
            self.error = "env 没有相机推流接口（离线 FakeEnv？）"
            print("[cam] %s，跳过真实取帧" % self.error, flush=True)
            return False
        self.env = env
        try:
            registered = list(env.get_camera_names())
        except Exception as e:                              # noqa: BLE001
            self.error = "get_camera_names 失败: %s: %s" % (type(e).__name__, e)
            print("[cam] %s（确认 OrcaStudio 已进入运行模式）" % self.error, flush=True)
            return False
        if not registered:
            self.error = ("后端未注册任何相机（env.get_camera_names() 为空）；"
                          "确认仿真处于「运行」状态")
            print("[cam] %s" % self.error, flush=True)
            return False
        try:
            self.manager = env.get_recorder_manager()
        except Exception as e:                              # noqa: BLE001
            self.error = "get_recorder_manager 失败: %s: %s" % (type(e).__name__, e)
            print("[cam] %s" % self.error, flush=True)
            return False

        for key in self.names:
            want = str(self.targets.get(key, key))
            name = self._resolve(registered, want)
            if name is None:
                print("[cam] 后端没有相机 %s（想要 %s）；已注册: %s"
                      % (key, want, registered), flush=True)
                continue
            kwargs = dict(CAM_STREAM_KWARGS)
            # max_buffer_frames 是 recorder 专属参数（不是相机属性），默认 36000 太吃内存
            kwargs.setdefault("max_buffer_frames", CAM_MAX_BUFFER_FRAMES)
            try:
                env.start_streaming(name, **kwargs)
            except Exception as e:                          # noqa: BLE001
                print("[cam] start_streaming(%r, %s) 失败: %s: %s"
                      % (name, kwargs, type(e).__name__, e), flush=True)
                continue
            rec = self.manager.get_recorder(name, "color")
            if rec is None:
                print("[cam] start_streaming(%r) 后仍取不到 recorder —— 通常是彩色流没起"
                      "（确认传了 capture_rgb=True），或该相机没有彩色传感器" % name, flush=True)
                continue
            try:
                port = int(getattr(rec, "port", 0) or 0)
            except Exception:                               # noqa: BLE001
                port = 0
            self.streams[key] = {
                "key": key, "name": name, "rec": rec, "port": port,
                "pending": {}, "frames": {},
                "received": 0, "drops": 0, "missing": 0, "err": "",
            }

        if not self.streams:
            if not self.error:
                self.error = "两路相机一路都没起来"
            print("[cam] 未接上任何相机：%s" % self.error, flush=True)
            return False

        print("[cam] 已接上 %s（%s）"
              % (list(self.streams),
                 ", ".join("%s=%s:port%s" % (k, s["name"], s["port"])
                           for k, s in self.streams.items())), flush=True)
        self._wait_first_frame()
        return True

    def _wait_first_frame(self) -> bool:
        """开流后引擎要等 render 驱动才出帧，这里主动催几帧；超时只告警不中断。"""
        deadline = time.time() + CAM_WAIT_FIRST_FRAME
        while time.time() < deadline:
            for st in self.streams.values():
                if st["received"] > 0:
                    continue
                latest = self._latest(st)
                if latest is not None and latest >= 0 and latest not in st["pending"]:
                    self._submit(st, int(latest))
            self.harvest()
            if all(st["received"] > 0 for st in self.streams.values()):
                return True
            time.sleep(0.05)
        print("[cam] 等待首帧超时（%.0fs）；已连接但首帧未到，主循环会继续驱动"
              % CAM_WAIT_FIRST_FRAME, flush=True)
        return False

    # ---------------------------------------------------------------- 取帧
    def _submit(self, st: dict, idx: int):
        try:
            task_cls = _single_frame_task_cls()
        except Exception as e:                              # noqa: BLE001
            st["err"] = "导入 SingleFrameTask 失败: %s: %s" % (type(e).__name__, e)
            return
        try:
            st["pending"][idx] = st["rec"].submit_task(
                task_cls(idx, self._on_frame))
        except Exception as e:                              # noqa: BLE001
            st["err"] = "submit_task(idx=%d) 失败: %s: %s" % (idx, type(e).__name__, e)

    def request(self, idx: int):
        """请求第 idx 帧（不阻塞）。该帧已被引擎解掉则丢弃，避免静默错帧。"""
        if not self.streams:
            return
        for st in self.streams.values():
            latest = self._latest(st)
            if latest is not None and latest >= idx:
                st["drops"] += 1
                continue
            if idx in st["pending"] or idx in st["frames"]:
                continue
            self._submit(st, idx)

    def harvest(self):
        """收割已完成的 Future：ndarray 存进 frames[idx]，None 计丢帧。"""
        for st in self.streams.values():
            done = [i for i, f in st["pending"].items() if f.done()]
            for idx in done:
                fut = st["pending"].pop(idx)
                try:
                    arr = fut.result()
                except Exception as e:                      # noqa: BLE001
                    st["err"] = "取帧失败 idx=%d: %s: %s" % (idx, type(e).__name__, e)
                    st["drops"] += 1
                    continue
                if arr is None:
                    st["drops"] += 1
                    continue
                st["frames"][idx] = arr
                st["received"] += 1

    @staticmethod
    def _on_frame(_frame_entry, decoded_frame):
        """save_worker 线程回调（签名由 orca_gym 的 FrameCallback 决定，第一参用不到）。

        这里就把帧缩到 224 并转 CHW，省内存。
        """
        if decoded_frame is None:
            return None
        return to_chw_224(decoded_frame)

    def flush(self, base: int, n: int, timeout: float = CAM_FLUSH_TIMEOUT) -> dict:
        """集末把 [base, base+n) 的帧凑齐成 (n,3,224,224)。缺帧用「保持上一帧」补。"""
        out = {}
        if not self.streams:
            return out
        if n <= 0:
            return {k: np.zeros((0, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    for k in self.streams}
        hi = base + n
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.harvest()
            # 只等「还在飞」的请求；没提交过的（被守卫丢弃 / get_frame 返回 None）不会再来，
            # 等它就是白等到超时 —— 直接交给下面的「保持上一帧」补齐。
            if not any(any(base <= i < hi for i in st["pending"])
                       for st in self.streams.values()):
                break
            time.sleep(0.01)
        for key, st in self.streams.items():
            arr = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cur = None
            first = None                     # 本段第一张有效帧的序号
            miss = 0
            for i in range(n):
                f = st["frames"].get(base + i)
                if f is not None:
                    cur = f
                    if first is None:
                        first = i
                else:
                    miss += 1
                if cur is not None:
                    arr[i] = cur
            # 开头缺失的帧（采集启动瞬间推流还没出图）用第一张有效帧回填，
            # 避免留下全 0 黑帧污染数据；整段一帧都没收到时只能保持 0。
            if first is not None and first > 0:
                arr[:first] = arr[first]
            out[key] = arr
            st["missing"] = miss
            st["frames"].clear()             # 及时释放，别跨集累积
            st["pending"].clear()
        return out

    # ---------------------------------------------------------------- 收尾
    def stats(self) -> dict:
        s = {"active": self.active, "error": self.error, "cameras": {}}
        for key, st in self.streams.items():
            s["cameras"][key] = {"camera": st["name"], "port": st["port"],
                                 "received": st["received"], "drops": st["drops"],
                                 "missing": st["missing"], "pending": len(st["pending"]),
                                 "err": st["err"]}
        return s

    def latest_index(self) -> int:
        """各路 recorder 已收到的最新 simulate_index（都没收到则 -1）。"""
        best = -1
        for st in self.streams.values():
            latest = self._latest(st)
            if latest is not None:
                best = max(best, int(latest))
        return best

    def drop_summary(self) -> str:
        if not self.streams:
            return "(未接相机)"
        parts = []
        for key, st in self.streams.items():
            # 只作诊断：received/drops 是全程累计，missing 是最近一集的 hold-last 补帧数。
            # 不再折算百分比 —— 三者口径不同，混算出来的比率会虚高甚至为负。
            parts.append("%s[%s] 累计收%d 过晚丢弃%d 本集补帧%d%s"
                         % (key, st["name"], st["received"], st["drops"], st["missing"],
                            ("  " + st["err"]) if st["err"] else ""))
        return "; ".join(parts)

    def close(self):
        for st in list(self.streams.values()):
            try:
                if self.manager is not None:
                    self.manager.stop_recorder(st["name"])
            except Exception:                               # noqa: BLE001
                pass
            try:
                self.env.get_recorder_manager().set_streaming_enabled(st["name"], False)
            except Exception:                               # noqa: BLE001
                pass
            st["pending"].clear()
            st["frames"].clear()
        self.streams.clear()


# ============================== HDF5 录制 ==============================

class HDF5Recorder:
    """
    每个 episode 写成一个独立的 HDF5 文件（50Hz 逐帧）：<stem>_ep<NNNN><ext>

        obs/camera_top    (T,3,224,224) uint8
        obs/camera_wrist  (T,3,224,224) uint8
        obs/qpos          (T,7) float32   6 关节角 + 1 维夹爪当前真实开度（净开口，米）
        action            (T,7) float32   6 关节目标角 + 1 维夹爪目标开度（净开口，米）
        attrs: fps / ctrl_dt / episode_index / num_steps / success / cube_start / place_xy
               / task / seed / failure_reason / dropped_frames / camera_keys
    """

    def __init__(self, path: str, record_cameras: bool = True):
        self.record_cameras = record_cameras
        self._buf: dict[str, list] = {}
        self._imgs: dict[str, np.ndarray] = {}
        self._ep = 0
        parent = os.path.dirname(os.path.abspath(path))
        self._dir = parent or os.getcwd()
        os.makedirs(self._dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        self._stem = stem or "episode"
        self._ext = ext or ".hdf5"

    def _ep_path(self) -> str:
        return os.path.join(self._dir, "%s_ep%04d%s" % (self._stem, self._ep, self._ext))

    def start_episode(self):
        self._buf = {"qpos": [], "action": []}
        self._imgs = {}

    def discard_episode(self):
        """丢弃当前 episode：清缓冲、不落盘、**不推进编号**。

        编号不推进，下一集成功时才会占用同一编号，最终磁盘上就是连续的
        ep0000..epN 无空洞（失败的尝试不留文件）。
        """
        self._buf = {"qpos": [], "action": []}
        self._imgs = {}

    def record(self, qpos7, action7):
        self._buf["qpos"].append(np.asarray(qpos7, dtype=np.float32))
        self._buf["action"].append(np.asarray(action7, dtype=np.float32))

    def record_images(self, images: dict):
        """整集图像一次存入（key -> (T,3,224,224) uint8）。"""
        self._imgs = {str(k): np.asarray(v, dtype=np.uint8) for k, v in images.items()}

    def end_episode(self, success: bool, cube_start: dict, place_xy, *,
                    task: str = "", seed: int = -1, failure_reason: str = "",
                    drops: int = 0) -> tuple[str, int]:
        n = len(self._buf["qpos"])
        path = self._ep_path()
        with h5py.File(path, "w") as f:
            f.attrs["fps"] = 1.0 / CTRL_DT
            f.attrs["ctrl_dt"] = CTRL_DT
            f.attrs["episode_index"] = self._ep
            f.attrs["num_steps"] = n
            f.attrs["success"] = bool(success)
            f.attrs["cube_start"] = np.array(list(cube_start.values()), dtype=np.float64)
            f.attrs["place_xy"] = np.asarray(place_xy, dtype=np.float64)
            # 训练侧需要的元数据。h5py 的 attrs 写不了 None，统一换成 "" / -1。
            f.attrs["task"] = str(task or "")
            f.attrs["seed"] = int(seed)
            f.attrs["failure_reason"] = str(failure_reason or "")
            f.attrs["dropped_frames"] = int(drops)
            f.attrs["camera_keys"] = np.array(sorted(self._imgs),
                                              dtype=h5py.string_dtype(encoding="utf-8"))
            obs = f.create_group("obs")
            obs.create_dataset("qpos", data=np.stack(self._buf["qpos"]))
            f.create_dataset("action", data=np.stack(self._buf["action"]))
            if self.record_cameras:
                for key, arr in sorted(self._imgs.items()):
                    if arr.shape[0] != n:
                        print("[warn] 图像 %s 帧数 %d != qpos 帧数 %d，按较短的对齐"
                              % (key, arr.shape[0], n))
                        arr = arr[:n]
                    obs.create_dataset(key, data=arr, compression="lzf")
        self._ep += 1
        return path, n

    def close(self):
        """每个 episode 都是独立文件、写完即关，这里无需额外收尾。"""


# ============================== 主流程 ==============================

class OracleCollector:
    """episode 主循环：重置 -> 随机摆位 -> 依次抓放 -> 落盘。"""

    def __init__(self, env, recorder: HDF5Recorder, seed: int | None = None,
                 real_time: bool = True, diag: bool = False,
                 cameras=CAM_TARGETS, task: str = TASK_INSTRUCTION,
                 miss_prob: float = MISS_PROB):
        self.env = env
        self.diag = diag
        self.miss_prob = float(np.clip(miss_prob, 0.0, 1.0))   # 每次抓取注入抓偏噪声的概率
        self.m = env.gym._mjModel
        self.d = env.gym._mjData
        self.m.opt.timestep = TIME_STEP          # 显式对齐 50Hz 控制周期
        # real_time=True 时每拍按 CTRL_DT(20ms) 补足 sleep，物理按真实时间播放，
        # 编辑器视口才能看到运动；real_time=False 则尽快跑完（批量数采 / 离线测试）。
        self.real_time = real_time
        self.recorder = recorder
        self.seed = -1 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)
        self.task = str(task or "")
        self.arm = ArmModel(self.m, self.d)
        kp, fmax = self.arm.boost_gripper_force()      # 夹爪电机力量保底（低于 2FG7 原厂值才抬）
        print("[gripper] kp=%.0f N/m  forcerange=±%.0f N  过盈量=%.4f m" % (kp, fmax, GRASP_SQUEEZE))
        self.ik = OracleIKController(self.arm)
        self.randomizer = EpisodeRandomizer(self.m)
        self.cams = CameraCapture(self.m, self.d, targets=cameras,
                                  enabled=recorder.record_cameras)
        # 相机帧的绝对 simulate_index：必须跨 episode 全局单调（见 run_episode）
        self._sim_idx = -1
        self.cubes = self._locate_cubes()
        if len(self.cubes) < 3:
            raise RuntimeError("场景里没找齐 红/黄/绿 三个方块，只找到 %s" % list(self.cubes))
        if self.diag:
            self._diag_model_dump()

    def open_cameras(self) -> bool:
        """启动时开相机流（只调一次：WS 握手约 2s/路，不要每集重开）。

        还要把 render 门控调成「每拍一帧」：默认 render_fps=30 且异步节流，会跳帧，
        跳掉的帧取不到 → 图像与 action 错位。
        """
        for fn, arg in (("set_sync_render", True),
                        ("set_render_fps", int(round(1.0 / CTRL_DT)))):
            try:
                getattr(self.env, fn)(arg)
            except Exception as e:                          # noqa: BLE001
                print("[cam] env.%s(%r) 失败：%s: %s" % (fn, arg, type(e).__name__, e))
        ok = self.cams.open(self.env)
        if not ok:
            print("[cam] 未接上真实相机：%s" % self.cams.error)
        return ok

    def _diag_cameras(self):
        """打印后端已注册相机与各路推流属性（排查「连上了但没帧」）。"""
        try:
            registered = list(self.env.get_camera_names())
        except Exception as e:                              # noqa: BLE001
            print("[diag-cam] get_camera_names 失败：%s: %s" % (type(e).__name__, e))
            return
        print("[diag-cam] 已注册相机: %s" % registered)
        try:
            mgr = self.env.get_recorder_manager()
        except Exception as e:                              # noqa: BLE001
            print("[diag-cam] get_recorder_manager 失败：%s: %s" % (type(e).__name__, e))
            return
        for name in registered:
            try:
                props = mgr.get_camera_properties(name)
            except Exception as e:                          # noqa: BLE001
                print("[diag-cam] %s get_camera_properties 失败：%s: %s"
                      % (name, type(e).__name__, e))
                continue
            keys = ("color_port", "depth_port", "capture_rgb", "capture_depth",
                    "streaming_enabled", "width", "height", "camera_type")
            got = {k: getattr(props, k, "?") for k in keys}
            print("[diag-cam] %s: %s" % (name, got))
            if int(got.get("color_port") or 0) <= 0:
                print("[diag-cam] !! %s 没有彩色推流端口（prefab 里缺 VideoStreamingConfig），"
                      "需在编辑器里给它填一个 RGB 端口" % name)

    def _diag_model_dump(self):
        """打印「编辑器实际编译出来的模型」里指垫 / 方块的接触参数。

        离线拼接模型（/tmp/_combined_cubes_test.xml）只反映磁盘 XML，不反映 OrcaStudio
        重新编译后的结果。夹爪夹不夹得住完全由这里的 condim / friction / priority /
        contype / conaffinity 决定，所以实跑前必须先把这个 dump 打出来对一遍。
        """
        m = self.m
        print("[diag] timestep=%.5f s  cone=%s  integrator=%d"
              % (m.opt.timestep, mujoco.mjtCone(int(m.opt.cone)).name, m.opt.integrator))
        keys = ("finger", "pad", "cube")
        for g in range(m.ngeom):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if not any(k in bname.lower() for k in keys):
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "(无名)"
            fr = m.geom_friction[g]
            print("  geom=%-26s body=%-22s condim=%d priority=%d friction=[%.3f %.3f %.4f] "
                  "contype=%d conaffinity=%d solref=(%.4f %.2f)"
                  % (gname, bname, m.geom_condim[g], m.geom_priority[g],
                     fr[0], fr[1], fr[2], m.geom_contype[g], m.geom_conaffinity[g],
                     m.geom_solref[g][0], m.geom_solref[g][1]))

    def _pad_cube_contacts(self, cube_bid: int) -> tuple[int, float]:
        """(指垫↔目标方块的接触点数, 法向力合力 N)——判断夹爪是否真的夹住了方块。"""
        m, d = self.m, self.d
        n, fsum = 0, 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            bodies = [int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])]
            if cube_bid not in bodies:
                continue
            other = bodies[0] if bodies[1] == cube_bid else bodies[1]
            bname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or "").lower()
            if "finger" not in bname and "pad" not in bname:
                continue
            n += 1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            fsum += abs(float(f6[0]))
        return n, fsum

    def _diag_stage(self, color: str, stage: str, machine, ref):
        """打印「刚进入某阶段」时的搬运状态，定位「方块下滑 / 转动」发生在哪个阶段。

        ref = 合爪瞬间的基准 {dz: 方块中心 - TCP 的高度差, tilt: 方块倾角}；
        之后各阶段的「相对下滑」和「相对转角」都相对这个基准算。
        注意：本行是在该阶段的第一拍之后采样的，所以「进入 close」= 合爪瞬间的抓取深度，
        「进入 lift / transfer / lower / retreat」≈ 上一阶段的结束状态。
        """
        bid = machine.cube_bid
        cube = self.d.xpos[bid]
        tcp = self.arm.tcp_pos()
        tilt = cube_tilt_deg(self.d, bid)
        yaw = cube_yaw(self.d, bid)
        yaw_dev = math.degrees(yaw - round(yaw / (math.pi / 2.0)) * (math.pi / 2.0))
        ncon, fsum = self._pad_cube_contacts(bid)
        print("    [diag] %-6s 进入 %-8s 方块中心-TCP=%+.4f m  相对合爪下滑=%+.4f m  横向偏心=%.4f m"
              "  倾角=%.1f°(Δ%.1f°)  偏航偏差=%.1f°  开度=%.4f m  指垫接触=%d 力=%.1f N"
              % (color, stage, cube[2] - tcp[2],
                 (cube[2] - tcp[2] - ref["dz"]) if ref else 0.0,
                 float(np.linalg.norm(tcp[:2] - cube[:2])), tilt,
                 (tilt - ref["tilt"]) if ref else 0.0, yaw_dev,
                 float(self.arm.grip_state()[0]), ncon, fsum), flush=True)

    def _locate_cubes(self) -> dict[str, str]:
        bodies = names_of(self.m, mujoco.mjtObj.mjOBJ_BODY)
        out = {}
        for color in PICK_ORDER:
            hit = [b for b in bodies if "cube" in b.lower() and color in b.lower()]
            if hit:
                out[color] = hit[0]
        return out

    def _place_xy(self, idx: int) -> np.ndarray:
        ox, oy = PLACE_OFFSETS[idx]
        return np.array([PLACE_CENTER[0] + ox, PLACE_CENTER[1] + oy]) + self.randomizer.offset()

    def _zero_ctrl(self) -> np.ndarray:
        return np.zeros(self.m.nu, dtype=np.float64)

    def _apply_and_step(self, action: np.ndarray, sim_idx: int | None = None,
                        request_idr: bool = False):
        t0 = time.perf_counter()
        ctrl = self._zero_ctrl()
        ctrl[self.arm.act_ids] = action[:6]
        # action 的夹爪通道是「净开口（米）」，转成 2FG7 的位置指令 ctrl 再下发
        ctrl[self.arm.grip_act] = grip_ctrl_for_width(action[6])
        self.env.gym.set_ctrl(ctrl)
        self.env.gym.mj_step(FRAME_SKIP)
        self.env.gym.update_data()              # _mjData -> env.data；render() 推的是 env.data.qpos，漏了这步视口会定住不动
        if sim_idx is not None:
            # 把本拍的 simulate_index 写进帧头，相机才能按同一 idx 取到这一帧
            self.env.render(simulate_index=sim_idx, request_idr=request_idr)
        else:
            self.env.render()                   # 无相机路径：与改动前逐字节一致
        if self.real_time:
            dt = CTRL_DT - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def _state7(self) -> np.ndarray:
        return np.concatenate([self.arm.arm_qpos(), self.arm.grip_state()]).astype(np.float32)

    def run_episode(self, keep_failures: bool = False) -> bool:
        """跑一集。返回是否成功。

        keep_failures=False（默认）时失败集**不落盘**、编号不推进，由调用方换一组
        随机摆位重试；True 时失败集也写盘（success=False），便于事后排查失败原因。
        """
        self.env.reset()                                   # 回到初始帧
        self.m.opt.timestep = TIME_STEP                    # reset 后再次对齐 50Hz
        cube_start = self.randomizer.reset(self.d, self.rng)
        self.env.gym.set_ctrl(self._zero_ctrl())
        self.env.gym.mj_step(int(0.5 / TIME_STEP))         # 让方块落稳
        # 让 episode 的起点就是「夹爪张开」：专家全程张开起手，而 reset 默认 ctrl=0 会把
        # 夹爪夹到底（净开口 0.0268 m）。不先张开的话首帧观测就带一个闭爪状态，偏出分布。
        open_ctrl = self._zero_ctrl()
        open_ctrl[self.arm.grip_act] = grip_ctrl_for_width(GRIP_GAP_OPEN)
        self.env.gym.set_ctrl(open_ctrl)
        self.env.gym.mj_step(int(0.5 / TIME_STEP))         # 等夹爪张到位
        self.ik.capture_home_orientation()

        # 相机帧的 simulate_index 必须全局单调、跨 episode 不复用：env.reset() 内部会
        # 自己 render 并推进 env 的计数器，若每集从 0 重数，get_frame(0) 会命中上一集
        # 残留在 rolling buffer 里的旧帧 -> 静默错图。
        base = self._next_sim_index()
        sim_idx, t = base, 0
        cameras_on = self.cams.active

        self.recorder.start_episode()
        # 「动作之前」的首帧：空动作只让夹爪保持张开、机械臂保持 home（与 reset 后的静止态
        # 等价），但必须走一次 step+render，引擎才会真的产出一帧新图像供相机按 idx 取用。
        neutral = np.zeros(7, dtype=np.float64)
        neutral[6] = GRIP_GAP_OPEN
        self._apply_and_step(neutral,
                             sim_idx=sim_idx if cameras_on else None,
                             request_idr=True)

        ok, fail = True, ""
        for i, color in enumerate(PICK_ORDER):
            body = self.cubes[color]
            machine = WaypointPickPlace(self.ik, body, self._place_xy(i),
                                        CUBE_Z + TRAY_FLOOR_DZ)
            # 抓偏噪声：以 miss_prob 的概率给这次抓取的目标 XY 叠一个 3~5 cm 的随机偏移，
            # 故意让夹爪抓空，逼专家演示「抓空 -> 抬升 -> 张爪 -> 重读位姿 -> 二次抓取」。
            if self.rng.random() < self.miss_prob:
                ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
                mag = float(self.rng.uniform(MISS_OFF_MIN, MISS_OFF_MAX))
                machine.miss_offset = mag * np.array([math.cos(ang), math.sin(ang)])
                print("  [noise] %s 注入抓偏 %.3f m（方向 %.0f°）"
                      % (color, mag, math.degrees(ang)))
            guard = 0
            prev_stage, ref = None, None
            while not machine.done:
                # 数据约定：obs-before-action。一帧 = 当前帧的 obs（动作**之前**的 qpos +
                # 该帧图像）+ 基于它决策出的动作。专家本来就读动作之前的 sim 状态来规划，
                # 之前却把动作之后的 state 记在一起，导致 (obs, action) 因果倒置、观测≈动作。
                #
                # 1) 取当前帧（动作之前）的图像与 qpos —— 该帧由上面的预热或上一拍的 step 渲染
                if cameras_on:
                    self.cams.request(sim_idx)
                    self.cams.harvest()
                obs = self._state7()
                # 2) 专家基于「动作之前」的 sim 状态决策出本拍动作
                action, stage, _ = machine.step()
                # 3) 记录 (obs_before, action)
                self.recorder.record(obs, action)
                # 4) 施加动作：推进物理并渲染下一帧（= 下一拍的 obs）
                sim_idx += 1
                t += 1
                self._apply_and_step(action,
                                     sim_idx=sim_idx if cameras_on else None,
                                     request_idr=False)
                if stage != prev_stage:
                    if self.diag and prev_stage is not None:
                        self._diag_stage(color, stage, machine, ref)
                    if stage == "retract":    # 空抓抢救启动：抬升 -> 张爪 -> 重读位姿 -> 二次抓取
                        print("  [rescue] %s 检测到空抓（累计 %d 次）-> 原地抬升/张爪/重读位姿/二次抓取"
                              % (color, machine.miss_count))
                    if self.diag and stage == "close":   # 合爪瞬间记下基准，后面都相对它算下滑/转角
                        ref = {"dz": float(self.d.xpos[machine.cube_bid][2]
                                           - self.arm.tcp_pos()[2]),
                               "tilt": cube_tilt_deg(self.d, machine.cube_bid)}
                    prev_stage = stage
                guard += 1
                if guard > 6000:
                    print("  [warn] %s 状态机超时（stage=%s），跳过" % (color, stage))
                    ok = False
                    fail = fail or "timeout:%s" % color
                    break
            if machine.failed:
                # 空抓抢救失败（重试用尽仍抓空）或抢救中把方块碰倒 -> 本集作废，不再抓后面的
                ok = False
                fail = fail or "miss_grasp:%s" % color
                print("  [fail] %s 抓取抢救失败（累计空抓 %d，已重试 %d）——本集作废"
                      % (color, machine.miss_count, machine.retries))
                break
            tilt = cube_tilt_deg(self.d, machine.cube_bid)
            err = float(np.linalg.norm(self.d.xpos[machine.cube_bid][:2] - machine.place_xy))
            good = err < 0.05 and tilt < KNOCK_TILT_DEG
            ok = ok and good
            if not good and not fail:
                fail = ("knock:%s" % color) if tilt >= KNOCK_TILT_DEG else ("place_error:%s" % color)
            note = "OK" if good else ("碰倒 %.0f°" % tilt if tilt >= KNOCK_TILT_DEG else "偏离")
            print("  放置 %-6s 目标 %s  实际 %s  误差 %.4f m  倾角 %.1f°  %s"
                  % (color, np.round(machine.place_xy, 3),
                     np.round(self.d.xpos[machine.cube_bid][:2], 3), err, tilt, note))

        self._sim_idx = sim_idx - 1
        drops = 0
        if self.recorder.record_cameras:
            if cameras_on:
                self.recorder.record_images(self.cams.flush(base, t))
                # 「丢帧」= 本集有多少拍的图像是 hold-last 补的（flush 里的 miss）。
                # 不能用 drops 计数器的增量：那个值会把上一集的 missing 也带进来，出现负数。
                drops = sum(s["missing"] for s in self.cams.streams.values())
                total = t * max(1, len(self.cams.streams))
                print("[cam] 本集 %d 帧，hold-last 补帧 %d（%.2f%%）  %s"
                      % (t, drops, 100.0 * drops / max(1, total), self.cams.drop_summary()))
                if drops > 0.01 * max(1, total):
                    print("[cam] !! hold-last 补帧偏高：查相机采集分辨率（CameraCaptureConfig "
                          "Width/Height）是否已降到 320x240，以及 render_fps 是否=50")
            else:
                print("[cam] !! 相机没起来，本集写入全 0 占位图 —— 这批数据不能用于训练")
                self.recorder.record_images(
                    {k: np.zeros((t, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                     for k in self.cams.names})

        if ok or keep_failures:
            name, n = self.recorder.end_episode(ok, cube_start, PLACE_CENTER,
                                                task=self.task, seed=self.seed,
                                                failure_reason=fail, drops=drops)
            print("[episode] %s 写入 %d 帧  success=%s%s"
                  % (name, n, ok, ("  reason=%s" % fail) if fail else ""))
        else:
            self.recorder.discard_episode()
            print("[episode] 丢弃（不落盘，编号不推进）  reason=%s" % (fail or "unknown"))
        return ok

    def _next_sim_index(self) -> int:
        """取下一个可用的全局 simulate_index。

        env.render(simulate_index=k) 会把 k 写回 env 的内部计数器；取三者较大值 +1，
        保证跨 episode 严格单调递增。除了本进程的计数器，还要参考 recorder 已收到的
        最大 index —— 进程刚启动时若引擎侧已经推过帧，从 0 重数会命中那些旧帧。
        """
        try:
            env_idx = int(getattr(self.env, "_current_simulate_index", -1))
        except Exception:                                   # noqa: BLE001
            env_idx = -1
        return max(self._sim_idx + 1, env_idx + 1, self.cams.latest_index() + 1)


def build_env(addr: str):
    """构造本地物理环境（物理在本进程，OrcaStudio 只做可视化）。"""
    from orca_gym.scripts.sim_env import SimEnv
    return SimEnv(frame_skip=FRAME_SKIP, orcagym_addr=addr,
                  agent_names=["NoRobot"], time_step=TIME_STEP)


def scene_handshake(addr: str, stage: str):
    """与 run_sim_loop 一致的场景握手（视口打出「开始仿真/运行结束」提示）。"""
    try:
        from orca_gym.scripts import run_sim_loop
        run_sim_loop.sceneinfo(scene=None, stage=stage, orcagym_address=addr)
    except Exception as e:                       # 握手失败不影响物理
        print("[warn] scene handshake(%s) 失败：%s" % (stage, e))


def parse_cameras(spec: str):
    """把 "camera_top=camera_float,camera_wrist=gripper_camera" 解析成 ((key, 相机名), ...)。"""
    pairs = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, name = item.partition("=")
        key, name = key.strip(), name.strip() or key.strip()
        if key:
            pairs.append((key, name))
    return tuple(pairs) or CAM_TARGETS


def _fmt_hms(sec: float) -> str:
    """秒 -> 人类可读时长。"""
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return ("%dh%02dm%02ds" % (h, m, s)) if h else ("%dm%02ds" % (m, s))


def _progress(done: int, target: int, fails: int, elapsed: float) -> str:
    """一行式进度条：成功数 / 丢弃数 / 成功率 / 平均单集耗时 / 预计剩余。"""
    tries = done + fails
    width = 24
    filled = int(round(width * done / max(1, target)))
    rate = done / tries if tries else 0.0
    per = elapsed / tries if tries else 0.0
    return ("[%s] 成功 %d/%d  丢弃 %d  成功率 %.1f%%  均 %.1fs/集  已用 %s  剩余约 %s"
            % ("#" * filled + "-" * (width - filled), done, target, fails,
               rate * 100.0, per, _fmt_hms(elapsed), _fmt_hms(per * (target - done))))


def main():
    ap = argparse.ArgumentParser(description="Oracle 自动抓放专家数据采集")
    ap.add_argument("--addr", default=DEFAULT_ADDR, help="OrcaStudio gRPC 地址")
    ap.add_argument("--episodes", type=int, default=5,
                    help="要采集的**成功**条数（失败集被丢弃后会自动换随机摆位重试补足）")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="最多尝试次数（含失败），0 = episodes*5，兜底防止无限重试")
    ap.add_argument("--keep-failures", action="store_true",
                    help="失败集也写盘（success=False）便于排查；默认丢弃、不落盘")
    ap.add_argument("--output",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "hdf5", "oracle_pick.hdf5"))
    ap.add_argument("--seed", type=int, default=None, help="随机种子（复现用）")
    ap.add_argument("--no-cameras", action="store_true", help="不录图像，只录 qpos/action")
    ap.add_argument("--fast", action="store_true",
                    help="不按实时节流、尽快跑完（批量数采用；默认按 50Hz 实时播放，方便在编辑器里看）")
    ap.add_argument("--diag", action="store_true",
                    help="打印实跑诊断：模型的指垫/方块接触参数 + 各阶段的方块下滑量与转角")
    ap.add_argument("--cameras", default="%s=%s,%s=%s"
                    % (CAMERA_TOP, CAM_TARGETS[0][1], CAMERA_WRIST, CAM_TARGETS[1][1]),
                    help="obs key -> 场景相机名的映射，逗号分隔；默认 %s=%s,%s=%s"
                         % (CAMERA_TOP, CAM_TARGETS[0][1], CAMERA_WRIST, CAM_TARGETS[1][1]))
    ap.add_argument("--task", default=TASK_INSTRUCTION,
                    help="写进 episode attrs 的语言指令（供 ACT 语言条件用，不用可留空）")
    ap.add_argument("--miss-prob", type=float, default=MISS_PROB,
                    help="每次抓取注入「抓偏空抓」噪声的概率（默认 %.2f）；"
                         "冒烟时给 1.0 可强制每次都演示抢救动作" % MISS_PROB)
    args, _ = ap.parse_known_args()   # 容忍 OrcaLab 额外传入的参数

    cameras = parse_cameras(args.cameras)
    env = build_env(args.addr)
    scene_handshake(args.addr, "beginscene")
    rec = HDF5Recorder(args.output, record_cameras=not args.no_cameras)
    collector = OracleCollector(env, rec, seed=args.seed, real_time=not args.fast,
                                diag=args.diag, cameras=cameras, task=args.task,
                                miss_prob=args.miss_prob)
    if args.diag:
        collector._diag_cameras()
    collector.open_cameras()

    target = int(args.episodes)
    max_attempts = int(args.max_attempts) or max(target * 5, target)
    ok_count = fail_count = 0
    t0 = time.perf_counter()
    print("[批量] 目标成功 %d 条，最多尝试 %d 次；失败集默认丢弃并换随机摆位重试"
          % (target, max_attempts), flush=True)
    try:
        while ok_count < target and (ok_count + fail_count) < max_attempts:
            print("\n[第 %d 次尝试] %s"
                  % (ok_count + fail_count + 1,
                     _progress(ok_count, target, fail_count, time.perf_counter() - t0)),
                  flush=True)
            try:
                ok = collector.run_episode(keep_failures=args.keep_failures)
            except KeyboardInterrupt:
                raise
            except Exception as e:                          # noqa: BLE001
                # IK 奇异 / 方块被摆到不可达处 / 场景异常 —— 一律丢弃本集、换一组重来
                print("  [episode] 异常丢弃：%s: %s" % (type(e).__name__, e), flush=True)
                collector.recorder.discard_episode()
                ok = False
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            print(_progress(ok_count, target, fail_count, time.perf_counter() - t0), flush=True)
    except KeyboardInterrupt:
        print("\n中断，已成功落盘的 episode 会保留。")
    finally:
        scene_handshake(args.addr, "endscene")
        print("[cam] 全程 %s" % collector.cams.drop_summary())
        collector.cams.close()
        rec.close()
        print("完成：成功 %d/%d（尝试 %d 次，丢弃 %d），用时 %s，数据 -> %s_ep*.hdf5"
              % (ok_count, target, ok_count + fail_count, fail_count,
                 _fmt_hms(time.perf_counter() - t0),
                 os.path.splitext(args.output)[0]))
        if ok_count < target:
            print("[warn] 没凑满目标：看上面的失败 reason，或适当放宽判定阈值")


if __name__ == "__main__":
    main()
