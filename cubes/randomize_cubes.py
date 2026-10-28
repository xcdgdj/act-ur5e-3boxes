#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连接正在运行的 OrcaSim 仿真，把 cubes.xml 里的三个小方块（红/黄/绿）
随机重新摆放到「机械臂容易够到」的桌面区域。

原理：
  1. 通过 OrcaSim 的 gRPC 接口（默认 localhost:50051）读取当前全量 qpos/qvel；
  2. 找到三个方块对应的 free joint（名字形如 cube_*_free）；
  3. 在可达区域内随机采样三个互不重叠的位置（带随机 yaw），z 贴在桌面上；
  4. 写回 qpos 并把对应 qvel 归零，通过 SetQpos / SetQvel 推送回仿真。

用法：
    python randomize_cubes.py                 # 随机摆位一次
    python randomize_cubes.py --n 5 --interval 1.0
    python randomize_cubes.py --addr localhost:50051 --seed 42

依赖：orca_gym（orca-gym 包）。必须用和 OrcaStudio 版本匹配的那份——共用一个环境：
    /home/user/miniconda3/envs/orcalab/bin/python randomize_cubes.py
"""

import argparse
import math
import random
import sys
import time

import grpc

from orca_gym.protos import mjc_message_pb2 as pb
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

# ---- 采样区域（单位 m，世界坐标）----
# 机械臂「容易够到」的完整桌面区域（由 IK 网格扫描得出）：
#     X ∈ [-1.15, -0.77]   Y ∈ [-0.77, 0.12]
# 下面只取其中更紧凑的一块，让三个方块彼此挨得近一些（完整区域都在射程内）。
X_MIN, X_MAX = -1.07, -0.83
Y_MIN, Y_MAX = -0.44, -0.20

# 方块参数（与 cubes.xml 保持一致）
CUBE_HALF = 0.025          # 边长 50 mm -> half 0.025
TABLE_TOP_Z = 0.75         # 桌面顶面高度
CUBE_Z = TABLE_TOP_Z + CUBE_HALF + 0.0005   # 方块中心 z，略高于桌面避免初始穿透

# 关卡里 ur5e_table prefab 实例带了一个平移（见 random_pick.prefab 的 Instance patch），
# 所以仿真 qpos 里的世界坐标 = XML 坐标 + 该平移量；而上面的可达区域是在 XML 坐标系里算的，
# 写回 qpos 时必须补上这个平移，否则方块会被扔到桌子外。
# 这里不写死数值，改为运行时从仿真里读 table body 实际位置反推。
TABLE_XML_POS = (-0.97, -0.32)   # ur5e_table.xml 里 <body name="table"> 的 pos
BASE_XML_POS = (-0.55, -0.32)    # ur5e_table.xml 里 <body name="base"> 的 pos（机械臂底座）

# 两个方块中心的最小间距：夹爪全开时在合爪方向上的半宽实测 74 mm，
# 方块自身半宽 25 mm，所以相邻方块中心距离至少要 99 mm 才不碰；
# 取 120 mm，留 ~20 mm 余量（不管两个方块是什么相对方位都不会撞到）。
MIN_SEP = 0.12

# 机械臂底座附近留出禁入区（避免方块放到机械臂正下方）
KEEP_OUT_CENTER = (-0.55, -0.32)
KEEP_OUT_RADIUS = 0.20

FREE_JOINT_TYPE = 0        # mjtJoint: FREE = 0


def find_cube_joints(stub):
    """返回 [(joint_name, qpos_addr, qvel_addr), ...]，按 qpos 地址排序。"""
    resp = stub.QueryAllJoints(pb.QueryAllJointsRequest())
    joints = []
    for j in resp.joint_info:
        if j.type != FREE_JOINT_TYPE:
            continue
        if "cube" in j.name.lower() and j.name.lower().endswith("_free"):
            joints.append((j.name, j.qpos_idx_start, j.qvel_idx_start))
    joints.sort(key=lambda t: t[1])
    return joints


def find_cube_free_joints(mj_model):
    """从本地 MuJoCo 模型里挑出三个方块的 free joint 名字（可能带 agent 前缀，所以按特征匹配）。"""
    names = []
    for i in range(mj_model.njnt):
        if mj_model.jnt_type[i] != FREE_JOINT_TYPE:
            continue
        name = mj_model.joint(i).name or ""
        if "cube" in name.lower() and name.lower().endswith("_free"):
            names.append(name)
    names.sort()
    return names


def find_table_body_name(stub):
    """找到桌子 body 的完整名字（导入/实例化后可能带 ur5e_table_usda_ 之类的前缀）。"""
    resp = stub.QueryAllBodies(pb.QueryAllBodiesRequest())
    for b in resp.body_info:
        if b.name == "table" or b.name.endswith("_table"):
            return b.name
    return None


def world_offset(stub, verbose=True):
    """返回 (dx, dy)：把 XML 坐标换算成仿真世界坐标需要补的平移量。"""
    name = find_table_body_name(stub)
    if name is None:
        if verbose:
            print("警告：仿真里没找到 table body，按偏移 (0, 0) 处理")
        return 0.0, 0.0
    resp = stub.QueryBodyPosMatQuat(
        pb.QueryBodyPosMatQuatRequest(body_name_list=[name])
    )
    p = resp.body_pos_mat_quat_list[0].pos
    dx = p[0] - TABLE_XML_POS[0]
    dy = p[1] - TABLE_XML_POS[1]
    if verbose:
        print("table body = %s  世界坐标=(%.4f, %.4f)  ->  XML->世界 平移偏移=(%.4f, %.4f)"
              % (name, p[0], p[1], dx, dy))
    return dx, dy


def find_arm_base_body_name(mj_model):
    """在本地 MuJoCo 模型里找机械臂底座 body（排除夹爪的 gripper_base）。"""
    for i in range(mj_model.nbody):
        name = mj_model.body(i).name or ""
        if name.endswith("_base") and "gripper" not in name.lower():
            return name
    return None


def world_offset_from_model(mj_model, verbose=True):
    """给「物理跑在 python 侧」的本地模型用：从底座位置反推 XML->世界 平移偏移。"""
    name = find_arm_base_body_name(mj_model)
    if name is None:
        if verbose:
            print("警告：本地模型里没找到机械臂底座 body，按偏移 (0, 0) 处理")
        return 0.0, 0.0
    p = mj_model.body(name).pos
    dx = p[0] - BASE_XML_POS[0]
    dy = p[1] - BASE_XML_POS[1]
    if verbose:
        print("底座 body = %s  世界坐标=(%.4f, %.4f)  ->  XML->世界 平移偏移=(%.4f, %.4f)"
              % (name, p[0], p[1], dx, dy))
    return dx, dy



def sample_positions(n, rng):
    """在可达区域内采样 n 个互不重叠的 (x, y)。"""
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 20000:
            raise RuntimeError("无法在可达区域内采样出 %d 个互不重叠的位置" % n)
        x = rng.uniform(X_MIN, X_MAX)
        y = rng.uniform(Y_MIN, Y_MAX)
        if (x - KEEP_OUT_CENTER[0]) ** 2 + (y - KEEP_OUT_CENTER[1]) ** 2 < KEEP_OUT_RADIUS ** 2:
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= MIN_SEP ** 2 for px, py in pts):
            pts.append((x, y))
    return pts


def random_quat(rng):
    """绕世界 z 轴随机 yaw，返回 MuJoCo 顺序的四元数 (w, x, y, z)。"""
    yaw = rng.uniform(-math.pi, math.pi)
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def randomize_once(stub, rng, verbose=True):
    cube_joints = find_cube_joints(stub)
    if not cube_joints:
        sys.exit(
            "没有在仿真里找到方块（free joint 名字里含 'cube' 且以 '_free' 结尾）。\n"
            "请先在 OrcaStudio 中导入 cubes/cubes.xml，并把生成 prefab 拖进关卡，再启动仿真。"
        )
    if verbose:
        print("找到 %d 个方块关节：%s" % (len(cube_joints), [c[0] for c in cube_joints]))

    state = stub.QueryAllQposQvelQacc(pb.QueryAllQposQvelQaccRequest())
    qpos = list(state.qpos)
    qvel = list(state.qvel)

    dx, dy = world_offset(stub, verbose)

    placements = sample_positions(len(cube_joints), rng)
    for (name, qadr, vadr), (x, y) in zip(cube_joints, placements):
        quat = random_quat(rng)
        qpos[qadr:qadr + 7] = [x + dx, y + dy, CUBE_Z] + quat
        qvel[vadr:vadr + 6] = [0.0] * 6
        if verbose:
            print("  %-20s -> x=%+.3f y=%+.3f (z=%.3f)" % (name, x + dx, y + dy, CUBE_Z))

    stub.SetQpos(pb.SetQposRequest(qpos=qpos))
    stub.SetQvel(pb.SetQvelRequest(qvel=qvel))
    stub.MJ_Forward(pb.MJ_ForwardRequest())


def main():
    ap = argparse.ArgumentParser(description="连接运行中的 OrcaSim 仿真，随机摆放三个小方块")
    ap.add_argument("--addr", default="localhost:50051", help="OrcaSim gRPC 地址")
    ap.add_argument("--n", type=int, default=1, help="随机摆位次数（默认 1）")
    ap.add_argument("--interval", type=float, default=1.0, help="多次摆位之间的间隔秒数")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可选，用于复现）")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    channel = grpc.insecure_channel(args.addr)
    try:
        grpc.channel_ready_future(channel).result(timeout=3.0)
    except grpc.FutureTimeoutError:
        sys.exit(
            "连接不上 %s。请确认 OrcaStudio 里该关卡的仿真已经在运行"
            "（外部程序默认是 run_sim_loop）。" % args.addr
        )
    stub = GrpcServiceStub(channel)

    for i in range(args.n):
        if args.n > 1:
            print("[%d/%d]" % (i + 1, args.n))
        randomize_once(stub, rng)
        if i + 1 < args.n:
            time.sleep(args.interval)

    channel.close()


if __name__ == "__main__":
    main()
