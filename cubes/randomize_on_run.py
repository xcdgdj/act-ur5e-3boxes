#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「点运行就自动随机摆位」的入口脚本。

由 OrcaStudio 的 external_programs 调用（见 .orcalab/config.toml 里的
randomize_sim_loop），一般不需要手动跑。

做的事：在 SimEnv.reset_model 外面套一层 —— 仿真 reset 回初始帧之后，
立刻把 cubes.xml 里的三个方块随机摆到「机械臂容易够到」的区域，然后照常
进入 run_sim_loop 的仿真循环。以后每次点运行都是新的随机分布。

方块尺寸、采样区域、最小间距等参数统一从 randomize_cubes.py 读取，改一处即可。
"""

import random
import sys

import numpy as np

import orca_gym.scripts.sim_env as sim_env

from randomize_cubes import (
    CUBE_Z,
    find_cube_free_joints,
    random_quat,
    sample_positions,
    world_offset_from_model,
)


def randomize_cubes(env):
    """把仿真里所有 cube_*_free 关节随机摆位（在本地 mjData 上改，再同步给渲染）。"""
    names = find_cube_free_joints(env.gym._mjModel)
    if not names:
        print("[randomize_on_run] 场景里没找到 cube_*_free 关节，跳过随机摆位", flush=True)
        return

    # 关卡里 prefab 带了平移，qpos 是世界坐标；可达区域是在 XML 坐标系里算的，
    # 所以采样结果要补上这个平移，否则方块会落到桌子外面。
    dx, dy = world_offset_from_model(env.gym._mjModel)

    rng = random.Random()
    placements = sample_positions(len(names), rng)

    joint_qpos = {}
    for name, (x, y) in zip(names, placements):
        joint_qpos[name] = np.array([x + dx, y + dy, CUBE_Z] + random_quat(rng), dtype=np.float64)

    env.gym.set_joint_qpos(joint_qpos)
    env.gym.mj_forward()
    env.gym.update_data()      # 本地 mjData -> env.data，随后 render() 会推送到 OrcaStudio

    for name, (x, y) in zip(names, placements):
        print("[randomize_on_run] %-24s -> x=%+.3f y=%+.3f z=%.3f" % (name, x + dx, y + dy, CUBE_Z), flush=True)


_original_reset_model = sim_env.SimEnv.reset_model


def _reset_model_with_random_cubes(self):
    obs, info = _original_reset_model(self)
    randomize_cubes(self)
    return obs, info


sim_env.SimEnv.reset_model = _reset_model_with_random_cubes


if __name__ == "__main__":
    import orca_gym.scripts.run_sim_loop as run_sim_loop

    run_sim_loop.main()
