#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 oracle_collect.py 采出的 HDF5 转成 LeRobot v2.1 数据集（供 ACT 训练）。

环境
----
**必须在 ``orcalab_lerobot`` 环境跑**（lerobot 0.3.4+orca.1，数据集 v2.1）。
不要用 ``orcalab`` 环境：那边的 orca-gym 26.8.2 带相机推流但**没装 lerobot**。
本脚本只依赖 ``h5py`` / ``numpy`` / ``lerobot``，**不 import orca_gym**
（orcalab_lerobot 里的 orca-gym 26.7.3 没有 ``recorder/``，import 会失败）。

输入
----
``oracle_collect.py --output <前缀>`` 产出的每个 episode 一个文件：``<前缀>_ep000.hdf5`` …
    obs/qpos           (T,7) float32   6 关节角 + 夹爪净开口(m)
    action             (T,7) float32   6 关节目标角 + 夹爪目标开度(m)
    obs/camera_top     (T,3,224,224) uint8   空中俯视
    obs/camera_wrist   (T,3,224,224) uint8   腕部
    attrs: fps / task / success / seed / failure_reason / dropped_frames / cube_start / place_xy

输出
----
标准 LeRobot 数据集目录（``data/`` + ``videos/`` + ``meta/``），
可直接被 ``lerobot`` 的 ACT 训练配置读取。

用法
----
    /home/user/miniconda3/envs/orcalab_lerobot/bin/python to_lerobot.py \\
        --input oracle_pick \\
        --out /home/user/data/lerobot/pick_ur5e \\
        --repo-id local/pick_ur5e \\
        --only-success

注意：``--out`` 目录必须**不存在**（LeRobotDataset.create 用的是 exist_ok=False）。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

STATE7_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "gripper_width"]
IMG_NAMES = ["channels", "height", "width"]


def build_features(camera_keys: list[str], state_dim: int) -> dict:
    """按 HDF5 里实际存在的相机 key 生成 LeRobot features。"""
    feats = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,),
                              "names": STATE7_NAMES[:state_dim]},
        "action": {"dtype": "float32", "shape": (state_dim,),
                   "names": STATE7_NAMES[:state_dim]},
    }
    for key in camera_keys:
        feats["observation.images.%s" % key] = {
            "dtype": "video", "shape": (3, 224, 224), "names": IMG_NAMES}
    return feats


def list_episodes(prefix: str) -> list[str]:
    stem, ext = os.path.splitext(prefix)
    if not ext:
        ext, stem = ".hdf5", prefix
    return sorted(glob.glob("%s_ep*%s" % (stem, ext)))


def probe(files: list[str], only_success: bool):
    """先扫一遍：拿到相机 key 集合与可用 episode 列表。"""
    usable, camera_keys, skipped = [], None, []
    for path in files:
        with h5py.File(path, "r") as f:
            ok = bool(f.attrs.get("success", False))
            if only_success and not ok:
                skipped.append((path, "success=False"))
                continue
            if "obs/qpos" not in f or "action" not in f:
                skipped.append((path, "缺 obs/qpos 或 action"))
                continue
            keys = sorted(k for k in f["obs"].keys() if k.startswith("camera_"))
            if not keys:
                skipped.append((path, "没有相机数据（用了 --no-cameras？）"))
                continue
            if camera_keys is None:
                camera_keys = keys
            elif keys != camera_keys:
                skipped.append((path, "相机 key 不一致 %s != %s" % (keys, camera_keys)))
                continue
            usable.append(path)
    return usable, (camera_keys or []), skipped


def main():
    ap = argparse.ArgumentParser(description="HDF5 -> LeRobot v2.1 数据集")
    ap.add_argument("--input", required=True,
                    help="oracle_collect.py --output 的前缀，或直接给某个 .hdf5 文件")
    ap.add_argument("--out", required=True, help="输出数据集目录（必须不存在）")
    ap.add_argument("--repo-id", default="local/pick_ur5e", help="数据集 id，如 local/pick_ur5e")
    ap.add_argument("--fps", type=int, default=50, help="与控制频率一致，默认 50")
    ap.add_argument("--only-success", action="store_true", help="只转换 success=True 的 episode")
    ap.add_argument("--image-threads", type=int, default=4,
                    help="写 PNG 的后台线程数（0 = 主线程直写，最慢）")
    args = ap.parse_args()

    files = list_episodes(args.input)
    if not files:
        print("没找到 episode 文件：%s_ep*.hdf5" % os.path.splitext(args.input)[0])
        return 1
    usable, camera_keys, skipped = probe(files, args.only_success)
    for path, why in skipped:
        print("[skip] %s：%s" % (os.path.basename(path), why))
    if not usable:
        print("没有可转换的 episode（全部被跳过）。")
        return 1

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:                                  # noqa: BLE001
        print("import lerobot 失败：%s: %s\n"
              "请用 orcalab_lerobot 环境的 python 跑本脚本：\n"
              "  /home/user/miniconda3/envs/orcalab_lerobot/bin/python to_lerobot.py ..."
              % (type(e).__name__, e))
        return 1

    with h5py.File(usable[0], "r") as f:
        state_dim = int(f["obs/qpos"].shape[1])
    features = build_features(camera_keys, state_dim)
    print("features:\n  %s" % "\n  ".join(sorted(features)))
    print("待转换 %d 个 episode -> %s" % (len(usable), args.out))

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.out,
        robot_type="ur5e",
        use_videos=True,
        batch_encoding_size=1,           # 每集存完立即编码视频
        image_writer_threads=args.image_threads,
    )

    total_frames, total_dropped, used = 0, 0, 0
    for i, path in enumerate(usable):
        with h5py.File(path, "r") as f:
            qpos = np.asarray(f["obs/qpos"][:], dtype=np.float32)
            action = np.asarray(f["action"][:], dtype=np.float32)
            task = str(f.attrs.get("task", "") or "")
            dropped = int(f.attrs.get("dropped_frames", 0) or 0)
            n = min(len(qpos), len(action))
            images = {k: np.asarray(f["obs/%s" % k][:], dtype=np.uint8) for k in camera_keys}
            n = min([n] + [v.shape[0] for v in images.values()])
            if n <= 0:
                print("[skip] %s：帧数为 0" % os.path.basename(path))
                continue
            for t in range(n):
                frame = {"observation.state": qpos[t], "action": action[t]}
                for k, arr in images.items():
                    frame["observation.images.%s" % k] = arr[t]
                ds.add_frame(frame, task=task)   # task 是位置参数，不能塞进 frame
        ds.save_episode()
        total_frames += n
        total_dropped += dropped
        used += 1
        print("[%d/%d] %s  %d 帧  丢帧披露 %d" % (i + 1, len(usable),
                                                 os.path.basename(path), n, dropped))

    rate = 100.0 * total_dropped / max(1, total_frames)
    print("完成：%d 个 episode / %d 帧 -> %s" % (used, total_frames, args.out))
    print("采集侧披露的丢帧合计 %d（%.2f%%）" % (total_dropped, rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
