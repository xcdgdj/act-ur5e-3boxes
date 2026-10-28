#!/usr/bin/env python3
"""LeRobot 数据集验毒：帧对齐 + 数值健康度 + 与源 HDF5 交叉核对。

用法：
    python check_dataset.py                  # 随机抽 1 集做全量检查（含解码视频）
    python check_dataset.py --episode 7      # 指定某一集
    python check_dataset.py --n 3            # 随机抽 3 集
    python check_dataset.py --all            # 100 集只查帧数（不解码视频，快）
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DS = os.path.join(HERE, "lerobot", "pick_ur5e")
H5_TPL = os.path.join(HERE, "hdf5", "oracle_pick_ep%04d.hdf5")
CAMS = ("camera_top", "camera_wrist")
JUMP_WARN = 0.5          # qpos/action 相邻帧最大跳变告警阈值（rad）
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]


def _chunk(ep):
    return ep // 1000


def parquet_path(ds, ep):
    return os.path.join(ds, "data", "chunk-%03d" % _chunk(ep), "episode_%06d.parquet" % ep)


def video_path(ds, ep, cam):
    return os.path.join(ds, "videos", "chunk-%03d" % _chunk(ep),
                        "observation.images." + cam, "episode_%06d.mp4" % ep)


def count_video_frames(path):
    import av
    with av.open(path) as c:
        return sum(1 for _ in c.decode(video=0))


def first_frames(path, n=1):
    import av
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb24"))
            if len(out) >= n:
                break
    return out


def h5_frames(ep):
    """返回源 HDF5 里各键的帧数，文件不存在返回 None。"""
    import h5py
    p = H5_TPL % ep
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as f:
        d = {"qpos": f["obs/qpos"].shape[0], "action": f["action"].shape[0]}
        for cam in CAMS:
            k = "obs/" + cam
            if k in f:
                d[cam] = f[k].shape[0]
        d["_success"] = bool(f.attrs.get("success", True))
    return d


def print_matrix(title, arr, names):
    print("  " + title)
    hdr = "    idx | " + " ".join("%9s" % n for n in names)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for i, row in enumerate(arr):
        print("    %3d | " % i + " ".join("%9.5f" % v for v in row))


def check_one(ds, ep, full=True):
    print("=" * 78)
    print("episode %d" % ep)
    print("=" * 78)
    ok = True

    # ---- 1. parquet 本体 ----
    pp = parquet_path(ds, ep)
    if not os.path.exists(pp):
        print("  [x] parquet 缺失：%s" % pp)
        return False
    t = pq.read_table(pp)
    n_pq = t.num_rows
    state = np.asarray(t["observation.state"].to_pylist(), dtype=np.float64)
    action = np.asarray(t["action"].to_pylist(), dtype=np.float64)
    counts = {"qpos": n_pq, "action": n_pq}

    # ---- 2. 视频帧数 ----
    if full:
        for cam in CAMS:
            vp = video_path(ds, ep, cam)
            if not os.path.exists(vp):
                print("  [x] 视频缺失：%s" % vp)
                ok = False
                continue
            counts[cam] = count_video_frames(vp)

    # ---- 3. 源 HDF5 对照 ----
    h5 = h5_frames(ep)
    if h5:
        for k in ("qpos", "action", "camera_top", "camera_wrist"):
            if k in h5:
                counts["h5." + k] = h5[k]

    # ---- 4. 对齐判定 ----
    print("\n[帧数对齐]")
    for k, v in counts.items():
        print("    %-18s %7d" % (k, v))
    vals = set(counts.values())
    if len(vals) == 1:
        print("    -> [v] 全部一致：%d 帧" % n_pq)
    else:
        print("    -> [x] 不一致：min=%d max=%d 差=%d" % (min(vals), max(vals), max(vals) - min(vals)))
        ok = False

    # ---- 5. 数值健康度 ----
    print("\n[数值健康度]")
    for name, arr in (("qpos", state), ("action", action)):
        bad = (~np.isfinite(arr)).sum()
        if bad:
            print("    [x] %s 含 %d 个 NaN/Inf" % (name, bad))
            ok = False
        d = np.abs(np.diff(arr, axis=0))
        jmax, jidx = (d.max(), d.argmax()) if d.size else (0.0, -1)
        print("    %-6s shape=%s  finite=%s  相邻帧最大跳变=%.5f (第%d维/第%d帧)  范围=[%.3f, %.3f]"
              % (name, arr.shape, "OK" if not bad else "有NaN",
                 jmax, jidx % arr.shape[1], jidx // arr.shape[1], arr.min(), arr.max()))
        if jmax > JUMP_WARN:
            print("    [!] %s 跳变 %.5f 超过 %.2f，疑似异常突变" % (name, jmax, JUMP_WARN))
            ok = False
        if not (np.isfinite(arr).all()):
            ok = False

    # ---- 6. 前 5 帧数值 ----
    n5 = min(5, n_pq)
    print("\n[前 %d 帧 qpos（observation.state）]" % n5)
    print_matrix("", state[:n5], JOINTS)
    print("\n[前 %d 帧 action]" % n5)
    print_matrix("", action[:n5], JOINTS)

    # ---- 7. 首帧是否黑屏 ----
    if full:
        print("\n[首帧画面]")
        for cam in CAMS:
            vp = video_path(ds, ep, cam)
            if not os.path.exists(vp):
                continue
            fr = first_frames(vp, 1)
            if not fr:
                print("    [x] %s 解不出帧" % cam); ok = False; continue
            f = fr[0].astype(np.float32)
            nz = (f.max(axis=2) > 8).mean()
            print("    %-12s shape=%s  均值=%.1f  非黑像素占比=%.1f%%  %s"
                  % (cam, f.shape, f.mean(), nz * 100,
                     "[v]" if nz > 0.5 else "[!] 疑似黑屏/近黑"))
            if nz <= 0.5:
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    ds = a.dataset
    n_ep = len(glob.glob(os.path.join(ds, "data", "chunk-*", "*.parquet")))
    if n_ep == 0:
        print("数据集里没有 parquet：%s" % ds); sys.exit(1)
    print("数据集 %s（%d 集）" % (ds, n_ep))

    if a.all:
        print("\n[全量帧数对齐]")
        bad = []
        for ep in range(n_ep):
            t = pq.read_table(parquet_path(ds, ep))
            n = t.num_rows
            h5 = h5_frames(ep)
            ref = h5["qpos"] if h5 else None
            flag = "" if (ref is None or ref == n) else "  [x] 源 HDF5=%d" % ref
            if ref is not None and ref != n:
                bad.append(ep)
            print("    ep%04d  %6d 帧%s" % (ep, n, flag))
        print("\n  -> %s" % ("[v] 全部与源 HDF5 一致" if not bad
                             else "[x] 不一致的集：%s" % bad))
        sys.exit(0 if not bad else 1)

    if a.episode is not None:
        eps = [a.episode]
    else:
        rng = random.Random(a.seed)
        eps = rng.sample(range(n_ep), min(a.n, n_ep))
        print("随机抽取：%s" % eps)

    allok = True
    for ep in eps:
        allok &= check_one(ds, ep, full=True)
        print()
    print("=" * 78)
    print("结论：%s" % ("[v] 全部通过" if allok else "[x] 有问题，见上面 [x]/[!] 行"))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
