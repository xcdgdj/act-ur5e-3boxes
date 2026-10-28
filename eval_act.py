#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT 闭环评估：加载训练好的 checkpoint，在 OrcaStudio 仿真里连跑 N 集随机抓放。

与 oracle_collect.py 的关系
--------------------------
仿真侧（连 OrcaStudio、随机摆方块、取两路相机帧、50Hz 控制节拍）**完全复用**
``oracle_collect`` 的既有实现，只把「专家路点状态机」换成「ACT 策略输出」：

    obs   = [6 关节角 + 夹爪净开口(m)] + [camera_top, camera_wrist] 224x224 图像
    action= 同一个 7 维空间（6 关节目标角 + 夹爪目标净开口）

图像与 state 的预处理必须与训练时逐位一致：
    · 图像 uint8 (3,224,224) -> float32/255（训练集的 video 解码就是 [0,1]）
    · 归一化由 checkpoint 里自带的 mean/std buffer 完成，脚本不重复做

环境
----
必须用 ``orcalab_gpu``：它同时具备
    · torch 2.7.1+cu128 + lerobot 0.3.4（ACT 推理，要 GPU）
    · orca_gym 26.8.2 / mujoco / scipy / grpcio（连 OrcaStudio 跑物理 + 取相机帧）

前提
----
OrcaStudio 必须已进入「运行」模式（50051 在监听、两路相机在推流），否则直接报错退出。

用法
----
    /home/user/miniconda3/envs/orcalab_gpu/bin/python eval_act.py \\
        --ckpt /home/user/orca_train/act_pick_ur5e/checkpoints/080000/pretrained_model \\
        --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import oracle_collect as oc                                  # noqa: E402 复用仿真/相机/摆位
import mujoco                                                # noqa: E402
import torch                                                 # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy      # noqa: E402

CUBE_IN_PLACE_TOL = 0.05      # 与采集侧判定一致：方块 xy 距指定放置点 5 cm 内算放好
LIFT_EPS = 0.05               # 方块中心抬升超过 5 cm 才算「被抓住过」
STABLE_STEPS = 25             # 三个方块都在位且静止这么久 -> 提前判成功
STAB_VEL = 0.05               # 静止判定：方块线速度 (m/s) 上限
PRINT_EVERY = 250             # 每多少拍打一行进度
FALL_Z = -0.05                # 方块掉到桌面以下就认为掉出台面


class DummyRecorder:
    """OracleCollector 只需要 recorder.record_cameras 这一个开关，其余接口空实现。"""

    record_cameras = True

    def start_episode(self):
        pass

    def discard_episode(self):
        pass

    def record(self, *a, **k):
        pass

    def record_images(self, *a, **k):
        pass

    def end_episode(self, *a, **k):
        return ("", 0)

    def close(self):
        pass


class ActRunner:
    """ACT 策略封装：batch 组装 + 逐步取动作（内部 100 步 action chunk 队列）。

    ``temporal_ensemble_coeff`` 非 None 时启用 ACT 论文的 temporal ensembling：
    每一拍都重新推理整条 chunk，再对所有重叠预测做指数加权平均
    （权重 wᵢ = exp(-coeff·i)，越老的预测权重越大）。这条路径不走 action 队列，
    因此 ``needs_obs`` 恒为 True —— 每一拍都必须喂图像。
    """

    def __init__(self, ckpt: str, device: str = "cuda",
                 temporal_ensemble_coeff: float | None = None):
        self.policy = ACTPolicy.from_pretrained(ckpt)
        coeff = None if temporal_ensemble_coeff is None else float(temporal_ensemble_coeff)
        if coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            # from_pretrained 只在 config 已带 coeff 时才建 ensembler，
            # 这里在加载后补建，chunk_size 取自 checkpoint 自身配置。
            self.policy.config.temporal_ensemble_coeff = coeff
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                coeff, self.policy.config.chunk_size)
        self.temporal_ensemble_coeff = coeff
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        self.cfg = self.policy.config
        # 例如 ["observation.images.camera_top", "observation.images.camera_wrist"]
        self.image_keys = list(self.cfg.image_features or [])

    def reset(self):
        self.policy.reset()                                # 清空 action chunk 队列 / ensembler 状态

    @property
    def needs_obs(self) -> bool:
        """是否需要新的观测（图像）。

        普通模式：队列空了才推理；temporal ensembling：每拍都要推理。
        """
        if self.temporal_ensemble_coeff is not None:
            return True
        return len(self.policy._action_queue) == 0

    def _batch(self, state7: np.ndarray, images: dict[str, np.ndarray]) -> dict:
        batch = {
            "observation.state": torch.from_numpy(np.asarray(state7, dtype=np.float32))
            .unsqueeze(0).to(self.device)
        }
        for key in self.image_keys:
            tag = "camera_" + key.rsplit("camera_", 1)[-1]     # observation.images.camera_top -> camera_top
            img = images.get(tag)
            if img is None:
                raise RuntimeError("缺相机图像：%s（key=%s）" % (tag, key))
            t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
            batch[key] = t.unsqueeze(0).to(self.device)
        return batch

    @torch.inference_mode()
    def act(self, state7: np.ndarray, images: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """返回 7 维原始动作（rad / m）。队列非空时 images 可为 None。"""
        batch = self._batch(state7, images) if self.needs_obs else {}
        action = self.policy.select_action(batch)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def predict_chunk(self, state7: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        """一次前向拿完整 chunk（仅用于开跑前的输入约定自检）。"""
        return self.policy.predict_action_chunk(
            self._batch(state7, images))[0].detach().cpu().numpy().astype(np.float64)


def sanity_check(runner: ActRunner, hdf5_path: str, tol: float) -> bool:
    """用专家数据的第一帧喂策略，对比预测 chunk 与专家轨迹的 L1 误差。

    这一检查能一次性抓住「图像缩放 / 归一化 / state 顺序 / 动作含义」任一处约定错配 ——
    如果这里就对不上，闭环跑 20 集只会白跑。
    """
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        qpos = np.asarray(f["obs/qpos"][:], dtype=np.float32)
        action = np.asarray(f["action"][:], dtype=np.float32)
        images = {}
        for tag in ("camera_top", "camera_wrist"):
            if "obs/%s" % tag in f:
                images[tag] = np.asarray(f["obs/%s" % tag][0], dtype=np.uint8)
    n = min(len(qpos), len(action))
    chunk = runner.predict_chunk(qpos[0], images)          # (chunk_size, 7)
    k = min(len(chunk), n)
    ref = action[:k]
    pred = chunk[:k]
    l1 = np.abs(pred - ref)
    print("[sanity] 专家数据 %s 第 0 帧 -> 策略 chunk 前 %d 步"
          % (os.path.basename(hdf5_path), k))
    print("[sanity] chunk L1  mean=%.4f  max=%.4f  (关节 rad / 夹爪 m)"
          % (float(l1.mean()), float(l1.max())))
    print("[sanity] 逐维 L1  mean=%s" % np.round(l1.mean(axis=0), 4).tolist())
    print("[sanity] 首帧 qpos=%s" % np.round(qpos[0], 4).tolist())
    print("[sanity] 首步 专家=%s" % np.round(ref[0], 4).tolist())
    print("[sanity] 首步 策略=%s" % np.round(pred[0], 4).tolist())
    ok = float(l1.mean()) <= tol
    print("[sanity] %s（阈值 mean<=%.3f）" % ("通过" if ok else "!! 不通过", tol), flush=True)
    return ok


class ActEvaluator:
    """复用 OracleCollector 的仿真/相机基建，把专家状态机换成 ACT 策略。"""

    def __init__(self, env, runner: ActRunner, *, cameras=oc.CAM_TARGETS,
                 real_time: bool = True, max_steps: int = 3000, seed=None,
                 dump_dir: str | None = None, idr_every: int = 5, replan: int = 0,
                 force_open_dist: float = 0.0, force_open_near: int = 5,
                 force_open_hold: int = 15):
        self.env = env
        self.runner = runner
        self.max_steps = int(max_steps)
        self.dump_dir = dump_dir
        self.idr_every = max(0, int(idr_every))
        self.replan = max(0, int(replan))
        # temporal ensembling 每拍都要推理；此时不能再「每次推理都补 IDR」，
        # 否则 50Hz 下每帧都是关键帧，编码器扛不住。改走 idr_every 固定节拍。
        self.temporal_ensemble = getattr(runner, "temporal_ensemble_coeff", None) is not None
        # 事件兜底（方案 4）：TCP 贴近放置点且持续下令合爪 N 拍时，强制张爪，
        # 用于验证「观测别名导致的静止死锁」之外，策略的其余部分是否健康。
        self.force_open_dist = float(force_open_dist)
        self.force_open_near = max(1, int(force_open_near))
        self.force_open_hold = max(1, int(force_open_hold))
        self._place_z_tcp = float(oc.CUBE_Z + oc.TRAY_FLOOR_DZ + oc.GRASP_DZ + oc.LOWER_DZ)
        # 「下令合爪」的判据必须用指令宽度（专家合爪指令 0.045），不能用 GRIP_GAP_CLOSED(0.0268)。
        # 取全开/全闭的中点 0.0458：合爪 0.045 判为闭，张开 0.0648 判为开。
        self._closed_thr = 0.5 * (oc.GRIP_GAP_CLOSED + oc.GRIP_GAP_OPEN)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        self.collector = oc.OracleCollector(env, DummyRecorder(), seed=seed,
                                            real_time=real_time, cameras=cameras)
        c = self.collector
        self.m, self.d = c.m, c.d
        self.cubes = c.cubes
        self.bids = {color: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
                     for color, body in self.cubes.items()}
        if any(b < 0 for b in self.bids.values()):
            raise RuntimeError("方块 body 解析失败：%s" % self.bids)

    # ---------------------------------------------------------------- 相机
    def open_cameras(self) -> bool:
        return self.collector.open_cameras()

    def _latest_images(self, idx: int, timeout: float = 15.0,
                       strict: bool = False) -> tuple[dict[str, np.ndarray], int]:
        """取第 idx 拍的两路图像。返回 (图像字典, 实际用到的最大落后拍数)。

        strict=True（策略真正要读图的那一拍）：必须等到**正好第 idx 帧**，否则
        退化成「保持上一帧」，而采集侧的约定是「一帧数据 = 该拍的状态 + 该拍的图」，
        图错拍等于给策略喂错时序信息 —— 这是 ACT 闭环里最致命的输入错配。
        strict=False（兜底）：该拍还没到就用最近的一张（hold-last），只用于预热。
        """
        cams = self.collector.cams
        deadline = time.time() + timeout
        while True:
            out, missing = {}, []
            for key, st in cams.streams.items():
                if strict:
                    f = st["frames"].get(idx)
                else:
                    f = st["frames"].get(idx)
                    if f is None:
                        cands = [i for i in st["frames"] if i <= idx]
                        f = st["frames"][max(cands)] if cands else None
                if f is None:
                    missing.append(key)
                else:
                    out[key] = f
            if not missing:
                return out, 0
            if time.time() >= deadline:
                if not strict:
                    raise RuntimeError("相机 %s 没有可用帧（确认 OrcaStudio 在运行模式、推流正常）"
                                       % missing)
                break
            cams.harvest()
            time.sleep(0.005)
        # strict 超时：退化成 hold-last，把落后拍数报出去（不静默）
        lag = 0
        for key, st in cams.streams.items():
            if key in out:
                continue
            cands = [i for i in st["frames"] if i <= idx]
            if not cands:
                raise RuntimeError("相机 %s 连旧帧都没有（确认 OrcaStudio 在运行模式、推流正常）"
                                   % key)
            j = max(cands)
            out[key] = st["frames"][j]
            lag = max(lag, idx - j)
        return out, lag

    def _trim_frames(self, idx: int):
        """滚动缓存只保留最近 3 帧，避免长 episode 把内存吃满。"""
        for st in self.collector.cams.streams.values():
            for i in [i for i in st["frames"] if i < idx - 2]:
                st["frames"].pop(i, None)

    def _dump_frames(self, ep: int, t: int, images: dict[str, np.ndarray]):
        """把策略实际看到的图像存成 PNG（排查「策略不动」时对比训练数据）。"""
        from PIL import Image
        for tag, img in images.items():
            arr = np.transpose(np.asarray(img)[:3], (1, 2, 0))
            Image.fromarray(arr).save(os.path.join(self.dump_dir,
                                                   "ep%02d_t%04d_%s.png" % (ep, t, tag)))

    # ---------------------------------------------------------------- 单集
    def run_episode(self, index: int, verbose: bool = True) -> dict:
        c, env = self.collector, self.env
        env.reset()
        self.m.opt.timestep = oc.TIME_STEP
        c.randomizer.reset(self.d, c.rng)
        env.gym.set_ctrl(c._zero_ctrl())
        env.gym.mj_step(int(0.5 / oc.TIME_STEP))           # 让方块落稳
        # 与采集侧 reset 完全同口径：episode 起点就是「夹爪张开」。
        open_ctrl = c._zero_ctrl()
        open_ctrl[c.arm.grip_act] = oc.grip_ctrl_for_width(oc.GRIP_GAP_OPEN)
        env.gym.set_ctrl(open_ctrl)
        env.gym.mj_step(int(0.5 / oc.TIME_STEP))           # 等夹爪张到位
        self.runner.reset()

        place_xy = [c._place_xy(i) for i in range(3)]
        start = {color: self.d.xpos[bid].copy() for color, bid in self.bids.items()}
        peak = {color: float(start[color][2]) for color in self.bids}
        cams_on = c.cams.active

        t0 = time.perf_counter()
        base = c._next_sim_index()
        n_black = n_jump = n_infer = n_lag = max_lag = 0
        near_cnt = override_left = n_force = 0
        prev_action = None
        stable = 0
        timeout = True

        # 预热一拍：把 reset 后的首帧渲染出来（夹爪保持张开，与采集侧 reset 后同口径），
        # 让「相机帧」和「策略要读的 state」同拍对齐。
        # （数据约定 = obs-before-action：一帧 = 动作**之前**的 state + 该帧图像，
        #   配上接下来要施加的动作；评估必须与重新采集的训练数据同口径。）
        neutral = np.zeros(7, dtype=np.float64)
        neutral[6] = oc.GRIP_GAP_OPEN                   # 夹爪保持张开；arm 通道全 0 -> 保持 home
        cur = base                                      # cur = 当前 state 对应的帧号
        c._apply_and_step(neutral, sim_idx=cur if cams_on else None, request_idr=True)
        if cams_on:
            c.cams.request(cur)
            c.cams.harvest()

        for t in range(self.max_steps):
            if self.replan and t and t % self.replan == 0:
                # 到点就丢掉当前 chunk 的剩余动作，强制用「当前观测」重新推理一次。
                # 训练时是 100 步开环执行一个 chunk；这里可以调小来加快闭环纠偏。
                self.runner.reset()
            state = c._state7()                         # 与帧 cur 同拍的观测
            did_infer = self.runner.needs_obs
            if did_infer:
                images, lag = self._latest_images(cur, timeout=2.0, strict=True)
                if lag:
                    n_lag += 1
                    max_lag = max(max_lag, lag)
                if any(int(np.count_nonzero(img)) == 0 for img in images.values()):
                    n_black += 1
                action = self.runner.act(state, images)
                n_infer += 1
                if verbose:
                    print("    [infer %2d @t=%4d] state=%s"
                          % (n_infer, t, np.round(state, 4).tolist()), flush=True)
                    print("                       act  =%s"
                          % np.round(action, 4).tolist(), flush=True)
                if self.dump_dir:
                    self._dump_frames(index, t, images)
            else:
                action = self.runner.act(state)
            if prev_action is not None and float(np.abs(action[:6] - prev_action[:6]).max()) > 0.5:
                n_jump += 1
            prev_action = action[:6]
            # 与采集侧同口径的限幅：关节进关节限位，夹爪净开口进 [全闭, 全开]
            act7 = np.array(action, dtype=np.float64)
            act7[:6] = c.arm.clamp(act7[:6])
            act7[6] = float(np.clip(act7[6], oc.GRIP_GAP_CLOSED, oc.GRIP_GAP_OPEN))

            # ---- 事件兜底（方案 4）：强制张爪，打破「观测别名」静止死锁 ----
            # 触发条件：TCP 距某个放置点 < force_open_dist，且策略持续 force_open_near 拍
            # 都在下令合爪。触发后连续 force_open_hold 拍把夹爪覆盖成全开（保证方块脱手），
            # 张爪本身会改变图像 -> 观测不再静止 -> 策略可以重新推进。
            if self.force_open_dist > 0.0:
                tcp = c.arm.tcp_pos()
                dmin = min(float(np.linalg.norm(tcp - np.array([p[0], p[1], self._place_z_tcp])))
                           for p in place_xy)
                if verbose and t % 50 == 0 and dmin < 0.06:
                    print("    [near-place t=%4d] TCP 距放置点 %.4f m  夹爪指令=%.4f  连续合爪=%d"
                          % (t, dmin, act7[6], near_cnt), flush=True)
                if override_left > 0:
                    act7[6] = float(oc.GRIP_GAP_OPEN)
                    override_left -= 1
                elif dmin < self.force_open_dist and act7[6] <= self._closed_thr:
                    near_cnt += 1
                    if near_cnt >= self.force_open_near:
                        override_left = self.force_open_hold - 1
                        near_cnt = 0
                        n_force += 1
                        act7[6] = float(oc.GRIP_GAP_OPEN)
                        print("    [force-open %d @t=%4d] TCP 距放置点 %.4f m，"
                              "连续 %d 拍下令合爪 -> 强制张爪 %d 拍"
                              % (n_force, t + 1, dmin, self.force_open_near,
                                 self.force_open_hold), flush=True)
                else:
                    near_cnt = 0

            cur += 1                                    # 下一帧的帧号 = 本拍渲染的那一帧
            # 推流是 H.264 + 单个持久解码器：一旦传输里丢了 NAL，参考链断掉，后面全是花屏，
            # 只有下一个 IDR 才能重新同步（采集侧整集只发一次 IDR，之前流干净所以没暴露）。
            # 这里按固定节拍强制补 IDR，把「一次丢包 -> 整集花屏」压到最多 idr_every 帧。
            c._apply_and_step(act7, sim_idx=cur if cams_on else None,
                              request_idr=((did_infer and not self.temporal_ensemble) or
                                           (self.idr_every and cur % self.idr_every == 0)))
            if cams_on:
                c.cams.request(cur)
                c.cams.harvest()
                self._trim_frames(cur)

            for color, bid in self.bids.items():
                peak[color] = max(peak[color], float(self.d.xpos[bid][2]))
            if t % PRINT_EVERY == 0 and verbose:
                errs = [float(np.linalg.norm(self.d.xpos[self.bids[cl]][:2] - place_xy[i]))
                        for i, cl in enumerate(oc.PICK_ORDER)]
                print("    [t=%4d] 距放置点 %s m  夹爪开度=%.4f m"
                      % (t, " ".join("%.3f" % e for e in errs),
                         float(c.arm.grip_state()[0])), flush=True)

            in_place = [float(np.linalg.norm(self.d.xpos[self.bids[cl]][:2] - place_xy[i]))
                        < CUBE_IN_PLACE_TOL for i, cl in enumerate(oc.PICK_ORDER)]
            lifted = [peak[cl] - float(start[cl][2]) > LIFT_EPS for cl in oc.PICK_ORDER]
            vmax = max(float(np.linalg.norm(self.d.qvel[self._dof_of(bid):self._dof_of(bid) + 3]))
                       for bid in self.bids.values())
            stable = stable + 1 if (all(in_place) and all(lifted) and vmax < STAB_VEL) else 0
            if stable >= STABLE_STEPS:
                timeout = False
                break
        elapsed = time.perf_counter() - t0

        result = {"episode": index, "steps": t + 1, "seconds": round(elapsed, 1),
                  "infer_calls": n_infer, "black_frames": n_black, "jump_actions": n_jump,
                  "stale_obs": n_lag, "max_obs_lag": max_lag,
                  "force_open": n_force,
                  "timeout": timeout, "cubes": {}}
        for i, color in enumerate(oc.PICK_ORDER):
            bid = self.bids[color]
            pos = self.d.xpos[bid]
            err = float(np.linalg.norm(pos[:2] - place_xy[i]))
            result["cubes"][color] = {
                "place_err": round(err, 4),
                "in_place": bool(err < CUBE_IN_PLACE_TOL),
                "lifted": bool(peak[color] - float(start[color][2]) > LIFT_EPS),
                "final_z": round(float(pos[2]), 4),
                "fell": bool(float(pos[2]) < FALL_Z),
                "tilt_deg": round(float(oc.cube_tilt_deg(self.d, bid)), 1),
            }
        result["success"] = all(v["in_place"] for v in result["cubes"].values())
        return result

    def _dof_of(self, bid: int) -> int:
        """方块 free joint 的 dof 起始下标（速度用）。"""
        return int(self.m.body_dofadr[bid])

    def close(self):
        self.collector.cams.close()


def summarize(results: list[dict]) -> None:
    """打印总成功率 + 每个方块的成败 + 异常现象清单。"""
    ok = [r for r in results if r["success"]]
    print("\n" + "=" * 78)
    print("闭环评估结果（%d 集）" % len(results))
    print("=" * 78)
    print("整集成功率（3 个方块全部落到各自指定放置点 5 cm 内）：%d/%d = %.1f%%"
          % (len(ok), len(results), 100.0 * len(ok) / max(1, len(results))))
    for color in oc.PICK_ORDER:
        n = sum(1 for r in results if r["cubes"][color]["in_place"])
        nl = sum(1 for r in results if r["cubes"][color]["lifted"])
        print("  %-6s 放到位 %2d/%d   曾被抓起 %2d/%d   平均放置误差 %.4f m"
              % (color, n, len(results), nl, len(results),
                 float(np.mean([r["cubes"][color]["place_err"] for r in results]))))
    steps = [r["steps"] for r in results]
    print("单集步数 平均 %.0f 拍（%.1f s）  最长 %d 拍  超时集 %d"
          % (float(np.mean(steps)), float(np.mean([r["seconds"] for r in results])),
             max(steps), sum(1 for r in results if r["timeout"])))
    print("推理次数 平均 %.1f 次/集（=每 100 拍一次）" % np.mean([r["infer_calls"] for r in results]))
    print("全黑帧 %d   关节跳变(>0.5 rad/拍) %d   掉出台面 %d   观测帧落后 %d 次（最多 %d 拍）"
          % (sum(r["black_frames"] for r in results),
             sum(r["jump_actions"] for r in results),
             sum(1 for r in results for v in r["cubes"].values() if v["fell"]),
             sum(r["stale_obs"] for r in results),
             max([r["max_obs_lag"] for r in results] or [0])))

    print("\n逐集明细：")
    for r in results:
        tag = "成功" if r["success"] else "失败"
        detail = "  ".join("%s%s(%.3f)" % (
            "" if v["in_place"] else "-", color[0].upper(), v["place_err"])
            for color, v in r["cubes"].items())
        print("  #%02d %s  %4d 拍 %5.1fs  推理%3d  %s"
              % (r["episode"], tag, r["steps"], r["seconds"], r["infer_calls"], detail))

    print("\n异常现象：")
    anomalies = []
    for r in results:
        for color, v in r["cubes"].items():
            if v["fell"]:
                anomalies.append("#%02d %s 掉出台面" % (r["episode"], color))
            elif not v["lifted"]:
                anomalies.append("#%02d %s 从未被抓起（策略没碰到方块）" % (r["episode"], color))
            elif not v["in_place"]:
                anomalies.append("#%02d %s 抬起过但没放到位（误差 %.3f m，倾角 %.1f°）"
                                 % (r["episode"], color, v["place_err"], v["tilt_deg"]))
        if r["timeout"] and not r["success"]:
            anomalies.append("#%02d 跑满 %d 拍上限仍未放完" % (r["episode"], r["steps"]))
        if r["black_frames"]:
            anomalies.append("#%02d 出现 %d 次全黑帧（相机丢帧）" % (r["episode"], r["black_frames"]))
        if r["jump_actions"]:
            anomalies.append("#%02d 出现 %d 次关节大跳变（策略输出抖动）"
                             % (r["episode"], r["jump_actions"]))
    print("  " + ("\n  ".join(anomalies) if anomalies else "无"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="ACT 闭环评估（OrcaStudio 仿真内跑随机抓放）")
    ap.add_argument("--ckpt",
                    default="/home/user/orca_train/act_pick_ur5e/checkpoints/080000/pretrained_model",
                    help="checkpoint 的 pretrained_model 目录（含 config.json + model.safetensors）")
    ap.add_argument("--episodes", type=int, default=20, help="跑多少集（默认 20）")
    ap.add_argument("--max-steps", type=int, default=3000,
                    help="单集最长拍数（50Hz，3000 拍 = 60 s）")
    ap.add_argument("--addr", default=oc.DEFAULT_ADDR, help="OrcaStudio gRPC 地址")
    ap.add_argument("--device", default="cuda", help="推理设备")
    ap.add_argument("--seed", type=int, default=None, help="随机摆位种子（复现用）")
    ap.add_argument("--fast", action="store_true",
                    help="不按实时节流、尽快跑完（默认 50Hz 实时，相机不易丢帧）")
    ap.add_argument("--cameras", default="%s=%s,%s=%s"
                    % (oc.CAMERA_TOP, oc.CAM_TARGETS[0][1], oc.CAMERA_WRIST, oc.CAM_TARGETS[1][1]),
                    help="obs key -> 场景相机名 的映射，必须与采集时一致")
    ap.add_argument("--sanity-h5", default=os.path.join(here, "hdf5", "oracle_pick_ep0000.hdf5"),
                    help="用哪一集的专家数据做输入约定自检（传空串跳过）")
    ap.add_argument("--sanity-tol", type=float, default=0.30,
                    help="自检的 chunk L1 均值阈值（rad），超了就停止并报错")
    ap.add_argument("--force", action="store_true", help="自检不通过也照跑")
    ap.add_argument("--dump-frames", default="",
                    help="把策略每拍看到的图像存成 PNG 到这个目录（排查用，默认不存）")
    ap.add_argument("--idr-every", type=int, default=5,
                    help="每多少拍强制给推流补一个 IDR 关键帧（0=只在推理后补；丢包花屏的自救）")
    ap.add_argument("--replan", type=int, default=0,
                    help="每多少拍丢弃剩余 chunk 重新推理（0=按训练设置每 100 拍一次）")
    ap.add_argument("--temporal-ensemble-coeff", type=float, default=None,
                    help="启用 temporal ensembling 的指数权重系数（ACT 论文默认 0.01；"
                         "传 0 表示等权平均；不传则关闭）")
    ap.add_argument("--force-open-dist", type=float, default=0.0,
                    help="事件兜底：TCP 距放置点小于该值(m)且持续合爪时强制张爪；0=关闭")
    ap.add_argument("--force-open-near", type=int, default=5,
                    help="事件兜底：需要连续多少拍满足「贴近放置点 + 下令合爪」才触发")
    ap.add_argument("--force-open-hold", type=int, default=15,
                    help="事件兜底：触发后连续强制张爪多少拍")
    ap.add_argument("--out", default=os.path.join(here, "eval_results_act.json"),
                    help="逐集结果 JSON 落盘路径")
    args = ap.parse_args()

    cameras = oc.parse_cameras(args.cameras)
    print("[cfg] ckpt=%s" % args.ckpt)
    print("[cfg] 相机映射=%s  设备=%s  %s"
          % (cameras, args.device, "尽快跑" if args.fast else "50Hz 实时"))
    runner = ActRunner(args.ckpt, device=args.device,
                       temporal_ensemble_coeff=args.temporal_ensemble_coeff)
    print("[cfg] policy chunk_size=%d n_action_steps=%d 图像输入=%s"
          % (runner.cfg.chunk_size, runner.cfg.n_action_steps, runner.image_keys))
    print("[cfg] temporal_ensemble_coeff=%s%s"
          % (runner.temporal_ensemble_coeff,
             "（每拍重推理 + 指数加权平均）" if runner.temporal_ensemble_coeff is not None
             else "（关闭：按 chunk 顺序开环执行）"))

    if args.sanity_h5 and os.path.exists(args.sanity_h5):
        if not sanity_check(runner, args.sanity_h5, args.sanity_tol) and not args.force:
            print("自检未通过：输入约定可能错配，先别跑闭环。确认无误可加 --force 跳过。")
            return 2

    env = oc.build_env(args.addr)
    oc.scene_handshake(args.addr, "beginscene")
    ev = ActEvaluator(env, runner, cameras=cameras, real_time=not args.fast,
                      max_steps=args.max_steps, seed=args.seed,
                      dump_dir=args.dump_frames or None, idr_every=args.idr_every,
                      replan=args.replan,
                      force_open_dist=args.force_open_dist,
                      force_open_near=args.force_open_near,
                      force_open_hold=args.force_open_hold)
    if not ev.open_cameras():
        print("相机没接上（%s），策略拿不到图像，退出。" % ev.collector.cams.error)
        return 3

    results = []
    t0 = time.perf_counter()
    try:
        for k in range(args.episodes):
            print("\n[第 %d/%d 集]" % (k + 1, args.episodes), flush=True)
            try:
                r = ev.run_episode(k, verbose=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:                          # noqa: BLE001
                print("  [episode] 异常：%s: %s" % (type(e).__name__, e), flush=True)
                r = None
            if r is not None:
                results.append(r)
                print("  -> %s  %d 拍 / %.1fs  推理 %d 次  方块误差 %s"
                      % ("成功" if r["success"] else "失败", r["steps"], r["seconds"],
                         r["infer_calls"],
                         " ".join("%.3f" % v["place_err"] for v in r["cubes"].values())),
                      flush=True)
            n_ok = sum(1 for x in results if x["success"])
            print("  [进度] 累计成功 %d/%d  已用 %s"
                  % (n_ok, len(results), oc._fmt_hms(time.perf_counter() - t0)), flush=True)
    except KeyboardInterrupt:
        print("\n中断：已跑完的结果下面照样汇总。")
    finally:
        oc.scene_handshake(args.addr, "endscene")
        print("[cam] %s" % ev.collector.cams.drop_summary())
        ev.close()

    if results:
        summarize(results)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "episodes": args.episodes,
                       "success_rate": sum(1 for r in results if r["success"]) / len(results),
                       "results": results}, f, indent=2, ensure_ascii=False)
        print("\n逐集结果已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
