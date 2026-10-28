# UR5e 随机抓取 — ACT 模仿学习（OrcaSim / MuJoCo）

在 OrcaStudio（MuJoCo）仿真里，用 UR5e + 两指夹爪把桌面上**随机摆放**的红/黄/绿三个方块依次抓取，并放到托盘的三个指定槽位。

整条链路：**专家数据采集 → LeRobot v2.1 数据集 → ACT 训练 → 闭环评估**，全部脚本都在这一个仓库里。

---

## 1. 结果速览

### 闭环抓放成功率（OrcaStudio 实时仿真，每集 3 个方块）

| 权重 | 集数 | 整集成功率 | 方块级成功 | 平均放置误差 | 倾角 | 兜底触发 |
|---|---|---|---|---|---|---|
| **400K（final）** | 3 | **3/3 = 100%** | 9/9 | 5 ~ 29 mm | 0° | 0 次 |
| **400K（final）** | 10 | **9/10 = 90%** | 27/30 | 5 ~ 47 mm | 0° | 0 次 |
| 375K（对照） | 10 | 8/10 = 80% | 24/30 | 6 ~ 39 mm | 0° | 0 次 |

> 两轮 10 集使用**同一 seed（2026）**，方块摆位逐集一致，因此可直接逐集对比。
> 「整集成功」= 三个方块全部落到各自指定放置点 5 cm 内。

### 训练收敛

| 指标 | 数值 |
|---|---|
| 总步数 | 400,000 |
| epoch | 22.24（每 epoch 17,988 步，batch 8，数据集 143,907 帧） |
| loss | 2.111（step 800）→ **0.018**（step 400K） |
| 单步耗时 | 0.027 s（约 37 步/s） |
| 总耗时 | 3 小时 03 分（单卡 RTX 5070 Ti 16 GB） |
| 学习率 | 1e-5 恒定（backbone 同为 1e-5） |

### 数据集

| 指标 | 旧版 | 本仓库（v2） |
|---|---|---|
| 集数 | 100 | 100 |
| 总帧数 | 287,667 | **143,907** |
| 单集中位帧数 | ~2,877 | **1,423** |
| 静止段占比（相邻帧关节变化 < 1e-4 rad） | **28.1%** | **0.3%** |
| 相机 | top + wrist | top + wrist |

---

## 2. 仓库结构

```
.
├── README.md
├── requirements.txt
├── oracle_collect.py          # 专家数据采集（路点状态机 + 相机录制 + HDF5）
├── to_lerobot.py              # HDF5 -> LeRobot v2.1 数据集
├── eval_act.py                # ACT 闭环评估
├── check_dataset.py           # HDF5 数据集体检（静止段/帧数统计）
├── random_pick.prefab         # OrcaStudio 关卡（场景根节点）
├── cubes/
│   ├── cubes.xml              # 三个方块 + 托盘的 MuJoCo 定义
│   ├── randomize_cubes.py     # 随机摆位（供采集脚本调用）
│   └── randomize_on_run.py    # 独立运行的随机摆位脚本
├── ur5e_table/
│   ├── ur5e_table.xml         # UR5e + 桌子 + 托盘
│   └── assets/                # 网格资源（.obj/.stl）
├── model/                     # ACT 400K 权重（Git LFS）
│   ├── model.safetensors
│   ├── config.json
│   └── train_config.json
└── results/
    ├── eval_400k_3ep.json
    ├── eval_400k_10ep.json
    ├── eval_375k_10ep.json
    └── collect_100ep.log
```

### 2.1 本机归档目录（全部产物集中在一处）

代码、数据、权重、结果统一归档在 `/home/user/act`，仓库本身只是其中的 `github_repo/` 子目录：

```
/home/user/act/
├── random_pick/            代码 + 场景（已软链回 OrcaSim/Levels/random_pick，OrcaStudio 可直接加载）
├── dataset_v2_no_alias/    100 集数据：hdf5/ 原始 + lerobot_pick_v2/ 训练用
├── act_pick_ur5e_v3/       ACT 400K 权重：025000 … 400000 + last
├── github_repo/            本仓库
├── results/                闭环评估 JSON 与完整日志
├── train_logs/  train/     训练日志、显存/内存看门狗
└── baselines/              专家回放基线（基建等价性验证）
```

对应到第 4 节的命令，本机实际取值是：

| 参数 | 本机路径 |
|---|---|
| `--dataset.root` | `/home/user/act/dataset_v2_no_alias/lerobot_pick_v2` |
| `--output_dir` | `/home/user/act/act_pick_ur5e_v3` |
| `--ckpt` | `/home/user/act/act_pick_ur5e_v3/checkpoints/last/pretrained_model` |
| 采集 `--output` | `/home/user/act/dataset_v2_no_alias/hdf5/pick.hdf5` |

> 场景目录必须是 OrcaStudio 工程 `Levels/` 下的真实路径或软链，否则编辑器加载不到关卡：
> `ln -sfn /home/user/act/random_pick /home/user/orca_studio/OrcaSim/OrcaSim/Levels/random_pick`

---

## 3. 环境依赖

两个 conda 环境（采集/评估需要连 OrcaStudio，训练只需要 lerobot）：

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `orcalab_gpu` | 采集、闭环评估 | torch 2.7.1+cu128, lerobot 0.3.4, mujoco 3.12.0, h5py 3.16.0, grpcio 1.66.1, av 17.0.1, orca_gym |
| `orcalab_lerobot` | HDF5 → LeRobot 转换 | lerobot 0.3.4, pyav, torch |

```bash
pip install -r requirements.txt
```

`orca_gym` 随 OrcaStudio 分发（用于连接编辑器里跑着的仿真进程），不在 PyPI 上，需要单独安装。

硬件：单卡 GPU（训练用 16 GB 显存足够，batch 8 时占用约 2.7 GB）、30 GB 内存。

---

## 4. 复现步骤

### 4.1 搭建场景

1. 打开 OrcaStudio，导入 `random_pick.prefab`（关卡根节点）。
2. 导入 `cubes/cubes.xml` 生成 prefab，拖进关卡中 **`Orca Gym Env` 子树下**。
   > 注意：只有挂在 `Orca Gym Env` 下的实体才会被导出到 MuJoCo 模型；放在场景根节点下会导致导出的 out.xml 缺关节/连杆，Python 侧找不到实体。
3. `ur5e_table` 同理挂在 `Orca Gym Env` 下。
4. 配置两路相机，映射名必须与脚本一致：

   | 相机 | OrcaStudio 侧名称 | 数据集字段 |
   |---|---|---|
   | 顶视 | `camera_float` | `observation.images.camera_top` |
   | 腕部 | `gripper_camera` | `observation.images.camera_wrist` |

5. 进入**运行模式**（50051 端口监听、相机开始推流）。

### 4.2 采集专家数据

```bash
python oracle_collect.py \
  --episodes 100 \
  --seed 20261027 \
  --output /path/to/hdf5/pick.hdf5
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--episodes` | 目标成功集数（失败会重采，不计入） |
| `--seed` | 闭环可复现的种子；**必须显式给**，默认 `None` 走 OS 熵源就不可复现 |
| `--miss-prob` | 注入抓偏的概率（默认 0.15），用于让策略学会从空抓中恢复 |
| `--fast` | 关闭实时节流（仅调试用，会丢帧） |

产出：`pick_ep0000.hdf5 …`，每集含 `obs/qpos`(T,7)、`action`(T,7)、`obs/camera_top`、`obs/camera_wrist`、`obs/cube_*`，属性里记录 seed。

**实测**：100 集 / 143,907 帧 / 51 分 45 秒（约 31 s/集，50 Hz 实时节流）。

### 4.3 获取 LeRobot v2.1 数据集

完整数据集（100 集 / 143,907 帧 / 1.5 GB）体积超出 GitHub 免费 LFS 配额，因此**不放在 git 仓库里**，而是作为 Release 附件提供：

```bash
# 下载并解压，得到 lerobot_pick_v2/ 目录
curl -L -o lerobot_pick_v2.tar \
  https://github.com/xcdgdj/act-ur5e-3boxes/releases/download/v1.0.0/lerobot_pick_v2.tar
tar xf lerobot_pick_v2.tar
```

解压后目录结构为 `lerobot_pick_v2/{data,videos,meta}`，直接作为 `--dataset.root` 使用即可（见 4.4）。

如果要**从零重采**，则做完 4.2 后用下面的命令把 HDF5 转成 LeRobot 格式：

```bash
python to_lerobot.py \
  --input  /path/to/hdf5/pick \
  --out    /path/to/lerobot_pick_v2 \
  --repo-id local/pick_v2_no_alias \
  --only-success
```

产出 100 个 parquet + 200 个 mp4（top/wrist 各 100），fps 50，约 1.5 GB。

### 4.4 训练 ACT

```bash
python -m lerobot.scripts.train \
  --dataset.repo_id=local/pick_v2_no_alias \
  --dataset.root=/path/to/lerobot_pick_v2 \
  --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 --policy.n_obs_steps=1 \
  --policy.use_vae=true \
  --policy.dim_model=512 --policy.n_encoder_layers=4 --policy.n_decoder_layers=1 \
  --policy.optimizer_lr=1e-5 --policy.optimizer_lr_backbone=1e-5 \
  --batch_size=8 --steps=400000 --save_freq=25000 --log_freq=100 \
  --num_workers=4 --seed=1000 \
  --output_dir=/path/to/act_pick_ur5e_v3 --job_name=act_v3 --wandb.enable=false
```

> `--policy.push_to_hub=false` 必须加，否则会因缺 `repo_id` 校验报错。

### 4.5 闭环评估

先让 OrcaStudio 进入运行模式，然后：

```bash
python eval_act.py \
  --ckpt ./model \
  --episodes 10 \
  --temporal-ensemble-coeff 0.01 \
  --idr-every 5 \
  --seed 2026 \
  --out results/eval_10ep.json
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--temporal-ensemble-coeff` | 开启 ACT temporal ensembling（推荐 0.01）；不填则按 chunk 顺序开环执行 |
| `--idr-every` | 每 N 拍请求一次关键帧；ensemble 模式下每拍都推理，不能每拍都请求 IDR |
| `--max-steps` | 单集上限（默认 3000 拍 = 60 s） |
| `--force-open-dist` | 事件兜底：TCP 距放置点小于该值且持续合爪 N 拍则强制张爪；**默认 0 = 关闭**，本仓库结果均为关闭状态 |
| `--sanity-h5` | 开跑前用专家首帧对比 chunk L1，检查输入约定是否错配 |

---

## 5. 关键设计

### 5.1 为什么数据集要「无静止段」

这是本项目最重要的一条经验。

旧版采集脚本里，直线插值阶段走完后还会继续等 IK 收敛判定 `ready() = reached() and settled()`。但位置伺服存在 5~7 mm 的重力稳态误差，永远大于位置容差，于是每个阶段都要**空等到 250 拍上限**才切下一阶段。结果：**28.1% 的帧是纯静止段**，单集 2,877 帧里有 800 多帧手臂一动不动。

这会造成**观测别名（observational aliasing）**：同一组 `(state, image)` 在数据集里既对应「保持不动」，又对应「张开夹爪 / 移动」，标签自相矛盾。ACT 的 `n_obs_steps=1` 意味着策略是 `(state, image)` 的确定性函数 —— 恒定输入 → 恒定输出 → 手臂不动 → 观测不变，数学上构成一个**稳定死锁吸引子**。实测就是这个现象：闭环中策略 100% 忠实地复现专家的「保持」段，`act ≈ state`，手臂永久冻结，换模型、加平滑、改 chunk 消耗策略全都无效。

**修法**（`oracle_collect.py`）：

1. 直线阶段走完后只做 **固定 5 拍 settle**（`SETTLE_MAX_STEPS`），不再等 IK 收敛；
2. 夹爪保持拍数按物理响应裁剪：合爪从张开位到夹住方块约 1 拍（20 ms），空手合到底约 7 拍，因此 `CLOSE_HOLD_STEPS = 10`、`OPEN_HOLD_STEPS = 5`（原来是 60 / 15）。

效果：静止段 28.1% → **0.3%**，单集帧数 2,877 → 1,423，专家抓放成功率不降。

### 5.2 动作与观测空间

| 项 | 定义 |
|---|---|
| 观测 `observation.state` | 6 个关节角（rad）+ 夹爪净开口（m），共 7 维 |
| 动作 `action` | 同一 7 维空间的**绝对目标**：关节目标角 + 夹爪目标净开口 |
| 夹爪动作语义 | 净开口（m）；全开 0.0648，合爪指令 0.045 |
| 控制频率 | 50 Hz（`CTRL_DT = 0.02`） |
| 归一化 | `VISUAL/STATE/ACTION` 全部 `MEAN_STD`，图像用 ImageNet stats，统计量存在 checkpoint 里 |

专家动作是速率限幅的绝对关节目标：`action[k] = qpos[k] + clip(target - qpos[k], ±0.2)`。

### 5.3 训练超参

| 超参 | 值 |
|---|---|
| 结构 | ACT，`dim_model=512`，encoder 4 层，decoder 1 层 |
| action chunk | `chunk_size=50`，`n_action_steps=50`，`n_obs_steps=1` |
| VAE | 开启，`latent_dim=32` |
| 优化器 | AdamW，lr `1e-5`（backbone 同为 `1e-5`），weight decay `1e-4`，lr 恒定不衰减 |
| batch | 8 |
| 步数 | 400,000（22.24 epoch） |
| seed | 1000（模型初始化 + 数据 shuffle） |
| dataloader | `num_workers=4`，shuffle buffer 2,000，视频后端 pyav |
| 存档 | 每 25,000 步 |

### 5.4 评估链路细节

- 每拍都重新推理整条 chunk，对重叠预测做指数加权（`temporal_ensemble_coeff=0.01`，权重 `wᵢ = exp(-0.01·i)`）
- 单次推理约 90 ms，因此单集墙钟 130~190 s（纯物理只需 30 s）；部署若在意节拍，可退回 chunk 队列模式（每 50 拍推理一次），速度约快 10 倍
- 判据：方块 xy 距指定放置点 < 5 cm 算放到位；中心抬升 > 5 cm 算曾被抓起

---

## 6. 复现时踩过的坑（建议先读）

| 现象 | 原因 | 处理 |
|---|---|---|
| 画面不动 / 定格在初始姿态 | 外部仿真循环步进 `mj_step` 后未同步 `env.data` | 显式调用 `env.gym.update_data()` |
| 画面不更新 | 仿真循环没有按真实物理时间步进 | 加入与 `CTRL_DT` 匹配的实时节流 |
| 导出的 out.xml 缺关节/连杆 | 实体没挂在 `Orca Gym Env` 子树下 | 场景里移到该子树 |
| 部件完全不可见 | 网格导入管线只为 `.obj` 生成渲染产物，`.stl` 不生成 | 视觉体用 `.obj` |
| 碰撞体可见 / 视觉体不碰撞 | 导入器约定：`contype=0 & conaffinity=0` 的 geom 导出为 `*_vis`，带碰撞的导出 `*_col` 且渲染隐藏 | 视觉体和碰撞体拆成两个 geom |
| 改完 XML 参数不生效 | 编辑器内存里仍是旧参数 | 重启 OrcaStudio 重新加载场景 |
| 方块被摆到桌面外 | prefab 实例带平移 patch | 用场景内桌子的实际世界位置反推偏移再叠加 |
| 工程目录里塞入大量图片后启动卡死 | AssetProcessor 扫描全部新增文件 | 数据集放工程目录外 |

---

## 7. 已知问题

**残余冻结（约 10% 集数）**：个别集里策略在放置区附近出现 `act ≈ state`（差值 < 0.005 rad）、夹爪保持张开、跑满 3000 拍上限，导致部分方块从头到尾没被碰过。已确认：

- 与 ckpt 新旧无关（400K 与 375K 都会发生，只是触发的集不同）
- 不是脚本顺序锁死（同一 seed 同摆位，375K 能过 400K 失败的那一集）
- 推测是**训练集里不存在的状态组合**（例如空抓手停在放置区、或托盘中方块状态与台面残留方块同时出现的组合），属于任务结构单一导致的分布外冻结

可能的改进方向：采集侧随机化抓取顺序与放置槽位对应关系、注入更多「抓取失败后重试」的负样本。

---

## 8. 引用与致谢

- 仿真：OrcaStudio（MuJoCo 后端）
- 策略与训练框架：[lerobot](https://github.com/huggingface/lerobot) 的 ACT 实现
- ACT 论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao et al., 2023)
