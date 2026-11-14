# Causal-WAM-Veomni

基于 VeOmni 的 Causal Cosmos3 视频 / 动作训练与推理。当前三专家路径支持 **IDM、FDM、Video 三种独立模式**，暂不开放 `joint` 或 Policy。默认以 Cosmos3-Edge 为起点，使用 55D 动作 / 状态接口和 Ascend NPU。

Text Reasoner 加载预训练权重并冻结；video expert 保留 28 层、hidden 2048、FFN 9216；action expert 随机初始化为 14 层、hidden 1024、FFN 4608，依次对应 video 第 2、4、…、28 层。纯 Video recipe 不分配 action expert。视频保持 `main` 的 Scheme B block-causal、teacher-forcing 和历史窗口规则；预训练权重只是初始化，不替代 causal 训练。

旧双专家 FD / ID / Policy 训练及 Policy rollout 仍作为兼容与基线保留，使用原来的非 `triple` 配置。

以下命令均在 `Causal-WAM-Veomni/` 根目录执行，`/path/to/...` 需替换为实际路径。

## 1. 环境配置

已验证环境（2026-11-14）：

| 项目 | 版本 / 路径 |
| --- | --- |
| 硬件 / 架构 | Ascend 910B3 / aarch64 |
| CANN | 9.1.0 |
| Python | 3.12.10 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0.post6 |
| Transformers / Diffusers | 5.16.1 / 0.39.0 |
| PyAV / Datasets | 15.1.0 / 4.8.5 |
| 虚拟环境 | `/mnt/sfs_turbo/zheng/envs/veomni-cosmos3` |

复用当前机器环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/zheng/envs/veomni-cosmos3/bin/activate
export LD_PRELOAD=/mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-framework-py312/lib/libstdc++.so.6
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
```

切换环境时不要沿用其他 Python 环境的 `torch/lib` 路径，尤其是 Python 3.13 的 `LD_LIBRARY_PATH`；本项目使用 Python 3.12。先通过 `npu-smi info` 确认空闲卡，再设置 `ASCEND_RT_VISIBLE_DEVICES`。

新建环境使用仓库依赖（需预装兼容的驱动、CANN 和 uv）：

```bash
uv sync --locked --extra npu_aarch64  # x86_64 改用 --extra npu
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

`pyproject.toml` 当前锁定 Diffusers 0.37.0、Datasets 2.21.0 等，与上表现有环境不同；新环境需重新验证。当前视频读写使用 PyAV。安装细节：[ARM](docs/get_started/installation/install_ascend_arm.md) / [x86](docs/get_started/installation/install_ascend_x86.md)。

## 2. 代码结构

```text
Causal-WAM-Veomni/
├── tasks/
│   ├── train_causal_cosmos3.py          # 训练入口、资源路径解析、梯度累积
│   └── infer/infer_causal_cosmos3.py    # 单卡推理 CLI
├── configs/
│   ├── dit/causal_cosmos3_edge_triple_action.yaml # 独立 IDM
│   ├── dit/causal_cosmos3_edge_triple_fdm.yaml    # 独立 FDM
│   ├── dit/causal_cosmos3_edge_triple_video.yaml  # 纯 Video
│   ├── data/action_datasets.yaml       # 动作数据源、采样、验证集划分
│   ├── data/video_datasets.yaml        # 视频数据源、采样、验证集划分
│   ├── infer/causal_cosmos3_triple.yaml # 三专家推理资源与采样参数
│   └── model_configs/causal_cosmos3/    # Edge / Nano / Super 结构与 condition 配置
├── veomni/
│   ├── data/causal_cosmos3/
│   │   ├── readers/                    # LeRobot 读取、视频解码、初始观测
│   │   ├── transforms/                 # 动作归一化、相机布局、文本 tokenization
│   │   └── datasets/                   # 候选窗口、源采样、packing、验证集划分
│   ├── models/diffusers/causal_cosmos3/
│   │   ├── cosmos3_condition/          # VAE 编码、时间步 / 噪声、训练输入准备
│   │   ├── cosmos3_transformer/        # 主网络、权重加载、参数冻结、loss
│   │   ├── cosmos3_core/               # Attention、mRoPE、packing、VAE、采样器
│   │   └── inference.py                # 模型加载、分块生成、rollout、结果导出
│   └── trainer/causal_cosmos3.py        # 接入 DiTTrainer，训练与定期验证
├── scripts/causal_cosmos3/             # 网络校验、旧 Cosmos checkpoint 转换
└── train.sh                            # 通用 torchrun 启动脚本
```

## 3. 训练配置修改

### 三种独立模式

所有模式均保留观测视频历史；“纯文本条件”指不读取 action/state。

| 模式 | 训练与梯度 | 推理 | 训练配置 |
| --- | --- | --- | --- |
| IDM | text/history 视频 FM + action FM；state 只进入 action，action loss 不更新 video | 先部分视频去噪，再固定视觉 KV 生成 action | [triple_action.yaml](configs/dit/causal_cosmos3_edge_triple_action.yaml) |
| FDM | 视频 FM；text、action/state、两者同时，各 1/3；video loss 可训练 clean action/state encoder，不训练动作去噪头 | clean action 条件固定，只对视频去噪 | [triple_fdm.yaml](configs/dit/causal_cosmos3_edge_triple_fdm.yaml) |
| Video | 仅 text/history 视频 FM，跳过 action/state | 生成 clean future video | [triple_video.yaml](configs/dit/causal_cosmos3_edge_triple_video.yaml) |

三专家的 `data.sample.mode` 使用 `inverse_dynamics` 或 `forward_dynamics`；纯 Video 使用 `data.stage: video`。`joint` 和混合任务 pack 会显式报错。FDM 的三种条件由 `data.sample.fd_condition_weights: [1, 1, 1]` 控制。

IDM 的视频监督使用正常抽样时间的 GT 加噪输入；action 的视觉缓存单独编码，默认固定在 `sigma_stop=0.9`。前 20% 训练使用 GT noisy-future cache，接下来 20% 将模型短 rollout cache 的概率提高到 50%，之后保持。缓存混合不会改变视频 / 动作 FM 输入与标签，也不反传采样轨迹。视频监督、缓存构建和推理 rollout 都只读 text/history，state 只给 action expert。

FDM 独立训练不依赖 IDM 初始化条件编码器；但 FDM checkpoint 的动作去噪头没有受监督，不能将它当成训练完成的 IDM。三专家和旧双专家 DCP 不可直接互相恢复 optimizer；T/V 权重可通过显式 warm-start 加载。细节见[训练文档](docs/usage/causal_cosmos3_training.md)。

### 模型与数据资源

`model.model_path` 指向完整预训练目录：

```text
/path/to/Cosmos3-Edge/
├── transformer/       # 预训练网络权重
├── text_tokenizer/    # 文本 tokenizer
└── vae/               # 训练 VAE
```

先修改以下两份 YAML：

| 配置文件 | 字段 | 修改内容 |
| --- | --- | --- |
| 上表选中的训练 YAML | `model.model_path` | 完整预训练目录；当前机器可用 `/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge` |
| 同上 | `model.config_path` / `model.condition_model_path` | 网络结构 / condition JSON；换 Nano、Super 时同步更换结构和权重 |
| 同上 | `data.train_path` | 数据源 YAML 路径 |
| 同上 | `train.checkpoint.output_dir` | 本次训练输出目录 |
| `configs/data/action_datasets.yaml` | `sources[].roots` | LeRobot v3 数据集根目录，支持列表和 glob |
| 同上 | `sources[].statistics` | 实测动作归一化统计；支持 `'{root}/meta/causal_action_stats.json'` |
| 同上 | `sources[].cameras` | head / left / right 对应的真实视频列名 |
| 同上 | `sources[].columns` | state、target、mask 列及 `target_offset`；按机器人数据语义配置 |

数据 YAML 只配置 `sampling / validation / sources`；`stage / geometry / candidates / sample` 在 DiT YAML 的 `data` 块中配置。入口函数组装完整数据配置后显式传给 Trainer；输出目录的 `dataset_config.yaml` 保存组装结果，`veomni_cli.yaml` 保存框架参数。

数据需包含 LeRobot v3 的 `meta/`、`data/`、`videos/` 和非空任务文本；IDM / FDM 另外要求统一的 55D 动作 / 状态、有效通道 mask 和统计文件。Video 不读取动作列或动作统计。子任务文本及边界可由 `meta/episodes.jsonl` 提供。示例数据 YAML 中存在占位路径，启动前需替换或删除无效源；Video 默认使用 `configs/data/video_datasets.yaml`。

### 常改参数

| 所在配置 | 字段 | 当前默认 / 作用 |
| --- | --- | --- |
| 训练 YAML | `model.optimizer.lr` | `5e-5` |
| 训练 YAML | `model.model_config.train_generation_only` | `true`，选择 GEN 相关训练参数 |
| 训练 YAML | `model.accelerator.dp_shard_size` | `8`，每个 FSDP 分片组的卡数；4 卡训练改为 `4` |
| 训练 YAML | `train.grad_accum_iter` | `2`，每次更新累积的 pack 数 / DP rank |
| 训练 YAML | `train.max_steps` / `num_train_epochs` | `10000` / `1`；max_steps 为每个 epoch 的优化器更新上限 |
| 训练 YAML | `data.max_seq_len` / `dyn_bsz_buffer_size` | `54000` / `8`，packing 预算 / 候选缓冲数 |
| 训练 YAML | `data.dataloader.num_workers` | `8` |
| 训练 YAML | `train.eval_steps` / `train.checkpoint.save_steps` | 均为 `1000`；设 `0` 关闭对应的 step 触发 |
| 训练 YAML | `train.wandb.enable` | `true`；离线设 `WANDB_MODE=offline`，或将该字段设为 `false` |
| 数据 YAML | `sampling` | `sample_uniform`；`source_weighted` 时使用 `sources[].weight` |
| 三专家 action 训练 YAML | `data.sample.mode` / `data.sample.mode_weights` | IDM 为 `inverse_dynamics` / `[1]`，FDM 为 `forward_dynamics` / `[1]` |
| FDM 训练 YAML | `data.sample.fd_condition_weights` | `[1, 1, 1]`，依次为纯 text、纯 action/state、两者同时 |
| 数据 YAML | `validation` | 按 episode 留出 5%，每源最多 100 个，seed 为 73；移除 `split_ratio` 关闭划分与定期验证 |

窗口与阶段设置（均在 DiT 训练 YAML 的 `data` 块）：

| 字段 | IDM / FDM 短窗默认 | 长窗示例 |
| --- | --- | --- |
| `data.candidates.mode` | `frame_window` | `segmented` |
| `data.candidates.max_actions` | `160` | `1792` |
| `data.candidates.overlap` | `0` | `321` |
| `data.sample.full_history` | `true` | `false` |
| `data.sample.history_range` | `[1, 32]` | `[1, 32]` |

- `max_actions: -1`：按完整子任务取样；无子任务时使用完整 episode。`use_subtask: false` 时始终按完整 episode。
- 所有模式的短窗口统一在尾部重复末值补齐到整 block；不扩展真实窗口范围，补齐的 action 步和纯补齐视频 latent 不参与监督。
- `micro_batch_size` 保持 `1`；指定 `grad_accum_iter` 时省略 `global_batch_size`。8 卡、累积 2 次对应 16 个 pack / 更新，pack 内样本数可变。
- 当前训练使用至少 2 卡、FSDP2、BF16、meta 初始化，SP / CP 保持 `1`。`cast_forward_inputs` 保持 `false`。
- 单个样本超过 `max_seq_len` 时仍会独立成包；长窗口需同时评估显存。

旧双专家的 [action recipe](configs/dit/causal_cosmos3_edge_action.yaml) 继续使用 FD / ID / Policy 的 `joint` 混采；该默认值不适用于三专家。

## 4. 启动训练与恢复

当前环境使用 Python 模块启动，避免依赖独立的 `torchrun` 可执行文件。

选择一个模式后启动。以下以 IDM 为例；FDM / Video 分别替换为上表对应 YAML，并使用匹配的数据源配置：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_triple_action.yaml \
  --model.model_path /path/to/Cosmos3-Edge \
  --data.train_path /path/to/action_sources.yaml \
  --train.checkpoint.output_dir outputs/triple_idm
```

6 卡、30 步短跑（卡号按实际空闲情况修改；CLI 参数覆盖 YAML）：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=6 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_triple_action.yaml \
  --model.model_path /path/to/Cosmos3-Edge \
  --data.train_path /path/to/action_sources.yaml \
  --model.accelerator.dp_shard_size 6 \
  --data.max_seq_len 8192 --data.dyn_bsz_buffer_size 2 \
  --data.dataloader.num_workers 2 \
  --train.max_steps 30 --train.grad_accum_iter 1 --train.eval_steps 0 \
  --train.checkpoint.save_steps 30 \
  --train.checkpoint.output_dir outputs/triple_smoke \
  --train.wandb.enable false
```

窗口长度在所选 YAML 的 `data.candidates.max_actions` 中设置，不能用不存在的候选窗口 CLI 字段覆盖。三专家直接选择对应 YAML；现有 [train_action.sh](scripts/causal_cosmos3/training_scripts/train_action.sh) / [train_video.sh](scripts/causal_cosmos3/training_scripts/train_video.sh) 脚本仍选择旧双专家 recipe。

已安装 `torchrun` 的环境也可使用通用脚本，日志写入 `log.txt`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash train.sh tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_triple_action.yaml \
  --model.model_path /path/to/Cosmos3-Edge --data.train_path /path/to/action_sources.yaml \
  --model.accelerator.dp_shard_size 4
```

多机使用 `train.sh`，各节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`、`NPROC_PER_NODE`；总 DP 卡数需能被 `dp_shard_size` 整除。

恢复训练：在原启动命令末尾加上以下参数，并保持模型、GEN 参数选择和优化器配置一致。

```bash
--train.checkpoint.load_path outputs/triple_idm/checkpoints/global_step_1000
```

DCP 恢复模型、优化器、调度器和训练 RNG；Cosmos3 数据流采用 restart，不保证逐样本完全重放。继续训练时需保证新的更新上限大于已恢复的 step。定期验证指标写入 `<output_dir>/validation.jsonl`，按所选任务记录；旧双专家另有 `eval/policy`。

## 5. 推理使用

### 准备完整训练权重

VeOmni DCP 先导出 safetensors，并保留该次训练的实际 `config.json`：

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir outputs/triple_idm/checkpoints/global_step_1000 \
  --save-dir outputs/triple_idm_1000_hf
cp outputs/triple_idm/model_assets/config.json outputs/triple_idm_1000_hf/config.json
```

已有完整 native / HF 训练权重可直接使用。旧 Cosmos-framework DCP 使用 [专用转换脚本](scripts/causal_cosmos3/convert_inference_checkpoint.py)，命令见[推理文档](docs/usage/causal_cosmos3_inference.md)。

### 修改推理配置

编辑 [三专家推理配置](configs/infer/causal_cosmos3_triple.yaml)；现有统计与机器人列名为 1030 POC 示例，需替换为实际训练资源。`tokenizer` 可设为训练根目录的 `text_tokenizer/`，`vae` 使用匹配训练的 `vae/`。

| 字段 | 要求 |
| --- | --- |
| `tokenizer` / `vae` | 匹配训练；推理 VAE 为包含 `config.json` 和权重的 Diffusers 目录 |
| `statistics` | 对应机器人、数据源的实测统计 |
| `source.cameras` / `source.columns` | 与输入数据列一致；核对 state / target / mask / offset |
| `geometry` / `source.resolution` / `source.viewpoint` | 与训练一致；默认每块 32 个 action，video stride 为 4 |
| `device` / `dtype` | 单卡 `npu:0` / `bfloat16` |
| `attention_backend` / `ops_implementation` | 与训练一致；当前为 `npu_tnd` + `torch_npu` RoPE / RMSNorm |
| `sampling` | Video / FDM 默认 20 步 UniPC；IDM 默认 4 步部分视频 Euler 到 σ=0.9、额外 1 次视觉 cache 编码、再独立 20 步 action UniPC |

### 运行

| 模式 | 行为 |
| --- | --- |
| `teacher_forcing --task inverse_dynamics` | 测得的视频历史 → 部分未来视觉 cache → action；state 仅进入 action |
| `teacher_forcing --task forward_dynamics` | 固定 clean action，选择 text / action / both 条件生成视频 |
| `teacher_forcing --task video` | text 与视频历史生成视频，不读取 action/state 列和统计 |

三专家拒绝 `rollout` / `policy`。`--task all` 只是分别执行三种推理，不是 joint 训练，也不意味着某个独立 checkpoint 已训练所有任务。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # 物理卡映射为逻辑 npu:0

# IDM：使用训练完成的 IDM checkpoint
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple.yaml \
  --checkpoint outputs/triple_idm_1000_hf \
  --mode teacher_forcing --task inverse_dynamics \
  --dataset /path/to/lerobot_v3 --episode 0 --start 0 --max-actions 64 \
  --partial-video-steps 4 --video-stop-sigma 0.9 --action-steps 20 \
  --output outputs/idm_episode0

# FDM：用 FDM checkpoint；text / action / both 三选一
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple.yaml \
  --checkpoint /path/to/triple_fdm_hf \
  --mode teacher_forcing --task forward_dynamics --fdm-conditioning both \
  --dataset /path/to/lerobot_v3 --episode 0 --max-actions 64 \
  --output outputs/fdm_episode0

# Video：--max-actions 在该任务中表示源视频帧间隔数
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple.yaml \
  --checkpoint /path/to/triple_video_hf \
  --mode teacher_forcing --task video \
  --dataset /path/to/lerobot_v3 --episode 0 --max-actions 32 \
  --output outputs/video_episode0
```

- 常用覆盖参数：`--video-steps 20 --action-steps 20 --history 32 --seed 42 --guidance 1`。`--steps` 仅控制旧双专家 sampler；三专家不接受联合蒸馏 fixed-step schedule。
- IDM 的 σ=0.9 future 仍有噪声，不进行未来像素解码，输出 `observation.mp4` 和动作结果；FDM 输出的视频为预测、动作标为条件；Video 不生成动作曲线。各任务均记录 `run.json` / `blocks.jsonl`，包含阶段耗时、cache 调用与显存统计。已完成的输出目录不可重复使用。
- 旧双专家使用 `configs/infer/causal_cosmos3.yaml`，保留 FD / ID / Policy 的 `teacher_forcing` 和仅限 Policy 的 `rollout`；初始 NPZ、反馈语义和命令见[推理文档](docs/usage/causal_cosmos3_inference.md)。

## 6. 真实 NPU 短跑验证

2026-11-14 使用物理 NPU 2–7，共 6 张 Ascend 910B3，以真实 Edge 权重和 `RoboTwin__adjust_bottle/clean` 数据分别训练三个模式。启用 BF16、NPU TND attention、FSDP2 和梯度 checkpoint；每个模式完成 30 次 optimizer 更新，并从完整 DCP 恢复后额外训练 1 步。

| 模式 | 首步 → 第 30 步 total loss | 第 30 步 video / action loss | rank0 Torch 峰值分配量（GiB） | DCP 恢复到第 31 步 |
| --- | --- | --- | --- | --- |
| Video | 8.61 → 1.93 | 1.93 / 不监督 | 9.16 | 通过 |
| FDM | 11.63 → 2.46 | 2.46 / 不监督 | 9.59 | 通过 |
| IDM | 30.94 → 5.39 | 2.34 / 3.05 | 9.47 | 通过 |

本次未出现非有限 loss / grad、OOM 或 FSDP/HCCL 错误。FDM checkpoint 的 action/state 编码投影存在非零且有限的 Adam 动量，未受监督的动作输出头无 optimizer state。另修复了学习率日志将 `5e-5` 舍入成 `0.00` 的显示问题。

测试使用 256 分辨率桶、384×320 拼接画布、5 个 latent / 两个预测 block、packing 预算 8192、候选缓冲 2、每 rank 每步 1 个 pack。IDM / FDM 窗口为 64 个动作步；Video 为 32 个源帧间隔，经 stride 2 得到相同视频帧数。上述显存为 rank0 本地 Torch allocator 峰值，不是全卡最大 HBM。测试只验证短窗口运行与恢复，不代表长上下文、生成质量或 RoboTwin 闭环成功率。

本机日志、完整 YAML、checkpoint 和审计结果保留在 `outputs/triple_npu_smoke_20261114_051638/`；该运行目录不提交到 Git。

## 7. 详细文档

[训练与验证](docs/usage/causal_cosmos3_training.md) · [网络实现](docs/usage/causal_cosmos3_network.md) · [独立推理](docs/usage/causal_cosmos3_inference.md) · [Checkpoint 导出](docs/usage/checkpoint_conversion.md)

基于 VeOmni；原始项目元信息见 `pyproject.toml`，许可证见 [LICENSE](LICENSE)。Cosmos3 模型代码的声明见 [NOTICE](veomni/models/diffusers/causal_cosmos3/NOTICE.txt)。
