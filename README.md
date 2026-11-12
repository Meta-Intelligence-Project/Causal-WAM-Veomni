# Causal-WAM-Veomni

基于 VeOmni 的 Causal Cosmos3 视频 / 动作联合训练与推理，支持 FD（动作预测视频）、ID（视频预测动作）和 Policy（联合预测）。默认配置为 Cosmos3-Edge、55D 动作 / 状态接口、Ascend NPU。

以下命令均在 `Causal-WAM-Veomni/` 根目录执行，`/path/to/...` 需替换为实际路径。

## 1. 环境配置

当前使用环境（2026-10-29）：

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
│   ├── dit/causal_cosmos3_edge_action.yaml     # Action 模型、窗口、优化器、并行和训练参数
│   ├── dit/causal_cosmos3_edge_video.yaml      # Video 模型、分块、优化器、并行和训练参数
│   ├── data/action_datasets.yaml        # Action 数据源、源采样、验证集划分
│   ├── data/video_datasets.yaml         # Video 数据源、源采样、验证集划分
│   ├── infer/causal_cosmos3.yaml        # 推理资源、输入语义、采样参数
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
| `configs/dit/causal_cosmos3_edge_action.yaml` | `model.model_path` | 完整预训练目录；当前机器可用 `/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge` |
| 同上 | `model.config_path` / `model.condition_model_path` | 网络结构 / condition JSON；换 Nano、Super 时同步更换结构和权重 |
| 同上 | `data.train_path` | 数据源 YAML 路径 |
| 同上 | `data.stage` / `geometry` / `candidates` / `sample` | 训练阶段、时空几何、候选和窗口采样；四项都位于训练 YAML 的 `data` 下 |
| 同上 | `train.checkpoint.output_dir` | 本次训练输出目录 |
| `configs/data/action_datasets.yaml` | `sources[].roots` | LeRobot v3 数据集根目录，支持列表和 glob |
| 同上 | `sources[].statistics` | 实测动作归一化统计；支持 `'{root}/meta/causal_action_stats.json'` |
| 同上 | `sources[].cameras` | head / left / right 对应的真实视频列名 |
| 同上 | `sources[].columns` | state、target、mask 列及 `target_offset`；按机器人数据语义配置 |

数据需包含 LeRobot v3 的 `meta/`、`data/`、`videos/`；动作 / 状态统一为 55D，并提供有效通道 mask 和非空任务文本。子任务文本及边界可由 `meta/episodes.jsonl` 提供。

数据源 YAML 的顶层只接受 `sampling`、`validation`、`sources`。`stage`、`geometry`、
`candidates` 和 `sample` 统一配置在训练 YAML 的 `data` 下；源文件残留这些字段会明确报错，
避免覆盖本次实验。入口组装最终数据配置后交给 Trainer，输出目录同时保存
`dataset_config.yaml` 以记录生效参数；它只保存配置，不保存数据张量或流缓存。

### 常改参数

| 所在配置 | 字段 | 当前默认 / 作用 |
| --- | --- | --- |
| 训练 YAML | `model.optimizer.lr` | `5e-5` |
| 训练 YAML | `model.model_config.train_generation_only` | `true`，选择 GEN 相关训练参数 |
| 训练 YAML | `model.model_config.enable_tbptt` | `false`；Action 开启连续 episode 分段训练，Video 开启原有 clip 内分段梯度策略 |
| 训练 YAML | `model.model_config.tbptt_segment_steps` | `256`；Action 每段推进的控制步数，Video 的梯度截断时间段跨度；正整数且为16的倍数，Action 按决策间隔D对齐，Video 按原预测块对齐 |
| 训练 YAML | `data.sample.decision_stride_steps` | Action TBPTT 决策间隔D，缺省跟随H；Policy TBPTT配方使用16，每次仍预测H32 |
| 训练 YAML | `model.condition_model_cfg.vae_reset_prefix_cache_mb` | 独立窗口路径的冻结 VAE 首帧补齐快照预算，`4096` MiB / rank；episode 路径另有有限流状态 |
| 训练 YAML | `model.accelerator.dp_shard_size` | `8`，每个 FSDP 分片组的卡数；4 卡训练改为 `4` |
| 训练 YAML | `model.accelerator.gradient_checkpointing` | `recompute_layers: -1`、`selective_recompute_layers: 28`，Edge各层保存attention结果、重算其余算子；后者设`0`恢复全量重算 |
| 训练 YAML | `train.micro_batch_size` | 默认 `1`；Action episode TBPTT 可设 `2` 等正整数，一次合并不同流的完整训练段；其他路径保持 `1` |
| 启动环境 | `CLOSE_MATMUL_K_SHIFT` | NPU 上 Action TBPTT 的 `micro_batch_size>1` 必须在启动 Python 前设为 `1`；Action 启动脚本默认导出此值 |
| 训练 YAML | `train.grad_accum_iter` | `2`，每 rank 每次更新串行累积的 microbatch 数；不会扩大一次矩阵计算的 batch |
| 训练 YAML | `train.gc_steps` | `0`，新训练进程保留自动 GC；DataLoader workers 启动时独立开启 GC，避免解码循环对象积累 |
| 训练 YAML | `train.max_steps` / `num_train_epochs` | `10000` / `1`；max_steps 为每个 epoch 的优化器更新上限 |
| 训练 YAML | `data.max_seq_len` / `dyn_bsz_buffer_size` | `54000` / `8`，独立窗口的 packing 预算 / 缓冲数；episode TBPTT 不据此拆段或丢监督 |
| 训练 YAML | `data.dataloader.num_workers` / `worker_num_threads` / `prefetch_factor` | `4` / `1` / `2`，每 rank 的 worker 数 / 每 worker 的 Torch 线程数 / 预取数 |
| 数据 YAML | `sources[].read.decode_threads` | `2`，每个视频解码器的线程数，避免多 rank 自动扩展线程池 |
| 训练 YAML | `train.eval_steps` / `train.checkpoint.save_steps` | 均为 `1000`；设 `0` 关闭对应的 step 触发 |
| 训练 YAML | `train.wandb.enable` | `true`；离线设 `WANDB_MODE=offline`，或将该字段设为 `false` |
| 数据 YAML | `sampling` | `sample_uniform`；`source_weighted` 时使用 `sources[].weight` |
| 训练 YAML | `data.sample.mode` / `mode_weights` | `joint` / `[1, 1, 1]`，依次为 FD、ID、Policy；也可固定单任务 |
| 数据 YAML | `validation` | 按 episode 留出 5%，每源最多 100 个，seed 为 73；移除 `split_ratio` 关闭划分与定期验证 |

每个 Action 决策保留以下窗口参数（均在训练 YAML 的 `data` 下）：

| 字段 | 默认 / 作用 |
| --- | --- |
| `data.sample.history_steps` | `1024`，当前锚点之前的原始 action 时间间隔数 |
| `data.sample.action_horizon` | `32`，一次决策的未来动作数 |
| `data.candidates.start_stride` | `1`，枚举决策锚点的间隔，包含 episode 尾部 |
| `data.candidates.use_subtask` | `true`，按锚点选描述；回退时 WARNING 打印完整 task 文本 |

- 每个决策使用一个当前状态锚点，历史逐 latent 因果可见，未来视频和动作组成一个联合决策。默认得到 65 个历史 latent、2 个未来 latent、32 个动作和 1 个状态。
- P/H 均需按 VAE 组对齐；负时间历史重复 episode 首帧，未来不足 H 时补齐且不监督补齐部分。`max_actions` 默认从 H 推导，显式指定时必须等于 H。
- Action 不再提供 `layout_mode`、`full_history`、`history_range` 或 segmented 训练分支。Video 预训练的独立配置仍支持通用分块与随机历史，见[训练说明](docs/usage/causal_cosmos3_training.md)。
- 指定 `grad_accum_iter` 时省略 `global_batch_size`。全梯度基线和 Video 路径的 `micro_batch_size` 保持 `1`。Action episode TBPTT 可用 `micro_batch_size: 2`、`grad_accum_iter: 1`，将每卡两条独立流的段合入一次 DiT 前后向；每次更新消费 `DP卡数 × micro_batch_size × grad_accum_iter` 个段，短段保留实际决策数。
- B>1 由主进程从不同 worker lane 各取一个 FIFO 队首，要求每 rank 至少 B 条活跃 lane（由 worker 数和该 rank 持有的 episode 数共同限制）。同一流前后依赖的两段不能在一次前向中并排。历史、文本、attention 和 VAE 状态分别隔离；P1024/H32、S256 及既有梯度截断规则保持。
- 当前训练使用至少 2 卡、FSDP2、BF16、meta 初始化，SP / CP 保持 `1`。`cast_forward_inputs` 保持 `false`。
- 单个样本超过 `max_seq_len` 时仍会独立成包；长窗口需同时评估显存。
- NPU BF16 VAE 将对称空间补零交给原生 Conv3D，并省去零 padding 的复制；首帧＋4 帧因果分组、归一化和精度保持原样，其他设备／精度沿用 Diffusers 路径。训练不将 latent、embedding 或卷积特征缓存到磁盘。

Action 与独立 Video 预训练共用 TBPTT 开关和长度参数。在训练 YAML 的
`model.model_config` 中只需开启：

```yaml
enable_tbptt: true
```

`tbptt_segment_steps` 表示 **TBPTT 时间段跨度**，默认 **256步**，必须是正整数且为
**16的倍数**。内部只读属性 `tbptt_segment_latents = steps / 16` 换算 latent 跨度，
默认对应16个 latent 时间间隔；它不表示保留多少历史。
Action 中 steps 表示原始动作/控制步，开启 TBPTT 时要求数据几何满足
`video_stride * temporal_compression == 16`。`data.sample.decision_stride_steps`
独立控制决策间隔 D，缺省跟随预测长度 H；D 必须是16的倍数且能整除 H，段跨度 S 必须能被 D 整除。
S256/H32/D32 每段最多8个决策；RoboTwin TBPTT Policy 配方使用 D16，
每段最多16个决策，每次仍监督32条 action 和2个未来视频 latent。
文本边界或 episode 尾部可以提前结束一段。
每个决策都使用自己的实测状态锚点、
FD/ID/policy 采样和视频/动作 loss；同段共享 clean 历史编码，段边界截断历史编码梯度。
P1024 仍限制每个决策可见的历史为65个 latent；这个参数控制分段跨度，不替代 P。
这是直接可读的缓存长度；历史 hidden 已含更早观测的信息，不等于切断更早的前向信息。

数据按完整 episode 分配给 DP rank，再分给 worker lane；段内顺序和跨段 continuation
由 FIFO 保持，不再 greedy 重排。每次 episode 访问轮换合法 tick 相位，避免固定间隔 D
永久漏掉其他锚点。当前要求 `sampling: sample_uniform`；`source_weighted` 明确报错。
这里实际按 rank/worker 的 FIFO 段槽位消费；各 lane 的 episode 长度不同会改变长期窗口频率，
不保证旧版全数据 candidate 严格均匀。导出配置自动记录 `training_sampling_unit="episode_lane_fifo"`。
极短 episode 的合法相位更少，也会更频繁重访其锚点。
最终文本 token 改变会切 context，并清空旧 VAE/DiT 流状态后重新预填充历史。
重叠监督的原始目标会读到 `end_tick + H - D`，例如首个 S256/H32/D16 段读到272步，
下一段仍从256步开始；共享观测校验后复用，避免重复编码。每个决策的历史只到自己的锚点，
额外读到的未来目标不会提前成为可见历史。可用 `--decision-stride-steps 16` 覆盖间隔。

段间保留的是 bounded、已 detach 的层输入 hidden，下一段使用当前 norm/K/V 权重读取。
optimizer 更新后，旧 hidden 仍可能来自更新前权重，这是额外训练近似；每段 loss 按所有
rank 和梯度累积内的实际决策数归一化，短段不额外放大。缓存仅驻留内存，resume restart
清空全部流状态。默认 `enable_tbptt: false` 仍走原独立窗口全梯度训练。

NPU 的 B>1 训练还要求 `CLOSE_MATMUL_K_SHIFT=1`；未设置或显式设为其他值时，
Trainer 会在模型构建前报错。Action 启动脚本默认导出 `1`，直接使用 Python/torchrun
则需要自行提前 export。B1/B2 对照也应使用相同设置。本机完整 Edge 的相同 prepared
payload 审计中，关闭 Matmul K-shift 错峰后，B2 相对 B1 的完整梯度 L2 误差由约
15.6% 降至 0.275%，该次视频/动作输出及全部下一段缓存最大绝对差为0；梯度仍有
BF16 舍入差异。这是当前 NPU 软件栈的实测结果，不是所有 Ascend 版本或输入形状的
逐 bit 等价保证；详细条件见[多流训练说明](docs/usage/causal_cosmos3_training.md#npu-matmul-numerical-setting)。

独立 `stage: video` 中，同一参数控制 clip 内每个梯度截断时间段的跨度，
使用固定的等效时序标尺：16 steps = 1 latent；
steps 不等于 MP4 原始帧数，原始视频的 FPS 下采样规则保持不变。
内部 latent 跨度 `K = steps / 16` 必须是每个样本 `latent_block_size` 的整数倍。
例如256步对应 K16，block2 时每8个预测块为一个 chunk，独立首 latent 归第一个 chunk。
每层跨 chunk 的 clean 历史 hidden 在 norm 之前 detach，norm/K/V 读出参数仍学习；
所有预测块的视频目标继续监督，原前向值和注意力可见性不变。这里 K 是换算后的时间分段跨度，
不保证每个预测块都保留滑动的最近 K 个历史 latent 的编码梯度。

独立 Video 路径仍在一个 clip 内监督所有目标，不新增跨 clip 流状态。两种方式都默认关闭，
属于有意改变梯度的训练近似，性能和收敛需要分别测量。旧独立窗口 recent-K 仅保留在
standalone 模型诊断调用中，不再是 Action Trainer 的 TBPTT 路径。

配置只接受 `tbptt_segment_steps`，导出也只保存这个 steps 字段。
旧 `tbptt_keep_history_steps`、`tbptt_keep_history_latents` 及更早的 Video-only
长度字段不做兼容或自动迁移；用于 YAML、命令行或 JSON 配置时都会明确报错，
避免旧名称掩盖时间段跨度的语义或混用单位。

新 Action TBPTT 模型会自动记录 `history_encoding="episode_stream"`，推理据此使用连续
VAE 和稳定的流 RoPE 坐标；旧 window 模型保留旧编码方式。Episode-stream 支持 eager/RTC，
目前不支持 NPU replay graph，开启时明确报错。不能用旧窗口 graph 静默替代。
观测触发锚点需保持 16 tick 的 VAE 相位；固定 RTC S16/R24 满足此条件，自适应触发需要
保持同样的相位对齐。
Episode 入口异常会使当前流失效，必须重新 reset。旧离线 `teacher_forcing` / `rollout`
入口仅接受 window checkpoint；episode-stream 模型应使用在线服务入口。
配置和验证边界见[训练说明](docs/usage/causal_cosmos3_training.md#optional-history-gradient-truncation-tbptt)。

## 4. 启动训练与恢复

当前环境使用 Python 模块启动，避免依赖独立的 `torchrun` 可执行文件。

8 卡训练：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export CLOSE_MATMUL_K_SHIFT=1
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml
```

也可以直接通过 Action 脚本切换 TBPTT，无需修改 YAML（两条命令择一运行）：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 OUTPUT_DIR=outputs/gy_action_tbptt \
  bash scripts/causal_cosmos3/training_scripts/train_action.sh --enable-tbptt true

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 OUTPUT_DIR=outputs/gy_action_full \
  bash scripts/causal_cosmos3/training_scripts/train_action.sh --enable-tbptt false
```

`--enable-tbptt` 覆盖 YAML 的开关；省略时使用 YAML。可选
`--tbptt-segment-steps 256` 覆盖段跨度。模型、数据和输出路径可通过
`MODEL_PATH`、`DATA_CONFIG`、`OUTPUT_DIR` 设置；数据 YAML 内需填写真实数据路径。
脚本按可见卡数设置进程数和分片卡数，使用当前环境的 `torchrun`。

要合并每卡两条独立 episode 流，使用 B2、累积1；脚本会默认设置上述 NPU 环境变量：

```bash
OUTPUT_DIR=outputs/gy_action_tbptt_b2 \
  bash scripts/causal_cosmos3/training_scripts/train_action.sh \
  --enable-tbptt true --train.micro_batch_size 2 --train.grad_accum_iter 1
```

直接启动时，在 Python 之前显式设置环境变量，设备数和分片数按实际资源配置：

```bash
export CLOSE_MATMUL_K_SHIFT=1
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml \
  --enable-tbptt true --train.micro_batch_size 2 --train.grad_accum_iter 1
```

4 卡短跑（CLI 参数覆盖 YAML）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml \
  --model.accelerator.dp_shard_size 4 \
  --train.max_steps 10 --train.grad_accum_iter 1 \
  --train.checkpoint.output_dir outputs/causal_cosmos3_smoke \
  --train.wandb.enable false
```

已安装 `torchrun` 的环境也可使用现有脚本，日志写入 `log.txt`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash train.sh tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml \
  --model.accelerator.dp_shard_size 4
```

多机使用 `train.sh`，各节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`、`NPROC_PER_NODE`；总 DP 卡数需能被 `dp_shard_size` 整除。

恢复训练：在原启动命令末尾加上以下参数，并保持模型、GEN 参数选择和优化器配置一致。

```bash
--train.checkpoint.load_path outputs/causal_cosmos3_edge/checkpoints/global_step_1000
```

DCP 恢复模型、优化器、调度器和训练状态；数据固定使用 restart，恢复 epoch 的数据流重新开始。定期验证指标写入 `<output_dir>/validation.jsonl`，按 `eval/fd`、`eval/id`、`eval/policy` 分别记录。

## 5. 推理使用

与当前 Action 训练布局一致的实测观测、streaming VAE、异步和 RTC 服务见[在线推理说明](docs/usage/causal_cosmos3_serving.md)，P/H 和预处理需匹配训练。以下工具用于显式离线多块诊断与生成像素反馈实验，其分块上下文不是在线决策窗口。

训练、在线服务和相机预处理默认统一使用 `resolution=256`。当前 4:3 三视图输入拼成 **384×320（高×宽）**：上方头部视图 256×320，下方左右腕部各 128×160；客户端读取服务端公布的分辨率。

在线服务默认复用同一 episode 中未变化的首帧补齐前缀：VAE 保留独立卷积检查点，DiT 保留各层 K/V；后续只重算变化后的历史后缀、当前状态和未来。P1024 下，t16/t32 分别可复用 64/63 个历史时间 latent，输入长度和训练配方不变。首次准备仍有成本；reset 重新建缓存，历史窗口填满后释放。`--no-reuse-reset-prefix` 可关闭这两项复用做同配置对比。

### 准备完整训练权重

VeOmni DCP 先导出 safetensors，并保留该次训练的实际 `config.json`：

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir outputs/causal_cosmos3_edge/checkpoints/global_step_1000 \
  --save-dir outputs/causal_cosmos3_edge_1000_hf
cp outputs/causal_cosmos3_edge/model_assets/config.json \
  outputs/causal_cosmos3_edge_1000_hf/config.json
```

已有完整 native / HF 训练权重可直接使用。旧 Cosmos-framework DCP 使用 [专用转换脚本](scripts/causal_cosmos3/convert_inference_checkpoint.py)，命令见[推理文档](docs/usage/causal_cosmos3_inference.md)。

### 修改推理配置

编辑 `configs/infer/causal_cosmos3.yaml`；其中现有资源路径为 1030 POC 示例。

| 字段 | 要求 |
| --- | --- |
| `tokenizer` / `vae` | 匹配训练；推理 VAE 为包含 `config.json` 和权重的 Diffusers 目录 |
| `statistics` | 对应机器人、数据源的实测统计 |
| `source.cameras` / `source.columns` | 与输入数据列一致；核对 state / target / mask / offset |
| `geometry` / `source.resolution` / `source.viewpoint` | 与训练一致；默认每块 32 个 action，video stride 为 4 |
| `device` / `dtype` | 单卡 `npu:0` / `bfloat16` |
| `attention_backend` / `ops_implementation` | 与训练一致；当前为 `npu_tnd` + `torch_npu` RoPE / RMSNorm |
| `sampling` | 默认 20 步 UniPC、guidance=1、history=32；蒸馏权重可配置 fixed-step schedule |

### 运行

本节离线命令仅适用于 `history_encoding="window"` 的模型；episode-stream checkpoint 会明确报错。

| 模式 | 行为 |
| --- | --- |
| `teacher_forcing` | 每块使用真实历史与状态；`--task all` 分别运行 FD / ID / Policy |
| `rollout` | 仅用初始图像 / 状态，后续反馈生成像素和预测动作；仅支持 `policy` |

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # 物理卡映射为逻辑 npu:0

# 使用真实历史，预览三种任务
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/causal_cosmos3_edge_1000_hf \
  --mode teacher_forcing --task all \
  --dataset /path/to/lerobot_v3 --episode 0 --start 0 --max-actions 64 \
  --output outputs/teacher_episode0

# 从初始观测自主生成
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/causal_cosmos3_edge_1000_hf \
  --mode rollout --task policy \
  --dataset /path/to/lerobot_v3 --episode 0 --start 0 --max-actions 366 \
  --output outputs/rollout_episode0
```

- 常用覆盖参数：`--steps 20 --history 32 --seed 42 --guidance 1`；`--max-actions` 为动作步数。
- 初始 NPZ：将 `--dataset ... --episode ... --start ...` 换成 `--input-npz /path/to/initial.npz --fps 30`。内容为物理量 `state[55]`、RGB/HWC uint8 相机图像和非空 prompt；自定义机器人需配置有效通道 mask，详见推理文档。
- 输出包含 `generated.mp4`、`actions.npz`、`actions.svg`、`run.json`、`blocks.jsonl`；`--task all` 按任务创建子目录。已完成的输出目录不可重复使用。

## 6. 详细文档

[训练与验证](docs/usage/causal_cosmos3_training.md) · [网络实现](docs/usage/causal_cosmos3_network.md) · [独立推理](docs/usage/causal_cosmos3_inference.md) · [Checkpoint 导出](docs/usage/checkpoint_conversion.md)

基于 VeOmni；原始项目元信息见 `pyproject.toml`，许可证见 [LICENSE](LICENSE)。Cosmos3 模型代码的声明见 [NOTICE](veomni/models/diffusers/causal_cosmos3/NOTICE.txt)。
