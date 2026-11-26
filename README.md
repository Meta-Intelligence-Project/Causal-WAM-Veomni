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
│   ├── dit/causal_cosmos3_edge_action.yaml     # 模型、优化器、并行、训练参数
│   ├── data/action_datasets.yaml       # 动作数据源、采样、验证集划分
│   ├── data/video_datasets.yaml        # 视频数据源、采样、验证集划分
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
| 同上 | `train.checkpoint.output_dir` | 本次训练输出目录 |
| `configs/data/action_datasets.yaml` | `sources[].roots` | LeRobot v3 数据集根目录，支持列表和 glob |
| 同上 | `sources[].statistics` | 实测动作归一化统计；支持 `'{root}/meta/causal_action_stats.json'` |
| 同上 | `sources[].cameras` | head / left / right 对应的真实视频列名 |
| 同上 | `sources[].columns` | state、target、mask 列及 `target_offset`；按机器人数据语义配置 |

数据 YAML 只配置 `sampling / validation / sources`；`stage / geometry / candidates / sample` 在 DiT YAML 的 `data` 块中配置。入口函数组装完整数据配置后显式传给 Trainer；输出目录的 `dataset_config.yaml` 保存组装结果，`veomni_cli.yaml` 保存框架参数。

数据需包含 LeRobot v3 的 `meta/`、`data/`、`videos/`；动作 / 状态统一为 55D，并提供有效通道 mask 和非空任务文本。子任务文本及边界可由 `meta/episodes.jsonl` 提供。

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
| 训练 YAML | `data.sample.mode` / `data.sample.mode_weights` | `joint` / `[1, 1, 1]`，依次为 FD、ID、Policy；也可固定单任务 |
| 训练 YAML | `data.sample.enable_cfg` / `caption_dropout_prob` | `false` / `0.15`；默认关闭 CFG 训练，传 `--cfg` 开启 Policy 指令随机置空（验证集固定不置空） |
| 训练 YAML | `model.model_config.enable_txt_steering` / `txt_steering_scale` | `false` / `1.3`；默认关闭跨模态文本注意力引导，传 `--txt-steering` 开启（线性生效区间 `[250, 1000]`） |
| 数据 YAML | `validation` | 按 episode 留出 5%，每源最多 100 个，seed 为 73；移除 `split_ratio` 关闭划分与定期验证 |

窗口与阶段设置（均在 DiT 训练 YAML 的 `data` 块）：

| 字段 | Stage 1 默认 | Stage 2 示例 |
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

## 4. 启动训练与恢复

当前环境使用 Python 模块启动，避免依赖独立的 `torchrun` 可执行文件。

8 卡训练：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml
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

DCP 恢复模型、优化器、调度器和训练状态；动态数据加载器尚不保证逐样本完全重放。定期验证指标写入 `<output_dir>/validation.jsonl`，按 `eval/fd`、`eval/id`、`eval/policy` 分别记录；开启 `--cfg` 时额外在相同噪声与时间步下对照无条件空指令分支，记录 `eval/policy/loss_uncond`、`eval/policy/text_gain`（$= \text{loss}_{\text{uncond}} - \text{loss}_{\text{cond}}$）及分模态 `video_text_gain` / `action_text_gain`，用于在线监控模型是否真正利用文本指令。

## 5. 推理使用

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
| `sampling` | 默认 20 步 UniPC、`enable_cfg: false`（等效 `cfg=1.0`）、`enable_txt_steering: null`（自动继承 checkpoint `config.json`）、`shift=1.0`、`history=32`；传 `--cfg` 启用预置 CFG 参数（`video_guidance=3.0`、`action_guidance=1.5`、`guidance_interval=[250, 1000]`、`normalize_cfg=true` 及同构空指令 JSON）；蒸馏权重可配置 fixed-step schedule |

### 运行

| 模式 | 行为 |
| --- | --- |
| `teacher_forcing` | 每块使用真实历史与状态；`--task all` 分别运行 FD / ID / Policy |
| `rollout` | 仅编码初始图像一次，后续在 latent 空间直接反馈生成视频潜变量并反馈最后有效动作，末尾统一解码 MP4；仅支持 `policy` |

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

- 常用覆盖参数：`--steps 20 --shift 3.0 --history 32 --seed 42`；CFG 显式开关 `--cfg` / `--no-cfg`（及 `--guidance`、`--video-guidance`、`--action-guidance`）；Text Steering 显式覆盖 `--txt-steering` / `--no-txt-steering`（及 `--txt-steering-scale`、`--txt-steering-min-timestep`、`--txt-steering-max-timestep`）；`--max-actions` 为动作步数。
- 初始 NPZ：将 `--dataset ... --episode ... --start ...` 换成 `--input-npz /path/to/initial.npz --fps 30`。内容为物理量 `state[55]`、RGB/HWC uint8 相机图像和非空 prompt；自定义机器人需配置有效通道 mask，详见推理文档。
- 输出包含 `generated.mp4`、`actions.npz`、`actions.svg`、`run.json`、`blocks.jsonl`；`--task all` 按任务创建子目录。已完成的输出目录不可重复使用。

## 6. 指令跟随增强（`main-cfg-txt-steering` 分支 Best Practice）

针对多视角 / 长历史后训练中视觉与历史帧先验过强、有效指令 token 被稀释导致的 Policy“不跟指令”问题，本分支提供两套正交互补机制：

- **CFG（外部残差引导）**：训练期按 `caption_dropout_prob=0.15` 将 Policy 样本的任务描述随机置空为同构 JSON（`"actions": []`，保留相机画幅 / FPS 元数据，验证集固定不置空）；推理期分离视频与动作引导强度（默认 `video_guidance=3.0`、`action_guidance=1.5`，防止动作超调抖动），配合 `guidance_interval=[250, 1000]` 与 `normalize_cfg=true`。
- **Text Steering（内部跨模态注意力引导）**：仅对预测块（`role=1`）在模型时间 $t \in (250, 1000]$ 内线性缩放**任务指令区间**（`text_steering_mask=True`）的 `gen -> text` 跨模态 logits $\gamma(t)$（峰值默认 `1.3`）。对话模板特殊 token（Attention Sink）、固定的多视角 / 分辨率 JSON 壳子、无条件空指令分支（`"actions": []`），以及 $t \le 250$ 低噪细化阶段和所有历史 / 条件 / 状态块均保持 `1.0`，避免 Sink 膨胀挤占视觉/状态预算或固定视角文本稀释动作指令。

### 推荐训练与评测配方

1. **标准 Policy 后训练 + 评测（推荐）**
   - 训练同时开启 `--cfg --txt-steering`，窗口模式保持 `candidates.mode: segmented`（按子任务边界切窗，避免跨阶段图文错位）；若任务存在明显多目标视觉歧义，可将 `data.sample.mode_weights` 中 Policy 权重适当调高（如 `[0.5, 0.5, 2.0]`）：
   ```bash
   # 1) 训练：开启 Policy CFG 指令置空 + 跨模态 Text Steering
   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
   python -m torch.distributed.run --standalone --nproc_per_node=8 \
     tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml \
     --cfg --caption-dropout-prob 0.15 \
     --txt-steering --txt-steering-scale 1.3 \
     --train.checkpoint.output_dir outputs/causal_cosmos3_cfg_steered

   # 2) 导出：务必同步拷贝 model_assets/config.json（保留训练时的 txt_steering 配置供推理自动继承）
   python scripts/merge_dcp_to_hf.py \
     --load-dir outputs/causal_cosmos3_cfg_steered/checkpoints/global_step_10000 \
     --save-dir outputs/causal_cosmos3_cfg_steered_10000_hf
   cp outputs/causal_cosmos3_cfg_steered/model_assets/config.json \
     outputs/causal_cosmos3_cfg_steered_10000_hf/config.json

   # 3) 推理评测：开启 --cfg 与分辨率匹配的 --shift（txt_steering 自动继承 config.json）
   ASCEND_RT_VISIBLE_DEVICES=0 python tasks/infer/infer_causal_cosmos3.py \
     --config configs/infer/causal_cosmos3.yaml \
     --checkpoint outputs/causal_cosmos3_cfg_steered_10000_hf \
     --mode rollout --task policy \
     --cfg --shift 3.0 \
     --dataset /path/to/lerobot_v3 --episode 0 --start 0 --max-actions 160 \
     --output outputs/rollout_cfg_steered_ep0
   ```
2. **存量旧 Checkpoint（未经 CFG 训练）零重训调优**
   - 旧权重未学习过空指令无条件分支，不建议直接开高倍率 `--cfg`；可单独开启 `--shift 3.0 --txt-steering --txt-steering-scale 1.2`（在 `1.15 ~ 1.35` 范围内扫描）做 Training-Free 指令敏感度验证，需要去掉时传 `--no-txt-steering` 即可对照。

### 实验调参必看注意事项

- **显式主开关机制（防止调参静默失效）**：`--cfg` 与 `--txt-steering` 均设计了显式布尔主开关保护。当 YAML 或 Checkpoint 中对应开关为 `false` 时，**如果命令行只传数值参数（例如仅传 `--guidance 3.0` 或仅传 `--txt-steering-scale 1.5`）而未传 `--cfg` / `--txt-steering`，功能仍保持关闭状态不会生效**。跑消融实验时请务必同时传入主开关，并在输出目录的 `run.json`（`config.sampling`）中核对最终生效值。
- **推理 `--shift` 与时间步区间的强联动（重要）**：训练侧 `configs/model_configs/causal_cosmos3/condition.json` 按分辨率使用时间步偏移（`256` 分辨率为 `shift: 3`，`480` 分辨率为 `shift: 5`），而推理配置为兼容历史默认保留 `shift: 1.0`。由于 `txt_steering` 在 `[250, 1000]` 上是线性爬坡（例如 $t=500$ 时 $\gamma \approx 1.10$，$t=850$ 时才达 $\gamma \approx 1.24$），且 CFG 仅在 `guidance_interval`（默认 `[250, 1000]`）内介入，若推理沿用 `shift=1.0`，约一半采样步会落在 $t < 500$ 的弱引导区间。**正式评测 `256` 分辨率模型时请显式传入 `--shift 3.0`（`480` 分辨率传 `--shift 5.0`）**，既与训练噪声分布严格对齐，也让 CFG 和 Text Steering 在高噪规划阶段充分发力。

## 7. 详细文档

[训练与验证](docs/usage/causal_cosmos3_training.md) · [网络实现](docs/usage/causal_cosmos3_network.md) · [独立推理](docs/usage/causal_cosmos3_inference.md) · [Checkpoint 导出](docs/usage/checkpoint_conversion.md)

基于 VeOmni；原始项目元信息见 `pyproject.toml`，许可证见 [LICENSE](LICENSE)。Cosmos3 模型代码的声明见 [NOTICE](veomni/models/diffusers/causal_cosmos3/NOTICE.txt)。
