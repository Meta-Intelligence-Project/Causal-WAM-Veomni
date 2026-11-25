# Frame-wise 训练与 dense subtask caption 交接

记录日期：2026-11-25。当前实现基线：`991403e2`（`feat/frame-wise-sampling`）。

工作区绝对路径：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/worktrees/frame-wise-sampling
```

本文记录已完成的读取与训练接入，以及下一步参考 SWAM dense subtask caption 分支的改造方案。**dense caption 选择逻辑尚未实现**，当前仍使用 anchor 帧的文本。下面的代码行号对应此基线；后续以函数名定位为准。

## 1. 已完成与后续范围

当前已有独立的 `LeRobotFrameDataset`、`ActionFrameDataset`、`VideoFrameDataset`，并已接入现有 causal 训练、Geometry 分块、policy/action、validation 和 packing。旧 `ActionDataset` / `VideoDataset` 保持独立。

`991403e2` 增加了固定样本数组包与有限 epoch：

- `data.samples_per_pack` 与 `data.max_seq_len` 同层级；配置前者后，`max_seq_len` 和 `dyn_bsz_buffer_size` 不参与训练组包。
- `train.micro_batch_size=1` 仍表示一个 pack，`grad_accum_iter` 表示每 rank 每次更新的 pack 数。
- 所有 source 为 `frame_anchor` 且 `sampling=sample_uniform` 时，可完整遍历有限候选集。尾部重复当轮打乱排列的前缀，补齐一次完整更新。
- 未配置 `samples_per_pack` 时继续使用原 token-budget / `source_weighted` 路径。固定 pack 目前不接受 `source_weighted`。

接下来拟迁移的是 **SWAM 按窗口实际抽到的视频帧选择一段 subtask caption 的语义**。保留现有 forward 窗口、候选数量、短 episode 处理与训练模型；窗口允许跨 subtask，不按语义边界裁剪。双向 `multi_anchor`、DreamZero shard 缓存/抽样，以及逐 block 切换文本，均不在这一步范围内。

## 2. 当前读取逻辑

### 2.1 候选窗口

训练子类要求 `past_frames=0`，从 anchor `t` 向后读取；`anchor_stride` 控制候选起点间隔。索引只存 episode 候选数量的前缀和，按需将候选编号映射成 episode 与 `t`。

- 长 episode：只枚举能够容纳完整 `future_frames` 的起点。
- 短 episode：存在有效转移时保留一个起点，按实际长度读取并补到最近的完整 block；不固定补到最大 horizon。
- 没有有效转移的单帧 episode 不进入训练候选。
- 原始检查用的基类允许每帧作为 anchor，包括 terminal observation；这与两个训练子类的候选集合不同。

例如，`target_offset=0/1`、`future_frames=128`、`anchor_stride=1` 时，一个包含 L 个观测的 episode 有 `max(1, L-128)` 个训练候选（要求 L≥2）。这不是“episode 每一帧都产生训练窗口”。

### 2.2 原始帧、action 与补齐

Action 的一步表示观测 `j → j+1` 的转移，目标来自 `j + target_offset` 行。因此 128 个 action 间隔需要 129 个原始观测位置；末端观测用于最后一步目标/转移对齐。

默认 Geometry：`actions_per_block=32`、`video_stride=4`、`temporal_compression=4`。完整窗口对应：

```text
原始观测位置：t ... t+128，共 129 个
视频抽帧位置：t, t+4, ... t+128，共 33 个
action：128 个
processor 输出的 block state：4 个，每个 block 使用自己的起始 state
VAE latent 时间长度：1 + (33-1)/4 = 9
```

短 episode 若只有 64 个观测（0…63），从 t=0 开始、`target_offset=0/1` 时只有 63 个有效 action 间隔，补到 64 个 action，即两个 block。视频是 17 帧，补齐查询重复实际终点；不会补成 128 个 action / 33 帧视频。有效性 mask 排除补齐位置。

Action 读取端在一次 payload I/O 前就确定有效终点和 block 长度；state/action 重复末端有效值，时间戳与有效性 mask 在该处生成。`ActionProcessor` 随后完成既有的 block 编码、映射、归一化等处理，不再重新决定窗口。

Video 子类不读取 action/state；按 `video_stride × temporal_compression × latent_block_size` 对齐实际前缀。未显式设置 stride 时，source FPS>16 使用 stride=2，否则为 1；自动规则仅在 `auto_video_stride and window.video_stride == 1` 时生效。Action 的 stride 由 Geometry 决定，二者应分开理解。

### 2.3 当前文本选择

`read_frame_window()` 将 timestamp、标签和所需 state/action 列放在同一次 projected Parquet read 中，但标签目前只以 `anchor_labels` 返回。

`LeRobotFrameDataset._caption()` 当前只根据 t 帧的 task/subtask ID 查文本。存在且启用 subtask 列时使用 subtask 文本；关闭 subtask 或数据没有该列时使用 task 文本。窗口跨过其他 subtask 时，文本仍属于 t。

已有 `FrameLabelConfig` 支持 root 内 ID→文本表、可选 episode 作用域及 JSONL 文本定义。ID 必须是有效整数，0 有效，未知 ID 报错；不支持此前已撤回的 null/ragged label 回退。

### 2.4 当前仓库阅读顺序

下列路径均相对于工作区根目录，可直接打开对应链接。

| 顺序 | 代码位置 | 关注内容 |
| --- | --- | --- |
| 1 | [`tasks/train_causal_cosmos3.py::parse_causal_cosmos3_args`](../../../tasks/train_causal_cosmos3.py#L25) | 组装 recipe 和 source YAML；实验字段进入 `dataset_config`。 |
| 2 | [`datasets/pipeline.py::build_dataset`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L186) | 根据 stage 和候选模式构造 Action/Video Frame Dataset。 |
| 3 | [`frame_dataset.py::FrameCandidateConfig / FrameLabelConfig`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L29) | 窗口参数、逐帧标签列与文本表契约。 |
| 4 | [`LeRobotFrameDataset::_forward_candidate_counts / locate`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L158) | 候选计数、前缀索引；Action/Video 分别在 `_candidate_ranges` 中提供可用长度。 |
| 5 | [`LeRobotFrameDataset::plan_read_window / read_sample / _caption`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L220) | 生成读取计划、一次读取 payload、当前 anchor caption。下一步主要修改位置。 |
| 6 | [`readers/lerobot.py::plan_frame_window / read_frame_window`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L374) | 有效终点、采样索引、一次 projected read、补齐、mask、标签返回。 |
| 7 | [`ActionFrameDataset::make_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L450) | 先按 Geometry 读到最终长度，再解码、processor、history/policy、prompt 和 token layout。 |
| 8 | [`VideoFrameDataset::plan_read_window / make_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L539) | 自动 stride、视频 block 对齐、有效 latent 数与 token layout。 |
| 9 | [`transforms/processing.py::Geometry / ActionProcessor`](../../../veomni/data/causal_cosmos3/transforms/processing.py#L35) | action/video 时序比例、block state、编码与归一化。 |
| 10 | [`LeRobotFrameDataset::prepare_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L299) 与 [`format_prompt`](../../../veomni/data/causal_cosmos3/transforms/layout.py#L183) | 使用最终 caption 格式化并 tokenize；文本长度进入真实 token 计数。 |
| 11 | [`datasets/validation.py::iter_validation_batches`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L59) | frame 验证直接复用 `make_sample`，强制 full history；identity 包含最终 caption。 |
| 12 | [`datasets/pipeline.py::FrameEpochSampler / build_frame_epoch_dataloader`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L52) 与 [`trainer/causal_cosmos3.py::_build_dataset / _build_dataloader`](../../../veomni/trainer/causal_cosmos3.py#L178) | 固定 B 组包、尾部重复、划分验证集后计算 epoch 步数。caption 改造应复用此链路。 |

配置与现有说明：

- [DiT recipe](../../../configs/dit/causal_cosmos3_edge_action_frame.yaml)：`data.stage/geometry/candidates/sample/samples_per_pack` 与训练参数。
- [source YAML](../../../configs/data/frame_datasets.yaml)：sampling、可选 validation、sources，以及来源相关的 labels/path/statistics 配置。
- [frame dataset 当前契约](frame_dataset.md)；[整体设计](../../design/lerobot_frame_dataset.md)。这些文档当前描述 anchor 文本，实施后需同步更新。

## 3. SWAM：优先参考的本地代码

SWAM 根目录：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/SWAM
```

**实际数据入口位于 `rlinf/data/datasets/dreamzero/`。** 同一仓库下还有 `dreamzero/groot/...` 副本，不能仅凭目录名把它当成当前 SWAM 的数据实现。

### 3.1 核心参考：dense caption resolver

完整路径：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/SWAM/rlinf/data/datasets/dreamzero/dense_subtasks.py
```

- `DenseSubtaskIndex.load()`，约第 68 行：读取 dense segment 标注。有效段筛选在约第 291 行，按 episode、start、end 排序在约第 333 行。
- `DenseSubtaskIndex.resolve()`，第 378 行：核心选择规则。先对实际采样的 `video_indices` 去重，然后分别统计每段半开区间 `[start_frame, end_frame)` 中包含多少个采样位置；选择计数最大的段。并列时 `np.argmax` 取排序后的第一个段。没有覆盖才返回 None。
- 该分支保留 `visual_clear=true`、`segment_type=manipulation` 且 caption 非空的段。这些是 SWAM dense 文件的字段与过滤规则；本仓库现有 ID 表没有这些字段，不应自行假定或添加同类过滤。
- `caption_granularity=episode` 会走另一条整 episode 文本分支。本次拟参考的是 `subtask` 粒度 resolver，不能将所有 unified_dense 配置都理解为按窗口选 subtask。

### 3.2 调用顺序与配置传递

| 相对 SWAM 根目录的路径 | 函数 / 行号 | 用途 |
| --- | --- | --- |
| `rlinf/data/datasets/dreamzero/dreamzero.py` | `_attach_dense_subtask_instruction`，1928；`_materialize_parquet_sample`，1941 | 先构造并 clip video/state/action 索引，再传实际 `video_idx` 解析 caption；发生在视频解码之前，不依赖像素。 |
| `rlinf/data/datasets/dreamzero/dreamzero.py` | `_build_modality_dict`，2430；dense text 分支，2511 | dense caption 优先写入 language 字段，再交给后续 transform。 |
| `rlinf/data/datasets/dreamzero/mixed.py` | `_build_dataset_bundle`，417；配置传递，541–551 | `subtask_mode`、`subtask_file`、caption 字段与 granularity 的入口。 |
| `rlinf/data/datasets/dreamzero/sampling_strategy.py` | `sample_video_indices`，93；`sample_action_indices`，171；`sample_state_indices`，243 | 窗口索引怎么生成；与 dense caption resolver 是两个独立环节。这里的双向 multi-anchor 暂不迁移。 |
| `examples/sft/config/pretrain.yaml` | source 配置，44–49；sampling 配置，443 起 | 本地示例中机器人 source 的 unified_dense 使用 `caption_granularity: episode`；multi-anchor 示例为 macro stride=48、video offsets=0,6,…,42。示例配置不等于所有训练均使用 subtask 粒度。 |

建议按 `resolve → _materialize_parquet_sample → _build_modality_dict → mixed.py` 阅读。完整的 materialize 调用文件路径为：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/SWAM/rlinf/data/datasets/dreamzero/dreamzero.py
```

SWAM 传入的是 clip 后的索引，resolver 再 `unique`；重复的末端索引不会被重复计票。本仓库已有显式 padding mask，拟直接只让有效视频采样位置计票。两者在边界上不保证完全相同，需在实现和测试中写清楚。

## 4. DreamZero：窗口与抽样的对照路径

DreamZero 根目录：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/dreamzero
```

| 相对 DreamZero 根目录的路径 | 函数 / 行号 | 已核对行为与用途 |
| --- | --- | --- |
| `groot/vla/data/dataset/lerobot_sharded.py` | `_uniform_sample_from_language_ranges`，1131 | 本地 DROID 分支以原始 language anchor 值约束，向前/向后扩展完整 chunk，排序成实际窗口。实际代码为 chunk 步长 24、视频间隔 3；附近有未同步的 32/8 注释，以代码为准。它不是 dense caption resolver。 |
| `groot/vla/data/dataset/lerobot_sharded.py` | `generate_shards`，397 | 以整个 episode 分组到 shard，默认目标约 10000 个候选 step；shard 不是 subtask。 |
| `groot/vla/data/dataset/lerobot_sharded.py` | mixture 的 `__iter__`，1465；候选过滤，1492；shuffle/cap，1507–1509 | 先按 episode 与 step filter 收集候选，再限制尾部起点、shuffle，最后按 `num_steps_per_shard * shard_sampling_rate` 截取。这会影响实际消费条数。 |
| `groot/vla/data/dataset/lerobot.py` | `_get_step_filter`，975 | 有 `meta/step_filter.jsonl` 时排除其中列出的 step；没有文件则初始保留所有 step。 |
| `groot/vla/configs/data/dreamzero/droid_relative.yaml` | 13–15 | 本地示例 `max_chunk_size=5`、`shard_sampling_rate=0.1`。这些配置不能直接当成本仓库的样本计数规则。 |

完整的窗口参考路径：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/dreamzero/groot/vla/data/dataset/lerobot_sharded.py
```

上述实现用于解释此前讨论的向前/向后取帧、过滤与 shard 抽样。下一步 caption 改造不引入这些机制，也不宣称候选数与 DreamZero 相同。

## 5. 下一步建议实施方案（尚未执行）

### 5.1 用现有逐帧 ID 表达连续段

复用现有 task/subtask ID→文本映射。对读取窗口中的逐帧 subtask ID 按连续相同值划段，再统计每段覆盖的有效视频采样位置数量。不要直接按 subtask ID 对整窗求众数：同一个 ID 在中间被其他段打断后重新出现，应作为两个独立段参与比较。

例如 stride=4，实际有效视频索引为 `[0,4,8,12,16]`：

```text
逐帧连续段：[0,4) A；[4,13) B；[13,17) A
覆盖的视频采样点：A 第一段=1，B=3，A 第二段=1
当前 anchor 文本=A；改造后窗口文本=B
```

即使 stride 跳过中间很短的一段，也必须根据完整逐帧 ID 序列识别边界，不能只在下采样后的标签上划段。并列取时间最早的段。补齐位置不计票；不根据 action 数量、原始区间时长或解码像素计票。

这复用了 SWAM 的“视频采样覆盖最多的一段”选择语义，数据载体仍是本仓库已有的逐帧 ID。无需搬入完整的 dense interval Parquet loader、质量过滤或新 Dataset。

### 5.2 主要改动位置

1. **`readers/lerobot.py::read_frame_window`**：同一次 projected read 已读到范围内的标签列。将需要的窗口标签序列与原始观测时间轴对齐返回，让上层拿到划段依据；沿用现有 stride、有效终点和 masks。避免为文本再读一次 Parquet，避免冗余返回可由现有字段推导的索引副本。
2. **`frame_dataset.py::LeRobotFrameDataset.read_sample / _caption`**：在窗口索引确定后，根据连续段覆盖选择 subtask，更新 `caption/subtask_index/caption_kind`。使用 Action 读取端根据 target_offset 修正后的有效终点，以及 Video 最终采用的 stride。两个训练子类共用该逻辑。
3. **同步测试、注释和现有说明**：目前写明 anchor text、或断言 caption 等于第一帧文本的位置需要调整；validation 继续调用同一个 `make_sample`，不另外实现 caption 规则。

保持 task-only 数据路径；已有的 `labels.subtask_column: null` 是关闭 subtask 读取的配置，不表示允许帧标签值为 null。无需为第一版增加 caption 策略开关或额外 config 类。若实施时发现确有必要新增训练构造字段，应放 recipe 的 `data` 下面；source YAML 继续负责标签来源与映射。

caption 必须在 `prepare_sample → format_prompt → tokenize → token_layout` 之前确定。这样文本 token 数会自然更新，后续 Geometry packing、网络、causal mask、policy loss、训练回调都可复用。

### 5.3 需要明确的影响

- `samples_per_pack` 的 B、候选数量、每轮步数、action/state/video/mask 应保持既有规则。
- `source_weighted` 的抽样算法与 source 权重不变；文本长度改变时，token-budget 路径的 pack 组合可能随之变化。
- validation identity 包含 caption，修改文本会改变其派生的固定噪声 seed；同一新版样本重复验证仍应确定一致。不能把新旧文本下 loss 完全一致作为验收标准。
- 没有新证据时，不重新加入此前删除的负索引接口、nullable/ragged label 支持、重复 padding 或无实际作用的配置。

## 6. 验证与交接材料

### 6.1 下一步测试重点

测试放在现有 [frame dataset 测试](../../../tests/data/test_causal_cosmos3_frame_dataset.py) 中，覆盖实际语义：

- anchor 所在段与覆盖最多段不同；按视频采样点计数与按原始帧数计数会得出不同答案。
- 非连续重复 ID 分段统计；并列时选最早段；stride 跳过短段仍保留原始边界。
- 短 episode / 尾部 padding 不参与投票，Action target_offset 与 Video 自动 stride 使用实际有效范围。
- task-only、ID=0、未知 ID 的既有契约保持；同一次 payload read 完成标签选择，不额外解码视频。
- Action/Video 共用选择逻辑，train/validation 对同一窗口使用相同 caption，prompt/token_length 跟随最终文本。
- 候选集合、Geometry 编码和固定 pack/epoch 回归继续通过。

当前较直接的定位点：`test_one_projected_read_anchor_labels_and_output_ownership`（233 行）、`test_four_blocks_use_four_states_and_keep_frame_labels`（581 行）、`test_video_short_window_reads_labels_only_and_clips_decode`（1101 行）。它们仍含 anchor 文本预期，实施时需重新审视。

### 6.2 现有真实数据及验证产物

可复用数据：

```text
/mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__turn_switch/clean
/mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__open_laptop/clean
/mnt/sfs_turbo/public/datasets/egosuite_processed/EgoStandard-body__Add_Syrup_to_Coffee
```

已检查这些数据有逐帧整数 task/subtask ID，文本来自 `meta/tasks.parquet` 与 `meta/subtasks.parquet`。优先复用现有映射，不凭空扩展标注格式。

已有读取/组包验证产物在仓库外：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/artifacts/frame-pack-epoch-validation-20261124/
  report.json
  check_real_data.py
  provenance.json
  run-final.log
  trainer-report.json
```

其中包括真实 Parquet/NumPy action oracle 与 PyAV 视频对照、两轮 epoch、模拟 DP 分片、worker=0/2、固定 B=8 与尾部重复检查，以及真实 Trainer 配置解析/划分/loader/LR hook 检查。Trainer 检查使用 CPU eager 覆盖并替换了模型 runtime，未加载真实权重；Geometry 部分使用形状匹配的测试 latent。它们不等同于多卡训练或真实 VAE/完整模型的 loss/梯度验收。

`991403e2` 提交前执行：

```bash
make quality
python -m pytest tests/data/test_causal_cosmos3_frame_dataset.py tests/data/test_causal_cosmos3_pipeline.py -q
make -C docs html SPHINXOPTS=-W
```

风格检查与文档构建通过。两份测试共 438 passed、4 xfailed、1 failed。唯一失败为旧测试 `test_getitem_skips_errors_but_make_sample_propagates` 的日志捕获计数：预期 4，实际 8；在提交前 HEAD `5d3363ad` 的独立归档目录也复现，未混入本次功能修复。

本机可用 CPU Python 为 `/tmp/veomni-issue13-venv/bin/python`；ruff 位于 `/mnt/sfs_turbo/zheng/envs/veomni-cosmos3/bin/ruff`；Sphinx 位于 `/tmp/veomni-issue13-venv/bin/sphinx-build`。这些是本地环境路径，不是项目依赖要求。后续实现完成后应先跑相关回归，再在真实训练环境做 VAE/模型执行检查。
