# Frame-wise 训练与 dense subtask caption 交接


双向采集验收与后续事项见第 11 节，最新 caption 规则见第 12 节，可选块起点 subtask 匹配见第 13 节；第 1～7 节包含迁移前基线与历史阶段记录，现行接口以 [frame dataset 契约](frame_dataset.md) 为准。
更新日期：2026-11-26。迁移基线：`991403e2`（`feat/frame-wise-sampling`）。

工作区绝对路径：

```text
/mnt/sfs_turbo/zheng/Veomni-Wam/worktrees/frame-wise-sampling
```

本文记录 `991403e2` 基线的读取与训练接入，以及参考 SWAM 的后续改造方案。**第 2–6 节保留 caption 阶段的交接背景；第 7 节保留上一轮 caption 实现记录（该轮未提交修改已清理）；第 8 节是用户修正后的完整目标，优先于此前只迁移 caption 的范围。第 9、10 节分别记录独立窗口采集与正式接入，第 11 节记录最新验收和遗留问题。** 代码定位以函数名为准。

## 1. 已完成与后续范围

当前已有独立的 `LeRobotFrameDataset`、`ActionFrameDataset`、`VideoFrameDataset`，并已接入现有 causal 训练、Geometry 分块、policy/action、validation 和 packing。旧 `ActionDataset` / `VideoDataset` 保持独立。

`991403e2` 增加了固定样本数组包与有限 epoch：

- `data.samples_per_pack` 与 `data.max_seq_len` 同层级；配置前者后，`max_seq_len` 和 `dyn_bsz_buffer_size` 不参与训练组包。
- `train.micro_batch_size=1` 仍表示一个 pack，`grad_accum_iter` 表示每 rank 每次更新的 pack 数。
- 所有 source 为 `frame_anchor` 且 `sampling=sample_uniform` 时，可完整遍历有限候选集。尾部重复当轮打乱排列的前缀，补齐一次完整更新。
- 未配置 `samples_per_pack` 时继续使用原 token-budget / `source_weighted` 路径。固定 pack 目前不接受 `source_weighted`。

用户已修正迁移范围：**既迁移 SWAM 的 multi-anchor 取窗口逻辑，也迁移窗口确定后按视频帧覆盖选择 subtask 文本的规则。** 不再要求保留现有 forward 窗口、候选起点集合或短 episode 的窗口构造方式。现有独立 Dataset 类、Geometry、causal/policy 训练、mask、packing 与 validation 链路继续复用；DreamZero shard 缓存/抽样及逐 block 切换文本不属于此目标。实施边界见第 8 节。

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

`FrameLabelConfig` 支持 root 内 ID→文本表及 JSONL 文本定义，不再保留 episode 作用域。task/subtask ID 直接转换为整数，0 有效，未知 ID 报错；不支持此前已撤回的 null/ragged label 回退。

### 2.4 当前仓库阅读顺序

下列路径均相对于工作区根目录，可直接打开对应链接。

| 顺序 | 代码位置 | 关注内容 |
| --- | --- | --- |
| 1 | [`tasks/train_causal_cosmos3.py::parse_causal_cosmos3_args`](../../../tasks/train_causal_cosmos3.py#L25) | 组装 recipe 和 source YAML；实验字段进入 `dataset_config`。 |
| 2 | [`datasets/pipeline.py::build_dataset`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L186) | 根据 stage 和候选模式构造 Action/Video Frame Dataset。 |
| 3 | [`frame_dataset.py::FrameCandidateConfig / FrameLabelConfig`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L35) | 窗口参数、逐帧标签列与文本表契约。 |
| 4 | [`LeRobotFrameDataset::_candidate_ranges / locate`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L161) | 候选计数、前缀索引；Action/Video 分别在 `_candidate_ranges` 中提供可用长度。 |
| 5 | [`LeRobotFrameDataset::plan_read_window / read_sample / _caption`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L214) | 生成读取计划、一次读取 payload、当前 anchor caption。下一步主要修改位置。 |
| 6 | [`readers/lerobot.py::plan_frame_window / read_frame_window`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L377) | 有效终点、采样索引、一次 projected read、补齐、mask、标签返回。 |
| 7 | [`ActionFrameDataset::make_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L497) | 先按 Geometry 读到最终长度，再解码、processor、history/policy、prompt 和 token layout。 |
| 8 | [`VideoFrameDataset::plan_read_window / make_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L582) | 自动 stride、视频 block 对齐、有效 latent 数与 token layout。 |
| 9 | [`transforms/processing.py::Geometry / ActionProcessor`](../../../veomni/data/causal_cosmos3/transforms/processing.py#L35) | action/video 时序比例、block state、编码与归一化。 |
| 10 | [`LeRobotFrameDataset::prepare_sample`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L349) 与 [`format_prompt`](../../../veomni/data/causal_cosmos3/transforms/layout.py#L183) | 使用最终 caption 格式化并 tokenize；文本长度进入真实 token 计数。 |
| 11 | [`datasets/validation.py::iter_validation_batches`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L76) | frame 验证直接复用 `make_sample`，强制 full history；identity 包含最终 caption。 |
| 12 | [`datasets/pipeline.py::FrameEpochSampler / build_frame_epoch_dataloader`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L52) 与 [`trainer/causal_cosmos3.py::_build_dataset / _build_dataloader`](../../../veomni/trainer/causal_cosmos3.py#L179) | 固定 B 组包、尾部重复、划分验证集后计算 epoch 步数。caption 改造应复用此链路。 |

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
| `rlinf/data/datasets/dreamzero/sampling_strategy.py` | `sample_video_indices`，93；`sample_action_indices`，171；`sample_state_indices`，243 | 窗口索引怎么生成；与 dense caption resolver 是两个独立环节。这里的双向 multi-anchor 已纳入第 8 节目标。 |
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

上述实现用于对照向前/向后取帧、过滤与 shard 抽样。下一步需参考 SWAM 的 multi-anchor 实现迁移窗口；不引入 DreamZero 的 shard 机制，也不宣称候选数与 DreamZero 相同。

## 5. Caption 阶段方案（工作区实现见第 7 节）

本节只记录 caption 阶段的改造；它不代表修正后的完整迁移范围。后续还需要第 8 节的取窗口与候选索引改造。

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

- 仅迁移 caption 时，`samples_per_pack` 的 B、候选数量、每轮步数、action/state/video/mask 保持既有规则。完整窗口迁移会改变候选/窗口，epoch 步数必须按新的 Dataset 长度重新计算。
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

## 7. 上一轮 caption 阶段实施记录（2026-11-25，已清理）

以下保留历史实现与验证记录。按用户要求，该轮未提交的 caption 代码、测试和配套文档修改已清理；当前工作区尚未接入窗口 caption 选择，不能据此认定完整迁移已经完成。

`read_frame_window` 在同一次 projected read 后返回与观测时间轴对齐的 frame_labels，保留原 anchor_labels。`LeRobotFrameDataset._caption` 根据完整逐帧 subtask ID 划连续段，以有效视频采样点计数，选覆盖最多的段，并列选最早段。非连续重复 ID 不合并，padding 不计票，也不会额外计入网格外被 clip 到的终点。选择完成后，Dataset 移除临时 frame_labels，返回最终 caption/subtask_index/caption_kind；task-only 使用原 anchor task。

Action 和 Video 共用该逻辑，分别使用 target_offset 修正后的有效终点与最终 video_stride。最终文本先于 prompt/tokenize/token_layout 确定，validation 继续直接复用 make_sample。没有新增配置、候选过滤、区间 loader 或重复解码。

回归覆盖连续段、跳过短段、并列、采样点与时长差异、补齐、offset、自动/显式 stride、ID 契约、task-only、单次 payload read、文本 token 数及重复验证。现行接口以[数据契约](frame_dataset.md#按帧标注)为准；真实 VAE/模型 loss 和梯度仍需在训练环境验收。

实施后的两份测试为 **459 passed、4 xfailed、1 failed**，比修改前新增 21 个参数用例。唯一失败仍为第 6.2 节的旧日志计数测试，修改前基线已重新复现；未修改旧 Dataset 的异常处理。`make quality`、Sphinx `-W` 构建通过，子代理对所提供代码进行静态审查，未发现问题；该审查未独立执行检查。

仓库外验证产物：`/mnt/sfs_turbo/zheng/Veomni-Wam/artifacts/dense-subtask-caption-20261125/`。`caption-oracle.json` 对上述三个真实数据源的 Action（offset=0/3）与 Video 共 69 个窗口，使用独立逐段遍历和视频采样位置集合计数核对 caption。这些窗口均只有一个连续标注段，跨段选择由新增合成回归覆盖。`payload-oracle.json` 对 turn_switch 的 3 个 episode 核对独立 NumPy action 编码和 PyAV 视频像素，像素完全一致，action 最大绝对误差约 `9.16e-8`。验证脚本和日志保存在同一目录；这些检查不包含真实 VAE 或多卡模型训练。

## 8. 用户修正后的目标：窗口与 caption 一起迁移

### 8.1 取窗口的目标行为

从采样器给出的原始帧 t 出发，先尝试 t 所在块，再依次尝试前一块、后一块、前两块、后两块，最多保留配置允许的块数。每个块的 video/action/state 对齐，最终按时间顺序组织，再添加视频末端观测。某一侧越过 episode 边界或遇到不同的原始 task/language 起点标签时，按 SWAM 规则停止该侧扩展；完整块检查、最后一帧补充/回退及 action/state 数量一致性检查需一起对照参考代码。

不将 dense subtask 段边界直接作为窗口裁剪边界。窗口先由原始 task/language 与时序规则决定，随后才按实际视频采样覆盖选择 dense subtask 文本。

用户已确认并允许以下标注使用顺序，适用于 dense subtask caption 路径：

1. **选窗口时使用原始 task/language 标签。** 比较新块起点与原始 t 的标签，决定该侧是否继续扩展；保持 SWAM 的块起点检查规则，不擅自增加整块逐帧同标签检查。
2. **窗口确定后再选择 dense subtask 文本。** 根据最终实际视频帧号统计各连续 subtask 段的覆盖，选覆盖最多的一段。dense subtask 的边界不参与前一步选窗。
3. **允许窗口跨多个 dense subtask。** 原始 task 在整个 episode 内相同时，task 标签检查不会限制扩展，窗口主要受 episode 边界、完整块要求和最大块数限制。
4. **“提前知道标签”是处理顺序要求，不是额外 I/O 要求。** 选块时能访问所需原始标签即可，可复用已读标签或缓存，不要求额外读取一次 Parquet，更不要求加载全部数据。已确认整个 episode 的原始 task 相同时，无需再逐帧读取它来判断一致性。

后续测试需分别覆盖“原始 task 相同、dense subtask 变化时允许跨段”和“原始 task 改变时停止对应一侧扩展”，避免混淆这两套标注的作用。

复用本框架 Geometry 中的单位。以 Action 默认的每块 32 步、video stride=4、最多 4 块为例，t=64 且 episode/原始标签允许时，取块顺序为 `64 → 32 → 96 → 0`，排序后得到：

```text
原始块起点：0、32、64、96
视频：0、4、8、……、128，共 33 帧
action：0～127，共 128 步
state：0、32、64、96，共 4 个
最初抽中的 t=64；最终送入 VAE 的第一帧为 0。
```

SWAM 本地预训练示例的 48 步/块和 video stride=6 是其具体 Geometry 参数；迁移算法不要求本框架同时改成这两个值。Video 子类需要使用自身的 stride、temporal compression 和 latent block 定义推导块跨度，不能套用 action 列或 action processor。

### 8.2 候选起点与短样本必须一起审视

取消“仅完整前向起点，短 episode 只保留一个起点”作为新路径的固定约束。SWAM 以逐帧位置为抽样入口；靠近末尾的 t 也可能通过前面的完整块形成有效样本。一个起点无效，也不等于该 episode 没有有效样本。

SWAM 对无法形成完整、对齐样本的起点抛出 EmptyTemporalSampleError，然后重新抽取位置。完整迁移必须处理这一行为，并说明它与本框架有限 epoch、确定性 validation 的关系；不能悄悄沿用旧 forward 过滤，也不能宣称随机重采样仍保证每个原始起点都对应一条有效且唯一的窗口。特别短、连一块都取不出的 episode，不能继续用旧路径“重复末端补出一块”冒充 SWAM 的有效完整块。

读取时选多少个真实完整块，与随后为训练张量补齐长度是两个环节。需保留补齐位置的有效性 mask；现有最近 block 补齐与 SWAM batch 固定长度补齐存在差异，不能将其中一个当作另一个已经迁移。caption 按实际窗口的视频位置选择，不将后续 padding 计入覆盖数。

### 8.3 必须修改与可以复用的位置

| 位置 | 后续工作 |
| --- | --- |
| `frame_dataset.py::_candidate_ranges / locate / plan_read_window` | 更新逐帧候选入口及双向分块计划，区分原始抽中 t 与实际读取的第一帧；不再以旧 forward horizon 直接判断候选合法性。 |
| `readers/lerobot.py::FrameWindow / read_frame_window` | 支持实际窗口早于 t 开始；按选中的块对齐读取 state/action/video 与标签，保留原始 t 的身份及实际采样位置；不能把负的相对偏移再次当成前向长度。 |
| `frame_dataset.py::ActionFrameDataset / VideoFrameDataset.make_sample` | 使用新窗口，适配 real interval、block 数和有效性 mask；沿用各自 Geometry、processor、history、conditioning 和 token layout。 |
| `frame_dataset.py::_caption` | 复用工作区已实现的连续段覆盖选择，覆盖位置以新窗口的实际视频索引为准，重新核对原点与 stride 的假设。 |
| `datasets/validation.py`、split 与固定 epoch loader | 验证候选集和无效起点处理，确保验证确定、DP 不失步、Dataset 长度与 epoch 步数一致；不能直接保留旧候选数量断言。 |
| recipe、数据契约与测试 | 明确配置中的最大块数/块跨度和既有 future_frames 的关系；更新候选计数、左右扩展、短窗口、target_offset、task 边界、caption 和 epoch 验证。 |

网络的 causal mask、policy 模式、loss 以及固定 samples_per_pack 的语义继续复用。新窗口按实际时间排序后再进入既有训练，不因向前取到数据而变成双向 attention。

### 8.4 首要参考代码

以下路径相对于 `/mnt/sfs_turbo/zheng/Veomni-Wam/manipulation/SWAM/`：

- `rlinf/data/datasets/dreamzero/sampling_strategy.py`：`sample_video_indices`、`sample_action_indices`、`sample_state_indices`、`sample_temporal_indices`、`require_multi_anchor_temporal_indices`。窗口扩展、末端处理和跨模态一致性应整体阅读。
- `rlinf/data/datasets/dreamzero/dreamzero.py`：`_read_episode_language_labels`、`_temporal_offsets_for_frame`、`_materialize_parquet_sample`、`__getitem__`。核对原始标签、相对/绝对索引转换、文本选择发生时机和无效起点重采样。
- `rlinf/data/datasets/dreamzero/dense_subtasks.py::DenseSubtaskIndex.resolve`：对最终实际视频索引选择 subtask 文本。
- `rlinf/data/datasets/dreamzero/dreamzero.py::DreamZeroCollator.collate_batch` 与 `mixed.py` 的 temporal_pad_lengths 构造：核对选窗之后的 batch 补齐及有效性 mask。

## 9. 已 review 的第一阶段：独立窗口索引采集

在现有 `readers/lerobot.py` 中增加 `sample_multi_anchor_indices`，只负责原始 t 两侧按块采集、原始 task 起点检查、视频末端补充/回退和 action/state 对齐索引。所有索引都是 episode 内绝对帧号，保留原始 t；无效窗口抛出 EmptyTemporalSampleError。接口与示例见[独立采集契约](frame_dataset.md#独立双向窗口索引采集)。

选窗与末端处理直接移植第 8.4 节 SWAM 的 `sample_video_indices`，保留原来的 `add_step_set`、`back_done`、`fwd_done` 结构；将固定 8 帧改为 Geometry 推导的每块采样点数，并沿用本框架绝对索引与 target_offset 接口。action/state 由同一组选中块生成，不重复执行选窗。

第一阶段仅完成独立采集函数，用户已逐段 review。第二阶段接入见第 10 节，最新验收与后续事项见第 11 节。

## 10. 正式接入：双向窗口、无效起点与 dense caption

- Action/Video 子类按帧枚举所有足够长 episode 的候选，包含末帧；最小块和 target_offset 的 eligibility 与自动 split 同步。采集上限改为 candidates.max_blocks（默认 4），与 mode/anchor_stride 统一配置和校验；每块跨度由各自 Geometry 推导，不再从 future_frames 换算。旧 frame_window/segmented 配置不变。
- FrameWindow.block_span 与 max_blocks 标识训练采集计划，past/future 只记录实际窗口跨度。read_frame_window 一次读取有界双向范围中的原始 task、完整 subtask 和 payload，调用已 review 的采集器；返回原始 t、实际 past/future、绝对视频位置及对齐数据，不建立全 episode payload 缓存。
- read_sample 根据最终视频采样点在连续 subtask 段中的覆盖选择 caption，并列取最早；dense 边界不限制选窗。原始检查与训练统一通过 Action/Video 子类，共用 max_blocks、Geometry 和窗口文本；基类不再直接实例化，也不再提供 past_frames/future_frames 配置。
- Dataset[index] 当前在读取失败后顺序尝试下一候选，不随机重采样；make_sample/read_sample 仍是指定起点的直接接口。candidate_index 记录实际采用的候选，不保证实际窗口唯一。
- validation 直接调用 make_sample，固定任务模式和完整 history，保持训练 RNG 不变；不随机换起点，也不再记录单独的重采样编号。DP 轮数及末轮 weight 规则复用，读取失败按既有评估错误同步处理。
- 较短样本仅返回真实完整块，复用可变长度 packing，不新增 SWAM batch 时间长度 padding；保留既有 mask。源数据检查脚本按实际窗口首帧和完整块验证 Parquet/NumPy action 与 PyAV 像素。

第二阶段历史验证（随机重采样版本）：两套 CPU 回归 502 passed、4 xfailed、1 failed；唯一失败是旧路径 test_getitem_skips_errors_but_make_sample_propagates 的日志计数，在本轮修改前也已复现。删除了用户已移除的采样器类型防御所对应的 9 个旧参数校验用例，配置入口的校验测试保留。随后补充了纯视频重采样和无效起点的有限 epoch/worker 覆盖，结果保存在产物日志。

三个真实数据根目录共 63 个 Action（offset=0/3）与 Video 窗口，对照原 SWAM 视频索引、独立逐段 caption 投票和 NumPy block action 编码；其中 42 个窗口早于原始 t 开始，8 次请求发生重采样。18 个边界窗口另用 PyAV 独立解码，像素完全一致，action 最大绝对误差约 1.19e-7。这些真实窗口只有一个 subtask 段，跨段与并列由合成回归覆盖。审查结论 safe，审查者另以 mock 源帧值检查 12,000 组随机参数的单次读取与索引对齐。

产物位于仓库外 `/mnt/sfs_turbo/zheng/Veomni-Wam/artifacts/multi-anchor-integration-20261125/`。未运行真实 VAE/完整模型多卡训练；Dataset RNG 保存恢复不改变 Trainer 的既有 restart-data checkpoint 策略。以上为第二阶段历史验证记录，最新验收与后续事项见第 11 节。

## 11. 本轮验收与后续事项（2026-11-26）

- 本地 4 个数据集、102 个 episode，在相同 geometry 和 subtask 连续段输入下，对照原始 SWAM 采集函数及 resolver：104,934 个有效窗口、3,147 个跨 subtask 窗口，索引和 caption 均无差异。SWAM 的标注质量过滤、可选文本拼接和 episode 粒度分支不在等价性结论内。
- 保存 8 条经过真实 Dataset、Cosmos tokenizer 和原生 DataLoader/Collator 的样本、prompt、张量及白底预览；原始视频张量共 224 帧、324 张相机帧，经独立解码逐像素核验，最大误差为 0。产物位于仓库外 `/mnt/sfs_turbo/zheng/Veomni-Wam/artifacts/subtask-swam-review-20261126/`。未运行真实 VAE 或模型训练。
- CPU 数据回归为 509 passed、4 xfailed、1 failed；唯一失败是旧 pipeline 的日志重复计数断言，已在未修改的 HEAD 快照复现。
- 用户确认保留当前递归顺序跳过逻辑，连续失败达到递归上限后终止属于已接受行为。
- **已修复验证候选筛选：** 验证器初始化时调用 build_validation_indices，用 read_sample(video=False) 排除无法成块的 anchor，仅保存原始有效索引；按有效数量重新计算轮数后再分片。后续验证复用这份列表，其他读取或标注错误仍传播并同步到各 rank；不重采样、不缓存样本张量。

## 12. 最新 caption 规则：按时序拼接 subtask

用户确认继续允许窗口跨 dense subtask，保留原来的多数票 `_caption`，新增独立 `_concat_subtask_caption`。统一配置 data.sample.subtask_mode，取值 task / subtask_concat / subtask_single，默认 subtask_concat。task 不读取 subtask 标签或文本表，只使用原始 t 的 task 文本；subtask_concat 使用本节拼接规则；subtask_single 使用第 13 节采样规则和原始 t 的 subtask 文本。缺少或关闭 subtask 列时，两种 subtask 模式回退到 task。原多数票函数保留为第 11 节的对照实现，不再作为训练模式。

新函数按窗口内完整逐帧标签划连续段；首尾覆盖至少 window.block_span 个原始帧间隔才保留，中间段全部保留。末端观测不计时长，也不单独形成一个段。文本去掉首尾空白，末字符是字母或数字时补 `.`，已有标点则保留，再按时序用空格拼接，非连续同 ID 不去重；所有段被过滤或未使用 subtask 时退回 task。

样本新增 subtask_indices 列表记录最终文本的 ID 顺序；仅保留一个 ID 时 subtask_index 为该整数，多段或 task 回退时为 None。第 11 节真实数据 caption 等价性结论对应旧多数票模式，不代表新拼接模式仍与 SWAM resolver 等价。

## 13. 可选块起点 subtask 匹配

用户确认采用 SWAM/DreamZero 的块起点检查粒度，新增与原采集函数并列的 sample_multi_anchor_indices_with_subtask，复用原双向采集循环。data.sample.subtask_mode=subtask_single 时，候选块起点的 task 或 subtask 与原始 t 不同，就停止该侧扩展。原始 t 所在块内部及末端观测仍可跨 subtask，不增加完整连续段裁剪。

reader 使用同一次有界读取中的 subtask 标签，不增加 Parquet 读取；缺少或关闭 subtask 列时沿用 task 采样。该模式同时固定使用原始 t 的 subtask 文本，不再独立组合采样匹配和文本拼接；候选枚举和无效起点处理沿用现有规则。
