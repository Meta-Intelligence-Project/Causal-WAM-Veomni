# 按帧采样与 action/video 样本处理

新旧数据类保持独立。`LeRobotFrameDataset` 是抽象公共基类，管理前缀索引、读取计划、按帧标签、公共 history 和 RNG 状态；`ActionFrameDataset` 和 `VideoFrameDataset` 分别增加动作与纯视频的双向完整块采集及训练样本字段。旧 `LeRobotDataset → ActionDataset / VideoDataset` 的索引和区间标注规则保持不变。

训练 dataloader 支持原有 token 预算组包，也支持 `samples_per_pack` 固定样本数的有限 epoch。模型验证在 `full_history=True` 时复用现有 Validator；`build_dataset` 和 `iter_frame_samples` 也可检查样本。完整权重与真实数据下的 VAE、loss 和梯度对照尚待联调。

## 类与索引

`LeRobotFrameDataset` 直接继承 PyTorch Dataset。初始化读取 metadata、相机配置、文本表和 source FPS，不读取帧 payload、不展开逐窗口表。前缀数量只随 episode 数量增长，`locate(index)` 用二分查找得到元数据行号和当前帧 t。调用方传入 `[0, len(dataset))` 内的整数候选编号，`locate` 只负责定位。

数据检查和训练统一使用 ActionFrameDataset / VideoFrameDataset：read_sample 返回原始 payload，make_sample 继续进行训练预处理；两者使用相同的 SWAM multi-anchor 窗口规则：

- `candidates` 统一配置 `mode=frame_anchor`、候选间隔 `anchor_stride` 和两侧合计的块数上限 `max_blocks`。max_blocks 默认 4，必须是正整数。Action/Video 共用这些默认值和校验。Dataset 不再提供 past_frames/future_frames 或单独的固定帧窗配置。
- Action 的块跨度 S=`actions_per_block`；Video 的 S=`最终 video_stride × temporal_compression × latent_block_size`。S 由各自 geometry 推导，最多采集 K 个真实完整块，不再把帧数预算换算为块数。
- 能容纳至少一个完整块及末端观测的 episode，按 `0, anchor_stride, … < L` 枚举所有起点，包括末帧。Action 同时考虑 target_offset 的可用范围；过短 episode 不产生候选。自动 split 使用相同的最小块要求。
- 从原始 t 开始，依次尝试 `t、t-S、t+S、t-2S、t+2S…`；某侧越界或块起点的原始 task 与 t 不同，就停止该侧。默认 subtask_concat 不使用 dense subtask 选窗；subtask_single 时，块起点的 subtask 与 t 不同也会停止该侧，块内部和末端观测仍允许跨段。
- 原始起点不是有效窗口的保证。`read_sample/make_sample` 对无效起点抛出 EmptyTemporalSampleError；当前 `dataset[index]` 在读取失败后顺序尝试下一候选，越过末尾时回到 0，不随机重采样。

迁移旧 `frame_anchor` 配置时，删除 candidates 中的 past_frames/future_frames，改设 candidates.max_blocks。Action 的 `future_frames=128, actions_per_block=32` 对应 `max_blocks=4`；Video 需按原来的 S 换算 `ceil(128/S)`，S=8/16 时分别为 16/8 块。固定块数后，改变 video_stride 会改变时间跨度，不改变块数上限。非 `frame_anchor` 的 frame_window/segmented 配置与采样逻辑保持原样；混合数据源分别配置各自的 candidates，max_blocks 只用于 frame_anchor；source.candidates.max_blocks 可以覆盖公共上限。

候选数与旧 forward 路径不同。有限 epoch 遍历的是请求的候选槽位；跳过无效起点可能重复某个实际窗口，不承诺每个原始起点对应唯一有效窗口。返回的 candidate_index 是实际采用的候选。

## 直接读取与样本处理

```python
from veomni.data.causal_cosmos3 import ActionFrameDataset, ActionProcessor, FrameCandidateConfig

dataset = ActionFrameDataset(
    "/path/to/lerobot_v3",
    source="robotwin",
    viewpoint="concat_view",
    candidates=FrameCandidateConfig(max_blocks=4, anchor_stride=1),
    processor=ActionProcessor("/path/to/measured_block_statistics.json"),
    tokenize=tokenize_prompt,  # 实际训练 tokenizer 的回调
    mode="policy",
)
raw = dataset.read_sample(0)
sample = dataset.make_sample(0, mode="policy", history=9)
```

`processor=None` 时可以使用 `read_sample` 检查原始 action/state，不能生成处理后样本。`ActionProcessor(None)` 可用于未归一化的数据检查；配置构建 action 样本要求实测统计量。`tokenize` 在原始检查中可选，处理后样本要求显式回调。

## 接口职责

下表中的实现均由框架提供；调用方可直接使用原始读取或处理后的样本接口，无须实现新的模型输入契约。

| 接口 | 输入输出及调用方 |
| --- | --- |
| [`LeRobotFrameDataset._candidate_ranges(episode_lengths)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L161) | 计算每个 episode 的首 anchor 和数量，由初始化调用。基类计算步进起点；action/video 子类过滤不足一个完整块的 episode，其余按帧枚举。 |
| [`locate(index)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L203) | 返回元数据行号和 t，不读取 payload；由读取计划及异常上下文调用。 |
| [`plan_read_window(index)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L214) | 子类必须实现，返回含块 geometry 和 max_blocks 的 FrameWindow。可调用基类实现取得存储计划，FrameWindow 允许直接修改字段；Action 设置动作块的 block_span 和 max_blocks，Video 按源 FPS 选择 stride 并推导 block_span。由 read_sample 调用，具体窗口由 reader 选定。 |
| [`_concat_subtask_caption(task_id, subtask_ids, block_span)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L248) | 框架提供，由 read_sample 在 subtask_mode=subtask_concat 时调用。block_span 单位为原始帧间隔，subtask_ids 包含末端观测标签或为 None；返回 caption、整数 task ID、按时序保留的 subtask ID 列表和 caption_kind。没有可用 subtask 时返回 task 文本及空列表。 |
| [`read_sample(index, *, video=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L281) | Action/Video 共用的原始 payload 接口，返回与训练相同的完整块、时间 mask、时间戳和窗口 caption。默认遵循实例 video 设置，也可显式跳过解码；不单独配置检查窗口。 |
| [`prepare_sample(sample)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L349) | 只生成 prompt 并按需 tokenize；元数据已由 read_sample 一次写入。由基类及两个子类的 make_sample 调用。 |
| [`_decode_sample_video(sample, real_intervals)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L339) | 基类提供处理后视频解码。子类传入实际窗口时间轴和有效帧间隔数，查询按 window.video_stride 抽取，写回 sample.video。由两个子类的 make_sample 调用。 |
| [`select_history(num_latents, *, explicit=None, rng=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L386) | 基类提供 latent 帧单位的 history 选择。显式值和 full_history 不抽随机数，否则使用传入 RNG 或按需初始化实例 RNG；由两个子类的 make_sample 调用。 |
| [`state_dict()` / `load_state_dict(state)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L397) | 基类统一保存、恢复实例 RNG，返回含 rng 的字典；由外层 Dataset wrapper 转发调用。 |
| [`read_frame_window(window, ..., block_size=None)`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L390) | 由 read_sample 调用。有 block_span 时，一次读取有界范围的标签和 payload，按原始 task 选块并返回实际数据；底层 reader 仍可检查显式 FrameWindow 的边界 mask，不做动作编码。 |
| [`ActionFrameDataset.make_sample(index, *, mode=None, history=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L497) | 产生 block state/action、prompt/token IDs、conditioning、latent 索引和 token 数。由 `dataset[index]`、数据检查与模型验证调用。 |
| [`VideoFrameDataset.make_sample(index, *, history=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L591) | 产生纯视频 block 对齐样本、history、首 latent 条件、有效 latent 数及 token 数。由 `dataset[index]`、数据检查与模型验证调用。 |
| [`build_validation_indices(dataset)`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L60) | 框架提供，在验证器初始化时调用一次；按 source/root 和原始候选顺序返回 (dataset, index) 列表。frame 候选用 read_sample(index, video=False) 筛选，只过滤 EmptyTemporalSampleError；旧窗口候选直接保留。没有有效候选时抛 ValueError，其他读取或标注错误传播。不保存样本张量。 |
| [`iter_frame_samples(dataset, rank, world_size, *, raw=False)`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L42) | 有限枚举样本，raw=True 调用 `read_sample`，否则调用 Dataset；用于数据检查，不是模型验证循环。 |

相关实现见 [帧数据类](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py)、[block 编码](../../../veomni/data/causal_cosmos3/transforms/processing.py) 和 [数据枚举](../../../veomni/data/causal_cosmos3/datasets/validation.py)。

一次动作样本的调用顺序：

1. plan_read_window 保留原始 t、max_blocks 和 block_span；read_frame_window 一次投影读取 t 两侧至多 K 个块及 offset 目标所需的有界范围，不保留整个 episode 的 payload 缓存。
2. 同一批已读原始 task 标签交给 sample_multi_anchor_indices；subtask_mode=subtask_single 且启用了 subtask 列时，read_sample 显式传入 subtask_column，reader 复用同批标签并改用 sample_multi_anchor_indices_with_subtask。同时确定完整块、视频末端及对齐的 action/state 索引。reader 返回实际窗口，window.anchor 始终是原始 t，实际首帧是 anchor-past，video_frame_indices 是 episode 内绝对帧号。
3. 根据 data.sample.subtask_mode 生成文本：task 不读取 subtask 列和文本表，使用原始 t 的 task 文本；subtask_concat 调用 `_concat_subtask_caption(task_id, subtask_ids, block_span)` 按时序拼接；subtask_single 使用原始 t 的 subtask 文本。subtask 列关闭或 auto 未找到时，两种 subtask 模式均回退到 task。原始列名仍由 FrameLabelConfig 配置。
4. 复用 ActionProcessor 按各 block 起始 state 编码动作、归一化，再生成 prompt/token IDs 和 Geometry token 数。

## 模型验证

[`iter_validation_batches()`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L76) 接受 ActionFrameDataset 和 VideoFrameDataset，要求 sample.full_history=True。自动划分仍以 episode 为单位，验证先筛选 held-out episode 的有效候选，再进行 DP 分片，不受训练 source 权重影响。

验证器初始化时调用 build_validation_indices；各 rank 按相同顺序扫描 frame 候选，调用 read_sample(index, video=False)，只捕获 EmptyTemporalSampleError 并排除该 anchor。只保存数据集引用和原始 index，不缓存视频或样本张量。筛选会增加一次 Parquet 读取，不解码视频、不消费 Dataset RNG；其他错误在初始化时经跨 rank 同步后抛出。全部候选被过滤时直接报错。旧窗口数据集不进行这次读取筛选。

CausalCosmos3Validator 缓存这份列表，以有效数量计算 window_count 和每个 mode 的轮数；每次验证共用该列表，不重新筛选。独立调用 iter_validation_batches 时，未提供 indices 才会先构建列表。过滤发生在分片之前，各卡正常验证自己的有效样本，尾部仍沿用重复样本且 weight=0 的补齐规则。

随后按以下顺序准备每条数据：

1. 固定当前任务模式，直接调用 frame 子类的 make_sample(index, mode=mode)；video 不传 mode，joint 按 FD、ID、policy 各遍历一次。未显式配置 sample.mode 时，Validator 从实际 Dataset 获取默认 mode，frame action 默认为 policy。
2. 复用 Dataset 的读取、窗口 caption、block 编码和完整 history。正式准备阶段只读取已分配的有效候选，保留其原始 index，不随机重采样，不初始化或推进训练 Dataset RNG。
3. 从返回样本生成 record：identity 包含 source/root、episode、candidate_index、caption、anchor、实际 past/future、max_blocks、video_stride 和实际长度。window 保存 Reader 返回的 FrameWindow；history_blocks 直接来自样本。固定完整 history 不生成 history_seed。
4. 按 identity 和 mode 派生独立的 time/video/action seeds，返回现有 batch/record/weight 协议。每个 pack 包含一个样本；各 DP rank 轮数相同，末轮重复样本的 weight=0。
5. 交给现有 [CausalCosmos3Validator.evaluate()](../../../veomni/trainer/causal_cosmos3.py#L418) 完成 VAE 编码、固定噪声准备、causal forward 和 loss 汇总。筛选后的 frame 读取或处理异常仍交由现有跨 rank 错误同步，不转换成空样本。预先过滤的 anchor 不进入 loss/count；原有 skipped_count 继续表示运行时整轮跳过的真实候选。

例如完整 128 步动作窗口按默认 geometry 产生 33 帧视频、9 个 latent，验证 history_blocks=9。较短窗口的 history 取实际完整块对应的 latent 数。模型输入与训练使用同一 make_sample 路径，旧 Dataset 的验证 identity、history_seed 和噪声 seeds 保持原有规则。

## 原始数据检查与训练

LeRobotFrameDataset 只提供公共实现，不能直接实例化。检查原始数据时调用 ActionFrameDataset / VideoFrameDataset.read_sample；不需要 tokenizer 或动作 processor，窗口仍由 candidates.max_blocks 和具体子类的 Geometry 决定，与训练和 validation 一致。FrameWindow.past/future 仅记录 reader 选定后的实际跨度，不是采样配置。

ActionFrameDataset / VideoFrameDataset 的 read_sample 已使用双向完整块。采集不足 K 块时返回真实块数，不把末端重复成新的有效块；所有保留的观测、视频和 action 时间位置均有效。有效性 mask 继续沿用。后续 geometry packing 按每条样本的实际长度工作，不把同一 pack 的样本补成相同时间长度。

原始 t 可能位于窗口中间或末端；VAE 首帧、首视频 latent 条件和 block action/state 参考使用实际窗口首帧。anchor_index 是原始 t 在最终观测时间轴中的位置，video_frame_indices 包含末端观测。state-derived action 的 target_offset 只应用一次，并严格检查目标边界。

## block 训练字段

每个 block 使用自己的起始 state。动作编码、归一化、有效 latent 数及 action/state latent 索引复用现有 `ActionProcessor`，不再使用单一 state[t] 编码整个窗口。

| 字段 | 含义 |
| --- | --- |
| `states / anchors / state_mask` | 每 block 一个归一化状态、未归一化参考状态和映射后的状态通道 mask。完整原始 state 保留在 raw 接口，不随处理后样本返回。 |
| `action / action_mask / action_valid_steps` | block 对齐的相对动作、有效通道和有效时间位置。 |
| `real_action_steps / num_valid_video_latents` | 有效原始动作数及参与视频监督的 latent 数。 |
| `state_action_indexes / state_latent_indexes / action_frame_ids` | block state 和 action 与视频 latent 的对应关系。 |
| `mode / history_blocks` | FD、ID、policy 或按权重抽取的 joint 模式；history_blocks 沿用现有命名，单位是 latent 帧。 |
| `condition_frame_indexes_vision / condition_frame_indexes_action` | policy 仅条件化首视频 latent，不将 action 设为条件。FD/ID 使用现有各自规则。 |
| `token_layout / token_length` | 文本、clean/target 视频两份、action 和 state 的实际 token 数。 |

当前 geometry 为 actions_per_block=32、video_stride=4、temporal_compression=4 时，完整 128 步窗口产生 33 帧视频、9 个视频 latent 和 4 个 block state。

统计量继续描述相对每个 block 起点的动作，必须匹配模板和 geometry。明确声明 `action_reference=current_frame` 的旧实验统计量会被拒绝；没有该声明的既有 block 统计量仍可使用。

`seed/rank/rng`、history 配置及校验、`_get_rng`、`select_history` 和 `state_dict/load_state_dict` 统一由基类管理，两个子类直接继承。Action 默认 full_history=True，Video 默认 False；history_range 默认 (1, 32)，RNG 的 seed/rank/worker 规则沿用原实现。有效起点显式传入 mode/history 不消耗 RNG；固定 mode 且 full_history=True 时，有效起点也不初始化或消耗 RNG。原始读取也不初始化或消耗 RNG，未初始化时所有帧数据类的状态统一为 `{"rng": None}`。这里的保存恢复指 Dataset.state_dict/load_state_dict；Trainer 继续使用既有 restart-data 策略，训练 checkpoint 恢复不承诺逐样本重放。

## 纯视频子类

`VideoFrameDataset` 继承原始读取和按帧标签，不读取 action/state 列，不依赖动作统计量。默认自动模式以 video_stride=1 构造读取计划，仅在基础 stride 为 1 时按 FPS 调整：source_fps > 16 时 stride=2，否则 stride=1。显式指定 `video_stride` 时使用固定间隔。

```python
from veomni.data.causal_cosmos3 import FrameCandidateConfig, VideoFrameDataset

dataset = VideoFrameDataset(
    "/path/to/lerobot_v3",
    source="robotwin",
    viewpoint="ego_view",
    candidates=FrameCandidateConfig(max_blocks=4, anchor_stride=1),
    latent_block_size=2,
    temporal_compression=4,
    tokenize=tokenize_prompt,
    full_history=False,
    history_range=(1, 32),
)
raw = dataset.read_sample(0)
sample = dataset.make_sample(0, history=9)
```

一次视频样本的调用顺序：

1. 根据 source FPS 或显式 stride 推导 S，K 直接取 max_blocks，一次投影读取有界范围内的 timestamp/task/subtask；不读 action/state。
2. 复用双向选块与末端处理，返回真实完整块和窗口 caption，按实际视频位置解码。
3. 写入 mode、latent block_size、history、首 latent 条件和有效 latent 数，生成 prompt/token IDs 及 video_token_layout。

VAE 首帧使用最终排序后的第一帧，可能早于原始 t。样本不包含 action/state 或动作条件字段。tokenize 在处理后样本中必需，在 raw 接口中可省略。

工厂在 `stage=video, candidates.mode=frame_anchor` 时构建此子类，接受旧 video 的 geometry 和 history 配置。实验配置仍写在 DiT recipe 的 `data` 段，例如：

```yaml
data:
  stage: video
  geometry:
    latent_block_size: 2
    temporal_compression: 4
    spatial_compression: 16
    patch_size: 2
  candidates:
    mode: frame_anchor
    max_blocks: 4
    anchor_stride: 1
  sample:
    full_history: false
    history_range: [1, 32]
```

纯视频检查同样调用 VideoFrameDataset.read_sample，与 make_sample 使用同一套块采集和 caption。自动 episode split 对 action/video 子类都排除无法容纳一个完整块及末端观测的 episode。

## 按帧标注

默认读取 t 的 `task_index → meta/tasks.parquet(task_index, task)`。info 声明 `subtask_index/subtask_id` 时，查 `meta/subtasks.parquet(subtask_index, subtask)`；没有 subtask 列或显式关闭时使用 task 文本。task_id 和 subtask_ids 直接转换为整数，均在当前数据根目录内按 ID 查表；未知 ID 报错，0 是有效 ID。`FrameLabelConfig` 支持配置列名和文本表，不再提供 task/subtask 的 episode 作用域配置。

JSONL 可以作为 ID→文本定义表，配置 `subtask_records_key` 等字段即可；新类不根据起止区间判断当前 subtask。只有区间标注的旧数据使用旧数据类。

采样和文本统一由 DiT 配置的 `data.sample.subtask_mode` 决定：

| 值 | 采样 | 文本 |
| --- | --- | --- |
| task | 仅检查原始 task 的块起点 | 使用原始 t 的 task 文本，不读取 subtask 列和文本表 |
| subtask_concat（默认） | 仅检查原始 task 的块起点，允许跨 subtask | 按下述规则拼接窗口内的 subtask 文本 |
| subtask_single | 同时检查 task/subtask 的块起点 | 使用原始 t 的 subtask 文本，不投票、不拼接 |

两种 subtask 模式遇到未启用或不存在的 subtask 列时，都回退到 task。显式指定的 subtask 列或文本表仍必须存在；task 模式不访问这些 subtask 定义。subtask_single 沿用 SWAM 的块起点检查，块内部和末端观测仍可能跨段，不按连续段裁剪整块。

subtask_concat 先按完整逐帧 subtask ID 划连续段，再由 read_sample 调用独立的 `_concat_subtask_caption(task_id, subtask_ids, block_span)`：

- 首尾段在实际窗口内覆盖至少一个 block 才保留；中间段无论多短都保留，包括被视频 stride 跳过的短段。
- 覆盖长度以原始帧间隔计，阈值使用 window.block_span，不要求 subtask 边界与 block 边界对齐。末端观测不增加时长，也不单独形成一个 subtask 段。
- 保留的文本去掉首尾空白，末字符是字母或数字时补 `.`，已有标点则保留，再按时序用空格拼接；同一 ID 在非连续段出现时分别保留，不去重。
- 所有段都被过滤时，使用原始 t 的 task 文本。

例如每块跨度为 8，窗口内各段覆盖 `3 → 26 → 3` 个原始帧间隔时，仅保留中间段；覆盖 `9 → 1 → 14` 时，三段均保留。caption 先于 prompt/tokenize/token_layout 确定，原始检查、训练和 validation 共用规则。sample.subtask_indices 按时序保存参与最终文本的 ID，task 回退时为 []；只有一个 ID 时 sample.subtask_index 为该整数，多段或 task 回退时为 None。

```yaml
data:
  sample:
    subtask_mode: subtask_concat  # task / subtask_concat / subtask_single
```

`source.labels` 只配置 task/subtask 的列名和文本表映射。工厂只将 subtask_mode 传给 frame_anchor 类，混用旧窗口数据源时也不会改变旧类的 caption 逻辑。直接构造 ActionFrameDataset 或 VideoFrameDataset 时，传入同名参数。

原来的 [`_caption(task_id, subtask_ids, video_offsets)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L229) 保留为多数票对照函数，不再作为训练配置选项。旧 frame_window/segmented 路径保持原样。

## 配置与检查

[DiT 配置](../../../configs/dit/causal_cosmos3_edge_action_frame.yaml) 的 `data` 段设置 stage、geometry、candidates 和 sample，采用 `anchor_stride=1 / max_blocks=4 / mode=policy`。它通过 `data.train_path` 引用[数据源配置](../../../configs/data/frame_datasets.yaml)，后者只负责 sampling、可选 validation 和 sources（包含路径、统计量及标签列配置）。两份 YAML 经现有入口组装成 Dataset 配置，再传给 `build_dataset(config, tokenize)`；训练 dataloader 复用现有组包路径，不再按 sample_kind 拒绝帧样本。

未配置 `samples_per_pack` 时，多源混合的长度回调统一返回样本的真实 `token_length`；`source_weighted` 仍按样本权重混合，不按 token 数调整权重。`ActionBatchingStrategy.put_item` 接收帧样本，按同一 token 预算组包；超预算单样本保留单独成包的行为。`collate_action_pack` 将每个 pack 包装为 `{"samples": [...]}`。模型验证要求 full_history=True，并复用现有 episode 划分、验证回调和模型执行链路。

### 固定样本数与有限 epoch

在 DiT recipe 中配置：

```yaml
data:
  samples_per_pack: 8
  max_seq_len: 54000  # 配置 samples_per_pack 后失效
train:
  micro_batch_size: 1  # 一个 pack
  grad_accum_iter: 2  # 每 rank 每次更新两个 pack
  max_steps: null
  num_train_epochs: 3
```

数据源 YAML 仍只配置来源与采样，使用 `sampling: sample_uniform`，所有 source 的有效候选模式必须为 `frame_anchor`。action/video 均可使用；不配置该字段时保留原有动态组包和 `source_weighted` 路径。

设 B 为每 pack 样本数，D 为 DP rank 数，A 为梯度累积次数。框架的 `global_batch_size=D*A` 仍以 pack 计数；实际每次更新为 `B*D*A` 条样本。按 episode 划分验证集后，N 个训练候选对应 `ceil(N/(B*D*A))` 步。sampler 用 seed+epoch 打乱，先覆盖全部请求候选槽位，再重复当前排列的前缀补齐完整更新；Dataset 跳过无效起点后返回实际样本，不减少 sampler 的槽位数。重复样本正常参与 loss，训练末尾没有额外零权重或 dummy 数据。即使 N 小于一次更新，也会补齐一次更新。

`max_seq_len` 和 `dyn_bsz_buffer_size` 不参与此路径，不会因为 token 数拆包或截断。`max_steps: null` 完整遍历；显式正数只限制每个 epoch 的步数，LR scheduler 沿用有效步数乘 epoch 数。validation 保持原有单样本 pack 及零权重补齐规则，不使用训练的 B。

这些接口均由框架提供，无须模型实现新的钩子：

| 接口 | 调用方与作用 |
| --- | --- |
| [`get_samples_per_pack(config)`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L37) | Trainer 的 [`_setup`](../../../veomni/trainer/causal_cosmos3.py#L124) 和 [`build_action_dataloader`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L301) 调用，校验正整数及有效采样模式；未配置返回 None。 |
| [`FrameEpochSampler`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L52) | `build_frame_epoch_dataloader` 构造，只调整父 sampler 的 num_samples/total_size，使每 rank 样本数整除 B*A；shuffle、分片、set_epoch 和状态接口继承 StatefulDistributedSampler。 |
| [`collate_frame_step(samples, ...)`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L63) | 原生 loader 调用，将一次更新的 B*A 条样本切成 A 个 pack，再逐 pack 调用传入的 collator。默认返回 list[packs]；训练返回 list[dict(samples=...)]。 |
| [`build_frame_epoch_dataloader(...)`](../../../veomni/data/causal_cosmos3/datasets/pipeline.py#L70) | Trainer 的 [`_build_dataloader`](../../../veomni/trainer/causal_cosmos3.py#L227) 和 `build_action_dataloader` 共用，返回有限的 DistributedDataloader；每次迭代返回一次更新所需的全部 pack。 |

一次更新沿用 `Dataset → collate_frame_step → collate_action_pack → Condition / geometry packing → causal forward / loss → gradient accumulation → optimizer`。`set_epoch` 由现有 DiT 训练循环调用；checkpoint 沿用 Cosmos3 的恢复进度与 RNG、重新开始数据流策略。CPU 测试覆盖多 rank 全候选遍历、短数据集补齐、epoch shuffle、action/video、多 worker、配置覆盖及真实 Trainer 的验证划分和步数。

[CPU 测试](../../../tests/data/test_causal_cosmos3_frame_dataset.py) 检查原始读取、标签、索引内存，并核对逐帧候选、双向块、caption 投票，并对照相同实际窗口的旧 block 编码、时间戳、conditioning 和 token 数。action/video 两种阶段均经过真实 native dataloader，覆盖 sample_uniform/source_weighted、每步 microbatch 数、token 预算及超预算单样本；验证测试覆盖完整 history、单次 payload 读取、任务模式、固定 seeds、RNG 状态保持、DP 等长轮次及零权重补齐；另以固定 latent 替代 VAE 权重，实际运行 Condition、packing、tiny Cosmos3Model forward、loss 和 Validator，检查 action/video 默认 mode 与重复评估结果。

[源数据检查脚本](../../../scripts/causal_cosmos3/validate_frame_dataset.py) 用独立 Parquet/NumPy block 编码及 PyAV 解码核对样本，使用显式检查 tokenizer，不加载模型或 checkpoint：

```bash
PYTHONPATH=. python scripts/causal_cosmos3/validate_frame_dataset.py \
  --root /path/to/lerobot_v3 --source robotwin --viewpoint concat_view \
  --max-blocks 4 --report /tmp/frame-validation.json
```

从 state 生成目标时传 `--columns '{"target":"state_unified","target_mask":"mask_state","target_offset":1}'`。`--labels` 接受 FrameLabelConfig JSON；`--statistics` 可指定实测 block 统计量。多 worker 检查要求系统允许本地 socket 通信。

## 独立双向窗口索引采集

[`sample_multi_anchor_indices`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L566) 是框架提供的确定性索引采集函数，由 read_frame_window 在同一次 payload 读取后调用，也可独立调用。Dataset/训练已通过该 reader 使用双向窗口。

选窗循环和末端回退直接改编自 SWAM 的 `sampling_strategy.py::sample_video_indices`，保留其先向前、再向后扩展的结构；适配本框架的每块视频采样点数、绝对帧索引和 action target_offset。action/state 从同一组已选块生成。

输入 anchor、length 均以 episode 内原始帧为单位，`max_blocks` 是块数上限。`block_span` 对 Action 取 `Geometry.actions_per_block`，对 Video 取最终 `video_stride × temporal_compression × latent_block_size`；不在采集函数内假定每块固定 8 个视频采样点。

`task_labels[frame]` 直接访问已有原始 task ID 或 language 标签，可传数组或按 episode 帧号索引的缓存。只有已确认整个 episode 的原始 task 相同时，调用方才可显式传 None 跳过查询。subtask 不替代原始 task；需要额外比较时，通过下述独立入口传入 subtask_labels。采集函数不读取 Parquet 或解码视频。

采集依次尝试 t、t-S、t+S、t-2S、t+2S，S=block_span。某侧越界或块起点 task 与原始 t 不同，即停止该侧；不增加整块逐帧同 task 检查。完整视频块先采 `[0, video_stride, …, S-video_stride]`，排序后再追加末端观测。末端不存在时，回退最后一块并保留其首帧作为前一块的末端；连一块都不能保留时抛出 EmptyTemporalSampleError，不补齐出有效块。

默认 target_offset=0 返回 action/state 索引；action j 的目标行是 j+target_offset，必须与相邻观测同时合法。只选择一次块起点，再统一生成 video/action/state 索引；目标行越界时抛出相同异常。`target_offset=None` 表示纯视频采集，action_indices/state_indices 都为 None，不依赖 action 列或 processor。该函数不替代读取端 ActionColumns 对目标列与 mask 的配置校验。

返回的 MultiAnchorIndices 包含原始 anchor、按时间排序的 block_starts、video_indices 和 action_indices，state_indices 为 action 路径的块起点。所有索引均为 episode 内绝对帧号，调用方不能再次加 anchor。函数不消耗 RNG、不重采样、不 padding；无效起点抛出 EmptyTemporalSampleError。

```python
from veomni.data.causal_cosmos3.readers.lerobot import sample_multi_anchor_indices

indices = sample_multi_anchor_indices(
    64, 200,
    block_span=32,
    video_stride=4,
    max_blocks=4,
    task_labels=None,  # 调用方已确认整个 episode 的原始 task 相同
)
assert indices.anchor == 64
assert indices.block_starts.tolist() == [0, 32, 64, 96]
assert indices.video_indices.tolist() == list(range(0, 129, 4))
assert indices.action_indices.tolist() == list(range(128))
```

### 同时匹配块起点的 subtask

[`sample_multi_anchor_indices_with_subtask`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L676) 是框架提供的并列入口，复用原函数的双向循环、末端处理及 action/state 对齐，只增加候选块起点与原始 t 的 subtask 比较。调用者传入相同的 anchor、length、block_span、video_stride、max_blocks、task_labels、target_offset，另传可按 episode 绝对帧号访问的 subtask_labels；返回相同的 MultiAnchorIndices，无可用块时仍抛 EmptyTemporalSampleError。

Dataset 通过 data.sample.subtask_mode=subtask_single 启用：

```yaml
data:
  sample:
    subtask_mode: subtask_single
```

read_sample 将已解析的 subtask 列名传给 read_frame_window；reader 从同一次有界投影读取构建标签索引，选择新入口，不增加 Parquet 读取。缺少或显式关闭 subtask 列时，仍按 task 采样。该模式同时固定使用原始 t 的 subtask 文本；不再单独配置采样匹配和文本拼接。

该模式与 SWAM/DreamZero 的 language 块起点检查粒度一致，不检查块内部和末端观测的 subtask，也不保证窗口位于同一连续段。例如跨度为 16，subtask 0 在第 20 帧切为 1，t=0 时可以保留起点 0、16 的两个块，窗口为 [0,32]，在起点 32 停止扩展。候选数量和无效起点的处理沿用现有规则。

现有 [CPU 测试](../../../tests/data/test_causal_cosmos3_frame_dataset.py) 覆盖扩展顺序、task 边界、末端回退、末帧起点、无效短样本、不同块跨度/stride、state 目标 offset 和纯视频采集。Dataset 测试覆盖实际读取、caption、RNG 恢复、validation 确定性及有限 epoch。
