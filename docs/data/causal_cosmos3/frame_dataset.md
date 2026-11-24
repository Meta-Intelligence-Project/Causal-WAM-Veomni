# 按帧采样与 action/video 样本处理

新旧数据类保持独立。`LeRobotFrameDataset` 管理前缀索引、原始窗口、按帧标签、公共 history 和 RNG 状态；`ActionFrameDataset` 和 `VideoFrameDataset` 分别增加动作与纯视频的前向候选约束、block 对齐及训练样本字段。旧 `LeRobotDataset → ActionDataset / VideoDataset` 的索引和区间标注规则保持不变。

训练 dataloader 已允许 `frame_anchor` 通过现有动态组包和 collate。模型验证在 `full_history=True` 时复用现有 Validator；`build_dataset` 和 `iter_frame_samples` 也可检查样本。完整权重与真实数据下的 VAE、loss 和梯度对照尚待联调。

## 类与索引

`LeRobotFrameDataset` 直接继承 PyTorch Dataset。初始化读取 metadata、相机配置、文本表和 source FPS，不读取帧 payload、不展开逐窗口表。前缀数量只随 episode 数量增长，`locate(index)` 用二分查找得到元数据行号和当前帧 t。调用方传入 `[0, len(dataset))` 内的整数候选编号，`locate` 只负责定位。

基类用于原始视频和标签检查，每个非空 episode 的每帧都可成为 anchor，包括最后一帧和单帧 episode。两个训练样本子类只接受 `past_frames=0` 和至少包含两个观测的 episode：

- `ActionFrameDataset` 要求 `future_frames` 是 `actions_per_block` 的正整数倍；`VideoFrameDataset` 接受任意正整数未来帧间隔数，在处理时补齐。
- 长 episode 只保留能容纳完整未来窗口的起点，按 `anchor_stride` 前进，不额外插入右对齐尾窗。
- 短 episode 保留一个合法起点，在样本处理时补到最近的完整 block。
- 单帧 episode 不产生候选；动作子类还根据 `target_offset` 同时限定起点和终点。

这些规则与有限长度的旧 `frame_window + start_stride` 一致，但候选索引仍为前缀数组，没有逐窗口展开。

## 直接读取与样本处理

```python
from veomni.data.causal_cosmos3 import ActionFrameDataset, ActionProcessor, FrameCandidateConfig

dataset = ActionFrameDataset(
    "/path/to/lerobot_v3",
    source="robotwin",
    viewpoint="concat_view",
    candidates=FrameCandidateConfig(past_frames=0, future_frames=128, anchor_stride=1),
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
| [`LeRobotFrameDataset._candidate_ranges(lengths)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L154) | 计算每个 episode 的首 anchor 和数量，由初始化调用。基类允许所有帧；action/video 子类覆盖为合法前向起点。 |
| [`_forward_candidate_counts(available)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L158) | 基类提供公共前向窗口数量公式。两个子类分别计算合法帧间隔数后调用；长 episode 保留完整窗口，短 episode 保留一个起点。 |
| [`locate(index)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L210) | 返回元数据行号和 t，不读取 payload；由读取计划及异常上下文调用。 |
| [`plan_read_window(index)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L220) | 返回 `FrameWindow` 物理读取计划；动作子类增加动作列，video 子类按源 FPS 选择抽帧间隔。由 `read_sample` 调用。 |
| [`read_sample(index, *, video=None, block_size=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L254) | 默认返回原始窗口、时间 mask、时间戳和 t 的文本。动作 `make_sample` 传入 block_size，由首次读取直接完成 block 补齐。默认遵循实例 video 设置，也可显式跳过解码。 |
| [`prepare_sample(sample)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L299) | 只生成 prompt 并按需 tokenize；元数据已由 read_sample 一次写入。由基类及两个子类的 make_sample 调用。 |
| [`_decode_sample_video(sample, real_intervals)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L289) | 基类提供处理后视频解码。子类传入补齐后的时间轴和有效帧间隔数；查询按 window.video_stride 抽取并截到真实终点，写回 sample.video。由两个子类的 make_sample 调用。 |
| [`select_history(num_latents, *, explicit=None, rng=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L334) | 基类提供 latent 帧单位的 history 选择。显式值和 full_history 不抽随机数，否则使用传入 RNG 或按需初始化实例 RNG；由两个子类的 make_sample 调用。 |
| [`state_dict()` / `load_state_dict(state)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L345) | 基类统一保存、恢复实例 RNG，返回含 rng 的字典；由外层 Dataset wrapper 转发调用。 |
| [`read_frame_window(window, ..., block_size=None)`](../../../veomni/data/causal_cosmos3/readers/lerobot.py#L389) | 由 read_sample 调用，读取前确定有效终点和最终长度。block_size 为前向 action 的源帧间隔数；指定后一次完成 action/state 补齐、时间轴和 mask，不做动作编码。 |
| [`ActionFrameDataset.make_sample(index, *, mode=None, history=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L450) | 产生 block state/action、prompt/token IDs、conditioning、latent 索引和 token 数。由 `dataset[index]`、数据检查与模型验证调用。 |
| [`VideoFrameDataset.make_sample(index, *, history=None)`](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py#L547) | 产生纯视频 block 对齐样本、history、首 latent 条件、有效 latent 数及 token 数。由 `dataset[index]`、数据检查与模型验证调用。 |
| [`iter_frame_samples(dataset, rank, world_size, *, raw=False)`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L41) | 有限枚举样本，raw=True 调用 `read_sample`，否则调用 Dataset；用于数据检查，不是模型验证循环。 |

相关实现见 [帧数据类](../../../veomni/data/causal_cosmos3/datasets/frame_dataset.py)、[block 编码](../../../veomni/data/causal_cosmos3/transforms/processing.py) 和 [数据枚举](../../../veomni/data/causal_cosmos3/datasets/validation.py)。

一次动作样本的调用顺序：

1. `read_sample(video=False, block_size=geometry.actions_per_block)` 先根据 episode 长度和 target_offset 算出有效 action 数，再向上对齐到整 block。同一次 Parquet 投影读取直接生成最终 state/action、时间戳、有效性 mask 和 t 的标签，不生成较长的中间补齐窗口。
2. 按实际有效终点截断视频查询时间，解码处理后样本需要的视频帧。
3. 复用 `ActionProcessor.__call__`，按每个 block 起点编码动作和归一化。
4. 生成 prompt/token IDs，并使用 `Geometry.token_layout` 计算实际长度。

## 模型验证

[`iter_validation_batches()`](../../../veomni/data/causal_cosmos3/datasets/validation.py#L59) 接受 ActionFrameDataset 和 VideoFrameDataset，要求 sample.full_history=True。自动划分仍以 episode 为单位，验证枚举 held-out episode 的全部合法 anchor，不受训练 source 权重影响。

验证按以下顺序准备每条数据：

1. 固定当前任务模式，直接调用 frame 子类的 make_sample；joint 按 FD、ID、policy 各遍历一次，video 只遍历一次。未显式配置 sample.mode 时，Validator 从实际 Dataset 获取默认 mode，frame action 默认为 policy。
2. 复用 Dataset 的读取、anchor 标签、补齐、block 编码和完整 history。每条样本只读取一次 payload，不预先重算 latent 数，也不初始化或推进 Dataset RNG。
3. 从返回样本生成 record：identity 包含 source/root、episode、candidate_index、caption、anchor、past、请求的 future、video_stride 和实际/补齐长度。window 保存 Reader 返回的 FrameWindow；history_blocks 直接来自样本。固定完整 history 不生成 history_seed。
4. 按 identity 和 mode 派生独立的 time/video/action seeds，返回现有 batch/record/weight 协议。每个 pack 包含一个样本；各 DP rank 轮数相同，末轮重复样本的 weight=0。
5. 交给现有 [CausalCosmos3Validator.evaluate()](../../../veomni/trainer/causal_cosmos3.py#L381) 完成 VAE 编码、固定噪声准备、causal forward 和 loss 汇总。frame 读取或处理异常直接交由现有跨 rank 错误同步，不转换成空样本。

例如完整 128 步动作窗口按默认 geometry 产生 33 帧视频、9 个 latent，验证 history_blocks=9。短 episode 的 history 取实际补齐后的 latent 数。模型输入与训练使用同一 make_sample 路径，旧 Dataset 的验证 identity、history_seed 和噪声 seeds 保持原有规则。

## 原始窗口与补齐

不传 block_size 的原始 `read_sample` 保持 `[t-k,t+a]` 请求：state 有 k+a+1 行，action 有 k+a 行，视频从 t-k 按 `video_stride` 抽帧。基类要求 k 能被 video_stride 整除，保证 anchor 在抽帧网格上，不在 raw 层写死 VAE 的时间压缩率。

越界观测重复 episode 首尾帧并设置时间 mask。原始 action 只在相邻观测和偏移后的目标行都存在时有效。`target_offset` 只应用一次；原生 action 要求 offset=0，state 目标要求使用 state mask。

原始 `timestamps` 为各观测位置对应的真实源时间，越界位置重复端点。动作训练读取通过 block_size 直接构造最终时间轴，在有效终点之后按存储 FPS 递增，视频查询仍截到有效终点。

block_size 读取用于 ActionFrameDataset 的合法前向候选，要求 past_frames=0 且至少一个有效 action，由数据类初始化和候选索引保证。reader 直接将补齐目标指向最后一个有效 action，将补齐 state 指向有效终点。这同时避免末行无 outgoing action 时的 NaN/零四元数进入编码。有效目标仍严格检查。

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

`seed/rank/rng`、history 配置及校验、`_get_rng`、`select_history` 和 `state_dict/load_state_dict` 统一由基类管理，两个子类直接继承。Action 默认 full_history=True，Video 默认 False；history_range 默认 (1, 32)，RNG 的 seed/rank/worker 规则沿用原实现。显式传入 mode/history 不消耗采样 RNG；固定 mode 且 full_history=True 时也不初始化或消耗 RNG。原始读取也不初始化或消耗 RNG，未初始化时所有帧数据类的状态统一为 `{"rng": None}`。

## 纯视频子类

`VideoFrameDataset` 继承原始读取和按帧标签，不读取 action/state 列，不依赖动作统计量。默认自动模式以 video_stride=1 构造读取计划，仅在基础 stride 为 1 时按 FPS 调整：source_fps > 16 时 stride=2，否则 stride=1。显式指定 `video_stride` 时使用固定间隔。

```python
from veomni.data.causal_cosmos3 import FrameCandidateConfig, VideoFrameDataset

dataset = VideoFrameDataset(
    "/path/to/lerobot_v3",
    source="robotwin",
    viewpoint="ego_view",
    candidates=FrameCandidateConfig(future_frames=128, anchor_stride=1),
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

1. `read_sample(video=False)` 投影读取时间戳和 t 的标签，保留原始请求与边界 mask。
2. 将有效帧间隔数补到 `video_stride × temporal_compression × latent_block_size` 的整数倍，延伸训练时间轴。
3. 将视频查询时间截到有效终点，只解码最终样本需要的帧；额外位置重复终点画面。
4. 写入 `mode="video"`、`block_size`、`history_blocks`、首 latent 条件和 `num_valid_video_latents`，生成 prompt/token IDs。
5. 使用现有 `video_token_layout` 计算文本与 clean/target 两份视频 token 数。history 单位仍是 latent 帧；`full_history=True` 使用补齐后的全部 latent。

VAE 首帧为 t；补齐画面通过有效 latent 数限制监督范围。样本不包含 action/state 或动作条件字段。`tokenize` 在处理后样本中必需，在 raw 接口中可省略。

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
    past_frames: 0
    future_frames: 128
    anchor_stride: 1
  sample:
    full_history: false
    history_range: [1, 32]
```

原始双向窗口检查继续直接实例化 `LeRobotFrameDataset`，不通过 video 训练样本分支。自动 episode split 对 action/video 子类都排除没有有效转移的 episode。

## 按帧标注

默认读取 t 的 `task_index → meta/tasks.parquet(task_index, task)`。info 声明 `subtask_index/subtask_id` 时，查 `meta/subtasks.parquet(subtask_index, subtask)`；没有 subtask 列或显式关闭时使用 task 文本。读取的 ID 必须是有效的整数，未知 ID 报错，0 是有效 ID。`FrameLabelConfig` 支持配置列名、文本表和 episode 作用域。

JSONL 可以作为 ID→文本定义表，配置 `subtask_records_key` 等字段即可；新类不根据起止区间判断当前 subtask。只有区间标注的旧数据使用旧数据类。

一个样本只选择 t 的一段文本。窗口可以跨 subtask，不自动按 subtask 截断，也没有逐 block 文本切换。

## 配置与检查

[DiT 配置](../../../configs/dit/causal_cosmos3_edge_action_frame.yaml) 的 `data` 段设置 stage、geometry、candidates 和 sample，采用 `past_frames=0 / future_frames=128 / anchor_stride=1 / mode=policy`。它通过 `data.train_path` 引用[数据源配置](../../../configs/data/frame_datasets.yaml)，后者只负责 sampling、可选 validation 和 sources（包含路径、统计量及标签列配置）。两份 YAML 经现有入口组装成 Dataset 配置，再传给 `build_dataset(config, tokenize)`；训练 dataloader 复用现有组包路径，不再按 sample_kind 拒绝帧样本。

多源混合的长度回调统一返回样本的真实 `token_length`；`source_weighted` 仍按样本权重混合，不按 token 数调整权重。`ActionBatchingStrategy.put_item` 接收帧样本，按同一 token 预算组包；超预算单样本保留单独成包的行为。`collate_action_pack` 将每个 pack 包装为 `{"samples": [...]}`。模型验证要求 full_history=True，并复用现有 episode 划分、验证回调和模型执行链路。

[CPU 测试](../../../tests/data/test_causal_cosmos3_frame_dataset.py) 检查原始读取、标签、索引内存，并对照旧路径的候选顺序、block 编码、补齐、时间戳、conditioning 和 token 数。action/video 两种阶段均经过真实 native dataloader，覆盖 sample_uniform/source_weighted、每步 microbatch 数、token 预算及超预算单样本；验证测试覆盖完整 history、单次 payload 读取、任务模式、固定 seeds、RNG 状态保持、DP 等长轮次及零权重补齐；另以固定 latent 替代 VAE 权重，实际运行 Condition、packing、tiny Cosmos3Model forward、loss 和 Validator，检查 action/video 默认 mode 与重复评估结果。

[源数据检查脚本](../../../scripts/causal_cosmos3/validate_frame_dataset.py) 用独立 Parquet/NumPy block 编码及 PyAV 解码核对样本，使用显式检查 tokenizer，不加载模型或 checkpoint：

```bash
PYTHONPATH=. python scripts/causal_cosmos3/validate_frame_dataset.py \
  --root /path/to/lerobot_v3 --source robotwin --viewpoint concat_view \
  --future-frames 128 --report /tmp/frame-validation.json
```

从 state 生成目标时传 `--columns '{"target":"state_unified","target_mask":"mask_state","target_offset":1}'`。`--labels` 接受 FrameLabelConfig JSON；`--statistics` 可指定实测 block 统计量。多 worker 检查要求系统允许本地 socket 通信。
