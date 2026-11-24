# 独立帧数据类与现有 block 训练

帧路径保留独立的 `LeRobotFrameDataset → ActionFrameDataset / VideoFrameDataset`，不合并到旧 Dataset，也不替换旧窗口索引。新的采样/标签规则与已有训练语义分别由对应组件负责。

## 职责

| 组件 | 负责的行为 |
| --- | --- |
| `LeRobotFrameDataset` | 元数据、相机、前缀定位、原始窗口、t 的 task/subtask 文本、公共前向窗口计数、处理后视频解码、history 和 RNG 状态。 |
| `ActionFrameDataset` | 合法前向 action 候选、block 对齐、policy/history 和最终样本字段。 |
| `VideoFrameDataset` | 合法前向 video 候选、按 FPS 抽帧、latent block 对齐、history 和纯视频样本字段。 |
| `read_frame_window` | 单次投影读取原始数据；动作训练传入 block_size，在读取前确定有效终点和最终 block 长度，一次生成补齐数据、有效性标记及时间戳。 |
| 现有 `ActionProcessor` / `Geometry` | 每 block 的 state 参考、动作编码、归一化、latent 索引和 token 数。 |

两个子类直接继承基类的 history 选择及 RNG 保存/恢复，只在各自初始化中传入默认配置。候选的合法范围和 padding 规则仍由子类负责，共同的窗口计数公式和视频查询截断/解码由基类实现。

详细接口、调用顺序和配置见[数据契约](../data/causal_cosmos3/frame_dataset.md)。

## 当前完成范围

1. 保留新的独立数据类、O(episodes) 索引、按需读取和按帧标注。
2. 移除单 state[t] 动作编码，改为复用现有 block 编码和统计量。
3. 实现 past_frames=0 的前向窗口、合法起点和短 episode 尾部对齐，补齐训练样本字段。

训练 dataloader 已允许 frame_anchor 使用现有动态组包和 collate。full_history=True 的模型验证直接复用 frame make_sample 和现有 Validator，不增加 history RNG，不重复计算窗口补齐。真实 VAE/模型链路、loss/梯度对照，以及双向历史窗口和左侧 padding 的 latent/attention 语义留待后续处理。

## 采样与标注

原始基类允许每个非空 episode 的每帧作为 anchor，用于数据检查。action/video 子类只接受有有效转移的 episode：长 episode 使用可容纳完整未来窗口的起点，短 episode 保留一个起点。每个 episode 只存首起点和累计候选数量，取样时反算 t。

按帧 task/subtask ID 和文本查找保留在新路径中，不恢复旧区间推断。文本来自 t，整个样本共享一段文本；跨 subtask 的固定窗口仍可存在。这项数据语义与 action 的 block 参考状态无关。

## 动作和视频对齐

在 actions_per_block=32、video_stride=4、temporal_compression=4 时，128 个 action 对应 33 帧视频和 4 个 block state。每个 block 的动作相对于该 block 的起点 state 编码，VAE 首帧为 t。

短窗口的 action 重复最后一个有效目标，state 重复有效终点。视频查询时间也截到该终点，不能仅在 loss 中忽略补齐位置而让后续真实帧进入输入。state-derived action 的 target_offset 同时决定合法起点和可用终点。

默认 raw 接口保留完整请求及 episode 边界 mask。动作 make_sample 在首次读取时传入 actions_per_block，直接得到整 block 的 action/state 和训练时间轴，不再截断、重复补齐。处理后样本只保留 block state，沿用现有 action/state mask 与索引字段；不继续维护单 state 的 states_mask/action_reference 分支。

纯视频子类复用相同原始读取，但不读取动作列或统计量。默认沿用旧 VideoDataset 的 source_fps > 16 时 stride=2、否则 stride=1，也可显式指定固定 video_stride。有效窗口补到完整 latent block，视频查询截到真实终点；首 latent 为条件，有效 latent 数限制监督范围。随机 history 可恢复，训练 dataloader 采用与 action 相同的组包路径。

## 验证

数据测试用独立数值预期验证多个 block 的参考状态，并与旧 ActionDataset / VideoDataset 对照候选顺序、编码、padding、时间轴、conditioning 和 token 数。按帧标签用零 ID、未知 ID、episode 作用域和 JSONL 文本定义分别覆盖。

真实数据脚本独立读取 Parquet、用 NumPy 编码每个 block，并用 PyAV 比较视频像素。当前这些检查不代表模型训练入口已接通。

## 配置归属

实验参数 stage、geometry、candidates 和 sample 位于 `configs/dit/causal_cosmos3_edge_action_frame.yaml` 的 data 段。`configs/data/frame_datasets.yaml` 只描述数据源与混合采样，可按需配置 validation。沿用现有入口的组装流程，不在 source YAML 重复实验参数。
