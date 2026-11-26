# 独立帧数据类与现有 block 训练

帧路径保留独立的 `LeRobotFrameDataset → ActionFrameDataset / VideoFrameDataset`，不合并到旧 Dataset，也不替换旧窗口索引。新的采样/标签规则与已有训练语义分别由对应组件负责。

## 职责

| 组件 | 负责的行为 |
| --- | --- |
| `LeRobotFrameDataset` | 抽象公共基类：元数据、相机、前缀定位、读取计划、窗口 caption、公共逐帧索引、处理后视频解码、history 和 RNG 状态。 |
| `ActionFrameDataset` | 逐帧 action 候选、双向完整块、policy/history 和最终样本字段。 |
| `VideoFrameDataset` | 逐帧 video 候选、按 FPS 抽帧、双向完整 latent block、history 和纯视频样本字段。 |
| `read_frame_window` | 单次投影读取有界双向范围，复用原始 task 标签选块，返回对齐 payload、实际视频索引、逐帧标签及时间戳。 |
| 现有 `ActionProcessor` / `Geometry` | 每 block 的 state 参考、动作编码、归一化、latent 索引和 token 数。 |

两个子类直接继承基类的 history 选择及 RNG 保存/恢复，只在各自初始化中传入默认配置。子类按各自 Geometry 过滤不足一块的 episode；基类管理公共逐帧索引和视频解码。

详细接口、调用顺序和配置见[数据契约](../data/causal_cosmos3/frame_dataset.md)。

## 当前完成范围

1. 保留新的独立数据类、O(episodes) 索引、按需读取和按帧标注。
2. 移除单 state[t] 动作编码，改为复用现有 block 编码和统计量。
3. 接入 SWAM 双向完整块采集、逐帧候选及按实际视频覆盖选择 dense caption。

训练 dataloader 已允许 frame_anchor 使用现有动态组包和 collate。当前 Dataset[index] 读取失败后顺序尝试下一候选。full_history=True 的模型验证直接调用 frame make_sample，不换起点、不消耗训练 RNG，读取失败交给现有 Validator 同步报错。真实 VAE/完整模型链路及 loss/梯度对照仍需在训练环境验收。

## 采样与标注

原始数据检查和训练统一通过 action/video 子类，基类不直接实例化。子类只接受能容纳一个完整块及末端观测的 episode，按 anchor_stride 枚举其全部帧，包含末帧。每个 episode 只存首起点和累计候选数量，取样时反算 t。

按帧 task/subtask ID 和文本查找保留在新路径中，不恢复旧区间推断。选窗比较候选块起点与原始 t 的 task；dense subtask 不参与裁窗。窗口确定后，按完整逐帧标签划连续段，以实际视频采样点计数，选覆盖最多的一段，并列选最早段；非连续同 ID 不合并，整个样本共享一段文本。这项数据语义与 action 的 block 参考状态无关。

## 动作和视频对齐

在 actions_per_block=32、video_stride=4、temporal_compression=4 时，128 个 action 对应 33 帧视频和 4 个 block state。每个 block 的动作相对于该 block 的起点 state 编码，VAE 首帧为最终窗口的最早帧，可能早于原始 t。

训练窗口仅保留真实完整块；末端观测缺失时按 SWAM 回退最后一个块，连一块都无法保留则抛 EmptyTemporalSampleError。target_offset 目标仍需合法。较短样本按实际长度进入 packing，不重复末端补出有效块。

read_sample 检查原始 payload，make_sample 继续预处理；两者使用同一套 max_blocks 和窗口 caption，不再配置 past_frames/future_frames。reader 一次读取 t 两侧有界范围，复用其中的标签与 payload，不缓存全 episode。处理后样本仍使用现有 block state/action mask 与 latent 索引。

纯视频子类不读取动作列或统计量；默认 source_fps > 16 时 stride=2，否则 stride=1，也可显式指定 stride。自身的 stride/compression/latent block 决定块跨度。随机 history 可恢复，与 action 共用组包路径。

## 验证

数据测试用独立数值预期验证多个 block 的参考状态，并与旧 ActionDataset / VideoDataset 核对相同实际窗口的编码、时间轴、conditioning 和 token 数，并单独验证双向候选、caption、顺序跳过无效起点、验证错误传播及 history RNG 恢复。按帧标签用零 ID、未知 ID、root 内 ID 映射、自定义列名和 JSONL 文本定义分别覆盖。

真实数据脚本独立读取 Parquet、用 NumPy 编码每个 block，并用 PyAV 比较视频像素。这些 CPU 检查不替代真实 VAE/完整模型的训练验收。

## 配置归属

实验参数 stage、geometry、candidates、sample 和 samples_per_pack 位于 `configs/dit/causal_cosmos3_edge_action_frame.yaml` 的 data 段。`configs/data/frame_datasets.yaml` 只描述数据源与混合采样，可按需配置 validation。沿用现有入口的组装流程，不在 source YAML 重复实验参数。frame_anchor 的 candidates 统一配置 mode/anchor_stride/max_blocks，max_blocks 控制两侧合计的块数上限，geometry 决定每块跨度；不再配置 candidates.past_frames/future_frames。旧 frame_window/segmented 不变。

`samples_per_pack` 配合 sample_uniform 启用有限候选 epoch：每 pack 固定 B 条样本，max_seq_len 失效，micro_batch_size 仍为一个 pack。每轮步数为验证划分后 `ceil(N/(B*D*A))`，尾部重复当前 shuffle 前缀补到完整更新，样本正常参与 loss；Dataset 跳过无效起点后返回实际样本，保持请求槽位数，不保证实际窗口唯一。复用原生 DistributedDataloader 和父 sampler 的 shuffle/set_epoch/分片，只扩展 sampler 的补齐长度；未配置时 source_weighted 与 token 预算路径保持原行为。详见[固定样本组包](../data/causal_cosmos3/frame_dataset.md#固定样本数与有限-epoch)。
