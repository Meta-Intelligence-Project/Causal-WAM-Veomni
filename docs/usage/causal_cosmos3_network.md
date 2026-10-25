# Cosmos3 网络构建与固定输入验证

本模块提供原生网络、结构配置和权重加载。模型运行时无需安装或加入 `cosmos-framework` 的路径。真实数据训练、VAE/tokenizer 条件处理、dataloader 和完整 DiT recipe 已接入，使用方式见 [Causal Cosmos3 training](causal_cosmos3_training.md)。

## 选择结构

使用 `configs/model_configs/causal_cosmos3/causal_cosmos3_{edge,nano,super}.json`。`variant` 只是标识；更改该字符串不会自动改变网络参数。完整配置可通过 `config.save_pretrained()` 保存并重新构建。

| 结构 | Edge | Nano | Super |
| --- | --- | --- | --- |
| hidden / intermediate | 2048 / 9216 | 4096 / 12288 | 5120 / 25600 |
| 层数 | 28 | 36 | 64 |
| Q heads / KV heads / head dim | 16 / 8 / 128 | 32 / 8 / 128 | 64 / 8 / 128 |
| MLP | ReLU² | SwiGLU | SwiGLU |
| 文本 QK norm | 无 | 有 | 有 |
| generation 读取文本 K 的额外 norm | 有 | 无 | 无 |
| RMS epsilon / RoPE theta | 1e-5 / 1e8 | 1e-6 / 5e6 | 1e-6 / 5e6 |
| 初始 action 接口 | domain，64 维 | domain，64 维 | none |

三个型号都是双路径 dense MoT，历史参数名 `mlp_moe_gen` 不表示稀疏专家。三者都使用 interleaved mRoPE，section 为 `[24,20,20]`。Super 的 Q/O 投影内部宽度为 8192，不能使用 hidden size 5120 代替。

Edge/Nano 配置来自原始 transformer 配置和权重索引；Super 来自 Cosmos 的 Qwen3-VL-32B 配置与 Super SFT 配置。Super 默认关闭 action，与审阅的基线一致；需要 action 时显式选择 shared 接口及维度。当前网络范围不包含视觉编码器、VAE、音频分支。

## 加载权重

Edge 原始权重选择 `Cosmos3-Edge/transformer` 组件目录，不能换成根目录索引。新配置与原始 Diffusers 配置不是同一配置类。

```python
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model

ops = OpsImplementationConfig(
    attn_implementation="eager",
    rms_norm_implementation="torch_npu",
    rotary_pos_emb_implementation="torch_npu",
)
model = build_foundation_model(
    config_path="configs/model_configs/causal_cosmos3/causal_cosmos3_edge.json",
    weights_path="/path/to/Cosmos3-Edge/transformer",
    config_kwargs={"weight_loading": {"initialize_missing_patterns": []}},
    torch_dtype="bfloat16",
    init_device="npu",
    ops_implementation=ops,
)
print(model.weight_loading_report)
```

Cosmos3 的 attention 后端由 forward 的 `attention_backend` 指定；上例的框架级 `attn_implementation` 不改变 causal 可见性。当前仅支持 SP/CP=1。

上例的 norm/RoPE 算子选择用于对齐 Edge 的 Ascend 基线。Nano/Super 基线使用 eager norm/RoPE，验证脚本按型号选择对应实现。BF16 下 eager 与融合 RoPE 的舍入不同；不能把两种算子混用后的全网误差直接归因于 causal mask。VeOmni 的融合 RoPE 当前复用 eager backward，与旧融合 backward 的 BF16 梯度可能存在数值差异。

参数名与 transformer safetensors 一致。VeOmni 的现有 tensor converter 扩展用于覆盖审计：未配置跳过的额外参数、重复参数及 shape 不符均报错，不裁剪、不补零。`weight_loading.initialize_missing_patterns` 不配置时，缺失参数按模型规则初始化，并在 warning 和加载报告中逐项列出；显式设置为空列表时要求参数齐全，上例使用这一严格模式。指定非空 glob 列表时，仅允许初始化匹配的缺失参数，其他缺失参数报错。

使用原始权重初始化 causal action 时，将配置设置为 `action_interface="shared"`、`action_dim=55`、`state_dim=55`，或使用自己明确选定的模板宽度。通过本次加载的 `weight_loading` 配置跳过旧 action/audio 参数，并限制新增参数的初始化范围：

```python
from veomni.models.checkpoint.weights import load_model_weights

model = build_foundation_model(
    config_path="configs/model_configs/causal_cosmos3/causal_cosmos3_edge.json",
    config_kwargs={
        "action_interface": "shared",
        "action_dim": 55,
        "state_dim": 55,
        "weight_loading": {
            "skip_patterns": ["audio_*", "action_*"],
            "initialize_missing_patterns": ["action_*", "state_*"],
        },
    },
    init_device="meta",
    torch_dtype="bfloat16",
    ops_implementation=ops,
)
load_model_weights(model, "/path/to/Cosmos3-Edge/transformer", init_device="npu")
print(model.weight_loading_report)
```

这是本次加载的一次性策略，不随模型配置导出。从 domain 切换到 shared 接口时，加载兼容参数、跳过旧 action 参数并显式初始化新增的 action/state 参数；报告记录加载数量、跳过和初始化的参数名称。Nano 的原始音频头也需明确配置跳过。保存后的完整 shared 模型可通过 `initialize_missing_patterns: []` 严格校验恢复。上述两个步骤是网络验证入口；训练的 FSDP 包装、加载顺序由 VeOmni runtime 管理。

新建的 `action_modality_embed` 使用标准差 `hidden_size**-0.5`、边界 ±3σ 的截断正态；`state_modality_embed` 使用相同标准差的普通正态，与 Ascend causal action 基线一致。这同时适用于普通构造和预训练加载后补初始化；完整 checkpoint 中已有的参数值不会重新初始化或截断。

训练保存/恢复使用 VeOmni 原生 DCP。本实现没有 Cosmos DCP reader，也没有把完整恢复改为预训练初始化。FSDP2 的层边界为 `Cosmos3DecoderLayer`；完整 DCP 恢复跳过 HF 权重初始化。动态 loader 的已有精确重放用例在零 worker 和两个 worker 下均预期失败，DCP 恢复不能直接保证后续样本顺序一致，详见 [训练与 checkpoint 说明](causal_cosmos3_training.md#checkpoints)。

## 张量接口

模型使用 `CausalLayout` 和张量，不依赖旧 `PackedSequence`。layout 中每个位置对应一个 packed token：

| 字段 | 含义 |
| --- | --- |
| `sample_ids` | 所属样本，从 0 开始 |
| `block_ids` | 文本和独立首 latent 为 -1，后续 latent 为 `(frame-1)//block_size` |
| `modalities` | text=0、video=1、action=2、state=3 |
| `roles` | text=-1、clean video=0、target=1、condition=2、state=3 |
| `history_blocks` | 每个样本允许读取的 clean video 历史 block 数 |

以上字段使用 CPU Python tuple。每种模态的 payload 按它在 layout 中出现的顺序排列。模型入口：

- `input_ids`：`[N_text]`。
- `video_tokens`：`[N_video, patch_size² * latent_channels]`。`patchify_video([C,T,H,W], p)` 保留原实现的 channel-last patch 顺序，支持空间补齐；`unpatchify_video()` 还原并裁剪。
- `video_timesteps`、`action_timesteps`：各模态每个 token 一个原始 0..1000 时间；模型乘 `timestep_scale`。clean、condition、state 使用 0。block 内视频/动作时间对齐由调用方负责。
- `position_ids`：`[3,N_packed]`，已经含 FPS 调整、时空位置和模态偏移。模型只执行 mRoPE，不在网络内重新推导数据几何。
- `action_tokens`、`state_tokens`：`[N_modality, configured_dim]`，以及同 shape 的 boolean 有效维度 mask。domain 接口额外需要每个 action token 的 `action_domains`。

返回 `Cosmos3Output.video/action`，顺序与各自 payload 一致，包含 clean/condition 位置。只监督 role=1 的位置；action 无效维度输出为零。`output_text_logits=True` 可额外计算文本 logits，`output_hidden_states=True` 返回最后归一化后的 packed hidden states。

网络不采样噪声；训练条件模型负责 VAE 编码、加噪和监督信息组装。传入 `targets` 时，网络返回 FP32 video/action loss 字典：逐样本按 active elements 归一化，再对 pack 内样本取平均。clean history 的投影、时间编码和主干计算始终保留梯度。

## Causal 可见性

文本 UND 在每个样本内使用 causal attention。GEN 只能读取同一样本的 token，具体规则如下：

| 查询位置 | 可读取的 KV |
| --- | --- |
| clean video | 文本、历史 clean video、同 block 的 clean video |
| target（video/action） | 文本、历史 clean video、同 block 的 target、condition 和 state |
| condition（video/action） | 文本、历史 clean video、同 block 的 condition 和 state |
| state | 文本、同 block 的 state |

历史仅包含 `history_blocks` 范围内的先前 clean video block，不包含历史 action/state。当前 target 不读取同 block 的 clean 副本或未来 block；condition 不读取 target，clean video 不读取 action/state，避免多层 attention 间接泄露目标信息。

首个 video latent 使用独立 block `-1`，后续 latent 按 block 分组；调用方负责构造 block 和时间位置。clean KV 保留梯度，NPU 分组 attention 对共享 KV 的梯度使用 FP32 累加。`dense`、`grouped` 和 `npu_tnd` 遵循相同可见性规则。

## 运行验证

从 VeOmni 根目录执行。使用正常的 VeOmni NPU 环境（Transformers 5.16.1），无需旧 Cosmos 环境。

```bash
# 小模型：CPU reference；可切换 Edge/Nano/Super 和 video/fd/id/policy
python -m scripts.causal_cosmos3.validate_network --tiny --task policy \
  --config configs/model_configs/causal_cosmos3/causal_cosmos3_super.json

# 原始 Edge：NPU BF16，短 causal video 的前向和反向
python -m scripts.causal_cosmos3.validate_network \
  --device npu --dtype bfloat16 --backend npu_tnd \
  --weights /path/to/Cosmos3-Edge/transformer

# 原始 Edge 初始化新的 shared action/state 接口
python -m scripts.causal_cosmos3.validate_network \
  --device npu --dtype bfloat16 --backend npu_tnd --task policy \
  --pretrained-initialization \
  --weights /path/to/Cosmos3-Edge/transformer

pytest tests/models/diffusers/test_causal_cosmos3.py -q
pytest tests/models/base/test_auto_registry.py -k cosmos3 -q
```

`dense` 是小规模 pair-mask reference；`grouped` 是 PyTorch 分组 reference。这两个参考后端的文本 UND 使用逐样本 causal SDPA。

`npu_tnd` 同时选择 UND 和 GEN 的 Ascend 实现：UND 将各样本的文本打包为一次 TND causal attention，使用原生 GQA、累计序列长度和 `sparse_mode=2` 的固定 2048×2048 压缩 causal mask（支持更长的文本序列）；GEN 使用无 mask 分组 attention，重复 KV 的梯度先 FP32 累加再转换回输入精度。每个 batch 的索引和序列长度只在层循环外构建一次。UND 一次调用最多支持 1000 个非空文本序列、合计 1,000,000 个文本 token。
