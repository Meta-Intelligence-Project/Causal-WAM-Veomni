# EgoSuite JSON A/B 部署与验收

Set `data.train_path` to the JSON manifest and configure
`data.manifest_options.source_defaults` in the training recipe. A manifest
is a nonempty list of records with `lerobot_uuid`, an absolute
`lerobot_path`, and the selected integer `episodes`. The trainer uses the
UUID as the source name and preserves the original episode indices.
`dataset_name` may be included as a label.

### 在新服务器部署 Ego A/B

新服务器已有同一个仓库时，从 GitHub 拉取 `ego-action` 并在该分支启动训练。JSON 适配在
`tasks/train_causal_cosmos3.py` 的配置解析入口，直接读取清单并生成现有 DataLoader
使用的 `sources`。原有数据源 YAML 仍可使用。

```bash
git fetch origin
git switch ego-action
```

JSON 里填写新服务器能访问的 LeRobot 子数据集目录，以及该目录下真实的 episode ID。
同一个 `dataset_name` 可以对应多个目录，每条记录的 UUID 必须唯一。

```json
[
  {
    "lerobot_uuid": "000b0d14-edcf-49dd-a9ad-2f2a9bebc794",
    "lerobot_path": "/efs-mi-east4-4/EmbodiedDatasets/egosuite_processed/v0/standard/EgoStandard__Wipe_Air_Conditioner_Casing",
    "dataset_name": "EgoSuiteProcessed_V0",
    "episodes": [0, 1]
  }
]
```

以 [A 配置](../../configs/dit/causal_cosmos3_ego_json_a.yaml) 和
[B 配置](../../configs/dit/causal_cosmos3_ego_json_b.yaml) 为模板，在两份配置中设置：

| 字段 | 新服务器填写的值 |
| --- | --- |
| `model.model_path` | Cosmos3-Edge 4B checkpoint 的根目录，包含 `transformer/`、`vae/`、`text_tokenizer/` |
| `data.train_path` | 上面 JSON 清单的绝对路径，A/B 使用同一份 |
| `data.manifest_options.source_defaults.statistics` | 实测的 Ego state/action 统计 JSON。统计维度、模板和窗口几何须与配置匹配，优先使用该训练划分计算的统计 |
| `model.accelerator.dp_shard_size` | 默认为 8，每组的 DP 大小须能被分片大小整除 |
| `train.checkpoint.output_dir` | 两组分别使用可写且不同的目录。默认是 `outputs/ego_json_ab/A` 和 `outputs/ego_json_ab/B` |

模板使用 Ascend NPU。如果新服务器使用其他加速器，沿用该机器已可用的 Cosmos3
attention 后端、RoPE 和 RMSNorm 配置。对于同格式的 EgoSuite 数据，保留头部左相机
`observation.images.head_left`、`state_unified`、`action_unified`、
`mask_state`、`mask_action` 和 `target_offset: 0`。其中 offset 为 0 的前提是
`action_unified[t]` 已表示下一步目标。

两组使用 `stage: action`、`sample.mode: policy`、`sample_uniform` 及相同的
episode 列表。视频损失权重均为 10，动作损失权重 A 为 0、B 为 10，B 监督共享
55 维模板里的双腕位姿索引 14 至 27。两组模板均为 10,000 步，默认不从清单拆分
验证集。如需留出验证集，在两份配置中同步设置 `manifest_options.validation`。

### 首次运行验收

1. 确认模型组件、统计文件、JSON 和 JSON 内的每个 LeRobot 根目录均可读。
   目录下应有 LeRobot 元数据、Parquet 与视频；清单内的 episode ID 应与元数据一致。
2. 确认每组都能解码视频、读到 state/action，并检查双腕维度有有效动作 mask。
3. 在已激活 CANN 和训练环境的仓库根目录，各跑一次 1 步 smoke test。下面是每组
   使用 8 张 NPU 时顺序运行的示例；实际卡数与分片配置应一致。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for arm in a b; do
    torchrun --standalone --nproc-per-node=8 tasks/train_causal_cosmos3.py \
        "configs/dit/causal_cosmos3_ego_json_${arm}.yaml" \
        --train.max_steps 1 \
        --train.checkpoint.output_dir "outputs/ego_json_smoke/${arm}" || break
done
```

确认两组都完成一个 optimizer step 后，正式训练时去掉两个 smoke 覆盖参数。
同机并行时为 A、B 分配互不重叠的 NPU，并分别保存日志。
