# RoboTwin client/server 闭环推理

本适配参考 Cosmos Framework 的 `docs/causal_action_server.md` 和
`cosmos_framework/inference/causal_action/`，保留其
`cosmos-causal-action-v2` WebSocket + MessagePack/NumPy 协议、原生 joint/EEF
字典、完整 block 反馈和客户端坐标转换，模型端使用 VeOmni 的
[CausalCosmos3Pipeline](../../veomni/models/diffusers/causal_cosmos3/inference.py#L120)。
运行时不依赖 Cosmos Framework checkout。

## 两端启动

server 在 VeOmni 推理环境运行，client 在已经能运行 RoboTwin 的仿真环境运行。
两者可以分处不同机器。server 加载完整 native/HF action checkpoint 或原生 DCP 的 `global_step_N/model/ckpt`；旧格式权重先按
[离线推理文档](causal_cosmos3_inference.md)转换。

在两端各自的 Python 环境安装额外的传输依赖；这是独立的可选工具依赖列表，
无需在 RoboTwin 环境安装 VeOmni、Torch 或模型依赖：

```bash
uv pip install -r tasks/infer/robotwin/requirements.txt
```

client 的 RoboTwin 环境还需 NumPy、OpenCV 和带 `libx264` 的 FFmpeg 命令行程序，
用于保存真实观测视频。server 使用 VeOmni 原有的 PyAV 编码依赖。

1. 编辑 [配置模板](../../configs/infer/causal_cosmos3_robotwin.yaml)，填写
   `model_config`、`tokenizer`、`vae`、`statistics`。`model_config` 必须是该次训练保存的 `config.json`，DCP 加载时尤其需要。`geometry`、`source.resolution`、`fps`、
   `ops_implementation` 必须与训练对应；`fps` 是对齐后的 action/control FPS，
   视频 FPS 为 `fps / video_stride`。Edge/Nano/Super 架构由 checkpoint 的
   `config.json` 决定，tokenizer 必须匹配该模型。
2. 在仓库根目录启动单进程模型服务，Ascend 环境先按本机设置加载驱动和 toolkit：

   ```bash
   python -m tasks.infer.serve_robotwin \
     --config configs/infer/causal_cosmos3_robotwin.yaml \
     --checkpoint /path/to/global_step_N/model/ckpt \
     --device npu:0 --host 0.0.0.0 --port 8000
   ```

   等待 `Causal policy listening`。支持 `--statistics`、`--robot-config`、
   `--fps`、`--steps`、`--history`、`--seed` 覆盖；`--history` 与 VeOmni 原生
   推理一致，单位是 latent 帧。默认 32 actions/block、10 步采样、guidance=1，
   每轮执行完整 32 步。`sampling.kv_cache: true` 默认开启跨请求缓存，也可显式传 `--kv-cache`。无需分布式初始化。

   本机已配置的 adjust_bottle checkpoint 可直接启动：

   ```bash
   bash scripts/causal_cosmos3/infer_scripts/serve_robotwin.sh
   ```

   可用 `PYTHON_BIN`、`INFER_CONFIG`、`CHECKPOINT_PATH`、`ASCEND_RT_VISIBLE_DEVICES`、`HOST`、`PORT` 环境变量覆盖启动参数。
3. 在仿真机器运行独立入口，路径指向这个仓库：

   ```bash
   python /path/to/Causal-WAM-Veomni/tasks/infer/robotwin_client.py \
     --robotwin-root /path/to/RoboTwin \
     --server-uri ws://SERVER_IP:8000 \
     --task-name adjust_bottle --task-config demo_clean \
     --control-mode joint --state-source command \
     --eef-dataset-root /path/to/training_robotwin_dataset \
     --seed 100000 --episodes 1 --max-steps 64 \
     --output /path/to/outputs/robotwin_sim
   ```

   `--eef-dataset-root` 从 `meta/info.json` 的 `custom.eef_enrichment` 读取
   坐标系和 `T_world_base`。也可传 `--eef-config /path/to/eef.json`：

   ```json
   {
     "eef_pose_frame": "base",
     "T_world_base": {
       "position": [0, 0, 0],
       "quaternion_wxyz": [1, 0, 0, 0]
     }
   }
   ```

   上述变换值只是结构示例，必须替换成数据生成时的真实变换。训练 EEF 使用世界
   坐标时，配置为 `{"eef_pose_frame": "world"}`。即使执行 joint 模式，state
   mask 仍可能包含 EEF，因此客户端要求显式声明 EEF 数据坐标约定。

## 状态、动作与掩码

server 从统计文件 `provenance.sources` 解析精确的 `state_mask` 和 `action_mask`。
如果统计包含多个不同机器人/掩码，或缺少来源记录，必须通过 `--robot-config`
提供 JSON，字段为两个长度 55 的 0/1 数组 `state_mask`、`action_mask`，以及可选的
`left_arm_dof`、`right_arm_dof`（默认均为 6，支持 1..7）。也可直接写入 YAML
的 `robotwin` 字段。掩码采用训练的 canonical unified55 顺序，不能用混合统计的
`valid` 并集替代。

服务端验证统计覆盖有效通道、相对动作所需状态锚点和统计中声明的 block/stride
几何。joint 模式要求两个手臂和夹爪的动作通道完整；EEF 模式要求两组 EEF 和夹爪
完整。缺失模式不会出现在连接 metadata 的 `available_control_modes` 中。

客户端 `state_source=command` 使用 `get_obs()['joint_action']` 的 drive target；
`measured` 使用实测关节 qpos。每轮发送 `joint_action` 和 `endpose` 字典，服务端
组装 unified55，再使用训练统计归一化。返回的动作已经反归一化并完成 delta 解码，
是绝对目标。joint 排列为左臂、左夹爪、右臂、右夹爪；EEF 为每臂
`xyz + xyzw + gripper`，总宽度 16。

EEF 协议始终使用数据集坐标的 `xyz+xyzw`。客户端负责与 RoboTwin 世界坐标
`xyz+wxyz` 双向转换，服务端不重复转换。动作通过 RoboTwin 原有
`take_action(..., action_type='qpos'/'ee')` 执行。

## 请求生命周期与历史

连接成功后 server 先发送 metadata。一个 server 同时接收一个仿真客户端，其他连接
以 1013 拒绝；断线后清空 episode 状态，保留已加载模型。
server 的独立 I/O 线程每 20 秒发送 ping，20 秒未收到 pong 时关闭连接，帮助发现
异常失联的客户端；正在运行的推理会完成后再释放模型会话。
退出服务时会停止监听、关闭已有连接并等待正在执行的推理结束，再清空会话状态。

1. `reset` 请求包含 `prompt`、`control_mode` 和可选的 `record_video`。
2. 第一轮请求包含三路原始 RGB `obs`、原生 `state`、`observation_step=0`。
3. 客户端执行完整 block，逐动作采集图像。下一请求包含新的 32 帧图像、最后状态、
   `executed_steps=32`、`observation_step=32`，依此类推。
4. server 抽取第 4、8、…、32 帧，逐相机按训练的 resolution bucket、画布矩形和
   PyAV BICUBIC 预处理，再用真实观测历史和最新状态采样下一块。
5. 重复/乱序请求、不完整反馈、mask 注入、未 reset 的 prompt/mode 变更被拒绝。
   模型执行失败会丢弃会话状态并返回 `reset_required=true`，客户端需重新 reset。

推理响应包含 `action`、`execute_steps`、`observation_step`、`revision` 和
`server_timing.infer_ms`；首个响应还包含完整的 `structured_prompt`。
协议客户端也可以直接导入
[CausalPolicyClient](../../tasks/infer/robotwin/client.py#L158)。

默认缓存路径仅编码本请求新增的真实 RGB：首帧单独编码，后续 8 张抽帧图像分为两个 4 帧
chunk，保留 Wan VAE 各层因果卷积状态。得到的两个 latent 作为一个完整训练 block
联合 prefill，各层成功后提交 observed KV。Transformer text KV 每个 CFG 分支只预填一次，
video KV 跨请求复用；denoise 只计算当前 video/action/state，不重算历史投影和 FFN。

缓存按 `sampling.history` 保留最近的 latent 帧（默认 32），沿用 VeOmni 的帧预算语义；
Cosmos 参考项目的 `history` 是训练块数，二者数值不能直接等同。淘汰后保留 episode
绝对 mRoPE 时间坐标，state 和 action 从本轮实际观测边界对齐。当前 state 只用于本轮
预测和相对动作解码，不写入视觉历史；预测视频、预测动作均不作为下一轮真实反馈。
reset、断线或模型异常会清空 VAE、各 CFG 分支的 KV 和时间游标。连接 metadata 的
`kv_cache=true`、`implementation=veomni-online-policy-kv` 可用于确认服务模式。

`--no-kv-cache` 提供重新编码有界 RGB 窗口的调试路径；长 episode 裁剪时会重建首帧，
因此不作为持久缓存的严格数值基准。FP32 数值测试采用相同噪声和完整实测 episode
作为对照。BF16 因矩阵形状和核分组变化，不保证逐位一致。
不支持半 block 执行或 RTC 异步规划；提前成功或到达任务上限时，客户端直接结束，
不向 server 填补未执行的帧。

## 仿真与产物

客户端读取 `--robotwin-root` 下的 `scripts/eval_policy_xpolicylab.py`，复用其
任务配置、环境创建、instruction 和清理辅助函数，不进入原来的 RPC 循环。
不传 `--instruction` 时先执行专家过程生成文本，再以相同 seed 重建初始场景；
专家观测不会发送到模型。可用 `--instruction "..."` 覆盖并跳过该过程。

`--task-name`、`--task-config`、`--control-mode` 支持逗号分隔，顺序执行笛卡尔积。
每组输出到 `<output>/<task>/<config>/<mode>/`，包含 `metrics.json` 和
`seed<seed>/` 下的 `action_joint.npy` 或 `action_eef.npy`、`requests.json`、
`structured_prompt.json`、`info.json`、真实观测 `video.mp4` 和预测 `prediction.mp4`。
同一路径再次运行会覆盖本轮文件，应为不同实验选择不同输出目录。
metrics 记录已完成/成功轮数和 `success_rate`；不稳定 seed 与专家失败的 seed 不计入
成功率分母。`--max-steps` 单位是 `take_action()` 次数；client 场景 seed 和 server
采样 seed 各自独立。

启用视频记录时 server 只累计首个观测 latent 和每轮预测 latent，结束后收到
`finish_video` 才整段解码一次并返回 MP4。视频按 `fps/video_stride` 播放，包含首帧；
它可能包含最后一个 block 内未执行的预测。记录所需 CPU 内存随 episode 长度增长。
`--video-fps` 只控制客户端观测视频播放速度，不调度控制循环。

## 验证范围

[测试](../../tests/models/diffusers/test_causal_cosmos3_robotwin.py)使用 CPU 原生采样器、
替代 VAE、假仿真环境和真实本机 WebSocket 连接，检查跨请求 text/observed KV 复用、反馈抽帧、历史裁剪、绝对动作
重锚定、掩码、错误恢复、单连接和重连，以及 MessagePack NumPy 编码。
另有原生小模型在线/完整历史采样数值对照、CFG 分支和多次淘汰检查，以及真实小型 Wan VAE 的流式/整段编码对照。
本机使用 step 2300 DCP 真实训练权重，在 NPU 上通过本地 WebSocket 回放 adjust_bottle
`episode_index=5` 的三轮实测 RGB/state（第 0、32、64 步），验证了缓存对象跨请求复用、
一次 text prefill、3 latent 历史预算的实际淘汰和断线清理。结果记录在
`outputs/robotwin_kv_validation_20261111/session.json`。此验证不等同于 RoboTwin/SAPIEN
闭环仿真；尚未评估真实仿真成功率。
