# RoboTwin IDM 闭环测试

服务端在 VeOmni 推理环境加载模型，客户端在已有的 RoboTwin/SAPIEN 仿真环境运行。
两端通过 WebSocket 交换 MessagePack/NumPy 数据，可以部署在不同机器。
本接入移植 [PR #21](https://github.com/Meta-Intelligence-Project/Causal-WAM-Veomni/pull/21)
的协议、Unified55 adapter 和客户端，推理计算复用本分支的
[原生观测会话](causal_cosmos3_inference.md#real-feedback-session)。

## 推理方式

默认任务为 `inverse_dynamics`，每个 block 先独立预测视频，再固定生成的视频预测动作。
视频阶段只读取文本、真实视觉历史和视频噪声，不调用 Policy 先验，不输入动作或 state。
动作阶段使用本分支训练时的可见性，state 可以读取当前动作；不缓存固定的 state KV。
`--task policy` 则联合生成视频和动作，使用同一套真实反馈会话。

当前使用窗口 VAE 编码，默认 `sampling.kv_cache: false`。
可用 `--kv-cache` 开启缓存：每轮观测编码后，
只预填充文本和真实历史视频的 KV，并在该轮视频/动作去噪与各自的 CFG 分支内复用。
当前视频、动作和 state 仍逐步计算，保留本分支的 attention 规则。
新观测到来后重建缓存，不跨滑动窗口复用；真实 RGB 历史和 VAE 编码方式保持不变。
`--no-kv-cache` 使用完整重算参考路径；`observation_mode: streaming` 仍会在加载权重前报错。
训练的数据管线和 loss 不变，缓存不增加模型参数。详见 [KV cache 设计](../design/causal_cosmos3_kv_cache.md)。
真实 step-4000 权重的 BF16 回放虽然加速约 1.8 倍，但关节最大差为 0.022 rad，
视频 latent 也未通过预设误差检查，因此保留为显式开启的选项，需用目标 checkpoint 验证。

## 安装与配置

在两端各自激活的环境中安装传输依赖：

```bash
python -m pip install -r tasks/client_server/requirements.txt
```

客户端另外需要能运行 RoboTwin 的 SAPIEN、NumPy、PyYAML、OpenCV，以及支持
`libx264` 的 FFmpeg。客户端不导入 VeOmni 或 Torch。当前 runner 复用 RoboTwin checkout
中的 `scripts/eval_policy_xpolicylab.py` 任务、配置和指令生成 helper。

[服务配置](../../configs/server/causal_cosmos3_server.yaml) 与
[本分支 IDM 离线配置](../../configs/infer/causal_cosmos3_idm_robotwin.yaml)保持一致：
55D 接口、32 actions/block、video_stride=4、时间压缩=4、64 latent 历史、20 步采样、
plain prompt，以及匹配的 tokenizer/VAE/statistics 路径。在线 action/control FPS 为 30，
模型图像 FPS 为 7.5。更换 checkpoint 时须确认对应的训练资源和 prompt 格式。

服务接受完整 native/HF checkpoint、VeOmni `global_step_N`，或内部 `model/ckpt`
目录。原生 DCP 自动查找该次训练的 `model_assets/config.json`，也保留推理 YAML 的
`model_config` 显式覆盖能力，无需向 checkpoint 写入额外文件。

## 启动服务端

在仓库根目录、已激活的 Torch/CANN 环境运行：

```bash
CHECKPOINT_PATH=/path/to/run/checkpoints/global_step_4000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
HOST=0.0.0.0 PORT=8000 \
bash scripts/causal_cosmos3/server_scripts/server.sh
```

可通过 `PYTHON_BIN` 选择当前机器的推理 Python，通过 `INFER_CONFIG` 选择其他 YAML。
也可直接运行：

```bash
python -m tasks.client_server.server.server \
  --config configs/server/causal_cosmos3_server.yaml \
  --checkpoint /path/to/run/checkpoints/global_step_4000 \
  --host 0.0.0.0 --port 8000 --task inverse_dynamics
```

`--fps`、`--steps`、`--history`、`--seed`、`--statistics`、`--device`、`--dtype` 和
`--attention-backend`、`--kv-cache` / `--no-kv-cache` 支持显式覆盖。
使用单进程 Python，不使用多 rank torchrun。
缺失或无效 FPS 会在读取权重前报错；`--adapter-config` 可提供 JSON 格式的精确
state/action masks 和双臂 DOF，默认从 statistics 的一致源 mask 中读取。

## 启动 RoboTwin 客户端

在仿真机器激活已有 RoboTwin 环境，然后运行本仓库入口：

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
SERVER_URI=ws://MODEL_SERVER_IP:8000 \
TASK_NAME=adjust_bottle TASK_CONFIG=demo_clean \
CONTROL_MODE=joint STATE_SOURCE=command \
EPISODES=1 MAX_STEPS=65 \
OUTPUT_DIR=outputs/robotwin_idm_closed_loop \
bash scripts/causal_cosmos3/client_scripts/robotwin_client.sh
```

65 步上限可以覆盖至少两轮完整反馈；任务提前成功时会正常停止。
默认用专家 rollout 生成任务指令，并在模型执行前重建同 seed 的初始场景；专家图像不会
送入模型。仅做启动检查时，可设置 `INSTRUCTION='Adjust the bottle.'` 跳过专家指令生成。
`--task-name`、`--task-config` 和 `--control-mode` 支持逗号分隔的多组任务。

默认 `--state-source command` 匹配 processed RoboTwin 数据中的关节命令目标口径。
EEF 从世界系 wxyz 转为训练数据的共享 ALOHA 基座系 xyzw；EEF 动作执行前反向转换。
`--control-mode eef` 仅在训练 action mask 覆盖双臂完整 EEF 和夹爪时可用。
当前适配的基座变换匹配已有 processed ALOHA 数据；其他机器人需提供自己的设备适配。

## 反馈与记录

初始请求提供每个相机的一帧 RGB 和当前状态。执行完整 32 个动作后，客户端提供每个
动作执行后的真实图像和最后一个状态。服务端抽取第 4、8、…、32 张图像，拼接成
8 张新模型帧，调用原生会话的 `observe()`，再调用 `predict()`。
历史只由真实反馈更新，预测图像/动作不会自动成为下一轮观测。

`observation_step` 从 0 开始，每次递增 32，模型时间为 `observation_step / fps`。
重复、乱序、不完整反馈和中途修改指令会被拒绝。任务在块内结束时，客户端直接停止，
不发送半块反馈。采样失败后必须 reset；断开连接也会清空 episode 状态。

客户端输出包括：

- `video.mp4`：真实观测视频。
- `prediction.mp4`：生成视频预览，整段 latent 解码，可能包含末块尚未执行的预测。
- `action_joint.npy` / `action_eef.npy`：收到的完整预测块；实际执行数以 `executed_steps` 为准。
- `prompt.txt`：第一次预测实际使用的输入文本。只有 structured prompt 才额外保存 JSON。
- `requests.json`：请求步数、服务端处理与客户端往返耗时。
- `info.json` / `metrics.json`：seed、任务、执行步数、成功状态和汇总。

FPS 定义模型/数据时间轴与视频播放速度；仿真的 `take_action()` 负责实际执行和规划，
这些耗时不代表机器人能以 30 Hz 实时闭环。模型服务成功响应也不代表任务成功。
通用设备接口及调用职责见[客户端接口](causal_policy_client.md)。

## 验证

```bash
python -m pytest -q tests/models/diffusers/test_causal_cosmos3_robotwin.py \
  tests/models/diffusers/test_causal_cosmos3_inference.py
```

测试覆盖真实 WebSocket、多轮真实反馈、reset/断连/失败清理、plain prompt、55D 适配、
RoboTwin 设备循环，以及小模型在线输出与直接原生会话的逐值一致性。
假设备和小模型测试不等同于真实 SAPIEN 闭环；真实任务需使用上述两端环境另行运行。
