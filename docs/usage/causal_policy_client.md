# 通用策略客户端与设备接口

[客户端 runtime.py](../../tasks/client_server/client/runtime.py)提供 WebSocket 通信、
通用 episode 循环、动作校验、请求计时和文件记录。它不导入 RoboTwin、SAPIEN 或 Torch。
设备接入通过 [RobotDevice](../../tasks/client_server/client/runtime.py#L224) 协议完成；
具体设备可以按结构实现这四个方法，无需继承或修改通用循环。

## 设备负责的接口

| 方法 | 实现方与输入输出约定 | 调用位置与效果 |
| --- | --- | --- |
| [observe()](../../tasks/client_server/client/runtime.py#L227) | 设备实现，返回 `(images, state)`。图像为 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路 RGB uint8 `[H,W,3]`；state 为 unified55 协议的 `joint_action/endpose` 字典。图像应脱离渲染器缓冲区，状态应与这次观测对应。 | 通用循环在[初始时](../../tasks/client_server/client/runtime.py#L351)和[每个动作后](../../tasks/client_server/client/runtime.py#L386)调用，得到实际反馈；下一次请求使用完整动作块的图像序列与最后一个状态。 |
| [execute(action)](../../tasks/client_server/client/runtime.py#L231) | 设备实现，接收一个绝对动作目标。joint 顺序为左臂、左夹爪、右臂、右夹爪；EEF 为左右各一组 `xyz+xyzw+gripper`，共 16 维。控制模式须与 client 一致。 | 通用循环[逐目标调用](../../tasks/client_server/client/runtime.py#L385)。设备负责将协议坐标和控制方式转换为自身接口；返回后，下一次 `observe()` 应读取执行后的结果。 |
| [is_done()](../../tasks/client_server/client/runtime.py#L235) | 设备实现，返回 bool，统一决定成功、步数上限等停止条件。 | 通用循环在[初始时](../../tasks/client_server/client/runtime.py#L350)、[每个动作后](../../tasks/client_server/client/runtime.py#L390)和[动作块结束时](../../tasks/client_server/client/runtime.py#L396)调用。中途结束时不补齐动作或观测，也不发送半块反馈。 |
| [is_success()](../../tasks/client_server/client/runtime.py#L239) | 设备实现，返回当前任务是否成功。 | 通用循环[构建最终结果时](../../tasks/client_server/client/runtime.py#L417)调用；若触发 EEF 低运动主动失败，最终 `success` 为 false。 |

坐标变换由设备拥有：服务端使用训练数据坐标的 EEF `xyz+xyzw`，
`observe()` 转换输入，`execute()` 转换输出。服务端和通用循环不重复进行世界/基座转换。
真实设备还需自行实现动作执行等待和图像/状态同步；视频播放 FPS 不负责控制调度。

## 通用 episode 流程

[run_episode()](../../tasks/client_server/client/runtime.py#L321) 是框架提供的循环，
设备通过上述接口接入，无需复制该函数：

1. 使用固定的 `prompt` reset 服务端会话；连接和设备环境的创建、关闭由调用者负责。
2. 若设备尚未结束，读取初始图像和状态，发送 `observation_step=0`。
3. 接收动作块，经 [validate_action_chunk()](../../tasks/client_server/client/runtime.py#L278)
   检查执行步数、动作形状和有限值，记录 `infer_ms` 与 `client_round_trip_ms`。
4. 保存协议坐标下的完整预测动作块。可选执行
   [detect_eef_low_motion()](../../tasks/client_server/client/runtime.py#L298)：
   每臂分别计算相对预测块首帧的最大位置、旋转和夹爪偏差；仅当两臂均低于
   0.01 米、3 度和 0.1 的独立阈值时提前判失败。旋转比较对 q/-q 符号不敏感。
5. 对动作块逐目标处理：
   - 调用设备的 [execute()](../../tasks/client_server/client/runtime.py#L385)。
   - 调用设备的 [observe()](../../tasks/client_server/client/runtime.py#L386) 获取执行后的图像和状态。
   - 调用设备的 [is_done()](../../tasks/client_server/client/runtime.py#L390) 决定是否停止；完整执行后反馈所有新图像和最后的状态。
6. 结束后保存预测视频和 `requests.json`，返回执行步数、成功状态、耗时、记录与文件路径。

`client_round_trip_ms` 覆盖 `client.infer(request)` 的完整时间，包括封包、发送、
服务端处理、响应接收和解包；观测读取、动作执行和本地文件保存不在这段计时内。
reset 和最终预测视频请求不作为动作预测记录写入 `requests.json`。

`InferenceClient.reset()` 发送 `reset=True`；服务端的
[`CausalInferenceSession.reset()`](../../tasks/client_server/server/session.py#L100)
替换当前 episode 的观测序列，丢弃所有 CFG 分支的文字/历史视频 KV、流式 VAE
特征、历史 RGB/latent、录像 latent 和进度计数。IDM 单次采样的视频/状态 KV
仅存在于采样调用中，不跨请求保存。模型权重仍驻留；底层设备内存池也可能保留
已释放空间。reset 响应确认清理完成，新观测到来后才重新编码、构建 KV。
这不会重置客户端的机器人环境，设备环境的 reset 由调用方负责。

## 记录与结果

设置 `output` 后，通用循环保存：

- `action_joint.npy` 或 `action_eef.npy`：完整预测块，最后一块可能包含尚未执行的目标。
- `video.mp4`：初始观测和每个已执行目标后的三视角拼接视频。
- `prediction.mp4`：服务端生成预览；支持整段 latent 解码，也兼容逐块 RGB 响应。
- `structured_prompt.json`：服务端首次响应提供的结构化提示。
- `requests.json`：动作预测请求的边界步数和服务端/客户端耗时。

函数同时返回 `requests` 列表和 `requests_path`，调用者可保留文件路径并自行生成
任务专属的 `info.json`、`metrics.json`。`output=None` 时不写文件，仍返回请求记录。
[record_actions()](../../tasks/client_server/client/runtime.py#L268) 在退出时保存已经收到的动作块，
包括执行过程中发生异常的情况；异常不会被通用循环吞掉。

## RoboTwin 接入示例

RoboTwin 的 [RoboTwinDevice](../../tasks/client_server/client/robotwin/simulator.py#L137)
实现四个设备方法。[rollout()](../../tasks/client_server/client/robotwin/simulator.py#L165)
保留现有入口签名，创建这个设备后调用通用循环。
任务创建、专家指令生成、seed、环境清理和成功率汇总仍由 RoboTwin runner 管理。

已创建并初始化 `env` 后，可以这样接入：

```python
from tasks.client_server.client.runtime import InferenceClient, run_episode
from tasks.client_server.client.robotwin.simulator import RoboTwinDevice

device = RoboTwinDevice(env, control_mode="joint", state_source="command")
with InferenceClient("ws://127.0.0.1:8000", control_mode="joint") as client:
    result = run_episode(
        device,
        client,
        prompt=env.get_instruction(),
        output="outputs/robotwin_episode",
    )
```

现有 [RoboTwin 客户端测试](../../tests/models/diffusers/test_causal_cosmos3_robotwin.py)
通过 `simulator.rollout()` 调用该通用流程，覆盖逐动作实际图像反馈、末步状态及块内提前结束。
完整启动和产物说明见 [RoboTwin 使用文档](causal_cosmos3_robotwin.md)。
