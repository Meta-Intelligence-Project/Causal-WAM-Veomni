# Piper 纯异步 RTC

Piper 根据服务端配置选择串行或异步规划。Piper POC 基线使用串行规划；可选的异步实现仅参考
`origin/gy_branch@410700febc508e881520e22982289dbacfeaf2a6` 中
`serving/async_session.py` 的决策与交接规则。服务器使用原有普通前向采样，
不包含 VJP、反向求导、动作融合或旧前缀复制，也不改变采样步数、CFG、残差复用等推理参数。
纯异步减少推理导致的停顿，但不保证两批预测之间的速度或加速度连续。

## 配置与兼容条件

异步执行由服务端 YAML 的 `async_infer` 统一指定。若要开启，修改
[Piper POC 服务配置](../../configs/server/causal_cosmos3_server_poc.yaml)：

```yaml
async_infer:
  rtc_enabled: true
  rtc_delay: 16
```

| 字段 | 缺省值 | 含义 |
| --- | --- | --- |
| `async_infer.rtc_enabled` | `false` | 启用纯异步；`false` 使用串行执行路径 |
| `async_infer.rtc_delay` | `16` | 新请求发出后继续执行旧计划的动作步数 |

POC 基线、RoboTwin 服务和 Edge Policy 推理示例均设为 `false`。
Edge Policy 的此配置仅在用它启动在线服务时生效，离线预览不调度机械臂。
省略整个 `async_infer` 时使用表中缺省值；不接受 null、未知字段或错误类型。
修改后需重启服务并重新连接客户端；reset 仅清空会话历史，不更改配置。

服务端通过 metadata 的 `async_infer` 下发实际设置；`rtc_async` 仅表示服务端能力。
Piper 在连接机械臂前读取并校验 metadata，不再从客户端 JSON 或 CLI 选择 RTC。
请求不再携带 RTC 配置字段；服务端拒绝客户端覆盖，保证调度和会话模式一致。
旧客户端 JSON 中的 `rtc_enabled` / `rtc_delay` 应移到服务端上述区块，
`--rtc`、`--no-rtc` 和 `--rtc-delay` 已移除。新客户端连接未提供 `async_infer`
metadata 的旧服务时采用串行模式；旧客户端发送 RTC 字段会被新服务拒绝。

机械臂参数仍使用 [Piper 配置](../../configs/client/causal_cosmos3_piper.json)：

| 字段 | 基线配置值 | 含义 |
| --- | --- | --- |
| `smooth_method` | `"spline"` | 串行基线使用平滑；开启 RTC 时必须改为 `"none"` |
| `action_dt` | `0.02` 秒 | 配置的控制步间隔，不等于机械臂真实到位耗时 |
| `max_chunks` | `70` | 模型请求次数；`0` 表示持续运行 |

设 `H=actions_per_block`，规划间隔在客户端和服务端代码中均固定为 `S=H/2`。
`rtc_delay=d` 必须是正整数，开启 RTC 时还必须满足 `d≤S`；
例如 `H=32` 允许 `1…16`，`H=64` 允许 `1…32`。服务端启动时校验设置和能力，
客户端按 metadata 再次检查；不合法值不会自动截断。关闭 RTC 时预留步数不参与调度。

```bash
python -m tasks.client_server.client.piper \
  --config-file configs/client/causal_cosmos3_piper.json
```

异步模式要求 Policy、streaming 观测、KV cache、CPU/CUDA、grouped/dense attention，
且关闭 NPU graph；`H` 必须为偶数，`S` 必须对齐 `actions_per_latent`。
Piper 仅支持 joint 控制、连续执行、无插值平滑、无预测视频导出。
不支持的组合会报错；EEF、逐轮交互、轨迹平滑或 RoboTwin 整块循环须在服务端
设 `async_infer.rtc_enabled: false`。RoboTwin 通用循环连接 RTC 服务会在执行前报错，
不会擅自切换服务端模式。

更改 prompt 前必须 reset；reset 会清空任务线程及服务端观测历史。
连接重试和 metadata 接收会检查停止文件；串行和异步模式的 reset、首次推理、
后续推理及块末上传等待均响应停止请求。
停止或异常退出时先关闭连接以解除阻塞的 RPC，再取消排队请求并回收线程；正常结束保持排空顺序。
预测视频仅在正常结束时导出，导出期间仍响应停止文件和信号；停止时不再发起新请求。
`report.json` 的 `async_infer` 记录本次连接实际使用的服务端设置。

## 执行流水

`H=32、S=16、d=16` 时，以下动作编号从 1 开始，决策时刻按已完成步数计：

```text
决策时刻         0              16               32               48
执行动作         ├── A[1:16] ───┼── A[17:32] ────┼── B[17:32] ────┼── C[17:32] ──>
模型请求         A（首次等待）   B（后台推理）      C（后台推理）      D（后台推理）
观测快照         初始状态       已执行16步        已执行32步        已执行48步
交接                             执行旧动作16步    使用B后半段        使用C后半段
```

1. 首次请求 A 必须等待完成，随后执行 A。
2. 完成第 16 步时，用实际状态和刚执行的 16 步相机反馈提交 B。
   单个后台 RPC 线程负责请求，主线程继续执行 A 剩余 16 步。
3. 完成第 32 步时取 B。B 的前 16 步对应已经执行过的时间区间，直接跳过；
   执行 B 后 16 步，同时用第 32 步的真实观测提交 C。
4. 依次重复。如果回复未到，等待回复后继续，不重复发动作、不快进跳过更多动作。
   该等待可能造成实际机械臂停顿；模拟执行不能证明真实运动连续性。

`d<S` 时，在每个规划窗口内仍于第 `S` 步提交下一请求，继续执行旧计划 `d` 步后交接。
首批执行 `S+d` 步，中间各批执行 `S` 步，末批执行 `H-d` 步；
单批运行则执行完整 `H` 步。因此 `N≥1` 次请求累计执行 `H+(N-1)S` 步，
`H=32` 时 15 次请求是 256 步，480 步需要 29 次请求；关闭 RTC 则 15 次请求为 480 步。

## 观测编码与 KV cache

流式 VAE 只接收一次实际相机时间序列：首帧初始化，之后每次决策推进 `S` 步。
网络按 `video_stride` 选择观测端点，但控制步计数不变。
`H=32、video_stride=4、temporal_compression=4` 时，每 16 步反馈 4 帧，得到 1 个新 latent。

Transformer 的完整块历史保持两个相位：决策步 `0,32,64,…` 与 `16,48,80,…`。
每个相位再次被使用时，追加过去 `H` 步的真实观测 latent，并按原 history 上限淘汰历史。
预测动作和预测视频不会写入测量 KV；两个相位共用一次 VAE 编码结果，CFG 分支拥有各自 KV。
更改 `d` 不改变观测或 KV 的刷新频率。任意推理失败后必须 reset，避免继续使用部分更新的缓存。

实现入口：[客户端协议](../../tasks/client_server/client/piper/protocol.py#L119)、
[服务端会话](../../tasks/client_server/server/session.py#L240)、
[客户端启动与配置](../../tasks/client_server/client/piper/__main__.py#L78)。

## 单元测试覆盖

单元测试覆盖开关、配置边界、不同 H/d 的无遗漏无重复执行、迟到回复、停止处理、
观测分组与相位 KV 推进、CFG/fixed-step 普通采样，以及串行回退。

## 本机 Cobot 网页接入

桌面“真机网站”仍由 `/home/agilex/cobot_magic/start_visualization.sh` 启动，
页面地址为 `http://localhost:5000/`。网页的 Cosmos 面板通过
[集成后端](../../scripts/causal_cosmos3/web_integration/cosmos_web.py)
启动当前仓库的 `server.sh` 和 `piper.sh`，不再启动旧 Cosmos 推理代码。
网页自身使用 aloha 环境；模型使用 `infer-wam-veomni`，Piper 使用 `piper_new`。
Piper 仍依赖本机 `lingbot-va-piper/client` 中的硬件驱动适配。

`MODEL_CONFIGS` 显式绑定 checkpoint 与服务 YAML；目前绑定
`ckpts/1104-command-32delta` 和 `configs/server/causal_cosmos3_server_poc.yaml`。
添加模型时必须核对其统计文件、Tokenizer、VAE 和动作几何，不能仅扫描权重目录后
自动套用其他模型的配置。旧版 DCP manifest 不用于新版网页的模型发现。

网页保留已保存的任务、速度、平滑及夹爪设置，并将旧版
`action_representation=relative, execute_steps=32` 配置读为新版设置。
其他动作表示或截短执行不会被静默迁移。RTC 开关和延迟保存于网页 `server` 字段，
启动时写入服务配置的 `async_infer`；默认取绑定 YAML 的值。
采样步数必须覆盖 YAML 中 `residual_reuse_full_steps` 的最大索引。

每次启动均在 `/home/agilex/cobot_magic/logs/cosmos/server-*.json` 保存有效配置
（JSON 也是合法 YAML），并通过绝对路径传给服务端。网页修改不会改变运行中的快照。
服务日志为 `server.log`，客户端日志为 `client.log`；任务结果目录的
`inference_config.json` 同时保存网页配置与实际服务配置。

网页通过 PID、进程启动时刻和系统启动 ID 记录服务归属，重启网页后仍可停止自己
启动的服务；外部终端启动的 VeOmni 服务仍由原终端停止。
外部服务日志约定为 `veomni_server.log`。网页不会为状态检查建立模型 WebSocket。
旧 Cosmos 进程仍参与占用检测，但不会作为新版 Piper 的执行目标。

安装命令：

```bash
python scripts/causal_cosmos3/web_integration/install.py
```

安装器先验证目标文件与已记录基线一致，再备份并替换三个集成文件。
备份位于 `/home/agilex/cobot_magic/outputs/veomni-piper-web-*`。
安装后须在推理、采集、归零等操作均停止时重启网页服务并刷新页面。
恢复备份中的同名文件及原网页配置、重启网页即可回滚。

RTC 与客户端执行参数必须联合校验：`smooth_method=none`、关节控制、
`save_prediction_video=false`。网页保存配置时拒绝不兼容组合，并在启动客户端前
依据实际运行服务的 RTC 设置再次检查（包括外部服务）。页面显示冲突并禁用执行按钮。
从旧网页迁移时，若要保留 spline/EEF/预测视频功能，应关闭 RTC 并重启模型服务；
若继续使用 RTC，则显式关闭这些功能。`piper.sh --check` 仅验证本地配置和依赖，
不读取服务端 metadata，不能代替这项联合检查。

## 已选定的真机基线（2026-11-23）

服务端使用 `causal_cosmos3_server_poc.yaml`，客户端使用
`causal_cosmos3_piper.json`，权重为 `ckpts/1104-command-32delta`。
此基线由用户选定，后续实验应明确记录相对于它的改动。

- 串行执行：`rtc_enabled=false`。
- 10 步去噪：`steps=10`；residual reuse 完整刷新索引为 `[0,3,6,9]`（从 0 开始）。
- spline 平滑：`smooth_factor=0.1`、`v_max=3`。
- MOVE_J 速度比例 `0.25`、`action_dt=0.02` 秒、到位容差 3 度。
- 模型动作频率 30 Hz；左右夹爪系数均为 1.0，关闭阈值 0.02。
- 最多 70 次预测请求，不保存预测视频。

仓库客户端配置已与本机网页保存的任务和运动参数对齐。网页已有保存配置会覆盖
YAML 中的可编辑采样/RTC 参数及客户端默认值；恢复基线时需同时核对网页保存值。

## 推理与真机执行耗时

Piper 任务结束、普通异常或收到停止请求后，在结果目录写入 `piper_timings.json`，
`report.json` 包含其路径及汇总，客户端日志打印 `PIPER_TIMINGS`。
计时事件暂存内存，关闭后台请求后统一保存，不在每个动作上写文件；强制杀进程不能保证保存。
旧版 `timings.json` 的 `server_infer_s` 和 `action_exec_s` 也会填入有效记录。

- `request_submitted`：客户端提交请求的时间；`request_completed` 包含客户端往返耗时
  （封包、网络、服务处理、解包）和服务端 `infer_ms`。后者为整个请求处理，非单独 denoise 内核计时。
- `command`：每条控制指令发送前的时间；`arrival_wait`：到位等待耗时及结果。
- `execution_start/end`：本轮执行段边界，包含发送、观测、等待到位和执行节流，
  不包含首次请求等待或动作文件保存。中途停止的执行段只保留已有事件，不伪造完成时间。
- `handoff_due`：旧计划执行完成、最后到位等待结束的时刻；`handoff_received`：主线程拿到
  新预测的时刻及实际等待 future 的耗时。前者不包含之后的日志、最后观测及其他应用开销，
  防止用这些额外延迟掩盖模型预测迟到。

对每次非首次且已有完成记录的 RTC 交接，比较：

```text
request_available_ms = 请求完成时刻 − 提交时刻（包含线程排队）
execution_overlap_ms = 旧计划执行完成时刻 − 提交时刻
slack_ms = execution_overlap_ms − request_available_ms
uncovered_inference_ms = max(0, -slack_ms)
```

`slack_ms >= 0` 表示该次预测及时返回；负值表示推理未被执行时间完全覆盖。
查看每次交接、迟到比例和 P95，不只比较两个平均值。
首轮必须等待、未完成请求和没有执行截止点的请求不计入交接判定；原始事件仍保存。
`command_interval_ms` 包括跨块间隔，可检查实际停顿；到位等待变长也会扩大重叠窗口，
因此及时返回不等价于运动平滑或真实频率达标。

仓库默认仍是上节串行基线。网页的 RTC 实验需将 `server.rtc_enabled` 改为 `true`，
`client.smooth_method` 改为 `none`，保存后重启模型服务。
