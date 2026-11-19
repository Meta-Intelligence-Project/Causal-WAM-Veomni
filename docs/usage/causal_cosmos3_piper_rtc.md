# Piper 纯异步 RTC

Piper 根据服务端配置选择串行或异步规划。Piper POC 服务示例启用异步规划；它仅参考
`origin/gy_branch@410700febc508e881520e22982289dbacfeaf2a6` 中
`serving/async_session.py` 的决策与交接规则。服务器使用原有普通前向采样，
不包含 VJP、反向求导、动作融合或旧前缀复制，也不改变采样步数、CFG、残差复用等推理参数。
纯异步减少推理导致的停顿，但不保证两批预测之间的速度或加速度连续。

## 配置与兼容条件

异步执行由服务端 YAML 的 `async_infer` 统一指定，例如
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

POC YAML 显式设为 `true`；RoboTwin 服务和 Edge Policy 推理示例设为 `false`。
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

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `smooth_method` | `"none"` | 异步时每个模型动作对应一条控制指令 |
| `action_dt` | `1/30` 秒 | 控制步的最低执行周期，不等于机械臂真实到位耗时 |
| `max_chunks` | `15` | 模型请求次数；`0` 表示持续运行 |

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
