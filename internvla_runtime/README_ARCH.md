# internvla_runtime 架构说明（从头说）

## 一、项目里本来有什么

- **InternNav**：视觉语言导航，模型 **InternVLA-N1-DualVLN**（system1 + system2），输入 RGB + 深度 + 指令，输出离散动作或轨迹。
- **真实世界用法**：脚本 `scripts/realworld/http_internvla_server.py` 用 **HTTP** 暴露一个接口：客户端上传一帧图像 + 深度，服务端跑模型，返回动作/轨迹。机器人端（如 Unitree Go2）跑 `http_internvla_client.py`（ROS2），把相机和里程计数据 POST 到该 HTTP 服务。

即：**原来 = HTTP server（带模型的电脑）+ HTTP client（Go2 上的 ROS2 节点）**。

---

## 二、同事加了什么（internvla_runtime）

同事加了一套 **gRPC** 方案，和 HTTP 并行存在：

- **proto**：定义两个 RPC——`InitEpisode(session_id, instruction)` 和 `Step(session_id, rgb, depth, camera_pose)`，返回动作或轨迹。
- **Python worker**：实现上述 gRPC 服务，内部还是调同一个 `InternVLAN1AsyncAgent`，等价于「把 HTTP 换成 gRPC」。
- **C++ server**（可选）：设计上是「机器人侧 C++ 收图、预处理，再通过 gRPC 调 Python worker」；当前 C++ 端仍是 TODO，未真正连上 Python。

所以：**internvla_runtime = 用 gRPC 暴露同一套 InternVLA 能力**，不是用 C++ 替代 Python，而是 C++（若启用）做前端、Python 做模型推理。

---

## 三、为什么 instruction 是客户端给的，不是服务端？

- **设计意图**：  
  - **服务端（worker）**：只负责「加载模型 + 按请求做推理」，不关心这局任务是什么。  
  - **客户端（Go2 / 任务方）**：知道「这一局要做什么」（例如「去门口」「去红色椅子」），所以由**客户端在每局开始前**通过 `InitEpisode(session_id, instruction)` 把 instruction 发给服务端；之后同一次会话的多次 `Step` 都用这条指令。

- 这样做的原因：  
  - 同一台 worker 可以服务多机、多任务，不同 session_id 不同 instruction。  
  - 任务由使用方（客户端）决定，服务端保持无状态、只做推理。

所以：**instruction 从 client 来是刻意的设计**，不是漏在 server 上配置。

---

## 四、当前推荐用法（已补全后的状态）

- **Python worker**：已补全 args（model_path、plan_step_gap、camera_intrinsic 等）、InitEpisode 存 instruction、Step 传 intrinsic + 按 session 取 instruction、warmup。可直接用。
- **C++ server**：仍是可选；若要用，需先 `gen_proto_cpp.sh`，再用 CMake 编译（已用 pkg-config 兜底 gRPC）。
- **真实世界部署**：workbench（本机）跑 **worker**（或 HTTP server），Go2 跑 **grpc_internvla_client.py**（或 http_internvla_client.py）。详见 `readme.txt` 与 **DEPLOY.md**。

---

## 五、目录与角色速览

| 路径 | 作用 |
|------|------|
| `proto/internvla.proto` | gRPC 接口：InitEpisode、Step |
| `python/worker_server.py` | Python gRPC 服务，调 InternVLA agent，**当前主用** |
| `python/client_example.py` | 最小示例客户端 |
| `cpp/server/main.cpp` | C++ 服务端（可选，目前未接 Python） |
| `readme.txt` | 生成 proto、启 worker、调用方式、Go2 客户端、C++ 构建 |
| **DEPLOY.md** | 本机 workbench + Go2 客户端部署步骤 |

---

## 六、离散动作含义与「为什么收到 [5] 不动」

模型（InternVLA-N1-DualVLN）输出的 **discrete_action** 与客户端行为对应关系如下（定义在 `internnav/agent/internvla_n1_agent_realworld.py` 的 `actions2idx`）：

| 值 | 含义 | 客户端行为 |
|----|------|------------|
| 0 | STOP | 不更新目标（相当于停） |
| 1 | ↑ 前进 | 更新目标：前进 0.25 m |
| 2 | ← 左转 | 更新目标：左转 15° |
| 3 | → 右转 | 更新目标：右转 15° |
| 5 | ↓ | **不更新目标，机器人不动** |
| 9 | （未在 agent 中定义） | **不更新目标，机器人不动** |

客户端（`grpc_internvla_client.py`）中约定：**仅当 `discrete_action` 不是 [5] 且不是 [9] 时才调用 `incremental_change_goal()`**，因此收到 [5] 或 [9] 时不会更新 `homo_goal`，底盘控制就不会产生位移/转向，看起来就像「没动」。

**如何确认：**

1. **看客户端打印**：运行 gRPC 客户端时已打印 `response` 和「含义」行，若一直显示 `[5] → ↓ (客户端视为不移动)`，说明服务端确实在返回 5。
2. **看服务端（worker）**：在 `worker_server.py` 或 agent 内打印 `output.output_action` 或模型 S2 的文本输出（如 "↓"），可确认是模型一直生成「↓」。
3. **若希望收到 5 时也动**：可修改客户端逻辑，例如把 5 映射为「后退一小步」或改为「不跳过 5、也调用 incremental_change_goal」（需在 `incremental_change_goal` 里为 5 实现具体行为）。

**为何模型可能一直输出 5**：可能原因包括当前图像下模型判断「先不动/观察」、指令与训练分布差异、或 S2 阶段偏好输出「↓」。可尝试换更简短的指令（如 "go forward"）或检查 worker 端日志里模型原始输出。

---

## 七、小结

- **internvla_runtime** = gRPC 版的 InternVLA 推理服务；**instruction 由客户端通过 InitEpisode 传入**是设计如此。
- **discrete_action**：0=STOP, 1=前进, 2=左转, 3=右转, 5/9=客户端不更新目标（机器人不动）；详见上文第六节。
- 你这边已做的修改：补全 worker 的 args 与 Step/InitEpisode 逻辑、C++ 构建与 pkg-config、Go2 gRPC 客户端与部署说明；同事原先的「args 不全、instruction 写死」等问题已修复并记录在 **changelog/2026-3-7.md**。
