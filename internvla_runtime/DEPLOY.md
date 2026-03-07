# 真实世界部署：Workbench（服务器） + Unitree Go2（客户端）

## 拓扑

```
[ 本机 Workbench ]                     [ Unitree Go2 ]
  - 跑 InternVLA 模型                    - 相机 + 底盘 + ROS2
  - 暴露 gRPC（或 HTTP）                 - 跑客户端节点，发图/位姿，收动作/轨迹
         ↑                                      |
         |         同一局域网（WiFi/有线）        |
         +--------------------------------------+
```

- **服务端**：放在本机（workbench），有 GPU、能跑模型。
- **客户端**：在 Go2 上或与 Go2 同网的一台机子上，跑 ROS2 节点，把相机和 odom 发给服务端，把收到的动作/轨迹发给底盘。

---

## 方式一：gRPC（推荐）

### 1. Workbench（本机）

在项目根目录执行（端口可改，与客户端一致即可）：

```bash
cd /path/to/InternNav
# 首次或改过 proto 后
bash internvla_runtime/gen_proto.sh

# 启动 gRPC worker（默认端口 50052，若用 5801 则加 --port 5801）
python internvla_runtime/python/worker_server.py \
  --model_path checkpoints/InternVLA-N1-DualVLN \
  --port 5801
```

确保本机防火墙放行 5801（或你用的端口），且 Go2 能访问本机 IP（如 `192.168.1.100`）。

### 2. Go2（客户端）

在 **Go2 的 ROS2 工作空间** 下（含 `controllers`、`thread_utils`、`cv_bridge` 等依赖）：

```bash
# 指定 workbench 的 IP 和端口，以及本局导航指令
python scripts/realworld/grpc_internvla_client.py \
  --grpc 192.168.1.100:5801 \
  --instruction "Walk to the door and stop"
```

- `192.168.1.100` 换成你 workbench 的实际 IP。
- `--instruction` 可省略，会用脚本内默认指令。

### 3. 检查

- Workbench：终端里看到 “InternVLA Python worker listening on [::]:5801” 且无报错。
- Go2：能收到相机和 odom 时，客户端会周期性打 “idx: … after grpc …” 和 response；底盘应随动作/轨迹运动。

---

## 方式二：HTTP（与原有方式一致）

### 1. Workbench（本机）

```bash
cd /path/to/InternNav
python scripts/realworld/http_internvla_server.py \
  --model_path checkpoints/InternVLA-N1-DualVLN
# 默认监听 0.0.0.0:5801
```

### 2. Go2（客户端）

在 Go2 的 ROS2 工作空间下，用原来的 HTTP 客户端（需把代码里的 `url` 改成 workbench 的地址，例如 `http://192.168.1.100:5801/eval_dual`），或使用已配置好该 URL 的 `http_internvla_client.py`。

---

## 端口与地址小结

| 角色 | 默认端口 | 说明 |
|------|----------|------|
| gRPC worker | 50052（可 `--port 5801`） | 与客户端 `--grpc <workbench_ip>:5801` 一致即可 |
| HTTP server | 5801 | 客户端 POST 到 `http://<workbench_ip>:5801/eval_dual` |

---

## 常见问题

- **Go2 连不上 workbench**：检查同一网段、防火墙、workbench IP 是否写对。
- **instruction 怎么改**：gRPC 用 `--instruction "…"`；或改 `grpc_internvla_client.py` 里的 `DEFAULT_INSTRUCTION`。
- **模型路径**：两方式都建议用 `checkpoints/InternVLA-N1-DualVLN`（带 system1 的 DualVLN）。
