# HA-VLN ↔ InternNav 桥接接口规格

本文档分别描述 **InternNav DualVLN 推理** 和 **HA-VLN HASimulator** 两侧的输入/输出/动作空间，
以及当前 HTTP 桥接的接口现状和已知问题，作为重新设计中间件的参考。

---

## 一、InternNav DualVLN 推理

### 1.1 动作空间

```python
class action_code(IntEnum):
    STOP     = 0
    FORWARD  = 1
    LEFT     = 2
    RIGHT    = 3
    LOOKUP   = 4   # 仅当 has_pitch_actions 时使用
    LOOKDOWN = 5   # 见下方特殊处理
```

**符号映射（LLM 输出 → action code）：**

| 符号 | action code | 备注 |
|------|------------|------|
| `STOP` | 0 | 停止 |
| `↑` | 1 (FORWARD) | 前进 |
| `←` | 2 (LEFT) | 左转 |
| `→` | 3 (RIGHT) | 右转 |
| `↓` | 5 (LOOKDOWN) | 有 lookdown 支持时 |
| `↓` | 1 (FORWARD) | 无 lookdown 支持时退化为前进 |

**LOOKDOWN 的两种模式（原版 evaluator）：**

| 条件 | 行为 | 是否 step 环境 |
|------|------|---------------|
| `has_pitch_actions` | 真实执行 2× `env.step(LOOKDOWN)` 改变俯仰角，之后 2× `LOOKUP` 恢复 | 是 |
| 两者皆无 | `↓` 被映射为 FORWARD 或用 depth fallback 选方向 | 否（作为 FORWARD step） |

**注意**：原版 evaluator 没有 `has_external_lookdown_views` 概念。
HA-VLN 服务端的外挂 lookdown 传感器数据（`--enable-lookdown-sensors`）
在原版 evaluator 中**不被使用**，可选用 `--disable-lookdown-sensors` 减少带宽。

**LOOKDOWN followup 的相机偏移**：
LOOKDOWN followup 后相机累积 -30° pitch 偏移是原版设计行为（与 R2R 训练一致），
不需要额外的恢复机制。模型对适度的 pitch 偏移有足够鲁棒性。

### 1.2 双系统架构

#### System 2（VLM 高层决策）

**模型：** `InternVLAN1ForCausalLM`（Qwen2.5-VL 架构）

**输入：**

| 输入项 | 格式 | 说明 |
|--------|------|------|
| 当前帧 RGB | PIL Image, resize 到 `(384, 384)` | 当前观测 |
| 历史帧 RGB | 最多 `num_history=8` 帧 | 通过 `np.linspace(0, step_id-1, 8)` 均匀采样 |
| 指令文本 | str | episode instruction |
| Prompt | 聊天模板 | 含 `<image>` token 占位符 |

**Prompt 模板（首次决策）：**
```
You are an autonomous navigation assistant. Your task is to <instruction>.
Where should you go next to stay on track? Please output the next waypoint's
coordinates in the image. Please output STOP when you have successfully completed the task.
These are your historical observations: <image> <image> ... .
<conjunction> <image>.
```

**Prompt 模板（lookdown followup）：**
```
You are still solving the same navigation instruction: <instruction>.
You requested a look-down view. Based only on this look-down image,
output the next waypoint's coordinates in the image, or output STOP if the task is complete.
```

**输出解析：**

| LLM 输出 | 解析结果 | 说明 |
|----------|---------|------|
| 含数字（如 `"142, 235"`） | `pixel_goal = [235, 142]` (x, y) | 进入 System 1 模式 |
| 纯方向符号（如 `"←←←←"`） | `action_seq = [2, 2, 2, 2]` | 离散动作序列 |
| `STOP` | `action_seq = [0]` | 结束 |

**生成参数：** `max_new_tokens=128, do_sample=False, use_cache=True`

#### System 1（NavDP 局部规划）

**触发条件：** `mode == "dual_system"` 且 `has_lookdown_support`，且 S2 输出了 pixel goal

**输入：**

| 输入项 | 格式 | 说明 |
|--------|------|------|
| `traj_latents` | `[B, N_QUERY, hidden_dim]` | S2 的 `generate_latents(output_ids, pixel_values, image_grid_thw)` |
| `images_dp` | `[B, 2, 224, 224, C]`, bfloat16, [0,1] | `[goal_image, current_image]` |
| `depths_dp` | `[B, 2, 224, 224, 1]`, bfloat16 | 对应深度 |

**输出：** `dp_actions` → `traj_to_actions()` → `list[int]` (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)

**轨迹到动作转换参数：**
- `step_size = 0.25 m`, `turn_angle = 15°`, `lookahead = 4`
- `MAX_STEPS = 8`（单次 pixel goal 周期最多 8 步环境动作）
- `MAX_LOCAL_STEPS = 4`（每次 S1 调用最多取前 4 个动作）

### 1.3 环境接口需求

evaluator 对 `self.env` 的调用：

| 方法 | 返回 | 说明 |
|------|------|------|
| `reset()` | `Dict[str, Any]` 或 `None` | 观测 dict 或结束信号 |
| `step(action: int)` | `(obs, reward, done, info)` | action 为 int 0-5 |
| `get_metrics()` | `Dict` | 至少含 `success`, `spl`, `oracle_success`, `distance_to_goal`, `top_down_map` |
| `get_current_episode()` | 对象 | 需有 `.scene_id`, `.episode_id`, `.instruction.instruction_text` |
| `get_capabilities()` | `Dict` | `has_external_lookdown_views` 等 |
| `.is_running` | `bool` | 控制外层循环 |
| `.episodes` | `list` | 长度用于 tqdm 进度条 |

**观测 Dict 结构：**

| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `rgb` | `(H, W, 3)` | uint8 | 前向 RGB |
| `depth` | `(H, W)` 或 `(H, W, 1)` | float32 | 深度，归一化 [0,1] |
| `lookdown_rgb` | `(H, W, 3)` | uint8 | 可选，俯视 RGB |
| `lookdown_depth` | `(H, W)` 或 `(H, W, 1)` | float32 | 可选，俯视深度 |
| `gps` | `(2,)` | float | 可选，用于 ShortestPathFollower |
| `compass` | `(1,)` | float | 可选 |

### 1.4 观测处理流程

```
RGB: obs["rgb"] → PIL Image → resize(384, 384) → 送入 S2

Depth: obs["depth"] → reshape(H, W) → filter_depth(blur=None)
       → × (max_depth - min_depth) + min_depth → × 1000 (转 mm)

NavDP 用 Depth: depth_mm → PIL I;16 → preprocess_depth_image_v2(224×224)
               → ÷ 1000 → clip(max=5.0) → 单位 m
```

---

## 二、HA-VLN HASimulator

### 2.1 动作空间

**原始 HA-VLN 配置 (`HAVLNCE_task.yaml`)：**
```yaml
ACTION_SPACE_CONFIG: v0
POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
```

**InternNav 对齐配置 (`HAVLNCE_task_internnav.yaml`)：**
```yaml
ACTION_SPACE_CONFIG: v1
TILT_ANGLE: 15
POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN]
```

| 动作 | 字符串名称 | 整数码（HTTP） | 备注 |
|------|-----------|---------------|------|
| STOP | `"STOP"` | 0 | |
| MOVE_FORWARD | `"MOVE_FORWARD"` | 1 | |
| TURN_LEFT | `"TURN_LEFT"` | 2 | |
| TURN_RIGHT | `"TURN_RIGHT"` | 3 | |
| LOOK_UP | `"LOOK_UP"` | 4 | internnav 配置新增，pitch +15° |
| LOOK_DOWN | `"LOOK_DOWN"` | 5 | internnav 配置新增，pitch -15° |

pitch 动作在 HTTP 服务端通过 `env.habitat_env.step()` 执行，绕过人类动画更新。

### 2.2 观测空间

**原始配置 → InternNav 对齐配置：**

| 传感器 | UUID | 原始分辨率 | 对齐分辨率 | 其他 |
|--------|------|-----------|-----------|------|
| RGB_SENSOR | `rgb` | 224×224, HFOV 90° | **640×480, HFOV 79°** | |
| DEPTH_SENSOR | `depth` | 256×256 | **640×480, HFOV 79°** | min=0.0, max=10.0 |
| GPS_SENSOR | `gps` | - | **(2,) float** | 新增 |
| COMPASS_SENSOR | `compass` | - | **(1,) float** | 新增 |
| INSTRUCTION_SENSOR | `instruction` | - | - | 文本指令 |
| SHORTEST_PATH_SENSOR | `shortest_path_sensor` | (1,) | (1,) | 最短路下一步 |
| VLN_ORACLE_PROGRESS_SENSOR | `progress` | (1,) | (1,) | 到目标的进度 |

**可选 lookdown 传感器（`--enable-lookdown-sensors` 时添加，v1 模式下不推荐）：**

| 传感器 | UUID | 配置 |
|--------|------|------|
| RGB_LOOKDOWN_SENSOR | `lookdown_rgb` | 与 RGB_SENSOR 同分辨率，pitch 倾斜 30° |
| DEPTH_LOOKDOWN_SENSOR | `lookdown_depth` | 与 DEPTH_SENSOR 同分辨率，pitch 倾斜 30° |

### 2.3 Human-Aware 特性

**数据路径：**
- `HUMAN_GLB_PATH`: 3D 人体模型 (HAPS 2.0 GLB 文件目录)
- `HUMAN_INFO_PATH`: 人体运动标注 JSON
- `RECOMPUTE_NAVMESH_PATH`: navmesh 缓存

**人体动画机制：**
- 子线程每 0.1s 发送 `REFRESH_HUMAN_MODEL` 信号
- `step()` 中 `_handle_signals()` 更新人体姿态
- 每人体 120 帧 SMPL 序列循环播放

### 2.4 指标

**标准 VLN 指标：**

| 指标 | 说明 |
|------|------|
| `distance_to_goal` | 到目标欧氏距离 (m) |
| `success` | 是否在 3m 内到达 (0/1) |
| `spl` | Success weighted by Path Length |
| `oracle_success` | 路径上任意点是否曾到达目标 |
| `path_length` | 实际路径长度 (m) |
| `steps_taken` | 步数 |
| `collisions` | `{count, is_collision}` |
| `collisions_detail` | `{count, is_collision, steps: [bool]}` |

**Human-Aware 指标：**

| 指标 | 说明 |
|------|------|
| `distance_to_human` | `{viewpoint: [distance, angle]}` |
| TCR | `max(0, collisions - baseline_collisions)` |
| CR | `min(TCR, 1)` |
| SR | `success × (TCR == 0)` |

---

## 三、HTTP 桥接现状（原版 HEAD）

### 3.1 服务端 (`havln_http_env_server.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metadata` | GET | `{total_episodes, max_episodes, capabilities}` |
| `/reset` | POST | 返回初始观测或 `{finished: true}` |
| `/step` | POST | `{action: 0-5}` → 返回新观测 |
| `/close` | POST | 关闭环境 |

**动作映射：** 客户端发 int 0-5 → 服务端查 `ACTION_MAP` → 字符串传给 env
- 常规动作 (0-3)：通过 `env.step()` 执行（含人类动画更新）
- pitch 动作 (4-5)：通过 `env.habitat_env.step()` 执行（绕过人类动画更新）

**观测编码：**
- RGB: `PIL Image → PNG bytes → base64` (`rgb_png_b64`)
- Depth: `np.save → base64` (`depth_npy_b64`)
- GPS: `float list` (`gps`)
- Compass: `float` (`compass`)
- Lookdown (可选): 同 RGB/Depth (`lookdown_rgb_png_b64`, `lookdown_depth_npy_b64`)

**capabilities 字段：**
```json
{
  "has_external_lookdown_views": bool,
  "lookdown_degrees": float,
  "split": str,
  "has_gps": bool,
  "has_compass": bool,
  "rgb_width": int, "rgb_height": int, "rgb_hfov": int,
  "depth_width": int, "depth_height": int
}
```

### 3.2 客户端 (`havln_http_env.py`)

发 int action code 给服务端。v1 模式下 action 4/5 直接发送（由服务端处理 pitch）。
解码 obs 包括 rgb、depth、gps、compass、lookdown（如有）。

### 3.3 episode 控制

- 服务端按 `episodes` 列表顺序逐个 serve
- `--max-episodes N` 控制最大 episode 数
- `--scene-filter` / `--episode-filter` 支持按场景/episode 过滤
- 客户端调用 `reset()` 获取下一个 episode，收到 `{finished: true}` 停止

---

## 四、关键不匹配与设计要点

### 4.1 动作空间 — 已对齐

| | InternNav | HA-VLN (internnav 配置) |
|---|-----------|------------------------|
| 动作数 | 6 (含 LOOKUP, LOOKDOWN) | **6** (v1, 含 LOOK_UP, LOOK_DOWN) |
| ACTION_SPACE_CONFIG | v1 | **v1** |
| TILT_ANGLE | 15° | **15°** |
| 编码 | int 0-5 | int 0-5（HTTP）/ 字符串（内部） |

**服务端处理：** pitch 动作 (4/5) 通过 `env.habitat_env.step()` 执行，绕过
`HAVLNCEDaggerEnv._handle_signals()`，不推进人类动画帧。

**步数倍增：** 每个 evaluator 导航步额外执行 4 次 pitch step (2×LOOKDOWN + 2×LOOKUP)，
`MAX_EPISODE_STEPS` 已设为 5000 以提供足够余量。

### 4.2 观测格式 — 已对齐

| | InternNav 训练 | HA-VLN (internnav 配置) |
|---|---------------|------------------------|
| RGB 分辨率 | 640×480 | **640×480** |
| RGB HFOV | 79° | **79°** |
| Depth 分辨率 | 640×480 | **640×480** |
| Depth HFOV | 79° | **79°** |
| Depth 值域 | [0, 1], min=0.0, max=10.0 | **[0, 1], min=0.0, max=10.0** |
| GPS/Compass | 有 | **有**（服务端编码，客户端解码） |
| top_down_map | 可选（save_video 用） | HTTP 模式下不提供（None） |

### 4.3 Episode 控制

- 服务端 `--max-episodes` 控制最大 episode 数
- 服务端 `--scene-filter` / `--episode-filter` 支持按场景/episode 过滤
- 客户端收到 `{finished: true}` 自动停止
- 客户端 `max_eval_episodes` 仅影响 tqdm 进度条长度，截断依赖服务端

### 4.4 GPS/Compass — 已实现

服务端 task config 添加了 `GPS_SENSOR` 和 `COMPASS_SENSOR`，
HTTP 响应中编码 `gps` (float[2]) 和 `compass` (float)，客户端解码后注入 obs。
- `has_pose_sensors = True` → system2-only 模式的 ShortestPathFollower 可用
- dual_system 模式同样可用（NavDP 不依赖但不影响）
