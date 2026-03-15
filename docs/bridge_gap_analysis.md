# HA-VLN ↔ InternNav 差距分析与对齐方案

本文档基于 `bridge_interface_spec.md` 中已记录的两侧接口规格，
系统梳理所有已知不匹配项，并给出逐项对齐方案。

---

## 一、差距总览

| # | 差距项 | InternNav (R2R 训练) | HA-VLN (当前) | 影响程度 | 对齐方案 |
|---|--------|---------------------|--------------|---------|---------|
| G1 | RGB 分辨率 | 640×480 | 224×224 | **致命** | 方案 A |
| G2 | HFOV | 79° | 90° | **致命** | 方案 A |
| G3 | Depth 分辨率 | 640×480 | 256×256 | **严重** | 方案 A |
| G4 | Depth 范围 | min=0.0 max=10.0 | 无显式配置 | 中等 | 方案 A |
| G5 | 动作空间 | v1 (6 动作含 pitch) | v0 (4 动作) | **严重** | 方案 B |
| G6 | GPS/Compass | 有（lab_sensors） | 无 | 低\* | 方案 A 附带 |
| G7 | Episode 控制 | 客户端驱动 | 服务端 `--max-episodes` | 低 | 服务端已支持 |
| G8 | top_down_map | 可选（save_video 用） | HTTP 模式不传输 | 无\*\* | 暂不处理 |

\* GPS/Compass 仅在 system2-only 模式下影响 ShortestPathFollower；dual_system 模式用 NavDP 不需要。
但为完整性仍建议在 HTTP 桥接中传输。

\*\* top_down_map 仅用于 save_video 可视化，不影响模型推理。

---

## 二、G1-G4：传感器配置不匹配（根因分析）

### 2.1 问题本质

InternNav 的 InternVLA-N1-DualVLN 模型训练于 `vln_r2r.yaml` 配置：

```yaml
# InternNav 训练配置
rgb_sensor:   {width: 640, height: 480, hfov: 79}
depth_sensor: {width: 640, height: 480, hfov: 79, min_depth: 0.0, max_depth: 10.0}
```

评估时图像处理链路：
```
训练: 640×480 RGB → resize(384,384) → S2 模型     [下采样，保留细节]
HA-VLN: 224×224 RGB → resize(384,384) → S2 模型   [上采样，图像模糊！]
```

### 2.2 FOV 的影响

79° 和 90° HFOV 的视角差异导致：
- 相同场景中物体的表观大小不同（90° 下物体更小更远）
- 空间几何关系改变（边缘畸变程度不同）
- 深度值与像素位置的映射关系变化（影响 NavDP 的 pixel→3D 投影）

### 2.3 深度传感器问题

- 分辨率不匹配（256×256 vs RGB 的 224×224），存在对齐误差
- 无显式 min_depth/max_depth 配置，依赖默认值

### 2.4 直接后果

S2 (VLM) 在 HA-VLN 图像上：
- **无法识别** 训练时学会的视觉 landmark → 退化为只输出离散动作 (←→↑)
- **不输出 ↓ (LOOKDOWN)** → 不触发 lookdown followup prompt
- **不输出坐标** → S1 (NavDP) 永远不被调用
- pixel_goal_ratio ≈ 0%，完全退化为 system2-only 离散控制

---

## 三、G5：动作空间不匹配

### 3.1 现状

| | InternNav | HA-VLN |
|---|-----------|--------|
| config | `action_space_config: v1` | `ACTION_SPACE_CONFIG: v0` |
| 动作数 | 6: STOP/FWD/LEFT/RIGHT/LOOKUP/LOOKDOWN | 4: STOP/FWD/LEFT/RIGHT |
| tilt_angle | 15° | 无 |

### 3.2 当前 workaround

HTTP 服务端通过 `--enable-lookdown-sensors` 添加固定 30° 俯视传感器。
evaluator 检测 `has_external_lookdown_views=True`，用外挂传感器数据代替物理 pitch 动作。

### 3.3 差异

**训练时 (has_pitch_actions=True)：**
```
每个非 LOOKDOWN 步骤中:
  1. env.step(LOOKDOWN) × 2 → 相机倾斜 30°
  2. 获取俯视图（物理渲染）
  3. env.step(LOOKUP) × 2 → 相机恢复水平
```

**HTTP 桥接 (has_external_lookdown_views=True)：**
```
每个步骤中:
  1. 观测已包含固定角度的 lookdown_rgb/lookdown_depth
  2. 不需要额外的 env.step
```

两者在 **相同传感器参数** 下应产生相同图像（相机位置和倾斜角一致）。
但如果传感器参数不同（G1-G4），lookdown 图像也会不同。

### 3.4 为什么仍需方案 B

虽然外挂传感器在理论上等效，但方案 B (v1 动作空间) 可以：
1. 使 `has_pitch_actions=True`，与训练 evaluator 路径完全一致
2. 避免外挂传感器可能的细微差异（如传感器注册顺序、初始化时机）
3. 但需要处理 **人类动画加速问题**（每个 pitch step 都会触发 `_handle_signals`）

---

## 四、G6：GPS/Compass

### 4.1 HA-VLN Habitat 支持情况

已验证 HA-VLN 的 Habitat 版本内建了标准 GPS/Compass 传感器：

```python
registry.get_sensor('GPSSensor')     → EpisodicGPSSensor (uuid='gps')
registry.get_sensor('CompassSensor') → EpisodicCompassSensor (uuid='compass')
```

只需在 task config 的 SENSORS 列表中添加即可激活。

### 4.2 实现方式

**HA-VLN 侧：**
- task config 添加 `GPS_SENSOR, COMPASS_SENSOR` 到 `TASK.SENSORS`
- HTTP server 在 obs_payload 中编码 `gps` 和 `compass`

**InternNav 侧：**
- `havln_http_env.py` 解码 gps/compass（极小改动，数行代码）
- `vln_ha_r2r.yaml` 添加 lab_sensors 声明

### 4.3 优先级

- dual_system 模式：**不需要**（NavDP 不依赖 GPS/Compass）
- system2-only 模式：**需要**（ShortestPathFollower 依赖）
- 建议：在方案 A 中顺带实现，成本极低

---

## 五、G7：Episode 控制

### 5.1 现状

- 服务端 `--max-episodes` 控制最大 episode 数 → **已可用**
- 客户端 `max_eval_episodes` 仅影响 tqdm 进度条长度，不真正截断循环 → **已知 bug**
- Episode 列表由 Habitat dataset split 决定

### 5.2 解决方案

**服务端控制**（推荐，不改 InternNav）：
- 使用 `--max-episodes N` 直接限制服务端 serve 的 episode 数
- 客户端收到 `{finished: true}` 自动停止

**场景/Episode 过滤**（可选增强）：
- 在服务端添加 `--scene-filter` / `--episode-filter` 参数
- 按 scene_id 或 episode_id 过滤 episode 列表

---

## 六、对齐方案详情

### 方案 A：传感器对齐 + GPS/Compass（Phase 1）

**原则：改 HA-VLN，不动 InternNav 核心代码**

**HA-VLN 改动：**

1. 新建 `HASimulator/config/HAVLNCE_task_internnav.yaml`
   - 基于 `HAVLNCE_task.yaml`
   - RGB: 640×480, HFOV 79°
   - Depth: 640×480, HFOV 79°, min=0.0, max=10.0
   - SENSORS 添加 GPS_SENSOR, COMPASS_SENSOR

2. 新建 `agent/config/internnav_bridge.yaml`
   - `BASE_TASK_CONFIG_PATH` 指向新 task config

3. 修改 `scripts/havln_http_env_server.py`
   - obs_payload 中编码 gps/compass
   - 默认使用 `config/internnav_bridge.yaml`

**InternNav 改动（仅配置 + 桥接 env）：**

4. 更新 `scripts/eval/configs/vln_ha_r2r.yaml` — 传感器参数对齐
5. 更新 `internnav/env/havln_http_env.py` — 解码 gps/compass（约 5 行代码）

### 方案 B：v1 动作空间（Phase 2）— 已实施

**HA-VLN 改动（已完成）：**

1. `HAVLNCE_task_internnav.yaml`：
   - `ACTION_SPACE_CONFIG: v1`
   - `TILT_ANGLE: 15`
   - `POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN]`
   - `MAX_EPISODE_STEPS: 5000`（v1 下每个导航步额外 4 pitch 步骤，需 5× 余量）

2. `havln_http_env_server.py`：
   - ACTION_MAP 扩展支持 action 4 (`LOOK_UP`) 和 5 (`LOOK_DOWN`)
   - pitch 动作通过 `env.habitat_env.step()` 直接调用底层 Habitat，
     **绕过** `HAVLNCEDaggerEnv._handle_signals()`，不推进人类动画帧
   - `/step` 端点添加 try/except 错误保护，防止返回空响应

**InternNav 改动（已完成，仅配置）：**

3. `vln_ha_r2r.yaml` 已包含 look_up/look_down actions + `tilt_angle: 15`
4. `havln_http_env.py` 的 step() 本身就发送 int action code，天然支持 4/5

**运行注意事项：**

- 可选使用 `--disable-lookdown-sensors`：原版 evaluator 不使用外挂传感器
  （没有 `has_external_lookdown_views` 概念），仅通过物理 pitch 获取 lookdown 视图，
  关闭可减少 HTTP 响应中无用的 640×480 lookdown 图像传输
- 正常导航步骤使用物理 pitch 获取 lookdown（与 R2R 训练一致）：
  env.step(LOOKDOWN)×2 → 获取俯视图 → env.step(LOOKUP)×2 → 恢复
- LOOKDOWN followup 后，相机会累积 -30° 偏移（这是原版 evaluator 的设计行为，
  与标准 R2R 评测一致）
- 每个 evaluator 导航步 = 1 常规 + 4 pitch = 5 次 Habitat env.step()
  500 evaluator 步 ≈ 2500 Habitat 步，MAX_EPISODE_STEPS 设为 5000 足够

---

## 七、验证计划与初步结果

### Phase 1+2 验证（方案 A+B）— 已完成

1. **传感器分辨率验证**：check_sim 图像确认为 640×480 ✅
2. **v1 动作空间**：LOOK_UP(4)/LOOK_DOWN(5) 服务端已支持 ✅
3. **人类动画**：pitch 动作绕过 `_handle_signals()`，不加速人类帧 ✅
4. **MAX_EPISODE_STEPS**：从 500 增至 5000，避免 pitch 步骤耗尽限额 ✅
5. **已知 warning**：PyTorch 模型加载的 `copying from a non-meta parameter` 警告为无害噪音

### Phase 3 — 使用原版 evaluator 验证（关键结论）

⚠️ **之前尝试的 evaluator 修改（外挂传感器优先 + MAX_CONSECUTIVE_LOOKDOWNS）
被证实有害——导致 R2R 运行也出现转圈问题。已完全回滚。**

**原版 evaluator + HA-VLN 桥接的实际表现**（32 episodes 跑批结果）：

1. **双系统管线完全打通** ✅
   - S2 正确输出 `↓` → LOOKDOWN followup → 模型输出 pixel goal 坐标
   - 例：`step_id: 0 output text: ↓` → `step_id: 0 output text: 251 338`
   - S1 (NavDP) 被成功调用，生成 `local_actions` 序列
   - pixel_goal → S1 planning → 执行 → 重新 S2 决策 的循环正常运转

2. **模型行为合理** ✅
   - 混合使用离散动作（←→↑）和 pixel goal + S1 本地规划
   - 会主动 STOP
   - 在某些 episode 达到 oracle_success=1.0（曾经过目标点附近）

3. **已知局限**（模型层面，非桥接问题）：
   - success rate 暂时较低（停在离目标 >3m 处）
   - 某些场景仍有长时间转圈行为
   - HA-VLN 场景含人类 3D 模型，与 R2R 训练数据有域差异

### save_video 修复（仅影响视频录制，不改推理/动作逻辑）

**根因**：`vis_frames` 的填充依赖 `info['top_down_map']`，但 HTTP 桥接模式下
metrics 中不包含 `top_down_map`（永远为 None），导致 `vis_frames` 始终为空。

**修复**：
1. 当 `top_down_map` 不可用时，降级为纯 RGB 帧（不含顶视图合成）
2. 去除 `metrics['success'] == 1.0` 条件，所有 episode 均保存视频
3. 添加 `len(vis_frames) > 0` 保护，避免空视频写入

视频路径：`{output_path}/vis_{epoch}/{scene_id}/{episode_id:04d}.mp4`

### 待验证

1. 确认 save_video 修复后视频正确保存
2. 通过视频回放分析模型的导航质量，定位 success rate 低的原因
3. 对比基准：同配置跑 InternNav 原生 R2R（无人类）确认模型本身正常

---

## 八、运行命令速查

### HA-VLN HTTP Server

```bash
cd /root/backup/HA-VLN/agent
conda run -n havlnce python ../scripts/havln_http_env_server.py \
  --config-path config/internnav_bridge.yaml \
  --split val_unseen \
  --max-episodes 32 \
  --port 8899
```

可用选项：
- `--scene-filter x8F5xyUWy9e,zsNo4HB9uLZ`：按场景过滤
- `--episode-filter 1,2,505`：按 episode ID 过滤
- `--max-episodes -1`：serve 全部 episode
- `--disable-lookdown-sensors`：关闭外挂俯视传感器（**不推荐**，会导致 LOOKDOWN followup 使用倾斜视图）

### InternNav DualVLN 客户端

```bash
cd /root/backup/InternNav

# 少量 episode 调试（3 episodes + vis_debug 录像）
conda run -n internnav python scripts/eval/eval.py \
  --config scripts/eval/configs/havln_http_dual_system_debug_cfg.py

# 32 episodes 正式评测
conda run -n internnav python scripts/eval/eval.py \
  --config scripts/eval/configs/havln_http_dual_system_32_cfg.py
```

### HA-VLN RandomAgent 基线

```bash
cd /root/backup/HA-VLN/agent/VLN-CE
conda run -n havlnce python run.py \
  --exp-config config/nonlearning_internnav.yaml \
  --run-type eval
```
