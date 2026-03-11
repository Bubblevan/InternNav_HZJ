# DualVLN Habitat Architecture Walkthrough

本文用“代码数据流”的方式解释 InternNav 里 DualVLN 在 Habitat VLN-CE R2R 评估中的工作路径，重点回答以下问题：

1. 输入数据到底有哪些
2. 这些数据是如何进入 InternNav 的
3. System 2 和 System 1 分别在哪里执行
4. Habitat 闭环版本和 realworld agent 版本有什么关系与差异

## 1. 先看总流程

如果你运行的是：

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
```

那么主路径是：

1. `scripts/eval/eval.py`
2. `Evaluator.init(...)`
3. `HabitatVLNEvaluator`
4. `HabitatEnv`
5. `HabitatVLNEvaluator._run_eval_dual_system()`
6. 在 evaluator 内直接调用 DualVLN 模型：
   - `model.generate(...)` 作为 System 2
   - `model.generate_latents(...)` 作为从 S2 文本输出到 S1 条件 latent 的桥接
   - `model.generate_traj(...)` 作为 System 1
7. 将离散动作回送给 Habitat 环境继续闭环

最关键的一点是：

> 目前 Habitat VLN-CE 这条评估链，并不是先走 `internnav/agent/internvla_n1_agent.py` 再走模型，而是 evaluator 里直接调模型。

这也是为什么仿真 profiling 最应该插在 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`，而不是只插在 `internnav/agent/internvla_n1_agent.py`。

## 2. 输入数据有哪些

### 2.1 数据文件层

Habitat R2R 评估所需的数据主要有两部分：

1. 标注 json
   路径在 `scripts/eval/configs/vln_r2r.yaml`
   例如 `data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz`

2. 场景资源
   路径在 `data/scene_data/...`
   其中包含：
   - `.glb`
   - `.navmesh`
   - `.house`

这条评估链不依赖 LeRobot 的 `traj_data/r2r`。`traj_data` 主要用于训练和离线数据管线，而不是 Habitat 的在线闭环评估。

### 2.2 每一步传给模型的运行时输入

在闭环过程中，模型真正看到的输入包括：

1. 当前正前方 RGB 图像
2. 当前正前方 depth
3. 由历史帧采样得到的多张历史 RGB
4. 指令文本 instruction
5. 一个额外取到的 look-down 视角 RGB/depth

需要注意：

- System 2 的文本生成主要使用正前方 RGB 和历史 RGB
- System 1 的局部动作生成使用 look-down 视角和从 S2 产生的 latent

## 3. 配置如何进入代码

入口配置是：

- `scripts/eval/configs/habitat_dual_system_cfg.py`
- `scripts/eval/configs/vln_r2r.yaml`

### 3.1 `eval.py`

`scripts/eval/eval.py` 负责读取 python config 文件，拿到 `eval_cfg`，然后通过：

```python
evaluator = Evaluator.init(evaluator_cfg)
evaluator.eval()
```

创建具体 evaluator。

### 3.2 `HabitatVLNEvaluator.__init__`

在 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` 中：

- 先读取 `cfg.eval_settings`
- 再加载 Habitat yaml
- 然后把 dataset path / scenes dir / split 等覆盖项写回 Habitat config
- 再通过 `DistributedEvaluator` 初始化环境
- 最后加载 DualVLN checkpoint

也就是说，这里同时把“仿真环境配置”和“模型推理配置”接到了一起。

## 4. Habitat 环境如何提供 episode

`internnav/env/habitat_env.py` 是一个很薄的 wrapper。

它的职责主要有三个：

1. 构造底层 `habitat.Env`
2. 从 Habitat dataset 中取出所有 episode
3. 做项目层面的 episode 过滤和顺序控制

在 `generate_episodes()` 中，它会：

- 按 scene 分组
- 读取 `progress.json`，跳过已完成 episode
- 根据 `allowed_scene_ids`
- 根据 `allowed_episode_ids`
- 根据 `max_eval_episodes`
  过滤出本次实际要评估的 episode 列表

然后 `reset()` 时，不是让 Habitat 自己随机选 episode，而是手动把 `current_episode` 切到这份过滤后的列表里。

## 5. 闭环主循环在哪里

核心闭环在：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
- 方法：`_run_eval_dual_system()`

这段代码的逻辑可以拆成四步。

### 5.1 取环境观测

每个循环步从 Habitat 拿到：

- `observations["rgb"]`
- `observations["depth"]`
- `observations["gps"]`
- `observations["compass"]`

其中：

- `rgb` 是当前前视图
- `depth` 是当前前视 depth
- `gps + compass` 被记录进 profiling / replay，用作 pose 信息

### 5.2 构造 S2 输入

如果当前没有残余动作可执行，也没有仍在生效的 pixel goal，那么会触发一次新的 S2 决策。

这一步会：

1. 把当前前视 RGB resize 到模型期望分辨率
2. 从历史帧里按 `num_history` 采样若干帧
3. 把 instruction 和多张图像按 Qwen-VL 风格组织成聊天模板
4. 调用：

```python
model.generate(...)
```

这就是当前 Habitat 版本的 System 2。

它的输出有两种可能：

1. 文本里是离散动作，例如 `↑` `←` `STOP`
2. 文本里是像素坐标，例如一个 pixel goal

### 5.3 S2 输出为离散动作时

如果 `model.generate(...)` 的文本结果不包含坐标，只包含离散动作，那么流程很直接：

- 用正则把文本解析成动作序列
- 从中取一个或若干个离散动作
- 直接送给 Habitat 环境执行

这时不会进入 System 1。

### 5.4 S2 输出为 pixel goal 时

如果 `model.generate(...)` 输出的是像素坐标，那么会进入双系统交接。

这一步分成两段：

#### 段 A：`generate_latents(...)`

先把 S2 的文本输出和对应视觉输入转成 trajectory latent：

```python
traj_latents = model.generate_latents(output_ids, pixel_values, image_grid_thw)
```

这一步本质上还是承接 System 2 的输出，是“从语言侧输出过渡到运动侧条件表示”的桥梁。

#### 段 B：`generate_traj(...)`

然后 evaluator 会额外取一个 look-down 视角，构造成：

- `images_dp`
- `depths_dp`

再调用：

```python
dp_actions = model.generate_traj(traj_latents, images_dp, depths_dp)
```

这一步就是当前闭环里的 System 1。

它输出的是局部轨迹，再通过 `traj_to_actions(...)` 转成离散动作序列。

随后会连续执行数步局部动作，直到：

- 达到最大 forward 步数
- 或局部动作输出 `STOP`
- 或需要重新回到 S2 做新决策

## 6. 为什么 evaluator 里会额外取 look-down 图像

你会看到 evaluator 在很多步里会额外执行两次：

- `LOOKDOWN`
- `LOOKDOWN`

然后在后面再执行两次 `LOOKUP`

原因是当前 System 1 的局部轨迹预测更依赖自上而下、更利于几何判断的视角。也就是说：

- 正前方图像更像是给 System 2 做“我下一步该往哪里去”的高层决策
- look-down 图像更像是给 System 1 做“我接下来几步该怎么走”的近场运动生成

这也是为什么 replay 导出时不能只存正前方图，而必须连 look-down RGB/depth 一起存下来，否则后续离线 benchmark 不能严格复现 S1 输入。

## 7. `internvla_n1_agent.py` 和当前 Habitat evaluator 的关系

项目里确实还有一个更“agent 化”的实现：

- `internnav/agent/internvla_n1_agent.py`

它提供了：

- `should_infer_s2()`
- `policy.s2_step(...)`
- `policy.s1_step_latent(...)`
- `step_no_infer(...)`

这个版本非常适合 realworld 或通用 agent 场景，因为它把大小脑逻辑抽象成了 agent 接口。

但对于当前 Habitat VLN-CE evaluator 来说，实际运行主路径是 evaluator 里直接调用模型，不是通过这个 agent 走的。

你可以把两者理解为：

- `internvla_n1_agent.py`：项目层的通用 agent 封装
- `habitat_vln_evaluator.py`：为了做 Habitat VLN-CE 评估而写的“直接调模型”的专用闭环实现

## 8. policy 层是怎么定义大小脑接口的

如果想看更“标准”的大小脑接口定义，可以看：

- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`

这里可以看到三类关键方法：

1. `step_no_infer(...)`
   只推进历史，不触发推理

2. `s2_step(...)`
   输入：
   - 当前 RGB
   - depth
   - pose
   - instruction
   - intrinsic
   - `look_down` 标志

   输出：
   - 若是高层离散决策，则返回 `output_action`
   - 若是 pixel goal，则返回：
     - `output_pixel`
     - `output_latent`

3. `s1_step_latent(...)`
   输入：
   - RGB / depth（通常是为局部规划准备的格式）
   - latent

   输出：
   - 一段局部离散动作序列

这套接口和 evaluator 中直接调 `generate / generate_latents / generate_traj` 的逻辑是对应的，只是 evaluator 当前把这三步手动展开了。

## 9. replay benchmark 为什么是必要的

闭环 Habitat 评估里，模型推理之外还混着很多东西：

- Habitat 环境步进
- look-down / look-up 视角切换
- 图像与 depth 预处理
- Python 控制流和同步等待

因此闭环总延迟不等于模型纯推理延迟。

这就是为什么新增了：

- `scripts/eval/tools/benchmark_dualvln_replay.py`

这个脚本的目的，是把闭环中保存下来的固定观测重新喂给 DualVLN，只测：

- `S2 generate`
- `S2 latent`
- `S1 generate_traj`
- 总推理时间
- 显存
- TPS
- 输出一致性

这样后续做 KV cache、backend 或量化优化时，才能明确知道收益是否真的来自模型本身。

## 10. 当前应该如何理解大小脑分工

可以把当前 DualVLN 粗略理解成：

- **System 2**
  负责高层语义决策
  输入是 instruction + 当前视图 + 历史视图
  输出要么是离散动作，要么是 pixel goal

- **System 1**
  负责短程局部运动生成
  输入是：
  - 来自 S2 的 latent
  - 当前更适合近场规划的 look-down 视觉输入
  输出是连续几步局部离散动作

所以并不是两个完全割裂的模型，而是：

> S2 先决定“我要去哪”，S1 再负责“我接下来几步怎么走过去”。

## 11. 读代码的推荐顺序

如果你之后想自己 trace 一遍，建议按这个顺序读：

1. `scripts/eval/eval.py`
2. `scripts/eval/configs/habitat_dual_system_cfg.py`
3. `scripts/eval/configs/vln_r2r.yaml`
4. `internnav/env/habitat_env.py`
5. `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`
6. `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
7. `internnav/model/basemodel/internvla_n1/internvla_n1.py`
8. `internnav/model/utils/vln_utils.py`

如果你的目标是分析 realworld 版，再补读：

9. `internnav/agent/internvla_n1_agent.py`
10. `internnav/agent/internvla_n1_agent_realworld.py`

## 12. 一句话总结

当前 Habitat VLN-CE R2R 版 DualVLN 的数据流可以概括为：

> Habitat 提供 RGB/depth/gps/compass 与 instruction，evaluator 先用前视图和历史图跑 System 2 文本决策；若得到离散动作则直接执行，若得到 pixel goal 则进一步生成 latent，并配合 look-down 视图进入 System 1 生成局部动作，最后再把动作送回 Habitat 闭环执行。
