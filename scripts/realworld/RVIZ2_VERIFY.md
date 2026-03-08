# 使用 RViz2 做可视化验证

在同一网络环境下的开发机上，用 RViz2 订阅机器人端发布的 ROS 2 话题，可验证彩色图、深度图、里程计是否正常。

## 前置条件

- 已安装 ROS 2（与机器人端同版本）
- 机器人端已启动并发布话题
- 开发机与机器人在同一网络，且 ROS 2 域 ID 一致（`ROS_DOMAIN_ID`）

## 步骤

### 1. 设置 ROS 2 环境

```bash
source /opt/ros/<distro>/setup.bash   # 如 humble: source /opt/ros/humble/setup.bash
```

### 2. 确认话题存在

```bash
ros2 topic list
```

应能看到类似：

- `/camera/camera/color/image_raw`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/odom_bridge`

### 3. 启动 RViz2

```bash
rviz2
```

### 4. 设置 Fixed Frame

- 左侧 **Global Options** → **Fixed Frame** 设为 `odom` 或 `camera_link`（根据实际 TF 树）

### 5. 添加显示项

点击左下角 **Add**，依次添加：

| 显示类型 | 话题名称 | 说明 |
|---------|----------|------|
| **Image** | `/camera/camera/color/image_raw` | RGB 彩色图像 |
| **Image** | `/camera/camera/aligned_depth_to_color/image_raw` | 对齐的深度图像 |
| **Odometry** | `/odom_bridge` | 机器人里程计（轨迹） |

### 6. 配置 Image 显示

- 每个 **Image** 显示项会打开一个图像窗口
- 若看不到图像，检查 **Topic** 是否选对，以及 **Reliability** 是否与发布端一致（如 `BEST_EFFORT`）

### 7. 配置 Odometry 显示

- **Topic** 选 `/odom_bridge`
- 可调整 **Shape**、**Color**、**Line Width** 等
- 若能看到轨迹，说明里程计正常

## 验证结果

- **彩色图**：能看到实时 RGB 画面
- **深度图**：能看到灰度深度图（近处亮、远处暗）
- **里程计轨迹**：能看到机器人移动轨迹

若三者都正常，说明机器人端这条链路基本 OK。

## 常见问题

1. **看不到图像**：检查 QoS，发布端若用 `BEST_EFFORT`，订阅端也需匹配
2. **Fixed Frame 报错**：确认 TF 树中有对应 frame，可用 `ros2 run tf2_tools view_frames` 查看
3. **跨机看不到话题**：确认 `ROS_DOMAIN_ID` 一致，网络互通
