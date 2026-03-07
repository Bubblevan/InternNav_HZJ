"""
Unitree Go2 + 真相机，通过 gRPC 调用 InternVLA worker。

与 http_internvla_client.py 相同的 ROS2 流程（Go2 底盘 + 相机 + 里程计），仅把 HTTP 换成 gRPC。

前置：先启动 gRPC worker，例如
  python internvla_runtime/python/worker_server.py --port 5801

本脚本需在 Go2 的 ROS2 工作空间下运行（与 http 客户端相同依赖：controllers, thread_utils 等）。
话题：/camera/... 相机，/odom_bridge 里程计，/cmd_vel_bridge 底盘控制。
"""
import copy
import io
import math
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
import grpc
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image

# 让 gRPC 生成的 stub 可被 import（项目根 = 上两级）
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "internvla_runtime" / "python"))
import internvla_pb2
import internvla_pb2_grpc

# 离散动作含义（与 internnav/agent/internvla_n1_agent_realworld.py 中 actions2idx 一致）
# 客户端对 5、9 不更新目标，故机器人不移动；其余会调用 incremental_change_goal 驱动底盘
DISCRETE_ACTION_NAMES = {
    0: "STOP",
    1: "↑ 前进",
    2: "← 左转15°",
    3: "→ 右转15°",
    5: "↓ (客户端视为不移动)",
    9: "(客户端视为不移动)",
}

frame_data = {}
frame_idx = 0
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# gRPC 配置（可按需改为参数）
GRPC_TARGET = "127.0.0.1:5801"
SESSION_ID = "go2_robot"  # Unitree Go2 本机会话
DEFAULT_INSTRUCTION = "Go forward and avoid obstacles. Stop when you reach the goal."

# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

_grpc_channel = None
_grpc_stub = None


def _get_stub():
    global _grpc_channel, _grpc_stub
    if _grpc_stub is None:
        _grpc_channel = grpc.insecure_channel(GRPC_TARGET)
        _grpc_stub = internvla_pb2_grpc.InternVLAServiceStub(_grpc_channel)
    return _grpc_stub


def _odom_to_camera_pose(odom):
    """odom = [x, y, yaw] -> 4x4 camera/robot pose (world to body)."""
    x, y, yaw = odom[0], odom[1], odom[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:2, 3] = [x, y]
    return T.astype(np.float32)


def grpc_dual_sys_eval(rgb_image, depth_image, camera_pose_4x4, reset):
    """
    用 gRPC 调用 worker：可选 InitEpisode(reset)，再 Step。
    rgb_image: (H,W,3) uint8, depth_image: (H,W) float32 米, camera_pose_4x4: (4,4) float32.
    返回与 HTTP 相同格式: {'discrete_action': [...]} 或 {'trajectory': [[x,y],...], 'pixel_goal': [x,y]}
    """
    global policy_init, http_idx, first_running_time
    stub = _get_stub()

    if reset:
        policy_init = False
        init_req = internvla_pb2.InitRequest(session_id=SESSION_ID, instruction=DEFAULT_INSTRUCTION)
        stub.InitEpisode(init_req)

    h, w = rgb_image.shape[0], rgb_image.shape[1]
    if rgb_image.dtype != np.uint8:
        rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    if depth_image.dtype != np.float32:
        depth_image = np.asarray(depth_image, dtype=np.float32)

    step_req = internvla_pb2.StepRequest(
        session_id=SESSION_ID,
        width=w,
        height=h,
        rgb=rgb_image.tobytes(),
        depth=depth_image.tobytes(),
        camera_pose=camera_pose_4x4.flatten().tolist(),
    )
    start = time.time()
    response = stub.Step(step_req)
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after grpc {time.time() - start:.3f}s")

    out = {}
    if response.has_action:
        out["discrete_action"] = list(response.action)
        names = [DISCRETE_ACTION_NAMES.get(a, f"未知{a}") for a in response.action]
        print(f"  → 含义: {names}  (仅 0,1,2,3 会驱动机器人；5/9 不更新目标故不动)")
    else:
        traj = np.array(response.trajectory, dtype=np.float64)
        if traj.size % 3 == 0:
            traj = traj.reshape(-1, 3)
        else:
            traj = traj.reshape(-1, 2)
        out["trajectory"] = traj.tolist()
        if response.pixel_x or response.pixel_y:
            out["pixel_goal"] = [response.pixel_x, response.pixel_y]
    print(f"response {out}")
    return out


def control_thread():
    global desired_v, desired_w
    while True:
        global current_control_mode
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        time.sleep(0.1)


def planning_thread():
    global trajs_in_world

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
        odom_rw_lock.release_read()

        if odom_infer is not None and infer_rgb is not None and infer_depth is not None:
            global frame_data, policy_init
            frame_data[http_idx] = {
                "infer_rgb": copy.deepcopy(infer_rgb),
                "infer_depth": copy.deepcopy(infer_depth),
                "infer_odom": copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]

            camera_pose = _odom_to_camera_pose(odom_infer)
            response = grpc_dual_sys_eval(infer_rgb, infer_depth, camera_pose, policy_init)
            policy_init = False

            global current_control_mode
            traj_len = 0.0
            if "trajectory" in response:
                trajectory = response["trajectory"]
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"traj len {traj_len}")
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]
                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                print(f"{time.time()} update traj")
                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
            elif "discrete_action" in response:
                actions = response["discrete_action"]
                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
        else:
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb: {infer_rgb is not None} depth: {infer_depth is not None}"
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__("go2_manager")

        rgb_down_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_down_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )
        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, "/odom_bridge", self.odom_callback, qos_profile)
        self.control_pub = self.create_publisher(Twist, "/cmd_vel_bridge", 5)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0
        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0
        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

    def rgb_forward_callback(self, rgb_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8")[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth_pil = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth_pil.save(depth_bytes, format="PNG")
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time
        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time
        rgb_depth_rw_lock.release_write()
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw
        self.control_pub.publish(request)


if __name__ == "__main__":
    global DEFAULT_INSTRUCTION, GRPC_TARGET
    import argparse
    ap = argparse.ArgumentParser(description="Go2 + InternVLA gRPC client")
    ap.add_argument("--instruction", "-i", type=str, default=DEFAULT_INSTRUCTION, help="导航指令，在 InitEpisode 时发给 worker")
    ap.add_argument("--grpc", type=str, default=GRPC_TARGET, help="Worker 地址，如 127.0.0.1:5801")
    args = ap.parse_args()
    # 让 planning 线程使用命令行传入的 instruction / 地址
    DEFAULT_INSTRUCTION = args.instruction
    GRPC_TARGET = args.grpc

    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager()
        control_thread_instance.start()
        planning_thread_instance.start()
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()
