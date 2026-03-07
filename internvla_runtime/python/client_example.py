#!/usr/bin/env python3
"""
调用 InternVLA gRPC 服务的示例客户端。

前提：Python worker 已启动，例如：
  python internvla_runtime/python/worker_server.py --port 5801

运行本示例（在项目根目录）：
  python internvla_runtime/python/client_example.py --port 5801
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import grpc

# 让 import internvla_pb2 能找到（与 worker_server 一致）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import internvla_pb2
import internvla_pb2_grpc


def run(host="127.0.0.1", port=5801, session_id="demo"):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = internvla_pb2_grpc.InternVLAServiceStub(channel)

    # 1) 可选：设置本局导航指令（不调则 Step 里用默认 "Go forward."）
    init_req = internvla_pb2.InitRequest(
        session_id=session_id,
        instruction="Walk to the door and stop.",
    )
    init_resp = stub.InitEpisode(init_req)
    print(f"InitEpisode: ok={init_resp.ok}")

    # 2) 发一帧：RGB + Depth + 4x4 相机位姿
    w, h = 640, 480
    rgb = np.zeros((h, w, 3), dtype=np.uint8)  # 示例用全黑，实际请填相机图像
    depth = np.zeros((h, w), dtype=np.float32)  # 示例用全 0，实际请填深度图
    camera_pose = np.eye(4, dtype=np.float32)   # 4x4 单位阵，实际请填真实位姿

    step_req = internvla_pb2.StepRequest(
        session_id=session_id,
        width=w,
        height=h,
        rgb=rgb.tobytes(),
        depth=depth.tobytes(),
        camera_pose=camera_pose.flatten().tolist(),
    )
    step_resp = stub.Step(step_req)

    if step_resp.has_action:
        print(f"Step: discrete_action={list(step_resp.action)}")
    else:
        print(f"Step: trajectory (len={len(step_resp.trajectory)}), pixel_goal=({step_resp.pixel_x}, {step_resp.pixel_y})")

    return step_resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="Worker 地址")
    ap.add_argument("--port", type=int, default=5801, help="Worker 端口")
    ap.add_argument("--session_id", default="demo", help="会话 ID，与 InitEpisode 一致")
    args = ap.parse_args()
    run(host=args.host, port=args.port, session_id=args.session_id)


if __name__ == "__main__":
    main()
