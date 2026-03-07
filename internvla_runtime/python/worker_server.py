import argparse
import grpc
import numpy as np
from concurrent import futures

import internvla_pb2
import internvla_pb2_grpc

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# 默认相机内参（与 http_internvla_server 一致，可按实际相机覆盖）
DEFAULT_CAMERA_INTRINSIC = np.array([
    [386.5, 0.0, 328.9, 0.0],
    [0.0, 386.5, 244, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


class Worker(internvla_pb2_grpc.InternVLAServiceServicer):
    def __init__(self, args):
        self.args = args
        self.agent = InternVLAN1AsyncAgent(args)
        self._session_instruction = {}  # session_id -> instruction（由 InitEpisode 写入）

        # 与 http server 一致：warmup 一次，避免首包过慢
        self.agent.step(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.float32),
            np.eye(4),
            "hello",
            intrinsic=args.camera_intrinsic,
        )
        self.agent.reset()

    def InitEpisode(self, request, context):
        """客户端通过本 RPC 传入 instruction：InitEpisode(session_id=..., instruction="你的导航指令")。"""
        if request.session_id:
            self._session_instruction[request.session_id] = request.instruction or "Go forward."
        response = internvla_pb2.InitResponse()
        response.ok = True
        return response

    def Step(self, request, context):
        w = request.width
        h = request.height

        rgb = np.frombuffer(request.rgb, dtype=np.uint8).reshape(h, w, 3)
        depth = np.frombuffer(request.depth, dtype=np.float32).reshape(h, w)
        camera_pose = np.array(request.camera_pose, dtype=np.float64).reshape(4, 4)

        instruction = self._session_instruction.get(
            request.session_id, "Go forward."
        )

        output = self.agent.step(
            rgb,
            depth,
            camera_pose,
            instruction,
            intrinsic=self.args.camera_intrinsic,
        )

        response = internvla_pb2.StepResponse()

        if output.output_action is not None:
            response.has_action = True
            for a in output.output_action:
                response.action.append(int(a))
        else:
            response.has_trajectory = True
            traj = np.asarray(output.output_trajectory).flatten()
            for v in traj:
                response.trajectory.append(float(v))
            if output.output_pixel is not None and len(output.output_pixel) >= 2:
                response.pixel_x = int(output.output_pixel[0])
                response.pixel_y = int(output.output_pixel[1])

        return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    args.camera_intrinsic = DEFAULT_CAMERA_INTRINSIC
    return args


def serve():
    args = parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    internvla_pb2_grpc.add_InternVLAServiceServicer_to_server(Worker(args), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"InternVLA Python worker listening on [::]:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
