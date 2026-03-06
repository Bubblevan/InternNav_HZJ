import grpc
import numpy as np
from concurrent import futures

import internvla_pb2
import internvla_pb2_grpc

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


class Worker(internvla_pb2_grpc.InternVLAServiceServicer):

    def __init__(self):

        args = {}

        self.agent = InternVLAN1AsyncAgent(args)

    def Step(self, request, context):

        w = request.width
        h = request.height

        rgb = np.frombuffer(request.rgb, dtype=np.uint8)
        rgb = rgb.reshape(h, w, 3)

        depth = np.frombuffer(request.depth, dtype=np.float32)
        depth = depth.reshape(h, w)

        camera_pose = np.array(request.camera_pose).reshape(4,4)

        output = self.agent.step(
            rgb,
            depth,
            camera_pose,
            "instruction",
        )

        response = internvla_pb2.StepResponse()

        if output.output_action is not None:

            response.has_action = True

            for a in output.output_action:
                response.action.append(a)

        else:

            response.has_trajectory = True

            traj = output.output_trajectory.flatten()

            for v in traj:
                response.trajectory.append(float(v))

        return response


def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1)
    )

    internvla_pb2_grpc.add_InternVLAServiceServicer_to_server(
        Worker(), server
    )

    server.add_insecure_port('[::]:50052')

    server.start()

    server.wait_for_termination()


if __name__ == "__main__":

    serve()