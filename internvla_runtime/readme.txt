1) 生成 gRPC Python 代码（首次或修改 proto 后执行一次）

   需安装: pip install grpcio-tools

    在项目根目录执行:
      bash internvla_runtime/gen_proto.sh

    会生成:
      internvla_runtime/python/internvla_pb2.py
      internvla_runtime/python/internvla_pb2_grpc.py

2) 启动 Python worker

    需在项目根目录执行（以便 import internnav）:
      cd /path/to/InternNav
      python internvla_runtime/python/worker_server.py
    或:
      cd internvla_runtime/python && PYTHONPATH=../.. python worker_server.py

    可选参数: --model_path /path/to/InternVLA-N1-DualVLN --port 5801

2b) 如何调用（Use the service）

    流程: InitEpisode(可选) → 多次 Step。
    - InitEpisode(session_id, instruction): 为本局设置导航指令。
    - Step(session_id, width, height, rgb_bytes, depth_bytes, camera_pose): 送一帧 RGB、深度、4x4 位姿，返回离散动作 action 或轨迹 trajectory + pixel_goal。

    用示例客户端测试（worker 已启动且端口 5801）:
      python internvla_runtime/python/client_example.py --port 5801
    可改 --host 连到其他机器上的 worker。

    自写客户端: 用 grpc 连到 host:port，InternVLAServiceStub 调用 InitEpisode / Step，请求字段见 proto/internvla.proto。

    接真相机（Unitree Go2 + ROS2）：与 http_internvla_client 同流程，改用 gRPC 的脚本：
      scripts/realworld/grpc_internvla_client.py
    需在 Go2 的 ROS2 工作空间（含 controllers、thread_utils 等），先起 worker（如 --port 5801），
    再在 Go2 上运行该节点；可改脚本内 GRPC_TARGET、DEFAULT_INSTRUCTION，或命令行 --grpc、--instruction。

    部署步骤（workbench 起 worker，Go2 起客户端）：见 DEPLOY.md。

3) 编译并启动 C++ server（可选）

    依赖: OpenCV, Protobuf, gRPC, CMake, pkg-config。例如 Ubuntu:
      sudo apt install cmake pkg-config libopencv-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

    在项目根目录依次执行:
      bash internvla_runtime/gen_proto_cpp.sh
      cd internvla_runtime && mkdir -p build && cd build && cmake .. && make

    会生成可执行文件 build/server，运行:
      ./server

    注意: 不要用 g++ -o xxx main.cpp 这种单条命令编译，会缺 gRPC/OpenCV 等链接。

常见问题
- 若出现 "g++: no input files": -o 后面应是输出可执行名，再写源文件；且本项目需用 CMake 构建。
- 若 sudo apt 报 rtl8812au-dkms 错误: 与本次编译无关，是系统里该驱动包未配置好。可暂时跳过:
  sudo mv /var/crash/rtl8812au-dkms.0.crash /var/crash/rtl8812au-dkms.0.crash.bak
  sudo dpkg --configure -a
  或不需要该 WiFi 驱动时: sudo apt remove rtl8812au-dkms

