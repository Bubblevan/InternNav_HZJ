cmake_minimum_required(VERSION 3.10)

project(internvla_server)

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

add_executable(server
    cpp/server/main.cpp
)

target_link_libraries(server
    gRPC::grpc++
    protobuf::libprotobuf
    ${OpenCV_LIBS}
)