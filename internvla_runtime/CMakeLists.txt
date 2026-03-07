cmake_minimum_required(VERSION 3.10)

project(internvla_server)

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)

# Ubuntu 等系统自带的 libgrpc++-dev 不提供 gRPCConfig.cmake，用 pkg-config 兜底
find_package(gRPC QUIET)
if(NOT gRPC_FOUND)
  pkg_check_modules(GRPC REQUIRED grpc++)
  add_library(gRPC::grpc++ INTERFACE IMPORTED)
  set_target_properties(gRPC::grpc++ PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GRPC_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${GRPC_LIBRARIES}"
  )
endif()

# 需先执行 gen_proto_cpp.sh 生成 cpp/gen/*.pb.*
set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/gen)
add_executable(server
    cpp/server/main.cpp
    ${GEN_DIR}/internvla.pb.cc
    ${GEN_DIR}/internvla.grpc.pb.cc
)
target_include_directories(server PRIVATE ${GEN_DIR})

target_link_libraries(server
    gRPC::grpc++
    protobuf::libprotobuf
    ${OpenCV_LIBS}
)