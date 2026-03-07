#!/usr/bin/env bash
# 从 proto 生成 C++：internvla.pb.* 与 internvla.grpc.pb.*
# 需已安装 protobuf-compiler-grpc 或 grpc 的 protoc 插件
# 在项目根目录执行：bash internvla_runtime/gen_proto_cpp.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=internvla_runtime/cpp/gen
mkdir -p "$OUT"

# 若系统有 grpc_cpp_plugin（apt: protobuf-compiler-grpc 或从 grpc 安装）
if command -v grpc_cpp_plugin &>/dev/null; then
  PLUGIN="grpc_cpp_plugin"
elif command -v grpc_cpp_plugin.exe &>/dev/null; then
  PLUGIN="grpc_cpp_plugin.exe"
else
  echo "请先安装 gRPC C++ 插件，例如: sudo apt install protobuf-compiler-grpc"
  exit 1
fi

protoc -I internvla_runtime/proto \
  --cpp_out="$OUT" \
  --grpc_out="$OUT" \
  --plugin=protoc-gen-grpc="$(which $PLUGIN)" \
  internvla_runtime/proto/internvla.proto

echo "Generated C++ files in $OUT"
ls -la "$OUT"
