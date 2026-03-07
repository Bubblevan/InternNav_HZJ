#!/usr/bin/env bash
# 从 proto 生成 Python：internvla_pb2.py、internvla_pb2_grpc.py
# 在项目根目录执行：bash internvla_runtime/gen_proto.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m grpc_tools.protoc \
  -I internvla_runtime/proto \
  --python_out=internvla_runtime/python \
  --grpc_python_out=internvla_runtime/python \
  internvla_runtime/proto/internvla.proto

echo "Generated: internvla_runtime/python/internvla_pb2.py, internvla_pb2_grpc.py"
