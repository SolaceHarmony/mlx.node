#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
OUT_DIR="$ROOT_DIR/labs/bin"
mkdir -p "$OUT_DIR"

LIB="$ROOT_DIR/node/build/Release/mlx_core.a"
if [[ ! -f "$LIB" ]]; then
  echo "Building mlx_core.a via node-gyp…"
  (cd "$ROOT_DIR/node" && npx node-gyp build)
fi

COMMON_INC=("-I" "$ROOT_DIR/node/vendor" "-I" "$ROOT_DIR")

echo "Building standalone CPU sanity…"
clang++ -std=c++17 -O2 -fexceptions -DFMT_HEADER_ONLY \
  "${COMMON_INC[@]}" \
  "$ROOT_DIR/labs/cpp/standalone_cpu.cpp" \
  "$LIB" \
  -framework Metal -framework Accelerate -framework Foundation -framework QuartzCore \
  -o "$OUT_DIR/cpu_sanity"

echo "CPU binary: $OUT_DIR/cpu_sanity"

echo "Building standalone GPU sanity…"
clang++ -std=c++17 -O2 -fexceptions -DFMT_HEADER_ONLY \
  "${COMMON_INC[@]}" \
  "$ROOT_DIR/labs/cpp/standalone_gpu.cpp" \
  "$LIB" \
  -framework Metal -framework Accelerate -framework Foundation -framework QuartzCore \
  -o "$OUT_DIR/gpu_sanity"

echo "GPU binary: $OUT_DIR/gpu_sanity"

echo "Run: $OUT_DIR/cpu_sanity ; and: $OUT_DIR/gpu_sanity"
