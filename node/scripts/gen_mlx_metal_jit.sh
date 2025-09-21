#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?root dir}"
OUT_DIR="${2:?out dir}"
CC_BIN="${3:-/usr/bin/clang}"

SCRIPT="${ROOT_DIR}/mlx/backend/metal/make_compiled_preamble.sh"
if [[ ! -f "$SCRIPT" ]]; then
  echo "make_compiled_preamble.sh not found: $SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Core preambles
srcs=(
  utils
  unary_ops
  binary_ops
  ternary_ops
  reduce_utils
  indexing/scatter
  indexing/gather
  indexing/gather_front
  indexing/gather_axis
  indexing/scatter_axis
  hadamard
)

# JIT kernels (MLX_METAL_JIT on)
srcs+=(
  arange
  copy
  unary
  binary
  binary_two
  fft
  logsumexp
  ternary
  softmax
  scan
  sort
  reduce
  steel/gemm/gemm
  steel/gemm/kernels/steel_gemm_fused
  steel/gemm/kernels/steel_gemm_masked
  steel/gemm/kernels/steel_gemm_gather
  steel/gemm/kernels/steel_gemm_splitk
  steel/gemm/kernels/steel_gemm_segmented
  steel/conv/conv
  steel/conv/kernels/steel_conv
  steel/conv/kernels/steel_conv_general
  quantized_utils
  quantized
  fp4_quantized
  gemv_masked
)

for s in "${srcs[@]}"; do
  bash "$SCRIPT" "$OUT_DIR" "$CC_BIN" "$ROOT_DIR" "$s"
done

echo "Generated $(ls "$OUT_DIR" | wc -l) JIT source files into $OUT_DIR" >&2
