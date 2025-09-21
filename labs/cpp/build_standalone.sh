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

PLATFORM=$(uname -s || echo unknown)

# Try to find system libfmt; if not found, fall back to header-only
FMT_CFLAGS=()
FMT_LDFLAGS=()
USE_FMT_HEADER_ONLY=1

if command -v pkg-config >/dev/null 2>&1; then
  if pkg-config --exists fmt; then
    FMT_CFLAGS=( $(pkg-config --cflags fmt) )
    FMT_LDFLAGS=( $(pkg-config --libs fmt) )
    USE_FMT_HEADER_ONLY=0
  fi
fi

# Probe common install prefixes when pkg-config is missing or incomplete
if [[ $USE_FMT_HEADER_ONLY -eq 1 ]]; then
  for prefix in /opt/homebrew /usr/local /opt/local /usr; do
    if [[ -f "$prefix/include/fmt/format.h" ]]; then
      if [[ -f "$prefix/lib/libfmt.dylib" || -f "$prefix/lib/libfmt.a" || -f "$prefix/lib/libfmt.so" ]]; then
        FMT_CFLAGS=("-I" "$prefix/include")
        FMT_LDFLAGS=("-L" "$prefix/lib" "-lfmt")
        USE_FMT_HEADER_ONLY=0
        break
      fi
    fi
  done
fi

COMMON_INC=("-I" "$ROOT_DIR/node/vendor" "-I" "$ROOT_DIR" ${FMT_CFLAGS[@]:-})

echo "Building standalone CPU sanity…"
CPU_DEFS=()
if [[ $USE_FMT_HEADER_ONLY -eq 1 ]]; then
  CPU_DEFS+=("-DFMT_HEADER_ONLY")
fi

FRAMEWORKS=()
if [[ "$PLATFORM" == "Darwin" ]]; then
  FRAMEWORKS=( -framework Metal -framework Accelerate -framework Foundation -framework QuartzCore )
fi

clang++ -std=c++17 -O2 -fexceptions ${CPU_DEFS[@]} \
  "${COMMON_INC[@]}" \
  "$ROOT_DIR/labs/cpp/standalone_cpu.cpp" \
  "$LIB" \
  ${FRAMEWORKS[@]} \
  ${FMT_LDFLAGS[@]:-} \
  -o "$OUT_DIR/cpu_sanity"

echo "CPU binary: $OUT_DIR/cpu_sanity"

echo "Building standalone GPU sanity…"
GPU_DEFS=()
if [[ $USE_FMT_HEADER_ONLY -eq 1 ]]; then
  GPU_DEFS+=("-DFMT_HEADER_ONLY")
fi
clang++ -std=c++17 -O2 -fexceptions ${GPU_DEFS[@]} \
  "${COMMON_INC[@]}" \
  "$ROOT_DIR/labs/cpp/standalone_gpu.cpp" \
  "$LIB" \
  ${FRAMEWORKS[@]} \
  ${FMT_LDFLAGS[@]:-} \
  -o "$OUT_DIR/gpu_sanity"

echo "GPU binary: $OUT_DIR/gpu_sanity"

echo "Run: $OUT_DIR/cpu_sanity ; and: $OUT_DIR/gpu_sanity"

if [[ $USE_FMT_HEADER_ONLY -eq 1 ]]; then
  cat <<INFO
Note: libfmt not found on system; using header-only mode. If you prefer a shared
libfmt, install it and re-run:
  - macOS (Homebrew):   brew install fmt
  - macOS (MacPorts):   sudo port install libfmt11
  - Debian/Ubuntu:      sudo apt install libfmt-dev
  - Conda (all):        conda install -c conda-forge fmt
  - vcpkg:              ./vcpkg install fmt
INFO
fi
