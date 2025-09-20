#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL_DIR="${ROOT_DIR}/vendor/mlx/backend/metal/kernels"
BUILD_DIR="${KERNEL_DIR}/.metallib_build"
AIR_DIR="${BUILD_DIR}/air"
METALLIB_PATH="${KERNEL_DIR}/mlx.metallib"

rm -rf "${BUILD_DIR}"
mkdir -p "${AIR_DIR}"

METAL_VERSION=$(zsh -c 'echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P - | tail -1 | tr -d "\n"')
echo "Detected Metal version: ${METAL_VERSION}"

if [[ -z "${METAL_VERSION}" ]]; then
  echo "Failed to determine Metal version" >&2
  exit 1
fi

if (( METAL_VERSION >= 320 )); then
  VERSION_INCLUDE="${KERNEL_DIR}/metal_3_1"
else
  VERSION_INCLUDE="${KERNEL_DIR}/metal_3_0"
fi

FLAGS=(-Wall -Wextra -fno-fast-math -Wno-c++17-extensions)
if [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
  FLAGS+=("-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}")
fi

mapfile -t METAL_FILES < <(find "${KERNEL_DIR}" -name '*.metal' -type f | sort)
AIR_FILES=()

for metal_file in "${METAL_FILES[@]}"; do
  rel_path="${metal_file#${KERNEL_DIR}/}"
  if [[ "${rel_path}" == "fence.metal" && ${METAL_VERSION} -lt 320 ]]; then
    echo "Skipping fence.metal for Metal version ${METAL_VERSION}"
    continue
  fi
  output_file="${AIR_DIR}/$(basename "${metal_file}" .metal).air"
  echo "Compiling ${rel_path} -> ${output_file}" 
  xcrun -sdk macosx metal "${FLAGS[@]}" \
    -I"${ROOT_DIR}/vendor" \
    -I"${KERNEL_DIR}" \
    -I"${VERSION_INCLUDE}" \
    -c "${metal_file}" \
    -o "${output_file}"
  AIR_FILES+=("${output_file}")
done

if [[ ${#AIR_FILES[@]} -eq 0 ]]; then
  echo "No Metal kernels were compiled" >&2
  exit 1
fi

echo "Linking ${#AIR_FILES[@]} kernels into ${METALLIB_PATH}" 
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "${METALLIB_PATH}"

rm -rf "${BUILD_DIR}"

echo "Metallib built at ${METALLIB_PATH}"
