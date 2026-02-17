#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL_DIR="${ROOT_DIR}/vendor/mlx/backend/metal/kernels"
BUILD_DIR="${KERNEL_DIR}/.metallib_build"
AIR_DIR="${BUILD_DIR}/air"
METALLIB_PATH="${KERNEL_DIR}/mlx.metallib"

rm -rf "${BUILD_DIR}"
mkdir -p "${AIR_DIR}"

METAL_VERSION=$(echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P - 2>/dev/null | tail -1 | tr -d " \n")
echo "Detected Metal version: ${METAL_VERSION}"

if [ -z "${METAL_VERSION}" ]; then
  echo "Failed to determine Metal version" >&2
  exit 1
fi

if [ "${METAL_VERSION}" -ge 320 ]; then
  VERSION_INCLUDE="${KERNEL_DIR}/metal_3_1"
else
  VERSION_INCLUDE="${KERNEL_DIR}/metal_3_0"
fi

COMMON_FLAGS="-Wall -Wextra -fno-fast-math -Wno-c++17-extensions"
if [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]; then
  COMMON_FLAGS="${COMMON_FLAGS} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
fi

count=0
for metal_file in "${KERNEL_DIR}"/*.metal; do
  [ -f "${metal_file}" ] || continue
  rel_path="$(basename "${metal_file}")"
  if [ "${rel_path}" = "fence.metal" ] && [ "${METAL_VERSION}" -lt 320 ]; then
    echo "Skipping fence.metal for Metal version ${METAL_VERSION}"
    continue
  fi
  output_file="${AIR_DIR}/$(basename "${metal_file}" .metal).air"
  echo "Compiling ${rel_path}"
  xcrun -sdk macosx metal ${COMMON_FLAGS} \
    -I"${ROOT_DIR}/vendor" \
    -I"${KERNEL_DIR}" \
    -I"${VERSION_INCLUDE}" \
    -c "${metal_file}" \
    -o "${output_file}"
  count=$((count + 1))
done

if [ "${count}" -eq 0 ]; then
  echo "No Metal kernels were compiled" >&2
  exit 1
fi

echo "Linking ${count} kernels into ${METALLIB_PATH}"
xcrun -sdk macosx metallib "${AIR_DIR}"/*.air -o "${METALLIB_PATH}"

rm -rf "${BUILD_DIR}"

echo "Metallib built at ${METALLIB_PATH}"
