#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP_PATH="${ROOT_DIR}/build/build_metallib.stamp"

mkdir -p "${ROOT_DIR}/build"

if [ "$(uname)" = "Darwin" ]; then
  "${SCRIPT_DIR}/build_metallib.sh"
fi

touch "${STAMP_PATH}"
