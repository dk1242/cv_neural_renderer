#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "Configuring project (out-of-source build)..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR"

echo "Building project..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)"

EXE="$BUILD_DIR/app/VisionEngineApp"
[ -x "$EXE" ] || { echo "Error: executable not found: $EXE" >&2; exit 1; }

# On hybrid-GPU laptops, force the discrete NVIDIA GPU via PRIME offload:
env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia "$EXE"
