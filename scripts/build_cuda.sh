#!/bin/bash
# Atlas Phase 2 — build the CUDA target directly on the lab A6000 box.
#
# No scheduler here: this node (Suramar) has 2x NVIDIA RTX A6000 (sm_86) attached
# directly, no SLURM/modules. The loop is just:
#   edit engine/cuda/  ->  scripts/build_cuda.sh  ->  scripts/test_cuda.sh  ->  iterate
#
# Run from the repo root. Requires nvcc on PATH (CUDA 12.x) and cmake.

set -euo pipefail

cd "$(dirname "$0")/.."

# Out-of-source build dir dedicated to the CUDA configuration, kept separate from any CPU
# build tree. Release: the kernels are the point; debug asserts aren't needed to compile.
cmake -B build-cuda -S . \
    -DATLAS_USE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda --parallel

echo "build_cuda: done -> build-cuda/"
