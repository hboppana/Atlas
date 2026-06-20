#!/bin/bash
# Atlas Phase 2 — build the CUDA target directly on the lab A6000 box.
#
# No scheduler here: this node (Suramar) has 2x NVIDIA RTX A6000 (sm_86) attached
# directly, no SLURM/modules. The loop is just:
#   edit engine/cuda/  ->  scripts/build_cuda.sh  ->  scripts/test_cuda.sh  ->  iterate
#
# Run from the repo root. Requires CUDA 12.x and cmake. nvcc need not be on PATH: this
# box ships the toolkit at /usr/local/cuda (-> CUDA 12.6) but doesn't put bin/ on PATH,
# so we locate it here. Override with CUDA_HOME=/path scripts/build_cuda.sh if needed.

set -euo pipefail

cd "$(dirname "$0")/.."

# Resolve the CUDA toolkit root: explicit CUDA_HOME wins, else nvcc already on PATH, else
# the conventional /usr/local/cuda symlink that the lab box provides.
CUDA_HOME="${CUDA_HOME:-$(command -v nvcc >/dev/null 2>&1 && dirname "$(dirname "$(command -v nvcc)")" || echo /usr/local/cuda)}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "build_cuda: nvcc not found under CUDA_HOME=$CUDA_HOME" >&2
    echo "build_cuda: set CUDA_HOME to your CUDA 12.x toolkit root and retry" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "build_cuda: using $("$CUDA_HOME/bin/nvcc" --version | grep -oE 'release [0-9.]+') at $CUDA_HOME"

# Pin cmake: the system cmake at /usr/local/bin is 3.18.2, below CMakeLists' 3.20 floor.
# The pip --user install at ~/.local/bin/cmake (3.20.5) satisfies it but isn't on PATH in
# bare/non-login shells (CI, cron). Prefer it explicitly; override with CMAKE=/path/to/cmake.
CMAKE="${CMAKE:-$([ -x "$HOME/.local/bin/cmake" ] && echo "$HOME/.local/bin/cmake" || command -v cmake)}"
echo "build_cuda: using cmake $("$CMAKE" --version | head -1 | grep -oE '[0-9.]+') at $CMAKE"

# Out-of-source build dir dedicated to the CUDA configuration, kept separate from any CPU
# build tree. Release: the kernels are the point; debug asserts aren't needed to compile.
# CUDAToolkit_ROOT/CMAKE_CUDA_COMPILER make CMake's CUDA detection independent of PATH.
"$CMAKE" -B build-cuda -S . \
    -DATLAS_USE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"

"$CMAKE" --build build-cuda --parallel

echo "build_cuda: done -> build-cuda/"
