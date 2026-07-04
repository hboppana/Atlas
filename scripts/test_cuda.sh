#!/bin/bash
# Atlas Phase 2 — run the CUDA bring-up test directly on the lab A6000 box.
#
# Runs after scripts/build_cuda.sh has produced build-cuda/. Validates Step 1's infra with
# the no-math round-trip: bit-exact H2D/D2H, a launched kernel, the diff harness, and that
# CUDA_CHECK aborts on an injected failure. Blob-free — no 4.4 GB weight blob needed.
#
# The box is shared and both A6000s are usually busy, so pin to one card to avoid contention:
#   CUDA_VISIBLE_DEVICES=1 scripts/test_cuda.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Match the cmake that built build-cuda/ (3.20 pip --user install), not the 3.18 system one.
# Override with CTEST=/path/to/ctest. See scripts/build_cuda.sh for the same pinning rationale.
CTEST="${CTEST:-$([ -x "$HOME/.local/bin/ctest" ] && echo "$HOME/.local/bin/ctest" || command -v ctest)}"

nvidia-smi

# Step 1 device bring-up + Step 2 matmul + Step 3 RMSNorm validation. Later kernel tests
# join this -R filter.
"$CTEST" --test-dir build-cuda -R 'test_device|test_matmul|test_rmsnorm' --output-on-failure

echo "test_cuda: done"
