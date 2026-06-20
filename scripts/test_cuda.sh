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

nvidia-smi

# Just the device bring-up test for now; later kernel tests join this target and the -R
# filter can widen (e.g. -R 'test_device|test_matmul').
ctest --test-dir build-cuda -R test_device --output-on-failure

echo "test_cuda: done"
