#!/bin/bash
# Atlas Phase 2 — run the CUDA bring-up test on a HiPerGator A6000 node.
#
# Runs after slurm/build_cuda.sh has produced build-cuda/. Validates Step 1's infra with the
# no-math round-trip: bit-exact H2D/D2H, a launched kernel, the diff harness, and that
# CUDA_CHECK aborts on an injected failure. Blob-free — no 4.4 GB weight blob needed.
#
# Fill the same placeholders as build_cuda.sh before the first sbatch:
#   <TODO:ACCOUNT> <TODO:QOS> <TODO:PARTITION> <TODO:CUDA_MOD>

#SBATCH --job-name=atlas-test-cuda
#SBATCH --account=<TODO:ACCOUNT>
#SBATCH --qos=<TODO:QOS>
#SBATCH --partition=<TODO:PARTITION>
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:10:00
#SBATCH --output=slurm-test-cuda-%j.out

set -euo pipefail

module purge
module load <TODO:CUDA_MOD>
module load cmake

nvidia-smi

# Just the device bring-up test for now; later kernel tests join this target and the -R
# filter can widen (e.g. -R 'test_device|test_matmul').
ctest --test-dir build-cuda -R test_device --output-on-failure

echo "test_cuda: done"
