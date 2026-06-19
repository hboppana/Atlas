#!/bin/bash
# Atlas Phase 2 — build the CUDA target on a HiPerGator A6000 node.
#
# CUDA is never compiled on the Windows dev machine; the loop is:
#   edit engine/cuda/ locally  ->  git push / rsync to HiPerGator
#     ->  sbatch slurm/build_cuda.sh     # this script: configure + compile on a GPU node
#     ->  sbatch slurm/test_cuda.sh      # run test_device, report pass
#     ->  iterate
#
# BEFORE THE FIRST sbatch, fill the placeholders below (values are cluster-specific and not
# yet confirmed):
#   <TODO:ACCOUNT>    your SLURM allocation/account (sacctmgr show assoc user=$USER)
#   <TODO:QOS>        QOS if your account needs one (else delete the --qos line)
#   <TODO:PARTITION>  GPU partition name (e.g. gpu / hpg-ai — check `sinfo`)
#   <TODO:CUDA_MOD>   exact CUDA module (e.g. cuda/12.4.1 — `module spider cuda`)
# A6000 = Ampere, sm_86; that arch is pinned in CMakeLists.txt (CMAKE_CUDA_ARCHITECTURES 86).

#SBATCH --job-name=atlas-build-cuda
#SBATCH --account=<TODO:ACCOUNT>
#SBATCH --qos=<TODO:QOS>
#SBATCH --partition=<TODO:PARTITION>
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:20:00
#SBATCH --output=slurm-build-cuda-%j.out

set -euo pipefail

module purge
module load <TODO:CUDA_MOD>
module load cmake

# Out-of-source build dir dedicated to the CUDA configuration (kept separate from any CPU
# build tree). Release: the kernels are the point; debug asserts aren't needed to compile.
cmake -B build-cuda -S . \
    -DATLAS_USE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda --parallel

echo "build_cuda: done -> build-cuda/"
