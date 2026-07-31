#!/bin/bash
# Atlas Phase 2 · Step 8 — run the FastAPI inference server (docs/14-bridge-serving.md).
#
# Loads the tokenizer, the mmap'd weights and (on the CUDA build) the 4.4 GB device blob
# ONCE at startup, then serves POST /generate as an SSE token stream.
#
#   scripts/run_server.sh                          # auto: build-cuda/ module if present
#   ATLAS_DEVICE=cpu scripts/run_server.sh         # force the CPU engine (~11 s/token)
#   CUDA_VISIBLE_DEVICES=1 scripts/run_server.sh   # pin a card; the box is shared
#
#   curl -N -X POST localhost:8000/generate \
#        -H 'content-type: application/json' \
#        -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
#
# Requires the extension module: scripts/build_cuda.sh (GPU) or a CPU build configured
# with -DATLAS_BUILD_PYTHON=ON.

set -euo pipefail

cd "$(dirname "$0")/.."

# The interpreter that has the Phase 2 deps AND matches the module's ABI. Override with
# PYTHON=/path/to/python (e.g. a conda env). See docs/14 on this box's env situation.
PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3)}"

if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "run_server: fastapi/uvicorn missing for $PYTHON" >&2
    echo "run_server: pip install -r requirements.txt" >&2
    exit 1
fi

echo "run_server: python  $PYTHON"
echo "run_server: device  ${ATLAS_DEVICE:-auto} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"
echo "run_server: startup loads the weights before the first request; give it a moment."

exec "$PYTHON" -m server.serve
