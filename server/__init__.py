"""Atlas Phase 2 serving layer — the pybind11 bridge and the FastAPI streaming server.

See docs/14-bridge-serving.md. The public surface is:

    server.config.Config   paths, device, caps, host/port (all ATLAS_* env-overridable)
    server.bridge.Engine   load once, then generate() / stream()
    server.serve.app       FastAPI app: POST /generate (SSE), GET /health

Nothing here reimplements inference; it all calls into the C++/CUDA engine.
"""
