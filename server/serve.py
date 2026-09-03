"""FastAPI server — POST /generate (SSE streaming), POST /tokenize and GET /health.

docs/14-bridge-serving.md. This module owns HTTP concerns only; all inference goes through
server.bridge.Engine, which goes through the C++/CUDA engine.

    scripts/run_server.sh
    curl -N -X POST localhost:8000/generate \
         -H 'content-type: application/json' \
         -d '{"prompt": "The capital of France is", "max_new_tokens": 8}'
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .bridge import Done, Engine, EngineError, TokenEvent
from .config import Config

log = logging.getLogger("atlas.serve")


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw text to count; no chat template is applied.")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Raw prompt text; no chat template is applied.")
    max_new_tokens: int = Field(default=32, ge=1)
    stream: bool = Field(default=True, description="Stream tokens as SSE, or return the whole completion.")


def _sse(payload: dict | str) -> str:
    """One SSE event. Terminal marker is the literal `data: [DONE]`."""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the engine ONCE at startup.

    Startup pays the mmap and the ~4.4 GB H2D upload before the first request, and a bad
    weights path fails at boot rather than on request one.
    """
    cfg: Config = getattr(app.state, "config", None) or Config.from_env()
    app.state.config = cfg
    # An engine set before startup is reused rather than loaded again -- the tests inject
    # their session engine this way, and a second load would mean a second 4.4 GB upload.
    injected = getattr(app.state, "engine", None)
    engine = injected or Engine.load(cfg)
    app.state.engine = engine
    log.info("engine ready: device=%s module=%s", engine.device, engine.module_path.name)
    try:
        yield
    finally:
        if injected is None:
            app.state.engine = None


app = FastAPI(title="Atlas inference server", version="0.2.0", lifespan=lifespan)


@app.exception_handler(EngineError)
async def _engine_error(_: Request, exc: EngineError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health(request: Request) -> dict:
    engine: Engine | None = getattr(request.app.state, "engine", None)
    return {
        "status": "ok" if engine else "loading",
        "device": engine.device if engine else None,
        "has_cuda": engine.has_cuda if engine else None,
        "model_loaded": engine is not None,
        "max_new_tokens": request.app.state.config.max_new_tokens,
    }


@app.post("/tokenize")
async def tokenize(request: Request, body: TokenizeRequest) -> dict:
    """Token count from the tokenizer that will actually do the encoding.

    Phase 4's agent budgets its prompt against a 2048-token window with no KV cache, so the
    count has to be real: docs/17 measured what a word-count heuristic costs (334/3489 chunks
    silently over the window, and a re-chunk to fix it). The tokenizer is already loaded
    here; asking it is cheaper than guessing anywhere else.
    """
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded")
    return {"n_tokens": len(engine.encode(body.text))}


def _checked(request: Request, body: GenerateRequest) -> Engine:
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded")
    # Capped here rather than in the pydantic model because the cap is per-deployment
    # config, not a schema constant. No KV cache means cost grows with length, so an
    # unbounded request would be a minutes-long response.
    if body.max_new_tokens > engine.max_new_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"max_new_tokens={body.max_new_tokens} exceeds the server cap of {engine.max_new_tokens}",
        )
    return engine


@app.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    engine = _checked(request, body)

    if not body.stream:
        done = await anyio.to_thread.run_sync(engine.generate, body.prompt, body.max_new_tokens)
        return {
            "text": done.text,
            "ids": done.ids,
            "generated": done.generated,
            "finish_reason": done.finish_reason,
            "elapsed_s": round(done.elapsed_s, 4),
        }

    return StreamingResponse(
        _sse_stream(engine, body),
        media_type="text/event-stream",
        # Streaming dies quietly behind a buffering reverse proxy; say so explicitly.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse_stream(engine: Engine, body: GenerateRequest) -> AsyncIterator[str]:
    """Bridge the engine's blocking iterator to an async SSE response.

    The generation itself blocks in C++, so it is pulled on a worker thread; the bridge's
    own worker thread is what makes tokens available before the completion is finished.
    Closing this generator (client disconnect) closes the bridge iterator, which trips the
    docs/13 on_token early-stop -- generation halts instead of running to max_new_tokens
    for a response nobody will read.
    """
    events: Iterator = engine.stream(body.prompt, body.max_new_tokens)
    finished = False
    try:
        while True:
            event = await anyio.to_thread.run_sync(lambda: next(events, None))
            if event is None:
                break
            if isinstance(event, TokenEvent):
                yield _sse({"index": event.index, "id": event.id, "token": event.token})
            elif isinstance(event, Done):
                finished = True
                yield _sse(
                    {
                        "done": True,
                        "generated": event.generated,
                        "finish_reason": event.finish_reason,
                        "elapsed_s": round(event.elapsed_s, 4),
                    }
                )
                yield _sse("[DONE]")
    finally:
        if not finished:
            log.info("client disconnected mid-stream; stopping generation (finish_reason=disconnect)")
        events.close()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.from_env()
    app.state.config = cfg
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
