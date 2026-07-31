"""Serving configuration — docs/14-bridge-serving.md.

Every field is overridable from the environment with an ATLAS_-prefixed variable, and the
path defaults are resolved relative to the repo root so the server runs from anywhere.

There is deliberately NO temperature/top_p knob: decoding is greedy (docs/13), and a
config field that is read and ignored is a lie in the API surface. It lands with sampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# server/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

DEVICES = ("auto", "cuda", "cpu")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


@dataclass(frozen=True)
class Config:
    # weights/ is gitignored and populated by scripts/download_weights.py +
    # scripts/convert_weights.py; reference/tokenizer/ is committed.
    weights_dir: Path = REPO_ROOT / "weights" / "tinyllama-1.1b-chat"
    tokenizer_dir: Path = REPO_ROOT / "reference" / "tokenizer"

    # Explicit path to the built _atlas_engine*.so. None => bridge.py discovers it,
    # preferring build-cuda/python/ over build/python/.
    module_path: Path | None = None

    # "auto" takes CUDA when the loaded module has it. "cuda" against a CPU-only module is
    # a startup error, not a silent downgrade -- a server that quietly runs 200x slower
    # than asked is worse than one that refuses to start.
    device: str = "auto"

    # Hard cap. Requests above it are rejected at validation rather than turning into a
    # minutes-long response: the engine has no KV cache, so cost grows with length.
    max_new_tokens: int = 64

    # Local by default. This server has no auth and holds a 4.4 GB model; it is not
    # something to bind to 0.0.0.0 without deciding to.
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def model_bin(self) -> Path:
        return self.weights_dir / "model.f32.bin"

    @property
    def model_manifest(self) -> Path:
        return self.weights_dir / "model.manifest.txt"

    @property
    def vocab_file(self) -> Path:
        return self.tokenizer_dir / "vocab.txt"

    @property
    def merges_file(self) -> Path:
        return self.tokenizer_dir / "merges.txt"

    @classmethod
    def from_env(cls) -> "Config":
        module_path = os.environ.get("ATLAS_ENGINE_MODULE")
        cfg = cls(
            weights_dir=_env_path("ATLAS_WEIGHTS_DIR", cls.weights_dir),
            tokenizer_dir=_env_path("ATLAS_TOKENIZER_DIR", cls.tokenizer_dir),
            module_path=Path(module_path).expanduser().resolve() if module_path else None,
            device=os.environ.get("ATLAS_DEVICE", cls.device).lower(),
            max_new_tokens=_env_int("ATLAS_MAX_NEW_TOKENS", cls.max_new_tokens),
            host=os.environ.get("ATLAS_HOST", cls.host),
            port=_env_int("ATLAS_PORT", cls.port),
        )
        if cfg.device not in DEVICES:
            raise ValueError(f"ATLAS_DEVICE={cfg.device!r} is not one of {DEVICES}")
        if cfg.max_new_tokens < 1:
            raise ValueError(f"ATLAS_MAX_NEW_TOKENS={cfg.max_new_tokens} must be >= 1")
        return cfg
