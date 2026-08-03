"""pybind11 bridge — the only Python module that knows _atlas_engine exists.

docs/14-bridge-serving.md. Everything above this file (serve.py, and later rag/, agent/,
mcp/) talks to `Engine`. Python never reimplements inference: this is encode -> generate ->
decode over the C++/CUDA engine and nothing else.
"""

from __future__ import annotations

import importlib.util
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Union

from .config import REPO_ROOT, Config

# One 4.4 GB model, one device blob, per-call scratch, and every launch on the default
# stream: two concurrent generations would interleave kernels with no correctness argument
# behind them. Requests therefore QUEUE. This is a deliberate property of a single-model,
# no-batching server (docs/14), not an oversight -- concurrency arrives with batching.
_GENERATE_LOCK = threading.Lock()

# Sentinel pushed by the worker thread when generation has finished, failed, or stopped.
_DONE = object()


@dataclass(frozen=True)
class TokenEvent:
    """One generated token.

    `index` is the token's position in the generated sequence, so it always matches `id`.
    It is monotonically increasing but may skip: when a token completes a multi-byte UTF-8
    character that a previous token began, the held-back text is merged into this event
    (see `_decode_prefix`).
    """

    index: int
    id: int
    token: str


@dataclass(frozen=True)
class Done:
    """Terminal event, always last unless the generation raised.

    Only reached when the stream ran to completion, so finish_reason is "eos" or "length";
    a disconnect closes the iterator and there is nobody left to hand a Done to (serve.py
    logs that case instead).
    """

    finish_reason: str  # "eos" | "length"
    generated: int
    text: str
    ids: list[int]
    elapsed_s: float


Event = Union[TokenEvent, Done]


class EngineError(RuntimeError):
    pass


def _candidate_module_paths(cfg: Config) -> list[Path]:
    """Where to look for the extension, most-preferred first.

    The module is a build artifact, not an installed package, and *which* .so is loaded is
    what selects the device (docs/14): build-cuda/ has GpuModel, build/ does not.
    """
    if cfg.module_path is not None:
        return [cfg.module_path]
    found: list[Path] = []
    for tree in ("build-cuda", "build"):
        found.extend(sorted((REPO_ROOT / tree / "python").glob("_atlas_engine*.so")))
    return found


def load_native_module(cfg: Config) -> tuple[ModuleType, Path]:
    """Import _atlas_engine from the build tree. Returns (module, path)."""
    candidates = _candidate_module_paths(cfg)
    if not candidates:
        raise EngineError(
            "the _atlas_engine extension module was not found under build-cuda/python/ or "
            "build/python/.\nBuild it with:\n"
            "  scripts/build_cuda.sh                     # GPU (lab box)\n"
            "  cmake -B build -S . -DATLAS_BUILD_PYTHON=ON && cmake --build build\n"
            "or point ATLAS_ENGINE_MODULE at an existing .so."
        )
    path = candidates[0]
    if not path.exists():
        raise EngineError(f"ATLAS_ENGINE_MODULE={path} does not exist")
    spec = importlib.util.spec_from_file_location("_atlas_engine", path)
    if spec is None or spec.loader is None:
        raise EngineError(f"{path} is not an importable extension module")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:
        # An .so built against a different Python (an ABI mismatch) is an unavailable
        # module, not a crash: EngineError keeps the SKIP-green discipline in tests and
        # gives the server a stated reason at startup. Rebuild against this interpreter.
        raise EngineError(f"cannot load {path}: {exc}") from exc
    return module, path


class Engine:
    """Tokenizer + model, loaded once, with a streaming greedy-decode entry point."""

    def __init__(self, native: ModuleType, module_path: Path, cfg: Config) -> None:
        self._native = native
        self._cfg = cfg
        self.module_path = module_path

        for required in (cfg.vocab_file, cfg.merges_file, cfg.model_bin, cfg.model_manifest):
            if not required.exists():
                raise EngineError(f"missing engine artifact: {required}")

        self._tok = native.Tokenizer.load(str(cfg.vocab_file), str(cfg.merges_file))

        want_cuda = cfg.device in ("auto", "cuda")
        if cfg.device == "cuda" and not native.has_cuda:
            raise EngineError(
                f"device='cuda' was requested but {module_path.name} was built without CUDA. "
                "Build with scripts/build_cuda.sh, or set ATLAS_DEVICE=cpu."
            )

        # The CPU Model is loaded either way: on the CUDA path GpuModel holds a non-owning
        # pointer into its mmap, so it must stay referenced for the process' lifetime.
        self._model = native.Model.load(str(cfg.model_bin), str(cfg.model_manifest))
        if want_cuda and native.has_cuda:
            # ~0.3 s / 4.4 GB H2D upload, paid once here rather than per request.
            self._runner = native.GpuModel.create(self._model)
            self.device = "cuda"
        else:
            self._runner = self._model
            self.device = "cpu"

    @classmethod
    def load(cls, cfg: Config | None = None) -> "Engine":
        cfg = cfg or Config.from_env()
        native, path = load_native_module(cfg)
        return cls(native, path, cfg)

    @property
    def has_cuda(self) -> bool:
        return bool(self._native.has_cuda)

    @property
    def max_new_tokens(self) -> int:
        return self._cfg.max_new_tokens

    @property
    def runner(self) -> object:
        """The bound native model actually used for decoding (GpuModel or Model).

        Exposed so tests can exercise the raw generate()/on_token contract directly rather
        than only through stream(); normal callers should not need it.
        """
        return self._runner

    def encode(self, prompt: str) -> list[int]:
        return self._tok.encode(prompt)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def _decode_prefix(self, ids: list[int]) -> str | None:
        """Decode a growing prefix of the generated ids, or None if it ends mid-character.

        Tokenizer::decode is NOT a per-token map: it reassembles <0xNN> byte-fallback runs
        into raw bytes and strips the single leading space introduced by normalization
        (engine/include/tokenizer.h). Decoding tokens one at a time would therefore drop the
        first token's spacing and split multi-byte UTF-8 characters. The rule is to decode
        the whole prefix each step and emit the delta; a prefix that ends inside a UTF-8
        sequence fails to convert to str and is held back until the next token completes it.

        O(n) per token against an O(n) forward pass over the same n -- free.
        """
        try:
            return self._tok.decode(ids)
        except UnicodeDecodeError:
            return None

    def stream(self, prompt: str, max_new_tokens: int) -> Iterator[Event]:
        """Yield TokenEvents as they are produced, then a terminal Done.

        The C++ generate() call blocks for the whole completion, so it runs on a worker
        thread and hands tokens back through a queue. Abandoning the iterator (an HTTP
        client disconnecting) sets the cancel flag, and the next on_token returns False --
        the docs/13 early-stop contract, which stops the loop *before* it pays for another
        forward. At ~0.054 s/token with no KV cache, that matters.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        prompt_ids = self.encode(prompt)
        if max_new_tokens == 0:
            yield Done("length", 0, "", [], 0.0)
            return

        events: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        failure: list[BaseException] = []
        started = time.perf_counter()

        def on_token(token_id: int) -> bool:
            events.put(token_id)
            return not cancelled.is_set()

        def worker() -> None:
            try:
                self._runner.generate(prompt_ids, max_new_tokens, on_token)
            except BaseException as exc:  # re-raised in the consumer's thread below
                failure.append(exc)
            finally:
                events.put(_DONE)

        # Held for the whole generation, so concurrent callers queue rather than interleave.
        _GENERATE_LOCK.acquire()
        thread = threading.Thread(target=worker, name="atlas-generate", daemon=True)
        thread.start()
        generated: list[int] = []
        sent = ""
        drained = False
        try:
            while True:
                item = events.get()
                if item is _DONE:
                    drained = True
                    break
                generated.append(item)
                text = self._decode_prefix(generated)
                if text is None or text == sent:
                    continue  # incomplete UTF-8: merge into the next event
                yield TokenEvent(index=len(generated) - 1, id=item, token=text[len(sent):])
                sent = text
        finally:
            # Reached on normal completion AND on GeneratorExit (consumer went away). Only
            # the latter needs cancelling and draining; `drained` is what tells them apart,
            # and it is also what makes finish_reason below trustworthy.
            if not drained:
                cancelled.set()
                while events.get() is not _DONE:
                    pass
            thread.join()
            _GENERATE_LOCK.release()

        if failure:
            raise failure[0]

        if len(generated) < max_new_tokens:
            # generate() stops short only at EOS, which it excludes from the result.
            finish_reason = "eos"
        else:
            finish_reason = "length"

        yield Done(
            finish_reason=finish_reason,
            generated=len(generated),
            text=sent,
            ids=generated,
            elapsed_s=time.perf_counter() - started,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> Done:
        """Drain stream() and return its terminal event.

        Deliberately built on stream() rather than calling generate() directly: one decode
        path means the batch and streamed texts cannot disagree.
        """
        done: Done | None = None
        for event in self.stream(prompt, max_new_tokens):
            if isinstance(event, Done):
                done = event
        assert done is not None, "stream() must end with a Done event"
        return done
