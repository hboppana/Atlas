"""Bridge validation — the Python path must return EXACTLY what the C++ path returns.

docs/14-bridge-serving.md. This file deliberately **imports no FastAPI**: it is the
boundary that keeps the two layers of Step 8 separable, so a red test names one layer.

The bar is discrete, not a tolerance — no new numerics are introduced by a binding, so
every assertion here is exact equality against values docs/13 already recorded.
"""

from __future__ import annotations

import gc
import threading
import time

import pytest

from server.bridge import Done, Engine, TokenEvent
from server.tests.conftest import FIRST_TOKEN, ORACLE_IDS, ORACLE_TEXT, PROMPT, PROMPT_IDS


@pytest.fixture(scope="session")
def streamed(engine: Engine) -> list:
    """One 8-token streamed completion, reused by several assertions.

    Session-scoped because on the CPU path a token costs ~11 s (docs/13); on the GPU path
    the whole thing is ~0.43 s.
    """
    return list(engine.stream(PROMPT, 8))


# --- tokenizer ------------------------------------------------------------------------


def test_encode_matches_reference_ids(engine: Engine) -> None:
    assert engine.encode(PROMPT) == PROMPT_IDS


def test_decode_round_trips(engine: Engine) -> None:
    assert engine.decode(engine.encode(PROMPT)) == PROMPT


def test_decode_oracle_ids(engine: Engine) -> None:
    assert engine.decode(ORACLE_IDS) == ORACLE_TEXT


# --- the oracle -----------------------------------------------------------------------


def test_generates_the_oracle_sequence(streamed: list) -> None:
    done = streamed[-1]
    assert isinstance(done, Done)
    assert done.ids == ORACLE_IDS


def test_first_token_is_paris(streamed: list) -> None:
    """Asserted separately so a forward-pass regression is distinguishable from a loop one."""
    assert streamed[0].id == FIRST_TOKEN


def test_completion_text(streamed: list) -> None:
    assert streamed[-1].text == ORACLE_TEXT


def test_finish_reason_and_count(streamed: list) -> None:
    done = streamed[-1]
    assert done.finish_reason == "length"
    assert done.generated == 8


def test_token_indices_are_monotonic(streamed: list) -> None:
    indices = [e.index for e in streamed if isinstance(e, TokenEvent)]
    assert indices == sorted(set(indices))
    assert indices[-1] == 7


def test_stream_equals_batch(engine: Engine, streamed: list) -> None:
    """The assertion that catches the incremental-detokenization trap.

    Per-token decode would drop the first token's leading space and split multi-byte UTF-8;
    only prefix-decode-and-delta makes the concatenation equal the batch text.
    """
    concatenated = "".join(e.token for e in streamed if isinstance(e, TokenEvent))
    assert concatenated == ORACLE_TEXT
    assert concatenated == engine.generate(PROMPT, 8).text


# --- budget + stopping ----------------------------------------------------------------


def test_zero_tokens_runs_no_forward(engine: Engine) -> None:
    started = time.perf_counter()
    done = engine.generate(PROMPT, 0)
    assert done.generated == 0 and done.ids == [] and done.text == ""
    assert time.perf_counter() - started < 1.0  # no forward was run


def test_budget_is_honoured(engine: Engine) -> None:
    assert engine.generate(PROMPT, 4).ids == ORACLE_IDS[:4]


def test_on_token_early_stop(engine: Engine) -> None:
    """The raw docs/13 contract: returning False stops before the next forward."""
    seen: list[int] = []

    def stop_after_two(token_id: int) -> bool:
        seen.append(token_id)
        return len(seen) < 2

    generated = engine.runner.generate(PROMPT_IDS, 8, stop_after_two)
    assert generated == ORACLE_IDS[:2]
    assert seen == ORACLE_IDS[:2]


def test_on_token_returning_none_keeps_going(engine: Engine) -> None:
    """A Python function that falls off the end returns None; that must not stop the loop."""
    generated = engine.runner.generate(PROMPT_IDS, 4, lambda token_id: None)
    assert generated == ORACLE_IDS[:4]


def test_callback_exception_propagates(engine: Engine) -> None:
    """It must surface as a Python exception, not unwind through CUDA or be swallowed."""

    def boom(token_id: int) -> bool:
        raise ValueError("callback blew up")

    with pytest.raises(ValueError, match="callback blew up"):
        engine.runner.generate(PROMPT_IDS, 4, boom)


def test_abandoning_the_stream_stops_generation(engine: Engine, streamed: list) -> None:
    """Closing the iterator must trip on_token's early stop, not run to max_new_tokens."""
    full_elapsed = streamed[-1].elapsed_s

    started = time.perf_counter()
    events = engine.stream(PROMPT, 8)
    for _ in range(2):
        next(events)
    events.close()
    elapsed = time.perf_counter() - started

    assert elapsed < full_elapsed * 0.75, "abandoning the stream did not stop generation early"
    assert not any(t.name == "atlas-generate" for t in threading.enumerate())


def test_lock_is_released_after_abandonment(engine: Engine) -> None:
    """A dropped consumer must not wedge the server's single generation slot."""
    events = engine.stream(PROMPT, 8)
    next(events)
    events.close()
    assert engine.generate(PROMPT, 1).ids == ORACLE_IDS[:1]


# --- boundary hazards -----------------------------------------------------------------


def test_gil_is_released_during_generate(engine: Engine) -> None:
    """Another thread must make progress while the C++ loop runs.

    Fails loudly if the gil_scoped_release around generate() is ever dropped — which would
    manifest in production as the FastAPI event loop freezing for the whole completion.
    """
    done = threading.Event()
    ticks = 0

    def run() -> None:
        try:
            engine.runner.generate(PROMPT_IDS, 2, None)
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while not done.wait(timeout=0.001):
        ticks += 1
    worker.join()
    assert ticks > 10, "the interpreter was blocked while generate() ran"


def test_gpu_model_keeps_its_model_alive(engine: Engine, native, config) -> None:
    """The keep_alive<0,1> regression test.

    GpuModel holds a NON-OWNING const Model* into the Model's mmap. Without keep_alive,
    the idiomatic one-liner below drops the last reference to the Model and leaves the
    GpuModel reading unmapped memory — garbage logits or a segfault, far from the cause.
    """
    if not native.has_cuda:
        pytest.skip("CPU-only module: GpuModel is not bound")

    gpu = native.GpuModel.create(
        native.Model.load(str(config.model_bin), str(config.model_manifest))
    )
    gc.collect()
    assert gpu.generate(PROMPT_IDS, 4, None) == ORACLE_IDS[:4]


def test_module_reports_its_device(engine: Engine, native) -> None:
    assert engine.device == ("cuda" if native.has_cuda else "cpu")
    assert engine.has_cuda == native.has_cuda
