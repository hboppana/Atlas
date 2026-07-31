"""Shared fixtures + the SKIP-green discipline for server/tests.

Mirrors the C++ tests (test_forward_gpu / test_generate_gpu): missing extension module or
missing weight blob is a green SKIP, not a failure, so a checkout without the 4.4 GB
gitignored blob still runs the suite.
"""

from __future__ import annotations

import pytest

from server.bridge import Engine, EngineError, load_native_module
from server.config import Config

# The reference prompt and the oracle values docs/13 recorded on the A6000 (Suramar,
# sm_86, 2026-07-24), reproduced by the CPU engine. These are exact, not tolerances.
PROMPT = "The capital of France is"
PROMPT_IDS = [1, 450, 7483, 310, 3444, 338]
ORACLE_IDS = [3681, 29889, 13, 13, 29906, 29889, 350, 29889]
ORACLE_TEXT = "Paris.\n\n2. B."
FIRST_TOKEN = 3681  # "Paris"


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.from_env()


@pytest.fixture(scope="session")
def native(config: Config):
    try:
        module, _ = load_native_module(config)
    except EngineError as exc:
        pytest.skip(f"extension module unavailable: {exc}")
    return module


@pytest.fixture(scope="session")
def engine(config: Config, native) -> Engine:
    """One Engine for the whole session — loading costs an mmap and a 4.4 GB upload."""
    for artifact in (config.model_bin, config.model_manifest, config.vocab_file, config.merges_file):
        if not artifact.exists():
            pytest.skip(f"missing engine artifact: {artifact}")
    return Engine.load(config)
