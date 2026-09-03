"""The engine client against a stub server — docs/19-agent-bringup.md § The engine client.

No weights and no GPU: a tiny FastAPI app stands in for the Phase 2 server and is driven
through fastapi's TestClient (httpx is already a Phase 2 test dep). What is under test is
the *client's* policy — the cap read from /health, the single 503 retry, AgentError on
anything else, and the token-count fallback that must never masquerade as a measurement.
"""

from __future__ import annotations

import pytest

from agent.engine import (
    DEFAULT_MAX_NEW_TOKENS,
    AgentError,
    DryRunEngineClient,
    FakeEngineClient,
    HttpEngineClient,
    estimate_tokens,
)

fastapi = pytest.importorskip("fastapi", reason="Phase 2 dependency; pip install -r requirements.txt")
pytest.importorskip("httpx", reason="Phase 2 test dependency; pip install -r requirements.txt")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class StubServer:
    """The Phase 2 endpoints the agent uses, with the failure modes it must survive."""

    def __init__(self, *, cap: int = 32, fail_times: int = 0, status: int = 503) -> None:
        self.cap = cap
        self.fail_times = fail_times
        self.status = status
        self.requests: list[dict] = []
        self.app = FastAPI()

        @self.app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "device": "cpu", "max_new_tokens": self.cap}

        @self.app.post("/generate")
        async def generate(body: dict) -> dict:
            self.requests.append(body)
            if self.fail_times > 0:
                self.fail_times -= 1
                raise HTTPException(status_code=self.status, detail="engine not loaded")
            return {"text": "Paris.", "generated": 2, "finish_reason": "length"}

        @self.app.post("/tokenize")
        async def tokenize(body: dict) -> dict:
            return {"n_tokens": len(str(body["text"]).split())}

    def client(self) -> HttpEngineClient:
        return HttpEngineClient("http://stub", client=TestClient(self.app))


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    """The retry pause is real seconds on the lab box and nothing here needs to feel them."""
    monkeypatch.setattr("agent.engine.RETRY_PAUSE_S", 0.0)


def test_cap_is_read_from_health_and_clamps_the_request() -> None:
    stub = StubServer(cap=32)
    client = stub.client()
    assert client.cap() == 32
    client.complete("hello", max_new_tokens=1000)
    # The server answers an over-cap request with 422; the client reads the cap instead.
    assert stub.requests[-1]["max_new_tokens"] == 32
    assert stub.requests[-1]["stream"] is False


def test_cap_falls_back_to_the_documented_default_when_health_is_unreachable() -> None:
    client = HttpEngineClient("http://127.0.0.1:9", client=_DeadClient())
    assert client.cap() == DEFAULT_MAX_NEW_TOKENS


def test_503_is_retried_once_then_succeeds() -> None:
    stub = StubServer(fail_times=1, status=503)  # "the engine is still loading"
    assert stub.client().complete("hello", 8) == "Paris."
    assert len(stub.requests) == 2


def test_persistent_503_raises_agent_error() -> None:
    stub = StubServer(fail_times=5, status=503)
    with pytest.raises(AgentError, match="503"):
        stub.client().complete("hello", 8)
    assert len(stub.requests) == 2  # one retry, not a loop


def test_500_raises_agent_error_without_retrying() -> None:
    stub = StubServer(fail_times=5, status=500)
    with pytest.raises(AgentError, match="500"):
        stub.client().complete("hello", 8)
    assert len(stub.requests) == 1


def test_transport_failure_raises_agent_error() -> None:
    client = HttpEngineClient("http://127.0.0.1:9", client=_DeadClient())
    with pytest.raises(AgentError):
        client.complete("hello", 8)


def test_count_tokens_uses_the_server() -> None:
    count = StubServer().client().count_tokens("one two three four")
    assert (count.n_tokens, count.exact) == (4, True)


def test_count_tokens_falls_back_conservatively_and_says_so() -> None:
    client = HttpEngineClient("http://127.0.0.1:9", client=_DeadClient())
    text = "a" * 320
    count = client.count_tokens(text)
    assert count.exact is False
    assert count == estimate_tokens(text)
    # Conservative: the estimate must not UNDER-count relative to ~4 chars/token, or a
    # budget decision made without the tokenizer would overflow the window.
    assert count.n_tokens >= len(text) / 4


def test_fake_client_records_prompts_and_clamps() -> None:
    fake = FakeEngineClient("an answer", max_new_tokens=16)
    assert fake.complete("<|user|>hi</s>", 64) == "an answer"
    assert fake.prompts == ["<|user|>hi</s>"]
    assert fake.budgets == [16]
    assert fake.count_tokens("one two three").n_tokens == 3


def test_dry_run_client_never_generates_but_still_counts() -> None:
    stub = StubServer()
    dry = DryRunEngineClient(stub.client())
    output = dry.complete("prompt text", 64)
    assert "dry run" in output
    assert dry.prompts == ["prompt text"]
    assert stub.requests == []  # the whole point: /generate was never called
    assert dry.count_tokens("one two").n_tokens == 2


class _DeadClient:
    """A transport that always fails, standing in for "no server on this box"."""

    def get(self, path, **kwargs):
        raise ConnectionError(f"connection refused: {path}")

    def post(self, path, **kwargs):
        raise ConnectionError(f"connection refused: {path}")
