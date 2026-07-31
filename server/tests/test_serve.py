"""FastAPI endpoint validation — docs/14-bridge-serving.md.

The HTTP layer only: that /generate streams rather than buffers, that the streamed text
equals the batch text equals the bridge's, and that validation and disconnect behave.
Engine correctness is test_bridge.py's job.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.config import Config
from server.serve import app
from server.tests.conftest import ORACLE_IDS, ORACLE_TEXT, PROMPT


@pytest.fixture(scope="module")
def client(engine, config: Config):
    """A TestClient over the app, reusing the session Engine rather than loading a second.

    A second Engine would mean a second 4.4 GB upload, and the lifespan handler's own load
    is exercised by scripts/run_server.sh, not here.
    """
    app.state.config = config
    app.state.engine = engine  # lifespan reuses a pre-set engine instead of loading again
    with TestClient(app) as test_client:
        assert app.state.engine is engine
        yield test_client


def _events(response) -> list[dict | str]:
    """Parse an SSE body into its payloads, preserving the terminal [DONE] marker."""
    out: list[dict | str] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_health(client, engine) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == engine.device
    assert body["has_cuda"] == engine.has_cuda


def test_stream_yields_tokens_then_done(client) -> None:
    response = client.post("/generate", json={"prompt": PROMPT, "max_new_tokens": 8})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response)
    assert events[-1] == "[DONE]"

    terminal = events[-2]
    assert terminal["done"] is True
    assert terminal["generated"] == 8
    assert terminal["finish_reason"] == "length"

    tokens = events[:-2]
    assert [e["id"] for e in tokens] == ORACLE_IDS
    assert [e["index"] for e in tokens] == sorted(e["index"] for e in tokens)
    assert "".join(e["token"] for e in tokens) == ORACLE_TEXT


def test_non_streaming_matches_streaming(client) -> None:
    body = client.post(
        "/generate", json={"prompt": PROMPT, "max_new_tokens": 8, "stream": False}
    ).json()
    assert body["text"] == ORACLE_TEXT
    assert body["ids"] == ORACLE_IDS
    assert body["finish_reason"] == "length"
    assert body["elapsed_s"] > 0


def test_disconnect_stops_generation(client, engine) -> None:
    """Abandoning the response must free the generation slot rather than run to completion.

    A wedged slot would show up as the next request hanging on the bridge's lock, so the
    follow-up call is the assertion.
    """
    with client.stream("POST", "/generate", json={"prompt": PROMPT, "max_new_tokens": 8}) as r:
        for _ in zip(range(2), r.iter_lines()):
            pass
    assert engine.generate(PROMPT, 1).ids == ORACLE_IDS[:1]


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},                                  # empty prompt
        {"max_new_tokens": 8},                           # missing prompt
        {"prompt": PROMPT, "max_new_tokens": 0},         # below the floor
        {"prompt": PROMPT, "max_new_tokens": 100_000},   # above the server cap
    ],
)
def test_validation_rejects(client, payload) -> None:
    assert client.post("/generate", json=payload).status_code == 422
