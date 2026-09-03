"""The agent's link to Atlas's own engine — docs/19-agent-bringup.md.

The agent reasons with TinyLlama **through the Phase 2 server**, over HTTP, rather than by
importing `server.bridge`. The in-process import is faster and tempting; it also means the
agent can only run where a 4.4 GB blob and a built `.so` are, which breaks the CPU-only test
rule and quietly makes Phase 5's MCP server the *third* thing that loads the engine. One
process owns the weights.

Two operations, and the second one is the interesting one:

    complete(prompt, max_new_tokens) -> text     POST /generate  (stream=False)
    count_tokens(text)               -> TokenCount   POST /tokenize

Token counting is not a nicety. There is no KV cache and the window is 2048, so prompt
length is the dominant cost of a run (docs/14: ~49 ms/token at length ~38, growing
quadratically). Phase 3 Step 3 already paid for guessing at token counts with a word
heuristic — 334/3489 chunks silently over MiniLM's window, and a re-chunk to fix it. So the
agent asks the tokenizer that will actually do the encoding, and when it cannot reach it the
count is marked inexact rather than trusted.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "AgentError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DryRunEngineClient",
    "EngineClient",
    "estimate_tokens",
    "FakeEngineClient",
    "HttpEngineClient",
    "TokenCount",
]

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Not paranoia: k=5 puts ~1000 tokens of evidence in front of a model with no KV cache, and
# a 64-token completion on top of that is tens of seconds of forward passes.
DEFAULT_TIMEOUT = 300.0
# The server's own default cap (server/config.py). Used only when /health is unreachable.
DEFAULT_MAX_NEW_TOKENS = 64
# Deliberately below the ~4 chars/token English average, so the fallback OVER-estimates and
# a budget decision made without the tokenizer errs towards a shorter prompt.
FALLBACK_CHARS_PER_TOKEN = 3.2
# One retry, because a 503 from /generate means "the engine is still loading", which is a
# state that resolves on its own. Anything else is a real error and is raised.
RETRY_PAUSE_S = 2.0

DRY_RUN_OUTPUT = "(dry run: the engine was not called)"


class AgentError(RuntimeError):
    """A failure the graph should record in `state['error']` rather than crash on."""


@dataclass(frozen=True)
class TokenCount:
    n_tokens: int
    exact: bool  # False => the fallback estimate, because the server was unreachable

    def __int__(self) -> int:
        return self.n_tokens


class EngineClient(Protocol):
    """What a node is allowed to ask of the engine. Everything else is `rag/` or `engine/`."""

    def complete(self, prompt: str, max_new_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> TokenCount: ...

    def cap(self) -> int:
        """The server's max_new_tokens ceiling; requests are clamped to it."""


def estimate_tokens(text: str) -> TokenCount:
    """The no-server fallback. Conservative by construction — see FALLBACK_CHARS_PER_TOKEN."""
    return TokenCount(n_tokens=int(math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)), exact=False)


class HttpEngineClient:
    """The real client. `httpx` is already a Phase 2 dependency, so nothing new lands here."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client=None,
    ) -> None:
        # Same ATLAS_-prefixed-env posture as server/config.py.
        self.base_url = (base_url or os.environ.get("ATLAS_AGENT_ENGINE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self._client = client  # an httpx.Client, or a TestClient in the tests
        self._cap: int | None = None

    # -- plumbing ----------------------------------------------------------------------

    @property
    def http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _post(self, path: str, payload: dict):
        try:
            # The timeout lives on the client (set in `http`), not on the call: fastapi's
            # TestClient rejects a per-request `timeout`, and the tests drive the real client.
            return self.http.post(path, json=payload)
        except Exception as exc:  # transport failure: connection refused, DNS, timeout
            raise AgentError(f"POST {self.base_url}{path}: {exc}") from exc

    def close(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            client.close()

    # -- the cap -----------------------------------------------------------------------

    def cap(self) -> int:
        """Read `/health`'s `max_new_tokens` once and clamp to it thereafter.

        Read lazily rather than in `__init__` so constructing a client on a laptop with no
        server is not itself an error — that is what makes `--dry-run` work. The server
        answers an over-cap request with 422, which is correct behaviour; a graph that dies
        on a config mismatch it could have read is a graph that wastes a lab session.
        """
        if self._cap is None:
            self._cap = DEFAULT_MAX_NEW_TOKENS
            try:
                response = self.http.get("/health")
                if response.status_code == 200:
                    value = response.json().get("max_new_tokens")
                    if isinstance(value, int) and value > 0:
                        self._cap = value
            except Exception:
                pass  # unreachable server => the documented default; /generate will say so
        return self._cap

    # -- the two operations ------------------------------------------------------------

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        budget = max(1, min(int(max_new_tokens), self.cap()))
        payload = {"prompt": prompt, "max_new_tokens": budget, "stream": False}
        # stream=False: the next node needs the finished text, and SSE parsing buys nothing
        # here. Streaming to a *user* is a Phase 5 concern; the endpoint already supports it.
        response = self._post("/generate", payload)
        if response.status_code == 503:
            time.sleep(RETRY_PAUSE_S)
            response = self._post("/generate", payload)
        if response.status_code != 200:
            raise AgentError(f"/generate returned {response.status_code}: {_detail(response)}")
        return str(response.json().get("text") or "")

    def count_tokens(self, text: str) -> TokenCount:
        if not text:
            return TokenCount(n_tokens=0, exact=True)
        try:
            response = self._post("/tokenize", {"text": text})
            if response.status_code == 200:
                return TokenCount(n_tokens=int(response.json()["n_tokens"]), exact=True)
        except AgentError:
            pass
        # A budget decision is never silently based on a guess: `exact` travels with the
        # number and lands in the state and the trace.
        return estimate_tokens(text)


def _detail(response) -> str:
    try:
        return str(response.json().get("detail") or response.text)[:200]
    except Exception:
        return str(getattr(response, "text", ""))[:200]


class FakeEngineClient:
    """Records the prompts it was handed and returns a canned completion.

    Every graph test uses this. Same injection discipline as `rag/retrieve.py`'s `embedder=`
    / `store=`, and for the same reason: the orchestration logic must be testable without a
    GPU, a server or the corpus.

    Its `count_tokens` counts whitespace words — exact for the fake's purposes, and it makes
    "a 200-token chunk" something a test can construct.
    """

    def __init__(self, completion: str = "a grounded answer", *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> None:
        self.completion = completion
        self.prompts: list[str] = []
        self.budgets: list[int] = []
        self._cap = int(max_new_tokens)

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        self.budgets.append(max(1, min(int(max_new_tokens), self._cap)))
        return self.completion

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(n_tokens=len(text.split()), exact=True)

    def cap(self) -> int:
        return self._cap


class DryRunEngineClient:
    """`--dry-run`: retrieve, budget and render the prompt, never generate.

    Token counting still goes to `/tokenize` when a server happens to be up, and falls back
    to the estimate when it is not, so the budget can be inspected on a checkout with no GPU
    without paying 30 s of generation for every look.
    """

    def __init__(self, counter: EngineClient | None = None) -> None:
        self._counter = counter if counter is not None else HttpEngineClient()
        self.prompts: list[str] = []

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return DRY_RUN_OUTPUT

    def count_tokens(self, text: str) -> TokenCount:
        return self._counter.count_tokens(text)

    def cap(self) -> int:
        return self._counter.cap()
