"""The synthesis prompt, its source markers, and the token budget — docs/19-agent-bringup.md.

Prompt text is data. It changes far more often than node logic and Steps 2-4 each add
templates of their own, so it lives here rather than in `nodes.py` — the same separation
argument the Phase 4 skill makes everywhere else.

Two facts drive everything in this file.

**The server applies no chat template.** `GenerateRequest.prompt` is documented as "Raw
prompt text; no chat template is applied", and the weights are TinyLlama-1.1B-Chat-v1.0,
tuned on the Zephyr format. So the agent owns the template and it has to be the real one.

**There is no KV cache and the window is 2048.** `max_position = 2048`
(engine/include/model.h) and docs/14 measured ~49 ms/token at length ~38 with cost growing
quadratically. Chunks are packed to TARGET_TOKENS = 200 (rag/chunk.py), so a default k=5 is
~1000 tokens of evidence in front of a model that re-runs the full forward pass per emitted
token. Prompt length is the dominant cost of a run and nothing upstream protects it, so the
agent budgets — see `budget_docs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from rag.retrieve import Result

from .engine import TokenCount

__all__ = [
    "Budget",
    "MAX_POSITION",
    "NO_EVIDENCE_ANSWER",
    "PROMPT_TOKEN_BUDGET",
    "SYSTEM_PROMPT",
    "budget_docs",
    "build_prompt",
    "render_source",
]

# engine/include/model.h. The prompt plus the completion must fit under this.
MAX_POSITION = 2048
# Leaves headroom under MAX_POSITION for max_new_tokens plus the template itself.
PROMPT_TOKEN_BUDGET = 1536

SYSTEM_PROMPT = (
    "You answer questions using only the excerpts provided. "
    "If they do not contain the answer, say so."
)

# What `synthesize` answers with when retrieval came back empty — without calling the
# engine. A RAG system that answers from parametric memory when the corpus has nothing is
# worse than one that says so, and 1.1B is exactly the size where that failure is fluent.
NO_EVIDENCE_ANSWER = "No relevant passages were found in the corpus for this question."


def render_source(result: Result, index: int) -> str:
    """One numbered excerpt block. `index` is 1-based and is what Step 3 resolves.

    The bracketed marker costs a handful of tokens now and saves the whole "which chunk did
    that sentence come from" problem later: `cite_sources` maps `[n]` back to a `chunk_id`.
    """
    year = f" ({result.year})" if result.year else ""
    section = f", {result.section}" if result.section else ""
    pages = (
        f"p{result.page_start}"
        if result.page_start == result.page_end
        else f"pp{result.page_start}-{result.page_end}"
    )
    header = f"[{index}] {result.title or result.paper_id}{year}{section}, {pages}"
    return f"{header}\n{' '.join(result.text.split())}"


def build_prompt(query: str, docs: Sequence[Result]) -> str:
    """The Zephyr-format prompt TinyLlama-1.1B-Chat was tuned on.

    The `</s>` turn terminators and the trailing `<|assistant|>` are not decoration: without
    them the raw model continues the *user* turn instead of answering it.
    """
    excerpts = "\n\n".join(render_source(doc, index) for index, doc in enumerate(docs, start=1))
    user = f"{excerpts}\n\nQuestion: {query}" if excerpts else f"Question: {query}"
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"


@dataclass(frozen=True)
class Budget:
    prompt: str
    used: list[Result]
    dropped: list[Result]
    n_tokens: int
    exact: bool  # False => the count came from the no-server estimate
    limit: int


def budget_docs(
    query: str,
    docs: Sequence[Result],
    count_tokens: Callable[[str], TokenCount],
    *,
    max_new_tokens: int = 0,
    limit: int = PROMPT_TOKEN_BUDGET,
) -> Budget:
    """Pack excerpts in MMR rank order until the next one would breach the budget.

    Dropping the *lowest-ranked* evidence whole is the only defensible truncation. Cutting a
    chunk mid-sentence hands a small model a mangled context with no signal that anything was
    removed, which is exactly how a 1.1B model starts inventing.

    The count comes from the engine's own tokenizer (`/tokenize`); `exact` travels with it so
    a budget decision made from the fallback estimate is never mistaken for a measurement.
    The whole prompt is re-counted per candidate rather than summing per-chunk counts —
    tokenization is not additive across a join, and at k<=5 this is a handful of calls.
    """
    # The completion shares the 2048-token window with the prompt, so the effective ceiling
    # is whichever of the two limits binds first.
    limit = max(1, min(int(limit), MAX_POSITION - max(0, int(max_new_tokens))))

    prompt = build_prompt(query, [])
    count = count_tokens(prompt)
    used: list[Result] = []
    dropped: list[Result] = list(docs)

    for position, doc in enumerate(docs):
        candidate = build_prompt(query, used + [doc])
        candidate_count = count_tokens(candidate)
        if candidate_count.n_tokens > limit:
            # `used` empty here means rank 0 alone does not fit. Keeping it anyway would
            # blow the window; keeping nothing reaches synthesize with an honest
            # empty-evidence state instead.
            break
        used.append(doc)
        dropped = list(docs[position + 1 :])
        prompt, count = candidate, candidate_count

    return Budget(
        prompt=prompt,
        used=used,
        dropped=dropped,
        n_tokens=count.n_tokens,
        exact=count.exact,
        limit=limit,
    )
