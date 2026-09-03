"""The state contract every Phase 4 node is written against — docs/19-agent-bringup.md.

The *whole* Phase 4 schema is defined here in Step 1, even though Step 1 populates only the
retrieval and synthesis keys. A TypedDict that grows a key per step is a contract nobody
can read, and four revisions of it is four rounds of churn in the nodes.

Three decisions are load-bearing:

  * `reasoning_steps` is the one reducer field. Every other key is last-write-wins, which is
    what `output` wants; the trace is the one thing nodes must *append* to. Step 2's loop
    visits `retrieve` twice and plain assignment would erase the first visit's trace — a bug
    that is invisible until the loop exists, so the reducer goes in before the loop does.
  * `retrieved_docs` holds `rag.retrieve.Result` objects, not dicts. Phase 3 made `Result`
    frozen for a stated reason ("a typo in a key should be an AttributeError at the call
    site, not an empty citation inside an answer"), and it already carries everything
    `cite_sources` needs plus `.citation()`.
  * `total=False` everywhere, and nodes return *partial* dicts. LangGraph merges what a node
    returns; a node that returns the whole state is a node that can silently clobber a key
    it never thought about.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rag.retrieve import Result

__all__ = ["AgentState", "Result"]


class AgentState(TypedDict, total=False):
    # --- input ---
    query: str  # the user's question, verbatim
    k: int  # retrieval breadth for this run
    max_new_tokens: int  # generation budget for this run

    # --- retrieval (Step 1) ---
    retrieved_docs: list[Result]  # MMR-ordered, rank 0 first
    retrieval_attempts: int  # Step 2's loop guard; Step 1 always leaves it 1

    # --- relevance (Step 2) ---
    relevance: float | None  # judged sufficiency of retrieved_docs
    needs_retry: bool

    # --- synthesis (Step 1) ---
    output: str  # the answer text
    prompt: str  # the exact prompt sent to the engine
    used_docs: list[Result]  # the subset that survived the token budget
    prompt_tokens: int  # what the budget actually spent
    prompt_tokens_exact: bool  # False => counted by the fallback estimate, not the tokenizer

    # --- citations (Step 3) ---
    citations: list[str]

    # --- always ---
    reasoning_steps: Annotated[list[str], operator.add]
    error: str | None


def initial_state(query: str, *, k: int, max_new_tokens: int) -> AgentState:
    """The entry state for one run. One question in, one run — there is no history here."""
    return AgentState(
        query=query,
        k=int(k),
        max_new_tokens=int(max_new_tokens),
        retrieval_attempts=0,
        reasoning_steps=[],
        error=None,
    )
