"""`python -m agent.cli "question"` — one question, one run. docs/19-agent-bringup.md.

    python -m agent.cli "how does flash attention avoid materializing the attention matrix"
    python -m agent.cli --k 3 --max-new-tokens 48 --json "what is a paged kv cache"
    python -m agent.cli --dry-run "..."      # retrieve + build the prompt, never generate

A real run needs the Phase 2 server up (scripts/run_server.sh; docs/14). `--dry-run` is the
only way to exercise the whole graph on a checkout with no GPU and no server, and it is how
the prompt budget gets inspected without paying a generation for every look.

There is no history, no session and no multi-turn here: that is a Phase 5 surface.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from rag.ingest import Paths
from rag.retrieve import Result

from .engine import DryRunEngineClient, HttpEngineClient
from .nodes import DEFAULT_K, DEFAULT_MAX_NEW_TOKENS, corpus_retriever
from .state import initial_state


def _jsonable(value):
    if isinstance(value, Result):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.cli",
        description="Ask the Atlas corpus a question: retrieve -> synthesize on Atlas's own engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="the question, in quotes")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"chunks to retrieve (default: {DEFAULT_K}; see docs/19 on why it is not rag's 5)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"generation budget, clamped to the server cap (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument("--data-dir", type=Path, default=Paths.default().data_dir, help="corpus root (default: data/)")
    parser.add_argument("--engine-url", help="Phase 2 server base URL (default: $ATLAS_AGENT_ENGINE_URL or :8000)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="retrieve and build the prompt but never call the engine",
    )
    parser.add_argument("--json", action="store_true", help="emit the full final state as JSON")
    parser.add_argument("--show-prompt", action="store_true", help="print the exact prompt sent to the engine")
    args = parser.parse_args(argv)

    from .graph import build_graph  # imports langgraph; keep it off --help

    http = HttpEngineClient(args.engine_url)
    engine = DryRunEngineClient(http) if args.dry_run else http
    paths = Paths(data_dir=args.data_dir.expanduser().resolve())
    graph = build_graph(engine=engine, retriever=corpus_retriever(paths))

    started = time.perf_counter()
    final = graph.invoke(initial_state(args.query, k=args.k, max_new_tokens=args.max_new_tokens))
    elapsed = time.perf_counter() - started

    if args.json:
        payload = {key: _jsonable(value) for key, value in final.items()}
        payload["elapsed_s"] = round(elapsed, 3)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if args.show_prompt and final.get("prompt"):
            print(final["prompt"])
            print("-" * 72)
        print(f"\nQ: {args.query}\n")
        print(final.get("output") or "(no output)")
        used = final.get("used_docs") or []
        if used:
            print("\nsources:")
            for index, doc in enumerate(used, start=1):
                print(f"  [{index}] {doc.citation()}")
        print("\ntrace:")
        for step in final.get("reasoning_steps") or []:
            print(f"  - {step}")
        print(f"  - {elapsed:.2f}s wall")

    if final.get("error"):
        sys.stdout.flush()  # or the stderr line lands ahead of the trace in a pipe
        print(f"error: {final['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
