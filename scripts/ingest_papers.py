#!/usr/bin/env python3
"""Atlas Phase 3 — corpus ingestion CLI (docs/15, docs/16, docs/17).

    scripts/ingest_papers.py --fetch                 # arxiv_ids.txt -> data/papers/*.pdf
    scripts/ingest_papers.py --fetch my_ids.txt      # a different seed list
    scripts/ingest_papers.py --ingest                # PDFs -> data/extracted/*.json
    scripts/ingest_papers.py --chunk                 # extracted -> data/chunks/*.json
    scripts/ingest_papers.py --chunk --exact-tokens  # ...counting real MiniLM wordpieces
    scripts/ingest_papers.py --embed                 # chunks -> vectors -> data/chroma/
    scripts/ingest_papers.py --fetch --ingest --chunk --embed   # the pipeline, in order
    scripts/ingest_papers.py --report                # counts, chunk/embed stats + failures
    scripts/ingest_papers.py --embed --force         # re-embed everything
    scripts/ingest_papers.py --query "flash attention tiling" -k 5      # store smoke path
    scripts/ingest_papers.py --search "how is the kv cache paged" -k 5  # the real retriever
    scripts/ingest_papers.py --search "..." --topic quantization --year-min 2023
    scripts/ingest_papers.py --eval                  # recall@k over rag/eval/queries.json

One CLI, one manifest, one report. Each phase is a separate status transition, so their
failures stay isolated; each is idempotent, so re-running is a no-op and a crash costs at
most one paper. Acquisition talks to arXiv (3 s between requests by default); extraction,
chunking and embedding are pure local compute and need no network after the encoder is
cached. The lab box is shared — pin a card with `CUDA_VISIBLE_DEVICES=0`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rag.chunk import chunk_papers  # noqa: E402
from rag.ingest import (  # noqa: E402  (path bootstrap must precede the import)
    DEFAULT_DELAY,
    IngestError,
    PaperError,
    Paths,
    extract_papers,
    fetch_papers,
    read_id_list,
    report,
)

DEFAULT_ID_LIST = REPO_ROOT / "scripts" / "arxiv_ids.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv PDFs and extract them to structured text with provenance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fetch",
        nargs="?",
        const=str(DEFAULT_ID_LIST),
        metavar="ID_LIST",
        help=f"download PDFs for the arXiv IDs in ID_LIST (default: {DEFAULT_ID_LIST.name})",
    )
    parser.add_argument("--ingest", action="store_true", help="extract PDFs to data/extracted/*.json")
    parser.add_argument("--chunk", action="store_true", help="chunk extracted docs to data/chunks/*.json")
    parser.add_argument(
        "--exact-tokens",
        action="store_true",
        help="chunk against the real MiniLM tokenizer instead of the word heuristic "
        "(how the corpus is built; the library default stays dependency-free)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="embed chunks to data/embeddings/*.npz, derive topics, upsert into data/chroma/",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="smoke query: plain top-k cosine over the store (MMR/reranking are Step 4)",
    )
    parser.add_argument(
        "--search",
        metavar="TEXT",
        help="retrieve: top-k cosine then MMR for diversity, with optional filters (Step 4)",
    )
    # These four default to None rather than to `rag.retrieve`'s constants so that importing
    # numpy is deferred to a run that actually retrieves — `--report` stays dependency-free.
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="results for --query/--search (default: 5)"
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        help="candidates pulled before MMR selects among them (default: 40)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_mult",
        type=float,
        help="MMR relevance/diversity trade-off, 1.0 = plain top-k (default: 0.7)",
    )
    parser.add_argument(
        "--max-per-paper",
        type=int,
        help="cap on chunks from one paper, 0 disables (default: 2)",
    )
    parser.add_argument("--topic", help="filter by a derived topic label (see --report)")
    parser.add_argument(
        "--paper", action="append", metavar="ID", help="filter to a paper_id (repeatable)"
    )
    parser.add_argument("--section", help="filter to one section title, e.g. Method")
    parser.add_argument("--year-min", type=int, help="filter to papers published in or after YEAR")
    parser.add_argument("--year-max", type=int, help="filter to papers published in or before YEAR")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run rag/eval/queries.json and print recall@k, MRR and the miss list",
    )
    parser.add_argument(
        "--device",
        help="torch device for the encoder (default: cuda when available, else cpu)",
    )
    parser.add_argument("--report", action="store_true", help="print manifest counts and failures")
    parser.add_argument("--force", action="store_true", help="ignore skip rules and redo the work")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Paths.default().data_dir,
        help="corpus root (default: data/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"seconds between arXiv requests (default: {DEFAULT_DELAY}); arXiv asks for >= 3",
    )
    args = parser.parse_args(argv)

    stages = (args.fetch, args.ingest, args.chunk, args.embed)
    if not (any(stages) or args.report or args.query or args.search or args.eval):
        parser.error(
            "nothing to do: pass --fetch, --ingest, --chunk, --embed, --query, --search, "
            "--eval and/or --report"
        )

    paths = Paths(data_dir=args.data_dir.expanduser().resolve())
    status = 0

    try:
        if args.fetch:
            paper_ids = read_id_list(Path(args.fetch))
            print(f"seed list {args.fetch}: {len(paper_ids)} ID(s)")
            fetch_papers(paper_ids, paths, force=args.force, delay=args.delay)
        if args.ingest:
            extract_papers(paths, force=args.force)
        if args.chunk:
            counter = {}
            if args.exact_tokens:
                from rag.chunk import EXACT_COUNTER
                from rag.embed import minilm_token_counter

                print("chunk: counting with the real MiniLM tokenizer")
                counter = {
                    "token_counter": minilm_token_counter(device=args.device or "cpu"),
                    "token_counter_name": EXACT_COUNTER,
                }
            chunk_papers(paths, force=args.force, **counter)
        elif args.exact_tokens:
            parser.error("--exact-tokens only means something with --chunk")
        if args.embed:
            from rag.embed import embed_papers  # numpy/chromadb stay off the --report path

            try:
                embed_papers(paths, device=args.device, force=args.force)
            except PaperError as exc:
                # Per-paper failures are recorded state; the run finished, the exit code says
                # so, and --report below names every paper that failed.
                print(f"embed: {exc}", file=sys.stderr)
                status = 1
        if args.query:
            from rag.embed import query_store

            results = query_store(paths, args.query, device=args.device, k=args.top_k)
            print(f"\nquery: {args.query!r}  (top {len(results)})")
            for rank, hit in enumerate(results, start=1):
                metadata = hit["metadata"]
                print(
                    f"  {rank}. {hit['score']:.3f}  {hit['chunk_id']}  "
                    f"p{metadata.get('page_start')}-{metadata.get('page_end')}  "
                    f"[{metadata.get('section') or 'n/a'}]  {metadata.get('title') or ''}"
                )
                print(f"       {' '.join(hit['text'].split())[:200]}")
        if args.search:
            from rag.retrieve import DEFAULT_FETCH_K, DEFAULT_LAMBDA, MAX_PER_PAPER
            from rag.retrieve import format_results, retrieve

            results = retrieve(
                paths,
                args.search,
                k=args.top_k,
                fetch_k=DEFAULT_FETCH_K if args.fetch_k is None else args.fetch_k,
                lambda_mult=DEFAULT_LAMBDA if args.lambda_mult is None else args.lambda_mult,
                max_per_paper=(
                    MAX_PER_PAPER if args.max_per_paper is None else args.max_per_paper
                ),
                paper_id=args.paper,
                topic_label=args.topic,
                year_min=args.year_min,
                year_max=args.year_max,
                section=args.section,
                device=args.device,
            )
            print(format_results(args.search, results))
        if args.eval:
            from rag.eval import format_report, run_eval
            from rag.retrieve import DEFAULT_LAMBDA

            chosen = DEFAULT_LAMBDA if args.lambda_mult is None else args.lambda_mult
            # Both settings, always: MMR's cost in recall is measured, never assumed.
            for lambda_mult, label in ((1.0, "lambda=1.0 (plain top-k)"), (chosen, f"lambda={chosen}")):
                data = run_eval(paths, k=args.top_k, lambda_mult=lambda_mult, device=args.device)
                print()
                print(format_report(data, label=label))
        if args.report or any(stages):
            print()
            print(report(paths))
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — the manifest is current; re-run to resume", file=sys.stderr)
        return 130

    return status


if __name__ == "__main__":
    raise SystemExit(main())
