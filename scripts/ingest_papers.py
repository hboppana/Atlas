#!/usr/bin/env python3
"""Atlas Phase 3 · Step 1 — corpus ingestion CLI (docs/15-rag-ingest-extraction.md).

    scripts/ingest_papers.py --fetch                 # arxiv_ids.txt -> data/papers/*.pdf
    scripts/ingest_papers.py --fetch my_ids.txt      # a different seed list
    scripts/ingest_papers.py --ingest                # PDFs -> data/extracted/*.json
    scripts/ingest_papers.py --chunk                 # extracted -> data/chunks/*.json
    scripts/ingest_papers.py --fetch --ingest --chunk # the whole pipeline, in order
    scripts/ingest_papers.py --report                # counts, chunk stats + failures
    scripts/ingest_papers.py --chunk --force         # re-chunk everything

Every phase is idempotent: re-running is a no-op, and a crash costs at most one paper.
Acquisition talks to arXiv (3 s between requests by default); extraction and chunking are
pure local compute and need no network.
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

    if not (args.fetch or args.ingest or args.chunk or args.report):
        parser.error("nothing to do: pass --fetch, --ingest, --chunk and/or --report")

    paths = Paths(data_dir=args.data_dir.expanduser().resolve())

    try:
        if args.fetch:
            paper_ids = read_id_list(Path(args.fetch))
            print(f"seed list {args.fetch}: {len(paper_ids)} ID(s)")
            fetch_papers(paper_ids, paths, force=args.force, delay=args.delay)
        if args.ingest:
            extract_papers(paths, force=args.force)
        if args.chunk:
            chunk_papers(paths, force=args.force)
        if args.report or args.fetch or args.ingest or args.chunk:
            print()
            print(report(paths))
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — the manifest is current; re-run to resume", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
