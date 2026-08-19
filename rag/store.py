"""ChromaDB vector store — docs/17-rag-embeddings-store.md § ChromaDB store.

Thin on purpose: Chroma is a dependency to be isolated, not a framework to build on. Every
call the rest of Atlas makes into the vector database goes through `ChunkStore`, so Step 4's
retrieval, Phase 4's agent and Phase 5's MCP tools all depend on this surface rather than on
Chroma's.

Two rules are load-bearing:

  * **Chroma must never embed.** Its default silently downloads its own ONNX MiniLM and
    would embed queries with a *different* pipeline than the corpus — an invisible
    train/serve skew that mostly works, which is what makes it dangerous. Atlas always
    passes explicit vectors produced by `rag.embed`. Note that `embedding_function=None` is
    *not* sufficient on chromadb >= 1.5: it is treated as "unspecified" and the default is
    installed anyway (measured — it downloaded the 79 MB ONNX bundle). So the collection is
    built with `ExplicitVectorsOnly`, an embedding function that raises, turning any code
    path that forgets to pass vectors into a loud error instead of silent skew.
  * Writes are **delete-by-paper, then add**. A re-chunked paper usually has a different
    number of chunks, so an upsert keyed on `chunk_id` leaves orphaned high-index rows from
    the previous run — rows that still retrieve, and that cite page spans which no longer
    exist. An orphan is worse than a missing row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .ingest import IngestError

COLLECTION_NAME = "atlas_chunks"
# Chroma rejects oversized writes; the corpus is chunked into batches well under its limit.
MAX_BATCH = 2000

# Cosine, so the unit-norm vectors written by rag.embed need no renormalization anywhere.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - exercised on a checkout without chromadb
        raise IngestError("chromadb is not installed — `pip install -r requirements.txt`") from exc
    return chromadb


def _explicit_vectors_only():
    """An embedding function that refuses to embed.

    Chroma requires *an* embedding function on a collection and quietly substitutes its
    bundled ONNX MiniLM when none is given. Handing it one that raises keeps the encoder
    question answered in exactly one place — `rag.embed` — and makes a missing
    `embeddings=` argument fail immediately instead of producing rows encoded by a
    different model than everything around them.
    """
    from chromadb.api.types import EmbeddingFunction

    class ExplicitVectorsOnly(EmbeddingFunction):
        def __init__(self) -> None:
            # Deliberately not `super().__init__()`: Chroma's base implementation exists
            # only to deprecation-warn subclasses that never wrote one.
            pass

        @staticmethod
        def name() -> str:
            return "atlas_explicit_vectors_only"

        def __call__(self, input):  # noqa: A002 - Chroma dictates the parameter name
            raise IngestError(
                "Chroma tried to embed text itself. Every add/query in Atlas must pass "
                "explicit vectors from rag.embed — see docs/17-rag-embeddings-store.md."
            )

        def get_config(self) -> dict:
            return {}

        @staticmethod
        def build_from_config(config: dict) -> "ExplicitVectorsOnly":
            return ExplicitVectorsOnly()

    return ExplicitVectorsOnly()


class ChunkStore:
    """`data/chroma/` as a persistent collection of chunk rows.

    document = the chunk text (so Chroma's own `documents` field is what Phase 4 quotes)
    id       = `<paper_id>:<index>`, already the format `cite_sources` expects
    metadata = scalars only, built by `rag.embed.chunk_metadata`
    """

    def __init__(self, chroma_dir: Path, *, name: str = COLLECTION_NAME) -> None:
        chromadb = _chromadb()
        self.path = Path(chroma_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata=dict(COLLECTION_METADATA),
            embedding_function=_explicit_vectors_only(),  # never let Chroma embed anything
        )

    # -- reads ---------------------------------------------------------------------------

    def count(self) -> int:
        return int(self.collection.count())

    def paper_summary(self) -> dict[str, tuple[int | None, str | None]]:
        """paper_id -> (topic_id, topic_label) currently stored. One `get` over metadata is
        enough to decide, per paper, between a full rewrite and a topic-only update."""
        rows = self.collection.get(include=["metadatas"])
        summary: dict[str, tuple[int | None, str | None]] = {}
        for metadata in rows.get("metadatas") or []:
            metadata = metadata or {}
            paper_id = metadata.get("paper_id")
            if paper_id is None:
                continue
            summary.setdefault(
                str(paper_id),
                (metadata.get("topic_id"), metadata.get("topic_label")),
            )
        return summary

    def paper_rows(self, paper_id: str) -> tuple[list[str], list[dict]]:
        rows = self.collection.get(where={"paper_id": paper_id}, include=["metadatas"])
        return list(rows.get("ids") or []), [dict(m or {}) for m in rows.get("metadatas") or []]

    # -- writes --------------------------------------------------------------------------

    def delete_paper(self, paper_id: str) -> None:
        self.collection.delete(where={"paper_id": paper_id})

    def replace_paper(
        self,
        paper_id: str,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict],
    ) -> int:
        """Destructive-then-additive: the only write path for a paper's chunks."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if not (len(ids) == len(documents) == len(metadatas) == len(vectors)):
            raise IngestError(
                f"{paper_id}: ids/documents/metadatas/vectors disagree "
                f"({len(ids)}/{len(documents)}/{len(metadatas)}/{len(vectors)})"
            )
        for metadata in metadatas:
            _check_scalars(paper_id, metadata)

        self.delete_paper(paper_id)
        for start in range(0, len(ids), MAX_BATCH):
            stop = start + MAX_BATCH
            self.collection.add(
                ids=list(ids[start:stop]),
                documents=list(documents[start:stop]),
                embeddings=vectors[start:stop].tolist(),
                metadatas=[dict(metadata) for metadata in metadatas[start:stop]],
            )
        return len(ids)

    def set_topic(self, paper_id: str, *, topic_id: int, topic_label: str) -> int:
        """Metadata-only rewrite of the two derived fields. Chroma replaces a row's metadata
        wholesale on update, so the stored dict is read back and merged rather than passed
        as a fragment."""
        ids, metadatas = self.paper_rows(paper_id)
        if not ids:
            return 0
        merged = []
        for metadata in metadatas:
            metadata["topic_id"] = int(topic_id)
            metadata["topic_label"] = str(topic_label)
            merged.append(metadata)
        for start in range(0, len(ids), MAX_BATCH):
            stop = start + MAX_BATCH
            self.collection.update(ids=ids[start:stop], metadatas=merged[start:stop])
        return len(ids)

    # -- query ---------------------------------------------------------------------------

    def query(
        self,
        vector: np.ndarray,
        *,
        k: int = 5,
        where: dict | None = None,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Plain top-k cosine over an explicit query vector — the ranking Chroma itself
        produces, and the smoke path that proves the store is wired correctly.

        `include_vectors` asks Chroma for the stored rows as well, which is what
        `rag.retrieve` needs for its chunk-to-chunk similarities. Reading those vectors back
        from here rather than from the `.npz` cache keeps the searched rows and the
        diversified rows the same rows — the cache is for work that must not re-run the
        model, not for the hot path.
        """
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        include = ["documents", "metadatas", "distances"]
        if include_vectors:
            include.append("embeddings")
        rows = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=max(1, int(k)),
            where=where or None,
            include=include,
        )
        ids = (rows.get("ids") or [[]])[0]
        documents = (rows.get("documents") or [[]])[0]
        metadatas = (rows.get("metadatas") or [[]])[0]
        distances = (rows.get("distances") or [[]])[0]
        # Chroma returns `embeddings: None` (not an empty list) when it was not requested.
        embeddings = (rows.get("embeddings") if include_vectors else None) or [[]]
        embeddings = embeddings[0] if len(embeddings) else []
        hits = [
            {
                "chunk_id": ids[index],
                "text": documents[index] if index < len(documents) else "",
                "metadata": dict(metadatas[index] or {}) if index < len(metadatas) else {},
                # cosine space: Chroma reports distance, and 1 - d is the similarity
                "score": 1.0 - float(distances[index]) if index < len(distances) else None,
            }
            for index in range(len(ids))
        ]
        if include_vectors:
            for index, hit in enumerate(hits):
                row = embeddings[index] if index < len(embeddings) else None
                hit["vector"] = np.asarray(row, dtype=np.float32) if row is not None else None
        return hits

    def reset(self) -> None:
        """Drop the collection. Used by tests and by a metadata-schema change."""
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata=dict(COLLECTION_METADATA),
            embedding_function=_explicit_vectors_only(),
        )


def _check_scalars(paper_id: str, metadata: dict) -> None:
    """Chroma rejects lists and None with an opaque error deep in its write path; failing
    here names the paper and the key instead."""
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool)):
            raise IngestError(
                f"{paper_id}: metadata[{key!r}] is {type(value).__name__}; "
                "Chroma metadata must be str | int | float | bool"
            )


__all__ = ["COLLECTION_NAME", "MAX_BATCH", "ChunkStore"]
