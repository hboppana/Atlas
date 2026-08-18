"""Local embeddings + derived topics — docs/17-rag-embeddings-store.md.

    data/chunks/<paper_id>.json  ->  data/embeddings/<paper_id>.npz  ->  data/chroma/

`--embed` is two passes, and the split is the shape of the step:

    pass 1  per paper    load chunks, embed, cache to .npz, mark the manifest.
                         Resumable, skippable, one paper's failure costs one paper.
    pass 2  corpus-level mean-pool paper vectors, cluster, label the clusters, then write
                         every chunk into Chroma with topic metadata attached.

A cluster is a statement about the corpus, so topics cannot be computed per paper. Pass 2
therefore reruns over the whole embedded set and rewrites topic metadata in place for
papers whose vectors did not change — a metadata-only update, never a re-embed.

Everything that is not literally "call the model" runs against an injected `Embedder`, so
the cache format, the skip rules, k-means, the labelling and the Chroma write path are all
testable on a checkout with no `sentence-transformers` installed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .ingest import (
    CHUNKED,
    EMBEDDED,
    FAILED,
    IngestError,
    Log,
    Manifest,
    PaperError,
    Paths,
    _utc_now,
    sha256_file,
    write_json,
)

# The pinned encoder. Weights are downloaded once from the HF hub at this revision and
# cached; every run after that is offline — the same posture as Phase 1's reference weights.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBED_DIM = 384
# MiniLM truncates silently past this many wordpieces. Step 2 budgeted 200 tokens against
# it with a word heuristic; `truncation_stats` below checks that budget with the real
# tokenizer on every run rather than assuming it.
MAX_SEQ_TOKENS = 256
BATCH_SIZE = 64

# Bumping this re-embeds the corpus on the next `--embed` — same role as CHUNKER_VERSION.
EMBEDDER_VERSION = 1

# Topic derivation. k is chosen by silhouette over this range, capped so a cluster is never
# built from fewer than MIN_PAPERS_PER_TOPIC papers on average.
MIN_K = 2
MAX_K = 8
MIN_PAPERS_PER_TOPIC = 4
KMEANS_SEED = 0
KMEANS_MAX_ITER = 100
TOPIC_LABEL_TERMS = 3

# Sections whose chunks stand in for "the abstract". Papers where `Abstract` is not its own
# line keep the abstract inside front matter (see docs/16), so both are read.
ABSTRACT_SECTIONS = ("abstract", "front matter")


class Embedder:
    """The seam. `encode` returns (n, dim) float32 rows that are already L2-normalized;
    `count_tokens` returns the encoder's true token count per text (not an estimate)."""

    dim: int = EMBED_DIM

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def count_tokens(self, texts: Sequence[str]) -> list[int]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# the real encoder
# --------------------------------------------------------------------------------------


class MiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 through sentence-transformers. Loaded lazily so importing this
    module — and therefore `rag.ingest`'s report — costs nothing without torch installed."""

    def __init__(self, *, device: str | None = None, revision: str = MODEL_REVISION) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised on a checkout without deps
            raise IngestError(
                "sentence-transformers is not installed — `pip install -r requirements.txt`"
            ) from exc

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                device = "cpu"

        self.device = device
        self.model = SentenceTransformer(MODEL_NAME, revision=revision, device=device)
        self.model.eval()
        # Renamed in sentence-transformers 5; the old name still works but warns.
        dimension = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(dimension())
        if self.dim != EMBED_DIM:  # pragma: no cover - would mean the pin moved under us
            raise IngestError(f"{MODEL_NAME} returned dim {self.dim}, expected {EMBED_DIM}")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            vectors = self.model.encode(
                list(texts),
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,  # normalize once, here, and never again
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        encoded = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]


def minilm_token_counter(*, device: str = "cpu") -> Callable[[str], int]:
    """The real wordpiece counter, shaped for `chunk_papers(token_counter=...)`.

    Chunking's *default* stays the word heuristic on purpose: making the chunker's output
    depend on whether sentence-transformers happens to be installed would be a
    reproducibility bug. This exists for the one case docs/17 describes — the truncation
    check finding chunks over 256 wordpieces, which is repaired by re-chunking with this
    counter injected and CHUNKER_VERSION bumped.
    """
    embedder = MiniLMEmbedder(device=device)
    return lambda text: embedder.count_tokens([text])[0]


# --------------------------------------------------------------------------------------
# embedding one paper
# --------------------------------------------------------------------------------------


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class PaperVectors:
    """One paper's cached rows, in `chunk_index` order."""

    paper_id: str
    ids: list[str]
    vectors: np.ndarray  # (n_chunks, dim) float32, unit-norm
    token_counts: np.ndarray  # (n_chunks,) int32, true encoder tokens
    meta: dict


def embed_texts(texts: Sequence[str], embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
    """-> (vectors, token_counts), both row-aligned to `texts`.

    Batches are formed over length-sorted texts to cut padding waste, then the results are
    scattered back into input order: chunk order on disk is always `chunk_index` order.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    vectors: np.ndarray | None = None
    tokens = np.zeros((len(texts),), dtype=np.int32)

    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start : start + BATCH_SIZE]
        batch_texts = [texts[index] for index in batch]
        encoded = np.asarray(embedder.encode(batch_texts), dtype=np.float32)
        if encoded.ndim != 2 or len(encoded) != len(batch):
            raise PaperError(f"embedder returned {encoded.shape}, expected ({len(batch)}, dim)")
        if vectors is None:
            vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
        vectors[batch] = encoded
        tokens[batch] = np.asarray(embedder.count_tokens(batch_texts), dtype=np.int32)

    assert vectors is not None
    # Cheap insurance: the contract says rows arrive normalized, and Step 4's MMR is a
    # plain dot product that silently degrades if they are not.
    return _l2_normalize(vectors), tokens


def embed_chunk_file(
    payload: dict,
    embedder: Embedder,
    *,
    source_sha256: str = "",
    embedder_version: int = EMBEDDER_VERSION,
) -> PaperVectors:
    """Chunk-file payload -> the cache contents. Pure apart from calling the encoder."""
    paper_id = payload.get("paper_id")
    if not paper_id:
        raise PaperError("chunk file has no paper_id")
    chunks = payload.get("chunks") or []
    if not chunks:
        raise PaperError("chunk file has no chunks")

    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    vectors, tokens = embed_texts([str(chunk.get("text") or "") for chunk in chunks], embedder)
    meta = {
        "paper_id": paper_id,
        "embedder_version": embedder_version,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "chunker_version": payload.get("chunker_version"),
        "source_sha256": source_sha256,
        "dim": int(vectors.shape[1]),
    }
    return PaperVectors(paper_id, ids, vectors, tokens, meta)


def write_cache(path: Path, cached: PaperVectors) -> None:
    """`np.savez` with the docs/17 keys plus `token_counts`, so the truncation histogram in
    `--report` never needs the model reloaded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.array(cached.ids, dtype="U"),
        vectors=cached.vectors.astype(np.float32),
        token_counts=cached.token_counts.astype(np.int32),
        meta=np.array(json.dumps(cached.meta, sort_keys=True)),
    )


def load_cache(path: Path) -> PaperVectors:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        tokens = (
            data["token_counts"].astype(np.int32)
            if "token_counts" in data.files
            else np.zeros((len(data["ids"]),), dtype=np.int32)
        )
        return PaperVectors(
            paper_id=str(meta.get("paper_id") or path.stem),
            ids=[str(value) for value in data["ids"]],
            vectors=data["vectors"].astype(np.float32),
            token_counts=tokens,
            meta=meta,
        )


def truncation_stats(token_counts: Sequence[int]) -> dict:
    """The one measurement Step 3 owes Step 2: how the real wordpiece counts sat against
    MiniLM's 256-token window. `over` > 0 means chunks were silently truncated."""
    counts = np.asarray(list(token_counts), dtype=np.int32)
    if counts.size == 0:
        return {"n": 0, "max": 0, "p99": 0, "over": 0, "limit": MAX_SEQ_TOKENS}
    return {
        "n": int(counts.size),
        "max": int(counts.max()),
        "p99": int(np.percentile(counts, 99)),
        "over": int((counts > MAX_SEQ_TOKENS).sum()),
        "limit": MAX_SEQ_TOKENS,
    }


def format_truncation(stats: dict) -> str:
    return (
        f"tokens: max {stats['max']}, p99 {stats['p99']}, "
        f"over {stats['limit']}: {stats['over']} / {stats['n']}"
    )


# --------------------------------------------------------------------------------------
# k-means, from scratch (docs/17 § Derived topics)
# --------------------------------------------------------------------------------------


def _sq_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(n, k) squared euclidean. On unit vectors this orders identically to cosine."""
    return (
        (vectors**2).sum(axis=1)[:, None]
        - 2.0 * vectors @ centroids.T
        + (centroids**2).sum(axis=1)[None, :]
    )


def _kmeans_plusplus(vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(vectors)
    first = int(rng.integers(n))
    chosen = [first]
    closest = ((vectors - vectors[first]) ** 2).sum(axis=1)
    while len(chosen) < k:
        total = float(closest.sum())
        if total <= 0.0:  # every point already coincides with a centre
            pick = int(rng.integers(n))
        else:
            pick = int(rng.choice(n, p=closest / total))
        chosen.append(pick)
        closest = np.minimum(closest, ((vectors - vectors[pick]) ** 2).sum(axis=1))
    return vectors[chosen].copy()


def kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int = KMEANS_SEED,
    max_iter: int = KMEANS_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ init and a fixed seed -> (labels, centroids).

    ~40 lines against a large transitive dependency tree for 43 points in 384 dimensions;
    the from-scratch ethos and the dependency budget point the same way here.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(vectors)
    if n == 0:
        raise IngestError("k-means needs at least one vector")
    k = max(1, min(k, n))

    rng = np.random.default_rng(seed)
    centroids = _kmeans_plusplus(vectors, k, rng)
    labels = np.zeros(n, dtype=np.int64)

    for step in range(max_iter):
        distances = _sq_distances(vectors, centroids)
        new_labels = distances.argmin(axis=1)
        if step and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
            else:
                # Re-seed an emptied cluster on the worst-served point rather than carrying
                # a dead centroid: k was chosen deliberately, so k clusters must come back.
                orphan = int(distances.min(axis=1).argmax())
                centroids[cluster] = vectors[orphan]
                labels[orphan] = cluster
    return labels.astype(np.int64), centroids.astype(np.float32)


def silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over euclidean distances. A singleton cluster scores 0, not 1 —
    otherwise k-selection is won by shattering the corpus into single-paper clusters."""
    vectors = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    if len(vectors) < 2 or len(clusters) < 2:
        return 0.0

    distances = np.sqrt(np.maximum(_sq_distances(vectors, vectors), 0.0))
    scores = []
    for index in range(len(vectors)):
        same = labels == labels[index]
        same[index] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = float(distances[index][same].mean())
        b = min(
            float(distances[index][labels == cluster].mean())
            for cluster in clusters
            if cluster != labels[index]
        )
        scale = max(a, b)
        scores.append((b - a) / scale if scale > 0 else 0.0)
    return float(np.mean(scores))


def choose_k(vectors: np.ndarray, *, seed: int = KMEANS_SEED) -> tuple[int, np.ndarray, float]:
    """Best mean silhouette over k in [MIN_K, MAX_K], capped at n // MIN_PAPERS_PER_TOPIC.

    A hand-picked k on a corpus that is meant to broaden is a constant that goes stale
    without ever failing. A corpus too small to cluster falls back to a single topic.
    """
    n = len(vectors)
    upper = min(MAX_K, n // MIN_PAPERS_PER_TOPIC)
    if upper < MIN_K:
        return 1, np.zeros(n, dtype=np.int64), 0.0

    best_k, best_labels, best_score = 1, np.zeros(n, dtype=np.int64), -2.0
    for k in range(MIN_K, upper + 1):
        labels, _ = kmeans(vectors, k, seed=seed)
        score = silhouette(vectors, labels)
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    return best_k, best_labels, best_score


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters largest-first, ties broken by first member, so `topic_id` does not
    hop between runs on an unchanged corpus. (It still moves when the corpus does — see the
    instability note in docs/17.)"""
    labels = np.asarray(labels)
    order = sorted(
        np.unique(labels),
        key=lambda cluster: (-int((labels == cluster).sum()), int(np.argmax(labels == cluster))),
    )
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([remap[int(value)] for value in labels], dtype=np.int64)


# --------------------------------------------------------------------------------------
# topic labels from term statistics
# --------------------------------------------------------------------------------------

_TERM = re.compile(r"[a-z][a-z0-9+\-]{2,}")

STOPWORDS = frozenset(
    """
    the and for with that this from are was were which their they our all can has have had
    not but its into than then when where while these those such over under between more
    most also very both each other some any one two three new used using use uses used
    based show shows shown given give gives given via per due upon been being does did done
    however therefore thus hence although though because about across after before during
    within without toward towards among against able only same much many few less least
    first second third work works paper papers method methods approach approaches result
    results model models data set sets task tasks propose proposed proposes present presents
    study studies experiment experiments evaluate evaluation performance state art
    """.split()
)


def _terms(text: str) -> list[str]:
    """Unigrams and bigrams. A bigram is kept only when neither half is a stopword, so
    `attention kernels` survives and `of attention` never forms."""
    words = _TERM.findall(text.lower())
    unigrams = [word for word in words if word not in STOPWORDS]
    bigrams = [
        f"{left} {right}"
        for left, right in zip(words, words[1:])
        if left not in STOPWORDS and right not in STOPWORDS
    ]
    return unigrams + bigrams


def paper_label_text(document: dict | None, chunks: Sequence[dict]) -> str:
    """Title + abstract-ish chunks: the part of a paper that says what it is about."""
    parts: list[str] = []
    if document and document.get("title"):
        parts.append(str(document["title"]))
    abstract = [
        str(chunk.get("text") or "")
        for chunk in chunks
        if str(chunk.get("section") or "").strip().lower() in ABSTRACT_SECTIONS
    ]
    if not abstract:  # no detectable abstract (2/43 papers) — the opening chunks stand in
        abstract = [str(chunk.get("text") or "") for chunk in chunks[:2]]
    parts.extend(abstract)
    return "\n".join(parts)


def label_clusters(
    paper_terms: Sequence[Counter],
    labels: Sequence[int],
    *,
    top_n: int = TOPIC_LABEL_TERMS,
) -> list[dict]:
    """-> per cluster: `label` plus the scored terms behind it.

    Score is `tf_in_cluster / (tf_in_corpus + 1)`: a term the cluster owns approaches 1, a
    term the whole corpus shares is divided down to noise. A term must also appear in at
    least two of the cluster's papers, so one chatty paper cannot name the cluster alone.
    """
    corpus = Counter()
    for counts in paper_terms:
        corpus.update(counts)

    out: list[dict] = []
    for cluster in sorted(set(int(value) for value in labels)):
        members = [index for index, value in enumerate(labels) if int(value) == cluster]
        tf = Counter()
        df = Counter()
        for index in members:
            tf.update(paper_terms[index])
            df.update(paper_terms[index].keys())
        min_df = min(2, len(members))

        scored = sorted(
            (
                (count / (corpus[term] + 1.0), term)
                for term, count in tf.items()
                if df[term] >= min_df
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )

        chosen: list[dict] = []
        for score, term in scored:
            # `attention` and `attention kernels` say the same thing once; keep the first.
            if any(term in taken["term"] or taken["term"] in term for taken in chosen):
                continue
            chosen.append({"term": term, "score": round(float(score), 4)})
            if len(chosen) == top_n:
                break

        out.append(
            {
                "topic_id": cluster,
                "label": ", ".join(item["term"] for item in chosen) or f"topic {cluster}",
                "terms": chosen,
            }
        )
    return out


def derive_topics(
    paper_ids: Sequence[str],
    paper_vectors: np.ndarray,
    paper_terms: Sequence[Counter],
    *,
    seed: int = KMEANS_SEED,
) -> dict:
    """Cluster paper vectors, label the clusters, and describe the result well enough that
    a stale downstream `topic_id` is diagnosable — hence `derived_at` and the member lists."""
    k, labels, score = choose_k(paper_vectors, seed=seed)
    labels = _canonical_labels(labels)
    clusters = label_clusters(paper_terms, labels)
    for cluster in clusters:
        cluster["paper_ids"] = [
            paper_ids[index]
            for index, value in enumerate(labels)
            if int(value) == cluster["topic_id"]
        ]
    return {
        "derived_at": _utc_now(),
        "k": int(k),
        "silhouette": round(float(score), 4),
        "n_papers": len(paper_ids),
        "model_name": MODEL_NAME,
        "seed": seed,
        "assignments": {paper_id: int(labels[index]) for index, paper_id in enumerate(paper_ids)},
        "topics": clusters,
    }


# --------------------------------------------------------------------------------------
# the two passes
# --------------------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _embed_one(
    paths: Paths,
    paper_id: str,
    embedder: Embedder,
    *,
    embedder_version: int,
) -> PaperVectors:
    payload = _load_json(paths.chunk_path(paper_id))
    cached = embed_chunk_file(
        payload,
        embedder,
        source_sha256=sha256_file(paths.chunk_path(paper_id)),
        embedder_version=embedder_version,
    )
    write_cache(paths.embedding_path(paper_id), cached)
    return cached


def embed_papers(
    paths: Paths,
    *,
    embedder: Embedder | None = None,
    device: str | None = None,
    force: bool = False,
    embedder_version: int = EMBEDDER_VERSION,
    log: Log = print,
) -> Manifest:
    """`--embed`: pass 1 per paper, then pass 2 over the corpus. Returns the manifest.

    Skip key is `chunk file sha256` + `embedder_version` (the chunk file carries
    `chunker_version`, so a chunker bump changes its hash and re-embeds). A per-paper
    failure is recorded state; the remaining papers still finish.
    """
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(paths.manifest_path)

    candidates = [
        paper_id
        for paper_id, record in sorted(manifest.records.items())
        if record.get("status") in (CHUNKED, EMBEDDED) and paths.chunk_path(paper_id).exists()
    ]
    if not candidates:
        log(f"embed: no chunked papers under {paths.chunks_dir}")
        return manifest

    log(f"embed: {len(candidates)} chunked paper(s) on disk")
    lazy = embedder is None
    dirty: set[str] = set()
    failed = False

    for paper_id in candidates:
        record = manifest.get(paper_id) or {}
        digest = sha256_file(paths.chunk_path(paper_id))
        cache_path = paths.embedding_path(paper_id)

        current = (
            record.get("status") == EMBEDDED
            and record.get("chunk_sha256") == digest
            and record.get("embedder_version") == embedder_version
            and cache_path.exists()
        )
        if not force and current:
            log(f"  skip     {paper_id}  (unchanged)")
            continue

        if lazy and embedder is None:  # pay for the model only once something needs embedding
            embedder = MiniLMEmbedder(device=device)
            log(f"  model    {MODEL_NAME} on {getattr(embedder, 'device', '?')}")

        try:
            cached = _embed_one(paths, paper_id, embedder, embedder_version=embedder_version)
        except (PaperError, json.JSONDecodeError, KeyError, ValueError) as exc:
            cache_path.unlink(missing_ok=True)  # never leave a stale cache behind
            manifest.record(
                paper_id,
                status=FAILED,
                chunk_sha256=digest,
                error=f"embed: {exc}",
            )
            log(f"  FAILED   {paper_id}  {exc}")
            failed = True
            continue

        manifest.record(
            paper_id,
            status=EMBEDDED,
            chunk_sha256=digest,
            embedded_at=_utc_now(),
            embedder_version=embedder_version,
            embedding_count=len(cached.ids),
            error=None,
        )
        dirty.add(paper_id)
        log(f"  embed    {paper_id}  {len(cached.ids)} vector(s)")

    if not dirty:
        log("embed: nothing to re-embed")

    embedded = [
        paper_id
        for paper_id, record in sorted(manifest.records.items())
        if record.get("status") == EMBEDDED and paths.embedding_path(paper_id).exists()
    ]
    if embedded:
        _corpus_pass(paths, manifest, embedded, dirty, log=log)

    if failed:
        raise PaperError(f"{len(manifest.failures())} paper(s) failed — see --report")
    return manifest


def _corpus_pass(
    paths: Paths,
    manifest: Manifest,
    embedded: Sequence[str],
    dirty: set[str],
    *,
    log: Log = print,
) -> dict:
    """Pass 2: topics over the whole embedded set, then the Chroma write."""
    from .store import ChunkStore  # local: Chroma stays out of the import path of --report

    caches = {paper_id: load_cache(paths.embedding_path(paper_id)) for paper_id in embedded}
    chunk_files = {paper_id: _load_json(paths.chunk_path(paper_id)) for paper_id in embedded}
    documents = {
        paper_id: (
            _load_json(paths.document_path(paper_id))
            if paths.document_path(paper_id).exists()
            else None
        )
        for paper_id in embedded
    }

    stats = truncation_stats(
        [count for cache in caches.values() for count in cache.token_counts.tolist()]
    )
    log(f"  {format_truncation(stats)}")
    if stats["over"]:
        log(
            f"  WARNING  {stats['over']} chunk(s) exceed {MAX_SEQ_TOKENS} wordpieces and were "
            "truncated — re-chunk with rag.embed.minilm_token_counter and bump CHUNKER_VERSION"
        )

    # Paper vector = mean of its chunk vectors, renormalized. Abstract-only would be sharper
    # but not every paper has a detectable abstract; the mean is the robust choice.
    paper_vectors = _l2_normalize(
        np.stack([caches[paper_id].vectors.mean(axis=0) for paper_id in embedded])
    )
    paper_terms = [
        Counter(
            _terms(
                paper_label_text(documents[paper_id], chunk_files[paper_id].get("chunks") or [])
            )
        )
        for paper_id in embedded
    ]

    topics = derive_topics(list(embedded), paper_vectors, paper_terms)
    write_json(paths.topics_path, topics)
    labels_by_id = {topic["topic_id"]: topic["label"] for topic in topics["topics"]}
    log(f"  topics   k={topics['k']} silhouette={topics['silhouette']:.3f}")
    for topic in topics["topics"]:
        log(f"    topic_{topic['topic_id']}  {len(topic['paper_ids']):>3} paper(s)  {topic['label']}")

    store = ChunkStore(paths.chroma_dir)
    present = store.paper_summary()
    written = updated = 0

    for paper_id in embedded:
        topic_id = topics["assignments"][paper_id]
        topic_label = labels_by_id[topic_id]
        cache = caches[paper_id]
        chunks = chunk_files[paper_id].get("chunks") or []
        record = manifest.get(paper_id) or {}

        if paper_id in dirty or paper_id not in present:
            metadatas = [
                chunk_metadata(
                    chunk,
                    documents[paper_id],
                    chunker_version=chunk_files[paper_id].get("chunker_version"),
                    embedder_version=record.get("embedder_version"),
                    topic_id=topic_id,
                    topic_label=topic_label,
                )
                for chunk in chunks
            ]
            store.replace_paper(
                paper_id,
                ids=cache.ids,
                documents=[str(chunk.get("text") or "") for chunk in chunks],
                vectors=cache.vectors,
                metadatas=metadatas,
            )
            written += 1
        elif present[paper_id] != (topic_id, topic_label):
            # Vectors did not change, only the corpus around them did: rewrite the two
            # derived fields in place rather than re-running the model.
            store.set_topic(paper_id, topic_id=topic_id, topic_label=topic_label)
            updated += 1

        if record.get("topic_id") != topic_id:
            manifest.record(paper_id, topic_id=topic_id, topic_label=topic_label)

    stale = [paper_id for paper_id in present if paper_id not in set(embedded)]
    for paper_id in stale:
        store.delete_paper(paper_id)

    log(
        f"  store    {store.count()} chunk(s) in {paths.chroma_dir.name}/ "
        f"({written} written, {updated} retopiced, {len(stale)} removed)"
    )
    return topics


def chunk_metadata(
    chunk: dict,
    document: dict | None,
    *,
    chunker_version: int | None,
    embedder_version: int | None,
    topic_id: int | None = None,
    topic_label: str | None = None,
) -> dict:
    """The Chroma metadata row — scalars only, because Chroma rejects lists and None.

    `title`/`year`/`arxiv_categories` are looked up from the extracted document at insert
    time rather than duplicated into every chunk file on disk.
    """
    document = document or {}
    page_span = chunk.get("page_span") or [0, 0]
    categories = document.get("arxiv_categories") or []

    metadata: dict[str, str | int | float | bool] = {
        "paper_id": str(chunk.get("chunk_id", ":")).rsplit(":", 1)[0],
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "section": str(chunk.get("section") or ""),
        "section_index": int(chunk.get("section_index") or 0),
        "page_start": int(page_span[0]),
        "page_end": int(page_span[-1]),
        "n_tokens_est": int(chunk.get("n_tokens_est") or 0),
        "title": str(document.get("title") or ""),
        "arxiv_categories": ",".join(str(value) for value in categories),
    }
    if document.get("year") is not None:
        metadata["year"] = int(document["year"])
    if chunker_version is not None:
        metadata["chunker_version"] = int(chunker_version)
    if embedder_version is not None:
        metadata["embedder_version"] = int(embedder_version)
    if topic_id is not None:
        metadata["topic_id"] = int(topic_id)
        metadata["topic_label"] = str(topic_label or "")
    return metadata


# --------------------------------------------------------------------------------------
# the smoke query path (Step 4 owns retrieval policy; this only proves the store works)
# --------------------------------------------------------------------------------------


def query_store(
    paths: Paths,
    text: str,
    *,
    embedder: Embedder | None = None,
    device: str | None = None,
    k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Plain top-k cosine. Deliberately not MMR, not reranked, not filtered by policy —
    that is Step 4. This exists so the store can be proven before the selection logic."""
    from .store import ChunkStore

    embedder = embedder or MiniLMEmbedder(device=device)
    vector = _l2_normalize(np.asarray(embedder.encode([text]), dtype=np.float32))[0]
    return ChunkStore(paths.chroma_dir).query(vector, k=k, where=where)


# --------------------------------------------------------------------------------------
# report (called from rag.ingest.report, which must not hard-depend on numpy/chromadb)
# --------------------------------------------------------------------------------------


def report_lines(paths: Paths) -> list[str]:
    """Embedding + topic sections of `--report`. Never raises: a missing cache or a missing
    chromadb is a line of output, not a crashed report."""
    lines: list[str] = []
    caches = sorted(paths.embeddings_dir.glob("*.npz")) if paths.embeddings_dir.exists() else []
    if caches:
        counts: list[int] = []
        vectors = 0
        for path in caches:
            try:
                cached = load_cache(path)
            except (OSError, ValueError, KeyError):
                continue
            vectors += len(cached.ids)
            counts.extend(cached.token_counts.tolist())
        lines.append("")
        lines.append(f"embeddings: {vectors} vector(s) across {len(caches)} paper(s)")
        if any(counts):
            lines.append(f"  {format_truncation(truncation_stats(counts))}")

    if paths.topics_path.exists():
        try:
            topics = _load_json(paths.topics_path)
        except (OSError, json.JSONDecodeError):
            topics = None
        if topics:
            lines.append("")
            lines.append(
                f"topics: k={topics.get('k')} silhouette={topics.get('silhouette')} "
                f"derived {topics.get('derived_at')}"
            )
            for topic in topics.get("topics") or []:
                lines.append(
                    f"  topic_{topic.get('topic_id')}  "
                    f"{len(topic.get('paper_ids') or []):>3} paper(s)  {topic.get('label')}"
                )

    if paths.chroma_dir.exists():
        try:
            from .store import ChunkStore

            store = ChunkStore(paths.chroma_dir)
            lines.append("")
            lines.append(f"store: {store.count()} chunk(s) in collection {store.name}")
        except Exception as exc:  # chromadb missing, or a collection this build cannot open
            lines.append("")
            lines.append(f"store: unavailable ({type(exc).__name__}: {exc})")
    return lines


__all__ = [
    "BATCH_SIZE",
    "EMBEDDER_VERSION",
    "EMBED_DIM",
    "MAX_SEQ_TOKENS",
    "MODEL_NAME",
    "MODEL_REVISION",
    "Embedder",
    "MiniLMEmbedder",
    "PaperVectors",
    "chunk_metadata",
    "choose_k",
    "derive_topics",
    "embed_chunk_file",
    "embed_papers",
    "embed_texts",
    "kmeans",
    "label_clusters",
    "load_cache",
    "minilm_token_counter",
    "paper_label_text",
    "query_store",
    "report_lines",
    "silhouette",
    "truncation_stats",
    "write_cache",
]
