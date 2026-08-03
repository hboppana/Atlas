"""Atlas Phase 3 RAG pipeline.

See docs/15-rag-ingest-extraction.md. Step 1 is on disk; the rest lands step by step:

    rag.ingest     acquire arXiv PDFs -> data/papers/, extract -> data/extracted/*.json
    rag.chunk      (Step 2) section-aware chunking over the extracted documents
    rag.embed      (Step 3) local sentence-transformers embeddings + derived topics
    rag.store      (Step 3) ChromaDB persistence
    rag.retrieve   (Step 4) MMR retrieval

Everything is local: no hosted embedding or LLM API is involved at any stage.
"""
