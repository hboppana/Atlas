---
name: atlas-mcp
description: Phase 5 — the MCP server (mcp/) that exposes Atlas as tools to any MCP client. Covers tool registration (search_papers, ask_corpus, ingest_document, run_inference), the stdio transport, and server config. Use when writing or reviewing mcp/ (server, tools, config) or wiring Atlas into an MCP client.
---

# Atlas Phase 5 — MCP Server

Expose the whole Atlas system — inference engine, RAG corpus, agent — as tools any MCP
client can call. **Prerequisite:** Phases 1–4 work. This is the integration capstone; it
wraps existing capabilities, it does not reimplement them. Read `atlas-architecture` first.

## Server (mcp/)

| File | Responsibility |
|------|----------------|
| `server.py` | MCP server entrypoint — tool registration, request handling, **stdio transport** |
| `tools.py` | the registered tools (below) |
| `config.py` | server config — model path, corpus path, embedding model, port |
| `__init__.py` | package marker |

## Registered tools

| Tool | Backed by |
|------|-----------|
| `search_papers` | Phase 3 RAG retrieval (`rag/retrieve.py`) |
| `ask_corpus` | Phase 4 LangGraph agent (full retrieve→synthesize→cite flow) |
| `ingest_document` | Phase 3 ingestion (`rag/ingest.py`) |
| `run_inference` | Phase 2 serving engine (`server/`) — raw model generation |

## Conventions

- **Thin wrapper.** Each tool delegates to the layer that already implements it. The MCP
  layer's job is the protocol surface: schema definitions, request/response handling,
  transport — not new model/RAG/agent logic.
- **stdio transport** is the default, so any MCP client (e.g. Claude Desktop, an IDE) can
  spawn and talk to the server.
- Tool schemas should be precise and self-describing — clear names, typed arguments,
  helpful descriptions — since they are the public contract to arbitrary clients.
- Uncomment the Phase 5 dep in `requirements.txt`: `mcp`.
