import { useState } from "react";

const REPO = {
  name: "atlas",
  type: "dir",
  desc: "Full-stack AI system: C++/CUDA inference → RAG → LangGraph → MCP",
  children: [
    { name: "README.md", type: "file", desc: "Project overview, architecture diagram, build instructions, benchmark results" },
    { name: "CMakeLists.txt", type: "file", desc: "Build config — handles CPU-only and CUDA paths" },
    { name: ".gitignore", type: "file", desc: "Ignore weights/, build/, data/papers/, .env" },
    {
      name: "engine/",
      type: "dir",
      desc: "C++/CUDA inference engine — the foundation layer",
      phase: 1,
      children: [
        {
          name: "include/",
          type: "dir",
          desc: "Header files for the inference engine",
          children: [
            { name: "model.h", type: "file", desc: "Model struct — layer weights, hyperparams, forward() signature" },
            { name: "tokenizer.h", type: "file", desc: "BPE tokenizer — encode/decode, vocab loading" },
            { name: "tensor.h", type: "file", desc: "Lightweight tensor class — shape, strides, data pointer, basic ops" },
            { name: "quantize.h", type: "file", desc: "INT8 quantization — scale/zero-point computation, dequant" },
          ],
        },
        {
          name: "src/",
          type: "dir",
          desc: "C++ implementation files",
          children: [
            { name: "model.cpp", type: "file", desc: "CPU forward pass — embedding lookup, attention, FFN, layer norm" },
            { name: "tokenizer.cpp", type: "file", desc: "BPE merge logic, special token handling, vocab from binary file" },
            { name: "tensor.cpp", type: "file", desc: "Tensor memory management, reshape, mmap weight loading" },
            { name: "quantize.cpp", type: "file", desc: "Post-training quantization — FP32 → INT8 weight conversion" },
            { name: "main.cpp", type: "file", desc: "CLI entry point — load model, run inference, print tokens" },
          ],
        },
        {
          name: "cuda/",
          type: "dir",
          desc: "CUDA kernels — the performance layer",
          phase: 2,
          children: [
            { name: "matmul.cu", type: "file", desc: "Tiled matrix multiply — shared memory, coalesced access, A6000 tuned" },
            { name: "attention.cu", type: "file", desc: "Fused attention kernel — Q·K^T softmax V in one launch" },
            { name: "layernorm.cu", type: "file", desc: "Fused layer norm — single-pass mean+variance, warp-level reduction" },
            { name: "kernels.cu", type: "file", desc: "Utility kernels — GELU, residual add, embedding lookup" },
          ],
        },
        {
          name: "tests/",
          type: "dir",
          desc: "Correctness tests against HuggingFace reference",
          children: [
            { name: "test_tokenizer.cpp", type: "file", desc: "BPE encode/decode round-trip, edge cases (empty, unicode)" },
            { name: "test_forward.cpp", type: "file", desc: "Compare output logits vs HuggingFace within tolerance" },
            { name: "test_quantize.cpp", type: "file", desc: "Quantized vs full-precision accuracy delta" },
          ],
        },
      ],
    },
    {
      name: "server/",
      type: "dir",
      desc: "Python bridge + FastAPI serving layer",
      phase: 2,
      children: [
        { name: "__init__.py", type: "file", desc: "" },
        { name: "bridge.py", type: "file", desc: "pybind11 bridge — exposes C++ engine.generate() to Python" },
        { name: "serve.py", type: "file", desc: "FastAPI server — /generate endpoint, streaming token output" },
        { name: "config.py", type: "file", desc: "Model path, max tokens, temperature, device selection" },
      ],
    },
    {
      name: "rag/",
      type: "dir",
      desc: "RAG pipeline — document ingestion to retrieval",
      phase: 3,
      children: [
        { name: "__init__.py", type: "file", desc: "" },
        { name: "ingest.py", type: "file", desc: "PDF parsing + chunking — section-aware splits (abstract, methods, results)" },
        { name: "embed.py", type: "file", desc: "Local embeddings via sentence-transformers (all-MiniLM-L6-v2), runs on GPU" },
        { name: "store.py", type: "file", desc: "ChromaDB vector store — insert, index, persist to disk" },
        { name: "retrieve.py", type: "file", desc: "Semantic retrieval + reranking — top-k chunks with MMR diversity" },
      ],
    },
    {
      name: "agent/",
      type: "dir",
      desc: "LangGraph multi-step agent orchestration",
      phase: 4,
      children: [
        { name: "__init__.py", type: "file", desc: "" },
        { name: "graph.py", type: "file", desc: "LangGraph state machine — defines node transitions and conditional edges" },
        { name: "nodes.py", type: "file", desc: "Agent nodes: retrieve, evaluate_relevance, synthesize, cite_sources" },
        { name: "tools.py", type: "file", desc: "Tool definitions — search_papers, query_corpus, summarize, compare" },
        { name: "state.py", type: "file", desc: "TypedDict state schema — query, retrieved_docs, reasoning_steps, output" },
      ],
    },
    {
      name: "mcp/",
      type: "dir",
      desc: "MCP server — exposes Forge as a tool to any MCP client",
      phase: 5,
      children: [
        { name: "__init__.py", type: "file", desc: "" },
        { name: "server.py", type: "file", desc: "MCP server entrypoint — tool registration, request handling, stdio transport" },
        { name: "tools.py", type: "file", desc: "Registered tools: search_papers, ask_corpus, ingest_document, run_inference" },
        { name: "config.py", type: "file", desc: "Server config — model path, corpus path, embedding model, port" },
      ],
    },
    {
      name: "scripts/",
      type: "dir",
      desc: "Utility scripts for setup, benchmarking, data prep",
      children: [
        { name: "download_weights.py", type: "file", desc: "Pull model weights from HuggingFace (GPT-2 or small Llama)" },
        { name: "convert_weights.py", type: "file", desc: "Convert .safetensors → raw binary for C++ mmap loading" },
        { name: "validate.py", type: "file", desc: "Compare Atlas output vs HuggingFace for correctness verification" },
        { name: "benchmark.py", type: "file", desc: "Measure tokens/sec, latency, memory usage across all phases" },
        { name: "ingest_papers.py", type: "file", desc: "Bulk ingest arXiv PDFs on federated learning / differential privacy" },
      ],
    },
    {
      name: "slurm/",
      type: "dir",
      desc: "HiPerGator SLURM job scripts",
      children: [
        { name: "build_cuda.sh", type: "file", desc: "SLURM job — compile CUDA kernels on A6000 GPU node" },
        { name: "benchmark.sh", type: "file", desc: "SLURM job — run full benchmark suite, log to results/" },
        { name: "embed_corpus.sh", type: "file", desc: "SLURM job — generate embeddings for full paper corpus on GPU" },
      ],
    },
    {
      name: "data/",
      type: "dir",
      desc: "Paper corpus and processed chunks (gitignored except structure)",
      children: [
        { name: "papers/", type: "dir", desc: "Raw PDFs from arXiv / Semantic Scholar", children: [] },
        { name: "chunks/", type: "dir", desc: "Processed text chunks with metadata", children: [] },
      ],
    },
    { name: "weights/", type: "dir", desc: "Model weights — gitignored, populated by download_weights.py", children: [] },
    { name: "requirements.txt", type: "file", desc: "Python deps — fastapi, langchain, langgraph, chromadb, sentence-transformers, mcp" },
    { name: "pyproject.toml", type: "file", desc: "Project metadata, build config, optional CUDA extras" },
  ],
};

const PHASES = [
  { num: 1, label: "C++ inference", color: "#7F77DD", weeks: "Weeks 1–3" },
  { num: 2, label: "CUDA + serving", color: "#D85A30", weeks: "Weeks 3–5" },
  { num: 3, label: "RAG pipeline", color: "#1D9E75", weeks: "Weeks 5–6" },
  { num: 4, label: "LangGraph agent", color: "#BA7517", weeks: "Weeks 6–7" },
  { num: 5, label: "MCP server", color: "#378ADD", weeks: "Week 7–8" },
];

function FileIcon({ type }) {
  const s = { width: 16, height: 16, flexShrink: 0 };
  if (type === "dir") {
    return (
      <svg style={s} viewBox="0 0 16 16" fill="none">
        <path d="M1.5 3C1.5 2.17 2.17 1.5 3 1.5h3.17a1.5 1.5 0 011.06.44l.94.94a.5.5 0 00.35.15H13c.83 0 1.5.67 1.5 1.5v7c0 .83-.67 1.5-1.5 1.5H3c-.83 0-1.5-.67-1.5-1.5V3z" fill="#BA7517" opacity="0.85"/>
      </svg>
    );
  }
  const ext = type === "file" ? "" : type;
  const colors = {
    "": "#888780",
  };
  return (
    <svg style={s} viewBox="0 0 16 16" fill="none">
      <path d="M3 1.5h6.59a1 1 0 01.7.29l2.42 2.42a1 1 0 01.29.7V13a1.5 1.5 0 01-1.5 1.5H3A1.5 1.5 0 011.5 13V3A1.5 1.5 0 013 1.5z" fill={colors[""] } opacity="0.5"/>
    </svg>
  );
}

function PhaseBadge({ phase }) {
  const p = PHASES.find(ph => ph.num === phase);
  if (!p) return null;
  return (
    <span style={{
      fontSize: 10,
      fontWeight: 500,
      padding: "1px 7px",
      borderRadius: 99,
      background: p.color + "1A",
      color: p.color,
      marginLeft: 8,
      whiteSpace: "nowrap",
      letterSpacing: "0.02em",
    }}>
      Phase {p.num}
    </span>
  );
}

function TreeNode({ node, depth = 0 }) {
  const [open, setOpen] = useState(depth < 1);
  const isDir = node.type === "dir";
  const hasChildren = isDir && node.children && node.children.length > 0;

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      <div
        onClick={() => hasChildren && setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "flex-start",
          padding: "5px 0 5px " + (depth * 20 + 8) + "px",
          cursor: hasChildren ? "pointer" : "default",
          borderRadius: 4,
          transition: "background 0.12s",
          gap: 6,
        }}
        onMouseEnter={e => { if (hasChildren) e.currentTarget.style.background = "rgba(128,128,120,0.07)"; }}
        onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
      >
        {hasChildren ? (
          <span style={{
            width: 16, height: 16, flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 11, color: "var(--color-text-tertiary)",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s",
            marginTop: 1,
          }}>▶</span>
        ) : (
          <span style={{ width: 16, flexShrink: 0 }} />
        )}
        <FileIcon type={node.type} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{
            fontSize: 13,
            fontWeight: isDir ? 500 : 400,
            color: "var(--color-text-primary)",
          }}>
            {node.name}
          </span>
          {node.phase && <PhaseBadge phase={node.phase} />}
          {node.desc && (
            <span style={{
              fontSize: 12,
              color: "var(--color-text-tertiary)",
              marginLeft: 8,
            }}>
              {node.desc}
            </span>
          )}
        </div>
      </div>
      {open && hasChildren && (
        <div>
          {node.children.map((child, i) => (
            <TreeNode key={child.name + i} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function AtlasRepoTree() {
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "24px 0" }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 28,
          fontWeight: 500,
          color: "var(--color-text-primary)",
          margin: 0,
          letterSpacing: "-0.02em",
        }}>
          atlas
        </h1>
        <p style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 13,
          color: "var(--color-text-secondary)",
          margin: "4px 0 0",
        }}>
          From CUDA kernels to autonomous agents — every layer, from scratch.
        </p>
      </div>

      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 6,
        marginBottom: 16,
        padding: "10px 12px",
        background: "var(--color-background-secondary)",
        borderRadius: 8,
      }}>
        {PHASES.map(p => (
          <div key={p.num} style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11,
            fontFamily: "'IBM Plex Mono', monospace",
            color: "var(--color-text-secondary)",
            padding: "3px 8px",
            borderRadius: 4,
            background: p.color + "0D",
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: 99,
              background: p.color, opacity: 0.8,
            }} />
            <span style={{ fontWeight: 500, color: p.color }}>{p.num}</span>
            <span>{p.label}</span>
            <span style={{ opacity: 0.6 }}>· {p.weeks}</span>
          </div>
        ))}
      </div>

      <div style={{
        border: "1px solid var(--color-border-tertiary)",
        borderRadius: 10,
        padding: "8px 0",
        background: "var(--color-background-primary)",
      }}>
        <TreeNode node={REPO} depth={0} />
      </div>

      <div style={{
        marginTop: 14,
        padding: "10px 14px",
        background: "var(--color-background-secondary)",
        borderRadius: 8,
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 11,
        color: "var(--color-text-tertiary)",
        lineHeight: 1.6,
      }}>
        <span style={{ fontWeight: 500, color: "var(--color-text-secondary)" }}>Stack: </span>
        C++ · CUDA · Python · FastAPI · pybind11 · LangGraph · ChromaDB · sentence-transformers · MCP · HiPerGator (NVIDIA A6000)
      </div>
    </div>
  );
}