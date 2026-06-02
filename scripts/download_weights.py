"""
download_weights.py - Phase 1, step 0: establish ground truth.

One-time setup that downloads TinyLlama-1.1B-Chat and captures a golden
reference forward pass on the CPU in FP32. The from-scratch C++ engine in
engine/ is validated against the artifacts this script produces:

  weights/tinyllama-1.1b-chat/    full HF snapshot (safetensors, tokenizer, config)
  reference/config.json           pinned architecture hyperparameters
  reference/prompt.txt            the fixed prompt used for validation
  reference/token_ids.npy         tokenizer(prompt) -> int32 ids    (oracle for test_tokenizer)
  reference/logits.npy            model(prompt) -> float32 logits   (oracle for test_forward)

Everything is forced to FP32 on CPU so the reference is deterministic and
directly comparable to the C++ CPU forward pass.

Setup:
  pip install -r requirements.txt
Run from the repo root:
  python scripts/download_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# A fixed, deliberately simple prompt. We use the RAW string (no chat template)
# so the C++ tokenizer only has to reproduce plain SentencePiece/BPE encoding
# for now. The Llama tokenizer prepends BOS (id 1) by default; the C++ side
# must match that. The chat template can come later once the engine is correct.
PROMPT = "The capital of France is"

# Repo root is the parent of the scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights" / "tinyllama-1.1b-chat"
REF_DIR = ROOT / "reference"

# Architecture values we expect for TinyLlama-1.1B-Chat-v1.0. The script asserts
# the downloaded config matches, so a surprise upstream change is caught here
# instead of silently corrupting the C++ port.
EXPECTED = {
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,  # GQA: 8 query heads share each KV head
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


def main() -> None:
    # Imports are inside main so the module-level docstring/constants are
    # readable without the heavy deps installed.
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)

    print(f"[1/5] Downloading {MODEL_ID} -> {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(WEIGHTS_DIR),
        # Grab only what we need: safetensors weights, all json config, and the
        # SentencePiece model. Skip the duplicate pytorch_model.bin to save ~2GB.
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )

    print("[2/5] Loading tokenizer and model (FP32, CPU)")
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # Pin and verify the architecture.
    cfg = model.config
    print("[3/5] Architecture config:")
    mismatches = []
    for key, want in EXPECTED.items():
        got = getattr(cfg, key, None)
        ok = got == want
        if not ok:
            mismatches.append((key, want, got))
        flag = "ok " if ok else "!! "
        print(f"        {flag}{key:26s} = {got}  (expected {want})")
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"        .. head_dim (derived)     = {head_dim}")
    if mismatches:
        raise SystemExit(
            "Config mismatch vs EXPECTED; resolve before porting to C++:\n"
            + "\n".join(f"  {k}: expected {w}, got {g}" for k, w, g in mismatches)
        )

    REF_DIR.mkdir(parents=True, exist_ok=True)
    # Save the pinned config the C++ engine will hardcode/parse.
    pinned = {k: getattr(cfg, k) for k in EXPECTED}
    pinned["head_dim"] = head_dim
    pinned["torch_dtype"] = "float32"
    pinned["model_id"] = MODEL_ID
    (REF_DIR / "config.json").write_text(json.dumps(pinned, indent=2))

    print(f"[4/5] Tokenizing prompt: {PROMPT!r}")
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    ids = input_ids[0].to(torch.int32).numpy()
    print(f"        token ids: {ids.tolist()}")
    print(f"        decoded:   {[tokenizer.decode([i]) for i in ids.tolist()]}")
    (REF_DIR / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    np.save(REF_DIR / "token_ids.npy", ids)

    print("[5/5] Running reference forward pass")
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0].to(torch.float32).numpy()  # [seq, vocab]
    np.save(REF_DIR / "logits.npy", logits)
    print(f"        logits shape: {logits.shape}  dtype: {logits.dtype}")

    # Human-readable sanity check: what does the model predict next?
    last = logits[-1]
    topk = last.argsort()[-5:][::-1]
    print("        top-5 next-token predictions:")
    for rank, tok in enumerate(topk, 1):
        print(f"          {rank}. id={int(tok):5d}  {tokenizer.decode([int(tok)])!r}  logit={last[tok]:.4f}")

    print("\nDone. Golden reference written to:", REF_DIR)
    print("These artifacts are the oracle for engine/tests/.")


if __name__ == "__main__":
    main()
