"""
convert_weights.py - Phase 1, step 3: safetensors (BF16) -> flat FP32 blob.

Same division of labor as the tokenizer export: Python parses the complex format
ONCE, the C++ engine reads a flat, dependency-free blob. The engine never parses
safetensors JSON or decodes BF16.

Reads:
  weights/tinyllama-1.1b-chat/model.safetensors   201 tensors, all BF16
  reference/config.json                            pinned architecture (validation oracle)

Writes (both under weights/tinyllama-1.1b-chat/, gitignored):
  model.f32.bin        every tensor upcast to FP32, little-endian, concatenated in a
                       fixed order. ~4.4 GB. Regenerated locally; never committed.
  model.manifest.txt   one tensor per line: "name byte_offset ndim d0 d1 ...".
                       Line-oriented and diffable; the engine reads this to know where
                       each tensor lives in the blob.

Design notes:
  - numpy-only, no torch/safetensors dependency. We parse the safetensors header by
    hand and upcast BF16 -> FP32 with a bit shift: BF16 is the top 16 bits of an FP32,
    so (u16 << 16) reinterpreted as float32 is the exact value (lossless for normals,
    subnormals, inf and nan alike). This matches what HF did building logits.npy with
    dtype=torch.float32.
  - Weights keep their PyTorch [out_features, in_features] layout (no transpose at
    convert time); the C++ forward pass computes y = x @ W^T directly via its `linear`
    helper. Trivial converter, matches the reference's own computation.
  - The blob is written in a fixed, self-consistent order (embed, then layers 0..21,
    then final norm, then lm_head); only the manifest offsets matter to the engine, but
    a stable order keeps diffs and debugging sane.

Run from the repo root:
  python scripts/convert_weights.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights" / "tinyllama-1.1b-chat"
SAFETENSORS = WEIGHTS_DIR / "model.safetensors"
REF_CONFIG = ROOT / "reference" / "config.json"

BIN_OUT = WEIGHTS_DIR / "model.f32.bin"
MANIFEST_OUT = WEIGHTS_DIR / "model.manifest.txt"

# Stream the blob in chunks so we never hold the whole 4.4 GB in memory at once.
WRITE_CHUNK_BYTES = 64 * 1024 * 1024


def load_safetensors_header(path: Path) -> dict:
    """Parse the safetensors header: 8-byte little-endian length, then that many bytes
    of JSON. Tensor `data_offsets` are byte ranges into the data block that follows."""
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def expected_tensors(cfg: dict) -> dict[str, list[int]]:
    """The 201 tensors TinyLlama must have, with their shapes, derived from the pinned
    config. Used to validate the download hasn't drifted before we spend minutes
    converting it. PyTorch nn.Linear stores weights as [out_features, in_features]."""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]  # 4 * 64 = 256 (GQA 8:1)

    want: dict[str, list[int]] = {"model.embed_tokens.weight": [vocab, h]}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        want[p + "input_layernorm.weight"] = [h]
        want[p + "self_attn.q_proj.weight"] = [h, h]
        want[p + "self_attn.k_proj.weight"] = [kv_dim, h]
        want[p + "self_attn.v_proj.weight"] = [kv_dim, h]
        want[p + "self_attn.o_proj.weight"] = [h, h]
        want[p + "post_attention_layernorm.weight"] = [h]
        want[p + "mlp.gate_proj.weight"] = [inter, h]
        want[p + "mlp.up_proj.weight"] = [inter, h]
        want[p + "mlp.down_proj.weight"] = [h, inter]
    want["model.norm.weight"] = [h]
    want["lm_head.weight"] = [vocab, h]
    return want


def write_order(cfg: dict) -> list[str]:
    """Fixed order the blob is written in: embed, then each layer's 9 tensors, then the
    final norm, then lm_head. Only the manifest offsets bind the engine, but a stable
    order keeps the blob reproducible and diffs meaningful."""
    order = ["model.embed_tokens.weight"]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        order += [
            p + "input_layernorm.weight",
            p + "self_attn.q_proj.weight",
            p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
        ]
    order += ["model.norm.weight", "lm_head.weight"]
    return order


def validate(header: dict, want: dict[str, list[int]]) -> None:
    """Fail loudly if the safetensors tensor set, shapes, or dtypes don't match what the
    C++ port was written against. Cheaper to catch here than to debug divergent logits."""
    problems: list[str] = []

    got_names = set(header)
    missing = sorted(want.keys() - got_names)
    extra = sorted(got_names - want.keys())
    for name in missing:
        problems.append(f"missing tensor: {name}")
    for name in extra:
        problems.append(f"unexpected tensor: {name}")

    for name, shape in want.items():
        entry = header.get(name)
        if entry is None:
            continue  # already reported as missing
        if entry["dtype"] != "BF16":
            problems.append(f"{name}: dtype {entry['dtype']}, expected BF16")
        if entry["shape"] != shape:
            problems.append(f"{name}: shape {entry['shape']}, expected {shape}")

    if problems:
        raise SystemExit(
            "Safetensors validation failed (resolve before converting):\n  "
            + "\n  ".join(problems)
        )

    print(f"[validate] ok: {len(want)} tensors, all BF16, shapes match reference/config.json")


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Upcast a BF16 byte buffer to FP32 losslessly: zero-extend each 16-bit value into
    the high half of a 32-bit word and reinterpret as float32."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def main() -> None:
    if not SAFETENSORS.exists():
        raise SystemExit(
            f"{SAFETENSORS} not found. Run scripts/download_weights.py first."
        )
    cfg = json.loads(REF_CONFIG.read_text())

    print(f"[1/3] Parsing safetensors header: {SAFETENSORS}")
    header = load_safetensors_header(SAFETENSORS)
    want = expected_tensors(cfg)
    validate(header, want)

    # Byte offset of the data block: 8-byte length prefix + the JSON header.
    with SAFETENSORS.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
    data_base = 8 + header_len

    order = write_order(cfg)
    assert set(order) == set(want), "write order must cover exactly the expected tensors"

    print(f"[2/3] Writing FP32 blob -> {BIN_OUT}")
    manifest_lines: list[str] = []
    out_offset = 0  # byte offset into model.f32.bin (FP32)

    with SAFETENSORS.open("rb") as src, BIN_OUT.open("wb") as dst:
        for name in order:
            entry = header[name]
            start, end = entry["data_offsets"]
            shape = entry["shape"]
            n_elem = int(np.prod(shape)) if shape else 1

            # Read this tensor's BF16 bytes straight from the data block.
            src.seek(data_base + start)
            raw = src.read(end - start)
            assert len(raw) == n_elem * 2, f"{name}: short read ({len(raw)} != {n_elem*2})"

            f32 = bf16_bytes_to_f32(raw)
            assert f32.size == n_elem, f"{name}: element count mismatch"

            # Stream out in chunks; ensure little-endian on disk regardless of host.
            f32.astype("<f4", copy=False).tofile(dst)

            ndim = len(shape)
            dims = " ".join(str(d) for d in shape)
            manifest_lines.append(f"{name} {out_offset} {ndim} {dims}".rstrip())
            out_offset += n_elem * 4

    MANIFEST_OUT.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    total_gb = out_offset / 1e9
    print(f"[3/3] Done: {len(order)} tensors, {out_offset} bytes ({total_gb:.2f} GB)")
    print(f"        blob:     {BIN_OUT}")
    print(f"        manifest: {MANIFEST_OUT}  ({len(manifest_lines)} lines)")


if __name__ == "__main__":
    sys.exit(main())
