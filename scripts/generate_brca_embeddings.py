#!/usr/bin/env python
"""Generate DNA language-model embeddings for the BRCA VEP eval set (TabBench-Bio "local").

For each requested model (``evo2`` / ``ntv2`` / ``dnabert2``) this:

1. Reads ``ref_sequence``, ``var_sequence`` and ``func.class`` from the
   TinyHumanTransformer eval CSV (every other column is ignored).
2. Computes the **mean-pooled** embedding of the reference and of the variant
   sequence, then **concatenates** them: ``[mean(ref) ‖ mean(var)]``. With evo2-7b
   (``blocks.26.mlp.l3``, 4096-d residual stream) that is a 8192-long vector;
   ntv2-500m → 2048; dnabert2 → 1536.
3. Writes one TabBench-Bio "local" dataset file per model with three columns:
   ``embedding`` (the vector as a comma-separated float string), ``group_id``
   (a stable reference-sequence hash used for grouped splitting), and ``y``
   (``func.class``: FUNC / LOF / INT — multiclass).

The embedding recipes (checkpoints, evo2 layer, ``add_special_tokens=False``
tokenization, masked mean-pooling, DNABERT-2 Triton patch) mirror the proven
``DNA_Foundation_Models_Inversion_Attack_Benchmark/generate/*`` scripts that produced
the GABBA embeddings, so outputs are directly comparable.

GPU only (evo2 needs transformer-engine/CUDA). Run via ``generate_brca_embeddings.sbatch``.
Quick smoke check: ``--models dnabert2 --limit 8 --out-dir /tmp/brca_smoke``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LOG = logging.getLogger("brca_embed")

# --- model identifiers (match the inversion-attack generate configs) -----------------
EVO2_CHECKPOINT = "evo2_7b"
EVO2_LAYER = "blocks.26.mlp.l3"
NTV2_CHECKPOINT = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DNABERT2_CHECKPOINT = "zhihan1996/DNABERT-2-117M"

# --- source-CSV columns we use (everything else is ignored) --------------------------
REF_COL = "ref_sequence"
VAR_COL = "var_sequence"
TARGET_COL = "func.class"

#: Output filename per model (relative to --out-dir).
OUT_FILENAME = {
    "evo2": "brca_evo2_vep.csv",
    "ntv2": "brca_ntv2_vep.csv",
    "dnabert2": "brca_dnabert2_vep.csv",
}


def _center_crop(seq: str, max_len: int | None) -> str:
    """Center-crop ``seq`` to ``max_len`` chars (no-op if shorter / ``max_len`` is None).

    BRCA variants are centered in the window, so a center-crop keeps the variant.
    """
    if max_len is None or len(seq) <= max_len:
        return seq
    mid = len(seq) // 2
    half = max_len // 2
    start = max(0, mid - half)
    return seq[start : start + max_len]


# ======================================================================================
# evo2
# ======================================================================================
def embed_evo2(seqs: list[str], device: str, batch_size: int, max_len: int | None) -> np.ndarray:
    """Mean-pooled evo2 embeddings (one row per sequence), batched.

    evo2's tokenizer is character-level, so equal-length inputs tokenize to equal-length
    token tensors and need no padding (matches ``embed_batch_evo2`` in the inversion repo).
    """
    from evo2 import Evo2

    LOG.info("Loading evo2 model %s (layer %s) ...", EVO2_CHECKPOINT, EVO2_LAYER)
    model = Evo2(EVO2_CHECKPOINT)
    cropped = [_center_crop(s, max_len) for s in seqs]

    out: list[np.ndarray] = []
    n = len(cropped)
    for start in range(0, n, batch_size):
        batch = cropped[start : start + batch_size]
        token_lists = [model.tokenizer.tokenize(s) for s in batch]
        length = len(token_lists[0])
        assert all(len(t) == length for t in token_lists), (
            "evo2 batch requires equal-length tokenizations; "
            f"got lengths={sorted({len(t) for t in token_lists})}"
        )
        input_ids = torch.tensor(token_lists, dtype=torch.long).to(device)
        with torch.no_grad():
            _, emb = model(input_ids, return_embeddings=True, layer_names=[EVO2_LAYER])
        assert EVO2_LAYER in emb, f"Layer {EVO2_LAYER!r} not in evo2 output"
        arr = emb[EVO2_LAYER].detach().float().cpu().numpy()  # [B, L, D]
        out.extend(arr[i].mean(axis=0) for i in range(arr.shape[0]))
        done = min(start + batch_size, n)
        if done % (batch_size * 25) == 0 or done == n:
            LOG.info("  evo2: %d/%d", done, n)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(out)


# ======================================================================================
# Nucleotide Transformer v2
# ======================================================================================
def embed_ntv2(seqs: list[str], device: str, batch_size: int) -> np.ndarray:
    """Mean-pooled NTv2 embeddings (last hidden state, masked mean over real tokens)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    LOG.info("Loading NTv2 model %s ...", NTV2_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(NTV2_CHECKPOINT, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(NTV2_CHECKPOINT, trust_remote_code=True)
    model.to(device).eval()

    out: list[np.ndarray] = []
    n = len(seqs)
    for start in range(0, n, batch_size):
        batch = seqs[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last = outputs.hidden_states[-1]  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).to(last.dtype)  # [B, L, 1]
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = (summed / counts).detach().float().cpu().numpy()  # [B, D]
        out.extend(pooled[i] for i in range(pooled.shape[0]))
        done = min(start + batch_size, n)
        if done % (batch_size * 25) == 0 or done == n:
            LOG.info("  ntv2: %d/%d", done, n)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(out)


# ======================================================================================
# DNABERT-2
# ======================================================================================
def _patch_triton_flash_attn() -> None:
    """Patch DNABERT-2's cached ``flash_attn_triton.py`` for Triton >= 3.x compatibility.

    The bundled file uses deprecated ``tl.dot(..., trans_a/trans_b=True)`` kwargs removed
    in newer Triton; replace them with the equivalent ``tl.dot(tl.trans(x), y)`` calls.
    Copied verbatim from the inversion-attack generate script.
    """
    for name, mod in list(sys.modules.items()):
        if "flash_attn_triton" not in name or not hasattr(mod, "__file__") or mod.__file__ is None:
            continue
        fpath = Path(mod.__file__)
        if not fpath.exists():
            continue
        src = fpath.read_text()
        if "trans_b=True" not in src and "trans_a=True" not in src:
            continue
        patched = src
        patched = patched.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")
        patched = patched.replace("tl.dot(do, v, trans_b=True)", "tl.dot(do, tl.trans(v))")
        patched = patched.replace(
            "tl.dot(p.to(do.dtype), do, trans_a=True)",
            "tl.dot(tl.trans(p.to(do.dtype)), do)",
        )
        patched = patched.replace("tl.dot(ds, q, trans_a=True)", "tl.dot(tl.trans(ds), q)")
        fpath.write_text(patched)
        LOG.info("Patched %s for Triton 3.x compatibility", fpath)


def embed_dnabert2(seqs: list[str], device: str) -> np.ndarray:
    """Mean-pooled DNABERT-2 embeddings (per-sequence; BPE token counts vary by sequence)."""
    import importlib

    from transformers import AutoModel, AutoTokenizer, BertConfig

    LOG.info("Loading DNABERT-2 model %s ...", DNABERT2_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_CHECKPOINT, trust_remote_code=True)
    config = BertConfig.from_pretrained(DNABERT2_CHECKPOINT)
    model = AutoModel.from_pretrained(DNABERT2_CHECKPOINT, config=config, trust_remote_code=True)

    # Patch & reload the Triton flash-attn module so the first forward compiles fixed kernels.
    _patch_triton_flash_attn()
    for mod_name in list(sys.modules):
        if "flash_attn_triton" in mod_name:
            importlib.reload(sys.modules[mod_name])

    model.to(device).eval()

    out: list[np.ndarray] = []
    n = len(seqs)
    for i, seq in enumerate(seqs):
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            hidden = model(inputs)[0]  # [1, num_tokens, D]
        out.append(hidden[0].detach().float().cpu().numpy().mean(axis=0))
        if (i + 1) % 200 == 0 or (i + 1) == n:
            LOG.info("  dnabert2: %d/%d", i + 1, n)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(out)


# ======================================================================================
# driver
# ======================================================================================
_EMBEDDERS = {
    "evo2": lambda seqs, args: embed_evo2(seqs, args.device, args.evo2_batch_size, args.evo2_max_len),
    "ntv2": lambda seqs, args: embed_ntv2(seqs, args.device, args.ntv2_batch_size),
    "dnabert2": lambda seqs, args: embed_dnabert2(seqs, args.device),
}


def _write_dataset(
    ref_emb: np.ndarray,
    var_emb: np.ndarray,
    y: pd.Series,
    group_ids: list[str],
    out_path: Path,
) -> None:
    """Write concatenated embeddings, biological group IDs, and targets."""
    assert ref_emb.shape == var_emb.shape, f"ref/var shape mismatch: {ref_emb.shape} vs {var_emb.shape}"
    concat = np.concatenate([ref_emb, var_emb], axis=1)  # [N, 2D]
    LOG.info("  concatenated embedding: %s (ref %d + var %d)", concat.shape, ref_emb.shape[1], var_emb.shape[1])
    # Format each row as a comma-separated float string (TabBench-Bio local embedding_column format).
    strings = [",".join(f"{v:.6f}" for v in row) for row in concat]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"embedding": strings, "group_id": group_ids, "y": y.to_numpy()}
    ).to_csv(out_path, index=False)
    LOG.info("  wrote %d rows x %d-d embedding -> %s", concat.shape[0], concat.shape[1], out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-csv",
        default="/weka/pfeifer/ppu738/TinyHumanTransformer/data/ctx8192/brca_eval.csv",
        help="TinyHumanTransformer BRCA eval CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="/weka/pfeifer/ppu738/TabBench-Bio/src/tabbench_bio/bio/data/local",
        help="Directory to write the per-model dataset CSVs into.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(OUT_FILENAME),
        default=["dnabert2", "ntv2", "evo2"],  # cheap -> expensive (fail-fast)
        help="Which models to embed with.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evo2-batch-size", type=int, default=4)
    parser.add_argument("--ntv2-batch-size", type=int, default=16)
    parser.add_argument(
        "--evo2-max-len", type=int, default=8192,
        help="Center-crop sequences to this many bp before evo2 (its context). 0/neg = no crop.",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, only embed the first N rows (smoke test).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.evo2_max_len <= 0:
        args.evo2_max_len = None

    if not torch.cuda.is_available():
        LOG.warning("CUDA not available — embedding on CPU will be very slow (and evo2 likely fails).")

    LOG.info("Reading %s", args.input_csv)
    df = pd.read_csv(args.input_csv, usecols=[REF_COL, VAR_COL, TARGET_COL])
    if args.limit > 0:
        df = df.head(args.limit)
    LOG.info("Rows: %d | target '%s' counts: %s", len(df), TARGET_COL, df[TARGET_COL].value_counts().to_dict())

    ref_seqs = df[REF_COL].astype(str).tolist()
    var_seqs = df[VAR_COL].astype(str).tolist()
    y = df[TARGET_COL].astype(str).str.strip()
    # Equal reference contexts identify the same local genomic locus. Hash rather than
    # persist the sequence itself so grouping metadata cannot accidentally become a feature.
    group_ids = [hashlib.sha256(seq.encode()).hexdigest()[:20] for seq in ref_seqs]

    for model_key in args.models:
        LOG.info("=" * 80)
        LOG.info("Model: %s", model_key)
        embed = _EMBEDDERS[model_key]
        LOG.info("Embedding %d reference sequences ...", len(ref_seqs))
        ref_emb = embed(ref_seqs, args)
        LOG.info("Embedding %d variant sequences ...", len(var_seqs))
        var_emb = embed(var_seqs, args)
        out_path = Path(args.out_dir) / OUT_FILENAME[model_key]
        _write_dataset(ref_emb, var_emb, y, group_ids, out_path)

    LOG.info("Done. Models: %s", ", ".join(args.models))


if __name__ == "__main__":
    main()
