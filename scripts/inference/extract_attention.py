#!/usr/bin/env python3
"""GPU attention-map extractor (Step 1 of the pipeline).

For every (model, dataset, question), runs one prefill forward pass with
`output_attentions=True`, identifies the question-token (Q) query positions
and vision-token (V) key positions from the input-id sequence, slices the
attention tensor along those positions, and writes a per-question .npz
with the keys consumed by `compute_correlations_npz.py`:

  attn_question  (L, H, |V|)  float16  = mean over q in Q of A[l,h,q,v]
  token_ids      (T,)         int32
  token_types    (T,)         uint8    = enum tag per token

Per-VLM cost on a single H100 80 GB: ~5-10 min for <=4B models and
~15-30 min for 7-8B models on the full SalChartQA corpus (n=5,999).

This driver is the minimal re-implementation of the original GPU
extractor used in the paper (the original was deleted from the working
tree once the attention-map cache stabilised). It produces the same
`.npz` schema that `compute_correlations_npz.py` consumes.

Usage:
    python scripts/inference/extract_attention.py \\
        --model 8B \\
        --dataset salchartqa \\
        --out attention_maps/

Recognised --model keys are listed in scripts/lib/model_constants.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Repo-relative imports (run from repo root or add scripts/ to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.extract_attention_helpers import (
    load_model as load_vlm_model,
    _prepare_inputs_qwen,
    _prepare_inputs_internvl2,
)
from lib.model_constants import get_model_config

TOKEN_TYPE = {
    "other": 0, "vision": 1, "question": 2, "suffix": 3, "answer_prefix": 4,
}

DATASET_PATHS = {
    "salchartqa": {
        "image_dir":  Path("data/salchartqa/raw_img"),
        "questions":  Path("data/salchartqa/image_questions.json"),
    },
    "taskvis": {
        "image_dir":  Path("data/taskvis/massvis"),
        "questions":  Path("data/taskvis/image_questions.json"),
    },
}

PROMPT_SUFFIX = "Answer concisely and immediately."
GEN_PREFIX    = "Answer:"


def _iter_dataset(dataset: str):
    """Yield (sample_idx, image_name, image_path, qid, question_text)
    in the same iteration order used by the cached NPZs (sorted q-id
    within dict-iteration order over images)."""
    cfg = DATASET_PATHS[dataset]
    img_qs = json.loads(cfg["questions"].read_text())
    sample_idx = 0
    for image_name, qs in img_qs.items():
        image_path = cfg["image_dir"] / image_name
        for qid in sorted(qs.keys()):
            q = qs[qid]
            text = q["question"] if isinstance(q, dict) and "question" in q else str(q)
            yield sample_idx, image_name, image_path, qid, text
            sample_idx += 1


def _build_messages_qwen(image_path: Path, question: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": f"{question} {PROMPT_SUFFIX}"},
        ],
    }]


def _build_messages_internvl(image_path: Path, question: str) -> list[dict]:
    return [{"role": "user",
             "text": f"{question} {PROMPT_SUFFIX}",
             "image_path": str(image_path)}]


def _identify_token_roles(input_ids: torch.Tensor, processor: Any,
                          family: str) -> dict:
    """Return slices for vision (V), question (Q), suffix, and answer-prefix
    tokens, plus a per-token type-tag array."""
    ids = input_ids[0].cpu().tolist()
    T = len(ids)
    tags = np.full(T, TOKEN_TYPE["other"], dtype=np.uint8)
    v_idx, q_idx, suf_idx, ans_idx = [], [], [], []

    if family == "qwen":
        # Qwen vision tokens are between <|vision_start|> and <|vision_end|>.
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        vstart = tok.convert_tokens_to_ids("<|vision_start|>")
        vend   = tok.convert_tokens_to_ids("<|vision_end|>")
        ipad   = tok.convert_tokens_to_ids("<|image_pad|>")
        in_v = False
        for i, t in enumerate(ids):
            if t == vstart: in_v = True;  continue
            if t == vend:   in_v = False; continue
            if in_v or t == ipad:
                v_idx.append(i); tags[i] = TOKEN_TYPE["vision"]
        # Question = the user-content text run between vision tokens and the
        # assistant marker. Detection here uses the suffix string as anchor.
        text = tok.decode(ids, skip_special_tokens=False)
    else:
        # InternVL: vision = runs of <IMG_CONTEXT> id, question = text in
        # the user role before <|im_end|>.
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        try:
            ictx = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        except Exception:
            ictx = -1
        for i, t in enumerate(ids):
            if t == ictx:
                v_idx.append(i); tags[i] = TOKEN_TYPE["vision"]

    # Question / suffix / answer-prefix detection by sub-string matching on
    # the decoded string. We mark Q as everything between the end of the
    # vision run and the start of the suffix; suffix and answer-prefix are
    # tagged for completeness.
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    decoded = tok.decode(ids, skip_special_tokens=False)
    suffix_pos = decoded.find(PROMPT_SUFFIX)
    ans_pos    = decoded.find(GEN_PREFIX, suffix_pos if suffix_pos >= 0 else 0)
    # Map character positions back to token indices via a re-encoding pass:
    # token i ends at the character offset processor offsets give us.
    if hasattr(tok, "decode"):
        # Cheap approximation: tag by re-encoding the suffix and prefix and
        # finding their token-id sub-sequences in `ids`.
        def _find_subseq(haystack, needle):
            n = len(needle)
            for i in range(len(haystack) - n + 1):
                if haystack[i:i + n] == needle:
                    return slice(i, i + n)
            return None
        suf_ids = tok.encode(" " + PROMPT_SUFFIX, add_special_tokens=False)
        ans_ids = tok.encode(" " + GEN_PREFIX,    add_special_tokens=False)
        suf_sl = _find_subseq(ids, suf_ids)
        ans_sl = _find_subseq(ids, ans_ids)
        if suf_sl is not None:
            tags[suf_sl] = TOKEN_TYPE["suffix"]
        if ans_sl is not None:
            tags[ans_sl] = TOKEN_TYPE["answer_prefix"]
        # Question = everything between the last vision-token index and the
        # start of the suffix (exclusive).
        if v_idx and suf_sl is not None:
            q_start = max(v_idx) + 1
            q_stop  = suf_sl.start
            if q_start < q_stop:
                for i in range(q_start, q_stop):
                    if tags[i] == TOKEN_TYPE["other"]:
                        q_idx.append(i)
                        tags[i] = TOKEN_TYPE["question"]

    return {
        "v_idx": np.asarray(v_idx, dtype=np.int64),
        "q_idx": np.asarray(q_idx, dtype=np.int64),
        "tags":  tags,
    }


@torch.no_grad()
def extract_one(model: Any, processor: Any, family: str,
                image_path: Path, question: str, device: str = "cuda") -> dict:
    """Single forward pass; returns the dict to save."""
    if family == "qwen":
        messages = _build_messages_qwen(image_path, question)
        inputs = _prepare_inputs_qwen(processor, messages, str(image_path),
                                      GEN_PREFIX, device)
    else:
        messages = _build_messages_internvl(image_path, question)
        inputs = _prepare_inputs_internvl2(processor, model, messages,
                                           str(image_path), GEN_PREFIX, device)

    out = model(**inputs, output_attentions=True, return_dict=True,
                use_cache=False)

    attn = torch.stack([a[0] for a in out.attentions], dim=0)   # (L, H, T, T)
    L, H, T, _ = attn.shape

    roles = _identify_token_roles(inputs["input_ids"], processor, family)
    v_idx, q_idx, tags = roles["v_idx"], roles["q_idx"], roles["tags"]
    if v_idx.size == 0 or q_idx.size == 0:
        raise RuntimeError(
            f"Could not identify vision ({v_idx.size}) or question "
            f"({q_idx.size}) token spans; check tokenizer special tokens."
        )

    qv = attn[:, :, q_idx, :][:, :, :, v_idx]                   # (L, H, |Q|, |V|)
    aq = qv.mean(dim=2).to(torch.float16).cpu().numpy()         # (L, H, |V|)

    return {
        "attn_question":           aq,
        "token_ids":               inputs["input_ids"][0].cpu().numpy().astype(np.int32),
        "token_types":             tags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model key (see scripts/lib/model_constants.py).")
    ap.add_argument("--dataset", choices=list(DATASET_PATHS.keys()),
                    default="salchartqa")
    ap.add_argument("--out", type=Path, default=Path("attention_maps"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = get_model_config(args.model)
    family = cfg.get("model_family", "qwen")
    out_dir = args.out / args.dataset / "image_suffix_question" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {cfg['model_id']} (family={family})")
    model, processor, _ = load_vlm_model(args.model, device="cuda")
    model.eval()

    n_done = 0
    for sample_idx, image_name, image_path, qid, question in _iter_dataset(args.dataset):
        if sample_idx < args.start:
            continue
        if args.limit is not None and n_done >= args.limit:
            break
        out_path = out_dir / f"{args.dataset}_{sample_idx}.npz"
        if out_path.exists():
            n_done += 1
            continue
        try:
            blob = extract_one(model, processor, family, image_path, question)
        except Exception as e:
            print(f"  sample_idx={sample_idx} {image_name} {qid}: FAILED: {e}")
            continue
        np.savez_compressed(out_path, **blob)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} done (last sample_idx={sample_idx})", flush=True)

    print(f"Saved {n_done} attention maps to {out_dir}")


if __name__ == "__main__":
    main()
