#!/usr/bin/env python3
"""Mean-ablate the top human-correlated vision heads on SalChartQA.

This script does two things for each requested model:

1. Rank vision heads by their mean correlation to the SalChartQA `mean_all`
   human fixation map across the usable SalChartQA subset.
2. Evaluate baseline vs ablated SalChartQA correctness, where the top-k heads
   are ablated together by replacing each selected head's pre-output-projection
   activation with that head's dataset-wide mean activation vector over
   SalChartQA.

The ranking side is built from the same assets used in
`data_cache_parallel`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from tqdm import tqdm

ROOT = Path("./")
LVLM_ROOT = Path("./scripts/lib")
sys.path.insert(0, str(LVLM_ROOT))

from attention.constants import get_model_config
from attention.extract_attention import (
    load_model as load_vlm_model,
    _prepare_inputs_internvl2,
    _prepare_inputs_qwen,
)
from correctness_final.evaluate_ordering import load_salchartqa_ground_truth, relaxed_match, _normalize_quotes
from utils.data_loaders import SalChartQALoader
from utils.internvl2_utils import load_internvl2_image


ATTN_BASE = Path("./cache/lvlm-chart/attention_maps/salchartqa/image_suffix_question")
HUMAN_MAP_DIR = ROOT / "data_cache" / "salchartqa"
CORRECTNESS_DIR = Path("./data/lvlm-chart/correctness")
OUT_DIR = ROOT / "4_ablation" / "results"
RANKING_DIR = OUT_DIR / "head_rankings"
MEAN_ACTIVATION_DIR = OUT_DIR / "mean_activations"
MEAN_ATTN_PATTERN_DIR = OUT_DIR / "mean_attn_patterns"
EVAL_DIR = OUT_DIR / "correctness"

COMMON_GRID = (28, 28)
GRID_SIZE = COMMON_GRID[0] * COMMON_GRID[1]  # 784
ATTN_SIGMA = 1.0
SUFFIX = "Answer concisely and immediately."
GENERATION_PREFIX = "Answer:"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.9
DEFAULT_NUM_RESPONSES = 5

INSTRUCT_MODELS = [
    "2.5-3B",
    "2.5-7B",
    "2B",
    "4B",
    "8B",
    "internvl3-1b-instruct",
    "internvl3-2b-instruct",
    "internvl3-8b-instruct",
    "internvl3.5-1b-instruct",
    "internvl3.5-2b-instruct",
    "internvl3.5-4b-instruct",
    "internvl3.5-8b-instruct",
]

TILE_CONFIGS = {
    "internvl3-1b-instruct": (256, True),
    "internvl3-2b-instruct": (256, True),
    "internvl3-8b-instruct": (256, True),
    "internvl3.5-1b-instruct": (256, True),
    "internvl3.5-2b-instruct": (256, True),
    "internvl3.5-4b-instruct": (256, True),
    "internvl3.5-8b-instruct": (256, True),
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    raw_idx: int
    image_path: str
    image_name: str
    question: str
    question_id: str


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bilinear_downsample_map(full_map: np.ndarray, gh: int, gw: int) -> np.ndarray:
    img = Image.fromarray(np.asarray(full_map, dtype=np.float32), mode="F")
    resized = np.asarray(img.resize((gw, gh), resample=Image.Resampling.BILINEAR), dtype=np.float64)
    total = float(resized.sum())
    if total > 1e-10:
        resized /= total
    return resized


def normalize_rows(matrix: np.ndarray, corr_type: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    if corr_type == "spearman":
        ranked = np.empty_like(arr, dtype=np.float64)
        for i in range(arr.shape[0]):
            ranked[i] = rankdata(arr[i])
        arr = ranked
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    valid = std[:, 0] > 1e-10
    out = np.zeros_like(arr, dtype=np.float64)
    out[valid] = (arr[valid] - mean[valid]) / std[valid]
    return out, valid


def corr_matrix(a: np.ndarray, b: np.ndarray, corr_type: str) -> np.ndarray:
    a_norm, a_valid = normalize_rows(a, corr_type)
    b_norm, b_valid = normalize_rows(b, corr_type)
    n = a.shape[1]
    cc = (a_norm @ b_norm.T) / n
    cc[~a_valid, :] = np.nan
    cc[:, ~b_valid] = np.nan
    return cc


def infer_grid_shape(n_vision: int, img_h: int, img_w: int) -> tuple[int, int] | None:
    gh, gw = round(img_h / 28), round(img_w / 28)
    if gh * gw == n_vision:
        return gh, gw
    aspect = img_w / img_h
    best = None
    best_err = float("inf")
    for h in range(1, n_vision + 1):
        if n_vision % h != 0:
            continue
        w = n_vision // h
        err = abs(w / h - aspect)
        if err < best_err:
            best_err = err
            best = (h, w)
    return best


def load_one_attn(npz_path: Path, model_key: str, num_layers: int, num_heads: int, img_h: int, img_w: int) -> np.ndarray | None:
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    if "attn_question" not in data:
        return None

    attn = data["attn_question"].astype(np.float32)
    token_types = data["token_types"][: attn.shape[2]]
    vision_mask = token_types == 0
    n_vision = int(vision_mask.sum())
    attn = attn[:, :, vision_mask]

    if attn.shape[0] != num_layers or attn.shape[1] != num_heads:
        return None

    tile_cfg = TILE_CONFIGS.get(model_key)
    if tile_cfg is not None:
        tokens_per_tile, has_thumbnail = tile_cfg
        tile_side = int(round(tokens_per_tile ** 0.5))
        num_tiles = n_vision // tokens_per_tile
        if num_tiles * tokens_per_tile != n_vision or num_tiles == 0:
            return None
        display_tiles = num_tiles - 1 if (has_thumbnail and num_tiles > 1) else num_tiles
        usable_tokens = display_tiles * tokens_per_tile
        aspect = img_w / img_h
        best = None
        best_err = float("inf")
        for h in range(1, display_tiles + 1):
            if display_tiles % h != 0:
                continue
            w = display_tiles // h
            err = abs(w / h - aspect)
            if err < best_err:
                best_err = err
                best = (h * tile_side, w * tile_side)
        if best is None:
            return None
        gh, gw = best
        if usable_tokens != n_vision:
            attn = attn[:, :, :usable_tokens]
        tile_h, tile_w = gh // tile_side, gw // tile_side
        attn = attn.reshape(num_layers, num_heads, tile_h, tile_w, tile_side, tile_side)
        attn = attn.transpose(0, 1, 2, 4, 3, 5).reshape(num_layers, num_heads, gh, gw)
    else:
        grid_shape = infer_grid_shape(n_vision, img_h, img_w)
        if grid_shape is None or grid_shape[0] * grid_shape[1] != n_vision:
            return None
        gh, gw = grid_shape
        attn = attn.reshape(num_layers, num_heads, gh, gw)

    if (gh, gw) != COMMON_GRID:
        tensor = torch.from_numpy(attn).reshape(1, num_layers * num_heads, gh, gw)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=COMMON_GRID,
            mode="bilinear",
            align_corners=False,
        )
        attn = tensor.reshape(num_layers, num_heads, *COMMON_GRID).numpy()
    return attn


def natural_sort_key(text: str) -> list[Any]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


def parse_ts(name: str) -> str:
    match = re.search(r"_(\d{8}_\d{6})(?:_fixed)?\.json$", name)
    return match.group(1) if match else ""


def choose_correctness_file(model_key: str) -> Path:
    pattern = "correctness_*_salchartqa_x5_n5999_*.json"
    candidates: list[tuple[bool, str, Path]] = []
    for path in CORRECTNESS_DIR.glob(pattern):
        name = path.name.lower()
        key = model_key.lower()
        if not re.match(rf"^correctness_{re.escape(key)}_salchartqa_x5_n5999_", name):
            continue
        candidates.append((name.endswith("_fixed.json"), parse_ts(name), path))
    if not candidates:
        raise FileNotFoundError(f"No matching SalChartQA x5 correctness file found for {model_key}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[-1][2]


def build_sample_records(sample_limit: int | None = None) -> list[SampleRecord]:
    loader = SalChartQALoader()
    records: list[SampleRecord] = []
    for idx in range(len(loader)):
        sample = loader[idx]
        human_map_path = HUMAN_MAP_DIR / f"{sample.sample_id}.npz"
        if not human_map_path.exists():
            continue
        records.append(
            SampleRecord(
                sample_id=sample.sample_id,
                raw_idx=int(sample.metadata["raw_idx"]),
                image_path=sample.image_path,
                image_name=sample.metadata["image_name"],
                question=sample.question,
                question_id=sample.metadata["question_id"],
            )
        )
        if sample_limit is not None and len(records) >= sample_limit:
            break
    return records


def get_attn_npz_path(model_key: str, sample: SampleRecord) -> Path:
    return ATTN_BASE / model_key / f"salchartqa_{sample.raw_idx}.npz"


def compute_head_ranking(
    model_key: str,
    corr_type: str,
    sample_limit: int | None,
    force: bool,
) -> pd.DataFrame:
    RANKING_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = RANKING_DIR / f"{model_key}__salchartqa__{corr_type}__mean_all_heads.csv"
    if ranking_path.exists() and not force:
        return pd.read_csv(ranking_path)

    cfg = get_model_config(model_key)
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    sums = np.zeros((num_layers, num_heads), dtype=np.float64)
    counts = np.zeros((num_layers, num_heads), dtype=np.int64)

    samples = build_sample_records(sample_limit=sample_limit)
    for sample in tqdm(samples, desc=f"Ranking heads for {model_key}"):
        human_npz = HUMAN_MAP_DIR / f"{sample.sample_id}.npz"
        attn_npz = get_attn_npz_path(model_key, sample)
        if not attn_npz.exists():
            continue

        human_data = np.load(human_npz)
        human_map = np.asarray(human_data["mean_all"], dtype=np.float64)
        img_h = int(human_data["image_h"])
        img_w = int(human_data["image_w"])
        attn = load_one_attn(attn_npz, model_key, num_layers, num_heads, img_h, img_w)
        if attn is None:
            continue

        for layer in range(num_layers):
            for head in range(num_heads):
                attn[layer, head] = gaussian_filter(attn[layer, head], sigma=ATTN_SIGMA)
                total = float(attn[layer, head].sum())
                if total > 1e-10:
                    attn[layer, head] /= total

        gh, gw = attn.shape[-2:]
        target = bilinear_downsample_map(human_map, gh, gw).reshape(1, gh * gw)
        heads_flat = attn.reshape(num_layers * num_heads, gh * gw)
        cc = corr_matrix(heads_flat, target, corr_type)[:, 0].reshape(num_layers, num_heads)
        valid = np.isfinite(cc)
        sums[valid] += cc[valid]
        counts[valid] += 1

    rows = []
    global_head = 0
    for layer in range(num_layers):
        for head in range(num_heads):
            count = int(counts[layer, head])
            rows.append(
                {
                    "model_key": model_key,
                    "corr_type": corr_type,
                    "layer": layer,
                    "head": head,
                    "global_head": global_head,
                    "n_samples": count,
                    "mean_corr": float(sums[layer, head] / count) if count > 0 else np.nan,
                }
            )
            global_head += 1

    df = pd.DataFrame(rows).sort_values(["mean_corr", "layer", "head"], ascending=[False, True, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(ranking_path, index=False)
    return df


def select_top_heads(df: pd.DataFrame, top_k: int) -> list[HeadRef]:
    top = df.dropna(subset=["mean_corr"]).head(top_k)
    return [HeadRef(layer=int(r.layer), head=int(r.head)) for r in top.itertuples(index=False)]


def select_bottom_heads(df: pd.DataFrame, bottom_k: int) -> list[HeadRef]:
    bottom = (
        df.dropna(subset=["mean_corr"])
        .sort_values(["mean_corr", "layer", "head"], ascending=[True, True, True])
        .head(bottom_k)
    )
    return [HeadRef(layer=int(r.layer), head=int(r.head)) for r in bottom.itertuples(index=False)]


def select_random_heads(
    df: pd.DataFrame,
    k: int,
    rng: random.Random,
    excluded: set[tuple[int, int]] | None = None,
) -> list[HeadRef]:
    excluded = excluded or set()
    pool = [
        (int(r.layer), int(r.head))
        for r in df.dropna(subset=["mean_corr"]).itertuples(index=False)
        if (int(r.layer), int(r.head)) not in excluded
    ]
    if len(pool) < k:
        raise RuntimeError(f"Not enough available heads to sample {k} random heads; pool={len(pool)}")
    chosen = rng.sample(pool, k)
    return [HeadRef(layer=layer, head=head) for layer, head in chosen]


def build_head_sets(
    model_key: str,
    ranking_df: pd.DataFrame,
    top_k: int,
    seed: int,
) -> dict[str, list[HeadRef]]:
    top = select_top_heads(ranking_df, top_k)
    bottom = select_bottom_heads(ranking_df, top_k)
    excluded = {(h.layer, h.head) for h in top} | {(h.layer, h.head) for h in bottom}

    rng_a = random.Random(f"{model_key}|random5_a|{seed}")
    random_a = select_random_heads(ranking_df, top_k, rng=rng_a, excluded=excluded)
    excluded_b = excluded | {(h.layer, h.head) for h in random_a}
    rng_b = random.Random(f"{model_key}|random5_b|{seed}")
    random_b = select_random_heads(ranking_df, top_k, rng=rng_b, excluded=excluded_b)

    return {
        "top5": top,
        "bottom5": bottom,
        "random5_a": random_a,
        "random5_b": random_b,
    }


def discover_lm_attention_modules(model: Any, expected_layers: int) -> list[tuple[int, str, Any, Any, int]]:
    named_modules = dict(model.named_modules())
    proj_names = ("o_proj", "out_proj", "proj")

    candidates: list[tuple[str, Any, Any, int]] = []

    for name, module in named_modules.items():
        lname = name.lower()
        if "language_model" not in lname:
            continue
        if "vision" in lname or "visual" in lname or "vit" in lname:
            continue
        if "self_attn" not in lname and "attention" not in lname:
            continue
        num_heads = getattr(module, "num_heads", None)
        if not isinstance(num_heads, int) or num_heads <= 1:
            # Fallback: infer num_heads from o_proj weight shape and head_dim.
            # o_proj.weight has shape [hidden_size, num_heads * head_dim], so
            # the INPUT dimension (shape[1]) gives num_heads * head_dim.
            head_dim = getattr(module, "head_dim", None)
            proj_candidate = getattr(module, "o_proj", None) or getattr(module, "out_proj", None)
            if isinstance(head_dim, int) and head_dim > 0 and proj_candidate is not None:
                weight = getattr(proj_candidate, "weight", None)
                if weight is not None:
                    num_heads = weight.shape[1] // head_dim
            if not isinstance(num_heads, int) or num_heads <= 1:
                continue
        proj = None
        for proj_name in proj_names:
            proj = getattr(module, proj_name, None)
            if proj is not None:
                break
        if proj is None:
            continue
        candidates.append((name, module, proj, num_heads))

    candidates.sort(key=lambda item: natural_sort_key(item[0]))
    if len(candidates) != expected_layers:
        names = [c[0] for c in candidates]
        raise RuntimeError(
            f"Could not identify exactly {expected_layers} LM attention layers; found {len(candidates)}.\n"
            f"Candidates: {names}"
        )
    return [(i, name, module, proj, num_heads) for i, (name, module, proj, num_heads) in enumerate(candidates)]


class MeanAblation:
    def __init__(
        self,
        model: Any,
        selected_heads: list[HeadRef],
        expected_layers: int,
        head_means: dict[int, np.ndarray],
        ablation_scope: str,
    ):
        per_layer: dict[int, set[int]] = {}
        for ref in selected_heads:
            per_layer.setdefault(ref.layer, set()).add(ref.head)
        self.per_layer = per_layer
        self.head_means = head_means
        self.ablation_scope = ablation_scope
        self.handles: list[Any] = []
        self.modules = discover_lm_attention_modules(model, expected_layers)
        self.current_vision_mask: torch.Tensor | None = None

    def set_vision_mask(self, mask: torch.Tensor | None) -> None:
        self.current_vision_mask = mask

    def __enter__(self) -> "MeanAblation":
        for layer_idx, name, _attn_module, proj_module, num_heads in self.modules:
            heads = self.per_layer.get(layer_idx)
            if not heads:
                continue
            layer_means = self.head_means.get(layer_idx)
            if layer_means is None:
                raise RuntimeError(f"Missing cached mean activation for layer {layer_idx} ({name})")

            def pre_hook(
                module: Any,
                args: tuple[Any, ...],
                *,
                heads: set[int] = heads,
                num_heads: int = num_heads,
                layer_means: np.ndarray = layer_means,
            ):
                if not args:
                    return None
                x = args[0]
                if not torch.is_tensor(x):
                    return None
                hidden = x.shape[-1]
                if hidden % num_heads != 0:
                    return None
                head_dim = hidden // num_heads
                x_heads = x.view(*x.shape[:-1], num_heads, head_dim)
                for head in heads:
                    mean_vec = torch.as_tensor(layer_means[head], dtype=x.dtype, device=x.device)
                    if self.ablation_scope == "all_tokens":
                        x_heads[..., head, :] = mean_vec
                    else:
                        vision_mask = self.current_vision_mask
                        if vision_mask is None:
                            return None
                        # Move mask to same device as tensor (multi-GPU)
                        vision_mask = vision_mask.to(x.device)
                        if x_heads.shape[0] != 1 or x_heads.shape[1] != vision_mask.shape[0]:
                            return None
                        x_heads[0, vision_mask, head, :] = mean_vec
                return (x_heads.reshape(*x.shape),) + args[1:]

            self.handles.append(proj_module.register_forward_pre_hook(pre_hook, with_kwargs=False))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class MeanActivationAccumulator:
    def __init__(self, model: Any, expected_layers: int, ablation_scope: str):
        self.modules = discover_lm_attention_modules(model, expected_layers)
        self.handles: list[Any] = []
        self.sums: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.ablation_scope = ablation_scope
        self.current_vision_mask: torch.Tensor | None = None

    def set_vision_mask(self, mask: torch.Tensor | None) -> None:
        self.current_vision_mask = mask

    def __enter__(self) -> "MeanActivationAccumulator":
        for layer_idx, _name, _attn_module, proj_module, num_heads in self.modules:
            def pre_hook(module: Any, args: tuple[Any, ...], *, layer_idx: int = layer_idx, num_heads: int = num_heads):
                if not args:
                    return None
                x = args[0]
                if not torch.is_tensor(x):
                    return None
                hidden = x.shape[-1]
                if hidden % num_heads != 0:
                    return None
                head_dim = hidden // num_heads
                x_heads = x.view(*x.shape[:-1], num_heads, head_dim)
                if self.ablation_scope == "all_tokens":
                    selected = x_heads.reshape(-1, num_heads, head_dim)
                else:
                    vision_mask = self.current_vision_mask
                    if vision_mask is None:
                        return None
                    # Move mask to same device as tensor (multi-GPU)
                    vision_mask = vision_mask.to(x.device)
                    if x_heads.shape[0] != 1 or x_heads.shape[1] != vision_mask.shape[0]:
                        return None
                    selected = x_heads[0, vision_mask, :, :]
                if selected.numel() == 0:
                    return None
                layer_sum = selected.sum(dim=0).detach().to(torch.float64).cpu()
                layer_count = int(selected.shape[0])
                if layer_idx not in self.sums:
                    self.sums[layer_idx] = layer_sum
                    self.counts[layer_idx] = layer_count
                else:
                    self.sums[layer_idx] += layer_sum
                    self.counts[layer_idx] += layer_count
                return None

            self.handles.append(proj_module.register_forward_pre_hook(pre_hook, with_kwargs=False))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> dict[int, np.ndarray]:
        return {
            layer_idx: (layer_sum / self.counts[layer_idx]).numpy().astype(np.float32)
            for layer_idx, layer_sum in self.sums.items()
        }


def compute_mean_attention_patterns(
    model_key: str,
    corr_type: str,
    sample_limit: int | None,
    force: bool,
) -> dict[int, np.ndarray]:
    """Compute dataset-mean spatial attention pattern per head from saved attention maps.

    Returns dict mapping layer_idx -> array of shape [num_heads, GRID_SIZE].
    Each row is a normalized probability distribution over the 28x28 vision grid.
    """
    MEAN_ATTN_PATTERN_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MEAN_ATTN_PATTERN_DIR / f"{model_key}__salchartqa__{corr_type}__mean_attn_pattern.npz"
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        return {int(key.split("_")[1]): cached[key] for key in cached.files}

    cfg = get_model_config(model_key)
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]

    sums = np.zeros((num_layers, num_heads, GRID_SIZE), dtype=np.float64)
    counts = np.zeros((num_layers, num_heads), dtype=np.int64)

    samples = build_sample_records(sample_limit=sample_limit)
    for sample in tqdm(samples, desc=f"Computing mean attn patterns for {model_key}"):
        attn_npz = get_attn_npz_path(model_key, sample)
        if not attn_npz.exists():
            continue
        human_npz = HUMAN_MAP_DIR / f"{sample.sample_id}.npz"
        if not human_npz.exists():
            continue
        human_data = np.load(human_npz)
        img_h = int(human_data["image_h"])
        img_w = int(human_data["image_w"])
        attn = load_one_attn(attn_npz, model_key, num_layers, num_heads, img_h, img_w)
        if attn is None:
            continue
        # attn: [num_layers, num_heads, 28, 28] on common grid
        for layer in range(num_layers):
            for head in range(num_heads):
                a = gaussian_filter(attn[layer, head], sigma=ATTN_SIGMA)
                total = float(a.sum())
                if total > 1e-10:
                    a /= total
                sums[layer, head] += a.flatten()
                counts[layer, head] += 1

    result: dict[int, np.ndarray] = {}
    for layer in range(num_layers):
        patterns = np.zeros((num_heads, GRID_SIZE), dtype=np.float32)
        for head in range(num_heads):
            if counts[layer, head] > 0:
                pat = sums[layer, head] / counts[layer, head]
                pat_sum = pat.sum()
                if pat_sum > 1e-10:
                    pat /= pat_sum
                patterns[head] = pat.astype(np.float32)
        result[layer] = patterns

    np.savez(cache_path, **{f"layer_{k}": v for k, v in result.items()})
    print(f"Saved mean attention patterns to {cache_path}")
    return result


def _resize_pattern_to_vision(mean_pat_flat: torch.Tensor, n_vision: int,
                               img_h: int | None = None, img_w: int | None = None) -> torch.Tensor:
    """Resize a [784] mean pattern on 28x28 grid to [n_vision] matching actual vision tokens."""
    gh_src, gw_src = COMMON_GRID
    pat_2d = mean_pat_flat.reshape(1, 1, gh_src, gw_src)

    # Try to infer target grid shape
    target_shape = None
    if img_h is not None and img_w is not None:
        target_shape = infer_grid_shape(n_vision, img_h, img_w)
    if target_shape is None:
        # Fallback: find best (h, w) with h*w == n_vision and reasonable aspect
        import math
        best, best_err = None, float("inf")
        for h in range(1, n_vision + 1):
            if n_vision % h != 0:
                continue
            w = n_vision // h
            err = abs(w / h - 1.0)  # prefer squarish
            if err < best_err:
                best_err = err
                best = (h, w)
        target_shape = best

    if target_shape is not None and target_shape[0] * target_shape[1] == n_vision:
        gh_tgt, gw_tgt = target_shape
        resized = torch.nn.functional.interpolate(
            pat_2d, size=(gh_tgt, gw_tgt), mode="bilinear", align_corners=False,
        )
    else:
        # Last resort: 1D interpolation
        resized = torch.nn.functional.interpolate(
            pat_2d.reshape(1, 1, 1, GRID_SIZE), size=(1, n_vision),
            mode="bilinear", align_corners=False,
        )
    out = resized.reshape(n_vision)
    out_sum = out.sum()
    if out_sum > 1e-10:
        out = out / out_sum
    return out


class AttentionPatternAblation:
    """Replace attention-to-vision patterns with dataset-mean patterns for selected heads.

    Unlike MeanAblation (which replaces the full pre-o_proj activation), this only
    modifies WHERE each head attends over vision tokens, keeping value processing intact.
    Tests whether each head's per-question spatial selectivity matters.

    Implementation: patches each ablated layer's attention forward to force
    output_attentions=True, then uses the delta approach:
      delta_output = o_proj(reshape((modified_weights - original_weights) @ V))
    Only the vision-token columns of the attention matrix change for ablated heads.
    The total attention to vision tokens is preserved; only the spatial distribution changes.
    """

    def __init__(
        self,
        model: Any,
        selected_heads: list[HeadRef],
        expected_layers: int,
        mean_attn_patterns: dict[int, np.ndarray] | None = None,
        uniform: bool = False,
    ):
        per_layer: dict[int, set[int]] = {}
        for ref in selected_heads:
            per_layer.setdefault(ref.layer, set()).add(ref.head)
        self.per_layer = per_layer
        self.mean_attn_patterns = mean_attn_patterns
        self.uniform = uniform
        self.handles: list[Any] = []
        self.modules = discover_lm_attention_modules(model, expected_layers)
        self.current_vision_mask: torch.Tensor | None = None
        self.current_img_hw: tuple[int, int] | None = None
        self._original_forwards: dict[int, Any] = {}
        self.model = model

    def set_vision_mask(self, mask: torch.Tensor | None) -> None:
        self.current_vision_mask = mask

    def set_image_hw(self, hw: tuple[int, int] | None) -> None:
        self.current_img_hw = hw

    def __enter__(self) -> "AttentionPatternAblation":
        ablation_ref = self  # capture for closures

        for layer_idx, name, attn_module, proj_module, num_heads in self.modules:
            heads = self.per_layer.get(layer_idx)
            if not heads:
                continue
            if self.uniform:
                patterns = None  # not needed for uniform
            else:
                patterns = self.mean_attn_patterns.get(layer_idx)
                if patterns is None:
                    raise RuntimeError(f"Missing mean attention pattern for layer {layer_idx} ({name})")

            original_forward = attn_module.forward
            self._original_forwards[layer_idx] = (attn_module, original_forward)

            def _make_patched(
                orig_fwd,
                *,
                _layer_idx: int = layer_idx,
                _heads: set[int] = heads,
                _patterns: np.ndarray | None = patterns,
                _num_heads: int = num_heads,
                _o_proj=proj_module,
                _uniform: bool = self.uniform,
            ):
                def patched_forward(*args, **kwargs):
                    # Force attention weight output
                    kwargs["output_attentions"] = True
                    output = orig_fwd(*args, **kwargs)

                    if not isinstance(output, tuple) or len(output) < 3:
                        return output

                    attn_output = output[0]   # [batch, seq_q, hidden] (after o_proj)
                    attn_weights = output[1]   # [batch, num_heads, seq_q, seq_kv]
                    past_kv = output[2]

                    if attn_weights is None:
                        return output

                    vision_mask = ablation_ref.current_vision_mask
                    if vision_mask is None:
                        return output

                    # --- extract V from KV cache ---
                    value = None
                    try:
                        if hasattr(past_kv, "value_cache"):
                            value = past_kv.value_cache[_layer_idx]
                        elif isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
                            value = past_kv[1]
                    except (IndexError, KeyError, TypeError):
                        pass
                    if value is None:
                        return output

                    # value: [batch, num_kv_heads, seq_kv, head_dim]
                    num_kv_heads = value.shape[1]
                    head_dim = value.shape[-1]
                    seq_kv = value.shape[2]
                    bsz = value.shape[0]
                    seq_q = attn_weights.shape[2]

                    # Expand V for GQA
                    if num_kv_heads != _num_heads:
                        num_groups = _num_heads // num_kv_heads
                        value_exp = value.unsqueeze(2).expand(
                            -1, -1, num_groups, -1, -1
                        ).reshape(bsz, _num_heads, seq_kv, head_dim)
                    else:
                        value_exp = value

                    # Map vision_mask to KV length
                    vm = vision_mask.to(attn_weights.device)
                    if vm.shape[0] < seq_kv:
                        padded = torch.zeros(seq_kv, dtype=torch.bool, device=vm.device)
                        padded[: vm.shape[0]] = vm
                        vm = padded
                    elif vm.shape[0] > seq_kv:
                        vm = vm[:seq_kv]

                    vision_indices = vm.nonzero(as_tuple=False).squeeze(-1)
                    n_vision = vision_indices.shape[0]
                    if n_vision == 0:
                        return output

                    img_hw = ablation_ref.current_img_hw
                    img_h = img_hw[0] if img_hw else None
                    img_w = img_hw[1] if img_hw else None

                    # Accumulate per-head delta: delta_attn_out for ablated heads
                    # Shape: [batch, seq_q, num_heads * head_dim]
                    delta_concat = torch.zeros(
                        bsz, seq_q, _num_heads * head_dim,
                        dtype=attn_output.dtype, device=attn_output.device,
                    )

                    for head in _heads:
                        if _uniform:
                            # Uniform: 1/n_vision for each vision token
                            replacement = torch.full(
                                (n_vision,), 1.0 / n_vision,
                                dtype=attn_weights.dtype, device=attn_weights.device,
                            )
                        else:
                            mean_pat = torch.as_tensor(
                                _patterns[head], dtype=attn_weights.dtype, device=attn_weights.device,
                            )
                            replacement = _resize_pattern_to_vision(
                                mean_pat, n_vision, img_h, img_w,
                            )  # [n_vision], sums to 1

                        # Original attention to vision tokens
                        orig_vision = attn_weights[:, head, :, vision_indices]      # [B, Sq, Nv]
                        orig_vision_total = orig_vision.sum(dim=-1, keepdim=True)   # [B, Sq, 1]

                        # New vision attention: same total, replacement spatial distribution
                        new_vision = replacement.unsqueeze(0).unsqueeze(0) * orig_vision_total

                        # Delta attention at vision positions
                        delta_vision = new_vision - orig_vision                     # [B, Sq, Nv]

                        # V at vision positions for this head
                        v_head_vision = value_exp[:, head, vision_indices, :]       # [B, Nv, Hd]

                        # Delta head output = delta_vision @ V_vision
                        delta_head = torch.bmm(delta_vision, v_head_vision)         # [B, Sq, Hd]

                        # Place into correct head slot
                        h_start = head * head_dim
                        h_end = h_start + head_dim
                        delta_concat[:, :, h_start:h_end] = delta_head

                    # delta through o_proj (linear, so additive)
                    delta_output = _o_proj(delta_concat)
                    new_attn_output = attn_output + delta_output

                    return (new_attn_output,) + output[1:]

                return patched_forward

            attn_module.forward = _make_patched(original_forward)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for layer_idx, (attn_module, orig_forward) in self._original_forwards.items():
            attn_module.forward = orig_forward
        self._original_forwards.clear()


def _infer_device(model: Any, model_family: str) -> str:
    """Return the appropriate device string for preparing inputs.

    For multi-GPU InternVL models (device_map='auto'), returns 'auto' so that
    _prepare_inputs_internvl2 places tensors on the correct per-submodule
    devices.  For Qwen multi-GPU, returns the device of the model's embedding
    layer (e.g. 'cuda:0').  For single-GPU, returns 'cuda'.
    """
    if torch.cuda.device_count() > 1:
        if model_family == "internvl2":
            return "auto"
        # Qwen with device_map='auto': find the embedding device
        try:
            embed = model.get_input_embeddings()
            return str(next(embed.parameters()).device)
        except (StopIteration, AttributeError):
            pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_messages(question: str, image_path: str, model_family: str) -> list[dict[str, Any]]:
    if model_family == "internvl2":
        return [{"role": "internvl2_text", "text": f"<image>\n{SUFFIX} {question}"}]
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": SUFFIX + " "},
            {"type": "text", "text": question},
        ],
    }]


def generate_responses(
    model: Any,
    processor: Any,
    config: dict[str, Any],
    sample: SampleRecord,
    n: int,
    seed_base: int,
    ablation: MeanAblation | AttentionPatternAblation | None = None,
) -> list[str]:
    model_family = config.get("model_family", "qwen")
    device = _infer_device(model, model_family)
    inputs, vision_mask = prepare_forward_inputs(processor, model, config, sample, device)

    # For InternVL, generate() does not accept image_flags (only forward() does).
    gen_inputs = {k: v for k, v in inputs.items() if k != "image_flags"}

    responses = []
    with torch.no_grad():
        for i in range(n):
            set_seed(seed_base + i)
            if ablation is not None:
                ablation.set_vision_mask(vision_mask)
                if isinstance(ablation, AttentionPatternAblation):
                    img = Image.open(sample.image_path)
                    ablation.set_image_hw((img.height, img.width))
            output_ids = model.generate(**gen_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
            if ablation is not None:
                ablation.set_vision_mask(None)
                if isinstance(ablation, AttentionPatternAblation):
                    ablation.set_image_hw(None)

            if model_family == "internvl2":
                responses.append(processor.decode(output_ids[0], skip_special_tokens=True).strip())
            else:
                generated_ids = output_ids[:, gen_inputs["input_ids"].shape[1] :]
                responses.append(processor.decode(generated_ids[0], skip_special_tokens=True).strip())
    return responses


def prepare_forward_inputs(
    processor: Any,
    model: Any,
    config: dict[str, Any],
    sample: SampleRecord,
    device: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    model_family = config.get("model_family", "qwen")
    messages = build_messages(sample.question, sample.image_path, model_family=model_family)
    if model_family == "internvl2":
        inputs = _prepare_inputs_internvl2(
            processor=processor,
            model=model,
            messages=messages,
            image_path=sample.image_path,
            generation_prefix=GENERATION_PREFIX,
            device=device,
        )
        input_ids = inputs["input_ids"][0]
        img_context_token_id = int(config["img_context_token_id"])
        vision_mask = input_ids.eq(img_context_token_id)
        return inputs, vision_mask
    if model_family == "qwen":
        inputs = _prepare_inputs_qwen(
            processor=processor,
            messages=messages,
            image_path=sample.image_path,
            generation_prefix=GENERATION_PREFIX,
            device=device,
        )
        tokenizer = processor.tokenizer
        input_ids = inputs["input_ids"][0]
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        start_positions = (input_ids == vision_start_id).nonzero(as_tuple=False).flatten().tolist()
        end_positions = (input_ids == vision_end_id).nonzero(as_tuple=False).flatten().tolist()
        if start_positions and end_positions:
            start = start_positions[0]
            end = end_positions[0]
            if end > start + 1:
                vision_mask[start + 1 : end] = True
        return inputs, vision_mask
    raise NotImplementedError(f"Mean-activation collection not implemented for model family {model_family}")


def collect_dataset_mean_activations(
    model_key: str,
    model: Any,
    processor: Any,
    config: dict[str, Any],
    samples: list[SampleRecord],
    device: str,
    force: bool,
    ablation_scope: str,
) -> dict[int, np.ndarray]:
    MEAN_ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MEAN_ACTIVATION_DIR / f"{model_key}__salchartqa__{ablation_scope}__dataset_mean_head_activation.npz"
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        return {
            int(key.split("_")[1]): np.asarray(cached[key], dtype=np.float32)
            for key in cached.files
        }

    model_family = config.get("model_family", "qwen")
    effective_device = _infer_device(model, model_family)
    accumulator = MeanActivationAccumulator(model, expected_layers=config["num_layers"], ablation_scope=ablation_scope)
    with accumulator, torch.no_grad():
        for sample in tqdm(samples, desc=f"Collecting mean activations for {model_key}"):
            inputs, vision_mask = prepare_forward_inputs(processor, model, config, sample, effective_device)
            accumulator.set_vision_mask(vision_mask)
            if model_family == "internvl2":
                model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_flags=inputs["image_flags"],
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            elif model_family == "qwen":
                model(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            else:
                raise NotImplementedError(f"Mean-activation collection not implemented for model family {model_family}")
            accumulator.set_vision_mask(None)

    means = accumulator.finalize()
    np.savez(cache_path, **{f"layer_{layer_idx}": arr for layer_idx, arr in means.items()})
    return means


def evaluate_one_condition(
    model: Any,
    processor: Any,
    config: dict[str, Any],
    samples: list[SampleRecord],
    gt_lookup: dict[str, dict[str, str]],
    num_responses: int,
    seed: int,
    ablation: MeanAblation | None,
) -> dict[str, Any]:
    results = []
    skipped_no_gt = 0

    ctx = ablation if ablation is not None else contextlib.nullcontext()
    with ctx:
        for sample_idx, sample in enumerate(tqdm(samples, desc="Evaluating correctness")):
            gt = gt_lookup.get(sample.image_name, {}).get(_normalize_quotes(sample.question))
            if gt is None:
                skipped_no_gt += 1
                continue
            seed_base = seed + sample_idx * 1000
            responses = generate_responses(
                model=model,
                processor=processor,
                config=config,
                sample=sample,
                n=num_responses,
                seed_base=seed_base,
                ablation=ablation,
            )
            correct = [relaxed_match(resp, gt) for resp in responses]
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "image_name": sample.image_name,
                    "question": sample.question,
                    "ground_truth": gt,
                    "responses": responses,
                    "correct": correct,
                    "num_correct": int(sum(correct)),
                }
            )

    total = len(results)
    if total > 0:
        mean_accuracy = sum(r["num_correct"] / num_responses for r in results) / total
        strict_accuracy = sum(1 for r in results if r["num_correct"] == num_responses) / total
        majority_accuracy = sum(1 for r in results if r["num_correct"] > num_responses / 2) / total
        any_correct_accuracy = sum(1 for r in results if r["num_correct"] > 0) / total
    else:
        mean_accuracy = strict_accuracy = majority_accuracy = any_correct_accuracy = None

    return {
        "summary": {
            "total_samples": total,
            "skipped_no_gt": skipped_no_gt,
            "num_responses_per_question": num_responses,
            "mean_accuracy": mean_accuracy,
            "strict_accuracy": strict_accuracy,
            "majority_accuracy": majority_accuracy,
            "any_correct_accuracy": any_correct_accuracy,
        },
        "results": results,
    }


def load_baseline_condition(correctness_path: Path, sample_ids: set[str]) -> dict[str, Any]:
    with open(correctness_path) as f:
        raw = json.load(f)

    results = [r for r in raw["results"] if r["sample_id"] in sample_ids]
    n = int(raw.get("num_responses", DEFAULT_NUM_RESPONSES))
    total = len(results)
    if total > 0:
        mean_accuracy = sum(r["num_correct"] / n for r in results) / total
        strict_accuracy = sum(1 for r in results if r["num_correct"] == n) / total
        majority_accuracy = sum(1 for r in results if r["num_correct"] > n / 2) / total
        any_correct_accuracy = sum(1 for r in results if r["num_correct"] > 0) / total
    else:
        mean_accuracy = strict_accuracy = majority_accuracy = any_correct_accuracy = None

    return {
        "correctness_file": str(correctness_path),
        "summary": {
            "total_samples": total,
            "skipped_no_gt": 0,
            "num_responses_per_question": n,
            "mean_accuracy": mean_accuracy,
            "strict_accuracy": strict_accuracy,
            "majority_accuracy": majority_accuracy,
            "any_correct_accuracy": any_correct_accuracy,
        },
    }


def build_result_payload(
    model_key: str,
    corr_type: str,
    ranking_df: pd.DataFrame,
    selected_heads: list[HeadRef],
    head_set_name: str,
    ablation_scope: str,
    baseline: dict[str, Any],
    ablated: dict[str, Any],
    eval_sample_ids: list[str],
    ablation_type: str = "mean_activation",
) -> dict[str, Any]:
    base = baseline["summary"]
    abl = ablated["summary"]
    ranking_rows = []
    top_df = ranking_df.head(len(selected_heads))
    for row in top_df.itertuples(index=False):
        ranking_rows.append(
            {
                "rank": int(row.rank),
                "layer": int(row.layer),
                "head": int(row.head),
                "global_head": int(row.global_head),
                "mean_corr": float(row.mean_corr),
                "n_samples": int(row.n_samples),
            }
        )

    abl_type_label = {
        "mean_activation": "dataset_mean_replace_pre_output_projection",
        "mean_attention": "dataset_mean_replace_attention_pattern",
        "uniform_attention": "uniform_replace_attention_pattern",
    }.get(ablation_type, ablation_type)

    return {
        "model_key": model_key,
        "dataset": "salchartqa",
        "corr_type": corr_type,
        "ablation_type": abl_type_label,
        "ablation_scope": ablation_scope,
        "head_set_name": head_set_name,
        "top_k_heads": len(selected_heads),
        "selected_heads": [{"layer": h.layer, "head": h.head} for h in selected_heads],
        "selected_head_ranking": ranking_rows,
        "evaluated_at": datetime.now().isoformat(),
        "evaluation_sample_ids": eval_sample_ids,
        "baseline_correctness_file": baseline.get("correctness_file"),
        "baseline_not_rerun": True,
        "baseline_summary": baseline["summary"],
        "ablated": ablated,
        "delta_summary": {
            "mean_accuracy": None if base["mean_accuracy"] is None else float(abl["mean_accuracy"] - base["mean_accuracy"]),
            "strict_accuracy": None if base["strict_accuracy"] is None else float(abl["strict_accuracy"] - base["strict_accuracy"]),
            "majority_accuracy": None if base["majority_accuracy"] is None else float(abl["majority_accuracy"] - base["majority_accuracy"]),
            "any_correct_accuracy": None if base["any_correct_accuracy"] is None else float(abl["any_correct_accuracy"] - base["any_correct_accuracy"]),
        },
    }


def save_payload(payload: dict[str, Any], model_key: str, head_set_name: str, top_k: int, worker_suffix: str = "", corr_type: str = "pearson") -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corr_tag = f"_{corr_type}" if corr_type != "pearson" else ""
    out_path = EVAL_DIR / f"salchartqa_{head_set_name}_top{top_k}_mean_ablation{corr_tag}__{model_key}{worker_suffix}__{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def print_short_summary(payload: dict[str, Any], out_path: Path) -> None:
    base = payload["baseline_summary"]
    abl = payload["ablated"]["summary"]
    delta = payload["delta_summary"]
    print(f"\nModel: {payload['model_key']} | Set: {payload['head_set_name']}")
    print(f"Top heads: {payload['selected_heads']}")
    print(f"Baseline mean acc: {base['mean_accuracy']:.4f}" if base["mean_accuracy"] is not None else "Baseline mean acc: N/A")
    print(f"Ablated mean acc:  {abl['mean_accuracy']:.4f}" if abl["mean_accuracy"] is not None else "Ablated mean acc:  N/A")
    print(f"Delta mean acc:    {delta['mean_accuracy']:+.4f}" if delta["mean_accuracy"] is not None else "Delta mean acc:    N/A")
    print(f"Baseline majority: {base['majority_accuracy']:.4f}" if base["majority_accuracy"] is not None else "Baseline majority: N/A")
    print(f"Ablated majority:  {abl['majority_accuracy']:.4f}" if abl["majority_accuracy"] is not None else "Ablated majority:  N/A")
    print(f"Delta majority:    {delta['majority_accuracy']:+.4f}" if delta["majority_accuracy"] is not None else "Delta majority:    N/A")
    print(f"Saved {out_path}")


def run_for_model(args: argparse.Namespace, model_key: str) -> list[Path] | None:
    ranking_df = compute_head_ranking(
        model_key=model_key,
        corr_type=args.corr_type,
        sample_limit=args.rank_sample_limit,
        force=args.force_recompute_ranks,
    )
    head_sets = build_head_sets(model_key, ranking_df, args.top_k, args.seed)
    if args.head_sets is not None:
        head_sets = {k: v for k, v in head_sets.items() if k in args.head_sets}

    if args.rank_only:
        print(f"{model_key}: saved ranking only")
        return None

    baseline_path = choose_correctness_file(model_key)
    baseline_raw_ids = set()
    with open(baseline_path) as f:
        baseline_raw = json.load(f)
    baseline_raw_ids = {r["sample_id"] for r in baseline_raw["results"]}

    candidate_samples = build_sample_records(sample_limit=None)
    eval_samples = [s for s in candidate_samples if s.sample_id in baseline_raw_ids]
    if args.eval_sample_ids_file is not None:
        with open(args.eval_sample_ids_file) as f:
            restrict_ids = set(json.load(f))
        eval_samples = [s for s in eval_samples if s.sample_id in restrict_ids]
        print(f"Restricted to {len(eval_samples)} samples from {args.eval_sample_ids_file}")
    if args.eval_sample_limit is not None:
        eval_samples = eval_samples[: args.eval_sample_limit]
    eval_sample_ids = {s.sample_id for s in eval_samples}
    baseline = load_baseline_condition(baseline_path, eval_sample_ids)
    gt_lookup = load_salchartqa_ground_truth()

    model, processor, config = load_vlm_model(model_key, device=args.device)

    ablation_type = args.ablation_type

    # Prepare ablation-type-specific data
    if ablation_type == "mean_activation":
        mean_source_samples = candidate_samples if args.mean_sample_limit is None else candidate_samples[: args.mean_sample_limit]
        head_means = collect_dataset_mean_activations(
            model_key=model_key,
            model=model,
            processor=processor,
            config=config,
            samples=mean_source_samples,
            device=args.device,
            force=args.force_recompute_means,
            ablation_scope=args.ablation_scope,
        )
        mean_attn_patterns = None
    elif ablation_type == "mean_attention":
        head_means = None
        mean_attn_patterns = compute_mean_attention_patterns(
            model_key=model_key,
            corr_type=args.corr_type,
            sample_limit=args.rank_sample_limit,
            force=args.force_recompute_means,
        )
    elif ablation_type == "uniform_attention":
        head_means = None
        mean_attn_patterns = None
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    out_paths: list[Path] = []
    for idx, (head_set_name, selected_heads) in enumerate(head_sets.items()):
        if ablation_type == "mean_activation":
            abl = MeanAblation(
                model,
                selected_heads,
                expected_layers=config["num_layers"],
                head_means=head_means,
                ablation_scope=args.ablation_scope,
            )
        elif ablation_type in ("mean_attention", "uniform_attention"):
            abl = AttentionPatternAblation(
                model,
                selected_heads,
                expected_layers=config["num_layers"],
                mean_attn_patterns=mean_attn_patterns,
                uniform=ablation_type == "uniform_attention",
            )

        ablated = evaluate_one_condition(
            model=model,
            processor=processor,
            config=config,
            samples=eval_samples,
            gt_lookup=gt_lookup,
            num_responses=baseline["summary"]["num_responses_per_question"],
            seed=args.seed + idx * 100_000,
            ablation=abl,
        )

        payload = build_result_payload(
            model_key=model_key,
            corr_type=args.corr_type,
            ranking_df=ranking_df,
            selected_heads=selected_heads,
            head_set_name=head_set_name,
            ablation_scope=args.ablation_scope,
            baseline=baseline,
            ablated=ablated,
            eval_sample_ids=[s.sample_id for s in eval_samples],
            ablation_type=ablation_type,
        )
        out_path = save_payload(payload, model_key=model_key, head_set_name=head_set_name, top_k=args.top_k, corr_type=args.corr_type)
        print_short_summary(payload, out_path)
        out_paths.append(out_path)
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys to run. Use 'all' for all instruct models.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top-ranked heads to ablate together.",
    )
    parser.add_argument(
        "--corr-type",
        choices=["pearson", "spearman", "s_img"],
        default="pearson",
        help="Correlation type for ranking heads against mean_all human maps.",
    )
    parser.add_argument(
        "--rank-sample-limit",
        type=int,
        default=None,
        help="Optional cap on usable SalChartQA samples when computing head rankings.",
    )
    parser.add_argument(
        "--eval-sample-limit",
        type=int,
        default=None,
        help="Optional cap on usable SalChartQA samples for correctness evaluation.",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=DEFAULT_NUM_RESPONSES,
        help="Number of stochastic responses per question during correctness evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to the model loader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used for matched baseline vs ablated sampling.",
    )
    parser.add_argument(
        "--eval-sample-ids-file",
        type=str,
        default=None,
        help="Path to JSON file with list of sample_id strings to restrict evaluation to.",
    )
    parser.add_argument(
        "--ablation-scope",
        choices=["vision_tokens", "all_tokens"],
        default="vision_tokens",
        help="Whether to replace selected LM-head outputs only at vision-token positions or across the full sequence. Only used with --ablation-type mean_activation.",
    )
    parser.add_argument(
        "--ablation-type",
        choices=["mean_activation", "mean_attention", "uniform_attention"],
        default="mean_activation",
        help=(
            "mean_activation: replace pre-o_proj activation with dataset mean (current default). "
            "mean_attention: replace attention-to-vision spatial pattern with dataset mean "
            "(keeps value processing intact, tests spatial selectivity specifically)."
        ),
    )
    parser.add_argument(
        "--mean-sample-limit",
        type=int,
        default=None,
        help="Optional cap on usable SalChartQA samples when estimating dataset-wide head means.",
    )
    parser.add_argument(
        "--head-sets",
        nargs="+",
        default=None,
        help="Which head sets to evaluate (default: all four). E.g. --head-sets top5 bottom5",
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Only compute/save head rankings; skip correctness evaluation.",
    )
    parser.add_argument(
        "--force-recompute-ranks",
        action="store_true",
        help="Recompute head rankings even if a cached CSV exists.",
    )
    parser.add_argument(
        "--force-recompute-means",
        action="store_true",
        help="Recompute cached dataset-wide mean head activations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = INSTRUCT_MODELS if args.models == ["all"] else args.models
    invalid = [m for m in models if m not in INSTRUCT_MODELS]
    if invalid:
        raise SystemExit(f"Unsupported model(s): {invalid}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for model_key in models:
        model_outputs = run_for_model(args, model_key)
        if model_outputs:
            produced.extend(model_outputs)
    if produced:
        print("\nOutputs:")
        for path in produced:
            print(path)


if __name__ == "__main__":
    main()
