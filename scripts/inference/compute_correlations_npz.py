#!/usr/bin/env python3
"""Compute per-head correlation NPZs for InternVL instruct models.

Produces the same format as recompute_correlations.py:
  {model}_mean_gaze_correlations_sigma{sigma}.npz
  {model}_mean_gaze_sample_results_sigma{sigma}.json

CPU-only. ~10-30 min per model on a single core.

Usage:
  python compute_instruct_correlations_npz.py [--model internvl3-1b-instruct] [--dataset salchartqa]
  python compute_instruct_correlations_npz.py --all  # run all 7 instruct models × 2 datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import erf

sys.path.insert(0, "./scripts/lib")
from correctness_final.evaluate_ordering import relaxed_match

# ── Paths ──
ATTN_BASE = Path("./cache/lvlm-chart/attention_maps")
CORR_OUT = Path("./data/lvlm-chart/correlations")
SALCHARTQA_DIR = Path("./data/salchartqa")
TASKVIS_DIR = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis")
TASKVIS_IMAGE_DIR = Path("./cache/lvlm-chart/eye_gaze_datasets/massvis")
DETERMINED_GT_PATH = Path("./data/lvlm-chart/correctness/determined_ground_truth_fixed.json")
RECLASSIFIED_GT_PATH = Path("./data/lvlm-chart/ground_truth/gpt-5-mini-2025-08-07_merged_all.json")

WORKER_SIGMA_PX = 19.0  # Both datasets, per SalChartQA paper
ATTN_SIGMA = 1.0
SOURCE_TOKEN = "question"

INSTRUCT_MODELS = [
    "internvl3-1b-instruct", "internvl3-2b-instruct", "internvl3-8b-instruct",
    "internvl3.5-1b-instruct", "internvl3.5-2b-instruct",
    "internvl3.5-4b-instruct", "internvl3.5-8b-instruct",
]


# ── Grid inference ──

def infer_tile_grid(n_vision, img_h, img_w, tokens_per_tile=256, has_thumbnail=True):
    tile_side = int(round(tokens_per_tile ** 0.5))
    n_tiles = n_vision // tokens_per_tile
    if n_tiles * tokens_per_tile != n_vision or n_tiles == 0:
        return None
    n_dynamic = n_tiles - 1 if (has_thumbnail and n_tiles > 1) else n_tiles
    n_vision_use = n_dynamic * tokens_per_tile
    aspect = img_w / img_h
    best, best_err = None, float("inf")
    for th in range(1, n_dynamic + 1):
        if n_dynamic % th == 0:
            tw = n_dynamic // th
            err = abs((tw / th) - aspect)
            if err < best_err:
                best_err = err
                best = (th * tile_side, tw * tile_side)
    if best is None:
        return None
    return best[0], best[1], n_vision_use


def get_attn_grid(data, n_layers, n_heads, img_h, img_w):
    attn_key = f"attn_{SOURCE_TOKEN}"
    if attn_key not in data:
        return None
    attn = data[attn_key].astype(np.float32)
    tt = data["token_types"][:attn.shape[2]]
    vision_mask = tt == 0
    attn_vision = attn[:, :, vision_mask]
    n_vision = int(vision_mask.sum())
    if attn_vision.shape[0] != n_layers or attn_vision.shape[1] != n_heads:
        return None

    tile_info = infer_tile_grid(n_vision, img_h, img_w)
    if tile_info is None:
        return None
    gh, gw, n_vision_use = tile_info
    if n_vision_use != n_vision:
        attn_vision = attn_vision[:, :, :n_vision_use]
    tile_side = 16
    tiles_h, tiles_w = gh // tile_side, gw // tile_side
    attn_2d = (attn_vision
               .reshape(n_layers, n_heads, tiles_h, tiles_w, tile_side, tile_side)
               .transpose(0, 1, 2, 4, 3, 5)
               .reshape(n_layers, n_heads, gh, gw)
               .astype(np.float64))
    return attn_2d, gh, gw


# ── Fixation loading ──

def gaussian_clicks_to_grid(fixations, gh, gw, img_h, img_w, sigma):
    h_edges = np.linspace(0, img_h, gh + 1)
    w_edges = np.linspace(0, img_w, gw + 1)
    s = sigma * np.sqrt(2)
    grid = np.zeros((gh, gw), dtype=np.float64)
    for x, y in fixations:
        row_contrib = (erf((h_edges[1:] - y) / s) - erf((h_edges[:-1] - y) / s)) / 2
        col_contrib = (erf((w_edges[1:] - x) / s) - erf((w_edges[:-1] - x) / s)) / 2
        grid += row_contrib[:, None] * col_contrib[None, :]
    return grid


def normalize_rows(M):
    mean = M.mean(axis=1, keepdims=True)
    std = M.std(axis=1, keepdims=True)
    valid = std.ravel() > 1e-10
    out = np.zeros_like(M)
    out[valid] = (M[valid] - mean[valid]) / std[valid]
    return out, valid


# ── SalChartQA loading ──

def load_salchartqa_samples():
    with open(SALCHARTQA_DIR / "image_questions.json") as f:
        iq = json.load(f)
    samples = []
    for image_name, qdict in iq.items():
        for q_id in qdict:
            samples.append({"idx": len(samples), "image_name": image_name, "question_id": q_id})
    return samples


def load_gt():
    with open(DETERMINED_GT_PATH) as f:
        det = json.load(f)
    gt = {}
    for key, val in det["gt_lookup"].items():
        if val.get("in_confident_subset") and val.get("gt"):
            gt[key] = val["gt"]
    with open(RECLASSIFIED_GT_PATH) as f:
        merged = json.load(f)
    for key, val in merged.items():
        gk = f"{val['image_name']}_{val['question_id']}"
        if gk not in gt:
            g = val["worker_ground_truth"]
            if isinstance(g, list): g = g[0]
            gt[gk] = str(g).strip()
    return gt


def load_worker_responses():
    wr = {}
    with open(SALCHARTQA_DIR / "unified_approved.csv") as f:
        for row in csv.DictReader(f):
            stem = Path(row["image_name"]).stem
            wr[(stem, row["participant_id"])] = row["answer"]
    return wr


def load_salchartqa_workers(image_name, question_id, gh, gw, img_h, img_w, gt, wr):
    stem = os.path.splitext(image_name)[0]
    fix_dir = SALCHARTQA_DIR / "fixationByVis" / stem / question_id
    if not fix_dir.is_dir():
        return []
    gt_key = f"{image_name}_{question_id}"
    gt_str = gt.get(gt_key) or gt.get(f"{stem}.png_{question_id}")

    grids = []
    for label in ["True", "False"]:
        folder = fix_dir / label
        if not folder.is_dir():
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".csv"):
                continue
            wid = fname[:-4]
            fixations = []
            with open(folder / fname) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        fixations.append((int(parts[0]), int(parts[1])))
            if fixations:
                grid = gaussian_clicks_to_grid(fixations, gh, gw, img_h, img_w, WORKER_SIGMA_PX)
                s = grid.sum()
                if s > 1e-10:
                    grid /= s
                grids.append(grid)
    return grids


# ── TaskVIS loading ──

def load_taskvis_samples():
    image_files = {}
    with open(TASKVIS_DIR / "images.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            image_files[int(parts[0])] = parts[1]
    samples = []
    with open(TASKVIS_DIR / "tasks.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            image_id = int(parts[0])
            samples.append({"idx": len(samples), "image_id": image_id,
                            "task_type": parts[1], "image_name": image_files[image_id]})
    return samples


def load_taskvis_meta():
    tasktypes = {}
    with open(TASKVIS_DIR / "tasktypes.txt") as f:
        header = f.readline().strip().split("\t")
        img_ids = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split("\t")
            tasktypes[parts[0]] = {iid: t for iid, t in zip(img_ids, parts[1:])}
    answers = {}
    with open(TASKVIS_DIR / "answers.txt") as f:
        header = f.readline().strip().split("\t")
        img_ids = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split("\t")
            answers[parts[0]] = {iid: int(v) == 1 for iid, v in zip(img_ids, parts[1:])}
    return tasktypes, answers, sorted(tasktypes.keys())


def load_taskvis_workers(image_id, task_type, gh, gw, img_h, img_w, tasktypes, answers, participants):
    grids = []
    for pid in participants:
        if tasktypes.get(pid, {}).get(image_id) != task_type:
            continue
        fix_path = TASKVIS_DIR / "fixations" / f"rec_p{pid[1:]}_fix_{image_id}.tsv"
        if not fix_path.exists():
            continue
        fixations = []
        with open(fix_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] == "Duration":
                    break
                if len(parts) >= 4:
                    fixations.append((int(parts[2]), int(parts[3])))
        if fixations:
            grid = gaussian_clicks_to_grid(fixations, gh, gw, img_h, img_w, WORKER_SIGMA_PX)
            s = grid.sum()
            if s > 1e-10:
                grid /= s
            grids.append(grid)
    return grids


# ── Main processing ──

def process_model(model_key, dataset, sigma=ATTN_SIGMA):
    print(f"\n{'='*60}")
    print(f"Model: {model_key}, Dataset: {dataset}, Sigma: {sigma}")
    print(f"{'='*60}")

    attn_dir = ATTN_BASE / dataset / "image_suffix_question" / model_key
    if not attn_dir.exists():
        print(f"  No attention dir: {attn_dir}")
        return

    # Probe for n_layers, n_heads
    attn_key = f"attn_{SOURCE_TOKEN}"
    probe = None
    for npz_path in sorted(attn_dir.glob("*.npz"))[:10]:
        p = np.load(str(npz_path))
        if attn_key in p:
            probe = p
            break
    if probe is None:
        print(f"  No files with {attn_key}")
        return
    n_layers, n_heads = probe[attn_key].shape[:2]
    print(f"  {n_layers} layers × {n_heads} heads = {n_layers * n_heads} total")

    # Load dataset-specific data
    if dataset == "salchartqa":
        samples = load_salchartqa_samples()
        gt = load_gt()
        wr = load_worker_responses()
        image_dir = SALCHARTQA_DIR / "raw_img"
        taskvis_meta = None
    else:
        samples = load_taskvis_samples()
        taskvis_meta = load_taskvis_meta()
        image_dir = TASKVIS_IMAGE_DIR
        gt = wr = None

    n_samples = len(samples)
    cc_mean_all = np.full((n_samples, n_layers, n_heads), np.nan, dtype=np.float32)
    cc_mean_correct = np.full((n_samples, n_layers, n_heads), np.nan, dtype=np.float32)
    cc_mean_incorrect = np.full((n_samples, n_layers, n_heads), np.nan, dtype=np.float32)
    cc_max_individual = np.full((n_samples, n_layers, n_heads), np.nan, dtype=np.float32)

    sample_results = []
    image_size_cache = {}
    t_start = time.time()
    n_processed = 0

    for sample in samples:
        idx = sample["idx"]

        if dataset == "salchartqa":
            npz_path = attn_dir / f"salchartqa_{idx}.npz"
            image_name = sample["image_name"]
        else:
            npz_path = attn_dir / f"taskvis_{sample['image_id']}_{sample['task_type']}.npz"
            image_name = sample["image_name"]

        if not npz_path.exists():
            continue

        data = np.load(str(npz_path))

        if image_name not in image_size_cache:
            img_path = image_dir / image_name
            if not img_path.exists():
                continue
            img = Image.open(img_path)
            image_size_cache[image_name] = (img.height, img.width)
        img_h, img_w = image_size_cache[image_name]

        result = get_attn_grid(data, n_layers, n_heads, img_h, img_w)
        if result is None:
            continue
        attn_2d, gh, gw = result
        n_pixels = gh * gw

        # Smooth attention
        attn_smooth = np.empty_like(attn_2d)
        for l in range(n_layers):
            for h in range(n_heads):
                head = attn_2d[l, h]
                if head.sum() < 0.01:
                    attn_smooth[l, h] = 0.0
                    continue
                sm = gaussian_filter(head, sigma=sigma)
                ss = sm.sum()
                attn_smooth[l, h] = sm / ss if ss > 1e-10 else 0.0

        # Load workers
        if dataset == "salchartqa":
            worker_grids = load_salchartqa_workers(
                image_name, sample["question_id"], gh, gw, img_h, img_w, gt, wr)
        else:
            worker_grids = load_taskvis_workers(
                sample["image_id"], sample["task_type"], gh, gw, img_h, img_w,
                *taskvis_meta)

        if len(worker_grids) < 2:
            continue

        # Mean gaze maps
        mean_all = np.mean(worker_grids, axis=0)
        s = mean_all.sum()
        if s > 1e-10:
            mean_all /= s

        # Flatten and normalize for correlation
        mean_all_flat = mean_all.ravel().reshape(1, -1)
        mean_all_normed, mean_all_valid = normalize_rows(mean_all_flat)

        attn_flat = attn_smooth.reshape(n_layers * n_heads, n_pixels)
        attn_normed, attn_valid = normalize_rows(attn_flat)

        cc = (attn_normed @ mean_all_normed.T) / n_pixels
        cc[~attn_valid, :] = np.nan
        cc[:, ~mean_all_valid] = np.nan
        cc_mean_all[idx] = cc.reshape(n_layers, n_heads)

        # Max individual worker correlation
        for wg in worker_grids:
            wg_flat = wg.ravel().reshape(1, -1)
            wg_normed, wg_valid = normalize_rows(wg_flat)
            cc_w = (attn_normed @ wg_normed.T) / n_pixels
            cc_w[~attn_valid, :] = np.nan
            cc_w[:, ~wg_valid] = np.nan
            cc_w = cc_w.reshape(n_layers, n_heads)
            cc_max_individual[idx] = np.fmax(cc_max_individual[idx], cc_w)

        n_processed += 1
        sample_results.append({
            "idx": idx,
            "n_workers": len(worker_grids),
            "grid_shape": [gh, gw],
            "cc_mean_all_max_head": float(np.nanmax(cc_mean_all[idx])),
            "cc_max_individual_max_head": float(np.nanmax(cc_max_individual[idx])),
            "image_name": image_name,
            "question_id": sample.get("question_id", sample.get("task_type", "")),
        })

        if n_processed % 500 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed
            eta = (n_samples - n_processed) / rate if rate > 0 else 0
            print(f"  {n_processed}/{n_samples} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t_start
    print(f"  Done: {n_processed}/{n_samples} in {elapsed:.0f}s")

    # Save
    out_dir = CORR_OUT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_out = out_dir / f"{model_key}_mean_gaze_correlations_sigma{sigma}.npz"
    np.savez_compressed(str(npz_out),
                        cc_mean_all_per_sample=cc_mean_all,
                        cc_mean_correct_per_sample=cc_mean_correct,
                        cc_mean_incorrect_per_sample=cc_mean_incorrect,
                        cc_max_individual_per_sample=cc_max_individual,
                        attn_layer_indices=np.arange(n_layers),
                        n_layers=n_layers, n_heads=n_heads)

    json_out = out_dir / f"{model_key}_mean_gaze_sample_results_sigma{sigma}.json"
    with open(json_out, "w") as f:
        json.dump({"model": model_key, "dataset": dataset,
                    "n_processed": n_processed, "sample_results": sample_results}, f)

    print(f"  Saved {npz_out.name}")
    print(f"  Saved {json_out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="salchartqa",
                        choices=["salchartqa", "taskvis"])
    parser.add_argument("--all", action="store_true",
                        help="Run all instruct models × both datasets")
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    if args.all:
        for ds in ["salchartqa", "taskvis"]:
            for mk in INSTRUCT_MODELS:
                process_model(mk, ds, args.sigma)
    elif args.model:
        process_model(args.model, args.dataset, args.sigma)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
