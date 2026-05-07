#!/usr/bin/env python3
"""Compute instruct-model and human baseline correlations for TaskVIS and SalChartQA.

Outputs:
- summary CSV with one row per dataset x row_type x model/baseline x corr_type x metric
- optional per-sample CSV for debugging/auditing

Notes
- Model-human comparisons downsample human fixations to the model's vision-token grid.
- Human-human baselines operate at native image resolution.
- SalChartQA uses the fixed determined ground truth to identify the 4,556 clean usable
  subset and to score worker/model correctness on that subset.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.stats import rankdata

sys.path.insert(0, "./scripts/lib")
sys.path.insert(0, "./scripts/lib/corr_plots")

from correctness_final.evaluate_ordering import relaxed_match  # type: ignore
import recompute_correlations as rc  # type: ignore


ROOT = Path("./")
OUT_DIR = ROOT / "tasks_executed" / "correlation_instruct_human"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECTNESS_DIR = Path("./data/lvlm-chart/correctness")
SALCHARTQA_DIR = Path("./data/salchartqa")
TASKVIS_DIR = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis")
TASKVIS_IMAGE_DIR = Path("./cache/lvlm-chart/eye_gaze_datasets/massvis")
ATTN_BASE = Path("./cache/lvlm-chart/attention_maps")
HF_CACHE = Path("./cache/hf")

WORKER_SIGMA_PX = {
    "taskvis": 19.0,
    "salchartqa": 19.0,  # SalChartQA paper (Wang et al. 2024): σ=19px ≈ 1° visual angle
}
ATTN_SIGMA = 1.0
ATTN_MIN_SUM = 0.01
SOURCE_TOKEN = "question"

MODEL_INFO = {
    "2.5-3B": {
        "model_name": "Qwen2.5-VL-3B-Instruct",
        "hf_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen",
    },
    "2.5-7B": {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "hf_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen",
    },
    "2B": {
        "model_name": "Qwen3-VL-2B-Instruct",
        "hf_model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "family": "qwen",
    },
    "4B": {
        "model_name": "Qwen3-VL-4B-Instruct",
        "hf_model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "qwen",
    },
    "8B": {
        "model_name": "Qwen3-VL-8B-Instruct",
        "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen",
    },
    "internvl3-1b-instruct": {
        "model_name": "InternVL3-1B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3-1B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3-2b-instruct": {
        "model_name": "InternVL3-2B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3-2B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3-8b-instruct": {
        "model_name": "InternVL3-8B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3-8B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3.5-1b-instruct": {
        "model_name": "InternVL3.5-1B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3_5-1B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3.5-2b-instruct": {
        "model_name": "InternVL3.5-2B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3_5-2B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3.5-4b-instruct": {
        "model_name": "InternVL3.5-4B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3_5-4B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
    "internvl3.5-8b-instruct": {
        "model_name": "InternVL3.5-8B-Instruct",
        "hf_model_id": "OpenGVLab/InternVL3_5-8B-Instruct",
        "family": "internvl",
        "tokens_per_tile": 256,
        "has_thumbnail": True,
    },
}


@dataclass
class WorkerRecord:
    worker_id: str
    fixations: list[tuple[int, int]]
    is_correct: bool | None


def gaussian_clicks_native(fixations, img_h, img_w, sigma):
    grid = np.zeros((img_h, img_w), dtype=np.float32)
    for x, y in fixations:
        if 0 <= x < img_w and 0 <= y < img_h:
            grid[y, x] += 1.0
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    s = float(grid.sum())
    if s > 1e-10:
        grid /= s
    return grid.astype(np.float64, copy=False)


def bilinear_downsample_map(full_map: np.ndarray, gh: int, gw: int):
    arr = np.asarray(full_map, dtype=np.float32)
    img = Image.fromarray(arr, mode="F")
    resized = img.resize((gw, gh), resample=Image.Resampling.BILINEAR)
    out = np.asarray(resized, dtype=np.float64)
    s = float(out.sum())
    if s > 1e-10:
        out /= s
    return out


def normalize_rows(M: np.ndarray, corr_type: str):
    X = np.asarray(M, dtype=np.float64)
    if corr_type == "spearman":
        ranked = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            ranked[i] = rankdata(X[i])
        X = ranked
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    valid = std[:, 0] > 1e-10
    out = np.zeros_like(X, dtype=np.float64)
    out[valid] = (X[valid] - mean[valid]) / std[valid]
    return out, valid


def corr_matrix(A: np.ndarray, B: np.ndarray, corr_type: str):
    a_norm, a_valid = normalize_rows(A, corr_type)
    b_norm, b_valid = normalize_rows(B, corr_type)
    n = A.shape[1]
    cc = (a_norm @ b_norm.T) / n
    cc[~a_valid, :] = np.nan
    cc[:, ~b_valid] = np.nan
    return cc


def summarize(values: list[float]):
    arr = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def parse_ts(ts: str):
    m = re.search(r"_(\d{8}_\d{6})(?:_fixed)?\.json$", ts)
    return m.group(1) if m else ""


def choose_correctness_file(dataset: str, model_key: str) -> Path | None:
    if dataset == "taskvis":
        n_tag = "n90"
        pattern = f"correctness_*_{dataset}_x5_{n_tag}_*.json"
    else:
        n_tag = "n5999"
        pattern = f"correctness_*_{dataset}_x5_{n_tag}_*.json"

    candidates = []
    for path in CORRECTNESS_DIR.glob(pattern):
        name = path.name.lower()
        key = model_key.lower()
        if key in {"2b", "4b", "8b"}:
            if not re.match(rf"^correctness_{re.escape(key)}_{dataset}_x5_{n_tag}_", name):
                continue
        elif key in {"2.5-3b", "2.5-7b"}:
            if not re.match(rf"^correctness_{re.escape(key)}_{dataset}_x5_{n_tag}_", name):
                continue
        else:
            if not re.match(rf"^correctness_{re.escape(key)}_{dataset}_x5_{n_tag}_", name):
                continue
        fixed = name.endswith("_fixed.json")
        candidates.append((fixed, parse_ts(name), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2].name))
    return candidates[-1][2]


def load_taskvis():
    image_files = {}
    with open(TASKVIS_DIR / "images.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            image_files[int(parts[0])] = parts[1]

    tasks = {}
    with open(TASKVIS_DIR / "tasks.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            tasks[(int(parts[0]), parts[1])] = parts[2]

    participant_tasks = {}
    with open(TASKVIS_DIR / "tasktypes.txt") as f:
        header = f.readline().strip().split("\t")
        image_ids = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split("\t")
            pid = parts[0]
            participant_tasks[pid] = {}
            for img_id, task_type in zip(image_ids, parts[1:]):
                participant_tasks[pid][img_id] = task_type

    participant_answers = {}
    with open(TASKVIS_DIR / "answers.txt") as f:
        header = f.readline().strip().split("\t")
        image_ids = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split("\t")
            pid = parts[0]
            participant_answers[pid] = {}
            for img_id, ans in zip(image_ids, parts[1:]):
                try:
                    participant_answers[pid][img_id] = int(ans) == 1
                except ValueError:
                    participant_answers[pid][img_id] = None

    samples = []
    for (image_id, task_type), question in sorted(tasks.items()):
        samples.append(
            {
                "sample_id": f"taskvis_{image_id}_{task_type}",
                "idx": len(samples),
                "image_id": image_id,
                "task_type": task_type,
                "question": question,
                "image_name": image_files[image_id],
            }
        )

    worker_skill_values = {}
    for pid, answers in participant_answers.items():
        vals = [v for v in answers.values() if v is not None]
        worker_skill_values[pid] = float(np.mean(vals)) if vals else np.nan

    return {
        "samples": samples,
        "participant_tasks": participant_tasks,
        "participant_answers": participant_answers,
        "worker_skill": worker_skill_values,
        "image_dir": TASKVIS_IMAGE_DIR,
    }


def load_taskvis_fixation(fix_path: Path):
    fixations = []
    with open(fix_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == "Duration":
                break
            if len(parts) >= 4:
                fixations.append((int(parts[2]), int(parts[3])))
    return fixations


def get_taskvis_workers(sample, taskvis_data):
    workers = []
    image_id = sample["image_id"]
    task_type = sample["task_type"]
    participant_tasks = taskvis_data["participant_tasks"]
    participant_answers = taskvis_data["participant_answers"]
    for pid in sorted(participant_tasks.keys()):
        if participant_tasks.get(pid, {}).get(image_id) != task_type:
            continue
        pid_num = pid[1:]
        fix_path = TASKVIS_DIR / "fixations" / f"rec_p{pid_num}_fix_{image_id}.tsv"
        if not fix_path.exists():
            continue
        fixations = load_taskvis_fixation(fix_path)
        if not fixations:
            continue
        workers.append(WorkerRecord(pid, fixations, participant_answers.get(pid, {}).get(image_id)))
    return workers


def load_salchartqa():
    with open(SALCHARTQA_DIR / "image_questions.json") as f:
        image_questions = json.load(f)
    with open(CORRECTNESS_DIR / "determined_ground_truth_fixed.json") as f:
        det_gt = json.load(f)
    gt_lookup = det_gt["gt_lookup"]

    usable_lookup = {}
    gt_answer_lookup = {}
    for key, val in gt_lookup.items():
        usable_lookup[key] = bool(val.get("in_confident_subset")) and not bool(val.get("is_questionable"))
        gt_answer_lookup[key] = val.get("gt")

    worker_rows = []
    with open(SALCHARTQA_DIR / "unified_approved.csv") as f:
        for row in csv.DictReader(f):
            worker_rows.append(row)

    worker_answer_lookup = {}
    worker_correct_rows = []
    for row in worker_rows:
        img_name = row["image_name"]
        question = row["question"]
        q_id = None
        for cand in ["Q0", "Q1"]:
            if image_questions.get(img_name, {}).get(cand) == question:
                q_id = cand
                break
        if q_id is None:
            continue
        gt_key = f"{img_name}_{q_id}"
        usable = usable_lookup.get(gt_key, False)
        answer = row["answer"]
        worker_id = row["participant_id"]
        worker_answer_lookup[(Path(img_name).stem, q_id, worker_id)] = answer
        is_correct = None
        gt_answer = gt_answer_lookup.get(gt_key)
        if usable and gt_answer:
            is_correct = relaxed_match(answer, gt_answer)
        worker_correct_rows.append((worker_id, gt_key, is_correct, usable))

    skill_vals = {}
    for worker_id, gt_key, is_correct, usable in worker_correct_rows:
        if not usable or is_correct is None:
            continue
        skill_vals.setdefault(worker_id, []).append(bool(is_correct))
    worker_skill = {
        wid: float(np.mean(vals)) if vals else np.nan
        for wid, vals in skill_vals.items()
    }

    samples = []
    for img_name, qdict in image_questions.items():
        img_stem = Path(img_name).stem
        for q_id, question in qdict.items():
            gt_key = f"{img_name}_{q_id}"
            if not usable_lookup.get(gt_key, False):
                continue
            samples.append(
                {
                    "sample_id": f"salchartqa_{img_stem}_{q_id}",
                    "idx": len(samples),
                    "image_name": img_name,
                    "question_id": q_id,
                    "question": question,
                    "gt_key": gt_key,
                }
            )

    return {
        "samples": samples,
        "gt_answer_lookup": gt_answer_lookup,
        "worker_answer_lookup": worker_answer_lookup,
        "worker_skill": worker_skill,
        "image_dir": SALCHARTQA_DIR / "raw_img",
    }


def get_salchartqa_workers(sample, sal_data):
    image_name = sample["image_name"]
    q_id = sample["question_id"]
    image_stem = Path(image_name).stem
    fix_dir = SALCHARTQA_DIR / "fixationByVis" / image_stem / q_id
    if not fix_dir.is_dir():
        return []
    gt_answer = sal_data["gt_answer_lookup"].get(sample["gt_key"])
    workers = []
    for folder_name in ["True", "False"]:
        folder = fix_dir / folder_name
        if not folder.is_dir():
            continue
        for fname in sorted(folder.glob("*.csv")):
            worker_id = fname.stem
            fixations = []
            with open(fname) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        fixations.append((int(parts[0]), int(parts[1])))
            if not fixations:
                continue
            answer = sal_data["worker_answer_lookup"].get((image_stem, q_id, worker_id), "")
            is_correct = relaxed_match(str(answer), gt_answer) if gt_answer else None
            workers.append(WorkerRecord(worker_id, fixations, is_correct))
    return workers


def get_image_size(image_path: Path, cache: dict[str, tuple[int, int]]):
    key = str(image_path)
    if key not in cache:
        img = Image.open(image_path)
        cache[key] = (img.height, img.width)
    return cache[key]


def load_attention_grid(npz_path: Path, img_h: int, img_w: int, model_cfg: dict):
    data = np.load(npz_path)
    attn_key = f"attn_{SOURCE_TOKEN}"
    attn = data[attn_key].astype(np.float32)
    n_layers, n_heads = attn.shape[:2]
    result = rc.get_attn_grid(data, n_layers, n_heads, img_h, img_w, model_cfg)
    if result is None:
        return None
    return result


def compute_model_metrics(attn_2d, worker_maps_full, corr_type):
    n_layers, n_heads, gh, gw = attn_2d.shape
    n_heads_total = n_layers * n_heads
    attn_transformed = rc.transform_attention_perhead(attn_2d, sigma=ATTN_SIGMA, min_sum=ATTN_MIN_SUM)
    attn_flat = attn_transformed.reshape(n_heads_total, gh * gw)

    worker_grids = []
    for worker_id, full_map, is_correct in worker_maps_full:
        grid = bilinear_downsample_map(full_map, gh, gw)
        worker_grids.append((worker_id, grid, is_correct))

    all_grids = [g for _, g, _ in worker_grids]
    correct_grids = [g for _, g, c in worker_grids if c is True]
    incorrect_grids = [g for _, g, c in worker_grids if c is False]

    targets = {
        "max_any_human": np.array([g.ravel() for g in all_grids], dtype=np.float64),
        "mean_all_humans": np.mean(all_grids, axis=0).ravel().reshape(1, -1),
        "mean_correct_humans": None if not correct_grids else np.mean(correct_grids, axis=0).ravel().reshape(1, -1),
        "mean_incorrect_humans": None if not incorrect_grids else np.mean(incorrect_grids, axis=0).ravel().reshape(1, -1),
    }

    out = {}
    for metric_name, target in targets.items():
        if target is None:
            out[metric_name] = np.nan
            continue
        cc = corr_matrix(attn_flat, target, corr_type)
        if metric_name == "max_any_human":
            with np.errstate(invalid="ignore"):
                per_head = np.nanmax(cc, axis=1)
        else:
            per_head = cc[:, 0]
        with np.errstate(invalid="ignore"):
            out[metric_name] = float(np.nanmax(per_head))
    return out


def compute_human_baselines(native, worker_skill, corr_type):
    worker_ids = [wid for wid, _, _ in native]
    skills = np.array([worker_skill.get(wid, np.nan) for wid in worker_ids], dtype=np.float64)
    mats = np.array([g.ravel() for _, g, _ in native], dtype=np.float64)
    out = {
        "max_any_human__max_any_human": (np.nan, np.nan),
        "max_any_human__mean_all_humans": (np.nan, np.nan),
        "max_any_human__mean_correct_humans": (np.nan, np.nan),
        "max_any_human__mean_incorrect_humans": (np.nan, np.nan),
    }

    sample_best_values = {k: [] for k in out}
    sample_best_skills = {k: [] for k in out}

    n_workers = len(native)
    for lhs_idx in range(n_workers):
        lhs = mats[lhs_idx:lhs_idx + 1]
        lhs_correct = native[lhs_idx][2]

        other_idx = [i for i in range(n_workers) if i != lhs_idx]
        if other_idx:
            target = mats[other_idx]
            cc = corr_matrix(lhs, target, corr_type)
            val = float(np.nanmax(cc))
            sample_best_values["max_any_human__max_any_human"].append(val)
            sample_best_skills["max_any_human__max_any_human"].append(skills[lhs_idx])

        target = mats[other_idx]
        if target.size:
            mean_target = target.mean(axis=0, keepdims=True)
            cc = corr_matrix(lhs, mean_target, corr_type)
            sample_best_values["max_any_human__mean_all_humans"].append(float(cc[0, 0]))
            sample_best_skills["max_any_human__mean_all_humans"].append(skills[lhs_idx])

        corr_idx = [i for i, (_, _, c) in enumerate(native) if c is True and i != lhs_idx]
        if corr_idx:
            mean_target = mats[corr_idx].mean(axis=0, keepdims=True)
            cc = corr_matrix(lhs, mean_target, corr_type)
            sample_best_values["max_any_human__mean_correct_humans"].append(float(cc[0, 0]))
            sample_best_skills["max_any_human__mean_correct_humans"].append(skills[lhs_idx])

        incorr_idx = [i for i, (_, _, c) in enumerate(native) if c is False and i != lhs_idx]
        if incorr_idx:
            mean_target = mats[incorr_idx].mean(axis=0, keepdims=True)
            cc = corr_matrix(lhs, mean_target, corr_type)
            sample_best_values["max_any_human__mean_incorrect_humans"].append(float(cc[0, 0]))
            sample_best_skills["max_any_human__mean_incorrect_humans"].append(skills[lhs_idx])

    for key in out:
        vals = sample_best_values[key]
        if not vals:
            continue
        best_idx = int(np.nanargmax(vals))
        out[key] = (vals[best_idx], sample_best_skills[key][best_idx])
    return out


def build_worker_maps_full(worker_records, dataset_name, img_h, img_w):
    sigma = WORKER_SIGMA_PX[dataset_name]
    native = []
    for wr in worker_records:
        grid = gaussian_clicks_native(wr.fixations, img_h, img_w, sigma)
        native.append((wr.worker_id, grid, wr.is_correct))
    return native


def compute_model_accuracy(dataset: str, model_key: str, sample_ids: set[str], sal_gt_lookup=None):
    corr_file = choose_correctness_file(dataset, model_key)
    if corr_file is None:
        return {
            "correctness_file": None,
            "mean_accuracy": np.nan,
            "majority_accuracy": np.nan,
            "strict_accuracy": np.nan,
            "any_correct_accuracy": np.nan,
        }
    with open(corr_file) as f:
        data = json.load(f)
    rows = []
    for row in data.get("results", []):
        sid = row["sample_id"]
        if sid not in sample_ids:
            continue
        responses = row.get("responses", [])
        if dataset == "salchartqa":
            gt = sal_gt_lookup.get(sid)
            if not gt:
                continue
            correct = [relaxed_match(resp, gt) for resp in responses]
        else:
            correct = row.get("correct", [])
        if not correct:
            continue
        frac = float(np.mean(correct))
        rows.append(
            {
                "mean_accuracy": frac,
                "majority_accuracy": float(frac > 0.5),
                "strict_accuracy": float(frac == 1.0),
                "any_correct_accuracy": float(frac > 0.0),
            }
        )
    if not rows:
        return {
            "correctness_file": str(corr_file),
            "mean_accuracy": np.nan,
            "majority_accuracy": np.nan,
            "strict_accuracy": np.nan,
            "any_correct_accuracy": np.nan,
        }
    df = pd.DataFrame(rows)
    return {
        "correctness_file": str(corr_file),
        "mean_accuracy": float(df["mean_accuracy"].mean()),
        "majority_accuracy": float(df["majority_accuracy"].mean()),
        "strict_accuracy": float(df["strict_accuracy"].mean()),
        "any_correct_accuracy": float(df["any_correct_accuracy"].mean()),
    }


def build_sal_sample_gt_lookup(samples, sal_data):
    out = {}
    for sample in samples:
        out[sample["sample_id"]] = sal_data["gt_answer_lookup"].get(sample["gt_key"])
    return out


def main():
    image_size_cache = {}

    taskvis_data = load_taskvis()
    sal_data = load_salchartqa()
    sal_sample_gt = build_sal_sample_gt_lookup(sal_data["samples"], sal_data)

    datasets = {
        "taskvis": taskvis_data,
        "salchartqa": sal_data,
    }

    coverage_rows = []
    summary_rows = []
    sample_rows = []
    worker_rows = []

    for dataset_name, dataset_data in datasets.items():
        samples = dataset_data["samples"]
        sample_ids = {s["sample_id"] for s in samples}
        print(f"\nDataset {dataset_name}: {len(samples)} samples")

        for model_key, model_cfg in MODEL_INFO.items():
            attn_dir = ATTN_BASE / dataset_name / "image_suffix_question" / model_key
            npz_count = len(list(attn_dir.glob("*.npz"))) if attn_dir.exists() else 0
            expected = len(samples)
            missing_raw = npz_count < expected
            coverage_rows.append(
                {
                    "dataset": dataset_name,
                    "model_key": model_key,
                    "model_name": model_cfg["model_name"],
                    "hf_model_id": model_cfg["hf_model_id"],
                    "attention_dir": str(attn_dir),
                    "npz_count": npz_count,
                    "expected_samples": expected,
                    "missing_raw_attention": missing_raw,
                }
            )

        # Human baselines per dataset
        human_values = {
            "pearson": {
                "max_any_human__max_any_human": [],
                "max_any_human__mean_all_humans": [],
                "max_any_human__mean_correct_humans": [],
                "max_any_human__mean_incorrect_humans": [],
            },
            "spearman": {
                "max_any_human__max_any_human": [],
                "max_any_human__mean_all_humans": [],
                "max_any_human__mean_correct_humans": [],
                "max_any_human__mean_incorrect_humans": [],
            },
        }
        human_skills = {
            corr_type: {metric: [] for metric in human_values[corr_type]}
            for corr_type in human_values
        }

        for sample in samples:
            image_path = dataset_data["image_dir"] / sample["image_name"]
            img_h, img_w = get_image_size(image_path, image_size_cache)
            workers = (
                get_taskvis_workers(sample, taskvis_data)
                if dataset_name == "taskvis"
                else get_salchartqa_workers(sample, sal_data)
            )
            workers = [w for w in workers if w.is_correct is not None]
            if len(workers) < 2:
                continue
            native_maps = build_worker_maps_full(workers, dataset_name, img_h, img_w)
            for corr_type in ["pearson", "spearman"]:
                hb = compute_human_baselines(native_maps, dataset_data["worker_skill"], corr_type)
                for metric_key, (value, skill) in hb.items():
                    if not np.isnan(value):
                        human_values[corr_type][metric_key].append(value)
                        human_skills[corr_type][metric_key].append(skill)
                        sample_rows.append(
                            {
                                "dataset": dataset_name,
                                "row_type": "human_baseline",
                                "entity_key": "human_baseline",
                                "entity_name": "Human Baseline",
                                "hf_model_id": None,
                                "corr_type": corr_type,
                                "metric": metric_key,
                                "sample_id": sample["sample_id"],
                                "value": value,
                                "selected_lhs_skill": skill,
                            }
                        )
                # Save full per-worker leave-one-out values as raw data.
                worker_ids = [wid for wid, _, _ in native_maps]
                mats = np.array([g.ravel() for _, g, _ in native_maps], dtype=np.float64)
                for lhs_idx, (wid, _, is_correct) in enumerate(native_maps):
                    lhs = mats[lhs_idx:lhs_idx + 1]
                    lhs_skill = dataset_data["worker_skill"].get(wid, np.nan)
                    other_idx = [i for i in range(len(native_maps)) if i != lhs_idx]
                    if other_idx:
                        cc = corr_matrix(lhs, mats[other_idx], corr_type)
                        worker_rows.append(
                            {
                                "dataset": dataset_name,
                                "corr_type": corr_type,
                                "sample_id": sample["sample_id"],
                                "lhs_worker_id": wid,
                                "lhs_is_correct": is_correct,
                                "lhs_skill": lhs_skill,
                                "metric": "max_any_human__max_any_human",
                                "value": float(np.nanmax(cc)),
                            }
                        )
                        mean_target = mats[other_idx].mean(axis=0, keepdims=True)
                        cc = corr_matrix(lhs, mean_target, corr_type)
                        worker_rows.append(
                            {
                                "dataset": dataset_name,
                                "corr_type": corr_type,
                                "sample_id": sample["sample_id"],
                                "lhs_worker_id": wid,
                                "lhs_is_correct": is_correct,
                                "lhs_skill": lhs_skill,
                                "metric": "max_any_human__mean_all_humans",
                                "value": float(cc[0, 0]),
                            }
                        )
                    corr_idx = [i for i, (_, _, c) in enumerate(native_maps) if c is True and i != lhs_idx]
                    if corr_idx:
                        cc = corr_matrix(lhs, mats[corr_idx].mean(axis=0, keepdims=True), corr_type)
                        worker_rows.append(
                            {
                                "dataset": dataset_name,
                                "corr_type": corr_type,
                                "sample_id": sample["sample_id"],
                                "lhs_worker_id": wid,
                                "lhs_is_correct": is_correct,
                                "lhs_skill": lhs_skill,
                                "metric": "max_any_human__mean_correct_humans",
                                "value": float(cc[0, 0]),
                            }
                        )
                    incorr_idx = [i for i, (_, _, c) in enumerate(native_maps) if c is False and i != lhs_idx]
                    if incorr_idx:
                        cc = corr_matrix(lhs, mats[incorr_idx].mean(axis=0, keepdims=True), corr_type)
                        worker_rows.append(
                            {
                                "dataset": dataset_name,
                                "corr_type": corr_type,
                                "sample_id": sample["sample_id"],
                                "lhs_worker_id": wid,
                                "lhs_is_correct": is_correct,
                                "lhs_skill": lhs_skill,
                                "metric": "max_any_human__mean_incorrect_humans",
                                "value": float(cc[0, 0]),
                            }
                        )

        for corr_type, metrics in human_values.items():
            for metric_key, vals in metrics.items():
                row = {
                    "dataset": dataset_name,
                    "row_type": "human_baseline",
                    "entity_key": "human_baseline",
                    "entity_name": "Human Baseline",
                    "hf_model_id": None,
                    "corr_type": corr_type,
                    "metric": metric_key,
                    "attention_sigma": ATTN_SIGMA,
                    "human_comparison_resolution": "native",
                    "correctness_standard": "taskvis_answers" if dataset_name == "taskvis" else "salchartqa_determined_ground_truth_fixed_usable_subset",
                }
                row.update({f"value_{k}": v for k, v in summarize(vals).items()})
                row.update({f"lhs_skill_{k}": v for k, v in summarize(human_skills[corr_type][metric_key]).items()})
                summary_rows.append(row)

        # Model rows per dataset
        for model_key, model_cfg in MODEL_INFO.items():
            attn_dir = ATTN_BASE / dataset_name / "image_suffix_question" / model_key
            if not attn_dir.exists():
                continue
            acc = compute_model_accuracy(
                dataset_name,
                model_key,
                sample_ids,
                sal_gt_lookup=sal_sample_gt if dataset_name == "salchartqa" else None,
            )
            model_values = {
                "pearson": {
                    "max_any_head__max_any_human": [],
                    "max_any_head__mean_all_humans": [],
                    "max_any_head__mean_correct_humans": [],
                    "max_any_head__mean_incorrect_humans": [],
                },
                "spearman": {
                    "max_any_head__max_any_human": [],
                    "max_any_head__mean_all_humans": [],
                    "max_any_head__mean_correct_humans": [],
                    "max_any_head__mean_incorrect_humans": [],
                },
            }

            print(f"  Processing {dataset_name} / {model_key}")
            for sample in samples:
                image_path = dataset_data["image_dir"] / sample["image_name"]
                img_h, img_w = get_image_size(image_path, image_size_cache)
                workers = (
                    get_taskvis_workers(sample, taskvis_data)
                    if dataset_name == "taskvis"
                    else get_salchartqa_workers(sample, sal_data)
                )
                workers = [w for w in workers if w.is_correct is not None]
                if len(workers) < 2:
                    continue
                native_maps = build_worker_maps_full(workers, dataset_name, img_h, img_w)

                if dataset_name == "taskvis":
                    npz_path = attn_dir / f"taskvis_{sample['image_id']}_{sample['task_type']}.npz"
                else:
                    npz_path = attn_dir / f"salchartqa_{sample['idx']}.npz"
                if not npz_path.exists():
                    continue

                loaded = load_attention_grid(npz_path, img_h, img_w, model_cfg)
                if loaded is None:
                    continue
                attn_2d, gh, gw = loaded
                for corr_type in ["pearson", "spearman"]:
                    metrics = compute_model_metrics(attn_2d, native_maps, corr_type)
                    mapped = {
                        "max_any_head__max_any_human": metrics["max_any_human"],
                        "max_any_head__mean_all_humans": metrics["mean_all_humans"],
                        "max_any_head__mean_correct_humans": metrics["mean_correct_humans"],
                        "max_any_head__mean_incorrect_humans": metrics["mean_incorrect_humans"],
                    }
                    for metric_key, value in mapped.items():
                        if not np.isnan(value):
                            model_values[corr_type][metric_key].append(value)
                            sample_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "row_type": "model",
                                    "entity_key": model_key,
                                    "entity_name": model_cfg["model_name"],
                                    "hf_model_id": model_cfg["hf_model_id"],
                                    "corr_type": corr_type,
                                    "metric": metric_key,
                                    "sample_id": sample["sample_id"],
                                    "value": value,
                                    "selected_lhs_skill": np.nan,
                                }
                            )

            for corr_type, metrics in model_values.items():
                for metric_key, vals in metrics.items():
                    row = {
                        "dataset": dataset_name,
                        "row_type": "model",
                        "entity_key": model_key,
                        "entity_name": model_cfg["model_name"],
                        "hf_model_id": model_cfg["hf_model_id"],
                        "corr_type": corr_type,
                        "metric": metric_key,
                        "attention_sigma": ATTN_SIGMA,
                        "human_comparison_resolution": "model_grid",
                        "correctness_standard": "taskvis_answers" if dataset_name == "taskvis" else "salchartqa_determined_ground_truth_fixed_usable_subset",
                        "correctness_file": acc["correctness_file"],
                        "model_mean_accuracy": acc["mean_accuracy"],
                        "model_majority_accuracy": acc["majority_accuracy"],
                        "model_strict_accuracy": acc["strict_accuracy"],
                        "model_any_correct_accuracy": acc["any_correct_accuracy"],
                    }
                    row.update({f"value_{k}": v for k, v in summarize(vals).items()})
                    summary_rows.append(row)

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["dataset", "model_key"])
    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "row_type", "entity_key", "corr_type", "metric"])
    sample_df = pd.DataFrame(sample_rows).sort_values(["dataset", "row_type", "entity_key", "corr_type", "metric", "sample_id"])
    worker_df = pd.DataFrame(worker_rows).sort_values(["dataset", "corr_type", "sample_id", "lhs_worker_id", "metric"])

    coverage_path = OUT_DIR / "coverage.csv"
    summary_path = OUT_DIR / "summary.csv"
    sample_path = OUT_DIR / "per_sample.csv"
    worker_path = OUT_DIR / "human_worker_raw.csv"
    manifest_path = OUT_DIR / "manifest.json"
    coverage_df.to_csv(coverage_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    worker_df.to_csv(worker_path, index=False)

    manifest = {
        "output_dir": str(OUT_DIR),
        "procedure": {
            "attention_smoothing_sigma_token_units": ATTN_SIGMA,
            "human_sigma_px": WORKER_SIGMA_PX,
            "model_human_comparison": "full-resolution human fixation-density map, bilinear downsampled to each model/example token grid",
            "human_human_comparison": "native image resolution",
            "correlation_types": ["pearson", "spearman"],
            "salchartqa_correctness_standard": "determined_ground_truth_fixed usable subset",
            "taskvis_correctness_standard": "answers.txt",
        },
        "inputs": {
            "attention_base": str(ATTN_BASE),
            "correctness_dir": str(CORRECTNESS_DIR),
            "salchartqa_dir": str(SALCHARTQA_DIR),
            "taskvis_dir": str(TASKVIS_DIR),
        },
        "files": {
            "coverage_csv": str(coverage_path),
            "summary_csv": str(summary_path),
            "per_sample_csv": str(sample_path),
            "human_worker_raw_csv": str(worker_path),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    missing = coverage_df[coverage_df["missing_raw_attention"]]
    print("\nSaved:")
    print(coverage_path)
    print(summary_path)
    print(sample_path)
    print(worker_path)
    print(manifest_path)
    if len(missing):
        print("\nModels with missing raw attention:")
        print(missing[["dataset", "model_key", "npz_count", "expected_samples"]].to_string(index=False))
    else:
        print("\nNo missing raw attention among the requested instruct-model set.")


if __name__ == "__main__":
    main()
