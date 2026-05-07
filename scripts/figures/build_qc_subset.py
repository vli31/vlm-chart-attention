#!/usr/bin/env python3
"""Build broader QC list: samples where ≥3 of 5 sources agree on the GT
(ChartQA original + SalChartQA worker plurality + 3 runs of gpt-5-mini).
Writes final_paper_figures/broad_qc_ids.json with a list of sample_idx.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, "./")
from correctness import relaxed_match  # noqa: E402

ROOT = Path("./")
GPT5 = ROOT / "gpt-5-mini-2025-08-07_merged_all.json"
SC_Q = Path("./data/lvlm-chart/SalChartQA/image_questions.json")
OUT  = ROOT / "final_paper_figures" / "broad_qc_ids.json"

with open(GPT5) as f:
    gpt5 = json.load(f)
with open(SC_Q) as f:
    img_qs = json.load(f)


def to_str(x):
    if x is None: return ""
    if isinstance(x, list): return ", ".join(str(i) for i in x)
    return str(x)


_UNANSWERABLE_CATS = {"unanswerable", "no_answer", "no_data",
                      "cannot_answer", "unable_to_answer"}
_AMBIGUOUS_CATS    = {"ambiguous"}


def _flag_category(f):
    if not f: return None
    if isinstance(f, dict): return f.get("category")
    if isinstance(f, str):
        try:
            d = eval(f, {})
            return d.get("category") if isinstance(d, dict) else None
        except Exception: return None
    return None


def gpt5_flag_status(entry):
    """Return (any_unanswerable, any_ambiguous) across the 3 gpt5 runs."""
    cats = [_flag_category(entry.get(f"run{i}_flag_error")) for i in range(1, 4)]
    cats = [c for c in cats if c]
    return (any(c in _UNANSWERABLE_CATS for c in cats),
            any(c in _AMBIGUOUS_CATS for c in cats))


def majority_agree(entry):
    """Return (gt, group_size) where group_size is the size of the largest
    agreeing group across {chartqa_gt, worker_gt, gpt5 runs 1–3}."""
    answers = [
        to_str(entry.get("chartqa_ground_truth", "")),
        to_str(entry.get("worker_ground_truth", "")),
        to_str(entry.get("run1_answer", "")),
        to_str(entry.get("run2_answer", "")),
        to_str(entry.get("run3_answer", "")),
    ]
    groups = []
    for i, ans in enumerate(answers):
        if not ans: continue
        matched = False
        for g in groups:
            if relaxed_match(ans, answers[g[0]]):
                g.append(i); matched = True; break
        if not matched:
            groups.append([i])
    if not groups:
        return "", 0
    largest = max(groups, key=len)
    return answers[largest[0]], len(largest)


# Build (image_name, question_id) → sample_idx using same iteration as the
# correlation pipeline (sorted Q-ids within dict-order images).
sample_lookup = {}
sample_idx = 0
for image_name, qs in img_qs.items():
    for q in sorted(qs.keys()):
        sample_lookup[(image_name, q)] = sample_idx
        sample_idx += 1
total_samples = sample_idx
print(f"sample_idx universe: {total_samples}")

THRESHOLDS = [5, 4, 3]
counts = {t: 0 for t in THRESHOLDS}
ids_at = {t: [] for t in THRESHOLDS}
ids_strict_3 = []   # ≥3 agree AND no run flagged unanswerable AND no ambiguous
n_no_gt = 0
n_no_idx = 0
n_dropped_unanswerable = 0
n_dropped_ambiguous = 0
for key, entry in gpt5.items():
    img = entry.get("image_name")
    qid = entry.get("question_id")
    sid = sample_lookup.get((img, qid))
    if sid is None:
        n_no_idx += 1; continue
    gt, k = majority_agree(entry)
    if not gt:
        n_no_gt += 1; continue
    any_unans, any_ambig = gpt5_flag_status(entry)
    for t in THRESHOLDS:
        if k >= t:
            counts[t] += 1
            ids_at[t].append(sid)
    if k >= 3:
        if any_unans:
            n_dropped_unanswerable += 1
        elif any_ambig:
            n_dropped_ambiguous += 1
        else:
            ids_strict_3.append(sid)

print(f"  GPT5 entries with no GT determinable: {n_no_gt}")
print(f"  GPT5 entries whose (img,qid) not in image_questions: {n_no_idx}")
for t in THRESHOLDS:
    print(f"  agreement ≥{t}-of-5: {counts[t]} samples")
print(f"  ≥3-agree AND no unanswerable AND no ambiguous: {len(ids_strict_3)} samples")
print(f"  dropped because any run flagged unanswerable: {n_dropped_unanswerable}")
print(f"  dropped because any run flagged ambiguous:    {n_dropped_ambiguous}")

# Also load the cached original in_qc_subset (4556 samples) and use that
# as the active QC list so the existing figure scripts pick it up.
DETERMINED_GT = Path(
    "./data/lvlm-chart"
    "/correctness/determined_ground_truth_fixed.json"
)
det = json.load(open(DETERMINED_GT))["gt_lookup"]
ids_in_qc_subset = []
for key, val in det.items():
    if val.get("in_confident_subset") and not val.get("is_questionable", False):
        img = val["image_name"]; qid = key.replace(f"{img}_", "")
        sid = sample_lookup.get((img, qid))
        if sid is not None:
            ids_in_qc_subset.append(sid)
ids_in_qc_subset = sorted(ids_in_qc_subset)
print(f"  cached in_qc_subset (active QC for figures):    {len(ids_in_qc_subset)} samples")

with open(OUT, "w") as f:
    json.dump({
        "total_samples": total_samples,
        "n_at_5_agree": counts[5],
        "n_at_4_agree": counts[4],
        "n_at_3_agree": counts[3],
        "n_at_3_agree_strict": len(ids_strict_3),
        # active list used by fig scripts → original in_qc_subset (4556)
        "ids_at_3_agree": ids_in_qc_subset,
        # alternates kept for reference
        "ids_in_qc_subset": ids_in_qc_subset,
        "ids_at_3_agree_strict_broad": sorted(ids_strict_3),  # 5427
        "ids_at_3_agree_loose": sorted(ids_at[3]),            # 5732
    }, f, indent=2)
print(f"saved {OUT}")
