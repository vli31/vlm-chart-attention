#!/usr/bin/env python3
"""Paper figures: head-equalized alignment vs accuracy for all 5 saliency
metrics (CC, SIM, NSS, AUC, KL).

Produces four files in final_paper_figures/fig3_other_metrics/:
  salchartqa_renorm16.{png,pdf}   salchartqa_native.{png,pdf}
  taskvis_renorm16.{png,pdf}      taskvis_native.{png,pdf}

Each figure has one row of five panels (CC, SIM, NSS, AUC, KL) sharing one
legend on the right.

Conventions
-----------
* Per-sample max-head metric arrays come from the consistent renorm
  pipeline (load → resize to 16x16 → smooth attention at sigma=1 in
  16x16-grid units → compute metric vs gaze rasterized at 16x16):
    per_sample_metrics_5way/{model}_renorm16_salchartqa.npz
    per_sample_metrics_5way/{model}_renorm16_taskvis.npz
* QC subset for SalChartQA = ids_in_qc_subset (n=4556).
* Head-equalized 300-head bootstrap (1000 reps) on x-axis.
* Question-set bootstrap (2000 reps) on y-axis (model accuracy).
* Joint bootstrap (10000 reps) for the Pearson r CI; permutation test
  (10000 reps) for the p-value.
* Human point: matches the source/mechanism used by fig3 — per-sample
  mean-LOO of the *same metric*, paired with per-sample human accuracy,
  then bootstrap-resampled over samples (2000 reps). Files:
    human_human_ceiling/human_loo_per_worker_5metric.csv
    human_human_ceiling/taskvis_loo_per_worker_5metric.csv
* KL is NOT transformed: lower = better, so a "good" model sits to the
  LEFT of the panel and the regression slope is typically negative.
* The KL convention in the codebase is KL(gaze ‖ head): gaze is the GT
  distribution, head is the predicted; missing fixated regions hurt the
  most.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path("./")
DATA_5WAY = ROOT / "per_sample_metrics_5way"
HUMAN_LOO_SC = ROOT / "human_human_ceiling" / "human_loo_per_worker_5metric.csv"
HUMAN_LOO_TV = ROOT / "human_human_ceiling" / "taskvis_loo_per_worker_5metric.csv"
QC_PATH = ROOT / "final_paper_figures" / "broad_qc_ids.json"
SC_UNIFIED = ROOT / "downloaded_data/salchartqa/SalChartQA/unified_approved.csv"
TV_GT = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis/taskvis_ground_truth.json")
CORRECTNESS_DIR = Path("./data/lvlm-chart/correctness")
OUT_DIR = ROOT / "final_paper_figures" / "fig3_other_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DRAW = 300
N_BOOT_HEAD = 1000
N_BOOT_Q = 2000
N_BOOT_JOINT = 10000
N_PERM = 10000
SEED = 42

METRICS = ["cc", "sim", "nss", "auc", "kl"]
METRIC_DISPLAY = {
    "cc":  "Max-Head CC",
    "sim": "Max-Head SIM",
    "nss": "Max-Head NSS",
    "auc": "Max-Head AUC-Judd",
    "kl":  "Max-Head KL(gaze‖head)",
}
# Higher = better for cc/sim/nss/auc; lower = better for kl
HIGHER_BETTER = {"cc": True, "sim": True, "nss": True, "auc": True, "kl": False}

MODEL_ORDER = [
    "2.5-3B", "2.5-7B", "2B", "4B", "8B",
    "internvl3-1b", "internvl3-2b", "internvl3-8b",
    "internvl3.5-1b", "internvl3.5-2b", "internvl3.5-4b", "internvl3.5-8b",
]
MODEL_LABELS = {
    "2.5-3B": "Q2.5-3B", "2.5-7B": "Q2.5-7B",
    "2B": "Q3-2B", "4B": "Q3-4B", "8B": "Q3-8B",
    "internvl3-1b": "IV3-1B", "internvl3-2b": "IV3-2B", "internvl3-8b": "IV3-8B",
    "internvl3.5-1b": "IV3.5-1B", "internvl3.5-2b": "IV3.5-2B",
    "internvl3.5-4b": "IV3.5-4B", "internvl3.5-8b": "IV3.5-8B",
}
MODEL_PARAMS_B = {
    "2.5-3B": 3, "2.5-7B": 7, "2B": 2, "4B": 4, "8B": 8,
    "internvl3-1b": 1, "internvl3-2b": 2, "internvl3-8b": 8,
    "internvl3.5-1b": 1, "internvl3.5-2b": 2,
    "internvl3.5-4b": 4, "internvl3.5-8b": 8,
}
FAMILY_COLORS = {
    "Qwen2.5-VL":  "#BB52A6",
    "Qwen3-VL":    "#3540A8",
    "InternVL3":   "#D17B30",
    "InternVL3.5": "#A23E1A",
}
FAMILY_MARKERS = {
    "Qwen2.5-VL": "o", "Qwen3-VL": "o",
    "InternVL3": "D", "InternVL3.5": "D",
}


def family(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"


def marker_size(params_b):
    return 55.0 * float(params_b)


# correctness filenames use these keys for SalChartQA / TaskVis
CORR_KEY = {m: m for m in MODEL_ORDER}
# For correctness JSONs, IV3 models use *-instruct in filename
CORRECTNESS_NAME_TV = {
    "internvl3-1b":   "internvl3-1b-instruct",
    "internvl3-2b":   "internvl3-2b-instruct",
    "internvl3-8b":   "internvl3-8b-instruct",
    "internvl3.5-1b": "internvl3.5-1b-instruct",
    "internvl3.5-2b": "internvl3.5-2b-instruct",
    "internvl3.5-4b": "internvl3.5-4b-instruct",
    "internvl3.5-8b": "internvl3.5-8b-instruct",
}


def correctness_path(model_key, dataset):
    if dataset == "salchartqa":
        ck = model_key
        if model_key.startswith("internvl3"):  # legacy: filename uses bare lc
            ck = model_key
        cands = sorted(CORRECTNESS_DIR.glob(
            f"correctness_{ck.lower()}_salchartqa_x5_n5999_*_fixed.json"))
    else:
        ck = CORRECTNESS_NAME_TV.get(model_key, model_key)
        cands = sorted(CORRECTNESS_DIR.glob(
            f"correctness_{ck.lower()}_taskvis_x5_n90_*.json"))
        if not cands:
            cands = sorted(CORRECTNESS_DIR.glob(
                f"correctness_{model_key.lower()}_taskvis_x5_n90_*.json"))
    return cands[-1] if cands else None


def per_sample_correct(model_key, dataset):
    """Return arrays length = total samples in correctness file:
    correct[i] = bool (true if num_correct > 3); sample_id[i] string.
    """
    p = correctness_path(model_key, dataset)
    if p is None:
        return None, None
    with open(p) as f:
        data = json.load(f)
    n = len(data["results"])
    correct = np.zeros(n, dtype=bool)
    sids = []
    for i, r in enumerate(data["results"]):
        correct[i] = sum(1 for c in r.get("correct", []) if c) > 3
        sids.append(r.get("sample_id", str(i)))
    return correct, np.array(sids, dtype=object)


def head_bootstrap_max(arr, n_draw, n_boot, rng):
    """Mean over samples of max(metric over n_draw heads), n_boot reps.
    Treats NaN heads as -inf so they never win the max."""
    arr_inf = np.where(np.isnan(arr), -np.inf, arr)
    n_s, n_total = arr.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_total, size=n_draw)
        mx = np.max(arr_inf[:, idx], axis=1)
        valid = np.isfinite(mx)
        out[b] = float(mx[valid].mean()) if valid.any() else np.nan
    return out


def head_bootstrap_min(arr, n_draw, n_boot, rng):
    """Lower-is-better version: mean over samples of MIN over n_draw heads."""
    arr_inf = np.where(np.isnan(arr), np.inf, arr)
    n_s, n_total = arr.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_total, size=n_draw)
        mn = np.min(arr_inf[:, idx], axis=1)
        valid = np.isfinite(mn)
        out[b] = float(mn[valid].mean()) if valid.any() else np.nan
    return out


def question_bootstrap_acc(correct_arr, n_boot, rng):
    n = len(correct_arr)
    arr = correct_arr.astype(np.float32)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = float(arr[idx].mean())
    return out


def joint_bootstrap_r_dist(rows, n_boot, rng):
    M = len(rows)
    x_b = np.stack([r["x_boot"] for r in rows], axis=0)
    y_b = np.stack([r["y_boot"] for r in rows], axis=0)
    Nx, Ny = x_b.shape[1], y_b.shape[1]
    rs = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, Nx, size=M)
        iy = rng.integers(0, Ny, size=M)
        x = x_b[np.arange(M), ix]
        y = y_b[np.arange(M), iy]
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        rs[b] = (x * y).sum() / d if d > 0 else np.nan
    return rs[np.isfinite(rs)]


def permutation_pvalue_pearsonr(xs, ys, n_perm, rng):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xc = xs - xs.mean(); yc = ys - ys.mean()
    denom_x = np.sqrt((xc * xc).sum())
    r_obs = (xc * yc).sum() / (denom_x * np.sqrt((yc * yc).sum()))
    r_abs = abs(r_obs)
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(yc)
        r_p = (xc * yp).sum() / (denom_x * np.sqrt((yp * yp).sum()))
        if abs(r_p) >= r_abs:
            hits += 1
    return float(r_obs), (1 + hits) / (1 + n_perm)


# ── Per-sample human accuracy for the human point ─────────────────
def load_sc_human_acc():
    """Returns {(image_name, question_text): mean is_correct}"""
    uni = pd.read_csv(SC_UNIFIED)
    return uni.groupby(["image_name", "question"])["is_correct"].mean().to_dict()


def load_tv_human_acc():
    """Returns {sample_id: human_accuracy.accuracy}."""
    gt = json.loads(TV_GT.read_text())
    return {s["sample_id"]: s["human_accuracy"]["accuracy"] for s in gt["samples"]}


def human_xy_with_ci(arr, n_boot, seed_offset):
    """arr: (N, 2) of [x, y] per sample; returns mean + percentile CIs from
    paired sample-resampling."""
    rng = np.random.default_rng(SEED + seed_offset)
    n = len(arr)
    x_b = np.empty(n_boot); y_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_b[b] = arr[idx, 0].mean()
        y_b[b] = arr[idx, 1].mean()
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()),
            float(np.percentile(x_b, 2.5)), float(np.percentile(x_b, 97.5)),
            float(np.percentile(y_b, 2.5)), float(np.percentile(y_b, 97.5)),
            x_b.astype(np.float64), y_b.astype(np.float64))


def get_human_point(dataset, metric):
    """Return (x_mean, y_mean, x_lo, x_hi, y_lo, y_hi, x_boot, y_boot) using
    the same per-sample LOO + per-sample human-acc paired-bootstrap mechanism
    as fig3, but for *this* metric. y is per-sample human accuracy."""
    col = f"mean_loo_{metric}"
    if dataset == "salchartqa":
        loo = pd.read_csv(HUMAN_LOO_SC)
        loo = loo[["image_name", "question_id", col]].rename(columns={col: "loo"})
        # Map (image_name, question_id) → human accuracy via question text
        acc_lookup = load_sc_human_acc()
        # We need a (image_name, question_id) → question text mapping
        cat = pd.read_csv(ROOT / "salchartqa_question_categories.csv")
        qtxt = {(r["image_name"], r["question_id"]): r["question"]
                for _, r in cat.iterrows()}
        loo["question_text"] = loo.apply(
            lambda r: qtxt.get((r["image_name"], r["question_id"]), None),
            axis=1)
        loo["human_acc"] = loo.apply(
            lambda r: float(acc_lookup.get((r["image_name"], r["question_text"]),
                                           np.nan))
            if r["question_text"] is not None else np.nan,
            axis=1)
        pair = loo[["loo", "human_acc"]].dropna().to_numpy()
    else:  # taskvis
        loo = pd.read_csv(HUMAN_LOO_TV)
        loo = loo[["sample_id", col]].rename(columns={col: "loo"})
        acc_lookup = load_tv_human_acc()
        loo["human_acc"] = loo["sample_id"].map(acc_lookup)
        pair = loo[["loo", "human_acc"]].dropna().to_numpy()
    return human_xy_with_ci(pair, N_BOOT_Q, seed_offset=hash(metric) % 100)


# ── Per-model bootstrap on the model side ─────────────────────────
def load_sc_model_data(model_key, resolution, metric, qc_ids_arr):
    """Load (max-head metric per sample, correctness per sample) on the QC
    subset. Returns (arr, correct, n_qc, n_heads) or None.
    """
    p = DATA_5WAY / f"{model_key}_{resolution}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    sids = d["sample_ids"].astype(np.int64)
    keep = np.isin(sids, qc_ids_arr) & (d["n_workers"] >= 3)
    arr = d[metric].astype(np.float64)[keep]
    sids_keep = sids[keep]
    # Per-sample correctness via sample_idx into the 5999-row JSON
    correct, _ = per_sample_correct(model_key, "salchartqa")
    if correct is None:
        return None
    corr_keep = correct[sids_keep]
    return arr, corr_keep, int(keep.sum()), int(arr.shape[1])


def load_tv_model_data(model_key, resolution, metric):
    """Load (max-head metric per sample, correctness per sample) on TaskVis."""
    p = DATA_5WAY / f"{model_key}_taskvis_{resolution}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    sids = np.asarray(d["sample_ids"], dtype=str)
    keep = d["n_workers"] >= 2
    arr = d[metric].astype(np.float64)[keep]
    sids_keep = sids[keep]
    correct, sid_corr = per_sample_correct(model_key, "taskvis")
    if correct is None:
        return None
    sid_to_correct = dict(zip(sid_corr, correct))
    corr_keep = np.array([sid_to_correct.get(s, False) for s in sids_keep],
                          dtype=bool)
    # drop samples not present in correctness file
    have = np.array([s in sid_to_correct for s in sids_keep])
    arr = arr[have]; corr_keep = corr_keep[have]
    return arr, corr_keep, int(have.sum()), int(arr.shape[1])


def build_one_figure(dataset, mode):
    """mode: 'renorm16' (everything resized to 16x16, attn σ=1 in 16x16 units)
    or 'native' (each sample at its native attention grid, attn σ=1 in native
    grid units). Both pipelines use chart-image-pixel canvas for gaze."""
    rng = np.random.default_rng(SEED)
    qc_ids_arr = np.array(
        [int(x) for x in json.loads(QC_PATH.read_text())["ids_in_qc_subset"]],
        dtype=np.int64)

    print(f"\n=== {dataset.upper()} ({mode}) ===", flush=True)

    rows_by_metric = {m: [] for m in METRICS}
    for mk in MODEL_ORDER:
        # Both pipelines use corrected chart-img canvas + sigma=1 attn smoothing
        p = DATA_5WAY / f"{mk}_{mode}_{dataset}.npz"
        if not p.exists():
            print(f"  skip {mk}: {p} missing")
            continue
        d = np.load(p)
        if dataset == "salchartqa":
            sids = d["sample_ids"].astype(np.int64)
            keep = np.isin(sids, qc_ids_arr) & (d["n_workers"] >= 3)
            sids_keep = sids[keep]
            correct, _ = per_sample_correct(mk, "salchartqa")
            if correct is None:
                print(f"  skip {mk}: no correctness")
                continue
            corr_keep = correct[sids_keep]
        else:
            sids = np.asarray(d["sample_ids"], dtype=str)
            keep = d["n_workers"] >= 2
            sids_keep = sids[keep]
            correct, sid_corr = per_sample_correct(mk, "taskvis")
            if correct is None:
                print(f"  skip {mk}: no correctness")
                continue
            sid2c = dict(zip(sid_corr, correct))
            have = np.array([s in sid2c for s in sids_keep])
            keep_idx = np.where(keep)[0][have]
            keep = np.zeros_like(keep); keep[keep_idx] = True
            sids_keep = sids[keep]
            corr_keep = np.array([sid2c[s] for s in sids_keep], dtype=bool)

        if int(keep.sum()) == 0:
            print(f"  skip {mk}: 0 valid samples")
            continue

        n_qc = int(keep.sum())
        # Question-set bootstrap on accuracy (shared across all 4 metrics)
        q_b = question_bootstrap_acc(corr_keep, N_BOOT_Q, rng)
        y_mean = float(corr_keep.mean())
        y_lo = float(np.percentile(q_b, 2.5))
        y_hi = float(np.percentile(q_b, 97.5))

        for metric in METRICS:
            arr = d[metric].astype(np.float64)[keep]
            n_heads = int(arr.shape[1])
            if HIGHER_BETTER[metric]:
                head_b = head_bootstrap_max(arr, N_DRAW, N_BOOT_HEAD, rng)
            else:
                head_b = head_bootstrap_min(arr, N_DRAW, N_BOOT_HEAD, rng)
            x_mean = float(np.nanmean(head_b))
            x_lo = float(np.nanpercentile(head_b, 2.5))
            x_hi = float(np.nanpercentile(head_b, 97.5))
            rows_by_metric[metric].append({
                "model": mk, "fam": family(mk),
                "params_b": MODEL_PARAMS_B[mk],
                "x_mean": x_mean, "x_lo": x_lo, "x_hi": x_hi,
                "x_boot": head_b.astype(np.float64),
                "y_mean": y_mean, "y_lo": y_lo, "y_hi": y_hi,
                "y_boot": q_b.astype(np.float64),
                "n_qc": n_qc, "n_heads": n_heads,
            })
        print(f"  {mk:18s} N={n_qc} acc={y_mean:.3f}", flush=True)

    # Human points (one per metric)
    human_pts = {}
    for metric in METRICS:
        try:
            (hx, hy, hx_lo, hx_hi, hy_lo, hy_hi, hx_b, hy_b) = get_human_point(
                dataset, metric)
        except Exception as e:
            print(f"  skip human point for {metric}: {e}")
            continue
        human_pts[metric] = {
            "x_mean": hx, "y_mean": hy,
            "x_lo": hx_lo, "x_hi": hx_hi, "y_lo": hy_lo, "y_hi": hy_hi,
            "x_boot": hx_b, "y_boot": hy_b,
        }
        print(f"  human {metric}: x={hx:.3f} y={hy:.3f}")

    # ── Plot: 1 row × 4 columns
    plt.rcParams.update({
        "font.size": 24, "axes.labelsize": 26, "axes.titlesize": 30,
        "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 24,
        "axes.linewidth": 1.4,
    })
    fig, axes = plt.subplots(1, 5, figsize=(34, 8))
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.16, top=0.84, wspace=0.34)

    # Shared y-limits across panels for direct accuracy comparison
    all_y = []
    for metric in METRICS:
        for r in rows_by_metric[metric]:
            all_y.extend([r["y_mean"], r["y_lo"], r["y_hi"]])
        if metric in human_pts:
            all_y.extend([human_pts[metric]["y_mean"],
                          human_pts[metric]["y_lo"],
                          human_pts[metric]["y_hi"]])
    if all_y:
        y_lo_g = min(all_y) - 0.05
        y_hi_g = max(all_y) + 0.05
    else:
        y_lo_g, y_hi_g = 0, 1

    r_summary = []
    for ax, metric in zip(axes, METRICS):
        rows = rows_by_metric[metric]
        if not rows:
            ax.set_visible(False)
            continue
        xs = np.array([r["x_mean"] for r in rows])
        ys = np.array([r["y_mean"] for r in rows])

        # Linear fit
        if len(xs) >= 3:
            z = np.polyfit(xs, ys, 1)
            x_lo_lim = min(xs.min(),
                           human_pts[metric]["x_mean"] if metric in human_pts else xs.min())
            x_hi_lim = max(xs.max(),
                           human_pts[metric]["x_mean"] if metric in human_pts else xs.max())
            xline = np.linspace(x_lo_lim - 0.04 * (x_hi_lim - x_lo_lim + 1e-6),
                                x_hi_lim + 0.04 * (x_hi_lim - x_lo_lim + 1e-6), 100)
            ax.plot(xline, np.polyval(z, xline),
                    color="#666", linestyle="--", lw=1.4, alpha=0.75, zorder=1)
            r_obs, p_perm = permutation_pvalue_pearsonr(xs, ys, N_PERM, rng)
            r_dist = joint_bootstrap_r_dist(rows, N_BOOT_JOINT, rng)
            r_lo = float(np.percentile(r_dist, 2.5))
            r_hi = float(np.percentile(r_dist, 97.5))
        else:
            r_obs = float("nan"); p_perm = float("nan")
            r_lo = r_hi = float("nan")

        r_summary.append({
            "dataset": dataset,
            "metric": metric, "n_models": len(xs),
            "r": r_obs, "perm_p": p_perm, "r_lo_95": r_lo, "r_hi_95": r_hi,
        })

        # Per-model points
        text_objs = []
        for rec in rows:
            fam = rec["fam"]
            ax.errorbar(rec["x_mean"], rec["y_mean"],
                        xerr=[[rec["x_mean"] - rec["x_lo"]],
                              [rec["x_hi"] - rec["x_mean"]]],
                        yerr=[[rec["y_mean"] - rec["y_lo"]],
                              [rec["y_hi"] - rec["y_mean"]]],
                        fmt="none", ecolor=FAMILY_COLORS[fam],
                        elinewidth=1.2, capsize=3, alpha=0.85, zorder=3)
            ax.scatter(rec["x_mean"], rec["y_mean"],
                       color=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam],
                       s=marker_size(rec["params_b"]),
                       edgecolor="black", linewidth=0.7,
                       alpha=0.85, zorder=5)
            # Param-count digit centered on the bubble (white text on
            # black-stroked outline for legibility on any bubble fill).
            # No adjust_text — the auto-arranger occasionally pushed a
            # label off the panel.
            label_fontsize = 11 if rec["params_b"] <= 2 else 14
            t = ax.text(rec["x_mean"], rec["y_mean"], f"{rec['params_b']}",
                        fontsize=label_fontsize, color="white",
                        fontweight="bold", zorder=7,
                        ha="center", va="center")
            t.set_path_effects([
                path_effects.withStroke(linewidth=1.6, foreground="black")])
            text_objs.append(t)

        # Human point
        if metric in human_pts:
            hp = human_pts[metric]
            ax.errorbar(hp["x_mean"], hp["y_mean"],
                        xerr=[[hp["x_mean"] - hp["x_lo"]],
                              [hp["x_hi"] - hp["x_mean"]]],
                        yerr=[[hp["y_mean"] - hp["y_lo"]],
                              [hp["y_hi"] - hp["y_mean"]]],
                        fmt="none", ecolor="black",
                        elinewidth=1.4, capsize=3, alpha=0.85, zorder=9)
            ax.scatter(hp["x_mean"], hp["y_mean"], marker="*",
                       s=marker_size(8) * 3.0, color="black",
                       edgecolor="white", linewidth=1.6, zorder=10)

        ax.set_xlabel(METRIC_DISPLAY[metric], fontsize=24)
        if metric == METRICS[0]:
            ax.set_ylabel("Accuracy", fontsize=24)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        all_x = list(xs)
        if metric in human_pts:
            all_x.append(human_pts[metric]["x_mean"])
        x_pad = 0.05 * (max(all_x) - min(all_x) + 1e-6)
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        ax.set_ylim(y_lo_g, y_hi_g)

        if not np.isnan(p_perm):
            if p_perm < 0.001:    p_str = "p < 0.001"
            elif p_perm < 0.01:   p_str = "p < 0.01"
            elif p_perm < 0.05:   p_str = "p < 0.05"
            else:                 p_str = f"p = {p_perm:.3f}"
            ax.text(0.96, 0.04 if HIGHER_BETTER[metric] else 0.96,
                    f"r = {r_obs:+.2f}\n{p_str}",
                    transform=ax.transAxes,
                    ha="right", va="bottom" if HIGHER_BETTER[metric] else "top",
                    fontsize=20, linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#999", alpha=0.92))

    # Legend
    legend_handles = []
    for mk in MODEL_ORDER:
        if not any(r["model"] == mk for m in METRICS for r in rows_by_metric[m]):
            continue
        fam = family(mk)
        legend_handles.append(Line2D([0], [0], marker=FAMILY_MARKERS[fam],
                                     color="w",
                                     markerfacecolor=FAMILY_COLORS[fam],
                                     markersize=np.sqrt(marker_size(MODEL_PARAMS_B[mk])),
                                     markeredgecolor="black",
                                     markeredgewidth=0.6,
                                     label=MODEL_LABELS[mk]))
    legend_handles.append(Line2D([0], [0], marker="*", color="w",
                                 markerfacecolor="black",
                                 markersize=np.sqrt(marker_size(8) * 3.0),
                                 markeredgecolor="white",
                                 markeredgewidth=1.4,
                                 label="Human"))
    fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.997, 0.5),
               title="Model", title_fontsize=26,
               fontsize=22, frameon=True, framealpha=0.95, edgecolor="#999",
               labelspacing=0.30, borderpad=0.30, handletextpad=0.5,
               borderaxespad=0.2)

    # No title — caption in the LaTeX appendix carries the dataset and mode.

    out_png = OUT_DIR / f"{dataset}_{mode}.png"
    out_pdf = OUT_DIR / f"{dataset}_{mode}.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")
    return r_summary


def main():
    all_summaries = []
    for dataset in ("salchartqa", "taskvis"):
        for mode in ("renorm16", "native"):
            s = build_one_figure(dataset, mode)
            for r in s:
                r["mode"] = mode
            all_summaries.extend(s)

    out = OUT_DIR / "v2_r_summary.json"
    with open(out, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nsaved {out}")
    print("\n=== r summary ===")
    for r in all_summaries:
        print(f"  {r['dataset']:11s} {r.get('mode', ''):8s} {r['metric']:4s}  "
              f"r={r['r']:+.3f}  p={r['perm_p']:.4f}  "
              f"95% CI [{r['r_lo_95']:+.3f}, {r['r_hi_95']:+.3f}]  "
              f"N={r['n_models']}")


if __name__ == "__main__":
    main()
