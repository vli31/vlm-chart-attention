#!/usr/bin/env python3
"""Paper figure: head-equalized VLM-human attention alignment vs model accuracy
on SalChartQA (left) and TaskVis (right). 12 VLMs per panel + human star.

Horizontal err: 95% CI from 300-head bootstrap, 1000 reps.
Vertical err:   95% CI from question-set bootstrap on accuracy, 2000 reps.
Alignment metric: max-head CC vs mean gaze, head-equalized.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
import matplotlib.patheffects as path_effects

ROOT = Path("./")
CORR_BASE = Path("./data/lvlm-chart/correlations")
CORRECTNESS_DIR = Path("./data/lvlm-chart/correctness")
OUT_DIR = ROOT / "final_paper_figures"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "fig3.png"
OUT_PDF = OUT_DIR / "fig3.pdf"

N_DRAW = 300
N_BOOT_HEAD = 1000
N_BOOT_Q = 2000
N_BOOT_JOINT = 10000  # for bootstrap CI on r (propagates per-model error bars)
N_PERM = 10000        # for permutation test of H0: ρ=0
SEED = 42
SIGMA = 1.0

MODEL_ORDER = [
    "2.5-3B", "2.5-7B", "2B", "4B", "8B",
    "internvl3-1b-instruct", "internvl3-2b-instruct", "internvl3-8b-instruct",
    "internvl3.5-1b-instruct", "internvl3.5-2b-instruct",
    "internvl3.5-4b-instruct", "internvl3.5-8b-instruct",
]
CORR_KEY_MAP = {
    "internvl3-1b-instruct": "internvl3-1b",
    "internvl3-2b-instruct": "internvl3-2b",
    "internvl3-8b-instruct": "internvl3-8b",
    "internvl3.5-1b-instruct": "internvl3.5-1b",
    "internvl3.5-2b-instruct": "internvl3.5-2b",
    "internvl3.5-4b-instruct": "internvl3.5-4b",
    "internvl3.5-8b-instruct": "internvl3.5-8b",
}
MODEL_LABELS = {
    "2.5-3B": "Q2.5-3B", "2.5-7B": "Q2.5-7B",
    "2B": "Q3-2B", "4B": "Q3-4B", "8B": "Q3-8B",
    "internvl3-1b-instruct": "IV3-1B", "internvl3-2b-instruct": "IV3-2B",
    "internvl3-8b-instruct": "IV3-8B",
    "internvl3.5-1b-instruct": "IV3.5-1B", "internvl3.5-2b-instruct": "IV3.5-2B",
    "internvl3.5-4b-instruct": "IV3.5-4B", "internvl3.5-8b-instruct": "IV3.5-8B",
}
MODEL_PARAMS_B = {
    "2.5-3B": 3, "2.5-7B": 7,
    "2B": 2, "4B": 4, "8B": 8,
    "internvl3-1b-instruct": 1, "internvl3-2b-instruct": 2,
    "internvl3-8b-instruct": 8,
    "internvl3.5-1b-instruct": 1, "internvl3.5-2b-instruct": 2,
    "internvl3.5-4b-instruct": 4, "internvl3.5-8b-instruct": 8,
}
# Same hue per vendor, different shade per version (Qwen → purples, Intern → oranges).
FAMILY_COLORS = {
    "Qwen2.5-VL":  "#BB52A6",   # magenta (pink-leaning purple, hue ~318)
    "Qwen3-VL":    "#3540A8",   # blue-purple indigo
    "InternVL3":   "#D17B30",   # mid burnt orange
    "InternVL3.5": "#A23E1A",   # deep burnt orange
}
FAMILY_MARKERS = {
    "Qwen2.5-VL": "o", "Qwen3-VL": "o",
    "InternVL3": "D", "InternVL3.5": "D",
}


def marker_size(params_b):
    """Marker area strictly proportional to parameter count (no offset),
    so 8B is exactly 8x the area of 1B."""
    return 90.0 * float(params_b)  # 1B → 90, 8B → 720


# Per-(dataset, model) label position overrides — only used when a bubble
# would otherwise be occluded by a near-identical neighbour. Values are
# (dx, dy) offsets in data coords from the bubble centre. A thin leader
# line is drawn from the offset label back to the bubble centre.
LABEL_OVERRIDES = {
    # IV3.5-4B and IV3.5-8B sit at nearly identical (x, y) on TaskVis;
    # the 4B's bubble (smaller) is hidden behind 8B (larger), so we
    # pull the "4" label out into empty whitespace to the upper-right
    # with a thin leader line back to the bubble centre.
    ("taskvis", "internvl3.5-4b-instruct"): (+0.045, +0.070),
}


def family(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"


def correctness_path(model_key, dataset):
    """Find latest correctness JSON for the (model, dataset).
    Prefer the -instruct correctness file when both exist, since the
    correlation/alignment files are computed from the -instruct model.
    """
    ck = CORR_KEY_MAP.get(model_key, model_key)
    if dataset == "salchartqa":
        # Try -instruct (full model_key) first, fall back to stripped ck.
        cands = sorted(CORRECTNESS_DIR.glob(
            f"correctness_{model_key.lower()}_salchartqa_x5_n5999_*_fixed.json"))
        if not cands:
            cands = sorted(CORRECTNESS_DIR.glob(
                f"correctness_{ck.lower()}_salchartqa_x5_n5999_*_fixed.json"))
    else:
        cands = sorted(CORRECTNESS_DIR.glob(
            f"correctness_{model_key.lower()}_taskvis_x5_n90_*.json"))
        if not cands:
            cands = sorted(CORRECTNESS_DIR.glob(
                f"correctness_{ck.lower()}_taskvis_x5_n90_*.json"))
    return cands[-1] if cands else None


# Broad QC for SalChartQA = ≥3-of-5 GT-source agreement (5732 of 5999 samples)
_BROAD_QC_PATH = Path("./final_paper_figures/broad_qc_ids.json")
if _BROAD_QC_PATH.exists():
    with open(_BROAD_QC_PATH) as _bf:
        _BROAD_QC_IDS = set(int(x) for x in json.load(_bf)["ids_at_3_agree"])
else:
    _BROAD_QC_IDS = None


def per_sample_correct(model_key, dataset):
    """Return (correct_bool_array_over_samples, qc_mask) where correct = (num_correct>3)."""
    p = correctness_path(model_key, dataset)
    if p is None:
        return None, None
    with open(p) as f:
        data = json.load(f)
    correct = []
    qc = []
    for sample_idx, r in enumerate(data.get("results", [])):
        if dataset == "salchartqa":
            if _BROAD_QC_IDS is not None:
                # broad QC: ≥3-of-5 GT agreement
                in_qc = sample_idx in _BROAD_QC_IDS
            else:
                in_qc = (r.get("in_confident_subset") and
                         not r.get("is_questionable", False))
            if not in_qc:
                qc.append(False)
                correct.append(False)
                continue
            qc.append(True)
            n = sum(1 for c in r.get("correct", []) if c)
        else:
            qc.append(True)
            n = int(r.get("num_correct", sum(1 for c in r.get("correct", []) if c)))
        correct.append(n > 3)
    return np.asarray(correct, dtype=bool), np.asarray(qc, dtype=bool)


def load_corr_array(model_key, dataset):
    """Returns cc (n_qc, L*H) at sigma=1.0, restricted to QC samples (and matched to correctness order)."""
    ck = CORR_KEY_MAP.get(model_key, model_key)
    p = CORR_BASE / dataset / f"{ck}_mean_gaze_correlations_sigma{SIGMA}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    cc = d["cc_mean_all_per_sample"]  # (S, L, H)
    n_s, n_l, n_h = cc.shape
    flat = cc.reshape(n_s, n_l * n_h).astype(np.float64)
    flat = np.where(np.isnan(flat), -np.inf, flat)
    return flat


def head_bootstrap_max(cc_flat, n_draw, n_boot, rng):
    """Mean over samples of max(CC over n_draw randomly-sampled heads), n_boot reps."""
    n_s, n_total = cc_flat.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_total, size=n_draw)
        mx = np.max(cc_flat[:, idx], axis=1)
        valid = np.isfinite(mx)
        out[b] = float(mx[valid].mean()) if valid.any() else np.nan
    return out


def question_bootstrap_acc(correct_arr, n_boot, rng):
    """Bootstrap mean over question set."""
    out = np.empty(n_boot)
    n = len(correct_arr)
    arr = correct_arr.astype(np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = float(arr[idx].mean())
    return out


def joint_bootstrap_r_dist(rows, n_boot, rng):
    """Propagate per-model x/y bootstrap uncertainty into r.
    Each replicate draws one x and one y per model from that model's bootstrap
    distributions, then computes Pearson r over the M models.
    Returns the bootstrap distribution of r (used for a CI on r)."""
    M = len(rows)
    x_b = np.stack([r["x_boot"] for r in rows], axis=0)  # (M, Nx)
    y_b = np.stack([r["y_boot"] for r in rows], axis=0)  # (M, Ny)
    Nx = x_b.shape[1]; Ny = y_b.shape[1]
    rs = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, Nx, size=M)
        iy = rng.integers(0, Ny, size=M)
        x = x_b[np.arange(M), ix]
        y = y_b[np.arange(M), iy]
        x = x - x.mean(); y = y - y.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        rs[b] = (x * y).sum() / denom if denom > 0 else np.nan
    return rs[np.isfinite(rs)]


def permutation_pvalue_pearsonr(xs, ys, n_perm, rng):
    """Two-sided permutation p for H0: x and y are independent.
    Permute y across the M models, recompute r, repeat n_perm times.
    p = (1 + #{|r_perm| >= |r_obs|}) / (1 + n_perm)  (add-one for unbiasedness)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    M = len(xs)
    xc = xs - xs.mean()
    yc = ys - ys.mean()
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


def human_accuracy_taskvis():
    """Mean across 90 samples of `human_accuracy` field in any taskvis correctness file."""
    cands = sorted(CORRECTNESS_DIR.glob("correctness_2.5-3b_taskvis_x5_n90_*.json"))
    if not cands:
        return None
    with open(cands[-1]) as f:
        data = json.load(f)
    vals = [r.get("human_accuracy") for r in data.get("results", [])
            if r.get("human_accuracy") is not None]
    return float(np.mean(vals)) if vals else None


def human_accuracy_salchartqa():
    """Mean is_correct from unified_approved.csv if available; else fall back to
    the value baked into bootstrap_5metric_2res_tests.json (~0.807)."""
    csv_path = ROOT / "downloaded_data/salchartqa/SalChartQA/unified_approved.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "is_correct" in df.columns:
            return float(df["is_correct"].mean())
    # fallback
    p = ROOT / "bootstrap_5metric_2res" / "bootstrap_5metric_2res_tests.json"
    if p.exists():
        with open(p) as f:
            return float(json.load(f).get("human_accuracy", np.nan))
    return None


def main():
    rng = np.random.default_rng(SEED)

    # Human baseline: per-sample mean LOO CC (each worker vs the mean of
    # OTHER workers' gaze maps, averaged over workers per sample). x = mean
    # over samples; y = mean over samples of per-sample human accuracy.
    # Both axes have bootstrap CIs computed by paired-resampling samples.
    import pandas as _pd
    h_xy = {}
    h_xy_boot = {}

    # ── SalChartQA: per-sample mean_loo + per-sample human accuracy ──
    sc_loo = _pd.read_csv(ROOT / "human_human_ceiling" / "human_loo_per_worker.csv")
    uni = _pd.read_csv(ROOT / "downloaded_data/salchartqa/SalChartQA/unified_approved.csv")
    iq = json.loads((ROOT / "downloaded_data/salchartqa/SalChartQA/image_questions.json").read_text())
    # Build (image_name, question_id) -> human_acc per sample
    qtext_lookup = {(img, qid): qt for img, qs in iq.items() for qid, qt in qs.items()}
    uni_acc = uni.groupby(["image_name", "question"])["is_correct"].mean()
    sc_loo["question_text"] = sc_loo.apply(
        lambda r: qtext_lookup.get((r["image_name"], r["question_id"]), None), axis=1)
    sc_loo["human_acc"] = sc_loo.apply(
        lambda r: float(uni_acc.get((r["image_name"], r["question_text"]), np.nan))
        if r["question_text"] is not None else np.nan, axis=1)
    sc_pair = sc_loo[["mean_loo", "human_acc"]].dropna().to_numpy()
    print(f"  SalChartQA human pairs: {len(sc_pair)} / {len(sc_loo)}")

    # ── TaskVis: per-sample mean_loo + per-sample human accuracy from GT ──
    tv_loo = _pd.read_csv(ROOT / "human_human_ceiling" / "taskvis_loo_per_worker.csv")
    tv_gt = json.loads((Path("./cache/lvlm-chart"
                             "/eye_gaze_datasets/taskvis/taskvis_ground_truth.json")).read_text())
    tv_acc_lookup = {s["sample_id"]: s["human_accuracy"]["accuracy"]
                     for s in tv_gt["samples"]}
    tv_loo["human_acc"] = tv_loo["sample_id"].map(tv_acc_lookup)
    tv_pair = tv_loo[["mean_loo", "human_acc"]].dropna().to_numpy()
    print(f"  TaskVis human pairs:    {len(tv_pair)} / {len(tv_loo)}")

    def human_xy_with_ci(arr, n_boot=N_BOOT_Q, seed=SEED):
        n = len(arr)
        rng_h = np.random.default_rng(seed + 7)
        x_b = np.empty(n_boot); y_b = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng_h.integers(0, n, size=n)
            x_b[b] = arr[idx, 0].mean()
            y_b[b] = arr[idx, 1].mean()
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()),
                float(np.percentile(x_b, 2.5)), float(np.percentile(x_b, 97.5)),
                float(np.percentile(y_b, 2.5)), float(np.percentile(y_b, 97.5)))

    sc_x, sc_y, sc_x_lo, sc_x_hi, sc_y_lo, sc_y_hi = human_xy_with_ci(sc_pair)
    tv_x, tv_y, tv_x_lo, tv_x_hi, tv_y_lo, tv_y_hi = human_xy_with_ci(tv_pair)

    h_xy["salchartqa"] = (sc_x, sc_y)
    h_xy["taskvis"]    = (tv_x, tv_y)
    h_xy_boot["salchartqa"] = (sc_x_lo, sc_x_hi, sc_y_lo, sc_y_hi)
    h_xy_boot["taskvis"]    = (tv_x_lo, tv_x_hi, tv_y_lo, tv_y_hi)
    h_align = {k: v[0] for k, v in h_xy.items()}
    h_acc   = {k: v[1] for k, v in h_xy.items()}
    print(f"  human alignment: {h_align}")
    print(f"  human accuracy:  {h_acc}")
    print(f"  human x CI sc: [{sc_x_lo:.3f}, {sc_x_hi:.3f}]  y CI sc: [{sc_y_lo:.3f}, {sc_y_hi:.3f}]")
    print(f"  human x CI tv: [{tv_x_lo:.3f}, {tv_x_hi:.3f}]  y CI tv: [{tv_y_lo:.3f}, {tv_y_hi:.3f}]")

    per_panel = {}
    for dataset in ("salchartqa", "taskvis"):
        print(f"\n=== {dataset.upper()} ===")
        rows = []
        for mk in MODEL_ORDER:
            cc_flat = load_corr_array(mk, dataset)
            if cc_flat is None:
                print(f"  skip {mk}: no correlation file")
                continue
            correct, qc = per_sample_correct(mk, dataset)
            if correct is None:
                print(f"  skip {mk}: no correctness file")
                continue
            n_s_corr = cc_flat.shape[0]
            qc_short = qc[:n_s_corr]
            correct_short = correct[:n_s_corr]
            cc_qc = cc_flat[qc_short]
            corr_qc = correct_short[qc_short]

            head_b = head_bootstrap_max(cc_qc, N_DRAW, N_BOOT_HEAD, rng)
            q_b = question_bootstrap_acc(corr_qc, N_BOOT_Q, rng)

            rows.append({
                "model": mk,
                "fam": family(mk),
                "params_b": MODEL_PARAMS_B[mk],
                "x_mean": float(np.nanmean(head_b)),
                "x_lo":   float(np.nanpercentile(head_b, 2.5)),
                "x_hi":   float(np.nanpercentile(head_b, 97.5)),
                "x_boot": head_b.astype(np.float64),
                "y_mean": float(corr_qc.mean()),
                "y_lo":   float(np.percentile(q_b, 2.5)),
                "y_hi":   float(np.percentile(q_b, 97.5)),
                "y_boot": q_b.astype(np.float64),
                "n_qc":   int(qc_short.sum()),
                "n_heads": int(cc_flat.shape[1]),
            })
            print(f"  {mk:30s}  align={rows[-1]['x_mean']:.3f}  "
                  f"acc={rows[-1]['y_mean']:.3f}  N_qc={rows[-1]['n_qc']}  "
                  f"H={rows[-1]['n_heads']}")
        per_panel[dataset] = rows

    # ── Plot ──
    plt.rcParams.update({
        "font.size": 44, "axes.labelsize": 48, "axes.titlesize": 54,
        "xtick.labelsize": 42, "ytick.labelsize": 42, "legend.fontsize": 46,
        "axes.linewidth": 1.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(28, 13))
    # Tight wspace: the two panels share x and y limits, so they read as one
    # combined plot rather than two disconnected ones.
    fig.subplots_adjust(left=0.05, right=0.82, bottom=0.13, top=0.92, wspace=0.06)
    titles = {"salchartqa": "SalChartQA", "taskvis": "TaskVis"}
    # Shared x-axis across both panels — make sure 0.45 is included on the
    # left so the human point's CI never gets clipped.
    all_x_global = []
    for ds_ in ("salchartqa", "taskvis"):
        all_x_global += [r["x_mean"] for r in per_panel[ds_]]
        all_x_global += [r["x_lo"]   for r in per_panel[ds_]]
        all_x_global += [r["x_hi"]   for r in per_panel[ds_]]
        if h_align[ds_] is not None:
            all_x_global.append(h_align[ds_])
            all_x_global.append(h_xy_boot[ds_][0])
            all_x_global.append(h_xy_boot[ds_][1])
    x_lo_g = min(min(all_x_global), 0.45) - 0.02
    x_hi_g = max(all_x_global) + 0.04
    for ax, dataset in zip(axes, ("salchartqa", "taskvis")):
        rows = per_panel[dataset]
        xs = np.array([r["x_mean"] for r in rows])
        ys = np.array([r["y_mean"] for r in rows])
        # Linear fit
        if len(xs) >= 3:
            z = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs.min(), h_align[dataset]) - 0.04,
                                max(xs.max(), h_align[dataset]) + 0.04, 100)
            ax.plot(xline, np.polyval(z, xline),
                    color="#666", linestyle="--", lw=1.6, alpha=0.75, zorder=1)
            r, p = permutation_pvalue_pearsonr(xs, ys, N_PERM, rng)
            r_dist = joint_bootstrap_r_dist(rows, N_BOOT_JOINT, rng)
            r_lo = float(np.percentile(r_dist, 2.5))
            r_hi = float(np.percentile(r_dist, 97.5))
        else:
            r = float("nan"); p = float("nan"); r_lo = r_hi = float("nan")

        # Per-model points (numeric param-count labels auto-placed below)
        text_objs = []
        bubble_artists = []
        for rec in rows:
            fam = rec["fam"]
            ax.errorbar(rec["x_mean"], rec["y_mean"],
                        xerr=[[rec["x_mean"] - rec["x_lo"]],
                              [rec["x_hi"] - rec["x_mean"]]],
                        yerr=[[rec["y_mean"] - rec["y_lo"]],
                              [rec["y_hi"] - rec["y_mean"]]],
                        fmt="none", ecolor=FAMILY_COLORS[fam],
                        elinewidth=1.6, capsize=4, alpha=0.85, zorder=3)
            sc = ax.scatter(rec["x_mean"], rec["y_mean"],
                       color=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam],
                       s=marker_size(rec["params_b"]),
                       edgecolor="black", linewidth=0.9,
                       alpha=0.85, zorder=5)
            bubble_artists.append(sc)
            # Param-count digit centered inside the bubble — white text on
            # a black stroke so it's legible on any family colour, no need
            # for adjust_text auto-arrangement. For (dataset, model) pairs
            # in LABEL_OVERRIDES, the digit is moved to an offset position
            # with a thin leader line back to the bubble centre.
            label_fontsize = 18 if rec["params_b"] <= 2 else 22
            override = LABEL_OVERRIDES.get((dataset, rec["model"]))
            if override is None:
                t = ax.text(rec["x_mean"], rec["y_mean"], f"{rec['params_b']}",
                            fontsize=label_fontsize, color="white",
                            fontweight="bold", zorder=7,
                            ha="center", va="center")
                t.set_path_effects([
                    path_effects.withStroke(linewidth=2.0, foreground="black")])
            else:
                dx, dy = override
                tx = rec["x_mean"] + dx; ty = rec["y_mean"] + dy
                # Leader line from offset label to bubble centre
                ax.plot([tx, rec["x_mean"]], [ty, rec["y_mean"]],
                        color="black", lw=0.9, alpha=0.8, zorder=6)
                t = ax.text(tx, ty, f"{rec['params_b']}",
                            fontsize=label_fontsize, color=FAMILY_COLORS[fam],
                            fontweight="bold", zorder=7,
                            ha="center", va="center")
                t.set_path_effects([
                    path_effects.withStroke(linewidth=2.4, foreground="white")])
            text_objs.append(t)
        # Human star with bootstrap-CI error bars on both axes (paired
        # sample-resampling of mean_loo and per-sample human accuracy)
        if h_align[dataset] is not None and h_acc[dataset] is not None:
            x_lo, x_hi, y_lo, y_hi = h_xy_boot[dataset]
            ax.errorbar(h_align[dataset], h_acc[dataset],
                        xerr=[[h_align[dataset] - x_lo],
                              [x_hi - h_align[dataset]]],
                        yerr=[[h_acc[dataset] - y_lo],
                              [y_hi - h_acc[dataset]]],
                        fmt="none", ecolor="black",
                        elinewidth=1.6, capsize=4, alpha=0.85, zorder=9)
            ax.scatter(h_align[dataset], h_acc[dataset], marker="*",
                       s=marker_size(8) * 4.0, color="black",
                       edgecolor="white", linewidth=2.0, zorder=10)

        ax.set_xlabel("Max Head-Human Correlation", fontsize=46)
        ax.set_title(titles[dataset], fontsize=52, pad=14)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        ax.set_box_aspect(1)  # force square panels
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        # x-limits: SHARED across both panels (includes 0.45 on the left)
        ax.set_xlim(x_lo_g, x_hi_g)
        # y-limits: SHARED across both panels for direct accuracy comparison
        all_y_global = []
        for ds_ in ("salchartqa", "taskvis"):
            all_y_global += [r["y_mean"] for r in per_panel[ds_]]
            all_y_global += [r["y_lo"]   for r in per_panel[ds_]]
            all_y_global += [r["y_hi"]   for r in per_panel[ds_]]
            if h_acc[ds_] is not None:
                all_y_global.append(h_acc[ds_])
        y_lo_g = min(all_y_global) - 0.06
        y_hi_g = max(all_y_global) + 0.06
        ax.set_ylim(y_lo_g, y_hi_g)
        # Combined y-axis: only the LEFT panel shows the y-label and tick
        # numbers; the right panel keeps tick marks but hides numbers.
        if dataset == "salchartqa":
            ax.set_ylabel("Accuracy", fontsize=46)
        else:
            ax.tick_params(axis="y", labelleft=False)
        # r (point estimate) + permutation p in lower-right
        if p < 0.001:    p_str = "p < 0.001"
        elif p < 0.01:   p_str = "p < 0.01"
        elif p < 0.05:   p_str = "p < 0.05"
        else:            p_str = f"p = {p:.3f}"
        ax.text(0.96, 0.04,
                f"r = {r:.2f}\n{p_str}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=36, linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#999", alpha=0.92))

    # ── Single legend on the right: one entry per model (color+marker+size all encoded) ──
    legend_handles = []
    for mk in MODEL_ORDER:
        fam = family(mk)
        legend_handles.append(Line2D([0], [0], marker=FAMILY_MARKERS[fam], color="w",
                                     markerfacecolor=FAMILY_COLORS[fam],
                                     markersize=np.sqrt(marker_size(MODEL_PARAMS_B[mk])),
                                     markeredgecolor="black", markeredgewidth=0.7,
                                     label=MODEL_LABELS[mk]))
    legend_handles.append(Line2D([0], [0], marker="*", color="w",
                                 markerfacecolor="black",
                                 markersize=np.sqrt(marker_size(8) * 4.0),
                                 markeredgecolor="white", markeredgewidth=1.6,
                                 label="Human"))
    model_leg = fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.995, 0.5),
               title="Model", title_fontsize=52,
               fontsize=46, frameon=True, framealpha=0.95, edgecolor="#999",
               labelspacing=0.25, borderpad=0.25, handletextpad=0.5,
               borderaxespad=0.2)
    fig.add_artist(model_leg)


    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
