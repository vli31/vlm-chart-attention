#!/usr/bin/env python3
"""Companion to fig5 — dot plot of *absolute* accuracy under each condition
(baseline + 4 ablation head sets), per model, with 95% bootstrap CIs.

One panel per model (12 panels in a 3×4 grid). X axis = condition; Y axis =
mean accuracy. Family-coloured panel borders.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("./results/correctness")
OUT = Path("./final_paper_figures")
RNG = np.random.default_rng(42)
N_BOOT = 2000

ALL = [
    ("2.5-3B","Q2.5-3B"), ("2.5-7B","Q2.5-7B"),
    ("2B","Q3-2B"), ("4B","Q3-4B"), ("8B","Q3-8B"),
    ("internvl3-1b-instruct","IV3-1B"), ("internvl3-2b-instruct","IV3-2B"),
    ("internvl3-8b-instruct","IV3-8B"),
    ("internvl3.5-1b-instruct","IV3.5-1B"), ("internvl3.5-2b-instruct","IV3.5-2B"),
    ("internvl3.5-4b-instruct","IV3.5-4B"), ("internvl3.5-8b-instruct","IV3.5-8B"),
]
KINDS = [("baseline","Base"), ("top5","Top-5"), ("bottom5","Bot-5"),
         ("random5_a","Rnd A"), ("random5_b","Rnd B")]
KIND_COLORS = {
    "baseline":"#222222",
    "top5":"#D62728",       # red
    "bottom5":"#1F77B4",    # blue
    "random5_a":"#7F7F7F",  # grey
    "random5_b":"#BCBCBC",  # light grey
}

# concentration metric: M_for_levels at K=1, p=0.9, expressed as fraction of H.
# Smaller fraction = MORE concentrated (a few heads always win for top-1).
CONC_DATA = json.load(open(Path("./final_paper_figures/fig4-2_data.json")))
def concentration_pct(mk):
    """Fraction of heads needed to cover 90% of samples' top-1 head."""
    aln_key = {"internvl3-1b-instruct":"internvl3-1b","internvl3-2b-instruct":"internvl3-2b",
               "internvl3-8b-instruct":"internvl3-8b",
               "internvl3.5-1b-instruct":"internvl3.5-1b","internvl3.5-2b-instruct":"internvl3.5-2b",
               "internvl3.5-4b-instruct":"internvl3.5-4b","internvl3.5-8b-instruct":"internvl3.5-8b",
              }.get(mk, mk)
    rec = CONC_DATA[aln_key]
    return rec["per_k"]["1"]["M_for_levels"]["0.9"] / rec["H"]
FAMILY_COLORS = {"Qwen2.5-VL":"#BB52A6","Qwen3-VL":"#3540A8",
                 "InternVL3":"#D17B30","InternVL3.5":"#A23E1A"}
def family_of(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"


def latest_canonical(kind, mk):
    """Latest run at N=4556 with method = pre_output_projection / all_tokens."""
    pat = re.compile(rf"salchartqa_{re.escape(kind)}_top5_mean_ablation__{re.escape(mk)}__(\d+_\d+)\.json$")
    cands = []
    for f in os.listdir(P):
        m = pat.match(f)
        if not m: continue
        d = json.load(open(P / f))
        cands.append((m.group(1), f, d, len(d["evaluation_sample_ids"])))
    if not cands: return None
    for ts, f, d, n in sorted(cands, reverse=True):
        if n == 4556 and d.get("ablation_type") == "dataset_mean_replace_pre_output_projection" \
           and d.get("ablation_scope") == "all_tokens":
            return d
    for ts, f, d, n in sorted(cands, reverse=True):
        if n == 4556:
            return d
    return None


def load_baseline(d):
    bp = Path(d["baseline_correctness_file"])
    alt = Path(str(bp).replace("/lab/", "/LABS/lab/"))
    for p in (bp, alt):
        try:
            if p.exists():
                bd = json.load(open(p))
                out = {}
                for r in bd["results"]:
                    nc = r.get("num_correct")
                    if nc is None: continue
                    out[r["sample_id"]] = float(nc) / 5.0
                return out
        except PermissionError: continue
    return None

def per_sample_acc(results):
    out = {}
    for r in results:
        nc = r.get("num_correct")
        if nc is None: continue
        out[r["sample_id"]] = float(nc) / 5.0
    return out

def per_trial_correct(results):
    """Returns dict sample_id -> list of 5 bools (correct[i] for trial i).
    Skips samples with missing entries."""
    out = {}
    for r in results:
        c = r.get("correct")
        if c is None or len(c) != 5: continue
        out[r["sample_id"]] = [bool(x) for x in c]
    return out

import math

# Hierarchical bootstrap CI on aggregate accuracy:
# For each replicate, resample samples WITH REPLACEMENT, and for each
# resampled sample also resample its 5 trials WITH REPLACEMENT, then take
# the grand mean. CI from the bootstrap distribution. This captures BOTH
# the per-sample variability ("which test items?") and the per-trial
# variability ("which decoder seed?") — the standard test-set + decoder
# uncertainty most readers care about.
def hierarchical_bootstrap_ci(sample_trials, n_boot=1000, ci=95):
    """sample_trials: 2D array [N, T] of per-sample, per-trial booleans.
    Returns (mean, lo, hi)."""
    arr = np.asarray(sample_trials, dtype=np.float64)
    N, T = arr.shape
    if N == 0 or T == 0:
        return 0.0, 0.0, 0.0
    boots = np.empty(n_boot, dtype=np.float64)
    rows = np.arange(N)[:, None]
    for b in range(n_boot):
        si = RNG.integers(0, N, size=N)
        ti = RNG.integers(0, T, size=(N, T))
        boots[b] = arr[si][rows, ti].mean()
    lo, hi = np.percentile(boots, [(100-ci)/2, 100 - (100-ci)/2])
    return float(arr.mean()), float(lo), float(hi)


def hierarchical_paired_p(arr_a, arr_b, n_boot=1000):
    """Paired hierarchical bootstrap test of mean(arr_a) > mean(arr_b).
    arr_a, arr_b: 2D arrays [N, T] aligned on samples (and trials, but we
    use only sample-cluster bootstrap with paired (sample, trial-resample)).
    Returns (mean_diff, two_sided_p, ci_lo, ci_hi).
    The p is bootstrap-based: fraction of bootstrap reps where the mean of
    (a-b) flips sign relative to the observed mean.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    if a.shape != b.shape: return float("nan"), float("nan"), float("nan"), float("nan")
    N, T = a.shape
    if N == 0 or T == 0: return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = a - b                                # [N, T] paired differences
    obs_mean = float(diffs.mean())
    sign = 1 if obs_mean >= 0 else -1
    boots = np.empty(n_boot)
    rows = np.arange(N)[:, None]
    n_opposite = 0
    for k in range(n_boot):
        si = RNG.integers(0, N, size=N)
        ti = RNG.integers(0, T, size=(N, T))
        m = diffs[si][rows, ti].mean()
        boots[k] = m
        if m * sign <= 0:
            n_opposite += 1
    # two-sided p ≈ 2 * P(boot disagrees with observed sign)
    p_two = float(2.0 * (n_opposite / n_boot))
    p_two = min(p_two, 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs_mean, p_two, float(lo), float(hi)


# ── compute mean accuracy + per-trial CI per (model, condition) ──
# CI is across the 5 stochastic generations (responses_per_q): for each
# trial t in 0..4 take mean accuracy over samples; 95% CI from those 5
# trial-level means using t-dist df=4.
def load_baseline_trials(d):
    bp = Path(d["baseline_correctness_file"])
    alt = Path(str(bp).replace("/lab/", "/LABS/lab/"))
    for p in (bp, alt):
        try:
            if p.exists():
                bd = json.load(open(p))
                return per_trial_correct(bd["results"])
        except PermissionError: continue
    return None

records = {}
for mk, _ in ALL:
    records[mk] = {}
    any_d = None
    for kind, _ in KINDS[1:]:
        d = latest_canonical(kind, mk)
        if d is not None:
            any_d = d; break
    if any_d is None: continue

    bs_trials = load_baseline_trials(any_d)
    if bs_trials is None: continue
    sids_eval = [r["sample_id"] for r in any_d["ablated"]["results"]]
    sids_used = [s for s in sids_eval if s in bs_trials]
    bs_arr = np.array([bs_trials[s] for s in sids_used], dtype=np.float64)
    m, lo, hi = hierarchical_bootstrap_ci(bs_arr)
    records[mk]["baseline"] = {"mean": m, "lo": lo, "hi": hi,
                               "n_samples": len(sids_used)}

    # cache the per-sample × trial arrays for kind comparisons
    kind_arrs = {"baseline": (bs_arr, sids_used)}
    for kind, _ in KINDS[1:]:
        d = latest_canonical(kind, mk)
        if d is None: continue
        ab_trials = per_trial_correct(d["ablated"]["results"])
        sids = [r["sample_id"] for r in d["ablated"]["results"] if r["sample_id"] in ab_trials]
        ab_arr = np.array([ab_trials[s] for s in sids], dtype=np.float64)
        m, lo, hi = hierarchical_bootstrap_ci(ab_arr)
        records[mk][kind] = {"mean": m, "lo": lo, "hi": hi,
                             "n_samples": len(sids)}
        kind_arrs[kind] = (ab_arr, sids)

    # paired statistical tests:
    #   Top-5 vs Baseline (does ablation matter?)
    #   Top-5 vs Random A  (is the Top-5 effect SPECIFIC?)
    def _paired(a_kind, b_kind):
        if a_kind not in kind_arrs or b_kind not in kind_arrs:
            return None
        a_arr, sa = kind_arrs[a_kind]; b_arr, sb = kind_arrs[b_kind]
        sids_common = [s for s in sa if s in set(sb)]
        if len(sids_common) < 50: return None
        idx_a = {s: i for i, s in enumerate(sa)}
        idx_b = {s: i for i, s in enumerate(sb)}
        ai = np.array([idx_a[s] for s in sids_common])
        bi = np.array([idx_b[s] for s in sids_common])
        a_p = a_arr[ai]; b_p = b_arr[bi]
        return hierarchical_paired_p(a_p, b_p)
    records[mk]["test_top5_vs_baseline"] = _paired("top5", "baseline")
    records[mk]["test_top5_vs_random_a"] = _paired("top5", "random5_a")

# Sort models by concentration (most concentrated → least)
ALL_SORTED = sorted([m for m in ALL if records.get(m[0], {}).get("baseline")],
                    key=lambda mk_disp: concentration_pct(mk_disp[0]))

# ── plot single row, per-panel y-axes, sorted by concentration ──
n_kinds = len(KINDS)
xs = np.linspace(0, 1, n_kinds)
n_models = len(ALL_SORTED)
fig, axes = plt.subplots(1, n_models, figsize=(1.20 * n_models + 1.0, 5.0),
                         sharey=False, sharex=True)
fig.subplots_adjust(wspace=0.55)

for i, (mk, disp) in enumerate(ALL_SORTED):
    ax = axes[i]
    means = [records[mk][k]["mean"] for k, _ in KINDS]
    los   = [records[mk][k]["lo"]   for k, _ in KINDS]
    his   = [records[mk][k]["hi"]   for k, _ in KINDS]
    cs    = [KIND_COLORS[k]         for k, _ in KINDS]

    ax.plot(xs, means, color="#bbb", lw=0.8, alpha=0.6, zorder=1)
    for x, m, lo, hi, c in zip(xs, means, los, his, cs):
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]],
                    fmt="o", color=c, ecolor=c, mfc=c, mec="black", mew=0.4,
                    ms=5.0, capsize=3.0, elinewidth=1.4, lw=1.0, zorder=3)
    base = records[mk]["baseline"]["mean"]
    ax.axhline(base, color="#222", ls="--", lw=0.6, alpha=0.35, zorder=0)

    fam_color = FAMILY_COLORS[family_of(mk)]
    ax.set_title(disp,
                 color=fam_color, fontweight="bold", fontsize=15, pad=6)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.10, 1.10)

    # Strict baseline alignment: baseline always at the same fractional y-pos.
    # frac=0.70 (slightly above middle) lets us include ticks both below and
    # above baseline naturally without bloating the axis.
    BASE_FRAC = 0.70
    safety = 0.02
    drop_below = max(base - min(los), 0.0) + safety
    rise_above = max(max(his) - base, 0.0) + safety
    s_data = max(drop_below / BASE_FRAC, rise_above / (1.0 - BASE_FRAC))

    # Pick smallest step (0.1, 0.2, or 0.5) that yields 2-3 ticks within
    # bounds, expanding span as needed to ensure ≥2 multiples of step are in.
    def _ticks_in_range(b0, t0, st):
        n_lo = math.ceil(b0 / st - 1e-9)
        n_hi = math.floor(t0 / st + 1e-9)
        return [k * st for k in range(n_lo, n_hi + 1)]

    def _min_s_for_2_ticks(step_):
        # Compute s_k (minimum s required to include each multiple of step_).
        ks = range(int((base - 1.5) // step_) - 1, int((base + 1.5) // step_) + 2)
        s_list = []
        for k in ks:
            v = k * step_
            if v <= base:
                s_k = (base - v) / BASE_FRAC
            else:
                s_k = (v - base) / (1.0 - BASE_FRAC)
            s_list.append(max(s_k, 0.0))
        s_list.sort()
        return s_list[1] if len(s_list) > 1 else s_list[0]

    chosen_step = None
    chosen_s    = None
    for step_try in [0.10, 0.20, 0.50]:
        s_for_2 = _min_s_for_2_ticks(step_try)
        s = max(s_data, s_for_2)
        bot_t = base - BASE_FRAC * s
        top_t = base + (1.0 - BASE_FRAC) * s
        n_ticks = len(_ticks_in_range(bot_t, top_t, step_try))
        if 2 <= n_ticks <= 3:
            chosen_step = step_try
            chosen_s    = s
            break
    if chosen_step is None:
        chosen_step, chosen_s = 0.5, s_data
    bottom = base - BASE_FRAC * chosen_s
    top    = base + (1.0 - BASE_FRAC) * chosen_s
    ax.set_ylim(bottom, top)

    ax.grid(True, axis="y", alpha=0.20, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.ticker import MultipleLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(MultipleLocator(chosen_step))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="minor", length=0)

axes[0].set_ylabel("Mean accuracy", fontsize=15)

# global legend (one entry per condition) at top, with extra space below it
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], marker="o", linestyle="none",
                         markerfacecolor=KIND_COLORS[k], markeredgecolor="black",
                         markersize=10, label=lbl)
                  for k, lbl in KINDS]
fig.legend(handles=legend_handles, loc="upper center", ncol=len(KINDS),
           frameon=True, fancybox=False, edgecolor="#444",
           fontsize=14, bbox_to_anchor=(0.5, 1.04))

fig.suptitle(
    r"Fig 5b — SalChartQA accuracy under head-set ablations (N${=}4556$, 95% hierarchical bootstrap CI: samples $\times$ 5 stochastic generations). "
    r"Panels left→right sorted by head-concentration (left = most concentrated).",
    fontsize=13, y=1.13,
)
fig.tight_layout(rect=(0, 0, 1, 0.80))
png = OUT / "fig5b.png"
pdf = OUT / "fig5b.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"saved {png} and {pdf}")

# also save numbers
(OUT / "fig5b_data.json").write_text(json.dumps(records, indent=2))
print(f"saved {OUT / 'fig5b_data.json'}")


# ── fig5b-3: same as the old fig5b-2 layout (per-model dot plot, shared y)
#    but excluding the three least-concentrated models. Kept here so it runs
#    before the sys.exit() guard that disables the legacy fig5b-2 output.
import matplotlib.pyplot as _plt
def _cond_records(mk):
    return [rec for rec in records[mk].values()
            if isinstance(rec, dict) and "lo" in rec]
KIND_LONG = {"baseline": "Baseline", "top5": "Top-5", "bottom5": "Bottom-5",
             "random5_a": "Random A", "random5_b": "Random B"}
g_pad = 0.02
ALL_SORTED_3 = ALL_SORTED[:-3]
n_models_3 = len(ALL_SORTED_3)
fig3, axes3 = _plt.subplots(1, n_models_3, figsize=(2.60 * n_models_3 + 2.4, 8.6),
                            sharey=True, sharex=True)
fig3.subplots_adjust(wspace=0.10)

global_lo3 = min(min(rec["lo"] for rec in _cond_records(mk))
                 for mk, _ in ALL_SORTED_3)
global_hi3 = max(max(rec["hi"] for rec in _cond_records(mk))
                 for mk, _ in ALL_SORTED_3)
g_bottom3 = math.floor(max(0.0, global_lo3 - g_pad) / 0.10) * 0.10
g_top3    = math.ceil (min(1.0, global_hi3 + g_pad) / 0.10) * 0.10

for i, (mk, disp) in enumerate(ALL_SORTED_3):
    ax = axes3[i]
    means = [records[mk][k]["mean"] for k, _ in KINDS]
    los   = [records[mk][k]["lo"]   for k, _ in KINDS]
    his   = [records[mk][k]["hi"]   for k, _ in KINDS]
    cs    = [KIND_COLORS[k]         for k, _ in KINDS]

    ax.plot(xs, means, color="#bbb", lw=1.0, alpha=0.6, zorder=1)
    for x, m, lo, hi, c in zip(xs, means, los, his, cs):
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]],
                    fmt="none", ecolor="#222",
                    capsize=12.0, capthick=3.5, elinewidth=5.5, zorder=2)
        ax.scatter([x], [m], marker="o", s=220, c=c,
                   edgecolors="black", linewidths=1.0, zorder=3)
    base = records[mk]["baseline"]["mean"]
    ax.axhline(base, color="#222", ls="--", lw=0.9, alpha=0.40, zorder=0)

    fam_color = FAMILY_COLORS[family_of(mk)]
    fam_short = {"Qwen2.5-VL": "Q2.5-VL", "Qwen3-VL": "Q3-VL",
                 "InternVL3":  "IV3",     "InternVL3.5": "IV3.5"}[family_of(mk)]
    size_str = disp.split("-")[-1]
    ax.set_title(f"{fam_short}\n{size_str}",
                 color=fam_color, fontweight="bold", fontsize=38, pad=12)
    ax.set_xticks([])
    ax.set_xlim(-0.10, 1.10)
    ax.tick_params(axis="y", labelsize=38)
    ax.grid(True, axis="y", alpha=0.20, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes3[0].set_ylim(g_bottom3, g_top3)
g_span3 = g_top3 - g_bottom3
g_step3 = 0.20 if g_span3 < 0.50 else (0.25 if g_span3 < 0.75 else 0.30)
axes3[0].yaxis.set_major_locator(MultipleLocator(g_step3))
axes3[0].yaxis.set_minor_locator(NullLocator())
axes3[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
for a in axes3:
    a.yaxis.set_minor_locator(NullLocator())
axes3[0].set_ylabel("Mean accuracy", fontsize=42)

fig3_legend_handles = [Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor=KIND_COLORS[k], markeredgecolor="black",
                              markersize=26, label=KIND_LONG[k])
                       for k, _ in KINDS]
fig3.tight_layout(rect=(0, 0.18, 1, 1.0))
fig3.legend(handles=fig3_legend_handles, loc="lower center", ncol=len(KINDS),
            frameon=True, fancybox=True,
            fontsize=38, bbox_to_anchor=(0.5, -0.01))

pdf3 = OUT / "fig5b-3.pdf"
fig3.savefig(pdf3, bbox_inches="tight")
print(f"saved {pdf3}")


# ── fig5b-2: DISABLED in this script. The current fig5b-2 is a 3-panel
#    head-layer / head-pool figure produced by build_fig5b2_layers.py.
#    Stop here so this script doesn't overwrite that file.
import sys as _sys
_sys.exit(0)

import matplotlib.pyplot as _plt
fig2, axes2 = _plt.subplots(1, n_models, figsize=(1.55 * n_models + 2.0, 7.4),
                            sharey=True, sharex=True)
fig2.subplots_adjust(wspace=0.10)

# Only consider per-condition entries (dicts with 'lo'/'hi'), skip test tuples.
def _cond_records(mk):
    return [rec for rec in records[mk].values()
            if isinstance(rec, dict) and "lo" in rec]
global_lo = min(min(rec["lo"] for rec in _cond_records(mk))
                for mk, _ in ALL_SORTED)
global_hi = max(max(rec["hi"] for rec in _cond_records(mk))
                for mk, _ in ALL_SORTED)
g_pad = 0.02
g_bottom = math.floor(max(0.0, global_lo - g_pad) / 0.10) * 0.10
g_top    = math.ceil (min(1.0, global_hi + g_pad) / 0.10) * 0.10

for i, (mk, disp) in enumerate(ALL_SORTED):
    ax = axes2[i]
    means = [records[mk][k]["mean"] for k, _ in KINDS]
    los   = [records[mk][k]["lo"]   for k, _ in KINDS]
    his   = [records[mk][k]["hi"]   for k, _ in KINDS]
    cs    = [KIND_COLORS[k]         for k, _ in KINDS]

    ax.plot(xs, means, color="#bbb", lw=0.8, alpha=0.6, zorder=1)
    for x, m, lo, hi, c in zip(xs, means, los, his, cs):
        # Error bars drawn in dark grey behind the colored dot so they stand
        # out against any marker color. Bar HEIGHTS are exactly [lo, hi];
        # only line/cap visual weight is scaled up so they're perceptible
        # at the compressed shared y-scale.
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]],
                    fmt="none", ecolor="#222",
                    capsize=8.0, capthick=2.5, elinewidth=4.0, zorder=2)
        ax.scatter([x], [m], marker="o", s=80, c=c,
                   edgecolors="black", linewidths=0.7, zorder=3)
    base = records[mk]["baseline"]["mean"]
    ax.axhline(base, color="#222", ls="--", lw=0.7, alpha=0.40, zorder=0)

    fam_color = FAMILY_COLORS[family_of(mk)]
    # Title: family on line 1, size on line 2 (e.g. "Q3-VL\n2B").
    fam_short = {"Qwen2.5-VL": "Q2.5-VL", "Qwen3-VL": "Q3-VL",
                 "InternVL3":  "IV3",     "InternVL3.5": "IV3.5"}[family_of(mk)]
    size_str = disp.split("-")[-1]   # e.g. "2B"
    ax.set_title(f"{fam_short}\n{size_str}",
                 color=fam_color, fontweight="bold", fontsize=22, pad=10)
    ax.set_xticks([])
    ax.set_xlim(-0.10, 1.10)
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(True, axis="y", alpha=0.20, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (Stats: per-model paired hierarchical bootstrap of Top-5 vs Random A
    # is computed in records[mk]["test_top5_vs_random_a"] but not annotated
    # on the figure to keep it clean.)

axes2[0].set_ylim(g_bottom, g_top)
g_span = g_top - g_bottom
# one more tick than fig5b: aim for ~4 ticks rather than ~3
g_step = 0.20 if g_span < 0.50 else (0.25 if g_span < 0.75 else 0.30)
axes2[0].yaxis.set_major_locator(MultipleLocator(g_step))
axes2[0].yaxis.set_minor_locator(NullLocator())
axes2[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
for a in axes2:
    a.yaxis.set_minor_locator(NullLocator())
axes2[0].set_ylabel("Mean accuracy", fontsize=28)

# Legend at the bottom: conditions + significance key.
KIND_LONG = {"baseline": "Baseline", "top5": "Top-5", "bottom5": "Bottom-5",
             "random5_a": "Random A", "random5_b": "Random B"}
fig2_legend_handles = [Line2D([0], [0], marker="o", linestyle="none",
                              markerfacecolor=KIND_COLORS[k], markeredgecolor="black",
                              markersize=18, label=KIND_LONG[k])
                       for k, _ in KINDS]
# Significance-key entry as a text-only legend handle (uses a transparent line)
sig_handle = Line2D([0], [0], color="none", lw=0,
    label=r"$^{***}p{<}0.001$    $^{**}p{<}0.01$    $^{*}p{<}0.05$    n.s. otherwise"
          "\n(Top-5 vs Random A, paired hierarchical bootstrap)")
fig2.legend(handles=fig2_legend_handles, loc="lower center", ncol=len(KINDS),
            frameon=True, fancybox=True,
            fontsize=24, bbox_to_anchor=(0.5, 0.0))
fig2.tight_layout(rect=(0, 0.10, 1, 1.0))

png2 = OUT / "fig5b-2.png"
pdf2 = OUT / "fig5b-2.pdf"
fig2.savefig(png2, dpi=300, bbox_inches="tight")
fig2.savefig(pdf2, bbox_inches="tight")
print(f"saved {png2} and {pdf2}")


