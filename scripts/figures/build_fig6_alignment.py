#!/usr/bin/env python3
"""fig5c variant — scatter coloured by per-question Q3-8B model accuracy.

Differences vs build_fig5c.py:
  - Scatter color = Q3-8B model accuracy per question (mean correct over
    the 5 sampled responses), instead of human accuracy.
  - X label: 'Inter-human gaze $r$' (no parenthetical).
  - Y label: 'Q3-8B max head--human $r$' (no parenthetical).
  - All fonts bumped except the r=0.48 / p<... annotation box.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import pearsonr, gaussian_kde

ROOT = Path("./")
APIN = ROOT / "aligned_pop_internal"
CORR = Path("./data/lvlm-chart/correlations/salchartqa")
CORRECT_DIR = Path("./data/lvlm-chart/correctness")
OUT  = ROOT / "final_paper_figures"
QC_PATH = OUT / "broad_qc_ids.json"
HLOO_CSV = ROOT / "human_human_ceiling" / "human_loo_per_worker.csv"
CEIL_CSV = ROOT / "human_human_ceiling" / "human_ceiling_t16_broad.csv"

Q38B_CORRECT = CORRECT_DIR / "correctness_8b_salchartqa_x5_n5999_20260317_051045_fixed.json"

ALL_MODELS_FULL = [
    "2.5-3B", "2.5-7B",
    "2B", "4B", "8B",
    "internvl3-1b", "internvl3-2b", "internvl3-8b",
    "internvl3.5-1b", "internvl3.5-2b", "internvl3.5-4b", "internvl3.5-8b",
]
EXCLUDE = {"internvl3-1b", "internvl3-2b", "internvl3.5-1b"}
KEEP_MODELS = [m for m in ALL_MODELS_FULL if m not in EXCLUDE]

DISPLAY = {
    "2.5-3B":"Q2.5-3B","2.5-7B":"Q2.5-7B",
    "2B":"Q3-2B","4B":"Q3-4B","8B":"Q3-8B",
    "internvl3-1b":"IV3-1B","internvl3-2b":"IV3-2B","internvl3-8b":"IV3-8B",
    "internvl3.5-1b":"IV3.5-1B","internvl3.5-2b":"IV3.5-2B",
    "internvl3.5-4b":"IV3.5-4B","internvl3.5-8b":"IV3.5-8B",
}
FAMILY_COLORS = {
    "Qwen2.5-VL":  "#BB52A6",
    "Qwen3-VL":    "#3540A8",
    "InternVL3":   "#D17B30",
    "InternVL3.5": "#A23E1A",
    "Human":       "#222222",
}
def family_of(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"


# Font sizes (everything bumped except R_ANNOT_FS).
TICK_FS    = 42
AXIS_FS    = 42
LEGEND_FS  = 42
YLABEL_FS  = 40
CB_FS      = 36
ANNO_FS    = 36
R_ANNOT_FS = 40   # unchanged per request


# ── data ────────────────────────────────────────────────────────────
qc = json.loads(QC_PATH.read_text())
qc_ids = set(int(x) for x in qc["ids_in_qc_subset"])

ridge_data = {}
for mk in KEEP_MODELS:
    p = CORR / f"{mk}_mean_gaze_correlations_sigma1.0.npz"
    if not p.exists(): continue
    z = np.load(p)
    cc_mean = z["cc_mean_all_per_sample"]
    cc_maxw = z["cc_max_individual_per_sample"]
    N = cc_mean.shape[0]
    keep_mask = np.array([i in qc_ids for i in range(N)])
    cc_mean_q = cc_mean[keep_mask]
    cc_maxw_q = cc_maxw[keep_mask]
    cm_flat = np.where(np.isnan(cc_mean_q), -np.inf, cc_mean_q).reshape(cc_mean_q.shape[0], -1)
    mw_flat = np.where(np.isnan(cc_maxw_q), -np.inf, cc_maxw_q).reshape(cc_maxw_q.shape[0], -1)
    max_head_mean = cm_flat.max(axis=1)
    max_any_human = mw_flat.max(axis=1)
    sids_kept = np.array([i for i in range(N) if i in qc_ids])
    finite = np.isfinite(max_head_mean) & np.isfinite(max_any_human)
    rec = {
        "sample_idx": sids_kept[finite],
        "max_head_mean": max_head_mean[finite],
        "max_any_human": max_any_human[finite],
    }
    p_csv = APIN / f"{mk}_per_sample.csv"
    if p_csv.exists():
        df_pw = pd.read_csv(p_csv)
        if "sample_idx" in df_pw.columns:
            df_pw = df_pw[df_pw["sample_idx"].isin(qc_ids)]
        col = "avg_max_cc_per_worker_aligned"
        if col in df_pw.columns:
            v = df_pw[col].dropna().values
            if len(v) >= 5:
                rec["avg_max_per_worker"] = v
    ridge_data[mk] = rec

hloo = pd.read_csv(HLOO_CSV)
hloo = hloo[hloo["sample_idx"].isin(qc_ids)]
hh_avg = hloo["mean_loo"].dropna().values
hh_max = hloo["max_loo"].dropna().values
HH_AVG_MED = float(np.median(hh_avg))
HH_MAX_MED = float(np.median(hh_max))

ceil = pd.read_csv(CEIL_CSV)[["sample_idx", "image_name", "question_id"]].dropna()
ceil = ceil[ceil["sample_idx"].isin(qc_ids)]
gaze_r = pd.read_csv(CEIL_CSV)[["sample_idx", "loo_cc"]].dropna()
gaze_r = gaze_r[gaze_r["sample_idx"].isin(qc_ids)]
gaze_r_map = dict(zip(gaze_r["sample_idx"].astype(int).values,
                      gaze_r["loo_cc"].astype(float).values))

# ── per-question Q3-8B model accuracy ───────────────────────────────
# correctness JSON has one entry per question with `image_name`, sample_id
# of form 'salchartqa_<imageid>_Q<n>', and a `correct` list of bools.
correct_doc = json.loads(Q38B_CORRECT.read_text())
model_acc_by_key = {}
for r in correct_doc["results"]:
    sid = r.get("sample_id", "")
    img = r.get("image_name")
    parts = sid.rsplit("_", 1)  # ['salchartqa_<imgid>', 'Q<n>']
    if len(parts) != 2 or not parts[1].startswith("Q"):
        continue
    qid = parts[1]
    correct = r.get("correct", [])
    if not correct:
        continue
    acc = float(np.mean([1.0 if c else 0.0 for c in correct]))
    model_acc_by_key[(img, qid)] = acc

ceil_full = pd.read_csv(CEIL_CSV)[["sample_idx", "image_name", "question_id"]].dropna()
ceil_full = ceil_full[ceil_full["sample_idx"].isin(qc_ids)]
acc_map = {}
for sid, img, qid in zip(ceil_full["sample_idx"].astype(int).values,
                         ceil_full["image_name"].values,
                         ceil_full["question_id"].values):
    acc_map[sid] = model_acc_by_key.get((img, qid), np.nan)


# ── layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28.0, 11.0))
gs = fig.add_gridspec(1, 5,
                      width_ratios=[1.30, 0.55, 1.55, 0.10, 0.55],
                      wspace=0.0)
axS = fig.add_subplot(gs[0, 0])
axV = fig.add_subplot(gs[0, 2])
axL = fig.add_subplot(gs[0, 4])
axL.set_axis_off()


# ── Region 1: legend ────────────────────────────────────────────────
strat_handles = [
    Line2D([0], [0], color="#222", lw=4.4, linestyle="-",
           label="Avg over viewers'\nbest-match head"),
    Line2D([0], [0], color="#222", lw=3.8, linestyle="--",
           label="Best head vs.\nmean-viewer gaze"),
    Line2D([0], [0], color="#222", lw=3.0, linestyle=":",
           label="Max over\n(head, viewer)"),
]
axL.legend(handles=strat_handles,
           loc="center left",
           bbox_to_anchor=(0.0, 0.5),
           fontsize=LEGEND_FS,
           frameon=True, edgecolor="#999", framealpha=0.95,
           labelspacing=1.0, handlelength=2.4, handletextpad=0.6,
           borderpad=0.8, borderaxespad=0.0,
           alignment="left")


# ── Region 2: scatter for Q3-8B coloured by Q3-8B model accuracy ────
SCATTER_MK = "8B"
rec = ridge_data[SCATTER_MK]
sids = rec["sample_idx"]
y_mh = rec["max_head_mean"]
g_r  = np.array([gaze_r_map.get(int(s), np.nan) for s in sids])
acc  = np.array([acc_map.get(int(s), np.nan) for s in sids])
mask = np.isfinite(g_r) & np.isfinite(y_mh) & np.isfinite(acc)
g_r, y_mh, acc = g_r[mask], y_mh[mask], acc[mask]
r_pearson, p_pearson = pearsonr(g_r, y_mh)

# Snap to the 6 discrete accuracy values {0, 0.2, ..., 1.0} (5 trials/q).
acc_bin = np.round(acc * 5) / 5

# Plot in count-descending order so the rarest bins (typically the low-acc
# points) end up drawn on top of the dense high-acc cloud.
bin_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
counts = np.array([int((acc_bin == b).sum()) for b in bin_levels])
draw_order = np.argsort(-counts)            # most-common first
order = np.concatenate([np.where(acc_bin == bin_levels[i])[0] for i in draw_order])
g_r_o, y_mh_o, acc_o = g_r[order], y_mh[order], acc_bin[order]

# Discrete magenta palette (RdPu): low-accuracy = saturated dark magenta
# (drawn on top so it pops), high-accuracy = pale pink (the background bulk).
disc_colors = plt.cm.RdPu(np.array([0.95, 0.78, 0.60, 0.42, 0.28, 0.16]))
cmap_disc = ListedColormap(disc_colors)
boundaries = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
norm_disc = BoundaryNorm(boundaries, cmap_disc.N)

# Uniform marker size and alpha; rare low-accuracy points are simply drawn
# on top (via the order above) so they read clearly against the bulk.
sc = axS.scatter(g_r_o, y_mh_o, c=acc_o, cmap=cmap_disc, norm=norm_disc,
                 s=20, alpha=0.70, linewidths=0)

slope, intercept = np.polyfit(g_r, y_mh, 1)
xs = np.linspace(g_r.min(), g_r.max(), 200)
axS.plot(xs, slope * xs + intercept, color="#222", lw=2.6, alpha=0.85)

def _format_p(p):
    if p < 1e-300:  return r"p $< 10^{-300}$"
    if p < 0.001:   return r"p $< 0.001$"
    return rf"p $= {p:.3f}$"

axS.text(0.06, 0.95,
         rf"$r = {abs(r_pearson):.2f}$" "\n" + _format_p(p_pearson),
         transform=axS.transAxes, ha="left", va="top",
         fontsize=R_ANNOT_FS, color="#222",
         bbox=dict(facecolor="white", edgecolor="#999", alpha=0.97,
                   boxstyle="round,pad=0.55"))

axS.set_xlim(-0.02, 1.0)
axS.set_ylim(-0.02, 1.0)
axS.set_xticks([0.0, 0.5, 1.0])
axS.set_yticks([0.0, 0.5, 1.0])
axS.tick_params(axis="both", labelsize=TICK_FS)
axS.set_xlabel(r"Inter-human gaze $r$", fontsize=AXIS_FS)
axS.set_ylabel(r"Q3-8B max head--human $r$", fontsize=AXIS_FS)
axS.grid(True, alpha=0.20, lw=0.5)
axS.spines["top"].set_visible(False)
axS.spines["right"].set_visible(False)

cb_left, cb_bottom, cb_w, cb_h = 0.55, 0.09, 0.42, 0.045
patch_ax = axS.inset_axes(
    [cb_left - 0.015, cb_bottom - 0.07, cb_w + 0.030, cb_h + 0.13])
patch_ax.set_axis_off()
patch_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=patch_ax.transAxes,
                                 facecolor="white", edgecolor="#aaa",
                                 linewidth=0.8, zorder=4))
cax = axS.inset_axes([cb_left, cb_bottom, cb_w, cb_h])
cbar = fig.colorbar(sc, cax=cax, orientation="horizontal",
                    ticks=[0.0, 1.0], spacing="proportional")
cbar.ax.set_xticklabels(["0", "1"])
cbar.ax.tick_params(labelsize=CB_FS)
cbar.ax.set_title("Model accuracy", fontsize=CB_FS, pad=8)
cax.set_zorder(5)
cbar.outline.set_zorder(5)


# ── Region 3: half-violins ──────────────────────────────────────────
ROW_ORDER = KEEP_MODELS[::-1] + ["Human"]
N_ROWS = len(ROW_ORDER)
ROW_HEIGHT = 1.0
y_of = {row: i * ROW_HEIGHT for i, row in enumerate(ROW_ORDER)}

DENSITY_AMP = 0.85
xgrid = np.linspace(-0.02, 1.0, 400)
STRATEGIES = [
    ("avg_max_per_worker", "-",  0.55, 1.6),
    ("max_head_mean",      "--", 0.30, 1.4),
    ("max_any_human",      ":",  0.18, 1.2),
]
HUMAN_STRATEGIES = {"avg_max_per_worker": hh_avg, "max_any_human": hh_max}

for row in ROW_ORDER:
    is_human = (row == "Human")
    fam = "Human" if is_human else family_of(row)
    color = FAMILY_COLORS[fam]
    y0 = y_of[row]
    for key, ls, alpha_fill, lw in STRATEGIES:
        if is_human:
            vals = HUMAN_STRATEGIES.get(key)
        else:
            vals = (ridge_data.get(row) or {}).get(key)
        if vals is None or len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.08)
        d = kde(xgrid)
        if d.max() <= 0: continue
        d_n = d / d.max() * DENSITY_AMP
        axV.fill_between(xgrid, y0, y0 + d_n,
                         color=color, alpha=alpha_fill, linewidth=0)
        axV.plot(xgrid, y0 + d_n, color=color, lw=lw,
                 linestyle=ls, alpha=0.95)

axV.axvline(HH_AVG_MED, color=FAMILY_COLORS["Human"], lw=1.4, ls="-",  alpha=0.55)
axV.axvline(HH_MAX_MED, color=FAMILY_COLORS["Human"], lw=1.4, ls="--", alpha=0.65)

y_anno = (N_ROWS - 1) * ROW_HEIGHT + DENSITY_AMP + 0.20
axV.text(HH_AVG_MED - 0.012, y_anno, f"avg = {HH_AVG_MED:.2f}",
         ha="right", va="bottom", fontsize=ANNO_FS,
         color=FAMILY_COLORS["Human"], style="italic", fontweight="bold")
axV.text(HH_MAX_MED + 0.012, y_anno, f"max = {HH_MAX_MED:.2f}",
         ha="left", va="bottom", fontsize=ANNO_FS,
         color=FAMILY_COLORS["Human"], style="italic", fontweight="bold")

axV.set_yticks([y_of[r] + 0.18 for r in ROW_ORDER])
ylabels = [DISPLAY.get(r, r) for r in ROW_ORDER]
ylabel_colors = [FAMILY_COLORS["Human" if r == "Human" else family_of(r)]
                 for r in ROW_ORDER]
axV.set_yticklabels(ylabels, fontsize=YLABEL_FS)
for tick, c in zip(axV.get_yticklabels(), ylabel_colors):
    tick.set_color(c); tick.set_fontweight("bold")
axV.set_ylim(-0.4, (N_ROWS - 1) * ROW_HEIGHT + DENSITY_AMP + 1.05)
axV.set_xlim(-0.02, 1.0)
axV.set_xticks([0.0, 0.5, 1.0])
axV.tick_params(axis="x", labelsize=TICK_FS)
axV.set_xlabel(r"Per-question max head--human $r$", fontsize=AXIS_FS)
axV.grid(True, axis="x", alpha=0.20, lw=0.5)
axV.spines["top"].set_visible(False)
axV.spines["right"].set_visible(False)
axV.spines["left"].set_visible(False)


fig.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.13, wspace=0.45)

png = OUT / "fig5c_modelacc.png"
pdf = OUT / "fig5c_modelacc.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"saved {png}\nsaved {pdf}")
print(f"\nQ3-8B native: Pearson r={r_pearson:+.3f}  n={mask.sum()}")
print(f"Q3-8B per-q model accuracy: range [{acc.min():.2f}, {acc.max():.2f}]  "
      f"mean={acc.mean():.3f}")
