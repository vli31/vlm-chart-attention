#!/usr/bin/env python3
"""fig6 — same per-model dot plot of ablation accuracy as the legacy fig5b-3,
but with panels sorted left→right by descending baseline accuracy instead of
by head concentration.

Reuses the cached records in fig5b_data.json (produced by build_fig5_dot.py)
so we don't re-run the hierarchical bootstrap.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter, NullLocator

OUT = Path("./final_paper_figures")
records = json.load(open(OUT / "fig5b_data.json"))

ALL = [
    ("2.5-3B", "Q2.5-3B"), ("2.5-7B", "Q2.5-7B"),
    ("2B", "Q3-2B"), ("4B", "Q3-4B"), ("8B", "Q3-8B"),
    ("internvl3-1b-instruct", "IV3-1B"), ("internvl3-2b-instruct", "IV3-2B"),
    ("internvl3-8b-instruct", "IV3-8B"),
    ("internvl3.5-1b-instruct", "IV3.5-1B"), ("internvl3.5-2b-instruct", "IV3.5-2B"),
    ("internvl3.5-4b-instruct", "IV3.5-4B"), ("internvl3.5-8b-instruct", "IV3.5-8B"),
]
KINDS = [("baseline", "Base"), ("top5", "Top-5"), ("bottom5", "Bot-5"),
         ("random5_a", "Rnd A"), ("random5_b", "Rnd B")]
KIND_LONG = {"baseline": "Baseline", "top5": "Top-5", "bottom5": "Bottom-5",
             "random5_a": "Random A", "random5_b": "Random B"}
KIND_COLORS = {
    "baseline":  "#222222",
    "top5":      "#D62728",   # red
    "bottom5":   "#1F77B4",   # blue
    "random5_a": "#7F7F7F",   # grey
    "random5_b": "#BCBCBC",   # light grey
}
FAMILY_COLORS = {"Qwen2.5-VL": "#BB52A6", "Qwen3-VL": "#3540A8",
                 "InternVL3": "#D17B30", "InternVL3.5": "#A23E1A"}

def family_of(mk: str) -> str:
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"

# Drop the same 3 models excluded from the original fig6 (the three lowest-
# accuracy / least-concentrated models: IV3-1B, IV3-2B, IV3.5-1B).
EXCLUDE = {"internvl3-1b-instruct", "internvl3-2b-instruct", "internvl3.5-1b-instruct"}
USED = [(mk, disp) for mk, disp in ALL
        if mk not in EXCLUDE and records.get(mk, {}).get("baseline")]

# Sort by descending baseline accuracy.
USED_SORTED = sorted(USED, key=lambda x: -records[x[0]]["baseline"]["mean"])

n_kinds = len(KINDS)
xs = np.linspace(0, 1, n_kinds)
n_models = len(USED_SORTED)

fig, axes = plt.subplots(1, n_models, figsize=(2.60 * n_models + 2.4, 8.6),
                         sharey=True, sharex=True)
fig.subplots_adjust(wspace=0.10)

def _cond_records(mk):
    return [rec for rec in records[mk].values()
            if isinstance(rec, dict) and "lo" in rec]

g_pad = 0.02
global_lo = min(min(rec["lo"] for rec in _cond_records(mk)) for mk, _ in USED_SORTED)
global_hi = max(max(rec["hi"] for rec in _cond_records(mk)) for mk, _ in USED_SORTED)
g_bottom = math.floor(max(0.0, global_lo - g_pad) / 0.10) * 0.10
g_top    = math.ceil (min(1.0, global_hi + g_pad) / 0.10) * 0.10

for i, (mk, disp) in enumerate(USED_SORTED):
    ax = axes[i]
    means = [records[mk][k]["mean"] for k, _ in KINDS]
    los   = [records[mk][k]["lo"]   for k, _ in KINDS]
    his   = [records[mk][k]["hi"]   for k, _ in KINDS]
    cs    = [KIND_COLORS[k]         for k, _ in KINDS]

    # No connecting line between markers (per request).
    for x, m, lo, hi, c in zip(xs, means, los, his, cs):
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]],
                    fmt="none", ecolor="#222",
                    capsize=12.0, capthick=3.5, elinewidth=5.5, zorder=2)
        ax.scatter([x], [m], marker="o", s=220, c=c,
                   edgecolors="black", linewidths=1.0, zorder=3)
    base = records[mk]["baseline"]["mean"]
    ax.axhline(base, color="#222", ls="--", lw=0.9, alpha=0.40, zorder=0)

    fam_short = {"Qwen2.5-VL": "Q2.5-VL", "Qwen3-VL": "Q3-VL",
                 "InternVL3":  "IV3",     "InternVL3.5": "IV3.5"}[family_of(mk)]
    size_str = disp.split("-")[-1]
    ax.set_title(f"{fam_short}\n{size_str}",
                 color="#222", fontweight="bold", fontsize=38, pad=12)
    ax.set_xticks([])
    ax.set_xlim(-0.10, 1.10)
    ax.tick_params(axis="y", labelsize=38)
    ax.grid(True, axis="y", alpha=0.20, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylim(g_bottom, g_top)
g_span = g_top - g_bottom
g_step = 0.20 if g_span < 0.50 else (0.25 if g_span < 0.75 else 0.30)
axes[0].yaxis.set_major_locator(MultipleLocator(g_step))
axes[0].yaxis.set_minor_locator(NullLocator())
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
for a in axes:
    a.yaxis.set_minor_locator(NullLocator())
axes[0].set_ylabel("Accuracy", fontsize=42)

legend_handles = [Line2D([0], [0], marker="o", linestyle="none",
                         markerfacecolor=KIND_COLORS[k], markeredgecolor="black",
                         markersize=26, label=KIND_LONG[k])
                  for k, _ in KINDS]
fig.tight_layout(rect=(0, 0.18, 1, 1.0))
fig.legend(handles=legend_handles, loc="lower center", ncol=len(KINDS),
           frameon=True, fancybox=True,
           fontsize=38, bbox_to_anchor=(0.5, -0.01))

pdf = OUT / "fig6.pdf"
png = OUT / "fig6.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=200, bbox_inches="tight")
print(f"saved {pdf} and {png}")
print("Order (highest → lowest baseline accuracy):")
for mk, disp in USED_SORTED:
    print(f"  {disp:10s}  baseline={records[mk]['baseline']['mean']:.4f}")
