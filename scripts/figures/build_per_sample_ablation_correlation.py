#!/usr/bin/env python3
"""For each (model, ablation kind):
  - Per-sample paired Δaccuracy
  - Per-sample alignment as (a) max over ALL heads, (b) max over ABLATED heads
  - Pearson r(Δ, alignment) for both, with bootstrap 95% CI

Tests whether the samples that lose accuracy under ablation are the same ones
where the ablated heads are well-aligned with human gaze on that sample.

Writes data to fig5_data.json and produces fig5b.pdf (a 2-panel companion to
fig5: per-sample correlation r with bootstrap CI per model).
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("./results/correctness")
ALN = Path(".//per_sample_metrics_broad")
IQ  = Path("./data/lvlm-chart/SalChartQA/image_questions.json")
OUT = Path("./final_paper_figures")

ALL_MODELS = [
    "2.5-3B", "2.5-7B",
    "2B", "4B", "8B",
    "internvl3-1b-instruct", "internvl3-2b-instruct", "internvl3-8b-instruct",
    "internvl3.5-1b-instruct", "internvl3.5-2b-instruct",
    "internvl3.5-4b-instruct", "internvl3.5-8b-instruct",
]
DISPLAY = {
    "2.5-3B":"Q2.5-3B", "2.5-7B":"Q2.5-7B",
    "2B":"Q3-2B", "4B":"Q3-4B", "8B":"Q3-8B",
    "internvl3-1b-instruct":"IV3-1B", "internvl3-2b-instruct":"IV3-2B",
    "internvl3-8b-instruct":"IV3-8B",
    "internvl3.5-1b-instruct":"IV3.5-1B", "internvl3.5-2b-instruct":"IV3.5-2B",
    "internvl3.5-4b-instruct":"IV3.5-4B", "internvl3.5-8b-instruct":"IV3.5-8B",
}
ALN_KEY = {
    "internvl3-1b-instruct":"internvl3-1b","internvl3-2b-instruct":"internvl3-2b",
    "internvl3-8b-instruct":"internvl3-8b",
    "internvl3.5-1b-instruct":"internvl3.5-1b","internvl3.5-2b-instruct":"internvl3.5-2b",
    "internvl3.5-4b-instruct":"internvl3.5-4b","internvl3.5-8b-instruct":"internvl3.5-8b",
}
KINDS = [("top5","Top-5"), ("bottom5","Bot-5"),
         ("random5_a","Rnd A"), ("random5_b","Rnd B")]
KIND_COLORS = {"top5":"#D62728","bottom5":"#9467BD","random5_a":"#7F7F7F","random5_b":"#BCBCBC"}
KIND_MARKERS = {"top5":"D","bottom5":"s","random5_a":"o","random5_b":"^"}
FAMILY_COLORS = {"Qwen2.5-VL":"#BB52A6","Qwen3-VL":"#3540A8",
                 "InternVL3":"#D17B30","InternVL3.5":"#A23E1A"}
def family_of(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"

RNG = np.random.default_rng(42)
N_BOOT = 1000

# build sample_id (string) -> canonical idx (int) map
_iq = json.load(open(IQ))
SID_TO_IDX = {}
i = 0
for img, qd in _iq.items():
    stem = img[:-4] if img.endswith(".png") else img
    for q in qd:
        SID_TO_IDX[f"salchartqa_{stem}_{q}"] = i
        i += 1


def latest_canonical(kind, mk):
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
            return (f, d, "all_tokens")
    for ts, f, d, n in sorted(cands, reverse=True):
        if n == 4556 and d.get("ablation_type") == "dataset_mean_replace_pre_output_projection":
            return (f, d, d.get("ablation_scope"))
    for ts, f, d, n in sorted(cands, reverse=True):
        if n == 4556:
            return (f, d, d.get("ablation_scope") + "*")
    ts, f, d, n = max(cands, key=lambda x: (x[3], x[0]))
    return (f, d, f"{d.get('ablation_scope')}*N={n}")


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

def boot_pearson(x, y, n_boot=N_BOOT, ci=95):
    """Bootstrap CI for Pearson r."""
    n = len(x)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        a, b = x[idx], y[idx]
        a = a - a.mean(); b = b - b.mean()
        d = float(np.sqrt((a*a).sum() * (b*b).sum()))
        rs[i] = (a*b).sum() / d if d > 0 else 0.0
    a, b = x - x.mean(), y - y.mean()
    d = float(np.sqrt((a*a).sum() * (b*b).sum()))
    r = float((a*b).sum() / d) if d > 0 else 0.0
    lo, hi = np.percentile(rs, [(100-ci)/2, 100 - (100-ci)/2])
    return r, float(lo), float(hi)

def boot_mean(x, n_boot=N_BOOT, ci=95):
    n = len(x)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        out[i] = x[idx].mean()
    lo, hi = np.percentile(out, [(100-ci)/2, 100 - (100-ci)/2])
    return float(x.mean()), float(lo), float(hi)


# ── compute ──
records = {}
for mk in ALL_MODELS:
    aln_key = ALN_KEY.get(mk, mk)
    p_aln = ALN / f"{aln_key}_16.npz"
    if not p_aln.exists():
        print(f"{mk}: alignment file missing"); continue
    z = np.load(p_aln)
    cc = z["cc"]                     # [N_aln, n_heads_total]
    sids_aln = z["sample_ids"].astype(int)
    n_layers, n_heads = int(z["n_layers"]), int(z["n_heads"])
    aln_max_all = np.where(np.isnan(cc), -np.inf, cc).max(axis=1)
    aln_idx = {int(sid): int(i) for i, sid in enumerate(sids_aln)}

    records[mk] = {}
    for kind, _ in KINDS:
        pick = latest_canonical(kind, mk)
        if pick is None: continue
        f, d, scope_used = pick
        ab_per = per_sample_acc(d["ablated"]["results"])
        bs_per = load_baseline(d)
        if bs_per is None: continue
        # per-sample, in ablation-eval order
        sids_eval = [r["sample_id"] for r in d["ablated"]["results"]]
        diffs, maxall, maxabl, base_v = [], [], [], []
        # ablated head indices (flattened)
        sel = d.get("selected_heads", [])
        sel_idx = [int(s["layer"]) * n_heads + int(s["head"]) for s in sel]
        for sid in sids_eval:
            if sid not in bs_per or sid not in ab_per: continue
            sidx = SID_TO_IDX.get(sid)
            if sidx is None or sidx not in aln_idx: continue
            row = aln_idx[sidx]
            mall = aln_max_all[row]
            if not np.isfinite(mall): continue
            head_vals = cc[row, sel_idx]
            head_vals = head_vals[np.isfinite(head_vals)]
            if len(head_vals) == 0: continue
            mabl = float(head_vals.max())
            diffs.append(ab_per[sid] - bs_per[sid])
            maxall.append(float(mall))
            maxabl.append(mabl)
            base_v.append(bs_per[sid])
        if len(diffs) < 50:
            print(f"{mk:28s} {kind:10s} too few paired samples ({len(diffs)})"); continue
        diffs = np.array(diffs); maxall = np.array(maxall); maxabl = np.array(maxabl)

        m_d, m_lo, m_hi = boot_mean(diffs * 100)
        r_all, ra_lo, ra_hi = boot_pearson(maxall, diffs)
        r_abl, rb_lo, rb_hi = boot_pearson(maxabl, diffs)
        records[mk][kind] = {
            "n": int(len(diffs)),
            "scope_used": scope_used,
            "baseline_mean": float(np.mean(base_v)),
            "mean_delta_pp": m_d, "ci95_lo_pp": m_lo, "ci95_hi_pp": m_hi,
            "r_max_all":      {"r": r_all, "ci_lo": ra_lo, "ci_hi": ra_hi},
            "r_max_ablated":  {"r": r_abl, "ci_lo": rb_lo, "ci_hi": rb_hi},
        }
        print(f"{mk:28s} {kind:10s} n={len(diffs):4d} {scope_used:>14s} "
              f"Δ={m_d:+6.2f}pp [{m_lo:+5.2f},{m_hi:+5.2f}]  "
              f"r_all={r_all:+.3f} [{ra_lo:+.3f},{ra_hi:+.3f}]  "
              f"r_abl={r_abl:+.3f} [{rb_lo:+.3f},{rb_hi:+.3f}]")

(OUT / "fig5_data.json").write_text(json.dumps(records, indent=2))
print(f"wrote {OUT / 'fig5_data.json'}")


# ── replot fig5: 2 panels (left = mean Δ, right = r over ablated heads) ──
def baseline_for(mk):
    for k, _ in KINDS:
        if k in records.get(mk, {}):
            return records[mk][k]["baseline_mean"]
    return 0.0
sorted_models = sorted([m for m in ALL_MODELS if records.get(m)], key=baseline_for)
n_models = len(sorted_models)
y_pos = np.arange(n_models)[::-1]
offs = np.linspace(-0.30, 0.30, len(KINDS))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.55 * n_models + 1.6),
                         gridspec_kw={"width_ratios": [1.4, 1.0]}, sharey=True)
axL, axR = axes

# left: Δaccuracy
for ki, (kind, label) in enumerate(KINDS):
    xs, lo, hi, ys = [], [], [], []
    for i, mk in enumerate(sorted_models):
        rec = records[mk].get(kind);
        if rec is None: continue
        xs.append(rec["mean_delta_pp"]); lo.append(rec["ci95_lo_pp"])
        hi.append(rec["ci95_hi_pp"]); ys.append(y_pos[i] + offs[ki])
    xs = np.array(xs); lo = np.array(lo); hi = np.array(hi); ys = np.array(ys)
    axL.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt="none",
                 ecolor=KIND_COLORS[kind], lw=1.2, capsize=2.5, alpha=0.85, zorder=2)
    axL.scatter(xs, ys, marker=KIND_MARKERS[kind], s=42,
                c=KIND_COLORS[kind], edgecolors="black", lw=0.5,
                label=label, zorder=3)
axL.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
axL.set_xlabel("Δ accuracy (pp), ablated − baseline", fontsize=11)
axL.set_title("Mean Δaccuracy (95% CI, paired)", fontsize=11.5)
axL.grid(True, axis="x", alpha=0.25, lw=0.5)
axL.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=9.5,
           ncol=2, title="Ablated head set", title_fontsize=9.5)

# right: r(Δ, max-ablated-head CC) per sample
for ki, (kind, label) in enumerate(KINDS):
    xs, lo, hi, ys = [], [], [], []
    for i, mk in enumerate(sorted_models):
        rec = records[mk].get(kind)
        if rec is None: continue
        ra = rec["r_max_ablated"]
        xs.append(ra["r"]); lo.append(ra["ci_lo"]); hi.append(ra["ci_hi"])
        ys.append(y_pos[i] + offs[ki])
    xs = np.array(xs); lo = np.array(lo); hi = np.array(hi); ys = np.array(ys)
    axR.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt="none",
                 ecolor=KIND_COLORS[kind], lw=1.2, capsize=2.5, alpha=0.85, zorder=2)
    axR.scatter(xs, ys, marker=KIND_MARKERS[kind], s=42,
                c=KIND_COLORS[kind], edgecolors="black", lw=0.5, zorder=3)
axR.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
axR.set_xlabel(r"per-sample $r$(Δ, max CC of ablated heads)", fontsize=11)
axR.set_title("Per-sample alignment-vs-Δ correlation (95% CI)", fontsize=11.5)
axR.grid(True, axis="x", alpha=0.25, lw=0.5)

# y labels (left axis only) with family colours
axL.set_yticks(y_pos)
axL.set_yticklabels([DISPLAY[mk] for mk in sorted_models], fontsize=10.5)
for tick, mk in zip(axL.get_yticklabels(), sorted_models):
    tick.set_color(FAMILY_COLORS[family_of(mk)])
    tick.set_fontweight("bold")
# baseline annotation on right side of the right panel
for i, mk in enumerate(sorted_models):
    base = baseline_for(mk)
    axR.text(1.02, y_pos[i], f"{base:.2f}",
             transform=axR.get_yaxis_transform(), ha="left", va="center",
             fontsize=9, color="#555")
axR.text(1.02, n_models - 0.4, "baseline", transform=axR.get_yaxis_transform(),
         ha="left", va="center", fontsize=8.5, color="#444", style="italic")

for ax in (axL, axR):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.6, n_models - 0.4)

fig.suptitle("Fig 5 — Head-set ablation impact on SalChartQA accuracy (n${=}4556$, paired)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT / "fig5.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig5.pdf", bbox_inches="tight")
print(f"saved {OUT / 'fig5.png'} and {OUT / 'fig5.pdf'}")
