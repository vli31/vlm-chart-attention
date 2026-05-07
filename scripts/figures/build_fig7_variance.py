#!/usr/bin/env python3
"""Box-plot version of tab-3 (variance attribution).

Aggregates each variance-attribution column over the 12 VLMs (one box per column);
overlays a black star for the Human row when defined; annotates each column with
the corresponding coefficient from the *final joint* regression.

Coefficient definitions (per model, then mean ± SD across 12 models):
  - Continuous predictors (Image cplx., Q. # words, Gaze r, # Workers,
    Human acc., Human RT, Model acc.): standardized β in a joint OLS
    of y on ALL continuous predictors. Reported as β.
  - Categorical predictors (Image ID, Plot type, Q. cat.): partial R²
    (added R² when the block is included on top of the rest of the
    blocks already in the joint model). Image FE is excluded for the
    Plot type / Q. cat. partials because both are constant within an
    image and would be perfectly collinear.

Run with --from-cache to skip the 9-min regression and re-render the figure
from the saved JSON.

Outputs:
  ./final_paper_figures/tab3_boxplot.png
  ./final_paper_figures/tab3_boxplot.pdf
  ./final_paper_figures/tab3_boxplot_coefficients.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.transforms import ScaledTranslation
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--from-cache", action="store_true",
                help="Skip regressions; load coefficients from saved JSON.")
args_cli = ap.parse_args()

ROOT = Path("./")
DATA = ROOT / "per_sample_metrics_broad"
CEIL = ROOT / "human_human_ceiling" / "human_ceiling_t16_broad.csv"
HOMO = ROOT / "human_homogeneity_per_q.csv"
IMGC = ROOT / "image_complexity_cache.csv"
HLOO = ROOT / "human_human_ceiling" / "human_loo_per_worker.csv"
CORR_DIR = Path("./data/lvlm-chart/correctness")
OUT = ROOT / "final_paper_figures"
OUT.mkdir(exist_ok=True)
QC = json.loads((OUT / "broad_qc_ids.json").read_text())
QC_IDS = set(int(x) for x in QC["ids_in_qc_subset"])

MODEL_KEYS = [
    "2.5-3B", "2.5-7B",
    "2B", "4B", "8B",
    "internvl3-1b", "internvl3-2b", "internvl3-8b",
    "internvl3.5-1b", "internvl3.5-2b", "internvl3.5-4b", "internvl3.5-8b",
]
DISPLAY = {
    "2.5-3B": "Q2.5-3B", "2.5-7B": "Q2.5-7B",
    "2B": "Q3-2B", "4B": "Q3-4B", "8B": "Q3-8B",
    "internvl3-1b": "IV3-1B", "internvl3-2b": "IV3-2B", "internvl3-8b": "IV3-8B",
    "internvl3.5-1b": "IV3.5-1B", "internvl3.5-2b": "IV3.5-2B",
    "internvl3.5-4b": "IV3.5-4B", "internvl3.5-8b": "IV3.5-8B",
}
CORR_TOKEN = {
    "2.5-3B": "2.5-3b", "2.5-7B": "2.5-7b",
    "2B": "2b", "4B": "4b", "8B": "8b",
    "internvl3-1b":  "internvl3-1b-instruct",
    "internvl3-2b":  "internvl3-2b-instruct",
    "internvl3-8b":  "internvl3-8b-instruct",
    "internvl3.5-1b": "internvl3.5-1b-instruct",
    "internvl3.5-2b": "internvl3.5-2b-instruct",
    "internvl3.5-4b": "internvl3.5-4b-instruct",
    "internvl3.5-8b": "internvl3.5-8b-instruct",
}

CONT_KEYS = ["imgcplx", "qwords", "gaze", "nworkers", "hacc", "hrt", "macc"]
CAT_KEYS  = ["image", "plottype", "qcat"]
ALL_KEYS  = ["image", "plottype", "imgcplx", "qcat", "qwords",
             "gaze", "nworkers", "hacc", "hrt", "macc", "residual"]

CACHE_PATH = OUT / "tab3_boxplot_coefficients.json"


# ── compute_or_load_coefficients ────────────────────────────────────
def compute_coefficients():
    """Load data, fit joint regressions, aggregate, save cache, return dict."""

    def load_max_cc(mk):
        d = np.load(DATA / f"{mk}_16.npz")
        cc = d["cc"]; nw = d["n_workers"]; sids = d["sample_ids"].astype(int)
        cc_clean = np.where(np.isnan(cc), -np.inf, cc)
        max_cc = cc_clean.max(axis=1).astype(np.float32)
        bad = (nw < 1) | (~np.isfinite(max_cc))
        max_cc[bad] = np.nan
        return pd.DataFrame({"sample_idx": sids, mk: max_cc})

    print("loading max-head CC for 12 models ...", flush=True)
    df = None
    for mk in MODEL_KEYS:
        one = load_max_cc(mk)
        df = one if df is None else df.merge(one, on="sample_idx", how="outer")

    ceil = pd.read_csv(CEIL)[["sample_idx", "loo_cc", "image_type", "qcategory",
                              "image_name", "question_id"]].rename(columns={"loo_cc": "Human"})
    df = df.merge(ceil, on="sample_idx", how="inner")
    df = df.merge(pd.read_csv(IMGC)[["image_name", "frac_nonbg"]], on="image_name", how="left")
    homo = pd.read_csv(HOMO).assign(image_name=lambda x: x["image_id"].astype(str) + ".png")
    df = df.merge(homo[["image_name", "question_id", "p_correct",
                        "mean_duration_ms", "q_n_words"]],
                  on=["image_name", "question_id"], how="left")
    df = df.merge(pd.read_csv(HLOO)[["sample_idx", "mean_loo"]],
                  on="sample_idx", how="left")

    def find_corr_file(token):
        pat = f"correctness_{token}_salchartqa_x5_n5999_"
        cands = [p for p in CORR_DIR.iterdir()
                 if p.name.startswith(pat) and p.name.endswith("_fixed.json")]
        if not cands:
            cands = [p for p in CORR_DIR.iterdir()
                     if p.name.startswith(pat) and p.name.endswith(".json")
                     and "fixed" not in p.name]
        return sorted(cands)[-1] if cands else None

    def per_q_acc_map(corr_file):
        bd = json.loads(corr_file.read_text())
        return {r["sample_id"]: float(r["num_correct"]) / 5.0
                for r in bd["results"] if r.get("num_correct") is not None}

    for mk in MODEL_KEYS:
        cf = find_corr_file(CORR_TOKEN[mk])
        if cf is None:
            df[f"macc_{mk}"] = np.nan; continue
        acc_map = per_q_acc_map(cf)
        sids = (df["image_name"].astype(str).str.replace(".png", "", regex=False)
                + "_" + df["question_id"].astype(str))
        df[f"macc_{mk}"] = ("salchartqa_" + sids).map(acc_map)

    df = df[df["sample_idx"].isin(QC_IDS)].reset_index(drop=True)
    ENT = MODEL_KEYS + ["mean_loo"]
    FEAT_BASE = ["frac_nonbg", "p_correct", "mean_duration_ms", "q_n_words",
                 "image_type", "qcategory", "image_name", "Human"]
    MACC = [f"macc_{mk}" for mk in MODEL_KEYS]
    df = df.dropna(subset=ENT + FEAT_BASE + MACC).reset_index(drop=True)

    nw_one = np.load(DATA / f"{MODEL_KEYS[0]}_16.npz")
    nw_map = pd.DataFrame({"sample_idx": nw_one["sample_ids"].astype(int),
                           "n_workers": nw_one["n_workers"].astype(np.float64)})
    df = (df.merge(nw_map, on="sample_idx", how="left")
            .dropna(subset=["n_workers"]).reset_index(drop=True))
    N = len(df)
    print(f"  N = {N} samples after QC + NaN drop", flush=True)

    def one_hot(series):
        cats = pd.Categorical(series.astype(str))
        X = np.zeros((len(series), len(cats.categories)), dtype=np.float64)
        X[np.arange(len(series)), cats.codes] = 1.0
        return X[:, 1:]

    def col(name):
        return df[name].to_numpy(dtype=np.float64).reshape(-1, 1)

    img_x   = one_hot(df["image_name"])
    ptype_x = one_hot(df["image_type"])
    qcat_x  = one_hot(df["qcategory"])
    ic_x    = col("frac_nonbg")
    qw_x    = col("q_n_words")
    gaze_x  = col("Human")
    nw_x    = col("n_workers")
    hacc_x  = col("p_correct")
    hrt_x   = col("mean_duration_ms")

    print(f"  block widths: image={img_x.shape[1]} plottype={ptype_x.shape[1]} "
          f"qcat={qcat_x.shape[1]}", flush=True)

    def r2_alone(y0, sst, X):
        Xc = X - X.mean(axis=0, keepdims=True)
        beta, *_ = np.linalg.lstsq(Xc, y0, rcond=None)
        return float(((Xc @ beta) ** 2).sum()) / sst

    def r2_joint(y0, sst, blocks):
        X_all = np.concatenate(blocks, axis=1)
        Xc = X_all - X_all.mean(axis=0, keepdims=True)
        beta, *_ = np.linalg.lstsq(Xc, y0, rcond=None)
        return float(((Xc @ beta) ** 2).sum()) / sst

    CONT_COL_MAP = {"imgcplx": ic_x, "qwords": qw_x, "gaze": gaze_x,
                    "nworkers": nw_x, "hacc": hacc_x, "hrt": hrt_x}
    CONT_ORDER = ["imgcplx", "qwords", "gaze", "nworkers", "hacc", "hrt"]

    def standardized_beta_continuous(y0, sst, model_macc_x):
        cols = [CONT_COL_MAP[k] for k in CONT_ORDER] + [model_macc_x]
        X = np.concatenate(cols, axis=1)
        Xc = X - X.mean(axis=0, keepdims=True)
        sd_x = Xc.std(axis=0, ddof=0)
        sd_y = float(np.sqrt(sst / len(y0)))
        beta, *_ = np.linalg.lstsq(Xc, y0, rcond=None)
        beta_z = beta * sd_x / max(sd_y, 1e-12)
        return dict(zip(CONT_KEYS, beta_z.tolist()))

    def partial_r2_categorical(y0, sst, model_macc_x):
        cont = [CONT_COL_MAP[k] for k in CONT_ORDER] + [model_macc_x]
        out = {}
        full     = [img_x, ptype_x, qcat_x] + cont
        base     = [ptype_x, qcat_x] + cont
        full_pt  = [ptype_x, qcat_x] + cont
        base_pt  = [qcat_x] + cont
        full_qc  = [ptype_x, qcat_x] + cont
        base_qc  = [ptype_x] + cont
        out["image"]    = max(r2_joint(y0, sst, full)    - r2_joint(y0, sst, base),    0.0)
        out["plottype"] = max(r2_joint(y0, sst, full_pt) - r2_joint(y0, sst, base_pt), 0.0)
        out["qcat"]     = max(r2_joint(y0, sst, full_qc) - r2_joint(y0, sst, base_qc), 0.0)
        return out

    def standalone_all(y0, sst, model_macc_x):
        return {
            "image":    r2_alone(y0, sst, img_x),
            "plottype": r2_alone(y0, sst, ptype_x),
            "imgcplx":  r2_alone(y0, sst, ic_x),
            "qcat":     r2_alone(y0, sst, qcat_x),
            "qwords":   r2_alone(y0, sst, qw_x),
            "gaze":     r2_alone(y0, sst, gaze_x),
            "nworkers": r2_alone(y0, sst, nw_x),
            "hacc":     r2_alone(y0, sst, hacc_x),
            "hrt":      r2_alone(y0, sst, hrt_x),
            "macc":     r2_alone(y0, sst, model_macc_x),
            "residual": max(1.0 - r2_joint(y0, sst,
                                           [img_x, ptype_x, ic_x, qcat_x, qw_x,
                                            gaze_x, nw_x, hacc_x, hrt_x, model_macc_x]),
                            0.0),
        }

    print("\nfitting joint regressions per model ...", flush=True)
    beta_per, partR_per, alone_per = {}, {}, {}
    for mk in MODEL_KEYS:
        y = df[mk].to_numpy(dtype=np.float64)
        y0 = y - y.mean()
        sst = float((y0 * y0).sum())
        macc_x = col(f"macc_{mk}")
        beta_per[mk]  = standardized_beta_continuous(y0, sst, macc_x)
        partR_per[mk] = partial_r2_categorical(y0, sst, macc_x)
        alone_per[mk] = standalone_all(y0, sst, macc_x)
        print(f"  {DISPLAY[mk]:<10}", flush=True)

    def aggregate(per_model_dict, keys):
        out = {}
        for k in keys:
            vals = np.array([per_model_dict[mk][k] for mk in MODEL_KEYS])
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                      "per_model": {mk: float(per_model_dict[mk][k]) for mk in MODEL_KEYS}}
        return out

    beta_agg  = aggregate(beta_per,  CONT_KEYS)
    partR_agg = aggregate(partR_per, CAT_KEYS)

    # Human row standalone R²
    y_h = df["mean_loo"].to_numpy(dtype=np.float64); y_h0 = y_h - y_h.mean()
    sst_h = float((y_h0 * y_h0).sum())
    human = standalone_all(y_h0, sst_h, hacc_x)
    full_h = [img_x, ptype_x, ic_x, qcat_x, qw_x, nw_x, hacc_x, hrt_x]
    human["residual"] = max(1.0 - r2_joint(y_h0, sst_h, full_h), 0.0)
    human["gaze"] = None
    human["macc"] = None

    out_json = {
        "n_samples": int(N),
        "models": MODEL_KEYS,
        "continuous_standardized_beta": beta_agg,
        "categorical_partial_r2": partR_agg,
        "standalone_r2_per_model": alone_per,
        "human_standalone_r2": human,
    }
    CACHE_PATH.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"saved {CACHE_PATH}", flush=True)
    return out_json


def load_coefficients():
    return json.loads(CACHE_PATH.read_text())


# ── compute or load ─────────────────────────────────────────────────
if args_cli.from_cache:
    print("--from-cache: loading saved coefficients", flush=True)
    res = load_coefficients()
else:
    res = compute_coefficients()

N               = int(res["n_samples"])
beta_agg        = res["continuous_standardized_beta"]
partR_agg       = res["categorical_partial_r2"]
alone_per_model = {mk: {k: float(v) for k, v in d.items()}
                   for mk, d in res["standalone_r2_per_model"].items()}
human_alone     = res["human_standalone_r2"]


print("\n=== Final-regression coefficients (mean ± SD across 12 VLMs) ===", flush=True)
print("  Continuous (standardized β, joint OLS on 7 continuous predictors):")
for k, label in [("imgcplx", "Image cplx."), ("qwords", "Q. # words"),
                 ("gaze", "Gaze r"), ("nworkers", "# Workers"),
                 ("hacc", "Human acc."), ("hrt", "Human RT"),
                 ("macc", "Model acc.")]:
    a = beta_agg[k]
    print(f"    {label:<14} β = {a['mean']:+.3f} ± {a['std']:.3f}")
print("  Categorical (partial R², %, joint model — image FE excluded for "
      "plottype/qcat to avoid collinearity):")
for k, label in [("image", "Image ID"), ("plottype", "Plot type"),
                 ("qcat", "Q. cat.")]:
    a = partR_agg[k]
    print(f"    {label:<14} ΔR² = {a['mean']*100:5.2f} ± {a['std']*100:.2f} %")


# ── plotting ────────────────────────────────────────────────────────
columns = [
    "Image ID", "Inter-human\nGaze $r$", "Plot Type", "Q. Cat.", "Human Acc.",
    "Q. # Words", "# Workers", "Human RT", "Model Acc.", "All\nPredictors",
]
COL_KEYS_ORDER = [
    "image", "gaze", "plottype", "qcat", "hacc",
    "qwords", "nworkers", "hrt", "macc", "all",
]

GROUP_GREEN  = ("#a5d6a7", "#1b5e20")
GROUP_BLUE   = ("#90caf9", "#0d47a1")
GROUP_PURPLE = ("#ce93d8", "#4a148c")
GROUP_GREY   = ("#bdbdbd", "#212121")
group_for = {
    "Image ID":              GROUP_GREEN,
    "Plot Type":             GROUP_GREEN,
    "Q. Cat.":               GROUP_GREEN,
    "Q. # Words":            GROUP_GREEN,
    "Inter-human\nGaze $r$": GROUP_BLUE,
    "# Workers":             GROUP_BLUE,
    "Human Acc.":            GROUP_BLUE,
    "Human RT":              GROUP_BLUE,
    "Model Acc.":            GROUP_PURPLE,
    "All\nPredictors":       GROUP_GREY,
}


def col_value(per_model_block, k):
    """Standalone R² for each block; 'all' = 1 − residual (joint R² of all blocks)."""
    if k == "all":
        return (1.0 - float(per_model_block["residual"])) * 100.0
    return float(per_model_block[k]) * 100.0


data_matrix = np.array([
    [col_value(alone_per_model[mk], k) for k in COL_KEYS_ORDER]
    for mk in MODEL_KEYS
])
human_row = np.array([
    np.nan if (k != "all" and human_alone[k] is None)
    else col_value(human_alone, k)
    for k in COL_KEYS_ORDER
])

plt.rcParams.update({
    "font.size": 38,
    "axes.labelsize": 44,
    "xtick.labelsize": 38,
    "ytick.labelsize": 36,
})

fig = plt.figure(figsize=(30, 9))
# height_ratios chosen so the per-unit data scale matches across the broken
# axis: ax_top range = 17, ax_bot range = 34, ratio 1:2.
gs = fig.add_gridspec(
    2, 1,
    height_ratios=[1, 2],
    hspace=0.05,
    left=0.075, right=0.994, top=0.985, bottom=0.30,
)
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0])

positions = np.arange(1, len(columns) + 1)
rng = np.random.default_rng(seed=2)
DOT_COLOR = "#3b3b3b"


def draw(ax):
    box_data = [data_matrix[:, i] for i in range(len(columns))]
    bp = ax.boxplot(
        box_data, positions=positions, widths=0.66,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#000", linewidth=4.0),
        whiskerprops=dict(linewidth=2.4),
        capprops=dict(linewidth=2.4),
    )
    whisker_pairs = list(zip(bp["whiskers"][0::2], bp["whiskers"][1::2]))
    cap_pairs     = list(zip(bp["caps"][0::2],     bp["caps"][1::2]))
    for patch, whisks, caps, med, col_label in zip(
        bp["boxes"], whisker_pairs, cap_pairs, bp["medians"], columns
    ):
        fill, edge = group_for[col_label]
        patch.set_facecolor(fill); patch.set_alpha(0.55)
        patch.set_edgecolor(edge); patch.set_linewidth(3.0); patch.set_zorder(2)
        for w in whisks: w.set_color(edge); w.set_linewidth(2.4); w.set_zorder(2)
        for c in caps:   c.set_color(edge); c.set_linewidth(2.4); c.set_zorder(2)
        med.set_zorder(4)

    for j in range(len(columns)):
        x = positions[j] + rng.uniform(-0.16, 0.16, size=data_matrix.shape[0])
        ax.scatter(x, data_matrix[:, j], s=95, c=DOT_COLOR,
                   edgecolor="white", linewidth=1.0, alpha=0.85, zorder=3)

    for j in range(len(columns)):
        if not np.isnan(human_row[j]):
            ax.scatter(positions[j], human_row[j], marker="*",
                       s=1500, color="black",
                       edgecolor="white", linewidth=2.6, zorder=6)

    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.9)
    ax.set_axisbelow(True)


draw(ax_top)
draw(ax_bot)

# Symmetric per-unit scale across the broken axis: ax_top covers 17 units
# (67--84), ax_bot covers 34 units (-2--32), height_ratios 1:2.
# ax_bot ymax=32 keeps the "30" tick-label clear of the broken-axis break marks.
ax_top.set_ylim(67, 84)
ax_bot.set_ylim(-2.0, 32.0)
ax_top.set_yticks([70, 80])
ax_bot.set_yticks([0, 10, 20, 30])

ax_top.spines["bottom"].set_visible(False)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)

# Broken-axis break marks — slightly larger so they remain clearly visible
# now that the y-axis sits closer to the figure's left edge.
d = 0.014
kw = dict(color="k", clip_on=False, linewidth=2.4)
ax_top.plot([-d, +d], [-0.06, +0.06], transform=ax_top.transAxes, **kw)
ax_bot.plot([-d, +d], [1 - 0.02, 1 + 0.02], transform=ax_bot.transAxes, **kw)

ax_bot.set_xticks(positions)
ax_bot.set_xticklabels(columns, rotation=28, ha="right")
for tick_label, col_label in zip(ax_bot.get_xticklabels(), columns):
    tick_label.set_color(group_for[col_label][1])
    tick_label.set_fontweight("semibold")

dx_shift = ScaledTranslation(18 / 72.0, 0, fig.dpi_scale_trans)
for label in ax_bot.get_xticklabels():
    label.set_transform(label.get_transform() + dx_shift)

ax_bot.set_xlim(0.4, len(columns) + 0.6)

# Two-line y-axis label, rotated 90° via two separate fig.text calls so the
# lines sit tightly stacked (matplotlib's set_ylabel + "\n" with rotation=90
# spreads the lines too far apart). Placed close to the axis on the left.
LABEL_X_FIG = 0.014
fig.text(LABEL_X_FIG, 0.69, "Standalone $R^2$",
         rotation=90, ha="center", va="center", fontsize=38)
fig.text(LABEL_X_FIG + 0.020, 0.69, "(% variance)",
         rotation=90, ha="center", va="center", fontsize=38)

# ── legend (boxed, horizontal, near top of plot body) ───────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=DOT_COLOR, markeredgecolor="white",
           markersize=14, label="VLM"),
    Line2D([0], [0], marker="*", color="none",
           markerfacecolor="black", markeredgecolor="white",
           markeredgewidth=1.4, markersize=28, label="Human Baseline"),
]
# Place at the very top of the chart (ax_top, the broken-axis upper panel).
legend = ax_top.legend(
    handles=legend_handles,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.98),
    frameon=True,                   # default rounded fancybox + default grey edge
    facecolor="white",
    fontsize=34,
    handletextpad=0.4,
    columnspacing=1.6,
    borderpad=0.5,
)

out_base = OUT / "tab3_boxplot"
plt.savefig(str(out_base) + ".png", dpi=170, bbox_inches="tight",
            pad_inches=0.15, facecolor="white")
plt.savefig(str(out_base) + ".pdf", bbox_inches="tight",
            pad_inches=0.15, facecolor="white")
plt.close()
print(f"saved {out_base}.png\nsaved {out_base}.pdf")
