#!/usr/bin/env python3
"""Variance-decomposition / prediction results (analogue of fig4 + tab-3) for
the other saliency metrics (CC, SIM, NSS, AUC, KL) at both 16x16 and native
attention-grid resolution.

For each (metric, resolution):
  - Per-model max-metric per sample, from per_sample_metrics_5way/
  - Joined with: human leave-one-out CC ceiling, image complexity, plot type,
    question category, # words in question, # workers / sample, human
    accuracy on Q, human response time
  - QC subset (n=4556 ids_in_qc_subset; final n drops slightly after NaN
    feature filtering)
  - Standalone R² per block (image identity, plot type, image complexity,
    q-category, q # words, gaze CC, n workers, human acc, human RT) and joint
    residual = 1 - R²(all 9 blocks together).
  - Cross-model correlation matrix (12 VLMs + Human gaze-CC ceiling).

Outputs:
  final_paper_figures/fig4_other_metrics/data_{metric}_{res}.json
  final_paper_figures/fig4_other_metrics/tab3_{metric}_{res}.tex
  final_paper_figures/fig4_other_metrics/summary.json   (compact view of all)

Notes:
  - The "Human" gaze entity uses loo_cc for *every* metric (since loo_NSS etc.
    are not precomputed). It represents inter-human gaze agreement.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

ROOT = Path("./")
DATA_5WAY = ROOT / "per_sample_metrics_5way"
CEIL = ROOT / "human_human_ceiling" / "human_ceiling_t16_broad.csv"
HOMO = ROOT / "human_homogeneity_per_q.csv"
IMGC = ROOT / "image_complexity_cache.csv"
IQ   = Path("./data/lvlm-chart/SalChartQA/image_questions.json")
QC_PATH = ROOT / "final_paper_figures" / "broad_qc_ids.json"
OUT = ROOT / "final_paper_figures" / "fig4_other_metrics"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_KEYS = [
    "2.5-3B", "2.5-7B",
    "2B", "4B", "8B",
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
GROUP_AFTER = {"2.5-7B", "8B", "internvl3-8b"}
FAMILY_COLORS = {
    "Qwen2.5-VL":  "BB52A6",
    "Qwen3-VL":    "3540A8",
    "InternVL3":   "D17B30",
    "InternVL3.5": "A23E1A",
}
def family_of(mk):
    if mk.startswith("internvl3.5"): return "InternVL3.5"
    if mk.startswith("internvl3"):   return "InternVL3"
    if mk.startswith("2.5-"):        return "Qwen2.5-VL"
    return "Qwen3-VL"


METRICS = ["cc", "sim", "nss", "auc", "kl"]
METRIC_DISPLAY = {"cc": "CC", "sim": "SIM", "nss": "NSS", "auc": "AUC",
                  "kl": "KL"}
RESOLUTIONS = ["16", "native"]


def load_max_metric(model_key: str, res: str, metric: str, qc_ids):
    """Return DataFrame with columns sample_idx, <model_key>=per-question
    best-aligned head under the given metric. For KL (lower = better) we
    take the per-sample MIN over heads; for the other metrics (higher =
    better) we take the per-sample MAX."""
    p = DATA_5WAY / f"{model_key}_{res}.npz"
    d = np.load(p)
    arr = d[metric].astype(np.float64)
    nw = d["n_workers"]
    sids = d["sample_ids"].astype(int)
    if metric == "kl":
        # lower KL = better-aligned head; use min over heads, NaN-safe.
        arr_clean = np.where(np.isnan(arr), np.inf, arr)
        best_v = arr_clean.min(axis=1).astype(np.float32)
        bad = (nw < 1) | (~np.isfinite(best_v))
    else:
        arr_clean = np.where(np.isnan(arr), -np.inf, arr)
        best_v = arr_clean.max(axis=1).astype(np.float32)
        bad = (nw < 1) | (~np.isfinite(best_v))
    best_v[bad] = np.nan
    out = pd.DataFrame({"sample_idx": sids, model_key: best_v,
                        f"{model_key}__nw": nw.astype(np.float64)})
    # restrict to QC subset
    return out[out["sample_idx"].isin(qc_ids)].reset_index(drop=True)


def one_hot(series: pd.Series) -> np.ndarray:
    cats = pd.Categorical(series.astype(str))
    X = np.zeros((len(series), len(cats.categories)), dtype=np.float64)
    X[np.arange(len(series)), cats.codes] = 1.0
    return X[:, 1:]


def residualize_by_group(arr, codes, n_groups):
    if arr.ndim == 1:
        arr = arr[:, None]; squeeze = True
    else:
        squeeze = False
    out = arr.copy()
    sums = np.zeros((n_groups, arr.shape[1]), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, codes, arr)
    np.add.at(counts, codes, 1)
    means = sums / np.maximum(counts, 1)[:, None]
    out -= means[codes]
    return out[:, 0] if squeeze else out


def r2_alone_centered(y0, sst, X):
    Xc = X - X.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(Xc, y0, rcond=None)
    return float(((Xc @ beta) ** 2).sum()) / sst


def r2_full(y0, sst, X_concat):
    """R² of y on all blocks together (image identity already absorbed
    elsewhere: this version takes y *centered* and X_concat *centered*)."""
    Xc = X_concat - X_concat.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(Xc, y0, rcond=None)
    return float(((Xc @ beta) ** 2).sum()) / sst


# ── load shared features once ─────────────────────────────────────
print("loading QC ids ...")
qc_ids = set(int(x) for x in json.loads(QC_PATH.read_text())["ids_in_qc_subset"])
print(f"  QC subset: n={len(qc_ids)}")

print("loading human ceiling ...")
ceil = pd.read_csv(CEIL)
ceil_use = ceil[["sample_idx", "loo_cc", "image_type", "qcategory",
                 "image_name", "question_id"]].rename(columns={"loo_cc": "Human"})

print("loading image complexity ...")
imgc = pd.read_csv(IMGC)[["image_name", "frac_nonbg"]]

print("loading human task metrics ...")
homo = pd.read_csv(HOMO)
homo = homo.assign(image_name=homo["image_id"].astype(str) + ".png")
homo_use = homo[["image_name", "question_id", "p_correct",
                 "mean_duration_ms", "q_n_words"]]


# ── per (metric, resolution) prediction analysis ───────────────────
COMP_TERMS = [
    "more", "less", "greater", "smaller", "fewer", "highest", "lowest",
    "than", "most", "least", "max", "maximum", "min", "minimum",
    "compare", "difference", "between", "exceed", "above", "below",
    "larger", "smallest", "biggest",
]

ALONE_KEYS = [
    "image_alone",          # image identity (absorbed by demeaning)
    "plottype_alone",
    "imgcomplexity_alone",
    "qcat_alone",
    "qcomplexity_alone",
    "gaze_alone",
    "nworkers_alone",
    "hacc_alone",
    "hrt_alone",
    "residual_full",
]

VAR_COLS = [
    ("image_alone",          r"\makecell{Image\\ID}",            ("Greens",  0.03, 0.45)),
    ("plottype_alone",       r"\makecell{Plot\\type}",           ("Greens",  0.03, 0.45)),
    ("imgcomplexity_alone",  r"\makecell{Image\\cplx.}",         ("Greens",  0.03, 0.45)),
    ("qcat_alone",           r"\makecell{Q.\\cat.}",             ("Greens",  0.03, 0.45)),
    ("qcomplexity_alone",    r"\makecell{Q.\\\#\,words}",        ("Greens",  0.03, 0.45)),
    ("gaze_alone",           r"\makecell{Gaze\\CC}",             ("Blues",   0.03, 0.45)),
    ("nworkers_alone",       r"\makecell{\#\,Workers\\/\,sample}",("Blues",   0.03, 0.45)),
    ("hacc_alone",           r"\makecell{Human\\acc.}",          ("Blues",   0.03, 0.45)),
    ("hrt_alone",            r"\makecell{Human\\RT}",            ("Blues",   0.03, 0.45)),
    ("residual_full",        r"\makecell{Resid.\\(unexpl.)}",    ("Greys",   0.02, 0.40)),
]
GROUP_HEADERS = [
    (5, r"\textbf{Image / question level (\% var.)}"),
    (4, r"\textbf{Human-side (\% var.)}"),
    (1, r""),
]
CMAPS = {name: matplotlib.colormaps[name] for name in ("Greens", "Blues", "Greys", "Purples")}


def cell_hex(v, lo_v, hi_v, cmap_tuple):
    name, cmap_lo, cmap_hi = cmap_tuple
    if hi_v == lo_v:
        norm = 0.5
    else:
        norm = (v - lo_v) / (hi_v - lo_v)
    pos = cmap_lo + norm * (cmap_hi - cmap_lo)
    r, g, b, _ = CMAPS[name](pos)
    return f"{int(round(r*255)):02X}{int(round(g*255)):02X}{int(round(b*255)):02X}"


def render_table(metric, res, n_samples, alone_shares):
    var_matrix = {mk: {key: alone_shares[mk][key] * 100.0 for key, _, _ in VAR_COLS}
                  for mk in MODEL_KEYS}
    var_minmax = {key: (min(var_matrix[mk][key] for mk in MODEL_KEYS),
                        max(var_matrix[mk][key] for mk in MODEL_KEYS))
                  for key, _, _ in VAR_COLS}

    def fmt_var_cell(mk, key, cmap_tuple):
        v = var_matrix[mk][key]
        lo, hi = var_minmax[key]
        return f"\\cellcolor[HTML]{{{cell_hex(v, lo, hi, cmap_tuple)}}}{v:.1f}"

    lines = []
    for mk in MODEL_KEYS:
        fam = family_of(mk)
        label = f"\\textcolor[HTML]{{{FAMILY_COLORS[fam]}}}{{{MODEL_LABELS[mk]}}}"
        cells = [fmt_var_cell(mk, key, cmap) for key, _, cmap in VAR_COLS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if mk in GROUP_AFTER:
            lines.append(r"\hline")

    sub_hdr = " & ".join([f"\\textbf{{{h}}}" for _, h, _ in VAR_COLS])
    group_cells = []
    n_so_far = 0
    for n, label in GROUP_HEADERS:
        n_so_far += n
        sep = "|" if n_so_far < sum(g[0] for g in GROUP_HEADERS) else ""
        group_cells.append(f"\\multicolumn{{{n}}}{{c{sep}}}{{{label}}}")
    group_hdr = " & ".join(group_cells)
    col_spec = "l|rrrrr|rrrr|r"

    metric_disp = METRIC_DISPLAY[metric]
    res_disp = "16{$\\times$}16 grid" if res == "16" else "native attention grid"

    best_word = "min" if metric == "kl" else "max"
    return r"""% Preamble:  \usepackage{colortbl}\usepackage[table]{xcolor}\usepackage{multirow}\usepackage{makecell}
% Variance attribution for """ + best_word + r"""-head """ + metric_disp + r""" at """ + res_disp + r""".
% Companion to tab-3 (CC, 16x16) — same conventions, different metric/resolution.
\begin{table}[ht]
\centering
\caption{Per-sample """ + best_word + r"""-head/human-gaze """ + metric_disp + r""" (""" + res_disp + r""", n${=}""" + str(n_samples) + r"""$): variance attribution. Each cell is the standalone $R^2$ (\% variance explained by that block alone); columns don't sum to 100 because blocks are correlated. \textbf{Resid.}~$=1-R^2$(all blocks jointly). Cell shading is per-column min--max with separate colormaps per conceptual group: image-level (Greens), human-side (Blues), residual (Greys).}
\label{tab:variance_""" + f"{metric}_{res}" + r"""}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\hline
\multirow{2}{*}{\textbf{Model}} & """ + group_hdr + r""" \\
\cline{2-11}
 & """ + sub_hdr + r""" \\
\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}%
}
\end{table}
"""


def run_one(metric: str, res: str):
    print(f"\n=== {metric.upper()} @ {res} ===", flush=True)

    # Load all 12 models' max-metric
    df = None
    for mk in MODEL_KEYS:
        one = load_max_metric(mk, res, metric, qc_ids)
        # Drop the helper nw column from the merge to avoid duplicates
        if df is None:
            df = one[["sample_idx", mk, f"{mk}__nw"]].rename(
                columns={f"{mk}__nw": "n_workers"})
        else:
            df = df.merge(one[["sample_idx", mk]], on="sample_idx", how="outer")
    print(f"  merged 12-model rows: {len(df)}")

    df = df.merge(ceil_use, on="sample_idx", how="inner")
    df = df.merge(imgc, on="image_name", how="left")
    df = df.merge(homo_use, on=["image_name", "question_id"], how="left")

    ENT = MODEL_KEYS + ["Human"]
    FEAT = ["frac_nonbg", "p_correct", "mean_duration_ms", "q_n_words",
            "image_type", "qcategory", "image_name"]
    df_use = df.dropna(subset=ENT + FEAT).reset_index(drop=True)
    N = len(df_use)
    print(f"  rows after drop NaN: {N}")

    # ─── Correlation matrix (13 entities) ───────────────────────────
    M = np.stack([df_use[k].to_numpy(dtype=np.float64) for k in ENT], axis=1)
    Mc = M - M.mean(axis=0, keepdims=True)
    Mn = Mc / (np.linalg.norm(Mc, axis=0, keepdims=True) + 1e-12)
    R = Mn.T @ Mn

    HUMAN_IDX = ENT.index("Human")
    r_with_human = R[:, HUMAN_IDX]
    r_other_mean = np.zeros(len(MODEL_KEYS))
    r_other_se   = np.zeros(len(MODEL_KEYS))
    for k in range(len(MODEL_KEYS)):
        others = [R[k, j] for j in range(len(MODEL_KEYS)) if j != k]
        arr = np.array(others)
        r_other_mean[k] = arr.mean()
        r_other_se[k]   = arr.std(ddof=1) / np.sqrt(len(arr))

    # ─── Standalone R² per predictor block ──────────────────────────
    img_codes = pd.Categorical(df_use["image_name"]).codes.astype(np.int64)
    img_n_groups = int(img_codes.max()) + 1

    def col_resid(arr):
        return residualize_by_group(arr, img_codes, img_n_groups)

    gaze_x   = df_use["Human"].to_numpy(dtype=np.float64).reshape(-1, 1)
    nwork_x  = df_use["n_workers"].to_numpy(dtype=np.float64).reshape(-1, 1)
    hacc_x   = df_use["p_correct"].to_numpy(dtype=np.float64).reshape(-1, 1)
    hrt_x    = df_use["mean_duration_ms"].to_numpy(dtype=np.float64).reshape(-1, 1)
    qcat_x   = one_hot(df_use["qcategory"])
    ptype_x  = one_hot(df_use["image_type"])
    imgc_x   = df_use["frac_nonbg"].to_numpy(dtype=np.float64).reshape(-1, 1)
    qn_x     = df_use["q_n_words"].to_numpy(dtype=np.float64).reshape(-1, 1)

    # within-image residualized blocks
    BLOCKS_R = [col_resid(b) for b in
                (gaze_x, nwork_x, hacc_x, hrt_x, qcat_x, qn_x)]
    # Plot type and image complexity are constant within image — use raw
    # one-hot / scalar for "alone" R² but they don't enter the joint
    # within-image residual fit (image identity already absorbs them).
    # To compute residual_full, we want R² when EVERY block is in jointly.
    # Joint design = image one-hot + plot type one-hot + image complexity
    # + qcat + qn + gaze + nworkers + hacc + hrt. Equivalent to:
    #   image absorbs first → then residualize y by image → fit on within-
    #   image-residualized {qcat, qn, gaze, nworkers, hacc, hrt} jointly.
    # plot type and image cplx are fully absorbed by image identity, so
    # adding them adds 0 explained var → joint r² = image_var + within-img r².

    alone_shares = {}
    image_absorbed = {}
    for mk in MODEL_KEYS:
        y = df_use[mk].to_numpy(dtype=np.float64)
        y0 = y - y.mean()
        sst = float((y0 * y0).sum())
        yr = col_resid(y0)
        ss_within = float((yr * yr).sum())
        r2_img = 1.0 - ss_within / sst
        image_absorbed[mk] = r2_img

        r2_plottype = r2_alone_centered(y0, sst, ptype_x)
        r2_imgc     = r2_alone_centered(y0, sst, imgc_x)
        r2_qcat     = r2_alone_centered(y0, sst, qcat_x)
        r2_qn       = r2_alone_centered(y0, sst, qn_x)
        r2_gaze     = r2_alone_centered(y0, sst, gaze_x)
        r2_nw       = r2_alone_centered(y0, sst, nwork_x)
        r2_hacc     = r2_alone_centered(y0, sst, hacc_x)
        r2_hrt      = r2_alone_centered(y0, sst, hrt_x)

        # Joint R² with all blocks (image identity + 5 within-image blocks)
        X_within_all = np.concatenate(BLOCKS_R, axis=1)
        beta, *_ = np.linalg.lstsq(X_within_all, yr, rcond=None)
        ss_within_explained = float(((X_within_all @ beta) ** 2).sum())
        ss_image = sst - ss_within
        r2_full_all = (ss_image + ss_within_explained) / sst
        residual_full = max(1.0 - r2_full_all, 0.0)

        alone_shares[mk] = {
            "image_alone":         r2_img,
            "plottype_alone":      r2_plottype,
            "imgcomplexity_alone": r2_imgc,
            "qcat_alone":          r2_qcat,
            "qcomplexity_alone":   r2_qn,
            "gaze_alone":          r2_gaze,
            "nworkers_alone":      r2_nw,
            "hacc_alone":          r2_hacc,
            "hrt_alone":           r2_hrt,
            "residual_full":       residual_full,
        }
        print(f"  {mk:18s} img={r2_img*100:4.1f} pt={r2_plottype*100:4.1f} "
              f"imgc={r2_imgc*100:4.1f} qcat={r2_qcat*100:4.1f} "
              f"qn={r2_qn*100:4.1f} gaze={r2_gaze*100:4.1f} "
              f"nw={r2_nw*100:4.2f} hacc={r2_hacc*100:4.2f} "
              f"hrt={r2_hrt*100:4.2f} resid={residual_full*100:4.1f}")

    # ─── Save data + LaTeX table ─────────────────────────────────────
    summary = {
        "metric": metric, "resolution": res,
        "n_samples_used": int(N),
        "labels": [MODEL_LABELS[m] for m in MODEL_KEYS] + ["Human"],
        "correlation_matrix": R.tolist(),
        "components_alone": ALONE_KEYS,
        "variance_alone": {mk: {k: float(v) for k, v in alone_shares[mk].items()}
                           for mk in MODEL_KEYS},
        "image_absorbed": {mk: float(image_absorbed[mk]) for mk in MODEL_KEYS},
        "corr_with_human_loo": {mk: float(R[i, HUMAN_IDX])
                                for i, mk in enumerate(MODEL_KEYS)},
        "corr_with_other_models": {mk: {"mean": float(r_other_mean[i]),
                                        "se":   float(r_other_se[i])}
                                   for i, mk in enumerate(MODEL_KEYS)},
    }
    out_json = OUT / f"data_{metric}_{res}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {out_json}")

    tex = render_table(metric, res, N, alone_shares)
    out_tex = OUT / f"tab3_{metric}_{res}.tex"
    out_tex.write_text(tex)
    print(f"  saved {out_tex}")

    return summary


def main():
    all_summaries = {}
    for metric in METRICS:
        for res in RESOLUTIONS:
            s = run_one(metric, res)
            all_summaries[f"{metric}_{res}"] = {
                "n_samples_used": s["n_samples_used"],
                "variance_alone_mean_over_models": {
                    k: float(np.mean([s["variance_alone"][mk][k]
                                      for mk in MODEL_KEYS]))
                    for k in ALONE_KEYS
                },
                "image_absorbed_mean": float(np.mean(list(s["image_absorbed"].values()))),
                "corr_with_human_mean": float(np.mean(list(s["corr_with_human_loo"].values()))),
            }
    with open(OUT / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nsaved {OUT / 'summary.json'}")
    print("\n=== compact summary (mean variance share over 12 models, %) ===")
    print(f"{'metric/res':14s} {'image':>6s} {'plot':>6s} {'imgc':>6s} "
          f"{'qcat':>6s} {'qn':>6s} {'gaze':>6s} {'nw':>6s} "
          f"{'hacc':>6s} {'hrt':>6s} {'resid':>6s} {'r_hum':>7s} {'N':>5s}")
    for key, s in all_summaries.items():
        v = s["variance_alone_mean_over_models"]
        print(f"{key:14s} "
              f"{v['image_alone']*100:6.1f} {v['plottype_alone']*100:6.1f} "
              f"{v['imgcomplexity_alone']*100:6.1f} {v['qcat_alone']*100:6.1f} "
              f"{v['qcomplexity_alone']*100:6.1f} {v['gaze_alone']*100:6.1f} "
              f"{v['nworkers_alone']*100:6.2f} {v['hacc_alone']*100:6.2f} "
              f"{v['hrt_alone']*100:6.2f} {v['residual_full']*100:6.1f} "
              f"{s['corr_with_human_mean']:+7.3f} {s['n_samples_used']:5d}")


if __name__ == "__main__":
    main()
