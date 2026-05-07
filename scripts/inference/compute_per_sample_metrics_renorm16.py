#!/usr/bin/env python3
"""Per-sample max-head 5-metric pipeline, renormalized to a common 16x16 grid
with consistent attention smoothing (sigma=1 grid unit) — corrected for the
canvas-mismatch bug that affected the previous TaskVis runs.

Key differences vs per_sample_metrics_5way{,_taskvis}.py:
  * Attention is **always renormalized to 16x16** before smoothing (so all
    models / samples use the same canonical grid, even though they have
    different numbers of vision tokens at native resolution).
  * Attention is smoothed at sigma=1.0 in the renormalized (16x16) grid
    space, matching the convention used in
    `final-chart/lib/corr_plots/recompute_correlations.py`
    (transform_attention_perhead, sigma=1.0).
  * Gaze is rasterized at 16x16 from raw fixations using erf-based
    Gaussian discretization with WORKER_SIGMA_PX=19 in chart-image-pixel
    space. (Same as reference.)
  * **TaskVis: chart-image dims come from the actual extracted MASSVIS
    chart png** (taskvis_charts/), NOT the screen (1280x1024). This was
    the source of the wrong-direction TaskVis-native correlations.

Outputs:
  per_sample_metrics_5way/{model}_renorm16_{salchartqa,taskvis}.npz
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import erf

ROOT = Path("./")
ATTN_BASE = Path("./cache/lvlm-chart/attention_maps")
SC_DIR = ROOT / "downloaded_data/salchartqa/SalChartQA"
SC_IMG_DIR = SC_DIR / "raw_img"
SC_FIX_DIR = SC_DIR / "fixationByVis"
TV_ZIP = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis/taskvis.zip")
TV_GT  = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis/taskvis_ground_truth.json")
TV_CHART_DIR = ROOT / "taskvis_charts"
SC_CORR_BASE = Path("./data/lvlm-chart/correlations/salchartqa")
TV_CORR_BASE = Path("./data/lvlm-chart/correlations/taskvis")
OUT_DIR = ROOT / "per_sample_metrics_5way"
OUT_DIR.mkdir(exist_ok=True)

WORKER_SIGMA_PX = 19.0  # gaze fixation smoothing, in chart-image pixels
ATTN_SIGMA_GRID = 1.0   # attention smoothing, in renormalized grid units
TARGET = 16             # renormalized grid size
SOURCE_TOKEN = "question"

sys.path.insert(0, str(ROOT))
import head_alignment_per_worker as hapw  # for load_attn_grid + resize_all + MODEL_REGISTRY

MODEL_ORDER = [
    "2.5-3B", "2.5-7B", "2B", "4B", "8B",
    "internvl3-1b", "internvl3-2b", "internvl3-8b",
    "internvl3.5-1b", "internvl3.5-2b", "internvl3.5-4b", "internvl3.5-8b",
]

# TaskVis attention dirs use *-instruct for InternVL
TV_ATTN_DIR = {
    "2.5-3B": "2.5-3B", "2.5-7B": "2.5-7B",
    "2B": "2B", "4B": "4B", "8B": "8B",
    "internvl3-1b":   "internvl3-1b-instruct",
    "internvl3-2b":   "internvl3-2b-instruct",
    "internvl3-8b":   "internvl3-8b-instruct",
    "internvl3.5-1b": "internvl3.5-1b-instruct",
    "internvl3.5-2b": "internvl3.5-2b-instruct",
    "internvl3.5-4b": "internvl3.5-4b-instruct",
    "internvl3.5-8b": "internvl3.5-8b-instruct",
}
TV_CORR_PREFIX = dict(TV_ATTN_DIR)


# ── Metric primitives (per-head, attention is HxN) ────────────────
def gaussian_clicks_to_grid(fixations, gh, gw, img_h, img_w, sigma):
    """Reference erf-based gaussian rasterization of fixations onto a (gh, gw)
    grid. Fixations are (x, y) in chart-image pixel coordinates."""
    if not fixations:
        return np.zeros((gh, gw), dtype=np.float64)
    h_edges = np.linspace(0, img_h, gh + 1)
    w_edges = np.linspace(0, img_w, gw + 1)
    s = sigma * np.sqrt(2)
    grid = np.zeros((gh, gw), dtype=np.float64)
    for x, y in fixations:
        row = (erf((h_edges[1:] - y) / s) - erf((h_edges[:-1] - y) / s)) / 2
        col = (erf((w_edges[1:] - x) / s) - erf((w_edges[:-1] - x) / s)) / 2
        grid += row[:, None] * col[None, :]
    return grid


def fix_to_grid_indices(xs, ys, img_w, img_h, gh, gw):
    """Closest-grid-cell indices for each fixation. Used for NSS/AUC."""
    xn = np.clip(np.asarray(xs) / img_w, 0.0, 1.0 - 1e-9)
    yn = np.clip(np.asarray(ys) / img_h, 0.0, 1.0 - 1e-9)
    return (yn * gh).astype(int), (xn * gw).astype(int)


def auc_judd_per_head(flat_HxN, fix_indices):
    H, n_pix = flat_HxN.shape
    if len(fix_indices) == 0: return np.full(H, 0.5)
    pos_mask = np.zeros(n_pix, dtype=bool)
    pos_mask[fix_indices] = True
    n_pos = int(pos_mask.sum()); n_neg = n_pix - n_pos
    if n_pos == 0 or n_neg == 0: return np.full(H, 0.5)
    aucs = np.empty(H)
    for h in range(H):
        ranks = flat_HxN[h].argsort().argsort() + 1
        sum_pos = ranks[pos_mask].sum()
        u = sum_pos - n_pos * (n_pos + 1) / 2
        aucs[h] = u / (n_pos * n_neg)
    return aucs


def kl_div_per_head(head_flat, gaze_flat):
    """KL(gaze || head)."""
    eps = 1e-9
    g = gaze_flat / max(gaze_flat.sum(), eps)
    g = g + eps; g = g / g.sum()
    sums = head_flat.sum(axis=1, keepdims=True)
    p = head_flat / np.where(sums > 1e-12, sums, 1.0)
    p = p + eps; p = p / p.sum(axis=1, keepdims=True)
    return (g[None, :] * (np.log(g[None, :]) - np.log(p))).sum(axis=1)


# ── Common per-sample metric computation ──────────────────────────
def compute_metrics(attn_renorm_LH_T_T, gaze_T_T, fix_idx_in_T):
    """attn_renorm_LH_T_T: (LH, gh, gw) attention (already smoothed) to
    compute 5 metrics against gaze_T_T (gh, gw). gh/gw can be 16 (renorm)
    or any native shape."""
    LH = attn_renorm_LH_T_T.shape[0]
    flat = attn_renorm_LH_T_T.reshape(LH, -1).astype(np.float64)
    gaze = gaze_T_T.ravel()

    # CC
    c = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(c, axis=1)
    cn = c / np.where(norms > 1e-12, norms, 1.0)[:, None]
    gz = gaze - gaze.mean(); gn = gz / max(np.linalg.norm(gz), 1e-12)
    cc = (cn @ gn).astype(np.float32)

    # SIM
    gp = gaze / max(gaze.sum(), 1e-12)
    head_sums = flat.sum(axis=1, keepdims=True)
    head_p = flat / np.where(head_sums > 1e-12, head_sums, 1.0)
    sim = np.minimum(head_p, gp[None, :]).sum(axis=1).astype(np.float32)

    # KL(gaze || head)
    kl = kl_div_per_head(flat, gaze).astype(np.float32)

    # NSS + AUC (need fixation grid points)
    nss = np.full(LH, np.nan, dtype=np.float32)
    auc = np.full(LH, np.nan, dtype=np.float32)
    if len(fix_idx_in_T) > 0:
        mu = flat.mean(axis=1, keepdims=True)
        sd = flat.std(axis=1, keepdims=True)
        z = (flat - mu) / np.where(sd > 1e-12, sd, 1.0)
        nss[:] = z[:, fix_idx_in_T].mean(axis=1)
        auc[:] = auc_judd_per_head(flat, fix_idx_in_T)
    return cc, sim, nss, auc, kl


def smooth_attention_renorm(attn_native, target=TARGET, sigma=ATTN_SIGMA_GRID):
    """Resize each (gh,gw) head to (TARGET,TARGET) bilinearly, then smooth
    with sigma=1 in the renormalized grid space."""
    L, H, gh, gw = attn_native.shape
    flat_T = hapw.resize_all(attn_native, target)  # returns (LH, target, target)
    out = np.empty((L * H, target, target), dtype=np.float64)
    for i in range(L * H):
        head = flat_T[i]
        if head.sum() < 0.01:
            out[i] = 0.0
            continue
        sm = gaussian_filter(head.astype(np.float64), sigma=sigma)
        s = sm.sum()
        out[i] = sm / s if s > 1e-10 else 0.0
    return out  # (LH, target, target)


def smooth_attention_native(attn_native, sigma=ATTN_SIGMA_GRID):
    """Don't resize; smooth attention at sigma=1 IN NATIVE GRID UNITS.
    Returns (LH, gh, gw) with each head normalised to sum=1."""
    L, H, gh, gw = attn_native.shape
    out = np.empty((L * H, gh, gw), dtype=np.float64)
    for l in range(L):
        for h in range(H):
            head = attn_native[l, h].astype(np.float64)
            idx = l * H + h
            if head.sum() < 0.01:
                out[idx] = 0.0
                continue
            sm = gaussian_filter(head, sigma=sigma)
            s = sm.sum()
            out[idx] = sm / s if s > 1e-10 else 0.0
    return out


# ── SalChartQA ────────────────────────────────────────────────────
def run_salchartqa(model_key, mode="renorm16"):
    cfg = hapw.MODEL_REGISTRY[model_key]
    print(f"\n=== SC {mode}: {model_key} ===", flush=True)
    corr_g32 = SC_CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_correlations_sigma1.0_grid32.npz"
    corr_native = SC_CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_correlations_sigma1.0.npz"
    corr_path = corr_g32 if corr_g32.exists() else corr_native
    corr = np.load(corr_path)
    n_layers = int(corr["n_layers"]); n_heads = int(corr["n_heads"])
    LH = n_layers * n_heads
    sr_g32 = SC_CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_sample_results_sigma1.0_grid32.json"
    sr_native = SC_CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_sample_results_sigma1.0.json"
    sr_path = sr_g32 if sr_g32.exists() else sr_native
    meta = json.load(open(sr_path))["sample_results"]
    qc_json = ROOT / "final_paper_figures" / "broad_qc_ids.json"
    qc_ids = set(int(x) for x in json.load(open(qc_json))["ids_in_qc_subset"])
    qc = [m for m in meta if int(m["idx"]) in qc_ids]
    print(f"  QC samples: {len(qc)}, LH={LH}", flush=True)

    cc_arr  = np.full((len(qc), LH), np.nan, dtype=np.float32)
    sim_arr = np.full((len(qc), LH), np.nan, dtype=np.float32)
    nss_arr = np.full((len(qc), LH), np.nan, dtype=np.float32)
    auc_arr = np.full((len(qc), LH), np.nan, dtype=np.float32)
    kl_arr  = np.full((len(qc), LH), np.nan, dtype=np.float32)
    sample_ids = np.zeros(len(qc), dtype=np.int64)
    n_workers_arr = np.zeros(len(qc), dtype=np.int32)

    t0 = time.time()
    for ci, s in enumerate(qc):
        sample_ids[ci] = s["idx"]
        img_name = s["image_name"]; q_id = s["question_id"]
        image_id = img_name.replace(".png", "")
        npz = ATTN_BASE / "salchartqa/image_suffix_question" / cfg["attn_subdir"] / f"salchartqa_{s['idx']}.npz"
        img_path = SC_IMG_DIR / img_name
        if not (npz.exists() and img_path.exists()):
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # Attention loaded first to determine native (gh, gw) when mode=native
        attn_2d = hapw.load_attn_grid(npz, n_layers, n_heads, img_h, img_w, cfg)
        if attn_2d is None: continue
        gh_nat, gw_nat = attn_2d.shape[2:]
        if mode == "renorm16":
            gh_target, gw_target = TARGET, TARGET
        else:  # native
            gh_target, gw_target = gh_nat, gw_nat

        # Workers — gaze rasterized at the (gh_target, gw_target) grid
        worker_grids_T = []
        all_yi = []; all_xi = []
        fix_dir = SC_FIX_DIR / image_id / q_id
        for label in ("True", "False"):
            d = fix_dir / label
            if not d.exists(): continue
            for p in sorted(d.glob("*.csv")):
                try:
                    arr = np.loadtxt(p, delimiter=",", dtype=np.float64)
                except Exception:
                    continue
                if arr.size == 0: continue
                if arr.ndim == 1: arr = arr.reshape(1, -1)
                if arr.shape[1] < 2: continue
                xs = arr[:, 0]; ys = arr[:, 1]
                fix_pairs = list(zip(xs.tolist(), ys.tolist()))
                g = gaussian_clicks_to_grid(fix_pairs, gh_target, gw_target,
                                            img_h, img_w, WORKER_SIGMA_PX)
                if g.sum() <= 0: continue
                worker_grids_T.append(g / g.sum())
                yi, xi = fix_to_grid_indices(xs, ys, img_w, img_h, gh_target, gw_target)
                all_yi.append(yi); all_xi.append(xi)
        if len(worker_grids_T) < 3:
            continue
        n_workers_arr[ci] = len(worker_grids_T)
        gaze_T = np.mean(worker_grids_T, axis=0)
        s_g = gaze_T.sum()
        if s_g > 0: gaze_T = gaze_T / s_g
        if mode == "renorm16":
            attn_T = smooth_attention_renorm(attn_2d, TARGET, ATTN_SIGMA_GRID)
        else:
            attn_T = smooth_attention_native(attn_2d, ATTN_SIGMA_GRID)

        if all_yi:
            yi_all = np.concatenate(all_yi); xi_all = np.concatenate(all_xi)
            fidx = np.unique(yi_all * gw_target + xi_all)
        else:
            fidx = np.array([], dtype=np.int64)

        cc_, sim_, nss_, auc_, kl_ = compute_metrics(attn_T, gaze_T, fidx)
        cc_arr[ci]  = cc_; sim_arr[ci] = sim_; nss_arr[ci] = nss_
        auc_arr[ci] = auc_; kl_arr[ci]  = kl_

        if (ci + 1) % 500 == 0:
            print(f"  [{ci+1}/{len(qc)}] elapsed={time.time()-t0:.0f}s", flush=True)

    out = OUT_DIR / f"{model_key}_{mode}_salchartqa.npz"
    np.savez(out, cc=cc_arr, sim=sim_arr, nss=nss_arr, auc=auc_arr, kl=kl_arr,
             sample_ids=sample_ids, n_workers=n_workers_arr,
             n_layers=n_layers, n_heads=n_heads)
    print(f"saved {out}; valid={(n_workers_arr >= 3).sum()}")


# ── TaskVis ───────────────────────────────────────────────────────
def run_taskvis(model_key, mode="renorm16"):
    import zipfile
    cfg = hapw.MODEL_REGISTRY[model_key]
    cfg = {**cfg, "attn_subdir": TV_ATTN_DIR[model_key]}
    corr_pref = TV_CORR_PREFIX[model_key]
    print(f"\n=== TV {mode}: {model_key} ===", flush=True)

    corr = np.load(TV_CORR_BASE / f"{corr_pref}_mean_gaze_correlations_sigma1.0.npz")
    n_layers = int(corr["n_layers"]); n_heads = int(corr["n_heads"])
    LH = n_layers * n_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}, LH={LH}", flush=True)

    gt = json.loads(TV_GT.read_text())
    samples = gt["samples"]

    zf = zipfile.ZipFile(TV_ZIP)
    with zf.open("tasktypes.txt") as f:
        tt_df = pd.read_csv(f, sep="\t").set_index("user")

    cc_arr  = np.full((len(samples), LH), np.nan, dtype=np.float32)
    sim_arr = np.full((len(samples), LH), np.nan, dtype=np.float32)
    nss_arr = np.full((len(samples), LH), np.nan, dtype=np.float32)
    auc_arr = np.full((len(samples), LH), np.nan, dtype=np.float32)
    kl_arr  = np.full((len(samples), LH), np.nan, dtype=np.float32)
    sample_ids = np.zeros(len(samples), dtype=object)
    n_workers_arr = np.zeros(len(samples), dtype=np.int32)

    t0 = time.time()
    for ci, s in enumerate(samples):
        img = int(s["image_id"]); task = s["task_type"]
        sid = s["sample_id"]
        sample_ids[ci] = sid
        chart_file = s["image_filename"]
        chart_path = TV_CHART_DIR / chart_file
        if not chart_path.exists():
            print(f"  skip {sid}: chart {chart_file} missing")
            continue
        with Image.open(chart_path) as im:
            img_w, img_h = im.size

        npz_path = ATTN_BASE / "taskvis/image_suffix_question" / cfg["attn_subdir"] / f"taskvis_{img}_{task}.npz"
        if not npz_path.exists():
            continue

        # Determine native (gh_nat, gw_nat) by loading attention first
        attn_2d = hapw.load_attn_grid(npz_path, n_layers, n_heads, img_h, img_w, cfg)
        if attn_2d is None:
            print(f"  skip {sid}: load_attn_grid None")
            continue
        gh_nat, gw_nat = attn_2d.shape[2:]
        if mode == "renorm16":
            gh_target, gw_target = TARGET, TARGET
        else:
            gh_target, gw_target = gh_nat, gw_nat

        # Workers — fixations are in chart-image-pixel space (same coord
        # system as the actual chart png in TV_CHART_DIR/, which the model
        # was shown).
        col = str(img)
        pids = tt_df.index[tt_df[col] == task].tolist()
        worker_grids_T = []
        all_yi = []; all_xi = []
        for pid in pids:
            nn = pid.replace("P", "")
            name = f"fixations/rec_p{nn}_fix_{img}.tsv"
            try:
                with zf.open(name) as f:
                    txt = f.read().decode("utf-8", errors="ignore")
            except KeyError:
                continue
            xs = []; ys = []
            for ln in txt.strip().split("\n")[1:]:
                parts = ln.split("\t")
                if len(parts) < 4: continue
                if parts[0] == "Duration": break
                try: xv = float(parts[2]); yv = float(parts[3])
                except ValueError: continue
                xs.append(xv); ys.append(yv)
            if not xs: continue
            xs = np.array(xs); ys = np.array(ys)
            fix_pairs = list(zip(xs.tolist(), ys.tolist()))
            g = gaussian_clicks_to_grid(fix_pairs, gh_target, gw_target,
                                        img_h, img_w, WORKER_SIGMA_PX)
            if g.sum() <= 0: continue
            worker_grids_T.append(g / g.sum())
            yi, xi = fix_to_grid_indices(xs, ys, img_w, img_h, gh_target, gw_target)
            all_yi.append(yi); all_xi.append(xi)
        if len(worker_grids_T) < 2:
            continue
        n_workers_arr[ci] = len(worker_grids_T)
        gaze_T = np.mean(worker_grids_T, axis=0)
        s_g = gaze_T.sum()
        if s_g > 0: gaze_T = gaze_T / s_g

        if mode == "renorm16":
            attn_T = smooth_attention_renorm(attn_2d, TARGET, ATTN_SIGMA_GRID)
        else:
            attn_T = smooth_attention_native(attn_2d, ATTN_SIGMA_GRID)

        if all_yi:
            yi_all = np.concatenate(all_yi); xi_all = np.concatenate(all_xi)
            fidx = np.unique(yi_all * gw_target + xi_all)
        else:
            fidx = np.array([], dtype=np.int64)

        cc_, sim_, nss_, auc_, kl_ = compute_metrics(attn_T, gaze_T, fidx)
        cc_arr[ci]  = cc_; sim_arr[ci] = sim_; nss_arr[ci] = nss_
        auc_arr[ci] = auc_; kl_arr[ci]  = kl_

        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{len(samples)}] elapsed={time.time()-t0:.0f}s", flush=True)

    out = OUT_DIR / f"{model_key}_{mode}_taskvis.npz"
    np.savez(out, cc=cc_arr, sim=sim_arr, nss=nss_arr, auc=auc_arr, kl=kl_arr,
             sample_ids=np.array(sample_ids).astype(str),
             n_workers=n_workers_arr,
             n_layers=n_layers, n_heads=n_heads)
    print(f"saved {out}; valid={(n_workers_arr >= 2).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_ORDER)
    ap.add_argument("--dataset", required=True, choices=["salchartqa", "taskvis"])
    ap.add_argument("--mode", choices=["renorm16", "native"], default="renorm16")
    args = ap.parse_args()
    if args.dataset == "salchartqa":
        run_salchartqa(args.model, args.mode)
    else:
        run_taskvis(args.model, args.mode)


if __name__ == "__main__":
    main()
