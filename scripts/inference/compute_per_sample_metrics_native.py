#!/usr/bin/env python3
"""Per-sample, per-head, 5 metrics (CC, SIM, NSS, AUC, KL) at either:
  - target=16 (equalized 16x16 grid)
  - target=NATIVE (each sample's native attention grid)

Output:
  per_sample_metrics_5way/<model>_<resolution>.npz
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

ROOT = Path("./")
ATTN_BASE = Path("./cache/lvlm-chart/attention_maps/salchartqa/image_suffix_question")
CORR_BASE = Path("./data/lvlm-chart/correlations/salchartqa")
SC_DIR = ROOT / "downloaded_data/salchartqa/SalChartQA"
SC_IMG_DIR = SC_DIR / "raw_img"
SC_FIX_DIR = SC_DIR / "fixationByVis"
SC_UNIFIED = SC_DIR / "unified_approved.csv"
CAT_PATH = ROOT / "salchartqa_question_categories.csv"
OUT_DIR = ROOT / "per_sample_metrics_5way"
OUT_DIR.mkdir(exist_ok=True)
OUT_DIR_BROAD = ROOT / "per_sample_metrics_broad"
OUT_DIR_BROAD.mkdir(exist_ok=True)
BROAD_QC_PATH = ROOT / "final_paper_figures" / "broad_qc_ids.json"
GAZE_SIGMA_PX = 19.0
GAUSSIAN_SMOOTH_ATTN = True

sys.path.insert(0, str(ROOT))
import head_alignment_per_worker as hapw  # noqa


def load_correct_workers(image_id, q_id, bad_workers, fast_trials, img_name, q_text):
    correct_dir = SC_FIX_DIR / image_id / q_id / "True"
    if not correct_dir.exists(): return []
    out = []
    for csv_path in sorted(correct_dir.glob("*.csv")):
        wid = csv_path.stem
        if wid in bad_workers: continue
        if (wid, img_name, q_text) in fast_trials: continue
        try:
            d = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
        except Exception:
            continue
        if d.size == 0: continue
        if d.ndim == 1: d = d.reshape(1, -1)
        if d.shape[1] < 2: continue
        out.append((d[:, 0], d[:, 1]))
    return out


def fix_grid_idx(xs, ys, img_w, img_h, gh, gw):
    xn = np.clip(xs / img_w, 0.0, 1.0 - 1e-9)
    yn = np.clip(ys / img_h, 0.0, 1.0 - 1e-9)
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
        sum_pos_ranks = ranks[pos_mask].sum()
        u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
        aucs[h] = u / (n_pos * n_neg)
    return aucs


def kl_div_per_head(head_flat, gaze_flat):
    """KL(gaze || head). head_flat: (H, n_pix). gaze_flat: (n_pix,)."""
    eps = 1e-9
    g = gaze_flat / max(gaze_flat.sum(), eps)
    g = g + eps; g = g / g.sum()
    H = head_flat.shape[0]
    sums = head_flat.sum(axis=1, keepdims=True)
    p = head_flat / np.where(sums > 1e-12, sums, 1.0)
    p = p + eps; p = p / p.sum(axis=1, keepdims=True)
    return (g[None, :] * (np.log(g[None, :]) - np.log(p))).sum(axis=1)


def smooth_at_native(attn_2d, sigma_grid):
    """Gaussian smooth each (h, w) attention map in place (in given native resolution)."""
    L, H, gh, gw = attn_2d.shape
    return np.array([
        gaussian_filter(attn_2d[l, h], sigma=sigma_grid)
        for l in range(L) for h in range(H)
    ]).reshape(L, H, gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(hapw.MODEL_REGISTRY))
    ap.add_argument("--resolution", choices=["16", "native"], default="16")
    ap.add_argument("--qc", choices=["inqc", "broad"], default="inqc",
                    help="inqc = legacy in_qc_subset filter; broad = ≥3-of-5 "
                         "GT agreement set from broad_qc_ids.json")
    args = ap.parse_args()
    cfg = hapw.MODEL_REGISTRY[args.model]
    print(f"=== 5-METRIC PER SAMPLE: {args.model}  resolution={args.resolution} ===", flush=True)

    corr_grid32 = CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_correlations_sigma1.0_grid32.npz"
    corr_path = corr_grid32 if corr_grid32.exists() else CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_correlations_sigma1.0.npz"
    corr = np.load(corr_path)
    n_layers = int(corr["n_layers"]); n_heads = int(corr["n_heads"])
    LH = n_layers * n_heads
    json_grid32 = CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_sample_results_sigma1.0_grid32.json"
    json_path = json_grid32 if json_grid32.exists() else CORR_BASE / f"{cfg['corr_prefix']}_mean_gaze_sample_results_sigma1.0.json"
    with open(json_path) as f:
        meta = json.load(f)["sample_results"]
    if args.qc == "broad":
        with open(BROAD_QC_PATH) as bf:
            broad_ids = set(json.load(bf)["ids_at_3_agree"])
        qc_idx = [i for i, s in enumerate(meta) if s["idx"] in broad_ids]
        print(f"  broad QC samples: {len(qc_idx)} (≥3-of-5 GT agreement), LH={LH}", flush=True)
    else:
        if any("in_qc_subset" in s for s in meta[:5]):
            qc_idx = [i for i, s in enumerate(meta) if s.get("in_qc_subset")]
        else:
            qc_idx = [i for i, s in enumerate(meta) if s.get("n_correct", 0) >= 3]
        print(f"  in_qc QC samples: {len(qc_idx)}, LH={LH}", flush=True)

    uni = pd.read_csv(SC_UNIFIED)
    worker_acc = uni.groupby("participant_id")["is_correct"].mean()
    bad_workers = set(worker_acc[worker_acc < 0.50].index)
    fast = uni[(uni["is_correct"] == True) & (uni["total_duration"] < 2000)]
    fast_trials = set(zip(fast["participant_id"], fast["image_name"], fast["question"]))
    cat = pd.read_csv(CAT_PATH)
    qtxt = {(r["image_name"], r["question_id"]): r["question"] for _, r in cat.iterrows()}

    n_qc = len(qc_idx)
    cc_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    sim_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    nss_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    auc_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    kl_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    sample_ids = np.zeros(n_qc, dtype=np.int64)
    n_workers_arr = np.zeros(n_qc, dtype=np.int32)
    grid_shapes = np.zeros((n_qc, 2), dtype=np.int32)

    t0 = time.time()
    for ci, sample_idx in enumerate(qc_idx):
        s = meta[sample_idx]
        sample_ids[ci] = s["idx"]
        img_name = s["image_name"]; q_id = s["question_id"]
        image_id = img_name.replace(".png", "")
        npz = ATTN_BASE / cfg["attn_subdir"] / f"salchartqa_{s['idx']}.npz"
        img_path = SC_IMG_DIR / img_name
        if not (npz.exists() and img_path.exists()): continue
        with PILImage.open(img_path) as im:
            img_w, img_h = im.size
        q_text = qtxt.get((img_name, q_id), "")
        workers = load_correct_workers(image_id, q_id, bad_workers, fast_trials, img_name, q_text)
        if len(workers) < 3: continue
        attn_2d = hapw.load_attn_grid(npz, n_layers, n_heads, img_h, img_w, cfg)  # (L, H, gh_nat, gw_nat)
        if attn_2d is None: continue
        gh_nat, gw_nat = attn_2d.shape[2:]

        if args.resolution == "16":
            target_h = target_w = 16
            attn = hapw.resize_all(attn_2d, target_h)  # (LH, T, T)
            attn = attn.reshape(LH, target_h, target_w)
        else:  # native
            target_h, target_w = gh_nat, gw_nat
            attn = attn_2d.reshape(LH, target_h, target_w)
        grid_shapes[ci] = [target_h, target_w]

        if GAUSSIAN_SMOOTH_ATTN:
            sigma_grid = (GAZE_SIGMA_PX / max(img_w, img_h)) * max(target_h, target_w)
            attn = np.stack([gaussian_filter(a, sigma=sigma_grid) for a in attn])
        flat = attn.reshape(LH, -1).astype(np.float64)

        # mean gaze map at this resolution
        sigma_pix = (GAZE_SIGMA_PX / max(img_w, img_h)) * max(target_h, target_w)
        accum = np.zeros((target_h, target_w), dtype=np.float64); n_used = 0
        all_yi = []; all_xi = []
        for xs, ys in workers:
            if len(xs) == 0: continue
            yi, xi = fix_grid_idx(xs, ys, img_w, img_h, target_h, target_w)
            sal = np.zeros((target_h, target_w), dtype=np.float64)
            np.add.at(sal, (yi, xi), 1.0)
            sal = gaussian_filter(sal, sigma=sigma_pix)
            if sal.sum() > 0:
                accum += sal / sal.sum(); n_used += 1
                all_yi.append(yi); all_xi.append(xi)
        if n_used == 0: continue
        gaze = (accum / n_used).ravel()
        n_workers_arr[ci] = n_used

        # CC
        c = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(c, axis=1)
        cn = c / np.where(norms > 1e-12, norms, 1.0)[:, None]
        gz = gaze - gaze.mean(); gn = gz / max(np.linalg.norm(gz), 1e-12)
        cc_arr[ci] = (cn @ gn).astype(np.float32)

        # SIM
        gaze_p = gaze / max(gaze.sum(), 1e-12)
        head_sums = flat.sum(axis=1, keepdims=True)
        head_p = flat / np.where(head_sums > 1e-12, head_sums, 1.0)
        sim_arr[ci] = np.minimum(head_p, gaze_p[None, :]).sum(axis=1).astype(np.float32)

        # KL
        kl_arr[ci] = kl_div_per_head(flat, gaze).astype(np.float32)

        # NSS + AUC
        mu = flat.mean(axis=1, keepdims=True)
        sd = flat.std(axis=1, keepdims=True)
        z = (flat - mu) / np.where(sd > 1e-12, sd, 1.0)
        if all_yi:
            yi_all = np.concatenate(all_yi); xi_all = np.concatenate(all_xi)
            fidx = yi_all * target_w + xi_all
            nss_arr[ci] = z[:, fidx].mean(axis=1).astype(np.float32)
            unique_fix = np.unique(fidx)
            auc_arr[ci] = auc_judd_per_head(flat, unique_fix).astype(np.float32)

        if (ci + 1) % 500 == 0:
            print(f"  [{ci+1}/{n_qc}] elapsed={time.time()-t0:.0f}s "
                  f"rate={(ci+1)/(time.time()-t0):.1f}/s", flush=True)

    out_dir = OUT_DIR_BROAD if args.qc == "broad" else OUT_DIR
    out = out_dir / f"{args.model}_{args.resolution}.npz"
    np.savez(out, cc=cc_arr, sim=sim_arr, nss=nss_arr, auc=auc_arr, kl=kl_arr,
             sample_ids=sample_ids, n_workers=n_workers_arr,
             grid_shapes=grid_shapes,
             n_layers=n_layers, n_heads=n_heads)
    print(f"saved {out}")
    print(f"  N samples = {n_qc}, valid = {(n_workers_arr >= 3).sum()}")


if __name__ == "__main__":
    main()
