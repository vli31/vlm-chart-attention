#!/usr/bin/env python3
"""TaskVis equivalent of per_sample_metrics_5way.py: per-sample, per-head,
5 metrics (CC, SIM, NSS, AUC, KL) at the equalized 16x16 grid for all 12 VLMs.

Output: per_sample_metrics_5way/{model}_taskvis_16.npz
        with cc/sim/nss/auc/kl arrays of shape (n_samples, LH).

Mirrors per_sample_metrics_5way.py but for TaskVis (90 samples, 1 worker
group per (image_id, task_type)). Gaze coordinates are screen-space
(1280x1024); attention is loaded from the same TaskVis attention-map npz
files used by the existing taskvis CC correlations.
"""
from __future__ import annotations
import argparse, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path("./")
ATTN_BASE = Path("./cache/lvlm-chart/attention_maps/taskvis/image_suffix_question")
CORR_BASE = Path("./data/lvlm-chart/correlations/taskvis")
TV_ZIP = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis/taskvis.zip")
TV_GT  = Path("./cache/lvlm-chart/eye_gaze_datasets/taskvis/taskvis_ground_truth.json")
OUT_DIR = ROOT / "per_sample_metrics_5way"
OUT_DIR.mkdir(exist_ok=True)

GAZE_SIGMA_PX = 35.0
GAUSSIAN_SMOOTH_ATTN = True
SCREEN_W = 1280
SCREEN_H = 1024

sys.path.insert(0, str(ROOT))
import head_alignment_per_worker as hapw  # noqa


def fix_to_grid(xs, ys, img_w, img_h, gh, gw):
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


def parse_tsv_bytes(b):
    txt = b.decode("utf-8", errors="ignore")
    lines = txt.strip().split("\n")
    if not lines: return np.array([]), np.array([])
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 4: continue
        try: x = float(parts[2]); y = float(parts[3])
        except ValueError: continue
        rows.append((x, y))
    if not rows: return np.array([]), np.array([])
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


# `--model` uses the canonical (non-instruct) registry name. For TaskVis,
# the InternVL attention dirs and correlation prefixes use the *-instruct
# suffix, so we map the registry key to the TaskVis naming here.
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
TV_CORR_PREFIX = dict(TV_ATTN_DIR)  # same naming for TaskVis correlation files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TV_ATTN_DIR))
    ap.add_argument("--resolution", choices=["16", "native"], default="16")
    args = ap.parse_args()
    print(f"=== TASKVIS 5-METRIC PER SAMPLE: {args.model}  resolution={args.resolution} ===", flush=True)

    cfg = hapw.MODEL_REGISTRY[args.model]
    # tweak cfg to point at the TaskVis attention subdir we determined above
    cfg = {**cfg, "attn_subdir": TV_ATTN_DIR[args.model]}
    corr_pref = TV_CORR_PREFIX[args.model]

    corr_path = CORR_BASE / f"{corr_pref}_mean_gaze_correlations_sigma1.0.npz"
    corr = np.load(corr_path)
    n_layers = int(corr["n_layers"]); n_heads = int(corr["n_heads"])
    LH = n_layers * n_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}, LH={LH}")

    # TaskVis ground truth -> samples in fixed order
    gt = json.loads(TV_GT.read_text())
    samples = gt["samples"]
    print(f"  n samples = {len(samples)}")

    # Load fixations from zip once, group by (image_id, task_type)
    print("loading fixations from zip ...")
    zf = zipfile.ZipFile(TV_ZIP)
    with zf.open("tasktypes.txt") as f:
        tt_df = pd.read_csv(f, sep="\t").set_index("user")

    n_qc = len(samples)
    cc_arr  = np.full((n_qc, LH), np.nan, dtype=np.float32)
    sim_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    nss_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    auc_arr = np.full((n_qc, LH), np.nan, dtype=np.float32)
    kl_arr  = np.full((n_qc, LH), np.nan, dtype=np.float32)
    sample_ids_str = np.zeros(n_qc, dtype=object)
    n_workers_arr = np.zeros(n_qc, dtype=np.int32)
    grid_shapes = np.zeros((n_qc, 2), dtype=np.int32)

    t0 = time.time()
    for ci, s in enumerate(samples):
        img = int(s["image_id"]); task = s["task_type"]
        sid = s["sample_id"]
        sample_ids_str[ci] = sid

        npz_path = ATTN_BASE / cfg["attn_subdir"] / f"taskvis_{img}_{task}.npz"
        if not npz_path.exists():
            print(f"  skip {sid}: no attn npz")
            continue

        # Pull the workers for this (image, task)
        col = str(img)
        pids = tt_df.index[tt_df[col] == task].tolist()
        if len(pids) < 2:
            continue
        worker_xs = []; worker_ys = []
        for pid in pids:
            nn = pid.replace("P", "")
            name = f"fixations/rec_p{nn}_fix_{img}.tsv"
            try:
                with zf.open(name) as f:
                    xs, ys = parse_tsv_bytes(f.read())
            except KeyError:
                continue
            if len(xs) == 0: continue
            worker_xs.append(xs); worker_ys.append(ys)
        if len(worker_xs) < 2:
            continue

        # Load attention; resize to 16x16 or keep at native
        attn_2d = hapw.load_attn_grid(npz_path, n_layers, n_heads,
                                      SCREEN_H, SCREEN_W, cfg)
        if attn_2d is None:
            print(f"  skip {sid}: load_attn_grid returned None")
            continue
        gh_nat, gw_nat = attn_2d.shape[2:]
        if args.resolution == "16":
            target_h = target_w = 16
            attn = hapw.resize_all(attn_2d, target_h).reshape(LH, target_h, target_w)
        else:  # native
            target_h, target_w = gh_nat, gw_nat
            attn = attn_2d.reshape(LH, target_h, target_w)
        grid_shapes[ci] = [target_h, target_w]

        if GAUSSIAN_SMOOTH_ATTN:
            sigma_grid = (GAZE_SIGMA_PX / max(SCREEN_W, SCREEN_H)) * max(target_h, target_w)
            attn = np.stack([gaussian_filter(a, sigma=sigma_grid) for a in attn])
        flat = attn.reshape(LH, -1).astype(np.float64)

        # Mean gaze map at the same resolution
        sigma_pix = (GAZE_SIGMA_PX / max(SCREEN_W, SCREEN_H)) * max(target_h, target_w)
        accum = np.zeros((target_h, target_w), dtype=np.float64); n_used = 0
        all_yi = []; all_xi = []
        for xs, ys in zip(worker_xs, worker_ys):
            yi, xi = fix_to_grid(xs, ys, SCREEN_W, SCREEN_H, target_h, target_w)
            sal = np.zeros((target_h, target_w), dtype=np.float64)
            np.add.at(sal, (yi, xi), 1.0)
            sal = gaussian_filter(sal, sigma=sigma_pix)
            if sal.sum() > 0:
                accum += sal / sal.sum(); n_used += 1
                all_yi.append(yi); all_xi.append(xi)
        if n_used == 0:
            continue
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

        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{n_qc}] elapsed={time.time()-t0:.0f}s", flush=True)

    out = OUT_DIR / f"{args.model}_taskvis_{args.resolution}.npz"
    np.savez(out, cc=cc_arr, sim=sim_arr, nss=nss_arr, auc=auc_arr, kl=kl_arr,
             sample_ids=np.array(sample_ids_str, dtype=object).astype(str),
             n_workers=n_workers_arr,
             grid_shapes=grid_shapes,
             n_layers=n_layers, n_heads=n_heads)
    print(f"saved {out}")
    print(f"  N samples = {n_qc}, valid = {(n_workers_arr >= 2).sum()}")


if __name__ == "__main__":
    main()
