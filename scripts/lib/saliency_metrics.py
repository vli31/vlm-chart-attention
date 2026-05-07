"""
Standard saliency evaluation metrics.

All metrics operate on 2D numpy arrays (predicted and ground truth saliency maps).
Maps are expected to be non-negative. Functions handle normalization internally.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc


def _to_distribution(x):
    """Normalize to a probability distribution (sum=1)."""
    x = np.clip(x, 0, None).astype(np.float64)
    s = x.sum()
    if s > 0:
        return x / s
    return np.ones_like(x) / x.size


def _normalize_01(x):
    """Min-max normalize to [0, 1]."""
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def pearson_cc(pred, target):
    """Pearson Correlation Coefficient between two saliency maps."""
    p = pred.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    if p.std() == 0 or t.std() == 0:
        return 0.0
    return float(pearsonr(p, t)[0])


def spearman_cc(pred, target):
    """Spearman rank correlation between two saliency maps."""
    p = pred.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    if p.std() == 0 or t.std() == 0:
        return 0.0
    return float(spearmanr(p, t)[0])


def kl_divergence(pred, target):
    """KL divergence: KL(target || pred). Lower is better."""
    p = _to_distribution(pred)
    t = _to_distribution(target)
    eps = 1e-10
    p = np.clip(p, eps, None)
    t = np.clip(t, eps, None)
    return float(np.sum(t * np.log(t / p)))


def nss(pred, fixation_map):
    """
    Normalized Scanpath Saliency.

    pred: continuous saliency map
    fixation_map: binary or near-binary map (thresholded at >0)
    """
    pred = pred.astype(np.float64)
    fixations = (fixation_map.ravel() > 0)
    if fixations.sum() == 0:
        return 0.0
    pred_norm = pred.ravel()
    std = pred_norm.std()
    if std == 0:
        return 0.0
    pred_norm = (pred_norm - pred_norm.mean()) / std
    return float(pred_norm[fixations].mean())


def auc_judd(pred, fixation_map):
    """
    AUC-Judd: ROC-based AUC using fixation locations as positives.

    pred: continuous saliency map
    fixation_map: binary map (>0 = fixated)
    """
    pred = pred.ravel().astype(np.float64)
    fix = (fixation_map.ravel() > 0).astype(np.int32)
    if fix.sum() == 0 or fix.sum() == len(fix):
        return 0.5
    fpr, tpr, _ = roc_curve(fix, pred)
    return float(auc(fpr, tpr))


def sim(pred, target):
    """
    Similarity metric: sum of min(p_i, t_i) where both are distributions.
    Higher is better (max=1.0).
    """
    p = _to_distribution(pred)
    t = _to_distribution(target)
    return float(np.minimum(p, t).sum())


def evaluate_sample(pred, target, fixation_map=None):
    """Compute all metrics for a single sample pair.

    Args:
        pred: predicted saliency map
        target: ground truth saliency heatmap
        fixation_map: optional binary fixation map for NSS and AUC-Judd.
                      If None, uses target as fixation map.
    """
    fix = fixation_map if fixation_map is not None else target
    result = {
        "cc": pearson_cc(pred, target),
        "spearman": spearman_cc(pred, target),
        "kl": kl_divergence(pred, target),
        "nss": nss(pred, fix),
        "sim": sim(pred, target),
        "auc_judd": auc_judd(pred, fix),
    }
    return result


def evaluate_all(pred_maps, target_maps, fixation_maps=None):
    """
    Compute all metrics across a list of prediction/target pairs.

    Args:
        pred_maps: list of predicted saliency maps
        target_maps: list of ground truth saliency heatmaps
        fixation_maps: optional list of binary fixation maps for NSS/AUC-Judd.
                       If None, uses target_maps as fixation maps.

    Returns dict with per-metric mean, std, and per-sample values.
    """
    metric_keys = ["cc", "spearman", "kl", "nss", "sim", "auc_judd"]
    results = {k: [] for k in metric_keys}

    for i, (pred, target) in enumerate(zip(pred_maps, target_maps)):
        fix = fixation_maps[i] if fixation_maps is not None else None
        sample_metrics = evaluate_sample(pred, target, fixation_map=fix)
        for k, v in sample_metrics.items():
            results[k].append(v)

    summary = {}
    for k, vals in results.items():
        arr = np.array(vals)
        summary[f"{k}_mean"] = float(np.nanmean(arr))
        summary[f"{k}_std"] = float(np.nanstd(arr))
    summary["n_samples"] = len(pred_maps)
    summary["per_sample"] = results

    return summary
