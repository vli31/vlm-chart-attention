# Reproducibility for "Do Vision-Language Models Exhibit Human-Like Attention in Chart Question-Answering?"

This repository contains the code that produces every numerical result and
figure in the paper: GPU inference (attention extraction, response
generation, mean-ablation), CPU post-processing (per-head correlations,
five saliency metrics, the QC ground-truth subset), and per-figure
analysis scripts.

## Datasets

We use two publicly released chart-QA eye-gaze datasets. Download URLs:

- **SalChartQA** (Wang et al., 2024) — 5,999 chart QA items × ~12.5 viewers each, BubbleView clicks at $\sigma=19$ px:
  <https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3884>
- **TaskVis** (Polatsek et al., 2018) — 90 image–task pairs from MASSVIS, calibrated eye-tracker at $\sigma=32$ px:
  <http://vgg.fiit.stuba.sk/2018-02/taskvis/>

After downloading place SalChartQA at `data/salchartqa/` (we use the
per-image, per-question fixation files in `fixationByVis/`,
`raw_img/`, `image_questions.json`, `unified_approved.csv`) and TaskVis
at `data/taskvis/` (`massvis/` images plus the fixation tables).

## Models

All 12 evaluated VLMs are publicly released instruct checkpoints on
HuggingFace:

| Family       | Sizes evaluated  |
|--------------|------------------|
| Qwen2.5-VL   | 3B, 7B           |
| Qwen3-VL     | 2B, 4B, 8B       |
| InternVL3    | 1B, 2B, 8B       |
| InternVL3.5  | 1B, 2B, 4B, 8B   |

All models load via `transformers.AutoModelForCausalLM` (Qwen) /
`transformers.AutoModel` (InternVL) on a single 80 GB GPU. The exact
model IDs and per-model config (number of layers, heads, vision-token
grid construction) are in `scripts/lib/model_constants.py`.

## Environment

Python 3.10+. See `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For inference (steps 1-3) you also need:
# pip install torch transformers accelerate pillow
```

## Pipeline overview

```
data/
 ├─ salchartqa/                    (downloaded raw data)
 └─ taskvis/                       (downloaded raw data)

[ Step 1, GPU ]  per-VLM attention extraction
                      → attention_maps/<dataset>/<model>/<idx>.npz
                        keys: attn_question (L×H×T_v float16),
                              attn_im_start_assistant, token_ids, token_types

[ Step 2, GPU ]  per-VLM K=5 baseline-accuracy generation
                      → correctness/<model>_<dataset>_x5.json

[ Step 3, GPU ]  per-VLM 4 mean-ablation re-runs (Top-5/Bottom-5/2×Random-5)
                      → correctness/salchartqa_<ablation>__<model>_<ts>.json

[ Step 4, CPU ]  human saliency aggregation, ground-truth majority vote
                      → aggregated_saliency/<dataset>/<image>_<qid>.npz
                      → determined_ground_truth_fixed.json

[ Step 5, CPU ]  per-head correlation NPZ + 5-metric per-sample matrices
                      → correlations/<dataset>/<model>_mean_gaze_correlations_sigma1.0.npz
                      → per_sample_metrics_5way/<model>_{16,native}.npz

[ Step 6, CPU ]  QC subset definition
                      → broad_qc_ids.json   (n=4,556)

[ Step 7, CPU ]  Per-figure analysis (this repo)
                      → final_paper_figures/*.pdf, *.json, *.tex
```

Steps 1–3 are GPU-only. Steps 4–7 are CPU-only.

## Step-by-step code

### Step 1 — Attention extraction (GPU)

Goal: for every (model, question), record the LM-decoder attention
$A^{(l,h)}_{t,v}$ from each query-token position $t \in \mathcal{Q}$
(question tokens) onto each vision-token position $v \in \mathcal{V}$ at
the prefill step. Output is one `.npz` per question with keys
`attn_question` (shape $L \times H \times T_v$, fp16) and
`attn_im_start_assistant` (control), plus the token type tags.

Prompt format (every VLM, every question):
```
<image> + question + " Answer concisely and immediately." + " Answer:"
```

The exact model loading and tokenisation logic is in
`scripts/lib/extract_attention_helpers.py` (`load_model`,
`_prepare_inputs_qwen`, `_prepare_inputs_internvl2`). The driver script
that walks the dataset, runs one forward pass per question with
`output_attentions=True`, slices the attention tensor along
$\mathcal{Q} \times \mathcal{V}$, and saves the result was deleted from
the working tree once the attention-map cache stabilised. The minimal
re-implementation is:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np, torch
from scripts.lib.extract_attention_helpers import load_model, _prepare_inputs_qwen

model, processor, cfg = load_model(model_key)             # see model_constants.py
for sample_idx, (image, question) in enumerate(dataset):
    inputs, q_slice, v_slice = _prepare_inputs_qwen(processor, image, question)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, return_dict=True)
    # out.attentions: tuple of (B, H, T, T) per layer
    attn = torch.stack(out.attentions, dim=0)              # (L, B=1, H, T, T)
    attn_qv = attn[:, 0, :, q_slice, :][..., v_slice]      # (L, H, |Q|, |V|)
    attn_q  = attn_qv.mean(dim=2).cpu().to(torch.float16) # (L, H, |V|)
    np.savez(f"attention_maps/{dataset_name}/{model_key}/{dataset_name}_{sample_idx}.npz",
             attn_question=attn_q.numpy(),
             token_ids=inputs['input_ids'][0].cpu().numpy(),
             token_types=token_types_array)
```

Cost: ~5–10 min (≤4B model) / 15–30 min (7–8B model) per VLM on a
single H100 80GB for the full 5,999-question SalChartQA pass.

### Step 2 — K=5 baseline-accuracy generation (GPU)

Same prompt as step 1 but in `model.generate()` mode with $K=5$
samples, $T=0.7$, top-$p=0.9$, $\leq 64$ output tokens.
`scripts/inference/compute_correctness.py` is the working version of
this driver. It writes one JSON per (model, dataset) with all $K$
responses per question; correctness is then judged via
`relaxed_match()` (`scripts/lib/relaxed_match.py`) against the QC
ground truth (step 4). A question is marked correct if at least 4 of 5
generations match.

Cost: ~10–20 min (≤4B) / 30–45 min (7–8B) per VLM per dataset on a
single H100 80GB for SalChartQA.

### Step 3 — Mean-ablation re-runs (GPU)

`scripts/inference/run_mean_ablation.py` is the production ablation
driver. Per VLM, it:

1. Reads cached per-head correlation NPZs (`correlations/<model>_mean_gaze_correlations_sigma1.0.npz`, step 5) to rank LM-decoder heads by mean Pearson r against the human gaze map across the SalChartQA QC subset.
2. Selects the Top-5, Bottom-5, and two independent Random-5 head sets.
3. For each set, registers a mean-substitution forward hook on the head's pre-projection (pre-$W_O$) activation that replaces it with that head's mean activation vector across SalChartQA, and re-runs the $K=5$ generation pass on the full SalChartQA corpus.
4. Writes one JSON per (model, ablation) with the ablated responses and per-question correctness.

Usage:
```bash
python scripts/inference/run_mean_ablation.py \
    --models 8B \
    --ablation-scope all_tokens \
    --ablation-type mean_activation \
    --seed 1234
```

Cost: ~3 GPU-h for 7–8B models (4 ablations × 5,999 q × $K{=}5$);
~1.5–2 h for ≤4B models. SLURM-log-derived timestamps in the paper's
Compute Resources appendix (`apx:compute`).

### Step 4 — Human saliency aggregation + ground truth (CPU)

`scripts/inference/aggregate_human_saliency.py` reads per-worker
fixation files from SalChartQA / TaskVis and produces per-question
mean saliency maps at native resolution, with the worker-quality
filters described in Appendix A of the paper (worker accuracy ≥ 50%
plus per-trial duration ≥ 2,000 ms; last-row 0 ms fixations dropped).

`scripts/inference/determine_ground_truth.py` collects the five
sources of evidence per SalChartQA question (ChartQA original answer,
SalChartQA worker plurality, three independent runs of `gpt-5-mini`
2025-08-07) and writes `determined_ground_truth_fixed.json`. The five
runs of `gpt-5-mini` are obtained via API; the script formats the
prompts and parses the responses but does not include API keys.

### Step 5 — Per-head correlations + per-sample 5-metric matrices (CPU)

- `scripts/inference/compute_correlations_npz.py` and
  `compute_human_correlations.py` read the cached `attention_maps/`
  (step 1) plus the aggregated saliency (step 4), compute per-head,
  per-sample Pearson r between attention and gaze at the model's
  native vision-token grid (with $\sigma=1$ Gaussian smoothing on the
  grid), and write `<model>_mean_gaze_correlations_sigma1.0.npz`
  (shape $(N, L, H)$).
- `scripts/inference/compute_per_sample_metrics_native.py`,
  `compute_per_sample_metrics_renorm16.py`, and
  `compute_per_sample_metrics_taskvis.py` produce the per-sample
  $L \times H$ matrices for all five saliency metrics (CC, SIM, NSS,
  AUC-Judd, KL) at the native attention grid and on the common $16
  \times 16$ renormalised grid.
- `scripts/lib/saliency_metrics.py` is the underlying metrics
  implementation (CC, SIM, NSS, AUC-Judd, KL) with the convention that
  per-question best-aligned head = max over heads for similarity-style
  metrics, **min over heads for KL** (lower KL = more aligned).

Cost: 10–30 CPU-min per (model, dataset, resolution).

### Step 6 — QC ground-truth subset (CPU)

`scripts/figures/build_qc_subset.py` constructs the high-confidence
subset (`broad_qc_ids.json`) by intersecting the five evidence sources
per question and keeping questions where the largest agreeing cluster
has size ≥ 3 and no `gpt-5-mini` run flags the question as
unanswerable / ambiguous. Yields $n=4{,}556$ paired model/human items
out of $5{,}999$ total SalChartQA items.

### Step 7 — Per-figure scripts (CPU)

Each script writes its PDF/PNG and any data files into
`final_paper_figures/`.

#### Main-text figures

| Main figure | Caption topic | Script |
|---|---|---|
| **Figure 4 (`between.pdf`)** | Head-equalised alignment vs. accuracy on SalChartQA & TaskVis + per-layer mean r | `scripts/figures/build_fig4_alignment_vs_accuracy.py` (left + middle panels; the per-layer-r right panel is generated by the `_build_fig5b_records.py` helper) |
| **Figure 5 (`fig6.pdf`)** | Per-model SalChartQA accuracy under Top-5 / Bottom-5 / Random-5 mean ablation (9 of 12 VLMs that pass the specificity criterion) | `scripts/figures/build_fig5_ablation.py` (reads the post-ablation `correctness/*` JSON files via the bootstrap-record helper `scripts/figures/_build_fig5b_records.py`) |
| **Figure 6 (`fig5c_modelacc.pdf`)** | Per-question Q3-8B max-head r vs inter-human gaze r (scatter, coloured by Q3-8B accuracy) + per-VLM head-human r ridge plots vs Human reference | `scripts/figures/build_fig6_alignment.py` |
| **Figure 7 (`tab3_boxplot.pdf`)** | Standalone $R^2$ for each predictor of per-question max-head r, across 12 VLMs (boxes) + Human reference (stars) | `scripts/figures/build_fig7_variance.py` |

```bash
python scripts/figures/build_fig4_alignment_vs_accuracy.py
python scripts/figures/build_fig5_ablation.py
python scripts/figures/build_fig6_alignment.py
python scripts/figures/build_fig7_variance.py
```

**Figures 1, 2, and 3 of the main text** (the abstract chart-attention
examples, chart-correlation examples, and the saliency-baseline bar
plot with TaskVis example) are hand-composed in Inkscape from the
underlying attention-map and human-gaze saliency rasters; the rasters
themselves come from steps 1–4. The composition itself is not
auto-generated.

#### Appendix figures and tables

| Appendix item | Script |
|---|---|
| Per-question variance attribution at native and 16×16 grid (5 metrics × 2 resolutions = 10 tables) | `scripts/figures/build_variance_attribution.py` |
| Alternative-metric alignment-vs-accuracy panels (CC/SIM/NSS/AUC/KL × {native, 16×16}) | `scripts/figures/build_appendix_alt_metrics.py` |
| Per-sample alignment ↔ ablation drop correlation table | `scripts/figures/build_per_sample_ablation_correlation.py` |

## Compute totals

End-to-end SalChartQA reproduction across all 12 VLMs:
~30–50 GPU-hours of inference (steps 1–3) on a single 80 GB GPU
plus a few CPU-hours of analysis (steps 4–7) on a single 16-core
node. SLURM-log-derived per-VLM and per-stage breakdowns are in
Appendix C of the paper.

## License & citation

The per-figure scripts in this repo are released under the MIT
License (see `LICENSE`). The cited datasets and models retain their
original licenses; please cite them via the references in the paper.
