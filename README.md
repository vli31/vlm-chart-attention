# VLM-chart-attention reproducibility

Code release accompanying the paper *"Do Vision-Language Models Exhibit
Human-Like Attention in Chart Question-Answering?"*

## Datasets

- **SalChartQA** (Wang et al., 2024): <https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3884>
- **TaskVis** (Polatsek et al., 2018): <http://vgg.fiit.stuba.sk/2018-02/taskvis/>

## Models

All 12 evaluated VLMs are publicly released instruct checkpoints on
HuggingFace: Qwen2.5-VL (3B, 7B), Qwen3-VL (2B, 4B, 8B), InternVL3
(1B, 2B, 8B), and InternVL3.5 (1B, 2B, 4B, 8B). They load via
`transformers.AutoModelForCausalLM` / `AutoModel`; the largest model
(8B) fits on a single 80 GB GPU.

## Environment

Python 3.10+. CPU-only analysis: `pip install -r requirements.txt`.
GPU inference additionally needs `torch`, `transformers`,
`accelerate`, and `pillow`.

## Pipeline

The paper's results come from three GPU stages — attention extraction,
$K{=}5$ baseline-accuracy generation, and four mean-ablation re-runs —
followed by CPU-side per-head correlations, a five-metric per-sample
matrix, the QC ground-truth construction, and the per-figure analysis
scripts. Prompt format and generation settings are given in the paper
(`Section: Methods` and `Appendix: Reproducibility`).

End-to-end SalChartQA reproduction across all 12 VLMs takes
approximately 30–50 GPU-hours on a single 80 GB GPU plus a few
CPU-hours of analysis on a 16-core node.

## License

Per-figure scripts in this repository are released under the MIT
License. Cited datasets and models retain their original licenses.
