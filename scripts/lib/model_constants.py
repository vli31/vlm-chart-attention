"""Model configuration constants for VLM attention analysis.

Reconstructed from head ranking CSVs and model architecture specifications.
"""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any


class InputOrder(Enum):
    IMAGE_SUFFIX_QUESTION = "image_suffix_question"
    IMAGE_QUESTION_SUFFIX = "image_question_suffix"
    QUESTION_IMAGE_SUFFIX = "question_image_suffix"
    QUESTION_SUFFIX_IMAGE = "question_suffix_image"


ALL_INPUT_ORDERS = list(InputOrder)

SUFFIX = "Answer concisely and immediately."

# HF cache directory
HF_CACHE = "./cache/hf"


def _snapshot_path(hf_id: str) -> str:
    """Resolve HF model ID to local snapshot path if available, else return ID."""
    cache = Path(HF_CACHE)
    model_dir = cache / f"models--{hf_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        snaps = sorted(snapshots.iterdir())
        if snaps:
            return str(snaps[0])
    return hf_id


# Model configurations keyed by the short model_key used throughout the project.
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    # Qwen2.5-VL family
    "2.5-3B": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "model_family": "qwen",
        "num_layers": 36,
        "num_heads": 16,
    },
    "2.5-7B": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "model_family": "qwen",
        "num_layers": 28,
        "num_heads": 28,
    },
    # Qwen3-VL family
    "2B": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_family": "qwen",
        "num_layers": 28,
        "num_heads": 16,
    },
    "4B": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_family": "qwen",
        "num_layers": 36,
        "num_heads": 32,
    },
    "8B": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_family": "qwen",
        "num_layers": 36,
        "num_heads": 32,
    },
    # InternVL3 family
    "internvl3-1b-instruct": {
        "model_id": "OpenGVLab/InternVL3-1B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 24,
        "num_heads": 14,
        "img_context_token_id": 151667,
    },
    "internvl3-2b-instruct": {
        "model_id": "OpenGVLab/InternVL3-2B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 28,
        "num_heads": 12,
        "img_context_token_id": 151667,
    },
    "internvl3-8b-instruct": {
        "model_id": "OpenGVLab/InternVL3-8B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 28,
        "num_heads": 28,
        "img_context_token_id": 151667,
    },
    # InternVL3.5 family
    "internvl3.5-1b-instruct": {
        "model_id": "OpenGVLab/InternVL3_5-1B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 28,
        "num_heads": 16,
        "img_context_token_id": 151667,
    },
    "internvl3.5-2b-instruct": {
        "model_id": "OpenGVLab/InternVL3_5-2B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 28,
        "num_heads": 16,
        "img_context_token_id": 151667,
    },
    "internvl3.5-4b-instruct": {
        "model_id": "OpenGVLab/InternVL3_5-4B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 36,
        "num_heads": 32,
        "img_context_token_id": 151667,
    },
    "internvl3.5-8b-instruct": {
        "model_id": "OpenGVLab/InternVL3_5-8B-Instruct",
        "model_class": "AutoModel",
        "model_family": "internvl2",
        "num_layers": 36,
        "num_heads": 32,
        "img_context_token_id": 151667,
    },
}

ALL_MODEL_SIZES = list(_MODEL_CONFIGS.keys())


def get_model_config(model_key: str) -> dict[str, Any]:
    """Return configuration dict for the given model key.

    Resolves model_id to a local snapshot path if the HF cache contains it,
    avoiding network calls and disk-quota issues with refs/main.
    """
    if model_key not in _MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model key: {model_key!r}. "
            f"Available: {sorted(_MODEL_CONFIGS.keys())}"
        )
    cfg = dict(_MODEL_CONFIGS[model_key])
    cfg["model_id"] = _snapshot_path(cfg["model_id"])
    return cfg


def get_suffix(model_key: str) -> str:
    return SUFFIX


def get_generation_prefix(model_key: str) -> str:
    return "Answer:"
