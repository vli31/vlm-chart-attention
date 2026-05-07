"""Model loading and input preparation for VLM attention analysis.

Reconstructed minimal version with load_model, _prepare_inputs_qwen,
and _prepare_inputs_internvl2.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image

from .constants import get_model_config, HF_CACHE

# Set HF cache before importing transformers
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)


def load_model(model_key: str, device: str = "cuda") -> tuple[Any, Any, dict[str, Any]]:
    """Load a VLM model and processor.

    Returns:
        (model, processor, config) where config is the dict from get_model_config
        with model_family, num_layers, num_heads, etc.
    """
    from transformers import AutoProcessor

    config = get_model_config(model_key)
    model_id = config["model_id"]
    model_family = config.get("model_family", "qwen")

    print(f"Loading model: {model_id} (family={model_family})")

    if model_family == "internvl2":
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Wrap tokenizer to provide .tokenizer attribute (matching processor API)
        class _Proc:
            def __init__(self, t):
                self.tokenizer = t

            def decode(self, *args, **kwargs):
                return self.tokenizer.decode(*args, **kwargs)

        return model, _Proc(tokenizer), config

    else:
        # Qwen family
        model_class_name = config.get("model_class", "Qwen3VLForConditionalGeneration")
        if model_class_name == "Qwen2_5_VLForConditionalGeneration":
            from transformers import Qwen2_5_VLForConditionalGeneration
            ModelClass = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen3VLForConditionalGeneration
            ModelClass = Qwen3VLForConditionalGeneration

        model = ModelClass.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, config


def _prepare_inputs_qwen(
    processor: Any,
    messages: list[dict[str, Any]],
    image_path: str,
    generation_prefix: str,
    device: str,
) -> dict[str, Any]:
    """Prepare inputs for Qwen-family VLMs.

    Args:
        processor: HF AutoProcessor.
        messages: Chat messages in Qwen format.
        image_path: Path to the image file.
        generation_prefix: Text prefix for generation (e.g. "Answer:").
        device: Target device string.

    Returns:
        Dict of model inputs (input_ids, attention_mask, pixel_values, etc.).
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if generation_prefix:
        text += generation_prefix

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device == "auto":
        pass
    else:
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
    return inputs


def _prepare_inputs_internvl2(
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
    image_path: str,
    generation_prefix: str,
    device: str,
) -> dict[str, Any]:
    """Prepare inputs for InternVL2/3/3.5-family VLMs.

    Uses the model's own chat template to build the prompt with image tokens,
    matching how model.chat() works internally.

    Args:
        processor: Wrapped tokenizer with .tokenizer attribute.
        model: The InternVL model.
        messages: Chat messages (InternVL format with 'text' key).
        image_path: Path to the image file.
        generation_prefix: Text prefix for generation.
        device: Target device string or "auto".

    Returns:
        Dict with input_ids, attention_mask, pixel_values, image_flags.
    """
    from utils.internvl2_utils import load_internvl2_image

    tokenizer = processor.tokenizer

    # Build the raw question text from messages
    text_parts = []
    for msg in messages:
        if "text" in msg:
            text_parts.append(msg["text"])
        elif "content" in msg:
            text_parts.append(str(msg["content"]))
    question = " ".join(text_parts)

    # Load and preprocess image
    pixel_values = load_internvl2_image(image_path, max_num=12)
    if device != "auto":
        pixel_values = pixel_values.to(device).to(torch.bfloat16)
    else:
        pixel_values = pixel_values.to(torch.bfloat16)

    num_patches = pixel_values.shape[0]
    image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

    # Set up img_context_token_id on the model (as model.chat() does)
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    # Build prompt using the model's conversation template
    from transformers import GenerationConfig
    try:
        from conversation import get_conv_template
    except ImportError:
        # The dynamic module should provide this
        import importlib
        import sys
        # Find the dynamic module directory
        for path in sys.path:
            try:
                mod = importlib.import_module("conversation")
                get_conv_template = mod.get_conv_template
                break
            except (ImportError, ModuleNotFoundError):
                continue
        else:
            # Fallback: try from the HF modules cache
            import glob as _glob
            modules_dirs = _glob.glob(
                "./cache/hf_modules/transformers_modules/_*/conversation.py"
            )
            if modules_dirs:
                spec = importlib.util.spec_from_file_location("conversation", modules_dirs[0])
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                get_conv_template = mod.get_conv_template
            else:
                raise ImportError("Cannot find InternVL conversation module")

    # Ensure <image> tag is present
    if "<image>" not in question:
        question = "<image>\n" + question

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], generation_prefix if generation_prefix else None)
    query = template.get_prompt()

    # Replace <image> with actual image tokens
    num_image_token = model.num_image_token  # tokens per patch (256)
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    if device != "auto":
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_flags = image_flags.to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
    }
