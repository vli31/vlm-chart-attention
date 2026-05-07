#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate how different input orderings affect model correctness/accuracy.

Supports both ChartGaze (binary True/False via logits) and SalChartQA
(open-ended generation with relaxed matching).

Tests orderings:
- [image] [question] [suffix] (iqs)
- [question] [image] [suffix] (qis)
- [question] [suffix] [image] (qsi)

Saves correctness results to ./data/lvlm-chart/correctness

Usage:
    python evaluate_ordering.py --model 2B --num-samples 100
    python evaluate_ordering.py --model 2B --dataset salchartqa --num-samples 100
"""

import argparse
import csv
import json
import re
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Heavy imports deferred to avoid dependency on attention/utils when only
# relaxed_match / normalize_answer are needed.
def _lazy_imports():
    """Import heavy dependencies only when evaluation pipeline functions are called."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from attention.constants import (
        SUFFIX, InputOrder, ALL_INPUT_ORDERS, ALL_MODEL_SIZES,
        get_model_config, get_suffix, get_generation_prefix,
    )
    from utils.data_loaders import ChartGazeLoader, SalChartQALoader, DataSample
    from utils.constants import CORRECTNESS_DIR, SALCHARTQA_PERSISTENT_DIR
    return locals()


def load_model(model_size: str, device: str = "cuda"):
    """Load model and processor based on model family."""
    from transformers import AutoProcessor

    config = get_model_config(model_size)
    model_id = config["model_id"]
    model_class_name = config.get("model_class", "Qwen3VLForConditionalGeneration")
    model_family = config.get("model_family", "qwen")

    print(f"Loading model: {model_id} ({model_class_name}, family={model_family})")

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
        # Wrap to provide .tokenizer attribute
        class _Proc:
            def __init__(self, t): self.tokenizer = t
        return model, _Proc(tokenizer), config

    elif model_family == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, config

    else:
        # Qwen family
        if model_class_name == "Qwen3_5ForConditionalGeneration":
            from transformers import Qwen3_5ForConditionalGeneration
            ModelClass = Qwen3_5ForConditionalGeneration
        elif model_class_name == "Qwen2_5_VLForConditionalGeneration":
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


def build_messages(
    question: str,
    image_path: str,
    order: InputOrder,
    suffix: str = "Answer concisely and immediately.",
    model_family: str = "qwen",
) -> List[Dict[str, Any]]:
    """Build chat messages based on the specified order and model family."""
    if model_family in ("internvl2", "paligemma"):
        # For these models, use the extract_attention build_messages
        from attention.extract_attention import build_messages as _build
        return _build(question, image_path, order, suffix, model_family=model_family)

    image_content = {"type": "image", "image": image_path}
    question_content = {"type": "text", "text": question}
    suffix_content = {"type": "text", "text": suffix}

    if order == InputOrder.QUESTION_IMAGE_SUFFIX:
        content = [question_content, image_content, suffix_content]
    elif order == InputOrder.IMAGE_QUESTION_SUFFIX:
        content = [image_content, question_content, suffix_content]
    elif order == InputOrder.QUESTION_SUFFIX_IMAGE:
        content = [question_content, suffix_content, image_content]
    elif order == InputOrder.IMAGE_SUFFIX_QUESTION:
        content = [image_content, suffix_content, question_content]
    else:
        raise ValueError(f"Unknown order: {order}")

    return [{"role": "user", "content": content}]


# All tracked token variants, grouped by polarity
POSITIVE_TOKENS = ["True", "true", "TRUE", " True", "Yes"]
NEGATIVE_TOKENS = ["False", "false", "FALSE", " False", "No"]
ALL_TOKENS = POSITIVE_TOKENS + NEGATIVE_TOKENS


def get_token_ids(processor) -> Dict[str, int]:
    """Get token IDs for all tracked token variants.

    Tracks: True, true, TRUE, ' True', Yes, False, false, FALSE, ' False', No

    Returns:
        Dict mapping token strings to their IDs
    """
    token_ids = {}
    for word in ALL_TOKENS:
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        token_ids[word] = ids[0]
    return token_ids


def get_logit_probabilities(
    model,
    processor,
    image_path: str,
    question: str,
    order: InputOrder,
    token_ids: Dict[str, int],
    device: str = "cuda",
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Get logit probabilities for all tracked token variants.

    Tracks 10 tokens: True/true/TRUE/' True'/Yes (positive)
                      False/false/FALSE/' False'/No (negative)

    Args:
        model: The loaded model
        processor: The model processor
        image_path: Path to the image file
        question: The question text
        order: Input ordering enum
        token_ids: Dict mapping token strings to their IDs
        device: Device to run on
        config: Model configuration dict

    Returns:
        Dict with per-token probabilities and aggregated positive/negative
    """
    model_family = config.get("model_family", "qwen") if config else "qwen"
    messages = build_messages(question, image_path, order, model_family=model_family)

    if model_family == "internvl2":
        from utils.internvl2_utils import load_internvl2_image
        tokenizer = processor.tokenizer

        pixel_values = load_internvl2_image(image_path).to(torch.bfloat16).to(device)
        num_tiles = pixel_values.shape[0]

        prompt_text = messages[0]["text"]
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        image_tokens = "<img>" + IMG_CONTEXT_TOKEN * 256 * num_tiles + "</img>"
        prompt_text = prompt_text.replace("<image>\n", image_tokens + "\n", 1)

        chatml_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer(chatml_prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            next_token_logits = outputs.logits[:, -1, :]

    elif model_family == "paligemma":
        prompt_text = messages[0]["text"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

    else:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

    with torch.no_grad():

        # Extract logits for all tracked tokens
        logits = {name: next_token_logits[0, tid].item() for name, tid in token_ids.items()}

        # Compute normalized probabilities over all tracked tokens (sum to 1)
        all_logits = torch.tensor([[logits[t] for t in ALL_TOKENS]])
        all_probs = F.softmax(all_logits, dim=-1)
        probs_normalized = {t: all_probs[0, i].item() for i, t in enumerate(ALL_TOKENS)}

        # Aggregated positive/negative
        prob_positive = sum(probs_normalized[t] for t in POSITIVE_TOKENS)
        prob_negative = sum(probs_normalized[t] for t in NEGATIVE_TOKENS)

        # Total probability mass on all tracked tokens from full vocabulary
        full_probs = F.softmax(next_token_logits, dim=-1)
        prob_total = sum(full_probs[0, tid].item() for tid in token_ids.values())

    result = {}
    # Individual normalized probabilities for each token
    for t in ALL_TOKENS:
        result[f"prob|{t}"] = probs_normalized[t]
    # Aggregated positive/negative (sum to 1)
    result["prob_positive"] = prob_positive
    result["prob_negative"] = prob_negative
    # Total probability mass on tracked tokens (from full vocabulary softmax)
    result["prob_total"] = prob_total
    return result


def evaluate_sample(
    model,
    processor,
    sample: DataSample,
    order: InputOrder,
    token_ids: Dict[str, int],
    device: str = "cuda",
    temp_dir: Optional[str] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Evaluate a single sample with a given order.

    Args:
        model: The loaded model
        processor: The model processor
        sample: DataSample containing image and question
        order: Input ordering enum
        token_ids: Dict mapping token names ('True','False','Yes','No') to IDs
        device: Device to run on
        temp_dir: Temporary directory for saving images (if sample lacks image_path)
        config: Model configuration dict
    """
    # Use existing image_path if available, otherwise save to temp file
    if sample.image_path:
        image_path = sample.image_path
    else:
        # Save image to temp file
        temp_path = Path(temp_dir) / f"{sample.sample_id}.png"
        sample.image.save(temp_path)
        image_path = str(temp_path)

    result = get_logit_probabilities(
        model, processor, image_path, sample.question, order,
        token_ids, device=device, config=config
    )

    # Determine ground truth
    gt = None
    if sample.answer is not None:
        gt_lower = str(sample.answer).lower().strip()
        if gt_lower in ("true", "yes", "1"):
            gt = True
        elif gt_lower in ("false", "no", "0"):
            gt = False

    # Logit-based prediction using aggregated positive/negative probabilities
    logit_pred = result["prob_positive"] > result["prob_negative"]
    logit_correct = None
    if gt is not None:
        logit_correct = (logit_pred == gt)

    return {
        "sample_id": sample.sample_id,
        "split": sample.metadata.get("split", "unknown") if sample.metadata else "unknown",
        "question": sample.question,
        "ground_truth": sample.answer,
        **{k: v for k, v in result.items()},
        "logit_pred": logit_pred,
        "logit_correct": logit_correct,
    }


def evaluate_ordering(
    model,
    processor,
    samples: List[DataSample],
    order: InputOrder,
    token_ids: Dict[str, int],
    device: str = "cuda",
    temp_dir: Optional[str] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Evaluate all samples with a given ordering.

    Args:
        model: The loaded model
        processor: The model processor
        samples: List of DataSample objects
        order: Input ordering enum
        token_ids: Dict mapping token names ('True','False','Yes','No') to IDs
        device: Device to run on
        temp_dir: Temporary directory for saving images (if samples lack image_path)
        config: Model configuration dict
    """
    results = []

    for sample in tqdm(samples, desc=f"Evaluating {order.value}"):
        result = evaluate_sample(
            model, processor, sample, order,
            token_ids, device, temp_dir, config=config
        )
        results.append(result)

    # Compute statistics
    total = len(results)
    logit_correct_count = sum(1 for r in results if r["logit_correct"] is True)
    logit_incorrect_count = sum(1 for r in results if r["logit_correct"] is False)
    logit_evaluable = logit_correct_count + logit_incorrect_count

    return {
        "order": order.value,
        "total_samples": total,
        "logit_evaluable": logit_evaluable,
        "logit_correct": logit_correct_count,
        "logit_incorrect": logit_incorrect_count,
        "logit_accuracy": logit_correct_count / logit_evaluable if logit_evaluable > 0 else None,
        "results": results,
    }


# =============================================================================
# SalChartQA: generation-based evaluation
# =============================================================================

SALCHARTQA_PERSISTENT_DIR = Path("./data/salchartqa")


def _normalize_quotes(s: str) -> str:
    """Normalize quote characters for consistent matching."""
    return s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace('"', "'")


def load_salchartqa_ground_truth() -> Dict[str, Dict[str, str]]:
    """Load ground truth answers from chartqa_ground_truth.csv.

    Normalizes quote characters in questions so that single/double quote
    variations match consistently.

    Returns:
        Dict mapping image_name -> {normalized_question -> ground_truth_answer}
    """
    gt_path = SALCHARTQA_PERSISTENT_DIR / "chartqa_ground_truth.csv"
    gt = {}
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image_name"]
            if img not in gt:
                gt[img] = {}
            gt[img][_normalize_quotes(row["question"])] = row["ground_truth"]
    return gt


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    # Normalize en-dash/em-dash to hyphen-minus
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # Strip leading "answer:" prefix if present
    s = re.sub(r"^answer:\s*", "", s)
    # Remove trailing period
    s = s.rstrip(".")
    # Strip trailing asterisks (footnote markers from charts: "2015*", "2020**")
    s = s.rstrip("*")
    return s.strip()


def _normalize_date_format(s: str) -> str:
    """Normalize abbreviated year formats: May '21 -> may 2021, Q3'12 -> q3 2012."""
    # 'YY at end or after space/letter -> 20YY or 19YY
    def _expand_year(m):
        prefix = m.group(1)
        yy = int(m.group(2))
        yyyy = 2000 + yy if yy < 50 else 1900 + yy
        return f"{prefix}{yyyy}"
    s = re.sub(r"(\s|^)'(\d{2})\b", _expand_year, s)
    # Also handle Q3'12 (no space before tick) -> Q3 2012
    def _expand_year_space(m):
        prefix = m.group(1)
        yy = int(m.group(2))
        yyyy = 2000 + yy if yy < 50 else 1900 + yy
        return f"{prefix} {yyyy}"
    s = re.sub(r"([a-z]\d?)'(\d{2})\b", _expand_year_space, s)
    return s


def _strip_units(s: str) -> str:
    """Strip common units/suffixes from a normalized answer string."""
    s = s.replace(",", "")
    s = s.replace("%", "").replace("$", "").replace("\u20ac", "").replace("\u00a3", "")
    _UNIT_RE = re.compile(
        r"\s*(billion|million|thousand|trillion|bn|mn|m|k|b|kg|kt|gw|gwh|mw|mwh|twh|tkm|kwh|"
        r"liters?|litres?|tonnes?|tons?|dollars?|euros?|percent(?:age)?\s*(?:points?)?|pp|"
        r"u\.s\.\s*dollars?|"
        r"days?|friends?|years?|months?|hours?|minutes?|people|countries|times|"
        r"g|t|km|mi|lb|lbs|oz|cm|mm|ha|sq)\s*$",
        re.I,
    )
    # Strip repeatedly to handle stacked units like "billion tkm"
    prev = None
    while s != prev:
        prev = s
        s = _UNIT_RE.sub("", s).strip()
    return s


def try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number (handles %, commas, units, ratios)."""
    s = _strip_units(s.strip())
    s = s.lstrip("~")
    try:
        return float(s)
    except ValueError:
        pass
    # Try ratio notation: "24:19" or "24/19"
    # Exclude academic-year patterns like "2010/11", "18/19" where the separator
    # is "/" and denominator is a small 2-digit number that looks like a year suffix
    ratio_m = re.match(r"^(\d+\.?\d*)\s*([:/])\s*(\d+\.?\d*)$", s)
    if ratio_m:
        num_s, sep, denom_s = ratio_m.group(1), ratio_m.group(2), ratio_m.group(3)
        num, denom = float(num_s), float(denom_s)
        # Skip if it looks like an academic year with "/" separator
        is_acad_year = (
            sep == "/" and len(denom_s) <= 2 and denom == int(denom) and
            (len(num_s) == 2 or (len(num_s) == 4 and 1800 <= num <= 2100))
        )
        if denom != 0 and not is_acad_year:
            return num / denom
    return None


def _is_year(s: str) -> bool:
    """Check if a string looks like a year (4-digit integer 1800-2100)."""
    s = _strip_units(s).strip()
    try:
        n = float(s)
        return n == int(n) and 1800 <= n <= 2100
    except (ValueError, TypeError, OverflowError):
        return False


def _normalize_list_format(s: str) -> str:
    """Normalize list formatting: strip brackets, normalize separators."""
    s = re.sub(r"^\[|\]$", "", s).strip()
    s = re.sub(r"\s+and\s+", ", ", s)
    return s


def _extract_calc_result(s: str) -> Optional[str]:
    """Extract the result after a final '=' in an arithmetic expression.

    E.g. "105 - (38 + 33) = 34" -> "34", "58% - 31% = 27%" -> "27%"
    """
    if "=" not in s:
        return None
    result = s.rsplit("=", 1)[-1].strip()
    return result if result else None


def _strip_list_parentheticals(s: str) -> str:
    """Strip item-level parentheticals from list items.

    E.g. "micronesia (country), japan" -> "micronesia, japan"
    """
    return re.sub(r"\s*\([^)]*\)", "", s)


def relaxed_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth with relaxed criteria.

    Matching rules (in order):
    1. Exact match (case-insensitive, stripped)
    1b. Exact match after unit stripping
    1b2. Date format normalization
    1c. List-format normalization ([A, B] matches "A and B")
    1d. Set matching for comma-separated lists (order-insensitive)
    1e. Arithmetic result extraction (after '=')
    1f. List parenthetical stripping
    2. Numerical closeness (within 5% relative tolerance)
       - Years (4-digit integers 1800-2100) require exact match
    3. Numeric GT containment in verbose predictions
    4. Word-boundary containment (non-numeric GTs only, with guards)
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Guard against empty predictions/ground truths
    if not pred_norm or not gt_norm:
        return False

    # 1b. Exact match after unit stripping
    pred_stripped = _strip_units(pred_norm)
    gt_stripped = _strip_units(gt_norm)
    if pred_stripped and gt_stripped and pred_stripped == gt_stripped:
        return True

    # 1b2. Date format normalization: "May '21" == "May 2021" == "May 21"
    pred_date = _normalize_date_format(pred_norm)
    gt_date = _normalize_date_format(gt_norm)
    if pred_date != pred_norm or gt_date != gt_norm:
        if pred_date == gt_date:
            return True

    # 1b3. Range notation normalization: "50-59" == "50 to 59"
    def _normalize_range(s):
        return re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', s)
    pred_range = _normalize_range(pred_stripped)
    gt_range = _normalize_range(gt_stripped)
    if pred_range != pred_stripped or gt_range != gt_stripped:
        if pred_range == gt_range:
            return True

    # 1c. List format normalization: [A, B] matches "A and B"
    pred_list = _normalize_list_format(pred_norm)
    gt_list = _normalize_list_format(gt_norm)
    if pred_list == gt_list:
        return True

    # 1d. Set matching for comma-separated lists (order-insensitive)
    if "," in pred_list and "," in gt_list:
        pred_items = sorted(x.strip() for x in pred_list.split(",") if x.strip())
        gt_items = sorted(x.strip() for x in gt_list.split(",") if x.strip())
        if pred_items == gt_items:
            return True

    # 1e. Arithmetic result extraction: "105 - (38 + 33) = 34" matches GT "34"
    calc_result = _extract_calc_result(pred_norm)
    if calc_result:
        if relaxed_match(calc_result, ground_truth):
            return True

    # 1f. List parenthetical stripping: "[Micronesia (country), Japan]" matches "[Micronesia, Japan]"
    pred_no_paren = _strip_list_parentheticals(pred_list)
    gt_no_paren = _strip_list_parentheticals(gt_list)
    if pred_no_paren != pred_list or gt_no_paren != gt_list:
        if pred_no_paren == gt_no_paren:
            return True
        # Also try set matching on parenthetical-stripped forms
        if "," in pred_no_paren and "," in gt_no_paren:
            pred_items = sorted(x.strip() for x in pred_no_paren.split(",") if x.strip())
            gt_items = sorted(x.strip() for x in gt_no_paren.split(",") if x.strip())
            if pred_items == gt_items:
                return True

    # Numerical closeness (check before containment to handle numeric GTs properly)
    pred_num = try_parse_number(pred_norm)
    gt_num = try_parse_number(gt_norm)
    if pred_num is not None and gt_num is not None:
        # Years must match exactly — don't use tolerance for 4-digit year values
        if _is_year(gt_norm) or _is_year(pred_norm):
            return pred_num == gt_num
        if gt_num == 0:
            return pred_num == 0
        return abs(pred_num - gt_num) / abs(gt_num) < 0.05

    # Numeric GT containment: if GT is a number but pred is verbose text,
    # check if the GT number appears standalone in the prediction.
    # Boundary: not preceded by digit/dot/minus, not followed by digit, dot+digit,
    # or letter (to avoid "1990s", "2nd", "3parts")
    if gt_num is not None and pred_num is None:
        num_pat = r'(?<![\d.\-])' + re.escape(gt_stripped) + r'(?!\d)(?!\.\d)(?![a-zA-Z])'
        if re.search(num_pat, pred_norm):
            return True

    # Word-boundary containment (only for non-numeric ground truths)
    if gt_num is None:
        # For yes/no GTs: require pred to START with the answer to avoid
        # false positives where "no" appears incidentally in explanation.
        # Also handle bracket-wrapped responses like "[No, 87.3]" and quoted '"Yes"'
        if gt_norm in ("yes", "no"):
            # Strip leading brackets/quotes for yes/no check
            pred_clean = re.sub(r'^[\[\]"\']+', '', pred_norm).strip()
            if pred_clean.startswith(gt_norm) and (
                len(pred_clean) == len(gt_norm) or
                not pred_clean[len(gt_norm)].isalpha()
            ):
                return True
            return False

        # Determine if GT with commas is a phrase (not a real item list).
        # A "phrase GT" has commas but each segment has multiple words,
        # e.g. "Not documented, fate unknown" vs a real list like "A, B, C"
        gt_is_list = "," in gt_norm
        gt_is_phrase = False
        if gt_is_list:
            segments = [seg.strip() for seg in gt_norm.split(",") if seg.strip()]
            # If most segments have 2+ words, it's a phrase, not a list of items
            multi_word = sum(1 for seg in segments if " " in seg)
            gt_is_phrase = (len(segments) <= 3 and multi_word >= len(segments) / 2)

        # GT-in-pred containment
        if not gt_is_list or gt_is_phrase or len(pred_norm) <= len(gt_norm) * 3:
            pattern = r'(?<!\w)' + re.escape(gt_norm) + r'(?!\w)'
            if re.search(pattern, pred_norm):
                return True

        # Pred-in-GT: only if GT is NOT a list (avoid "26" matching "[26,15,11,5,3]")
        if not gt_is_list:
            pattern = r'(?<!\w)' + re.escape(pred_norm) + r'(?!\w)'
            if re.search(pattern, gt_norm):
                return True

        # Also try with list-normalized forms
        if pred_list != pred_norm or gt_list != gt_norm:
            gt_list_is_list = "," in gt_list
            gt_list_is_phrase = False
            if gt_list_is_list:
                segments = [seg.strip() for seg in gt_list.split(",") if seg.strip()]
                multi_word = sum(1 for seg in segments if " " in seg)
                gt_list_is_phrase = (len(segments) <= 3 and multi_word >= len(segments) / 2)

            if not gt_list_is_list or gt_list_is_phrase or len(pred_list) <= len(gt_list) * 3:
                pattern = r'(?<!\w)' + re.escape(gt_list) + r'(?!\w)'
                if re.search(pattern, pred_list):
                    return True
            if not gt_list_is_list:
                pattern = r'(?<!\w)' + re.escape(pred_list) + r'(?!\w)'
                if re.search(pattern, gt_list):
                    return True

    return False


def generate_response(
    model,
    processor,
    image_path: str,
    question: str,
    order: InputOrder,
    suffix: str,
    generation_prefix: Optional[str] = None,
    max_new_tokens: int = 32,
    device: str = "cuda",
    config: Dict[str, Any] = None,
) -> str:
    """Generate a text response from the model."""
    model_family = config.get("model_family", "qwen") if config else "qwen"
    messages = build_messages(question, image_path, order, suffix, model_family=model_family)

    if model_family == "internvl2":
        from utils.internvl2_utils import load_internvl2_image
        tokenizer = processor.tokenizer

        pixel_values = load_internvl2_image(image_path).to(torch.bfloat16).to(device)

        # Use model.chat() — InternVL's generate() returns only generated
        # tokens (not input+generated), so manual slicing doesn't work.
        chat_question = messages[0]["text"]
        if generation_prefix:
            chat_question += f"\n{generation_prefix}"
        gen_config = dict(max_new_tokens=max_new_tokens)
        with torch.no_grad():
            response = model.chat(
                tokenizer, pixel_values, chat_question,
                generation_config=gen_config,
            )
        return response

    elif model_family == "paligemma":
        prompt_text = messages[0]["text"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    else:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if generation_prefix:
            text += f" {generation_prefix}"

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def evaluate_sample_salchartqa(
    model,
    processor,
    sample: DataSample,
    order: InputOrder,
    ground_truth: str,
    suffix: str,
    generation_prefix: Optional[str],
    device: str = "cuda",
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Evaluate a single SalChartQA sample via generation."""
    image_path = sample.image_path

    response = generate_response(
        model, processor, image_path, sample.question, order,
        suffix, generation_prefix, device=device, config=config,
    )

    correct = relaxed_match(response, ground_truth)

    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "ground_truth": ground_truth,
        "response": response,
        "correct": correct,
    }


def evaluate_ordering_salchartqa(
    model,
    processor,
    samples: List[DataSample],
    order: InputOrder,
    gt_lookup: Dict[str, Dict[str, str]],
    suffix: str,
    generation_prefix: Optional[str],
    device: str = "cuda",
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Evaluate all SalChartQA samples with a given ordering."""
    results = []
    skipped = 0

    for sample in tqdm(samples, desc=f"Evaluating {order.value}"):
        # Look up ground truth
        img_name = sample.metadata.get("image_name", "")
        gt_for_img = gt_lookup.get(img_name, {})
        gt_answer = gt_for_img.get(_normalize_quotes(sample.question))

        if gt_answer is None:
            skipped += 1
            continue

        result = evaluate_sample_salchartqa(
            model, processor, sample, order, gt_answer,
            suffix, generation_prefix, device, config=config,
        )
        results.append(result)

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])

    return {
        "order": order.value,
        "total_samples": total,
        "skipped": skipped,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else None,
        "results": results,
    }


# =============================================================================
# ChartGaze: data loading
# =============================================================================

def load_chartgaze_both_splits(
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[DataSample]:
    """Load ChartGaze from both train and validation splits.

    Samples are drawn proportionally from each split, or all samples if num_samples is None.
    Uses class-level caching in ChartGazeLoader so HF dataset is only loaded once.

    Args:
        num_samples: Total number of samples to load (None = all)
        seed: Random seed for sampling

    Returns:
        List of DataSample objects with split info in metadata
    """
    print("Loading ChartGaze dataset (train + validation)...")

    # Load both splits - HF dataset is cached at class level, so second call is fast
    train_loader = ChartGazeLoader(data_split="train")
    val_loader = ChartGazeLoader(data_split="validation")

    train_size = len(train_loader)
    val_size = len(val_loader)
    total_size = train_size + val_size

    print(f"  Total: {total_size} samples (train: {train_size}, val: {val_size})")

    # Determine indices to sample
    if num_samples is None or num_samples >= total_size:
        train_indices = list(range(train_size))
        val_indices = list(range(val_size))
    else:
        # Sample proportionally from each split
        random.seed(seed)
        train_ratio = train_size / total_size
        n_train = int(num_samples * train_ratio)
        n_val = num_samples - n_train

        train_indices = random.sample(range(train_size), min(n_train, train_size))
        val_indices = random.sample(range(val_size), min(n_val, val_size))

        print(f"  Sampling {len(train_indices)} from train, {len(val_indices)} from validation")

    # Collect samples with split metadata
    samples = []
    for idx in train_indices:
        sample = train_loader[idx]
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["split"] = "train"
        samples.append(sample)

    for idx in val_indices:
        sample = val_loader[idx]
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["split"] = "validation"
        samples.append(sample)

    # Shuffle combined samples
    random.seed(seed)
    random.shuffle(samples)

    print(f"Loaded {len(samples)} samples total")
    return samples


def load_salchartqa_samples(
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> List[DataSample]:
    """Load SalChartQA samples."""
    print("Loading SalChartQA dataset...")
    loader = SalChartQALoader()
    total_size = len(loader)
    print(f"  Total: {total_size} image-question pairs")

    if num_samples is None or num_samples >= total_size:
        indices = list(range(total_size))
    else:
        random.seed(seed)
        indices = random.sample(range(total_size), num_samples)
        print(f"  Sampling {len(indices)} samples")

    samples = [loader[i] for i in indices]
    random.seed(seed)
    random.shuffle(samples)

    print(f"Loaded {len(samples)} samples")
    return samples


def run_evaluation(
    model_size: str,
    dataset: str = "chartgaze",
    num_samples: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda",
    orders: Optional[List[InputOrder]] = None,
) -> Dict[str, Any]:
    """Run full evaluation across all orderings."""
    if orders is None:
        orders = ALL_INPUT_ORDERS

    # Load model
    model, processor, config = load_model(model_size, device)

    suffix = get_suffix(dataset)
    generation_prefix = get_generation_prefix(dataset)
    print(f"Dataset: {dataset}")
    print(f"Suffix: {suffix!r}")
    if generation_prefix:
        print(f"Generation prefix: {generation_prefix!r}")

    if dataset == "chartgaze":
        return _run_chartgaze_evaluation(
            model, processor, config, model_size, num_samples, seed, device, orders,
        )
    elif dataset == "salchartqa":
        return _run_salchartqa_evaluation(
            model, processor, config, model_size, num_samples, seed, device, orders,
            suffix, generation_prefix,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _run_chartgaze_evaluation(
    model, processor, config,
    model_size: str,
    num_samples: Optional[int],
    seed: int,
    device: str,
    orders: List[InputOrder],
) -> Dict[str, Any]:
    """ChartGaze evaluation (logit-based True/False)."""
    token_ids = get_token_ids(processor)
    print(f"Token IDs - {', '.join(f'{repr(k)}: {v}' for k, v in token_ids.items())}")

    samples = load_chartgaze_both_splits(num_samples=num_samples, seed=seed)

    train_count = sum(1 for s in samples if s.metadata.get("split") == "train")
    val_count = sum(1 for s in samples if s.metadata.get("split") == "validation")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory for images: {temp_dir}")

        all_results = {}
        for order in orders:
            print(f"\n{'='*60}")
            print(f"Evaluating order: {order.value}")
            print(f"{'='*60}")

            order_results = evaluate_ordering(
                model, processor, samples, order,
                token_ids, device, temp_dir, config=config
            )
            all_results[order.value] = order_results

            print(f"\nResults for {order.value}:")
            if order_results["logit_accuracy"] is not None:
                print(f"  Logit accuracy: {order_results['logit_accuracy']:.1%} ({order_results['logit_correct']}/{order_results['logit_evaluable']})")
            else:
                print(f"  Logit accuracy: N/A (no evaluable samples)")

    return {
        "model": config["model_id"],
        "model_size": model_size,
        "dataset": "chartgaze",
        "splits": {"train": train_count, "validation": val_count},
        "num_samples": len(samples),
        "seed": seed,
        "evaluated_at": datetime.now().isoformat(),
        "orders": all_results,
    }


def _run_salchartqa_evaluation(
    model, processor, config,
    model_size: str,
    num_samples: Optional[int],
    seed: int,
    device: str,
    orders: List[InputOrder],
    suffix: str,
    generation_prefix: Optional[str],
) -> Dict[str, Any]:
    """SalChartQA evaluation (generation-based with relaxed matching)."""
    gt_lookup = load_salchartqa_ground_truth()
    print(f"Ground truth: {sum(len(v) for v in gt_lookup.values())} answers across {len(gt_lookup)} images")

    samples = load_salchartqa_samples(num_samples=num_samples, seed=seed)

    all_results = {}
    for order in orders:
        print(f"\n{'='*60}")
        print(f"Evaluating order: {order.value}")
        print(f"{'='*60}")

        order_results = evaluate_ordering_salchartqa(
            model, processor, samples, order,
            gt_lookup, suffix, generation_prefix, device,
            config=config,
        )
        all_results[order.value] = order_results

        print(f"\nResults for {order.value}:")
        if order_results["accuracy"] is not None:
            print(f"  Accuracy: {order_results['accuracy']:.1%} ({order_results['correct']}/{order_results['total_samples']})")
        else:
            print(f"  Accuracy: N/A")
        if order_results["skipped"] > 0:
            print(f"  Skipped (no GT): {order_results['skipped']}")

    return {
        "model": config["model_id"],
        "model_size": model_size,
        "dataset": "salchartqa",
        "num_samples": len(samples),
        "seed": seed,
        "evaluated_at": datetime.now().isoformat(),
        "orders": all_results,
    }


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save evaluation results with informative filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    model_short = results["model_size"].lower()
    dataset = results["dataset"]
    n_samples = results["num_samples"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"correctness_{model_short}_{dataset}_n{n_samples}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")
    return output_path


def print_summary(results: Dict[str, Any]):
    """Print a summary comparison of all orderings."""
    dataset = results.get("dataset", "chartgaze")

    print("\n" + "="*60)
    print(f"SUMMARY: Accuracy by Input Order ({dataset})")
    print("="*60)

    if "splits" in results:
        splits = results["splits"]
        print(f"  Samples: {results['num_samples']} (train: {splits['train']}, val: {splits['validation']})")
    else:
        print(f"  Samples: {results['num_samples']}")
    print()

    orders = results["orders"]

    if dataset == "chartgaze":
        print(f"  {'Order':<25s} | {'Logit Acc':>12s}")
        print(f"  {'-'*25}-+-{'-'*12}")
        for order_name, order_data in orders.items():
            logit_acc = order_data.get("logit_accuracy")
            logit_str = f"{logit_acc:.1%}" if logit_acc is not None else "N/A"
            print(f"  {order_name:<25s} | {logit_str:>12s}")
    else:
        print(f"  {'Order':<25s} | {'Accuracy':>12s}")
        print(f"  {'-'*25}-+-{'-'*12}")
        for order_name, order_data in orders.items():
            acc = order_data.get("accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            print(f"  {order_name:<25s} | {acc_str:>12s}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate input ordering impact on model correctness"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=ALL_MODEL_SIZES,
        help="Model size: 2B, 4B, 3.5-2B, or 3.5-4B"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chartgaze",
        choices=["chartgaze", "salchartqa"],
        help="Dataset to evaluate (default: chartgaze)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(CORRECTNESS_DIR),
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--orders",
        type=str,
        nargs="+",
        choices=[o.value for o in InputOrder],
        default=None,
        help="Specific orders to evaluate (default: all)"
    )

    args = parser.parse_args()

    # Parse orders
    orders = None
    if args.orders:
        orders = [InputOrder(o) for o in args.orders]

    # Run evaluation
    results = run_evaluation(
        model_size=args.model,
        dataset=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
        orders=orders,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_path = save_results(results, Path(args.output_dir))

    return output_path


if __name__ == "__main__":
    main()
