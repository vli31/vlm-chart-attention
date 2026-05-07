"""Compute correctness for yes/no questions using empirical sampling from Qwen 2B or 4B.

Usage:
    python compute_correctness.py --model 2b
    python compute_correctness.py --model 4b

    # With question-image ordering (question → image → suffix):
    python compute_correctness.py --model 4b --token-order question-image

Evaluates correctness via heuristic (checks if response contains yes/no).
Saves all samples for later LLM judge evaluation.

DataFrame columns saved:
- split: "train" or "validation"
- index: sample index within split
- question: the original question
- image_name: name of the image file
- correct_answer: ground truth answer
- samples_orig: JSON list of all samples for original prompt
- samples_suffix: JSON list of all samples for suffix prompt
- heuristic_prob_orig: fraction correct by heuristic (original)
- heuristic_prob_suffix: fraction correct by heuristic (suffix)
"""

import argparse
import json
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.datasets import ChartGazeLoader, CHARTGAZE_CACHE

PROMPT_SUFFIX = " Please answer just 'yes' or 'no'."


def heuristic_check(response: str, ground_truth: str) -> bool:
    """Check correctness using heuristics.

    Checks if response:
    1. Starts with yes/no (after normalization)
    2. Contains yes/no as a standalone word
    """
    response_lower = response.strip().lower()
    gt_lower = ground_truth.strip().lower()

    # Normalize ground truth to yes/no
    if gt_lower not in ("yes", "no"):
        if "yes" in gt_lower:
            gt_lower = "yes"
        elif "no" in gt_lower:
            gt_lower = "no"
        else:
            return False

    # Check 1: starts with yes/no
    if response_lower.startswith("yes") or response_lower.startswith("no"):
        response_answer = "yes" if response_lower.startswith("yes") else "no"
        return response_answer == gt_lower

    # Check 2: contains yes/no as standalone word
    yes_pattern = r'\byes\b'
    no_pattern = r'\bno\b'

    has_yes = bool(re.search(yes_pattern, response_lower))
    has_no = bool(re.search(no_pattern, response_lower))

    if has_yes and not has_no:
        return gt_lower == "yes"
    elif has_no and not has_yes:
        return gt_lower == "no"

    # Ambiguous or no clear answer
    return False


def generate_samples(
    model,
    processor,
    image,
    question: str,
    num_samples: int = 10,
    temperature: float = 0.7,
    token_order: str = "image-question",
    prompt_suffix: str = "",
) -> list[str]:
    """Generate multiple answer samples from the model.

    Args:
        token_order: "image-question" (default) or "question-image"
        prompt_suffix: suffix to append (only used with question-image order)
    """
    if token_order == "image-question":
        # Original order: image → question
        content = [{"type": "image", "image": image}, {"type": "text", "text": question}]
    else:  # question-image
        # New order: question → image → suffix
        content = [
            {"type": "text", "text": question},
            {"type": "image", "image": image},
        ]
        if prompt_suffix:
            content.append({"type": "text", "text": prompt_suffix})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_samples,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    samples = []

    for i in range(num_samples):
        generated_ids = outputs.sequences[i, input_len:]
        text = processor.decode(generated_ids, skip_special_tokens=True)
        samples.append(text.strip())

    return samples


def compute_heuristic_prob(samples: list[str], ground_truth: str) -> float:
    """Compute heuristic correctness probability."""
    heuristic_correct = sum(1 for s in samples if heuristic_check(s, ground_truth))
    return heuristic_correct / len(samples)


def process_split(
    split: str,
    model,
    processor,
    num_samples: int = 10,
    token_order: str = "image-question",
) -> pd.DataFrame:
    """Process a data split and return results DataFrame.

    Args:
        token_order: "image-question" (default) or "question-image"
    """
    loader = ChartGazeLoader(cache_dir=CHARTGAZE_CACHE, data_split=split)
    rows = []

    print(f"\nProcessing {split} split ({len(loader)} samples)...")
    print(f"Token order: {token_order}")

    for i, sample in enumerate(tqdm(loader)):
        gt = sample.answer

        if token_order == "image-question":
            # Original behavior: test with and without suffix appended
            question_orig = sample.question
            question_suffix = sample.question + PROMPT_SUFFIX

            samples_orig = generate_samples(
                model, processor, sample.image, question_orig, num_samples,
                token_order="image-question"
            )
            samples_suffix = generate_samples(
                model, processor, sample.image, question_suffix, num_samples,
                token_order="image-question"
            )
        else:  # question-image
            # New order: question → image → suffix
            # "orig" = question → image (no suffix)
            # "suffix" = question → image → suffix
            samples_orig = generate_samples(
                model, processor, sample.image, sample.question, num_samples,
                token_order="question-image", prompt_suffix=""
            )
            samples_suffix = generate_samples(
                model, processor, sample.image, sample.question, num_samples,
                token_order="question-image", prompt_suffix=PROMPT_SUFFIX
            )

        # Compute heuristic probabilities
        h_orig = compute_heuristic_prob(samples_orig, gt)
        h_suffix = compute_heuristic_prob(samples_suffix, gt)

        # Get image name from path
        image_name = str(sample.image).split("/")[-1] if hasattr(sample.image, "__str__") else f"image_{i}"

        rows.append({
            "split": split,
            "index": i,
            "question": sample.question,
            "image_name": image_name,
            "correct_answer": sample.answer,
            "samples_orig": json.dumps(samples_orig),
            "samples_suffix": json.dumps(samples_suffix),
            "heuristic_prob_orig": h_orig,
            "heuristic_prob_suffix": h_suffix,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from Qwen VL model")
    parser.add_argument("--model", type=str, required=True, choices=["2b", "4b"],
                        help="Model size to use (2b or 4b)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to generate per question (default: 10)")
    parser.add_argument("--token-order", type=str, choices=["image-question", "question-image"],
                        default="image-question",
                        help="Order of tokens: 'image-question' (default) or 'question-image'")
    args = parser.parse_args()

    model_ids = {
        "2b": "Qwen/Qwen3-VL-2B-Instruct",
        "4b": "Qwen/Qwen3-VL-4B-Instruct",
    }
    model_id = model_ids[args.model]

    # Load model on GPU
    print(f"Loading model: {model_id}")
    print(f"Token order: {args.token_order}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Process both splits
    df_train = process_split("train", model, processor, args.num_samples, args.token_order)
    df_val = process_split("validation", model, processor, args.num_samples, args.token_order)

    # Combine and save
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Output filename includes token order if not default
    if args.token_order == "question-image":
        output_file = f"correctness_{args.model}_qimg.csv"
    else:
        output_file = f"correctness_{args.model}.csv"

    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Print summary stats
    print("\n" + "=" * 60)
    print(f"SUMMARY - Qwen {args.model.upper()}")
    print("=" * 60)

    for split in ["train", "validation"]:
        split_df = df[df["split"] == split]
        print(f"\n=== {split.upper()} ({len(split_df)} samples) ===")
        print(f"  Original prompt - Heuristic: {(split_df['heuristic_prob_orig'] > 0.7).mean():.2%}")
        print(f"  With suffix     - Heuristic: {(split_df['heuristic_prob_suffix'] > 0.7).mean():.2%}")


if __name__ == "__main__":
    main()
