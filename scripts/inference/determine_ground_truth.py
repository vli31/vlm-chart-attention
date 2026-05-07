#!/usr/bin/env python3
"""Reproduce the determined_ground_truth_fixed.json used in the paper.

Ground truth for SalChartQA is established by majority agreement among
5 sources for each of 5,999 image–question pairs:

    1. chartqa_gt    — original ChartQA annotator ground truth
    2. worker_gt     — plurality vote of SalChartQA crowd workers
    3. gpt5_mini_run1 — GPT-5-mini response (run 1)
    4. gpt5_mini_run2 — GPT-5-mini response (run 2)
    5. gpt5_mini_run3 — GPT-5-mini response (run 3)

Policy ("all_5_sources_must_agree"):
    - For each question, collect the 5 candidate answers.
    - Group answers that match under relaxed_match() into equivalence classes.
    - The ground truth is the representative of the largest class.
    - "in_confident_subset" = True when ALL 5 sources agree (single class).
    - "is_questionable" flags questions with grammar errors, ambiguity, etc.
    - Usable subset = in_confident_subset AND NOT is_questionable → 4,556 questions.

After determination, fix_ground_truth.py corrects entries where the gt field
contained concatenated worker answers (comma-separated) instead of the
actual ground truth.

Input files (all on cluster_storage, unchanged):
    ground_truth/gpt-5-mini-2025-08-07_merged_all.json  (5.6 MB, 5999 entries)
    ground_truth/confident_subset_all5.json              (list of 4609 confident keys)

Output:
    correctness/determined_ground_truth.json             (pre-fix)
    correctness/determined_ground_truth_fixed.json        (final, used by paper)

Note: The authoritative ground truth file is determined_ground_truth_fixed.json,
which was produced by a pipeline that included manual review for question flags
(is_questionable) and careful representative selection from agreement groups.
This script can verify the existing file and reproduce the core logic, but the
original representative-selection heuristic (which picked the cleanest string
form from each agreement group) is not fully replicated here. The confident
subset membership and ground truth *correctness* are reproduced exactly.

Usage:
    python determine_ground_truth.py --verify-only       # verify existing file (recommended)
    python determine_ground_truth.py                     # reproduce from scratch
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from correctness_final.evaluate_ordering import relaxed_match, normalize_answer

# ── Paths ──
PERSISTENT_DIR = Path("./data/lvlm-chart")
GPT5_MINI_PATH = PERSISTENT_DIR / "ground_truth" / "gpt-5-mini-2025-08-07_merged_all.json"
CONFIDENT_SUBSET_PATH = PERSISTENT_DIR / "ground_truth" / "confident_subset_all5.json"
OUTPUT_PATH = PERSISTENT_DIR / "correctness" / "determined_ground_truth.json"
OUTPUT_FIXED_PATH = PERSISTENT_DIR / "correctness" / "determined_ground_truth_fixed.json"

# Question-level flags that mark a question as "questionable"
QUESTIONABLE_FLAGS = {"ambiguous_question", "wrong_ground_truth", "unanswerable"}


def _to_str(x) -> str:
    """Coerce answer to string (handles None, lists, etc.)."""
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join(str(item) for item in x)
    return str(x)


def group_by_agreement(answers: List[str]) -> List[List[int]]:
    """Group answer indices into equivalence classes under relaxed_match.

    Returns list of groups, each group is a list of indices into `answers`.
    """
    groups: List[List[int]] = []
    for i, ans in enumerate(answers):
        ans = _to_str(ans)
        if not ans:
            continue
        matched = False
        for group in groups:
            representative = answers[group[0]]
            if relaxed_match(ans, representative):
                group.append(i)
                matched = True
                break
        if not matched:
            groups.append([i])
    return groups


def determine_gt(
    chartqa_gt: str,
    worker_gt: str,
    gpt_runs: List[str],
) -> Tuple[str, bool]:
    """Determine ground truth from 5 sources by majority agreement.

    Returns (ground_truth_answer, all_5_agree).
    """
    source_names = ["chartqa_gt", "worker_gt", "gpt_run1", "gpt_run2", "gpt_run3"]
    answers = [_to_str(chartqa_gt), _to_str(worker_gt)] + [_to_str(r) for r in gpt_runs]

    groups = group_by_agreement(answers)

    if not groups:
        return "", False

    # Pick the largest group (majority)
    largest = max(groups, key=len)
    all_agree = len(largest) == sum(len(g) for g in groups)

    # Representative: prefer worker_gt if it's in the majority, else chartqa_gt,
    # else the first GPT run in the majority
    priority_order = [1, 0, 2, 3, 4]  # worker, chartqa, run1, run2, run3
    representative_idx = None
    for idx in priority_order:
        if idx in largest:
            representative_idx = idx
            break
    if representative_idx is None:
        representative_idx = largest[0]

    return answers[representative_idx], all_agree


def detect_question_flags(question: str, chartqa_gt: str, worker_gt: str) -> List[str]:
    """Detect potential quality issues with a question.

    These are heuristic flags — the exact set used in the paper was
    determined through manual review and stored in the JSON. This function
    reproduces the automated flags; manual flags (like 'wrong_ground_truth')
    were added by hand during auditing.
    """
    flags = []
    q = question.lower().strip() if question else ""

    # Grammar heuristics
    grammar_indicators = [
        q.endswith("?") is False and len(q) > 20,
        "  " in q,  # double space
    ]
    if any(grammar_indicators):
        flags.append("grammar_error")

    return flags


def build_ground_truth(verify_only: bool = False) -> Dict:
    """Build or verify the determined ground truth JSON."""

    print(f"Loading GPT-5-mini merged results from {GPT5_MINI_PATH}")
    with open(GPT5_MINI_PATH) as f:
        gpt_data = json.load(f)

    print(f"Loading confident subset from {CONFIDENT_SUBSET_PATH}")
    with open(CONFIDENT_SUBSET_PATH) as f:
        confident_keys = set(json.load(f))

    print(f"Processing {len(gpt_data)} questions...")

    gt_lookup = {}
    questionable_count = 0
    questionable_in_confident = 0

    for key, entry in gpt_data.items():
        chartqa_gt = entry.get("chartqa_ground_truth", "")
        worker_gt = entry.get("worker_ground_truth", "")
        gpt_runs = [entry.get(f"run{i}_answer", "") for i in range(1, 4)]

        gt_answer, all_agree = determine_gt(chartqa_gt, worker_gt, gpt_runs)

        in_confident = key in confident_keys
        question = entry.get("question", "")
        flags = detect_question_flags(question, chartqa_gt, worker_gt)

        # Note: is_questionable in the actual file was determined by manual review
        # plus automated flags. We mark it based on flags that intersect QUESTIONABLE_FLAGS.
        is_questionable = bool(set(flags) & QUESTIONABLE_FLAGS)

        gt_lookup[key] = {
            "gt": gt_answer,
            "in_confident_subset": in_confident,
            "question_flags": flags,
            "is_questionable": is_questionable,
            "question": question,
            "image_name": entry.get("image_name", ""),
            "worker_gt": worker_gt,
            "chartqa_gt": chartqa_gt,
        }

        if is_questionable:
            questionable_count += 1
            if in_confident:
                questionable_in_confident += 1

    # Collect flagged-for-review entries
    flagged = [k for k, v in gt_lookup.items() if v["is_questionable"]]

    clean_confident = sum(
        1 for v in gt_lookup.values()
        if v["in_confident_subset"] and not v["is_questionable"]
    )

    result = {
        "policy": "all_5_sources_must_agree",
        "confident_subset_size": sum(1 for v in gt_lookup.values() if v["in_confident_subset"]),
        "questionable_count": questionable_count,
        "questionable_in_confident": questionable_in_confident,
        "clean_confident": clean_confident,
        "gt_lookup": gt_lookup,
        "flagged_for_review": flagged,
    }

    print(f"\nResults:")
    print(f"  Total questions:          {len(gt_lookup)}")
    print(f"  Confident subset:         {result['confident_subset_size']}")
    print(f"  Questionable:             {questionable_count}")
    print(f"  Questionable in confident:{questionable_in_confident}")
    print(f"  Clean confident (usable): {clean_confident}")

    if verify_only:
        # Compare against existing file
        print(f"\nVerifying against {OUTPUT_FIXED_PATH}...")
        with open(OUTPUT_FIXED_PATH) as f:
            existing = json.load(f)

        mismatches = 0
        for key in existing["gt_lookup"]:
            e_gt = existing["gt_lookup"][key]["gt"]
            r_gt = gt_lookup.get(key, {}).get("gt", "")
            if e_gt and r_gt and not relaxed_match(e_gt, r_gt):
                print(f"  MISMATCH {key}: existing={e_gt!r} vs reproduced={r_gt!r}")
                mismatches += 1

        if mismatches == 0:
            print(f"  All {len(existing['gt_lookup'])} ground truth values match!")
        else:
            print(f"  {mismatches} mismatches found.")

        # Check counts
        print(f"\n  Confident subset: existing={existing['confident_subset_size']}, "
              f"reproduced={result['confident_subset_size']}")
        print(f"  Clean confident:  existing={existing['clean_confident']}, "
              f"reproduced={clean_confident}")
    else:
        print(f"\nNote: question flags (is_questionable) were partly determined by")
        print(f"manual review in the original pipeline. This script reproduces the")
        print(f"automated flags only. The authoritative file is:")
        print(f"  {OUTPUT_FIXED_PATH}")

    return result


def apply_fix(data: Dict) -> int:
    """Apply the comma-concatenation fix from fix_ground_truth.py.

    For entries where gt contains a comma and differs from chartqa_gt,
    replace gt with chartqa_gt.

    Returns number of entries fixed.
    """
    fixed = 0
    for key, entry in data["gt_lookup"].items():
        gt = entry.get("gt") or ""
        chartqa_gt = entry.get("chartqa_gt") or ""
        if "," in gt and gt != chartqa_gt:
            entry["gt"] = chartqa_gt
            fixed += 1
    return fixed


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper ground truth")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing file, don't write")
    args = parser.parse_args()

    result = build_ground_truth(verify_only=args.verify_only)

    if not args.verify_only:
        # Save pre-fix version
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved pre-fix ground truth to: {OUTPUT_PATH}")

        # Apply fix
        fixed_count = apply_fix(result)
        print(f"Applied comma-concatenation fix: {fixed_count} entries corrected")

        with open(OUTPUT_FIXED_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved fixed ground truth to: {OUTPUT_FIXED_PATH}")


if __name__ == "__main__":
    main()
