#!/usr/bin/env python3
"""
Compute aggregated saliency maps from worker fixation data.

For SalChartQA, uses the correctness determination from correctness_final
(not the original SalChartQA True/False folders).

Usage:
    python scripts/aggregate_saliency.py --dataset massvis
    python scripts/aggregate_saliency.py --dataset taskvis
    python scripts/aggregate_saliency.py --dataset salchartqa
    python scripts/aggregate_saliency.py --all
"""

import argparse
import json
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    print("scipy required: pip install scipy")
    sys.exit(1)

# ── Paths ──
TEMP_DIR = Path('./cache/lvlm-chart')
PERSISTENT_DIR = Path('./data/lvlm-chart')

MASSVIS_IMAGES_DIR = TEMP_DIR / 'eye_gaze_datasets' / 'massvis'
MASSVIS_FIXATIONS_DIR = TEMP_DIR / 'eye_gaze_datasets' / 'massvis_eyetracking' / 'csv_files' / 'fixationsByVis' / 'fixationsByVis'

TASKVIS_DIR = TEMP_DIR / 'eye_gaze_datasets' / 'taskvis'
SALCHARTQA_DIR = Path('./data/salchartqa')

CORRECTNESS_DIR = PERSISTENT_DIR / 'correctness'
GT_FILE = CORRECTNESS_DIR / 'determined_ground_truth_fixed.json'
GT_MERGED_FILE = PERSISTENT_DIR / 'ground_truth' / 'gpt-5-mini-2025-08-07_merged_all.json'

OUTPUT_BASE = TEMP_DIR / 'aggregated_saliency'

SIGMA = 25  # Gaussian blur sigma
TARGET_H, TARGET_W = 600, 850


# ── relaxed_match import (reuse from correctness_final) ──
sys.path.insert(0, './scripts/lib')
try:
    from correctness_final.evaluate_ordering import relaxed_match, normalize_answer
except ImportError:
    print("Warning: Could not import relaxed_match from correctness_final")
    def relaxed_match(pred, gt):
        return pred.strip().lower() == gt.strip().lower()
    def normalize_answer(s):
        return s.strip().lower()


def read_fixation_csv(path: Path) -> List[Tuple[float, float, float]]:
    """Read fixation CSV file. Returns list of (x, y, duration) tuples."""
    fixations = []
    try:
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        # Format: id, x, y, duration (or just x, y, duration)
                        if len(row) >= 4:
                            x, y, dur = float(row[1]), float(row[2]), float(row[3])
                        else:
                            x, y, dur = float(row[0]), float(row[1]), float(row[2])
                        if x >= 0 and y >= 0 and dur > 0:
                            fixations.append((x, y, dur))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"  Warning: Could not read {path}: {e}")
    return fixations


def read_taskvis_fixation_tsv(path: Path) -> List[Tuple[float, float, float]]:
    """Read TaskVIS fixation TSV file. Returns list of (x, y, duration) tuples."""
    fixations = []
    try:
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            rows = list(reader)
            for i, row in enumerate(rows):
                if len(row) >= 4:
                    try:
                        x = float(row[2])  # FixationPointX
                        y = float(row[3])  # FixationPointY
                        # Duration: difference to next timestamp, or use fixed duration
                        if i + 1 < len(rows):
                            try:
                                dur = float(rows[i+1][0]) - float(row[0])
                            except (ValueError, IndexError):
                                dur = 200  # default duration
                        else:
                            dur = 200
                        if x >= 0 and y >= 0 and dur > 0:
                            fixations.append((x, y, dur))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"  Warning: Could not read {path}: {e}")
    return fixations


def fixations_to_heatmap(fixations: List[Tuple[float, float, float]],
                          img_w: int, img_h: int,
                          target_h: int = TARGET_H, target_w: int = TARGET_W,
                          sigma: float = SIGMA) -> np.ndarray:
    """Convert fixation list [(x, y, duration), ...] to a blurred heatmap."""
    fix_map = np.zeros((target_h, target_w), dtype=np.float64)
    for x, y, dur in fixations:
        sx = int(x * target_w / img_w)
        sy = int(y * target_h / img_h)
        if 0 <= sx < target_w and 0 <= sy < target_h:
            fix_map[sy, sx] += dur
    if fix_map.max() > 0:
        heatmap = gaussian_filter(fix_map, sigma=sigma)
        heatmap = heatmap / heatmap.max()
    else:
        heatmap = fix_map
    return heatmap


def save_heatmap(heatmap: np.ndarray, path: Path):
    """Save heatmap as grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
    img.save(path)


def get_image_size(img_path: Path) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    with Image.open(img_path) as img:
        return img.size


# ═══════════════════════════════════════════════════════════════
# MASSVIS
# ═══════════════════════════════════════════════════════════════

def aggregate_massvis():
    """Aggregate MASSVIS fixations across all workers per visualization."""
    out_dir = OUTPUT_BASE / 'massvis' / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not MASSVIS_FIXATIONS_DIR.exists():
        print(f"MASSVIS fixations not found: {MASSVIS_FIXATIONS_DIR}")
        return

    vis_dirs = sorted([d for d in MASSVIS_FIXATIONS_DIR.iterdir() if d.is_dir()])
    print(f"Processing {len(vis_dirs)} MASSVIS visualizations...")

    for vis_dir in vis_dirs:
        vis_name = vis_dir.name
        out_path = out_dir / f'{vis_name}.png'

        if out_path.exists():
            continue

        img_path = MASSVIS_IMAGES_DIR / f'{vis_name}.png'
        if not img_path.exists():
            continue

        img_w, img_h = get_image_size(img_path)

        # Collect fixations from all workers (encoding phase)
        enc_dir = vis_dir / 'enc'
        if not enc_dir.exists():
            continue

        all_fixations = []
        for fix_file in enc_dir.iterdir():
            if fix_file.suffix == '.csv':
                fixations = read_fixation_csv(fix_file)
                all_fixations.extend(fixations)

        if all_fixations:
            heatmap = fixations_to_heatmap(all_fixations, img_w, img_h)
            save_heatmap(heatmap, out_path)

    count = len(list(out_dir.glob('*.png')))
    print(f"MASSVIS: {count} aggregated heatmaps saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════
# TASKVIS
# ═══════════════════════════════════════════════════════════════

def aggregate_taskvis():
    """Aggregate TaskVIS fixations per (image, task) for all/correct/incorrect."""
    out_all = OUTPUT_BASE / 'taskvis' / 'all'
    out_correct = OUTPUT_BASE / 'taskvis' / 'correct'
    out_incorrect = OUTPUT_BASE / 'taskvis' / 'incorrect'
    for d in [out_all, out_correct, out_incorrect]:
        d.mkdir(parents=True, exist_ok=True)

    # Load TaskVIS metadata
    # images.txt: imageID -> filename
    image_map = {}
    with open(TASKVIS_DIR / 'images.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == 'imageID':
                continue
            image_map[int(parts[0])] = parts[1]

    # tasks.txt: (imageID, task) -> question
    tasks = {}
    with open(TASKVIS_DIR / 'tasks.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == 'imageID':
                continue
            tasks[(int(parts[0]), parts[1])] = parts[2]

    # tasktypes.txt: participant -> {imageID: task_type}
    participant_tasks = {}
    with open(TASKVIS_DIR / 'tasktypes.txt') as f:
        header = f.readline().strip().split('\t')
        image_ids = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split('\t')
            user = parts[0]
            participant_tasks[user] = {}
            for i, task_type in enumerate(parts[1:]):
                participant_tasks[user][image_ids[i]] = task_type

    # answers.txt: participant -> {imageID: correct (0/1)}
    participant_answers = {}
    with open(TASKVIS_DIR / 'answers.txt') as f:
        header = f.readline().strip().split('\t')
        image_ids_ans = [int(x) for x in header[1:]]
        for line in f:
            parts = line.strip().split('\t')
            user = parts[0]
            participant_answers[user] = {}
            for i, ans in enumerate(parts[1:]):
                participant_answers[user][image_ids_ans[i]] = int(ans)

    print(f"TaskVIS: {len(image_map)} images, {len(tasks)} tasks, {len(participant_tasks)} participants")

    # For each (image_id, task_type), collect fixations from relevant participants
    for (img_id, task_type), question in sorted(tasks.items()):
        sample_id = f'taskvis_{img_id}_{task_type}'

        out_all_path = out_all / f'{sample_id}.png'
        out_corr_path = out_correct / f'{sample_id}.png'
        out_incorr_path = out_incorrect / f'{sample_id}.png'

        if out_all_path.exists() and out_corr_path.exists() and out_incorr_path.exists():
            continue

        img_filename = image_map[img_id]
        img_path = MASSVIS_IMAGES_DIR / img_filename
        if not img_path.exists():
            continue

        img_w, img_h = get_image_size(img_path)

        all_fix = []
        correct_fix = []
        incorrect_fix = []

        for user, user_tasks in participant_tasks.items():
            if user_tasks.get(img_id) != task_type:
                continue

            # Find fixation file
            user_num = user.replace('P', '')
            fix_path = TASKVIS_DIR / 'fixations' / f'rec_p{user_num}_fix_{img_id}.tsv'
            if not fix_path.exists():
                continue

            fixations = read_taskvis_fixation_tsv(fix_path)
            all_fix.extend(fixations)

            is_correct = participant_answers.get(user, {}).get(img_id, 0)
            if is_correct == 1:
                correct_fix.extend(fixations)
            else:
                incorrect_fix.extend(fixations)

        if all_fix and not out_all_path.exists():
            save_heatmap(fixations_to_heatmap(all_fix, img_w, img_h), out_all_path)
        if correct_fix and not out_corr_path.exists():
            save_heatmap(fixations_to_heatmap(correct_fix, img_w, img_h), out_corr_path)
        if incorrect_fix and not out_incorr_path.exists():
            save_heatmap(fixations_to_heatmap(incorrect_fix, img_w, img_h), out_incorr_path)

    counts = {k: len(list(d.glob('*.png'))) for k, d in
              [('all', out_all), ('correct', out_correct), ('incorrect', out_incorrect)]}
    print(f"TaskVIS aggregated: {counts}")


# ═══════════════════════════════════════════════════════════════
# SALCHARTQA
# ═══════════════════════════════════════════════════════════════

def aggregate_salchartqa():
    """Recompute SalChartQA correct/incorrect aggregated saliency using new GT."""
    out_all = OUTPUT_BASE / 'salchartqa' / 'all'
    out_correct = OUTPUT_BASE / 'salchartqa' / 'correct'
    out_incorrect = OUTPUT_BASE / 'salchartqa' / 'incorrect'
    for d in [out_all, out_correct, out_incorrect]:
        d.mkdir(parents=True, exist_ok=True)

    # Load determined ground truth
    with open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_lookup = gt_data['gt_lookup']
    print(f"Loaded GT for {len(gt_lookup)} questions")

    # Load unified_approved.csv for worker responses
    worker_responses = defaultdict(list)  # (image_stem, Q_id) -> [(participant_id, answer)]
    with open(SALCHARTQA_DIR / 'unified_approved.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('is_approved', 'True') != 'True':
                continue
            img_name = row['image_name']
            question = row['question']
            answer = row['answer']
            participant_id = row['participant_id']
            worker_responses[(img_name, question)].append({
                'participant_id': participant_id,
                'answer': answer,
            })

    # Load image_questions.json for question mapping
    with open(SALCHARTQA_DIR / 'image_questions.json') as f:
        image_questions = json.load(f)

    # Build question -> (image_name, Q_id) mapping
    question_to_key = {}
    for img_name, qdict in image_questions.items():
        for q_id, question in qdict.items():
            question_to_key[(img_name, question)] = (img_name, q_id)

    fixation_base = SALCHARTQA_DIR / 'fixationByVis'

    processed = 0
    skipped_no_gt = 0

    for img_name, qdict in sorted(image_questions.items()):
        img_stem = Path(img_name).stem
        img_path = SALCHARTQA_DIR / 'raw_img' / img_name

        if not img_path.exists():
            continue

        img_w, img_h = get_image_size(img_path)

        for q_id, question in qdict.items():
            gt_key = f'{img_name}_{q_id}'
            gt_info = gt_lookup.get(gt_key)

            out_all_path = out_all / f'{img_stem}_{q_id}.png'
            out_corr_path = out_correct / f'{img_stem}_{q_id}.png'
            out_incorr_path = out_incorrect / f'{img_stem}_{q_id}.png'

            # Skip if already computed
            if out_all_path.exists() and out_corr_path.exists():
                processed += 1
                continue

            # Collect all fixation files from both True/ and False/ subdirs
            fix_dir = fixation_base / img_stem / q_id
            if not fix_dir.exists():
                continue

            all_fix = []
            correct_fix = []
            incorrect_fix = []

            for correctness_dir in ['True', 'False']:
                sub_dir = fix_dir / correctness_dir
                if not sub_dir.exists():
                    continue
                for fix_file in sub_dir.iterdir():
                    if fix_file.suffix != '.csv':
                        continue
                    worker_id = fix_file.stem
                    fixations = read_fixation_csv(fix_file)
                    all_fix.extend(fixations)

                    # Determine new correctness
                    if gt_info and gt_info.get('in_confident_subset', False):
                        gt_answer = gt_info['gt']
                        # Find this worker's answer
                        worker_answer = None
                        for resp in worker_responses.get((img_name, question), []):
                            if resp['participant_id'] == worker_id:
                                worker_answer = resp['answer']
                                break

                        if worker_answer is not None:
                            is_correct = relaxed_match(worker_answer, gt_answer)
                            if is_correct:
                                correct_fix.extend(fixations)
                            else:
                                incorrect_fix.extend(fixations)
                        else:
                            # Can't determine - use original folder
                            if correctness_dir == 'True':
                                correct_fix.extend(fixations)
                            else:
                                incorrect_fix.extend(fixations)
                    else:
                        skipped_no_gt += 1
                        # Not in confident subset - use original folder assignment
                        if correctness_dir == 'True':
                            correct_fix.extend(fixations)
                        else:
                            incorrect_fix.extend(fixations)

            if all_fix and not out_all_path.exists():
                save_heatmap(fixations_to_heatmap(all_fix, img_w, img_h), out_all_path)
            if correct_fix and not out_corr_path.exists():
                save_heatmap(fixations_to_heatmap(correct_fix, img_w, img_h), out_corr_path)
            if incorrect_fix and not out_incorr_path.exists():
                save_heatmap(fixations_to_heatmap(incorrect_fix, img_w, img_h), out_incorr_path)

            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed} questions...")

    counts = {k: len(list(d.glob('*.png'))) for k, d in
              [('all', out_all), ('correct', out_correct), ('incorrect', out_incorrect)]}
    print(f"SalChartQA aggregated: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate saliency maps")
    parser.add_argument('--dataset', choices=['massvis', 'taskvis', 'salchartqa'],
                        help='Dataset to process')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    args = parser.parse_args()

    if args.all or args.dataset == 'massvis':
        print("=" * 60)
        print("Aggregating MASSVIS saliency...")
        print("=" * 60)
        aggregate_massvis()

    if args.all or args.dataset == 'taskvis':
        print("=" * 60)
        print("Aggregating TaskVIS saliency...")
        print("=" * 60)
        aggregate_taskvis()

    if args.all or args.dataset == 'salchartqa':
        print("=" * 60)
        print("Aggregating SalChartQA saliency (recomputed correctness)...")
        print("=" * 60)
        aggregate_salchartqa()


if __name__ == '__main__':
    main()
