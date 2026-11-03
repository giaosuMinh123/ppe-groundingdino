"""
Evaluation and visualization of GroundingDINO PPE detection results.

The script matches predictions against ground truth (GT) by IoU and produces,
for each run:
  * per-class Precision / Recall / F1 at the operating threshold,
  * per-class AP and mAP at a single IoU threshold,
  * optional COCO-style mAP@[0.5:0.95],
  * PR curves and confusion matrices,
  * a text report plus CSV / JSON summaries.
When several runs are given, it also draws side-by-side comparison charts.

--------------------------------------------------------------------------
Input
--------------------------------------------------------------------------
Ground truth (--gt_dir)
    Directory of YOLO-format .txt files:
        class_id x_center y_center width height     (normalized to 0-1)
    The file name (without extension) is the matching key. If the name
    contains "-", only the part after the last "-" is used as the key, which
    mirrors how the inference script builds its image ids.

Predictions (--pred_dir or --runs), produced by grounding_dino_ppe_optimized.py
    pred-<key>.txt   class_id x y w h          (no score; boxes that passed the
                                                display / operating threshold)
    pred-<key>.csv   class_id,x,y,w,h,score    (boxes that passed the low
                                                capture threshold, with scores)

Which file feeds which metric
    .csv  ->  AP, mAP, PR curves          (needs scores to rank detections)
    .txt  ->  Precision, Recall, F1, TP / FP / FN, confusion matrices
              (single operating point)
    If a run has no .csv, the .txt boxes are used with score = 1.0 for every
    box. P/R/F1 stay valid, but AP / PR curves degrade to a single point.

--------------------------------------------------------------------------
Output files (per run, inside <output_dir>/<run_name>/)
--------------------------------------------------------------------------
    report.txt                                   Human-readable per-class table
    per_class_metrics.csv                        Same table, machine-readable
    summary.json                                 Overall + per-class metrics
    per_class_precision_recall_f1_ap_bar.png     Grouped bars: P, R, F1, AP per class
    per_class_ap_bar.png                         AP per class with mAP reference line
    per_class_gt_vs_pred_box_count_bar.png       GT vs predicted box counts per class
    per_class_pr_curves_grid.png                 One PR curve per class (subplots)
    per_class_pr_curves_overlay.png              All per-class PR curves in one axes
    overall_pr_curve.png                         PR curve averaged over classes (area = mAP)
    confusion_matrix_with_background.png         Full confusion matrix (TP on diagonal)
    misclassified_gt_heatmap.png                 Missed GT covered by another class
  With --map_sweep:
    map_50_95.json                               mAP@[0.5:0.95], overall and per class
    per_class_ap_vs_iou_lines.png                AP of each class vs IoU threshold
    map_vs_iou_line.png                          mAP vs IoU threshold
    pr_curve_<class>_by_iou.png                  (with --iou_sweep_classes) PR curve of
                                                 one class at every IoU level
  Multiple runs (<output_dir>/comparison/):
    compare_runs_per_class_f1_bar.png
    compare_runs_per_class_recall_bar.png
    compare_runs_per_class_ap_bar.png
    compare_runs_overall_metrics_bar.png
    all_runs_summary.json

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
Single run:
    python evaluate_visualize.py --gt_dir path/to/labels --classes classes.txt \
        --pred_dir path/to/predictions --iou_threshold 0.5 --output_dir eval_report

Multiple runs (JSON mapping run name -> prediction directory, or a JSON file):
    python evaluate_visualize.py --gt_dir path/to/labels --classes classes.txt \
        --runs '{"combined": "out_combined", "per_class": "out_per_class"}' \
        --output_dir eval_report
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless backend: figures are only written to disk.
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Output file names are defined once so that the code and the docstring above
# stay in sync.
FILE_REPORT = "report.txt"
FILE_PER_CLASS_CSV = "per_class_metrics.csv"
FILE_SUMMARY_JSON = "summary.json"
FILE_PRF_AP_BAR = "per_class_precision_recall_f1_ap_bar.png"
FILE_AP_BAR = "per_class_ap_bar.png"
FILE_COUNT_BAR = "per_class_gt_vs_pred_box_count_bar.png"
FILE_PR_GRID = "per_class_pr_curves_grid.png"
FILE_PR_OVERLAY = "per_class_pr_curves_overlay.png"
FILE_PR_OVERALL = "overall_pr_curve.png"
FILE_CONFUSION_FULL = "confusion_matrix_with_background.png"
FILE_CONFUSION_MISCLASS = "misclassified_gt_heatmap.png"
FILE_MAP_JSON = "map_50_95.json"
FILE_AP_VS_IOU = "per_class_ap_vs_iou_lines.png"
FILE_MAP_VS_IOU = "map_vs_iou_line.png"
FILE_PR_BY_IOU_TEMPLATE = "pr_curve_{class_name}_by_iou.png"
COMPARE_DIR = "comparison"
FILE_COMPARE_F1 = "compare_runs_per_class_f1_bar.png"
FILE_COMPARE_RECALL = "compare_runs_per_class_recall_bar.png"
FILE_COMPARE_AP = "compare_runs_per_class_ap_bar.png"
FILE_COMPARE_OVERALL = "compare_runs_overall_metrics_bar.png"
FILE_COMPARE_JSON = "all_runs_summary.json"

FIG_DPI = 150

# A detection is (class_id, x_center, y_center, width, height, score).
# A ground-truth box is (class_id, x_center, y_center, width, height).


# --------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------
def load_class_names(classes_path: str) -> list[str]:
    """Read one class name per line (lower-cased); the line index is the class id."""
    with open(classes_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_path}")
    return names


def read_gt_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file. A missing file means an image without objects."""
    boxes = []
    if not path.exists():
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append((class_id, x, y, w, h))
    return boxes


def read_pred_txt_for_key(pred_dir: Path, key: str) -> list[tuple[int, float, float, float, float, float]]:
    """
    Read pred-<key>.txt: the boxes that passed the operating (display) threshold.

    Used for single-point metrics (Precision / Recall / F1, TP / FP / FN and
    both confusion matrices). The file carries no score, so every box gets
    score = 1.0; the score is irrelevant here because the keep/drop decision
    has already been made at inference time.
    """
    txt_path = pred_dir / f"pred-{key}.txt"
    detections = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                detections.append((class_id, x, y, w, h, 1.0))
    return detections


def read_pred_csv_for_key(pred_dir: Path, key: str) -> list[tuple[int, float, float, float, float, float]]:
    """
    Read pred-<key>.csv: all candidates above the low capture threshold, with scores.

    Used only for AP / mAP / PR curves, which need the full score-ranked list.
    If the .csv is missing, fall back to the .txt boxes with score = 1.0; AP and
    PR curves are then computed from a single operating point and are not
    meaningful.
    """
    csv_path = pred_dir / f"pred-{key}.csv"
    detections = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            next(f, None)  # Skip the header row.
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                class_id = int(float(parts[0]))
                x, y, w, h, score = map(float, parts[1:6])
                detections.append((class_id, x, y, w, h, score))
    elif (pred_dir / f"pred-{key}.txt").exists():
        detections = read_pred_txt_for_key(pred_dir, key)
    return detections


def build_gt_index(gt_dir: Path) -> dict[str, Path]:
    """
    Map matching key -> GT file path.

    A GT file may be named "209230.txt" or keep the original image name, e.g.
    "pexels-photo-209230.txt". Both reduce to the same key by taking the part
    after the last "-", which is how the inference script derives image ids.
    """
    index: dict[str, Path] = {}
    collisions = []
    for path in gt_dir.glob("*.txt"):
        key = path.stem.split("-")[-1] if "-" in path.stem else path.stem
        if key in index:
            collisions.append((key, index[key].name, path.name))
        index[key] = path
    if collisions:
        print(f"[warn] {len(collisions)} GT keys collide after shortening file names "
              f"(later files overwrite earlier ones), e.g. {collisions[:3]}")
    return index


def build_pred_keys(pred_dir: Path) -> set:
    """Collect the keys of all pred-<key>.txt / pred-<key>.csv files."""
    keys = set()
    for pattern in ("pred-*.txt", "pred-*.csv"):
        for path in pred_dir.glob(pattern):
            keys.add(path.stem[len("pred-"):])
    return keys


def match_keys(gt_dir: Path, pred_dir: Path):
    """Return (sorted keys present in both GT and predictions, GT index)."""
    gt_index = build_gt_index(gt_dir)
    pred_keys = build_pred_keys(pred_dir)
    gt_keys = set(gt_index.keys())

    common = sorted(gt_keys & pred_keys)
    missing_pred = gt_keys - pred_keys
    missing_gt = pred_keys - gt_keys

    print(f"[match] GT files: {len(gt_keys)} | Pred files: {len(pred_keys)} | Matched: {len(common)}")
    if missing_pred:
        print(f"[warn] {len(missing_pred)} images have GT but no prediction (e.g. {list(missing_pred)[:5]})")
    if missing_gt:
        print(f"[warn] {len(missing_gt)} predictions have no GT file (e.g. {list(missing_gt)[:5]})")
    if not common:
        raise RuntimeError(
            "No GT file could be matched with a prediction file. Check the file names: "
            "the part of the GT file name after the last '-' must equal the <key> in "
            "'pred-<key>.txt/.csv'. An empty prediction folder means inference did not run."
        )
    return common, gt_index


def validate_class_ids(gt_by_key, pred_by_key_txt, pred_by_key_csv, class_names):
    """
    Fail early with a clear message if any GT or prediction class id lies
    outside 0..len(class_names)-1, instead of crashing later with an IndexError.
    """
    n_classes = len(class_names)
    bad_gt, bad_pred = Counter(), Counter()

    for boxes in gt_by_key.values():
        for box in boxes:
            if box[0] < 0 or box[0] >= n_classes:
                bad_gt[box[0]] += 1
    for detections in list(pred_by_key_txt.values()) + list(pred_by_key_csv.values()):
        for det in detections:
            if det[0] < 0 or det[0] >= n_classes:
                bad_pred[det[0]] += 1

    if bad_gt or bad_pred:
        lines = [
            f"[ERROR] Class id outside the range declared in the classes file "
            f"({n_classes} classes, valid ids are 0..{n_classes - 1})."
        ]
        if bad_gt:
            lines.append(f"  - In ground truth : {dict(sorted(bad_gt.items()))} (class_id: occurrences)")
        if bad_pred:
            lines.append(f"  - In predictions  : {dict(sorted(bad_pred.items()))} (class_id: occurrences)")
        lines.append(
            "-> The dataset may have more classes than the classes file. Check the dataset's "
            "own class definition (e.g. data.yaml) and update the classes file."
        )
        raise RuntimeError("\n".join(lines))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def cxcywh_to_xyxy(box):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_xyxy(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# Metrics at a single operating point (uses the .txt predictions)
# --------------------------------------------------------------------------
def compute_pointwise_metrics(gt_by_key, pred_by_key, keys, class_names, iou_threshold):
    """
    Per-class TP / FP / FN, Precision, Recall and F1 at one operating point.

    Matching is greedy per image: same-class (prediction, GT) pairs with
    IoU >= threshold are matched in descending IoU order, one-to-one.
      TP : matched prediction.
      FP : prediction left unmatched.
      FN : GT left unmatched.
    For every FN the function also looks for a prediction of a DIFFERENT class
    that overlaps the GT with IoU >= threshold; the best one is recorded in
    `confusion[gt_class][pred_class]` (object found but labelled wrongly).
    """
    n_classes = len(class_names)
    tp = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    n_gt = np.zeros(n_classes, dtype=int)
    n_pred = np.zeros(n_classes, dtype=int)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for key in keys:
        gts = gt_by_key[key]
        preds = pred_by_key[key]

        for gt in gts:
            n_gt[gt[0]] += 1
        for pred in preds:
            n_pred[pred[0]] += 1

        gt_xyxy = [(gt[0], cxcywh_to_xyxy(gt[1:5])) for gt in gts]
        pred_xyxy = [(p[0], cxcywh_to_xyxy(p[1:5])) for p in preds]

        matched_gt = [False] * len(gt_xyxy)
        matched_pred = [False] * len(pred_xyxy)

        # Step 1: match same-class pairs, highest IoU first.
        candidates = []
        for pi, (pred_cls, pred_box) in enumerate(pred_xyxy):
            for gi, (gt_cls, gt_box) in enumerate(gt_xyxy):
                if pred_cls != gt_cls:
                    continue
                iou = iou_xyxy(pred_box, gt_box)
                if iou >= iou_threshold:
                    candidates.append((iou, pi, gi))
        candidates.sort(key=lambda c: -c[0])
        for _, pi, gi in candidates:
            if matched_pred[pi] or matched_gt[gi]:
                continue
            matched_pred[pi] = True
            matched_gt[gi] = True
            tp[gt_xyxy[gi][0]] += 1

        # Step 2: unmatched predictions are false positives.
        for pi, (pred_cls, _) in enumerate(pred_xyxy):
            if not matched_pred[pi]:
                fp[pred_cls] += 1

        # Step 3: unmatched GT are false negatives; record whether another
        # class's prediction covers them.
        for gi, (gt_cls, gt_box) in enumerate(gt_xyxy):
            if matched_gt[gi]:
                continue
            fn[gt_cls] += 1
            best_iou, best_pred_cls = 0.0, None
            for pred_cls, pred_box in pred_xyxy:
                if pred_cls == gt_cls:
                    continue
                iou = iou_xyxy(pred_box, gt_box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou, best_pred_cls = iou, pred_cls
            if best_pred_cls is not None:
                confusion[gt_cls][best_pred_cls] += 1

    precision = np.divide(tp, tp + fp, out=np.zeros(n_classes), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(n_classes), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=(precision + recall) > 0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt, "n_pred": n_pred,
        "precision": precision, "recall": recall, "f1": f1, "confusion": confusion,
    }


def compute_full_confusion_matrix(gt_by_key, pred_by_key, keys, class_names, iou_threshold):
    """
    Standard detection confusion matrix of size (n_classes + 1) x (n_classes + 1);
    the last row / column is "background".

      [i, i]   : correct detections of class i (TP).
      [i, j]   : GT of class i matched (IoU >= threshold) with a prediction of
                 class j != i (localized but mislabelled).
      [i, bg]  : GT of class i with no matching prediction (pure FN).
      [bg, j]  : prediction of class j with no matching GT (pure FP).

    Unlike compute_pointwise_metrics, matching here ignores the class so that
    mislabelled detections land in the off-diagonal cells.
    """
    n_classes = len(class_names)
    background = n_classes
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    for key in keys:
        gt_xyxy = [(gt[0], cxcywh_to_xyxy(gt[1:5])) for gt in gt_by_key[key]]
        pred_xyxy = [(p[0], cxcywh_to_xyxy(p[1:5])) for p in pred_by_key[key]]

        matched_gt = [False] * len(gt_xyxy)
        matched_pred = [False] * len(pred_xyxy)

        candidates = []
        for pi, (_, pred_box) in enumerate(pred_xyxy):
            for gi, (_, gt_box) in enumerate(gt_xyxy):
                iou = iou_xyxy(pred_box, gt_box)
                if iou >= iou_threshold:
                    candidates.append((iou, pi, gi))
        candidates.sort(key=lambda c: -c[0])
        for _, pi, gi in candidates:
            if matched_pred[pi] or matched_gt[gi]:
                continue
            matched_pred[pi] = True
            matched_gt[gi] = True
            matrix[gt_xyxy[gi][0]][pred_xyxy[pi][0]] += 1

        for gi, (gt_cls, _) in enumerate(gt_xyxy):
            if not matched_gt[gi]:
                matrix[gt_cls][background] += 1
        for pi, (pred_cls, _) in enumerate(pred_xyxy):
            if not matched_pred[pi]:
                matrix[background][pred_cls] += 1

    return matrix


# --------------------------------------------------------------------------
# AP / mAP / PR curves (uses the .csv predictions)
# --------------------------------------------------------------------------
def voc_ap(recall, precision) -> float:
    """Area under the PR curve with all-point interpolation (Pascal VOC 2010+ / COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_per_class(gt_by_key, pred_by_key, keys, class_names, iou_threshold):
    """
    Per-class AP at one IoU threshold.

    For each class, detections from all images are sorted by descending score
    and greedily matched to the unmatched GT box with the highest IoU (TP if
    IoU >= threshold, otherwise FP). A class without GT or without detections
    gets AP = 0.

    Returns:
        ap_per_class : (n_classes,) array of AP values.
        pr_curves    : {class_id: (recall_array, precision_array)}.
    """
    n_classes = len(class_names)
    ap_per_class = np.zeros(n_classes)
    pr_curves = {}

    for cid in range(n_classes):
        gt_by_image = {}
        n_gt_total = 0
        for key in keys:
            boxes = [cxcywh_to_xyxy(b[1:5]) for b in gt_by_key[key] if b[0] == cid]
            gt_by_image[key] = boxes
            n_gt_total += len(boxes)

        detections = []
        for key in keys:
            for det in pred_by_key[key]:
                if det[0] == cid:
                    detections.append((key, cxcywh_to_xyxy(det[1:5]), det[5]))

        if n_gt_total == 0 or not detections:
            pr_curves[cid] = (np.array([0.0]), np.array([0.0]))
            continue

        detections.sort(key=lambda d: -d[2])
        matched = {key: [False] * len(boxes) for key, boxes in gt_by_image.items()}

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        for i, (key, box, _) in enumerate(detections):
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_by_image.get(key, [])):
                if matched[key][j]:
                    continue
                iou = iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched[key][best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt_total
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap_per_class[cid] = voc_ap(recall, precision)
        pr_curves[cid] = (recall, precision)

    return ap_per_class, pr_curves


def interpolated_precision(recall, precision, grid):
    """
    Precision envelope of one PR curve sampled on a recall grid: the value at
    recall r is the highest precision reached at any recall >= r, and 0 when r
    is beyond the largest recall the class reaches.
    """
    envelope = np.maximum.accumulate(np.asarray(precision)[::-1])[::-1]
    idx = np.searchsorted(recall, grid, side="left")
    out = np.zeros(len(grid))
    valid = idx < len(recall)
    out[valid] = envelope[idx[valid]]
    return out


def compute_mean_pr_curve(pr_curves, n_classes, n_points=101):
    """
    Overall PR curve: at each recall value, the mean over all classes of the
    class's interpolated precision. A class without GT or detections counts as
    zero, exactly as it does in mAP. Returns (recall_grid, mean_precision).
    """
    grid = np.linspace(0.0, 1.0, n_points)
    per_class = np.array([
        interpolated_precision(*pr_curves[cid], grid) for cid in range(n_classes)
    ])
    return grid, per_class.mean(axis=0)


def compute_map_sweep(gt_by_key, pred_by_key_csv, keys, class_names,
                      iou_start=0.5, iou_end=0.95, iou_step=0.05):
    """
    COCO-style mAP@[0.5:0.95]: recompute per-class AP at 10 IoU levels
    (0.50, 0.55, ..., 0.95) and average. Roughly 10x slower than a single AP run.

    Returns a dict with:
        iou_levels, ap_matrix (n_levels x n_classes), ap_per_class_avg,
        map_overall, pr_curves_by_iou ({iou: {class_id: (recall, precision)}}).
    """
    iou_levels = np.round(np.arange(iou_start, iou_end + 1e-9, iou_step), 2)
    ap_matrix = []
    pr_curves_by_iou = {}
    for iou_thr in iou_levels:
        ap_per_class, pr_curves = compute_ap_per_class(
            gt_by_key, pred_by_key_csv, keys, class_names, float(iou_thr)
        )
        ap_matrix.append(ap_per_class)
        pr_curves_by_iou[float(iou_thr)] = pr_curves
    ap_matrix = np.array(ap_matrix)

    return {
        "iou_levels": iou_levels.tolist(),
        "ap_matrix": ap_matrix,
        "ap_per_class_avg": ap_matrix.mean(axis=0),
        "map_overall": float(ap_matrix.mean()),
        "pr_curves_by_iou": pr_curves_by_iou,
    }


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------
def write_report(run_name, class_names, metrics, ap_per_class, iou_threshold, out_dir: Path):
    """Write report.txt, per_class_metrics.csv and summary.json; return the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    n_gt, n_pred = metrics["n_gt"], metrics["n_pred"]
    precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]

    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    mean_ap = float(np.mean(ap_per_class))

    lines = []
    lines.append("=" * 78)
    lines.append(f"EVALUATION REPORT - run: {run_name}")
    lines.append("=" * 78)
    lines.append(f"IoU threshold     : {iou_threshold:g}")
    lines.append(f"Total GT boxes    : {int(n_gt.sum())}")
    lines.append(f"Total Pred boxes  : {int(n_pred.sum())}")
    lines.append(f"Total TP / FP / FN: {total_tp} / {total_fp} / {total_fn}")
    lines.append(f"Overall Precision : {overall_p:.4f}")
    lines.append(f"Overall Recall    : {overall_r:.4f}")
    lines.append(f"Overall F1        : {overall_f1:.4f}")
    lines.append(f"mAP               : {mean_ap:.4f}")
    lines.append("-" * 78)
    lines.append(f"{'ID':<3}{'Class':<15}{'GT':>6}{'Pred':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
                 f"{'Prec':>8}{'Rec':>8}{'F1':>8}{'AP':>8}")
    lines.append("-" * 78)
    for cid, name in enumerate(class_names):
        lines.append(
            f"{cid:<3}{name:<15}{n_gt[cid]:>6}{n_pred[cid]:>6}{tp[cid]:>6}{fp[cid]:>6}{fn[cid]:>6}"
            f"{precision[cid]:>8.3f}{recall[cid]:>8.3f}{f1[cid]:>8.3f}{ap_per_class[cid]:>8.3f}"
        )
    lines.append("=" * 78)
    lines.append("Data sources: AP and mAP come from pred-*.csv (all candidates above the capture threshold).")
    lines.append("              Pred, TP, FP, FN, Prec, Rec and F1 come from pred-*.txt (boxes above the display threshold).")

    report_text = "\n".join(lines)
    print(report_text)
    with open(out_dir / FILE_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_dir / FILE_PER_CLASS_CSV, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,gt,pred,tp,fp,fn,precision,recall,f1,ap\n")
        for cid, name in enumerate(class_names):
            f.write(
                f"{cid},{name},{n_gt[cid]},{n_pred[cid]},{tp[cid]},{fp[cid]},{fn[cid]},"
                f"{precision[cid]:.6f},{recall[cid]:.6f},{f1[cid]:.6f},{ap_per_class[cid]:.6f}\n"
            )

    summary = {
        "run_name": run_name,
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "overall_f1": overall_f1,
        "mAP": mean_ap,
        "per_class": {
            name: {
                "gt": int(n_gt[cid]), "pred": int(n_pred[cid]),
                "tp": int(tp[cid]), "fp": int(fp[cid]), "fn": int(fn[cid]),
                "precision": float(precision[cid]), "recall": float(recall[cid]),
                "f1": float(f1[cid]), "ap": float(ap_per_class[cid]),
            }
            for cid, name in enumerate(class_names)
        },
    }
    with open(out_dir / FILE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def _save_figure(fig, out_path: Path):
    """Apply tight layout, write the figure to disk and release it."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _set_class_ticks(ax, class_names, rotation=45):
    """Use class names as x tick labels (rotated so long names stay readable)."""
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=rotation, ha="right" if rotation < 90 else "center")


def _draw_grouped_bars(ax, series: dict, n_groups: int, group_width=0.8):
    """
    Draw grouped bars. `series` maps legend label -> list of `n_groups` values;
    group k is centered on x = k.
    """
    n_series = max(len(series), 1)
    bar_width = group_width / n_series
    x = np.arange(n_groups)
    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * bar_width
        ax.bar(x + offset, values, bar_width, label=label)


# --------------------------------------------------------------------------
# Charts: one run
# --------------------------------------------------------------------------
def plot_prf_ap_bar(class_names, metrics, ap_per_class, iou_threshold, out_path):
    """Grouped bars of Precision, Recall, F1 and AP for every class."""
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.6), 5))
    _draw_grouped_bars(ax, {
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "AP": ap_per_class,
    }, len(class_names))
    _set_class_ticks(ax, class_names)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 / AP by Class (IoU={iou_threshold:g})")
    ax.legend()
    _save_figure(fig, out_path)


def plot_ap_bar(class_names, ap_per_class, map_value, iou_threshold, out_path):
    """One AP bar per class with value labels and a dashed mAP reference line."""
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.7), 5.5))
    bars = ax.bar(x, ap_per_class, color="steelblue")
    for rect, value in zip(bars, ap_per_class):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.01, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(map_value, color="crimson", linestyle="--", linewidth=1.2,
               label=f"mAP = {map_value:.3f}")
    _set_class_ticks(ax, class_names)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel(f"AP@IoU={iou_threshold:g}")
    ax.set_title(f"AP@IoU={iou_threshold:g} by Class")
    ax.legend()
    _save_figure(fig, out_path)


def plot_gt_vs_pred_count(class_names, metrics, out_path):
    """Number of GT boxes versus predicted boxes for every class."""
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.6), 5))
    _draw_grouped_bars(ax, {
        "Ground truth": metrics["n_gt"],
        "Prediction": metrics["n_pred"],
    }, len(class_names))
    _set_class_ticks(ax, class_names)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of boxes")
    ax.set_title("GT vs Predicted Box Count by Class")
    ax.legend()
    _save_figure(fig, out_path)


def plot_pr_curves_grid(class_names, pr_curves, out_path):
    """One PR curve per class, each in its own subplot."""
    n = len(class_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.array(axes).reshape(-1)
    for cid, name in enumerate(class_names):
        ax = axes[cid]
        recall, precision = pr_curves[cid]
        ax.plot(recall, precision, linewidth=1.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("PR Curve by Class")
    _save_figure(fig, out_path)


def plot_pr_curves_overlay(class_names, pr_curves, out_path):
    """All per-class PR curves in one axes, one color per class."""
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab20")
    for cid, name in enumerate(class_names):
        recall, precision = pr_curves[cid]
        if len(recall) <= 1:
            continue  # Class without GT or detections: nothing to draw.
        ax.plot(recall, precision, linewidth=1.3, color=cmap(cid % 20), label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves, All Classes")
    ax.legend(fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(1.01, 0))
    _save_figure(fig, out_path)


def plot_overall_pr_curve(recall_grid, mean_precision, map_value, iou_threshold, out_path):
    """PR curve averaged over all classes; its area is the mAP."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_grid, mean_precision, linewidth=2, color="crimson",
            label=f"all classes  mAP@{iou_threshold:g} = {map_value:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve, All Classes (mean over classes)")
    ax.legend(loc="upper right")
    _save_figure(fig, out_path)


def plot_misclassified_gt_heatmap(class_names, confusion, out_path):
    """
    Heatmap of missed GT boxes that are covered by a prediction of another class.
    Row = true class of the GT, column = class the model predicted instead.
    """
    size = max(8, len(class_names) * 0.5)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(confusion, cmap="Reds")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Misclassified Ground Truth (IoU match, wrong class)")
    threshold = confusion.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = confusion[i, j]
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7,
                        color="white" if value > threshold else "black")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Number of boxes")
    _save_figure(fig, out_path)


def plot_confusion_matrix_with_background(class_names, matrix, out_path):
    """Full confusion matrix; the diagonal (TP) is outlined and bold."""
    labels = list(class_names) + ["background"]
    n = len(labels)
    size = max(9, n * 0.55)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Confusion Matrix (diagonal = TP)")
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if value > 0:
                is_diagonal = (i == j and i < len(class_names))
                ax.text(j, i, str(value), ha="center", va="center",
                        fontsize=8 if is_diagonal else 7,
                        fontweight="bold" if is_diagonal else "normal",
                        color="white" if value > vmax / 2 else "black")
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="limegreen", linewidth=1.5))
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Number of boxes")
    _save_figure(fig, out_path)


# --------------------------------------------------------------------------
# Charts: IoU sweep (mAP@[0.5:0.95])
# --------------------------------------------------------------------------
def plot_ap_vs_iou(class_names, sweep_result, out_path):
    """AP of every class as the IoU threshold increases."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")
    for cid, name in enumerate(class_names):
        ax.plot(sweep_result["iou_levels"], sweep_result["ap_matrix"][:, cid],
                marker="o", markersize=3, linewidth=1.2, color=cmap(cid % 20), label=name)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"AP vs IoU Threshold (mAP@[.5:.95] = {sweep_result['map_overall']:.3f})")
    ax.legend(fontsize=7, ncol=2, loc="upper right", bbox_to_anchor=(1.35, 1.0))
    _save_figure(fig, out_path)


def plot_map_vs_iou(class_names, sweep_result, out_path):
    """mAP (mean AP over classes) at each IoU threshold."""
    map_per_iou = sweep_result["ap_matrix"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(sweep_result["iou_levels"], map_per_iou,
            marker="o", markersize=5, linewidth=2, color="crimson")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel(f"mAP (mean AP over {len(class_names)} classes)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"mAP vs IoU Threshold (mAP@[.5:.95] = {sweep_result['map_overall']:.3f})")
    _save_figure(fig, out_path)


def plot_class_pr_by_iou(class_name, cid, pr_curves_by_iou, out_path):
    """Full PR curve of one class at every IoU level, in one axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    iou_levels = sorted(pr_curves_by_iou.keys())
    cmap = plt.get_cmap("viridis")
    for i, iou in enumerate(iou_levels):
        recall, precision = pr_curves_by_iou[iou][cid]
        if len(recall) <= 1:
            continue
        ax.plot(recall, precision, linewidth=1.4,
                color=cmap(i / max(len(iou_levels) - 1, 1)), label=f"IoU={iou:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve of '{class_name}' by IoU")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    _save_figure(fig, out_path)


# --------------------------------------------------------------------------
# Charts: comparison between runs
# --------------------------------------------------------------------------
def plot_run_comparison(class_names, run_summaries, metric_key, metric_label, out_path):
    """Grouped bars: one group per class, one bar per run, for one metric."""
    series = {
        run_name: [summary["per_class"][name][metric_key] for name in class_names]
        for run_name, summary in run_summaries.items()
    }
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.7), 5))
    _draw_grouped_bars(ax, series, len(class_names))
    _set_class_ticks(ax, class_names)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} by Class, Run Comparison")
    ax.legend()
    _save_figure(fig, out_path)


def plot_overall_comparison(run_summaries, out_path):
    """Grouped bars: overall Precision / Recall / F1 / mAP, one bar per run."""
    metric_keys = ["overall_precision", "overall_recall", "overall_f1", "mAP"]
    metric_labels = ["Precision", "Recall", "F1", "mAP"]
    series = {
        run_name: [summary[k] for k in metric_keys]
        for run_name, summary in run_summaries.items()
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_grouped_bars(ax, series, len(metric_keys))
    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics, Run Comparison")
    ax.legend()
    _save_figure(fig, out_path)


# --------------------------------------------------------------------------
# Per-run pipeline
# --------------------------------------------------------------------------
def evaluate_one_run(run_name, gt_dir, pred_dir, class_names, iou_threshold, output_dir,
                     map_sweep=False, iou_sweep_classes=None):
    """Evaluate one prediction directory, write report + charts, return the summary dict."""
    print(f"\n{'#' * 78}\n# RUN: {run_name}\n{'#' * 78}")
    keys, gt_index = match_keys(gt_dir, pred_dir)

    gt_by_key = {key: read_gt_file(gt_index[key]) for key in keys}

    # Two prediction sources serve two purposes and must not be mixed:
    #   .txt -> operating-point metrics (P / R / F1, TP / FP / FN, confusion matrices)
    #   .csv -> score-ranked metrics (AP, mAP, PR curves)
    # Computing P/R/F1 from the low-threshold .csv would count many low-score
    # candidates as false positives and misrepresent the operating point.
    pred_by_key_txt = {key: read_pred_txt_for_key(pred_dir, key) for key in keys}
    pred_by_key_csv = {key: read_pred_csv_for_key(pred_dir, key) for key in keys}

    validate_class_ids(gt_by_key, pred_by_key_txt, pred_by_key_csv, class_names)

    has_real_scores = any(det[5] != 1.0 for dets in pred_by_key_csv.values() for det in dets)
    if not has_real_scores:
        print("[warn] No .csv file with real scores was found: every prediction is treated as "
              "score = 1.0, so AP and PR curves are not meaningful.")

    metrics = compute_pointwise_metrics(gt_by_key, pred_by_key_txt, keys, class_names, iou_threshold)
    ap_per_class, pr_curves = compute_ap_per_class(
        gt_by_key, pred_by_key_csv, keys, class_names, iou_threshold
    )

    run_out_dir = output_dir / run_name
    summary = write_report(run_name, class_names, metrics, ap_per_class, iou_threshold, run_out_dir)

    plot_prf_ap_bar(class_names, metrics, ap_per_class, iou_threshold, run_out_dir / FILE_PRF_AP_BAR)
    plot_ap_bar(class_names, ap_per_class, summary["mAP"], iou_threshold, run_out_dir / FILE_AP_BAR)
    plot_gt_vs_pred_count(class_names, metrics, run_out_dir / FILE_COUNT_BAR)
    plot_pr_curves_grid(class_names, pr_curves, run_out_dir / FILE_PR_GRID)
    plot_pr_curves_overlay(class_names, pr_curves, run_out_dir / FILE_PR_OVERLAY)
    recall_grid, mean_precision = compute_mean_pr_curve(pr_curves, len(class_names))
    plot_overall_pr_curve(recall_grid, mean_precision, summary["mAP"], iou_threshold,
                          run_out_dir / FILE_PR_OVERALL)
    plot_misclassified_gt_heatmap(class_names, metrics["confusion"], run_out_dir / FILE_CONFUSION_MISCLASS)

    full_matrix = compute_full_confusion_matrix(gt_by_key, pred_by_key_txt, keys, class_names, iou_threshold)
    plot_confusion_matrix_with_background(class_names, full_matrix, run_out_dir / FILE_CONFUSION_FULL)

    if map_sweep:
        print("[map_sweep] Computing mAP@[0.5:0.95] (10 IoU levels, may take a few minutes on large data)...")
        sweep = compute_map_sweep(gt_by_key, pred_by_key_csv, keys, class_names)
        plot_ap_vs_iou(class_names, sweep, run_out_dir / FILE_AP_VS_IOU)
        plot_map_vs_iou(class_names, sweep, run_out_dir / FILE_MAP_VS_IOU)
        with open(run_out_dir / FILE_MAP_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "map_50_95_overall": sweep["map_overall"],
                "iou_levels": sweep["iou_levels"],
                "per_class": {name: float(sweep["ap_per_class_avg"][cid])
                              for cid, name in enumerate(class_names)},
            }, f, indent=2)
        print(f"[map_sweep] mAP@[0.5:0.95] = {sweep['map_overall']:.4f} "
              f"(mAP@{iou_threshold:g} = {summary['mAP']:.4f})")
        summary["mAP_50_95"] = sweep["map_overall"]

        if iou_sweep_classes:
            name_to_cid = {name: cid for cid, name in enumerate(class_names)}
            for name in iou_sweep_classes:
                name = name.strip().lower()
                if name not in name_to_cid:
                    print(f"[warn] --iou_sweep_classes: '{name}' is not in the classes file, skipped.")
                    continue
                safe_name = name.replace(" ", "_").replace("/", "_")
                out_path = run_out_dir / FILE_PR_BY_IOU_TEMPLATE.format(class_name=safe_name)
                plot_class_pr_by_iou(name, name_to_cid[name], sweep["pr_curves_by_iou"], out_path)
                print(f"[map_sweep] Saved PR curves by IoU for '{name}': {out_path}")

    print(f"[OK] Saved report and charts to: {run_out_dir}")
    return summary


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_runs(args, parser) -> dict:
    """Return {run_name: prediction_dir} from --runs (JSON string or file) or --pred_dir."""
    if args.runs:
        if Path(args.runs).is_file():
            with open(args.runs, "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            return json.loads(args.runs)
        except json.JSONDecodeError as exc:
            parser.error(f"--runs is neither an existing JSON file nor valid JSON text: {exc}")
    return {"run1": args.pred_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GroundingDINO PPE detections against YOLO ground truth and draw charts."
    )
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory with ground-truth .txt files (YOLO format).")
    parser.add_argument("--classes", type=str, required=True,
                        help="Path to classes.txt (one class name per line; line index = class id).")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Prediction directory of a single run (the run is named 'run1').")
    parser.add_argument(
        "--runs", type=str, default=None,
        help='JSON text or JSON file mapping run name -> prediction directory, e.g. '
             '\'{"combined": "out1", "per_class": "out2"}\'. Use it to compare several runs.',
    )
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold used for matching and AP (default: 0.5).")
    parser.add_argument(
        "--map_sweep", action="store_true",
        help="Also compute COCO-style mAP@[0.5:0.95] (about 10x slower; reporting only).",
    )
    parser.add_argument(
        "--iou_sweep_classes", type=str, default=None,
        help="Comma-separated class names; requires --map_sweep. Draws one extra chart per class "
             "with its full PR curve at every IoU level. Example: --iou_sweep_classes 'head,shoes'",
    )
    parser.add_argument("--output_dir", type=str, default="eval_report",
                        help="Directory where reports and charts are written (default: eval_report).")
    args = parser.parse_args()

    if not args.pred_dir and not args.runs:
        parser.error("Provide --pred_dir (single run) or --runs (multiple runs).")

    class_names = load_class_names(args.classes)
    print(f"[classes] {len(class_names)} classes: {class_names}")

    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args, parser)
    iou_sweep_classes = (
        [c.strip() for c in args.iou_sweep_classes.split(",") if c.strip()]
        if args.iou_sweep_classes else None
    )

    run_summaries = {}
    for run_name, pred_dir_str in runs.items():
        run_summaries[run_name] = evaluate_one_run(
            run_name, gt_dir, Path(pred_dir_str), class_names, args.iou_threshold, output_dir,
            map_sweep=args.map_sweep, iou_sweep_classes=iou_sweep_classes,
        )

    if len(run_summaries) > 1:
        comp_dir = output_dir / COMPARE_DIR
        comp_dir.mkdir(parents=True, exist_ok=True)
        plot_run_comparison(class_names, run_summaries, "f1", "F1", comp_dir / FILE_COMPARE_F1)
        plot_run_comparison(class_names, run_summaries, "recall", "Recall", comp_dir / FILE_COMPARE_RECALL)
        plot_run_comparison(class_names, run_summaries, "ap", "AP", comp_dir / FILE_COMPARE_AP)
        plot_overall_comparison(run_summaries, comp_dir / FILE_COMPARE_OVERALL)
        with open(comp_dir / FILE_COMPARE_JSON, "w", encoding="utf-8") as f:
            json.dump(run_summaries, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved comparison charts for {len(run_summaries)} runs to: {comp_dir}")

    print(f"\n{'=' * 78}\nDONE. Results are in: {output_dir.resolve()}\n{'=' * 78}")


if __name__ == "__main__":
    main()