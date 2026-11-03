import argparse
import csv
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "hea",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
}


def safe_int(value, default=None):
    if value is None:
        return default
    value = str(value).strip()
    if value == "":
        return default
    return int(value)


def safe_float(value, default=0.0):
    if value is None:
        return default
    value = str(value).strip()
    if value == "":
        return default
    return float(value)


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def read_comparison_csv(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_compare_folder(compare_dir):
    compare_dir = Path(compare_dir)
    csv_files = sorted(compare_dir.glob("comparison_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No comparison_*.csv files found in: {compare_dir}")

    dataset_stats = {
        "num_images": 0,
        "num_prediction_rows": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "gt_count_by_class": defaultdict(int),
    }

    per_class = {
        class_id: {
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_predictions": 0,
            "num_gt": 0,
        }
        for class_id in CLASS_NAMES
    }

    for csv_file in csv_files:
        dataset_stats["num_images"] += 1
        rows = read_comparison_csv(csv_file)

        seen_gt_keys_in_file = set()

        for row in rows:
            if row.get("row_type") != "prediction":
                continue

            dataset_stats["num_prediction_rows"] += 1

            pred_class_id = safe_int(row.get("pred_class_id"))
            status = (row.get("status") or "").strip().upper()

            if pred_class_id is not None and pred_class_id in per_class:
                per_class[pred_class_id]["num_predictions"] += 1

            if status == "TP":
                dataset_stats["total_tp"] += 1
                if pred_class_id is not None and pred_class_id in per_class:
                    per_class[pred_class_id]["tp"] += 1

                gt_class_id = safe_int(row.get("matched_gt_class_id"))
                gt_line_num = safe_int(row.get("matched_gt_line_num"))
                if gt_class_id is not None and gt_line_num is not None:
                    gt_key = (csv_file.name, gt_class_id, gt_line_num)
                    if gt_key not in seen_gt_keys_in_file:
                        seen_gt_keys_in_file.add(gt_key)
                        dataset_stats["gt_count_by_class"][gt_class_id] += 1
                        if gt_class_id in per_class:
                            per_class[gt_class_id]["num_gt"] += 1

            elif status == "FP":
                dataset_stats["total_fp"] += 1
                if pred_class_id is not None and pred_class_id in per_class:
                    per_class[pred_class_id]["fp"] += 1

        # To count FN, read unmatched GT directly from the paired TXT report if available,
        # but easier is infer from GT totals present in CSV:
        # TP rows already contribute GT counts for matched GT only.
        # So we also need unmatched GT rows.
        # We'll extract them from the sibling .txt report if available.
        txt_file = csv_file.with_suffix(".txt")
        if txt_file.exists():
            fn_lines = extract_fn_from_txt(txt_file)
            dataset_stats["total_fn"] += len(fn_lines)
            for gt_class_id, gt_line_num in fn_lines:
                dataset_stats["gt_count_by_class"][gt_class_id] += 1
                if gt_class_id in per_class:
                    per_class[gt_class_id]["fn"] += 1
                    per_class[gt_class_id]["num_gt"] += 1

        # For matched GT, FN not incremented here; only GT totals already counted.
        # After file loop, derive FN for classes from num_gt - tp when needed.

    # Rebuild class FN safely in case some files had no txt sibling
    for class_id, stats in per_class.items():
        inferred_fn = max(stats["num_gt"] - stats["tp"], 0)
        if stats["fn"] < inferred_fn:
            stats["fn"] = inferred_fn

    # Rebuild total FN from class sums for consistency
    dataset_stats["total_fn"] = sum(stats["fn"] for stats in per_class.values())
    total_gt = sum(stats["num_gt"] for stats in per_class.values())

    return csv_files, dataset_stats, per_class, total_gt


def extract_fn_from_txt(txt_path):
    """
    Read unmatched GT boxes from the comparison .txt report.

    Expected lines like:
    GT #00 | class=14 (shoes) | line=4 | box=(...)
    """
    fn_items = []
    lines = Path(txt_path).read_text(encoding="utf-8").splitlines()

    in_fn_section = False
    for line in lines:
        stripped = line.strip()

        if stripped == "UNMATCHED GROUND-TRUTH BOXES (FN)":
            in_fn_section = True
            continue

        if not in_fn_section:
            continue

        if stripped.startswith("GT #"):
            # Example: GT #00 | class=14 (shoes) | line=4 | box=(...)
            parts = [p.strip() for p in stripped.split("|")]
            gt_class_id = None
            gt_line_num = None

            for part in parts:
                if part.startswith("class="):
                    # class=14 (shoes)
                    right = part.split("=", 1)[1].strip()
                    gt_class_id = int(right.split()[0])
                elif part.startswith("line="):
                    gt_line_num = int(part.split("=", 1)[1].strip())

            if gt_class_id is not None and gt_line_num is not None:
                fn_items.append((gt_class_id, gt_line_num))

    return fn_items


def build_report_text(compare_dir, csv_files, dataset_stats, per_class, total_gt):
    total_tp = dataset_stats["total_tp"]
    total_fp = dataset_stats["total_fp"]
    total_fn = dataset_stats["total_fn"]

    p = precision(total_tp, total_fp)
    r = recall(total_tp, total_fn)
    f1 = f1_score(p, r)

    lines = []
    lines.append("=" * 100)
    lines.append("GROUNDINGDINO OVERALL DATASET EVALUATION")
    lines.append("=" * 100)
    lines.append(f"Compare folder        : {compare_dir}")
    lines.append(f"Comparison CSV files  : {len(csv_files)}")
    lines.append(f"Total images          : {dataset_stats['num_images']}")
    lines.append(f"Total predictions     : {dataset_stats['num_prediction_rows']}")
    lines.append(f"Total ground-truth    : {total_gt}")
    lines.append("-" * 100)
    lines.append(f"Total TP              : {total_tp}")
    lines.append(f"Total FP              : {total_fp}")
    lines.append(f"Total FN              : {total_fn}")
    lines.append(f"Overall Precision     : {p:.6f}")
    lines.append(f"Overall Recall        : {r:.6f}")
    lines.append(f"Overall F1-score      : {f1:.6f}")
    lines.append("=" * 100)
    lines.append("")
    lines.append("PER-CLASS METRICS")
    lines.append("-" * 100)
    lines.append(
        f"{'ID':<4} {'Class':<18} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'Precision':>12} {'Recall':>12} {'F1':>12}"
    )
    lines.append("-" * 100)

    for class_id in sorted(per_class.keys()):
        stats = per_class[class_id]
        cp = precision(stats["tp"], stats["fp"])
        cr = recall(stats["tp"], stats["fn"])
        cf1 = f1_score(cp, cr)

        lines.append(
            f"{class_id:<4} "
            f"{stats['class_name']:<18} "
            f"{stats['num_gt']:>6} "
            f"{stats['num_predictions']:>6} "
            f"{stats['tp']:>6} "
            f"{stats['fp']:>6} "
            f"{stats['fn']:>6} "
            f"{cp:>12.6f} "
            f"{cr:>12.6f} "
            f"{cf1:>12.6f}"
        )

    return "\n".join(lines)


def save_overall_csv(output_csv_path, per_class):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_id",
        "class_name",
        "num_gt",
        "num_predictions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_id in sorted(per_class.keys()):
            stats = per_class[class_id]
            cp = precision(stats["tp"], stats["fp"])
            cr = recall(stats["tp"], stats["fn"])
            cf1 = f1_score(cp, cr)

            writer.writerow({
                "class_id": class_id,
                "class_name": stats["class_name"],
                "num_gt": stats["num_gt"],
                "num_predictions": stats["num_predictions"],
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "precision": f"{cp:.6f}",
                "recall": f"{cr:.6f}",
                "f1": f"{cf1:.6f}",
            })

    print(f"Saved overall CSV to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Read all comparison_*.csv files and compute overall dataset evaluation"
    )
    parser.add_argument(
        "--compare_dir",
        required=True,
        help="Folder containing comparison_pred-*_vs_pexels-photo-*.csv files"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Folder to save overall_evaluation.txt and overall_evaluation.csv"
    )
    args = parser.parse_args()

    compare_dir = Path(args.compare_dir)
    output_dir = Path(args.output_dir) if args.output_dir else compare_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files, dataset_stats, per_class, total_gt = aggregate_compare_folder(compare_dir)

    report_text = build_report_text(
        compare_dir=compare_dir,
        csv_files=csv_files,
        dataset_stats=dataset_stats,
        per_class=per_class,
        total_gt=total_gt,
    )

    print(report_text)

    output_txt_path = output_dir / "overall_evaluation.txt"
    output_csv_path = output_dir / "overall_evaluation.csv"

    output_txt_path.write_text(report_text, encoding="utf-8")
    print(f"Saved overall TXT to: {output_txt_path}")

    save_overall_csv(output_csv_path, per_class)


if __name__ == "__main__":
    main()