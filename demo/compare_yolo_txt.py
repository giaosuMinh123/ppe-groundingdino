import argparse
import csv
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
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
}


def class_name(class_id):
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


def load_yolo_txt(txt_path):
    items = []
    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    with txt_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{txt_path} line {line_num}: expected 5 values, got {len(parts)} -> {line}"
                )

            items.append({
                "class_id": int(parts[0]),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
                "raw_line": line,
                "line_num": line_num,
            })

    return items


def cxcywh_to_xyxy(box):
    x = box["x_center"]
    y = box["y_center"]
    w = box["width"]
    h = box["height"]
    return (
        x - w / 2.0,
        y - h / 2.0,
        x + w / 2.0,
        y + h / 2.0,
    )


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = cxcywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = cxcywh_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def evaluate(preds, gts, iou_threshold=0.5):
    matched_gt = set()
    results = []

    tp = 0
    fp = 0

    for pred_idx, pred in enumerate(preds):
        pred_class = pred["class_id"]

        best_gt_idx = None
        best_iou = 0.0

        # same-class matching first
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            if gt["class_id"] != pred_class:
                continue

            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            tp += 1
            results.append({
                "pred_idx": pred_idx,
                "pred": pred,
                "status": "TP",
                "matched_gt_idx": best_gt_idx,
                "matched_gt": gts[best_gt_idx],
                "iou": best_iou,
            })
        else:
            # find best overlap across all GT for diagnostics
            best_any_gt_idx = None
            best_any_iou = 0.0
            for gt_idx, gt in enumerate(gts):
                score = iou(pred, gt)
                if score > best_any_iou:
                    best_any_iou = score
                    best_any_gt_idx = gt_idx

            fp += 1
            results.append({
                "pred_idx": pred_idx,
                "pred": pred,
                "status": "FP",
                "matched_gt_idx": None,
                "matched_gt": None,
                "iou": best_iou,
                "best_any_gt_idx": best_any_gt_idx,
                "best_any_gt": gts[best_any_gt_idx] if best_any_gt_idx is not None else None,
                "best_any_iou": best_any_iou,
            })

    fn_indices = [gt_idx for gt_idx in range(len(gts)) if gt_idx not in matched_gt]
    fn = len(fn_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "results": results,
        "fn_indices": fn_indices,
    }


def build_report_text(preds, gts, eval_result, iou_threshold, gt_path, pred_path):
    lines = []
    lines.append("=" * 80)
    lines.append("YOLO TXT COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"Ground truth  : {gt_path}")
    lines.append(f"Prediction    : {pred_path}")
    lines.append(f"IoU threshold : {iou_threshold}")
    lines.append(f"Predictions   : {len(preds)}")
    lines.append(f"Ground truth  : {len(gts)}")
    lines.append("-" * 80)
    lines.append(f"TP            : {eval_result['tp']}")
    lines.append(f"FP            : {eval_result['fp']}")
    lines.append(f"FN            : {eval_result['fn']}")
    lines.append(f"Precision     : {eval_result['precision']:.4f}")
    lines.append(f"Recall        : {eval_result['recall']:.4f}")
    lines.append(f"F1-score      : {eval_result['f1']:.4f}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("PREDICTION-BY-PREDICTION RESULTS")
    lines.append("-" * 80)

    for item in eval_result["results"]:
        pred = item["pred"]
        pred_cls = pred["class_id"]
        pred_name = class_name(pred_cls)

        lines.append(
            f"Pred #{item['pred_idx']:02d} | "
            f"class={pred_cls:2d} ({pred_name}) | "
            f"box=({pred['x_center']:.6f}, {pred['y_center']:.6f}, {pred['width']:.6f}, {pred['height']:.6f}) | "
            f"status={item['status']} | IoU={item['iou']:.4f}"
        )

        if item["status"] == "TP":
            gt = item["matched_gt"]
            lines.append(
                f"   matched GT -> class={gt['class_id']:2d} ({class_name(gt['class_id'])}) | "
                f"line={gt['line_num']} | "
                f"box=({gt['x_center']:.6f}, {gt['y_center']:.6f}, {gt['width']:.6f}, {gt['height']:.6f})"
            )
        else:
            best_any_gt = item.get("best_any_gt")
            if best_any_gt is not None:
                lines.append(
                    f"   best overlap -> class={best_any_gt['class_id']:2d} ({class_name(best_any_gt['class_id'])}) | "
                    f"line={best_any_gt['line_num']} | "
                    f"IoU={item.get('best_any_iou', 0.0):.4f}"
                )

    lines.append("")
    lines.append("UNMATCHED GROUND-TRUTH BOXES (FN)")
    lines.append("-" * 80)

    if not eval_result["fn_indices"]:
        lines.append("None")
    else:
        for gt_idx in eval_result["fn_indices"]:
            gt = gts[gt_idx]
            lines.append(
                f"GT #{gt_idx:02d} | "
                f"class={gt['class_id']:2d} ({class_name(gt['class_id'])}) | "
                f"line={gt['line_num']} | "
                f"box=({gt['x_center']:.6f}, {gt['y_center']:.6f}, {gt['width']:.6f}, {gt['height']:.6f})"
            )

    return "\n".join(lines)


def save_report_txt(output_txt_path, report_text):
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text(report_text, encoding="utf-8")
    print(f"Saved TXT report to: {output_txt_path}")


def save_report_csv(output_csv_path, eval_result):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_type",
        "pred_idx",
        "pred_line_num",
        "pred_class_id",
        "pred_class_name",
        "pred_x_center",
        "pred_y_center",
        "pred_width",
        "pred_height",
        "status",
        "matched_gt_idx",
        "matched_gt_line_num",
        "matched_gt_class_id",
        "matched_gt_class_name",
        "matched_gt_x_center",
        "matched_gt_y_center",
        "matched_gt_width",
        "matched_gt_height",
        "iou_same_class",
        "best_any_gt_idx",
        "best_any_gt_line_num",
        "best_any_gt_class_id",
        "best_any_gt_class_name",
        "best_any_iou",
    ]

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in eval_result["results"]:
            pred = item["pred"]
            matched_gt = item.get("matched_gt")
            best_any_gt = item.get("best_any_gt")

            row = {
                "row_type": "prediction",
                "pred_idx": item["pred_idx"],
                "pred_line_num": pred["line_num"],
                "pred_class_id": pred["class_id"],
                "pred_class_name": class_name(pred["class_id"]),
                "pred_x_center": f"{pred['x_center']:.6f}",
                "pred_y_center": f"{pred['y_center']:.6f}",
                "pred_width": f"{pred['width']:.6f}",
                "pred_height": f"{pred['height']:.6f}",
                "status": item["status"],
                "matched_gt_idx": "",
                "matched_gt_line_num": "",
                "matched_gt_class_id": "",
                "matched_gt_class_name": "",
                "matched_gt_x_center": "",
                "matched_gt_y_center": "",
                "matched_gt_width": "",
                "matched_gt_height": "",
                "iou_same_class": f"{item.get('iou', 0.0):.6f}",
                "best_any_gt_idx": "",
                "best_any_gt_line_num": "",
                "best_any_gt_class_id": "",
                "best_any_gt_class_name": "",
                "best_any_iou": f"{item.get('best_any_iou', 0.0):.6f}" if item["status"] == "FP" else "",
            }

            if matched_gt is not None:
                row.update({
                    "matched_gt_idx": item["matched_gt_idx"],
                    "matched_gt_line_num": matched_gt["line_num"],
                    "matched_gt_class_id": matched_gt["class_id"],
                    "matched_gt_class_name": class_name(matched_gt["class_id"]),
                    "matched_gt_x_center": f"{matched_gt['x_center']:.6f}",
                    "matched_gt_y_center": f"{matched_gt['y_center']:.6f}",
                    "matched_gt_width": f"{matched_gt['width']:.6f}",
                    "matched_gt_height": f"{matched_gt['height']:.6f}",
                })

            if best_any_gt is not None:
                row.update({
                    "best_any_gt_idx": item.get("best_any_gt_idx", ""),
                    "best_any_gt_line_num": best_any_gt["line_num"],
                    "best_any_gt_class_id": best_any_gt["class_id"],
                    "best_any_gt_class_name": class_name(best_any_gt["class_id"]),
                })

            writer.writerow(row)

    print(f"Saved CSV report to: {output_csv_path}")


def make_default_output_paths(gt_path, pred_path, output_dir=None):
    gt_stem = Path(gt_path).stem
    pred_stem = Path(pred_path).stem

    base_name = f"comparison_{pred_stem}_vs_{gt_stem}"

    if output_dir is None:
        out_dir = Path(pred_path).parent
    else:
        out_dir = Path(output_dir)

    return (
        out_dir / f"{base_name}.txt",
        out_dir / f"{base_name}.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO-format GT and prediction txt files")
    parser.add_argument("--gt", required=True, help="Path to ground-truth txt")
    parser.add_argument("--pred", required=True, help="Path to prediction txt")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold, default=0.5")
    parser.add_argument("--output_dir", default=None, help="Optional folder to save .txt and .csv reports")
    args = parser.parse_args()

    gts = load_yolo_txt(args.gt)
    preds = load_yolo_txt(args.pred)

    eval_result = evaluate(preds, gts, iou_threshold=args.iou)

    report_text = build_report_text(
        preds=preds,
        gts=gts,
        eval_result=eval_result,
        iou_threshold=args.iou,
        gt_path=args.gt,
        pred_path=args.pred,
    )

    print(report_text)

    output_txt_path, output_csv_path = make_default_output_paths(
        gt_path=args.gt,
        pred_path=args.pred,
        output_dir=args.output_dir,
    )

    save_report_txt(output_txt_path, report_text)
    save_report_csv(output_csv_path, eval_result)


if __name__ == "__main__":
    main()