"""
Draw ground-truth boxes on every image and save the results to a new folder.

Inputs
------
--image_dir   folder of images, e.g. pexels-photo-209230.jpg
--label_dir   folder of YOLO label files, e.g. pexels-photo-209230.txt
              (one box per line: class_id x_center y_center width height,
              all coordinates normalised to [0, 1])
--classes     classes.txt, one class name per line in id order
--output_dir  folder to create; each image is saved with its GT boxes drawn

An image and its label file are matched by file name without extension.
Images without a label file are still saved (no boxes) and are reported.

Usage:
    python draw_gt_boxes.py --image_dir "images" --label_dir "labels" --classes "classes.txt" --output_dir "GT"
"""

import argparse
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(classes_path: str) -> list[str]:
    """Read classes.txt (one name per line) in id order."""
    with open(classes_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_path}")
    return names


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into (class_id, x_center, y_center, width, height) tuples."""
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                print(f"[warn] {label_path.name} line {line_no}: expected 5 values, got {len(parts)}; skipped.")
                continue
            boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
    return boxes


def get_font(image_width: int):
    """Return a bold font whose size scales with the image width."""
    size = max(14, image_width // 80)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes, class_names: list[str], colors: dict[int, tuple]) -> Image.Image:
    """Draw each GT box with its class name on a copy of the image."""
    image = image.copy()
    W, H = image.size
    draw = ImageDraw.Draw(image)
    font = get_font(W)

    for cid, xc, yc, w, h in boxes:
        x0, y0 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x1, y1 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        color = colors.get(cid, (255, 0, 0))
        label = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, W // 400))
        try:
            bbox = draw.textbbox((x0, y0), label, font=font)
        except Exception:
            bbox = (x0, y0, x0 + 8 * len(label), y0 + 16)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser("Draw ground-truth boxes on images")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    class_names = load_class_names(args.classes)
    image_dir, label_dir, output_dir = Path(args.image_dir), Path(args.label_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One saturated colour per class id (evenly spaced hues), identical across all images.
    n = len(class_names)
    colors = {
        cid: tuple(int(c * 255) for c in colorsys.hsv_to_rgb((cid * 0.618) % 1.0, 0.9, 0.85))
        for cid in range(n)
    }

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[images] {len(image_paths)} images in {image_dir}")

    n_ok, n_no_label, n_boxes_total = 0, 0, 0
    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            boxes = read_yolo_labels(label_path)
        else:
            boxes = []
            n_no_label += 1
            print(f"[warn] no label file for {img_path.name}; saved without boxes.")

        image = Image.open(img_path).convert("RGB")
        drawn = draw_boxes(image, boxes, class_names, colors)
        drawn.save(output_dir / f"{img_path.stem}.jpg", quality=95)
        n_ok += 1
        n_boxes_total += len(boxes)

    print("=" * 60)
    print(f"Done. Images saved: {n_ok}, without label file: {n_no_label}, total GT boxes drawn: {n_boxes_total}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()