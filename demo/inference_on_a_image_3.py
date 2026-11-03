import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


# ============================================================
# 16 dataset classes
# dataset_name = what you want in your label system
# prompt_text  = what GroundingDINO should actually search for
# ============================================================
CLASS_DEFS = [
    {"id": 0,  "dataset_name": "person",       "prompt_text": "person"},
    {"id": 1,  "dataset_name": "ear",          "prompt_text": "human ear"},
    {"id": 2,  "dataset_name": "ear-mufs",     "prompt_text": "ear muffs"},
    {"id": 3,  "dataset_name": "face",         "prompt_text": "human face"},
    {"id": 4,  "dataset_name": "face-guard",   "prompt_text": "face shield"},
    {"id": 5,  "dataset_name": "face-mask",    "prompt_text": "protective face mask"},
    {"id": 6,  "dataset_name": "foot",         "prompt_text": "human foot"},
    {"id": 7,  "dataset_name": "tool",         "prompt_text": "hand tool"},
    {"id": 8,  "dataset_name": "glasses",      "prompt_text": "protective safety glasses"},
    {"id": 9,  "dataset_name": "gloves",       "prompt_text": "protective gloves"},
    {"id": 10, "dataset_name": "helmet",       "prompt_text": "safety helmet"},
    {"id": 11, "dataset_name": "hands",        "prompt_text": "human hand"},
    {"id": 12, "dataset_name": "hea",          "prompt_text": "human head"},
    {"id": 13, "dataset_name": "medical-suit", "prompt_text": "protective medical suit"},
    {"id": 14, "dataset_name": "shoes",        "prompt_text": "protective work shoes"},
    {"id": 15, "dataset_name": "safety-suit",  "prompt_text": "industrial safety suit"},
]

CLASS_THRESHOLDS = {
    0: 0.5,   # person
    1: 0.5,   # ear 
    2: 0.5,   # ear-mufs
    3: 0.5,   # face
    4: 0.5,   # face-guard
    5: 0.5,   # face-mask
    6: 0.5,   # foot
    7: 0.5,   # tool 
    8: 0.5,   # glasses
    9: 0.5,   # gloves 
    10: 0.5,  # helmet
    11: 0.5,  # hands
    12: 0.5,  # head
    13: 0.5,  # medical-suit
    14: 0.5,  # shoes
    15: 0.5,  # safety-suit
}


def get_device(cpu_only=False):
    if cpu_only or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = str(device)

    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)

    model.eval()
    model.to(device)
    return model


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]

    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes, labels):
        # normalized cxcywh -> pixel xyxy
        box = box * torch.tensor([W, H, W, H], dtype=box.dtype)
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        x0, y0, x1, y1 = box.tolist()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((x0, y0), label, font=font)
        else:
            tw, th = draw.textsize(label, font=font)
            text_bbox = (x0, y0, x0 + tw, y0 + th)

        draw.rectangle(text_bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)

    return image_pil


def get_grounding_output(model, image, caption, box_threshold, device):
    """
    Run GroundingDINO with ONE prompt only.
    Returns:
      boxes_filt: Tensor [N, 4] in normalized cx, cy, w, h
      scores_filt: Tensor [N]
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]   # (num_queries, 256)
    boxes = outputs["pred_boxes"][0]               # (num_queries, 4), normalized cxcywh

    max_scores = logits.max(dim=1)[0]
    filt_mask = max_scores > box_threshold

    boxes_filt = boxes[filt_mask].detach().cpu()
    scores_filt = max_scores[filt_mask].detach().cpu()

    return boxes_filt, scores_filt


def save_yolo_labels(output_path, class_ids, boxes):
    """
    Save YOLO-format txt:
      class_id x_center y_center width height
    """
    lines = []

    for class_id, box in zip(class_ids, boxes):
        x_center, y_center, width, height = box.tolist()
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved YOLO labels to: {output_path}")


def run_separate_class_prompts(model, image, device):
    all_boxes = []
    all_class_ids = []
    all_labels_for_draw = []
    all_scores = []

    for cls in CLASS_DEFS:
        class_id = cls["id"]
        dataset_name = cls["dataset_name"]
        prompt_text = cls["prompt_text"]
        class_threshold = CLASS_THRESHOLDS[class_id]

        boxes_filt, scores_filt = get_grounding_output(
            model=model,
            image=image,
            caption=prompt_text,
            box_threshold=class_threshold,
            device=device,
        )

        print(f"[INFO] class_id={class_id:2d}, prompt='{prompt_text}', detections={len(boxes_filt)}")

        for box, score in zip(boxes_filt, scores_filt):
            all_boxes.append(box)
            all_class_ids.append(class_id)
            all_labels_for_draw.append(f"{dataset_name}({score:.3f})")
            all_scores.append(float(score))

    return all_boxes, all_class_ids, all_labels_for_draw, all_scores


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def apply_nms_per_class(class_ids, boxes, labels, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return [], [], []

    kept_class_ids = []
    kept_boxes = []
    kept_labels = []

    unique_classes = sorted(set(class_ids))

    for cls in unique_classes:
        idxs = [i for i, c in enumerate(class_ids) if c == cls]
        cls_boxes = torch.stack([boxes[i] for i in idxs], dim=0)
        cls_scores = torch.tensor([scores[i] for i in idxs], dtype=torch.float32)

        cls_boxes_xyxy = box_cxcywh_to_xyxy(cls_boxes)
        keep = torch.ops.torchvision.nms(cls_boxes_xyxy, cls_scores, iou_threshold)

        for k in keep.tolist():
            original_idx = idxs[k]
            kept_class_ids.append(class_ids[original_idx])
            kept_boxes.append(boxes[original_idx])
            kept_labels.append(labels[original_idx])

    return kept_class_ids, kept_boxes, kept_labels


def make_output_stem(image_path):
    filename = os.path.basename(image_path)      # pexels-photo-209230.jpeg
    stem = os.path.splitext(filename)[0]         # pexels-photo-209230
    image_id = stem.split("-")[-1]               # 209230
    return f"pred-{image_id}"                    # pred-209230


def main():
    parser = argparse.ArgumentParser("Grounding DINO - 16 separate prompts", add_help=True)

    parser.add_argument(
        "--config_file", "-c",
        type=str,
        required=True,
        help="path to config file"
    )

    parser.add_argument(
        "--checkpoint", "--checkpoint_path", "-p",
        dest="checkpoint_path",
        type=str,
        required=True,
        help="path to checkpoint file"
    )

    parser.add_argument(
        "--image_path", "-i",
        type=str,
        required=True,
        help="path to image file"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="output directory"
    )

    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="box threshold (kept for compatibility; class-specific thresholds are used)"
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="run on CPU only"
    )

    # keep this only so old terminal commands do not break
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="ignored; this script now uses 16 separate built-in prompts"
    )

    args = parser.parse_args()

    if args.text_prompt is not None:
        print("[INFO] --text_prompt is ignored because this script uses 16 separate prompts automatically.")

    os.makedirs(args.output_dir, exist_ok=True)

    output_stem = make_output_stem(args.image_path)

    device = get_device(cpu_only=args.cpu_only)
    print(f"[INFO] Using device: {device}")

    image_pil, image = load_image(args.image_path)
    model = load_model(args.config_file, args.checkpoint_path, device)

    # save raw image
    raw_image_path = os.path.join(args.output_dir, "raw_image.jpg")
    image_pil.save(raw_image_path)
    print(f"Saved raw image to: {raw_image_path}")

    # run 16 separate prompts
    all_boxes, all_class_ids, all_labels_for_draw, all_scores = run_separate_class_prompts(
        model=model,
        image=image,
        device=device,
    )

    # per-class NMS
    all_class_ids, all_boxes, all_labels_for_draw = apply_nms_per_class(
        all_class_ids,
        all_boxes,
        all_labels_for_draw,
        all_scores,
        iou_threshold=0.5,
    )

    # save pred-xxxxx.txt
    pred_txt_path = os.path.join(args.output_dir, f"{output_stem}.txt")
    save_yolo_labels(pred_txt_path, all_class_ids, all_boxes)

    # save pred-xxxxx.jpg
    if len(all_boxes) > 0:
        boxes_tensor = torch.stack(all_boxes, dim=0)
    else:
        boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

    size = image_pil.size  # (W, H)
    pred_dict = {
        "boxes": boxes_tensor,
        "size": [size[1], size[0]],  # H, W
        "labels": all_labels_for_draw,
    }

    image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)
    pred_img_path = os.path.join(args.output_dir, f"{output_stem}.jpg")
    image_with_box.save(pred_img_path)
    print(f"Saved visualization to: {pred_img_path}")


if __name__ == "__main__":
    main()