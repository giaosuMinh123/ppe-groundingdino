import argparse
import os
from pathlib import Path   # ← THÊM DÒNG NÀY

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes, labels):
        # cxcywh -> xyxy
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = map(int, box)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        # Draw label
        font = ImageFont.load_default()
        try:
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font=font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, x0 + w, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white", font=font)
        except:
            pass

    return image_pil   # ← Chỉ trả về ảnh, bỏ mask vì không dùng


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model


def get_grounding_output(model,
    image,
    caption,
    box_threshold,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
    nms_threshold=0.5):   # Giữ nguyên hàm này của bạn (không thay đổi)
    assert text_threshold is not None or token_spans is not None, \
        "text_threshold and token_spans should not be None at the same time!"

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]              # (nq, 4), format cx,cy,w,h (normalized)

    # ------------------------------------------------------------------
    # NORMAL MODE (không dùng token_spans)
    # ------------------------------------------------------------------
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()

        # Lọc theo box threshold
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # Tạo phrase labels
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        scores = []

        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                tokenizer
            )
            score = logit.max().item()
            scores.append(score)

            if with_logits:
                pred_phrases.append(f"{pred_phrase}({score:.3f})")
            else:
                pred_phrases.append(pred_phrase)

        # ------------------ APPLY NMS ------------------
        if len(boxes_filt) > 0:
            # Chuyển từ cxcywh -> xyxy
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt)

            # Tensor score
            scores_tensor = torch.tensor(scores)

            # NMS
            keep = nms(boxes_xyxy, scores_tensor, nms_threshold)

            boxes_filt = boxes_filt[keep]
            pred_phrases = [pred_phrases[i] for i in keep.tolist()]

    # ------------------------------------------------------------------
    # GIVEN-PHRASE MODE (có token_spans)
    # ------------------------------------------------------------------
    else:
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device)  # (n_phrase, 256)

        logits_for_phrases = positive_maps @ logits.T  # (n_phrase, nq)

        all_boxes = []
        all_phrases = []
        all_scores = []

        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])

            filt_mask = logit_phr > box_threshold

            boxes_this = boxes[filt_mask]
            scores_this = logit_phr[filt_mask]

            for score in scores_this:
                all_scores.append(score.item())
                if with_logits:
                    all_phrases.append(f"{phrase}({score.item():.3f})")
                else:
                    all_phrases.append(phrase)

            all_boxes.append(boxes_this)

        if len(all_boxes) > 0 and sum(len(b) for b in all_boxes) > 0:
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()

            # Apply NMS
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt)
            scores_tensor = torch.tensor(all_scores)

            keep = nms(boxes_xyxy, scores_tensor, nms_threshold)

            boxes_filt = boxes_filt[keep]
            pred_phrases = [all_phrases[i] for i in keep.tolist()]
        else:
            boxes_filt = torch.empty((0, 4))
            pred_phrases = []

    return boxes_filt, pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--image_path", "-i", type=str, required=True)
    parser.add_argument("--text_prompt", "-t", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True)

    parser.add_argument("--box_threshold", type=float, default=0.5)
    parser.add_argument("--text_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--token_spans", type=str, default=None)
    parser.add_argument("--cpu-only", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load image & model
    image_pil, image = load_image(args.image_path)
    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)

    # Token spans handling
    token_spans = eval(args.token_spans) if args.token_spans is not None else None
    text_threshold = None if token_spans is not None else args.text_threshold

    # Inference
    boxes_filt, pred_phrases = get_grounding_output(
        model=model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=text_threshold,
        cpu_only=args.cpu_only,
        token_spans=token_spans,
        nms_threshold=args.nms_threshold
    )

    # Tạo tên file
    stem = Path(args.image_path).stem
    image_id = stem.split("-")[-1] if "-" in stem else stem
    output_stem = f"pred-{image_id}"

    # ====================== CLASS MAPPING ======================
    caption_clean = args.text_prompt.lower().strip()
    if caption_clean.endswith("."):
        caption_clean = caption_clean[:-1]
    
    class_names = [name.strip() for name in caption_clean.split(" . ")]
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    print(f"Detected {len(class_names)} classes: {class_names}")

    # ====================== SAVE TXT ======================
    txt_path = os.path.join(args.output_dir, f"{output_stem}.txt")
    lines = []

    for box, phrase in zip(boxes_filt, pred_phrases):
        clean_phrase = phrase.split('(')[0].strip()   # bỏ score
        class_id = class_name_to_id.get(clean_phrase)
        if class_id is None:
            print(f"[WARN] Unknown phrase skipped: {clean_phrase}")
            continue  # fallback = 0

        x_c, y_c, w, h = box.tolist()
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Saved TXT: {txt_path}")

    # ====================== SAVE IMAGE ======================
    boxes_tensor = boxes_filt if len(boxes_filt) > 0 else torch.empty((0, 4), dtype=torch.float32)
    
    pred_dict = {
        "boxes": boxes_tensor,
        "size": [image_pil.height, image_pil.width],  # H, W
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)
    img_path = os.path.join(args.output_dir, f"{output_stem}.jpg")
    image_with_box.save(img_path, quality=95)
    print(f"[OK] Saved Image: {img_path}")

    print(f"✅ Finished processing: {Path(args.image_path).name}")