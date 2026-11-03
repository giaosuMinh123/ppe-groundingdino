"""
GroundingDINO zero-shot inference for PPE detection.

Design notes
------------
1. Per-class NMS. Boxes are suppressed only against boxes of the SAME class,
   so a box of a sub-region class (head, face, hands) is never removed by an
   overlapping "person" box.
2. Class assignment through token_spans (character offsets of each phrase in
   the caption) instead of string-matching the phrase decoded by the model,
   which is unstable and silently drops boxes.
3. Class list and class ids are fixed. They are read from classes.txt and do
   not depend on the word order of the caption in a given run.
4. The model is loaded once and a whole image directory is processed.
5. Optional per-class display thresholds. Classes without an entry fall back
   to --box_threshold.
6. Label font size scales with the image width.
7. --prompt_mode selects how text is sent to the model:
     per_class (default): one forward pass per class, with a one-phrase
         caption such as "helmet .". Each phrase gets the model's full
         attention. Cost: N forward passes per image (N = number of classes).
     combined: a single caption "a . b . c ." and one forward pass. Faster;
         token_spans map each box back to its class.
8. --synonyms separates the official class name (used for the class id, the
   label and the .txt output; it must match the ground truth) from the search
   term actually sent to the model, e.g. class "face-guard" queried as
   "face shield".
9. Two thresholds with different roles:
     --capture_threshold (default 0.05): low cut-off that decides which
         candidates are written to the .csv, which keeps scores. The .csv is
         the input for PR curves and mAP, so it must not be truncated early.
     --box_threshold / --class_thresholds: display cut-off applied afterwards.
         It only affects the .txt and .jpg outputs, never the .csv.
10. --only_classes (optional) runs a subset of classes while keeping their
    ids from classes.txt. When omitted, all classes are run.

Synonym / description file (JSON). Only list the classes to override:
    {
      "face-guard": "face shield",
      "medical-suit": "hazmat suit",
      "safety-suit": "coverall"
    }

Usage (per_class, the default):
    python grounding_dino_ppe_optimized.py -c config.py -p weights.pth --classes classes.txt --image_dir path/to/images -o outputs --box_threshold 0.35 --nms_threshold 0.5

Usage (combined):
    python grounding_dino_ppe_optimized.py -c config.py -p weights.pth --classes classes.txt --image_dir path/to/images -o outputs --prompt_mode combined

classes.txt: one class name per line, in the exact order of the ids 0, 1, 2, ...
    person
    helmet
    gloves
    ...

Outputs per image (<stem> = "pred-<image id>"):
    <stem>.csv  every candidate >= capture_threshold: class_id,x_center,y_center,width,height,score
    <stem>.txt  boxes >= display threshold, YOLO format: class_id x_center y_center width height
    <stem>.jpg  the image with the displayed boxes drawn
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms, nms

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.vl_utils import create_positive_map_from_span

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Class list, synonyms, and caption / token_spans construction
# --------------------------------------------------------------------------
def load_class_names(classes_path: str) -> list[str]:
    """Read classes.txt (one name per line) and return lower-cased names in id order."""
    with open(classes_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_path}")
    return names


def load_synonyms(synonyms_arg: str | None, class_names: list[str]) -> dict[str, str]:
    """
    Build the mapping official class name -> search term sent to the model.

    The official name determines the class id, the .txt output and the label,
    so it must match the ground truth. The search term is only the text the
    model sees (e.g. class "face-guard" queried as "face shield").

    Args:
        synonyms_arg: path to a JSON file, or a JSON string, such as
            {"face-guard": "face shield", "medical-suit": "hazmat suit"}.
            None means no overrides.
        class_names: official class names from classes.txt.

    Returns:
        Dict covering every class. Classes without an override map to their
        own name. Keys that match no class are reported and ignored.
    """
    mapping = {name: name for name in class_names}
    if not synonyms_arg:
        return mapping

    if os.path.isfile(synonyms_arg):
        with open(synonyms_arg, "r", encoding="utf-8") as f:
            user_map = json.load(f)
    else:
        user_map = json.loads(synonyms_arg)

    for k, v in user_map.items():
        k_norm = k.strip().lower()
        if k_norm not in mapping:
            print(f"[warn] synonym key '{k}' does not match any class in classes.txt; ignored.")
            continue
        mapping[k_norm] = v.strip().lower()

    return mapping


def build_caption_and_token_spans(class_names: list[str]):
    """
    Join the phrases into one caption "a . b . c ." and locate each phrase.

    Returns:
        caption: the joined caption string.
        token_spans: for each phrase, [[start, end]] character offsets inside
            the caption. They are used to map model tokens back to classes,
            which does not depend on how the model decodes the phrase.
    """
    caption = ""
    token_spans = []
    for name in class_names:
        start = len(caption)
        caption += name
        end = len(caption)
        token_spans.append([[start, end]])
        caption += " . "
    caption = caption.strip()
    return caption, token_spans


# --------------------------------------------------------------------------
# Image / model loading
# --------------------------------------------------------------------------
def load_image(image_path: str):
    """Load an RGB image; return the PIL image and the normalised model input tensor."""
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


def resolve_device(cpu_only: bool) -> str:
    """Return "cpu" or "cuda"; fall back to CPU when CUDA is unavailable instead of crashing."""
    if cpu_only:
        return "cpu"
    if not torch.cuda.is_available():
        print("[warn] CUDA is not available; falling back to CPU (slower).")
        return "cpu"
    return "cuda"


def load_model(model_config_path: str, model_checkpoint_path: str, cpu_only: bool = False):
    """Build GroundingDINO from its config, load the checkpoint, and switch to eval mode."""
    args = SLConfig.fromfile(model_config_path)
    args.device = resolve_device(cpu_only)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"[load_model] {load_res}")
    model.eval()
    return model


# --------------------------------------------------------------------------
# Inference, mode "combined": one caption, one forward pass
# --------------------------------------------------------------------------
@torch.no_grad()
def get_grounding_output_multiclass(
    model,
    image: torch.Tensor,
    class_names: list[str],
    search_terms: list[str],
    capture_threshold: float,
    nms_threshold: float = 0.5,
    cpu_only: bool = False,
):
    """
    Run all classes in a single caption and apply NMS separately per class.

    Args:
        model: loaded GroundingDINO model.
        image: normalised image tensor from load_image.
        class_names: official class names (define the class ids).
        search_terms: phrases actually sent to the model; same length and
            order as class_names (may be synonyms or descriptions).
        capture_threshold: low cut-off that keeps candidates for later
            analysis (PR curve / mAP). The final display threshold is applied
            later, when the files are written, not here.
        nms_threshold: IoU above which a lower-scored box of the same class
            is suppressed.
        cpu_only: force CPU.

    Returns:
        boxes_filt: Tensor (N, 4), cxcywh normalised to [0, 1].
        class_ids: list[int] of length N, indices into class_names.
        scores: list[float] of length N.
    """
    caption, token_spans = build_caption_and_token_spans(search_terms)

    device = resolve_device(cpu_only)
    model = model.to(device)
    image = image.to(device)

    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, 256 text tokens)
    boxes = outputs["pred_boxes"][0]  # (num_queries, 4), cxcywh normalised

    positive_maps = create_positive_map_from_span(
        model.tokenizer(caption), token_span=token_spans
    ).to(device)  # (num_classes, 256): which tokens belong to which class

    # Per-class score of every query: average of that class's token scores.
    logits_for_class = positive_maps @ logits.T  # (num_classes, num_queries)

    all_boxes, all_scores, all_class_ids = [], [], []

    for class_id, logit_c in enumerate(logits_for_class):
        mask = logit_c > capture_threshold
        if mask.sum() == 0:
            continue
        boxes_c = boxes[mask].cpu()
        scores_c = logit_c[mask].cpu()

        all_boxes.append(boxes_c)
        all_scores.append(scores_c)
        all_class_ids.extend([class_id] * boxes_c.shape[0])

    if not all_boxes:
        return torch.empty((0, 4)), [], []

    boxes_cat = torch.cat(all_boxes, dim=0)
    scores_cat = torch.cat(all_scores, dim=0)
    class_ids_t = torch.tensor(all_class_ids, dtype=torch.int64)

    # Per-class NMS: a box only competes with boxes of the same class.
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cat)
    keep = batched_nms(boxes_xyxy, scores_cat, class_ids_t, nms_threshold)

    boxes_filt = boxes_cat[keep]
    class_ids = class_ids_t[keep].tolist()
    scores = scores_cat[keep].tolist()

    return boxes_filt, class_ids, scores


# --------------------------------------------------------------------------
# Inference, mode "per_class": one forward pass per class
# --------------------------------------------------------------------------
@torch.no_grad()
def get_grounding_output_per_class(
    model,
    image: torch.Tensor,
    class_names: list[str],
    search_terms: list[str],
    capture_threshold: float,
    nms_threshold: float = 0.5,
    cpu_only: bool = False,
):
    """
    Run a separate forward pass for each class, with a caption holding only
    that class's phrase (e.g. "helmet .").

    Because the caption has a single phrase, classes do not compete for
    attention inside one long caption. This helps small classes and classes
    that are sub-regions of another object (head, face, hands inside person).

    Cost: N forward passes per image (N = number of classes), so it is about
    N times slower than combined mode. A GPU is recommended for large datasets.

    Args and Returns: same as get_grounding_output_multiclass.
    """
    device = resolve_device(cpu_only)
    model = model.to(device)
    image = image.to(device)

    all_boxes, all_scores, all_class_ids = [], [], []

    for class_id, (name, term) in enumerate(zip(class_names, search_terms)):
        caption = term.lower().strip()
        if not caption.endswith("."):
            caption = caption + " ."

        outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, 256 text tokens)
        boxes = outputs["pred_boxes"][0]  # (num_queries, 4), cxcywh normalised

        # The caption holds one class only, so the score of a query is its
        # maximum over all text tokens; no token_spans or positive map needed.
        scores_per_query = logits.max(dim=1)[0]  # (num_queries,)

        mask = scores_per_query > capture_threshold
        if mask.sum() == 0:
            continue

        boxes_c = boxes[mask].cpu()
        scores_c = scores_per_query[mask].cpu()

        # NMS inside this class: merge duplicate boxes of the same object.
        # Other classes are unaffected because each class is handled separately.
        if boxes_c.shape[0] > 1:
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_c)
            keep = nms(boxes_xyxy, scores_c, nms_threshold)
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]

        all_boxes.append(boxes_c)
        all_scores.append(scores_c)
        all_class_ids.extend([class_id] * boxes_c.shape[0])

    if not all_boxes:
        return torch.empty((0, 4)), [], []

    boxes_filt = torch.cat(all_boxes, dim=0)
    class_ids = all_class_ids
    scores = torch.cat(all_scores, dim=0).tolist()

    return boxes_filt, class_ids, scores


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def get_font(image_width: int):
    """Return a bold font whose size scales with the image width."""
    size = max(14, image_width // 80)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def plot_boxes_to_image(image_pil, boxes, class_ids, scores, class_names):
    """Draw boxes with "class(score)" labels on the image; each class has a fixed random colour."""
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    font = get_font(W)

    rng = np.random.default_rng(42)
    colors = {cid: tuple(rng.integers(0, 255, size=3).tolist()) for cid in set(class_ids)}

    for box, cid, score in zip(boxes, class_ids, scores):
        # cxcywh (normalised) -> xyxy (pixels)
        box = box * torch.tensor([W, H, W, H], dtype=box.dtype)
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = map(int, box.tolist())
        color = colors[cid]
        label = f"{class_names[cid]}({score:.2f})"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, W // 400))
        try:
            bbox = draw.textbbox((x0, y0), label, font=font)
        except Exception:
            bbox = (x0, y0, x0 + 8 * len(label), y0 + 16)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)

    return image_pil


# --------------------------------------------------------------------------
# Per-image processing
# --------------------------------------------------------------------------
def process_one_image(model, image_path, class_names, output_dir, args):
    """
    Run inference on one image and write its .csv, .txt and .jpg outputs.

    Returns:
        Number of boxes written to the .txt file (boxes above the display threshold).
    """
    stem = Path(image_path).stem
    image_id = stem.split("-")[-1] if "-" in stem else stem
    output_stem = f"pred-{image_id}"

    image_pil, image = load_image(str(image_path))

    infer_fn = (
        get_grounding_output_per_class
        if args.prompt_mode == "per_class"
        else get_grounding_output_multiclass
    )

    # Run only the active classes. The model sees a shorter class list, so its
    # class indices are mapped back to the ids of classes.txt right after.
    # Inference uses the LOW capture threshold so every candidate that might
    # matter for PR curves / mAP is kept; the display threshold comes later.
    active = args.active_ids
    boxes_filt, class_ids, scores = infer_fn(
        model=model,
        image=image,
        class_names=[class_names[i] for i in active],
        search_terms=[args.search_terms[i] for i in active],
        capture_threshold=args.capture_threshold,
        nms_threshold=args.nms_threshold,
        cpu_only=args.cpu_only,
    )
    class_ids = [active[c] for c in class_ids]

    # ---- CSV with scores: ALL captured candidates ----
    # Source of truth for evaluate_visualize.py (PR curve / mAP); not cut by
    # the display threshold.
    csv_path = os.path.join(output_dir, f"{output_stem}.csv")
    csv_lines = ["class_id,x_center,y_center,width,height,score"]
    csv_lines += [
        f"{cid},{x_c:.6f},{y_c:.6f},{w:.6f},{h:.6f},{score:.6f}"
        for (x_c, y_c, w, h), cid, score in zip(boxes_filt.tolist(), class_ids, scores)
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # ---- Display subset: apply the real thresholds (per-class, else global) ----
    # Only this subset goes into the .txt and .jpg.
    display_idx = [
        i for i, (cid, score) in enumerate(zip(class_ids, scores))
        if score >= args.class_thresholds_dict.get(class_names[cid], args.box_threshold)
    ]
    boxes_disp = boxes_filt[display_idx] if len(display_idx) > 0 else torch.empty((0, 4))
    class_ids_disp = [class_ids[i] for i in display_idx]
    scores_disp = [scores[i] for i in display_idx]

    # ---- YOLO-format TXT (class ids are fixed across runs) ----
    txt_path = os.path.join(output_dir, f"{output_stem}.txt")
    lines = [
        f"{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        for (x_c, y_c, w, h), cid in zip(boxes_disp.tolist(), class_ids_disp)
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- Visualization: only the displayed boxes ----
    img_out = plot_boxes_to_image(image_pil.copy(), boxes_disp, class_ids_disp, scores_disp, class_names)
    img_path = os.path.join(output_dir, f"{output_stem}.jpg")
    img_out.save(img_path, quality=95)

    return len(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("GroundingDINO PPE detection - optimized")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True, help="Path to classes.txt (one class per line, fixed id order)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, help="Single image")
    group.add_argument("--image_dir", type=str, help="Folder of images (batch mode)")

    parser.add_argument("--output_dir", "-o", type=str, default="outputs")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["per_class", "combined"],
        default="per_class",
        help=(
            "per_class: one forward pass per class (full attention on each "
            "phrase, about N times slower). "
            "combined: one caption with all classes (faster; token_spans "
            "separate the classes)."
        ),
    )
    parser.add_argument(
        "--capture_threshold",
        type=float,
        default=0.05,
        help=(
            "Low threshold applied at inference to KEEP candidates in the .csv "
            "(they are not lost). Do not set it high: it is the net that feeds "
            "PR curves / mAP."
        ),
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.35,
        help="DISPLAY threshold: affects only the .txt and .jpg outputs, never the .csv.",
    )
    parser.add_argument(
        "--class_thresholds",
        type=str,
        default=None,
        help=(
            'JSON string, e.g. \'{"helmet": 0.25, "gloves": 0.3}\'. Classes not listed '
            "use --box_threshold. Also a DISPLAY threshold (txt/jpg), not applied to the .csv."
        ),
    )
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument(
        "--synonyms",
        type=str,
        default=None,
        help=(
            "Path to a JSON file, or a JSON string, mapping official class name "
            "-> text sent to the model, e.g. '{\"face-guard\": \"face shield\", "
            "\"medical-suit\": \"hazmat suit\"}'. Class ids and labels keep the "
            "official names from classes.txt; only the query text changes."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--only_classes", type=str, default=None,
        help='Optional: run only these classes (comma-separated), e.g. "face-guard,safety-vest". '
             "Class ids still follow classes.txt. Omit to run all classes.",
    )
    args = parser.parse_args()

    args.class_thresholds_dict = json.loads(args.class_thresholds) if args.class_thresholds else {}

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = load_class_names(args.classes)
    print(f"[classes] {len(class_names)} classes (fixed ids 0..{len(class_names)-1}): {class_names}")

    synonyms_map = load_synonyms(args.synonyms, class_names)
    args.search_terms = [synonyms_map[name] for name in class_names]
    changed = [f"{n} -> {s}" for n, s in zip(class_names, args.search_terms) if n != s]
    if changed:
        print(f"[synonyms] {len(changed)} classes use a replacement query text: {changed}")

    if args.only_classes:
        wanted = [n.strip().lower() for n in args.only_classes.split(",") if n.strip()]
        missing = [n for n in wanted if n not in class_names]
        if missing:
            raise ValueError(f"--only_classes contains classes not in classes.txt: {missing}")
        args.active_ids = [class_names.index(n) for n in wanted]
        print(f"[only_classes] running {len(args.active_ids)} classes: {[class_names[i] for i in args.active_ids]}")
    else:
        args.active_ids = list(range(len(class_names)))

    print("[model] loading (only once)...")
    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)

    if args.image_path:
        image_paths = [Path(args.image_path)]
    else:
        image_paths = sorted(
            p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMG_EXTS
        )
        print(f"[batch] found {len(image_paths)} images in {args.image_dir}")

    n_ok, n_fail, total_boxes = 0, 0, 0
    for img_path in image_paths:
        try:
            n_boxes = process_one_image(model, img_path, class_names, args.output_dir, args)
            total_boxes += n_boxes
            n_ok += 1
            print(f"[OK] {img_path.name}: {n_boxes} boxes")
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {img_path.name}: {e}")

    print("=" * 60)
    print(f"Done. Success: {n_ok}, Failed: {n_fail}, Total boxes: {total_boxes}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()