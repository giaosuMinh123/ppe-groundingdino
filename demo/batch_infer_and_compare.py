import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n" + "=" * 120)
    print("RUN:", " ".join(str(x) for x in cmd))
    print("=" * 120)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO inference + comparison for all images in a folder"
    )

    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable to use, default=current Python",
    )
    parser.add_argument(
        "--inference_script",
        required=True,
        help="Path to inference_on_a_image_3.py",
    )
    parser.add_argument(
        "--compare_script",
        required=True,
        help="Path to compare_yolo_txt.py",
    )
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to GroundingDINO config file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Folder containing pexels-photo-*.jpeg",
    )
    parser.add_argument(
        "--labels_dir",
        required=True,
        help="Folder containing pexels-photo-*.txt ground-truth labels",
    )
    parser.add_argument(
        "--predicted_dir",
        required=True,
        help="Folder to save prediction outputs",
    )
    parser.add_argument(
        "--compare_dir",
        required=True,
        help="Folder to save comparison outputs",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for comparison",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Use CPU only",
    )
    parser.add_argument(
    "--text_prompt",
    required=True,
    help="Text prompt for GroundingDINO",
    )

    args = parser.parse_args()

    python_exe = args.python_exe
    inference_script = Path(args.inference_script)
    compare_script = Path(args.compare_script)
    config_file = Path(args.config_file)
    checkpoint = args.checkpoint
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    predicted_dir = Path(args.predicted_dir)
    compare_dir = Path(args.compare_dir)

    predicted_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("pexels-photo-*.jpeg"))

    if not image_files:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_files)} images.")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for image_path in image_files:
        image_stem = image_path.stem                  # pexels-photo-209230
        image_id = image_stem.split("-")[-1]         # 209230

        gt_path = labels_dir / f"{image_stem}.txt"
        pred_path = predicted_dir / f"pred-{image_id}.txt"

        print("\n" + "#" * 120)
        print(f"PROCESSING: {image_path.name}")
        print("#" * 120)

        if not gt_path.exists():
            print(f"[SKIP] Missing GT label: {gt_path}")
            skipped_count += 1
            continue

        try:
            # 1) inference
            infer_cmd = [
                python_exe,
                str(inference_script),
                "--config_file", str(config_file),
                "--checkpoint_path", str(checkpoint),
                "--image_path", str(image_path),
                "--output_dir", str(predicted_dir),
                "--text_prompt", args.text_prompt,
            ]
            if args.cpu_only:
                infer_cmd.append("--cpu-only")

            run_command(infer_cmd)

            if not pred_path.exists():
                raise FileNotFoundError(f"Prediction file not found after inference: {pred_path}")

            # 2) compare
            compare_cmd = [
                python_exe,
                str(compare_script),
                "--gt", str(gt_path),
                "--pred", str(pred_path),
                "--iou", str(args.iou),
                "--output_dir", str(compare_dir),
            ]

            run_command(compare_cmd)

            success_count += 1

        except Exception as e:
            print(f"[FAILED] {image_path.name}: {e}")
            failed_count += 1

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)
    print(f"Success : {success_count}")
    print(f"Skipped : {skipped_count}")
    print(f"Failed  : {failed_count}")


if __name__ == "__main__":
    main()