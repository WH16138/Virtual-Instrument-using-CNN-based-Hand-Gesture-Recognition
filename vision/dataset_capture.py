import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision import dataset_capture_both, dataset_capture_cnn, dataset_capture_landmarks


def parse_args():
    parser = argparse.ArgumentParser(description="Capture VisionQuest gesture datasets.")
    parser.add_argument(
        "--mode",
        choices=["both", "landmarks", "cnn"],
        default="both",
        help="Dataset type to capture. both saves PNG crops and landmark vectors together.",
    )
    parser.add_argument("--dataset", default=None, help="Override output dataset directory.")
    parser.add_argument("--png-dataset", default=None, help="PNG output directory for both mode.")
    parser.add_argument("--landmark-dataset", default=None, help="Landmark output directory for both mode.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main():
    args = parse_args()
    forwarded = [
        "dataset_capture",
        "--camera",
        str(args.camera),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.dataset:
        forwarded.extend(["--dataset", args.dataset])

    old_argv = sys.argv
    try:
        if args.mode == "both":
            forwarded = [
                "dataset_capture",
                "--camera",
                str(args.camera),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ]
            if args.png_dataset:
                forwarded.extend(["--png-dataset", args.png_dataset])
            if args.landmark_dataset:
                forwarded.extend(["--landmark-dataset", args.landmark_dataset])
            sys.argv = forwarded
            dataset_capture_both.main()
        elif args.mode == "landmarks":
            sys.argv = forwarded
            dataset_capture_landmarks.main()
        else:
            sys.argv = forwarded
            dataset_capture_cnn.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
