import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.gesture_features import GESTURE_CLASSES, landmarks_to_feature_vector
from vision.hand_tracker import HandTracker


KEY_TO_LABEL = {
    ord("1"): "Fist",
    ord("2"): "Open_Palm",
    ord("3"): "V_Sign",
}


def ensure_dataset_dirs(dataset_dir):
    for label in GESTURE_CLASSES:
        (dataset_dir / label).mkdir(parents=True, exist_ok=True)


def get_counts(dataset_dir):
    return {
        label: len(list((dataset_dir / label).glob("*.npy")))
        for label in GESTURE_CLASSES
    }


def save_feature(dataset_dir, label, landmarks):
    feature = landmarks_to_feature_vector(landmarks)
    if feature is None:
        return None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = dataset_dir / label / f"{label}_{stamp}.npy"
    np.save(output_path, feature)
    return output_path


def draw_overlay(frame, active_label, counts, hand_count):
    lines = [
        "Landmark dataset capture",
        "1: Fist | 2: Open_Palm | 3: V_Sign | Q: quit",
        f"Detected hands: {hand_count}",
        f"Last saved: {active_label or '-'}",
        "Samples: " + "  ".join(f"{label}: {counts.get(label, 0)}" for label in GESTURE_CLASSES),
    ]

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 28 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Capture normalized landmark vectors for MLP training.")
    parser.add_argument("--dataset", default="dataset_landmarks", help="Landmark dataset output directory.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    ensure_dataset_dirs(dataset_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hand_tracker = HandTracker(max_num_hands=1)
    counts = get_counts(dataset_dir)
    active_label = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            detection = hand_tracker.detect_hands(frame)
            frame = hand_tracker.draw_hands(frame, detection)
            hand_landmarks = (detection.get("hand_landmarks") or [None])[0]

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in KEY_TO_LABEL and hand_landmarks is not None:
                label = KEY_TO_LABEL[key]
                saved_path = save_feature(dataset_dir, label, hand_landmarks)
                if saved_path is not None:
                    active_label = label
                    counts = get_counts(dataset_dir)
                    print(f"Saved {saved_path}")

            draw_overlay(frame, active_label, counts, len(detection.get("hand_landmarks") or []))
            cv2.imshow("Landmark Dataset Capture", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
