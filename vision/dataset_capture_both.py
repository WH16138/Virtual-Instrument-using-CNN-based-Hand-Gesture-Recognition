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
    ord("4"): "Gun_Sign",
    ord("5"): "OK_Sign",
}


def ensure_dataset_dirs(png_dataset_dir, landmark_dataset_dir):
    for label in GESTURE_CLASSES:
        (png_dataset_dir / label).mkdir(parents=True, exist_ok=True)
        (landmark_dataset_dir / label).mkdir(parents=True, exist_ok=True)


def get_counts(dataset_dir, suffix):
    return {
        label: len(list((dataset_dir / label).glob(f"*{suffix}")))
        for label in GESTURE_CLASSES
    }


def crop_hand_region(frame, landmarks, padding=24):
    if landmarks is None:
        return None

    height, width = frame.shape[:2]
    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height

    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, width - 1)
    y2 = min(int(ys.max()) + padding, height - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def save_sample(png_dataset_dir, landmark_dataset_dir, label, frame, landmarks):
    crop = crop_hand_region(frame, landmarks)
    feature = landmarks_to_feature_vector(landmarks)
    if crop is None or crop.size == 0 or feature is None:
        return None, None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = png_dataset_dir / label / f"{label}_{stamp}.png"
    npy_path = landmark_dataset_dir / label / f"{label}_{stamp}.npy"

    cv2.imwrite(str(png_path), crop)
    np.save(npy_path, feature)
    return png_path, npy_path


def draw_overlay(frame, active_label, png_counts, landmark_counts, crop):
    lines = [
        "Combined dataset capture",
        "Saves PNG crop + landmark NPY together",
        "1: Fist | 2: Open_Palm | 3: V_Sign | 4: Gun_Sign | 5: OK_Sign | Q: quit",
        f"Last saved: {active_label or '-'}",
        "PNG: " + "  ".join(f"{label}: {png_counts.get(label, 0)}" for label in GESTURE_CLASSES),
        "NPY: " + "  ".join(f"{label}: {landmark_counts.get(label, 0)}" for label in GESTURE_CLASSES),
    ]

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 28 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if crop is not None and crop.size > 0:
        preview_height = 140
        preview_width = max(1, int(crop.shape[1] * preview_height / crop.shape[0]))
        preview = cv2.resize(crop, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        x1 = frame.shape[1] - preview_width - 10
        y1 = 10
        frame[y1 : y1 + preview_height, x1 : x1 + preview_width] = preview
        cv2.rectangle(frame, (x1, y1), (x1 + preview_width, y1 + preview_height), (0, 255, 0), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture both PNG crops and landmark vectors.")
    parser.add_argument("--png-dataset", default="dataset", help="PNG/CNN dataset output directory.")
    parser.add_argument("--landmark-dataset", default="dataset_landmarks", help="Landmark/MLP dataset output directory.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main():
    args = parse_args()
    png_dataset_dir = Path(args.png_dataset)
    landmark_dataset_dir = Path(args.landmark_dataset)
    ensure_dataset_dirs(png_dataset_dir, landmark_dataset_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hand_tracker = HandTracker(max_num_hands=1)
    png_counts = get_counts(png_dataset_dir, ".png")
    landmark_counts = get_counts(landmark_dataset_dir, ".npy")
    active_label = None
    crop = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            detection = hand_tracker.detect_hands(clean_frame)
            landmarks = (detection.get("hand_landmarks") or [None])[0]
            crop = crop_hand_region(clean_frame, landmarks)
            frame = hand_tracker.draw_hands(frame, detection)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in KEY_TO_LABEL and landmarks is not None:
                label = KEY_TO_LABEL[key]
                png_path, npy_path = save_sample(png_dataset_dir, landmark_dataset_dir, label, clean_frame, landmarks)
                if png_path is not None and npy_path is not None:
                    active_label = label
                    png_counts = get_counts(png_dataset_dir, ".png")
                    landmark_counts = get_counts(landmark_dataset_dir, ".npy")
                    print(f"Saved {png_path}")
                    print(f"Saved {npy_path}")

            draw_overlay(frame, active_label, png_counts, landmark_counts, crop)
            cv2.imshow("Combined Dataset Capture", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
