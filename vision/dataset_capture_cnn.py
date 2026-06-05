import argparse
import datetime
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.gesture_features import GESTURE_CLASSES
from vision.hand_tracker import HandTracker


KEY_TO_LABEL = {
    ord("1"): "Fist",
    ord("2"): "Open_Palm",
    ord("3"): "V_Sign",
    ord("4"): "Gun_Sign",
    ord("5"): "OK_Sign",
}


def ensure_dataset_dirs(dataset_dir):
    for label in GESTURE_CLASSES:
        (dataset_dir / label).mkdir(parents=True, exist_ok=True)


def get_counts(dataset_dir):
    return {
        label: len(list((dataset_dir / label).glob("*.png")))
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


def save_crop(dataset_dir, label, crop):
    if crop is None or crop.size == 0:
        return None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = dataset_dir / label / f"{label}_{stamp}.png"
    cv2.imwrite(str(output_path), crop)
    return output_path


def draw_overlay(frame, active_label, counts, crop):
    lines = [
        "PNG/CNN dataset capture",
        "1: Fist | 2: Open_Palm | 3: V_Sign | 4: Gun_Sign | 5: OK_Sign | Q: quit",
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

    if crop is not None and crop.size > 0:
        preview_height = 140
        preview_width = max(1, int(crop.shape[1] * preview_height / crop.shape[0]))
        preview = cv2.resize(crop, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        x1 = frame.shape[1] - preview_width - 10
        y1 = 10
        frame[y1 : y1 + preview_height, x1 : x1 + preview_width] = preview
        cv2.rectangle(frame, (x1, y1), (x1 + preview_width, y1 + preview_height), (0, 255, 0), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture PNG hand crops for CNN training.")
    parser.add_argument("--dataset", default="dataset", help="PNG dataset output directory.")
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
    crop = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            detection = hand_tracker.detect_hands(clean_frame)
            hand_landmarks = (detection.get("hand_landmarks") or [None])[0]
            crop = crop_hand_region(clean_frame, hand_landmarks)
            frame = hand_tracker.draw_hands(frame, detection)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in KEY_TO_LABEL and crop is not None:
                label = KEY_TO_LABEL[key]
                saved_path = save_crop(dataset_dir, label, crop)
                if saved_path is not None:
                    active_label = label
                    counts = get_counts(dataset_dir)
                    print(f"Saved {saved_path}")

            draw_overlay(frame, active_label, counts, crop)
            cv2.imshow("PNG CNN Dataset Capture", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
