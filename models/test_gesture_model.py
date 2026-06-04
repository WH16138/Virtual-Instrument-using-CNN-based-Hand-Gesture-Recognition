import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.gesture_detector import GestureDetector
from vision.hand_tracker import HandTracker


def draw_prediction_panel(frame, title, gesture_info, x, y):
    gesture = gesture_info.get("gesture", "Unknown")
    smoothed = gesture_info.get("smoothed_gesture", "Unknown")
    confidence = gesture_info.get("confidence", 0.0)
    color = (0, 255, 0) if confidence >= 0.6 else (0, 180, 255)

    cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Class: {gesture}", (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Stable: {smoothed}", (x, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the trained gesture model on a live camera feed.")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "gesture_model.pkl"),
        help="Path to the trained Keras gesture model.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Requested camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Requested camera capture height.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Gesture model not found: {model_path}")

    hand_tracker = HandTracker(max_num_hands=2)
    gesture_detector = GestureDetector(str(model_path))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Gesture model test started.")
    print("Show your hand to the camera. Press Q to quit.")

    fps_clock = cv2.getTickCount()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - fps_clock, 1)
            fps_clock = cv2.getTickCount()

            hand_detection = hand_tracker.detect_hands(frame)
            frame = hand_tracker.draw_hands(frame, hand_detection)

            left_info = gesture_detector.detect_gesture(hand_detection.get("left_hand"), "left")
            right_info = gesture_detector.detect_gesture(hand_detection.get("right_hand"), "right")
            best_info = right_info if right_info["confidence"] > left_info["confidence"] else left_info

            draw_prediction_panel(frame, "Left hand", left_info, 10, 35)
            draw_prediction_panel(frame, "Right hand", right_info, 10, 150)

            best_color = (0, 255, 0) if best_info["confidence"] >= 0.6 else (0, 180, 255)
            cv2.putText(
                frame,
                f"Best: {best_info['smoothed_gesture']} ({best_info['confidence']:.2f})",
                (10, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                best_color,
                2,
            )
            cv2.putText(
                frame,
                f"Hands detected: {len(hand_detection.get('hand_landmarks') or [])} | FPS: {fps:.1f}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Gesture Model Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
