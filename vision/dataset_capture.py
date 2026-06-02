import cv2
import datetime
import importlib
import tempfile
from pathlib import Path

GESTURE_LABELS = {
    ord("1"): "Fist",
    ord("2"): "Open_Palm",
    ord("3"): "V_Sign"
}

DATASET_DIR = Path("dataset")
MODEL_FILENAME = "hand_landmarker.task"
MODEL_CANDIDATES = [
    Path("models") / MODEL_FILENAME,
    Path(MODEL_FILENAME),
    Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME,
]


def ensure_dataset_dirs():
    for label in GESTURE_LABELS.values():
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


def find_model_path():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    candidate_list = "\n".join(f"  - {path}" for path in MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"MediaPipe HandLandmarker 모델 파일을 찾을 수 없습니다.\n"
        f"다음 경로 중 하나에 '{MODEL_FILENAME}'를 놓고 다시 실행하세요:\n"
        f"{candidate_list}\n"
        "공식 문서: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python"
    )


def load_hand_landmarker():
    hand_module = importlib.import_module("mediapipe.tasks.python.vision.hand_landmarker")
    image_module = importlib.import_module("mediapipe.tasks.python.vision.core.image")
    model_path = find_model_path()

    hand_landmarker = hand_module.HandLandmarker.create_from_model_path(str(model_path))
    return hand_landmarker, image_module.Image, hand_module.HandLandmarksConnections.HAND_CONNECTIONS


def crop_hand_region(frame, hand_landmarks, padding=20):
    height, width = frame.shape[:2]
    xs = [lm.x * width for lm in hand_landmarks]
    ys = [lm.y * height for lm in hand_landmarks]

    x1 = max(int(min(xs) - padding), 0)
    y1 = max(int(min(ys) - padding), 0)
    x2 = min(int(max(xs) + padding), width - 1)
    y2 = min(int(max(ys) + padding), height - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def draw_hand_landmarks(frame, landmarks, connections):
    height, width = frame.shape[:2]
    points = [
        (min(max(int(lm.x * width), 0), width - 1), min(max(int(lm.y * height), 0), height - 1))
        for lm in landmarks
    ]

    for connection in connections:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    return frame


def save_hand_image(image, label):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = DATASET_DIR / label / f"{label}_{stamp}.png"
    cv2.imwrite(str(filename), image)
    return filename


def get_counts():
    return {
        label: len(list((DATASET_DIR / label).glob("*.png")))
        for label in GESTURE_LABELS.values()
    }


def draw_overlay(frame, hand_crop, counts):
    lines = [
        "Gesture dataset capture",
        "Press 1: Fist | 2: Open_Palm | 3: V_Sign",
        "Press Q: quit",
        "",
        f"Fist: {counts['Fist']}  Open_Palm: {counts['Open_Palm']}  V_Sign: {counts['V_Sign']}"
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    if hand_crop is not None:
        crop_height = 160
        crop_width = int(hand_crop.shape[1] * crop_height / hand_crop.shape[0])
        crop_preview = cv2.resize(hand_crop, (crop_width, crop_height))
        frame[10:10 + crop_height, frame.shape[1] - crop_width - 10:frame.shape[1] - 10] = crop_preview
        cv2.rectangle(
            frame,
            (frame.shape[1] - crop_width - 10, 10),
            (frame.shape[1] - 10, 10 + crop_height),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "Hand crop",
            (frame.shape[1] - crop_width - 10, 10 + crop_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame


def detect_hand_landmarks(frame, hand_landmarker, image_cls, temp_path):
    cv2.imwrite(str(temp_path), frame)
    mp_image = image_cls.create_from_file(str(temp_path))
    return hand_landmarker.detect(mp_image)


def main():
    ensure_dataset_dirs()

    try:
        hand_landmarker, image_cls, hand_connections = load_hand_landmarker()
    except FileNotFoundError as exc:
        print(exc)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    counts = get_counts()
    hand_crop = None
    frame_index = 0
    temp_image = Path(tempfile.gettempdir()) / "mediapipe_hand_frame.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        try:
            result = detect_hand_landmarks(frame, hand_landmarker, image_cls, temp_image)
        except Exception:
            result = None

        if result and result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_hand_landmarks(frame, hand_landmarks, hand_connections)
            hand_crop = crop_hand_region(frame, hand_landmarks)
        else:
            hand_crop = None

        frame = draw_overlay(frame, hand_crop, counts)
        cv2.imshow("Gesture Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key in GESTURE_LABELS and hand_crop is not None:
            label = GESTURE_LABELS[key]
            saved_path = save_hand_image(hand_crop, label)
            counts = get_counts()
            print(f"Saved {saved_path}")

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
