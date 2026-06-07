import cv2
import cv2
import importlib
import numpy as np
import sys
import tempfile
import types
from pathlib import Path

class HandTracker:
    """최신 MediaPipe Task API를 사용한 손 감지 및 랜드마크 추출"""

    def __init__(
        self,
        model_path=None,
        max_num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.model_path = Path(model_path) if model_path else self._find_model_path()
        self._install_tensorflow_doc_controls_stub()
        self._install_matplotlib_stub()
        self.hand_module = importlib.import_module("mediapipe.tasks.python.vision.hand_landmarker")
        self.image_module = importlib.import_module("mediapipe.tasks.python.vision.core.image")
        self.hand_landmarker = self.hand_module.HandLandmarker.create_from_model_path(
            str(self.model_path)
        )
        self.hand_connections = self.hand_module.HandLandmarksConnections.HAND_CONNECTIONS
        self.temp_image_path = Path(tempfile.gettempdir()) / "mediapipe_hand_tracker_frame.jpg"
        self.image_format = getattr(self.image_module, "ImageFormat", None)
        self.max_num_hands = max_num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

    @staticmethod
    def _install_tensorflow_doc_controls_stub():
        """Avoid TensorFlow DLL loading triggered by MediaPipe doc-only imports."""
        if "tensorflow" in sys.modules:
            return

        def decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda obj: obj

        class DocControlsStub(types.ModuleType):
            def __getattr__(self, _name):
                return decorator

        tensorflow_module = types.ModuleType("tensorflow")
        tools_module = types.ModuleType("tensorflow.tools")
        docs_module = types.ModuleType("tensorflow.tools.docs")
        doc_controls_module = DocControlsStub("tensorflow.tools.docs.doc_controls")

        docs_module.doc_controls = doc_controls_module
        tools_module.docs = docs_module
        tensorflow_module.tools = tools_module

        sys.modules["tensorflow"] = tensorflow_module
        sys.modules["tensorflow.tools"] = tools_module
        sys.modules["tensorflow.tools.docs"] = docs_module
        sys.modules["tensorflow.tools.docs.doc_controls"] = doc_controls_module

    @staticmethod
    def _install_matplotlib_stub():
        """Avoid bundling matplotlib for MediaPipe's unused plotting helper."""
        if "matplotlib" in sys.modules:
            return

        matplotlib_module = types.ModuleType("matplotlib")
        pyplot_module = types.ModuleType("matplotlib.pyplot")

        def _unused_plot(*_args, **_kwargs):
            return None

        pyplot_module.figure = _unused_plot
        pyplot_module.show = _unused_plot
        pyplot_module.close = _unused_plot
        pyplot_module.gca = _unused_plot
        pyplot_module.plot = _unused_plot
        pyplot_module.scatter = _unused_plot
        matplotlib_module.pyplot = pyplot_module

        sys.modules["matplotlib"] = matplotlib_module
        sys.modules["matplotlib.pyplot"] = pyplot_module
    def _find_model_path(self):
        candidates = [
            Path("models") / "hand_landmarker.task",
            Path("hand_landmarker.task"),
            Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "MediaPipe HandLandmarker 모델 파일(hand_landmarker.task)을 찾을 수 없습니다.\n"
            "리포지터리 루트 또는 models/ 폴더에 파일을 추가하세요.\n"
            "공식 문서: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python"
        )

    def _prepare_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.image_format is not None:
            try:
                return self.image_module.Image(
                    image_format=self.image_format.SRGB,
                    data=rgb_frame,
                )
            except Exception:
                pass

        cv2.imwrite(str(self.temp_image_path), frame)
        return self.image_module.Image.create_from_file(str(self.temp_image_path))

    def detect_hands(self, frame):
        """손 감지 및 랜드마크 추출"""
        left_hand = None
        right_hand = None
        handedness_list = []
        hand_landmarks_list = []

        mp_image = self._prepare_image(frame)
        results = self.hand_landmarker.detect(mp_image)

        if results and results.hand_landmarks and results.handedness:
            for landmarks, hand_info in zip(results.hand_landmarks, results.handedness):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                label = getattr(hand_info[0], "category_name", None) or getattr(hand_info[0], "display_name", None)
                label = label or "Unknown"
                handedness_list.append(label)
                hand_landmarks_list.append(coords)

                if label.lower() == "left":
                    left_hand = coords
                else:
                    right_hand = coords

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "handedness": handedness_list,
            "hand_landmarks": hand_landmarks_list,
        }

    def draw_hands(self, frame, detection_result):
        """손 랜드마크 그리기"""
        hand_landmarks_list = detection_result.get("hand_landmarks") or []
        height, width = frame.shape[:2]

        for landmarks in hand_landmarks_list:
            points = [
                (
                    min(max(int(lm[0] * width), 0), width - 1),
                    min(max(int(lm[1] * height), 0), height - 1),
                )
                for lm in landmarks
            ]

            for connection in self.hand_connections:
                start = points[connection.start]
                end = points[connection.end]
                cv2.line(frame, start, end, (0, 255, 0), 2)

            for point in points:
                cv2.circle(frame, point, 3, (0, 255, 255), -1)

        return frame
