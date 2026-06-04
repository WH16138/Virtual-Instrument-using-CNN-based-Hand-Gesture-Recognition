import pickle
from collections import deque
from pathlib import Path

import numpy as np

from vision.gesture_features import GESTURE_CLASSES, landmarks_to_feature_vector


class GestureDetector:
    """Classify MediaPipe hand landmarks.

    Default runtime uses a lightweight scikit-learn model saved as .pkl so the
    main app does not depend on TensorFlow native DLLs. Keras .keras models are
    still supported as an optional fallback when TensorFlow is available.
    """

    def __init__(self, model_path=None, smoothing_size=5):
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self.model_kind = None
        self.model_loaded = False
        self.gesture_classes = GESTURE_CLASSES
        self.smoothing_size = smoothing_size
        self.smoothing_windows = {}

        if self.model_path is not None:
            self._load_model(self.model_path)
        else:
            print("Warning: no gesture model found. Gesture classification is disabled.")

    def _resolve_model_path(self, model_path):
        if model_path:
            return Path(model_path)

        candidates = [
            Path("models") / "gesture_model.pkl",
            Path("models") / "gesture_model.keras",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_model(self, model_path):
        try:
            if model_path.suffix.lower() == ".pkl":
                with model_path.open("rb") as handle:
                    self.model = pickle.load(handle)
                self.model_kind = "sklearn"
            elif model_path.suffix.lower() == ".keras":
                import tensorflow as tf

                self.model = tf.keras.models.load_model(str(model_path))
                self.model_kind = "keras"
            else:
                print(f"Warning: unsupported gesture model format: {model_path}")
                return

            self.model_loaded = True
            print(f"Loaded gesture model: {model_path}")
        except Exception as exc:
            print(f"Warning: could not load gesture model: {model_path} ({exc})")
            self.model = None
            self.model_kind = None
            self.model_loaded = False

    def detect_gesture(self, hand_landmarks, stream_id="default"):
        if hand_landmarks is None or not self.model_loaded:
            return self._unknown()

        feature = landmarks_to_feature_vector(hand_landmarks)
        if feature is None:
            return self._unknown()

        try:
            probabilities = self._predict_probabilities(feature)
            confidence = float(np.max(probabilities))
            gesture_idx = int(np.argmax(probabilities))
            gesture = self.gesture_classes[gesture_idx]

            smoothing_window = self.smoothing_windows.setdefault(stream_id, deque(maxlen=self.smoothing_size))
            smoothing_window.append(gesture)
            smoothed_gesture = max(set(smoothing_window), key=list(smoothing_window).count)

            return {
                "gesture": gesture,
                "confidence": confidence,
                "smoothed_gesture": smoothed_gesture,
            }
        except Exception as exc:
            print(f"Gesture inference error: {exc}")
            return self._unknown()

    def _predict_probabilities(self, feature):
        x_value = np.expand_dims(feature, axis=0)
        if self.model_kind == "sklearn":
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_value)[0]
            prediction = int(self.model.predict(x_value)[0])
            probabilities = np.zeros(len(self.gesture_classes), dtype=np.float32)
            probabilities[prediction] = 1.0
            return probabilities

        predictions = self.model.predict(x_value, verbose=0)
        return predictions[0]

    @staticmethod
    def _unknown():
        return {
            "gesture": "Unknown",
            "confidence": 0.0,
            "smoothed_gesture": "Unknown",
        }
