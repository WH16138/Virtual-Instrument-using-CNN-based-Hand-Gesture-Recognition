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

    def __init__(
        self,
        model_path=None,
        smoothing_size=5,
        lower_confidence=0.62,
        lower_margin=0.18,
        high_confidence=0.78,
        high_margin=0.32,
        shot_confidence_bonus=0.06,
        shot_margin_bonus=0.08,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self.model_kind = None
        self.model_loaded = False
        self.gesture_classes = GESTURE_CLASSES
        self.smoothing_size = smoothing_size
        self.smoothing_windows = {}
        self.lower_confidence = lower_confidence
        self.lower_margin = lower_margin
        self.high_confidence = high_confidence
        self.high_margin = high_margin
        self.shot_gestures = {"V_Sign", "Gun_Sign"}
        self.shot_confidence_bonus = shot_confidence_bonus
        self.shot_margin_bonus = shot_margin_bonus
        self.active_streams = {}

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
            probabilities = self._align_probabilities(np.asarray(self._predict_probabilities(feature), dtype=np.float32))
            if probabilities is None:
                print(
                    "Warning: gesture model class count mismatch. "
                    f"Expected {len(self.gesture_classes)} classes."
                )
                return self._unknown()
            class_probabilities = self._class_probability_dict(probabilities)

            gesture_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[gesture_idx])
            second_confidence = self._second_best_probability(probabilities, gesture_idx)
            margin = confidence - second_confidence
            gesture = self.gesture_classes[gesture_idx]
            was_active = self.active_streams.get(stream_id, False)
            required_confidence = self.lower_confidence if was_active else self.high_confidence
            required_margin = self.lower_margin if was_active else self.high_margin
            if gesture in self.shot_gestures:
                required_confidence += self.shot_confidence_bonus
                required_margin += self.shot_margin_bonus

            if confidence < required_confidence or margin < required_margin:
                self._mark_stream_unknown(stream_id)
                return {
                    "gesture": "Unknown",
                    "confidence": confidence,
                    "smoothed_gesture": "Unknown",
                    "margin": margin,
                    "second_confidence": second_confidence,
                    "raw_gesture": gesture,
                    "class_probabilities": class_probabilities,
                }

            smoothing_window = self.smoothing_windows.setdefault(stream_id, deque(maxlen=self.smoothing_size))
            smoothing_window.append(gesture)
            smoothed_gesture = max(set(smoothing_window), key=list(smoothing_window).count)
            self.active_streams[stream_id] = True

            return {
                "gesture": gesture,
                "confidence": confidence,
                "smoothed_gesture": smoothed_gesture,
                "margin": margin,
                "second_confidence": second_confidence,
                "raw_gesture": gesture,
                "class_probabilities": class_probabilities,
            }
        except Exception as exc:
            print(f"Gesture inference error: {exc}")
            return self._unknown()

    def reset(self, stream_id=None):
        if stream_id is None:
            self.smoothing_windows.clear()
            self.active_streams.clear()
            return
        self.smoothing_windows.pop(stream_id, None)
        self.active_streams.pop(stream_id, None)

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

    def _mark_stream_unknown(self, stream_id):
        self.active_streams[stream_id] = False
        self.smoothing_windows.pop(stream_id, None)

    def _align_probabilities(self, probabilities):
        if probabilities.shape[0] == len(self.gesture_classes):
            return probabilities

        legacy_classes = ["Fist", "Open_Palm", "V_Sign", "OK_Sign"]
        if probabilities.shape[0] == len(legacy_classes) and "Gun_Sign" in self.gesture_classes:
            aligned = np.zeros(len(self.gesture_classes), dtype=np.float32)
            for legacy_index, label in enumerate(legacy_classes):
                if label in self.gesture_classes:
                    aligned[self.gesture_classes.index(label)] = probabilities[legacy_index]
            return aligned

        return None

    def _class_probability_dict(self, probabilities):
        return {
            label: float(probabilities[index])
            for index, label in enumerate(self.gesture_classes)
        }

    @staticmethod
    def _second_best_probability(probabilities, best_idx):
        if probabilities.shape[0] <= 1:
            return 0.0
        masked = probabilities.copy()
        masked[best_idx] = -1.0
        return float(np.max(masked))

    @staticmethod
    def _unknown():
        return {
            "gesture": "Unknown",
            "confidence": 0.0,
            "smoothed_gesture": "Unknown",
            "margin": 0.0,
            "second_confidence": 0.0,
            "raw_gesture": "Unknown",
            "class_probabilities": {},
        }
