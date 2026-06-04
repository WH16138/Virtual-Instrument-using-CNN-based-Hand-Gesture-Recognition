import datetime
from pathlib import Path

import numpy as np

from vision.gesture_features import GESTURE_CLASSES, landmarks_to_feature_vector


class DatasetCollector:
    """Save normalized landmark feature vectors for gesture training."""

    def __init__(self, dataset_dir="dataset_landmarks"):
        self.dataset_dir = Path(dataset_dir)
        self.gesture_classes = GESTURE_CLASSES
        for gesture_class in self.gesture_classes:
            (self.dataset_dir / gesture_class).mkdir(parents=True, exist_ok=True)

    def save_gesture_sample(self, gesture_label, hand_landmarks):
        if gesture_label not in self.gesture_classes:
            return False

        feature = landmarks_to_feature_vector(hand_landmarks)
        if feature is None:
            return False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.dataset_dir / gesture_label / f"{gesture_label}_{timestamp}.npy"
        np.save(output_path, feature)
        return True

    def save_gesture_image(self, gesture_label, hand_landmarks):
        """Backward-compatible alias for older callers."""
        return self.save_gesture_sample(gesture_label, hand_landmarks)

    def get_dataset_stats(self):
        return {
            gesture_class: len(list((self.dataset_dir / gesture_class).glob("*.npy")))
            for gesture_class in self.gesture_classes
        }
