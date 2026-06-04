import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.gesture_features import FEATURE_SIZE, GESTURE_CLASSES


class LandmarkGestureModelTrainer:
    """Train a lightweight sklearn MLP classifier from normalized landmark vectors."""

    def __init__(self, dataset_dir="dataset_landmarks", model_output_path="models/gesture_model.pkl"):
        self.dataset_dir = Path(dataset_dir)
        self.model_output_path = Path(model_output_path)
        self.gesture_classes = GESTURE_CLASSES
        self.class_to_idx = {gesture: idx for idx, gesture in enumerate(self.gesture_classes)}

    def load_dataset(self):
        x_values = []
        y_values = []

        for gesture_class in self.gesture_classes:
            class_dir = self.dataset_dir / gesture_class
            if not class_dir.exists():
                print(f"Warning: missing dataset directory: {class_dir}")
                continue

            for path in sorted(class_dir.glob("*.npy")):
                feature = np.load(path).astype(np.float32)
                if feature.shape != (FEATURE_SIZE,):
                    print(f"Skipping invalid feature shape {feature.shape}: {path}")
                    continue
                x_values.append(feature)
                y_values.append(self.class_to_idx[gesture_class])

        if not x_values:
            return np.empty((0, FEATURE_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.stack(x_values).astype(np.float32), np.asarray(y_values, dtype=np.int64)

    def build_model(self, max_iter=800):
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=42,
            ),
        )

    def train(self, max_iter=800, test_size=0.2):
        x_values, y_values = self.load_dataset()
        if len(x_values) == 0:
            raise RuntimeError(
                "No .npy landmark feature samples found. Run vision/dataset_capture.py first."
            )

        print("Dataset counts:")
        for idx, gesture_class in enumerate(self.gesture_classes):
            print(f"  {gesture_class}: {int(np.sum(y_values == idx))}")

        class_counts = np.bincount(y_values, minlength=len(self.gesture_classes))
        stratify = y_values if min(class_counts) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        model = self.build_model(max_iter=max_iter)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=list(range(len(self.gesture_classes))),
                target_names=self.gesture_classes,
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_output_path.open("wb") as handle:
            pickle.dump(model, handle)
        print(f"Saved landmark model: {self.model_output_path}")
        return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the landmark-vector gesture classifier.")
    parser.add_argument("--dataset", default="dataset_landmarks", help="Landmark dataset directory.")
    parser.add_argument("--output", default="models/gesture_model.pkl", help="Output pickle model path.")
    parser.add_argument("--max-iter", type=int, default=800)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = LandmarkGestureModelTrainer(args.dataset, args.output)
    trainer.train(max_iter=args.max_iter)


if __name__ == "__main__":
    main()
