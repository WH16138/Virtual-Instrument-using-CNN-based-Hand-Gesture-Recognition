import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.gesture_features import GESTURE_CLASSES


class CnnGestureModelTrainer:
    """Train a CNN classifier from legacy PNG hand-crop images."""

    def __init__(self, dataset_dir="dataset", model_output_path="models/gesture_model_cnn.keras"):
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

            for path in sorted(class_dir.glob("*.png")):
                image = cv2.imread(str(path))
                if image is None:
                    print(f"Skipping unreadable image: {path}")
                    continue
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255.0
                x_values.append(image)
                y_values.append(self.class_to_idx[gesture_class])

        if not x_values:
            return np.empty((0, 64, 64, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.stack(x_values).astype(np.float32), np.asarray(y_values, dtype=np.int64)

    def build_model(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(64, 64, 3)),
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.gesture_classes), activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, epochs=50, batch_size=16, test_size=0.2):
        x_values, y_values = self.load_dataset()
        if len(x_values) == 0:
            raise RuntimeError("No PNG samples found. Run vision/dataset_capture_cnn.py first.")

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

        augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.08),
                keras.layers.RandomZoom(0.08),
            ]
        )

        model = self.build_model()
        train_model = keras.Sequential([augmentation, model])
        train_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ]
        train_model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        predictions = np.argmax(train_model.predict(x_test, verbose=0), axis=1)
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
        train_model.save(self.model_output_path)
        print(f"Saved CNN model: {self.model_output_path}")
        return train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the legacy PNG/CNN gesture classifier.")
    parser.add_argument("--dataset", default="dataset", help="PNG dataset directory.")
    parser.add_argument("--output", default="models/gesture_model_cnn.keras", help="Output Keras model path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = CnnGestureModelTrainer(args.dataset, args.output)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
