import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VisionQuest gesture model.")
    parser.add_argument(
        "--mode",
        choices=["landmarks", "cnn"],
        default="landmarks",
        help="Training pipeline to use. landmarks is the current real-time default and does not require TensorFlow.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset directory.")
    parser.add_argument("--output", default=None, help="Override output model path.")
    parser.add_argument("--epochs", type=int, default=None, help="CNN epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="CNN batch size.")
    parser.add_argument("--max-iter", type=int, default=None, help="Landmark MLP max iterations.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "landmarks":
        from models.train_landmarks import LandmarkGestureModelTrainer

        trainer = LandmarkGestureModelTrainer(
            dataset_dir=args.dataset or "dataset_landmarks",
            model_output_path=args.output or "models/gesture_model.pkl",
        )
        trainer.train(max_iter=args.max_iter or 800)
        return

    from models.train_cnn import CnnGestureModelTrainer

    trainer = CnnGestureModelTrainer(
        dataset_dir=args.dataset or "dataset",
        model_output_path=args.output or "models/gesture_model_cnn.keras",
    )
    trainer.train(
        epochs=args.epochs or 50,
        batch_size=args.batch_size or 16,
    )


if __name__ == "__main__":
    main()
