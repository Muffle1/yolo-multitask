import argparse
import time
from pathlib import Path
import json
from typing import Any

from ultralytics import YOLO

from prepare_dataset.augmentation import Augmentation
from prepare_dataset.config import AugmentationConfig


def load_json_config(config_path: Path | str) -> AugmentationConfig:
	"""Load augmentation config JSON and return AugmentationConfig instance."""
	p = Path(config_path)
	if not p.exists():
		raise FileNotFoundError(f"Augmentation config not found: {p}")
	with open(p, "r", encoding="utf-8") as f:
		data: dict[str, Any] = json.load(f)

	return AugmentationConfig(
		horizontal_flip=data.get("horizontal_flip", True),
		vertical_flip=data.get("vertical_flip", False),
		rotation_range=tuple(data.get("rotation_range", (-15, 15))),
		zoom_range=data.get("zoom_range", 0.1),
		brightness_range=tuple(data.get("brightness_range", (0.8, 1.2))),
		contrast_range=tuple(data.get("contrast_range", (0.8, 1.2))),
		saturation_range=tuple(data.get("saturation_range", (0.8, 1.2))),
		hue_range=tuple(data.get("hue_range", (0.8, 1.2))),
		amount=data.get("amount", 1),
		label_format=data.get("label_format"),
	)


def train_model(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    print("Starting training:")
    print(f"  dataset = {args.data_dir}")
    print(f"  weights = {args.model}")
    model.train(
        data=str(args.data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )


def infer_model(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/infer")
    result = model.predict(
        source=str(args.source),
        save=True,
        project=str(output_dir),
        name=args.name,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf_thres,
        max_det=args.max_det,
    )
    print(f"Inference completed. Predictions saved under {output_dir / args.name}")
    if result:
        print(result)


def profile_model(args: argparse.Namespace) -> None:
    model = YOLO(args.weights)
    source = str(args.source)
    print("Profiling inference performance...")
    start_time = time.perf_counter()
    model.predict(
        source=source,
        save=False,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf_thres,
        max_det=args.max_det,
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Profile complete: elapsed={elapsed:.3f}s for source={source}")


def augment_dataset(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if getattr(args, "config", None) else Path("prepare_dataset/config.json")
    config = load_json_config(config_path)
    
    if getattr(args, "amount", None) is not None:
        config.amount = args.amount

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    image_extensions = [".jpg", ".jpeg", ".png"]
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions])
    print(f"Found {len(image_paths)} images to augment in {images_dir}")

    augmenter = Augmentation(config, output_dir)
    augmenter.augment(image_paths, labels_dir)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO multitask utility: train, infer, profile, augment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument("--data-dir", required=True, help="Path to dataset config or root dataset folder")
    train_parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights or model name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    train_parser.add_argument("--device", default="0", help="Device for training, e.g. 0 or cpu")
    train_parser.add_argument("--project", default="runs/train", help="Save directory for training results")
    train_parser.add_argument("--name", default="exp", help="Experiment name")

    infer_parser = subparsers.add_parser("infer", help="Run inference with a YOLO model")
    infer_parser.add_argument("--weights", required=True, help="Path to trained weights")
    infer_parser.add_argument("--source", required=True, help="Source image, folder, or video for inference")
    infer_parser.add_argument("--output-dir", default="runs/infer", help="Output folder for inference results")
    infer_parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    infer_parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    infer_parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image")
    infer_parser.add_argument("--device", default="0", help="Device for inference")
    infer_parser.add_argument("--name", default="exp", help="Inference run name")

    profile_parser = subparsers.add_parser("profile", help="Profile model inference speed")
    profile_parser.add_argument("--weights", required=True, help="Path to trained weights")
    profile_parser.add_argument("--source", required=True, help="Source image, folder, or video to profile")
    profile_parser.add_argument("--imgsz", type=int, default=640, help="Image size for profiling")
    profile_parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    profile_parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per image")
    profile_parser.add_argument("--device", default="0", help="Device for profiling")

    augment_parser = subparsers.add_parser("augment", help="Augment dataset images and labels")
    augment_parser.add_argument("--images-dir", required=True, help="Directory with input images")
    augment_parser.add_argument("--labels-dir", required=True, help="Directory with input label files")
    augment_parser.add_argument("--output-dir", default="augmented", help="Directory for augmented dataset")
    augment_parser.add_argument("--config", required=True, help="JSON config file for augmentation")
    augment_parser.add_argument("--amount", type=int, default=None, help="Number of augmented copies per image")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "infer":
        infer_model(args)
    elif args.command == "profile":
        profile_model(args)
    elif args.command == "augment":
        augment_dataset(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
