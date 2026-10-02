"""Run VRU-Detect inference on a folder and save annotated images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "person": (64, 180, 75),
    "rider": (255, 170, 0),
    "bike": (40, 130, 255),
}

DEFAULT_THRESHOLD_CANDIDATES = (
    Path("results/finetuned/threshold_recommendations.csv"),
    Path("results/finetuned/thresholds.json"),
    Path("results/threshold_recommendations.csv"),
)


def auto_device(requested: str | None = None) -> str:
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = str(requested).strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def load_class_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {int(key): str(value) for key, value in config["class_names"].items()}


def discover_threshold_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    for candidate in DEFAULT_THRESHOLD_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _apply_threshold(thresholds: dict[int, float], class_names: dict[int, str], key: Any, value: Any) -> None:
    name_to_id = {name: class_id for class_id, name in class_names.items()}
    key_str = str(key)
    if key_str.isdigit():
        class_id = int(key_str)
    else:
        class_id = name_to_id.get(key_str)
    if class_id in thresholds:
        thresholds[class_id] = float(value)


def load_thresholds(path: Path | None, default: float, class_names: dict[int, str]) -> dict[int, float]:
    thresholds = {class_id: default for class_id in class_names}
    if path is None:
        return thresholds
    if not path.exists():
        raise FileNotFoundError(f"Threshold file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = row.get("class_id") or row.get("class_name")
                value = row.get("threshold")
                if key is not None and value not in (None, ""):
                    _apply_threshold(thresholds, class_names, key, value)
        return thresholds

    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key, value in data.items():
            _apply_threshold(thresholds, class_names, key, value)
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                key = row.get("class_id") or row.get("class_name")
                value = row.get("threshold")
                if key is not None and value not in (None, ""):
                    _apply_threshold(thresholds, class_names, key, value)
    return thresholds


def find_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def draw_detection(
    image: Any,
    bbox: list[float],
    label: str,
    score: float,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {score:.2f}"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_y = max(y1, text_size[1] + baseline + 4)
    cv2.rectangle(
        image,
        (x1, text_y - text_size[1] - baseline - 4),
        (x1 + text_size[0] + 4, text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 2, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_demo(
    weights: str,
    image_dir: Path,
    output_dir: Path,
    class_names: dict[int, str],
    thresholds: dict[int, float],
    device: str | None = None,
) -> int:
    from ultralytics import YOLO

    images = find_images(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    resolved_device = auto_device(device)
    min_threshold = min(thresholds.values()) if thresholds else 0.25
    written = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        results = model.predict(
            source=str(image_path),
            conf=min_threshold,
            device=resolved_device,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                if class_id not in class_names:
                    continue
                score = float(box.conf.item())
                if score < thresholds.get(class_id, min_threshold):
                    continue
                class_name = class_names[class_id]
                color = COLORS.get(class_name, (255, 255, 255))
                draw_detection(
                    image,
                    [float(value) for value in box.xyxy[0].tolist()],
                    class_name,
                    score,
                    color,
                )
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), image)
        written += 1
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--thresholds-json", type=Path, default=None, help="JSON or CSV threshold file")
    parser.add_argument("--thresholds-file", type=Path, default=None, help="JSON or CSV threshold file")
    parser.add_argument("--device", default=None, help="Use 'auto' to select CUDA when available")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    class_names = load_class_names(args.classes)
    threshold_path = discover_threshold_path(args.thresholds_file or args.thresholds_json)
    thresholds = load_thresholds(threshold_path, args.threshold, class_names)
    written = run_demo(
        weights=args.weights,
        image_dir=args.images,
        output_dir=args.output_dir,
        class_names=class_names,
        thresholds=thresholds,
        device=args.device,
    )
    print(f"Wrote {written} annotated images to {args.output_dir}")


if __name__ == "__main__":
    main()
