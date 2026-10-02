"""Evaluate a COCO-pretrained YOLO baseline on processed VRU data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from PIL import Image

from src.metrics import (
    compute_precision_recall,
    compute_tercile_thresholds,
    size_bucket,
    stratified_eval,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def auto_device(requested: str | None = None) -> str:
    """Select a safe Ultralytics device string for local CPU/cloud runs."""
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = str(requested).strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def load_class_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def class_names_by_id(class_config: dict[str, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in class_config["class_names"].items()}


def load_conditions(processed_dir: Path) -> dict[str, dict[str, str]]:
    path = processed_dir / "conditions.csv"
    if not path.exists():
        return {}
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            metadata[image_id] = dict(row)
            metadata[Path(image_id).stem] = dict(row)
    return metadata


def find_processed_images(processed_dir: Path, split: str = "val", max_images: int | None = None) -> list[Path]:
    image_dir = processed_dir / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing processed image directory: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images is not None:
        images = images[:max_images]
    return images


def _read_yolo_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    records: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_id, x_center, y_center, width, height = stripped.split()[:5]
        records.append(
            (
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
        )
    return records


def _label_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return [
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    ]


def _areas_for_split(processed_dir: Path, split: str) -> list[float]:
    areas: list[float] = []
    image_dir = processed_dir / "images" / split
    label_dir = processed_dir / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return areas
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_yolo_label_file(label_path)
        if not labels:
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        for _, _, _, width, height in labels:
            areas.append(width * image_width * height * image_height)
    return areas


def load_ground_truths(
    processed_dir: Path,
    images: list[Path],
    split: str = "val",
    thresholds: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = compute_tercile_thresholds(_areas_for_split(processed_dir, "train"))
        if thresholds is None:
            thresholds = compute_tercile_thresholds(_areas_for_split(processed_dir, split))

    ground_truths: list[dict[str, Any]] = []
    label_dir = processed_dir / "labels" / split
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_yolo_label_file(label_path)
        if not labels:
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        for class_id, x_center, y_center, width, height in labels:
            bbox = _label_to_xyxy(x_center, y_center, width, height, image_width, image_height)
            area = width * image_width * height * image_height
            ground_truths.append(
                {
                    "image_id": image_path.name,
                    "class_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "size_bucket": size_bucket(area, thresholds),
                }
            )
    return ground_truths


def build_baseline_source_to_targets(class_config: dict[str, Any]) -> dict[int, list[int]]:
    class_id_by_name = {name: class_id for class_id, name in class_names_by_id(class_config).items()}
    source_to_targets: dict[int, list[int]] = {}
    for class_name, source_ids in class_config["baseline_coco_accept"].items():
        target_id = class_id_by_name[class_name]
        for source_id in source_ids:
            source_to_targets.setdefault(int(source_id), []).append(target_id)
    return source_to_targets


def build_finetuned_source_to_targets(class_config: dict[str, Any]) -> dict[int, list[int]]:
    return {class_id: [class_id] for class_id in class_names_by_id(class_config)}


def run_yolo_predictions(
    weights: str,
    images: list[Path],
    source_to_targets: dict[int, list[int]],
    conf: float = 0.25,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run Ultralytics YOLO and map model class IDs to project class IDs."""
    from ultralytics import YOLO

    model = YOLO(weights)
    resolved_device = auto_device(device)
    predictions: list[dict[str, Any]] = []
    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=conf,
            device=resolved_device,
            verbose=False,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None:
            continue
        for box in boxes:
            source_class = int(box.cls.item())
            if source_class not in source_to_targets:
                continue
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            score = float(box.conf.item())
            for target_class in source_to_targets[source_class]:
                predictions.append(
                    {
                        "image_id": image_path.name,
                        "class_id": int(target_class),
                        "bbox": xyxy,
                        "score": score,
                        "source_class_id": source_class,
                    }
                )
    return predictions


def metrics_to_rows(
    metrics: dict[Any, dict[str, float | int]],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, values in metrics.items():
        rows.append(
            {
                "class_id": class_id,
                "class_name": "all" if class_id == "__all__" else class_names.get(int(class_id), str(class_id)),
                **values,
            }
        )
    return rows


def save_metric_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    stratified_rows: list[dict[str, Any]],
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = ["class_id", "class_name", "precision", "recall", "tp", "fp", "fn"]
    stratified_columns = ["stratum", "value", *summary_columns]
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    stratified_df = pd.DataFrame(stratified_rows, columns=stratified_columns)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    stratified_df.to_csv(output_dir / "stratified.csv", index=False)
    write_metric_chart(stratified_df, output_dir / "stratified_chart.png", title)


def write_empty_chart(output_path: Path, title: str, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_metric_chart(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        write_empty_chart(output_path, title, "No stratified metrics available")
        return
    plot_df = df[df["class_id"] == "__all__"].copy()
    if plot_df.empty:
        write_empty_chart(output_path, title, "No aggregate stratified metrics available")
        return
    plot_df["label"] = plot_df["stratum"].astype(str) + "=" + plot_df["value"].astype(str)
    plot_df = plot_df.head(30)
    x_positions = range(len(plot_df))
    plt.figure(figsize=(max(10, len(plot_df) * 0.45), 5))
    plt.bar([x - 0.2 for x in x_positions], plot_df["precision"], width=0.4, label="precision")
    plt.bar([x + 0.2 for x in x_positions], plot_df["recall"], width=0.4, label="recall")
    plt.xticks(list(x_positions), plot_df["label"], rotation=70, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    class_names: dict[int, str],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes = sorted(class_names)
    summary = compute_precision_recall(
        predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        classes=classes,
    )
    summary_rows = metrics_to_rows(summary, class_names)
    stratified_rows = stratified_eval(
        predictions,
        ground_truths,
        metadata,
        iou_threshold=iou_threshold,
        classes=classes,
    )
    for row in stratified_rows:
        class_id = row["class_id"]
        row["class_name"] = "all" if class_id == "__all__" else class_names.get(int(class_id), str(class_id))
    return summary_rows, stratified_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="Use 'auto' to select CUDA when available")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    class_config = load_class_config(args.classes)
    class_names = class_names_by_id(class_config)
    images = find_processed_images(args.processed_dir, split="val", max_images=args.max_images)
    metadata = load_conditions(args.processed_dir)
    ground_truths = load_ground_truths(args.processed_dir, images, split="val")
    predictions = run_yolo_predictions(
        args.weights,
        images,
        source_to_targets=build_baseline_source_to_targets(class_config),
        conf=args.conf,
        device=args.device,
    )
    summary_rows, stratified_rows = evaluate_predictions(
        predictions,
        ground_truths,
        metadata,
        class_names,
        iou_threshold=args.iou,
    )
    save_metric_outputs(
        args.output_dir,
        summary_rows,
        stratified_rows,
        title="COCO-pretrained YOLO baseline stratified VRU metrics",
    )
    print(f"Wrote baseline metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
