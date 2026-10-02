"""Convert BDD100K Detection 2020 annotations into a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from src.metrics import compute_tercile_thresholds, size_bucket as metric_size_bucket


DEFAULT_CLASSES_PATH = Path("configs/classes.yaml")
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _box_values(box: dict[str, float] | list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    if isinstance(box, dict):
        return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    if len(box) != 4:
        raise ValueError("Box sequences must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def box_to_yolo(
    box: dict[str, float] | list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert an xyxy box to normalized YOLO xywh coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    x1, y1, x2, y2 = _box_values(box)
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box after clipping: {(x1, y1, x2, y2)}")

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def assign_size_bucket(area: float, thresholds: tuple[float, float] | None) -> str:
    """Assign a VRU box area to a size bucket."""
    return metric_size_bucket(area, thresholds)


def extract_conditions(frame: dict[str, Any]) -> dict[str, str]:
    """Extract BDD100K condition metadata from a frame."""
    attributes = frame.get("attributes") or {}
    return {
        "weather": str(attributes.get("weather", "unknown") or "unknown"),
        "timeofday": str(attributes.get("timeofday", "unknown") or "unknown"),
        "scene": str(attributes.get("scene", "unknown") or "unknown"),
    }


def _xyxy_clipped(
    box: dict[str, float] | list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _box_values(box)
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box after clipping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def convert_frame(
    frame: dict[str, Any],
    image_size: tuple[int, int],
    class_mapping: dict[str, int],
    split: str = "",
    thresholds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Convert one BDD100K frame to YOLO label lines and GT records."""
    image_width, image_height = image_size
    image_id = str(frame.get("name") or frame.get("image_id") or frame.get("id"))
    label_lines: list[str] = []
    ground_truths: list[dict[str, Any]] = []
    areas: list[float] = []

    for label in frame.get("labels", []) or []:
        category = label.get("category")
        box = label.get("box2d")
        if category not in class_mapping or not box:
            continue
        try:
            x1, y1, x2, y2 = _xyxy_clipped(box, image_width, image_height)
            x_center, y_center, width, height = box_to_yolo(box, image_width, image_height)
        except (KeyError, TypeError, ValueError):
            continue

        class_id = int(class_mapping[category])
        area = (x2 - x1) * (y2 - y1)
        bucket = assign_size_bucket(area, thresholds)
        areas.append(area)
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        ground_truths.append(
            {
                "image_id": image_id,
                "split": split,
                "class_id": class_id,
                "class_name": category,
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "size_bucket": bucket,
            }
        )

    return {
        "image_id": image_id,
        "labels": label_lines,
        "ground_truths": ground_truths,
        "areas": areas,
        "conditions": extract_conditions(frame),
    }


def load_class_config(path: Path = DEFAULT_CLASSES_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_label_path(raw_dir: Path, split: str) -> Path:
    candidates = [
        raw_dir / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
        raw_dir / "labels" / "det_20" / f"det_{split}.json",
        raw_dir / "bdd100k" / "labels" / f"det_{split}.json",
        raw_dir / "labels" / f"det_{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find det_{split}.json under supported label layouts in {raw_dir}"
    )


def image_roots(raw_dir: Path, split: str) -> list[Path]:
    return [
        raw_dir / "bdd100k" / "images" / "100k" / split,
        raw_dir / "images" / "100k" / split,
        raw_dir / "bdd100k" / "images" / split,
        raw_dir / "images" / split,
    ]


def find_image_path(raw_dir: Path, split: str, image_name: str) -> Path:
    image_path = Path(image_name)
    names = [image_path.name]
    if image_path.suffix == "":
        names.extend(f"{image_path.name}{ext}" for ext in DEFAULT_IMAGE_EXTENSIONS)

    for root in image_roots(raw_dir, split):
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find image {image_name!r} for split {split}")


def image_size_from_frame_or_file(frame: dict[str, Any], image_path: Path) -> tuple[int, int]:
    width = frame.get("width") or frame.get("image_width")
    height = frame.get("height") or frame.get("image_height")
    if width and height:
        return int(width), int(height)
    with Image.open(image_path) as image:
        return image.size


def link_or_copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        relative_source = os.path.relpath(source.resolve(), start=destination.parent.resolve())
        destination.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, destination)


def _write_dataset_yaml(out_dir: Path, class_names: dict[int, str]) -> None:
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)


def _load_frames(label_path: Path, max_images: int | None) -> list[dict[str, Any]]:
    with label_path.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)
    if not isinstance(frames, list):
        raise ValueError(f"Expected {label_path} to contain a list of frames")
    if max_images is not None:
        frames = frames[:max_images]
    return frames


def _image_level_bucket(ground_truths: list[dict[str, Any]]) -> str:
    if not ground_truths:
        return "none"
    order = {"unknown": -1, "small": 0, "medium": 1, "large": 2}
    return max(
        (str(gt.get("size_bucket", "unknown")) for gt in ground_truths),
        key=lambda bucket: order.get(bucket, -1),
    )


def convert_dataset(
    raw_dir: Path,
    out_dir: Path,
    splits: list[str],
    max_images: int | None = None,
    classes_path: Path = DEFAULT_CLASSES_PATH,
) -> dict[str, Any]:
    """Convert requested splits and return summary metadata."""
    class_config = load_class_config(classes_path)
    class_mapping = {str(key): int(value) for key, value in class_config["bdd_to_yolo"].items()}
    class_names = {int(key): str(value) for key, value in class_config["class_names"].items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    raw_converted: dict[str, list[dict[str, Any]]] = {}
    train_areas: list[float] = []

    for split in splits:
        label_path = find_label_path(raw_dir, split)
        frames = _load_frames(label_path, max_images)
        converted_split: list[dict[str, Any]] = []
        for frame in tqdm(frames, desc=f"Reading {split}"):
            image_name = str(frame.get("name") or frame.get("image_id") or frame.get("id"))
            image_path = find_image_path(raw_dir, split, image_name)
            image_size = image_size_from_frame_or_file(frame, image_path)
            converted = convert_frame(
                frame,
                image_size=image_size,
                class_mapping=class_mapping,
                split=split,
                thresholds=None,
            )
            converted["image_id"] = image_path.name
            converted["image_path"] = image_path
            converted["image_size"] = image_size
            converted_split.append(converted)
            if split == "train":
                train_areas.extend(converted["areas"])
        raw_converted[split] = converted_split

    thresholds = compute_tercile_thresholds(train_areas)
    if thresholds is None:
        all_areas = [
            area
            for converted_split in raw_converted.values()
            for converted in converted_split
            for area in converted["areas"]
        ]
        thresholds = compute_tercile_thresholds(all_areas)

    conditions_rows: list[dict[str, str]] = []
    total_images = 0
    total_boxes = 0

    for split, converted_split in raw_converted.items():
        for converted in tqdm(converted_split, desc=f"Writing {split}"):
            frame_ground_truths: list[dict[str, Any]] = []
            label_lines: list[str] = []
            for gt in converted["ground_truths"]:
                bucket = assign_size_bucket(float(gt["area"]), thresholds)
                gt["size_bucket"] = bucket
                class_id = int(gt["class_id"])
                x1, y1, x2, y2 = gt["bbox"]
                width, height = converted["image_size"]
                x_center, y_center, box_width, box_height = box_to_yolo(
                    (x1, y1, x2, y2),
                    width,
                    height,
                )
                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )
                frame_ground_truths.append(gt)

            image_id = str(converted["image_id"])
            label_path = out_dir / "labels" / split / f"{Path(image_id).stem}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(label_lines))
                if label_lines:
                    handle.write("\n")

            image_destination = out_dir / "images" / split / Path(image_id).name
            link_or_copy_image(Path(converted["image_path"]), image_destination)

            conditions = converted["conditions"]
            conditions_rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "weather": conditions["weather"],
                    "timeofday": conditions["timeofday"],
                    "scene": conditions["scene"],
                    "size_bucket": _image_level_bucket(frame_ground_truths),
                }
            )
            total_images += 1
            total_boxes += len(label_lines)

    with (out_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image_id", "split", "weather", "timeofday", "scene", "size_bucket"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(conditions_rows)

    _write_dataset_yaml(out_dir, class_names)

    return {
        "images": total_images,
        "boxes": total_boxes,
        "splits": splits,
        "size_thresholds": thresholds,
        "out_dir": str(out_dir),
    }


def parse_splits(value: str) -> list[str]:
    splits = [split.strip() for split in value.split(",") if split.strip()]
    invalid = sorted(set(splits) - {"train", "val"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported split(s): {', '.join(invalid)}")
    return splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--splits", type=parse_splits, default=["train", "val"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = convert_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        splits=args.splits,
        max_images=args.max_images,
        classes_path=args.classes,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
