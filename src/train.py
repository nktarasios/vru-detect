"""Fine-tune YOLO on the processed VRU-Detect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def auto_device(requested: str | None) -> str:
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = str(requested).strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def list_images(image_dir: Path, subset: int | None = None) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if subset is not None and subset < 0:
        raise ValueError("subset must be non-negative")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if subset is not None:
        images = images[:subset]
    return images


def has_vru_label(image_path: Path, processed_dir: Path, split: str) -> bool:
    label_path = processed_dir / "labels" / split / f"{image_path.stem}.txt"
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def write_image_list(path: Path, images: list[Path]) -> None:
    """Write absolute image paths without following symlinks.

    Ultralytics derives label paths by swapping ``images`` → ``labels`` in the
    image path. Resolving symlinks into ``data/raw/...`` would break that mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image in images:
            handle.write(f"{image.absolute()}\n")


def build_training_dataset_yaml(
    processed_dir: Path,
    class_config: dict[str, Any],
    work_dir: Path,
    train_subset: int | None,
    val_subset: int | None,
    oversample_factor: int,
) -> Path:
    """Create an Ultralytics dataset YAML with an oversampled train list."""
    train_images = list_images(processed_dir / "images" / "train", subset=train_subset)
    val_images = list_images(processed_dir / "images" / "val", subset=val_subset)

    oversampled_train: list[Path] = []
    for image in train_images:
        repeats = oversample_factor if has_vru_label(image, processed_dir, "train") else 1
        oversampled_train.extend([image] * max(1, repeats))

    train_list = work_dir / "train_oversampled.txt"
    val_list = work_dir / "val.txt"
    write_image_list(train_list, oversampled_train)
    write_image_list(val_list, val_images)

    class_names = {int(key): str(value) for key, value in class_config["class_names"].items()}
    dataset_yaml = {
        "path": str(processed_dir.resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "names": class_names,
    }
    dataset_yaml_path = work_dir / "dataset_oversampled.yaml"
    with dataset_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)
    return dataset_yaml_path


def train_model(
    config_path: Path,
    processed_dir: Path,
    classes_path: Path,
    model_override: str | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    subset_override: int | None = None,
) -> Any:
    """Run Ultralytics training and return the training result object."""
    from ultralytics import YOLO

    config = load_yaml(config_path)
    class_config = load_yaml(classes_path)

    project = Path(config.get("project", "models/runs"))
    name = str(config.get("name", "vru_finetune"))
    work_dir = project / "_lists" / name
    subset_config = config.get("subset") or {}
    train_subset = subset_override if subset_override is not None else subset_config.get("train")
    val_subset = subset_override if subset_override is not None else subset_config.get("val")
    resolved_device = auto_device(device_override if device_override is not None else config.get("device"))
    batch = int(config.get("batch", 4 if resolved_device == "cpu" else 8))
    imgsz = int(config.get("imgsz", 640))
    dataset_yaml_path = build_training_dataset_yaml(
        processed_dir=processed_dir,
        class_config=class_config,
        work_dir=work_dir,
        train_subset=train_subset,
        val_subset=val_subset,
        oversample_factor=max(1, int(config.get("vru_oversample_factor", 1))),
    )

    model = YOLO(model_override or config.get("model", "yolo26n.pt"))
    result = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs_override if epochs_override is not None else int(config.get("epochs", 15)),
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        patience=int(config.get("patience", 10)),
        workers=int(config.get("workers", 2)),
        project=str(project.resolve()),
        name=name,
        exist_ok=True,
    )
    # Promote best checkpoint to a stable path for Phase 3/5 scripts.
    import shutil
    best = Path(str(result.save_dir)) / "weights" / "best.pt" if hasattr(result, "save_dir") else None
    if best is None:
        # Fallback: newest best.pt under project
        candidates = sorted(project.rglob("weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        best = candidates[0] if candidates else None
    if best and best.exists():
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, models_dir / "vru_best.pt")
        print(f"Copied best weights to {models_dir / 'vru_best.pt'}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, help="Use 'auto' to select CUDA when available")
    parser.add_argument("--subset", type=int, default=None, help="Limit train and val image counts")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_model(
        config_path=args.config,
        processed_dir=args.processed_dir,
        classes_path=args.classes,
        model_override=args.model,
        epochs_override=args.epochs,
        device_override=args.device,
        subset_override=args.subset,
    )


if __name__ == "__main__":
    main()
