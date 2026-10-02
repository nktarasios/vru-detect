#!/usr/bin/env python3
"""Download a BDD100K VRU subset from the Hugging Face validation mirror.

Official ETH mirror (dl.cv.ethz.ch) is preferred when reachable. This script is
the automated fallback used when that host is unavailable.

Source dataset: https://huggingface.co/datasets/dgural/bdd100k
Original data license: BDD100K (educational / research / not-for-profit).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "dgural/bdd100k"

# FiftyOne / HF labels -> BDD-style categories used by this repo
LABEL_MAP = {
    "pedestrian": "person",
    "other person": "person",
    "person": "person",
    "rider": "rider",
    "bicycle": "bike",
    "bike": "bike",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motor",
    "motor": "motor",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
    "other vehicle": "other vehicle",
    "trailer": "trailer",
}
VRU = {"person", "rider", "bike"}


def to_bdd_frame(sample: dict) -> dict:
    name = Path(sample["filepath"]).name
    weather = sample.get("weather", {})
    timeofday = sample.get("timeofday", {})
    scene = sample.get("scene", {})
    width = int(sample.get("metadata", {}).get("width", 1280))
    height = int(sample.get("metadata", {}).get("height", 720))
    labels = []
    for i, det in enumerate((sample.get("detections") or {}).get("detections") or []):
        raw_cat = det.get("label")
        cat = LABEL_MAP.get(raw_cat, raw_cat)
        bb = det.get("bounding_box")
        if not bb or len(bb) != 4:
            continue
        x, y, bw, bh = bb
        labels.append(
            {
                "id": i,
                "category": cat,
                "attributes": {
                    "occluded": bool(det.get("occluded", False)),
                    "truncated": bool(det.get("truncated", False)),
                    "trafficLightColor": str(det.get("trafficLightColor", "none")),
                },
                "box2d": {
                    "x1": float(x) * width,
                    "y1": float(y) * height,
                    "x2": float(x + bw) * width,
                    "y2": float(y + bh) * height,
                },
            }
        )
    return {
        "name": name,
        "width": width,
        "height": height,
        "attributes": {
            "weather": weather.get("label", "undefined") if isinstance(weather, dict) else "undefined",
            "timeofday": timeofday.get("label", "undefined")
            if isinstance(timeofday, dict)
            else "undefined",
            "scene": scene.get("label", "undefined") if isinstance(scene, dict) else "undefined",
        },
        "labels": labels,
        "_filepath": sample["filepath"],
        "_has_vru": any(label["category"] in VRU for label in labels),
    }


def strip_private(frame: dict) -> dict:
    return {key: value for key, value in frame.items() if not key.startswith("_")}


def download_images(frames: list[dict], dest_root: Path, max_workers: int = 16) -> None:
    def _one(frame: dict) -> None:
        dest = dest_root / frame["name"]
        if dest.exists() and dest.stat().st_size > 0:
            return
        local = hf_hub_download(REPO, frame["_filepath"], repo_type="dataset")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, frame) for frame in frames]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 100 == 0:
                print(f"  downloaded {i}/{len(frames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/bdd100k"))
    parser.add_argument("--max-vru", type=int, default=1200)
    parser.add_argument("--max-bg", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    raw = args.raw_dir
    img_train = raw / "images" / "100k" / "train"
    img_val = raw / "images" / "100k" / "val"
    labels_dir = raw / "labels" / "det_20"
    for path in (img_train, img_val, labels_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("Downloading samples.json...")
    samples_path = hf_hub_download(REPO, "samples.json", repo_type="dataset")
    with open(samples_path, encoding="utf-8") as handle:
        samples = json.load(handle)["samples"]

    frames = [to_bdd_frame(sample) for sample in samples]
    vru_frames = [frame for frame in frames if frame["_has_vru"]]
    non_vru = [frame for frame in frames if not frame["_has_vru"]]
    print(f"VRU frames available: {len(vru_frames)} / {len(frames)}")

    random.seed(args.seed)
    random.shuffle(vru_frames)
    random.shuffle(non_vru)
    vru_frames = vru_frames[: args.max_vru]
    non_vru = non_vru[: args.max_bg]

    n_vru_val = max(100, int(0.2 * len(vru_frames)))
    n_bg_val = max(30, int(0.2 * len(non_vru)))
    train_frames = vru_frames[:-n_vru_val] + non_vru[:-n_bg_val]
    val_frames = vru_frames[-n_vru_val:] + non_vru[-n_bg_val:]
    random.shuffle(train_frames)
    random.shuffle(val_frames)

    with open(labels_dir / "det_train.json", "w", encoding="utf-8") as handle:
        json.dump([strip_private(frame) for frame in train_frames], handle)
    with open(labels_dir / "det_val.json", "w", encoding="utf-8") as handle:
        json.dump([strip_private(frame) for frame in val_frames], handle)

    print(f"Downloading train={len(train_frames)} val={len(val_frames)} images...")
    download_images(train_frames, img_train, max_workers=args.workers)
    download_images(val_frames, img_val, max_workers=args.workers)

    (raw / "SOURCE.txt").write_text(
        "Hugging Face mirror: dgural/bdd100k (BDD100K validation split).\n"
        "Label aliases: pedestrian->person, bicycle->bike, motorcycle->motor.\n"
        "Local train/val split created from the validation mirror for pipeline runs\n"
        "when the official ETH host is unreachable.\n"
        "License: BDD100K educational/research/not-for-profit terms.\n",
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    main()
