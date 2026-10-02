"""Shared detection metrics for VRU-Detect."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


Box = tuple[float, float, float, float]
Detection = dict[str, Any]


def _as_box(box: Iterable[float]) -> Box:
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def box_area(box: Iterable[float]) -> float:
    """Return area for an xyxy box."""
    x1, y1, x2, y2 = _as_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Compute intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = _as_box(box_a)
    bx1, by1, bx2, by2 = _as_box(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area((ax1, ay1, ax2, ay2)) + box_area((bx1, by1, bx2, by2)) - inter
    if union <= 0:
        return 0.0
    return inter / union


def size_bucket(area: float, thresholds: tuple[float, float] | None) -> str:
    """Assign an object area to small/medium/large using tercile thresholds."""
    if thresholds is None:
        return "unknown"
    low, high = thresholds
    if area <= low:
        return "small"
    if area <= high:
        return "medium"
    return "large"


def compute_tercile_thresholds(areas: Iterable[float]) -> tuple[float, float] | None:
    """Compute lower and upper tercile thresholds from positive areas."""
    sorted_areas = sorted(float(area) for area in areas if float(area) > 0)
    if not sorted_areas:
        return None
    n = len(sorted_areas)
    low_idx = min(n - 1, max(0, math.ceil(n / 3) - 1))
    high_idx = min(n - 1, max(0, math.ceil((2 * n) / 3) - 1))
    return sorted_areas[low_idx], sorted_areas[high_idx]


def _class_value(item: Detection) -> Any:
    return item.get("class_id", item.get("class_name", item.get("category")))


def match_predictions(
    predictions: list[Detection],
    ground_truths: list[Detection],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict[str, list[Detection]]:
    """Greedily match predictions to ground truth boxes by descending score.

    Each prediction and ground truth dictionary must include `image_id` and
    `bbox` keys. Class-aware matching uses `class_id`, falling back to
    `class_name` or `category`.
    """
    gt_by_image: dict[str, list[tuple[int, Detection]]] = defaultdict(list)
    for idx, gt in enumerate(ground_truths):
        gt_by_image[str(gt["image_id"])].append((idx, gt))

    used_gt: set[int] = set()
    matches: list[Detection] = []
    false_positives: list[Detection] = []

    sorted_predictions = sorted(
        predictions,
        key=lambda pred: float(pred.get("score", 1.0)),
        reverse=True,
    )
    for pred in sorted_predictions:
        pred_image = str(pred["image_id"])
        pred_class = _class_value(pred)
        best_iou = 0.0
        best_gt_idx: int | None = None
        best_gt: Detection | None = None

        for gt_idx, gt in gt_by_image.get(pred_image, []):
            if gt_idx in used_gt:
                continue
            if class_aware and pred_class != _class_value(gt):
                continue
            overlap = iou(pred["bbox"], gt["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_gt_idx = gt_idx
                best_gt = gt

        if best_gt_idx is not None and best_iou >= iou_threshold and best_gt is not None:
            used_gt.add(best_gt_idx)
            matches.append({"prediction": pred, "ground_truth": best_gt, "iou": best_iou})
        else:
            false_positives.append(pred)

    false_negatives = [
        gt for idx, gt in enumerate(ground_truths) if idx not in used_gt
    ]
    return {
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_precision_recall(
    predictions: list[Detection],
    ground_truths: list[Detection],
    iou_threshold: float = 0.5,
    classes: Iterable[Any] | None = None,
) -> dict[Any, dict[str, float | int]]:
    """Compute per-class and micro-averaged precision/recall."""
    if classes is None:
        classes = sorted(
            {_class_value(item) for item in predictions + ground_truths},
            key=lambda value: str(value),
        )

    results: dict[Any, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    for class_id in classes:
        class_predictions = [
            pred for pred in predictions if _class_value(pred) == class_id
        ]
        class_ground_truths = [
            gt for gt in ground_truths if _class_value(gt) == class_id
        ]
        matched = match_predictions(
            class_predictions,
            class_ground_truths,
            iou_threshold=iou_threshold,
            class_aware=False,
        )
        tp = len(matched["matches"])
        fp = len(matched["false_positives"])
        fn = len(matched["false_negatives"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        results[class_id] = {
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    results["__all__"] = {
        "precision": _safe_divide(total_tp, total_tp + total_fp),
        "recall": _safe_divide(total_tp, total_tp + total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
    return results


def _metadata_value(item: Detection, metadata: dict[str, dict[str, Any]], key: str) -> Any:
    if key in item:
        return item[key]
    image_id = str(item.get("image_id", ""))
    return metadata.get(image_id, {}).get(key, "unknown")


def stratified_eval(
    predictions: list[Detection],
    ground_truths: list[Detection],
    metadata: dict[str, dict[str, Any]],
    strata: Iterable[str] = ("timeofday", "weather", "scene", "size_bucket"),
    iou_threshold: float = 0.5,
    classes: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute precision/recall for metadata strata.

    For image-level strata, predictions and ground truths are filtered by image
    metadata. For object-level `size_bucket`, ground truths are filtered by
    bucket and predictions from the same images are included.
    """
    rows: list[dict[str, Any]] = []
    all_classes = list(classes) if classes is not None else None
    for stratum in strata:
        if stratum == "size_bucket":
            values = sorted(
                {
                    str(gt.get("size_bucket", "unknown"))
                    for gt in ground_truths
                }
            )
        else:
            values = sorted(
                {
                    str(_metadata_value(item, metadata, stratum))
                    for item in ground_truths + predictions
                }
            )

        for value in values:
            if stratum == "size_bucket":
                selected_gts = [
                    gt for gt in ground_truths if str(gt.get("size_bucket", "unknown")) == value
                ]
                selected_images = {str(gt["image_id"]) for gt in selected_gts}
                selected_preds = [
                    pred for pred in predictions if str(pred["image_id"]) in selected_images
                ]
            else:
                selected_gts = [
                    gt
                    for gt in ground_truths
                    if str(_metadata_value(gt, metadata, stratum)) == value
                ]
                selected_preds = [
                    pred
                    for pred in predictions
                    if str(_metadata_value(pred, metadata, stratum)) == value
                ]

            metrics = compute_precision_recall(
                selected_preds,
                selected_gts,
                iou_threshold=iou_threshold,
                classes=all_classes,
            )
            for class_id, class_metrics in metrics.items():
                rows.append(
                    {
                        "stratum": stratum,
                        "value": value,
                        "class_id": class_id,
                        **class_metrics,
                    }
                )
    return rows
