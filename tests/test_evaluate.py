from src.metrics import compute_precision_recall, iou, match_predictions, stratified_eval


def test_iou_computation() -> None:
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175


def test_precision_recall_matching() -> None:
    predictions = [
        {"image_id": "a.jpg", "class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": "a.jpg", "class_id": 0, "bbox": [20, 20, 30, 30], "score": 0.8},
        {"image_id": "a.jpg", "class_id": 1, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    ground_truths = [
        {"image_id": "a.jpg", "class_id": 0, "bbox": [0, 0, 10, 10]},
        {"image_id": "a.jpg", "class_id": 1, "bbox": [40, 40, 50, 50]},
    ]

    matched = match_predictions(predictions, ground_truths, iou_threshold=0.5)
    assert len(matched["matches"]) == 1
    assert len(matched["false_positives"]) == 2
    assert len(matched["false_negatives"]) == 1

    metrics = compute_precision_recall(
        predictions,
        ground_truths,
        iou_threshold=0.5,
        classes=[0, 1],
    )
    assert metrics[0]["precision"] == 0.5
    assert metrics[0]["recall"] == 1.0
    assert metrics[1]["precision"] == 0.0
    assert metrics[1]["recall"] == 0.0
    assert metrics["__all__"]["tp"] == 1
    assert metrics["__all__"]["fp"] == 2
    assert metrics["__all__"]["fn"] == 1


def test_stratified_eval_by_conditions() -> None:
    predictions = [
        {"image_id": "day.jpg", "class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": "night.jpg", "class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
    ]
    ground_truths = [
        {
            "image_id": "day.jpg",
            "class_id": 0,
            "bbox": [0, 0, 10, 10],
            "size_bucket": "small",
        },
        {
            "image_id": "night.jpg",
            "class_id": 0,
            "bbox": [20, 20, 30, 30],
            "size_bucket": "large",
        },
    ]
    metadata = {
        "day.jpg": {"timeofday": "daytime", "weather": "clear", "scene": "city"},
        "night.jpg": {"timeofday": "night", "weather": "clear", "scene": "city"},
    }

    rows = stratified_eval(
        predictions,
        ground_truths,
        metadata,
        strata=["timeofday"],
        classes=[0],
    )
    by_value = {row["value"]: row for row in rows if row["class_id"] == 0}

    assert by_value["daytime"]["recall"] == 1.0
    assert by_value["night"]["recall"] == 0.0
