import json
from pathlib import Path

from src.dataset import box_to_yolo, convert_frame, extract_conditions
from src.metrics import compute_tercile_thresholds, size_bucket


def test_box_to_yolo_conversion_math() -> None:
    x_center, y_center, width, height = box_to_yolo(
        {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
        image_width=100,
        image_height=100,
    )

    assert x_center == 0.2
    assert y_center == 0.4
    assert width == 0.2
    assert height == 0.4


def test_size_bucketing_terciles() -> None:
    thresholds = compute_tercile_thresholds([10, 20, 30, 40, 50, 60])

    assert thresholds == (20, 40)
    assert size_bucket(10, thresholds) == "small"
    assert size_bucket(30, thresholds) == "medium"
    assert size_bucket(60, thresholds) == "large"


def test_conditions_extraction() -> None:
    frame = {
        "attributes": {
            "weather": "rainy",
            "timeofday": "night",
            "scene": "highway",
        }
    }

    assert extract_conditions(frame) == {
        "weather": "rainy",
        "timeofday": "night",
        "scene": "highway",
    }


def test_convert_frame_filters_to_vru_classes() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "bdd_tiny.json"
    frame = json.loads(fixture_path.read_text(encoding="utf-8"))[0]

    converted = convert_frame(
        frame,
        image_size=(100, 80),
        class_mapping={"person": 0, "rider": 1, "bike": 2},
        split="train",
        thresholds=(100, 900),
    )

    assert converted["image_id"] == "synthetic_0001.jpg"
    assert converted["labels"] == ["0 0.200000 0.500000 0.200000 0.500000"]
    assert len(converted["ground_truths"]) == 1
    assert converted["ground_truths"][0]["class_name"] == "person"
    assert converted["ground_truths"][0]["size_bucket"] == "medium"
