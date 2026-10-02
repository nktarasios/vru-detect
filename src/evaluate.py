"""Evaluate fine-tuned VRU-Detect weights with threshold recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.baseline_eval import (
    build_finetuned_source_to_targets,
    class_names_by_id,
    evaluate_predictions,
    find_processed_images,
    load_class_config,
    load_conditions,
    load_ground_truths,
    run_yolo_predictions,
    save_metric_outputs,
)
from src.metrics import compute_precision_recall


DEFAULT_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]


def parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return DEFAULT_THRESHOLDS
    if isinstance(value, list):
        return sorted({float(item) for item in value})
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required")
    for threshold in thresholds:
        if threshold < 0 or threshold > 1:
            raise argparse.ArgumentTypeError("Thresholds must be between 0 and 1")
    return thresholds


def filter_predictions(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [pred for pred in predictions if float(pred.get("score", 1.0)) >= threshold]


def threshold_sweep(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    class_names: dict[int, str],
    thresholds: list[float],
    iou_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    classes = sorted(class_names)
    for threshold in thresholds:
        metrics = compute_precision_recall(
            filter_predictions(predictions, threshold),
            ground_truths,
            iou_threshold=iou_threshold,
            classes=classes,
        )
        for class_id, values in metrics.items():
            precision = float(values["precision"])
            recall = float(values["recall"])
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            rows.append(
                {
                    "threshold": threshold,
                    "class_id": class_id,
                    "class_name": "all" if class_id == "__all__" else class_names.get(int(class_id), str(class_id)),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": values["tp"],
                    "fp": values["fp"],
                    "fn": values["fn"],
                }
            )
    return pd.DataFrame(rows)


def recommend_thresholds(sweep_df: pd.DataFrame) -> pd.DataFrame:
    recommendations: list[dict[str, Any]] = []
    if sweep_df.empty or "class_name" not in sweep_df.columns:
        return pd.DataFrame(recommendations)
    for class_name in sorted(name for name in sweep_df["class_name"].unique() if name != "all"):
        class_df = sweep_df[sweep_df["class_name"] == class_name].copy()
        if class_df.empty:
            continue
        if class_name in {"person", "rider"}:
            max_recall = class_df["recall"].max()
            candidates = class_df[class_df["recall"] >= max_recall * 0.95]
            selected = candidates.sort_values(
                ["recall", "precision", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            rationale = (
                "Selected from thresholds within 95% of max recall, then highest precision; "
                "missed person/rider VRUs are prioritized over false positives."
            )
        else:
            selected = class_df.sort_values(
                ["f1", "recall", "precision"],
                ascending=[False, False, False],
            ).iloc[0]
            rationale = "Selected by best F1 with recall and precision as tie-breakers."
        recommendations.append(
            {
                "class_id": selected["class_id"],
                "class_name": class_name,
                "threshold": selected["threshold"],
                "precision": selected["precision"],
                "recall": selected["recall"],
                "f1": selected["f1"],
                "rationale": rationale,
            }
        )
    return pd.DataFrame(recommendations)


def write_threshold_justification(path: Path, recommendations: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Threshold justification",
        "",
        "Fine-tuned thresholds are selected per class from the validation sweep.",
        "For `person` and `rider`, recall is prioritized because missed VRUs are a higher-risk error than false positives.",
        "",
    ]
    if recommendations.empty:
        lines.append("No recommendations could be produced because the sweep was empty.")
    else:
        for row in recommendations.to_dict(orient="records"):
            lines.extend(
                [
                    f"## {row['class_name']}",
                    "",
                    f"- Recommended threshold: `{row['threshold']:.2f}`",
                    f"- Precision: `{row['precision']:.3f}`",
                    f"- Recall: `{row['recall']:.3f}`",
                    f"- F1: `{row['f1']:.3f}`",
                    f"- Rationale: {row['rationale']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_threshold_chart(path: Path, sweep_df: pd.DataFrame) -> None:
    """Plot precision/recall against confidence threshold for each VRU class."""
    if sweep_df.empty or "class_name" not in sweep_df.columns:
        return
    plot_df = sweep_df[sweep_df["class_name"] != "all"].copy()
    if plot_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for class_name in sorted(plot_df["class_name"].unique()):
        class_df = plot_df[plot_df["class_name"] == class_name].sort_values("threshold")
        plt.plot(class_df["threshold"], class_df["recall"], marker="o", label=f"{class_name} recall")
        plt.plot(class_df["threshold"], class_df["precision"], linestyle="--", marker="x", label=f"{class_name} precision")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Validation threshold sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_baseline_comparison(output_dir: Path, finetuned_summary: pd.DataFrame, baseline_path: Path) -> None:
    if not baseline_path.exists():
        return
    baseline = pd.read_csv(baseline_path)
    if baseline.empty or finetuned_summary.empty:
        return
    finetuned = finetuned_summary.copy()
    if "class_id" not in baseline.columns or "class_id" not in finetuned.columns:
        return
    baseline["class_id_key"] = baseline["class_id"].astype(str)
    finetuned["class_id_key"] = finetuned["class_id"].astype(str)
    comparison = baseline.merge(
        finetuned,
        on="class_id_key",
        suffixes=("_baseline", "_finetuned"),
    )
    if comparison.empty:
        return
    comparison = comparison[
        [
            "class_id_key",
            "class_name_baseline",
            "precision_baseline",
            "recall_baseline",
            "precision_finetuned",
            "recall_finetuned",
            "tp_baseline",
            "fp_baseline",
            "fn_baseline",
            "tp_finetuned",
            "fp_finetuned",
            "fn_finetuned",
        ]
    ].rename(columns={"class_id_key": "class_id", "class_name_baseline": "class_name"})
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    markdown_lines = [
        "| " + " | ".join(comparison.columns) + " |",
        "| " + " | ".join("---" for _ in comparison.columns) + " |",
    ]
    for row in comparison.astype(str).to_dict(orient="records"):
        markdown_lines.append("| " + " | ".join(row[column] for column in comparison.columns) + " |")
    (output_dir / "baseline_comparison.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="Use 'auto' to select CUDA when available")
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/finetuned"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("results/baseline/summary.csv"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    class_config = load_class_config(args.classes)
    class_names = class_names_by_id(class_config)
    images = find_processed_images(args.processed_dir, split="val", max_images=args.max_images)
    metadata = load_conditions(args.processed_dir)
    ground_truths = load_ground_truths(args.processed_dir, images, split="val")
    sweep_thresholds = sorted({float(args.conf), *args.thresholds})
    predictions = run_yolo_predictions(
        args.weights,
        images,
        source_to_targets=build_finetuned_source_to_targets(class_config),
        conf=min(sweep_thresholds),
        device=args.device,
    )

    summary_rows, stratified_rows = evaluate_predictions(
        filter_predictions(predictions, args.conf),
        ground_truths,
        metadata,
        class_names,
        iou_threshold=args.iou,
    )
    save_metric_outputs(
        args.output_dir,
        summary_rows,
        stratified_rows,
        title="Fine-tuned YOLO stratified VRU metrics",
    )

    sweep_df = threshold_sweep(predictions, ground_truths, class_names, sweep_thresholds, args.iou)
    recommendations = recommend_thresholds(sweep_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(args.output_dir / "threshold_sweep.csv", index=False)
    recommendations.to_csv(args.output_dir / "threshold_recommendations.csv", index=False)
    write_threshold_chart(args.output_dir / "threshold_sweep_chart.png", sweep_df)
    write_threshold_justification(args.output_dir / "threshold_justification.md", recommendations)
    write_baseline_comparison(args.output_dir, pd.DataFrame(summary_rows), args.baseline_summary)
    print(f"Wrote fine-tuned metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
