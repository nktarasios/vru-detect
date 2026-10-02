#!/usr/bin/env python3
"""Build portfolio-facing comparison charts from committed result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SUMMARY = ROOT / "results/baseline/summary.csv"
FINETUNED_SUMMARY = ROOT / "results/finetuned/summary.csv"
BASELINE_STRAT = ROOT / "results/baseline/stratified.csv"
FINETUNED_STRAT = ROOT / "results/finetuned/stratified.csv"
THRESHOLD_SWEEP = ROOT / "results/finetuned/threshold_sweep.csv"
OUT_DIR = ROOT / "results/portfolio"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f8fa",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "text.color": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def load_class_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "class_name" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "class_name"})
    return df[df["class_name"].isin(["person", "rider", "bike", "all"])].copy()


def plot_baseline_vs_finetuned() -> Path:
    baseline = load_class_summary(BASELINE_SUMMARY)
    finetuned = load_class_summary(FINETUNED_SUMMARY)
    classes = ["person", "rider", "bike", "all"]
    x = range(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b_p = [float(baseline.loc[baseline["class_name"] == c, "precision"].iloc[0]) for c in classes]
    b_r = [float(baseline.loc[baseline["class_name"] == c, "recall"].iloc[0]) for c in classes]
    f_p = [float(finetuned.loc[finetuned["class_name"] == c, "precision"].iloc[0]) for c in classes]
    f_r = [float(finetuned.loc[finetuned["class_name"] == c, "recall"].iloc[0]) for c in classes]

    ax.bar([i - 1.5 * width for i in x], b_p, width, label="Baseline precision", color="#7a8799")
    ax.bar([i - 0.5 * width for i in x], b_r, width, label="Baseline recall", color="#b7c0cc")
    ax.bar([i + 0.5 * width for i in x], f_p, width, label="Fine-tuned precision", color="#0b6e4f")
    ax.bar([i + 1.5 * width for i in x], f_r, width, label="Fine-tuned recall", color="#3bb273")

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("VRU-Detect: baseline vs fine-tuned (conf=0.25)")
    ax.legend(ncols=2, loc="upper right")
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    fig.tight_layout()
    out = OUT_DIR / "baseline_vs_finetuned.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _recall_for(df: pd.DataFrame, stratum: str, value: str, class_name: str = "all") -> float:
    rows = df[
        (df["stratum"] == stratum)
        & (df["value"] == value)
        & (df["class_name"] == class_name)
    ]
    if rows.empty:
        return 0.0
    return float(rows["recall"].iloc[0])


def plot_failure_modes() -> Path:
    baseline = pd.read_csv(BASELINE_STRAT)
    finetuned = pd.read_csv(FINETUNED_STRAT)

    buckets = [
        ("Night (all VRU)", "timeofday", "night", "all"),
        ("Night person", "timeofday", "night", "person"),
        ("Small objects", "size_bucket", "small", "all"),
        ("Medium objects", "size_bucket", "medium", "all"),
        ("City street", "scene", "city street", "all"),
    ]
    labels = [item[0] for item in buckets]
    b_vals = [_recall_for(baseline, s, v, c) for _, s, v, c in buckets]
    f_vals = [_recall_for(finetuned, s, v, c) for _, s, v, c in buckets]

    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.barh([i + 0.18 for i in y], b_vals, height=0.34, color="#7a8799", label="Baseline recall")
    ax.barh([i - 0.18 for i in y], f_vals, height=0.34, color="#0b6e4f", label="Fine-tuned recall")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_title("Where the detector still fails: hard-condition recall")
    ax.legend(loc="lower right")
    for i, (b, f) in enumerate(zip(b_vals, f_vals)):
        ax.text(min(0.97, max(b, f) + 0.02), i, f"{b:.3f} → {f:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "failure_modes_recall.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_threshold_tradeoff() -> Path:
    sweep = pd.read_csv(THRESHOLD_SWEEP)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    colors = {"precision": "#0b6e4f", "recall": "#c44900", "f1": "#2f5d8c"}
    recommendations = {"person": 0.05, "rider": 0.05, "bike": 0.15}

    for ax, class_name in zip(axes, ["person", "rider", "bike"]):
        subset = sweep[sweep["class_name"] == class_name].sort_values("threshold")
        for metric in ["precision", "recall", "f1"]:
            ax.plot(subset["threshold"], subset[metric], marker="o", linewidth=2, label=metric, color=colors[metric])
        rec = recommendations[class_name]
        ax.axvline(rec, color="#111111", linestyle="--", linewidth=1.2, label=f"chosen={rec:.2f}")
        ax.set_title(class_name)
        ax.set_xlabel("Confidence threshold")
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-class threshold tradeoff (fine-tuned model)", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "threshold_tradeoff.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def write_findings_markdown() -> Path:
    baseline = pd.read_csv(BASELINE_STRAT)
    finetuned = pd.read_csv(FINETUNED_STRAT)
    b_sum = load_class_summary(BASELINE_SUMMARY)
    f_sum = load_class_summary(FINETUNED_SUMMARY)

    def pr(df: pd.DataFrame, name: str) -> tuple[float, float]:
        row = df.loc[df["class_name"] == name].iloc[0]
        return float(row["precision"]), float(row["recall"])

    lines = [
        "# VRU-Detect Findings",
        "",
        "Generated from committed evaluation CSVs for portfolio review.",
        "",
        "## Headline",
        "",
        "Fine-tuning does not invent a perfect VRU detector. It changes the failure",
        "shape: rider false positives collapse, person recall rises, and the remaining",
        "risk concentrates in night scenes and small distant objects.",
        "",
        "## Class-level change (conf=0.25)",
        "",
        "| class | baseline P / R | fine-tuned P / R | product read |",
        "| --- | --- | --- | --- |",
    ]
    reads = {
        "person": "More pedestrians caught; precision dips slightly — acceptable if misses are costlier.",
        "rider": "Precision jumps; COCO person/bike confusion was the baseline pathology.",
        "bike": "Recall roughly doubles but stays low — still an open product risk.",
        "all": "Aggregate gains hide residual night/small-object weakness.",
    }
    for name in ["person", "rider", "bike", "all"]:
        bp, br = pr(b_sum, name)
        fp, fr = pr(f_sum, name)
        lines.append(
            f"| {name} | {bp:.3f} / {br:.3f} | {fp:.3f} / {fr:.3f} | {reads[name]} |"
        )

    lines.extend(
        [
            "",
            "## Hard-condition recall",
            "",
            "| condition | baseline recall | fine-tuned recall | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    conditions = [
        ("Night (all)", "timeofday", "night", "all"),
        ("Night person", "timeofday", "night", "person"),
        ("Small objects", "size_bucket", "small", "all"),
        ("Medium objects", "size_bucket", "medium", "all"),
        ("City street", "scene", "city street", "all"),
    ]
    for label, stratum, value, class_name in conditions:
        b = _recall_for(baseline, stratum, value, class_name)
        f = _recall_for(finetuned, stratum, value, class_name)
        lines.append(f"| {label} | {b:.3f} | {f:.3f} | {f - b:+.3f} |")

    lines.extend(
        [
            "",
            "## Product implications",
            "",
            "1. **Do not ship on aggregate mAP alone.** Night person recall and small-object",
            "   recall remain the safety-relevant bottlenecks after fine-tuning.",
            "2. **Thresholds are product decisions.** Person/rider use recall-first",
            "   operating points (`0.05`); bike uses best-F1 (`0.15`) because false bike",
            "   triggers were noisier relative to the recall gain.",
            "3. **The portfolio claim is methodological.** This repo shows how to expose",
            "   condition-specific risk and choose an operating point deliberately — not",
            "   that a CPU-trained nano model is deployment-ready.",
            "",
            "## Charts",
            "",
            "- `results/portfolio/baseline_vs_finetuned.png`",
            "- `results/portfolio/failure_modes_recall.png`",
            "- `results/portfolio/threshold_tradeoff.png`",
            "",
        ]
    )
    out = OUT_DIR / "FINDINGS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_baseline_vs_finetuned(),
        plot_failure_modes(),
        plot_threshold_tradeoff(),
        write_findings_markdown(),
    ]
    for path in outputs:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
