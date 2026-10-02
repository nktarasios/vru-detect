#!/usr/bin/env python3
"""Build a miss-concentration / residual-risk report from stratified CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/baseline/stratified.csv"
FINETUNED = ROOT / "results/finetuned/stratified.csv"
SUMMARY_B = ROOT / "results/baseline/summary.csv"
SUMMARY_F = ROOT / "results/finetuned/summary.csv"
OUT = ROOT / "results/portfolio/ERROR_REPORT.md"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fn"] = df["fn"].astype(int)
    df["tp"] = df["tp"].astype(int)
    df["fp"] = df["fp"].astype(int)
    df["recall"] = df["recall"].astype(float)
    df["precision"] = df["precision"].astype(float)
    return df


def top_miss_buckets(df: pd.DataFrame, class_name: str = "all", n: int = 8) -> pd.DataFrame:
    subset = df[(df["class_name"] == class_name) & (df["stratum"] != "size_bucket")].copy()
    # Prefer buckets with meaningful support
    subset["support"] = subset["tp"] + subset["fn"]
    subset = subset[subset["support"] >= 10]
    subset = subset.sort_values(["fn", "recall"], ascending=[False, True])
    return subset.head(n)


def size_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["stratum"] == "size_bucket") & (df["class_name"] == "all")].sort_values("value")


def class_summary() -> pd.DataFrame:
    b = pd.read_csv(SUMMARY_B)
    f = pd.read_csv(SUMMARY_F)
    rows = []
    for name in ["person", "rider", "bike", "all"]:
        br = b.loc[b["class_name"] == name].iloc[0]
        fr = f.loc[f["class_name"] == name].iloc[0]
        rows.append(
            {
                "class": name,
                "baseline_fn": int(br["fn"]),
                "finetuned_fn": int(fr["fn"]),
                "fn_delta": int(fr["fn"]) - int(br["fn"]),
                "baseline_recall": float(br["recall"]),
                "finetuned_recall": float(fr["recall"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    baseline = _load(BASELINE)
    finetuned = _load(FINETUNED)
    classes = class_summary()
    top_b = top_miss_buckets(baseline)
    top_f = top_miss_buckets(finetuned)
    size_b = size_table(baseline)
    size_f = size_table(finetuned)

    lines = [
        "# Error / residual-risk report",
        "",
        "This report concentrates on **false negatives (misses)** — the failure mode",
        "that matters most for VRU risk analysis.",
        "",
        "## Class-level misses (conf=0.25)",
        "",
        "| class | baseline FN | fine-tuned FN | FN delta | baseline recall | fine-tuned recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in classes.to_dict(orient="records"):
        lines.append(
            f"| {row['class']} | {row['baseline_fn']} | {row['finetuned_fn']} | {row['fn_delta']:+d} | "
            f"{row['baseline_recall']:.3f} | {row['finetuned_recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Where misses still concentrate after fine-tuning",
            "",
            "Highest-FN condition buckets (excluding size; support ≥ 10):",
            "",
            "| stratum | value | FN | recall | support |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_f.itertuples(index=False):
        support = int(row.tp + row.fn)
        lines.append(
            f"| {row.stratum} | {row.value} | {row.fn} | {row.recall:.3f} | {support} |"
        )

    lines.extend(
        [
            "",
            "## Size is the sharpest remaining knife-edge",
            "",
            "| size | baseline recall | fine-tuned recall | baseline FN | fine-tuned FN |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for value in ["small", "medium", "large"]:
        b = size_b.loc[size_b["value"] == value].iloc[0]
        f = size_f.loc[size_f["value"] == value].iloc[0]
        lines.append(
            f"| {value} | {float(b.recall):.3f} | {float(f.recall):.3f} | {int(b.fn)} | {int(f.fn)} |"
        )

    # Night person specifically
    def _get(df: pd.DataFrame, stratum: str, value: str, class_name: str) -> pd.Series:
        return df[
            (df["stratum"] == stratum)
            & (df["value"] == value)
            & (df["class_name"] == class_name)
        ].iloc[0]

    night_b = _get(baseline, "timeofday", "night", "person")
    night_f = _get(finetuned, "timeofday", "night", "person")
    lines.extend(
        [
            "",
            "## Spotlight: night pedestrians",
            "",
            f"- Baseline night-person recall: **{float(night_b.recall):.3f}** "
            f"(FN={int(night_b.fn)})",
            f"- Fine-tuned night-person recall: **{float(night_f.recall):.3f}** "
            f"(FN={int(night_f.fn)})",
            f"- Absolute gain: **{float(night_f.recall) - float(night_b.recall):+.3f}**",
            "",
            "Interpretation: fine-tuning helps, but night pedestrians remain a first-class",
            "product risk. Any claim about this model should lead with this residual gap, not hide",
            "it behind aggregate precision.",
            "",
            "## Baseline miss geography (for contrast)",
            "",
            "| stratum | value | FN | recall | support |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_b.itertuples(index=False):
        support = int(row.tp + row.fn)
        lines.append(
            f"| {row.stratum} | {row.value} | {row.fn} | {row.recall:.3f} | {support} |"
        )

    lines.extend(
        [
            "",
            "## Next experiments this report implies",
            "",
            "1. Night-heavy resampling or night-only fine-tune pass",
            "2. Higher resolution / SAHI-style tiling for small objects",
            "3. Rider/bike co-occurrence aware augmentation",
            "4. Keep publishing stratified FN tables — do not regress to mAP-only reporting",
            "",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    try:
        shown = OUT.relative_to(ROOT)
    except ValueError:
        shown = OUT
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
