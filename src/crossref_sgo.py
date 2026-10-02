"""Cross-reference VRU-Detect condition coverage with optional SGO-Audit data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


CONDITION_COLUMNS = ("timeofday", "weather", "scene")


def resolve_sgo_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path if cli_path.exists() else None
    env_path = os.environ.get("SGO_AUDIT_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    default = Path("data/sgo")
    return default if default.exists() else None


def write_missing_note(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / "sgo_audit_unavailable.md"
    note_path.write_text(
        "\n".join(
            [
                "# SGO-Audit cross-reference",
                "",
                "SGO-Audit data was not available.",
                "",
                "Checked `data/sgo/` and the `SGO_AUDIT_PATH` environment variable. "
                "No comparison was run. This is expected for environments where the optional "
                "audit export has not been provided.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def load_processed_conditions(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "conditions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed conditions file: {path}")
    return pd.read_csv(path)


def _load_json_records(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return pd.DataFrame(value)
        return pd.DataFrame([data])
    return pd.DataFrame()


def load_sgo_records(sgo_path: Path) -> pd.DataFrame:
    files: list[Path]
    if sgo_path.is_file():
        files = [sgo_path]
    else:
        files = sorted(
            [
                path
                for path in sgo_path.rglob("*")
                if path.suffix.lower() in {".csv", ".json"}
            ]
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        if path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path))
        elif path.suffix.lower() == ".json":
            frames.append(_load_json_records(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_sgo_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    lower_to_original = {column.lower(): column for column in normalized.columns}
    rename_map: dict[str, str] = {}
    aliases = {
        "timeofday": ["timeofday", "time_of_day", "lighting", "light", "illumination"],
        "weather": ["weather", "condition_weather"],
        "scene": ["scene", "road_scene", "environment"],
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower_to_original:
                rename_map[lower_to_original[candidate]] = target
                break
    normalized = normalized.rename(columns=rename_map)
    for column in CONDITION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "unknown"
        normalized[column] = normalized[column].fillna("unknown").astype(str)
    return normalized


def ensure_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in CONDITION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "unknown"
        normalized[column] = normalized[column].fillna("unknown").astype(str)
    return normalized


def distribution(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = ensure_condition_columns(df)
    rows: list[dict[str, Any]] = []
    for column in CONDITION_COLUMNS:
        counts = df[column].fillna("unknown").astype(str).value_counts(dropna=False)
        total = int(counts.sum())
        for value, count in counts.items():
            rows.append(
                {
                    "source": source,
                    "condition": column,
                    "value": value,
                    "count": int(count),
                    "share": 0.0 if total == 0 else float(count) / total,
                }
            )
    return pd.DataFrame(rows)


def compare_distributions(processed: pd.DataFrame, sgo: pd.DataFrame) -> pd.DataFrame:
    processed_dist = distribution(processed, "vru_detect")
    sgo_dist = distribution(sgo, "sgo_audit")
    merged = processed_dist.merge(
        sgo_dist,
        on=["condition", "value"],
        how="outer",
        suffixes=("_vru_detect", "_sgo_audit"),
    ).fillna(0)
    merged["share_delta_vru_minus_sgo"] = (
        merged["share_vru_detect"].astype(float) - merged["share_sgo_audit"].astype(float)
    )
    return merged


def write_summary(output_dir: Path, comparison: pd.DataFrame, sgo_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "condition_distribution_comparison.csv", index=False)
    lines = [
        "# SGO-Audit cross-reference",
        "",
        f"Compared processed VRU-Detect condition metadata with SGO-Audit data from `{sgo_path}`.",
        "",
        "Largest absolute distribution deltas:",
        "",
    ]
    top = comparison.copy()
    top["abs_delta"] = top["share_delta_vru_minus_sgo"].abs()
    for row in top.sort_values("abs_delta", ascending=False).head(10).to_dict(orient="records"):
        lines.append(
            f"- {row['condition']}={row['value']}: "
            f"VRU {row['share_vru_detect']:.3f}, "
            f"SGO {row['share_sgo_audit']:.3f}, "
            f"delta {row['share_delta_vru_minus_sgo']:.3f}"
        )
    (output_dir / "sgo_crossref_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sgo-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sgo_crossref"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sgo_path = resolve_sgo_path(args.sgo_path)
    if sgo_path is None:
        note_path = write_missing_note(args.output_dir)
        print(f"SGO-Audit data was not available; wrote {note_path}")
        return

    processed = load_processed_conditions(args.processed_dir)
    sgo = normalize_sgo_columns(load_sgo_records(sgo_path))
    if sgo.empty:
        note_path = write_missing_note(args.output_dir)
        print(f"SGO-Audit path contained no CSV/JSON records; wrote {note_path}")
        return
    comparison = compare_distributions(processed, sgo)
    write_summary(args.output_dir, comparison, sgo_path)
    print(f"Wrote SGO-Audit comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
