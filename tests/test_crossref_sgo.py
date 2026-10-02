"""Tests for the public NHTSA SGO lighting cross-reference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.crossref_sgo import (
    alignment_decision,
    build_crossref,
    classify_party,
    map_lighting,
    read_sgo_csv,
    submission_sort_key,
    write_outputs,
)


def _report(**overrides: str) -> dict[str, str]:
    row = {
        "Report ID": "1",
        "Report Version": "1",
        "Report Type": "1-Day",
        "Report Submission Date": "JAN-2024",
        "Same Incident ID": "",
        "Crash With": "Passenger Car",
        "Lighting": "Daylight",
        "Roadway Type": "Street",
        "Narrative": "",
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stratified(recalls: dict[tuple[str, str], float]) -> pd.DataFrame:
    rows = []
    for (bucket, class_name), recall in recalls.items():
        rows.append(
            {
                "stratum": "timeofday",
                "value": bucket,
                "class_name": class_name,
                "recall": recall,
                "tp": 1,
                "fn": 1,
            }
        )
    return pd.DataFrame(rows)


def _recall_grid(night_all: float = 0.2, day_all: float = 0.8) -> pd.DataFrame:
    rows = []
    for bucket, all_recall, person, rider, bike in (
        ("daytime", day_all, 0.70, 0.60, 0.40),
        ("dawn/dusk", 0.50, 0.55, 0.40, 0.30),
        ("night", night_all, 0.10, 0.15, 0.05),
    ):
        for class_name, recall in (
            ("all", all_recall),
            ("person", person),
            ("rider", rider),
            ("bike", bike),
        ):
            rows.append(
                {
                    "stratum": "timeofday",
                    "value": bucket,
                    "class_name": class_name,
                    "recall": recall,
                    "tp": 2,
                    "fn": 8 if bucket == "night" else 2,
                }
            )
    return pd.DataFrame(rows)


def test_read_sgo_csv_replaces_a_non_utf8_byte(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_bytes(b"Report ID,City\n1,Coeur d\xe2??Alene\n")
    frame = read_sgo_csv(path)
    assert frame.loc[0, "Report ID"] == "1"
    assert "Coeur" in frame.loc[0, "City"]


def test_submission_sort_key_is_chronological() -> None:
    assert submission_sort_key("DEC-2024") > submission_sort_key("JUN-2024")
    assert submission_sort_key("JAN-2025") > submission_sort_key("DEC-2024")
    assert submission_sort_key("JUN-2025") > submission_sort_key("MAY-2025")
    assert submission_sort_key("not-a-date") == -1


def test_party_and_lighting_codes() -> None:
    assert classify_party("Non-Motorist: Pedestrian") == "pedestrian"
    assert classify_party("  non-motorist:   cyclist ") == "cyclist"
    assert classify_party("Non-Motorist: Other") is None
    assert classify_party("Passenger Car") is None
    assert map_lighting("Daylight") == "daytime"
    assert map_lighting("Dawn / Dusk") == "dawn/dusk"
    assert map_lighting("Dark - Lighted") == "night"
    assert map_lighting("Dark - Not Lighted") == "night"
    assert map_lighting("Dark - Unknown Lighting") == "night"
    assert map_lighting("Unknown") == "unknown"
    assert map_lighting("") == "unknown"
    assert map_lighting("Other, see Narrative") == "other"


def test_alignment_rule_requires_unique_mode_and_minimum() -> None:
    lined_up = alignment_decision({"daytime": 1, "dawn/dusk": 0, "night": 4}, {"daytime": 0.8, "dawn/dusk": 0.5, "night": 0.2})
    assert lined_up["lines_up"] is True
    assert lined_up["mode"] == ["night"]

    missed = alignment_decision({"daytime": 5, "dawn/dusk": 1, "night": 2}, {"daytime": 0.8, "dawn/dusk": 0.5, "night": 0.2})
    assert missed["lines_up"] is False
    assert missed["mode"] == ["daytime"]
    assert missed["lowest_recall"] == ["night"]

    tie = alignment_decision({"daytime": 3, "dawn/dusk": 0, "night": 3}, {"daytime": 0.8, "dawn/dusk": 0.5, "night": 0.2})
    assert tie["lines_up"] is False


def test_reduction_keeps_latest_code_and_collapses_duplicate_filings(tmp_path: Path) -> None:
    archive = {
        "ADS": _frame(
            [
                _report(
                    **{
                        "Report ID": "A-1",
                        "Report Version": "1",
                        "Crash With": "Unknown",
                        "Lighting": "Unknown",
                        "Narrative": "no pedestrians involved",
                    }
                ),
                _report(
                    **{
                        "Report ID": "A-1",
                        "Report Version": "2",
                        "Report Submission Date": "MAR-2024",
                        "Crash With": "Non-Motorist: Pedestrian",
                        "Lighting": "Dark - Lighted",
                        "Roadway Type": "Intersection",
                        "Same Incident ID": "same-1",
                    }
                ),
                _report(
                    **{
                        "Report ID": "B-9",
                        "Report Version": "2",
                        "Report Submission Date": "JUN-2024",
                        "Crash With": "Non-Motorist: Pedestrian",
                        "Lighting": "Dark - Lighted",
                        "Roadway Type": "Intersection",
                        "Same Incident ID": "same-1",
                    }
                ),
                _report(
                    **{
                        "Report ID": "C-1",
                        "Crash With": "Non-Motorist: Cyclist",
                        "Lighting": "Daylight",
                        "Roadway Type": "Street",
                    }
                ),
                _report(
                    **{
                        "Report ID": "C-2",
                        "Crash With": "Non-Motorist: Cyclist",
                        "Lighting": "Twilight",
                        "Roadway Type": "Street",
                    }
                ),
                _report(
                    **{
                        "Report ID": "D-1",
                        "Report Type": "No New or Updated Incident Reports",
                        "Crash With": "Non-Motorist: Pedestrian",
                        "Lighting": "Daylight",
                    }
                ),
                _report(
                    **{
                        "Report ID": "E-1",
                        "Crash With": "Passenger Car",
                        "Lighting": "Night-unlit-invented",
                        "Narrative": "The vehicle struck a pedestrian in the crosswalk.",
                    }
                ),
                _report(
                    **{
                        "Report ID": "F-1",
                        "Crash With": "Non-Motorist: Other",
                        "Lighting": "Dark - Not Lighted",
                    }
                ),
            ]
        )
    }
    current = {
        "ADAS": _frame(
            [
                _report(
                    **{
                        "Report ID": "Z-1",
                        "Crash With": "Non-Motorist: Cyclist",
                        "Roadway Type": "Street",
                    }
                )
            ]
        ).drop(columns=["Lighting"])
    }
    result = build_crossref(
        archive,
        current,
        _recall_grid(),
        _recall_grid(night_all=0.15, day_all=0.75),
        retrieved_on="2026-10-02",
        archive_files=[{"system": "ADS", "url": "https://example.test/ads.csv", "bytes": 10, "sha256": "abc", "last_modified": ""}],
        current_files=[{"system": "ADAS", "url": "https://example.test/adas.csv", "bytes": 4, "sha256": "def", "last_modified": ""}],
    )

    parties = result.incidents["party"].tolist()
    assert parties.count("pedestrian") == 1
    assert parties.count("cyclist") == 2
    pedestrian = result.incidents[result.incidents["party"] == "pedestrian"].iloc[0]
    assert pedestrian["report_id"] == "B-9"
    assert pedestrian["lighting_bucket"] == "night"
    assert result.query["unmapped_lighting"] == ["Twilight"]
    assert result.query["current"]["lighting_included"] is False
    assert result.query["current"]["vru_incidents"] == 1
    assert result.query["archive"]["vru_codings_dropped_by_collapse"] == 0
    assert result.query["alignment"]["combined"]["lines_up"] is False
    assert result.query["alignment"]["pedestrian"]["lines_up"] is True

    text = result.finding_md
    assert "## Query" in text
    assert "Non-Motorist: Pedestrian" in text
    assert "Same Incident ID" in text
    assert "alignment rule is unmet" in text
    assert "alignment rule is met" in text
    assert "`abc`" in text
    assert "1 pedestrian, 2 cyclist" in text

    write_outputs(result, tmp_path)
    assert (tmp_path / "finding.md").exists()
    assert (tmp_path / "query.json").exists()
    assert (tmp_path / "lighting_comparison.csv").exists()
    assert (tmp_path / "roadway_counts.csv").exists()
    assert not (tmp_path / "sgo_audit_unavailable.md").exists()


def test_dark_codes_stay_distinct_inside_the_night_bucket() -> None:
    archive = {
        "ADS": _frame(
            [
                _report(
                    **{
                        "Report ID": "N-1",
                        "Crash With": "Non-Motorist: Pedestrian",
                        "Lighting": "Dark - Not Lighted",
                        "Roadway Type": "Rural Road",
                    }
                ),
                _report(
                    **{
                        "Report ID": "N-2",
                        "Crash With": "Non-Motorist: Cyclist",
                        "Lighting": "Dark - Lighted",
                        "Roadway Type": "Street",
                    }
                ),
            ]
        )
    }
    result = build_crossref(
        archive,
        {},
        _stratified(
            {
                ("daytime", "all"): 0.9,
                ("dawn/dusk", "all"): 0.5,
                ("night", "all"): 0.1,
                ("daytime", "person"): 0.9,
                ("dawn/dusk", "person"): 0.5,
                ("night", "person"): 0.1,
                ("daytime", "rider"): 0.9,
                ("dawn/dusk", "rider"): 0.5,
                ("night", "rider"): 0.2,
                ("daytime", "bike"): 0.4,
                ("dawn/dusk", "bike"): 0.3,
                ("night", "bike"): 0.05,
            }
        ),
        _stratified(
            {
                ("daytime", "all"): 0.4,
                ("dawn/dusk", "all"): 0.3,
                ("night", "all"): 0.2,
                ("daytime", "person"): 0.4,
                ("dawn/dusk", "person"): 0.3,
                ("night", "person"): 0.2,
                ("daytime", "rider"): 0.4,
                ("dawn/dusk", "rider"): 0.3,
                ("night", "rider"): 0.2,
                ("daytime", "bike"): 0.4,
                ("dawn/dusk", "bike"): 0.3,
                ("night", "bike"): 0.2,
            }
        ),
        retrieved_on="2026-10-02",
        archive_files=[],
        current_files=[],
    )
    night = result.lighting_comparison[result.lighting_comparison["lighting_bucket"] == "night"].iloc[0]
    assert int(night["sgo_incidents"]) == 2
    raw = {row.lighting: int(row.incidents) for row in result.lighting_raw.itertuples(index=False)}
    assert raw["Dark - Not Lighted"] == 1
    assert raw["Dark - Lighted"] == 1
    assert result.query["alignment"]["combined"]["lines_up"] is True
