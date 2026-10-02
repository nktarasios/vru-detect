"""Cross-reference VRU-Detect lighting recall with the public NHTSA SGO archive.

The check reads NHTSA's Standing General Order 2021-01 incident CSVs directly.
It does not depend on another repository. The lighting comparison uses the
pre-third-amendment archive (through 15 June 2025), which is the public file
set that contains a Lighting field. Third-amendment files are counted and kept
out of the lighting table when that field is absent.

This is a research comparison of published crash codes and a detector's
stratified recall. It is not a safety certification and not a claim about any
manufacturer's perception system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


USER_AGENT = "vru-detect-sgo-crossref/1.0 (research; public NHTSA SGO archive)"
NHTSA_PAGE = "https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting"
ARCHIVE_PREFIX = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/"
CURRENT_PREFIX = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/"

ARCHIVE_SOURCES: tuple[tuple[str, str], ...] = (
    ("ADS", ARCHIVE_PREFIX + "SGO-2021-01_Incident_Reports_ADS.csv"),
    ("ADAS", ARCHIVE_PREFIX + "SGO-2021-01_Incident_Reports_ADAS.csv"),
    ("OTHER", ARCHIVE_PREFIX + "SGO-2021-01_Incident_Reports_OTHER.csv"),
)
CURRENT_SOURCES: tuple[tuple[str, str], ...] = (
    ("ADS", CURRENT_PREFIX + "SGO-2021-01_Incident_Reports_ADS.csv"),
    ("ADAS", CURRENT_PREFIX + "SGO-2021-01_Incident_Reports_ADAS.csv"),
    ("OTHER", CURRENT_PREFIX + "SGO-2021-01_Incident_Reports_OTHER.csv"),
)

NO_NEW_REPORT = "No New or Updated Incident Reports"
KNOWN_LIGHTING_BUCKETS = ("daytime", "dawn/dusk", "night")
BUCKET_ORDER = ("daytime", "dawn/dusk", "night", "unknown", "other")
RECALL_CLASSES = ("all", "person", "rider", "bike")

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

LIGHTING_TO_BUCKET = {
    "daylight": "daytime",
    "dawn / dusk": "dawn/dusk",
    "dark - lighted": "night",
    "dark - not lighted": "night",
    "dark - unknown lighting": "night",
    "unknown": "unknown",
    "": "unknown",
    "other, see narrative": "other",
}

_PEDESTRIAN = re.compile(r"^non-motorist:\s*pedestrian$")
_CYCLIST = re.compile(r"^non-motorist:\s*cyclist$")

REQUIRED_COLUMNS = (
    "Report ID",
    "Report Version",
    "Report Type",
    "Same Incident ID",
    "Crash With",
    "Roadway Type",
)


@dataclass
class CrossrefResult:
    incidents: pd.DataFrame
    lighting_comparison: pd.DataFrame
    lighting_raw: pd.DataFrame
    roadway: pd.DataFrame
    query: dict[str, Any]
    finding_md: str


def submission_sort_key(value: str) -> int:
    """Map a MON-YYYY submission stamp to a chronological integer.

    Lexical order of month abbreviations is not chronological (DEC sorts before
    JAN). Unparsed values sort first so a parsed stamp wins a tie.
    """
    text = " ".join(str(value or "").strip().upper().split())
    parts = text.split("-")
    if len(parts) == 2 and parts[0] in MONTHS and parts[1].isdigit():
        return int(parts[1]) * 12 + MONTHS[parts[0]]
    return -1


def normalize_label(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def classify_party(crash_with: str) -> str | None:
    """Return pedestrian or cyclist for the published Crash With code.

    Matching is exact after whitespace and case normalization. Narrative text
    is intentionally ignored: phrases such as "no pedestrians involved" are
    not crashes with a pedestrian.
    """
    text = normalize_label(crash_with)
    if _PEDESTRIAN.match(text):
        return "pedestrian"
    if _CYCLIST.match(text):
        return "cyclist"
    return None


def map_lighting(value: str) -> str:
    """Map an SGO Lighting code onto a detector time-of-day bucket."""
    text = normalize_label(value)
    if text in LIGHTING_TO_BUCKET:
        return LIGHTING_TO_BUCKET[text]
    return "other"


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _as_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def prepare_reports(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Attach the sort keys used to keep one row per report and per incident."""
    _require_columns(df, REQUIRED_COLUMNS, system)
    frame = _as_text_frame(df)
    if "Report Submission Date" not in frame.columns:
        frame["Report Submission Date"] = ""
    if "Lighting" not in frame.columns:
        frame["Lighting"] = ""
    frame["system"] = system
    frame["_version"] = pd.to_numeric(frame["Report Version"], errors="coerce").fillna(0).astype(int)
    frame["_submitted"] = frame["Report Submission Date"].map(submission_sort_key)
    return frame


def latest_version(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest Report Version for each Report ID."""
    ordered = df.sort_values(
        ["Report ID", "_version", "_submitted", "Report ID"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates("Report ID", keep="first").reset_index(drop=True)


def exclude_non_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Drop monthly filings that report no crash."""
    keep = df["Report Type"].str.strip() != NO_NEW_REPORT
    return df.loc[keep].reset_index(drop=True)


def _prefer_row(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["_version", "_submitted", "Report ID"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def collapse_same_incident(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse duplicate filings of one crash.

    NHTSA's Same Incident ID links reports of the same crash, including filings
    by both the manufacturer and the operator. Rows with a blank Same Incident
    ID stay as their own incidents. Within a group, the kept row is the highest
    Report Version, then the latest MON-YYYY submission, then the lowest
    Report ID.
    """
    work = df.copy()
    work["_sid"] = work["Same Incident ID"].str.strip()
    grouped = work[work["_sid"] != ""]
    singles = work[work["_sid"] == ""]

    crash_with_disagreements = 0
    vru_codings_dropped = 0
    mixed_party_groups = 0
    vru_lighting_disagreements = 0
    kept_parts: list[pd.DataFrame] = []
    for _, group in grouped.groupby("_sid", sort=False):
        parties = {classify_party(value) for value in group["Crash With"]}
        named_parties = {party for party in parties if party}
        if len(named_parties) > 1:
            mixed_party_groups += 1
        crash_values = {value.strip() for value in group["Crash With"]}
        if len(crash_values) > 1:
            crash_with_disagreements += 1
        vru_rows = group[group["Crash With"].map(classify_party).notna()]
        if not vru_rows.empty:
            lighting_values = {value.strip() for value in vru_rows["Lighting"]}
            if len(lighting_values) > 1:
                vru_lighting_disagreements += 1
        kept = _prefer_row(group).iloc[[0]]
        if named_parties and classify_party(str(kept["Crash With"].iloc[0])) is None:
            vru_codings_dropped += 1
        kept_parts.append(kept)

    parts: list[pd.DataFrame] = [*kept_parts]
    if not singles.empty:
        parts.append(singles)
    collapsed = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]
    stats = {
        "same_incident_groups": int(grouped["_sid"].nunique()) if not grouped.empty else 0,
        "rows_without_same_incident_id": int(len(singles)),
        "rows_removed_by_collapse": int(len(df) - len(collapsed)),
        "groups_with_disagreeing_crash_with": crash_with_disagreements,
        "groups_with_mixed_pedestrian_and_cyclist": mixed_party_groups,
        "vru_groups_with_disagreeing_lighting": vru_lighting_disagreements,
        "vru_codings_dropped_by_collapse": vru_codings_dropped,
    }
    return collapsed.reset_index(drop=True), stats


def select_vru(df: pd.DataFrame) -> pd.DataFrame:
    """Keep incidents whose Crash With code is a pedestrian or a cyclist."""
    work = df.copy()
    work["party"] = work["Crash With"].map(classify_party)
    work = work[work["party"].notna()].copy()
    work["lighting_bucket"] = work["Lighting"].map(map_lighting)
    work["lighting"] = work["Lighting"].str.strip().replace("", "(blank)")
    work["roadway_type"] = work["Roadway Type"].str.strip().replace("", "(blank)")
    columns = [
        "system",
        "Report ID",
        "Same Incident ID",
        "Crash With",
        "party",
        "lighting",
        "lighting_bucket",
        "roadway_type",
    ]
    renamed = work[columns].rename(
        columns={
            "Report ID": "report_id",
            "Same Incident ID": "same_incident_id",
            "Crash With": "crash_with",
        }
    )
    return renamed.sort_values(
        ["party", "lighting", "roadway_type", "report_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def lighting_column_usable(df: pd.DataFrame) -> bool:
    if "Lighting" not in df.columns:
        return False
    return bool(df["Lighting"].fillna("").astype(str).str.strip().ne("").any())


def reduce_population(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply version, empty-report, and same-incident reductions."""
    prepared = [prepare_reports(frame, system) for system, frame in frames.items()]
    if not prepared:
        empty = pd.DataFrame()
        return empty, empty, {"rows_read": 0, "rows_by_system": {}}
    combined = pd.concat(prepared, ignore_index=True)
    rows_by_system = {system: int(len(frame)) for system, frame in frames.items()}
    latest = latest_version(combined)
    incidents = exclude_non_incidents(latest)
    collapsed, collapse_stats = collapse_same_incident(incidents)
    stats: dict[str, Any] = {
        "rows_read": int(len(combined)),
        "rows_by_system": rows_by_system,
        "unique_report_ids": int(combined["Report ID"].nunique()),
        "latest_version_rows": int(len(latest)),
        "after_excluding_no_new_reports": int(len(incidents)),
        "after_same_incident_collapse": int(len(collapsed)),
        **collapse_stats,
    }
    return collapsed, incidents, stats


def recall_lookup(stratified: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Index time-of-day precision/recall rows by (bucket, class)."""
    required = {"stratum", "value", "class_name", "recall", "tp", "fn"}
    missing = required.difference(stratified.columns)
    if missing:
        raise ValueError(f"Stratified recall file is missing columns: {', '.join(sorted(missing))}")
    rows = stratified[stratified["stratum"].astype(str) == "timeofday"]
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows.to_dict(orient="records"):
        tp = int(row["tp"])
        fn = int(row["fn"])
        lookup[(str(row["value"]), str(row["class_name"]))] = {
            "recall": float(row["recall"]),
            "support": float(tp + fn),
        }
    return lookup


def _count_bucket(incidents: pd.DataFrame, bucket: str, party: str | None = None) -> int:
    rows = incidents
    if party is not None:
        rows = rows[rows["party"] == party]
    return int((rows["lighting_bucket"] == bucket).sum())


def build_lighting_comparison(
    incidents: pd.DataFrame,
    finetuned: dict[tuple[str, str], dict[str, float]],
    baseline: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    known_total = int(incidents["lighting_bucket"].isin(KNOWN_LIGHTING_BUCKETS).sum()) if not incidents.empty else 0
    total = int(len(incidents))
    rows: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        row: dict[str, Any] = {
            "lighting_bucket": bucket,
            "sgo_incidents": _count_bucket(incidents, bucket),
            "sgo_pedestrian": _count_bucket(incidents, bucket, "pedestrian"),
            "sgo_cyclist": _count_bucket(incidents, bucket, "cyclist"),
        }
        row["sgo_share"] = 0.0 if total == 0 else row["sgo_incidents"] / total
        if bucket in KNOWN_LIGHTING_BUCKETS and known_total:
            row["sgo_share_of_known"] = row["sgo_incidents"] / known_total
        else:
            row["sgo_share_of_known"] = float("nan")
        for source_name, lookup in (("finetuned", finetuned), ("baseline", baseline)):
            for class_name in RECALL_CLASSES:
                metrics = lookup.get((bucket, class_name))
                prefix = f"{source_name}_{class_name}"
                if metrics is None:
                    row[f"{prefix}_recall"] = float("nan")
                    row[f"{prefix}_support"] = float("nan")
                else:
                    row[f"{prefix}_recall"] = metrics["recall"]
                    row[f"{prefix}_support"] = metrics["support"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_lighting_raw(incidents: pd.DataFrame) -> pd.DataFrame:
    if incidents.empty:
        return pd.DataFrame(columns=["lighting", "incidents", "pedestrian", "cyclist"])
    rows: list[dict[str, Any]] = []
    for lighting, group in incidents.groupby("lighting", sort=False):
        rows.append(
            {
                "lighting": lighting,
                "incidents": int(len(group)),
                "pedestrian": int((group["party"] == "pedestrian").sum()),
                "cyclist": int((group["party"] == "cyclist").sum()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["incidents", "lighting"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def build_roadway(incidents: pd.DataFrame) -> pd.DataFrame:
    if incidents.empty:
        return pd.DataFrame(columns=["roadway_type", "incidents", "pedestrian", "cyclist"])
    rows: list[dict[str, Any]] = []
    for roadway, group in incidents.groupby("roadway_type", sort=False):
        rows.append(
            {
                "roadway_type": roadway,
                "incidents": int(len(group)),
                "pedestrian": int((group["party"] == "pedestrian").sum()),
                "cyclist": int((group["party"] == "cyclist").sum()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["incidents", "roadway_type"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _mode_bucket(counts: Mapping[str, int]) -> list[str]:
    if not any(counts.get(bucket, 0) for bucket in KNOWN_LIGHTING_BUCKETS):
        return []
    top = max(counts.get(bucket, 0) for bucket in KNOWN_LIGHTING_BUCKETS)
    return [bucket for bucket in KNOWN_LIGHTING_BUCKETS if counts.get(bucket, 0) == top]


def _worst_recall(recall_by_bucket: Mapping[str, float]) -> list[str]:
    present = {bucket: recall_by_bucket[bucket] for bucket in KNOWN_LIGHTING_BUCKETS if bucket in recall_by_bucket}
    if not present:
        return []
    worst = min(present.values())
    return [bucket for bucket in KNOWN_LIGHTING_BUCKETS if bucket in present and present[bucket] == worst]


def alignment_decision(
    counts: Mapping[str, int],
    recall_by_bucket: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the pre-set alignment rule.

    The rule is met when exactly one known-lighting bucket holds the most
    incidents and that same bucket is the unique minimum of the paired recall
    series. Unknown lighting is excluded from the mode. A tie on either side
    leaves the rule unmet.
    """
    mode = _mode_bucket(counts)
    worst = _worst_recall(recall_by_bucket)
    lines_up = len(mode) == 1 and mode == worst
    return {"mode": mode, "lowest_recall": worst, "lines_up": lines_up}


def _party_counts(incidents: pd.DataFrame, party: str | None = None) -> dict[str, int]:
    rows = incidents if party is None else incidents[incidents["party"] == party]
    return {bucket: int((rows["lighting_bucket"] == bucket).sum()) for bucket in BUCKET_ORDER}


def _recall_series(
    lookup: dict[tuple[str, str], dict[str, float]],
    class_name: str,
) -> dict[str, float]:
    series: dict[str, float] = {}
    for bucket in KNOWN_LIGHTING_BUCKETS:
        metrics = lookup.get((bucket, class_name))
        if metrics is not None:
            series[bucket] = metrics["recall"]
    return series


def _fmt_recall(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_support(value: float | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _bucket_phrase(buckets: list[str]) -> str:
    if not buckets:
        return "unavailable"
    if len(buckets) == 1:
        return buckets[0]
    return " and ".join(buckets)


def _describe_slice(
    title: str,
    counts: Mapping[str, int],
    decision: dict[str, Any],
    recall_label: str,
    recall_by_bucket: Mapping[str, float],
    support_by_bucket: Mapping[str, float],
) -> str:
    known = sum(counts.get(bucket, 0) for bucket in KNOWN_LIGHTING_BUCKETS)
    unknown = counts.get("unknown", 0)
    other = counts.get("other", 0)
    total = known + unknown + other
    mode = _bucket_phrase(decision["mode"])
    worst = _bucket_phrase(decision["lowest_recall"])
    worst_key = decision["lowest_recall"][0] if len(decision["lowest_recall"]) == 1 else None
    recall_text = _fmt_recall(recall_by_bucket.get(worst_key) if worst_key else None)
    support_text = _fmt_support(support_by_bucket.get(worst_key) if worst_key else None)
    parts = [
        f"{title}: {total} incidents, {known} with a mapped lighting value "
        f"({', '.join(f'{counts.get(bucket, 0)} {bucket}' for bucket in KNOWN_LIGHTING_BUCKETS)}), "
        f"{unknown} unknown, {other} other."
    ]
    if decision["lines_up"]:
        parts.append(
            f"The alignment rule is met. The known-lighting mode and the lowest {recall_label} "
            f"are both {mode} (recall {recall_text}, support {support_text})."
        )
    else:
        parts.append(
            f"The alignment rule is unmet. The known-lighting mode is {mode}. "
            f"The lowest {recall_label} is {worst} (recall {recall_text}, support {support_text})."
        )
    if len(decision["mode"]) == 1 and unknown:
        mode_count = counts.get(decision["mode"][0], 0)
        if mode_count and unknown * 2 >= mode_count:
            gap = mode_count - unknown
            if gap == 0:
                relation = f"tied with the {mode} count of {mode_count}"
            elif gap == 1:
                relation = f"one below the {mode} count of {mode_count}"
            else:
                relation = f"against a {mode} count of {mode_count}"
            parts.append(f"Unknown lighting is {unknown} of {total} incidents, {relation}.")
    return _wrap(" ".join(parts))


def render_finding(query: dict[str, Any], incidents: pd.DataFrame, comparison: pd.DataFrame, lighting_raw: pd.DataFrame, roadway: pd.DataFrame) -> str:
    decisions = query["alignment"]
    archive = query["archive"]
    current = query["current"]
    lines: list[str] = [
        "# SGO cross-reference",
        "",
        "This note compares lighting on pedestrian and cyclist crashes in the public",
        "NHTSA Standing General Order 2021-01 archive with VRU-Detect recall by lighting.",
        "It is a research check of published codes against a stratified evaluation.",
        "It does not establish deployment readiness or safety certification, and it",
        "does not describe any manufacturer's perception system.",
        "",
        "## Result",
        "",
        _wrap(decisions["summary"]),
        "",
        _describe_slice(
            "Combined pedestrian and cyclist incidents",
            decisions["combined"]["counts"],
            decisions["combined"],
            "fine-tuned all-VRU recall",
            decisions["combined"]["recall"],
            decisions["combined"]["support"],
        ),
        "",
        _describe_slice(
            "Pedestrian incidents, paired with person recall",
            decisions["pedestrian"]["counts"],
            decisions["pedestrian"],
            "fine-tuned person recall",
            decisions["pedestrian"]["recall"],
            decisions["pedestrian"]["support"],
        ),
        "",
        _wrap(decisions["cyclist_paragraph"]),
        "",
        *_dark_code_lines(lighting_raw),
        "BDD's `night` label does not separate lit and unlit roads, so a night match",
        "is a match to that coarser bucket.",
        "",
        "## Query",
        "",
        f"Retrieved {query['retrieved_on']} from the public CSVs linked on",
        f"[NHTSA's SGO crash-reporting page]({NHTSA_PAGE}).",
        "No other repository is used.",
        "",
        "Lighting population: the archive published for reports under the General",
        "Order before the third amendment (incident reports through 15 June 2025).",
        "Those files have a Lighting column. The third-amendment files (reports from",
        "16 June 2025 onward) are a separate count. They enter the lighting table",
        "only when a Lighting column with values is present.",
        "",
        "Steps, in order:",
        "",
        "1. Read the ADS, Level 2 ADAS, and Other CSVs. Other holds reports whose",
        "   automation type was not classified as ADS or Level 2 ADAS.",
        "2. Keep the highest `Report Version` for each `Report ID`. Ties break toward",
        "   the later `Report Submission Date` (`MON-YYYY`, parsed as a date) and then",
        "   the lower `Report ID`.",
        "3. Drop rows whose `Report Type` is `No New or Updated Incident Reports`.",
        "4. Collapse rows that share a non-empty `Same Incident ID`, with the same",
        "   version and date rule. A blank `Same Incident ID` stays its own incident.",
        "5. Keep rows whose `Crash With` is `Non-Motorist: Pedestrian` or",
        "   `Non-Motorist: Cyclist` after case and whitespace normalization.",
        "   `Non-Motorist: Other` is excluded. Narrative text is not searched.",
        "6. Map Lighting onto the detector buckets: `Daylight` → daytime,",
        "   `Dawn / Dusk` → dawn/dusk, `Dark - Lighted`, `Dark - Not Lighted`, and",
        "   `Dark - Unknown Lighting` → night, `Unknown` or blank → unknown,",
        "   `Other, see Narrative` and any unrecognized code → other.",
        "",
        "Alignment rule, fixed before reading the outcome: among daytime, dawn/dusk,",
        "and night, the unique mode of incidents with a mapped lighting value is the",
        "same bucket as the unique minimum of the paired fine-tuned recall series.",
        "Unknown lighting is reported and left out of the mode. A tie on either side",
        "leaves the rule unmet. The combined check uses all-VRU recall. The",
        "pedestrian check uses person recall. The cyclist check uses rider recall and",
        "bike recall, and it meets the rule only when both class minima are that same",
        "single bucket.",
        "",
        "Detector recall is the committed stratified evaluation at confidence 0.25:",
        "`results/finetuned/stratified.csv` and `results/baseline/stratified.csv`.",
        "Support is true positives plus false negatives. The recall-first operating",
        "thresholds in `results/finetuned/threshold_recommendations.csv` (person and",
        "rider 0.05, bike 0.15) are a different operating point. Stratified recall at",
        "those thresholds is not in the committed tables, so this note uses 0.25.",
        "",
        "### Archive files",
        "",
        "| system | url | bytes | sha256 | last-modified |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in query["archive_files"]:
        lines.append(
            f"| {item['system']} | `{item['url']}` | {item['bytes']} | `{item['sha256']}` | {item.get('last_modified') or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "### Reduction counts (archive)",
            "",
            f"- Rows read: {archive['rows_read']} ({_system_counts(archive['rows_by_system'])}).",
            f"- Unique report IDs: {archive['unique_report_ids']}.",
            f"- After keeping the latest version: {archive['latest_version_rows']}.",
            f"- After dropping filings with no new incident: {archive['after_excluding_no_new_reports']}.",
            f"- After collapsing `Same Incident ID`: {archive['after_same_incident_collapse']} "
            f"({archive['rows_removed_by_collapse']} rows removed; "
            f"{archive['rows_without_same_incident_id']} incidents had a blank id).",
            f"- Groups whose member reports disagreed on `Crash With`: {archive['groups_with_disagreeing_crash_with']}.",
            f"- Groups that coded both a pedestrian and a cyclist: {archive['groups_with_mixed_pedestrian_and_cyclist']}.",
            f"- VRU groups whose lighting codes disagreed: {archive['vru_groups_with_disagreeing_lighting']}.",
            f"- Incidents where a pedestrian or cyclist code was dropped because the kept row used a different `Crash With`: {archive['vru_codings_dropped_by_collapse']}.",
            f"- Pedestrian or cyclist incidents kept from the archive: {archive['vru_incidents']} "
            f"({archive['vru_pedestrian']} pedestrian, {archive['vru_cyclist']} cyclist; "
            f"{_system_counts(archive['vru_by_system'])}).",
            "",
            "### Third-amendment files, excluded from the lighting table",
            "",
        ]
    )
    for item in query["current_files"]:
        lines.append(
            f"- {item['system']}: `{item['url']}` ({item['bytes']} bytes, sha256 `{item['sha256']}`, last-modified {item.get('last_modified') or 'n/a'})."
        )
    if current.get("lighting_included"):
        lines.append(
            "- These files contained Lighting values and were included in the lighting population."
        )
    else:
        lines.append(
            f"- Lighting column usable: {str(current.get('lighting_usable', False)).lower()}."
        )
        dropped = int(current.get("vru_codings_dropped_by_collapse", 0))
        dropped_sentence = ""
        if dropped:
            noun = "code" if dropped == 1 else "codes"
            group_word = "that group" if dropped == 1 else "those groups"
            dropped_sentence = (
                f" The same-incident collapse left out {dropped} pedestrian or cyclist {noun} "
                f"because the kept row of {group_word} used a different `Crash With`."
            )
        lines.append(
            textwrap.fill(
                "- Same reduction, then pedestrian or cyclist `Crash With`: "
                f"{current.get('vru_incidents', 0)} incidents "
                f"({current.get('vru_pedestrian', 0)} pedestrian, {current.get('vru_cyclist', 0)} cyclist) "
                f"from {current.get('rows_read', 0)} rows. They stay out of the lighting distribution "
                "because the file has no usable Lighting field."
                + dropped_sentence,
                width=88,
                subsequent_indent="  ",
            )
        )
    if query.get("unmapped_lighting"):
        rendered = ", ".join(f"`{value}`" for value in query["unmapped_lighting"])
        lines.extend(["", f"Unrecognized Lighting codes mapped to other: {rendered}."])
    lines.extend(
        [
            "",
            "## Lighting codes",
            "",
            "| SGO Lighting | incidents | pedestrian | cyclist |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    if lighting_raw.empty:
        lines.append("| (none) | 0 | 0 | 0 |")
    else:
        for row in lighting_raw.to_dict(orient="records"):
            lines.append(
                f"| {row['lighting']} | {int(row['incidents'])} | {int(row['pedestrian'])} | {int(row['cyclist'])} |"
            )
    lines.extend(
        [
            "",
            "## Lighting buckets and detector recall",
            "",
            "Share of known uses daytime + dawn/dusk + night as the denominator.",
            "Recall is fine-tuned, confidence 0.25. Baseline all-VRU recall is the last column.",
            "",
            "| bucket | SGO | pedestrian | cyclist | share of known | recall all | support | person recall | person support | baseline recall all |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.to_dict(orient="records"):
        share = row["sgo_share_of_known"]
        share_text = "—" if pd.isna(share) else f"{float(share):.1%}"
        lines.append(
            "| {bucket} | {sgo} | {ped} | {cyc} | {share} | {recall} | {support} | {person} | {person_support} | {baseline} |".format(
                bucket=row["lighting_bucket"],
                sgo=int(row["sgo_incidents"]),
                ped=int(row["sgo_pedestrian"]),
                cyc=int(row["sgo_cyclist"]),
                share=share_text,
                recall=_cell_recall(row, "finetuned_all_recall"),
                support=_cell_support(row, "finetuned_all_support"),
                person=_cell_recall(row, "finetuned_person_recall"),
                person_support=_cell_support(row, "finetuned_person_support"),
                baseline=_cell_recall(row, "baseline_all_recall"),
            )
        )
    lines.extend(
        [
            "",
            "Rider and bike recall, the classes paired with the cyclist count:",
            "",
            "| bucket | rider recall | rider support | bike recall | bike support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.to_dict(orient="records"):
        if row["lighting_bucket"] not in KNOWN_LIGHTING_BUCKETS:
            continue
        lines.append(
            "| {bucket} | {rider} | {rider_support} | {bike} | {bike_support} |".format(
                bucket=row["lighting_bucket"],
                rider=_cell_recall(row, "finetuned_rider_recall"),
                rider_support=_cell_support(row, "finetuned_rider_support"),
                bike=_cell_recall(row, "finetuned_bike_recall"),
                bike_support=_cell_support(row, "finetuned_bike_support"),
            )
        )
    lines.extend(
        [
            "",
            "## Roadway",
            "",
            "Roadway Type is extracted for the same pedestrian and cyclist incidents.",
            "It is not compared with BDD scene recall. SGO `Street` and `Intersection`",
            "are not BDD `city street` and `residential`, and forcing that join would",
            "invent a match the labels do not support.",
            "",
            "| Roadway Type | incidents | pedestrian | cyclist |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    if roadway.empty:
        lines.append("| (none) | 0 | 0 | 0 |")
    else:
        for row in roadway.to_dict(orient="records"):
            lines.append(
                f"| {row['roadway_type']} | {int(row['incidents'])} | {int(row['pedestrian'])} | {int(row['cyclist'])} |"
            )
    lines.extend(["", _wrap(_roadway_sentence(roadway, len(incidents))), ""])
    lines.extend(
        [
            "## Limits",
            "",
            "- The archive is crashes that named reporting entities submitted under",
            "  SGO 2021-01. NHTSA states that reporting thresholds differ for ADS and",
            "  Level 2 ADAS, that entities only report crashes they know about, and",
            "  that the files are not normalized by vehicles or miles traveled. The",
            "  system mix in the counts above is a reporting mix.",
            "- `Crash With` is one category in the published CSV. A crash coded as a",
            "  passenger car is outside this set even if a person was nearby.",
            "- The same crash can remain split when `Same Incident ID` is blank or wrong.",
            "  NHTSA documents that limitation.",
            "- Several lighting and roadway cells are small. The pedestrian known-lighting",
            "  total is especially thin, and unknown lighting is a large share of the",
            "  pedestrian rows.",
            "- Recall is from the 300-image BDD100K subsample at confidence 0.25, not",
            "  from the crashes themselves. A low-recall bucket can be safety-relevant",
            "  on its own; this note only asks whether that bucket is also where these",
            "  reported crashes sit.",
            "",
        ]
    )
    return "\n".join(lines)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def _wrap(text: str) -> str:
    return textwrap.fill(text, width=88)


def _system_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "none"
    preferred = ["ADS", "ADAS", "OTHER"]
    keys = [key for key in preferred if key in counts]
    keys.extend(key for key in counts if key not in keys)
    return ", ".join(f"{counts[key]} {key}" for key in keys)


def _cell_recall(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def _cell_support(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or pd.isna(value):
        return "—"
    return str(int(value))


def _dark_code_lines(lighting_raw: pd.DataFrame) -> list[str]:
    codes = ("Dark - Lighted", "Dark - Not Lighted", "Dark - Unknown Lighting")
    counts = {code: 0 for code in codes}
    if not lighting_raw.empty:
        for row in lighting_raw.to_dict(orient="records"):
            for code in codes:
                if normalize_label(str(row["lighting"])) == normalize_label(code):
                    counts[code] = int(row["incidents"])
    lines = ["Night is the sum of these dark SGO codes:", ""]
    lines.extend(f"- `{code}`: {counts[code]}" for code in codes)
    lines.append("")
    return lines


def _roadway_sentence(roadway: pd.DataFrame, total: int) -> str:
    if roadway.empty or total == 0:
        return "No roadway codes were available for this set."
    ordered = roadway.to_dict(orient="records")
    top = ordered[0]
    street = next((row for row in ordered if row["roadway_type"] == "Street"), None)
    intersection = next((row for row in ordered if row["roadway_type"] == "Intersection"), None)
    street_n = int(street["incidents"]) if street else 0
    intersection_n = int(intersection["incidents"]) if intersection else 0
    sentence = (
        f"`{top['roadway_type']}` is the most common roadway code "
        f"({int(top['incidents'])} of {total})."
    )
    if street_n or intersection_n:
        sentence += f" Street and Intersection together are {street_n + intersection_n} of {total}."
    highway = next((row for row in ordered if row["roadway_type"] == "Highway / Freeway"), None)
    if highway is not None and int(highway["incidents"]) == int(highway["pedestrian"]) and int(highway["cyclist"]) == 0:
        sentence += (
            f" All {int(highway['incidents'])} Highway / Freeway incidents in this set are pedestrians."
        )
    return sentence


def _decision_block(
    incidents: pd.DataFrame,
    party: str | None,
    lookup: dict[tuple[str, str], dict[str, float]],
    class_name: str,
) -> dict[str, Any]:
    counts = _party_counts(incidents, party)
    recall = _recall_series(lookup, class_name)
    support = {
        bucket: lookup[(bucket, class_name)]["support"]
        for bucket in KNOWN_LIGHTING_BUCKETS
        if (bucket, class_name) in lookup
    }
    decision = alignment_decision(counts, recall)
    decision["counts"] = counts
    decision["recall"] = recall
    decision["support"] = support
    return decision


def _cyclist_paragraph(incidents: pd.DataFrame, finetuned: dict[tuple[str, str], dict[str, float]]) -> str:
    counts = _party_counts(incidents, "cyclist")
    rider = alignment_decision(counts, _recall_series(finetuned, "rider"))
    bike = alignment_decision(counts, _recall_series(finetuned, "bike"))
    known = sum(counts.get(bucket, 0) for bucket in KNOWN_LIGHTING_BUCKETS)
    rider_recall = _recall_series(finetuned, "rider")
    bike_recall = _recall_series(finetuned, "bike")
    rider_support = {
        bucket: finetuned[(bucket, "rider")]["support"]
        for bucket in KNOWN_LIGHTING_BUCKETS
        if (bucket, "rider") in finetuned
    }
    bike_support = {
        bucket: finetuned[(bucket, "bike")]["support"]
        for bucket in KNOWN_LIGHTING_BUCKETS
        if (bucket, "bike") in finetuned
    }
    base = (
        f"Cyclist incidents, paired with rider and bike recall: "
        f"{int((incidents['party'] == 'cyclist').sum()) if not incidents.empty else 0} incidents, "
        f"{known} with a mapped lighting value "
        f"({', '.join(f'{counts.get(bucket, 0)} {bucket}' for bucket in KNOWN_LIGHTING_BUCKETS)}), "
        f"{counts.get('unknown', 0)} unknown, {counts.get('other', 0)} other."
    )
    rider_worst = _bucket_phrase(rider["lowest_recall"])
    bike_worst = _bucket_phrase(bike["lowest_recall"])
    mode = _bucket_phrase(rider["mode"])
    if rider["lines_up"] and bike["lines_up"] and rider["lowest_recall"] == bike["lowest_recall"]:
        return (
            f"{base} The alignment rule is met for both classes. The known-lighting mode "
            f"and both recall minima are {mode}."
        )
    rider_key = rider["lowest_recall"][0] if len(rider["lowest_recall"]) == 1 else None
    bike_key = bike["lowest_recall"][0] if len(bike["lowest_recall"]) == 1 else None
    return (
        f"{base} The alignment rule is unmet. The known-lighting mode is {mode}. "
        f"The lowest rider recall is {rider_worst} "
        f"(recall {_fmt_recall(rider_recall.get(rider_key) if rider_key else None)}, "
        f"support {_fmt_support(rider_support.get(rider_key) if rider_key else None)}). "
        f"The lowest bike recall is {bike_worst} "
        f"(recall {_fmt_recall(bike_recall.get(bike_key) if bike_key else None)}, "
        f"support {_fmt_support(bike_support.get(bike_key) if bike_key else None)})."
    )


def _summary_sentence(combined: dict[str, Any], pedestrian: dict[str, Any], cyclist_lines_up: bool) -> str:
    combined_text = "met" if combined["lines_up"] else "unmet"
    pedestrian_text = "met" if pedestrian["lines_up"] else "unmet"
    cyclist_text = "met" if cyclist_lines_up else "unmet"
    return (
        f"Primary check, pedestrian and cyclist incidents together versus all-VRU recall: "
        f"the alignment rule is {combined_text}. "
        f"Pedestrian incidents versus person recall: the alignment rule is {pedestrian_text}. "
        f"Cyclist incidents versus rider and bike recall: the alignment rule is {cyclist_text}."
    )


def _cyclist_lines_up(incidents: pd.DataFrame, finetuned: dict[tuple[str, str], dict[str, float]]) -> bool:
    counts = _party_counts(incidents, "cyclist")
    rider = alignment_decision(counts, _recall_series(finetuned, "rider"))
    bike = alignment_decision(counts, _recall_series(finetuned, "bike"))
    return bool(rider["lines_up"] and bike["lines_up"] and rider["lowest_recall"] == bike["lowest_recall"])


def build_crossref(
    archive_frames: Mapping[str, pd.DataFrame],
    current_frames: Mapping[str, pd.DataFrame],
    finetuned_stratified: pd.DataFrame,
    baseline_stratified: pd.DataFrame,
    *,
    retrieved_on: str,
    archive_files: list[dict[str, Any]],
    current_files: list[dict[str, Any]],
) -> CrossrefResult:
    collapsed, _pre_collapse, archive_stats = reduce_population(archive_frames)
    if "Lighting" not in collapsed.columns or not lighting_column_usable(collapsed):
        raise ValueError("Archive files have no usable Lighting column; the lighting comparison cannot run.")
    archive_incidents = select_vru(collapsed)
    current_collapsed, _, current_stats = reduce_population(current_frames) if current_frames else (pd.DataFrame(), pd.DataFrame(), {})
    current_usable = not current_collapsed.empty and lighting_column_usable(current_collapsed)
    if current_usable:
        current_vru = select_vru(current_collapsed)
        incidents = pd.concat([archive_incidents, current_vru], ignore_index=True)
        incidents = incidents.sort_values(
            ["party", "lighting", "roadway_type", "report_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        current_vru = (
            select_vru(current_collapsed) if not current_collapsed.empty else pd.DataFrame(columns=["party", "system"])
        )
        incidents = archive_incidents

    finetuned = recall_lookup(finetuned_stratified)
    baseline = recall_lookup(baseline_stratified)
    comparison = build_lighting_comparison(incidents, finetuned, baseline)
    lighting_raw = build_lighting_raw(incidents)
    roadway = build_roadway(incidents)

    unmapped = sorted(
        {
            str(value)
            for value in incidents["lighting"]
            if map_lighting(str(value)) == "other"
            and normalize_label(str(value)) not in {"other, see narrative", "(blank)"}
        }
    )
    archive_stats["vru_incidents"] = int(len(archive_incidents))
    archive_stats["vru_pedestrian"] = (
        int((archive_incidents["party"] == "pedestrian").sum()) if not archive_incidents.empty else 0
    )
    archive_stats["vru_cyclist"] = int((archive_incidents["party"] == "cyclist").sum()) if not archive_incidents.empty else 0
    archive_stats["vru_by_system"] = _value_counts(archive_incidents, "system")

    current_summary = {
        "rows_read": int(current_stats.get("rows_read", 0)),
        "lighting_usable": bool(current_usable),
        "lighting_included": bool(current_usable),
        "vru_incidents": 0 if current_usable else int(len(current_vru)),
        "vru_pedestrian": 0
        if current_usable or current_vru.empty
        else int((current_vru["party"] == "pedestrian").sum()),
        "vru_cyclist": 0 if current_usable or current_vru.empty else int((current_vru["party"] == "cyclist").sum()),
        "vru_codings_dropped_by_collapse": int(current_stats.get("vru_codings_dropped_by_collapse", 0)),
    }
    if current_usable:
        current_summary["vru_incidents_included"] = int(len(current_vru))

    combined = _decision_block(incidents, None, finetuned, "all")
    pedestrian = _decision_block(incidents, "pedestrian", finetuned, "person")
    cyclist_ok = _cyclist_lines_up(incidents, finetuned)
    alignment = {
        "summary": _summary_sentence(combined, pedestrian, cyclist_ok),
        "combined": combined,
        "pedestrian": pedestrian,
        "cyclist_lines_up": cyclist_ok,
        "cyclist_paragraph": _cyclist_paragraph(incidents, finetuned),
    }
    query: dict[str, Any] = {
        "retrieved_on": retrieved_on,
        "source_page": NHTSA_PAGE,
        "archive_files": archive_files,
        "current_files": current_files,
        "archive": archive_stats,
        "current": current_summary,
        "unmapped_lighting": unmapped,
        "alignment": {
            "summary": alignment["summary"],
            "combined": {
                "lines_up": combined["lines_up"],
                "mode": combined["mode"],
                "lowest_recall": combined["lowest_recall"],
                "counts": combined["counts"],
                "recall": combined["recall"],
                "support": combined["support"],
            },
            "pedestrian": {
                "lines_up": pedestrian["lines_up"],
                "mode": pedestrian["mode"],
                "lowest_recall": pedestrian["lowest_recall"],
                "counts": pedestrian["counts"],
                "recall": pedestrian["recall"],
                "support": pedestrian["support"],
            },
            "cyclist_lines_up": cyclist_ok,
            "cyclist_paragraph": alignment["cyclist_paragraph"],
        },
        "recall_confidence": 0.25,
        "crash_with_codes": ["Non-Motorist: Pedestrian", "Non-Motorist: Cyclist"],
    }
    # The stored query is what the note renders. Rebuild the alignment view the
    # renderer expects (full decision dicts live on `alignment` above).
    query["alignment"] = alignment
    finding = render_finding(query, incidents, comparison, lighting_raw, roadway)
    serializable = json.loads(json.dumps(query, default=_json_default))
    query_for_disk = serializable
    return CrossrefResult(
        incidents=incidents,
        lighting_comparison=comparison,
        lighting_raw=lighting_raw,
        roadway=roadway,
        query=query_for_disk,
        finding_md=finding,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, float):
        return value
    return str(value)


def write_outputs(result: CrossrefResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = output_dir / "sgo_audit_unavailable.md"
    if stale.exists():
        stale.unlink()
    result.incidents.to_csv(output_dir / "vru_incidents.csv", index=False)
    result.lighting_comparison.to_csv(output_dir / "lighting_comparison.csv", index=False)
    result.lighting_raw.to_csv(output_dir / "lighting_raw_counts.csv", index=False)
    result.roadway.to_csv(output_dir / "roadway_counts.csv", index=False)
    (output_dir / "query.json").write_text(json.dumps(result.query, indent=2) + "\n", encoding="utf-8")
    (output_dir / "finding.md").write_text(result.finding_md, encoding="utf-8")


def read_sgo_csv(path: Path) -> pd.DataFrame:
    # One published archive file contains a non-UTF-8 byte inside a city name
    # ("Coeur d…Alene"). Fields used here are unaffected; replace the bad byte
    # rather than guessing a second encoding for the whole file.
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def download_file(url: str, dest: Path, timeout: int = 180) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        last_modified = response.headers.get("Last-Modified", "")
    dest.write_bytes(payload)
    return {
        "url": url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "last_modified": last_modified,
    }


def fetch_sources(
    sources: tuple[tuple[str, str], ...],
    cache_dir: Path,
    refresh: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, Any]] = []
    for system, url in sources:
        filename = url.rsplit("/", 1)[-1]
        dest = cache_dir / f"{system.lower()}_{filename}"
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        if dest.exists() and meta_path.exists() and not refresh:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if digest != meta.get("sha256"):
                meta = download_file(url, dest)
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        else:
            meta = download_file(url, dest)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        record = {"system": system, **meta}
        manifest.append(record)
        frames[system] = read_sgo_csv(dest)
    return frames, manifest


def load_stratified(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing stratified recall table: {path}")
    return pd.read_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sgo_crossref"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/downloads/sgo"))
    parser.add_argument("--finetuned-stratified", type=Path, default=Path("results/finetuned/stratified.csv"))
    parser.add_argument("--baseline-stratified", type=Path, default=Path("results/baseline/stratified.csv"))
    parser.add_argument("--refresh", action="store_true", help="Redownload NHTSA CSVs even if a cache exists")
    parser.add_argument("--retrieved-on", default=None, help="Override the retrieval date (YYYY-MM-DD)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    retrieved_on = args.retrieved_on or date.today().isoformat()
    archive_frames, archive_files = fetch_sources(ARCHIVE_SOURCES, args.cache_dir / "archive", args.refresh)
    current_frames, current_files = fetch_sources(CURRENT_SOURCES, args.cache_dir / "current", args.refresh)
    result = build_crossref(
        archive_frames,
        current_frames,
        load_stratified(args.finetuned_stratified),
        load_stratified(args.baseline_stratified),
        retrieved_on=retrieved_on,
        archive_files=archive_files,
        current_files=current_files,
    )
    write_outputs(result, args.output_dir)
    print(f"Wrote SGO cross-reference to {args.output_dir}")
    print(result.query["alignment"]["summary"])


if __name__ == "__main__":
    main()
