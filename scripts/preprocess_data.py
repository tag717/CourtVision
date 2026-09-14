#!/usr/bin/env python3
"""
CourtVision - PL-NBA annotation preprocessing
---------------------
1. Recursively reads PL-NBA possession JSON files.
2. Flattens each annotated event into one metadata row.
3. Converts timestamps such as "05:50" -> 5.50 seconds.
4. Preserves the original annotation interval.
5. Creates an effective continuous event interval using the next event's
   start time as the current event's end time.

   This follows the CourtVision annotation policy:
   - Unlabeled gaps are treated as continuation of the previous event/state.
   - Overlaps are resolved by assuming the next event begins at its start time.
   - The final event keeps its original annotated end time.

6. Adds ML-friendly fields:
   - is_shot
   - shot_type
   - shot_outcome

7. Writes:
   - normalized event metadata CSV
   - dataset summary JSON

Example
-------
python scripts/preprocess_data.py --input data/raw --output data/metadata/events.csv --stats data/metadata/dataset_stats.json"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SHOT_LABELS = {
    "2ptShot",
    "3ptShot",
    "Layup",
    "FreeThrow",
    "PutBack",
    "Dunk",
}

# Normalization due to input file inconsistancy
LABEL_NORMALIZATION = {
    "Putback": "PutBack",
}

# Normalize only when the event itself is a shot.
SHOT_OUTCOME_MAP = {
    "made": "MADE",
    "missed": "MISSED",
    "outside": "MISSED",
    "fouled": "FOUL",
}


def parse_timestamp(timestamp: str) -> float:
    """
    Convert PL-NBA's SS:CC timestamp format to decimal seconds.

    Examples:
        "00:00" -> 0.00
        "05:50" -> 5.50
        "17:20" -> 17.20
    """
    if not isinstance(timestamp, str) or ":" not in timestamp:
        raise ValueError(f"Invalid timestamp: {timestamp!r}")

    seconds_part, decimal_part = timestamp.split(":", 1)

    if not seconds_part.isdigit() or not decimal_part.isdigit():
        raise ValueError(f"Invalid timestamp: {timestamp!r}")

    return int(seconds_part) + int(decimal_part) / (10 ** len(decimal_part))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_possession(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)

    required_top_level = {
        "video_id",
        "possession_id",
        "home_team",
        "away_team",
        "events",
    }
    missing = required_top_level - data.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level fields: {sorted(missing)}")

    events = data["events"]
    if not isinstance(events, list):
        raise ValueError(f"{path}: 'events' must be a list")

    parsed_events: list[dict[str, Any]] = []

    for original_index, event in enumerate(events):
        ts = event.get("timestamp_video", {})
        raw_start = parse_timestamp(ts.get("start"))
        raw_end = parse_timestamp(ts.get("end"))

        # if one event in that possession is invalid, only mark that event
        # invalid and continue
        if raw_end < raw_start:
            print(
                f"Skipping invalid event: {path.name} "
                f"{event.get('event_id')} "
                f"({raw_start} -> {raw_end})"
            )
            continue

        parsed_events.append(
            {
                "_original_index": original_index,
                "_raw_start": raw_start,
                "_raw_end": raw_end,
                "_event": event,
            }
        )

    # events with the same start time preserve their original JSON order.
    parsed_events.sort(key=lambda x: (x["_raw_start"], x["_original_index"]))

    rows: list[dict[str, Any]] = []

    for i, item in enumerate(parsed_events):
        event = item["_event"]
        raw_start = item["_raw_start"]
        raw_end = item["_raw_end"]

        if i < len(parsed_events) - 1:
            next_start = parsed_events[i + 1]["_raw_start"]

            if next_start > raw_start:
                effective_end = next_start
            else:
                # Simultaneous/overlapping event:
                # preserve this event's original interval
                effective_end = raw_end
        else:
            effective_end = raw_end

        effective_start = raw_start
        
        event_label_raw = event.get("event_label", "")
        event_label = LABEL_NORMALIZATION.get(event_label_raw, event_label_raw)
        raw_result = event.get("result")
        is_shot = event_label in SHOT_LABELS

        shot_type = event_label if is_shot else ""
        shot_outcome = (
            SHOT_OUTCOME_MAP.get(raw_result, "UNKNOWN")
            if is_shot
            else ""
        )

        is_valid_interval = int(effective_end > effective_start)
        # later for train valid_df = df[df["is_valid_interval"] == 1] will 
        # exclude invalid datas

        rows.append(
            {
                "source_file": path.name,
                "video_id": data["video_id"],
                "possession_id": data["possession_id"],
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                "event_id": event.get("event_id", ""),
                "event_label": event_label or "",
                "player": event.get("player", ""),
                "raw_result": raw_result or "",
                "caption": event.get("caption", ""),
                "raw_start_sec": raw_start,
                "raw_end_sec": raw_end,
                "raw_duration_sec": round(raw_end - raw_start, 4),
                "effective_start_sec": effective_start,
                "effective_end_sec": effective_end,
                "effective_duration_sec": round(
                    effective_end - effective_start, 4
                ),
                "is_valid_interval": is_valid_interval,
                "is_shot": int(is_shot),
                "shot_type": shot_type,
                "shot_outcome": shot_outcome,
            }
        )

    return rows


def build_stats(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    event_counts = Counter(row["event_label"] for row in rows)
    result_counts = Counter(row["raw_result"] for row in rows)

    event_result_counts = Counter(
        (row["event_label"], row["raw_result"]) for row in rows
    )

    shot_type_counts = Counter(
        row["shot_type"] for row in rows if row["is_shot"]
    )
    shot_outcome_counts = Counter(
        row["shot_outcome"] for row in rows if row["is_shot"]
    )

    shot_type_outcome_counts = Counter(
        (row["shot_type"], row["shot_outcome"])
        for row in rows
        if row["is_shot"]
    )

    zero_or_negative_effective = [
        {
            "video_id": row["video_id"],
            "possession_id": row["possession_id"],
            "event_id": row["event_id"],
            "event_label": row["event_label"],
            "effective_start_sec": row["effective_start_sec"],
            "effective_end_sec": row["effective_end_sec"],
        }
        for row in rows
        if row["effective_duration_sec"] <= 0
    ]

    unknown_shot_outcomes = [
        {
            "video_id": row["video_id"],
            "possession_id": row["possession_id"],
            "event_id": row["event_id"],
            "shot_type": row["shot_type"],
            "raw_result": row["raw_result"],
        }
        for row in rows
        if row["is_shot"] and row["shot_outcome"] == "UNKNOWN"
    ]

    return {
        "num_events": len(rows),
        "num_videos": len({row["video_id"] for row in rows}),
        "num_possessions": len({row["possession_id"] for row in rows}),
        "num_shot_events": sum(row["is_shot"] for row in rows),
        "event_counts": dict(sorted(event_counts.items())),
        "result_counts": dict(sorted(result_counts.items())),
        "event_result_counts": {
            f"{event} | {result}": count
            for (event, result), count in sorted(event_result_counts.items())
        },
        "shot_type_counts": dict(sorted(shot_type_counts.items())),
        "shot_outcome_counts": dict(sorted(shot_outcome_counts.items())),
        "shot_type_outcome_counts": {
            f"{shot_type} | {outcome}": count
            for (shot_type, outcome), count
            in sorted(shot_type_outcome_counts.items())
        },
        "zero_or_negative_effective_intervals": zero_or_negative_effective,
        "unknown_shot_outcomes": unknown_shot_outcomes,
        "parse_errors": errors,
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No valid annotation rows were generated.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize PL-NBA annotations for CourtVision."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing PL-NBA JSON annotation files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/events.csv"),
        help="Output normalized metadata CSV.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/metadata/dataset_stats.json"),
        help="Output dataset statistics JSON.",
    )

    args = parser.parse_args()

    json_files = sorted(args.input.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON annotation files found under: {args.input}"
        )

    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, path in enumerate(json_files, start=1):
        try:
            all_rows.extend(normalize_possession(path))

            if i % 100 == 0:
                print(f"Processed {i}/{len(json_files)} files...")

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    write_csv(all_rows, args.output)

    stats = build_stats(all_rows, errors)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Processed JSON files: {len(json_files)}")
    print(f"Generated event rows: {len(all_rows)}")
    print(f"Unique videos: {stats['num_videos']}")
    print(f"Unique possessions: {stats['num_possessions']}")
    print(f"Shot events: {stats['num_shot_events']}")
    print(f"Output CSV: {args.output}")
    print(f"Stats JSON: {args.stats}")

    if errors:
        print(f"WARNING: {len(errors)} files failed to parse.")
        print("See parse_errors in the stats JSON.")

    if stats["unknown_shot_outcomes"]:
        print(
            "WARNING: Found shot events with unmapped outcomes. "
            "Review unknown_shot_outcomes in the stats JSON."
        )

    if stats["zero_or_negative_effective_intervals"]:
        print(
            "WARNING: Found zero/negative effective intervals, usually caused "
            "by events with identical start times. Review them before training."
        )


if __name__ == "__main__":
    main()
