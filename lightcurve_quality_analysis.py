#!/usr/bin/env python3
"""Audit timing, cadence, and band quality in serialized detected sources."""

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def _finite_times(source, key):
    values = source.get(key, [])
    return sorted(float(value) for value in values if math.isfinite(float(value)))


def summarize_source(path, source_index, source):
    positive = _finite_times(source, "photometry_times")
    all_observations = sorted(set(positive + _finite_times(source, "non_detections_times")))
    bands = [str(value) for value in source.get("photometry_bands", [])]
    span = positive[-1] - positive[0] if len(positive) > 1 else 0.0
    all_span = all_observations[-1] - all_observations[0] if len(all_observations) > 1 else 0.0
    gaps = [right - left for left, right in zip(positive, positive[1:])]
    counts = {band: bands.count(band) for band in sorted(set(bands))}
    return {
        "file": str(path),
        "source_index": source_index,
        "z": source.get("z"),
        "t_exp": source.get("t_exp"),
        "n_positive": len(positive),
        "n_non_detections": len(source.get("non_detections_times", [])),
        "n_bands": len(set(bands)),
        "bands": ";".join(sorted(set(bands))),
        "positive_span_days": span,
        "all_observation_span_days": all_span,
        "max_positive_gap_days": max(gaps, default=0.0),
        "min_positive_gap_days": min(gaps, default=0.0),
        "positive_span_exceeds_14_days": span > 14.0,
        "all_observation_span_exceeds_14_days": all_span > 14.0,
        "band_counts": json.dumps(counts, sort_keys=True),
    }


def audit_file(path):
    data = json.loads(Path(path).read_text())
    sources = data.get("detected_sources", []) if isinstance(data, dict) else []
    return [summarize_source(path, index, source) for index, source in enumerate(sources)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", dest="pattern", default="results/**/*.json")
    parser.add_argument("--output", default="results/lightcurve_quality.csv")
    args = parser.parse_args()

    rows = []
    files = sorted(glob.glob(args.pattern, recursive=True))
    for filename in files:
        if filename.endswith(".metadata.json"):
            continue
        try:
            rows.extend(audit_file(filename))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Skipping {filename}: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file", "source_index", "z", "t_exp", "n_positive", "n_non_detections",
        "n_bands", "bands", "positive_span_days", "all_observation_span_days",
        "max_positive_gap_days", "min_positive_gap_days",
        "positive_span_exceeds_14_days", "all_observation_span_exceeds_14_days",
        "band_counts",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    violations = sum(row["positive_span_exceeds_14_days"] for row in rows)
    print(f"Wrote {len(rows)} detected-source rows to {output}")
    print(f"Positive-detection spans over 14 days: {violations}")


if __name__ == "__main__":
    main()
