#!/usr/bin/env python3
"""Audit serialized rate limits against published ZTF benchmarks.

This reads existing JSON results only; it does not import ZTF_sim or rerun a
survey simulation.
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path


PUBLISHED_FIXED_95 = 1775.0
PUBLISHED_ISOTROPIC_95 = 4029.0
PUBLISHED_DURATION_YEARS = 23.0 / 12.0


def summarize_result(path):
    data = json.loads(Path(path).read_text())
    rows = []
    for summary in data.get("rate_summaries", []):
        effective_vt = float(summary.get("effective_vt_gpc3_yr", float("nan")))
        if not math.isfinite(effective_vt) or effective_vt <= 0:
            continue
        rate_90 = -math.log(1.0 - 0.90) / effective_vt
        rate_95 = -math.log(1.0 - 0.95) / effective_vt
        duration = float(summary.get("survey_duration_years", float("nan")))
        duration_scaled_95 = rate_95 * duration / PUBLISHED_DURATION_YEARS
        rows.append(
            {
                "file": str(path),
                "transient_type": summary.get("transient_type"),
                "volumetric_rate": summary.get("volumetric_rate"),
                "n_simulated": summary.get("n_simulated", data.get("n_simulated")),
                "n_detected": summary.get("n_detected", data.get("n_detected")),
                "overall_efficiency": summary.get("overall_efficiency"),
                "z_max": summary.get("z_max"),
                "survey_duration_years": duration,
                "survey_omega_sr": summary.get("survey_omega_sr"),
                "effective_vt_gpc3_yr": effective_vt,
                "rate_90_zero_events": rate_90,
                "rate_95_zero_events": rate_95,
                "duration_scaled_95": duration_scaled_95,
                "ratio_to_fixed_1775": duration_scaled_95 / PUBLISHED_FIXED_95,
                "ratio_to_isotropic_4029": duration_scaled_95 / PUBLISHED_ISOTROPIC_95,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        dest="pattern",
        default="**/*.json",
        help="Recursive result-file glob.",
    )
    parser.add_argument("--output", default="results/rate_limit_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.pattern, recursive=True)):
        try:
            rows.extend(summarize_result(path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text("")

    print(f"Wrote {len(rows)} rate-summary rows to {output}")
    if rows:
        scaled = sorted(row["duration_scaled_95"] for row in rows)
        midpoint = scaled[len(scaled) // 2]
        print(f"Duration-scaled 95% median: {midpoint:.3g} Gpc^-3 yr^-1")
        print(f"Published fixed-angle benchmark: {PUBLISHED_FIXED_95:.3g}")
        print(f"Published isotropic benchmark: {PUBLISHED_ISOTROPIC_95:.3g}")


if __name__ == "__main__":
    main()
