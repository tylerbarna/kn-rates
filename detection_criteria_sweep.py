#!/usr/bin/env python3
"""Launch reproducible ZTF_sim runs over a detection-criteria grid."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DEFAULT_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 3.0, 4.0),
    "min_detections": (2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (3.0, 7.0, 14.0, 21.0),
    "min_time_separation_hours": (3.0, 12.0),
    "min_fade_rate": (0.0, 0.3, 0.6),
    "min_galactic_lat": (0.0, 15.0),
}


def parse_values(text, converter):
    return tuple(converter(value) for value in text.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/detection_criteria_sweep")
    parser.add_argument("--n-transients", type=int, default=100_000)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--model",
        default="Bu2026Fixed",
        choices=("Bu2026Fixed", "Bu2026Vary", "Metzger"),
    )
    parser.add_argument("--snr-thresholds", default="4,5,6")
    parser.add_argument("--snr-secondary-values", default="2,3,4")
    parser.add_argument("--min-detections-values", default="2,3,4")
    parser.add_argument("--min-detections-primary-values", default="1,2")
    parser.add_argument("--max-timespan-values", default="3,7,14,21")
    parser.add_argument("--min-time-separation-hours-values", default="3,12")
    parser.add_argument("--min-fade-rate-values", default="0,0.3,0.6")
    parser.add_argument("--min-galactic-lat-values", default="0,15")
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Run only this zero-based grid index, for use with Slurm arrays.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    combinations = list(
        itertools.product(
            parse_values(args.snr_thresholds, float),
            parse_values(args.snr_secondary_values, float),
            parse_values(args.min_detections_values, int),
            parse_values(args.min_detections_primary_values, int),
            parse_values(args.max_timespan_values, float),
            parse_values(args.min_time_separation_hours_values, float),
            parse_values(args.min_fade_rate_values, float),
            parse_values(args.min_galactic_lat_values, float),
        )
    )
    if args.array_index is None:
        indexed_combinations = enumerate(combinations)
    else:
        if not 0 <= args.array_index < len(combinations):
            parser.error(f"--array-index must be between 0 and {len(combinations) - 1}")
        indexed_combinations = [(args.array_index, combinations[args.array_index])]

    for index, (
        snr,
        secondary,
        minimum,
        primary,
        span,
        separation,
        fade,
        galactic_lat,
    ) in indexed_combinations:
        label = (
            f"snr{snr:g}_snr2{secondary:g}_ndet{minimum}_ndet1{primary}"
            f"_span{span:g}d_sep{separation:g}h_fade{fade:g}_b{galactic_lat:g}"
        )
        output_base = root / label / args.model / "simulation"
        command = [
            sys.executable,
            "ZTF_sim.py",
            "--output-base",
            str(output_base),
            "--n-transients",
            str(args.n_transients),
            "--n-runs",
            str(args.n_runs),
            "--n-processes",
            str(args.n_processes),
            "--seed",
            str(args.seed + index),
            "--model",
            args.model,
            "--snr-threshold",
            str(snr),
            "--snr-threshold-secondary",
            str(secondary),
            "--min-detections",
            str(minimum),
            "--min-detections-primary",
            str(primary),
            "--max-timespan-days",
            str(span),
            "--min-time-separation-hours",
            str(separation),
            "--min-fade-rate",
            str(fade),
            "--min-galactic-lat",
            str(galactic_lat),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
