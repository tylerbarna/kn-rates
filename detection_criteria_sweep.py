#!/usr/bin/env python3
"""Launch reproducible ZTF_sim runs over a small detection-criteria grid."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DEFAULT_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "min_detections": (2, 3, 4),
    "max_timespan_days": (7.0, 14.0, 21.0),
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
    parser.add_argument("--model", default="Bu2026Fixed", choices=("Bu2026Fixed", "Bu2026Vary", "Metzger"))
    parser.add_argument("--snr-thresholds", default="4,5,6")
    parser.add_argument("--min-detections-values", default="2,3,4")
    parser.add_argument("--max-timespan-values", default="7,14,21")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    combinations = itertools.product(
        parse_values(args.snr_thresholds, float),
        parse_values(args.min_detections_values, int),
        parse_values(args.max_timespan_values, float),
    )
    for index, (snr, minimum, span) in enumerate(combinations):
        label = f"snr{snr:g}_ndet{minimum}_span{span:g}d"
        output_base = root / label / args.model / "simulation"
        command = [
            sys.executable, "ZTF_sim.py",
            "--output-base", str(output_base),
            "--n-transients", str(args.n_transients),
            "--n-runs", str(args.n_runs),
            "--n-processes", str(args.n_processes),
            "--seed", str(args.seed + index),
            "--model", args.model,
            "--snr-threshold", str(snr),
            "--min-detections", str(minimum),
            "--max-timespan-days", str(span),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
