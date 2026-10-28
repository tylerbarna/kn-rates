#!/usr/bin/env python3
"""Aggregate large-scale kilonova simulation outputs and compute ZTF rate limits.

This script parses serialized simulation JSON outputs across model families
(`Metzger`, `Bu2026Fixed`, `Bu2026Vary`), calculates the Poisson upper limits:
    R_{CL} = -ln(1 - CL) / <VT>_eff

It also calculates duration-scaled rate limits to directly compare the 8-year
ZTF baseline (8.01 years) against the historical 23-month (1.92 years)
benchmark of Andreoni et al. (2020), arXiv:2008.00008.
"""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ANDREONI_FIXED_95 = 1775.0      # Kasen/Bulla fixed-parameter 95% CL limit
ANDREONI_ISOTROPIC_95 = 4029.0   # Bulla viewing-angle averaged (isotropic) 95% CL limit
ANDREONI_DURATION_YRS = 23.0 / 12.0  # 1.917 years (March 2018 - Feb 2020)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate kilonova rate limits from simulation results.")
    parser.add_argument(
        "--glob",
        dest="pattern",
        default="results/**/**/*.json",
        help="Glob pattern for result JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/01_large_scale_rate_limits/rate_limit_summary.csv",
        help="Path to save the aggregated summary table.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy marathon results from results/marathon/.",
    )
    return parser.parse_args()


def extract_model_from_path(file_path: str) -> str:
    """Infer the model identifier from file path or contents."""
    if "Metzger" in file_path:
        return "Metzger"
    if "Bu2026Fixed" in file_path or "ztf_fixed" in file_path:
        return "Bu2026Fixed"
    if "Bu2026Vary" in file_path:
        return "Bu2026Vary"
    return "Unknown"


def process_file(path: Path) -> List[Dict[str, Any]]:
    """Extract rate summary records from a single result file."""
    if path.name.endswith(".metadata.json"):
        return []
    
    try:
        content = json.loads(path.read_text())
    except Exception:
        return []

    # Check for metadata
    meta_path = Path(f"{path}.metadata.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    model = meta.get("model", extract_model_from_path(str(path)))
    rate_summaries = content.get("rate_summaries", []) if isinstance(content, dict) else (content if isinstance(content, list) else [])
    
    rows = []
    for rs in rate_summaries:
        eff_vt = float(rs.get("effective_vt_gpc3_yr", float("nan")))
        if not math.isfinite(eff_vt) or eff_vt <= 0:
            continue
        
        duration = float(rs.get("survey_duration_years", 8.0057))
        r90 = -math.log(1.0 - 0.90) / eff_vt
        r95 = -math.log(1.0 - 0.95) / eff_vt
        r99 = -math.log(1.0 - 0.99) / eff_vt
        
        # Scaling to 1.92 years equivalent: R_scaled = R_8yr * (T_8yr / T_1.92yr)
        r95_duration_scaled = r95 * (duration / ANDREONI_DURATION_YRS)

        n_det = int(rs.get("n_detected", content.get("n_detected", 0)))
        
        # Reference Andreoni benchmark forward-projected to 8 years for direct physical comparison
        if model == "Bu2026Fixed":
            andreoni_proj_8yr = ANDREONI_FIXED_95 * (ANDREONI_DURATION_YRS / duration)
        elif model == "Bu2026Vary":
            andreoni_proj_8yr = ANDREONI_ISOTROPIC_95 * (ANDREONI_DURATION_YRS / duration)
        else:
            andreoni_proj_8yr = 1000.0 * (ANDREONI_DURATION_YRS / duration)

        rows.append({
            "file": str(path),
            "model": model,
            "transient_type": rs.get("transient_type", "Kilonova"),
            "n_simulated": rs.get("n_simulated", content.get("n_simulated", 1_000_000)),
            "n_detected": n_det,
            "overall_efficiency": rs.get("overall_efficiency", 0.0),
            "survey_duration_years": duration,
            "effective_vt_gpc3_yr": eff_vt,
            "rate_90_upper_limit": r90,
            "rate_95_upper_limit": r95,
            "rate_99_upper_limit": r99,
            "rate_95_duration_scaled_1_92yr": r95_duration_scaled,
            "andreoni_projected_8yr_r95": andreoni_proj_8yr,
            "is_poisson_sparse": n_det < 5,
            "ratio_to_andreoni_fixed_1775": r95_duration_scaled / ANDREONI_FIXED_95,
            "ratio_to_andreoni_isotropic_4029": r95_duration_scaled / ANDREONI_ISOTROPIC_95,
        })
    return rows


def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern, recursive=True))
    rows = []
    for f in files:
        if "metadata" in f:
            continue
        if not args.include_legacy and "results/marathon" in f and "01_large_scale" in args.pattern:
            continue
        rows.extend(process_file(Path(f)))

    if not rows:
        # Fall back to existing benchmark CSV if available
        bench_csv = Path("results/rate_limit_benchmark.csv")
        if bench_csv.exists():
            print(f"No new JSON files matched pattern '{args.pattern}'. Ingesting {bench_csv}...")
            df = pd.read_csv(bench_csv)
            df["model"] = df["file"].apply(extract_model_from_path)
            rows = df.to_dict(orient="records")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid simulation summaries found.")
        return

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} simulation records to {out_csv}")

    print("\n" + "=" * 95)
    print("ZTF 8-YEAR KILONOVA RATE LIMIT BENCHMARK SUMMARY")
    print("=" * 95)
    
    grouped = df.groupby("model")
    summary_table = grouped.agg(
        n_runs=("effective_vt_gpc3_yr", "count"),
        n_detected_mean=("n_detected", "mean"),
        efficiency_mean=("overall_efficiency", "mean"),
        vt_eff_mean=("effective_vt_gpc3_yr", "mean"),
        vt_eff_median=("effective_vt_gpc3_yr", "median"),
        r95_8yr_median=("rate_95_upper_limit", "median"),
        r95_scaled_median=("rate_95_duration_scaled_1_92yr", "median"),
        sparse_runs=("is_poisson_sparse", "sum"),
    )
    print(summary_table.to_string())
    print("-" * 95)
    print("Direct 8-Year Physical Comparison (raw R_95 vs forward-projected Andreoni 2020):")
    print(f"  Fixed/Polar (Bu2026Fixed) : {ANDREONI_FIXED_95 * (ANDREONI_DURATION_YRS / 8.0057):.1f} Gpc^-3 yr^-1 (Andreoni 8-yr projected benchmark)")
    print(f"  Isotropic (Bu2026Vary)    : {ANDREONI_ISOTROPIC_95 * (ANDREONI_DURATION_YRS / 8.0057):.1f} Gpc^-3 yr^-1 (Andreoni 8-yr projected benchmark)")
    print("-" * 95)
    print(f"Historical 23-Month Benchmarks (Andreoni 2020 at 1.92 yr baseline):")
    print(f"  Fixed:     {ANDREONI_FIXED_95:.1f} Gpc^-3 yr^-1")
    print(f"  Isotropic: {ANDREONI_ISOTROPIC_95:.1f} Gpc^-3 yr^-1")
    print("=" * 95)


if __name__ == "__main__":
    main()
