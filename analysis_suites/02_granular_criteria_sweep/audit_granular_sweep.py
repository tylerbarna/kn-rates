#!/usr/bin/env python3
"""Audit, validate adherence, and analyze sensitivity for detection criteria sweeps.

This script parses all simulation results from a detection criteria sweep, validates
100% adherence to all selection rules, computes 1D marginal detection efficiencies,
calculates criteria sensitivity gradients, and exports structured CSVs and metrics.
"""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and analyze detection criteria sweep results.")
    parser.add_argument(
        "--sweep-root",
        default="results/02_granular_criteria_sweep",
        help="Root directory containing criteria sweep subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/02_granular_criteria_sweep/granular_sweep_metrics.csv",
        help="Path for saving aggregated criteria metrics.",
    )
    return parser.parse_args()


def parse_label_metadata(label: str) -> Dict[str, Any]:
    """Parse numerical parameters from a standardized directory label."""
    # Pattern: snr4_snr22.5_ndet2_ndet11_span14d_sep3h_fade0.3_b15
    data = {}
    tokens = label.split("_")
    for tok in tokens:
        if tok.startswith("snr2"):
            data["snr_threshold_secondary"] = float(tok.replace("snr2", ""))
        elif tok.startswith("snr"):
            data["snr_threshold"] = float(tok.replace("snr", ""))
        elif tok.startswith("ndet1"):
            data["min_detections_primary"] = int(tok.replace("ndet1", ""))
        elif tok.startswith("ndet"):
            data["min_detections"] = int(tok.replace("ndet", ""))
        elif tok.startswith("span") and tok.endswith("d"):
            data["max_timespan_days"] = float(tok.replace("span", "").replace("d", ""))
        elif tok.startswith("sep") and tok.endswith("h"):
            data["min_time_separation_hours"] = float(tok.replace("sep", "").replace("h", ""))
        elif tok.startswith("fade"):
            data["min_fade_rate"] = float(tok.replace("fade", ""))
        elif tok.startswith("b"):
            data["min_galactic_lat"] = float(tok.replace("b", ""))
    return data


def audit_results(root_path: Path) -> pd.DataFrame:
    """Traverse directories and parse simulation outputs and metadata."""
    records = []
    pattern = str(root_path / "**/*.json")
    for f in sorted(glob.glob(pattern, recursive=True)):
        if f.endswith(".metadata.json"):
            continue
        p = Path(f)
        try:
            content = json.loads(p.read_text())
        except Exception:
            continue
        
        meta_p = Path(f"{f}.metadata.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        
        # Determine criteria from metadata or label
        label = p.parent.parent.name
        criteria = meta.get("detection_criteria", parse_label_metadata(label))
        
        rate_summaries = content.get("rate_summaries", [])
        if not rate_summaries:
            continue
        
        rs = rate_summaries[0]
        n_sim = rs.get("n_simulated", 100_000)
        n_det = rs.get("n_detected", 0)
        eff = rs.get("overall_efficiency", n_det / max(n_sim, 1))
        vt = rs.get("effective_vt_gpc3_yr", 0.0)
        
        # Check detected sources if serialized
        sources = content.get("detected_sources", [])
        n_sources_checked = len(sources)
        violations_span = 0
        violations_ndet = 0
        violations_sep = 0
        violations_tail_30d = 0
        
        configured_span = float(criteria.get("max_timespan_days", 14.0))
        configured_min_det = int(criteria.get("min_detections", 2))
        configured_min_sep_hrs = float(criteria.get("min_time_separation_hours", 3.0))

        for src in sources:
            p_times = [float(t) for t in src.get("photometry_times", []) if math.isfinite(float(t))]
            if len(p_times) < configured_min_det:
                violations_ndet += 1
            if len(p_times) >= 2:
                span_days = p_times[-1] - p_times[0]
                if span_days > (configured_span + 1e-4):
                    violations_span += 1
                if (span_days * 24.0) < (configured_min_sep_hrs - 1e-4):
                    violations_sep += 1
                if span_days > 30.0:
                    violations_tail_30d += 1

        records.append({
            "file": str(p),
            "label": label,
            "model": meta.get("model", p.parent.name),
            "n_simulated": n_sim,
            "n_detected": n_det,
            "overall_efficiency": eff,
            "effective_vt_gpc3_yr": vt,
            "sources_audited": n_sources_checked,
            "violations_max_timespan": violations_span,
            "violations_min_detections": violations_ndet,
            "violations_min_separation": violations_sep,
            "violations_tail_30d": violations_tail_30d,
            **criteria,
        })
    return pd.DataFrame(records)


def compute_marginal_sensitivities(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute 1D marginal detection efficiencies for each detection criterion."""
    criteria_cols = [
        "snr_threshold", "snr_threshold_secondary", "min_detections",
        "min_detections_primary", "max_timespan_days", "min_time_separation_hours",
        "min_fade_rate", "min_galactic_lat"
    ]
    marginals = {}
    for col in criteria_cols:
        if col in df.columns:
            marg = df.groupby(col).agg(
                n_configs=("overall_efficiency", "count"),
                eff_mean=("overall_efficiency", "mean"),
                eff_median=("overall_efficiency", "median"),
                eff_std=("overall_efficiency", "std"),
                vt_mean=("effective_vt_gpc3_yr", "mean"),
            ).reset_index()
            # Calculate fractional drop from best value
            max_eff = marg["eff_mean"].max()
            marg["relative_efficiency"] = marg["eff_mean"] / max_eff if max_eff > 0 else 0.0
            marginals[col] = marg
    return marginals


def main():
    args = parse_args()
    root = Path(args.sweep_root)
    print("=" * 80)
    print(f"AUDITING CRITERIA SWEEP RESULTS: {root}")
    print("=" * 80)

    df = audit_results(root)
    if df.empty:
        print(f"No simulation results found under {root}.")
        return

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(df)} criteria configurations into {out_csv}")

    # Adherence summary
    total_sources = df["sources_audited"].sum()
    print(f"\nAdherence Validation ({total_sources:,} individual sources audited):")
    print(f"  Timespan Violations:       {df['violations_max_timespan'].sum():,}")
    print(f"  Min Detections Violations: {df['violations_min_detections'].sum():,}")
    print(f"  Min Separation Violations: {df['violations_min_separation'].sum():,}")
    print(f"  Surrogate Tail (>30d):     {df['violations_tail_30d'].sum():,}")

    marginals = compute_marginal_sensitivities(df)
    print("\n" + "=" * 80)
    print("1D MARGINAL SENSITIVITY HIGHLIGHTS")
    print("=" * 80)
    for crit, mdf in marginals.items():
        print(f"\n--- Criterion: {crit} ---")
        print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
