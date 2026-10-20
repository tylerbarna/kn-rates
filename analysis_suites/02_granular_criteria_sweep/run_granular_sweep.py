#!/usr/bin/env python3
"""Granular detection criteria grid sweep engine for ZTF kilonova simulations.

This module systematically evaluates how detection efficiency (epsilon) and
effective volume-time (<VT>_eff) respond to variations across a multidimensional
grid of observational and algorithmic selection cuts:
  1. Primary SNR threshold: [4.0, 5.0, 6.0]
  2. Secondary SNR threshold: [2.5, 3.0, 4.0]
  3. Minimum qualifying detections: [2, 3, 4]
  4. Minimum primary SNR detections: [1, 2]
  5. Maximum timespan window: [3.0, 7.0, 14.0, 21.0] days
  6. Minimum first-to-last separation: [1.0, 3.0, 6.0, 12.0] hours
  7. Minimum fade rate: [0.0, 0.15, 0.30, 0.50] mag/day
  8. Minimum absolute galactic latitude: [0.0, 10.0, 15.0, 20.0] deg
  9. Fast-transient requirement: [True, False]
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from survey_sim import (
    DetectionCriteria,
    FixedBu2026KilonovaPopulation,
    FixedMetzgerKilonovaPopulation,
    MetzgerKNModel,
    SimulationPipeline,
    SurveyStore,
    load_ztf_survey,
)
from survey_sim.fiesta_model import FiestaKNModel
from survey_sim.serialization import save_result


# Canonical 2,592-point grid (matching ZTFReST v2 sweep that ran in ~2 hours)
CANONICAL_2592_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 3.0, 4.0),
    "min_detections": (2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (3.0, 7.0, 14.0, 21.0),
    "min_time_separation_hours": (3.0, 12.0),
    "min_fade_rate": (0.0, 0.3, 0.6),
    "min_galactic_lat": (0.0, 15.0),
}

# Fine-grained 13,824-point grid
FINE_13824_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.5, 3.0, 4.0),
    "min_detections": (2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (3.0, 7.0, 14.0, 21.0),
    "min_time_separation_hours": (1.0, 3.0, 6.0, 12.0),
    "min_fade_rate": (0.0, 0.15, 0.30, 0.50),
    "min_galactic_lat": (0.0, 10.0, 15.0, 20.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run granular detection criteria grid sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default="results/02_granular_criteria_sweep")
    parser.add_argument("--model", default="Bu2026Fixed", choices=["Bu2026Fixed", "Bu2026Vary", "Metzger"])
    parser.add_argument("--grid-preset", default="canonical", choices=["canonical", "fine"],
                        help="Grid size: 'canonical' (2,592 points, ~2 hr on msismall) or 'fine' (13,824 points).")
    parser.add_argument("--n-transients", type=int, default=100_000)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Zero-based index to run a specific grid point under Slurm job array.",
    )
    parser.add_argument(
        "--slice-param",
        type=str,
        default=None,
        help="If set, only sweep across this single parameter holding all others at fiducial values.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without running simulation.")
    return parser.parse_args()


def get_grid_combinations(preset: str = "canonical") -> List[Dict[str, Any]]:
    """Generate all combinations in the selected multidimensional Cartesian grid."""
    grid_def = CANONICAL_2592_GRID if preset == "canonical" else FINE_13824_GRID
    keys = list(grid_def.keys())
    values_product = itertools.product(*(grid_def[k] for k in keys))
    combinations = []
    for vals in values_product:
        combinations.append(dict(zip(keys, vals)))
    return combinations


def get_fiducial_criteria() -> Dict[str, Any]:
    """Return standard ZTFReST-like fiducial criteria."""
    return {
        "snr_threshold": 5.0,
        "snr_threshold_secondary": 3.0,
        "min_detections": 2,
        "min_detections_primary": 1,
        "max_timespan_days": 14.0,
        "min_time_separation_hours": 3.0,
        "min_fade_rate": 0.30,
        "min_galactic_lat": 15.0,
        "require_fast_transient": True,
    }


def make_label(config: Dict[str, Any]) -> str:
    """Format configuration into a standardized directory label."""
    return (
        f"snr{config['snr_threshold']:g}_snr2{config['snr_threshold_secondary']:g}"
        f"_ndet{config['min_detections']}_ndet1{config['min_detections_primary']}"
        f"_span{config['max_timespan_days']:g}d_sep{config['min_time_separation_hours']:g}h"
        f"_fade{config['min_fade_rate']:g}_b{config['min_galactic_lat']:g}"
    )


def run_grid_point(
    survey: SurveyStore,
    model_name: str,
    criteria_cfg: Dict[str, Any],
    n_transients: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute simulation for a single criteria configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate population
    if model_name == "Bu2026Fixed":
        pop = FixedBu2026KilonovaPopulation(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=False, rate=1000.0, z_max=0.15
        )
        model = FiestaKNModel()
    elif model_name == "Bu2026Vary":
        pop = FixedBu2026KilonovaPopulation(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=True, rate=1000.0, z_max=0.30
        )
        model = FiestaKNModel()
    else:
        pop = FixedMetzgerKilonovaPopulation(
            mej=0.00126, vej=0.50, kappa=398.0, rate=1000.0, z_max=0.30
        )
        model = MetzgerKNModel()

    det = DetectionCriteria(
        snr_threshold=float(criteria_cfg["snr_threshold"]),
        snr_threshold_secondary=float(criteria_cfg["snr_threshold_secondary"]),
        min_detections=int(criteria_cfg["min_detections"]),
        min_detections_primary=int(criteria_cfg["min_detections_primary"]),
        max_timespan_days=float(criteria_cfg["max_timespan_days"]),
        min_time_separation_hours=float(criteria_cfg["min_time_separation_hours"]),
        require_fast_transient=criteria_cfg.get("require_fast_transient", True),
        min_rise_rate=0.0,
        min_fade_rate=float(criteria_cfg["min_fade_rate"]),
        min_galactic_lat=float(criteria_cfg["min_galactic_lat"]),
    )

    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": model},
        detection=det,
        n_transients=n_transients,
        seed=seed,
    )

    t_start = time.perf_counter()
    result = pipeline.run()
    elapsed = time.perf_counter() - t_start

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    fname = output_dir / f"sim_result_{timestamp}.json"
    save_result(result, str(fname))

    # Save metadata
    meta = {
        "model": model_name,
        "n_transients": n_transients,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "detection_criteria": criteria_cfg,
    }
    Path(f"{fname}.metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    rs = result.rate_summaries[0]
    return {
        "file": str(fname),
        "model": model_name,
        "n_simulated": result.n_simulated,
        "n_detected": result.n_detected,
        "efficiency": rs.overall_efficiency,
        "effective_vt_gpc3_yr": rs.effective_vt_gpc3_yr,
        "elapsed_seconds": elapsed,
        **criteria_cfg,
    }


def main():
    args = parse_args()
    grid_def = CANONICAL_2592_GRID if args.grid_preset == "canonical" else FINE_13824_GRID
    grid = get_grid_combinations(args.grid_preset)
    total_grid_points = len(grid)

    print("=" * 80)
    print("GRANULAR DETECTION CRITERIA GRID SWEEP ENGINE")
    print(f"Preset: {args.grid_preset} ({total_grid_points:,} points) | Model: {args.model}")
    print("=" * 80)

    if args.slice_param:
        if args.slice_param not in grid_def:
            sys.exit(f"Error: Unknown slice parameter '{args.slice_param}'. Valid options: {list(grid_def.keys())}")
        values = grid_def[args.slice_param]
        fiducial = get_fiducial_criteria()
        active_points = []
        for val in values:
            cfg = dict(fiducial)
            cfg[args.slice_param] = val
            active_points.append(cfg)
        print(f"Selected 1D Slice over '{args.slice_param}': {len(active_points)} points: {values}")
    elif args.array_index is not None:
        if not 0 <= args.array_index < total_grid_points:
            sys.exit(f"Error: --array-index must be between 0 and {total_grid_points - 1}")
        active_points = [grid[args.array_index]]
        print(f"Selected Slurm Array Task ID: {args.array_index} / {total_grid_points - 1}")
    else:
        active_points = grid
        print(f"Executing complete {total_grid_points}-point grid sweep.")

    if args.dry_run:
        print("Dry run requested. Displaying sample configurations:")
        for idx, pt in enumerate(active_points[:5]):
            print(f"  [{idx}] {make_label(pt)}")
        return

    print("Loading ZTF 8-year survey...")
    survey = load_ztf_survey(start="201803", end="202603", nside=64)

    records = []
    root = Path(args.output_root)
    for idx, cfg in enumerate(active_points):
        label = make_label(cfg)
        out_dir = root / label / args.model
        pt_seed = args.seed + (args.array_index if args.array_index is not None else idx)
        print(f"\n[{idx+1}/{len(active_points)}] Evaluating {label}...")
        res = run_grid_point(survey, args.model, cfg, args.n_transients, pt_seed, out_dir)
        records.append(res)
        print(f"  Detected: {res['n_detected']:,} / {res['n_simulated']:,} | Eff: {res['efficiency']:.6f} | Runtime: {res['elapsed_seconds']:.2f}s")

    summary_file = root / f"sweep_summary_{args.model}.csv"
    pd.DataFrame(records).to_csv(summary_file, index=False)
    print(f"\nSaved sweep summary to {summary_file}")


if __name__ == "__main__":
    main()
