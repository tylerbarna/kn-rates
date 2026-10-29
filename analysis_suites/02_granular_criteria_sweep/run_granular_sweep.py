#!/usr/bin/env python3
"""Granular detection criteria grid sweep engine for ZTF kilonova simulations.

This module systematically evaluates how detection efficiency (epsilon) and
effective volume-time (<VT>_eff) respond to variations across a multidimensional
grid of observational and algorithmic selection cuts.

All 10 parameters defined as detection criteria are included:
  1. Primary SNR threshold (snr_threshold)
  2. Secondary SNR threshold (snr_threshold_secondary)
  3. Minimum qualifying detections (min_detections)
  4. Minimum primary SNR detections (min_detections_primary)
  5. Maximum timespan window (max_timespan_days)
  6. Minimum first-to-last separation (min_time_separation_hours)
  7. Minimum rise rate (min_rise_rate)
  8. Minimum fade rate (min_fade_rate)
  9. Absolute galactic latitude cut (min_galactic_lat)
  10. Fast-transient requirement flag (require_fast_transient)

HPC MSI Cluster Scheduling Architecture:
  - Hard cluster constraint: At most 5,000 jobs submitted at one time per user.
  - Scheduler execution profile: MSI schedules and runs tasks in chunks of 20-50 concurrent jobs.
  - Architecture: Automatic task-chunking maps arbitrarily large grids into <= 5,000 Slurm array tasks.
    Each task processes a chunk of grid points sequentially, throttled to %50 concurrency.
"""

import argparse
import gc
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

try:
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
    from survey_sim.serialization import load_result, save_result
except ImportError:
    DetectionCriteria = None
    FixedBu2026KilonovaPopulation = None
    FixedMetzgerKilonovaPopulation = None
    MetzgerKNModel = None
    SimulationPipeline = None
    SurveyStore = None
    load_ztf_survey = None
    FiestaKNModel = None
    load_result = None
    save_result = None


# Expanded parameter search bounds across all 10 detection criteria
EXPANDED_CRITERIA_BOUNDS = {
    "snr_threshold": (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0),
    "snr_threshold_secondary": (2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    "min_detections": (1, 2, 3, 4, 5, 6),
    "min_detections_primary": (1, 2, 3),
    "max_timespan_days": (1.0, 3.0, 7.0, 14.0, 21.0, 28.0),
    "min_time_separation_hours": (0.5, 1.0, 3.0, 6.0, 12.0, 24.0),
    "min_rise_rate": (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    "min_fade_rate": (0.0, 0.15, 0.30, 0.50, 0.80, 1.0),
    "min_galactic_lat": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    "require_fast_transient": (True, False),
}

# Standard ZTFReST-like fiducial criteria
FIDUCIAL_CRITERIA = {
    "snr_threshold": 5.0,
    "snr_threshold_secondary": 3.0,
    "min_detections": 2,
    "min_detections_primary": 1,
    "max_timespan_days": 14.0,
    "min_time_separation_hours": 3.0,
    "min_rise_rate": 0.0,
    "min_fade_rate": 0.30,
    "min_galactic_lat": 15.0,
    "require_fast_transient": True,
}

# Legacy Canonical 2,592-point grid (8 parameters)
CANONICAL_2592_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 3.0, 4.0),
    "min_detections": (2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (3.0, 7.0, 14.0, 21.0),
    "min_time_separation_hours": (3.0, 12.0),
    "min_rise_rate": (0.0,),
    "min_fade_rate": (0.0, 0.3, 0.6),
    "min_galactic_lat": (0.0, 15.0),
    "require_fast_transient": (True,),
}

# Broad MSI-Compliant Grid: 68,400 physically valid points across all 10 criteria.
# Expanded parameter scopes across secondary SNR, detections, rise/fade rates (including 0.1 rise and 0.3 fade), and galactic lat.
# Automatically chunked into chunks of 15 points -> 4,560 tasks <= 5,000 MSI submission limit.
BROAD_GRID = {
    "snr_threshold": (3.5, 4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 2.5, 3.0, 4.0),
    "min_detections": (1, 2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (2.0, 7.0, 14.0, 28.0),
    "min_time_separation_hours": (0.5, 2.0, 6.0),
    "min_rise_rate": (0.0, 0.1, 0.5, 1.0),
    "min_fade_rate": (0.0, 0.20, 0.30, 0.50, 0.80),
    "min_galactic_lat": (0.0, 10.0, 20.0),
    "require_fast_transient": (True, False),
}

# Fine Grid: 22,464 physically valid points.
# Automatically chunked into chunks of 5 points -> 4,493 tasks <= 5,000.
FINE_GRID = {
    "snr_threshold": (4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 2.5, 3.0, 4.0),
    "min_detections": (1, 2, 3, 4),
    "min_detections_primary": (1, 2),
    "max_timespan_days": (3.0, 7.0, 14.0, 28.0),
    "min_time_separation_hours": (1.0, 6.0),
    "min_rise_rate": (0.0, 0.1, 0.5, 1.0),
    "min_fade_rate": (0.0, 0.30, 0.60),
    "min_galactic_lat": (0.0, 10.0, 20.0),
    "require_fast_transient": (True, False),
}

# Ultra-Fine Grid: 2,138,460 physically valid points across full scope.
# Maximum granular exploration across all 10 parameters with refined step sizes:
# - Primary SNR: 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 (step size 0.5)
# - Secondary SNR: 2.0, 2.5, 3.0, 3.5, 4.0 (step size 0.5)
# - Cadence Range: N_min in 1..5, N_prim in 1..3
# - Temporal Window: 2.0, 7.0, 14.0, 28.0 days
# - Asteroid Veto: 0.5, 1.0, 2.0, 3.0, 6.0 hours (includes 1.0h and 3.0h fiducial)
# - Rise Rate: 0.0, 0.1, 0.2, 0.3, 0.5, 1.0 mag/day
# - Fade Rate: 0.0 to 1.0 at 0.1 intervals (11 points: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# - Galactic Lat: 0.0, 5.0, 10.0, 15.0, 20.0 deg (step size 5.0 deg, includes 15.0 deg fiducial)
# Automatically chunked into chunks of 428 points -> 4,997 tasks <= 5,000.
ULTRA_FINE_GRID = {
    "snr_threshold": (3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
    "snr_threshold_secondary": (2.0, 2.5, 3.0, 3.5, 4.0),
    "min_detections": (1, 2, 3, 4, 5),
    "min_detections_primary": (1, 2, 3),
    "max_timespan_days": (2.0, 7.0, 14.0, 28.0),
    "min_time_separation_hours": (0.5, 1.0, 2.0, 3.0, 6.0),
    "min_rise_rate": (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
    "min_fade_rate": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    "min_galactic_lat": (0.0, 5.0, 10.0, 15.0, 20.0),
    "require_fast_transient": (True, False),
}

# Legacy Ultra-Fine Grid: 158,460 physically valid points (preserved for backward compatibility).
LEGACY_ULTRA_FINE_GRID = {
    "snr_threshold": (3.5, 4.0, 5.0, 6.0),
    "snr_threshold_secondary": (2.0, 2.5, 3.0, 3.5, 4.0),
    "min_detections": (1, 2, 3, 4, 5),
    "min_detections_primary": (1, 2, 3),
    "max_timespan_days": (2.0, 7.0, 14.0, 28.0),
    "min_time_separation_hours": (0.5, 2.0, 6.0),
    "min_rise_rate": (0.0, 0.1, 0.5, 1.0),
    "min_fade_rate": (0.0, 0.20, 0.30, 0.50, 0.80),
    "min_galactic_lat": (0.0, 10.0, 20.0),
    "require_fast_transient": (True, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run granular detection criteria grid sweep compliant with MSI Slurm limits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default="results/02_granular_criteria_sweep")
    parser.add_argument("--model", default="Bu2026Fixed", choices=["Bu2026Fixed", "Bu2026Vary", "Metzger"])
    parser.add_argument(
        "--grid-preset",
        default="ultra_fine",
        choices=["ultra_fine", "broad", "fine", "canonical", "legacy_ultra_fine", "all_slices"],
        help=(
            "Grid configuration preset: "
            "'ultra_fine' (2,138,460 points across all 10 criteria, default); "
            "'broad' (68,400 points, chunks of 15 -> 4,560 jobs <=5,000 MSI limit); "
            "'fine' (22,464 points, chunks of 5 -> 4,493 jobs); "
            "'legacy_ultra_fine' (158,460 points, chunks of 35 -> 4,528 jobs); "
            "'canonical' (2,592 legacy points); "
            "'all_slices' (union of 1D marginal slices across all 10 criteria)."
        ),
    )
    parser.add_argument("--n-transients", type=int, default=1_000_000, help="Number of synthetic transients injected (default: 1,000,000 for high statistics).")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Zero-based task index under Slurm job array (maps to a chunk of points).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=15,
        help="Number of grid points executed per task (default: 15 points per task for broad preset).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5000,
        help="Maximum Slurm array tasks allowed at submission (MSI limit: 5,000).",
    )
    parser.add_argument(
        "--slice-param",
        type=str,
        default=None,
        help="If set, only sweep across this single parameter holding all others at fiducial values.",
    )
    parser.add_argument(
        "--exclude-non-detections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Omit bulky 8-year non-detection history arrays from serialized JSONs (saves >90%% disk space; default: True).",
    )
    parser.add_argument(
        "--compact-summary-only",
        action="store_true",
        help="Save only rate summaries and detection metrics without lightcurves (~0.4 KB/point, >99.9%% disk savings).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Optional high-capacity scratch directory (e.g., $TMPDIR or /scratch.global/$USER) to write raw simulation outputs.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Force recomputation even if output JSON exists.")
    parser.add_argument("--print-plan", action="store_true", help="Print grid partitioning plan and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print sample configurations without running.")
    return parser.parse_args()


def is_physically_valid(config: Dict[str, Any]) -> bool:
    """Filter out unphysical or redundant criteria combinations."""
    # 1. Secondary SNR threshold cannot be more stringent than primary
    if config.get("snr_threshold_secondary", 0.0) > config.get("snr_threshold", 0.0):
        return False
    # 2. Number of primary detections cannot exceed total minimum detections
    if config.get("min_detections_primary", 0) > config.get("min_detections", 0):
        return False
    # 3. If fast transient check is disabled, rise and fade cuts are inactive
    if not config.get("require_fast_transient", True):
        if config.get("min_fade_rate", 0.0) != 0.0 or config.get("min_rise_rate", 0.0) != 0.0:
            return False
    # 4. If min_detections == 1, time separation, rise rate, and fade rate cannot be calculated from 1 point
    if config.get("min_detections", 2) == 1:
        if config.get("min_time_separation_hours", 0.0) > 0.5:
            return False
        if config.get("min_rise_rate", 0.0) > 0.0 or config.get("min_fade_rate", 0.0) > 0.0:
            return False
    return True


def get_fiducial_criteria() -> Dict[str, Any]:
    """Return standard ZTFReST-like fiducial criteria."""
    return dict(FIDUCIAL_CRITERIA)


def get_grid_combinations(preset: str = "broad") -> List[Dict[str, Any]]:
    """Generate all physically valid combinations in the selected multidimensional grid."""
    if preset == "canonical":
        raw_grid = CANONICAL_2592_GRID
    elif preset == "fine":
        raw_grid = FINE_GRID
    elif preset == "ultra_fine":
        raw_grid = ULTRA_FINE_GRID
    elif preset == "legacy_ultra_fine":
        raw_grid = LEGACY_ULTRA_FINE_GRID
    elif preset == "all_slices":
        fiducial = get_fiducial_criteria()
        seen = set()
        combinations = []
        for param, values in EXPANDED_CRITERIA_BOUNDS.items():
            for val in values:
                cfg = dict(fiducial)
                cfg[param] = val
                if is_physically_valid(cfg):
                    key = tuple(sorted((k, v) for k, v in cfg.items()))
                    if key not in seen:
                        seen.add(key)
                        combinations.append(cfg)
        return combinations
    else:  # "broad" (default)
        raw_grid = BROAD_GRID

    keys = list(raw_grid.keys())
    values_product = itertools.product(*(raw_grid[k] for k in keys))
    combinations = []
    for vals in values_product:
        cfg = dict(zip(keys, vals))
        if is_physically_valid(cfg):
            combinations.append(cfg)
    return combinations


def calculate_chunking(
    total_points: int,
    chunk_size: Optional[int] = None,
    max_tasks: int = 5000,
) -> Tuple[int, int]:
    """Determine chunk_size and n_tasks guaranteeing n_tasks <= max_tasks."""
    if chunk_size is not None and chunk_size > 0:
        actual_chunk = chunk_size
        n_tasks = math.ceil(total_points / actual_chunk)
        if n_tasks > max_tasks:
            actual_chunk = math.ceil(total_points / max_tasks)
            n_tasks = math.ceil(total_points / actual_chunk)
    else:
        if total_points <= max_tasks:
            actual_chunk = 1
            n_tasks = total_points
        else:
            actual_chunk = math.ceil(total_points / max_tasks)
            n_tasks = math.ceil(total_points / actual_chunk)
    return actual_chunk, n_tasks


def make_label(config: Dict[str, Any]) -> str:
    """Format configuration into a standardized directory label including all 10 criteria."""
    fast_int = 1 if config.get("require_fast_transient", True) else 0
    rise = config.get("min_rise_rate", 0.0)
    return (
        f"snr{config['snr_threshold']:g}_snr2{config['snr_threshold_secondary']:g}"
        f"_ndet{config['min_detections']}_ndet1{config['min_detections_primary']}"
        f"_span{config['max_timespan_days']:g}d_sep{config['min_time_separation_hours']:g}h"
        f"_fade{config['min_fade_rate']:g}_rise{rise:g}_fast{fast_int}_b{config['min_galactic_lat']:g}"
    )


def save_sweep_result(
    result: Any,
    filepath: Union[str, Path],
    exclude_non_detections: bool = True,
    compact_summary_only: bool = False,
) -> None:
    """Save simulation result with options to omit non-detections or full lightcurves to minimize disk usage."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if compact_summary_only:
        rate_summaries = []
        for rs in getattr(result, "rate_summaries", []):
            rate_summaries.append({
                "transient_type": getattr(rs, "transient_type", "Kilonova"),
                "volumetric_rate": getattr(rs, "volumetric_rate", 1000.0),
                "detections_per_year": getattr(rs, "detections_per_year", 0.0),
                "detections_total": getattr(rs, "detections_total", 0.0),
                "overall_efficiency": getattr(rs, "overall_efficiency", 0.0),
                "n_simulated": getattr(rs, "n_simulated", getattr(result, "n_simulated", 0)),
                "n_detected": getattr(rs, "n_detected", getattr(result, "n_detected", 0)),
                "z_max": getattr(rs, "z_max", 0.15),
                "survey_omega_sr": getattr(rs, "survey_omega_sr", 0.0),
                "survey_duration_years": getattr(rs, "survey_duration_years", 8.0),
                "effective_vt_gpc3_yr": getattr(rs, "effective_vt_gpc3_yr", 0.0),
            })
        payload = {
            "type": "SimulationResult",
            "n_simulated": getattr(result, "n_simulated", 0),
            "n_detected": getattr(result, "n_detected", 0),
            "rate_summaries": rate_summaries,
            "detected_sources": [],
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        return

    # Standard serialization
    if save_result is not None:
        save_result(result, str(filepath), overwrite=True)
        if exclude_non_detections:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for src in data.get("detected_sources", []):
                    src["non_detections_times"] = []
                    src["non_detections_depths"] = []
                    src["non_detections_bands"] = []
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass


def run_grid_point(
    survey: Any,
    model_name: str,
    criteria_cfg: Dict[str, Any],
    n_transients: int,
    seed: int,
    output_dir: Path,
    force_rerun: bool = False,
    exclude_non_detections: bool = True,
    compact_summary_only: bool = False,
) -> Dict[str, Any]:
    """Execute simulation for a single criteria configuration with resumption support."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing result file for resumption
    if not force_rerun:
        existing_results = sorted(output_dir.glob("sim_result_*.json"))
        existing_results = [f for f in existing_results if not f.name.endswith(".metadata.json")]
        if existing_results:
            latest = existing_results[-1]
            try:
                res_data = json.loads(latest.read_text())
                rs = res_data.get("rate_summaries", [{}])[0]
                meta_file = Path(f"{latest}.metadata.json")
                elapsed = 0.0
                if meta_file.exists():
                    try:
                        elapsed = json.loads(meta_file.read_text()).get("elapsed_seconds", 0.0)
                    except Exception:
                        pass
                return {
                    "file": str(latest),
                    "model": model_name,
                    "n_simulated": res_data.get("n_simulated", n_transients),
                    "n_detected": res_data.get("n_detected", rs.get("n_detected", 0)),
                    "efficiency": rs.get("overall_efficiency", 0.0),
                    "effective_vt_gpc3_yr": rs.get("effective_vt_gpc3_yr", 0.0),
                    "elapsed_seconds": elapsed,
                    "cached": True,
                    **criteria_cfg,
                }
            except Exception:
                pass  # Fall through to simulate if file is corrupt

    if SimulationPipeline is None or load_ztf_survey is None:
        raise RuntimeError("survey_sim is not installed in the current environment.")

    # Instantiate population
    if model_name == "Bu2026Fixed":
        pop = FixedBu2026KilonovaPopulation(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=False, rate=1000.0, z_max=0.15
        )
        model = FiestaKNModel(survey="ztf")
    elif model_name == "Bu2026Vary":
        pop = FixedBu2026KilonovaPopulation(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=True, rate=1000.0, z_max=0.30
        )
        model = FiestaKNModel(survey="ztf")
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
        require_fast_transient=bool(criteria_cfg.get("require_fast_transient", True)),
        min_rise_rate=float(criteria_cfg.get("min_rise_rate", 0.0)),
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
    save_sweep_result(
        result=result,
        filepath=fname,
        exclude_non_detections=exclude_non_detections,
        compact_summary_only=compact_summary_only,
    )

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
        "cached": False,
        **criteria_cfg,
    }


def main():
    args = parse_args()
    grid = get_grid_combinations(args.grid_preset)
    total_points = len(grid)

    chunk_size, n_tasks = calculate_chunking(
        total_points=total_points,
        chunk_size=args.chunk_size,
        max_tasks=args.max_tasks,
    )

    if args.print_plan:
        print("=" * 80)
        print("MSI HPC JOB ARRAY PARTITIONING PLAN")
        print("=" * 80)
        print(f"Grid Preset:            {args.grid_preset}")
        print(f"Total Valid Grid Points:{total_points:,}")
        print(f"Max Tasks Threshold:    {args.max_tasks:,} (MSI limit: 5,000)")
        print(f"Points Per Task (Chunk):{chunk_size}")
        print(f"Total Array Tasks:      {n_tasks:,}")
        print(f"Slurm Array Directive:  #SBATCH --array=0-{n_tasks - 1}")
        print(f"Scheduler Throttling:   Unthrottled (Slurm cluster queue manages MSI node execution)")
        print(f"Est. Job Time Per Task: ~{chunk_size * 2:.1f} minutes (100k sources) / ~{chunk_size * 40:.1f} minutes (1M sources)")
        print("=" * 80)
        return

    print("=" * 80)
    print("GRANULAR DETECTION CRITERIA GRID SWEEP ENGINE (ALL 10 CRITERIA)")
    print(f"Preset: {args.grid_preset} ({total_points:,} valid points) | Model: {args.model}")
    print(f"Chunk Size: {chunk_size} points/task | Slurm Tasks: {n_tasks:,} <= {args.max_tasks:,}")
    print("=" * 80)

    if args.slice_param:
        if args.slice_param not in EXPANDED_CRITERIA_BOUNDS:
            sys.exit(
                f"Error: Unknown slice parameter '{args.slice_param}'. Valid options: {list(EXPANDED_CRITERIA_BOUNDS.keys())}"
            )
        values = EXPANDED_CRITERIA_BOUNDS[args.slice_param]
        fiducial = get_fiducial_criteria()
        active_points = []
        for val in values:
            cfg = dict(fiducial)
            cfg[args.slice_param] = val
            if is_physically_valid(cfg):
                active_points.append(cfg)
        print(f"Selected 1D Slice over '{args.slice_param}': {len(active_points)} points: {values}")
    elif args.array_index is not None:
        if not 0 <= args.array_index < n_tasks:
            sys.exit(f"Error: --array-index must be between 0 and {n_tasks - 1} (chunk_size={chunk_size})")
        start_idx = args.array_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_points)
        active_points = grid[start_idx:end_idx]
        print(
            f"Selected Slurm Array Task ID: {args.array_index} / {n_tasks - 1} "
            f"(Grid slice [{start_idx}:{end_idx}], {len(active_points)} points)"
        )
    else:
        active_points = grid
        print(f"Executing complete {total_points}-point grid sweep locally across {n_tasks} chunk(s).")

    if args.dry_run:
        print(f"Dry run requested. Displaying sample configurations (showing up to 5 of {len(active_points)}):")
        for idx, pt in enumerate(active_points[:5]):
            print(f"  [{idx}] {make_label(pt)}")
        return

    if load_ztf_survey is None:
        sys.exit("Error: survey_sim module not found. Run under conda environment 'kn_rates_dev' or on MSI.")

    print("Loading ZTF 8-year survey...")
    survey = load_ztf_survey(start="201803", end="202603", nside=64)

    records = []
    # If scratch directory specified, write heavy simulation files to scratch to preserve home quota
    scratch_root = Path(args.scratch_dir) if args.scratch_dir else None
    sim_root = scratch_root if scratch_root else Path(args.output_root)
    proj_root = Path(args.output_root)
    proj_root.mkdir(parents=True, exist_ok=True)
    if scratch_root:
        scratch_root.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed + (args.array_index * 1000 if args.array_index is not None else 0)

    for idx, cfg in enumerate(active_points):
        label = make_label(cfg)
        out_dir = sim_root / label / args.model
        pt_seed = base_seed + idx
        print(f"\n[{idx + 1}/{len(active_points)}] Evaluating {label}...")
        res = run_grid_point(
            survey=survey,
            model_name=args.model,
            criteria_cfg=cfg,
            n_transients=args.n_transients,
            seed=pt_seed,
            output_dir=out_dir,
            force_rerun=args.force_rerun,
            exclude_non_detections=args.exclude_non_detections,
            compact_summary_only=args.compact_summary_only,
        )
        records.append(res)
        status = "CACHED" if res.get("cached") else f"{res['elapsed_seconds']:.2f}s"
        print(
            f"  Detected: {res['n_detected']:,} / {res['n_simulated']:,} | "
            f"Eff: {res['efficiency']:.6f} | "
            f"VT: {res['effective_vt_gpc3_yr']:.4e} | Status: {status}"
        )
        del res
        gc.collect()

    # Save summary
    if pd is not None:
        task_suffix = f"_task{args.array_index}" if args.array_index is not None else ""
        df = pd.DataFrame(records)
        summary_file = proj_root / f"sweep_summary_{args.model}{task_suffix}.csv"
        df.to_csv(summary_file, index=False)
        print(f"\nSaved sweep summary to {summary_file}")
        if scratch_root and scratch_root != proj_root:
            scratch_summary = scratch_root / f"sweep_summary_{args.model}{task_suffix}.csv"
            df.to_csv(scratch_summary, index=False)


if __name__ == "__main__":
    main()
