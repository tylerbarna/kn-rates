#!/usr/bin/env python3
"""Candidate generation engine for the 'Simulate-Once, Filter-Many' criteria sweep.

This script executes large-scale Monte Carlo transient simulations under a minimally
restrictive (loose bounding) detection criteria envelope across all three canonical
kilonova models (Bu2026Fixed, Bu2026Vary, Metzger).

By saving only candidate detected lightcurves that pass the loose envelope (typically
~1,000 - 3,000 events out of 1,000,000 injections), the downstream filtering engine
can evaluate tens of thousands of criteria combinations in seconds without re-running
the computationally expensive radiative transfer and survey pointing matching steps.

Supported Execution Modes:
  1. 'fiducial': Pinned canonical AT2017gfo physical parameters (1 run per model family).
  2. 'ensemble': Multiple independent 1M-transient samples (10 - 100 samples per model),
     supporting either Monte Carlo seed variations or randomized ejecta parameter draws.
"""

import argparse
import datetime
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
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
    from survey_sim.serialization import save_result
    HAS_SURVEY_SIM = True
except ImportError:
    DetectionCriteria = None
    FixedBu2026KilonovaPopulation = None
    FixedMetzgerKilonovaPopulation = None
    MetzgerKNModel = None
    SimulationPipeline = None
    SurveyStore = None
    load_ztf_survey = None
    FiestaKNModel = None
    save_result = None
    HAS_SURVEY_SIM = False


# Canonical model choices
MODELS = ["Bu2026Fixed", "Bu2026Vary", "Metzger"]

# Minimally restrictive loose bounding criteria envelope.
# Ensures every transient that could pass ANY configuration in the 68,400-point
# grid (or ultra-fine 158,460-point grid) is captured.
LOOSE_BOUNDING_CRITERIA = {
    "snr_threshold": 3.0,               # Below grid minimum of 3.5
    "snr_threshold_secondary": 2.0,     # Matches grid minimum of 2.0
    "min_detections": 1,               # Lowest in grid (1 detection)
    "min_detections_primary": 1,       # Lowest in grid (1 detection)
    "max_timespan_days": 35.0,         # Exceeds grid maximum of 28.0 days
    "min_time_separation_hours": 0.0,  # Below grid minimum of 0.5 hours
    "min_rise_rate": 0.0,              # Inactive (0.0 mag/day)
    "min_fade_rate": 0.0,              # Inactive (0.0 mag/day)
    "min_galactic_lat": 0.0,           # Full sky (no galactic plane cut)
    "require_fast_transient": False,   # Disabled
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate loose-envelope candidate lightcurves for criteria sweeps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "Bu2026Fixed", "Bu2026Vary", "Metzger"],
        help="Model to simulate ('all' executes Bu2026Fixed, Bu2026Vary, and Metzger sequentially).",
    )
    parser.add_argument(
        "--mode",
        default="fiducial",
        choices=["fiducial", "ensemble"],
        help="Simulation mode: 'fiducial' (canonical AT2017gfo parameters) or 'ensemble' (multi-sample statistical distribution).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of independent 1M samples per model under 'ensemble' mode (recommended: 10 - 100, default: 20).",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Specific zero-based sample index to run (for Slurm array mapping).",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Global array task index mapping across models and samples (task_id = model_idx * n_samples + sample_id).",
    )
    parser.add_argument(
        "--n-transients",
        type=int,
        default=1_000_000,
        help="Number of synthetic transients injected per simulation run.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=4,
        help="Number of parallel worker processes for SimulationPipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261020,
        help="Base random seed.",
    )
    parser.add_argument(
        "--randomize-params",
        action="store_true",
        help="Sample ejecta parameters uniformly within realistic physical bounds per sample (similar to Suite 01).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/02_granular_criteria_sweep/candidates",
        help="Output directory for candidate JSON result files.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Optional high-capacity scratch directory (e.g. $TMPDIR or /scratch.global/$USER).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Overwrite existing candidate files.",
    )
    return parser.parse_args()


def get_model_bounds(model_name: str, z_max: float = 0.15) -> Dict[str, Tuple[float, float]]:
    """Return physical bounds for parameter randomization matching Suite 01."""
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        return {
            "log10_mej_dyn": (-2.5, -1.301),   # 0.003 to 0.05 M_sun
            "v_ej_dyn": (0.12, 0.35),           # 0.12c to 0.35c
            "ye_dyn": (0.10, 0.25),             # Low Ye (lanthanide rich)
            "log10_mej_wind": (-2.0, -0.699),  # 0.01 to 0.20 M_sun
            "v_ej_wind": (0.05, 0.20),          # 0.05c to 0.20c
            "ye_wind": (0.25, 0.45),            # High Ye (lanthanide poor)
            "z_max": (z_max, z_max),
        }
    elif model_name == "Metzger":
        return {
            "mej": (0.001, 0.010),              # 0.001 to 0.010 M_sun
            "vej": (0.30, 0.70),                # 0.30c to 0.70c
            "kappa": (10.0, 500.0),             # Opacity cm^2/g
            "z_max": (z_max, z_max),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def build_population_instance(
    model_name: str,
    rate: float = 1000.0,
    randomize_params: bool = False,
    rng: Optional[Any] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Build population and model instances with fiducial or randomized parameters."""
    if not HAS_SURVEY_SIM:
        return None, None, {}

    if rng is None and np is not None:
        rng = np.random.default_rng()

    z_max = 0.15 if model_name in {"Bu2026Fixed", "Bu2026Vary"} else 0.30

    if randomize_params and rng is not None:
        bounds = get_model_bounds(model_name, z_max=z_max)
        params = {k: float(rng.uniform(low, high)) for k, (low, high) in bounds.items() if k != "z_max"}
        params["rate"] = rate
        params["z_max"] = z_max
        if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
            params["vary_inclination"] = (model_name == "Bu2026Vary")
            if model_name == "Bu2026Fixed":
                params["inclination_em"] = 0.45
            pop = FixedBu2026KilonovaPopulation(**params)
            model = FiestaKNModel(survey="ztf")
        else:
            pop = FixedMetzgerKilonovaPopulation(**params)
            model = MetzgerKNModel()
        return pop, model, params

    # Canonical fiducial AT2017gfo parameters
    if model_name == "Bu2026Fixed":
        params = dict(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=False, rate=rate, z_max=0.15
        )
        pop = FixedBu2026KilonovaPopulation(**params)
        model = FiestaKNModel(survey="ztf")
    elif model_name == "Bu2026Vary":
        params = dict(
            log10_mej_dyn=-1.7, v_ej_dyn=0.20, ye_dyn=0.15,
            log10_mej_wind=-1.1, v_ej_wind=0.10, ye_wind=0.35,
            inclination_em=0.45, vary_inclination=True, rate=rate, z_max=0.15
        )
        pop = FixedBu2026KilonovaPopulation(**params)
        model = FiestaKNModel(survey="ztf")
    else:  # Metzger
        params = dict(
            mej=0.00126, vej=0.50, kappa=398.0, rate=rate, z_max=0.30
        )
        pop = FixedMetzgerKilonovaPopulation(**params)
        model = MetzgerKNModel()

    return pop, model, params


def run_candidate_simulation(
    survey: Any,
    model_name: str,
    n_transients: int,
    seed: int,
    output_file: Path,
    randomize_params: bool = False,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Execute a single loose-criteria simulation run and save candidate detections."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file = Path(f"{output_file}.metadata.json")

    if not force_rerun and output_file.exists():
        print(f"  [CACHE HIT] Candidate file already exists: {output_file}")
        try:
            data = json.loads(output_file.read_text())
            return {
                "file": str(output_file),
                "model": model_name,
                "n_simulated": data.get("n_simulated", n_transients),
                "n_candidates": len(data.get("detected_sources", [])),
                "cached": True,
            }
        except Exception:
            pass  # Fall through to re-simulate if corrupted

    if not HAS_SURVEY_SIM:
        raise RuntimeError("survey_sim is not installed in the current environment.")

    rng = np.random.default_rng(seed) if np is not None else None
    pop, model_inst, params = build_population_instance(
        model_name=model_name,
        randomize_params=randomize_params,
        rng=rng,
    )

    det = DetectionCriteria(
        snr_threshold=float(LOOSE_BOUNDING_CRITERIA["snr_threshold"]),
        snr_threshold_secondary=float(LOOSE_BOUNDING_CRITERIA["snr_threshold_secondary"]),
        min_detections=int(LOOSE_BOUNDING_CRITERIA["min_detections"]),
        min_detections_primary=int(LOOSE_BOUNDING_CRITERIA["min_detections_primary"]),
        max_timespan_days=float(LOOSE_BOUNDING_CRITERIA["max_timespan_days"]),
        min_time_separation_hours=float(LOOSE_BOUNDING_CRITERIA["min_time_separation_hours"]),
        require_fast_transient=bool(LOOSE_BOUNDING_CRITERIA["require_fast_transient"]),
        min_rise_rate=float(LOOSE_BOUNDING_CRITERIA["min_rise_rate"]),
        min_fade_rate=float(LOOSE_BOUNDING_CRITERIA["min_fade_rate"]),
        min_galactic_lat=float(LOOSE_BOUNDING_CRITERIA["min_galactic_lat"]),
    )

    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": model_inst},
        detection=det,
        n_transients=n_transients,
        seed=seed,
    )

    t0 = time.perf_counter()
    result = pipeline.run()
    elapsed = time.perf_counter() - t0

    # Save candidate result
    save_result(result, str(output_file), overwrite=True)

    # Strip heavy non-detection arrays to keep candidate files under ~2 MB
    try:
        data = json.loads(output_file.read_text())
        for src in data.get("detected_sources", []):
            src["non_detections_times"] = []
            src["non_detections_depths"] = []
            src["non_detections_bands"] = []
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    # Save metadata
    rs = result.rate_summaries[0] if getattr(result, "rate_summaries", None) else None
    meta = {
        "model": model_name,
        "n_transients": n_transients,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "params": params,
        "randomize_params": randomize_params,
        "loose_criteria": LOOSE_BOUNDING_CRITERIA,
        "n_candidates": len(result.detected_sources) if hasattr(result, "detected_sources") else 0,
        "effective_vt_gpc3_yr": getattr(rs, "effective_vt_gpc3_yr", 0.0) if rs else 0.0,
        "overall_efficiency": getattr(rs, "overall_efficiency", 0.0) if rs else 0.0,
    }
    meta_file.write_text(json.dumps(meta, indent=2) + "\n")

    n_cand = len(result.detected_sources) if hasattr(result, "detected_sources") else 0
    print(f"  [DONE] Simulated {n_transients:,} transients in {elapsed:.1f}s -> {n_cand:,} candidate detections saved to {output_file.name}")

    return {
        "file": str(output_file),
        "model": model_name,
        "n_simulated": n_transients,
        "n_candidates": n_cand,
        "elapsed_seconds": elapsed,
        "cached": False,
    }


def main():
    args = parse_args()
    target_models = MODELS if args.model == "all" else [args.model]

    # Handle Slurm global task_id mapping: task_id = model_idx * n_samples + sample_id
    if args.task_id is not None:
        model_idx = args.task_id // args.n_samples
        sample_id = args.task_id % args.n_samples
        if model_idx >= len(MODELS):
            sys.exit(f"Task ID {args.task_id} exceeds available models ({len(MODELS)}) x samples ({args.n_samples}).")
        target_models = [MODELS[model_idx]]
        args.sample_id = sample_id
        args.mode = "ensemble"
        print(f"Slurm Global Task ID {args.task_id} -> Model: {target_models[0]}, Sample: #{sample_id}/{args.n_samples}")

    out_base = Path(args.scratch_dir) if args.scratch_dir else Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SIMULATE-ONCE CANDIDATE GENERATION ENGINE")
    print(f"Models:          {target_models}")
    print(f"Mode:            {args.mode} (N_samples per model: {args.n_samples if args.mode == 'ensemble' else 1})")
    print(f"Transients/Run:  {args.n_transients:,}")
    print(f"RandomizeParams: {args.randomize_params}")
    print(f"Output Directory:{out_base}")
    print("=" * 80)

    if not HAS_SURVEY_SIM:
        print("Error: survey_sim module not found. Run under conda environment 'kn_rates_dev' or on MSI cluster.")
        sys.exit(1)

    print("Loading ZTF 8-year survey footprint (start: 201803, end: 202603, nside: 64)...")
    survey = load_ztf_survey(start="201803", end="202603", nside=64)

    for model_name in target_models:
        model_dir = out_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "fiducial":
            sample_ids = [0]
        elif args.sample_id is not None:
            sample_ids = [args.sample_id]
        else:
            sample_ids = list(range(args.n_samples))

        for sid in sample_ids:
            pt_seed = args.seed + (MODELS.index(model_name) * 100_000) + (sid * 1_007)
            if args.mode == "fiducial":
                cand_file = model_dir / f"candidates_{model_name}_fiducial.json"
            else:
                cand_file = model_dir / f"candidates_{model_name}_sample_{sid:03d}.json"

            print(f"\n[{model_name}] Generating candidate sample #{sid:03d} (seed: {pt_seed})...")
            run_candidate_simulation(
                survey=survey,
                model_name=model_name,
                n_transients=args.n_transients,
                seed=pt_seed,
                output_file=cand_file,
                randomize_params=args.randomize_params,
                force_rerun=args.force_rerun,
            )
            gc.collect()

    print("\n" + "=" * 80)
    print("Candidate generation complete across all requested models.")
    print(f"All candidate lightcurves ready in: {out_base}")
    print("Next step: Run 'filter_criteria_grid.py' to evaluate criteria across the complete grid.")
    print("=" * 80)


if __name__ == "__main__":
    main()
