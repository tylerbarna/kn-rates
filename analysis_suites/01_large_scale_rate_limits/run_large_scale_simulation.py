#!/usr/bin/env python3
"""Large-scale simulation runner for ZTF 8-year kilonova rate limits.

This module executes high-statistics Monte Carlo simulations ($N \\ge 10^5-10^6$
transients per realization) across three canonical kilonova models:
- Metzger: Semi-analytic 1-component model with fiducial lanthanide-free opacity
- Bu2026Fixed: 2-component numerical model (dynamical + wind) at fixed viewing angle (0.45 rad)
- Bu2026Vary: 2-component numerical model with isotropic viewing angle distribution (flat in cos iota)

The simulations interface directly with `survey_sim` and the full 8-year ZTF survey
dataset (2018-03 through 2026-03), applying standardized detection criteria:
  - Primary SNR threshold >= 5.0
  - Secondary SNR threshold >= 3.0
  - Minimum 2 positive detections
  - Qualifying timespan <= 14 days
  - Minimum time separation between first and last detection >= 3 hours
  - Fast-transient fade rate >= 0.3 mag/day
  - Galactic latitude cut |b| >= 15.0 deg
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the large-scale simulation runner."""
    parser = argparse.ArgumentParser(
        description="Run large-scale ZTF 8-year kilonova simulations for rate limit determination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Bu2026Fixed",
        choices=["Bu2026Fixed", "Bu2026Vary", "Metzger"],
        help="Kilonova model family to simulate.",
    )
    parser.add_argument(
        "--n-transients",
        type=int,
        default=10_000_000,
        help="Number of synthetic transients injected per simulation run (default: 10,000,000 for high-statistics 8-year rate limits).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of independent simulation runs to execute.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1000.0,
        help="Injected fiducial volumetric event rate in Gpc^-3 yr^-1.",
    )
    parser.add_argument(
        "--vary-rate",
        action="store_true",
        help="Randomize volumetric rate uniformly between 100 and 2000 Gpc^-3 yr^-1 per run.",
    )
    parser.add_argument(
        "--randomize-params",
        action="store_true",
        help="Sample model ejecta parameters uniformly within realistic physical bounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducible Monte Carlo generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/01_large_scale_rate_limits",
        help="Base directory for serialized JSON and metadata results.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Optional array task index (defaults to SLURM_ARRAY_TASK_ID if available).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing result files.",
    )
    parser.set_defaults(overwrite=True)
        "--z-max",
        type=float,
        default=0.15,
        help="Maximum redshift cutoff for injected transients (default: 0.15 to match ZTF horizon and prevent sample dilution).",
    )
    # Detection criteria parameter overrides
    parser.add_argument("--snr-threshold", type=float, default=5.0, help="Primary detection SNR threshold.")
    parser.add_argument("--snr-threshold-secondary", type=float, default=3.0, help="Secondary detection SNR threshold.")
    parser.add_argument("--min-detections", type=int, default=2, help="Minimum qualifying positive detections.")
    parser.add_argument("--min-detections-primary", type=int, default=1, help="Minimum detections at primary SNR.")
    parser.add_argument("--max-timespan-days", type=float, default=14.0, help="Maximum detection span in days.")
    parser.add_argument("--min-time-separation-hours", type=float, default=3.0, help="Minimum separation between first and last detections.")
    parser.add_argument("--min-fade-rate", type=float, default=0.3, help="Minimum decline rate in mag/day.")
    parser.add_argument("--min-galactic-lat", type=float, default=15.0, help="Minimum absolute galactic latitude in deg.")
    parser.add_argument(
        "--no-require-fast-transient",
        dest="require_fast_transient",
        action="store_false",
        help="Disable fast-transient rate cut filter.",
    )
    parser.set_defaults(require_fast_transient=True)
    return parser.parse_args()


def get_model_bounds(model_name: str, z_max: float = 0.15) -> Dict[str, tuple]:
    """Retrieve physical prior bounds for kilonova model parameters.

    Bounds are calibrated against numerical relativity simulations of binary neutron
    star mergers (e.g. Radice+18, Dietrich+20) to prevent unphysically sub-luminous
    ejecta masses (M < 10^-3 M_sun) that cause artificial detection volume collapse.

    Args:
        model_name: The identifier for the model family.
        z_max: Cosmological redshift horizon cutoff (default: 0.15).

    Returns:
        Dictionary mapping parameter names to (min, max) bounds.
    """
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        return {
            "log10_mej_dyn": (-2.5, -1.301),    # ~0.003 to 0.05 M_sun (BNS dynamical ejecta)
            "v_ej_dyn": (0.12, 0.35),            # 0.12c to 0.35c
            "ye_dyn": (0.15, 0.35),              # Electron fraction (lanthanide rich to poor)
            "log10_mej_wind": (-2.0, -0.550),   # ~0.01 to 0.28 M_sun (secular disk wind)
            "v_ej_wind": (0.05, 0.15),           # 0.05c to 0.15c
            "ye_wind": (0.20, 0.40),             # Lanthanide-poor wind
            "inclination_em": (0.0, math.pi / 2.0),
            "z_max": (z_max, z_max + 0.05),     # Horizon-matched to avoid volume dilution
        }
    if model_name == "Metzger":
        return {
            "mej": (0.001, 0.010),              # 0.001 to 0.010 M_sun (realistic fast ejecta)
            "vej": (0.30, 0.70),                # 0.30c to 0.70c
            "kappa": (10.0, 500.0),             # Opacity in cm^2 g^-1 (lanthanide-free to moderate)
            "z_max": (z_max, z_max + 0.05),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def build_population(
    model_name: str,
    rate: float,
    randomize_params: bool = False,
    z_max: float = 0.15,
    rng: Optional[np.random.Generator] = None,
    **overrides: Any,
):
    """Instantiate a survey_sim kilonova population configured for the specified model."""
    if rng is None:
        rng = np.random.default_rng()

    if randomize_params:
        bounds = get_model_bounds(model_name, z_max=z_max)
        params = {
            k: float(rng.uniform(low, high))
            for k, (low, high) in bounds.items()
            if k != "z_max"
        }
        params["rate"] = rate
        params["z_max"] = z_max
        if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
            params["vary_inclination"] = (model_name == "Bu2026Vary")
            if model_name == "Bu2026Fixed":
                params["inclination_em"] = 0.45  # Keep polar viewing angle fixed (iota=26 deg)
        params.update(overrides)
        print(f"  [Randomized Draw] {model_name} params: {params}")
        if model_name == "Metzger":
            return FixedMetzgerKilonovaPopulation(**params)
        return FixedBu2026KilonovaPopulation(**params)

    # Standard canonical fiducial population parameters
    if model_name == "Bu2026Fixed":
        params = dict(
            log10_mej_dyn=-1.7,
            v_ej_dyn=0.20,
            ye_dyn=0.15,
            log10_mej_wind=-1.1,
            v_ej_wind=0.10,
            ye_wind=0.35,
            inclination_em=0.45,  # ~26 degrees (polar line-of-sight)
            vary_inclination=False,
            rate=rate,
            z_max=z_max,
        )
        params.update(overrides)
        return FixedBu2026KilonovaPopulation(**params)

    if model_name == "Bu2026Vary":
        params = dict(
            log10_mej_dyn=-1.7,
            v_ej_dyn=0.20,
            ye_dyn=0.15,
            log10_mej_wind=-1.1,
            v_ej_wind=0.10,
            ye_wind=0.35,
            inclination_em=0.45,
            vary_inclination=True,  # Flat in cos(iota)
            rate=rate,
            z_max=z_max,  # Set to 0.15 (matching Fixed) to eliminate 85% Monte Carlo sample dilution
        )
        params.update(overrides)
        return FixedBu2026KilonovaPopulation(**params)

    if model_name == "Metzger":
        params = dict(
            mej=0.00126,
            vej=0.50,
            kappa=398.0,
            rate=rate,
            z_max=z_max,
        )
        params.update(overrides)
        return FixedMetzgerKilonovaPopulation(**params)

    raise ValueError(f"Unknown model: {model_name}")


def build_lightcurve_model(model_name: str):
    """Instantiate the lightcurve generator for the specified model family."""
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        return FiestaKNModel()
    if model_name == "Metzger":
        return MetzgerKNModel()
    raise ValueError(f"Unknown model: {model_name}")


def create_detection_criteria(args: argparse.Namespace) -> DetectionCriteria:
    """Create DetectionCriteria object from parsed CLI arguments."""
    return DetectionCriteria(
        snr_threshold=args.snr_threshold,
        snr_threshold_secondary=args.snr_threshold_secondary,
        min_detections=args.min_detections,
        min_detections_primary=args.min_detections_primary,
        max_timespan_days=args.max_timespan_days,
        min_time_separation_hours=args.min_time_separation_hours,
        require_fast_transient=args.require_fast_transient,
        min_rise_rate=0.0,
        min_fade_rate=args.min_fade_rate,
        min_galactic_lat=args.min_galactic_lat,
    )


def run_single_simulation(
    run_idx: int,
    args: argparse.Namespace,
    survey: SurveyStore,
    seed: int,
) -> Dict[str, Any]:
    """Execute a single Monte Carlo simulation realization and record outputs."""
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rate = args.rate if not args.vary_rate else float(np.random.uniform(100.0, 2000.0))
    pop = build_population(
        args.model,
        rate=rate,
        randomize_params=args.randomize_params,
        z_max=args.z_max,
        rng=np.random.default_rng(seed),
    )
    model_instance = build_lightcurve_model(args.model)
    det_criteria = create_detection_criteria(args)

    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": model_instance},
        detection=det_criteria,
        n_transients=args.n_transients,
        seed=seed,
    )

    t_start = time.perf_counter()
    result = pipeline.run()
    elapsed = time.perf_counter() - t_start

    task_id = args.task_id if args.task_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    task_suffix = f"_task_{task_id}" if task_id is not None else f"_pid_{os.getpid()}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fname = out_dir / f"ztf_{args.model}_{args.n_transients//1000}k{task_suffix}_run_{run_idx}_{timestamp}.json"
    save_result(result, str(fname), overwrite=args.overwrite)

    # Save detailed metadata
    metadata = {
        "model": args.model,
        "n_transients": args.n_transients,
        "run_index": run_idx,
        "task_id": str(task_id) if task_id is not None else None,
        "seed": seed,
        "z_max": args.z_max,
        "randomize_params": args.randomize_params,
        "elapsed_seconds": elapsed,
        "detection_criteria": {
            "snr_threshold": args.snr_threshold,
            "snr_threshold_secondary": args.snr_threshold_secondary,
            "min_detections": args.min_detections,
            "min_detections_primary": args.min_detections_primary,
            "max_timespan_days": args.max_timespan_days,
            "min_time_separation_hours": args.min_time_separation_hours,
            "min_fade_rate": args.min_fade_rate,
            "min_galactic_lat": args.min_galactic_lat,
            "require_fast_transient": args.require_fast_transient,
        },
    }
    Path(f"{fname}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rate_summaries = []
    for rs in result.rate_summaries:
        eff_vt = rs.effective_vt_gpc3_yr
        r95 = -math.log(0.05) / eff_vt if (math.isfinite(eff_vt) and eff_vt > 0) else float("nan")
        rate_summaries.append({
            "transient_type": rs.transient_type,
            "volumetric_rate": rs.volumetric_rate,
            "n_detected": rs.n_detected,
            "n_simulated": rs.n_simulated,
            "overall_efficiency": rs.overall_efficiency,
            "survey_duration_years": rs.survey_duration_years,
            "effective_vt_gpc3_yr": eff_vt,
            "rate_95_upper_limit": r95,
        })

    return {
        "run_idx": run_idx,
        "file": str(fname),
        "model": args.model,
        "seed": seed,
        "n_simulated": result.n_simulated,
        "n_detected": result.n_detected,
        "elapsed_seconds": elapsed,
        "rate_summaries": rate_summaries,
    }


def main():
    args = parse_args()
    print("=" * 80)
    print("ZTF 8-YEAR LARGE-SCALE KILONOVA RATE LIMIT SIMULATION")
    print(f"Model: {args.model} | Transients: {args.n_transients:,} | Runs: {args.n_runs}")
    print("=" * 80)

    print("Loading full 8-year ZTF survey dataset (2018-03 to 2026-03, NSIDE=64)...")
    t0 = time.perf_counter()
    survey = load_ztf_survey(start="201803", end="202603", nside=64)
    print(f"Survey loaded in {time.perf_counter() - t0:.2f}s:")
    print(f"  Total observations: {survey.n_observations:,}")
    print(f"  MJD range:          {survey.mjd_range[0]:.2f} - {survey.mjd_range[1]:.2f}")
    print(f"  Duration:           {survey.duration_years:.2f} years")
    print(f"  Photometric bands:  {', '.join(survey.bands)}")

    base_seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))

    results = []
    for i in range(args.n_runs):
        seed = base_seed + i * 10007
        print(f"\nExecuting run {i+1}/{args.n_runs} [Seed: {seed}]...")
        res = run_single_simulation(i, args, survey, seed)
        results.append(res)
        rs = res["rate_summaries"][0]
        print(f"  Detected: {res['n_detected']} / {res['n_simulated']} (eff: {rs['overall_efficiency']:.6f})")
        print(f"  VT_eff:   {rs['effective_vt_gpc3_yr']:.4e} Gpc^3 yr")
        print(f"  R_95%:    {rs['rate_95_upper_limit']:.1f} Gpc^-3 yr^-1")
        print(f"  Runtime:  {res['elapsed_seconds']:.2f}s")

    print("\n" + "=" * 80)
    print("SIMULATION BATCH COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
