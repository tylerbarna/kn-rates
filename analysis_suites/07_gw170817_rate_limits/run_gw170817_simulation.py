#!/usr/bin/env python3
"""Run large-scale ZTF 8-year kilonova simulations with pinned GW170817 parameters.

This module simulates an ensemble of synthetic kilonovae whose physical ejecta
parameters and geometries are strictly pinned to commonly accepted values for
the canonical binary neutron star merger GW170817 (AT2017gfo).

Evaluated Model Archetypes:
1. GW170817_Bu2026_FixedView:
   Multi-component numerical radiative transfer surrogate (Fiesta / Bulla 2019/2026)
   with tuned GW170817 ejecta:
     - log10(M_dyn / M_sun) = -1.80 (~0.016 M_sun)
     - v_dyn = 0.20 c
     - Y_e,dyn = 0.15 (lanthanide-rich tidal dynamical component)
     - log10(M_wind / M_sun) = -1.10 (~0.079 M_sun)
     - v_wind = 0.10 c
     - Y_e,wind = 0.35 (lanthanide-poor disk wind component)
     - Inclination iota = 0.45 rad (~25.8 deg), matching HST astrometry and VLBI constraints.
   Citations: Bulla (2019), Anand et al. (2023), Dietrich et al. (2020), Andreoni et al. (2020).

2. GW170817_Bu2026_Isotropic:
   Exact same GW170817 intrinsic ejecta masses and electron fractions as above,
   but observed at an isotropic viewing angle distribution (uniform in cos iota).
   Isolates the viewing angle suppression effect under identical ejecta assumptions.

3. GW170817_Metzger_Blue:
   Semi-analytic single-component model (Metzger 2017/2019) calibrated to the
   prominent early-time optical peak of AT2017gfo:
     - M_ej = 0.020 M_sun
     - v_ej = 0.25 c
     - kappa = 0.50 cm^2 g^-1 (lanthanide-free opacity)
   Citations: Villar et al. (2017), Cowperthwaite et al. (2017), Metzger (2017, 2019).

4. GW170817_Metzger_Mean:
   Semi-analytic single-component model calibrated to the combined / mass-weighted
   ejecta of AT2017gfo with moderate effective opacity:
     - M_ej = 0.040 M_sun
     - v_ej = 0.15 c
     - kappa = 1.00 cm^2 g^-1 (mixed/moderate lanthanide opacity)
   Citations: Villar et al. (2017), Kasen et al. (2017), Metzger (2019).
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


# Canonical parameters pinned to GW170817 literature constraints
GW170817_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "GW170817_Bu2026_FixedView": {
        "model_family": "Bu2026",
        "log10_mej_dyn": -1.80,     # ~0.0158 M_sun dynamical ejecta
        "v_ej_dyn": 0.20,           # 0.20c
        "ye_dyn": 0.15,             # Lanthanide-rich equatorial tidal component
        "log10_mej_wind": -1.10,    # ~0.0794 M_sun secular post-merger disk wind
        "v_ej_wind": 0.10,          # 0.10c
        "ye_wind": 0.35,            # Lanthanide-poor polar/disk wind component
        "inclination_em": 0.45,     # ~25.8 deg (HST astrometry & VLBI superluminal jet constraint)
        "vary_inclination": False,
        "description": "2-component numerical model pinned to GW170817 observed at fixed polar line-of-sight (iota=26 deg)",
        "citations": [
            "Bulla (2019, MNRAS 489:5037, arXiv:1906.04205)",
            "Anand et al. (2023, arXiv:2307.11080)",
            "Dietrich et al. (2020, Science 370:1450, arXiv:2002.11355)",
            "Andreoni et al. (2020, ApJ 904:155, arXiv:2008.00008)",
        ],
    },
    "GW170817_Bu2026_Isotropic": {
        "model_family": "Bu2026",
        "log10_mej_dyn": -1.80,
        "v_ej_dyn": 0.20,
        "ye_dyn": 0.15,
        "log10_mej_wind": -1.10,
        "v_ej_wind": 0.10,
        "ye_wind": 0.35,
        "inclination_em": 0.45,
        "vary_inclination": True,   # Uniform distribution in cos(iota)
        "description": "2-component numerical model with GW170817 ejecta observed at isotropic (angle-averaged) orientations",
        "citations": [
            "Bulla (2019, MNRAS 489:5037, arXiv:1906.04205)",
            "Andreoni et al. (2020, ApJ 904:155, arXiv:2008.00008)",
        ],
    },
    "GW170817_Metzger_Blue": {
        "model_family": "Metzger",
        "mej": 0.020,               # 0.020 M_sun fast lanthanide-free ejecta
        "vej": 0.25,                # 0.25c
        "kappa": 0.50,              # 0.50 cm^2 g^-1 (lanthanide-free opacity)
        "description": "Semi-analytic 1-component model tuned to early blue optical peak of GW170817 (lanthanide-free)",
        "citations": [
            "Villar et al. (2017, ApJL 851:L21, arXiv:1710.11576)",
            "Cowperthwaite et al. (2017, ApJL 848:L17, arXiv:1710.05840)",
            "Metzger (2017, Living Rev Relativ 23:1, arXiv:1910.01617)",
        ],
    },
    "GW170817_Metzger_Mean": {
        "model_family": "Metzger",
        "mej": 0.040,               # 0.040 M_sun total mass-weighted ejecta
        "vej": 0.15,                # 0.15c
        "kappa": 1.00,              # 1.00 cm^2 g^-1 (moderate lanthanide opacity)
        "description": "Semi-analytic 1-component model tuned to total mass-weighted average ejecta of GW170817",
        "citations": [
            "Villar et al. (2017, ApJL 851:L21, arXiv:1710.11576)",
            "Kasen et al. (2017, Nature 551:80, arXiv:1710.05463)",
            "Metzger (2019, Living Rev Relativ 23:1, arXiv:1910.01617)",
        ],
    },
}

# Alias mapping for command-line convenience
MODEL_ALIASES: Dict[str, str] = {
    "gw170817_bu2026_fixedview": "GW170817_Bu2026_FixedView",
    "gw170817_bu2026_fixed": "GW170817_Bu2026_FixedView",
    "bu2026fixed": "GW170817_Bu2026_FixedView",
    "bu2026_fixed": "GW170817_Bu2026_FixedView",
    "gw170817_bu2026_isotropic": "GW170817_Bu2026_Isotropic",
    "bu2026isotropic": "GW170817_Bu2026_Isotropic",
    "bu2026_vary": "GW170817_Bu2026_Isotropic",
    "bu2026vary": "GW170817_Bu2026_Isotropic",
    "gw170817_metzger_blue": "GW170817_Metzger_Blue",
    "metzgerblue": "GW170817_Metzger_Blue",
    "metzger_blue": "GW170817_Metzger_Blue",
    "gw170817_metzger_mean": "GW170817_Metzger_Mean",
    "metzgermean": "GW170817_Metzger_Mean",
    "metzger_mean": "GW170817_Metzger_Mean",
    "metzger": "GW170817_Metzger_Blue",
}


def canonicalize_model_name(raw_name: str) -> str:
    """Normalize model string to canonical identifier."""
    cleaned = raw_name.strip().lower()
    if cleaned in MODEL_ALIASES:
        return MODEL_ALIASES[cleaned]
    for canon in GW170817_PARAMETERS:
        if canon.lower() == cleaned:
            return canon
    raise ValueError(
        f"Unsupported model '{raw_name}'. Available canonical models: {list(GW170817_PARAMETERS.keys())}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the GW170817 simulation runner."""
    parser = argparse.ArgumentParser(
        description="Run ZTF 8-year kilonova rate limits simulations pinned to GW170817 parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GW170817_Bu2026_FixedView",
        help=f"Model archetype to simulate. Choices: {list(GW170817_PARAMETERS.keys())}",
    )
    parser.add_argument(
        "--n-transients",
        type=int,
        default=1_000_000,
        help="Number of synthetic transients injected per simulation run.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of independent simulation realizations.",
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible Monte Carlo generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/07_gw170817_rate_limits",
        help="Directory where results and metadata are serialized.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Optional array task index (defaults to SLURM_ARRAY_TASK_ID).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing result files.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--z-max",
        type=float,
        default=0.15,
        help="Redshift cutoff (default: 0.15 to avoid sample dilution while enclosing ZTF horizon).",
    )
    # Standard detection criteria matching Suite 01
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


def build_gw170817_population(
    canonical_model: str,
    rate: float = 1000.0,
    z_max: float = 0.15,
    **overrides: Any,
):
    """Instantiate the population generator with pinned GW170817 parameters."""
    cfg = dict(GW170817_PARAMETERS[canonical_model])
    family = cfg.pop("model_family")
    cfg.pop("description", None)
    cfg.pop("citations", None)

    cfg["rate"] = rate
    cfg["z_max"] = z_max
    cfg.update(overrides)

    if family == "Bu2026":
        return FixedBu2026KilonovaPopulation(**cfg)
    elif family == "Metzger":
        return FixedMetzgerKilonovaPopulation(**cfg)
    else:
        raise ValueError(f"Unknown model family: {family}")


def build_gw170817_lightcurve_model(canonical_model: str):
    """Instantiate the lightcurve generator for the specified model family."""
    family = GW170817_PARAMETERS[canonical_model]["model_family"]
    if family == "Bu2026":
        return FiestaKNModel(survey="ztf")
    elif family == "Metzger":
        return MetzgerKNModel()
    else:
        raise ValueError(f"Unknown model family: {family}")


def create_detection_criteria(args: argparse.Namespace) -> DetectionCriteria:
    """Build the standardized ZTF observational detection criteria."""
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


def run_single_realization(
    run_idx: int,
    args: argparse.Namespace,
    canonical_model: str,
    survey: SurveyStore,
    seed: int,
) -> Dict[str, Any]:
    """Execute a single Monte Carlo realization and serialize outputs."""
    out_dir = Path(args.output_dir) / canonical_model
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = build_gw170817_population(canonical_model, rate=args.rate, z_max=args.z_max)
    lc_model = build_gw170817_lightcurve_model(canonical_model)
    det_criteria = create_detection_criteria(args)

    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": lc_model},
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
    fname = out_dir / f"ztf_{canonical_model}_{args.n_transients//1000}k{task_suffix}_run_{run_idx}_{timestamp}.json"
    save_result(result, str(fname), overwrite=args.overwrite)

    # Detailed metadata recording exact parameters, citations, and execution context
    model_meta = GW170817_PARAMETERS[canonical_model]
    metadata = {
        "model": canonical_model,
        "model_family": model_meta["model_family"],
        "parameters": {k: v for k, v in model_meta.items() if k not in {"description", "citations"}},
        "description": model_meta["description"],
        "citations": model_meta["citations"],
        "n_transients": args.n_transients,
        "run_index": run_idx,
        "task_id": str(task_id) if task_id is not None else None,
        "seed": seed,
        "z_max": args.z_max,
        "rate": args.rate,
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
            "n_simulated": rs.n_simulated,
            "n_detected": rs.n_detected,
            "overall_efficiency": rs.overall_efficiency,
            "effective_vt_gpc3_yr": eff_vt,
            "rate_95_upper_limit": r95,
            "rate_95_zero_events": r95,
        })

    summary_entry = {
        "file": str(fname),
        "model": canonical_model,
        "seed": seed,
        "task_id": str(task_id) if task_id is not None else None,
        "n_simulated": result.n_simulated,
        "n_detected": result.n_detected,
        "efficiency": result.n_detected / max(result.n_simulated, 1),
        "elapsed_seconds": elapsed,
        "rate_summaries": rate_summaries,
    }
    return summary_entry


def main():
    args = parse_args()
    canonical_model = canonicalize_model_name(args.model)

    print("=" * 80)
    print("SUITE 07: FIXED-PARAMETER GW170817 KILONOVA SIMULATION RUNNER")
    print("=" * 80)
    print(f"Model Archetype:  {canonical_model}")
    print(f"Description:      {GW170817_PARAMETERS[canonical_model]['description']}")
    print(f"Transients/Run:   {args.n_transients:,}")
    print(f"Realization Runs: {args.n_runs}")
    print(f"Redshift Cutoff:  z_max = {args.z_max}")
    print(f"Base Seed:        {args.seed}")
    print(f"Output Directory: {args.output_dir}/{canonical_model}")
    print("Parameters:")
    for k, v in GW170817_PARAMETERS[canonical_model].items():
        if k not in {"citations", "description"}:
            print(f"  - {k}: {v}")
    print("=" * 80)

    print("Loading ZTF 8-year survey observations (201803 to 202603)...")
    survey = load_ztf_survey(start="201803", end="202603", nside=64)
    print(f"Loaded survey: {survey.n_observations:,} observations across {survey.duration_years:.2f} years.")

    base_seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2**31
    task_id_int = int(args.task_id) if (args.task_id is not None and args.task_id.isdigit()) else 0

    summaries = []
    for r in range(args.n_runs):
        run_seed = (base_seed + task_id_int * 10007 + r * 9973) % 2**31
        print(f"\n[Realization {r+1}/{args.n_runs}] Starting run with seed {run_seed}...")
        summary = run_single_realization(r, args, canonical_model, survey, seed=run_seed)
        summaries.append(summary)
        print(f"  --> Completed in {summary['elapsed_seconds']:.2f}s | "
              f"Detected: {summary['n_detected']:,}/{summary['n_simulated']:,} "
              f"({summary['efficiency']*100:.3f}%)")
        for rs in summary["rate_summaries"]:
            print(f"      VT_eff: {rs['effective_vt_gpc3_yr']:.4e} Gpc^3 yr | "
                  f"R_95%: {rs['rate_95_upper_limit']:.1f} Gpc^-3 yr^-1")

    print("\n" + "=" * 80)
    print(f"SUITE 07 COMPLETED: {len(summaries)} realization(s) serialized.")
    print("=" * 80)


if __name__ == "__main__":
    main()
