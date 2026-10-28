#!/usr/bin/env python3
"""Compare detection performance: Broad Parameter Priors vs. AT2017gfo-Like Kilonovae.

This module evaluates how the observational detection criteria interact with:
  1. Broad-Swath Population Priors:
     Continuous sampling across wide ejecta mass ranges (10^-4 to 0.28 M_sun),
     lanthanide fractions (Y_e 0.15 to 0.40), expansion velocities (0.05c to 0.35c),
     and viewing angles (polar to edge-on).
  2. AT2017gfo-Like Fiducial Benchmark:
     Tuned archetype reproducing GW170817 optical/NIR lightcurves:
       - M_ej,dyn = 0.016 M_sun, v_dyn = 0.20c, Y_e,dyn = 0.15
       - M_ej,wind = 0.079 M_sun, v_wind = 0.10c, Y_e,wind = 0.35
       - Inclination iota = 0.45 rad (~26 deg)

Key Scientific Questions Addressed:
  - Why is AT2017gfo detection efficiency ~3-5x higher than population-averaged efficiency?
  - How much of this difference is driven by viewing angle bias (polar vs. isotropic)
    versus ejecta mass/luminosity distribution?
  - How do the inferred volumetric rate limits R_95% diverge between archetype and population assumptions?
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from survey_sim import (
    DetectionCriteria,
    FixedBu2026KilonovaPopulation,
    SimulationPipeline,
    SurveyStore,
    load_ztf_survey,
)
from survey_sim.fiesta_model import FiestaKNModel
from survey_sim.serialization import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparison between broad prior distributions and AT2017gfo-like kilonovae.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-transients", type=int, default=100_000, help="Transients per simulation run.")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of Monte Carlo realizations per configuration.")
    parser.add_argument("--output-dir", default="results/03_prior_vs_at2017gfo")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing result files.",
    )
    parser.set_defaults(overwrite=True)
    return parser.parse_args()


def get_criteria_regimes() -> Dict[str, Dict[str, Any]]:
    """Define representative detection criteria regimes."""
    return {
        "Baseline_ZTFReST": {
            "snr_threshold": 5.0,
            "snr_threshold_secondary": 3.0,
            "min_detections": 2,
            "min_detections_primary": 1,
            "max_timespan_days": 14.0,
            "min_time_separation_hours": 3.0,
            "min_fade_rate": 0.30,
            "min_galactic_lat": 15.0,
            "require_fast_transient": True,
        },
        "Relaxed_HighEfficiency": {
            "snr_threshold": 4.0,
            "snr_threshold_secondary": 2.5,
            "min_detections": 2,
            "min_detections_primary": 1,
            "max_timespan_days": 21.0,
            "min_time_separation_hours": 1.0,
            "min_fade_rate": 0.15,
            "min_galactic_lat": 10.0,
            "require_fast_transient": True,
        },
        "Strict_HighPurity": {
            "snr_threshold": 6.0,
            "snr_threshold_secondary": 4.0,
            "min_detections": 3,
            "min_detections_primary": 2,
            "max_timespan_days": 7.0,
            "min_time_separation_hours": 6.0,
            "min_fade_rate": 0.50,
            "min_galactic_lat": 20.0,
            "require_fast_transient": True,
        },
    }


def get_population_configurations() -> Dict[str, Dict[str, Any]]:
    """Define the comparative populations."""
    return {
        # 1. AT2017gfo Fiducial (Fixed viewing angle 26 deg)
        "AT2017gfo_FixedView": {
            "log10_mej_dyn": -1.80,    # ~0.016 M_sun
            "v_ej_dyn": 0.20,
            "ye_dyn": 0.15,
            "log10_mej_wind": -1.10,   # ~0.079 M_sun
            "v_ej_wind": 0.10,
            "ye_wind": 0.35,
            "inclination_em": 0.45,   # ~26 deg (GW170817 geometry)
            "vary_inclination": False,
            "rate": 1000.0,
            "z_max": 0.15,
            "randomize": False,
            "description": "AT2017gfo tuned parameters observed along polar line-of-sight (iota=26 deg)",
        },
        # 2. AT2017gfo Ejecta with Isotropic Viewing Angles
        "AT2017gfo_IsotropicView": {
            "log10_mej_dyn": -1.80,
            "v_ej_dyn": 0.20,
            "ye_dyn": 0.15,
            "log10_mej_wind": -1.10,
            "v_ej_wind": 0.10,
            "ye_wind": 0.35,
            "inclination_em": 0.45,
            "vary_inclination": True,   # Uniform in cos(iota)
            "rate": 1000.0,
            "z_max": 0.25,
            "randomize": False,
            "description": "AT2017gfo ejecta masses with isotropic (angle-averaged) viewing distribution",
        },
        # 3. Broad Prior - Fixed Viewing Angle
        "BroadPrior_FixedView": {
            "rate": 1000.0,
            "z_max": 0.20,
            "vary_inclination": False,
            "randomize": True,
            "description": "Randomized ejecta parameters across physical prior bounds, fixed iota=26 deg",
        },
        # 4. Broad Prior - Full Isotropic Parameter Swath
        "BroadPrior_Isotropic": {
            "rate": 1000.0,
            "z_max": 0.30,
            "vary_inclination": True,
            "randomize": True,
            "description": "Full multidimensional prior: ejecta masses, velocities, Y_e, and cos(iota) randomized",
        },
    }


def sample_random_broad_params(rng: np.random.Generator, vary_incl: bool) -> Dict[str, Any]:
    """Sample ejecta parameters across broad physical prior ranges."""
    params = {
        "log10_mej_dyn": float(rng.uniform(-4.0, -1.301)),
        "v_ej_dyn": float(rng.uniform(0.12, 0.35)),
        "ye_dyn": float(rng.uniform(0.15, 0.35)),
        "log10_mej_wind": float(rng.uniform(-4.0, -0.550)),
        "v_ej_wind": float(rng.uniform(0.05, 0.15)),
        "ye_wind": float(rng.uniform(0.20, 0.40)),
        "inclination_em": float(rng.uniform(0.0, math.pi / 2.0)) if not vary_incl else 0.45,
        "vary_inclination": vary_incl,
    }
    return params


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("KILONOVA POPULATION COMPARISON: BROAD SWATH PRIORS vs. AT2017gfo ARCHETYPE")
    print(f"Transients: {args.n_transients:,} | Runs per config: {args.n_runs}")
    print("=" * 85)

    print("Loading full 8-year ZTF survey dataset...")
    survey = load_ztf_survey(start="201803", end="202603", nside=64)
    model = FiestaKNModel()

    criteria_regimes = get_criteria_regimes()
    populations = get_population_configurations()

    records = []
    run_counter = 0
    total_combinations = len(criteria_regimes) * len(populations) * args.n_runs

    for crit_name, crit_cfg in criteria_regimes.items():
        det = DetectionCriteria(
            snr_threshold=crit_cfg["snr_threshold"],
            snr_threshold_secondary=crit_cfg["snr_threshold_secondary"],
            min_detections=crit_cfg["min_detections"],
            min_detections_primary=crit_cfg["min_detections_primary"],
            max_timespan_days=crit_cfg["max_timespan_days"],
            min_time_separation_hours=crit_cfg["min_time_separation_hours"],
            require_fast_transient=crit_cfg["require_fast_transient"],
            min_rise_rate=0.0,
            min_fade_rate=crit_cfg["min_fade_rate"],
            min_galactic_lat=crit_cfg["min_galactic_lat"],
        )

        for pop_name, pop_cfg in populations.items():
            for r in range(args.n_runs):
                run_counter += 1
                seed = args.seed + run_counter * 1009
                rng = np.random.default_rng(seed)

                # Build population
                if pop_cfg["randomize"]:
                    sample_kwargs = sample_random_broad_params(rng, pop_cfg["vary_inclination"])
                    sample_kwargs["rate"] = pop_cfg["rate"]
                    sample_kwargs["z_max"] = pop_cfg["z_max"]
                    pop = FixedBu2026KilonovaPopulation(**sample_kwargs)
                else:
                    pop_kwargs = {
                        k: v for k, v in pop_cfg.items()
                        if k not in ["randomize", "description"]
                    }
                    pop = FixedBu2026KilonovaPopulation(**pop_kwargs)

                pipeline = SimulationPipeline(
                    survey=survey,
                    populations=[pop],
                    models={"Kilonova": model},
                    detection=det,
                    n_transients=args.n_transients,
                    seed=seed,
                )

                t0 = time.perf_counter()
                result = pipeline.run()
                elapsed = time.perf_counter() - t0

                rs = result.rate_summaries[0]
                eff_vt = rs.effective_vt_gpc3_yr
                r95 = -math.log(0.05) / eff_vt if (eff_vt > 0 and math.isfinite(eff_vt)) else float("nan")

                # Serialize result
                fname = out_dir / f"sim_{crit_name}_{pop_name}_run{r}.json"
                save_result(result, str(fname), overwrite=args.overwrite)

                record = {
                    "criteria_regime": crit_name,
                    "population_type": pop_name,
                    "description": pop_cfg["description"],
                    "run_idx": r,
                    "seed": seed,
                    "n_simulated": result.n_simulated,
                    "n_detected": result.n_detected,
                    "efficiency": rs.overall_efficiency,
                    "effective_vt_gpc3_yr": eff_vt,
                    "rate_95_upper_limit": r95,
                    "elapsed_seconds": elapsed,
                    **crit_cfg,
                }
                records.append(record)
                print(f"[{run_counter}/{total_combinations}] {crit_name:22s} | {pop_name:23s} | Det: {result.n_detected:4d} | Eff: {rs.overall_efficiency:8.5f} | R95: {r95:7.1f} | {elapsed:.2f}s")

    df = pd.DataFrame(records)
    summary_csv = out_dir / "prior_vs_at2017gfo_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\nWrote full comparison summary to {summary_csv}")

    print("\n" + "=" * 95)
    print("POPULATION vs ARCHETYPE PERFORMANCE COMPARISON SUMMARY")
    print("=" * 95)
    summary_table = df.groupby(["criteria_regime", "population_type"]).agg(
        n_detected_mean=("n_detected", "mean"),
        eff_mean=("efficiency", "mean"),
        vt_eff_mean=("effective_vt_gpc3_yr", "mean"),
        r95_mean=("rate_95_upper_limit", "mean"),
    )
    print(summary_table.to_string())
    print("=" * 95)


if __name__ == "__main__":
    main()
