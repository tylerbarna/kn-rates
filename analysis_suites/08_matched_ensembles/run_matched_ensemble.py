#!/usr/bin/env python3
"""Suite 08: Matched-exposure ZTF 8-year kilonova rate ensembles (WP-5).

Runs high-statistics injection-recovery ensembles matched to the real ZTF archival search
(``real-ztf-search``) with zero modifications to the frozen ``survey-sim`` package:

1. Loads restricted searched-exposure HDF5 logs exported by ``knsearch.exposure_export`` when
   ``--exposure-dir`` is provided (falling back to ``load_ztf_survey`` only when omitted).
2. Enforces pure fade-only selection (``min_rise_rate = 99.0``) matching ``knsearch.criteria``.
3. Supports all five Paper I model tiers via stratified parameter draws in ``knrates.ensembles``:
   - ``Bu2026_Polar``
   - ``Bu2026_Isotropic``
   - ``AT2017gfo_Posterior_Polar``
   - ``AT2017gfo_Posterior_Isotropic``
   - ``BNS_NR_Population``
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knrates.ensembles import (
    FADE_ONLY_MIN_RISE_RATE,
    SUITE08_TIERS,
    draw_tier_parameters,
    load_posterior_samples,
    resolve_local_boom_hdf5_paths,
)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tier", choices=list(SUITE08_TIERS), default="Bu2026_Polar",
                    help="Paper I kilonova model tier.")
    ap.add_argument("--exposure-dir", type=str, default=None,
                    help="Directory of searched monthly ZTF HDF5 files (ztf_YYYYMM.h5) from knsearch.exposure_export.")
    ap.add_argument("--posterior-chain", type=str, default=None,
                    help="Optional CSV/JSON file of AT2017gfo posterior samples for AT2017gfo_Posterior_* tiers.")
    ap.add_argument("--start", type=str, default="201803", help="Start month (YYYYMM).")
    ap.add_argument("--end", type=str, default="202603", help="End month (YYYYMM, inclusive).")
    ap.add_argument("--nside", type=int, default=64, help="HEALPix NSIDE for spatial indexing.")
    ap.add_argument("--n-transients", type=int, default=500_000,
                    help="Synthetic kilonovae injected per task (200 tasks x 500k = 10^8 per tier).")
    ap.add_argument("--task-index", type=int, default=None,
                    help="0-based draw index within [0, n-tasks) (defaults to SLURM_ARRAY_TASK_ID %% n_tasks).")
    ap.add_argument("--n-tasks", type=int, default=200,
                    help="Total stratified parameter draws per tier across the ensemble.")
    ap.add_argument("--ensemble-seed", type=int, default=20261125,
                    help="Shared ensemble seed for deterministic Latin-hypercube stratification across tasks.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Per-task Monte Carlo injection seed (defaults to ensemble_seed + 10007 * (task_index + 1)).")
    ap.add_argument("--rate", type=float, default=1000.0, help="Fiducial injected rate [Gpc^-3 yr^-1].")
    ap.add_argument("--z-max", type=float, default=0.15, help="Comoving injection horizon z_max.")
    ap.add_argument("--output-dir", type=str, default="results/08_matched_ensembles",
                    help="Root directory for Suite 08 JSON + .metadata.json outputs.")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    ap.set_defaults(overwrite=True)

    # Photometric criteria (parity with knsearch.criteria)
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--snr-threshold-secondary", type=float, default=3.0)
    ap.add_argument("--min-detections", type=int, default=2)
    ap.add_argument("--min-detections-primary", type=int, default=1)
    ap.add_argument("--max-timespan-days", type=float, default=14.0)
    ap.add_argument("--min-time-separation-hours", type=float, default=3.0)
    ap.add_argument("--min-fade-rate", type=float, default=0.30)
    ap.add_argument("--min-rise-rate", type=float, default=FADE_ONLY_MIN_RISE_RATE,
                    help="Set to 99.0 to disable survey-sim's rise-OR-fade branch for pure fade-only selection.")
    ap.add_argument("--min-galactic-lat", type=float, default=15.0)
    return ap.parse_args(argv)


def load_matched_survey(args: argparse.Namespace):
    from survey_sim import SurveyStore, load_ztf_survey

    if args.exposure_dir:
        paths = resolve_local_boom_hdf5_paths(Path(args.exposure_dir), start=args.start, end=args.end)
        print(f"Loading {len(paths)} searched ZTF HDF5 files from {args.exposure_dir} [{args.start}..{args.end}]...")
        return SurveyStore.from_ztf_boom(paths, nside=args.nside), "searched_local_hdf5", len(paths)
    print(f"WARNING: --exposure-dir not set; falling back to full survey-sim ZTF store [{args.start}..{args.end}].")
    return load_ztf_survey(start=args.start, end=args.end, nside=args.nside), "default_ztf_boom", None


def main(argv=None) -> int:
    args = parse_args(argv)

    from survey_sim import DetectionCriteria, FixedBu2026KilonovaPopulation, SimulationPipeline
    from survey_sim.fiesta_model import FiestaKNModel
    from survey_sim.serialization import save_result

    raw_slurm = os.environ.get("SLURM_ARRAY_TASK_ID")
    if args.task_index is not None:
        task_index = int(args.task_index)
    elif raw_slurm is not None:
        task_index = int(raw_slurm) % int(args.n_tasks)
    else:
        task_index = 0

    tier_offset = SUITE08_TIERS.index(args.tier) * 1_000_003
    mc_seed = int(args.seed) if args.seed is not None else int(args.ensemble_seed + tier_offset + (task_index + 1) * 10_007)

    post_df = load_posterior_samples(Path(args.posterior_chain)) if args.posterior_chain else None
    pop_params = draw_tier_parameters(
        args.tier,
        task_index=task_index,
        n_tasks=args.n_tasks,
        ensemble_seed=args.ensemble_seed,
        rate=args.rate,
        z_max=args.z_max,
        posterior_samples=post_df,
    )

    survey, exposure_source, n_h5 = load_matched_survey(args)
    pop = FixedBu2026KilonovaPopulation(**pop_params)
    kn_model = FiestaKNModel(survey="ztf")
    det = DetectionCriteria(
        snr_threshold=args.snr_threshold,
        snr_threshold_secondary=args.snr_threshold_secondary,
        min_detections=args.min_detections,
        min_detections_primary=args.min_detections_primary,
        max_timespan_days=args.max_timespan_days,
        min_time_separation_hours=args.min_time_separation_hours,
        require_fast_transient=True,
        min_rise_rate=args.min_rise_rate,
        min_fade_rate=args.min_fade_rate,
        min_galactic_lat=args.min_galactic_lat,
    )

    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": kn_model},
        detection=det,
        n_transients=args.n_transients,
        seed=mc_seed,
    )

    t0 = time.perf_counter()
    result = pipeline.run()
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output_dir) / args.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
    fname = out_dir / f"ztf_{args.tier}_{args.n_transients // 1000}k_task_{task_index:03d}_run_0_{stamp}.json"
    save_result(result, str(fname), overwrite=args.overwrite)

    crit_dict: Dict[str, Any] = {
        "snr_threshold": args.snr_threshold,
        "snr_threshold_secondary": args.snr_threshold_secondary,
        "min_detections": args.min_detections,
        "min_detections_primary": args.min_detections_primary,
        "max_timespan_days": args.max_timespan_days,
        "min_time_separation_hours": args.min_time_separation_hours,
        "min_rise_rate": args.min_rise_rate,
        "min_fade_rate": args.min_fade_rate,
        "min_galactic_lat": args.min_galactic_lat,
        "require_fast_transient": True,
    }
    metadata = {
        "suite": "08",
        "model": args.tier,
        "model_family": "Bu2026",
        "n_transients": args.n_transients,
        "run_index": 0,
        "task_id": f"{task_index:03d}",
        "n_tasks": args.n_tasks,
        "ensemble_seed": args.ensemble_seed,
        "seed": mc_seed,
        "z_max": args.z_max,
        "rate": args.rate,
        "randomize_params": args.tier not in ("Bu2026_Polar", "Bu2026_Isotropic"),
        "exposure_source": exposure_source,
        "exposure_dir": args.exposure_dir,
        "n_exposure_files": n_h5,
        "posterior_chain": args.posterior_chain,
        "elapsed_seconds": elapsed,
        "parameters": pop_params,
        "detection_criteria": crit_dict,
    }
    Path(f"{fname}.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rs = result.rate_summaries[0]
    r95 = -math.log(0.05) / rs.effective_vt_gpc3_yr if rs.effective_vt_gpc3_yr > 0 else float("nan")
    print(
        f"[Suite 08] tier={args.tier} task={task_index:03d}/{args.n_tasks} "
        f"det={result.n_detected}/{result.n_simulated} VT={rs.effective_vt_gpc3_yr:.4e} Gpc^3 yr "
        f"R95={r95:.1f} Gpc^-3 yr^-1 ({elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
