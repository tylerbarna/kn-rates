# import fiesta
from survey_sim import FixedBu2026KilonovaPopulation, FixedMetzgerKilonovaPopulation, SimulationPipeline, load_ztf_survey, DetectionCriteria, Bu2026KilonovaPopulation, SurveyStore, MetzgerKNModel
from survey_sim.fiesta_model import FiestaKNModel
from survey_sim.serialization import save_result, load_result
import os 
import sys
from contextlib import redirect_stdout
import io
import math
import matplotlib.pyplot as plt
import numpy as np  
from scipy.stats import cumfreq


# from fiesta.surrogates import download_recommended_surrogates
# try:
#     download_recommended_surrogates()
# except Exception as e:
#     print(f"Warning: Could not download recommended surrogates:\n{e}")
#     pass
import argparse
import datetime
from pathlib import Path
import time

DEFAULT_OUTPUT_BASE = "results/ztf_10_sim_result"
DEFAULT_N_TRANSIENTS = 1_000_000
DEFAULT_N_RUNS = 10
DEFAULT_N_PROCESSES = DEFAULT_N_RUNS


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ZTF kilonova simulation.")
    parser.add_argument(
        "--output-base",
        default=DEFAULT_OUTPUT_BASE,
        help="Base output filename for result JSON files.",
    )
    parser.add_argument(
        "--n-transients",
        type=int,
        default=DEFAULT_N_TRANSIENTS,
        help="Number of transients to simulate in each run.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of simulation runs to execute.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=DEFAULT_N_PROCESSES,
        help="Number of parallel processes to use (default: same as n-runs).",
    )
    parser.add_argument(
        "--vary-rate",
        action="store_true",
        help="If set, vary the volumetric rate for each run (default: fixed rate).",
    )
    parser.add_argument(
        "--randomize-params",
        action="store_true",
        help="If set, randomize kilonova parameters per run from reasonable model bounds.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Bu2026Fixed",
        help="Kilonova model to use (default: Bu2026Fixed).",
        choices=["Bu2026Fixed", "Bu2026Vary", "Metzger"]
    )
    parser.add_argument(
        "--debug-cadence",
        action="store_true",
        help="Force a perfect 0.5-day cadence at RA=0, Dec=0 for debugging detection criteria.",
    )
    return parser.parse_args()


args = parse_args()
output_base = Path(args.output_base)
output_base.parent.mkdir(parents=True, exist_ok=True)
vary_rate = args.vary_rate
randomize_params = args.randomize_params
debug_cadence = args.debug_cadence
model = args.model

if debug_cadence:
    print("Debug cadence enabled: forcing perfect 0.5-day cadence at RA=0, Dec=0.")
    os.environ["DEBUG_CADENCE"] = "0.5"
    os.environ["DEBUG_MAG"] = "30.0"

def _parameter_bounds_for_model(model_name: str):
    """Return reasonable physical bounds for kilonova parameters.

    These values are broad enough to span the expected range of realistic
    kilonova ejecta and viewing-angle configurations without going outside
    physically plausible limits. The default behavior remains unchanged unless
    the caller opts into randomization.
    """
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        return {
            "log10_mej_dyn": (-4.0, -1.3010299956639813),
            "v_ej_dyn": (0.12, 0.35),
            "ye_dyn": (0.15, 0.35),
            "log10_mej_wind": (-4.0, -0.5502131530142267),
            "v_ej_wind": (0.05, 0.15),
            "ye_wind": (0.2, 0.4),
            "inclination_em": (0.0, 1.5707963267948966),
            "z_max": (0.15, 0.35),
        }
    if model_name == "Metzger":
        return {
            "mej": (0.0005, 0.0050),
            "vej": (0.20, 0.80),
            "kappa": (50.0, 1000.0),
            "z_max": (0.15, 0.35),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _randomize_population_kwargs(model_name: str, rate: float, rng=None):
    """Sample a single realistic kilonova population from broad model bounds."""
    if rng is None:
        rng = np.random.default_rng()
    bounds = _parameter_bounds_for_model(model_name)
    kwargs = {key: float(rng.uniform(low, high)) for key, (low, high) in bounds.items() if key != "z_max"}
    kwargs["rate"] = rate
    kwargs["z_max"] = float(rng.uniform(*bounds["z_max"]))
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        kwargs["vary_inclination"] = model_name == "Bu2026Vary"
    return kwargs

survey = load_ztf_survey(nside=64);

print(f"  Observations: {survey.n_observations}")
print(f"  MJD range: {survey.mjd_range}")
print(f"  Duration: {survey.duration_years:.2f} years")
print(f"  Bands: {survey.bands}")



def _build_population(model_name: str, rate: float, randomize_params: bool = False, **kwargs):
    ## all parameters are keyword arguments that can be updated by passing them in as kwargs
    if model_name == "Bu2026Fixed":
        # Tuned AT2017gfo Bu2026 parameters (best g/r/i fit at t<4d)
        # log10_mej_dyn=-1.8 (slightly less dynamical ejecta)
        # inclination_em=0.45 rad (26 deg, consistent with GW170817 constraints)
        population_kwargs = dict(
            log10_mej_dyn=-1.7,
            v_ej_dyn=0.2,
            ye_dyn=0.15,
            log10_mej_wind=-1.1,
            v_ej_wind=0.1,
            ye_wind=0.35,
            inclination_em=0.45,
            vary_inclination=False,
            rate=rate,
            # z_max chosen well above the AT2017gfo-bright ZTF detection horizon.
            # compute_rate's integrand has already converged out here, so tightening
            # further would dilute MC stats without biasing VT_eff.
            z_max=0.15,
        )
        if randomize_params:
            population_kwargs = _randomize_population_kwargs(model_name, rate)
        population_kwargs.update(kwargs)
        return FixedBu2026KilonovaPopulation(**population_kwargs)
    if model_name == "Bu2026Vary":
        population_kwargs = dict(
            log10_mej_dyn=-1.7,
            v_ej_dyn=0.2,
            ye_dyn=0.15,
            log10_mej_wind=-1.1,
            v_ej_wind=0.1,
            ye_wind=0.35,
            inclination_em=0.45,
            vary_inclination=True,  # flat in cos(iota)
            rate=rate,
            z_max=0.3,
        )
        if randomize_params:
            population_kwargs = _randomize_population_kwargs(model_name, rate)
        population_kwargs.update(kwargs)
        return FixedBu2026KilonovaPopulation(**population_kwargs)
    if model_name == "Metzger":
        population_kwargs = dict(
            mej=0.00126,
            vej=0.50,
            kappa=398.0,
            rate=rate,
            z_max=0.3,
        )
        if randomize_params:
            population_kwargs = _randomize_population_kwargs(model_name, rate)
        population_kwargs.update(kwargs)
        return FixedMetzgerKilonovaPopulation(**population_kwargs)
    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str):
    if model_name in {"Bu2026Fixed", "Bu2026Vary"}:
        return FiestaKNModel()
    if model_name == "Metzger":
        return MetzgerKNModel()
    raise ValueError(f"Unsupported model: {model_name}")


def _make_detection_criteria():
    return DetectionCriteria(
        snr_threshold=5.0,
        snr_threshold_secondary=3.0,
        min_detections=2,
        min_detections_primary=1,
        max_timespan_days=14.0,
        min_time_separation_hours=3.0,
        require_fast_transient=True,
        min_rise_rate=0.0,
        min_fade_rate=0.3,
        min_galactic_lat=15.0,
    )


# Run pipeline in parallel
n_sims = args.n_runs
n_processes = args.n_processes
N = args.n_transients
print(f"\nRunning pipeline {n_sims} times across {n_processes} threads with {N} transients each (fixed incl=0.45 rad, tuned ejecta)...")
if randomize_params:
    print("  Parameter randomization: enabled")


def _run_instance(idx: int, seed=None):
    rate = 1000.0 if not vary_rate else np.random.uniform(100.0, 2000.0)
    pop_kwargs = {}
    if debug_cadence:
        pop_kwargs['fixed_coord'] = (0.0, 0.0)
    pop = _build_population(model, rate, randomize_params=randomize_params, **pop_kwargs)
    det = _make_detection_criteria()
    model_instance = _build_model(model)

    # Ensure seed isn't set to be the same for all processes when running in parallel, but allow for a fixed seed when running single-threaded for reproducibility.
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        # print(f"  Using seed {seed} for run {idx} (rate={rate:.1f} Gpc^-3 yr^-1)")
    else:
        seed = int(seed) + idx
    pipeline = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"Kilonova": model_instance},
        detection=det,
        n_transients=N,
        seed=seed,
    )
    run_start = time.perf_counter()
    result = pipeline.run()
    run_elapsed = time.perf_counter() - run_start
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fname = f"{output_base}_{datetime_str}.json"
    save_result(result, fname)
    # return a small summary dict (picklable)
    summaries = []
    for rs in result.rate_summaries:
        summaries.append({
            'transient_type': rs.transient_type,
            'volumetric_rate': rs.volumetric_rate,
            'n_detected': rs.n_detected,
            'n_simulated': rs.n_simulated,
            'overall_efficiency': rs.overall_efficiency,
            'survey_duration_years': rs.survey_duration_years,
            'effective_vt_gpc3_yr': rs.effective_vt_gpc3_yr,
        })
    return {
        'idx': idx,
        'fname': fname,
        'n_simulated': result.n_simulated,
        'n_detected': result.n_detected,
        'elapsed_seconds': run_elapsed,
        'summaries': summaries,
    }


def _format_elapsed(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    return f"{remaining_seconds:.1f}s"

if __name__ == '__main__':
    from multiprocessing import get_context
    ctx = get_context('spawn')
    batch_start = time.perf_counter()
    
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(_run_instance, list(range(n_sims)))
    batch_elapsed = time.perf_counter() - batch_start

    # Print aggregated summaries
    for res in results:
        print(f"\nResult file: {res['fname']}")
        print(f"  Simulated: {res['n_simulated']}")
        print(f"  Detected:  {res['n_detected']}")
        print(f"  Runtime:   {_format_elapsed(res['elapsed_seconds'])}")
        eff = res['n_detected'] / max(res['n_simulated'], 1)
        print(f"  Efficiency: {eff:.4f} ({eff*100:.2f}%)")
        print(f"\n--- Rate Summaries ---")
        for rs in res['summaries']:
            print(f"  {rs['transient_type']}: Vol rate={rs['volumetric_rate']:.1f} n_det={rs['n_detected']} n_sim={rs['n_simulated']} eff={rs['overall_efficiency']:.4f} VT={rs['effective_vt_gpc3_yr']:.4e}")

    print(f"\nTotal batch runtime: {_format_elapsed(batch_elapsed)}")
        
# upper_limits = [rs.upper_limit(percentile).rate_upper for percentile in np.linspace(0, 1, 100)]
# # print(upper_limits)
# ecdf = cumfreq(upper_limits, numbins=100)
# x = ecdf.lowerlimit + np.arange(ecdf.cumcount.size) * ecdf.binsize
# y = ecdf.cumcount / ecdf.cumcount[-1]
# plt.plot(x, y, linewidth=3)
# plt.xlabel('Upper Limit (Gpc^-3 yr^-1)')
# plt.xscale('log')
# plt.title('Cumulative Distribution of Upper Limits')
# # plt.legend()
# plt.grid(True, which='both')
# plt.savefig('upper_limits_ecdf.png')
