#!/usr/bin/env python3
"""Cluster-scale prior and parameter-bound sensitivity study for ZTF_sim."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


MODELS = ("Bu2026Fixed", "Bu2026Vary", "Metzger")


def import_ztf_prior_functions():
    """Import the existing prior functions without forwarding CLI arguments."""
    original_argv = sys.argv
    sys.argv = ["ZTF_sim.py"]
    try:
        import ZTF_sim
        from ZTF_sim import (
            _parameter_bounds_for_model,
            _randomize_population_kwargs,
        )
    finally:
        sys.argv = original_argv
    return (
        ZTF_sim,
        _parameter_bounds_for_model,
        _randomize_population_kwargs,
    )


def summarize_samples(values, lower, upper):
    values = np.asarray(values, dtype=float)
    return {
        "declared_lower": float(lower),
        "declared_upper": float(upper),
        "sample_min": float(values.min()),
        "sample_max": float(values.max()),
        "sample_mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "sample_q001": float(np.quantile(values, 0.001)),
        "sample_q01": float(np.quantile(values, 0.01)),
        "sample_q50": float(np.quantile(values, 0.50)),
        "sample_q99": float(np.quantile(values, 0.99)),
        "sample_q999": float(np.quantile(values, 0.999)),
        "below_bound": int(np.count_nonzero(values < lower)),
        "above_bound": int(np.count_nonzero(values > upper)),
    }


def analyze_model(model_name, n_samples, seed, rate, bounds_function, sampler):
    rng = np.random.default_rng(seed)
    bounds = bounds_function(model_name)
    samples = {parameter: np.empty(n_samples, dtype=np.float64) for parameter in bounds}
    violations = []

    for draw in range(n_samples):
        values = sampler(model_name, rate, rng=rng)
        for parameter, (lower, upper) in bounds.items():
            value = float(values[parameter])
            samples[parameter][draw] = value
            if not lower <= value <= upper:
                if len(violations) < 100:
                    violations.append({
                        "draw": draw,
                        "parameter": parameter,
                        "value": value,
                        "lower": lower,
                        "upper": upper,
                    })

    summary = {
        parameter: summarize_samples(values, *bounds[parameter])
        for parameter, values in samples.items()
    }
    return summary, samples, violations


def run_response_surface(
    model_name,
    samples,
    n_simulations,
    n_transients,
    rate,
    seed,
    ztf_module,
):
    """Run small simulations at prior draws to connect priors to detections."""
    n_simulations = min(n_simulations, len(next(iter(samples.values()))))
    rng = np.random.default_rng(seed)
    records = []
    detection = ztf_module._make_detection_criteria()

    for draw_index in range(n_simulations):
        parameters = {
            parameter: float(values[draw_index])
            for parameter, values in samples.items()
        }
        population = ztf_module._build_population(
            model_name,
            rate,
            randomize_params=False,
            **parameters,
        )
        pipeline = ztf_module.SimulationPipeline(
            survey=ztf_module.survey,
            populations=[population],
            models={"Kilonova": ztf_module._build_model(model_name)},
            detection=detection,
            n_transients=n_transients,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        result = pipeline.run()
        summary = result.rate_summaries[0]
        records.append(
            {
                "draw_index": draw_index,
                "n_simulated": result.n_simulated,
                "n_detected": result.n_detected,
                "overall_efficiency": summary.overall_efficiency,
                "effective_vt_gpc3_yr": summary.effective_vt_gpc3_yr,
                "parameters": parameters,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/prior_sensitivity")
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--rate", type=float, default=1000.0)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODELS,
        help="Model to analyze; repeat for multiple models (default: all).",
    )
    parser.add_argument("--run-simulations", action="store_true")
    parser.add_argument("--n-simulations", type=int, default=0)
    parser.add_argument("--n-transients", type=int, default=100_000)
    args = parser.parse_args()

    ztf_module, bounds_function, sampler = import_ztf_prior_functions()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(args.model or MODELS)
    metadata = {
        "n_samples": args.n_samples,
        "base_seed": args.seed,
        "rate": args.rate,
        "models": {},
    }

    for model_index, model_name in enumerate(models):
        summary, samples, violations = analyze_model(
            model_name,
            args.n_samples,
            args.seed + model_index,
            args.rate,
            bounds_function,
            sampler,
        )
        np.savez_compressed(output_dir / f"{model_name}.npz", **samples)
        model_metadata = {
            "seed": args.seed + model_index,
            "summary": summary,
            "n_recorded_violations": len(violations),
            "violations": violations,
        }
        if args.run_simulations and args.n_simulations > 0:
            records = run_response_surface(
                model_name,
                samples,
                args.n_simulations,
                args.n_transients,
                args.rate,
                args.seed + 10_000 + model_index,
                ztf_module,
            )
            (output_dir / f"{model_name}_response.json").write_text(
                json.dumps(records, indent=2)
            )
            model_metadata["n_response_simulations"] = len(records)
        metadata["models"][model_name] = model_metadata
        print(f"{model_name}: {args.n_samples:,} draws, {len(violations)} violations")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
