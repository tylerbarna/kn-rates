"""Research checks for ZTF_sim prior bounds and randomized parameters."""

import sys

import numpy as np
import pytest

_original_argv = sys.argv
sys.argv = ["ZTF_sim.py"]
try:
    from ZTF_sim import (
        _parameter_bounds_for_model,
        _randomize_population_kwargs,
    )
finally:
    sys.argv = _original_argv


MODELS = ("Bu2026Fixed", "Bu2026Vary", "Metzger")


@pytest.mark.parametrize("model_name", MODELS)
def test_bounds_are_finite_and_ordered(model_name):
    for lower, upper in _parameter_bounds_for_model(model_name).values():
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper


@pytest.mark.parametrize("model_name", MODELS)
def test_large_prior_draws_stay_inside_bounds(model_name):
    bounds = _parameter_bounds_for_model(model_name)
    rng = np.random.default_rng(20260930)

    for _ in range(100_000):
        draw = _randomize_population_kwargs(model_name, 1000.0, rng=rng)
        for parameter, (lower, upper) in bounds.items():
            assert lower <= draw[parameter] <= upper


@pytest.mark.parametrize("model_name", MODELS)
def test_prior_draws_are_reproducible(model_name):
    first = _randomize_population_kwargs(
        model_name, 1000.0, rng=np.random.default_rng(123)
    )
    second = _randomize_population_kwargs(
        model_name, 1000.0, rng=np.random.default_rng(123)
    )
    assert first == second


@pytest.mark.parametrize("model_name", MODELS)
def test_uniform_prior_means_are_reasonable(model_name):
    bounds = _parameter_bounds_for_model(model_name)
    rng = np.random.default_rng(456)
    values = {parameter: [] for parameter in bounds}

    for _ in range(20_000):
        draw = _randomize_population_kwargs(model_name, 1000.0, rng=rng)
        for parameter in bounds:
            values[parameter].append(draw[parameter])

    for parameter, (lower, upper) in bounds.items():
        expected = (lower + upper) / 2.0
        tolerance = 0.03 * (upper - lower)
        assert abs(np.mean(values[parameter]) - expected) < tolerance


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        _parameter_bounds_for_model("unknown")
