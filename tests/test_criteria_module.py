"""Unit tests for canonical knrates.criteria module and Suite 02 parity."""
import inspect
import random
import sys
import textwrap
from pathlib import Path

from knrates import criteria as krc

REPO_ROOT = Path(__file__).resolve().parents[1]
FILTER_DIR = REPO_ROOT / "analysis_suites" / "02_granular_criteria_sweep"


def test_fast_criteria_filter_reexports_canonical_band_wls_rates():
    sys.path.insert(0, str(FILTER_DIR))
    try:
        import fast_criteria_filter as fcf
    finally:
        sys.path.remove(str(FILTER_DIR))
    assert fcf.band_wls_rates is krc.band_wls_rates
    assert textwrap.dedent(inspect.getsource(fcf.band_wls_rates)) == textwrap.dedent(inspect.getsource(krc.band_wls_rates))


def test_evaluate_known_fading_kilonova():
    pts = [
        (60000.0, 8.0, 19.0, "g"),
        (60001.0, 5.2, 19.8, "g"),
        (60002.0, 3.5, 20.5, "g"),
    ]
    res = krc.evaluate(pts, krc.SUITE02_FIDUCIAL)
    assert res["passes"] is True
    assert res["n_qualifying"] == 3
    assert res["n_primary"] == 2
    assert res["max_fade_rate"] > 0.6


def test_evaluate_rejects_flat_variable():
    pts = [
        (60000.0, 8.0, 19.0, "r"),
        (60002.0, 7.5, 19.05, "r"),
    ]
    res = krc.evaluate(pts, krc.SUITE02_FIDUCIAL)
    assert res["pass_fast"] is False
    assert res["passes"] is False


def test_deterministic_on_random_curves():
    rng = random.Random(42)
    for _ in range(200):
        pts = [(rng.uniform(0, 10), rng.uniform(2, 20), rng.uniform(18, 22), rng.choice("gri"))
               for _ in range(rng.randint(1, 8))]
        r1 = krc.evaluate(pts, krc.SUITE02_FIDUCIAL)
        r2 = krc.evaluate(list(reversed(pts)), krc.SUITE02_FIDUCIAL)
        assert r1 == r2
