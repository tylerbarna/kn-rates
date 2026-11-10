"""Candidate-set compatibility and per-set VT normalisation in the Suite 02 filter."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_suites" / "02_granular_criteria_sweep"))
import fast_criteria_filter as F  # noqa: E402


def _cand(d: Path, name: str, loose_b: float, zmax: float, seed: int, eff=0.01, vt=0.0845):
    f = d / name
    f.write_text("{}")
    Path(f"{f}.metadata.json").write_text(json.dumps({
        "seed": seed, "params": {"z_max": zmax}, "loose_criteria": {"min_galactic_lat": loose_b},
        "overall_efficiency": eff, "effective_vt_gpc3_yr": vt}))
    return f


def test_legacy_and_duplicate_sets_are_skipped(tmp_path):
    a = _cand(tmp_path, "candidates_M_sample_000.json", 15.0, 0.15, 1)
    b = _cand(tmp_path, "candidates_M_fiducial.json", 15.0, 0.15, 1)       # same seed as sample 0
    c = _cand(tmp_path, "candidates_M_sample_001.json", 0.0, 0.15, 2)       # legacy: |b| not applied
    d = _cand(tmp_path, "candidates_M_sample_002.json", 15.0, 0.30, 3)      # legacy Metzger z_max
    e = _cand(tmp_path, "candidates_M_sample_003.json", 15.0, 0.15, 4)
    keep, skipped = F.filter_compatible_candidates([a, b, c, d, e])
    assert keep == [a, e]
    assert len(skipped) == 3


def test_vt_per_efficiency_from_metadata(tmp_path):
    f = _cand(tmp_path, "candidates_M_sample_000.json", 15.0, 0.15, 1, eff=0.01, vt=0.0845)
    assert np.isclose(F.vt_per_efficiency(f), 8.45)
