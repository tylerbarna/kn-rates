"""simulate_candidates must not reuse candidate files made with an older loose envelope or parameters."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_suites" / "02_granular_criteria_sweep"))
import simulate_candidates as SC  # noqa: E402

PARAMS = dict(log10_mej_dyn=-1.7, v_ej_dyn=0.2, rate=1000.0, z_max=0.15)


def _meta(tmp_path, **over):
    m = dict(model="Bu2026Fixed", n_transients=1_000_000, seed=42, params=PARAMS, randomize_params=False,
             loose_criteria=dict(SC.LOOSE_BOUNDING_CRITERIA))
    m.update(over)
    f = tmp_path / "candidates_Bu2026Fixed_sample_000.json.metadata.json"
    f.write_text(json.dumps(m))
    return f


def test_current_file_is_reused(tmp_path):
    assert SC.cache_mismatch(_meta(tmp_path), PARAMS, 1_000_000, 42, False) is None


def test_legacy_envelope_is_regenerated(tmp_path):
    legacy = dict(SC.LOOSE_BOUNDING_CRITERIA, min_galactic_lat=0.0)
    assert "loose envelope" in SC.cache_mismatch(_meta(tmp_path, loose_criteria=legacy), PARAMS, 1_000_000, 42, False)


def test_changed_zmax_seed_or_missing_metadata_is_regenerated(tmp_path):
    f = _meta(tmp_path, params=dict(PARAMS, z_max=0.3))
    assert "parameters" in SC.cache_mismatch(f, PARAMS, 1_000_000, 42, False)
    assert "seed" in SC.cache_mismatch(_meta(tmp_path), PARAMS, 1_000_000, 7, False)
    assert SC.cache_mismatch(tmp_path / "missing.metadata.json", PARAMS, 1_000_000, 42, False) == "no readable metadata"
