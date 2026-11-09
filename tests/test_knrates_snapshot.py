"""Tests for the results snapshot compiler/loader (knrates.collect / knrates.snapshot)."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knrates import collect, snapshot  # noqa: E402
from knrates.stats import LN20, pooled_vt, rate_upper_limit  # noqa: E402


def _write_run(base: Path, model: str, task: int, n_det: int, n_sim: int = 1000, vt_per_eff: float = 8.4,
               corrupt: bool = False):
    d = base / "01_large_scale_rate_limits" / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"ztf_{model}_1k_task_{task}_run_0_2026-10-28_21:{task:02d}:00.json"
    eff = n_det / n_sim
    srcs = [{"z": 0.02 + 0.001 * i, "t_exp": 58200.0 + 10 * i, "true_params": {"mej": 0.01, "inclination_EM": 0.45},
             "photometry_times": [58200.5 + 10 * i, 58201.0 + 10 * i], "photometry_mags": [19.5, 19.9],
             "photometry_bands": ["g", "r"]} for i in range(n_det)]
    payload = {"type": "SimulationResult", "n_simulated": n_sim, "n_detected": n_det,
               "rate_summaries": [{"transient_type": "Kilonova", "volumetric_rate": 1000.0,
                                   "overall_efficiency": round(eff, 5), "n_simulated": n_sim, "n_detected": n_det,
                                   "z_max": 0.15, "survey_omega_sr": 8.2, "survey_duration_years": 8.00571,
                                   "effective_vt_gpc3_yr": eff * vt_per_eff}],
               "detected_sources": srcs}
    text = json.dumps(payload)
    f.write_text(text[: len(text) // 2] if corrupt else text)
    meta = {"model": model, "n_transients": n_sim, "run_index": 0, "task_id": str(task), "seed": 1000 + task,
            "z_max": 0.15, "detection_criteria": {"snr_threshold": 5.0, "min_detections": 2}}
    Path(f"{f}.metadata.json").write_text(json.dumps(meta))
    return f


def _results(tmp_path, n_runs=3):
    res = tmp_path / "results"
    for t in range(n_runs):
        _write_run(res, "Bu2026Fixed", t, n_det=20 + t)
    return res


def test_stats_pooling_and_limit():
    import pandas as pd
    df = pd.DataFrame({"vt_gpc3_yr": [0.01, 0.03], "n_simulated": [100, 100], "n_detected": [1, 3]})
    p = pooled_vt(df)
    assert p["vt"] == pytest.approx(0.02)
    assert rate_upper_limit(p["vt"]) == pytest.approx(LN20 / 0.02)
    assert LN20 == pytest.approx(2.995732, rel=1e-6)


def test_build_write_and_load(tmp_path):
    res = _results(tmp_path)
    tables, manifest, errors = collect.build(res)
    assert errors == []
    assert len(tables["suite01_runs"]) == 3
    assert tables["suite01_texp_month"]["n_detected"].sum() == 20 + 21 + 22
    out = tmp_path / "snap.zip"
    collect.write_snapshot(tables, manifest, out)
    with zipfile.ZipFile(out) as zf:
        assert "manifest.json" in zf.namelist()
    snap = snapshot.load_snapshot(out)
    assert len(snap["suite01_runs"]) == 3
    assert snap.manifest["tables"]["suite01_runs"]["rows"] == 3


def test_cli_latest_timestamped_and_selection(tmp_path, monkeypatch):
    res = _results(tmp_path)
    out = tmp_path / "snapshots"
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", out)
    assert collect.main(["--results-dir", str(res), "--out-dir", str(out)]) == 0
    assert (out / collect.LATEST_NAME).exists()
    assert collect.main(["--results-dir", str(res), "--out-dir", str(out), "--timestamped"]) == 0
    stamped = [p for p in out.glob("knrates_results_*.zip") if p.name != collect.LATEST_NAME]
    assert len(stamped) == 1
    stamp = stamped[0].stem.replace("knrates_results_", "")
    assert snapshot.load_snapshot(stamp[:8]).path == stamped[0]
    assert snapshot.load_snapshot("newest").path == stamped[0]
    assert snapshot.load_snapshot("latest").path == out / collect.LATEST_NAME
    # second run re-uses unchanged suites from the cache
    assert collect.main(["--results-dir", str(res), "--out-dir", str(out)]) == 0
    assert "suite01" in snapshot.load_snapshot("latest").manifest["reused_from_previous"]


def test_refuses_to_shrink_and_skips_corrupt_files(tmp_path):
    res = _results(tmp_path)
    out = tmp_path / "snapshots"
    assert collect.main(["--results-dir", str(res), "--out-dir", str(out)]) == 0
    before = (out / collect.LATEST_NAME).read_bytes()
    # a job dies mid-write: one result file is truncated
    _write_run(res, "Bu2026Fixed", 2, n_det=22, corrupt=True)
    assert collect.main(["--results-dir", str(res), "--out-dir", str(out), "--force"]) == 1
    assert (out / collect.LATEST_NAME).read_bytes() == before  # previous snapshot untouched
    tables, manifest, errors = collect.build(res, snapshot.load_snapshot(out / collect.LATEST_NAME), force=True)
    assert any("truncated" in s or "JSONDecodeError" in s for s in manifest["skipped_files"])
    assert any("refusing" in e for e in errors)


def test_rejects_impossible_counts(tmp_path):
    res = tmp_path / "results"
    _write_run(res, "Bu2026Fixed", 0, n_det=5)
    f = next((res / "01_large_scale_rate_limits" / "Bu2026Fixed").glob("*0_2026*.json"))
    d = json.loads(f.read_text())
    d["rate_summaries"][0]["n_detected"] = 5000  # > n_simulated
    f.write_text(json.dumps(d))
    _, _, errors = collect.build(res)
    assert any("n_detected > n_simulated" in e for e in errors)
