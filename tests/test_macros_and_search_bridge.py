"""Tests for Two-Snapshot search integration and paper/numbers.tex macro generation."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from knrates import analysis as A, figures as F, macros as M, notebook as NB  # noqa: E402


def test_compute_and_write_paper1_macros_matches_analysis_and_ms_tex(tmp_path: Path) -> None:
    ctx = NB.setup("latest", quiet=True)
    vals = M.compute_paper1_macros(ctx=ctx)
    assert all(k.isalpha() for k in vals)

    # Verify macros match live computed quantities rather than stale string literals
    polar = ctx.lim.loc["Bu2026Fixed"]
    iso = ctx.lim.loc["Bu2026Vary"]
    cmp = A.andreoni_comparison(ctx.tm, ctx.runs, ["Bu2026Fixed", "Bu2026Vary"]).set_index("model")
    assert vals["RlimPolarIdeal"] == f"{float(polar['r95']):.0f}"
    assert vals["RlimIsotropicIdeal"] == f"{float(iso['r95']):.0f}"
    assert vals["VTPolar"] == f"{float(polar['vt_gpc3_yr']):.4f}"
    assert vals["VTIsotropic"] == f"{float(iso['vt_gpc3_yr']):.4f}"
    assert vals["VTGainFullOverWindow"] == f"{float(cmp.loc['Bu2026Fixed', 'vt_gain_full_over_window']):.1f}"

    out = M.write_paper1_macros(tmp_path / "numbers.tex", ctx=ctx)
    txt = out.read_text(encoding="utf-8")
    for k, v in vals.items():
        assert f"\\newcommand{{\\{k}}}{{{v}}}" in txt

    # Every macro defined in committed paper/numbers.tex and referenced in paper/ms.tex must exist in vals
    committed_numbers = (NB.REPO_ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    committed_keys = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", committed_numbers))
    assert committed_keys == set(vals.keys())
    ms_tex = (NB.REPO_ROOT / "paper" / "ms.tex").read_text(encoding="utf-8")
    used_macros = {m for m in re.findall(r"\\([A-Za-z]+)", ms_tex) if m in committed_keys}
    assert {"RlimPolarIdeal", "RlimIsotropicIdeal", "VTPolar", "VTIsotropic"} <= used_macros
    assert len(used_macros) >= 10


def test_search_snapshot_resolution_modes_and_figures(tmp_path: Path, monkeypatch) -> None:
    assert NB.load_search_snapshot(None) is None
    assert NB.load_search_snapshot("none") is None
    with pytest.raises(FileNotFoundError):
        NB.load_search_snapshot(tmp_path / "nonexistent_knsearch.zip")

    ctx = NB.setup("latest", search_snapshot="auto", quiet=True)
    if ctx.search_snap is not None:
        monkeypatch.setenv("KNRATES_SEARCH_SNAPSHOT", str(ctx.search_snap.path))
        via_env = NB.load_search_snapshot("auto")
        assert via_env is not None and via_env.path == ctx.search_snap.path
        fig1 = F.stage_a_cutflow(ctx.search_snap["stage_a_cutflow"])
        fig2 = F.rb_calibration_summary(ctx.search_snap["rb_calibration"])
        assert len(fig1.axes[0].lines) >= 1
        assert len(fig2.axes[0].lines) >= 1
        plt.close("all")
