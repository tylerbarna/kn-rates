"""Tests for Two-Snapshot search integration and paper/numbers.tex macro generation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from knrates import figures as F, macros as M, notebook as NB  # noqa: E402


def test_compute_and_write_paper1_macros(tmp_path: Path) -> None:
    ctx = NB.setup("latest", quiet=True)
    vals = M.compute_paper1_macros(ctx=ctx)
    assert all(k.isalpha() for k in vals)
    assert vals["RlimPolarIdeal"] == "192"
    assert vals["RlimIsotropicIdeal"] == "664"
    assert vals["VTPolar"] == "0.0156"
    assert vals["VTIsotropic"] == "0.0045"
    assert vals["VTGainFullOverWindow"] == "5.6"

    out = M.write_paper1_macros(tmp_path / "numbers.tex", ctx=ctx)
    txt = out.read_text(encoding="utf-8")
    assert r"\newcommand{\RlimPolarIdeal}{192}" in txt
    assert r"\newcommand{\VTGainFullOverWindow}{5.6}" in txt


def test_search_snapshot_auto_loads_and_figures_draw() -> None:
    ctx = NB.setup("latest", search_snapshot="auto", quiet=True)
    if ctx.search_snap is None:
        return  # allowed when real-ztf-search sibling is absent
    assert "stage_a_cutflow" in ctx.search_snap
    assert "rb_calibration" in ctx.search_snap
    fig1 = F.stage_a_cutflow(ctx.search_snap["stage_a_cutflow"])
    fig2 = F.rb_calibration_summary(ctx.search_snap["rb_calibration"])
    assert fig1 is not None and fig2 is not None
    plt.close("all")
