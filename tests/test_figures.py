"""Smoke test: every figure function draws from the committed snapshot without error."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from knrates import analysis as A, figures as F, notebook as NB  # noqa: E402
from knrates.stats import summarize_runs  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return NB.setup("latest", quiet=True)


def test_setup_keeps_only_suite01_possis(ctx):
    assert ctx.models == ["Bu2026Fixed", "Bu2026Vary"]
    assert not any("Metzger" in m or m.startswith("GW170817_") for m in ctx.runs.model.unique())


def test_all_figures_draw(ctx):
    runs, lim, M, snap = ctx.runs, ctx.lim, ctx.models, ctx.snap
    figs = [F.rate_limits(lim, M), F.distance_reach(ctx.zh, runs, M), F.sensitivity_timeline(ctx.tm, runs, M),
            F.andreoni_window(A.andreoni_comparison(ctx.tm, runs, M)),
            F.viewing_angle(ctx.src, runs, lim, snap.get("models_lightcurves")),
            F.convergence(pd.concat([A.subsampling_convergence(runs, m, [10_000, 1_000_000], 50) for m in M]), lim, M),
            F.zmax_check(ctx.zh, runs, M), F.at2017gfo_lightcurves(snap["models_lightcurves"])[0],
            F.criteria_one_at_a_time(snap["suite02_slices_1d"], snap["suite02_fiducial"]),
            F.criteria_ranking(A.criteria_response(snap["suite02_slices_1d"], snap["suite02_fiducial"])),
            F.criteria_pairs(snap["suite02_slices_2d"], snap["suite02_fiducial"], "Bu2026Fixed"),
            F.named_regimes(snap["suite02_named_configs"])]
    s3 = snap["suite03_runs"].rename(columns={"population_type": "model"})
    l3 = summarize_runs(s3, by=("criteria_regime", "model")).rename(columns={"model": "population_type"})
    figs.append(F.population_regimes(l3, sorted(set(l3.criteria_regime)), sorted(set(l3.population_type))))
    assert all(f is not None for f in figs)
    plt.close("all")


def test_table1_matches_limits(ctx):
    tab, tex, _ = F.table1(ctx.lim, ctx.runs, ctx.models)
    assert list(tab.model) == ctx.models
    assert (tab.set_index("model").r95 == ctx.lim.loc[ctx.models, "r95"]).all()


def test_andreoni_window_is_23_months(ctx):
    cmp = A.andreoni_comparison(ctx.tm, ctx.runs, ctx.models).set_index("model")
    win = A.vt_in_window(ctx.tm, ctx.runs, "Bu2026Fixed", *A.ANDREONI_WINDOW)
    assert win["n_months"] == 23
    assert cmp.loc["Bu2026Fixed", "r95_window"] == pytest.approx(win["r95"])
