"""Matched-exposure kilonova ensemble definitions, stratified parameter draws, and hierarchical pooling.

Why this module exists
----------------------
Paper I compares the empirical 8-year ZTF archival search (``real-ztf-search``) against matched
``survey-sim`` injection-recovery ensembles (Suite 08) under two strict parity constraints without
modifying the frozen ``survey-sim`` Rust/Python codebase:

1. **Fade-only selection parity (``FADE_ONLY_MIN_RISE_RATE = 99.0``):**
   In ``survey-sim``'s Rust ``DetectionCriteria``, setting ``min_rise_rate = 0.0`` enables an OR branch
   that passes non-fading rising sources whenever ``max_rise_rate >= 0.0``. Setting
   ``min_rise_rate = 99.0`` disables the rise branch cleanly while preserving ``min_fade_rate = 0.30``
   mag/day, matching ``knsearch.criteria`` and ``knrates.criteria``.
2. **Searched-exposure footprint parity:**
   ``resolve_local_boom_hdf5_paths`` locates monthly ``ztf_YYYYMM.h5`` files exported by
   ``knsearch.exposure_export`` (restricted to exposures actually present in the searched local BOOM
   archive) so ``survey_sim.SurveyStore.from_ztf_boom`` simulates the exact searched sky-time footprint.

To avoid importing ``jax`` or ``survey_sim`` in analysis notebooks and unit tests, this module depends
only on the Python standard library, ``numpy``, and ``pandas``.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .criteria import FIDUCIAL_CRITERIA
from .stats import rate_upper_limit

#: Disables survey-sim's rise-OR-fade branch (min_rise_rate=0.0 bug) without modifying survey-sim.
FADE_ONLY_MIN_RISE_RATE: float = 99.0

#: Canonical 9-key criteria dictionary for Suite 08 matched-exposure runs.
MATCHED_FIDUCIAL_CRITERIA: Dict[str, Any] = {
    **FIDUCIAL_CRITERIA,
    "min_rise_rate": FADE_ONLY_MIN_RISE_RATE,
}

#: Validated neural-surrogate training domain for FiestaKNModel / Bu2026 POSSIS grid.
#: Lower mass bounds avoid the unphysical 10^-4 M_sun sub-luminous tail (docs/review_2026-11.md).
BU2026_SURROGATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "log10_mej_dyn": (-3.0, -1.301),
    "v_ej_dyn": (0.12, 0.35),
    "ye_dyn": (0.15, 0.35),
    "log10_mej_wind": (-2.5, -0.550),
    "v_ej_wind": (0.05, 0.15),
    "ye_wind": (0.20, 0.40),
}

EJECTA_KEYS: Tuple[str, ...] = (
    "log10_mej_dyn",
    "v_ej_dyn",
    "ye_dyn",
    "log10_mej_wind",
    "v_ej_wind",
    "ye_wind",
)

#: Baseline AT2017gfo-like point used in Suite 01 and Suite 08 point-archetype tiers.
BASELINE_BU2026_EJECTA: Dict[str, float] = {
    "log10_mej_dyn": -1.70,
    "v_ej_dyn": 0.20,
    "ye_dyn": 0.15,
    "log10_mej_wind": -1.10,
    "v_ej_wind": 0.10,
    "ye_wind": 0.35,
}

#: Calibrated marginal mean and 1-sigma widths approximating the AT2017gfo POSSIS + VLBI posterior
#: when an external posterior sample file is not supplied via --posterior-chain.
AT2017GFO_POSTERIOR_GAUSSIAN: Dict[str, Tuple[float, float]] = {
    "log10_mej_dyn": (-1.77, 0.15),
    "v_ej_dyn": (0.21, 0.04),
    "ye_dyn": (0.17, 0.04),
    "log10_mej_wind": (-1.12, 0.18),
    "v_ej_wind": (0.09, 0.025),
    "ye_wind": (0.33, 0.04),
    "inclination_em": (0.45, math.radians(3.5)),  # Mooley+2022 VLBI jet inclination 25.8 +/- 3.5 deg
}

#: NR-calibrated binary neutron star ejecta prior box (Radice+2018; Dietrich+2020).
BNS_NR_PRIOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "log10_mej_dyn": (-2.50, -1.301),
    "v_ej_dyn": (0.12, 0.35),
    "ye_dyn": (0.15, 0.30),
    "log10_mej_wind": (-2.00, -0.60),
    "v_ej_wind": (0.05, 0.15),
    "ye_wind": (0.25, 0.40),
}

SUITE08_TIERS: Tuple[str, ...] = (
    "Bu2026_Polar",
    "Bu2026_Isotropic",
    "AT2017gfo_Posterior_Polar",
    "AT2017gfo_Posterior_Isotropic",
    "BNS_NR_Population",
)

_YYYYMM_RE = re.compile(r"(20\d{4})")


def latin_hypercube_unit(n_samples: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic ``(n_samples, dim)`` Latin hypercube in ``(0, 1)^dim``.

    All Slurm array workers sharing ``(n_samples, seed)`` generate the exact same stratified design
    and index their assigned row ``task_index``, guaranteeing low-discrepancy coverage of the 6D/7D
    parameter space without inter-node communication.
    """
    if n_samples < 1 or dim < 1:
        raise ValueError(f"n_samples ({n_samples}) and dim ({dim}) must be >= 1")
    rng = np.random.default_rng(seed)
    u = np.empty((n_samples, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n_samples)
        jitter = rng.uniform(0.0, 1.0, size=n_samples)
        u[:, j] = (perm + jitter) / float(n_samples)
    return u


def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    """Acklam rational approximation to the standard normal quantile function (max error < 1.2e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)

    plow = 0.02425
    phigh = 1.0 - plow
    out = np.empty_like(p)

    low = p < plow
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    mid = (~low) & (p <= phigh)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )

    high = p > phigh
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    return out


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_normal_from_unit(u: float, mean: float, sd: float, lo: float, hi: float) -> float:
    """Map ``u in (0, 1)`` to ``N(mean, sd^2)`` truncated strictly to ``[lo, hi]``."""
    p_lo = _norm_cdf((lo - mean) / sd)
    p_hi = _norm_cdf((hi - mean) / sd)
    p = p_lo + float(u) * max(p_hi - p_lo, 1e-12)
    val = mean + sd * float(_inv_norm_cdf(np.array([p]))[0])
    return float(np.clip(val, lo, hi))


def load_posterior_samples(path: Path) -> pd.DataFrame:
    """Load an external AT2017gfo posterior table (CSV or JSON records) and clip to surrogate bounds."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw)
    else:
        df = pd.read_csv(path)

    missing = [k for k in EJECTA_KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"Posterior sample file {path} missing required columns: {missing}")

    out = df.copy()
    for k, (lo, hi) in BU2026_SURROGATE_BOUNDS.items():
        out[k] = out[k].astype(float).clip(lower=lo, upper=hi)
    if "inclination_em" in out.columns:
        out["inclination_em"] = out["inclination_em"].astype(float).clip(lower=0.0, upper=math.pi / 2.0)
    return out.reset_index(drop=True)


def draw_tier_parameters(
    tier: str,
    task_index: int = 0,
    n_tasks: int = 200,
    ensemble_seed: int = 20261125,
    rate: float = 1000.0,
    z_max: float = 0.15,
    posterior_samples: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Return kwargs for ``FixedBu2026KilonovaPopulation`` for ``(tier, task_index)``."""
    if tier not in SUITE08_TIERS:
        raise ValueError(f"Unknown Suite 08 tier {tier!r}; expected one of {SUITE08_TIERS}")
    if not (0 <= task_index < max(n_tasks, 1)):
        raise ValueError(f"task_index={task_index} outside [0, {n_tasks})")

    if tier == "Bu2026_Polar":
        return {
            **BASELINE_BU2026_EJECTA,
            "inclination_em": 0.45,
            "vary_inclination": False,
            "rate": float(rate),
            "z_max": float(z_max),
        }

    if tier == "Bu2026_Isotropic":
        return {
            **BASELINE_BU2026_EJECTA,
            "inclination_em": 0.45,
            "vary_inclination": True,
            "rate": float(rate),
            "z_max": float(z_max),
        }

    if tier in ("AT2017gfo_Posterior_Polar", "AT2017gfo_Posterior_Isotropic"):
        vary_inc = tier.endswith("_Isotropic")
        if posterior_samples is not None and len(posterior_samples) > 0:
            rng = np.random.default_rng(ensemble_seed)
            order = rng.choice(len(posterior_samples), size=n_tasks, replace=len(posterior_samples) < n_tasks)
            row = posterior_samples.iloc[int(order[task_index])]
            params = {k: float(row[k]) for k in EJECTA_KEYS}
            inc = float(row["inclination_em"]) if ("inclination_em" in row and not vary_inc) else 0.45
        else:
            # Deterministic Latin-hypercube draw from truncated Gaussian AT2017gfo posterior
            u = latin_hypercube_unit(n_tasks, len(EJECTA_KEYS) + 1, seed=ensemble_seed)[task_index]
            params = {}
            for j, k in enumerate(EJECTA_KEYS):
                mu, sd = AT2017GFO_POSTERIOR_GAUSSIAN[k]
                lo, hi = BU2026_SURROGATE_BOUNDS[k]
                params[k] = _truncated_normal_from_unit(float(u[j]), mu, sd, lo, hi)
            if vary_inc:
                inc = 0.45
            else:
                mu_i, sd_i = AT2017GFO_POSTERIOR_GAUSSIAN["inclination_em"]
                inc = _truncated_normal_from_unit(float(u[len(EJECTA_KEYS)]), mu_i, sd_i, 0.0, math.pi / 2.0)

        return {
            **params,
            "inclination_em": float(inc),
            "vary_inclination": vary_inc,
            "rate": float(rate),
            "z_max": float(z_max),
        }

    # tier == "BNS_NR_Population": stratified Latin-hypercube draw across NR BNS ejecta box
    u = latin_hypercube_unit(n_tasks, len(EJECTA_KEYS), seed=ensemble_seed + 7919)[task_index]
    params = {}
    for j, k in enumerate(EJECTA_KEYS):
        lo_nr, hi_nr = BNS_NR_PRIOR_BOUNDS[k]
        lo_s, hi_s = BU2026_SURROGATE_BOUNDS[k]
        lo, hi = max(lo_nr, lo_s), min(hi_nr, hi_s)
        params[k] = float(lo + float(u[j]) * (hi - lo))
    return {
        **params,
        "inclination_em": 0.45,
        "vary_inclination": True,
        "rate": float(rate),
        "z_max": float(z_max),
    }


def resolve_local_boom_hdf5_paths(data_dir: Path, start: str = "201803", end: str = "202603") -> List[str]:
    """Return sorted ``.h5`` paths inside ``data_dir`` whose ``YYYYMM`` token lies in ``[start, end]``.

    Compatible with both ``ztf_YYYYMM.h5`` (``survey_sim.load_ztf_survey_local`` convention) and
    prefixed exports such as ``ztf_searched_YYYYMM.h5``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Exposure directory does not exist: {data_dir}")
    matched: List[Tuple[str, Path]] = []
    for p in sorted(data_dir.glob("*.h5")):
        m = _YYYYMM_RE.search(p.stem)
        if not m:
            continue
        ym = m.group(1)
        if start <= ym <= end:
            matched.append((ym, p))
    if not matched:
        raise FileNotFoundError(f"No monthly ZTF HDF5 files in {data_dir} matching [{start}, {end}]")
    matched.sort(key=lambda item: (item[0], item[1].name))
    return [str(p) for _, p in matched]


def summarize_ensemble_tiers(
    runs_df: pd.DataFrame,
    epsilon_pipe: float = 1.0,
    n_obs: int = 0,
    by: str = "model",
) -> pd.DataFrame:
    """Pool Suite 08 realisations into marginal ``<VT>_eff`` and epistemic 16-50-84% intervals.

    For fixed-parameter tiers (``Bu2026_Polar``, ``Bu2026_Isotropic``), the draw-to-draw spread reflects
    pure binomial Monte Carlo noise. For parameter-sampled tiers (``AT2017gfo_Posterior_*``,
    ``BNS_NR_Population``), each realisation represents an independent ejecta draw ``theta_k``, so the
    injection-weighted mean estimates the prior/posterior-marginalized ``<VT>`` while ``vt_p16..vt_p84``
    and ``r95_p16..r95_p84`` quantify the astrophysical/epistemic spread across draws.
    """
    if not (0.0 < epsilon_pipe <= 1.0):
        raise ValueError(f"epsilon_pipe must lie in (0, 1], got {epsilon_pipe}")

    rows = []
    for label, g in runs_df.groupby(by, sort=False):
        w = g["n_simulated"].to_numpy(float)
        vt_raw = g["vt_gpc3_yr"].to_numpy(float)
        vt_eff_draws = epsilon_pipe * vt_raw
        n_sim = float(w.sum())
        n_det = int(g["n_detected"].sum())
        k_runs = len(g)

        vt_marg = float(np.sum(w * vt_eff_draws) / n_sim) if n_sim > 0 else np.nan
        if k_runs >= 2 and n_sim > 0:
            # Cluster-robust SE of weighted mean across K independent draws
            resid = vt_eff_draws - vt_marg
            vt_se = float(math.sqrt(k_runs / (k_runs - 1.0) * np.sum((w * resid) ** 2)) / n_sim)
        else:
            vt_se = float(vt_marg / math.sqrt(n_det)) if n_det > 0 else np.nan

        p16, p50, p84 = (float(x) for x in np.percentile(vt_eff_draws, [16.0, 50.0, 84.0]))
        r90_marg = float(rate_upper_limit(vt_marg, 0.90, n_obs=n_obs)) if vt_marg > 0 else np.nan
        r95_marg = float(rate_upper_limit(vt_marg, 0.95, n_obs=n_obs)) if vt_marg > 0 else np.nan
        r99_marg = float(rate_upper_limit(vt_marg, 0.99, n_obs=n_obs)) if vt_marg > 0 else np.nan

        # Lower VT corresponds to higher R95 upper limit, so quantiles invert monotonically
        r95_p16 = float(rate_upper_limit(p84, 0.95, n_obs=n_obs)) if p84 > 0 else np.nan
        r95_p50 = float(rate_upper_limit(p50, 0.95, n_obs=n_obs)) if p50 > 0 else np.nan
        r95_p84 = float(rate_upper_limit(p16, 0.95, n_obs=n_obs)) if p16 > 0 else np.nan

        rows.append({
            by: label,
            "n_runs": k_runs,
            "n_simulated": int(n_sim),
            "n_detected": n_det,
            "epsilon_pipe": float(epsilon_pipe),
            "vt_marginal_gpc3_yr": vt_marg,
            "vt_marginal_se": vt_se,
            "vt_p16_gpc3_yr": p16,
            "vt_p50_gpc3_yr": p50,
            "vt_p84_gpc3_yr": p84,
            "r90_marginal": r90_marg,
            "r95_marginal": r95_marg,
            "r99_marginal": r99_marg,
            "r95_p16": r95_p16,
            "r95_p50": r95_p50,
            "r95_p84": r95_p84,
        })
    return pd.DataFrame(rows)
