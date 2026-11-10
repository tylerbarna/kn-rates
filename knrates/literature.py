"""Published reference values used for comparison, with sources.

Every number here was checked against the arXiv text of the cited paper
(literature review, 2026-11).  Values flagged ``verified=False`` could not be
confirmed from a primary source and must not be quoted without checking.
Rates are in Gpc^-3 yr^-1.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Optical kilonova rate upper limits
# ---------------------------------------------------------------------------
RATE_LIMITS = [
    dict(key="andreoni2020_gw170817", survey="ZTF (first 23 months)", ref="Andreoni et al. 2020",
         arxiv="2008.00008", model="GW170817-like", limit=1775.0, cl="95%", verified=True, data_end="2020-01-31",
         note="headline case: two 5-sigma detections; simsurvey injections; no Galactic-latitude cut"),
    dict(key="andreoni2020_possis_isotropic", survey="ZTF (first 23 months)", ref="Andreoni et al. 2020",
         arxiv="2008.00008", model="POSSIS, uniform viewing angle", limit=4029.0, cl="95% (implied)", verified=True,
         data_end="2020-01-31",
         note="CL not stated explicitly next to the number; method (3 recovered injections, Gehrels 1986) implies 95%"),
    dict(key="andreoni2020_tophat_m16_2det", survey="ZTF (first 23 months)", ref="Andreoni et al. 2020",
         arxiv="2008.00008", model="top-hat M=-16, 2 detections", limit=930.0, cl="95%", verified=True, data_end="2020-01-31", note=""),
    dict(key="andreoni2020_linear_decline_2det", survey="ZTF (first 23 months)", ref="Andreoni et al. 2020",
         arxiv="2008.00008", model="linear decline M=-16.5, 0.7 mag/d, 2 det.", limit=937.0, cl="95%",
         verified=True, data_end="2020-01-31", note=""),
    dict(key="andreoni2021_ztfrest_combined", survey="ZTF (ZTFReST + 2020 search)", ref="Andreoni et al. 2021",
         arxiv="2104.06352", model="GW170817-like", limit=900.0, cl="not stated (95% per secondary source)",
         verified=True, data_end="2021-03-03", note="combination of 1775 and 1904 by adding VT; CL not confirmed in v1 text"),
    dict(key="kasliwal2017_iptf", survey="iPTF", ref="Kasliwal et al. 2017", arxiv="1710.05436",
         model="GW170817-like", limit=800.0, cl="3-sigma", verified=True, note=""),
    dict(key="smartt2017_atlas", survey="ATLAS", ref="Smartt et al. 2017", arxiv="1710.05841",
         model="GW170817-like", limit=3.0e4, cl="95%", verified=True, note=""),
    dict(key="doctor2017_des", survey="DES-SN", ref="Doctor et al. 2017", arxiv="1611.08052",
         model="brightest model (M_i=-16.2)", limit=2.4e4, cl="90%", verified=True, note=""),
    dict(key="almualla2021_ztf_simulated", survey="ZTF simulated schedules", ref="Almualla et al. 2021",
         arxiv="2011.10421", model="POSSIS, cos-uniform viewing angle", limit=2749.0, cl="95%", verified=True,
         note="simulation study"),
]

# ---------------------------------------------------------------------------
# Gravitational-wave BNS merger rate (90% credible ranges, union over methods)
# ---------------------------------------------------------------------------
GW_BNS_RATES = [
    dict(key="gwtc3_bns_rate", ref="LVK GWTC-3 population (Abbott et al. 2023)", arxiv="2111.03634",
         low=10.0, high=1700.0, verified=True),
    dict(key="gwtc4_bns_rate", ref="LVK GWTC-4.0 population (2025)", arxiv="2508.18083",
         low=7.6, high=250.0, verified=True,
         note="union of FullPop-4.0 and BGP 90% intervals (arXiv v2); published version may quote 8.8-250 (unverified)"),
]

# ---------------------------------------------------------------------------
# AT2017gfo / GW170817 reference parameters
# ---------------------------------------------------------------------------
AT2017GFO = dict(
    distance_mpc=40.7, distance_ref="Cantiello et al. 2018 (SBF), arXiv:1801.06080",
    viewing_angle_deg=(19.0, 25.0), viewing_angle_ref="Mooley et al. 2022, arXiv:2210.06568 (VLBI); 14-28 deg in Mooley+2018",
    peak_abs_mag={"g": -15.96, "r": -15.97, "i": -15.71}, peak_ref="Drout et al. 2017, arXiv:1710.05443",
    decline_mag_per_day={"g": 1.14, "r": 0.95, "i": 0.57}, decline_ref="Drout et al. 2017 (max to +5.5 d)",
    villar17_components={
        "blue": dict(kappa=0.5, mej=0.020, vej=0.266),
        "purple": dict(kappa=3.0, mej=0.047, vej=0.152),
        "red": dict(kappa=10.0, mej=0.011, vej=0.137),
    },
    villar17_ref="Villar et al. 2017, arXiv:1710.11576 (three-component one-zone fit)",
)

# fiesta POSSIS surrogate training domain (Koehn et al. 2025, arXiv:2507.13807)
FIESTA_TRAINING_RANGE = {
    "log10_mej_dyn": (-3.0, -1.3), "v_ej_dyn": (0.12, 0.28), "ye_dyn": (0.15, 0.35),
    "log10_mej_wind": (-2.0, -0.9), "v_ej_wind": (0.05, 0.15), "ye_wind": (0.20, 0.40),
    "inclination_rad": (0.0, 1.5707963267948966), "time_days": (0.2, 26.0),
}

# Training box of the surrogate actually used by survey_sim ("Bu2026_MLP" metadata, read from the model files
# in the survey_sim environment, Nov 2026). It is wider than the ranges quoted in the fiesta paper above; the MLP
# does not clip, so parameters outside this box are silently extrapolated. Source-frame time 0.194-27.5 d.
BU2026_MLP_TRAINING_RANGE = {
    "log10_mej_dyn": (-4.0, -1.301), "v_ej_dyn": (0.12, 0.35), "ye_dyn": (0.15, 0.35),
    "log10_mej_wind": (-4.0, -0.550), "v_ej_wind": (0.05, 0.15), "ye_wind": (0.20, 0.40),
    "inclination_rad": (0.0, 1.5707963267948966), "time_days": (0.194, 27.5),
}

# Physically plausible grey opacities for r-process ejecta (Tanaka et al. 2020, arXiv:1906.08914)
KAPPA_RANGE_CM2_G = (0.5, 30.0)


# Other reference values drawn in figures (AT2017gfo / GW170817 properties)
OTHER_REFERENCES = [
    dict(key="mooley2022_gw170817_viewing_angle", ref="Mooley et al. 2022", arxiv="2210.06568",
         what="GW170817 viewing angle 19-25 deg (VLBI + HST)", low=19.0, high=25.0),
    dict(key="mooley2018_gw170817_viewing_angle", ref="Mooley et al. 2018", arxiv="1806.09693",
         what="GW170817 viewing angle 14-28 deg (VLBI superluminal motion)", low=14.0, high=28.0),
    dict(key="drout2017_at2017gfo_peak_g", ref="Drout et al. 2017", arxiv="1710.05443",
         what="AT2017gfo peak absolute magnitude, g", band="g", value=-15.96),
    dict(key="drout2017_at2017gfo_peak_r", ref="Drout et al. 2017", arxiv="1710.05443",
         what="AT2017gfo peak absolute magnitude, r", band="r", value=-15.97),
    dict(key="drout2017_at2017gfo_peak_i", ref="Drout et al. 2017", arxiv="1710.05443",
         what="AT2017gfo peak absolute magnitude, i", band="i", value=-15.71),
]

# Keys used before Nov 2026, still accepted by lookup()
_OLD_KEYS = {
    "andreoni20_gw170817": "andreoni2020_gw170817", "andreoni20_possis_iso": "andreoni2020_possis_isotropic",
    "andreoni20_tophat16_2det": "andreoni2020_tophat_m16_2det", "andreoni20_linear_2det": "andreoni2020_linear_decline_2det",
    "andreoni21_ztfrest_combined": "andreoni2021_ztfrest_combined", "kasliwal17_ptf": "kasliwal2017_iptf",
    "smartt17_atlas": "smartt2017_atlas", "doctor17_des": "doctor2017_des", "almualla21_ztf_sim": "almualla2021_ztf_simulated",
    "gwtc3": "gwtc3_bns_rate", "gwtc4": "gwtc4_bns_rate",
}


def rate_limits_table(verified_only: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(RATE_LIMITS)
    return df[df["verified"]] if verified_only else df


def _all_rows():
    return RATE_LIMITS + GW_BNS_RATES + OTHER_REFERENCES


def keys(kind: str = "all") -> list:
    """Selectable keys: ``"rate_limits"``, ``"gw_rates"``, ``"other"`` or ``"all"``."""
    rows = {"rate_limits": RATE_LIMITS, "gw_rates": GW_BNS_RATES, "other": OTHER_REFERENCES}.get(kind, _all_rows())
    return [r["key"] for r in rows]


def lookup(key: str) -> dict:
    key = _OLD_KEYS.get(key, key)
    for row in _all_rows():
        if row["key"] == key:
            return row
    raise KeyError(f"unknown literature key {key!r}; options: {keys()}")


def select(selected, kind: str = "all") -> list:
    """Validate a user selection of literature keys (in the user's order); unknown keys raise with the options."""
    allowed = set(keys(kind))
    out = []
    for k in selected:
        k = _OLD_KEYS.get(k, k)
        if k not in allowed:
            raise KeyError(f"{k!r} is not a valid choice here; options: {sorted(allowed)}")
        out.append(lookup(k))
    return out


def url(key: str) -> str:
    return f"https://arxiv.org/abs/{lookup(key)['arxiv']}"


def cite(key: str) -> str:
    """Markdown hyperlink to the publication behind ``key``."""
    r = lookup(key)
    return f"[{r['ref']}]({url(key)})"


def describe(key: str) -> str:
    """One-line description of what ``key`` contributes to a figure."""
    r = lookup(key)
    if "limit" in r:
        return f"{r['survey']}, {r['model']}: R < {r['limit']:g} Gpc^-3 yr^-1 ({r['cl']})"
    if "low" in r and "what" not in r:
        return f"BNS merger rate {r['low']:g}-{r['high']:g} Gpc^-3 yr^-1 (90%)"
    return r["what"]
