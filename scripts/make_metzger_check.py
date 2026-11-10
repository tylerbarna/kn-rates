#!/usr/bin/env python3
"""Light-curve check of the one-zone Metzger models against AT2017gfo.

Writes to results/model_lightcurves/ (picked up by `python -m knrates.collect --suites models`):
  * metzger_lightcurves.csv   -- survey_sim's Metzger light curves (absolute AB mag, g/r/i, no extinction), computed
                                 by the Rust helper in scripts/metzger_lightcurves/ with the exact library function
                                 survey_sim uses
  * at2017gfo_photometry.csv  -- PS1 g/r/i photometry of AT2017gfo as absolute magnitudes (data/at2017gfo/)

Requires cargo; the helper builds offline if survey_sim's dependencies are in the local cargo cache.
"""
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model_lightcurves"
D_L_MPC = 40.7                                   # Cantiello et al. 2018 (knrates.literature.AT2017GFO)
T0 = pd.Timestamp("2017-08-17T12:41:04.4")       # GW170817 merger time (UTC)


def metzger_curves() -> pd.DataFrame:
    crate = ROOT / "scripts" / "metzger_lightcurves"
    res = subprocess.run(["cargo", "run", "--release", "--quiet", "--manifest-path", str(crate / "Cargo.toml")],
                         check=True, capture_output=True, text=True)
    return pd.read_csv(io.StringIO(res.stdout))


def at2017gfo_photometry() -> pd.DataFrame:
    ph = pd.read_csv(ROOT / "data" / "at2017gfo" / "AT2017gfo_corrected_nmma.dat", sep=r"\s+",
                     names=["utc", "filter", "mag", "mag_err"])
    ph = ph[ph["filter"].isin(["ps1::g", "ps1::r", "ps1::i"]) & (ph.mag_err < 0.5)].copy()
    ph["band"] = ph["filter"].str[-1]
    ph["t_days"] = (pd.to_datetime(ph.utc) - T0).dt.total_seconds() / 86400
    ph["abs_mag"] = ph.mag - 5 * np.log10(D_L_MPC * 1e6 / 10)
    ph["param_set"] = "AT2017gfo_PS1"
    return ph[["param_set", "band", "t_days", "abs_mag", "mag_err"]].sort_values(["band", "t_days"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = metzger_curves()
    m.to_csv(OUT / "metzger_lightcurves.csv", index=False)
    p = at2017gfo_photometry()
    p.to_csv(OUT / "at2017gfo_photometry.csv", index=False)
    print(f"wrote {len(m)} model rows and {len(p)} photometry rows to {OUT}")
