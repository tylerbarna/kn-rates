#!/usr/bin/env python3
"""Tabulate rest-frame absolute-magnitude light curves of the simulated Bu2026 (fiesta) models.

Evaluates the same surrogate survey_sim uses (``FiestaKNModel(survey="ztf")``, Bu2026_MLP) at 10 pc and z≈0 for the
Suite 01 and Suite 07 ejecta parameters over a grid of viewing angles, and writes
``results/model_lightcurves/bu2026_lightcurves.csv`` (columns: param_set, inclination_deg, band, t_days, abs_mag).
The results snapshot (``python -m knrates.collect``) picks the file up as table ``models_lightcurves``.

Run in an environment with survey_sim + fiesta (e.g. ``conda activate surveysim``)::

    python scripts/make_model_lightcurves.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_SETS = {
    # Suite 01 Bu2026Fixed / Bu2026Vary ejecta (analysis_suites/01_large_scale_rate_limits/run_large_scale_simulation.py)
    "suite01": dict(log10_mej_dyn=-1.7, v_ej_dyn=0.20, Ye_dyn=0.15, log10_mej_wind=-1.1, v_ej_wind=0.10, Ye_wind=0.35),
    # Suite 07 GW170817_Bu2026_* ejecta (analysis_suites/07_gw170817_rate_limits/run_gw170817_simulation.py)
    "suite07": dict(log10_mej_dyn=-1.8, v_ej_dyn=0.20, Ye_dyn=0.15, log10_mej_wind=-1.1, v_ej_wind=0.10, Ye_wind=0.35),
}
INCLINATIONS_DEG = [0.0, 15.0, 25.8, 36.9, 45.6, 53.1, 60.0, 72.5, 84.3, 90.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/model_lightcurves/bu2026_lightcurves.csv")
    args = ap.parse_args()

    import jax.numpy as jnp
    from survey_sim.fiesta_model import FiestaKNModel

    fm = FiestaKNModel(survey="ztf")
    names = list(fm.model.parameter_names)
    rows = []
    for set_name, base in PARAM_SETS.items():
        for inc in INCLINATIONS_DEG:
            p = dict(base, inclination_EM=np.radians(inc))
            fp = {}
            for k in names:
                key = next((c for c in (k, k.replace("Ye", "ye"), k.replace("ye", "Ye"), k.lower()) if c in p), None)
                if key is None:
                    raise KeyError(f"surrogate parameter {k} not provided")
                fp[k] = jnp.float32(p[key])
            fp["luminosity_distance"] = jnp.float32(1e-5)  # 10 pc -> absolute magnitudes
            fp["redshift"] = jnp.float32(0.0)
            t, mags = fm.model.predict(fp)
            t = np.asarray(t, dtype=float)
            for filt, short in fm.reverse_map.items():
                if filt in mags:
                    m = np.asarray(mags[filt], dtype=float)
                    rows += [dict(param_set=set_name, inclination_deg=inc, band=short, t_days=ti, abs_mag=mi)
                             for ti, mi in zip(t, m)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows; parameters {names})")


if __name__ == "__main__":
    main()
