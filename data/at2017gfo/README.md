# AT2017gfo photometry

`AT2017gfo_corrected_nmma.dat` is copied unchanged from the NMMA repository
(`example_files/lightcurves/AT2017gfo_corrected.dat`, nmma commit `8fd36287`). Columns: UTC time, filter, AB
magnitude, uncertainty. The magnitudes are corrected for Milky Way extinction (the PS1 g values are 0.39 mag brighter
than in NMMA's uncorrected `AT2017gfo.dat`, i.e. A_g for E(B−V) ≈ 0.105).

Used by `scripts/make_metzger_check.py`, which converts the PS1 g/r/i points to absolute magnitudes with
d_L = 40.7 Mpc (Cantiello et al. 2018) and t0 = 2017-08-17 12:41:04.4 UTC (GW170817).
