# `kn-rates` Operator & Researcher Guide

This guide explains how a human researcher can inspect, re-run, and extend the ZTF 8-year kilonova rate simulation and manuscript pipeline without needing any AI agent tooling.

---

## 1. How the Ecosystem Fits Together

The ZTF kilonova rate measurement is split across four clean repositories:

1. **`survey-sim` (Frozen Engine):** Rust/Python/JAX Monte Carlo simulator (`SimulationPipeline`, `FiestaKNModel`, `SurveyStore`). Never edited directly by downstream scripts.
2. **`real-ztf-search` (Archival Data Search):** Runs Stages A–E on ZTF public + partnership + Caltech observations (`2018-03-17` – `2026-03-31`), exports the exact searched exposure HDF5 logs (`knsearch.exposure_export`), and compiles public-safe aggregate search metrics into `snapshots/knsearch_results_latest.zip`.
3. **`ztf-vetting` (Stage E Portal):** Authenticated React web portal where human reviewers classify Stage D survivors alongside single-blind synthetic kilonova injections.
4. **`kn-rates` (This Repo — Ensembles, Rate Inference & Paper I/II):** Runs the injection–recovery ensembles (`analysis_suites/`), compiles `snapshots/knrates_results_latest.zip`, loads both snapshots in `knrates.notebook.setup()`, generates all figures/tables/LaTeX macros (`paper/numbers.tex`), and builds `paper/ms.pdf`.

---

## 2. Everyday Local Commands (No Cluster Required)

Everything needed for notebooks, figures, tables, and manuscript compilation works offline from the version-controlled ZIP snapshots:

```bash
# 1. Install package in editable mode (once)
pip install -e .

# 2. Run the unit test suite (fast, no survey-sim/JAX required when ignoring Suite 07 engine test)
python3 -m pytest -q --ignore=tests/test_suite07_gw170817.py

# 3. Regenerate LaTeX prose macros (paper/numbers.tex) directly from the latest snapshot
python3 -m knrates.macros

# 4. Re-execute the Paper I & II figure notebooks and all analysis notebooks headlessly
python3 execute_notebooks.py notebooks/paper_figures.ipynb notebooks/companion/paper2_figures.ipynb
python3 execute_notebooks.py

# 5. Compile Paper I (AASTeX v6.31)
cd paper && tectonic -X compile ms.tex   # or latexmk -pdf ms.tex
```

---

## 3. Package Map (`knrates/`)

| Module | What It Does | Key Entry Points |
| :--- | :--- | :--- |
| `knrates/criteria.py` | Canonical 10-parameter kilonova selection criteria (`FIDUCIAL_CRITERIA`) and weighted least-squares per-band rate fitter (`band_wls_rates`), shared verbatim with `real-ztf-search`. | `band_wls_rates(pts)`, `evaluate(pts, cfg)`, `passes(pts, cfg)` |
| `knrates/snapshot.py` | Reads `snapshots/knrates_results_*.zip` into a dict-like `Snapshot` object of pandas DataFrames + `manifest.json`. | `load_snapshot("latest")` |
| `knrates/collect.py` | Compiles raw MSI `results/` JSON/CSV outputs into `snapshots/knrates_results_latest.zip` with schema, efficiency, and non-shrinking guards. | `python3 -m knrates.collect` |
| `knrates/stats.py` | Frequentist Poisson upper limits (`n_obs >= 0`), injection-weighted pooled $\langle VT \rangle$, Clopper–Pearson intervals, and hypergeometric sub-sampling. | `poisson_factor`, `rate_upper_limit`, `pooled_vt`, `summarize_runs` |
| `knrates/analysis.py` | Flat-$\Lambda$CDM distances/volumes, epoch-resolved $\langle VT \rangle(t)$, viewing-angle $\langle VT \rangle(\iota)$, criteria elasticity, and Andreoni+2020 comparisons. | `headline_limits`, `vt_by_epoch`, `inclination_efficiency`, `andreoni_comparison` |
| `knrates/ensembles.py` | Suite 08 matched-exposure configuration (`FADE_ONLY_MIN_RISE_RATE = 99.0`), deterministic Latin-hypercube draws across Paper I's 5 model tiers, and marginal/epistemic pooling. | `draw_tier_parameters`, `resolve_local_boom_hdf5_paths`, `summarize_ensemble_tiers` |
| `knrates/macros.py` | Writes strictly alphabetical `\newcommand` macros to `paper/numbers.tex` so manuscript prose never drifts from snapshot numbers. | `compute_paper1_macros`, `write_paper1_macros` |
| `knrates/figures.py` | One function per publication/diagnostic figure (used identically by interactive notebooks and `paper_figures.ipynb`). | `limit_summary`, `inclination_vt`, `vt_timeline`, `stage_a_cutflow`, ... |
| `knrates/notebook.py` | One-line notebook initializer loading both the simulation snapshot and (automatically when sibling `real-ztf-search` is present) the archival-search snapshot. | `ctx = setup("latest", search_snapshot="auto")` |

---

## 4. Running Suite 08 Matched Ensembles on MSI Agate

Once `real-ztf-search` exports the restricted monthly HDF5 exposure logs (`ztf_YYYYMM.h5`) for the searched ZTF observations:

```bash
# On MSI Agate login node:
conda activate kn_rates_dev

# Submit the 5-tier stratified Suite 08 Slurm array (200 draws x 500k injections per tier)
EXPOSURE_DIR=/path/to/ztf_boom_searched sbatch scripts/08_run_matched_ensembles.sh

# After array completion, rebuild and verify the version-controlled snapshot
python3 -m knrates.collect --timestamped --also-latest
git add snapshots/knrates_results_latest.zip && git commit -m "Update snapshot with Suite 08 matched ensembles"
```
