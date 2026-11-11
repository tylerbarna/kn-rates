# defunct/

Retired code and outputs, kept for provenance. Nothing here is used by the active pipeline, notebooks or tests
(except `superseded_suites/04_*/compare_with_andreoni2020.py`, whose helper is still unit-tested).

| Folder | What | Why retired |
|---|---|---|
| top level (`sim_*.sh`, `ZTF_sim_small.py`, …) | First-generation Slurm drivers | Superseded by `scripts/0N_*` (monolithic, over-allocated) |
| `preliminary_analyses/` | Root-level preliminary notebooks/scripts: detection-criteria sweeps v1/v2, light-curve quality, prior sensitivity, local analysis, rate-limit benchmark notebook, ad-hoc tests | Superseded by Suites 01–03/07 and the `knrates` notebooks |
| `notebook_generators/` | `build_notebooks.py`, `generate_paper_notebook.py`, `generate_v2_notebook.py`, Suite 05 `generate_notebook.py` | Wrote notebooks from Python strings containing hard-coded results and synthetic fall-back data; would overwrite the data-driven notebooks. Notebooks are now the source of truth. |
| `deprecated_scripts/` | Old criteria-sweep launchers (already disabled with `exit 1`), forwarders, prior-sensitivity and rate-limit-benchmark Slurm scripts | Replaced by `scripts/submit_fast_criteria_sweep.sh` etc. |
| `superseded_suites/` | Former Suites 04 (Andreoni comparison), 05 (sample convergence), 06 (inclination) | **Did not run simulations.** 05 and 06 binomially re-drew detection counts from hard-coded efficiencies / typed-in magnitude and VT tables; 04 used an inverted duration scaling and an unpublished "2020 Gpc⁻³ yr⁻¹" Metzger benchmark. Replaced by notebooks 04–06, which use the Suite 01/07 realisations. Their `results/0[456]_*` outputs must not be cited. |
| `notebooks_by_suite/` | The per-suite analysis notebooks (01–07, paper figures, telemetry) as of 2026-11-10 | Replaced by the notebooks in `notebooks/`, which follow the paper's argument and draw figures from `knrates/figures.py`. These still run (`python execute_notebooks.py defunct/notebooks_by_suite/*.ipynb`) and hold the longer per-figure explanations; the paper-figure notebook here has `SAVE_FIGURES = False`, so it never overwrites `paper/`. |
| `paper_figures_unsupported/` | Former paper Figures 4, 6, 7 | Plotted hard-coded arrays (archetype bias VTs, criteria elasticities, imposter purity and colour curves) with no simulation behind them. |
