# Decisions log

What was decided, why, and where it shows up. Numbers are not repeated here: `notebooks/00_overview.ipynb`
recomputes them from the current snapshot. Newest first. Add an entry whenever a choice changes what the
paper says or which runs count.

| Date | Decision | Why | Where |
|---|---|---|---|
| 2026-11-10 | Notebooks reorganised by the paper's argument (00–05 + `paper_figures`); the per-suite notebooks moved to `defunct/notebooks_by_suite/` (they still run). Each figure is defined once in `knrates/figures.py`. | The per-suite layout followed how runs were produced, not what they show; figure code existed in two or three copies. | `notebooks/`, `knrates/figures.py`, `knrates/notebook.py` |
| 2026-11-10 | Paper Fig. 1 compares with Andreoni et al. (2020) through 2020 January (23 months), not 2019 December. | Off-by-one in the date cut: the label read −34 % instead of −40 %, and the dashed line started a month early. | `knrates/figures.py::sensitivity_timeline` |
| 2026-11-10 | Suite 01 is the main analysis; Suite 07 (GW170817 parameter set) appears only as a robustness check. The paper's headline numbers come from Suite 01. | The two sets differ only by 0.1 dex in the dynamical ejecta mass; showing both side by side confused every figure. | `04_systematics` §3, `ms.tex` §3–4 |
| 2026-11-10 | Suite 02 filter keeps survey_sim's rule: a source passes the evolution test if it fades **or** rises fast enough; 0 switches a branch off. | Matches the simulation whose ⟨VT⟩ it is compared with. Consequence: a rise threshold *adds* volume, and fade = 0 means no evolution test. | `05_criteria_companion` |
| 2026-11-10 | Suite 02 candidate sets are reused only if their metadata matches the current envelope, parameters, seed and number of injections; otherwise they are re-simulated. | The first ensemble job silently reused sets made with the old envelope, and the filter rejected them all ([PR #23](https://github.com/tylerbarna/kn-rates/pull/23)). | `analysis_suites/02_*/simulate_candidates.py` |
| 2026-11 | One-zone (Metzger) models dropped from all results (`include_metzger=False` by default). | Compared with AT2017gfo photometry, the "blue" and "mean" presets are 1–3 mag too bright, and the κ = 398 template is ≈0.7 mag too faint in g and fades far too fast after 2 d. | `04_systematics` §4 (switch `INCLUDE_METZGER`) |
| 2026-11 | Andreoni et al. (2020)'s "first 23 months" taken as 2018 March – 2020 January. Published limits are carried to eight years with the ⟨VT⟩ growth we measure, never by survey duration. | The paper gives no dates. ZTF's ⟨VT⟩ grew faster than its duration, so duration scaling understates eight years. | `knrates/analysis.py::ANDREONI_WINDOW`, `02_survey_time_and_previous_searches` |
| 2026-11 | The Suite 01/07 POSSIS parameter sets are described as "AT2017gfo-like ejecta", not as a fit to AT2017gfo. A posterior-marginalised run is an open item. | The pinned parameters are not a published AT2017gfo fit. | `ms.tex` §3 (`\tbd`) |
| 2026-11 | Suite 03 single-draw `BroadPrior_*` populations retired; `BroadPopulation_Isotropic` draws parameters per transient. | A single draw per realisation is one parameter set, not a population. | `04_systematics` §6 |
| 2026-11 | Former Suites 04–06 retired; their outputs must not be cited. Their questions are answered from the Suite 01 realisations. | They ran no simulations: they re-drew counts from hard-coded efficiencies or typed-in tables. | `defunct/superseded_suites/` |
| 2026-11 | All limits assume N_obs = 0. The archival candidate search is a separate, still-open piece of work. | The repository contains no search of real ZTF data. | `ms.tex` §7 (`\tbd`) |
| 2026-11 | z_max = 0.15 kept for the existing runs. | 99.9 % of ⟨VT⟩ comes from z < 0.08; a smaller boundary would only reduce MC noise in new runs. | `04_systematics` §2 |

## Open questions (decide, then log here)

- Replace the fixed POSSIS parameters with draws from a published AT2017gfo posterior?
  The current model is brighter than AT2017gfo in r and i over the first two days (`04_systematics` §4).
- Re-run Suite 03 at scale before quoting a population-averaged limit: it currently has 3 realisations per cell.
- Fix or remove the AND-logic validation path in `fast_criteria_filter.py`, which disagrees with the production OR rule.
