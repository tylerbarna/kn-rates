"""Unit tests for the granular detection criteria sweep engine and MSI HPC compliance."""

import re
import unittest
from pathlib import Path

# Import the granular sweep functions
import sys
sys.path.insert(0, str(Path(__file__).parent / "analysis_suites" / "02_granular_criteria_sweep"))

from run_granular_sweep import (
    EXPANDED_CRITERIA_BOUNDS,
    FIDUCIAL_CRITERIA,
    get_grid_combinations,
    calculate_chunking,
    is_physically_valid,
    make_label,
    save_sweep_result,
)
from audit_granular_sweep import parse_label_metadata


class TestGranularCriteriaGrid(unittest.TestCase):
    def test_all_10_criteria_present_in_expanded_bounds(self):
        """Ensure all 10 items defined as detection criteria are in expanded bounds."""
        expected_keys = {
            "snr_threshold",
            "snr_threshold_secondary",
            "min_detections",
            "min_detections_primary",
            "max_timespan_days",
            "min_time_separation_hours",
            "min_rise_rate",
            "min_fade_rate",
            "min_galactic_lat",
            "require_fast_transient",
        }
        self.assertEqual(set(EXPANDED_CRITERIA_BOUNDS.keys()), expected_keys)
        self.assertEqual(set(FIDUCIAL_CRITERIA.keys()), expected_keys)

    def test_all_10_criteria_in_grid_combinations(self):
        """Ensure every grid combination dictionary contains all 10 keys."""
        grid = get_grid_combinations("broad")
        self.assertGreater(len(grid), 0)
        expected_keys = {
            "snr_threshold",
            "snr_threshold_secondary",
            "min_detections",
            "min_detections_primary",
            "max_timespan_days",
            "min_time_separation_hours",
            "min_rise_rate",
            "min_fade_rate",
            "min_galactic_lat",
            "require_fast_transient",
        }
        for pt in grid[:50]:  # check sample
            self.assertEqual(set(pt.keys()), expected_keys)

    def test_physical_validity_filtering(self):
        """Ensure unphysical or impossible criteria combinations are rejected."""
        # Valid config
        valid_cfg = dict(FIDUCIAL_CRITERIA)
        self.assertTrue(is_physically_valid(valid_cfg))

        # Secondary SNR > Primary SNR
        invalid_snr = dict(valid_cfg, snr_threshold=4.0, snr_threshold_secondary=5.0)
        self.assertFalse(is_physically_valid(invalid_snr))

        # Min Primary Detections > Min Total Detections
        invalid_ndet = dict(valid_cfg, min_detections=2, min_detections_primary=3)
        self.assertFalse(is_physically_valid(invalid_ndet))

        # Require fast transient False but non-zero fade rate
        invalid_fade = dict(valid_cfg, require_fast_transient=False, min_fade_rate=0.3)
        self.assertFalse(is_physically_valid(invalid_fade))

    def test_msi_hpc_compliance_all_presets(self):
        """Ensure all presets chunk into <= 5,000 Slurm array tasks."""
        presets = ["canonical", "broad", "fine", "ultra_fine", "legacy_ultra_fine", "all_slices"]
        for preset in presets:
            grid = get_grid_combinations(preset)
            total = len(grid)
            chunk_size, n_tasks = calculate_chunking(total, max_tasks=5000)
            self.assertLessEqual(
                n_tasks,
                5000,
                f"Preset {preset} produced {n_tasks} tasks which exceeds the 5,000 MSI limit!",
            )
            self.assertGreater(n_tasks, 0)
            self.assertGreater(chunk_size, 0)
            self.assertGreaterEqual(n_tasks * chunk_size, total)

    def test_broad_preset_chunking_compliance(self):
        """Verify the expanded broad grid (68,400 points) satisfies <= 5,000 tasks with chunk_size=15."""
        grid = get_grid_combinations("broad")
        total = len(grid)
        self.assertEqual(total, 68400)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=15, max_tasks=5000)
        self.assertEqual(chunk_size, 15)
        self.assertEqual(n_tasks, 4560)
        self.assertLessEqual(n_tasks, 5000)

    def test_ultra_fine_preset_chunking_compliance(self):
        """Verify the ultra-fine grid (2,138,460 points) satisfies <= 5,000 tasks with chunk_size=428."""
        grid = get_grid_combinations("ultra_fine")
        total = len(grid)
        self.assertEqual(total, 2138460)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=35, max_tasks=5000)
        self.assertEqual(chunk_size, 428)
        self.assertEqual(n_tasks, 4997)
        self.assertLessEqual(n_tasks, 5000)

    def test_legacy_ultra_fine_preset_chunking_compliance(self):
        """Verify the legacy ultra-fine grid (158,460 points) satisfies <= 5,000 tasks with chunk_size=35."""
        grid = get_grid_combinations("legacy_ultra_fine")
        total = len(grid)
        self.assertEqual(total, 158460)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=35, max_tasks=5000)
        self.assertEqual(chunk_size, 35)
        self.assertEqual(n_tasks, 4528)
        self.assertLessEqual(n_tasks, 5000)

    def test_fine_preset_chunking_compliance(self):
        """Verify the fine grid (22,464 points) satisfies <= 5,000 tasks with chunk_size=5."""
        grid = get_grid_combinations("fine")
        total = len(grid)
        self.assertEqual(total, 22464)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=5, max_tasks=5000)
        self.assertEqual(chunk_size, 5)
        self.assertEqual(n_tasks, 4493)
        self.assertLessEqual(n_tasks, 5000)

    def test_label_generation_and_parsing_roundtrip(self):
        """Ensure make_label and parse_label_metadata accurately serialize and deserialize all 10 criteria."""
        cfg = {
            "snr_threshold": 4.5,
            "snr_threshold_secondary": 2.5,
            "min_detections": 3,
            "min_detections_primary": 2,
            "max_timespan_days": 7.0,
            "min_time_separation_hours": 6.0,
            "min_rise_rate": 0.5,
            "min_fade_rate": 0.3,
            "min_galactic_lat": 15.0,
            "require_fast_transient": True,
        }
        label = make_label(cfg)
        self.assertIn("rise0.5", label)
        self.assertIn("fast1", label)
        self.assertIn("snr4.5", label)
        self.assertIn("snr22.5", label)

        parsed = parse_label_metadata(label)
        for k, v in cfg.items():
            self.assertEqual(parsed[k], v, f"Mismatch for key {k}: expected {v}, got {parsed[k]}")

    def test_slurm_script_directives(self):
        """Verify fast candidate simulation script has safe array directives protecting FairShare."""
        script_path = Path(__file__).parent / "scripts" / "02_run_fast_candidates_simulation.sh"
        self.assertTrue(script_path.exists())
        content = script_path.read_text()

        # Check array directive (supports unthrottled 0-N or throttled 0-N%M)
        match = re.search(r"#SBATCH\s+--array=0-(\d+)(?:%(\d+))?", content)
        self.assertIsNotNone(match, "Could not find #SBATCH --array directive in script!")
        max_idx = int(match.group(1))

        # Under the fast Simulate-Once architecture, array tasks are only 3 (fiducial) to 60 (ensemble)
        self.assertLessEqual(max_idx + 1, 60, "Fast criteria simulation array tasks must be <= 60 to protect FairShare!")

        # Verify legacy script is safely disabled with deprecation exit
        legacy_path = Path(__file__).parent / "scripts" / "02_run_granular_criteria_sweep.sh"
        self.assertTrue(legacy_path.exists())
        self.assertIn("DEPRECATED", legacy_path.read_text())

    def test_save_compact_summary_result(self):
        """Verify compact summary saves lightweight JSON (<1 KB) with full rate summary."""
        import tempfile
        import json

        class MockRS:
            transient_type = "Kilonova"
            volumetric_rate = 1000.0
            detections_per_year = 0.5
            detections_total = 4.0
            overall_efficiency = 0.0003
            n_simulated = 100000
            n_detected = 30
            z_max = 0.15
            survey_omega_sr = 2.1
            survey_duration_years = 8.0
            effective_vt_gpc3_yr = 4.2e-4

        class MockResult:
            n_simulated = 100000
            n_detected = 30
            rate_summaries = [MockRS()]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sim_result_test.json"
            save_sweep_result(
                result=MockResult(),
                filepath=out_file,
                compact_summary_only=True,
            )
            self.assertTrue(out_file.exists())
            sz = out_file.stat().st_size
            self.assertLess(sz, 1024, f"Compact JSON must be < 1 KB, got {sz} bytes")

            data = json.loads(out_file.read_text())
            self.assertEqual(data["n_simulated"], 100000)
            self.assertEqual(data["n_detected"], 30)
            self.assertEqual(len(data["rate_summaries"]), 1)
            self.assertEqual(data["rate_summaries"][0]["effective_vt_gpc3_yr"], 4.2e-4)


class TestFastCriteriaFilterAndCandidates(unittest.TestCase):
    """Test suite verifying the 'Simulate-Once, Filter-Many' candidate filtering engine."""

    def test_loose_bounding_criteria_encompasses_all_presets(self):
        """Ensure LOOSE_BOUNDING_CRITERIA is strictly more permissive than all search bounds."""
        from simulate_candidates import LOOSE_BOUNDING_CRITERIA

        # snr_threshold <= minimum snr in expanded bounds
        min_snr = min(EXPANDED_CRITERIA_BOUNDS["snr_threshold"])
        self.assertLessEqual(LOOSE_BOUNDING_CRITERIA["snr_threshold"], min_snr)

        # snr_threshold_secondary <= minimum secondary snr in expanded bounds
        min_snr2 = min(EXPANDED_CRITERIA_BOUNDS["snr_threshold_secondary"])
        self.assertLessEqual(LOOSE_BOUNDING_CRITERIA["snr_threshold_secondary"], min_snr2)

        # min_detections == 1
        self.assertEqual(LOOSE_BOUNDING_CRITERIA["min_detections"], 1)
        self.assertEqual(LOOSE_BOUNDING_CRITERIA["min_detections_primary"], 1)

        # max_timespan_days >= maximum timespan in expanded bounds
        max_span = max(EXPANDED_CRITERIA_BOUNDS["max_timespan_days"])
        self.assertGreaterEqual(LOOSE_BOUNDING_CRITERIA["max_timespan_days"], max_span)

        # min_time_separation_hours == 0
        self.assertEqual(LOOSE_BOUNDING_CRITERIA["min_time_separation_hours"], 0.0)

        # min_galactic_lat == 0
        self.assertEqual(LOOSE_BOUNDING_CRITERIA["min_galactic_lat"], 0.0)

        # require_fast_transient is False
        self.assertFalse(LOOSE_BOUNDING_CRITERIA["require_fast_transient"])

    def test_fast_filter_adherence_evaluation(self):
        """Verify evaluate_candidate_against_criteria accurately enforces all cuts."""
        import numpy as np
        from fast_criteria_filter import evaluate_candidate_against_criteria

        # Mock lightcurve with 3 detections:
        # Day 0.0: mag 20.0, SNR 6.5 (Primary)
        # Day 0.2: mag 20.1, SNR 4.0 (Secondary)
        # Day 1.0: mag 21.0, SNR 3.5 (Secondary, fade rate = (21-20.1)/0.8 = 1.125 mag/day)
        # Galactic lat = 18.0 deg
        lc = {
            "times": np.array([100.0, 100.2, 101.0]),
            "snrs": np.array([6.5, 4.0, 3.5]),
            "mags": np.array([20.0, 20.1, 21.0]),
            "abs_gal_lat": 18.0,
            "n_pts": 3,
        }

        # Passes standard fiducial criteria
        cfg = {
            "snr_threshold": 5.0,
            "snr_threshold_secondary": 3.0,
            "min_detections": 2,
            "min_detections_primary": 1,
            "max_timespan_days": 14.0,
            "min_time_separation_hours": 3.0,
            "min_rise_rate": 0.0,
            "min_fade_rate": 0.3,
            "min_galactic_lat": 15.0,
            "require_fast_transient": True,
        }
        self.assertTrue(evaluate_candidate_against_criteria(lc, cfg))

        # Fails primary SNR if threshold is 7.0
        cfg_high_snr = dict(cfg, snr_threshold=7.0)
        self.assertFalse(evaluate_candidate_against_criteria(lc, cfg_high_snr))

        # Fails min_detections if threshold is 4
        cfg_high_ndet = dict(cfg, min_detections=4)
        self.assertFalse(evaluate_candidate_against_criteria(lc, cfg_high_ndet))

        # Fails timespan if max_timespan_days is 0.5 days (actual span is 1.0 day)
        cfg_short_span = dict(cfg, max_timespan_days=0.5)
        # Detections [0, 1] have span 0.2 days (4.8h) and satisfy min_detections=2, min_prim=1!
        # Rolling window finds slice [0:2]!
        self.assertTrue(evaluate_candidate_against_criteria(lc, cfg_short_span))

        # Fails min_time_separation if min_separation is 30 hours (max span is 24h)
        cfg_long_sep = dict(cfg, min_time_separation_hours=30.0)
        self.assertFalse(evaluate_candidate_against_criteria(lc, cfg_long_sep))

        # Fails galactic latitude if cut is 20 deg (source is at 18 deg)
        cfg_high_b = dict(cfg, min_galactic_lat=20.0)
        self.assertFalse(evaluate_candidate_against_criteria(lc, cfg_high_b))

        # Fails fade rate if required fade rate is 2.0 mag/day (max fade is ~1.125 mag/day)
        cfg_high_fade = dict(cfg, min_fade_rate=2.0)
        self.assertFalse(evaluate_candidate_against_criteria(lc, cfg_high_fade))

    def test_multi_model_candidate_pool_and_filter(self):
        """Verify end-to-end CandidatePool loading and parallel filtering across all 3 models."""
        import tempfile
        import json
        import numpy as np
        from fast_criteria_filter import CandidatePool, run_filter_for_pool

        # Create mock candidate JSON
        mock_source = {
            "z": 0.05,
            "peak_abs_mag": -16.0,
            "t_exp": 60000.0,
            "transient_type": "Kilonova",
            "photometry_times": [60001.0, 60001.2, 60002.0],
            "photometry_mags": [19.0, 19.1, 20.0],
            "photometry_errs": [0.1, 0.15, 0.2],
            "photometry_snrs": [10.0, 7.0, 5.0],
            "photometry_bands": ["g", "r", "r"],
            "galactic_lat": 25.0,
        }
        mock_payload = {
            "type": "SimulationResult",
            "n_simulated": 100000,
            "n_detected": 1,
            "detected_sources": [mock_source],
        }

        test_grid = [
            {
                "snr_threshold": 5.0, "snr_threshold_secondary": 3.0,
                "min_detections": 2, "min_detections_primary": 1,
                "max_timespan_days": 14.0, "min_time_separation_hours": 3.0,
                "min_rise_rate": 0.0, "min_fade_rate": 0.3,
                "min_galactic_lat": 15.0, "require_fast_transient": True,
            },
            {
                "snr_threshold": 6.0, "snr_threshold_secondary": 4.0,
                "min_detections": 4, "min_detections_primary": 2,
                "max_timespan_days": 7.0, "min_time_separation_hours": 12.0,
                "min_rise_rate": 0.0, "min_fade_rate": 0.3,
                "min_galactic_lat": 15.0, "require_fast_transient": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for model in ["Bu2026Fixed", "Bu2026Vary", "Metzger"]:
                cand_file = Path(tmpdir) / f"candidates_{model}_fiducial.json"
                cand_file.write_text(json.dumps(mock_payload))

                pool = CandidatePool(cand_file, model)
                self.assertEqual(pool.n_candidates, 1)
                self.assertEqual(pool.n_simulated, 100000)

                df = run_filter_for_pool(pool, test_grid, n_workers=1)
                self.assertEqual(len(df), 2)
                self.assertEqual(df["model"].iloc[0], model)
                # First point should pass (n_det = 1)
                self.assertEqual(df["n_detected"].iloc[0], 1)
                self.assertAlmostEqual(df["overall_efficiency"].iloc[0], 1e-5)
                # Second point requires 4 detections, mock only has 3 -> fails (n_det = 0)
                self.assertEqual(df["n_detected"].iloc[1], 0)

    def test_candidate_file_discovery_layouts(self):
        """Verify find_candidate_files discovers candidate files across all supported directory layouts."""
        import tempfile
        from fast_criteria_filter import find_candidate_files

        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir) / "02_granular_criteria_sweep"
            cand_dir = suite_dir / "candidates"
            model_dir = cand_dir / "Bu2026Fixed"
            model_dir.mkdir(parents=True, exist_ok=True)

            f1 = model_dir / "candidates_Bu2026Fixed_fiducial.json"
            f1_meta = model_dir / "candidates_Bu2026Fixed_fiducial.json.metadata.json"
            f1.write_text("{}")
            f1_meta.write_text("{}")

            # Layout 1: direct model sub-directory inside candidates
            found1 = find_candidate_files(cand_dir, "Bu2026Fixed")
            self.assertEqual(len(found1), 1)
            self.assertEqual(found1[0], f1)

            # Layout 2: passing suite root directory (candidates/Bu2026Fixed/)
            found2 = find_candidate_files(suite_dir, "Bu2026Fixed")
            self.assertEqual(len(found2), 1)
            self.assertEqual(found2[0], f1)

            # Layout 3: flat candidate directory
            flat_dir = Path(tmpdir) / "flat_candidates"
            flat_dir.mkdir(parents=True, exist_ok=True)
            f_flat = flat_dir / "candidates_Metzger_sample_0.json"
            f_flat.write_text("{}")
            found3 = find_candidate_files(flat_dir, "Metzger")
            self.assertEqual(len(found3), 1)
            self.assertEqual(found3[0], f_flat)

            # Should return empty list if model does not exist
            found_none = find_candidate_files(cand_dir, "NonExistentModel")
            self.assertEqual(found_none, [])


if __name__ == "__main__":
    unittest.main()
