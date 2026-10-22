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
        presets = ["canonical", "broad", "fine", "ultra_fine", "all_slices"]
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
        """Verify the ultra-fine grid (158,460 points) satisfies <= 5,000 tasks with chunk_size=35."""
        grid = get_grid_combinations("ultra_fine")
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
        """Verify 02_run_granular_criteria_sweep.sh has correct array and throttle directives."""
        script_path = Path(__file__).parent / "scripts" / "02_run_granular_criteria_sweep.sh"
        self.assertTrue(script_path.exists())
        content = script_path.read_text()

        # Check array directive (supports unthrottled 0-N or throttled 0-N%M)
        match = re.search(r"#SBATCH\s+--array=0-(\d+)(?:%(\d+))?", content)
        self.assertIsNotNone(match, "Could not find #SBATCH --array directive in script!")
        max_idx = int(match.group(1))

        self.assertLessEqual(max_idx + 1, 5000, "Slurm array tasks must be <= 5,000 for MSI compliance!")

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


if __name__ == "__main__":
    unittest.main()
