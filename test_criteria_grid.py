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
        """Verify the expanded broad grid (42,480 points) satisfies <= 5,000 tasks with chunk_size=10."""
        grid = get_grid_combinations("broad")
        total = len(grid)
        self.assertEqual(total, 42480)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=10, max_tasks=5000)
        self.assertEqual(chunk_size, 10)
        self.assertEqual(n_tasks, 4248)
        self.assertLessEqual(n_tasks, 5000)

    def test_ultra_fine_preset_chunking_compliance(self):
        """Verify the ultra-fine grid (98,268 points) satisfies <= 5,000 tasks with chunk_size=20."""
        grid = get_grid_combinations("ultra_fine")
        total = len(grid)
        self.assertEqual(total, 98268)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=20, max_tasks=5000)
        self.assertEqual(chunk_size, 20)
        self.assertEqual(n_tasks, 4914)
        self.assertLessEqual(n_tasks, 5000)

    def test_fine_preset_chunking_compliance(self):
        """Verify the fine grid (17,280 points) satisfies <= 5,000 tasks with chunk_size=4."""
        grid = get_grid_combinations("fine")
        total = len(grid)
        self.assertEqual(total, 17280)
        chunk_size, n_tasks = calculate_chunking(total, chunk_size=4, max_tasks=5000)
        self.assertEqual(chunk_size, 4)
        self.assertEqual(n_tasks, 4320)
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


if __name__ == "__main__":
    unittest.main()
