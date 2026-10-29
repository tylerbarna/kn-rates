"""Unit tests for rate limit aggregation and Andreoni (2020) comparison logic."""

import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Provide stubs for pandas/numpy if running in a bare test environment
_HAS_PANDAS = True
try:
    import pandas as pd
    import numpy as np
except ImportError:
    _HAS_PANDAS = False
    sys.modules["pandas"] = MagicMock()
    sys.modules["numpy"] = MagicMock()

agg_module = importlib.import_module("analysis_suites.01_large_scale_rate_limits.aggregate_rate_limits")
comp_module = importlib.import_module("analysis_suites.04_historical_andreoni_comparison.compare_with_andreoni2020")
import rate_limit_benchmark


class TestAggregationAndComparison(unittest.TestCase):
    """Test rate limit column harmonization and aggregation functionality."""

    def test_extract_model_from_path(self):
        self.assertEqual(agg_module.extract_model_from_path("results/01_large_scale_rate_limits/Metzger/run_0.json"), "Metzger")
        self.assertEqual(agg_module.extract_model_from_path("results/01_large_scale_rate_limits/Bu2026Fixed/run_0.json"), "Bu2026Fixed")
        self.assertEqual(agg_module.extract_model_from_path("results/01_large_scale_rate_limits/Bu2026Vary/run_0.json"), "Bu2026Vary")
        self.assertEqual(agg_module.extract_model_from_path("results/ztf_fixed/run_1.json"), "Bu2026Fixed")
        self.assertEqual(agg_module.extract_model_from_path("results/bu2026_vary/run_1.json"), "Bu2026Vary")

    def test_process_file_includes_both_column_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = Path(tmpdir) / "Bu2026Fixed_test.json"
            sample_data = {
                "n_simulated": 1000000,
                "n_detected": 42,
                "rate_summaries": [
                    {
                        "transient_type": "Kilonova",
                        "survey_duration_years": 8.0057,
                        "effective_vt_gpc3_yr": 0.002,
                        "overall_efficiency": 0.00042,
                    }
                ],
            }
            sample_file.write_text(json.dumps(sample_data))

            rows = agg_module.process_file(sample_file)
            self.assertEqual(len(rows), 1)
            row = rows[0]

            expected_r95 = -math.log(0.05) / 0.002
            expected_r90 = -math.log(0.10) / 0.002
            expected_r99 = -math.log(0.01) / 0.002

            # Verify both naming conventions are present and equal
            self.assertIn("rate_95_upper_limit", row)
            self.assertIn("rate_95_zero_events", row)
            self.assertAlmostEqual(row["rate_95_upper_limit"], expected_r95, places=4)
            self.assertAlmostEqual(row["rate_95_zero_events"], expected_r95, places=4)

            self.assertIn("rate_90_upper_limit", row)
            self.assertIn("rate_90_zero_events", row)
            self.assertAlmostEqual(row["rate_90_upper_limit"], expected_r90, places=4)
            self.assertAlmostEqual(row["rate_90_zero_events"], expected_r90, places=4)

            self.assertIn("rate_99_upper_limit", row)
            self.assertIn("rate_99_zero_events", row)
            self.assertAlmostEqual(row["rate_99_upper_limit"], expected_r99, places=4)
            self.assertAlmostEqual(row["rate_99_zero_events"], expected_r99, places=4)

            self.assertIn("duration_scaled_95", row)
            self.assertIn("rate_95_duration_scaled_1_92yr", row)
            self.assertEqual(row["duration_scaled_95"], row["rate_95_duration_scaled_1_92yr"])

    def test_rate_limit_benchmark_summarize_result_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = Path(tmpdir) / "test_benchmark.json"
            sample_data = {
                "n_simulated": 1000000,
                "n_detected": 50,
                "rate_summaries": [
                    {
                        "transient_type": "Kilonova",
                        "survey_duration_years": 8.0,
                        "effective_vt_gpc3_yr": 0.005,
                        "overall_efficiency": 0.0005,
                    }
                ],
            }
            sample_file.write_text(json.dumps(sample_data))

            rows = rate_limit_benchmark.summarize_result(sample_file)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("rate_95_zero_events", row)
            self.assertIn("rate_95_upper_limit", row)
            self.assertIn("rate_90_zero_events", row)
            self.assertIn("rate_90_upper_limit", row)
            self.assertIn("duration_scaled_95", row)
            self.assertIn("rate_95_duration_scaled_1_92yr", row)
            self.assertEqual(row["rate_95_zero_events"], row["rate_95_upper_limit"])

    def test_extract_rate_95_logic(self):
        class MockSeries:
            def __init__(self, val):
                self.val = val
            def mean(self):
                return self.val

        class MockDataFrame(dict):
            @property
            def columns(self):
                return list(self.keys())

        # Test case 1: only rate_95_upper_limit present (the exact KeyError cause)
        df1 = MockDataFrame({"rate_95_upper_limit": MockSeries(1234.5)})
        self.assertAlmostEqual(comp_module._extract_rate_95(df1), 1234.5)

        # Test case 2: only rate_95_zero_events present
        df2 = MockDataFrame({"rate_95_zero_events": MockSeries(567.8)})
        self.assertAlmostEqual(comp_module._extract_rate_95(df2), 567.8)

        # Test case 3: only R_95 present
        df3 = MockDataFrame({"R_95": MockSeries(999.0)})
        self.assertAlmostEqual(comp_module._extract_rate_95(df3), 999.0)

        # Test case 4: only effective_vt_gpc3_yr present
        # In this case _extract_rate_95 calls (-np.log(0.05) / sub["effective_vt_gpc3_yr"]).mean()
        # which uses numpy if available or raises KeyError if missing columns
        df_empty = MockDataFrame({})
        with self.assertRaises(KeyError):
            comp_module._extract_rate_95(df_empty)


if __name__ == "__main__":
    unittest.main()
