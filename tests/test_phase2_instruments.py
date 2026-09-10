"""
Unit test suite verifying Phase 2A Real Gemma Instrument Validation engine and CLI script.
Tests execution end-to-end in mock mode without needing GPU/CUDA or OpenRouter API calls.
"""

import unittest
import os
import json
import shutil
import tempfile

from target.gemma import GemmaTargetInterface
from experiments.phase2_instrument_validation import Phase2InstrumentValidator


class TestPhase2Instruments(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="phase2_test_")
        self.mock_target = GemmaTargetInterface(mock=True)
        self.validator = Phase2InstrumentValidator(
            target_model=self.mock_target,
            mock=True,
            output_dir=self.temp_dir
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_validator_executes_all_11_tests_in_mock(self):
        """Verify that Phase2InstrumentValidator executes all 11 instrument validation tests end-to-end."""
        results = self.validator.execute_validation()

        self.assertIsNotNone(results)
        self.assertIn("metadata", results)
        self.assertIn("test_results", results)
        self.assertIn("summary_table", results)

        test_keys = [
            "test_1_residual_patching",
            "test_2_noop_stability",
            "test_3_dose_response",
            "test_4_random_control",
            "test_5_bidirectional",
            "test_6_attribution",
            "test_7_attention_head",
            "test_8_mlp",
            "test_9_component_comparison",
            "test_10_replication",
            "test_11_contrastive_control"
        ]

        for k in test_keys:
            self.assertIn(k, results["test_results"], f"Missing test result key '{k}'")
            self.assertEqual(results["test_results"][k]["status"], "PASS", f"Test '{k}' failed")

    def test_output_files_generated(self):
        """Verify that phase2_instrument_validation.json, summary.json, and PHASE2_REPORT.md are generated."""
        self.validator.execute_validation()

        json_path = os.path.join(self.temp_dir, "phase2_instrument_validation.json")
        sum_path = os.path.join(self.temp_dir, "phase2_instrument_validation_summary.json")
        report_path = os.path.join(self.temp_dir, "PHASE2_REPORT.md")

        self.assertTrue(os.path.exists(json_path), f"File {json_path} missing")
        self.assertTrue(os.path.exists(sum_path), f"File {sum_path} missing")
        self.assertTrue(os.path.exists(report_path), f"File {report_path} missing")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("test_results", data)

        with open(sum_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
            self.assertEqual(sum_data["overall_status"], "PASS")

        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
            self.assertIn("PHASE 2A — REAL GEMMA INSTRUMENT VALIDATION REPORT", report_text)
            self.assertIn("Instrument Status Summary Table", report_text)

    def test_noop_stability_assertion(self):
        """Verify residual no-op stability is verified as True."""
        results = self.validator.execute_validation()
        noop_res = results["test_results"]["test_2_noop_stability"]
        self.assertTrue(noop_res["no_op_stable"])


if __name__ == "__main__":
    unittest.main()
