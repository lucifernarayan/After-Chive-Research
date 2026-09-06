"""
Unit tests for Phase 4A safety assertions, hard mock guards, file hygiene, and data separation.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from scripts.phase4a_evaluation import run_phase4a_evaluation, _assert_real_target_output
from schemas.hypotheses import FailureCase


class TestPhase4ASafety(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cases_file = os.path.join(self.temp_dir.name, "test_cases.json")
        self.results_file = os.path.join(self.temp_dir.name, "test_results.json")
        self.summary_file = os.path.join(self.temp_dir.name, "test_summary.json")

        self.test_cases = [
            {
                "case_id": "test_001",
                "failure_family": "negation_instruction",
                "task_description": "Test Case 1",
                "original_prompt": "Prompt 1",
                "expected_original_behavior": "Behavior 1",
                "hidden_variant": "Hidden Prompt 1",
                "expected_hidden_behavior": "Behavior 1",
                "candidate_labels": ["Label1", "Label2"],
                "failure_description": "Failure 1"
            }
        ]

        with open(self.cases_file, "w", encoding="utf-8") as f:
            json.dump(self.test_cases, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("torch.cuda.is_available", return_value=False)
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key", "HF_TOKEN": "test_token"})
    def test_real_mode_assertions_cuda_missing(self, mock_cuda):
        """Test that running real evaluation without CUDA raises RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            run_phase4a_evaluation(
                mock_gemma=False,
                mock_gemini=False,
                cases_path=self.cases_file,
                results_path=self.results_file,
                summary_path=self.summary_file
            )
        self.assertIn("CUDA is unavailable", str(ctx.exception))

    @patch("torch.cuda.is_available", return_value=True)
    @patch.dict(os.environ, {"HF_TOKEN": "test_token"}, clear=True)
    def test_real_mode_assertions_api_key_missing(self, mock_cuda):
        """Test that running real evaluation without OPENROUTER_API_KEY raises RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            run_phase4a_evaluation(
                mock_gemma=False,
                mock_gemini=False,
                cases_path=self.cases_file,
                results_path=self.results_file,
                summary_path=self.summary_file
            )
        self.assertIn("OPENROUTER_API_KEY environment variable is missing", str(ctx.exception))

    @patch("torch.cuda.is_available", return_value=True)
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}, clear=True)
    def test_real_mode_assertions_hf_token_missing(self, mock_cuda):
        """Test that running real evaluation without HF_TOKEN raises RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            run_phase4a_evaluation(
                mock_gemma=False,
                mock_gemini=False,
                cases_path=self.cases_file,
                results_path=self.results_file,
                summary_path=self.summary_file
            )
        self.assertIn("HF_TOKEN environment variable is missing", str(ctx.exception))

    def test_mock_output_string_guard(self):
        """Test that mock output strings or synthetic logprobs trigger RuntimeError."""
        # 1. Mock Gemma response string
        with self.assertRaises(RuntimeError) as ctx:
            _assert_real_target_output("Prompt [Mock Gemma Response]", {"Label1": -1.0}, "c1", "test")
        self.assertIn("Mock response string detected", str(ctx.exception))

        # 2. Ablated response string
        with self.assertRaises(RuntimeError) as ctx:
            _assert_real_target_output("[Ablated Response] text", {"Label1": -1.0}, "c1", "test")
        self.assertIn("Mock response string detected", str(ctx.exception))

        # 3. Synthetic mock logprobs (-0.15, -3.65)
        with self.assertRaises(RuntimeError) as ctx:
            _assert_real_target_output("Valid text", {"Rome": -0.15, "Paris": -3.65}, "c1", "test")
        self.assertIn("Mock candidate logprobs detected", str(ctx.exception))

        # 4. Valid output should pass without raising
        _assert_real_target_output("Valid output text", {"Rome": -0.05, "Paris": -4.2}, "c1", "test")

    def test_mock_result_resume_refusal(self):
        """Test that real evaluation refuses to resume mock-contaminated result files."""
        mock_results = {
            "aggregate_metrics": {"execution_mode": "mock", "completed_cases": 1, "total_cases": 1},
            "cases": [
                {
                    "case_id": "test_001",
                    "actual_target_output": "[Mock Gemma Response]",
                    "input_case": {"model_response": "[Mock Gemma Response]"}
                }
            ]
        }
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(mock_results, f, indent=2)

        # Running real mode (will fail at CUDA check, but resume check happens before)
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key", "HF_TOKEN": "test_token"}):
                with self.assertRaises(RuntimeError) as ctx:
                    run_phase4a_evaluation(
                        mock_gemma=False,
                        mock_gemini=False,
                        cases_path=self.cases_file,
                        results_path=self.results_file,
                        summary_path=self.summary_file
                    )

    def test_explicit_mock_mode_allowed(self):
        """Test that explicit mock mode runs cleanly without raising real-mode safety errors."""
        res = run_phase4a_evaluation(
            mock_gemma=True,
            mock_gemini=True,
            cases_path=self.cases_file,
            results_path=self.results_file,
            summary_path=self.summary_file
        )
        self.assertEqual(res["execution_mode"], "mock")
        self.assertEqual(res["completed_cases"], 1)

    def test_data_separation(self):
        """Test that FailureCase schema contains only public fields and excludes hidden test data."""
        fields = FailureCase.model_fields.keys()
        self.assertIn("case_id", fields)
        self.assertIn("task_description", fields)
        self.assertIn("prompt", fields)
        self.assertIn("model_response", fields)
        self.assertIn("failure_description", fields)
        self.assertIn("expected_behavior", fields)

        # Verify hidden fields are completely absent
        self.assertNotIn("hidden_variant", fields)
        self.assertNotIn("expected_hidden_behavior", fields)
        self.assertNotIn("hidden_output", fields)
        self.assertNotIn("hidden_candidate_logprobs", fields)

    def test_case_limit_parameter(self):
        """Test that passing limit parameter caps the number of processed cases."""
        # Create a dataset with 3 test cases
        three_cases = [
            {
                "case_id": f"test_case_{i}",
                "failure_family": "negation_instruction",
                "task_description": f"Test Case {i}",
                "original_prompt": f"Prompt {i}",
                "expected_original_behavior": f"Behavior {i}",
                "hidden_variant": f"Hidden Prompt {i}",
                "expected_hidden_behavior": f"Behavior {i}",
                "candidate_labels": ["Label1", "Label2"],
                "failure_description": f"Failure {i}"
            }
            for i in range(1, 4)
        ]
        cases_3_file = os.path.join(self.temp_dir.name, "test_cases_3.json")
        with open(cases_3_file, "w", encoding="utf-8") as f:
            json.dump(three_cases, f, indent=2)

        res = run_phase4a_evaluation(
            mock_gemma=True,
            mock_gemini=True,
            cases_path=cases_3_file,
            results_path=self.results_file,
            summary_path=self.summary_file,
            limit=2
        )
        self.assertEqual(res["total_cases"], 2)
        self.assertEqual(res["completed_cases"], 2)

    def test_custom_output_paths_isolation(self):
        """Test that passing custom results_path starts with zero completed cases even if default results file exists."""
        # 1. Write an existing results file with completed case 'test_001'
        default_res_path = os.path.join(self.temp_dir.name, "default_results.json")
        default_sum_path = os.path.join(self.temp_dir.name, "default_summary.json")
        
        run_phase4a_evaluation(
            mock_gemma=True,
            mock_gemini=True,
            cases_path=self.cases_file,
            results_path=default_res_path,
            summary_path=default_sum_path
        )
        self.assertTrue(os.path.exists(default_res_path))

        # 2. Run evaluation pointing to NEW custom output paths
        custom_res_path = os.path.join(self.temp_dir.name, "validation_2case_results.json")
        custom_sum_path = os.path.join(self.temp_dir.name, "validation_2case_summary.json")

        res = run_phase4a_evaluation(
            mock_gemma=True,
            mock_gemini=True,
            cases_path=self.cases_file,
            results_path=custom_res_path,
            summary_path=custom_sum_path
        )

        # 3. Verify custom results file was created independently and processed case test_001 afresh
        self.assertTrue(os.path.exists(custom_res_path))
        self.assertTrue(os.path.exists(custom_sum_path))
        self.assertEqual(res["completed_cases"], 1)

        with open(custom_res_path, "r", encoding="utf-8") as f:
            custom_data = json.load(f)
        self.assertEqual(custom_data["cases"][0]["case_id"], "test_001")


if __name__ == "__main__":
    unittest.main()
