"""
Unit test verifying Phase 4A resume support and atomic per-case persistence.
"""

import os
import json
import tempfile
import unittest
from scripts.phase4a_evaluation import run_phase4a_evaluation


class TestPhase4AResume(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cases_file = os.path.join(self.temp_dir.name, "test_cases.json")
        self.results_file = os.path.join(self.temp_dir.name, "test_results.json")
        self.summary_file = os.path.join(self.temp_dir.name, "test_summary.json")

        # Create 3 synthetic test cases
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
            },
            {
                "case_id": "test_002",
                "failure_family": "factual_substitution",
                "task_description": "Test Case 2",
                "original_prompt": "Prompt 2",
                "expected_original_behavior": "Behavior 2",
                "hidden_variant": "Hidden Prompt 2",
                "expected_hidden_behavior": "Behavior 2",
                "candidate_labels": ["Label1", "Label2"],
                "failure_description": "Failure 2"
            },
            {
                "case_id": "test_003",
                "failure_family": "format_constraint",
                "task_description": "Test Case 3",
                "original_prompt": "Prompt 3",
                "expected_original_behavior": "Behavior 3",
                "hidden_variant": "Hidden Prompt 3",
                "expected_hidden_behavior": "Behavior 3",
                "candidate_labels": ["Label1", "Label2"],
                "failure_description": "Failure 3"
            }
        ]

        with open(self.cases_file, "w", encoding="utf-8") as f:
            json.dump(self.test_cases, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_interrupted_run_resumes_cleanly(self):
        # 1. Simulate an interrupted run where ONLY test_001 was completed
        initial_cases = [
            {
                "case_id": "test_001",
                "failure_family": "negation_instruction",
                "agent1_correct": True,
                "agent2_correct": True,
                "brier_score_agent1": 0.05,
                "brier_score_agent2": 0.02,
                "experiment_count": 3,
                "intervention_count": 2
            }
        ]
        initial_results = {
            "aggregate_metrics": {"completed_cases": 1, "total_cases": 3},
            "cases": initial_cases
        }
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(initial_results, f, indent=2)

        # 2. Run phase4a evaluation script targeting these paths
        res = run_phase4a_evaluation(
            mock_gemma=True,
            mock_gemini=True,
            cases_path=self.cases_file,
            results_path=self.results_file,
            summary_path=self.summary_file
        )

        # 3. Assert resume behavior
        self.assertEqual(res["completed_cases"], 3)
        self.assertEqual(res["total_cases"], 3)

        with open(self.results_file, "r", encoding="utf-8") as f:
            final_data = json.load(f)

        final_case_ids = [c["case_id"] for c in final_data["cases"]]
        self.assertEqual(final_case_ids, ["test_001", "test_002", "test_003"])
        
        # Verify initial case was preserved without overwrite
        self.assertEqual(final_data["cases"][0]["brier_score_agent1"], 0.05)


if __name__ == "__main__":
    unittest.main()
