"""
Unit test verifying candidate-token log-probability extraction in target interface and sandbox.
"""

import unittest
from target.gemma import GemmaTargetInterface
from interventions.sandbox import InvestigationSandbox, InvestigationBudget


class TestInstrumentation(unittest.TestCase):

    def setUp(self):
        self.mock_target = GemmaTargetInterface(mock=True)
        self.sandbox = InvestigationSandbox(
            target_model=self.mock_target,
            budget=InvestigationBudget(max_experiments=5, max_target_calls=15, max_interventions=8),
            case_id="case_test_instr"
        )

    def test_target_candidate_logprobs_mock(self):
        cands = ["Rome", "Paris"]
        text, logprobs = self.mock_target.run_inference("What is the capital of Italy?", candidate_tokens=cands)
        self.assertIsNotNone(logprobs)
        self.assertIn("Rome", logprobs)
        self.assertIn("Paris", logprobs)
        self.assertIsInstance(logprobs["Rome"], float)
        self.assertIsInstance(logprobs["Paris"], float)

    def test_sandbox_patch_candidate_logprobs(self):
        cands = ["Rome", "Paris"]
        res = self.sandbox.patch_activation(
            source_prompt="The capital of France is",
            target_prompt="The capital of Italy is",
            layer_idx=12,
            source_pos=3,
            target_pos=3,
            hypothesis_id=1,
            candidate_tokens=cands
        )
        self.assertIsNotNone(res.candidate_logprobs)
        self.assertIn("Rome", res.candidate_logprobs)
        self.assertIn("Paris", res.candidate_logprobs)

    def test_sandbox_ablate_candidate_logprobs(self):
        cands = ["Rome", "Paris"]
        res = self.sandbox.ablate_activation(
            prompt="The capital of Italy is",
            layer_idx=12,
            position_idx=3,
            hypothesis_id=2,
            candidate_tokens=cands
        )
        self.assertIsNotNone(res.candidate_logprobs)
        self.assertIn("Rome", res.candidate_logprobs)
        self.assertIn("Paris", res.candidate_logprobs)


if __name__ == "__main__":
    unittest.main()
