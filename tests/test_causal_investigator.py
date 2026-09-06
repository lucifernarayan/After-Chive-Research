"""
Unit tests for Agent #2 (CausalInvestigatorAgent) dynamic experiment proposal,
Pydantic validation, evidence update logic, and security firewall.
"""

import unittest
from unittest.mock import MagicMock, patch

from schemas.hypotheses import FailureCase, HypothesisSet, Hypothesis
from schemas.investigation import (
    ExperimentRequest, 
    ExperimentProposalSet, 
    ExperimentResult, 
    HypothesisEvidence,
    InvestigationBudget
)
from interventions.sandbox import InvestigationSandbox
from target.gemma import GemmaTargetInterface
from agents.causal_investigator import CausalInvestigatorAgent


class TestCausalInvestigator(unittest.TestCase):

    def setUp(self):
        self.sample_case = FailureCase(
            case_id="case_001",
            task_description="Negation test",
            prompt="Write a poem about trees. Do NOT mention leaves.",
            model_response="The trees have beautiful green leaves everywhere.",
            failure_description="Target model mentioned leaves despite negative constraint.",
            expected_behavior="Target model describes trees without mentioning leaves."
        )

        h1 = Hypothesis(
            claim="Residual activation of 'leaves' token overrides negative instruction.",
            mechanism_guess="Residual stream vectors for prompt noun 'leaves' override instruction vector at Layer 12.",
            confidence="high"
        )
        h2 = Hypothesis(
            claim="Instruction token routing fails in middle self-attention layers.",
            mechanism_guess="Attention routing from 'Do NOT' token is weak at Layer 8.",
            confidence="medium"
        )
        self.hypotheses = HypothesisSet(hypotheses=[h1, h2], most_likely=1)

    def test_mock_mode_uses_mock_proposal_generator(self):
        """Verify mock mode calls _generate_mock_experiment_requests and does not call OpenRouter."""
        agent = CausalInvestigatorAgent(mock=True)
        with patch.object(agent, '_generate_mock_experiment_requests', wraps=agent._generate_mock_experiment_requests) as mock_gen:
            with patch.object(agent, '_propose_experiments_real') as mock_real:
                proposals = agent.propose_experiments(self.sample_case, self.hypotheses)
                mock_gen.assert_called_once()
                mock_real.assert_not_called()
                self.assertEqual(len(proposals), 3)

    def test_real_mode_calls_real_proposal_generator(self):
        """Verify real mode calls _propose_experiments_real and does NOT call mock proposal generator."""
        agent = CausalInvestigatorAgent(mock=False)
        # Mock client initialization so mock=False is retained
        agent.client = MagicMock()
        agent.mock = False

        expected_reqs = [
            ExperimentRequest(
                experiment_id="exp_001",
                hypothesis_id=1,
                experiment_type="patch",
                prompt=self.sample_case.prompt,
                source_prompt="Trees grow high in the sky.",
                layer_idx=10,
                position_idx=2,
                rationale="Test patching"
            )
        ]

        with patch.object(agent, '_propose_experiments_real', return_value=expected_reqs) as mock_real:
            with patch.object(agent, '_generate_mock_experiment_requests') as mock_gen:
                proposals = agent.propose_experiments(self.sample_case, self.hypotheses)
                mock_real.assert_called_once()
                mock_gen.assert_not_called()
                self.assertEqual(proposals, expected_reqs)

    def test_real_mode_validates_structured_proposals(self):
        """Verify real mode parses structured OpenRouter completions into ExperimentProposalSet."""
        agent = CausalInvestigatorAgent(mock=False)
        agent.client = MagicMock()
        agent.mock = False

        proposal_data = ExperimentProposalSet(
            proposals=[
                ExperimentRequest(
                    experiment_id="exp_real_01",
                    hypothesis_id=1,
                    experiment_type="ablate",
                    prompt=self.sample_case.prompt,
                    layer_idx=14,
                    position_idx=4,
                    rationale="Dynamic ablation proposal"
                )
            ]
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=proposal_data))]
        agent.client.beta.chat.completions.parse.return_value = mock_completion

        requests = agent._propose_experiments_real(self.sample_case, self.hypotheses)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].experiment_type, "ablate")
        self.assertEqual(requests[0].layer_idx, 14)

    def test_malformed_proposal_rejection(self):
        """Verify malformed or invalid proposals (invalid tool type, bad hypothesis_id) raise RuntimeError/ValueError."""
        agent = CausalInvestigatorAgent(mock=False)
        agent.client = MagicMock()
        agent.mock = False

        # 1. Invalid tool type
        bad_tool = ExperimentProposalSet(
            proposals=[
                ExperimentRequest(
                    experiment_id="exp_bad",
                    hypothesis_id=1,
                    experiment_type="run_target",
                    prompt=self.sample_case.prompt,
                    rationale="Valid base"
                )
            ]
        )
        bad_tool.proposals[0].experiment_type = "invalid_python_tool"  # Force invalid tool

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=bad_tool))]
        agent.client.beta.chat.completions.parse.return_value = mock_completion

        with self.assertRaises((RuntimeError, ValueError)):
            agent._propose_experiments_real(self.sample_case, self.hypotheses)

        # 2. Invalid hypothesis ID out of bounds
        bad_hyp = ExperimentProposalSet(
            proposals=[
                ExperimentRequest(
                    experiment_id="exp_bad_hyp",
                    hypothesis_id=999,  # Out of range
                    experiment_type="patch",
                    prompt=self.sample_case.prompt,
                    rationale="Bad hyp ID"
                )
            ]
        )
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=bad_hyp))]
        agent.client.beta.chat.completions.parse.return_value = mock_completion

        with self.assertRaises((RuntimeError, ValueError)):
            agent._propose_experiments_real(self.sample_case, self.hypotheses)

    def test_evidence_update_zero_delta_unresolved(self):
        """Verify zero behavioral delta is treated as unresolved/non-informative, not automatic contradiction."""
        agent = CausalInvestigatorAgent(mock=True)
        ev = HypothesisEvidence(
            hypothesis_id=1,
            claim="Test claim",
            mechanism_guess="Test mechanism",
            prior_confidence="high",
            supporting_experiments=[],
            contradicting_experiments=[],
            evidence_type="intervention",
            evidence_strength="weak",
            updated_confidence="high",
            status="unresolved",
            rationale="Initial state"
        )

        res = ExperimentResult(
            investigation_id="case_001",
            experiment_id="exp_001",
            hypothesis_id=1,
            experiment_type="patch",
            prompt="Prompt",
            baseline_output="Out A",
            intervened_output="Out A",
            observed_behavioral_delta=0.0,
            timestamp="2026-09-07T00:00:00Z"
        )

        agent._update_evidence(ev, res)

        self.assertEqual(ev.status, "unresolved")
        self.assertEqual(ev.evidence_strength, "weak")
        self.assertNotIn("exp_001", ev.contradicting_experiments)
        self.assertIn("non-informative", ev.rationale.lower())

    def test_blind_prediction_prompt_removes_expected_behavior_shortcut(self):
        """Verify user content for blind prediction does NOT leak expected_behavior string."""
        agent = CausalInvestigatorAgent(mock=False)
        agent.client = MagicMock()
        agent.mock = False

        sandbox = InvestigationSandbox(
            target_model=GemmaTargetInterface(mock=True),
            budget=InvestigationBudget(max_experiments=2),
            case_id=self.sample_case.case_id
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(parsed=MagicMock(
            prediction_id="pred_1",
            case_id="case_001",
            investigator_type="causal_intervention",
            predicted_behavior="Adheres",
            predicted_label="No leaves",
            confidence=0.8,
            rationale="Evidence based",
            frozen_at="now"
        )))]
        agent.client.beta.chat.completions.parse.return_value = mock_completion

        agent.generate_blind_prediction(self.sample_case, self.hypotheses, sandbox)

        # Inspect messages passed to OpenRouter completion
        call_args = agent.client.beta.chat.completions.parse.call_args
        messages = call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

        self.assertNotIn(f"Expected Behavior: {self.sample_case.expected_behavior}", user_msg)
        self.assertNotIn("Expected Behavior:", user_msg)

    def test_budget_limits_enforced(self):
        """Verify that investigation halts when budget limits are exhausted."""
        agent = CausalInvestigatorAgent(mock=True)
        sandbox = InvestigationSandbox(
            target_model=GemmaTargetInterface(mock=True),
            budget=InvestigationBudget(max_experiments=1, max_target_calls=1, max_interventions=1),
            case_id=self.sample_case.case_id
        )

        record = agent.investigate(self.sample_case, self.hypotheses, sandbox)
        self.assertLessEqual(len(record.experiments), 1)


if __name__ == "__main__":
    unittest.main()
