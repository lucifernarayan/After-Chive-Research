"""
Validation experiment orchestration for Phase 3A Agent #2 Causal Investigator & Sandbox.
"""

from typing import Dict, Any
from schemas.hypotheses import FailureCase, HypothesisSet, Hypothesis
from target.gemma import GemmaTargetInterface
from interventions.sandbox import InvestigationSandbox, InvestigationBudget
from agents.causal_investigator import CausalInvestigatorAgent
from experiments.hypothesis_experiment import get_validation_cases


def get_synthetic_investigation_setup(mock_gemma: bool = True) -> tuple[FailureCase, HypothesisSet]:
    """Returns synthetic investigation case and corresponding Agent #1 hypotheses for Phase 3A testing."""
    case = get_validation_cases()[1] # case_syn_002 (Prefix Constraint / Capital QA)

    h1 = Hypothesis(
        claim="Residual stream activation vectors for 'France' persist into middle layers and override the prompt instruction.",
        mechanism_guess="Post-attention residual states in layers 10-16 encode strong 'France' feature representations.",
        confidence="high"
    )
    h2 = Hypothesis(
        claim="Self-attention heads fail to route the negation prefix token representation to the final sequence position.",
        mechanism_guess="Attention routing in layers 6-12 suppresses prefix instruction weights.",
        confidence="medium"
    )
    h3 = Hypothesis(
        claim="The model relies on direct logit shortcuts for country-capital word pairs in late MLP layers.",
        mechanism_guess="MLP logit channels directly trigger 'Paris' output tokens.",
        confidence="low"
    )

    hypotheses = HypothesisSet(hypotheses=[h1, h2, h3], most_likely=1)
    return case, hypotheses


class InvestigationExperimentRunner:
    """
    Orchestrates Phase 3A validation suite.
    """

    def __init__(self, agent: CausalInvestigatorAgent, target_model: GemmaTargetInterface):
        self.agent = agent
        self.target = target_model

    def run_phase3a_validation(self) -> Dict[str, Any]:
        case, hypotheses = get_synthetic_investigation_setup(mock_gemma=self.target.mock)
        
        # Configure conservative Phase 3A budget
        budget = InvestigationBudget(max_experiments=5, max_target_calls=15, max_interventions=8)
        sandbox = InvestigationSandbox(target_model=self.target, budget=budget, case_id=case.case_id)

        # Run investigation
        record = self.agent.investigate(case, hypotheses, sandbox)

        return {
            "investigation_id": record.investigation_id,
            "case_id": case.case_id,
            "experiments_executed": len(record.experiments),
            "hypothesis_updates": [h.model_dump() for h in record.hypothesis_evidence],
            "budget_status": record.budget_status.model_dump(),
            "record": record.model_dump()
        }
