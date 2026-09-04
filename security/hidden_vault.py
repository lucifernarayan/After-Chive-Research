"""
Security Blind Firewall: Hidden Test Vault & Architectural Isolation Engine.

Hard Isolation Rules:
- Agent #1 and Agent #2 NEVER receive access to HiddenTestVault data during investigation or prediction.
- Neither prediction function accepts hidden_variant objects as parameters.
- Querying hidden variant data prior to explicit prediction freeze raises a hard PermissionError.
"""

import datetime
from typing import Dict, Optional, Any
from schemas.predictions import BlindPrediction, PredictionScoringResult


class PermissionError(Exception):
    """Raised when an unauthorized agent or un-frozen workflow attempts to read hidden test data."""
    pass


class HiddenVariantFixture:
    """Encapsulates hidden variant test data outside investigator accessible environment."""
    def __init__(self, case_id: str, hidden_prompt: str, expected_label: str, expected_behavior: str):
        self.case_id = case_id
        self.hidden_prompt = hidden_prompt
        self.expected_label = expected_label
        self.expected_behavior = expected_behavior


class HiddenTestVault:
    """
    Architectural Blind Firewall holding hidden test variant fixtures.
    Enforces freeze lock prior to revealing or scoring outcomes.
    """

    def __init__(self):
        self._fixtures: Dict[str, HiddenVariantFixture] = {}
        self._is_frozen: Dict[str, bool] = {}

    def register_fixture(self, case_id: str, hidden_prompt: str, expected_label: str, expected_behavior: str):
        """Register a synthetic hidden test fixture for a case."""
        self._fixtures[case_id] = HiddenVariantFixture(case_id, hidden_prompt, expected_label, expected_behavior)
        self._is_frozen[case_id] = False

    def freeze_investigation(self, case_id: str):
        """Lock and freeze investigation state for a case."""
        if case_id in self._fixtures:
            self._is_frozen[case_id] = True

    def reveal_and_evaluate(
        self, 
        case_id: str, 
        agent1_pred: BlindPrediction, 
        agent2_pred: BlindPrediction
    ) -> PredictionScoringResult:
        """
        Reveal hidden variant outcome ONLY after predictions are frozen.
        Deterministically evaluates Agent #1 vs Agent #2 predictions against actual outcome.
        """
        if not self._is_frozen.get(case_id, False):
            raise PermissionError(
                f"SECURITY VIOLATION: Attempted to reveal hidden variant for '{case_id}' before investigation was frozen!"
            )

        fixture = self._fixtures.get(case_id)
        if not fixture:
            raise KeyError(f"No hidden test fixture registered for case '{case_id}'.")

        # Deterministic Label Matching
        actual_label = fixture.expected_label.strip().lower()
        
        a1_pred_label = agent1_pred.predicted_label.strip().lower()
        a2_pred_label = agent2_pred.predicted_label.strip().lower()

        a1_correct = (a1_pred_label == actual_label) or (actual_label in a1_pred_label)
        a2_correct = (a2_pred_label == actual_label) or (actual_label in a2_pred_label)

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return PredictionScoringResult(
            case_id=case_id,
            agent1_prediction=agent1_pred,
            agent2_prediction=agent2_pred,
            actual_hidden_label=fixture.expected_label,
            actual_hidden_behavior=fixture.expected_behavior,
            agent1_correct=a1_correct,
            agent2_correct=a2_correct,
            agent1_confidence=round(agent1_pred.confidence, 4),
            agent2_confidence=round(agent2_pred.confidence, 4),
            agent1_score=1.0 if a1_correct else 0.0,
            agent2_score=1.0 if a2_correct else 0.0,
            timestamp=timestamp
        )
