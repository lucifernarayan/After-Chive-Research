"""
Pydantic schemas for symmetric blind predictions and scoring evaluation.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class BlindPrediction(BaseModel):
    """
    Symmetric prediction record produced by an investigator at freeze time.
    MUST be generated before hidden variant outcome is revealed or executed.
    """
    prediction_id: str = Field(..., description="Unique prediction identifier")
    case_id: str = Field(..., description="Target failure case ID")
    investigator_type: Literal["transcript_only", "causal_intervention"] = Field(
        ..., description="Type of investigator ('transcript_only' for Agent #1, 'causal_intervention' for Agent #2)"
    )
    predicted_behavior: str = Field(..., description="Description of expected target model behavior on hidden variant")
    predicted_label: str = Field(..., description="Explicit predicted categorical label/token (e.g. 'Rome', 'Paris', 'Positive')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Predictive confidence score between 0.0 and 1.0")
    rationale: str = Field(..., description="Explanation based ONLY on evidence available at freeze time")
    frozen_at: str = Field(..., description="ISO 8601 timestamp when prediction was permanently frozen")


class PredictionScoringResult(BaseModel):
    """
    Deterministic scoring evaluation comparing Agent #1 and Agent #2 predictions
    against the actual hidden variant outcome.
    """
    case_id: str
    agent1_prediction: BlindPrediction
    agent2_prediction: BlindPrediction
    actual_hidden_label: str = Field(..., description="Ground truth outcome label from hidden variant execution")
    actual_hidden_behavior: str = Field(..., description="Ground truth full behavior text")
    
    agent1_correct: bool = Field(..., description="True if Agent #1 predicted label matches actual hidden label")
    agent2_correct: bool = Field(..., description="True if Agent #2 predicted label matches actual hidden label")
    
    agent1_confidence: float = Field(..., description="Agent #1 predictive confidence")
    agent2_confidence: float = Field(..., description="Agent #2 predictive confidence")
    
    agent1_score: float = Field(..., description="Numeric score for Agent #1 (1.0 for correct, 0.0 for incorrect)")
    agent2_score: float = Field(..., description="Numeric score for Agent #2 (1.0 for correct, 0.0 for incorrect)")
    
    timestamp: str = Field(..., description="ISO 8601 scoring timestamp")
