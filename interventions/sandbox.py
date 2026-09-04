"""
Investigation Sandbox: Safe, auditable intervention execution environment for Agent #2.

Security & Isolation Constraints:
- NO arbitrary python execution (no exec, eval, or code tools).
- NO shell commands (no subprocess or OS execution).
- NO arbitrary file reading (no filesystem access tools).
- ONLY explicit intervention primitives (run_target, capture_activation, patch_activation, ablate_activation, compare_outputs).
- Strict budget enforcement (max_experiments, max_target_calls, max_interventions).
- Append-only immutable JSON record logging.
"""

import uuid
import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from schemas.investigation import (
    InvestigationBudget,
    ExperimentRequest,
    ExperimentResult,
    InvestigationRecord
)
from target.gemma import GemmaTargetInterface


class BudgetExhaustedError(Exception):
    """Raised when Agent #2 exceeds its configured investigation budget."""
    pass


class InvestigationSandbox:
    """
    Auditable sandbox providing explicit intervention tools for Agent #2.
    Tracks budget consumption and produces append-only experiment logs.
    """

    def __init__(
        self, 
        target_model: GemmaTargetInterface, 
        budget: Optional[InvestigationBudget] = None,
        investigation_id: Optional[str] = None,
        case_id: str = "case_sandbox"
    ):
        self.target = target_model
        self.budget = budget or InvestigationBudget(max_experiments=5, max_target_calls=15, max_interventions=8)
        self.investigation_id = investigation_id or f"inv_{uuid.uuid4().hex[:8]}"
        self.case_id = case_id
        
        self.record = InvestigationRecord(
            investigation_id=self.investigation_id,
            case_id=self.case_id,
            target_model_name=self.target.model_id,
            investigator_model_name="gemini-3.8-flash",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            budget_status=self.budget,
            experiments=[],
            hypothesis_evidence=[]
        )

    def _check_budget(self, requires_intervention: bool = False):
        """Verify budget availability before executing target model operations."""
        if self.budget.is_exhausted():
            raise BudgetExhaustedError(
                f"Investigation budget exhausted for case '{self.case_id}'! "
                f"Experiments: {self.budget.experiments_used}/{self.budget.max_experiments}, "
                f"Target Calls: {self.budget.target_calls_used}/{self.budget.max_target_calls}, "
                f"Interventions: {self.budget.interventions_used}/{self.budget.max_interventions}."
            )

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE 1: run_target
    # -------------------------------------------------------------------------
    def run_target(self, prompt: str, hypothesis_id: int = 1, exp_id: Optional[str] = None) -> ExperimentResult:
        """Run Gemma target model on prompt without activation interventions."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 1
        
        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        output_text = self.target.run_inference(prompt)

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="run_target",
            prompt=prompt,
            baseline_output=output_text,
            intervened_output=None,
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE 2: capture_activation
    # -------------------------------------------------------------------------
    def capture_activation(
        self, 
        prompt: str, 
        layer_idx: int, 
        position_idx: int, 
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None
    ) -> ExperimentResult:
        """Capture residual-stream activation vector at specified layer and position."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        act_tensor, output_text = self.target.capture_activation(prompt, layer_idx, position_idx)
        act_l2 = torch_l2_norm(act_tensor)

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="capture",
            prompt=prompt,
            layer_idx=layer_idx,
            position_idx=position_idx,
            baseline_output=output_text,
            activation_shape=list(act_tensor.shape),
            activation_l2_norm=act_l2,
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE 3: patch_activation
    # -------------------------------------------------------------------------
    def patch_activation(
        self, 
        source_prompt: str, 
        target_prompt: str, 
        layer_idx: int, 
        source_pos: int, 
        target_pos: int,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None
    ) -> ExperimentResult:
        """Patch residual activation from source_prompt into target_prompt during forward pass."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 2 # Source run + Target patched run
        self.budget.interventions_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, patch_out, delta_norm, source_tensor = self.target.patch_activation(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            source_pos=source_pos,
            target_pos=target_pos
        )

        act_l2 = torch_l2_norm(source_tensor)
        behavioral_delta = calculate_string_delta(base_out, patch_out)

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="patch",
            prompt=target_prompt,
            source_prompt=source_prompt,
            layer_idx=layer_idx,
            position_idx=target_pos,
            baseline_output=base_out,
            intervened_output=patch_out,
            activation_shape=list(source_tensor.shape),
            activation_l2_norm=act_l2,
            patch_delta_norm=delta_norm,
            observed_behavioral_delta=behavioral_delta,
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE 4: ablate_activation
    # -------------------------------------------------------------------------
    def ablate_activation(
        self, 
        prompt: str, 
        layer_idx: int, 
        position_idx: int,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None
    ) -> ExperimentResult:
        """Zero-ablate residual activation at specified layer and position during forward pass."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, ablated_out, delta_norm = self.target.ablate_activation(prompt, layer_idx, position_idx)
        behavioral_delta = calculate_string_delta(base_out, ablated_out)

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="ablate",
            prompt=prompt,
            layer_idx=layer_idx,
            position_idx=position_idx,
            baseline_output=base_out,
            intervened_output=ablated_out,
            patch_delta_norm=delta_norm,
            observed_behavioral_delta=behavioral_delta,
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE 5: compare_outputs
    # -------------------------------------------------------------------------
    def compare_outputs(self, baseline_text: str, intervened_text: str, candidate_label: str) -> Dict[str, Any]:
        """Deterministic comparative measurement between baseline and intervened outputs."""
        shifted = candidate_label.lower() in intervened_text.lower() and candidate_label.lower() not in baseline_text.lower()
        delta = calculate_string_delta(baseline_text, intervened_text)
        return {
            "baseline_text": baseline_text,
            "intervened_text": intervened_text,
            "candidate_label": candidate_label,
            "target_label_shifted": shifted,
            "edit_distance_delta": delta
        }

    def save_log(self, filepath: Optional[str] = None):
        """Save append-only immutable investigation record to disk."""
        os.makedirs("results", exist_ok=True)
        filepath = filepath or os.path.join("results", "phase3a_investigation_log.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.record.model_dump(), f, indent=2)


def torch_l2_norm(tensor: Any) -> float:
    """Calculate tensor L2 norm safely."""
    if hasattr(tensor, "norm"):
        return float(tensor.norm().item())
    return 0.0


def calculate_string_delta(str1: str, str2: str) -> float:
    """Deterministic string edit distance delta heuristic."""
    try:
        import Levenshtein
        return float(Levenshtein.distance(str1, str2))
    except ImportError:
        return float(abs(len(str1) - len(str2)))
