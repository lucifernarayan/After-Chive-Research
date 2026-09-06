"""
Investigation Sandbox: Safe, auditable intervention execution environment for Agent #2.

Security & Isolation Constraints:
- NO arbitrary python execution (no exec, eval, or code tools).
- NO shell commands (no subprocess or OS execution).
- NO arbitrary file reading (no filesystem access tools).
- ONLY explicit intervention primitives (run_target, capture_activation, patch_activation, ablate_activation, compare_outputs).
- Strict budget enforcement (max_experiments, max_target_calls, max_interventions).
- Append-only immutable JSON record logging.

Scientific Evidence Classification:
- Tool interventions perform residual-stream manipulation at specific layers/positions.
- They do NOT isolate individual attention heads or specific sub-components.
- Interventions establish causal relevance of hidden representations, NOT proof of detailed component mechanisms.
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
from config import DEFAULT_INVESTIGATOR_MODEL


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
            investigator_model_name=DEFAULT_INVESTIGATOR_MODEL,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            budget_status=self.budget,
            experiments=[],
            hypothesis_evidence=[]
        )

    def _check_budget(self):
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
    def run_target(self, prompt: str, hypothesis_id: int = 1, exp_id: Optional[str] = None, candidate_tokens: Optional[List[str]] = None) -> ExperimentResult:
        """Run Gemma target model on prompt without activation interventions (Observational)."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 1
        
        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        output_text, cand_logprobs = self.target.run_inference(prompt, candidate_tokens=candidate_tokens)

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="run_target",
            prompt=prompt,
            baseline_output=output_text,
            intervened_output=None,
            candidate_logprobs=cand_logprobs,
            evidence_type="observational",
            evidence_strength="weak",
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
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Capture residual-stream activation vector at specified layer/position (Observational)."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        act_tensor, output_text, cand_logprobs = self.target.capture_activation(prompt, layer_idx, position_idx, candidate_tokens=candidate_tokens)
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
            candidate_logprobs=cand_logprobs,
            evidence_type="observational",
            evidence_strength="weak",
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
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Patch residual activation from source_prompt into target_prompt at layer_idx/position_idx (Intervention)."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 2 # Source run + Target patched run
        self.budget.interventions_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(target_prompt, candidate_tokens=candidate_tokens)
        _, patch_out, delta_norm, source_tensor, cand_logprobs = self.target.patch_activation(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            source_pos=source_pos,
            target_pos=target_pos,
            candidate_tokens=candidate_tokens
        )

        act_l2 = torch_l2_norm(source_tensor)
        behavioral_delta = calculate_string_delta(base_out, patch_out)
        delta_logprobs, margin_delta, eff_size = compute_logprob_metrics(base_logprobs, cand_logprobs, candidate_tokens, behavioral_delta)

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
            candidate_logprobs=cand_logprobs,
            candidate_delta_logprobs=delta_logprobs,
            candidate_margin_delta=margin_delta,
            effect_size=eff_size,
            evidence_type="intervention",
            evidence_strength="moderate" if (behavioral_delta > 0 or abs(margin_delta) > 0.5) else "weak",
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
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Zero-ablate residual-stream activation at layer_idx/position_idx (Intervention)."""
        self._check_budget()
        self.budget.experiments_used += 1
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(prompt, candidate_tokens=candidate_tokens)
        _, ablated_out, delta_norm, cand_logprobs = self.target.ablate_activation(
            prompt, layer_idx, position_idx, candidate_tokens=candidate_tokens
        )
        behavioral_delta = calculate_string_delta(base_out, ablated_out)
        delta_logprobs, margin_delta, eff_size = compute_logprob_metrics(base_logprobs, cand_logprobs, candidate_tokens, behavioral_delta)

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
            candidate_logprobs=cand_logprobs,
            candidate_delta_logprobs=delta_logprobs,
            candidate_margin_delta=margin_delta,
            effect_size=eff_size,
            evidence_type="intervention",
            evidence_strength="moderate" if (behavioral_delta > 0 or abs(margin_delta) > 0.5) else "weak",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: tokenize_prompt
    # -------------------------------------------------------------------------
    def tokenize_prompt(
        self, 
        prompt: str, 
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None
    ) -> ExperimentResult:
        """Tokenize prompt and return exact token alignment map (Observational)."""
        self._check_budget()
        self.budget.experiments_used += 1
        
        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        token_map = self.target.tokenize_prompt(prompt)
        base_out = f"Tokenized {len(token_map)} tokens."

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="tokenize_prompt",
            prompt=prompt,
            baseline_output=base_out,
            token_alignment=token_map,
            evidence_type="observational",
            evidence_strength="weak",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: layer_sweep
    # -------------------------------------------------------------------------
    def layer_sweep(
        self,
        prompt: str,
        start_layer: int = 0,
        end_layer: int = 24,
        step_layer: int = 4,
        position_idx: int = 0,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None,
        source_prompt: Optional[str] = None
    ) -> ExperimentResult:
        """Systematically sweep layers (start_layer to end_layer) for activation intervention."""
        self._check_budget()
        self.budget.experiments_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(prompt, candidate_tokens=candidate_tokens)
        sweep_records = []
        max_m_delta = 0.0

        for layer in range(start_layer, end_layer + 1, max(1, step_layer)):
            if self.budget.is_exhausted():
                break
            if source_prompt:
                _, p_out, d_norm, _, c_logprobs = self.target.patch_activation(
                    source_prompt=source_prompt, target_prompt=prompt, layer_idx=layer,
                    source_pos=position_idx, target_pos=position_idx, candidate_tokens=candidate_tokens
                )
                self.budget.target_calls_used += 2
                self.budget.interventions_used += 1
                b_delta = calculate_string_delta(base_out, p_out)
                out_text = p_out
            else:
                _, a_out, d_norm, c_logprobs = self.target.ablate_activation(
                    prompt=prompt, layer_idx=layer, position_idx=position_idx, candidate_tokens=candidate_tokens
                )
                self.budget.target_calls_used += 2
                self.budget.interventions_used += 1
                b_delta = calculate_string_delta(base_out, a_out)
                out_text = a_out

            d_logprobs, m_delta, eff = compute_logprob_metrics(base_logprobs, c_logprobs, candidate_tokens, b_delta)
            if abs(m_delta) > abs(max_m_delta):
                max_m_delta = m_delta

            sweep_records.append({
                "layer_idx": layer,
                "output": out_text,
                "delta_norm": d_norm,
                "behavioral_delta": b_delta,
                "candidate_logprobs": c_logprobs,
                "candidate_margin_delta": m_delta,
                "effect_size": eff
            })

        eff_size = min(1.0, max(-1.0, max_m_delta / 4.0)) if max_m_delta != 0.0 else 0.0

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="layer_sweep",
            prompt=prompt,
            source_prompt=source_prompt,
            position_idx=position_idx,
            baseline_output=base_out,
            intervened_output=f"Layer sweep completed across layers {start_layer}-{end_layer}.",
            sweep_results=sweep_records,
            candidate_logprobs=base_logprobs,
            candidate_margin_delta=max_m_delta,
            effect_size=eff_size,
            evidence_type="intervention",
            evidence_strength="moderate" if abs(max_m_delta) > 0.5 or len(sweep_records) > 0 else "weak",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: position_sweep
    # -------------------------------------------------------------------------
    def position_sweep(
        self,
        prompt: str,
        layer_idx: int = 12,
        start_pos: int = 0,
        end_pos: int = 8,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None,
        source_prompt: Optional[str] = None
    ) -> ExperimentResult:
        """Systematically sweep token positions (start_pos to end_pos) at layer_idx."""
        self._check_budget()
        self.budget.experiments_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(prompt, candidate_tokens=candidate_tokens)
        sweep_records = []
        max_m_delta = 0.0

        for pos in range(start_pos, end_pos + 1):
            if self.budget.is_exhausted():
                break
            if source_prompt:
                _, p_out, d_norm, _, c_logprobs = self.target.patch_activation(
                    source_prompt=source_prompt, target_prompt=prompt, layer_idx=layer_idx,
                    source_pos=pos, target_pos=pos, candidate_tokens=candidate_tokens
                )
                self.budget.target_calls_used += 2
                self.budget.interventions_used += 1
                b_delta = calculate_string_delta(base_out, p_out)
                out_text = p_out
            else:
                _, a_out, d_norm, c_logprobs = self.target.ablate_activation(
                    prompt=prompt, layer_idx=layer_idx, position_idx=pos, candidate_tokens=candidate_tokens
                )
                self.budget.target_calls_used += 2
                self.budget.interventions_used += 1
                b_delta = calculate_string_delta(base_out, a_out)
                out_text = a_out

            d_logprobs, m_delta, eff = compute_logprob_metrics(base_logprobs, c_logprobs, candidate_tokens, b_delta)
            if abs(m_delta) > abs(max_m_delta):
                max_m_delta = m_delta

            sweep_records.append({
                "position_idx": pos,
                "output": out_text,
                "delta_norm": d_norm,
                "behavioral_delta": b_delta,
                "candidate_logprobs": c_logprobs,
                "candidate_margin_delta": m_delta,
                "effect_size": eff
            })

        eff_size = min(1.0, max(-1.0, max_m_delta / 4.0)) if max_m_delta != 0.0 else 0.0

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="position_sweep",
            prompt=prompt,
            source_prompt=source_prompt,
            layer_idx=layer_idx,
            baseline_output=base_out,
            intervened_output=f"Position sweep completed across positions {start_pos}-{end_pos}.",
            sweep_results=sweep_records,
            candidate_logprobs=base_logprobs,
            candidate_margin_delta=max_m_delta,
            effect_size=eff_size,
            evidence_type="intervention",
            evidence_strength="moderate" if abs(max_m_delta) > 0.5 or len(sweep_records) > 0 else "weak",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: dose_response
    # -------------------------------------------------------------------------
    def dose_response(
        self,
        source_prompt: str,
        target_prompt: str,
        layer_idx: int = 12,
        source_pos: int = 0,
        target_pos: int = 0,
        alphas: Optional[List[float]] = None,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Measure dose-response curve using interpolated activation patching alpha in [0.0, 0.25, 0.5, 0.75, 1.0]."""
        self._check_budget()
        self.budget.experiments_used += 1

        alphas = alphas or [0.0, 0.25, 0.5, 0.75, 1.0]
        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(target_prompt, candidate_tokens=candidate_tokens)
        curve = {}
        max_m_delta = 0.0

        for a in alphas:
            if self.budget.is_exhausted():
                break
            _, p_out, d_norm, _, c_logprobs = self.target.patch_interpolation(
                source_prompt=source_prompt, target_prompt=target_prompt, layer_idx=layer_idx,
                source_pos=source_pos, target_pos=target_pos, alpha=a, candidate_tokens=candidate_tokens
            )
            self.budget.target_calls_used += 2
            self.budget.interventions_used += 1
            b_delta = calculate_string_delta(base_out, p_out)
            _, m_delta, _ = compute_logprob_metrics(base_logprobs, c_logprobs, candidate_tokens, b_delta)
            curve[f"{a:.2f}"] = m_delta
            if abs(m_delta) > abs(max_m_delta):
                max_m_delta = m_delta

        eff_size = min(1.0, max(-1.0, max_m_delta / 4.0))

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="dose_response",
            prompt=target_prompt,
            source_prompt=source_prompt,
            layer_idx=layer_idx,
            position_idx=target_pos,
            baseline_output=base_out,
            intervened_output=f"Dose response curve computed across alphas {alphas}.",
            dose_response_curve=curve,
            candidate_logprobs=base_logprobs,
            candidate_margin_delta=max_m_delta,
            effect_size=eff_size,
            evidence_type="mechanism_discriminating",
            evidence_strength="strong" if abs(max_m_delta) > 1.0 else "moderate",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: contrastive_control
    # -------------------------------------------------------------------------
    def contrastive_control(
        self,
        source_prompt: str,
        control_prompt: str,
        target_prompt: str,
        layer_idx: int = 12,
        source_pos: int = 0,
        control_pos: int = 0,
        target_pos: int = 0,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Compare activation patch from source_prompt vs contrastive control_prompt into target_prompt."""
        self._check_budget()
        self.budget.experiments_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_out, base_logprobs = self.target.run_inference(target_prompt, candidate_tokens=candidate_tokens)

        # Patch from Source
        _, src_out, src_dnorm, _, src_logprobs = self.target.patch_activation(
            source_prompt=source_prompt, target_prompt=target_prompt, layer_idx=layer_idx,
            source_pos=source_pos, target_pos=target_pos, candidate_tokens=candidate_tokens
        )
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        # Patch from Control
        _, ctrl_out, ctrl_dnorm, _, ctrl_logprobs = self.target.patch_activation(
            source_prompt=control_prompt, target_prompt=target_prompt, layer_idx=layer_idx,
            source_pos=control_pos, target_pos=target_pos, candidate_tokens=candidate_tokens
        )
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        src_b_delta = calculate_string_delta(base_out, src_out)
        _, src_m_delta, src_eff = compute_logprob_metrics(base_logprobs, src_logprobs, candidate_tokens, src_b_delta)

        ctrl_b_delta = calculate_string_delta(base_out, ctrl_out)
        _, ctrl_m_delta, ctrl_eff = compute_logprob_metrics(base_logprobs, ctrl_logprobs, candidate_tokens, ctrl_b_delta)

        net_margin_delta = round(src_m_delta - ctrl_m_delta, 4)
        eff_size = min(1.0, max(-1.0, net_margin_delta / 4.0))

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="contrastive_control",
            prompt=target_prompt,
            source_prompt=source_prompt,
            layer_idx=layer_idx,
            position_idx=target_pos,
            baseline_output=base_out,
            intervened_output=f"Source Out: '{src_out}' | Control Out: '{ctrl_out}'",
            candidate_logprobs=src_logprobs,
            candidate_margin_delta=net_margin_delta,
            effect_size=eff_size,
            sweep_results=[
                {"type": "source", "output": src_out, "margin_delta": src_m_delta, "delta_norm": src_dnorm},
                {"type": "control", "output": ctrl_out, "margin_delta": ctrl_m_delta, "delta_norm": ctrl_dnorm}
            ],
            evidence_type="mechanism_discriminating",
            evidence_strength="strong" if abs(net_margin_delta) > 1.0 else "moderate",
            timestamp=timestamp
        )

        self.record.experiments.append(result)
        self.save_log()
        return result

    # -------------------------------------------------------------------------
    # EXPLICIT TOOL INTERFACE: bidirectional_patch
    # -------------------------------------------------------------------------
    def bidirectional_patch(
        self,
        prompt_a: str,
        prompt_b: str,
        layer_idx: int = 12,
        pos_a: int = 0,
        pos_b: int = 0,
        hypothesis_id: int = 1,
        exp_id: Optional[str] = None,
        candidate_tokens: Optional[List[str]] = None
    ) -> ExperimentResult:
        """Bidirectional patching: patch A -> B and B -> A to verify symmetric causal transfer."""
        self._check_budget()
        self.budget.experiments_used += 1

        exp_id = exp_id or f"exp_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        base_a, base_logprobs_a = self.target.run_inference(prompt_a, candidate_tokens=candidate_tokens)
        base_b, base_logprobs_b = self.target.run_inference(prompt_b, candidate_tokens=candidate_tokens)

        # Patch A -> B
        _, out_a_into_b, norm_ab, _, logprobs_ab = self.target.patch_activation(
            source_prompt=prompt_a, target_prompt=prompt_b, layer_idx=layer_idx,
            source_pos=pos_a, target_pos=pos_b, candidate_tokens=candidate_tokens
        )
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        # Patch B -> A
        _, out_b_into_a, norm_ba, _, logprobs_ba = self.target.patch_activation(
            source_prompt=prompt_b, target_prompt=prompt_a, layer_idx=layer_idx,
            source_pos=pos_b, target_pos=pos_a, candidate_tokens=candidate_tokens
        )
        self.budget.target_calls_used += 2
        self.budget.interventions_used += 1

        b_delta_ab = calculate_string_delta(base_b, out_a_into_b)
        _, m_delta_ab, _ = compute_logprob_metrics(base_logprobs_b, logprobs_ab, candidate_tokens, b_delta_ab)

        b_delta_ba = calculate_string_delta(base_a, out_b_into_a)
        _, m_delta_ba, _ = compute_logprob_metrics(base_logprobs_a, logprobs_ba, candidate_tokens, b_delta_ba)

        avg_m_delta = round((abs(m_delta_ab) + abs(m_delta_ba)) / 2.0, 4)
        eff_size = min(1.0, max(-1.0, avg_m_delta / 4.0))

        result = ExperimentResult(
            investigation_id=self.investigation_id,
            experiment_id=exp_id,
            hypothesis_id=hypothesis_id,
            experiment_type="bidirectional_patch",
            prompt=prompt_b,
            source_prompt=prompt_a,
            layer_idx=layer_idx,
            position_idx=pos_b,
            baseline_output=f"Base A: '{base_a}' | Base B: '{base_b}'",
            intervened_output=f"A->B: '{out_a_into_b}' | B->A: '{out_b_into_a}'",
            candidate_logprobs=logprobs_ab,
            candidate_margin_delta=avg_m_delta,
            effect_size=eff_size,
            sweep_results=[
                {"direction": "A->B", "output": out_a_into_b, "margin_delta": m_delta_ab, "delta_norm": norm_ab},
                {"direction": "B->A", "output": out_b_into_a, "margin_delta": m_delta_ba, "delta_norm": norm_ba}
            ],
            evidence_type="mechanism_discriminating",
            evidence_strength="strong" if avg_m_delta > 1.0 else "moderate",
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


def compute_logprob_metrics(
    baseline_logprobs: Optional[Dict[str, float]], 
    intervened_logprobs: Optional[Dict[str, float]],
    candidate_tokens: Optional[List[str]],
    behavioral_delta: float
) -> Tuple[Optional[Dict[str, float]], float, float]:
    """
    Computes continuous candidate logprob metrics:
    - candidate_delta_logprobs: {cand: logP_int - logP_base}
    - candidate_margin_delta: (logP(c0) - logP(c1))_int - (logP(c0) - logP(c1))_base
    - effect_size: normalized score in [-1.0, 1.0]
    """
    if not baseline_logprobs or not intervened_logprobs or not candidate_tokens:
        eff_size = min(1.0, max(-1.0, behavioral_delta / 10.0))
        return None, 0.0, eff_size

    delta_logprobs = {}
    for cand in candidate_tokens:
        if cand in baseline_logprobs and cand in intervened_logprobs:
            delta_logprobs[cand] = round(intervened_logprobs[cand] - baseline_logprobs[cand], 4)

    margin_delta = 0.0
    if len(candidate_tokens) >= 2:
        c0, c1 = candidate_tokens[0], candidate_tokens[1]
        if c0 in baseline_logprobs and c1 in baseline_logprobs and c0 in intervened_logprobs and c1 in intervened_logprobs:
            base_margin = baseline_logprobs[c0] - baseline_logprobs[c1]
            int_margin = intervened_logprobs[c0] - intervened_logprobs[c1]
            margin_delta = round(int_margin - base_margin, 4)

    eff_size = min(1.0, max(-1.0, margin_delta / 4.0)) if margin_delta != 0.0 else min(1.0, max(-1.0, behavioral_delta / 10.0))
    return delta_logprobs, margin_delta, round(eff_size, 4)


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
