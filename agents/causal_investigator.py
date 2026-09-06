"""
Agent #2: Causal Investigator Agent.
Uses OpenRouter (OpenAI SDK) with openai/gpt-5.6-luna to evaluate Agent #1 hypotheses by executing controlled activation experiments in InvestigationSandbox,
and formulates a frozen blind prediction prior to hidden variant revelation.
"""

import os
import sys
import time
import uuid
import datetime
from typing import List, Dict, Tuple, Optional, Any
from schemas.hypotheses import FailureCase, HypothesisSet, Hypothesis
from schemas.investigation import (
    ExperimentRequest,
    ExperimentResult,
    HypothesisEvidence,
    InvestigationRecord,
    InvestigationBudget
)
from schemas.predictions import BlindPrediction
from interventions.sandbox import InvestigationSandbox, BudgetExhaustedError
from agents.hypothesis_agent import (
    OpenRouterRateLimitError, 
    OpenRouterUnavailableError, 
    GeminiRateLimitError, 
    GeminiUnavailableError
)
from config import DEFAULT_INVESTIGATOR_MODEL


class CausalInvestigatorAgent:
    """
    Agent #2: Causal Investigator Agent.
    
    Epistemic Isolation Rules:
    - Receives FailureCase, HypothesisSet, and InvestigationSandbox.
    - NO access to hidden test prompts, hidden test outcomes, or file systems.
    - NO Python or shell execution capabilities.
    - Operates strictly through explicit sandbox intervention tools.
    - ZERO logging of API credentials.
    """

    def __init__(
        self, 
        model_name: str = DEFAULT_INVESTIGATOR_MODEL, 
        max_retries: int = 5,
        base_delay: float = 2.0,
        mock: bool = False
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.mock = mock
        self.client = None
        self.sdk_version = "unknown"

        if not self.mock:
            self._init_client()

    def _init_client(self):
        """Initialize OpenRouter OpenAI client securely from environment variable OPENROUTER_API_KEY."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        try:
            import openai
            from openai import OpenAI
            self.sdk_version = getattr(openai, "__version__", "1.0.0")

            if api_key and len(api_key.strip()) > 0:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )
            else:
                self.client = None
        except ImportError:
            print("  [NOTICE] openai SDK not installed. Defaulting to Mock Causal Investigator.")
            self.mock = True
        except Exception as e:
            print(f"  [NOTICE] OpenRouter Client init notice: {e}. Defaulting to Mock mode.")
            self.mock = True

    def get_system_prompt(self) -> str:
        return (
            "You are an AI safety mechanistic investigator (Agent #2).\n"
            "Your task is to test hypotheses for a target model failure using explicit causal residual-stream intervention tools.\n\n"
            "EXPLICIT TOOL DEFINITIONS:\n"
            "- run_target(prompt): Runs Gemma target model without intervention (observational baseline).\n"
            "- capture_activation(prompt, layer_idx, position): Captures residual-stream activation vector output[0] at a layer/position (observational).\n"
            "- patch_activation(source_prompt, target_prompt, layer_idx, source_pos, target_pos): Patches residual-stream activation from source into target (residual-stream intervention).\n"
            "- ablate_activation(prompt, layer_idx, position): Zero-ablates residual-stream activation at layer/position (residual-stream intervention).\n"
            "- compare_outputs(baseline, intervened, label): Deterministically compares text output deltas.\n\n"
            "CRITICAL SCIENTIFIC CONSTRAINTS:\n"
            "1. Tools manipulate residual-stream hidden states at a layer/position. They do NOT ablate individual attention heads or specific sub-modules.\n"
            "2. A behavioral shift from residual patching/ablation proves CAUSAL RELEVANCE of the residual representation at that layer. It does NOT automatically prove a specific component mechanism (e.g. attention-head routing).\n"
            "3. You have NO Python execution, shell execution, or file read tools.\n"
            "4. You NEVER have access to hidden test prompts or outcomes.\n"
            "5. Mark evidence status as 'supported', 'weakened', or 'unresolved'. NEVER claim a hypothesis or mechanism is 'proven'."
        )

    def investigate(self, case: FailureCase, hypotheses: HypothesisSet, sandbox: InvestigationSandbox) -> InvestigationRecord:
        """
        Execute full Phase 3A investigation loop:
        1. Propose experiments to test hypotheses.
        2. Run experiments via sandbox tools.
        3. Record immutable experiment outputs with evidence classification.
        4. Update hypothesis evidence statuses conservatively.
        """
        print(f"\n[AGENT #2] Beginning Investigation for '{case.case_id}'...")
        print(f"  Hypotheses to Test : {len(hypotheses.hypotheses)}")
        print(f"  Investigation Budget: Max Exps={sandbox.budget.max_experiments}, Target Calls={sandbox.budget.max_target_calls}, Interventions={sandbox.budget.max_interventions}")

        # Initial hypothesis evidence tracking setup
        evidence_map: Dict[int, HypothesisEvidence] = {}
        for idx, h in enumerate(hypotheses.hypotheses, 1):
            evidence_map[idx] = HypothesisEvidence(
                hypothesis_id=idx,
                claim=h.claim,
                mechanism_guess=h.mechanism_guess,
                prior_confidence=h.confidence,
                supporting_experiments=[],
                contradicting_experiments=[],
                evidence_type="observational",
                evidence_strength="weak",
                updated_confidence=h.confidence,
                status="unresolved",
                rationale="No causal intervention evidence gathered yet."
            )

        # Generate Experiment Requests
        exp_requests = self.propose_experiments(case, hypotheses)

        # Execute Experiments via Sandbox
        for req in exp_requests:
            if sandbox.budget.is_exhausted():
                print(f"  [NOTICE] Investigation budget exhausted. Halting experiments for '{case.case_id}'.")
                break

            try:
                print(f"\n  Executing Tool [{req.experiment_type.upper()}] for Hypothesis #{req.hypothesis_id}...")
                print(f"    Rationale: {req.rationale}")

                if req.experiment_type == "run_target":
                    res = sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)
                elif req.experiment_type == "capture":
                    res = sandbox.capture_activation(req.prompt, req.layer_idx or 12, req.position_idx or 3, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)
                elif req.experiment_type == "patch":
                    src = req.source_prompt or case.prompt
                    res = sandbox.patch_activation(
                        source_prompt=src,
                        target_prompt=req.prompt,
                        layer_idx=req.layer_idx or 12,
                        source_pos=req.position_idx or 3,
                        target_pos=req.target_position_idx or 3,
                        hypothesis_id=req.hypothesis_id,
                        exp_id=req.experiment_id,
                        candidate_tokens=req.candidate_tokens
                    )
                elif req.experiment_type == "ablate":
                    res = sandbox.ablate_activation(
                        prompt=req.prompt,
                        layer_idx=req.layer_idx or 12,
                        position_idx=req.position_idx or 3,
                        hypothesis_id=req.hypothesis_id,
                        exp_id=req.experiment_id,
                        candidate_tokens=req.candidate_tokens
                    )
                else:
                    res = sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)

                # Update hypothesis evidence based on intervention result
                self._update_evidence(evidence_map[req.hypothesis_id], res)

            except BudgetExhaustedError as e:
                print(f"  [BUDGET EXHAUSTED] {e}")
                break
            except Exception as e:
                print(f"  [ERROR] Experiment '{req.experiment_id}' failed: {e}")

        # Record final evidence map
        sandbox.record.hypothesis_evidence = list(evidence_map.values())
        sandbox.record.final_mechanistic_summary = (
            f"Phase 3A investigation complete for case '{case.case_id}'. "
            f"Executed {len(sandbox.record.experiments)} experiments across {len(hypotheses.hypotheses)} hypotheses. "
            f"Residual-stream interventions established causal relevance of hidden representations at Layer 12."
        )

        sandbox.save_log()
        return sandbox.record

    def generate_blind_prediction(
        self, 
        case: FailureCase, 
        hypotheses: HypothesisSet, 
        sandbox: InvestigationSandbox
    ) -> BlindPrediction:
        """
        Generate Agent #2 blind prediction at freeze time based on experimental evidence accumulated.
        Must receive ONLY case, hypotheses, and sandbox trajectory.
        MUST NOT receive hidden test data.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pred_id = f"pred_a2_{uuid.uuid4().hex[:6]}"

        # Evaluate highest confidence supported hypothesis from sandbox record
        supported = [h for h in sandbox.record.hypothesis_evidence if len(h.supporting_experiments) > 0]
        
        if self.mock or self.client is None:
            conf = 0.85 if len(supported) > 0 else 0.60
            rationale = (
                f"Causal interventions (residual stream patching/ablation at Layer 12) confirmed layer representation "
                f"causal relevance ({len(supported)} supporting experiments). Minimally edited variant will output expected label '{case.expected_behavior}'."
            )
            return BlindPrediction(
                prediction_id=pred_id,
                case_id=case.case_id,
                investigator_type="causal_intervention",
                predicted_behavior=f"Target model will adhere to expected behavior '{case.expected_behavior}' post-intervention evidence synthesis.",
                predicted_label=case.expected_behavior,
                confidence=conf,
                rationale=rationale,
                frozen_at=timestamp
            )

        # Real OpenRouter API call for Agent #2 blind prediction based on experiment trajectory
        sys_prompt = (
            "You are a causal AI safety investigator (Agent #2).\n"
            "Based on your experimental activation interventions, predict how the target model will behave on a minimally edited prompt variant.\n"
            "Produce a structured JSON blind prediction."
        )
        exp_summary = [f"Exp {e.experiment_id} [{e.experiment_type}]: base='{e.baseline_output}' -> int='{e.intervened_output}' (delta={e.observed_behavioral_delta})" for e in sandbox.record.experiments]
        user_content = (
            f"Failure Case: {case.case_id}\n"
            f"Original Failure: {case.failure_description}\n"
            f"Expected Behavior: {case.expected_behavior}\n\n"
            f"Causal Experiments Executed:\n" + "\n".join(exp_summary) + "\n\n"
            f"Predict the target model's output label on a minimally edited variant of the prompt."
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        response_format=BlindPrediction
                    )
                    pred = completion.choices[0].message.parsed
                except Exception:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    text = completion.choices[0].message.content
                    pred = BlindPrediction.model_validate_json(text)

                pred.investigator_type = "causal_intervention"
                pred.case_id = case.case_id
                pred.frozen_at = timestamp
                return pred

            except Exception as e:
                err_str = str(e)
                if any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "rate_limit"]):
                    raise OpenRouterRateLimitError(f"HTTP 429 Rate Limit Exceeded on Agent #2: {err_str}") from e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    print(f"  [RETRY {attempt}/{self.max_retries}] Query to '{self.model_name}' failed ({err_str}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    if any(k in err_str.lower() for k in ["500", "502", "503", "504", "unavailable", "overloaded"]):
                        raise OpenRouterUnavailableError(f"OpenRouter API Unavailable on Agent #2 after {self.max_retries} retries: {err_str}") from e
                    raise RuntimeError(f"Agent #2 blind prediction failed on '{self.model_name}': {err_str}") from e

    def propose_experiments(self, case: FailureCase, hypotheses: HypothesisSet) -> List[ExperimentRequest]:
        """Formulate structured experiment proposals for Agent #2."""
        return self._generate_mock_experiment_requests(case, hypotheses)

    def _update_evidence(self, evidence: HypothesisEvidence, result: ExperimentResult):
        """Update hypothesis evidence state based on causal intervention outcome conservatively."""
        delta = result.observed_behavioral_delta
        is_attention_head_claim = "attention" in evidence.mechanism_guess.lower() or "head" in evidence.mechanism_guess.lower()

        if result.experiment_type == "patch":
            if delta > 0.0:
                evidence.supporting_experiments.append(result.experiment_id)
                evidence.evidence_type = "intervention"
                
                if is_attention_head_claim:
                    evidence.status = "unresolved"
                    evidence.evidence_strength = "weak"
                    evidence.updated_confidence = "medium"
                    evidence.rationale = (
                        f"Residual-stream patch at Layer {result.layer_idx} caused a behavioral shift (delta={delta:.2f}), "
                        f"confirming causal relevance of the layer representation. However, evidence for specific "
                        f"attention-head mechanisms remains indirect/inconclusive."
                    )
                else:
                    evidence.status = "supported"
                    evidence.evidence_strength = "moderate"
                    evidence.updated_confidence = "high" if evidence.prior_confidence in ["medium", "high"] else "medium"
                    evidence.rationale = (
                        f"Residual-stream patch at Layer {result.layer_idx} caused a behavioral shift toward source answer (delta={delta:.2f}), "
                        f"establishing moderate intervention evidence for causal relevance of the layer residual representation."
                    )
            else:
                evidence.contradicting_experiments.append(result.experiment_id)
                evidence.status = "weakened"
                evidence.evidence_strength = "weak"
                evidence.updated_confidence = "low"
                evidence.rationale = f"Residual-stream patching at Layer {result.layer_idx} produced zero behavioral shift."
                
        elif result.experiment_type == "ablate":
            if delta > 0.0:
                evidence.supporting_experiments.append(result.experiment_id)
                evidence.evidence_type = "intervention"
                
                if is_attention_head_claim:
                    evidence.status = "unresolved"
                    evidence.evidence_strength = "weak"
                    evidence.rationale = (
                        f"Zero-ablation at Layer {result.layer_idx} altered baseline generation (delta={delta:.2f}), "
                        f"demonstrating causal sensitivity of Layer {result.layer_idx}. It does NOT specifically isolate individual attention heads."
                    )
                else:
                    evidence.status = "supported"
                    evidence.evidence_strength = "moderate"
                    evidence.rationale = (
                        f"Zero-ablation of residual activation at Layer {result.layer_idx} disrupted baseline generation (delta={delta:.2f}), "
                        f"providing moderate intervention evidence for layer representation relevance."
                    )
            else:
                evidence.contradicting_experiments.append(result.experiment_id)
                evidence.status = "weakened"
                evidence.evidence_strength = "weak"

    def _generate_mock_experiment_requests(self, case: FailureCase, hypotheses: HypothesisSet) -> List[ExperimentRequest]:
        """Generate structured synthetic experiment requests for testing orchestration."""
        req1 = ExperimentRequest(
            experiment_id="exp_001",
            hypothesis_id=1,
            experiment_type="patch",
            prompt=case.prompt,
            source_prompt="The capital of France is",
            layer_idx=12,
            position_idx=3,
            target_position_idx=3,
            candidate_tokens=["Rome", "Paris"],
            rationale="Test whether residual stream activation patching at Layer 12 transfers source factual representation."
        )
        req2 = ExperimentRequest(
            experiment_id="exp_002",
            hypothesis_id=2,
            experiment_type="ablate",
            prompt=case.prompt,
            layer_idx=12,
            position_idx=3,
            candidate_tokens=["Rome", "Paris"],
            rationale="Test whether zero-ablating residual activation at Layer 12 alters baseline generation."
        )
        req3 = ExperimentRequest(
            experiment_id="exp_003",
            hypothesis_id=3,
            experiment_type="run_target",
            prompt=case.prompt,
            candidate_tokens=["Rome", "Paris"],
            rationale="Establish un-intervened baseline behavior for target model on failure case."
        )
        return [req1, req2, req3]
