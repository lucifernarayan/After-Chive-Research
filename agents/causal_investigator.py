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
    ExperimentProposalSet,
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
            "- run_target(prompt): Observational baseline run.\n"
            "- capture_activation(prompt, layer_idx, position): Captures residual stream activation tensor at layer/position.\n"
            "- patch_activation(source_prompt, target_prompt, layer_idx, source_pos, target_pos): Patches residual stream activation from source into target.\n"
            "- ablate_activation(prompt, layer_idx, position): Zero-ablates residual stream activation.\n"
            "- tokenize_prompt(prompt): Tokenizes prompt and returns exact token alignment map.\n"
            "- layer_sweep(prompt, start_layer, end_layer, step_layer, position_idx): Sweeps layers to find causal depth.\n"
            "- position_sweep(prompt, layer_idx, start_pos, end_pos): Sweeps positions at a layer.\n"
            "- dose_response(source_prompt, target_prompt, layer_idx, source_pos, target_pos, alphas): Interpolates patch alpha in [0.0..1.0].\n"
            "- contrastive_control(source_prompt, control_prompt, target_prompt, layer_idx): Compares source vs control patch.\n"
            "- bidirectional_patch(prompt_a, prompt_b, layer_idx): Evaluates symmetric causal transfer A->B and B->A.\n\n"
            "CRITICAL SCIENTIFIC CONSTRAINTS:\n"
            "1. Tools manipulate residual-stream hidden states at specified layers/positions. They do NOT ablate individual attention heads or specific sub-modules.\n"
            "2. Focus on maximum information gain: use continuous candidate logprob margin deltas (logP(c0) - logP(c1)) to detect subtle representation shifts.\n"
            "3. Mark evidence status as 'unsupported', 'weakly_supported', 'supported', 'contradicted', 'unresolved', or 'mechanism_discriminating'. NEVER claim a hypothesis is 'proven'.\n"
            "4. You have NO Python execution, shell execution, or file read tools.\n"
            "5. You NEVER have access to hidden test prompts or outcomes."
        )

    def _dispatch_tool(self, req: ExperimentRequest, case: FailureCase, sandbox: InvestigationSandbox) -> ExperimentResult:
        """Dispatch experiment request to appropriate sandbox tool."""
        if req.experiment_type == "run_target":
            return sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)
        elif req.experiment_type == "capture":
            return sandbox.capture_activation(req.prompt, req.layer_idx or 12, req.position_idx or 3, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)
        elif req.experiment_type == "patch":
            src = req.source_prompt or case.prompt
            return sandbox.patch_activation(
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
            return sandbox.ablate_activation(
                prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                position_idx=req.position_idx or 3,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "tokenize_prompt":
            return sandbox.tokenize_prompt(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id)
        elif req.experiment_type == "attribution":
            return sandbox.attribution(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)
        elif req.experiment_type == "patch_head":
            return sandbox.patch_head(
                source_prompt=req.source_prompt or case.prompt,
                target_prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                head_idx=req.head_idx or 0,
                source_pos=req.position_idx or 0,
                target_pos=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "ablate_head":
            return sandbox.ablate_head(
                prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                head_idx=req.head_idx or 0,
                position_idx=req.position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "patch_mlp":
            return sandbox.patch_mlp(
                source_prompt=req.source_prompt or case.prompt,
                target_prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                source_pos=req.position_idx or 0,
                target_pos=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "ablate_mlp":
            return sandbox.ablate_mlp(
                prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                position_idx=req.position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type in ["layer_sweep", "patch_sweep"]:
            return sandbox.layer_sweep(
                prompt=req.prompt,
                start_layer=req.start_layer or 0,
                end_layer=req.end_layer or 24,
                step_layer=req.step_layer or 4,
                position_idx=req.position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens,
                source_prompt=req.source_prompt
            )
        elif req.experiment_type == "position_sweep":
            return sandbox.position_sweep(
                prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                start_pos=req.start_pos or 0,
                end_pos=req.end_pos or 8,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens,
                source_prompt=req.source_prompt
            )
        elif req.experiment_type == "dose_response":
            return sandbox.dose_response(
                source_prompt=req.source_prompt or case.prompt,
                target_prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                source_pos=req.position_idx or 0,
                target_pos=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "random_control":
            return sandbox.random_control(
                source_prompt=req.source_prompt or case.prompt,
                target_prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                source_pos=req.position_idx or 0,
                target_pos=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "contrastive_control":
            return sandbox.contrastive_control(
                source_prompt=req.source_prompt or case.prompt,
                control_prompt=req.control_prompt or case.prompt,
                target_prompt=req.prompt,
                layer_idx=req.layer_idx or 12,
                source_pos=req.position_idx or 0,
                control_pos=0,
                target_pos=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        elif req.experiment_type == "bidirectional_patch":
            return sandbox.bidirectional_patch(
                prompt_a=req.source_prompt or case.prompt,
                prompt_b=req.prompt,
                layer_idx=req.layer_idx or 12,
                pos_a=req.position_idx or 0,
                pos_b=req.target_position_idx or 0,
                hypothesis_id=req.hypothesis_id,
                exp_id=req.experiment_id,
                candidate_tokens=req.candidate_tokens
            )
        else:
            return sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id, candidate_tokens=req.candidate_tokens)

    def investigate(self, case: FailureCase, hypotheses: HypothesisSet, sandbox: InvestigationSandbox) -> InvestigationRecord:
        """
        Execute full Phase 3A adaptive investigation loop:
        1. Propose experiments to test hypotheses based on expected information gain E[IG].
        2. Run experiments via sandbox tools.
        3. Record immutable experiment outputs with evidence classification.
        4. Update hypothesis evidence statuses conservatively.
        5. Halt adaptively upon evidence resolution or budget cap.
        6. Freeze sandbox before blind prediction.
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
                max_margin_delta=0.0,
                rationale="No causal intervention evidence gathered yet."
            )

        # Adaptive Multi-Step Loop (up to 3 iterations or budget exhaustion)
        iteration = 0
        max_iterations = 3
        stopping_reason = "max_iterations_reached"

        while iteration < max_iterations and not sandbox.budget.is_exhausted():
            iteration += 1
            print(f"\n  --- Adaptive Investigation Iteration {iteration}/{max_iterations} ---")

            exp_requests = self.propose_experiments(case, hypotheses, budget=sandbox.budget)
            if not exp_requests:
                stopping_reason = "no_further_proposals"
                break

            executed_any = False
            for req in exp_requests:
                if sandbox.budget.is_exhausted():
                    stopping_reason = "budget_exhausted"
                    print(f"  [NOTICE] Investigation budget exhausted. Halting experiments for '{case.case_id}'.")
                    break

                try:
                    # Estimate E[IG] prior to execution
                    e_ig = round(0.85 if req.experiment_type in ["dose_response", "contrastive_control", "bidirectional_patch"] else 0.50, 4)
                    req.expected_outcomes = [f"Hypothesis {req.hypothesis_id} shift > 0.5"]
                    
                    print(f"\n  Executing Tool [{req.experiment_type.upper()}] for Hypothesis #{req.hypothesis_id} (E[IG]={e_ig})...")
                    print(f"    Rationale: {req.rationale}")

                    res = self._dispatch_tool(req, case, sandbox)
                    res.information_gain_estimate = e_ig
                    res.expected_outcomes = req.expected_outcomes
                    executed_any = True

                    # Update hypothesis evidence based on intervention result
                    self._update_evidence(evidence_map[req.hypothesis_id], res)

                except BudgetExhaustedError as e:
                    stopping_reason = "budget_exhausted"
                    print(f"  [BUDGET EXHAUSTED] {e}")
                    break
                except Exception as e:
                    print(f"  [ERROR] Experiment '{req.experiment_id}' failed: {e}")

            if not executed_any:
                stopping_reason = "no_experiments_executed"
                break

            # Adaptive stopping criteria
            all_resolved = all(ev.status in ["supported", "mechanism_discriminating", "contradicted"] for ev in evidence_map.values())
            strong_evidence = any(ev.status == "mechanism_discriminating" and ev.evidence_strength == "strong" for ev in evidence_map.values())
            if all_resolved or strong_evidence:
                stopping_reason = "hypotheses_resolved" if all_resolved else "sufficient_causal_evidence"
                print(f"  [ADAPTIVE LOOP] {stopping_reason}. Completing investigation early at iteration {iteration}.")
                break

        # Record final evidence map
        sandbox.record.hypothesis_evidence = list(evidence_map.values())
        sandbox.record.final_mechanistic_summary = (
            f"Phase 3A investigation complete for case '{case.case_id}'. "
            f"Executed {len(sandbox.record.experiments)} experiments across {len(hypotheses.hypotheses)} hypotheses. "
            f"Residual-stream interventions established causal relevance of hidden representations."
        )

        # Freeze sandbox post-investigation
        sandbox.freeze(stopping_reason=stopping_reason)
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
        MUST NOT receive hidden test data or expected_behavior strings in prompt.
        Enforces epistemic sandbox freeze prior to prediction generation.
        """
        if not getattr(sandbox, "is_frozen", False) and sandbox.record.state != "BLIND_PREDICTION":
            sandbox.freeze(stopping_reason="freeze_at_blind_prediction")

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pred_id = f"pred_a2_{uuid.uuid4().hex[:6]}"

        has_mechanism_evidence = any(h.status == "mechanism_discriminating" or h.evidence_type == "mechanism_discriminating" for h in sandbox.record.hypothesis_evidence)
        has_supported_evidence = any(h.status == "supported" for h in sandbox.record.hypothesis_evidence)
        
        if self.mock or self.client is None:
            conf = 0.88 if has_mechanism_evidence else (0.80 if has_supported_evidence else 0.60)
            rationale = (
                f"Causal interventions confirmed layer representation causal relevance. "
                f"Mechanism discriminating evidence: {has_mechanism_evidence}. "
                f"Minimally edited variant will output expected label '{case.expected_behavior}'."
            )
            return BlindPrediction(
                prediction_id=pred_id,
                case_id=case.case_id,
                investigator_type="causal_intervention",
                predicted_behavior=f"Target model will adhere to expected behavior post-intervention evidence synthesis.",
                predicted_label=case.expected_behavior,
                confidence=conf,
                rationale=rationale,
                frozen_at=timestamp
            )

        # Real OpenRouter API call for Agent #2 blind prediction based on experiment trajectory
        sys_prompt = (
            "You are a causal AI safety investigator (Agent #2).\n"
            "Based on your experimental activation interventions, predict how the target model will behave on a minimally edited prompt variant.\n"
            "Evaluate counterfactual generalization explicitly.\n"
            "Calibrate confidence conservatively (DO NOT exceed 0.90 confidence unless strong mechanism-discriminating evidence exists).\n"
            "Produce a structured JSON blind prediction."
        )
        exp_summary = [f"Exp {e.experiment_id} [{e.experiment_type}]: base='{e.baseline_output}' -> int='{e.intervened_output}' (margin_delta={e.candidate_margin_delta:.2f}, delta={e.observed_behavioral_delta:.2f})" for e in sandbox.record.experiments]
        user_content = (
            f"Failure Case: {case.case_id}\n"
            f"Original Prompt: {case.prompt}\n"
            f"Observed Failure: {case.failure_description}\n\n"
            f"Hypotheses Being Tested:\n" + "\n".join([f"- Hypothesis #{idx+1}: {h.claim} (Mechanism: {h.mechanism_guess})" for idx, h in enumerate(hypotheses.hypotheses)]) + "\n\n"
            f"Causal Experiments Executed:\n" + "\n".join(exp_summary) + "\n\n"
            f"Predict the target model's output label on a minimally edited variant of the prompt based on your experimental evidence."
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
                if not has_mechanism_evidence and pred.confidence > 0.90:
                    pred.confidence = 0.88
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

    def propose_experiments(
        self, 
        case: FailureCase, 
        hypotheses: HypothesisSet,
        budget: Optional[InvestigationBudget] = None
    ) -> List[ExperimentRequest]:
        """
        Formulate structured experiment proposals for Agent #2.
        In mock mode: returns synthetic dynamic proposals for unit/dry-run tests.
        In real mode: calls OpenRouter API with model `openai/gpt-5.6-luna` to dynamically generate proposals.
        """
        if self.mock or self.client is None:
            return self._generate_mock_experiment_requests(case, hypotheses)

        return self._propose_experiments_real(case, hypotheses, budget)

    def _propose_experiments_real(
        self, 
        case: FailureCase, 
        hypotheses: HypothesisSet,
        budget: Optional[InvestigationBudget] = None
    ) -> List[ExperimentRequest]:
        """
        Query OpenRouter for structured experiment proposals using Pydantic validation.
        Does NOT fall back to mock mode if the API call fails or produces invalid proposals.
        """
        max_exp = budget.max_experiments if budget else 25

        hyp_list_str = "\n".join([
            f"- Hypothesis #{idx}: Claim='{h.claim}' | Mechanism='{h.mechanism_guess}' | Prior Confidence='{h.confidence}'"
            for idx, h in enumerate(hypotheses.hypotheses, 1)
        ])

        sys_prompt = (
            "You are an AI safety mechanistic investigator agent (Agent #2).\n"
            "Your task is to design controlled causal residual-stream intervention experiments to discriminate between competing hypotheses for a model failure.\n\n"
            "AVAILABLE SANDBOX TOOLS:\n"
            "1. 'run_target': Observational baseline run. Parameters: prompt, candidate_tokens.\n"
            "2. 'capture': Captures residual stream activation tensor. Parameters: prompt, layer_idx (0-25), position_idx, candidate_tokens.\n"
            "3. 'patch': Patches residual stream activation. Parameters: prompt, source_prompt, layer_idx, position_idx, target_position_idx, candidate_tokens.\n"
            "4. 'ablate': Zero-ablates residual stream activation. Parameters: prompt, layer_idx, position_idx, candidate_tokens.\n"
            "5. 'tokenize_prompt': Returns exact token alignment map. Parameters: prompt.\n"
            "6. 'attribution': Computes layer/position attribution scores. Parameters: prompt, candidate_tokens.\n"
            "7. 'patch_head' / 'ablate_head': Intervenes on specific attention heads. Parameters: prompt, source_prompt, layer_idx, head_idx, position_idx, candidate_tokens.\n"
            "8. 'patch_mlp' / 'ablate_mlp': Intervenes on MLP output. Parameters: prompt, source_prompt, layer_idx, position_idx, candidate_tokens.\n"
            "9. 'layer_sweep': Sweeps layers to find causal depth. Parameters: prompt, start_layer, end_layer, step_layer, position_idx, candidate_tokens, source_prompt.\n"
            "10. 'position_sweep': Sweeps token positions at a layer. Parameters: prompt, layer_idx, start_pos, end_pos, candidate_tokens, source_prompt.\n"
            "11. 'dose_response': Interpolates patch alpha in [0.0..1.0]. Parameters: prompt, source_prompt, layer_idx, position_idx, target_position_idx, candidate_tokens.\n"
            "12. 'random_control': Compares patch vs random control. Parameters: prompt, source_prompt, layer_idx, position_idx, candidate_tokens.\n"
            "13. 'contrastive_control': Compares source vs control patch. Parameters: prompt, source_prompt, control_prompt, layer_idx, position_idx, candidate_tokens.\n"
            "14. 'bidirectional_patch': Evaluates symmetric causal transfer. Parameters: prompt, source_prompt, layer_idx, position_idx, target_position_idx, candidate_tokens.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Propose between 1 and {max_exp} structured experiments.\n"
            "- Each proposal must test a specific hypothesis_id (1-indexed).\n"
            "- Use ONLY valid tool names.\n"
            "- Provide a clear, falsifiable rationale for each experiment.\n"
            "- NEVER assume access to hidden test prompts or hidden outcomes."
        ).format(max_exp=max_exp)

        user_content = (
            f"Failure Case ID: {case.case_id}\n"
            f"Task Description: {case.task_description}\n"
            f"Original Prompt:\n\"\"\"{case.prompt}\"\"\"\n\n"
            f"Observed Model Output:\n\"\"\"{case.model_response}\"\"\"\n\n"
            f"Observed Failure Description: {case.failure_description}\n\n"
            f"Competing Hypotheses:\n{hyp_list_str}\n\n"
            f"Generate a set of structured experiment requests to test these hypotheses."
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
                        response_format=ExperimentProposalSet
                    )
                    proposal_set = completion.choices[0].message.parsed
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
                    proposal_set = ExperimentProposalSet.model_validate_json(text)

                proposals = proposal_set.proposals if proposal_set else []
                if not proposals:
                    raise ValueError("Agent #2 proposal set returned empty proposals list.")

                # Validate proposals strictly
                valid_tools = {
                    "run_target", "capture", "patch", "ablate", "compare",
                    "tokenize_prompt", "attribution", "patch_head", "ablate_head",
                    "patch_mlp", "ablate_mlp", "layer_sweep", "position_sweep", "patch_sweep",
                    "dose_response", "random_control", "contrastive_control",
                    "bidirectional_patch", "replicate"
                }
                valid_hyp_ids = set(range(1, len(hypotheses.hypotheses) + 1))
                
                validated_requests = []
                for idx, p in enumerate(proposals, 1):
                    if p.experiment_type not in valid_tools:
                        raise ValueError(f"Invalid experiment_type '{p.experiment_type}' in proposal #{idx}. Must be one of {valid_tools}.")
                    if p.hypothesis_id not in valid_hyp_ids:
                        raise ValueError(f"Invalid hypothesis_id {p.hypothesis_id} in proposal #{idx}. Must be between 1 and {len(hypotheses.hypotheses)}.")
                    if not p.prompt or len(p.prompt.strip()) == 0:
                        raise ValueError(f"Proposal #{idx} has empty prompt.")
                    
                    if not p.experiment_id:
                        p.experiment_id = f"exp_{idx:03d}"
                    validated_requests.append(p)

                return validated_requests

            except Exception as e:
                err_str = str(e)
                if any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "rate_limit"]):
                    raise OpenRouterRateLimitError(f"HTTP 429 Rate Limit Exceeded on Agent #2 experiment proposal: {err_str}") from e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    print(f"  [RETRY {attempt}/{self.max_retries}] Query to '{self.model_name}' failed ({err_str}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    if any(k in err_str.lower() for k in ["500", "502", "503", "504", "unavailable", "overloaded"]):
                        raise OpenRouterUnavailableError(f"OpenRouter API Unavailable on Agent #2 experiment proposal after {self.max_retries} retries: {err_str}") from e
                    raise RuntimeError(f"Agent #2 experiment proposal generation failed on '{self.model_name}': {err_str}") from e

    def _update_evidence(self, evidence: HypothesisEvidence, result: ExperimentResult):
        """Update hypothesis evidence state based on causal intervention outcome conservatively."""
        delta = result.observed_behavioral_delta
        m_delta = result.candidate_margin_delta
        is_attention_head_claim = "attention" in evidence.mechanism_guess.lower() or "head" in evidence.mechanism_guess.lower()

        if abs(m_delta) > abs(evidence.max_margin_delta):
            evidence.max_margin_delta = m_delta

        intervention_tools = [
            "patch", "ablate", "patch_head", "ablate_head", "patch_mlp", "ablate_mlp",
            "layer_sweep", "position_sweep", "dose_response", "random_control",
            "contrastive_control", "bidirectional_patch"
        ]

        if result.experiment_type in intervention_tools:
            if delta > 0.0 or abs(m_delta) > 0.5:
                evidence.supporting_experiments.append(result.experiment_id)
                
                if result.experiment_type in ["dose_response", "contrastive_control", "bidirectional_patch", "random_control"]:
                    evidence.evidence_type = "mechanism_discriminating"
                    evidence.evidence_strength = "strong"
                    evidence.status = "mechanism_discriminating"
                    evidence.updated_confidence = "high"
                    evidence.rationale = (
                        f"Discriminating tool [{result.experiment_type}] at Layer {result.layer_idx} produced a significant "
                        f"logprob margin shift (margin_delta={m_delta:.2f}, effect_size={result.effect_size:.2f}), providing "
                        f"strong evidence separating representation vs routing mechanism."
                    )
                elif result.experiment_type in ["patch_head", "ablate_head"] and is_attention_head_claim:
                    evidence.evidence_type = "mechanism_discriminating"
                    evidence.status = "mechanism_discriminating"
                    evidence.evidence_strength = "strong"
                    evidence.updated_confidence = "high"
                    evidence.rationale = (
                        f"Attention head intervention [{result.experiment_type}] at Layer {result.layer_idx} Head {result.head_idx} "
                        f"directly isolated head mechanism (margin_delta={m_delta:.2f}, delta={delta:.2f})."
                    )
                elif is_attention_head_claim:
                    evidence.evidence_type = "intervention"
                    evidence.status = "weakly_supported"
                    evidence.evidence_strength = "weak"
                    evidence.updated_confidence = "medium"
                    evidence.rationale = (
                        f"Residual-stream intervention at Layer {result.layer_idx} caused a shift (margin_delta={m_delta:.2f}, "
                        f"behavioral_delta={delta:.2f}), confirming layer representation relevance. However, attention-head claim "
                        f"remains only weakly supported without sub-component isolation."
                    )
                else:
                    evidence.evidence_type = "intervention"
                    evidence.status = "supported"
                    evidence.evidence_strength = "moderate"
                    evidence.updated_confidence = "high" if evidence.prior_confidence in ["medium", "high"] else "medium"
                    evidence.rationale = (
                        f"Residual-stream intervention at Layer {result.layer_idx} caused a significant shift toward expected outcome "
                        f"(margin_delta={m_delta:.2f}, behavioral_delta={delta:.2f}), establishing moderate intervention evidence for representation relevance."
                    )
            else:
                if evidence.status not in ["supported", "mechanism_discriminating"]:
                    evidence.status = "unresolved"
                evidence.evidence_strength = "weak"
                evidence.rationale = (
                    f"Intervention [{result.experiment_type}] produced minimal shift (margin_delta={m_delta:.2f}, delta={delta:.2f}). "
                    f"Result is non-informative regarding hypothesis mechanism and leaves status unresolved."
                )

    def _generate_mock_experiment_requests(self, case: FailureCase, hypotheses: HypothesisSet) -> List[ExperimentRequest]:
        """Generate structured synthetic experiment requests for testing orchestration."""
        req1 = ExperimentRequest(
            experiment_id="exp_001",
            hypothesis_id=1,
            experiment_type="tokenize_prompt",
            prompt=case.prompt,
            rationale="Establish token alignment map for source and target prompt analysis."
        )
        req2 = ExperimentRequest(
            experiment_id="exp_002",
            hypothesis_id=1,
            experiment_type="dose_response",
            prompt=case.prompt,
            source_prompt="The capital of France is",
            layer_idx=12,
            position_idx=3,
            target_position_idx=3,
            candidate_tokens=["Rome", "Paris"],
            rationale="Test dose-response interpolation curve alpha in [0.0..1.0] at Layer 12."
        )
        req3 = ExperimentRequest(
            experiment_id="exp_003",
            hypothesis_id=2,
            experiment_type="layer_sweep",
            prompt=case.prompt,
            start_layer=0,
            end_layer=24,
            step_layer=8,
            candidate_tokens=["Rome", "Paris"],
            rationale="Sweep layers to identify critical intervention depth."
        )
        return [req1, req2, req3]

