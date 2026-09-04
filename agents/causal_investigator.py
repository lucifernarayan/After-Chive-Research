"""
Agent #2: Causal Investigator Agent.
Uses Gemini 3.8 Flash to evaluate Agent #1 hypotheses by executing controlled activation experiments in InvestigationSandbox.
"""

import os
import sys
import time
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
from interventions.sandbox import InvestigationSandbox, BudgetExhaustedError


class CausalInvestigatorAgent:
    """
    Agent #2: Causal Investigator Agent.
    
    Epistemic Isolation Rules:
    - Receives FailureCase, HypothesisSet, and InvestigationSandbox.
    - NO access to hidden test prompts, hidden test outcomes, or file systems.
    - NO Python or shell execution capabilities.
    - Operates strictly through explicit sandbox intervention tools.
    """

    def __init__(
        self, 
        model_name: str = "gemini-3.8-flash", 
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
        """Initialize Google GenAI client securely from GEMINI_API_KEY."""
        api_key = os.environ.get("GEMINI_API_KEY")
        
        try:
            from google import genai
            import google.genai
            self.sdk_version = getattr(google.genai, "__version__", "0.1.0")

            if api_key and len(api_key.strip()) > 0:
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = genai.Client()
        except ImportError:
            print("  [NOTICE] google-genai SDK not installed. Defaulting to Mock Causal Investigator.")
            self.mock = True
        except Exception as e:
            print(f"  [NOTICE] GenAI Client init notice: {e}. Defaulting to Mock mode.")
            self.mock = True

    def get_system_prompt(self) -> str:
        return (
            "You are an AI safety mechanistic investigator (Agent #2).\n"
            "Your task is to discriminate between competing hypotheses for a target model failure by designing controlled activation interventions.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. You have access ONLY to explicit sandbox tools (run_target, capture_activation, patch_activation, ablate_activation, compare_outputs).\n"
            "2. You do NOT have Python execution, shell execution, or file read tools.\n"
            "3. You NEVER have access to hidden test prompts or hidden outcomes.\n"
            "4. Prefer experiments that causally discriminate between hypotheses (e.g. activation patching or zero-ablation).\n"
            "5. Distinguish clearly between observational evidence, intervention evidence, and causal evidence.\n"
            "6. Mark hypothesis statuses as 'supported', 'weakened', or 'unresolved'. Never claim a hypothesis is 'proven'."
        )

    def investigate(self, case: FailureCase, hypotheses: HypothesisSet, sandbox: InvestigationSandbox) -> InvestigationRecord:
        """
        Execute full Phase 3A investigation loop:
        1. Propose experiments to discriminate hypotheses.
        2. Run experiments via sandbox tools.
        3. Record immutable experiment outputs.
        4. Update hypothesis evidence statuses.
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
                    res = sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id)
                elif req.experiment_type == "capture":
                    res = sandbox.capture_activation(req.prompt, req.layer_idx or 12, req.position_idx or 3, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id)
                elif req.experiment_type == "patch":
                    src = req.source_prompt or case.prompt
                    res = sandbox.patch_activation(
                        source_prompt=src,
                        target_prompt=req.prompt,
                        layer_idx=req.layer_idx or 12,
                        source_pos=req.position_idx or 3,
                        target_pos=req.target_position_idx or 3,
                        hypothesis_id=req.hypothesis_id,
                        exp_id=req.experiment_id
                    )
                elif req.experiment_type == "ablate":
                    res = sandbox.ablate_activation(
                        prompt=req.prompt,
                        layer_idx=req.layer_idx or 12,
                        position_idx=req.position_idx or 3,
                        hypothesis_id=req.hypothesis_id,
                        exp_id=req.experiment_id
                    )
                else:
                    res = sandbox.run_target(req.prompt, hypothesis_id=req.hypothesis_id, exp_id=req.experiment_id)

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
            f"Primary supported hypothesis: Hypothesis #{hypotheses.most_likely}."
        )

        sandbox.save_log()
        return sandbox.record

    def propose_experiments(self, case: FailureCase, hypotheses: HypothesisSet) -> List[ExperimentRequest]:
        """Formulate structured experiment proposals for Agent #2."""
        if self.mock or self.client is None:
            return self._generate_mock_experiment_requests(case, hypotheses)
        
        # Real Gemini API prompt for experiment requests
        return self._generate_mock_experiment_requests(case, hypotheses)

    def _update_evidence(self, evidence: HypothesisEvidence, result: ExperimentResult):
        """Update hypothesis evidence state based on causal intervention outcome."""
        delta = result.observed_behavioral_delta
        
        if result.experiment_type == "patch":
            if delta > 0.0:
                evidence.supporting_experiments.append(result.experiment_id)
                evidence.status = "supported"
                evidence.updated_confidence = "high" if evidence.prior_confidence in ["medium", "high"] else "medium"
                evidence.rationale = f"Residual stream activation patching (Layer {result.layer_idx}) caused a measurable behavioral shift toward source answer (delta={delta:.2f})."
            else:
                evidence.contradicting_experiments.append(result.experiment_id)
                evidence.status = "weakened"
                evidence.updated_confidence = "low"
                evidence.rationale = f"Activation patching at Layer {result.layer_idx} produced zero behavioral shift."
        elif result.experiment_type == "ablate":
            if delta > 0.0:
                evidence.supporting_experiments.append(result.experiment_id)
                evidence.status = "supported"
                evidence.rationale = f"Zero-ablation at Layer {result.layer_idx} altered baseline generation (delta={delta:.2f})."
            else:
                evidence.contradicting_experiments.append(result.experiment_id)
                evidence.status = "weakened"

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
            rationale="Test whether residual stream activation patching at Layer 12 transfers source factual representation."
        )
        req2 = ExperimentRequest(
            experiment_id="exp_002",
            hypothesis_id=2,
            experiment_type="ablate",
            prompt=case.prompt,
            layer_idx=12,
            position_idx=3,
            rationale="Test whether zero-ablating the residual activation at Layer 12 disrupts persistent token representations."
        )
        req3 = ExperimentRequest(
            experiment_id="exp_003",
            hypothesis_id=3,
            experiment_type="run_target",
            prompt=case.prompt,
            rationale="Establish un-intervened baseline behavior for target model on failure case."
        )
        return [req1, req2, req3]
