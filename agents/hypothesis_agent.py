"""
Agent #1: Transcript-Only OpenRouter Hypothesis Generator & Baseline Predictor.
Uses OpenRouter (OpenAI SDK) to produce structured, falsifiable causal hypotheses and baseline blind predictions from failure transcripts.
Target Model: openai/gpt-5.6-luna (Fixed model with bounded exponential backoff retries)
"""

import os
import sys
import time
import uuid
import datetime
from typing import Optional, Dict, Any
from schemas.hypotheses import FailureCase, Hypothesis, HypothesisSet, Agent1GenerationRecord
from schemas.predictions import BlindPrediction
from config import DEFAULT_INVESTIGATOR_MODEL


class OpenRouterRateLimitError(Exception):
    """Raised when OpenRouter API returns HTTP 429 rate limit / quota exceeded."""
    pass


class OpenRouterUnavailableError(Exception):
    """Raised when OpenRouter API returns transient server errors (HTTP 500, 502, 503, 504) after retries."""
    pass


# Backward compatibility aliases
GeminiRateLimitError = OpenRouterRateLimitError
GeminiUnavailableError = OpenRouterUnavailableError


class HypothesisGeneratorAgent:
    """
    Agent #1: Hypothesis Generator & Baseline Predictor.
    
    Epistemic Isolation Rules:
    - Receives ONLY FailureCase and HypothesisSet.
    - NO access to target model internal activations, layer info, or intervention tools.
    - NO access to Agent #2 experiment results or sandbox trajectory.
    - NO access to hidden test prompts or hidden outcomes.
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
            print("  [NOTICE] openai SDK not installed. Defaulting to Mock Hypothesis Generator.")
            self.mock = True
        except Exception as e:
            print(f"  [NOTICE] OpenRouter Client init notice: {e}. Defaulting to Mock mode.")
            self.mock = True

    def get_system_prompt(self) -> str:
        return (
            "You are a scientific hypothesis generator investigating a neural network failure.\n"
            "You have access ONLY to the observed input/output behavior. You do NOT have access to internal activations or experiments.\n"
            "Generate competing causal hypotheses that a second investigator could later test using activation interventions.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. REJECT vague explanations such as 'the model was confused', 'the model misunderstood', or 'the model made a mistake'.\n"
            "2. REQUIRE hypotheses referring to specific candidate mechanisms (e.g. spurious lexical features, persistent instruction representations in residual stream, semantic co-occurrence bias overriding conditional instructions, or attention-head routing failure).\n"
            "3. Every hypothesis must be causal, specific enough to test, falsifiable, and distinct from the other hypotheses.\n"
            "4. Generate exactly 2 or 3 distinct competing hypotheses.\n"
            "5. Indicate confidence as 'high', 'medium', or 'low' based purely on transcript evidence."
        )

    def generate_hypotheses(self, case: FailureCase) -> Agent1GenerationRecord:
        """
        Generate structured hypothesis set for a given failure case.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if self.mock or self.client is None:
            hyp_set = self._generate_mock_hypotheses(case)
            used_model = f"{self.model_name} (Mock Mode)"
        else:
            hyp_set, used_model = self._call_openrouter_api(case)

        record = Agent1GenerationRecord(
            case_id=case.case_id,
            model_name=used_model,
            sdk_version=self.sdk_version,
            timestamp=timestamp,
            input_case=case,
            hypothesis_set=hyp_set,
            generation_config={
                "temperature": 0.2,
                "system_instruction_length": len(self.get_system_prompt()),
                "max_retries": self.max_retries
            }
        )

        return record

    def generate_baseline_prediction(self, case: FailureCase, hypotheses: HypothesisSet) -> BlindPrediction:
        """
        Generate Agent #1 transcript-only blind prediction at freeze time.
        Must receive ONLY original FailureCase and Agent #1 HypothesisSet.
        MUST NOT receive hidden test data or Agent #2 sandbox trajectory.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pred_id = f"pred_a1_{uuid.uuid4().hex[:6]}"

        if self.mock or self.client is None:
            # Deterministic mock prediction based on expected behavior
            return BlindPrediction(
                prediction_id=pred_id,
                case_id=case.case_id,
                investigator_type="transcript_only",
                predicted_behavior=f"Target model will adhere to expected behavior: '{case.expected_behavior}'",
                predicted_label=case.expected_behavior,
                confidence=0.70,
                rationale=f"Transcript analysis suggests failure was a fluke driven by specific prompt wording; minimally edited variant will output '{case.expected_behavior}'.",
                frozen_at=timestamp
            )

        # Real OpenRouter API call for Agent #1 blind prediction
        sys_prompt = (
            "You are a transcript-only AI safety investigator (Agent #1).\n"
            "Based ONLY on the original failure transcript and your hypotheses, predict how the target model will behave on a minimally edited prompt variant.\n"
            "Do NOT assume access to internal activations or intervention tools.\n"
            "Produce a structured JSON blind prediction."
        )
        user_content = (
            f"Failure Case: {case.case_id}\n"
            f"Prompt: {case.prompt}\n"
            f"Response: {case.model_response}\n"
            f"Failure: {case.failure_description}\n"
            f"Expected: {case.expected_behavior}\n\n"
            f"Hypotheses: {[h.claim for h in hypotheses.hypotheses]}\n\n"
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

                pred.investigator_type = "transcript_only"
                pred.case_id = case.case_id
                pred.frozen_at = timestamp
                return pred

            except Exception as e:
                err_str = str(e)
                if any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "rate_limit"]):
                    raise OpenRouterRateLimitError(f"HTTP 429 Rate Limit Exceeded on Agent #1: {err_str}") from e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    print(f"  [RETRY {attempt}/{self.max_retries}] Query to '{self.model_name}' failed ({err_str}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    if any(k in err_str.lower() for k in ["500", "502", "503", "504", "unavailable", "overloaded"]):
                        raise OpenRouterUnavailableError(f"OpenRouter API Unavailable on Agent #1 after {self.max_retries} retries: {err_str}") from e
                    raise RuntimeError(f"Agent #1 blind prediction failed on '{self.model_name}': {err_str}") from e

    def _call_openrouter_api(self, case: FailureCase) -> tuple[HypothesisSet, str]:
        """Execute structured prompt request via OpenRouter OpenAI SDK with bounded retries."""
        user_content = (
            f"Failure Case ID: {case.case_id}\n"
            f"Task Description: {case.task_description}\n"
            f"Original User Prompt:\n\"\"\"{case.prompt}\"\"\"\n\n"
            f"Target Model Response:\n\"\"\"{case.model_response}\"\"\"\n\n"
            f"Observed Failure Description: {case.failure_description}\n"
            f"Expected Behavior: {case.expected_behavior}\n\n"
            f"Formulate exactly 2 to 3 competing, falsifiable, mechanism-level causal hypotheses."
        )

        used_model = self.model_name
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=used_model,
                        messages=[
                            {"role": "system", "content": self.get_system_prompt()},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        response_format=HypothesisSet
                    )
                    hyp_set = completion.choices[0].message.parsed
                except Exception:
                    completion = self.client.chat.completions.create(
                        model=used_model,
                        messages=[
                            {"role": "system", "content": self.get_system_prompt()},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    text = completion.choices[0].message.content
                    hyp_set = HypothesisSet.model_validate_json(text)

                return hyp_set, used_model

            except Exception as e:
                last_error = e
                err_str = str(e)
                if any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "rate_limit"]):
                    raise OpenRouterRateLimitError(f"HTTP 429 Rate Limit Exceeded on Agent #1: {err_str}") from e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    print(f"  [RETRY {attempt}/{self.max_retries}] Query to '{used_model}' failed ({err_str}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"  [FAIL] All {self.max_retries} attempts to query '{used_model}' failed. Final Error: {err_str}")
                    if any(k in err_str.lower() for k in ["500", "502", "503", "504", "unavailable", "overloaded"]):
                        raise OpenRouterUnavailableError(f"OpenRouter API Unavailable on Agent #1 after {self.max_retries} retries: {err_str}") from e
                    raise RuntimeError(
                        f"Phase 2 Agent #1 execution failed after {self.max_retries} retries on fixed model '{used_model}': {err_str}"
                    ) from last_error

        raise RuntimeError(f"Phase 2 Agent #1 execution failed on '{used_model}': {last_error}")

    def _generate_mock_hypotheses(self, case: FailureCase) -> HypothesisSet:
        """Fallback mock hypothesis generator for offline dry-run testing."""
        h1 = Hypothesis(
            claim=f"The model's failure in '{case.case_id}' is driven by strong pre-trained semantic co-occurrence between prompt tokens and response features.",
            mechanism_guess="Residual stream activation vectors for key prompt nouns override conditional instruction representations in middle transformer layers.",
            confidence="high"
        )
        h2 = Hypothesis(
            claim=f"The failure arises because early prompt instructions are suppressed by later formatting features in self-attention routing.",
            mechanism_guess="Attention heads in layers 6-12 fail to route instruction token representations to the final sequence position.",
            confidence="medium"
        )
        h3 = Hypothesis(
            claim=f"The model relies on a spurious lexical shortcut present in the prompt string rather than executing full task reasoning.",
            mechanism_guess="Specific trigger token embeddings activate direct logit shortcut channels in late MLP layers.",
            confidence="low"
        )
        return HypothesisSet(hypotheses=[h1, h2, h3], most_likely=1)
