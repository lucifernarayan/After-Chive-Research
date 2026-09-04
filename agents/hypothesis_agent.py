"""
Agent #1: Transcript-Only Gemini Hypothesis Generator.
Uses Google GenAI SDK to produce structured, falsifiable causal hypotheses from failure transcripts.
Default Model: gemini-3.8-flash
"""

import os
import sys
import datetime
from typing import Optional, Dict, Any
from schemas.hypotheses import FailureCase, Hypothesis, HypothesisSet, Agent1GenerationRecord


class HypothesisGeneratorAgent:
    """
    Agent #1: Hypothesis Generator.
    
    Epistemic Isolation Rules:
    - Receives ONLY FailureCase (prompt, response, failure description, expected behavior).
    - NO access to target model internal activations, layer info, or intervention tools.
    - NO access to hidden test prompts or hidden outcomes.
    - ZERO logging of API credentials.
    """

    def __init__(
        self, 
        model_name: str = "gemini-3.8-flash", 
        fallback_model: str = "gemini-3.8-flash",
        mock: bool = False
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.mock = mock
        self.client = None
        self.sdk_version = "unknown"

        if not self.mock:
            self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client securely from environment variable GEMINI_API_KEY."""
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
            print("  [NOTICE] google-genai SDK not installed. Defaulting to Mock Hypothesis Generator.")
            self.mock = True
        except Exception as e:
            print(f"  [NOTICE] GenAI Client init notice: {e}. Defaulting to Mock mode.")
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
            # Synthetic mock hypothesis generation conforming to Pydantic schema
            hyp_set = self._generate_mock_hypotheses(case)
            used_model = f"{self.model_name} (Mock Mode)"
        else:
            hyp_set, used_model = self._call_gemini_api(case)

        record = Agent1GenerationRecord(
            case_id=case.case_id,
            model_name=used_model,
            sdk_version=self.sdk_version,
            timestamp=timestamp,
            input_case=case,
            hypothesis_set=hyp_set,
            generation_config={
                "temperature": 0.2,
                "system_instruction_length": len(self.get_system_prompt())
            }
        )

        return record

    def _call_gemini_api(self, case: FailureCase) -> tuple[HypothesisSet, str]:
        """Execute structured prompt request via Google GenAI SDK."""
        from google.genai import types

        user_content = (
            f"Failure Case ID: {case.case_id}\n"
            f"Task Description: {case.task_description}\n"
            f"Original User Prompt:\n\"\"\"{case.prompt}\"\"\"\n\n"
            f"Target Model Response:\n\"\"\"{case.model_response}\"\"\"\n\n"
            f"Observed Failure Description: {case.failure_description}\n"
            f"Expected Behavior: {case.expected_behavior}\n\n"
            f"Formulate exactly 2 to 3 competing, falsifiable, mechanism-level causal hypotheses."
        )

        config = types.GenerateContentConfig(
            system_instruction=self.get_system_prompt(),
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=HypothesisSet
        )

        used_model = self.model_name
        try:
            response = self.client.models.generate_content(
                model=used_model,
                contents=user_content,
                config=config
            )
            if hasattr(response, "parsed") and response.parsed is not None:
                hyp_set = response.parsed
            else:
                hyp_set = HypothesisSet.model_validate_json(response.text)
        except Exception as e:
            print(f"  [NOTICE] Error querying {used_model}: {e}. Retrying with {self.fallback_model}...")
            used_model = self.fallback_model
            response = self.client.models.generate_content(
                model=used_model,
                contents=user_content,
                config=config
            )
            if hasattr(response, "parsed") and response.parsed is not None:
                hyp_set = response.parsed
            else:
                hyp_set = HypothesisSet.model_validate_json(response.text)

        return hyp_set, used_model

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
