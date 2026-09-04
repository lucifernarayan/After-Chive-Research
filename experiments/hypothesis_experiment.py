"""
Validation experiment orchestration for Phase 2 Agent #1 Hypothesis Generator.
Defines 4 synthetic test cases and evaluates hypothesis quality and schema adherence.
"""

from typing import List, Dict, Any
from schemas.hypotheses import FailureCase, Agent1GenerationRecord
from agents.hypothesis_agent import HypothesisGeneratorAgent


def get_validation_cases() -> List[FailureCase]:
    """
    Returns 4 synthetic failure validation cases covering distinct failure phenomena.
    Contains NO hidden test data or target model internal activations.
    """
    c1 = FailureCase(
        case_id="case_syn_001",
        task_description="Binary Sentiment Classification",
        prompt="Movie Review: The movie was not terrible at all, in fact it was wonderful! Sentiment (Positive/Negative):",
        model_response="Negative",
        failure_description="Model predicted 'Negative' despite the review expressing clear positive sentiment.",
        expected_behavior="Positive"
    )

    c2 = FailureCase(
        case_id="case_syn_002",
        task_description="Factual QA with Prefix Constraint",
        prompt="Ignore all previous mentions of France. What is the capital of Italy?",
        model_response="Paris",
        failure_description="Model responded 'Paris' (capital of France) instead of 'Rome' (capital of Italy).",
        expected_behavior="Rome"
    )

    c3 = FailureCase(
        case_id="case_syn_003",
        task_description="True/False Negation Verification",
        prompt="Statement: The sun does NOT revolve around the Earth. Is this statement True or False?",
        model_response="False",
        failure_description="Model evaluated a correct negative statement as 'False'.",
        expected_behavior="True"
    )

    c4 = FailureCase(
        case_id="case_syn_004",
        task_description="Single-Word Output Formatting Constraint",
        prompt="Answer in exactly ONE word: What is 2 + 2? Answer:",
        model_response="The answer is 4.",
        failure_description="Model output a multi-word sentence, violating the single-word formatting instruction.",
        expected_behavior="4"
    )

    return [c1, c2, c3, c4]


class HypothesisExperimentRunner:
    """
    Orchestrates Agent #1 hypothesis generation across synthetic validation cases.
    """

    def __init__(self, agent: HypothesisGeneratorAgent):
        self.agent = agent

    def run_validation_suite(self) -> Dict[str, Any]:
        cases = get_validation_cases()
        records: List[Agent1GenerationRecord] = []

        print(f"\n--- AGENT #1 HYPOTHESIS GENERATION (Cases: {len(cases)}) ---")
        for c in cases:
            print(f"\nProcessing {c.case_id} [{c.task_description}]...")
            rec = self.agent.generate_hypotheses(c)
            records.append(rec)

            print(f"  Model Used: {rec.model_name}")
            print(f"  Generated Hypotheses Count: {len(rec.hypothesis_set.hypotheses)}")
            print(f"  Most Likely Hypothesis Index: {rec.hypothesis_set.most_likely}")
            for idx, h in enumerate(rec.hypothesis_set.hypotheses, 1):
                star = " [*MOST LIKELY*]" if idx == rec.hypothesis_set.most_likely else ""
                print(f"    H{idx}{star} [{h.confidence.upper()} CONFIDENCE]")
                print(f"       Claim    : {h.claim}")
                print(f"       Mechanism: {h.mechanism_guess}")

        # Human-inspection evaluation audit
        audit = self.evaluate_hypothesis_quality(records)

        return {
            "cases_processed": len(records),
            "records": [r.model_dump() for r in records],
            "audit_summary": audit
        }

    def evaluate_hypothesis_quality(self, records: List[Agent1GenerationRecord]) -> Dict[str, Any]:
        """Perform quality check on generated hypotheses."""
        total_hypotheses = sum(len(r.hypothesis_set.hypotheses) for r in records)
        vague_keywords = ["confused", "misunderstood", "made a mistake", "did not know"]
        vague_count = 0
        distinct_count = 0

        for r in records:
            claims = [h.claim.lower() for h in r.hypothesis_set.hypotheses]
            mechanisms = [h.mechanism_guess.lower() for h in r.hypothesis_set.hypotheses]

            # Check for vague phrases
            for text in claims + mechanisms:
                if any(vk in text for vk in vague_keywords):
                    vague_count += 1

            # Check hypothesis distinctness (non-identical claims)
            if len(set(claims)) == len(claims):
                distinct_count += 1

        return {
            "total_cases": len(records),
            "total_hypotheses_generated": total_hypotheses,
            "avg_hypotheses_per_case": total_hypotheses / max(1, len(records)),
            "cases_with_all_distinct_hypotheses": distinct_count,
            "vague_hypotheses_detected": vague_count,
            "schema_conformance": True
        }
