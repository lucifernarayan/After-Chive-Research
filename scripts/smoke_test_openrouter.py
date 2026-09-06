"""
Standalone OpenRouter API Smoke Test.

Verifies that OpenRouter API client integration works with `openai/gpt-5.6-luna`
by making a single structured generation request for a synthetic FailureCase.

Usage:
  python scripts/smoke_test_openrouter.py          # Real API call (requires OPENROUTER_API_KEY)
  python scripts/smoke_test_openrouter.py --mock   # Offline mock validation (no API key needed)
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.hypotheses import FailureCase
from agents.hypothesis_agent import HypothesisGeneratorAgent
from config import DEFAULT_INVESTIGATOR_MODEL, DEFAULT_INVESTIGATOR_PROVIDER


def main():
    parser = argparse.ArgumentParser(description="OpenRouter API Smoke Test")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without making real API calls")
    args = parser.parse_args()

    print("=" * 70)
    print("  OPENROUTER API SMOKE TEST")
    print(f"  Provider: {DEFAULT_INVESTIGATOR_PROVIDER}")
    print(f"  Model   : {DEFAULT_INVESTIGATOR_MODEL}")
    print(f"  Mode    : {'MOCK' if args.mock else 'REAL API'}")
    print("=" * 70)

    if not args.mock:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key or len(api_key.strip()) == 0:
            print("\n[ERROR] OPENROUTER_API_KEY environment variable is missing!")
            print("Set OPENROUTER_API_KEY or run with --mock for offline testing.")
            sys.exit(1)
        print("\n[CHECK] OPENROUTER_API_KEY environment variable is PRESENT.")
    else:
        print("\n[CHECK] Running in --mock mode (no API key required).")

    synthetic_case = FailureCase(
        case_id="smoke_001",
        task_description="Negation Instruction Adherence Test",
        prompt="Write a review of a restaurant. Do NOT mention food.",
        model_response="The food was delicious and the pizza was outstanding.",
        failure_description="Target model violated negative constraint by explicitly discussing food.",
        expected_behavior="Target model describes atmosphere, service, or decor without food references."
    )

    print(f"\nSynthetic Case: {synthetic_case.case_id}")
    print(f"Prompt: {synthetic_case.prompt}")
    print(f"Observed Output: {synthetic_case.model_response}")

    print(f"\nInstantiating HypothesisGeneratorAgent (model='{DEFAULT_INVESTIGATOR_MODEL}', mock={args.mock})...")
    agent = HypothesisGeneratorAgent(model_name=DEFAULT_INVESTIGATOR_MODEL, mock=args.mock)

    print("\nExecuting single structured hypothesis generation request...")
    record = agent.generate_hypotheses(synthetic_case)

    print("\n[SUCCESS] Structured output received successfully!")
    print(f"  Used Model: {record.model_name}")
    print(f"  Hypotheses Generated: {len(record.hypothesis_set.hypotheses)}")
    for idx, h in enumerate(record.hypothesis_set.hypotheses, 1):
        print(f"\n  Hypothesis #{idx}:")
        print(f"    Claim      : {h.claim}")
        print(f"    Mechanism  : {h.mechanism_guess}")
        print(f"    Confidence : {h.confidence}")

    print("\n" + "=" * 70)
    print("  SMOKE TEST PASSED CLEANLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
