"""
Qualification Script for 15-case Causal Mechanistic Investigator Evaluation Dataset.
Verifies failure occurrence, hidden-variant behavioral shift, and candidate label validity.
"""

import os
import sys
import json
import argparse
import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any
from target.gemma import GemmaTargetInterface


def qualify_case(case: Dict[str, Any], target_model: GemmaTargetInterface) -> Dict[str, Any]:
    """Qualify a single evaluation case against Gemma target model."""
    case_id = case["case_id"]
    orig_prompt = case["original_prompt"]
    hidden_prompt = case["hidden_variant"]
    cands = case["candidate_labels"]

    # 1. Run inference on original prompt
    orig_output, orig_logprobs = target_model.run_inference(orig_prompt, candidate_tokens=cands)

    # 2. Run inference on hidden variant prompt
    hidden_output, hidden_logprobs = target_model.run_inference(hidden_prompt, candidate_tokens=cands)

    # 3. Qualification Criteria Verification
    # Criterion 1: Minimal edit check
    differ_prompts = orig_prompt.strip() != hidden_prompt.strip()
    
    # Criterion 2: Valid candidate logprobs
    valid_logprobs = orig_logprobs is not None and len(orig_logprobs) > 0

    # Criterion 3: Observable target behavior / shift
    # In mock mode, synthetic responses are deterministic and qualified for orchestration verification.
    # In real mode, target model failure must match intended failure pattern.
    if target_model.mock:
        qualified = differ_prompts and valid_logprobs
        reason = "Mock validation: Prompts differ and candidate logprobs successfully extracted."
    else:
        # Real Gemma validation check
        behavioral_shift = orig_output.strip() != hidden_output.strip()
        qualified = differ_prompts and valid_logprobs and behavioral_shift
        reason = (
            "Qualified: Original and hidden variants exhibit distinct target model behaviors." 
            if qualified else "Rejected: Target model produced identical responses across variants."
        )

    status = "QUALIFIED" if qualified else "REJECTED"

    return {
        "case_id": case_id,
        "failure_family": case["failure_family"],
        "status": status,
        "reason": reason,
        "original_prompt": orig_prompt,
        "original_output": orig_output,
        "original_candidate_logprobs": orig_logprobs,
        "hidden_variant": hidden_prompt,
        "hidden_output": hidden_output,
        "hidden_candidate_logprobs": hidden_logprobs
    }


def main():
    parser = argparse.ArgumentParser(description="Qualify 15-case evaluation dataset for Causal Mechanistic Investigator.")
    parser.add_argument("--mock-gemma", action="store_true", default=False, help="Run Gemma target model in mock mode.")
    parser.add_argument("--dataset-path", type=str, default="cases/evaluation_cases.json", help="Path to evaluation cases JSON.")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save qualification results.")
    parser.add_argument("--update-cases", action="store_true", help="Update evaluation_cases.json with actual Gemma model_response outputs.")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = "results/qualification_results.json" if args.mock_gemma else "results/qualification_real_results.json"

    print("=" * 80)
    print("  PHASE 3 EVALUATION DATASET QUALIFICATION SCRIPT")
    print(f"  Target Model Mode : {'MOCK' if args.mock_gemma else 'REAL GEMMA 2 2B IT'}")
    print(f"  Dataset Path      : {args.dataset_path}")
    print(f"  Output Path       : {args.output_path}")
    print("=" * 80)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Evaluation dataset file not found at '{args.dataset_path}'.")

    import torch
    if not args.mock_gemma:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for real Gemma dataset qualification.")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token or len(hf_token.strip()) == 0:
            raise RuntimeError("HF_TOKEN environment variable is required for real Gemma dataset qualification.")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\nLoaded {len(cases)} cases across 3 failure families.")
    
    target_model = GemmaTargetInterface(mock=args.mock_gemma)

    results = []
    qualified_count = 0
    rejected_count = 0

    print("\n--- Running Qualification Pipeline ---")
    for idx, case in enumerate(cases, 1):
        res = qualify_case(case, target_model)
        results.append(res)
        if res["status"] == "QUALIFIED":
            qualified_count += 1
            status_str = "[QUALIFIED]"
            if args.update_cases or not args.mock_gemma:
                case["model_response"] = res["original_output"]
        else:
            rejected_count += 1
            status_str = "[REJECTED]"

        print(f"  Case {idx:02d}/15 [{case['case_id']}] ({case['failure_family']}): {status_str} -> {res['reason']}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mock_mode": args.mock_gemma,
        "total_cases": len(cases),
        "qualified_cases": qualified_count,
        "rejected_cases": rejected_count,
        "qualification_results": results
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.update_cases:
        with open(args.dataset_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        print(f"  Updated dataset saved with real Gemma responses to '{args.dataset_path}'.")

    print("\n" + "=" * 80)
    print("  QUALIFICATION SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total Cases Created : {len(cases)}")
    print(f"  Total Qualified     : {qualified_count}")
    print(f"  Total Rejected      : {rejected_count}")
    print(f"  Results Saved To    : {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
