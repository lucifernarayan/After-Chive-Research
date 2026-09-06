"""
Qualification Script for 15-case Causal Mechanistic Investigator Evaluation Dataset.
Verifies original failure occurrence, hidden-variant behavioral shift, and candidate label validity on real Gemma 2 2B IT.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from target.gemma import GemmaTargetInterface


def check_original_failure(case: Dict[str, Any], orig_output: str) -> Tuple[bool, str]:
    """
    Verifies whether Gemma's actual response on original_prompt constitutes a genuine failure.
    Returns (original_failure_verified: bool, reason: str).
    """
    case_id = case["case_id"]
    family = case.get("failure_family", "")
    orig_clean = orig_output.strip()
    expected = case.get("expected_original_behavior", "").strip()

    if family == "negation_instruction":
        if case_id == "neg_001":
            if "rome" in orig_clean.lower() and "paris" not in orig_clean.lower():
                return False, "Model correctly answered 'Rome' adhering to negation constraint; original failure NOT reproduced."
            return True, f"Model outputted '{orig_clean}' failing negation constraint against France."
        elif case_id == "neg_002":
            if "strawberry" in orig_clean.lower() and "apple" not in orig_clean.lower():
                return False, "Model correctly answered 'Strawberry' adhering to negative constraint; original failure NOT reproduced."
            return True, f"Model outputted '{orig_clean}' mentioning forbidden fruit 'apple'."
        elif case_id == "neg_003":
            if orig_clean.lower().startswith("no"):
                return False, "Model correctly outputted 'No' adhering to restriction; original failure NOT reproduced."
            return True, f"Model outputted '{orig_clean}' violating 'No' restriction."
        elif case_id == "neg_004":
            if "sol" in orig_clean.lower() and "sun" not in orig_clean.lower():
                return False, "Model correctly answered 'Sol' adhering to negative constraint; original failure NOT reproduced."
            return True, f"Model outputted '{orig_clean}' mentioning forbidden word 'sun'."
        elif case_id == "neg_005":
            if "e" not in orig_clean.lower() and len(orig_clean) > 0:
                return False, "Model correctly avoided letter 'e'; original failure NOT reproduced."
            return True, f"Model outputted '{orig_clean}' which contains forbidden letter 'e'."
        else:
            if expected.lower() in orig_clean.lower():
                return False, f"Model outputted compliant behavior '{expected}'; failure NOT verified."
            return True, f"Model outputted '{orig_clean}', failing expected behavior '{expected}'."

    elif family == "factual_substitution":
        if expected.lower() in orig_clean.lower():
            return False, f"Model correctly followed contextual premise '{expected}'; original failure NOT reproduced."
        return True, f"Model outputted '{orig_clean}' instead of contextually substituted answer '{expected}'."

    elif family == "format_constraint":
        if case_id == "fmt_001":
            if orig_clean.isupper() and "YES" in orig_clean:
                return False, "Model correctly responded in ALL CAPS 'YES'; failure NOT verified."
            return True, f"Model outputted '{orig_clean}' violating ALL CAPS constraint."
        elif case_id == "fmt_002":
            try:
                data = json.loads(orig_clean)
                if isinstance(data, dict) and "result" in data:
                    return False, "Model correctly returned valid JSON object; failure NOT verified."
            except Exception:
                pass
            return True, f"Model outputted '{orig_clean}' violating JSON object constraint."
        elif case_id == "fmt_003":
            import string
            has_punct = any(ch in string.punctuation for ch in orig_clean)
            if not has_punct:
                return False, "Model correctly outputted answer without punctuation; failure NOT verified."
            return True, f"Model outputted '{orig_clean}' containing punctuation."
        elif case_id == "fmt_004":
            if orig_clean.startswith("CONFIRMED:"):
                return False, "Model correctly started with prefix 'CONFIRMED:'; failure NOT verified."
            return True, f"Model outputted '{orig_clean}' missing required prefix 'CONFIRMED:'."
        elif case_id == "fmt_005":
            if orig_clean.startswith("[") and orig_clean.endswith("]"):
                return False, "Model correctly enclosed answer in square brackets; failure NOT verified."
            return True, f"Model outputted '{orig_clean}' missing square brackets."
        else:
            if expected in orig_clean:
                return False, f"Model outputted expected format '{expected}'; failure NOT verified."
            return True, f"Model outputted '{orig_clean}', failing format constraint '{expected}'."

    else:
        if expected.lower() in orig_clean.lower():
            return False, f"Model outputted compliant answer '{expected}'; failure NOT verified."
        return True, f"Model outputted '{orig_clean}', failing expected answer '{expected}'."


def check_hidden_distinct(orig_output: str, hidden_output: str) -> Tuple[bool, str]:
    """
    Verifies whether hidden_variant prompt output differs meaningfully from original prompt output.
    Returns (hidden_behavior_distinct: bool, reason: str).
    """
    orig_clean = orig_output.strip().lower()
    hidden_clean = hidden_output.strip().lower()

    if orig_clean == hidden_clean:
        return False, f"Target model produced identical outputs ('{orig_output.strip()}') across original and hidden variants."
    
    from interventions.sandbox import calculate_string_delta
    delta = calculate_string_delta(orig_clean, hidden_clean)

    if delta < 1.0:
        return False, f"Target model outputs across variants ('{orig_output.strip()}' vs '{hidden_output.strip()}') differ only trivially."

    return True, f"Target model outputs exhibit meaningful behavioral difference ('{orig_output.strip()}' vs '{hidden_output.strip()}')."


def qualify_case(case: Dict[str, Any], target_model: GemmaTargetInterface) -> Dict[str, Any]:
    """Qualify a single evaluation case against Gemma target model."""
    case_id = case["case_id"]
    orig_prompt = case["original_prompt"]
    hidden_prompt = case["hidden_variant"]
    cands = case.get("candidate_labels", [])
    expected_orig = case.get("expected_original_behavior", "")
    expected_hidden = case.get("expected_hidden_behavior", "")

    # 1. Run inference on original prompt
    orig_output, orig_logprobs = target_model.run_inference(orig_prompt, candidate_tokens=cands)

    # 2. Run inference on hidden variant prompt
    hidden_output, hidden_logprobs = target_model.run_inference(hidden_prompt, candidate_tokens=cands)

    # 3. Qualification Criteria Verification
    differ_prompts = orig_prompt.strip() != hidden_prompt.strip()
    valid_logprobs = orig_logprobs is not None and len(orig_logprobs) > 0

    if target_model.mock:
        orig_fail_verified = True
        hidden_distinct = differ_prompts and valid_logprobs
        qualified = orig_fail_verified and hidden_distinct
        reason = "Mock validation: Prompts differ and candidate logprobs successfully extracted."
    else:
        orig_fail_verified, fail_reason = check_original_failure(case, orig_output)
        hidden_distinct, distinct_reason = check_hidden_distinct(orig_output, hidden_output)
        qualified = differ_prompts and valid_logprobs and orig_fail_verified and hidden_distinct

        if qualified:
            reason = "Qualified: Original failure verified and hidden variant output exhibits distinct target model behavior."
        elif not orig_fail_verified:
            reason = f"Rejected: {fail_reason}"
        elif not hidden_distinct:
            reason = f"Rejected: {distinct_reason}"
        else:
            reason = "Rejected: Candidate logprobs invalid or prompts do not differ."

    status = "QUALIFIED" if qualified else "REJECTED"

    return {
        "case_id": case_id,
        "failure_family": case["failure_family"],
        "status": status,
        "original_prompt": orig_prompt,
        "hidden_prompt": hidden_prompt,
        "expected_original_behavior": expected_orig,
        "actual_original_behavior": orig_output.strip(),
        "expected_hidden_behavior": expected_hidden,
        "actual_hidden_behavior": hidden_output.strip(),
        "original_failure_verified": orig_fail_verified,
        "hidden_behavior_distinct": hidden_distinct,
        "case_qualified": qualified,
        "qualification_reason": reason,
        "original_candidate_logprobs": orig_logprobs,
        "hidden_candidate_logprobs": hidden_logprobs
    }


def main():
    parser = argparse.ArgumentParser(description="Qualify 15-case evaluation dataset for Causal Mechanistic Investigator.")
    parser.add_argument("--mock-gemma", action="store_true", default=False, help="Run Gemma target model in mock mode.")
    parser.add_argument("--dataset-path", type=str, default="cases/evaluation_cases.json", help="Path to evaluation cases JSON.")
    parser.add_argument("--results-path", type=str, default=None, help="Path to save detailed qualification case results.")
    parser.add_argument("--summary-path", type=str, default=None, help="Path to save qualification summary JSON.")
    parser.add_argument("--output-path", type=str, default=None, help="Legacy alias for results-path.")
    parser.add_argument("--update-cases", action="store_true", help="Update evaluation_cases.json with actual Gemma model_response outputs.")
    args = parser.parse_args()

    results_path = args.results_path or args.output_path
    summary_path = args.summary_path

    if results_path is None:
        results_path = "results/qualification_results.json" if args.mock_gemma else "results/qualification_real_v2_results.json"
    if summary_path is None:
        summary_path = "results/qualification_summary.json" if args.mock_gemma else "results/qualification_real_v2_summary.json"

    print("=" * 80)
    print("  PHASE 3 EVALUATION DATASET QUALIFICATION SCRIPT")
    print(f"  Target Model Mode : {'MOCK' if args.mock_gemma else 'REAL GEMMA 2 2B IT'}")
    print(f"  Dataset Path      : {args.dataset_path}")
    print(f"  Results Path      : {results_path}")
    print(f"  Summary Path      : {summary_path}")
    print("=" * 80)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Evaluation dataset file not found at '{args.dataset_path}'.")

    import torch
    if not args.mock_gemma:
        if not torch.cuda.is_available():
            print("  [NOTICE] CUDA GPU not detected. Running Real Gemma 2 2B IT model on CPU...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token or len(hf_token.strip()) == 0:
            print("  [NOTICE] HF_TOKEN environment variable not set. Attempting Hugging Face model load...")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\nLoaded {len(cases)} cases across 3 failure families.")
    
    target_model = GemmaTargetInterface(mock=args.mock_gemma)

    case_records = []
    qualified_count = 0
    rejected_count = 0
    orig_fail_count = 0
    hidden_distinct_count = 0

    print("\n--- Running Qualification Pipeline ---")
    for idx, case in enumerate(cases, 1):
        res = qualify_case(case, target_model)
        case_records.append(res)
        
        if res["original_failure_verified"]:
            orig_fail_count += 1
        if res["hidden_behavior_distinct"]:
            hidden_distinct_count += 1

        if res["case_qualified"]:
            qualified_count += 1
            status_str = "[QUALIFIED]"
            if args.update_cases or not args.mock_gemma:
                case["model_response"] = res["actual_original_behavior"]
        else:
            rejected_count += 1
            status_str = "[REJECTED]"

        print(f"  Case {idx:02d}/15 [{case['case_id']}] ({case['failure_family']}): {status_str} -> {res['qualification_reason']}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    summary_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_model_name": target_model.model_id,
        "mock_mode": args.mock_gemma,
        "total_cases": len(cases),
        "original_failures_verified": orig_fail_count,
        "hidden_behavior_distinct": hidden_distinct_count,
        "fully_qualified_cases": qualified_count,
        "rejected_cases": rejected_count,
        "qualification_results": case_records
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(case_records, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    if args.update_cases:
        with open(args.dataset_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        print(f"  Updated dataset saved with real Gemma responses to '{args.dataset_path}'.")

    print("\n" + "=" * 80)
    print("  QUALIFICATION SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total Cases Evaluated        : {len(cases)}")
    print(f"  Original Failures Verified  : {orig_fail_count}")
    print(f"  Hidden Behaviors Distinct   : {hidden_distinct_count}")
    print(f"  Fully Qualified Cases       : {qualified_count}")
    print(f"  Rejected Cases              : {rejected_count}")
    print(f"  Detailed Results Saved To   : {results_path}")
    print(f"  Summary Report Saved To     : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
