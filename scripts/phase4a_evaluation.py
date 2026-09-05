"""
Phase 4A Execution Script: Real 15-Case Blind Comparative Evaluation.

Evaluates Agent #1 (Transcript-Only Baseline) vs Agent #2 (Causal Investigator with Sandbox)
on the frozen 15-case dataset across 3 failure families:
1. Negation / Instruction-Binding Failures (5 cases)
2. Factual / Entity-Substitution Failures (5 cases)
3. Output-Format / Constraint Failures (5 cases)

Target Model      : google/gemma-2-2b-it
Investigator Model: gemini-3.8-flash

Features:
- Atomic per-case persistence (saves result to disk immediately after each completed case).
- Automatic resume support (skips completed case_ids from existing results_path).
- Bounded API error handling (clean termination on rate limits / retries exhausted).
"""

import sys
import os
import time
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from target.gemma import GemmaTargetInterface
from agents.hypothesis_agent import HypothesisGeneratorAgent, GeminiRateLimitError, GeminiUnavailableError
from agents.causal_investigator import CausalInvestigatorAgent
from interventions.sandbox import InvestigationSandbox, InvestigationBudget
from security.hidden_vault import HiddenTestVault
from schemas.hypotheses import FailureCase


def _save_checkpoint(
    case_records: List[Dict[str, Any]], 
    case_specs: List[Dict[str, Any]], 
    results_path: str, 
    summary_path: str, 
    is_mock: bool, 
    start_total_time: float
) -> Dict[str, Any]:
    """Helper function to calculate aggregate metrics and save atomic checkpoint to disk."""
    total_cases = len(case_specs)
    completed_cases = len(case_records)
    total_elapsed = time.time() - start_total_time

    if completed_cases == 0:
        return {}

    a1_total_correct = sum(1 for c in case_records if c.get("agent1_correct", False))
    a2_total_correct = sum(1 for c in case_records if c.get("agent2_correct", False))

    a1_acc = a1_total_correct / completed_cases
    a2_acc = a2_total_correct / completed_cases
    delta_acc = a2_acc - a1_acc

    brier_scores_a1 = [c.get("brier_score_agent1", 0.0) for c in case_records]
    brier_scores_a2 = [c.get("brier_score_agent2", 0.0) for c in case_records]
    mean_brier_a1 = sum(brier_scores_a1) / completed_cases
    mean_brier_a2 = sum(brier_scores_a2) / completed_cases

    family_stats = {
        "negation_instruction": {"agent1_correct": 0, "agent2_correct": 0, "total": 0},
        "factual_substitution": {"agent1_correct": 0, "agent2_correct": 0, "total": 0},
        "format_constraint": {"agent1_correct": 0, "agent2_correct": 0, "total": 0}
    }

    both_correct = 0
    agent2_only = 0
    agent1_only = 0
    both_incorrect = 0
    total_experiments = 0
    total_interventions = 0

    for c in case_records:
        fam = c.get("failure_family")
        if fam in family_stats:
            family_stats[fam]["total"] += 1
            if c.get("agent1_correct"):
                family_stats[fam]["agent1_correct"] += 1
            if c.get("agent2_correct"):
                family_stats[fam]["agent2_correct"] += 1

        a1_c = c.get("agent1_correct", False)
        a2_c = c.get("agent2_correct", False)
        if a1_c and a2_c:
            both_correct += 1
        elif a2_c and not a1_c:
            agent2_only += 1
        elif a1_c and not a2_c:
            agent1_only += 1
        else:
            both_incorrect += 1

        total_experiments += c.get("experiment_count", 0)
        total_interventions += c.get("intervention_count", 0)

    by_family_summary = {}
    for fam, stats in family_stats.items():
        tot = stats["total"]
        if tot > 0:
            f_a1_acc = stats["agent1_correct"] / tot
            f_a2_acc = stats["agent2_correct"] / tot
            f_delta = f_a2_acc - f_a1_acc
        else:
            f_a1_acc, f_a2_acc, f_delta = 0.0, 0.0, 0.0
        
        by_family_summary[fam] = {
            "total_cases": tot,
            "agent1_correct": stats["agent1_correct"],
            "agent2_correct": stats["agent2_correct"],
            "agent1_accuracy": round(f_a1_acc, 4),
            "agent2_accuracy": round(f_a2_acc, 4),
            "accuracy_delta": round(f_delta, 4)
        }

    aggregate_metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_model_name": "google/gemma-2-2b-it",
        "investigator_model_name": "gemini-3.8-flash",
        "execution_mode": "mock" if is_mock else "real",
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "agent1_accuracy": round(a1_acc, 4),
        "agent2_accuracy": round(a2_acc, 4),
        "accuracy_delta": round(delta_acc, 4),
        "contingency_breakdown": {
            "both_correct": both_correct,
            "agent2_only_correct": agent2_only,
            "agent1_only_correct": agent1_only,
            "both_incorrect": both_incorrect
        },
        "brier_score": {
            "agent1_mean": round(mean_brier_a1, 4),
            "agent2_mean": round(mean_brier_a2, 4),
            "brier_delta": round(mean_brier_a2 - mean_brier_a1, 4)
        },
        "accuracy_by_family": by_family_summary,
        "trajectory_totals": {
            "total_experiments_executed": total_experiments,
            "total_interventions_executed": total_interventions,
            "avg_experiments_per_case": round(total_experiments / completed_cases, 2)
        },
        "total_elapsed_seconds": round(total_elapsed, 4)
    }

    full_output = {
        "aggregate_metrics": aggregate_metrics,
        "cases": case_records
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)

    return aggregate_metrics


def _assert_real_target_output(text: str, logprobs: Optional[Dict[str, float]], case_id: str, context: str):
    """Guard against mock output strings or synthetic logprobs during real evaluation."""
    if "[Mock Gemma Response]" in text or "[Ablated Response]" in text:
        raise RuntimeError(f"Mock response string detected in {context} for case '{case_id}': '{text}'")
    if logprobs is not None:
        vals = list(logprobs.values())
        if len(vals) >= 2 and vals[0] == -0.15 and vals[1] == -3.65:
            raise RuntimeError(f"Mock candidate logprobs detected in {context} for case '{case_id}': {logprobs}")


def run_phase4a_evaluation(
    mock_gemma: bool = False,
    mock_gemini: bool = False,
    cases_path: str = "cases/evaluation_cases.json",
    results_path: str = "results/phase4a_real_results.json",
    summary_path: str = "results/phase4a_real_summary.json"
) -> Dict[str, Any]:
    start_total_time = time.time()
    
    print("=" * 80)
    exec_type = "MOCK VALIDATION (Dry-Run Mode)" if (mock_gemma or mock_gemini) else "REAL GEMMA & GEMINI API EVALUATION"
    print("  PROJECT: CAUSAL MECHANISTIC INVESTIGATOR - PHASE 4A COMPARATIVE EVALUATION")
    print(f"  MODE: {exec_type}")
    print(f"  Dataset Path: {cases_path}")
    print("=" * 80)

    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"Evaluation cases dataset file not found at '{cases_path}'.")

    with open(cases_path, "r", encoding="utf-8") as f:
        case_specs = json.load(f)

    print(f"\nLoaded {len(case_specs)} evaluation cases from '{cases_path}'.")

    # Real-mode safety checks & assertions
    cuda_ok = torch.cuda.is_available()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_ok = hf_token is not None and len(hf_token.strip()) > 0
    api_key = os.environ.get("GEMINI_API_KEY")
    gemini_ok = api_key is not None and len(api_key.strip()) > 0

    print("REAL MODE ASSERTIONS:")
    print(f"  CUDA                  : {'PASS' if cuda_ok else 'FAIL'}")
    print(f"  HF_TOKEN              : {'PRESENT' if hf_ok else 'MISSING'}")
    print(f"  GEMINI_API_KEY        : {'PRESENT' if gemini_ok else 'MISSING'}")
    print(f"  Target mock mode      : {mock_gemma}")
    print(f"  Investigator mock mode: {mock_gemini}")

    if not (mock_gemma or mock_gemini):
        if not cuda_ok:
            raise RuntimeError("CUDA is unavailable. Real evaluation requires CUDA GPU.")
        if not hf_ok:
            raise RuntimeError("HF_TOKEN environment variable is missing. Real evaluation requires Hugging Face authentication.")
        if not gemini_ok:
            raise RuntimeError("GEMINI_API_KEY environment variable is missing. Real evaluation requires Gemini API key.")

    # 1. RESUME SUPPORT: Load existing completed cases from results_path if present
    case_records = []
    completed_case_ids = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_mode = existing_data.get("aggregate_metrics", {}).get("execution_mode")
            existing_cases = existing_data.get("cases", [])

            has_mock_responses = any(
                "[Mock Gemma Response]" in str(c.get("actual_target_output", "")) or
                "[Mock Gemma Response]" in str(c.get("input_case", {}).get("model_response", ""))
                for c in existing_cases
            )

            if not (mock_gemma or mock_gemini) and (existing_mode == "mock" or has_mock_responses):
                print(f"  [RESUME REFUSED] Existing results file '{results_path}' is mock-contaminated (execution_mode='{existing_mode}'). Starting fresh real evaluation.")
            else:
                for c in existing_cases:
                    c_id = c.get("case_id")
                    if c_id:
                        case_records.append(c)
                        completed_case_ids.add(c_id)
                print(f"  [RESUME ACTIVE] Found existing '{results_path}' with {len(completed_case_ids)} completed cases.")
        except Exception as e:
            print(f"  [RESUME NOTICE] Could not parse existing results at '{results_path}': {e}. Starting fresh.")

    # Instantiate Target Interface, Agents, and Vault
    target_interface = GemmaTargetInterface(model_id="google/gemma-2-2b-it", mock=mock_gemma)
    agent1 = HypothesisGeneratorAgent(model_name="gemini-3.8-flash", mock=mock_gemini)
    agent2 = CausalInvestigatorAgent(model_name="gemini-3.8-flash", mock=mock_gemini)
    vault = HiddenTestVault()

    if not (mock_gemma or mock_gemini):
        if target_interface.mock:
            raise RuntimeError("Target model interface is in mock mode despite real evaluation request.")
        if agent1.mock or agent2.mock:
            raise RuntimeError("Investigator agent is in mock mode despite real evaluation request.")

    print("\n" + "=" * 80)
    print("  BEGINNING EVALUATION LOOP WITH ATOMIC PERSISTENCE & RESUME SUPPORT")
    print("=" * 80)

    for idx, spec in enumerate(case_specs, 1):
        c_id = spec["case_id"]
        family = spec["failure_family"]
        cands = spec["candidate_labels"]
        
        # Resume Check: Skip if case already completed in previous run
        if c_id in completed_case_ids:
            print(f"\n[{idx:02d}/{len(case_specs)}] Case '{c_id}' ({family}): [SKIPPED - ALREADY COMPLETED]")
            continue

        print(f"\n[{idx:02d}/{len(case_specs)}] Case '{c_id}' ({family}): [EXECUTING]...")

        try:
            case_start = time.time()
            
            # Step A: Target model inference on original prompt
            orig_resp, orig_logprobs = target_interface.run_inference(spec["original_prompt"], candidate_tokens=cands)
            if not (mock_gemma or mock_gemini):
                _assert_real_target_output(orig_resp, orig_logprobs, c_id, "original target inference")

            failure_case = FailureCase(
                case_id=c_id,
                task_description=spec["task_description"],
                prompt=spec["original_prompt"],
                model_response=orig_resp,
                failure_description=spec["failure_description"],
                expected_behavior=spec["expected_original_behavior"]
            )

            # Step B: Register hidden variant in Security Vault
            vault.register_fixture(
                case_id=c_id,
                hidden_prompt=spec["hidden_variant"],
                expected_label=spec["expected_hidden_behavior"],
                expected_behavior=spec["expected_hidden_behavior"]
            )

            # Step C: Branch 1 - Agent #1 Baseline
            a1_record = agent1.generate_hypotheses(failure_case)
            a1_hypotheses = a1_record.hypothesis_set
            a1_pred = agent1.generate_baseline_prediction(failure_case, a1_hypotheses)

            # Step D: Branch 2 - Agent #2 Investigation Sandbox & Blind Prediction
            budget = InvestigationBudget(max_experiments=5, max_target_calls=15, max_interventions=8)
            sandbox = InvestigationSandbox(target_model=target_interface, budget=budget, case_id=c_id)
            a2_investigation_record = agent2.investigate(failure_case, a1_hypotheses, sandbox)
            a2_pred = agent2.generate_blind_prediction(failure_case, a1_hypotheses, sandbox)

            # Step E: Freeze investigation
            vault.freeze_investigation(c_id)

            # Step F: Target execution on hidden variant prompt
            actual_hidden_text, actual_hidden_logprobs = target_interface.run_inference(spec["hidden_variant"], candidate_tokens=cands)
            if not (mock_gemma or mock_gemini):
                _assert_real_target_output(actual_hidden_text, actual_hidden_logprobs, c_id, "hidden target inference")

            # Step G: Reveal & score
            scoring = vault.reveal_and_evaluate(c_id, a1_pred, a2_pred)
            case_elapsed = time.time() - case_start

            # Calculate Brier Scores
            y_true_a1 = 1.0 if scoring.agent1_correct else 0.0
            y_true_a2 = 1.0 if scoring.agent2_correct else 0.0
            brier_a1 = (a1_pred.confidence - y_true_a1) ** 2
            brier_a2 = (a2_pred.confidence - y_true_a2) ** 2

            exps_used = sandbox.budget.experiments_used
            intv_used = sandbox.budget.interventions_used

            case_record = {
                "case_id": c_id,
                "failure_family": family,
                "input_case": failure_case.model_dump(),
                "agent1_hypotheses": [h.model_dump() for h in a1_hypotheses.hypotheses],
                "agent1_prediction": a1_pred.model_dump(),
                "agent1_correct": scoring.agent1_correct,
                "agent1_confidence": scoring.agent1_confidence,
                "agent1_score": scoring.agent1_score,
                "agent2_experiment_trajectory": [e.model_dump() for e in sandbox.record.experiments],
                "agent2_final_evidence_state": [h.model_dump() for h in sandbox.record.hypothesis_evidence],
                "agent2_prediction": a2_pred.model_dump(),
                "agent2_correct": scoring.agent2_correct,
                "agent2_confidence": scoring.agent2_confidence,
                "agent2_score": scoring.agent2_score,
                "hidden_variant": {
                    "hidden_prompt": spec["hidden_variant"],
                    "expected_hidden_behavior": spec["expected_hidden_behavior"],
                    "candidate_labels": cands
                },
                "actual_target_output": actual_hidden_text,
                "actual_target_logprobs": actual_hidden_logprobs,
                "experiment_count": exps_used,
                "intervention_count": intv_used,
                "elapsed_time_seconds": round(case_elapsed, 4),
                "brier_score_agent1": round(brier_a1, 4),
                "brier_score_agent2": round(brier_a2, 4)
            }

            # Step H: ATOMIC PERSISTENCE - Save completed case to disk immediately!
            case_records.append(case_record)
            completed_case_ids.add(c_id)
            
            _save_checkpoint(case_records, case_specs, results_path, summary_path, mock_gemma or mock_gemini, start_total_time)
            
            print(f"  Agent #1 Prediction: '{a1_pred.predicted_label}' (Conf: {a1_pred.confidence:.2f}) -> Correct: {scoring.agent1_correct}")
            print(f"  Agent #2 Prediction: '{a2_pred.predicted_label}' (Conf: {a2_pred.confidence:.2f}) -> Correct: {scoring.agent2_correct}")
            print(f"  Ground Truth Label : '{scoring.actual_hidden_label}'")
            print(f"  [PERSISTED ATOMICALLY] Case '{c_id}' saved to disk. Total completed: {len(case_records)}/{len(case_specs)}.")

        except (GeminiRateLimitError, GeminiUnavailableError) as e:
            print(f"\n  [API TERMINATION] Case '{c_id}' interrupted by API restriction: {e}")
            print(f"  [CLEAN TERMINATION] Preserving {len(case_records)} completed cases to '{results_path}'.")
            break
        except KeyboardInterrupt:
            print(f"\n  [USER INTERRUPT] Evaluation paused manually on case '{c_id}'. Preserving {len(case_records)} completed cases.")
            break
        except Exception as e:
            print(f"\n  [EXECUTION ERROR] Case '{c_id}' failed: {e}. Preserving {len(case_records)} completed cases.")
            break

    # Save final aggregate metrics
    final_metrics = _save_checkpoint(case_records, case_specs, results_path, summary_path, mock_gemma or mock_gemini, start_total_time)

    print("\n" + "=" * 80)
    print("  PHASE 4A EVALUATION SUMMARY REPORT")
    print("=" * 80)
    if final_metrics:
        print(f"  Execution Mode            : {final_metrics['execution_mode'].upper()}")
        print(f"  Completed / Total Cases   : {final_metrics['completed_cases']} / {final_metrics['total_cases']}")
        print(f"  Agent #1 Accuracy (Base)  : {final_metrics['agent1_accuracy'] * 100:.1f}%")
        print(f"  Agent #2 Accuracy (Causal): {final_metrics['agent2_accuracy'] * 100:.1f}%")
        print(f"  Accuracy Delta (Delta Acc) : {final_metrics['accuracy_delta'] * 100:+.1f}%")
        print(f"  Full Results Saved To     : {results_path}")
        print(f"  Summary Metrics Saved To  : {summary_path}")
    else:
        print("  No cases were completed in this run.")
    print("=" * 80)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4A 15-Case Evaluation")
    parser.add_argument("--mock-gemma", action="store_true", help="Run Gemma target model in mock mode")
    parser.add_argument("--mock-gemini", action="store_true", help="Run Agent 1 and Agent 2 in mock mode")
    parser.add_argument("--cases-path", type=str, default="cases/evaluation_cases.json", help="Path to cases JSON")
    parser.add_argument("--results-path", type=str, default="results/phase4a_real_results.json", help="Path to detailed results JSON")
    parser.add_argument("--summary-path", type=str, default="results/phase4a_real_summary.json", help="Path to summary JSON")
    args = parser.parse_args()

    run_phase4a_evaluation(
        mock_gemma=args.mock_gemma,
        mock_gemini=args.mock_gemini,
        cases_path=args.cases_path,
        results_path=args.results_path,
        summary_path=args.summary_path
    )
