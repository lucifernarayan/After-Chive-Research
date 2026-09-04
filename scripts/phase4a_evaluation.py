"""
Phase 4A Execution Script: Real 15-Case Blind Comparative Evaluation.

Evaluates Agent #1 (Transcript-Only Baseline) vs Agent #2 (Causal Investigator with Sandbox)
on the frozen 15-case dataset across 3 failure families:
1. Negation / Instruction-Binding Failures (5 cases)
2. Factual / Entity-Substitution Failures (5 cases)
3. Output-Format / Constraint Failures (5 cases)

Target Model      : google/gemma-2-2b-it
Investigator Model: gemini-3.8-flash
"""

import sys
import os
import time
import json
import argparse
import datetime
from typing import Dict, List, Any
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from target.gemma import GemmaTargetInterface
from agents.hypothesis_agent import HypothesisGeneratorAgent
from agents.causal_investigator import CausalInvestigatorAgent
from interventions.sandbox import InvestigationSandbox, InvestigationBudget
from security.hidden_vault import HiddenTestVault
from schemas.hypotheses import FailureCase


def run_phase4a_evaluation(
    mock_gemma: bool = False,
    mock_gemini: bool = False,
    cases_path: str = "cases/evaluation_cases.json",
    results_path: str = "results/phase4a_results.json",
    summary_path: str = "results/phase4a_summary.json"
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

    device = "cuda" if torch.cuda.is_available() and not mock_gemma else "cpu"
    api_key = os.environ.get("GEMINI_API_KEY")
    key_present = api_key is not None and len(api_key.strip()) > 0

    print(f"  Target Model      : google/gemma-2-2b-it (Mode: {'MOCK' if mock_gemma else 'FULL HF TRANSFORMERS'})")
    print(f"  Investigator Model: gemini-3.8-flash (Mode: {'MOCK' if mock_gemini else 'REAL GEMINI API'})")
    print(f"  Compute Device    : {device}")
    print(f"  GEMINI_API_KEY    : {'PRESENT' if key_present else 'NOT SET'}")

    if not key_present and not mock_gemini:
        print("  [NOTICE] GEMINI_API_KEY not set. Defaulting Agent #1 & Agent #2 to mock mode.")
        mock_gemini = True

    # 1. Instantiate Gemma Target Interface
    target_interface = GemmaTargetInterface(model_id="google/gemma-2-2b-it", mock=mock_gemma)

    # 2. Instantiate Investigators
    agent1 = HypothesisGeneratorAgent(model_name="gemini-3.8-flash", mock=mock_gemini)
    agent2 = CausalInvestigatorAgent(model_name="gemini-3.8-flash", mock=mock_gemini)

    # 3. Instantiate Hidden Test Vault (Security Firewall)
    vault = HiddenTestVault()

    case_records = []
    
    # Tracking accumulators
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
    brier_scores_a1 = []
    brier_scores_a2 = []

    print("\n" + "=" * 80)
    print("  BEGINNING 15-CASE EVALUATION LOOP")
    print("=" * 80)

    for idx, spec in enumerate(case_specs, 1):
        case_start = time.time()
        c_id = spec["case_id"]
        family = spec["failure_family"]
        cands = spec["candidate_labels"]
        
        print(f"\n[{idx:02d}/15] Case '{c_id}' ({family})...")

        # Step A: Run target model on original prompt to establish model_response
        orig_resp, orig_logprobs = target_interface.run_inference(spec["original_prompt"], candidate_tokens=cands)

        # Construct FailureCase schema
        failure_case = FailureCase(
            case_id=c_id,
            task_description=spec["task_description"],
            prompt=spec["original_prompt"],
            model_response=orig_resp,
            failure_description=spec["failure_description"],
            expected_behavior=spec["expected_original_behavior"]
        )

        # Step B: Register hidden variant in Security Vault (Fixtures remain isolated)
        vault.register_fixture(
            case_id=c_id,
            hidden_prompt=spec["hidden_variant"],
            expected_label=spec["expected_hidden_behavior"],
            expected_behavior=spec["expected_hidden_behavior"]
        )

        # Step C: Branch 1 - Agent #1 Transcript-Only Hypotheses & Blind Prediction
        a1_record = agent1.generate_hypotheses(failure_case)
        a1_hypotheses = a1_record.hypothesis_set
        a1_pred = agent1.generate_baseline_prediction(failure_case, a1_hypotheses)

        # Step D: Branch 2 - Agent #2 Causal Investigator Sandbox & Blind Prediction
        budget = InvestigationBudget(max_experiments=5, max_target_calls=15, max_interventions=8)
        sandbox = InvestigationSandbox(target_model=target_interface, budget=budget, case_id=c_id)
        
        # Agent #2 investigation
        a2_investigation_record = agent2.investigate(failure_case, a1_hypotheses, sandbox)
        
        # Agent #2 blind prediction at freeze point
        a2_pred = agent2.generate_blind_prediction(failure_case, a1_hypotheses, sandbox)

        # Step E: Freeze investigation state for case
        vault.freeze_investigation(c_id)

        # Step F: Target model execution on hidden variant prompt
        actual_hidden_text, actual_hidden_logprobs = target_interface.run_inference(spec["hidden_variant"], candidate_tokens=cands)

        # Step G: Reveal hidden outcome & score predictions deterministically
        scoring = vault.reveal_and_evaluate(c_id, a1_pred, a2_pred)

        case_elapsed = time.time() - case_start

        # Calculate Brier Scores
        y_true_a1 = 1.0 if scoring.agent1_correct else 0.0
        y_true_a2 = 1.0 if scoring.agent2_correct else 0.0
        brier_a1 = (a1_pred.confidence - y_true_a1) ** 2
        brier_a2 = (a2_pred.confidence - y_true_a2) ** 2
        brier_scores_a1.append(brier_a1)
        brier_scores_a2.append(brier_a2)

        # Update accumulators
        exps_used = sandbox.budget.experiments_used
        intv_used = sandbox.budget.interventions_used
        total_experiments += exps_used
        total_interventions += intv_used

        if family in family_stats:
            family_stats[family]["total"] += 1
            if scoring.agent1_correct:
                family_stats[family]["agent1_correct"] += 1
            if scoring.agent2_correct:
                family_stats[family]["agent2_correct"] += 1

        if scoring.agent1_correct and scoring.agent2_correct:
            both_correct += 1
        elif scoring.agent2_correct and not scoring.agent1_correct:
            agent2_only += 1
        elif scoring.agent1_correct and not scoring.agent2_correct:
            agent1_only += 1
        else:
            both_incorrect += 1

        print(f"  Agent #1 Prediction: '{a1_pred.predicted_label}' (Conf: {a1_pred.confidence:.2f}) -> Correct: {scoring.agent1_correct}")
        print(f"  Agent #2 Prediction: '{a2_pred.predicted_label}' (Conf: {a2_pred.confidence:.2f}) -> Correct: {scoring.agent2_correct}")
        print(f"  Ground Truth Label : '{scoring.actual_hidden_label}'")
        print(f"  Trajectory Stats   : Exps={exps_used}, Interventions={intv_used}, Time={case_elapsed:.2f}s")

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
        case_records.append(case_record)

    total_elapsed = time.time() - start_total_time
    total_cases = len(case_specs)

    a1_total_correct = sum(1 for c in case_records if c["agent1_correct"])
    a2_total_correct = sum(1 for c in case_records if c["agent2_correct"])

    a1_acc = a1_total_correct / total_cases
    a2_acc = a2_total_correct / total_cases
    delta_acc = a2_acc - a1_acc

    mean_brier_a1 = sum(brier_scores_a1) / total_cases
    mean_brier_a2 = sum(brier_scores_a2) / total_cases

    # Family accuracy metrics
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

    # Aggregate Metrics Dictionary
    aggregate_metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_model_name": "google/gemma-2-2b-it",
        "investigator_model_name": "gemini-3.8-flash",
        "execution_mode": "mock" if (mock_gemma or mock_gemini) else "real",
        "total_cases": total_cases,
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
            "avg_experiments_per_case": round(total_experiments / total_cases, 2)
        },
        "total_elapsed_seconds": round(total_elapsed, 4)
    }

    # Prepare final output structure
    full_output = {
        "aggregate_metrics": aggregate_metrics,
        "cases": case_records
    }

    # Save results to disk
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("  PHASE 4A EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f"  Execution Mode            : {aggregate_metrics['execution_mode'].upper()}")
    print(f"  Total Benchmark Cases     : {total_cases}")
    print(f"  Agent #1 Accuracy (Base)  : {a1_acc * 100:.1f}% ({a1_total_correct}/{total_cases})")
    print(f"  Agent #2 Accuracy (Causal): {a2_acc * 100:.1f}% ({a2_total_correct}/{total_cases})")
    print(f"  Accuracy Delta (Delta Acc) : {delta_acc * 100:+.1f}%")
    print(f"  Contingency Matrix        : Both Correct={both_correct} | Agent2 Only={agent2_only} | Agent1 Only={agent1_only} | Both Incorrect={both_incorrect}")
    print(f"  Brier Score Mean          : Agent1={mean_brier_a1:.4f} | Agent2={mean_brier_a2:.4f}")
    print(f"  Trajectory Totals         : Exps={total_experiments} | Interventions={total_interventions} | Total Time={total_elapsed:.1f}s")
    print("-" * 80)
    print("  ACCURACY BY FAILURE FAMILY:")
    for fam, stats in by_family_summary.items():
        print(f"    - {fam:<25}: Agent1={stats['agent1_accuracy']*100:.0f}% | Agent2={stats['agent2_accuracy']*100:.0f}% | Delta Acc={stats['accuracy_delta']*100:+.0f}%")
    print("=" * 80)
    print(f"  Full Results Saved To     : {results_path}")
    print(f"  Summary Metrics Saved To  : {summary_path}")
    print("=" * 80)

    return aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4A 15-Case Evaluation")
    parser.add_argument("--mock-gemma", action="store_true", help="Run Gemma target model in mock mode")
    parser.add_argument("--mock-gemini", action="store_true", help="Run Agent 1 and Agent 2 in mock mode")
    parser.add_argument("--cases-path", type=str, default="cases/evaluation_cases.json", help="Path to cases JSON")
    parser.add_argument("--results-path", type=str, default="results/phase4a_results.json", help="Path to detailed results JSON")
    parser.add_argument("--summary-path", type=str, default="results/phase4a_summary.json", help="Path to summary JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.mock_gemma:
        print("  [NOTICE] No CUDA GPU detected locally. Setting --mock-gemma for mock validation.")
        args.mock_gemma = True

    run_phase4a_evaluation(
        mock_gemma=args.mock_gemma,
        mock_gemini=args.mock_gemini,
        cases_path=args.cases_path,
        results_path=args.results_path,
        summary_path=args.summary_path
    )
