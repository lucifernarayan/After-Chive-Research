"""
Phase 2A CLI Script: Real Gemma Instrument Validation Protocol.
Executes controlled 11-stage instrument validation on target model google/gemma-2-2b-it.

Usage:
  python scripts/phase2_instrument_validation.py [--model google/gemma-2-2b-it] [--mock] [--output-dir results/phase2]
"""

import sys
import os
import argparse
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.phase2_instrument_validation import Phase2InstrumentValidator
from target.gemma import GemmaTargetInterface


def main():
    parser = argparse.ArgumentParser(description="Phase 2A Real Gemma Instrument Validation Protocol")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Target model identifier")
    parser.add_argument("--mock", action="store_true", help="Run in dry-run mock mode without downloading model weights")
    parser.add_argument("--output-dir", type=str, default="results/phase2", help="Directory to save JSON & Markdown validation outputs")
    args = parser.parse_args()

    print("=" * 80)
    print("  PHASE 2A — REAL GEMMA INSTRUMENT VALIDATION LAUNCHER")
    print("=" * 80)
    print(f"  Target Model : {args.model}")
    print(f"  Mock Mode    : {args.mock}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  CUDA Avail   : {torch.cuda.is_available()}")
    print("=" * 80)

    # Initialize target model interface
    target = GemmaTargetInterface(model_id=args.model, mock=args.mock)
    validator = Phase2InstrumentValidator(target_model=target, mock=args.mock, output_dir=args.output_dir)

    results = validator.execute_validation()
    overall_status = results.get("test_results", {}).get("test_1_residual_patching", {}).get("status") == "PASS"

    sys.exit(0 if overall_status else 1)


if __name__ == "__main__":
    main()
