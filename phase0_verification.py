import sys
from scripts.phase0_verification import run_phase0_verification, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 Verification for Causal Mechanistic Investigator")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Target model ID")
    parser.add_argument("--mock", action="store_true", help="Run dry-run mock verification without downloading model weights")
    args = parser.parse_args()

    success = run_phase0_verification(target_model_id=args.model, mock=args.mock)
    sys.exit(0 if success else 1)
