"""
Central Configuration Constants for Causal Mechanistic Investigator.
Single source of truth for model identifiers and Phase 4A evaluation result paths.
"""

DEFAULT_TARGET_MODEL = "google/gemma-2-2b-it"
DEFAULT_INVESTIGATOR_PROVIDER = "openrouter"
DEFAULT_INVESTIGATOR_MODEL = "openai/gpt-5.6-luna"
MAX_EXPERIMENTS_PER_CASE = 25

# Phase 4A OpenRouter GPT-5.6 Luna Evaluation Result Paths
PHASE4A_RESULTS_PATH = "results/phase4a_openrouter_luna_real_results.json"
PHASE4A_SUMMARY_PATH = "results/phase4a_openrouter_luna_real_summary.json"

