# Project: Causal Mechanistic Investigator

Preliminary AI safety & mechanistic interpretability research prototype investigating:
> *When a target AI produces an unexpected or incorrect behavior, does giving a second AI investigator the ability to perform controlled causal interventions on the target model's internal activations improve its ability to predict how the target model will behave on an unseen, minimally edited version of the original input?*

## 1. Project Specifications

* **Target Model**: `google/gemma-2-2b-it` (Standardized for N≈15 preliminary benchmark).
* **Investigator Model**: `gemini-3.8-flash` via Google GenAI SDK (`google-genai`).
* **Experimental Conditions**:
  1. **Condition A**: Transcript-Only Baseline (Agent #1: No activation access, no intervention tools).
  2. **Condition B**: Causal-Intervention Investigator (Agent #2: Residual-stream activation patching, controlled sandbox, hard blind-prediction firewall).

---

## 2. Repository Structure

```
project/
├── agents/            # Agent #1 (agents/hypothesis_agent.py) & Agent #2 (agents/causal_investigator.py)
├── target/            # Target model loader (target/gemma.py) & activation hooks
├── interventions/     # Patching engine & sandbox (interventions/sandbox.py, interventions/patching.py)
├── experiments/       # Case controller & validation suites (experiments/investigation_experiment.py)
├── security/          # Blind firewall & hidden test vault (security/hidden_vault.py)
├── schemas/           # Pydantic schemas (schemas/predictions.py, schemas/investigation.py, schemas/hypotheses.py)
├── cases/             # Dataset of benchmark cases & variant suites (~15 cases)
├── results/           # Run logs, JSON outputs, & evaluation reports (results/phase3a_comparative_eval.json)
├── notebooks/         # Google Colab notebooks (notebooks/phase3a_investigator.ipynb)
├── scripts/           # Execution scripts (scripts/phase3a_investigator.py)
├── requirements.txt   # Locked Python dependencies
├── phase0_verification.py # Top-level Phase 0 verification runner
└── README.md          # Project documentation
```

---

## 3. Phase 0 Verification

Phase 0 verifies environment compatibility, PyTorch activation hooks on Gemma 2 2B IT, deterministic inference, and Google Gemini API setup.

```bash
python phase0_verification.py --mock
```

---

## 4. Phase 1: Causal Activation Patching Primitive

Phase 1 implements and validates `ActivationPatchingEngine` using native PyTorch forward hooks on residual stream hidden states (`model.model.layers[layer]`).

```bash
python scripts/phase1_patching.py --mock
```

---

## 5. Phase 2: Transcript-Only Gemini Hypothesis Agent (Agent #1)

Phase 2 implements Agent #1 (`HypothesisGeneratorAgent`), a transcript-only LLM hypothesis generator using `gemini-3.8-flash` via the `google-genai` SDK.

```bash
python scripts/phase2_hypothesis.py --mock
```

---

## 6. Phase 3A: Symmetric Comparative Pipeline & Investigation Sandbox

Phase 3A implements the symmetric comparative prediction and scoring pipeline:

```
                SAME FAILURE CASE
                       |
                Agent #1 hypotheses
                       |
             +---------+---------+
             |                   |
    Transcript-only       Causal investigator
         branch                  |
             |             bounded experiments
             |                   |
      FREEZE PREDICTION    FREEZE PREDICTION
             |                   |
             +---------+---------+
                       |
                SAME HIDDEN VARIANT
                       |
                 SAME SCORER
                       |
          +------------+------------+
          |                         |
    Agent #1 score             Agent #2 score
```

### Architecture & Security Controls:
1. **Symmetric Prediction Schema**: Agent #1 and Agent #2 produce structurally identical `BlindPrediction` objects (`prediction_id`, `case_id`, `investigator_type`, `predicted_behavior`, `predicted_label`, `confidence`, `rationale`, `frozen_at`).
2. **Epistemic Isolation Firewall**: Neither prediction function accepts `hidden_variant` objects. Agent #2 trajectory does NOT leak into Agent #1 context. Hidden test outcomes are revealed ONLY to `HiddenTestVault.reveal_and_evaluate()` after predictions are permanently frozen.
3. **Explicit Tool Boundary**: Agent #2 has **NO** arbitrary python, shell, or file read execution capabilities (`run_target`, `capture_activation`, `patch_activation`, `ablate_activation`, `compare_outputs` only).
4. **Investigation Budget**: Enforces strict budgets (`max_experiments=5`, `max_target_calls=15`, `max_interventions=8`).
5. **Deterministic Scorer Engine**: `HiddenTestVault` deterministically evaluates Agent #1 and Agent #2 predictions against actual hidden variant label (`agent1_correct`, `agent2_correct`, `agent1_score`, `agent2_score`).
6. **PHASE 3B STATUS**: **NOT IMPLEMENTED YET** (Hidden-test dataset scaling and benchmark evaluation belong to future phases).

### Running Phase 3A Validation:
```bash
# Dry-run mock mode
python scripts/phase3a_investigator.py --mock-gemma --mock-gemini

# Full execution (Requires CUDA GPU and GEMINI_API_KEY)
python scripts/phase3a_investigator.py
```

---

## 7. Methodological & Scientific Rules

1. **Scientific Distinction**: Successful activation patching demonstrates causal intervention capability. It does **NOT** constitute a discovered mechanism.
2. **Hard Blind Firewall**: Agent #2 and its intervention sandbox NEVER receive access to hidden test prompts, hidden outputs, or hidden activations until the prediction is explicitly frozen.
3. **Reproducibility**: All baseline generation calls run with `temperature=0`, fixed random seeds, and explicit token alignment logging.
