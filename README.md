# Project: Causal Mechanistic Investigator

Preliminary AI safety & mechanistic interpretability research prototype investigating:
> *When a target AI produces an unexpected or incorrect behavior, does giving a second AI investigator the ability to perform controlled causal interventions on the target model's internal activations improve its ability to predict how the target model will behave on an unseen, minimally edited version of the original input?*

## 1. Project Specifications

* **Target Model**: `google/gemma-2-2b-it` (Standardized for N≈15 preliminary benchmark).
* **Investigator Model**: `gemini-3.8-flash` via Google GenAI SDK (`google-genai`).
* **Experimental Conditions**:
  1. **Condition A**: Transcript-Only Baseline (No activation access, no intervention tools).
  2. **Condition B**: Causal-Intervention Investigator (Residual-stream activation patching, controlled sandbox, hard blind-prediction firewall).

---

## 2. Repository Structure

```
project/
├── agents/            # Agent #1 (Hypothesis) & Agent #2 (Causal Investigator: agents/causal_investigator.py)
├── target/            # Target model loader (target/gemma.py) & activation hooks
├── interventions/     # Patching engine & sandbox (interventions/sandbox.py, interventions/patching.py)
├── experiments/       # Case controller & validation suites (experiments/investigation_experiment.py)
├── security/          # Blind firewall, leakage checker
├── schemas/           # Pydantic schemas (schemas/investigation.py, schemas/hypotheses.py, schemas/patching.py)
├── cases/             # Dataset of benchmark cases & variant suites (~15 cases)
├── results/           # Run logs, JSON outputs, & evaluation reports (results/phase3a_investigation_log.json)
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

## 6. Phase 3A: Causal Investigator Agent (Agent #2) & Investigation Sandbox

Phase 3A implements Agent #2 (`CausalInvestigatorAgent`) and `InvestigationSandbox`.

### Architecture & Security Controls:
1. **Explicit Tool Boundary**: Agent #2 has **NO** arbitrary python, shell, or file read execution capabilities (`exec`, `eval`, `subprocess`, `os`, `open` are NOT exposed).
2. **Exposed Tools Only**:
   - `run_target(prompt)`
   - `capture_activation(prompt, layer_idx, position)`
   - `patch_activation(source_prompt, target_prompt, layer_idx, source_pos, target_pos)`
   - `ablate_activation(prompt, layer_idx, position)`
   - `compare_outputs(baseline_text, intervened_text, candidate_label)`
3. **Investigation Budget**: Enforces strict budgets (`max_experiments=5`, `max_target_calls=15`, `max_interventions=8`). Halts cleanly when budget is exhausted.
4. **Causal vs Observational Evidence**: Distinguishes observational activation correlation from causal intervention evidence (residual stream patching and zero-ablation). Updates hypothesis statuses to `supported`, `weakened`, or `unresolved` without declaring hypotheses "proven".
5. **Epistemic Isolation Boundary**: ZERO hidden-test prompts, outcomes, or metadata accessible to Agent #2.
6. **PHASE 3B STATUS**: **NOT IMPLEMENTED YET** (Hidden-test vault, blind prediction, freezing, and scoring belong to future phases).

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
