# Project: Causal Mechanistic Investigator

Preliminary AI safety & mechanistic interpretability research prototype investigating:
> *When a target AI produces an unexpected or incorrect behavior, does giving a second AI investigator the ability to perform controlled causal interventions on the target model's internal activations improve its ability to predict how the target model will behave on an unseen, minimally edited version of the original input?*

## 1. Project Specifications

* **Target Model**: `google/gemma-2-2b-it` (Standardized for N≈15 preliminary benchmark).
* **Investigator Model**: Gemini 2.5 Pro / Flash via Google GenAI SDK (`google-genai`).
* **Experimental Conditions**:
  1. **Condition A**: Transcript-Only Baseline (No activation access, no intervention tools).
  2. **Condition B**: Causal-Intervention Investigator (Residual-stream activation patching, controlled sandbox, hard blind-prediction firewall).

---

## 2. Repository Structure

```
project/
├── agents/            # Agent #1 (Hypothesis: agents/hypothesis_agent.py)
├── target/            # Target model loader, inference wrapper, & activation hooks
├── interventions/     # Modular intervention primitives (interventions/patching.py)
├── experiments/       # Case controller & validation experiments (experiments/hypothesis_experiment.py)
├── security/          # Hidden test vault, blind firewall, leakage checker
├── schemas/           # Pydantic schemas (schemas/hypotheses.py, schemas/patching.py)
├── cases/             # Dataset of benchmark cases & variant suites (~15 cases)
├── results/           # Run logs, JSON outputs, & evaluation reports
├── notebooks/         # Google Colab verification & experiment notebooks
├── scripts/           # Execution scripts (Phase 0, Phase 1, Phase 2 verification)
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

Phase 2 implements Agent #1 (`HypothesisGeneratorAgent`), a transcript-only LLM hypothesis generator using Gemini 2.5 Pro via the `google-genai` SDK.

### Features:
- **Strict Epistemic Isolation**: Zero tool access, zero internal activation access, zero hidden test access.
- **Mechanism-Level Specificity**: Rejects vague claims ("model was confused"). Requires hypotheses referring to specific mechanisms (spurious features, instruction persistence, semantic co-occurrence bias).
- **Structured Pydantic Output**: Outputs `HypothesisSet` schema containing 2-3 falsifiable hypotheses and a `most_likely` index.
- **Privacy Audit**: `GEMINI_API_KEY` accessed securely via environment variables; credentials are never printed or saved to logs.

### Running Phase 2:
```bash
# Dry-run mock mode
python scripts/phase2_hypothesis.py --mock

# Real Gemini API execution
python scripts/phase2_hypothesis.py --model gemini-2.5-pro
```

---

## 6. Methodological & Scientific Rules

1. **Scientific Distinction**: Successful activation patching demonstrates causal intervention capability. It does **NOT** constitute a discovered mechanism.
2. **Hard Blind Firewall**: Agent #2 and its intervention sandbox NEVER receive access to hidden test prompts, hidden outputs, or hidden activations until the prediction is explicitly frozen.
3. **Reproducibility**: All baseline generation calls run with `temperature=0`, fixed random seeds, and explicit token alignment logging.
