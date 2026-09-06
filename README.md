# Project: Causal Mechanistic Investigator

Preliminary AI safety & mechanistic interpretability research prototype investigating:
> *When a target AI produces an unexpected or incorrect behavior, does giving a second AI investigator the ability to perform controlled causal interventions on the target model's internal activations improve its ability to predict how the target model will behave on an unseen, minimally edited version of the original input?*

## 1. Project Specifications

* **Target Model**: `google/gemma-2-2b-it` (Standardized for N≈15 preliminary benchmark).
* **Investigator Model**: `gemini-3.7-flash` via Google GenAI SDK (`google-genai`).
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

### 5. Phase 2: Transcript-Only Gemini Hypothesis Agent (Agent #1)

Phase 2 implements Agent #1 (`HypothesisGeneratorAgent`), a transcript-only LLM hypothesis generator using `gemini-3.7-flash` via the `google-genai` SDK.

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

---

## 8. 15-Case Evaluation Dataset & Qualification Pipeline

A benchmark dataset of 15 concrete cases (`cases/evaluation_cases.json`) designed to measure whether causal intervention tools improve LLM behavior prediction on unseen prompt variants.

### Failure Families (5 cases each):
1. **Negation / Instruction-Binding Failures**: Negative constraints (e.g. "Do not mention X", "Without using letter Y") where target model attention/residual representations fail to enforce negative instructions.
2. **Factual Entity-Substitution Failures**: Counterfactual or alternate history premises (e.g. "Capital of Australia is Sydney") where target model defaults to pre-trained parametric associations.
3. **Output-Format / Constraint Failures**: Enclosure or formatting rules (e.g. ALL CAPS ONLY, JSON object formatting, zero punctuation) where target model violates structural output constraints.

### Hidden-Variant Design:
Each case pairs a visible failure prompt with a minimally edited hidden variant prompt. The hidden variant introduces a controlled perturbation (e.g. swapping target entities or constraints) to measure whether Agent #2's causal residual interventions allow predicting behavioral outcomes on unseen variants better than Agent #1's transcript reasoning alone.

### Qualification Criteria:
A case is classified as `QUALIFIED` via `scripts/qualify_cases.py` if and only if:
1. **Target Failure**: The target model actually exhibits the intended failure pattern on the original prompt.
2. **Measurable Shift**: The hidden variant produces a distinct, measurable behavioral outcome relative to the original prompt.
3. **Candidate Token Validity**: Candidate labels map to valid next-token log-probabilities.
4. **Minimal Edit**: Original and hidden variant prompts are non-identical but minimally edited.

### Qualification Execution:
```bash
# Run qualification in mock target mode
python scripts/qualify_cases.py --mock-gemma

# Run qualification against real Gemma 2 2B IT target model
python scripts/qualify_cases.py
```

### Primary Evaluation Metrics:
* **Accuracy Delta**: $\Delta Acc = Acc_{\text{Agent2}} - Acc_{\text{Agent1}}$ on frozen blind predictions for hidden variant outcomes.
* **Confidence Calibration**: Mean squared error / Brier score of predicted confidence relative to actual ground truth outcome.

---

## 9. Phase 4A: Real 15-Case Blind Comparative Evaluation

Phase 4A executes the full comparative benchmark across the frozen 15-case dataset (`cases/evaluation_cases.json`) using `gemini-3.7-flash` as the Gemini investigator model.

### API Consumption & Rate-Limit Protocol:
- **Investigator Model**: `gemini-3.7-flash`
- **Expected Requests**: Exactly **45 Gemini API `generate_content` calls** for 15 cases (3 calls/case: Agent 1 hypothesis generation, Agent 1 blind prediction, Agent 2 blind prediction).
- **Free-Tier Constraints**: The benchmark is executed strictly under free-tier API rate limits (15 RPM / 1500 RPD).
- **No Model Fallback**: If rate limits or quota limits occur, execution terminates cleanly with preserved atomic checkpoints. No silent model fallback or automatic model switching is performed.

### Evaluation Pipeline Flow per Case:
1. **Public Input**: Agent #1 and Agent #2 receive identical public case specs (task description, prompt, model response, failure description, expected behavior).
2. **Agent #1 (Transcript Baseline)**: Formulates 2–3 hypotheses and freezes a blind prediction without activation access.
3. **Agent #2 (Causal Investigator)**: Formulates hypotheses and executes bounded activation interventions (`run_target`, `capture`, `patch`, `ablate`) in `InvestigationSandbox`, then freezes a blind prediction before hidden variant revelation.
4. **Blind Firewall Lock**: `HiddenTestVault` locks prediction freeze state.
5. **Target Execution & Scoring**: Target model executes hidden variant prompt, and `HiddenTestVault.reveal_and_evaluate()` deterministically scores both predictions.

### Output Files:
* `results/phase4a_gemini37_real_results.json`: Detailed per-case records (hypotheses, experiment trajectories, predictions, hidden variant outcomes, candidate logprobs, elapsed times).
* `results/phase4a_gemini37_real_summary.json`: Summary metrics report ($\Delta Acc$, family breakdown, contingency breakdown, Brier scores).

### Execution Commands:
```bash
# Run dry-run mock mode validation
python scripts/phase4a_evaluation.py --mock-gemma --mock-gemini

# Run real benchmark evaluation (Requires GPU and GEMINI_API_KEY)
python scripts/phase4a_evaluation.py
```

### Key Scientific Limitations:
1. **Infrastructure vs Mechanistic Proof**: Phase 4A measures whether controlled causal interventions improve blind behavioral generalization ($\Delta Acc$). It does **NOT** prove that Agent #2 discovered exact internal mechanisms (e.g. attention head routing vs MLP channels).
2. **Bounded Budget Constraints**: Agent #2 is capped at 5 experiments per case and performs layer residual manipulations; it does not exhaustively search all attention head combinations or weight matrices.
3. **Epistemic Isolation Firewall**: Both agents are strictly firewalled from hidden variant prompts until after predictions are permanently frozen.

