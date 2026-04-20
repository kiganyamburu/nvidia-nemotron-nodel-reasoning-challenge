# Model-Based Evaluation Workflow Guide

## Overview

This guide walks through the complete transition from TF-IDF proxy evaluation to actual model-generation evaluation using Nemotron-3-Nano-30B.

## Stage 1: TF-IDF Proxy Baseline (Already Complete)

**Purpose:** Fast, low-cost baseline evaluation of prompt strategies without needing a GPU.

**Run:**

```bash
.\.venv\Scripts\python scripts\prompt_engineering.py
```

**Outputs:**

- `outputs/prompt_experiments_summary.csv` - Per-strategy accuracy (5 rows)
- `outputs/prompt_experiments_examples.csv` - Per-example predictions (7,125 rows)

**Key Results:**

- Mathematical problems: ~43% exact-match accuracy
- Other problem types: ~0% exact-match accuracy
- Few-shot strategy raises average similarity to 0.74 but doesn't improve exact-match

---

## Stage 2: Error Analysis (Already Complete)

**Purpose:** Stratify TF-IDF failures by problem type, answer family, and prompt length to understand what works and what doesn't.

**Run:**

```bash
.\.venv\Scripts\python scripts\analyze_prompt_errors.py
```

**Outputs:**

- `outputs/prompt_error_analysis.md` - Human-readable report
- `outputs/prompt_error_analysis_by_problem_type.csv` - Accuracy by problem type
- `outputs/prompt_error_analysis_by_family_and_length.csv` - Accuracy by answer family and prompt length
- `outputs/prompt_error_analysis_hard_cases.csv` - 15 hardest examples

**Key Insights:**

- Roman answers: some recovery with prompt engineering (1.85–2.26 strategies correct)
- Binary, numeric, text answers: near-zero success across all strategies
- Hard cases: mostly Cryptography-type prompts with perfect retrieval similarity but wrong answers

---

## Stage 3: Model-Based Evaluation (New - Your Next Step)

**Purpose:** Run the same 5 prompt strategies through the actual Nemotron model to validate whether prompt engineering improves real generation.

### Option A: Test with GPT-2 (Recommended for Initial Testing)

If you don't have immediate access to a GPU or want to validate the pipeline first:

```bash
.\.venv\Scripts\python scripts\test_generation_pipeline.py
```

This script:

- Loads GPT-2 (tiny, CPU-compatible)
- Tests the generation pipeline on 5 validation examples
- Confirms the code path works before running expensive Nemotron evaluation
- Takes ~30 seconds

**Expected Output:**

```
Testing model generation pipeline with GPT-2...
...
✓ Pipeline test successful!
Ready to run full model_generation_evaluation.py with Nemotron or other target model.
```

### Option B: Full Model Evaluation with Nemotron

Once you're ready for production evaluation on a GPU machine:

```bash
.\.venv\Scripts\python scripts\model_generation_evaluation.py
```

**Requirements:**

- GPU with sufficient VRAM:
  - 60GB+ for bfloat16 (full precision)
  - 15–20GB with 4-bit quantization (default, recommended)
- HuggingFace login (if using gated models):
  ```bash
  huggingface-cli login
  ```
  Then paste your access token from https://huggingface.co/settings/tokens

**What the Script Does:**

1. Loads validation set (1,425 examples)
2. For each strategy (minimal, instruction, cot, answer_format, few_shot):
   - Creates strategy-specific prompts
   - Calls `model.generate()` with temperature=0.0 (deterministic)
   - Normalizes generated answers and compares to gold answers
   - Tracks exact-match accuracy and generation length
3. Saves results to CSVs
4. Prints summary table

**Outputs:**

- `outputs/model_generation_summary.csv` - Per-strategy accuracy from actual model
- `outputs/model_generation_examples.csv` - Per-example predictions (7,125 rows with generated answers)

**Example Output:**

```
================================================================================
MODEL GENERATION EVALUATION SUMMARY
================================================================================
    strategy  exact_match  num_prompts  avg_generated_length  avg_num_tokens
        cot        0.1234         1425                 12.34             4.56
  instruction        0.1567         1425                 15.67             5.89
        ...
================================================================================
```

**Troubleshooting:**

If you encounter CUDA out-of-memory errors:

- Script automatically uses 4-bit quantization (should fit in 15–20GB VRAM)
- If still OOM: reduce generation batch size or use smaller model

If model download fails:

- Check internet connection
- Ensure HuggingFace token is valid: `huggingface-cli login`
- Model is 30B parameters (~60GB unquantized); requires stable connection

---

## Stage 4: Proxy vs. Model Comparison

**Purpose:** Validate whether prompt strategies that work on TF-IDF also improve real model generation.

**Run after Stage 3:**

```bash
.\.venv\Scripts\python scripts\compare_proxy_vs_model.py
```

**Outputs:**

- `outputs/comparison_proxy_vs_model_summary.csv` - Side-by-side accuracy
- `outputs/comparison_proxy_vs_model.md` - Detailed analysis
- `outputs/comparison_disagreements.csv` - Cases where proxy and model disagree

**What to Look For:**

✓ **Good Alignment:** If proxy and model agree on best strategy

- Proxy is reliable for strategy selection
- Confirms TF-IDF captures meaningful signal

⚠️ **Divergence:** If proxy best ≠ model best

- Proxy may not fully capture model behavior
- Need to prioritize model results for strategy selection

✓ **Model Outperformance:** If model correct where proxy fails

- Indicates model can reason beyond retrieval similarity
- May justify fine-tuning on best strategies

---

## Recommended Workflow

### Week 1: Validation & Testing

1. ✅ Run prompt_engineering.py (TF-IDF proxy)
2. ✅ Run analyze_prompt_errors.py (error analysis)
3. Run test_generation_pipeline.py (validate code path with GPT-2)

### Week 2: Production Evaluation

4. Run model_generation_evaluation.py (full Nemotron evaluation on GPU machine)
5. Run compare_proxy_vs_model.py (validate against TF-IDF proxy)

### Week 3: Iteration

6. Based on comparison results:
   - If model performs well: select best strategy, prepare for LoRA fine-tuning
   - If model struggles: analyze failure modes, iterate on prompt design
   - Consider few-shot templating refinements

---

## Technical Details

### Prompt Strategies

All 5 strategies use the same validation data and answer comparison logic:

```
Minimal:         {user_prompt}
Instruction:     ### Instruction: {user_prompt} \n\n ### Answer:
CoT:             Think step by step: {user_prompt} \n\n Answer:
AnswerFormat:    {user_prompt} \n\n Answer (short, exact):
FewShot:         Example 1: ... Answer: ... \n Example 2: ... \n Now answer: {user_prompt} \n Answer:
```

### Answer Normalization

Both proxy and model use identical answer normalization:

```python
normalized = answer.lower().strip()
```

This ensures fair comparison (exact-match is binary, 1 if normalized answers match).

### Generation Parameters

**Model generation:**

- `temperature=0.0` (deterministic, greedy sampling)
- `max_new_tokens=64` (sufficient for most reasoning answers)
- `top_p=0.95` (only if temperature > 0)

**Comparison metric:**

- Exact-match: binary (1 if normalized gold == normalized generated)
- No partial credit (unlike retrieval similarity)

---

## Output Files Summary

After completing all stages, you'll have:

```
outputs/
├── Stage 1 (TF-IDF Proxy):
│   ├── prompt_experiments_summary.csv           # 5 rows: per-strategy stats
│   └── prompt_experiments_examples.csv          # 7,125 rows: per-example predictions
│
├── Stage 2 (Error Analysis):
│   ├── prompt_error_analysis.md                 # Human-readable report
│   ├── prompt_error_analysis_by_problem_type.csv
│   ├── prompt_error_analysis_by_family_and_length.csv
│   └── prompt_error_analysis_hard_cases.csv     # 15 hardest examples
│
├── Stage 3 (Model Generation):
│   ├── model_generation_summary.csv             # 5 rows: per-strategy stats
│   └── model_generation_examples.csv            # 7,125 rows: per-example predictions
│
└── Stage 4 (Comparison):
    ├── comparison_proxy_vs_model_summary.csv    # Strategy comparison
    ├── comparison_proxy_vs_model.md             # Detailed insights
    └── comparison_disagreements.csv             # Proxy vs. model mismatches
```

---

## Next Steps After Evaluation

Once you've completed all 4 stages:

1. **Identify Best Strategy**
   - Use model results (not proxy) to determine best strategy
   - Cross-reference with problem type breakdowns

2. **Prepare for LoRA Fine-Tuning**
   - Select best strategy and problem types where model performs well
   - Prepare training data using selected strategy
   - Use `lora_pipeline.ipynb` to fine-tune

3. **Test on Held-Out Test Set**
   - Evaluate fine-tuned model on test.csv (3 examples)
   - Compare to baseline Nemotron without fine-tuning

4. **Iterate**
   - If fine-tuning helps: explore other hyperparameters (LoRA rank, learning rate)
   - If minimal improvement: consider other approaches (ensemble, mixture-of-experts, etc.)

---

## Questions & Troubleshooting

**Q: Can I run model evaluation without a GPU?**
A: The test_generation_pipeline.py with GPT-2 works on CPU. For Nemotron, you need a GPU (CUDA). CPU inference is ~100x slower; not practical for 1,425 examples.

**Q: Why does the model perform worse than TF-IDF?**
A: Common reasons:

- Model struggles with instruction-following on reasoning tasks
- Answer format in completions differs from gold answers
- Model generates verbose text that fails exact-match
- Consider post-processing (extract first number, first word, etc.)

**Q: Can I use a smaller model than Nemotron?**
A: Yes, but it should be instruction-tuned for best results. Examples:

- Llama-2-7B-Chat
- Mistral-7B-Instruct
- Phi-2 or Phi-3

Just change `model_name` in model_generation_evaluation.py.

**Q: How long does model evaluation take?**
A: ~2–4 hours for 1,425 examples × 5 strategies = 7,125 generations on a single A100 GPU.
With smaller GPU (3090): ~8–12 hours. With CPU: not practical.

**Q: Can I interrupt and resume?**
A: The current script doesn't support checkpointing. To resume: either wait for completion or modify script to save intermediate results.

---

## References

- **TF-IDF Proxy:** `scripts/prompt_engineering.py`
- **Error Analysis:** `scripts/analyze_prompt_errors.py`
- **Model Generation:** `scripts/model_generation_evaluation.py`
- **Comparison:** `scripts/compare_proxy_vs_model.py`
- **Testing:** `scripts/test_generation_pipeline.py`
- **LoRA Training:** `lora_pipeline.ipynb`
