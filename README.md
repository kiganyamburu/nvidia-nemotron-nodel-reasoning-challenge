# NVIDIA Nemotron Model Reasoning Challenge

## Overview

This repository contains the working files used to explore the Nemotron reasoning dataset, benchmark prompt strategies, and scaffold LoRA fine-tuning for Nemotron-3-Nano.

## Contents

**Data & Notebooks:**
- `train.csv` - training data with `id`, `prompt`, and `answer` columns (9,500 rows)
- `test.csv` - test data (3 rows)
- `data_analysis.ipynb` - dataset exploration, problem classification, baseline analysis
- `lora_pipeline.ipynb` - LoRA fine-tuning scaffold for Nemotron-3-Nano-30B with GPU detection

**Evaluation Scripts:**
- `scripts/prompt_engineering.py` - prompt-template experiments using TF-IDF retrieval proxy (fast, low-cost baseline)
- `scripts/analyze_prompt_errors.py` - detailed error analysis stratified by problem type, answer family, prompt length
- `scripts/model_generation_evaluation.py` - actual model-based evaluation using Nemotron model.generate()
- `scripts/compare_proxy_vs_model.py` - comparison of TF-IDF proxy results vs. real model outputs

**Utilities:**
- `scripts/inspect_prompts.py` - prompt tokenization debugging helper
- `scripts/test_vectorizer.py` - TF-IDF vocabulary smoke test
- `.gitignore` - ignores local environments, notebook checkpoints, and generated outputs

## Setup & Dependencies

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy scikit-learn torch transformers datasets peft trl accelerate safetensors huggingface_hub
```

All dependencies are pinned to compatible versions in the .venv for reproducibility.

## Evaluation Workflow

The project uses a **two-stage evaluation approach** to validate prompt engineering techniques:

### Stage 1: TF-IDF Proxy (Fast Baseline)
Run `scripts/prompt_engineering.py` to evaluate 5 prompt strategies (minimal, instruction, cot, answer_format, few_shot) using TF-IDF retrieval as a fast proxy for model responses.

```bash
.\.venv\Scripts\python scripts\prompt_engineering.py
```

**Outputs:**
- `outputs/prompt_experiments_summary.csv` - per-strategy accuracy and similarity
- `outputs/prompt_experiments_examples.csv` - per-example predictions

**Key findings from proxy evaluation:**
- Mathematical problems: ~43% exact-match (only domain with signal)
- Cryptography, Geometry, Logic: ~0% exact-match
- Roman numeral answers: some partial recovery with prompt strategies
- Binary, numeric, text answers: mostly unrecoverable via retrieval

### Stage 2: Detailed Error Analysis
Run `scripts/analyze_prompt_errors.py` to stratify failures by problem type and answer family/prompt length.

```bash
.\.venv\Scripts\python scripts\analyze_prompt_errors.py
```

**Outputs:**
- `outputs/prompt_error_analysis.md` - human-readable report with stratified tables
- `outputs/prompt_error_analysis_by_problem_type.csv` - accuracy stratified by problem type
- `outputs/prompt_error_analysis_by_family_and_length.csv` - accuracy stratified by answer family and prompt length
- `outputs/prompt_error_analysis_hard_cases.csv` - sample failure cases (15 hardest examples)

### Stage 3: Model-Based Evaluation (Real Generation)
Run `scripts/model_generation_evaluation.py` to evaluate the same 5 prompt strategies using the actual Nemotron model via `model.generate()`.

```bash
.\.venv\Scripts\python scripts/model_generation_evaluation.py
```

**Requirements:**
- GPU with sufficient VRAM (60GB+ for bfloat16, 15–20GB with 4-bit quantization)
- Access to Nemotron-3-Nano-30B on HuggingFace Hub

**Outputs:**
- `outputs/model_generation_summary.csv` - per-strategy accuracy from actual model
- `outputs/model_generation_examples.csv` - per-example predictions with generated answers

### Stage 4: Proxy vs. Model Comparison
Run `scripts/compare_proxy_vs_model.py` to validate whether prompt strategies that work on TF-IDF also improve real model generation.

```bash
.\.venv\Scripts\python scripts/compare_proxy_vs_model.py
```

**Outputs:**
- `outputs/comparison_proxy_vs_model_summary.csv` - side-by-side accuracy comparison
- `outputs/comparison_proxy_vs_model.md` - detailed analysis of strategy agreement and divergence
- `outputs/comparison_disagreements.csv` - cases where proxy and model disagree

**Key comparisons:**
- Which strategies improve with real model vs. proxy?
- Does proxy ranking match actual model ranking?
- Are there cases where model succeeds where proxy fails (and vice versa)?

## LoRA Fine-Tuning (Optional)

Once prompt validation is complete, use `lora_pipeline.ipynb` to fine-tune the model:

1. Open `lora_pipeline.ipynb` in Jupyter
2. The notebook auto-detects GPU availability and runs training only when CUDA is available
3. Training uses conservative LoRA config (LORA_R=16, LORA_ALPHA=32, LORA_DROPOUT=0.05) suitable for the 30B model
4. On CPU-only systems, the notebook includes a guarded smoke-test using a tiny model (no-op)

**Training config:**
- Model: Nemotron-3-Nano-30B
- Data: Training split (85% of 9,500 rows = 8,075 examples)
- Batch size: 1 (with gradient accumulation=16 for effective batch size 16)
- Learning rate: 1e-4
- Epochs: 1 (for iteration speed; increase for production)

## Generated Artifacts

All outputs are written to the `outputs/` directory and ignored by git:

```
outputs/
├── prompt_experiments_summary.csv              # Proxy: per-strategy stats
├── prompt_experiments_examples.csv             # Proxy: per-example predictions (7,125 rows)
├── prompt_error_analysis.md                    # Error stratification report
├── prompt_error_analysis_by_problem_type.csv   # Accuracy by problem type
├── prompt_error_analysis_by_family_and_length.csv # Accuracy by answer family & prompt length
├── prompt_error_analysis_hard_cases.csv        # Sample failures (15 rows)
├── model_generation_summary.csv                # Model: per-strategy stats
├── model_generation_examples.csv               # Model: per-example predictions (7,125 rows)
├── comparison_proxy_vs_model_summary.csv       # Side-by-side proxy vs. model comparison
├── comparison_proxy_vs_model.md                # Detailed proxy vs. model analysis
└── comparison_disagreements.csv                # Cases where proxy and model disagree
```

## Architecture & Design

**Problem Classification:**
- Prompt text classified into: Bit Manipulation, Cryptography, Logic Puzzles, Sequence Analysis, Mathematical, Geometry, Graph/Network, Geography/Navigation, Reasoning
- Uses regex word-boundary matching (e.g., `r"\bcalculate\b"`) to avoid false positives

**Answer Families:**
- Binary (4+ bits)
- Roman numerals
- Numeric (int or float)
- Text (default)

**Prompt Strategies (5):**
1. **minimal** - raw prompt only
2. **instruction** - adds "### Instruction:" template
3. **cot** - adds "Think step by step:" for chain-of-thought
4. **answer_format** - adds "Answer (short, exact):" format guidance
5. **few_shot** - adds 2-shot in-context examples

**Train/Validation Split:**
- 85% train (8,075 examples) - reserved for LoRA fine-tuning
- 15% validation (1,425 examples) - used for all prompt and model evaluation experiments

## Notes & Future Work

- TF-IDF proxy reveals signal only for Mathematical problems (~43% accuracy); other domains show ~0% exact-match.
- Model-based evaluation is critical to determine if prompt strategies improve real generation.
- Prompt strategies that help with TF-IDF may differ from those that help with actual model outputs (common proxy failure mode).
- Roman numeral answers show some recovery with prompt engineering; binary, numeric, text remain challenging.
- Next steps after validation: run fine-tuning with best-performing prompt + strategy combination, evaluate on held-out test set.

