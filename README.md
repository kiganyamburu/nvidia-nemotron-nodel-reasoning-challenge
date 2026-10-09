# NVIDIA Nemotron Model Reasoning Challenge

## Overview

This repository contains the working files used to explore the Nemotron reasoning dataset, benchmark prompt strategies, and scaffold LoRA fine-tuning for Nemotron-3-Nano.

## Contents

- `train.csv` - training data with `prompt` and `answer` columns
- `test.csv` - test data
- `data_analysis.ipynb` - dataset exploration, label/theme inspection, and baseline analysis
- `lora_pipeline.ipynb` - LoRA fine-tuning scaffold for Nemotron-3-Nano-30B
- `scripts/prompt_engineering.py` - prompt-template experiments using a TF-IDF retrieval proxy
- `scripts/inspect_prompts.py` - prompt debugging helper
- `scripts/test_vectorizer.py` - quick TF-IDF vocabulary smoke test
- `.gitignore` - ignores local environments, notebook checkpoints, and generated outputs

## Setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy scikit-learn torch transformers datasets peft trl accelerate safetensors huggingface_hub
```

## How To Run

Run the notebooks in order if you want the full workflow:

1. Open `data_analysis.ipynb` for EDA and baseline checks.
2. Open `lora_pipeline.ipynb` to review the LoRA training scaffold. The notebook is configured to run training only when CUDA is available.
3. Use `scripts/prompt_engineering.py` to reproduce prompt-template experiments and write CSV summaries under `outputs/`.

Example:

```bash
.\.venv\Scripts\python scripts\prompt_engineering.py
```

## Generated Artifacts

The repository writes experiment outputs to `outputs/`, including prompt experiment summaries and per-example predictions. These files are generated locally and are ignored by git.

## Notes

- Keep preprocessing and experiment settings reproducible with a fixed random seed.
- The LoRA notebook is a scaffold for GPU-backed runs; it includes a guarded smoke path when training is not enabled.
- The prompt-engineering script uses TF-IDF retrieval as a cheap proxy, not as a substitute for model-based evaluation.
