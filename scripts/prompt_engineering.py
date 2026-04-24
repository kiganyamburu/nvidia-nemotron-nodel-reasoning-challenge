#!/usr/bin/env python3
"""Quick prompt-engineering experiments using TF-IDF retrieval as a fast proxy.

This script:
- loads train.csv
- creates a train/validation split
- defines several prompt templates/strategies
- for each strategy, transforms prompts, fits a TF-IDF retriever on transformed train prompts,
  retrieves nearest training answers for transformed validation prompts, and measures exact-match
  accuracy as a cheap proxy for how well the prompt surface helps retrieval.
- writes a CSV with per-example predictions and a short summary per strategy.

Note: This is a low-cost proxy for prompt quality. Replace retrieval with model.generate when you
have a GPU and the model available to test the templates end-to-end.
"""

from pathlib import Path
import random
import csv
import re
import unicodedata
import argparse
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def normalize_answer(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s


def template_instruction(p: str) -> str:
    return f"### Instruction:\n{p}\n\n### Response:\n"


def template_minimal(p: str) -> str:
    return p


def template_cot(p: str) -> str:
    return f"Think step by step:\n{p}\nAnswer:"  # short CoT prefix


def template_answer_format(p: str) -> str:
    return f"{p}\nAnswer (short, exact):"


def template_few_shot(p: str, examples: List[Dict[str, str]]) -> str:
    # two-shot inline examples
    ex_text = []
    for ex in examples:
        ex_text.append(f"Q: {ex['prompt']}\nA: {ex['answer']}\n")
    examples_block = "\n".join(ex_text)
    return f"{examples_block}\nQ: {p}\nA:"


def run_strategy(
    name: str, transform_fn, train_df: pd.DataFrame, valid_df: pd.DataFrame
):
    # transform both train and validation prompts
    train_texts_series = train_df["prompt"].map(transform_fn).astype(str)
    valid_texts_series = valid_df["prompt"].map(transform_fn).astype(str)

    # drop empty transformed texts to avoid empty-vocabulary errors
    train_nonempty_mask = train_texts_series.str.strip().astype(bool)
    valid_nonempty_mask = valid_texts_series.str.strip().astype(bool)

    train_texts = train_texts_series[train_nonempty_mask].tolist()
    valid_texts = valid_texts_series[valid_nonempty_mask].tolist()

    # Align filtered dataframes
    train_df_filtered = train_df[train_nonempty_mask].reset_index(drop=True)
    valid_df_filtered = valid_df[valid_nonempty_mask].reset_index(drop=True)

    # If filtering removed all documents, return None summary
    if len(train_texts) == 0 or len(valid_texts) == 0:
        return {
            "strategy": name,
            "n_valid": len(valid_df),
            "exact_match": float("nan"),
            "avg_similarity": float("nan"),
        }, []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=8000,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        train_matrix = vectorizer.fit_transform(train_texts)
    except ValueError:
        # empty vocabulary or similar issue
        return {
            "strategy": name,
            "n_valid": len(valid_df),
            "exact_match": float("nan"),
            "avg_similarity": float("nan"),
        }, []
    valid_matrix = vectorizer.transform(valid_texts)

    sim = cosine_similarity(valid_matrix, train_matrix)
    best_idx = sim.argmax(axis=1)
    pred_answers = (
        train_df_filtered.iloc[best_idx]["answer"]
        .astype(str)
        .map(normalize_answer)
        .values
    )

    gold = valid_df_filtered["answer"].astype(str).map(normalize_answer).values
    exact = pred_answers == gold

    summary = {
        "strategy": name,
        "n_valid": len(valid_df),
        "exact_match": float(exact.mean()),
        "avg_similarity": float(sim.max(axis=1).mean()),
    }

    per_example = []
    for i, row in valid_df_filtered.reset_index(drop=True).iterrows():
        per_example.append(
            {
                "strategy": name,
                "id": row.get("id", i),
                "prompt": row["prompt"],
                "gold_answer": row["answer"],
                "pred_answer": pred_answers[i],
                "exact": bool(exact[i]),
                "similarity": float(sim[i].max()),
            }
        )

    return summary, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir", type=str, default=".", help="project root containing train.csv"
    )
    parser.add_argument("--out", type=str, default="outputs/prompt_experiments.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    proj = Path(args.project_dir)
    train_path = proj / "train.csv"
    if not train_path.exists():
        raise SystemExit(f"train.csv not found at {train_path}")

    df = pd.read_csv(train_path)
    # keep only prompt/answer
    df = df[["prompt", "answer"]].copy()

    # small validation split for fast experiments
    train_df, valid_df = train_test_split(
        df, test_size=0.15, random_state=args.seed, shuffle=True
    )

    # choose two random examples for few-shot template (from train)
    sample_examples = train_df.sample(n=2, random_state=args.seed)[
        ["prompt", "answer"]
    ].to_dict(orient="records")

    strategies = [
        ("minimal", template_minimal),
        ("instruction", template_instruction),
        ("cot", template_cot),
        ("answer_format", template_answer_format),
        ("few_shot", lambda p: template_few_shot(p, sample_examples)),
    ]

    all_summaries = []
    all_examples = []
    for name, fn in strategies:
        s, examples = run_strategy(name, fn, train_df, valid_df)
        all_summaries.append(s)
        all_examples.extend(examples)
        print(f"{name}: exact={s['exact_match']:.4f} avg_sim={s['avg_similarity']:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save per-example predictions
    examples_df = pd.DataFrame(all_examples)
    examples_df.to_csv(out_dir / "prompt_experiments_examples.csv", index=False)

    # save summary
    pd.DataFrame(all_summaries).to_csv(
        out_dir / "prompt_experiments_summary.csv", index=False
    )

    print("\nSaved outputs to", out_dir)


if __name__ == "__main__":
    main()
