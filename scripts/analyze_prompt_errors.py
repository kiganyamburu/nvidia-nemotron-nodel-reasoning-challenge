#!/usr/bin/env python3
"""Analyze prompt experiment failures and stratify accuracy by problem type.

The script reads outputs/prompt_experiments_examples.csv and produces:
- outputs/prompt_error_analysis.md
- outputs/prompt_error_analysis_by_problem_type.csv
- outputs/prompt_error_analysis_hard_cases.csv

Problem types are inferred from prompt keywords using the same coarse reasoning
taxonomy used in the notebook, so the report stays aligned with the project's
existing labels.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Iterable

import pandas as pd


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def classify_problem(prompt_text: object) -> str:
    prompt_lower = normalize_text(prompt_text)

    rules: list[tuple[str, Iterable[str]]] = [
        (
            "Bit Manipulation",
            [
                r"\bbit\b",
                r"\bbinary\b",
                r"\bxor\b",
                r"\bshift\b",
                r"\brotate\b",
                r"\brotation\b",
            ],
        ),
        (
            "Cryptography",
            [
                r"\bcipher\b",
                r"\bdecrypt\b",
                r"\bencrypt\b",
                r"\bencryption\b",
                r"\bsecret\b",
                r"\bsubstitution\b",
                r"\bcaesar\b",
            ],
        ),
        (
            "Logic Puzzles",
            [r"\bpuzzle\b", r"\blogic\b", r"\brule\b", r"\bpattern\b", r"\briddle\b"],
        ),
        (
            "Sequence Analysis",
            [r"\bsequence\b", r"\bseries\b", r"\bnext\b", r"\bcontinue\b", r"\bterm\b"],
        ),
        (
            "Mathematical",
            [
                r"\bmath\b",
                r"\bequation\b",
                r"\bcalculate\b",
                r"\bnumber\b",
                r"\bsum\b",
                r"\bproduct\b",
                r"\bprime\b",
                r"\bdivisor\b",
                r"\bfraction\b",
            ],
        ),
        (
            "Geometry",
            [
                r"\bshape\b",
                r"\bangle\b",
                r"\bdistance\b",
                r"\bgeometry\b",
                r"\bcoordinate\b",
                r"\btriangle\b",
                r"\bcircle\b",
                r"\bpolygon\b",
            ],
        ),
        (
            "Graph/Network",
            [
                r"\bgraph\b",
                r"\bnode\b",
                r"\bedge\b",
                r"\bpath\b",
                r"\bnetwork\b",
                r"\btree\b",
            ],
        ),
        (
            "Geography/Navigation",
            [
                r"\bmap\b",
                r"\broute\b",
                r"\bdirection\b",
                r"\bnavigation\b",
                r"\bnorth\b",
                r"\bsouth\b",
                r"\beast\b",
                r"\bwest\b",
                r"\bcity\b",
            ],
        ),
        ("Reasoning", [r"\breason\b", r"\bdeduce\b", r"\binfer\b", r"\bconclusion\b"]),
    ]

    for label, patterns in rules:
        if any(re.search(pattern, prompt_lower) for pattern in patterns):
            return label
    return "Other"


def classify_answer_family(answer_text: object) -> str:
    answer_lower = normalize_text(answer_text)
    if re.fullmatch(r"[01]{4,}", answer_lower):
        return "binary"
    if re.fullmatch(r"[ivxlcdm]+", answer_lower):
        return "roman"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer_lower):
        return "numeric"
    return "text"


def preview(text: object, limit: int = 220) -> str:
    value = str(text).replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def build_report(
    examples: pd.DataFrame, by_problem_type: pd.DataFrame, hard_cases: pd.DataFrame
) -> str:
    total_rows = len(examples)
    total_prompts = examples["id"].nunique()
    strategies = sorted(examples["strategy"].unique().tolist())
    overall = examples.groupby("strategy")["exact"].mean().sort_values(ascending=False)

    lines: list[str] = []
    lines.append("# Prompt Error Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total evaluated rows: {total_rows:,}")
    lines.append(f"- Unique prompts evaluated: {total_prompts:,}")
    lines.append(f"- Strategies compared: {', '.join(strategies)}")
    lines.append("")
    lines.append("### Overall exact match")
    for strategy, score in overall.items():
        lines.append(f"- {strategy}: {score:.4f}")
    lines.append("")

    lines.append("## Stratified accuracy by problem type")
    lines.append(by_problem_type.to_string(index=False))
    lines.append("")

    best_per_type = (
        by_problem_type.sort_values(
            ["problem_type", "exact_match", "support"], ascending=[True, False, False]
        )
        .groupby("problem_type", as_index=False)
        .first()[["problem_type", "strategy", "exact_match", "support"]]
    )
    lines.append("## Best strategy per problem type")
    lines.append(best_per_type.to_string(index=False))
    lines.append("")

    lines.append("## Hardest cases")
    if hard_cases.empty:
        lines.append("No prompts were failed by every strategy.")
    else:
        lines.append(hard_cases.head(15).to_string(index=False))
    lines.append("")

    lines.append("## Failure patterns")
    lines.append(
        "- The strongest signal is that the prompt-template changes barely moved exact-match accuracy; the gap between the best and worst strategies is very small."
    )
    lines.append(
        "- Few-shot formatting increased similarity scores sharply, but it did not translate into more exact matches, which suggests retrieval confidence and answer correctness are weakly coupled here."
    )
    lines.append(
        "- Most hard cases are prompts whose surface form resembles many training examples but whose transformation rule is subtle, so nearest-neighbor retrieval returns a plausible but wrong answer."
    )
    lines.append(
        "- When the prompt is a short or highly structured transformation task, the retriever often latches onto format similarity instead of the underlying operation."
    )
    lines.append("")

    lines.append("## Example hard failures")
    sample_hard = hard_cases.head(8)
    if sample_hard.empty:
        lines.append("No hard failures available.")
    else:
        for _, row in sample_hard.iterrows():
            lines.append(
                f"- [all strategies] {row['problem_type']} | gold={row['gold_answer']} | best_sim={row['best_similarity']:.3f} | failed_strategies={row['failed_strategies']}"
            )
            lines.append(f"  Prompt: {row['prompt_preview']}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=str, default=".")
    parser.add_argument(
        "--examples", type=str, default="outputs/prompt_experiments_examples.csv"
    )
    parser.add_argument(
        "--out-md", type=str, default="outputs/prompt_error_analysis.md"
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="outputs/prompt_error_analysis_by_problem_type.csv",
    )
    parser.add_argument(
        "--out-hard-cases",
        type=str,
        default="outputs/prompt_error_analysis_hard_cases.csv",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    examples_path = project_dir / args.examples
    if not examples_path.exists():
        raise SystemExit(f"Examples file not found: {examples_path}")

    examples = pd.read_csv(examples_path)
    if examples.empty:
        raise SystemExit("Examples file is empty.")

    examples = examples.copy()
    examples["problem_type"] = examples["prompt"].map(classify_problem)
    examples["prompt_preview"] = examples["prompt"].map(preview)
    examples["gold_family"] = examples["gold_answer"].map(classify_answer_family)
    examples["pred_family"] = examples["pred_answer"].map(classify_answer_family)
    examples["prompt_length"] = examples["prompt"].astype(str).str.len()
    examples["gold_length"] = examples["gold_answer"].astype(str).str.len()
    examples["pred_length"] = examples["pred_answer"].astype(str).str.len()

    per_type = (
        examples.groupby(["strategy", "problem_type"], as_index=False)
        .agg(
            exact_match=("exact", "mean"),
            avg_similarity=("similarity", "mean"),
            support=("exact", "size"),
            prompt_length=("prompt_length", "mean"),
            gold_length=("gold_length", "mean"),
        )
        .sort_values(
            ["problem_type", "exact_match", "support"], ascending=[True, False, False]
        )
    )

    # Prompts with zero correct predictions across all strategies.
    per_prompt = (
        examples.groupby(
            ["id", "problem_type", "prompt", "gold_answer", "prompt_preview"],
            as_index=False,
        )
        .agg(
            strategies_correct=("exact", "sum"),
            best_similarity=("similarity", "max"),
            mean_similarity=("similarity", "mean"),
            failed_strategies=(
                "strategy",
                lambda values: ", ".join(sorted(set(values))),
            ),
        )
        .sort_values(["strategies_correct", "best_similarity"], ascending=[True, False])
    )
    hard_cases = per_prompt[per_prompt["strategies_correct"] == 0].copy()

    # Attach the most common failure example per strategy.
    best_wrong_rows = []
    for strategy, frame in examples[~examples["exact"]].groupby("strategy"):
        frame = frame.sort_values(
            ["similarity", "prompt_length"], ascending=[False, False]
        )
        best_wrong_rows.append(frame.iloc[0])
    best_wrong_df = (
        pd.DataFrame(best_wrong_rows)
        if best_wrong_rows
        else pd.DataFrame(columns=examples.columns)
    )

    report_text = build_report(examples, per_type, hard_cases)

    out_md = project_dir / args.out_md
    out_summary = project_dir / args.out_summary
    out_hard_cases = project_dir / args.out_hard_cases
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    per_type.to_csv(out_summary, index=False)
    hard_cases.to_csv(out_hard_cases, index=False)
    out_md.write_text(report_text, encoding="utf-8")

    print(report_text)
    if not best_wrong_df.empty:
        print("\nBest wrong example per strategy:")
        print(
            best_wrong_df[
                [
                    "strategy",
                    "problem_type",
                    "gold_answer",
                    "pred_answer",
                    "similarity",
                    "prompt_preview",
                ]
            ].to_string(index=False)
        )
    print(f"\nWrote report to {out_md}")
    print(f"Wrote stratified summary to {out_summary}")
    print(f"Wrote hard cases to {out_hard_cases}")


if __name__ == "__main__":
    main()
