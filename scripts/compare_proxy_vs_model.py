"""
Comparison of TF-IDF proxy results vs. actual model generation.
Validates whether prompt strategies that improve TF-IDF retrieval also improve real model outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def compare_proxy_vs_model():
    """Compare TF-IDF proxy results with model generation results."""

    output_dir = Path("outputs")

    # Load TF-IDF proxy results
    try:
        df_proxy = pd.read_csv(output_dir / "prompt_experiments_examples.csv")
        print("Loaded TF-IDF proxy results")
    except FileNotFoundError:
        print("ERROR: TF-IDF proxy results not found. Run prompt_engineering.py first.")
        return

    # Load model generation results
    try:
        df_model = pd.read_csv(output_dir / "model_generation_examples.csv")
        print("Loaded model generation results")
    except FileNotFoundError:
        print(
            "ERROR: Model generation results not found. Run model_generation_evaluation.py first."
        )
        return

    # Merge on strategy, id to align results
    df_merged = pd.merge(
        df_proxy[["strategy", "id", "exact"]].rename(columns={"exact": "proxy_exact"}),
        df_model[["strategy", "id", "exact_match"]].rename(
            columns={"exact_match": "model_exact"}
        ),
        on=["strategy", "id"],
        how="inner",
    )

    print(
        f"\nAligned {len(df_merged)} predictions across {df_merged['strategy'].nunique()} strategies"
    )

    # ========================================================================
    # COMPARISON SUMMARY BY STRATEGY
    # ========================================================================

    summary_by_strategy = []
    for strategy in sorted(df_merged["strategy"].unique()):
        df_strat = df_merged[df_merged["strategy"] == strategy]

        proxy_acc = df_strat["proxy_exact"].mean()
        model_acc = df_strat["model_exact"].mean()

        # Agreement: both correct, both incorrect, disagreement
        both_correct = (
            (df_strat["proxy_exact"] == 1) & (df_strat["model_exact"] == 1)
        ).sum()
        both_wrong = (
            (df_strat["proxy_exact"] == 0) & (df_strat["model_exact"] == 0)
        ).sum()
        proxy_only = (
            (df_strat["proxy_exact"] == 1) & (df_strat["model_exact"] == 0)
        ).sum()
        model_only = (
            (df_strat["proxy_exact"] == 0) & (df_strat["model_exact"] == 1)
        ).sum()

        summary_by_strategy.append(
            {
                "strategy": strategy,
                "proxy_accuracy": proxy_acc,
                "model_accuracy": model_acc,
                "accuracy_delta": model_acc - proxy_acc,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "proxy_only": proxy_only,
                "model_only": model_only,
                "agreement_pct": ((both_correct + both_wrong) / len(df_strat) * 100),
                "num_predictions": len(df_strat),
            }
        )

    df_summary = pd.DataFrame(summary_by_strategy)

    # Save summary
    summary_path = output_dir / "comparison_proxy_vs_model_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary to {summary_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("PROXY VS. MODEL COMPARISON SUMMARY")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)

    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================

    markdown_report = "# Proxy vs. Model Comparison Report\n\n"

    markdown_report += "## Overview\n"
    markdown_report += f"- Total aligned predictions: {len(df_merged)}\n"
    markdown_report += f"- Strategies compared: {df_merged['strategy'].nunique()}\n\n"

    markdown_report += "## Key Findings\n\n"

    # Find best strategy on each
    best_proxy_idx = df_summary["proxy_accuracy"].idxmax()
    best_model_idx = df_summary["model_accuracy"].idxmax()

    best_proxy_strategy = df_summary.loc[best_proxy_idx, "strategy"]
    best_proxy_acc = df_summary.loc[best_proxy_idx, "proxy_accuracy"]

    best_model_strategy = df_summary.loc[best_model_idx, "strategy"]
    best_model_acc = df_summary.loc[best_model_idx, "model_accuracy"]

    markdown_report += f"- **Best TF-IDF proxy strategy:** {best_proxy_strategy} ({best_proxy_acc:.1%} accuracy)\n"
    markdown_report += f"- **Best model strategy:** {best_model_strategy} ({best_model_acc:.1%} accuracy)\n\n"

    # Strategy correlations
    markdown_report += "## Strategy Accuracy Comparison\n\n"
    markdown_report += "| Strategy | Proxy Acc | Model Acc | Delta | Agreement % |\n"
    markdown_report += "|----------|-----------|-----------|-------|-------------|\n"
    for _, row in df_summary.iterrows():
        markdown_report += f"| {row['strategy']} | {row['proxy_accuracy']:.1%} | {row['model_accuracy']:.1%} | {row['accuracy_delta']:+.1%} | {row['agreement_pct']:.1f}% |\n"

    markdown_report += "\n"

    # Agreement breakdown
    markdown_report += "## Agreement Breakdown\n\n"
    markdown_report += "Across all strategies:\n\n"

    total_both_correct = df_summary["both_correct"].sum()
    total_both_wrong = df_summary["both_wrong"].sum()
    total_proxy_only = df_summary["proxy_only"].sum()
    total_model_only = df_summary["model_only"].sum()
    total = len(df_merged)

    markdown_report += (
        f"- **Both correct:** {total_both_correct} ({total_both_correct/total:.1%})\n"
    )
    markdown_report += (
        f"- **Both wrong:** {total_both_wrong} ({total_both_wrong/total:.1%})\n"
    )
    markdown_report += f"- **Proxy correct, model wrong:** {total_proxy_only} ({total_proxy_only/total:.1%})\n"
    markdown_report += f"- **Proxy wrong, model correct:** {total_model_only} ({total_model_only/total:.1%})\n"
    markdown_report += f"- **Overall agreement:** {(total_both_correct + total_both_wrong)/total:.1%}\n\n"

    # Key insights
    markdown_report += "## Insights\n\n"

    if best_model_strategy != best_proxy_strategy:
        markdown_report += f"⚠️ **Strategy divergence:** TF-IDF proxy favors '{best_proxy_strategy}' but actual model performs best with '{best_model_strategy}'.\n"
        markdown_report += (
            f"   This suggests the proxy may not fully capture model behavior.\n\n"
        )
    else:
        markdown_report += f"✓ **Strategy agreement:** Both proxy and model agree '{best_model_strategy}' is best.\n\n"

    if total_model_only > total_proxy_only:
        markdown_report += f"✓ **Model outperforms proxy:** Model is correct on {total_model_only - total_proxy_only} cases where TF-IDF fails.\n"
        markdown_report += f"   This indicates the actual model can reason beyond simple retrieval similarity.\n\n"
    else:
        markdown_report += f"⚠️ **Proxy reliable:** TF-IDF is more often correct than model ({total_proxy_only} vs {total_model_only}).\n"
        markdown_report += f"   Model may be suffering from instruction-following or hallucination issues.\n\n"

    # Strategy ranking comparison
    proxy_ranking = df_summary.nlargest(5, "proxy_accuracy")["strategy"].tolist()
    model_ranking = df_summary.nlargest(5, "model_accuracy")["strategy"].tolist()

    markdown_report += "## Strategy Rankings\n\n"
    markdown_report += "**TF-IDF Proxy (top 5):**\n"
    for i, s in enumerate(proxy_ranking, 1):
        markdown_report += f"{i}. {s}\n"
    markdown_report += "\n**Actual Model (top 5):**\n"
    for i, s in enumerate(model_ranking, 1):
        markdown_report += f"{i}. {s}\n"

    markdown_report += "\n"

    # Save markdown report
    report_path = output_dir / "comparison_proxy_vs_model.md"
    with open(report_path, "w") as f:
        f.write(markdown_report)
    print(f"\nSaved detailed report to {report_path}")

    # ========================================================================
    # DETAILED DISAGREEMENT ANALYSIS
    # ========================================================================

    # Cases where model succeeds but proxy fails
    model_wins = df_merged[
        (df_merged["proxy_exact"] == 0) & (df_merged["model_exact"] == 1)
    ]
    proxy_wins = df_merged[
        (df_merged["proxy_exact"] == 1) & (df_merged["model_exact"] == 0)
    ]

    if len(model_wins) > 0:
        print(f"\nModel wins over proxy: {len(model_wins)} cases")
        print(f"  Sample strategies: {model_wins['strategy'].value_counts().to_dict()}")

    if len(proxy_wins) > 0:
        print(f"\nProxy wins over model: {len(proxy_wins)} cases")
        print(f"  Sample strategies: {proxy_wins['strategy'].value_counts().to_dict()}")

    # Save detailed disagreements
    detailed_path = output_dir / "comparison_disagreements.csv"
    df_merged.to_csv(detailed_path, index=False)
    print(f"Saved detailed disagreements to {detailed_path}")


if __name__ == "__main__":
    compare_proxy_vs_model()
