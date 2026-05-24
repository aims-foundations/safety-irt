# -*- coding: utf-8 -*-
"""
Cultural/Conceptual Mismatch Robustness Analysis
=================================================
Addresses reviewer concern that the cultural-mismatch finding rests on
a small manually-inspected sample (30 prompts).

Strategy: the full 2835-row dataset has category labels for every prompt.
We extend the qualitative observation to a quantitative claim:
  - Compute mean τ per category across ALL prompts × languages
  - Bootstrap 95% CIs to characterise uncertainty
  - Identify which categories show systematically elevated τ
  - Test category effect via Kruskal-Wallis (non-parametric ANOVA)

Outputs (irt_validations/results_cultural_mismatch/):
  category_tau_summary.csv     — mean |τ|, CI, n per category
  category_tau_by_lang.csv     — mean |τ| per category × language
  category_tau_boxplot.pdf/png — boxplot of τ distributions by category
  cultural_mismatch_summary.txt — plain-text summary for rebuttal

Usage:
  python cultural_mismatch_robustness.py
"""

import os
import sys
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr

try:
    from fig_style import apply_style, savefig, FULL_WIDTH, C_RED, C_BLUE, C_PURPLE
    apply_style()
except ImportError:
    C_RED, C_BLUE, C_PURPLE = "#c0392b", "#2471a3", "#7d3c98"
    FULL_WIDTH = 5.5
    def savefig(fig, path, **kw):
        fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, "..")
TQ_CSV     = os.path.join(REPO_ROOT, "model", "results", "multimetric_translation_v_DIF.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_cultural_mismatch")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

# Shortened category names for display
SHORT_NAMES = {
    "Violence & incitement":                                "Violence",
    "Theft":                                               "Theft",
    "Weapons":                                             "Weapons",
    "Hate speech & offensive language":                    "Hate speech",
    "Fraud & deception":                                   "Fraud",
    "Soliciting personally identifiable information":      "PII solicitation",
    "Discrimination & injustice":                          "Discrimination",
    "Non-violent unethical behavior (e.g. lying, cheating, etc.)": "Unethical behavior",
    "Bullying & harassment":                               "Bullying",
    "Substance abuse & banned substances":                 "Substance abuse",
    "Adult content":                                       "Adult content",
    "Self-harm":                                           "Self-harm",
    "Conspiracy theories & misinformation":                "Misinformation",
    "Property crime & vandalism":                          "Property crime",
    "Sexual exploitation & human trafficking":             "Sexual exploitation",
    "Child abuse":                                         "Child abuse",
    "Terrorism & organized crime":                         "Terrorism",
    "Animal abuse":                                        "Animal abuse",
}


def parse_tags(tag_str):
    """Return list of category strings from a tags cell."""
    try:
        return ast.literal_eval(str(tag_str))
    except Exception:
        return [str(tag_str)]


def load_data():
    df = pd.read_csv(TQ_CSV)
    df["id"] = df["id"].astype(str)
    # Explode multi-tag prompts so each row is one category label
    df["tags_list"] = df["category"].apply(parse_tags)
    df_exp = df.explode("tags_list").copy()
    df_exp["category_short"] = df_exp["tags_list"].map(SHORT_NAMES).fillna(df_exp["tags_list"])
    return df, df_exp


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    vals = np.asarray(values)
    if len(vals) < 3:
        return np.nan, np.nan
    means = [RNG.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo = (1 - ci) / 2
    return np.percentile(means, [lo * 100, (1 - lo) * 100])


def category_summary(df_exp):
    rows = []
    cats = df_exp.groupby("category_short")["tau"]
    for cat, vals in cats:
        v = vals.dropna().values
        ci_lo, ci_hi = bootstrap_ci(v)
        rows.append({
            "category":          cat,
            "n_obs":             len(v),
            "mean_tau":          round(float(v.mean()), 3),
            "mean_abs_tau":      round(float(np.abs(v).mean()), 3),
            "median_tau":        round(float(np.median(v)), 3),
            "std_tau":           round(float(v.std()), 3),
            "ci95_lo":           round(float(ci_lo), 3),
            "ci95_hi":           round(float(ci_hi), 3),
            "pct_positive":      round(float((v > 0).mean() * 100), 1),
        })
    return pd.DataFrame(rows).sort_values("mean_tau", ascending=False).reset_index(drop=True)


def kruskal_test(df_exp):
    groups = [g["tau"].dropna().values for _, g in df_exp.groupby("category_short") if len(g) >= 5]
    stat, p = kruskal(*groups)
    return stat, p, len(groups)


def category_by_language(df_exp):
    pivot = df_exp.pivot_table(values="tau", index="category_short",
                               columns="language", aggfunc="mean")
    return pivot.round(3)


def plot_boxplot(df_exp, summary):
    order = summary["category"].tolist()
    data  = [df_exp[df_exp["category_short"] == c]["tau"].dropna().values for c in order]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 1.0))
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    medianprops=dict(color="white", lw=1.2),
                    whiskerprops=dict(lw=0.7),
                    capprops=dict(lw=0.7),
                    boxprops=dict(lw=0.5))

    # Colour by mean τ: positive=red, negative=blue
    for patch, cat in zip(bp["boxes"], order):
        m = summary.loc[summary["category"] == cat, "mean_tau"].values[0]
        patch.set_facecolor(C_RED if m > 0.1 else (C_BLUE if m < -0.1 else C_PURPLE))
        patch.set_alpha(0.75)

    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel(r"$\tau_{iL}$ (cross-lingual safety gap)", fontsize=7)
    ax.set_title("τ distribution by harm category\n(all 315 prompts × 9 languages)", fontsize=8)
    fig.tight_layout()
    savefig(fig, os.path.join(OUT_DIR, "category_tau_boxplot"))


def main():
    print("Loading data...")
    df, df_exp = load_data()
    print(f"  {len(df)} prompt×language pairs; {df_exp['category_short'].nunique()} categories after explode.")

    # Kruskal-Wallis: is there a significant category effect?
    kw_stat, kw_p, n_groups = kruskal_test(df_exp)
    print(f"\nKruskal-Wallis across {n_groups} categories: H={kw_stat:.2f}, p={kw_p:.4e}")

    # Category summary with bootstrap CIs
    summary = category_summary(df_exp)
    summary_path = os.path.join(OUT_DIR, "category_tau_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Mean τ by category (all prompts × languages) ===")
    print(summary[["category", "n_obs", "mean_tau", "ci95_lo", "ci95_hi",
                    "pct_positive", "mean_abs_tau"]].to_string(index=False))

    # Per-language breakdown
    lang_pivot = category_by_language(df_exp)
    lang_pivot.to_csv(os.path.join(OUT_DIR, "category_tau_by_lang.csv"))

    # Plot
    plot_boxplot(df_exp, summary)

    # Identify "culturally sensitive" categories: top third by mean τ
    top_n = max(3, len(summary) // 3)
    top_cats = summary.head(top_n)
    bot_cats = summary.tail(top_n)
    gap = top_cats["mean_tau"].mean() - bot_cats["mean_tau"].mean()

    # Plain-text summary
    lines = ["=== Cultural/Conceptual Mismatch Robustness Summary ===\n"]
    lines.append(f"Dataset: {len(df)} prompt×language pairs, {df['id'].nunique()} unique prompts, 9 languages.")
    lines.append(f"Kruskal-Wallis test for category effect: H={kw_stat:.2f}, p={kw_p:.2e} ({n_groups} groups)\n")
    lines.append("Top categories by mean τ (95% bootstrap CI):")
    for _, r in top_cats.iterrows():
        lines.append(f"  {r.category:<40}  mean τ={r.mean_tau:+.3f}  95%CI=[{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]  n={r.n_obs}")
    lines.append("\nBottom categories by mean τ:")
    for _, r in bot_cats.iterrows():
        lines.append(f"  {r.category:<40}  mean τ={r.mean_tau:+.3f}  95%CI=[{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]  n={r.n_obs}")
    lines.append(f"\nMean τ gap (top vs bottom third): {gap:+.3f}")

    txt_path = os.path.join(OUT_DIR, "cultural_mismatch_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary → {txt_path}")


if __name__ == "__main__":
    main()
