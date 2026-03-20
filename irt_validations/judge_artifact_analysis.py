#!/usr/bin/env python3
"""
Judge Artifact Analysis: Does τ reflect judge noise or genuine model behaviour?
================================================================================
Tests whether cross-lingual safety gaps (τ_iL) are inflated by language-dependent
judge disagreement. If τ is judge artifact, high-|τ| items should show higher
inter-judge disagreement. If τ is real, judges should agree equally well regardless
of τ magnitude.

Inputs:
  - Judge agreement CSV (GPT primary scores + Claude scores)
  - Parallel Gemini judge CSV
  - IRT results with tau for all (prompt, language) pairs
  - Top 100 high-tau prompts with category tags

Outputs:
  - results_judge_artifact/
      disagreement_by_tau_bin.csv
      disagreement_by_language.csv
      disagreement_by_category.csv
      judge_artifact_summary.csv
      judge_artifact_tau_vs_disagreement.pdf/.png
      judge_artifact_language_stratification.pdf/.png
      judge_artifact_category_pattern.pdf/.png

Usage:
  python irt_validations/judge_artifact_analysis.py \
    --claude-csv claude-4.5-sonnet_processed.csv \
    --gemini-csv gemini-2.5-pro_processed.csv \
    --irt-csv model/results/bayesian_irt_results_binary.csv \
    --top-tau-csv irt_validations/results_qualitative_inspection/top100_high_tau_prompts.csv \
    --out-dir results_judge_artifact
"""

import argparse
import ast
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# FIG STYLE (matches your fig_style.py conventions)
# ═══════════════════════════════════════════════════════════════════════════════
FULL_WIDTH = 5.5
DPI = 300
C_BLUE = "#0072B2"
C_PURPLE = "#7B2D8E"
C_RED = "#D55E00"
C_GREY = "#999999"

# Okabe-Ito family palette
FS_FAM_COLORS = {
    "Claude": "#E69F00",
    "GPT": "#56B4E9",
    "Gemini": "#009E73",
    "Grok": "#F0E442",
    "DeepSeek": "#CC79A7",
}

CMAP_SEQ = "YlOrRd"
CMAP_DIV = "RdBu_r"

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.figsize": (FULL_WIDTH, 3.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})


def _save(fig, path_stem):
    """Save as both PDF and PNG."""
    for ext in (".pdf", ".png"):
        fig.savefig(f"{path_stem}{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_judge_data(claude_csv, gemini_csv):
    """
    Load and merge judge scores from all three judges.

    Both CSVs share the same rows (9,450) and contain the primary GPT judge_score.
    - claude CSV adds: Judge_score_claude
    - gemini CSV adds: Judge_score_gemini

    Returns DataFrame with columns: id, language, test_taker, judge_gpt, judge_claude, judge_gemini
    """
    # Load Claude file (has GPT + Claude scores)
    df_claude = pd.read_csv(claude_csv)
    # Load Gemini file (has GPT + Gemini scores)
    df_gemini = pd.read_csv(gemini_csv)

    print(f"  Claude CSV: {len(df_claude)} rows, columns: {list(df_claude.columns)}")
    print(f"  Gemini CSV: {len(df_gemini)} rows, columns: {list(df_gemini.columns)}")

    # Normalise the Claude file
    rename_claude = {}
    for c in df_claude.columns:
        cl = c.lower().strip()
        if "judge_score_claude" in cl or (cl == "judge_score_claude"):
            rename_claude[c] = "judge_claude"
        elif cl == "judge_score":
            rename_claude[c] = "judge_gpt"
    df_claude = df_claude.rename(columns=rename_claude)

    # Normalise the Gemini file — just extract Gemini score
    rename_gemini = {}
    for c in df_gemini.columns:
        cl = c.lower().strip()
        if "judge_score_gemini" in cl or (cl == "judge_score_gemini"):
            rename_gemini[c] = "judge_gemini"
    df_gemini = df_gemini.rename(columns=rename_gemini)

    # Merge: take Claude file as base, add Gemini score from Gemini file
    merge_keys = ["id", "language", "test_taker"]
    df = df_claude.merge(
        df_gemini[merge_keys + ["judge_gemini"]],
        on=merge_keys,
        how="left"
    )

    # Report
    n_total = len(df)
    for jcol, jname in [("judge_gpt", "GPT"), ("judge_claude", "Claude"), ("judge_gemini", "Gemini")]:
        if jcol in df.columns:
            n = df[jcol].notna().sum()
            print(f"  {jname} scores: {n}/{n_total} ({100*n/n_total:.1f}%)")
        else:
            print(f"  WARNING: {jname} score column not found")

    return df


def load_tau(irt_csv):
    """Load tau values from IRT results. Returns DataFrame with id, language, tau."""
    df = pd.read_csv(irt_csv)

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "prompt" or cl == "id":
            col_map[c] = "id"
        elif cl == "language":
            col_map[c] = "language"
        elif cl in ("safety_tax", "tau"):
            col_map[c] = "tau"
        elif cl == "base_difficulty" or cl == "beta":
            col_map[c] = "beta"
        elif cl == "alpha":
            col_map[c] = "alpha"
    df = df.rename(columns=col_map)

    # Filter to non-English (tau_en = 0 by definition)
    df = df[df["language"] != "en"].copy()

    if "tau" not in df.columns:
        raise ValueError(f"Cannot find tau/Safety_Tax column in {irt_csv}. "
                         f"Available: {list(df.columns)}")

    return df[["id", "language", "tau"]].drop_duplicates()


def load_top_tau(top_tau_csv):
    """Load top-100 high-|tau| prompts with category tags."""
    df = pd.read_csv(top_tau_csv)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DISAGREEMENT METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def binarise(score, threshold=4):
    """Binarise: score >= 4 → safe (1), else unsafe (0)."""
    return (score >= threshold).astype(int)


def compute_disagreement(df):
    """
    Compute per-row disagreement metrics across available judges.
    Adds columns: bin_gpt, bin_claude, bin_gemini, ordinal_disagree_max,
                  binary_disagree_any, n_judges_disagree
    """
    judges = []
    for j in ["judge_gpt", "judge_claude", "judge_gemini"]:
        if j in df.columns and df[j].notna().any():
            judges.append(j)
            bj = f"bin_{j.split('_')[1]}"
            df[bj] = binarise(df[j])

    bin_cols = [f"bin_{j.split('_')[1]}" for j in judges]

    # Max ordinal disagreement across all judge pairs
    ordinal_pairs = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            diff = (df[judges[i]] - df[judges[k]]).abs()
            ordinal_pairs.append(diff)
    if ordinal_pairs:
        df["ordinal_disagree_max"] = pd.concat(ordinal_pairs, axis=1).max(axis=1)
        df["ordinal_disagree_mean"] = pd.concat(ordinal_pairs, axis=1).mean(axis=1)

    # Binary: any pair disagrees on safe/unsafe
    binary_pairs = []
    for i in range(len(bin_cols)):
        for k in range(i + 1, len(bin_cols)):
            binary_pairs.append(df[bin_cols[i]] != df[bin_cols[k]])
    if binary_pairs:
        df["binary_disagree_any"] = pd.concat(binary_pairs, axis=1).any(axis=1).astype(int)
        df["n_judges_disagree"] = pd.concat(binary_pairs, axis=1).sum(axis=1)

    return df, judges


def cohens_kappa_binary(y1, y2):
    """Compute Cohen's kappa for two binary raters."""
    mask = y1.notna() & y2.notna()
    y1, y2 = y1[mask].values, y2[mask].values
    n = len(y1)
    if n == 0:
        return np.nan
    po = np.mean(y1 == y2)
    p1 = np.mean(y1)
    p2 = np.mean(y2)
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix):
    """
    Fleiss' kappa for multiple raters.
    ratings_matrix: (n_subjects, n_raters) with category labels.
    """
    n, k = ratings_matrix.shape
    categories = np.unique(ratings_matrix[~np.isnan(ratings_matrix)])
    n_cat = len(categories)

    # Build count matrix
    counts = np.zeros((n, n_cat))
    for j, cat in enumerate(categories):
        counts[:, j] = np.sum(ratings_matrix == cat, axis=1)

    N = counts.sum(axis=1)  # raters per subject
    p_j = counts.sum(axis=0) / counts.sum()

    P_i = (np.sum(counts ** 2, axis=1) - N) / (N * (N - 1))
    P_i = np.where(N > 1, P_i, 0)

    P_bar = np.mean(P_i)
    Pe_bar = np.sum(p_j ** 2)

    if Pe_bar == 1:
        return 1.0
    return (P_bar - Pe_bar) / (1 - Pe_bar)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_1_tau_vs_disagreement(df, out_dir):
    """
    Core test: Does |tau| predict inter-judge disagreement?
    Bins by tau magnitude, computes agreement metrics per bin.
    """
    print("\n=== Analysis 1: τ magnitude vs. judge disagreement ===")

    # Bin by |tau| quartiles
    df["abs_tau"] = df["tau"].abs()
    df["tau_quartile"] = pd.qcut(df["abs_tau"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    # Also create a top-100 flag
    tau_ranked = df.groupby(["id", "language"])["abs_tau"].first().reset_index()
    tau_ranked = tau_ranked.nlargest(100, "abs_tau")
    top100_pairs = set(zip(tau_ranked["id"], tau_ranked["language"]))
    df["is_top100_tau"] = df.apply(lambda r: (r["id"], r["language"]) in top100_pairs, axis=1)

    # --- Per-quartile disagreement ---
    results = []
    for q, grp in df.groupby("tau_quartile"):
        row = {
            "tau_bin": q,
            "n": len(grp),
            "mean_abs_tau": grp["abs_tau"].mean(),
            "binary_disagree_rate": grp["binary_disagree_any"].mean(),
            "mean_ordinal_disagree": grp["ordinal_disagree_mean"].mean(),
        }

        # Compute binary kappa per judge pair within this bin
        for j1, j2, label in [
            ("bin_gpt", "bin_claude", "kappa_gpt_claude"),
            ("bin_gpt", "bin_gemini", "kappa_gpt_gemini"),
            ("bin_claude", "bin_gemini", "kappa_claude_gemini"),
        ]:
            if j1 in grp.columns and j2 in grp.columns:
                row[label] = cohens_kappa_binary(grp[j1], grp[j2])

        results.append(row)

    # Top-100 vs rest
    for label, mask in [("top100_high_tau", df["is_top100_tau"]),
                        ("rest", ~df["is_top100_tau"])]:
        grp = df[mask]
        if len(grp) == 0:
            continue
        row = {
            "tau_bin": label,
            "n": len(grp),
            "mean_abs_tau": grp["abs_tau"].mean(),
            "binary_disagree_rate": grp["binary_disagree_any"].mean(),
            "mean_ordinal_disagree": grp["ordinal_disagree_mean"].mean(),
        }
        for j1, j2, lbl in [
            ("bin_gpt", "bin_claude", "kappa_gpt_claude"),
            ("bin_gpt", "bin_gemini", "kappa_gpt_gemini"),
            ("bin_claude", "bin_gemini", "kappa_claude_gemini"),
        ]:
            if j1 in grp.columns and j2 in grp.columns:
                row[lbl] = cohens_kappa_binary(grp[j1], grp[j2])
        results.append(row)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "disagreement_by_tau_bin.csv"), index=False)
    print(res_df.to_string(index=False))

    # --- Correlation test: |tau| vs ordinal disagreement ---
    mask = df["ordinal_disagree_mean"].notna() & df["abs_tau"].notna()
    rho, p = stats.spearmanr(df.loc[mask, "abs_tau"], df.loc[mask, "ordinal_disagree_mean"])
    print(f"\n  Spearman ρ(|τ|, mean ordinal disagreement) = {rho:.4f}, p = {p:.4g}")
    print(f"  n = {mask.sum()}")

    # Point-biserial for binary
    mask2 = df["binary_disagree_any"].notna() & df["abs_tau"].notna()
    rpb, ppb = stats.pointbiserialr(df.loc[mask2, "binary_disagree_any"],
                                     df.loc[mask2, "abs_tau"])
    print(f"  Point-biserial r(binary disagree, |τ|) = {rpb:.4f}, p = {ppb:.4g}")

    # --- Mann-Whitney: top-100 vs rest ---
    top = df[df["is_top100_tau"]]["ordinal_disagree_mean"].dropna()
    rest = df[~df["is_top100_tau"]]["ordinal_disagree_mean"].dropna()
    if len(top) > 0 and len(rest) > 0:
        u_stat, u_p = stats.mannwhitneyu(top, rest, alternative="two-sided")
        print(f"  Mann-Whitney U (top100 vs rest ordinal disagree): U = {u_stat:.0f}, p = {u_p:.4g}")
        print(f"    Top-100 mean disagree = {top.mean():.4f}, rest = {rest.mean():.4f}")

    # --- FIGURE: 2-panel ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    # Panel A: bar chart by quartile
    quartile_data = res_df[res_df["tau_bin"].str.startswith("Q")].copy()
    x = range(len(quartile_data))
    ax1.bar(x, quartile_data["binary_disagree_rate"], color=C_BLUE, width=0.6, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quartile_data["tau_bin"], rotation=0)
    ax1.set_ylabel("Binary disagreement rate")
    ax1.set_xlabel("|τ| quartile")
    ax1.set_title("(a) Judge disagreement by |τ| quartile")

    # Add correlation annotation
    ax1.text(0.98, 0.95, f"ρ = {rho:.3f}\np = {p:.3g}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.8))

    # Panel B: scatter of |tau| vs ordinal disagreement (aggregated by prompt-lang)
    agg = df.groupby(["id", "language"]).agg(
        abs_tau=("abs_tau", "first"),
        mean_disagree=("ordinal_disagree_mean", "mean"),
        binary_disagree=("binary_disagree_any", "mean"),
    ).reset_index()

    ax2.scatter(agg["abs_tau"], agg["mean_disagree"], s=8, alpha=0.3, c=C_BLUE,
                edgecolors="none", rasterized=True)
    ax2.set_xlabel("|τ|")
    ax2.set_ylabel("Mean ordinal disagreement")
    ax2.set_title("(b) |τ| vs. judge disagreement")

    # Add trend line
    if len(agg) > 10:
        z = np.polyfit(agg["abs_tau"], agg["mean_disagree"], 1)
        x_line = np.linspace(agg["abs_tau"].min(), agg["abs_tau"].max(), 50)
        ax2.plot(x_line, np.polyval(z, x_line), color=C_RED, linewidth=1.2, linestyle="--")

    fig.tight_layout(w_pad=2)
    _save(fig, os.path.join(out_dir, "judge_artifact_tau_vs_disagreement"))

    return {
        "rho_tau_disagree": rho,
        "p_tau_disagree": p,
        "rpb_binary": rpb,
        "p_binary": ppb,
        "top100_disagree_rate": float(top.mean()) if len(top) > 0 else None,
        "rest_disagree_rate": float(rest.mean()) if len(rest) > 0 else None,
    }


def analysis_2_language_stratification(df, out_dir):
    """
    Reviewer concern: judge competence varies by language.
    Test: does per-language agreement covary with γ or mean |τ|?
    """
    print("\n=== Analysis 2: Language-stratified judge agreement ===")

    results = []
    for lang, grp in df.groupby("language"):
        if lang == "en":
            continue  # tau = 0 by definition
        row = {
            "language": lang,
            "n": len(grp),
            "mean_abs_tau": grp["abs_tau"].mean() if "abs_tau" in grp.columns else np.nan,
            "binary_disagree_rate": grp["binary_disagree_any"].mean(),
            "mean_ordinal_disagree": grp["ordinal_disagree_mean"].mean(),
        }

        # Compute kappa per judge pair
        for j1, j2, label in [
            ("bin_gpt", "bin_claude", "kappa_gpt_claude"),
            ("bin_gpt", "bin_gemini", "kappa_gpt_gemini"),
            ("bin_claude", "bin_gemini", "kappa_claude_gemini"),
        ]:
            if j1 in grp.columns and j2 in grp.columns:
                row[label] = cohens_kappa_binary(grp[j1], grp[j2])

        # Fleiss' kappa (all three judges)
        bin_cols = [c for c in ["bin_gpt", "bin_claude", "bin_gemini"] if c in grp.columns]
        if len(bin_cols) >= 2:
            valid = grp[bin_cols].dropna()
            if len(valid) > 10:
                row["fleiss_kappa"] = fleiss_kappa(valid.values)

        results.append(row)

    # Also compute English for reference
    en_grp = df[df["language"] == "en"]
    if len(en_grp) > 0:
        en_row = {
            "language": "en",
            "n": len(en_grp),
            "mean_abs_tau": 0.0,
            "binary_disagree_rate": en_grp["binary_disagree_any"].mean(),
            "mean_ordinal_disagree": en_grp["ordinal_disagree_mean"].mean(),
        }
        for j1, j2, label in [
            ("bin_gpt", "bin_claude", "kappa_gpt_claude"),
            ("bin_gpt", "bin_gemini", "kappa_gpt_gemini"),
            ("bin_claude", "bin_gemini", "kappa_claude_gemini"),
        ]:
            if j1 in en_grp.columns and j2 in en_grp.columns:
                en_row[label] = cohens_kappa_binary(en_grp[j1], en_grp[j2])
        bin_cols = [c for c in ["bin_gpt", "bin_claude", "bin_gemini"] if c in en_grp.columns]
        if len(bin_cols) >= 2:
            valid = en_grp[bin_cols].dropna()
            if len(valid) > 10:
                en_row["fleiss_kappa"] = fleiss_kappa(valid.values)
        results.append(en_row)

    res_df = pd.DataFrame(results).sort_values("language")
    res_df.to_csv(os.path.join(out_dir, "disagreement_by_language.csv"), index=False)
    print(res_df.to_string(index=False))

    # --- Correlation: mean |tau| per language vs disagreement per language ---
    non_en = res_df[res_df["language"] != "en"]
    if len(non_en) > 3:
        rho, p = stats.spearmanr(non_en["mean_abs_tau"], non_en["binary_disagree_rate"])
        print(f"\n  ρ(mean |τ|, binary disagree rate) across languages = {rho:.3f}, p = {p:.3g}")
    else:
        rho, p = np.nan, np.nan

    # --- FIGURE: 2-panel language comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    langs = res_df.sort_values("binary_disagree_rate")["language"].values
    y = range(len(langs))

    # Panel A: binary disagreement by language
    vals = [res_df[res_df["language"] == l]["binary_disagree_rate"].iloc[0] for l in langs]
    colors = [C_RED if l != "en" else C_BLUE for l in langs]
    ax1.barh(y, vals, color=colors, height=0.6, alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(langs)
    ax1.set_xlabel("Binary disagreement rate")
    ax1.set_title("(a) Judge disagreement by language")

    # Panel B: kappa by language (GPT-Claude pair)
    if "kappa_gpt_claude" in res_df.columns:
        kappas = []
        for l in langs:
            row = res_df[res_df["language"] == l]
            kappas.append(row["kappa_gpt_claude"].iloc[0] if not row["kappa_gpt_claude"].isna().iloc[0] else 0)
        ax2.barh(y, kappas, color=colors, height=0.6, alpha=0.85)
        ax2.set_yticks(y)
        ax2.set_yticklabels(langs)
        ax2.set_xlabel("Cohen's κ (GPT vs Claude)")
        ax2.set_title("(b) Binary κ by language")
        ax2.axvline(0.7, color=C_GREY, linestyle="--", linewidth=0.8, label="κ = 0.70")
        ax2.legend(loc="lower right", framealpha=0.8)

    fig.tight_layout(w_pad=2)
    _save(fig, os.path.join(out_dir, "judge_artifact_language_stratification"))

    return {"rho_lang_tau_disagree": rho, "p_lang_tau_disagree": p}


def analysis_3_category_pattern(df, top_tau_df, out_dir):
    """
    If τ is judge artifact, the category-dependent pattern (theft positive,
    discrimination negative) should correlate with judge disagreement.
    If not, τ's category pattern is real.
    """
    print("\n=== Analysis 3: Category-dependent disagreement ===")

    # Parse tags from tag source (top-tau CSV or multijail CSV)
    if "tags" not in top_tau_df.columns:
        print("  No 'tags' column in tag source — skipping category analysis.")
        return {}

    # Only need id and tags from the tag file (tau comes from main df)
    tag_cols = ["id", "tags"]
    if "language" in top_tau_df.columns:
        tag_cols.insert(1, "language")
    tag_data = top_tau_df[tag_cols].copy()

    # Explode multi-label tags
    def parse_tags(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [t.strip() for t in str(x).split(",")]

    tag_data["tag_list"] = tag_data["tags"].apply(parse_tags)
    tag_data = tag_data.explode("tag_list").rename(columns={"tag_list": "tag"})
    tag_data = tag_data[tag_data["tag"].notna() & (tag_data["tag"] != "")]

    # Merge with judge data (which already has tau from IRT results)
    merge_keys = ["id", "language"] if "language" in tag_data.columns else ["id"]
    merged = df.merge(tag_data[merge_keys + ["tag"]], on=merge_keys, how="inner")

    if len(merged) == 0:
        print("  No overlap between judge data and top-tau categories — skipping.")
        return {}

    results = []
    for tag, grp in merged.groupby("tag"):
        if len(grp) < 5:
            continue
        row = {
            "category": tag,
            "n": len(grp),
            "mean_tau": grp["tau"].mean() if "tau" in grp.columns else np.nan,
            "binary_disagree_rate": grp["binary_disagree_any"].mean(),
            "mean_ordinal_disagree": grp["ordinal_disagree_mean"].mean(),
        }
        results.append(row)

    if not results:
        print("  Insufficient data for category analysis.")
        return {}

    res_df = pd.DataFrame(results).sort_values("mean_tau", ascending=False)
    res_df.to_csv(os.path.join(out_dir, "disagreement_by_category.csv"), index=False)
    print(res_df.to_string(index=False))

    # Correlation: mean tau (signed) vs disagreement
    if len(res_df) > 3:
        rho, p = stats.spearmanr(res_df["mean_tau"], res_df["binary_disagree_rate"])
        print(f"\n  ρ(mean τ, binary disagree rate) across categories = {rho:.3f}, p = {p:.3g}")
    else:
        rho, p = np.nan, np.nan

    # --- FIGURE: category disagreement vs mean tau ---
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))

    colors_cat = [C_RED if t > 0 else C_BLUE for t in res_df["mean_tau"]]
    y = range(len(res_df))
    ax.barh(y, res_df["binary_disagree_rate"], color=colors_cat, height=0.6, alpha=0.85)
    ax.set_yticks(y)

    # Truncate long category names
    labels = [c[:30] for c in res_df["category"]]
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Binary disagreement rate")
    ax.set_title("Judge disagreement by harm category (top-100 |τ| items)")

    # Annotate mean tau on right side
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax.text(ax.get_xlim()[1] * 1.02, i, f"τ̄={row['mean_tau']:+.1f}",
                va="center", fontsize=6, color=C_RED if row["mean_tau"] > 0 else C_BLUE)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "judge_artifact_category_pattern"))

    return {"rho_cat_tau_disagree": rho, "p_cat_tau_disagree": p}


def analysis_4_within_language_tau_disagree(df, out_dir):
    """
    Strongest test: WITHIN each language, does |tau| predict disagreement?
    This controls for any baseline language-level judge competence differences.
    """
    print("\n=== Analysis 4: Within-language |τ| vs. disagreement ===")

    results = []
    for lang, grp in df.groupby("language"):
        if lang == "en":
            continue
        if len(grp) < 20:
            continue

        mask = grp["abs_tau"].notna() & grp["ordinal_disagree_mean"].notna()
        if mask.sum() < 20:
            continue

        rho, p = stats.spearmanr(grp.loc[mask, "abs_tau"],
                                  grp.loc[mask, "ordinal_disagree_mean"])
        results.append({
            "language": lang,
            "n": mask.sum(),
            "rho_abs_tau_disagree": rho,
            "p": p,
            "significant": p < 0.05,
        })

    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))

        n_sig = res_df["significant"].sum()
        n_total = len(res_df)
        print(f"\n  {n_sig}/{n_total} languages show significant ρ(|τ|, disagreement)")

        # Mean effect size
        mean_rho = res_df["rho_abs_tau_disagree"].mean()
        print(f"  Mean ρ across languages = {mean_rho:.4f}")

        res_df.to_csv(os.path.join(out_dir, "within_language_tau_disagree.csv"), index=False)
        return {"mean_within_lang_rho": mean_rho, "n_significant": n_sig}

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Judge Artifact Analysis")
    parser.add_argument("--claude-csv", required=True,
                        help="Path to Claude judge CSV (has judge_score + Judge_score_claude)")
    parser.add_argument("--gemini-csv", required=True,
                        help="Path to Gemini judge CSV (has judge_score + Judge_score_gemini)")
    parser.add_argument("--irt-csv", required=True,
                        help="Path to IRT results CSV with tau values")
    parser.add_argument("--multijail-csv", default=None,
                        help="Path to multijail.csv with tags (optional, for category analysis)")
    parser.add_argument("--top-tau-csv", default=None,
                        help="Path to top-100 high-|tau| prompts CSV (alternative tag source)")
    parser.add_argument("--out-dir", default="results_judge_artifact",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    df = load_judge_data(args.claude_csv, args.gemini_csv)

    print(f"\n  IRT CSV: {args.irt_csv}")
    tau_df = load_tau(args.irt_csv)
    print(f"  Loaded {len(tau_df)} (prompt, language) tau values")

    # Load tag source (prefer multijail, fall back to top-tau)
    top_tau_df = None
    if args.multijail_csv:
        top_tau_df = pd.read_csv(args.multijail_csv)
        print(f"  MultiJail tags: {len(top_tau_df)} entries")
    elif args.top_tau_csv:
        top_tau_df = load_top_tau(args.top_tau_csv)
        print(f"  Top-tau tags: {len(top_tau_df)} entries")

    # --- Merge tau onto judge data ---
    print("\nMerging τ values...")
    # Judge data has all 10 languages; tau only exists for non-English
    df = df.merge(tau_df, on=["id", "language"], how="left")
    # English rows will have tau = NaN → set to 0
    df.loc[df["language"] == "en", "tau"] = 0.0

    n_with_tau = df["tau"].notna().sum()
    print(f"  {n_with_tau}/{len(df)} rows have τ values ({100*n_with_tau/len(df):.1f}%)")

    # --- Compute disagreement ---
    print("\nComputing disagreement metrics...")
    df, judges = compute_disagreement(df)
    print(f"  Judges available: {judges}")
    print(f"  Overall binary disagreement rate: {df['binary_disagree_any'].mean():.4f}")
    print(f"  Overall mean ordinal disagreement: {df['ordinal_disagree_mean'].mean():.4f}")

    # --- Run analyses ---
    summary = {}

    # Only analyse rows with tau values
    df_with_tau = df[df["tau"].notna()].copy()
    df_with_tau["abs_tau"] = df_with_tau["tau"].abs()
    print(f"\n  Analysing {len(df_with_tau)} rows with τ values")

    r1 = analysis_1_tau_vs_disagreement(df_with_tau, args.out_dir)
    summary.update(r1)

    # For language analysis, use all data (including English as reference)
    df["abs_tau"] = df["tau"].abs().fillna(0)
    r2 = analysis_2_language_stratification(df, args.out_dir)
    summary.update(r2)

    r3 = {}
    if top_tau_df is not None:
        r3 = analysis_3_category_pattern(df_with_tau, top_tau_df, args.out_dir)
    else:
        print("\n=== Analysis 3: SKIPPED (no tag source provided) ===")
        print("  Pass --multijail-csv or --top-tau-csv to enable category analysis.")
    summary.update(r3)

    r4 = analysis_4_within_language_tau_disagree(df_with_tau, args.out_dir)
    summary.update(r4)

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY: Judge Artifact Analysis")
    print("=" * 72)

    verdict_tau = "NO" if (summary.get("p_tau_disagree", 0) > 0.05 or
                           abs(summary.get("rho_tau_disagree", 0)) < 0.1) else "MAYBE"
    print(f"  |τ| predicts disagreement? → {verdict_tau}")
    print(f"    ρ(|τ|, disagree) = {summary.get('rho_tau_disagree', 'N/A')}")

    verdict_lang = "NO" if (summary.get("p_lang_tau_disagree", 0) > 0.05 or
                            abs(summary.get("rho_lang_tau_disagree", 0)) < 0.3) else "MAYBE"
    print(f"  Language-level: |τ| tracks disagreement? → {verdict_lang}")
    print(f"    ρ(mean |τ|, disagree by lang) = {summary.get('rho_lang_tau_disagree', 'N/A')}")

    if "mean_within_lang_rho" in summary:
        print(f"  Within-language mean ρ(|τ|, disagree) = {summary['mean_within_lang_rho']:.4f}")
        print(f"    Significant languages: {summary.get('n_significant', 'N/A')}/9")

    # Write summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.out_dir, "judge_artifact_summary.csv"), index=False)

    print(f"\n  All results saved to {args.out_dir}/")
    print("  Figures: judge_artifact_*.pdf / .png")


if __name__ == "__main__":
    # Allow running with args or with defaults for debugging
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python judge_artifact_analysis.py --claude-csv ... --gemini-csv ... --irt-csv ... [--top-tau-csv ...]")
        print("\nRun with --help for full options.")
        sys.exit(0)