#!/usr/bin/env python3
"""
Unidimensionality Robustness: Q3 residuals + Kendall's W
=========================================================
Addresses reviewer concern that EFA picks up "Refusal Bias" rather than
genuine unidimensional safety structure (Sanmi Koyejo critique).

Analysis 1: Yen's Q3 — residual correlations after removing the FULL IRT model.
  E[X] = σ(α_i × ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))
  Each (test_taker × language) is a separate "person" in Q3.
  If within-category Q3 ≈ between-category Q3, single factor suffices.

Analysis 2: Kendall's W — do models rank harm categories the same way?
  Low W = heterogeneous profiles → not a blunt refusal threshold.

Loads all parameters from irt.py output CSVs. Does NOT refit.

Required inputs (all from model/results/):
  --irt-csv       bayesian_irt_results_binary.csv  (prompt, language, β, τ, α)
  --theta-csv     theta_person_params.csv          (test_taker, θ)
  --delta-csv     delta_person_lang_params.csv      (test_taker, language, δ)
  --gamma-csv     gamma_lang_params.csv             (language, γ)
  --master-csv    Master_Passes0-9_Dataset.csv      (observed responses + tags)

Usage:
  python unidim_robustness.py
  # or override any path:
  python unidim_robustness.py --out-dir custom_output/
"""

import argparse
import ast
import os
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PATHS (same source as irt.py)
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
MASTER_CSV = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
# IRT outputs live in model/results/ relative to irt.py
IRT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "model", "results")
IRT_CSV = os.path.join(IRT_RESULTS_DIR, "bayesian_irt_results_binary.csv")
THETA_CSV = os.path.join(IRT_RESULTS_DIR, "theta_person_params.csv")
DELTA_CSV = os.path.join(IRT_RESULTS_DIR, "delta_person_params.csv")
GAMMA_CSV = os.path.join(IRT_RESULTS_DIR, "gamma_language_params.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG STYLE
# ═══════════════════════════════════════════════════════════════════════════════
FULL_WIDTH = 5.5
DPI = 300
C_BLUE = "#0072B2"
C_RED = "#D55E00"
C_GREY = "#999999"
CMAP_DIV = "RdBu_r"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.figsize": (FULL_WIDTH, 3.5),
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def _save(fig, path_stem):
    for ext in (".pdf", ".png"):
        fig.savefig(f"{path_stem}{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_tags(x):
    """Parse multi-label tags: \"['Theft', 'Fraud']\" → ['Theft', 'Fraud']."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        result = ast.literal_eval(x)
        return result if isinstance(result, list) else [str(result)]
    except (ValueError, SyntaxError):
        return [t.strip() for t in str(x).split(",") if t.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — all from irt.py output CSVs
# ═══════════════════════════════════════════════════════════════════════════════

def load_all(master_csv, irt_csv, theta_csv, delta_csv, gamma_csv):
    """
    Load observed data + all IRT parameters from their respective CSVs.

    Returns:
      raw_df      — master dataset (for observations and tags)
      student_col — column name for test-takers
      id_to_cats  — {prompt_id: [cat1, cat2, ...]}
      params      — dict of all IRT parameter lookups
    """
    # ── Master dataset ──
    print("  Loading master dataset...")
    df = pd.read_csv(master_csv, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['safe'] = (df['judge_score'] >= 4).astype(float)
    df['id'] = df['id'].apply(clean_id)
    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    print(f"  {len(df)} rows, {df[student_col].nunique()} test-takers, "
          f"{df['id'].nunique()} prompts, {df['language'].nunique()} languages")

    # ── θ_j from theta_person_params.csv ──
    theta_df = pd.read_csv(theta_csv)
    theta_dict = dict(zip(theta_df['test_taker'], theta_df['theta']))
    print(f"  θ: {len(theta_dict)} test-takers")

    # ── γ_L from gamma_lang_params.csv ──
    gamma_df = pd.read_csv(gamma_csv)
    gamma_dict = dict(zip(gamma_df['language'], gamma_df['gamma']))
    print(f"  γ: {gamma_dict}")

    # ── δ_jL from delta_person_lang_params.csv ──
    delta_df = pd.read_csv(delta_csv)
    delta_dict = {}
    for _, row in delta_df.iterrows():
        delta_dict[(row['test_taker'], row['language'])] = row['delta']
    print(f"  δ: {len(delta_dict)} (test_taker, language) pairs")

    # ── β_i, α_i, τ_iL from bayesian_irt_results_binary.csv ──
    irt_df = pd.read_csv(irt_csv)
    # Normalise column names
    col_map = {}
    for c in irt_df.columns:
        cl = c.lower().strip()
        if cl in ("prompt", "id"): col_map[c] = "id"
        elif cl == "language": col_map[c] = "language"
        elif cl in ("base_difficulty",): col_map[c] = "beta"
        elif cl in ("safety_tax",): col_map[c] = "tau"
        elif cl == "alpha": col_map[c] = "alpha"
    irt_df = irt_df.rename(columns=col_map)
    irt_df['id'] = irt_df['id'].apply(clean_id)

    beta_dict = irt_df.groupby('id')['beta'].first().to_dict()
    alpha_dict = irt_df.groupby('id')['alpha'].first().to_dict()
    tau_dict = {}
    for _, row in irt_df.iterrows():
        tau_dict[(row['id'], row['language'])] = row['tau']
    print(f"  β: {len(beta_dict)} items, τ: {len(tau_dict)} (item, lang) pairs")

    # ── Categories from tags column ──
    id_to_cats = {}
    if 'tags' in df.columns:
        tag_df = df[['id', 'tags']].drop_duplicates(subset='id')
        for _, row in tag_df.iterrows():
            id_to_cats[row['id']] = parse_tags(row['tags'])
        n_multi = sum(1 for v in id_to_cats.values() if len(v) > 1)
        print(f"  Categories: {len(id_to_cats)} prompts, {n_multi} multi-label")

    params = {
        'theta': theta_dict,   # test_taker → θ
        'gamma': gamma_dict,   # language → γ
        'delta': delta_dict,   # (test_taker, language) → δ
        'beta': beta_dict,     # prompt_id → β
        'alpha': alpha_dict,   # prompt_id → α
        'tau': tau_dict,       # (prompt_id, language) → τ
    }

    return df, student_col, id_to_cats, params


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: YEN'S Q3 (FULL MODEL)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_q3(raw_df, student_col, id_to_cats, params, out_dir):
    """
    Yen's Q3 using the FULL multi-group 2PL IRT model.

    Each (test_taker, language) = one "person". For each observation:
      E[X] = σ(α_i × ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))
      residual = observed − E[X]

    Q3(i, k) = Pearson corr(residual_i, residual_k) across all persons.
    """
    print("\n=== Analysis 1: Yen's Q3 Residual Correlations ===")
    print("  E[X] = σ(α_i × ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))")

    theta = params['theta']
    gamma = params['gamma']
    delta = params['delta']
    beta = params['beta']
    alpha = params['alpha']
    tau = params['tau']

    df = raw_df.copy()

    # Aggregate to mean P(safe) per (test_taker, language, prompt) across passes
    agg = df.groupby([student_col, 'language', 'id'])['safe'].mean().reset_index()
    print(f"  Raw observations: {len(agg)}")

    # Filter to rows with all parameters available
    agg = agg[agg['id'].isin(beta)]
    agg = agg[agg[student_col].isin(theta)]
    valid_langs = set(gamma.keys()) | {'en'}
    agg = agg[agg['language'].isin(valid_langs)]
    print(f"  After param filtering: {len(agg)}")

    # Compute expected value using full model (vectorised via lookup)
    agg['alpha'] = agg['id'].map(alpha)
    agg['theta'] = agg[student_col].map(theta)
    agg['beta'] = agg['id'].map(beta)
    agg['gamma'] = agg['language'].map(gamma).fillna(0.0)
    agg['tau'] = agg.apply(lambda r: tau.get((r['id'], r['language']), 0.0), axis=1)
    agg['delta'] = agg.apply(lambda r: delta.get((r[student_col], r['language']), 0.0), axis=1)

    agg['logit'] = agg['alpha'] * ((agg['theta'] + agg['delta']) -
                                    (agg['beta'] + agg['gamma'] + agg['tau']))
    agg['expected'] = 1.0 / (1.0 + np.exp(-agg['logit']))
    agg['residual'] = agg['safe'] - agg['expected']

    # Person = (test_taker, language)
    agg['person'] = agg[student_col] + "::" + agg['language']
    n_persons = agg['person'].nunique()
    n_items = agg['id'].nunique()
    print(f"  Q3 matrix: {n_persons} persons × {n_items} items")

    # Pivot: rows=persons, cols=prompts, values=residuals
    resid_matrix = agg.pivot_table(index='person', columns='id',
                                    values='residual', aggfunc='mean')

    # Drop zero-variance items
    item_std = resid_matrix.std()
    resid_matrix = resid_matrix.loc[:, item_std > 1e-8]
    valid_prompts = resid_matrix.columns.tolist()
    print(f"  After variance filter: {len(valid_prompts)} items")

    # Q3 = item×item correlation across persons
    resid_filled = resid_matrix.fillna(0).values
    q3_matrix = np.corrcoef(resid_filled.T)
    np.fill_diagonal(q3_matrix, np.nan)

    # Classify pairs: "within" if they share ANY tag
    within_q3, between_q3 = [], []
    cat_pair_q3 = {}

    for i, j in combinations(range(len(valid_prompts)), 2):
        val = q3_matrix[i, j]
        if np.isnan(val):
            continue
        p1, p2 = valid_prompts[i], valid_prompts[j]
        cats1 = set(id_to_cats.get(p1, []))
        cats2 = set(id_to_cats.get(p2, []))
        if not cats1 or not cats2:
            continue

        if cats1 & cats2:
            within_q3.append(val)
        else:
            between_q3.append(val)

        for c1 in cats1:
            for c2 in cats2:
                key = tuple(sorted([c1, c2]))
                cat_pair_q3.setdefault(key, []).append(val)

    within_q3 = np.array(within_q3)
    between_q3 = np.array(between_q3)

    print(f"\n  Within-category Q3:  mean={within_q3.mean():.4f}, "
          f"median={np.median(within_q3):.4f}, n={len(within_q3)}")
    print(f"  Between-category Q3: mean={between_q3.mean():.4f}, "
          f"median={np.median(between_q3):.4f}, n={len(between_q3)}")

    diff = within_q3.mean() - between_q3.mean()
    u_stat, u_p = stats.mannwhitneyu(within_q3, between_q3, alternative='two-sided')
    pooled_sd = np.sqrt((within_q3.std()**2 + between_q3.std()**2) / 2)
    d = diff / pooled_sd if pooled_sd > 0 else 0
    print(f"  Δ = {diff:.4f}, Cohen's d = {d:.4f}, p = {u_p:.4g}")

    # ── Save CSVs ──
    pd.DataFrame([
        {"group": "within", "mean": within_q3.mean(), "median": np.median(within_q3),
         "sd": within_q3.std(), "n": len(within_q3)},
        {"group": "between", "mean": between_q3.mean(), "median": np.median(between_q3),
         "sd": between_q3.std(), "n": len(between_q3)},
    ]).to_csv(os.path.join(out_dir, "q3_within_vs_between.csv"), index=False)

    all_cats = sorted({c for cats in id_to_cats.values() for c in cats})
    cat_matrix = pd.DataFrame(np.nan, index=all_cats, columns=all_cats)
    for (c1, c2), vals in cat_pair_q3.items():
        m = np.mean(vals)
        cat_matrix.loc[c1, c2] = m
        cat_matrix.loc[c2, c1] = m
    cat_matrix.to_csv(os.path.join(out_dir, "q3_category_pair_matrix.csv"))

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))

    lo = min(within_q3.min(), between_q3.min())
    hi = max(within_q3.max(), between_q3.max())
    bins = np.linspace(lo, hi, 40)
    ax1.hist(between_q3, bins=bins, alpha=0.6, color=C_BLUE, density=True,
             edgecolor='white', linewidth=0.3, label="Between-category")
    ax1.hist(within_q3, bins=bins, alpha=0.6, color=C_RED, density=True,
             edgecolor='white', linewidth=0.3, label="Within-category")
    ax1.axvline(between_q3.mean(), color=C_BLUE, ls='--', lw=0.8)
    ax1.axvline(within_q3.mean(), color=C_RED, ls='--', lw=0.8)
    ax1.set_xlabel("Q3 residual correlation")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Q3 distributions")
    ax1.legend(loc='upper right', framealpha=0.8)
    ax1.text(0.02, 0.95, f"Δ = {diff:.3f}\nd = {d:.3f}",
             transform=ax1.transAxes, va='top', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GREY, alpha=0.8))

    cats_show = [c for c in all_cats if cat_matrix.loc[c].notna().sum() > 1]
    sub = cat_matrix.loc[cats_show, cats_show].astype(float)
    short = [c[:20] for c in cats_show]
    im = ax2.imshow(sub.values, cmap=CMAP_DIV, aspect='auto', vmin=-0.15, vmax=0.15)
    ax2.set_xticks(range(len(short)))
    ax2.set_xticklabels(short, rotation=90, fontsize=4)
    ax2.set_yticks(range(len(short)))
    ax2.set_yticklabels(short, fontsize=4)
    ax2.set_title("(b) Q3 by category pair")
    plt.colorbar(im, ax=ax2, shrink=0.8, label="Mean Q3")

    fig.tight_layout(w_pad=1.5)
    _save(fig, os.path.join(out_dir, "unidim_q3_residuals"))

    return {"q3_within": within_q3.mean(), "q3_between": between_q3.mean(),
            "q3_diff": diff, "cohens_d": d, "mann_whitney_p": u_p}


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: KENDALL'S W
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kendall_w(raw_df, student_col, id_to_cats, out_dir):
    """
    Kendall's W: concordance of category rankings across test-takers.
    Low W = models disagree on which categories are hard → not a blunt threshold.
    Explodes multi-label tags.
    """
    print("\n=== Analysis 2: Kendall's W Category Concordance ===")

    df = raw_df.copy()

    # Explode multi-label tags
    df['tag_list'] = df['id'].map(id_to_cats)
    df = df.dropna(subset=['tag_list'])
    df = df[df['tag_list'].apply(len) > 0]
    df = df.explode('tag_list').rename(columns={'tag_list': 'category'})

    # Safe rate per (test_taker, category)
    cat_rates = df.groupby([student_col, 'category'])['safe'].mean().reset_index()
    rate_matrix = cat_rates.pivot(index=student_col, columns='category', values='safe')
    rate_matrix = rate_matrix.dropna(axis=1, how='all').dropna(axis=0, how='all')

    cats = rate_matrix.columns.tolist()
    n_judges = len(rate_matrix)
    n_items = len(cats)
    print(f"  {n_judges} test-takers × {n_items} categories")

    # Rank (1 = lowest safe rate = hardest to refuse)
    rank_matrix = rate_matrix.rank(axis=1, method='average')

    rank_sums = rank_matrix.sum(axis=0)
    S = ((rank_sums - rank_sums.mean()) ** 2).sum()
    W = 12 * S / (n_judges ** 2 * (n_items ** 3 - n_items))

    chi2 = n_judges * (n_items - 1) * W
    df_chi = n_items - 1
    p_val = 1 - stats.chi2.cdf(chi2, df_chi)

    print(f"  Kendall's W = {W:.4f}")
    print(f"  χ² = {chi2:.1f}, df = {df_chi}, p = {p_val:.4g}")

    results = pd.DataFrame({
        'category': cats,
        'mean_safe_rate': rate_matrix.mean(axis=0),
        'median_rank': rank_matrix.median(axis=0),
        'rank_iqr': rank_matrix.quantile(0.75, axis=0) - rank_matrix.quantile(0.25, axis=0),
    }).sort_values('median_rank')
    results.to_csv(os.path.join(out_dir, "kendall_w_results.csv"), index=False)
    print(f"\n  Category rankings:")
    print(results.to_string(index=False))

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    sorted_cats = results['category'].values
    box_data = [rank_matrix[cat].dropna().values for cat in sorted_cats]
    bp = ax.boxplot(box_data, vert=True, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color=C_RED, linewidth=1.2))
    for patch in bp['boxes']:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.5)
    ax.set_xticks(range(1, len(sorted_cats) + 1))
    ax.set_xticklabels([c[:22] for c in sorted_cats], rotation=60, ha='right', fontsize=5)
    ax.set_ylabel("Rank across test-takers")
    ax.set_title(f"Category rank variability (Kendall's W = {W:.3f})")
    ax.text(0.98, 0.95, f"W = {W:.3f}\np < {max(p_val, 1e-10):.2g}",
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_GREY, alpha=0.8))

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "unidim_kendall_w"))

    return {"kendall_w": W, "chi2": chi2, "p_val": p_val}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unidimensionality Robustness: Q3 + Kendall's W")
    parser.add_argument("--master-csv", default=MASTER_CSV)
    parser.add_argument("--irt-csv", default=IRT_CSV)
    parser.add_argument("--theta-csv", default=THETA_CSV)
    parser.add_argument("--delta-csv", default=DELTA_CSV)
    parser.add_argument("--gamma-csv", default=GAMMA_CSV)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_df, student_col, id_to_cats, params = load_all(
        args.master_csv, args.irt_csv, args.theta_csv,
        args.delta_csv, args.gamma_csv)

    r1 = compute_q3(raw_df, student_col, id_to_cats, params, args.out_dir)
    r2 = compute_kendall_w(raw_df, student_col, id_to_cats, args.out_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Q3: within={r1['q3_within']:.4f}, between={r1['q3_between']:.4f}, "
          f"Δ={r1['q3_diff']:.4f}, d={r1['cohens_d']:.4f}")
    print(f"  Kendall's W = {r2['kendall_w']:.4f}")

    holds = r1['q3_diff'] < 0.05 and r2['kendall_w'] < 0.7
    print(f"  VERDICT: Unidimensionality {'SUPPORTED' if holds else 'NEEDS DISCUSSION'}")

    pd.DataFrame([{**r1, **r2}]).to_csv(
        os.path.join(args.out_dir, "unidim_robustness_summary.csv"), index=False)
    print(f"\n  Results → {args.out_dir}/")


if __name__ == "__main__":
    main()