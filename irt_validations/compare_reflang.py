# -*- coding: utf-8 -*-
"""
Reference Language Sensitivity Analysis
========================================
Fits the 2PL IRT model with different reference languages and compares:
  1. θ rank correlation (Spearman + Pearson)
  2. β rank correlation
  3. Top-100 τ overlap (Jaccard)
  4. Top-100 τ overlap by direction (positive-only, negative-only)
  5. γ value comparison
  6. English-reversal count stability

Usage:
    python coreflang.py
    python coreflang.py --langs en zh ar
    python coreflang.py --skip-training  # use cached results
"""

import argparse
import os
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

RESULTS_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_results(ref_lang):
    """Load saved results for a given reference language."""
    d = os.path.join(RESULTS_ROOT, f"results_ref_{ref_lang}")
    if not os.path.exists(d):
        raise FileNotFoundError(f"No results for ref={ref_lang}. Run reflang.py --ref-lang {ref_lang} first.")

    theta = pd.read_csv(os.path.join(d, f"theta_person_params_ref_{ref_lang}.csv"))
    gamma = pd.read_csv(os.path.join(d, f"gamma_language_params_ref_{ref_lang}.csv"))
    results = pd.read_csv(os.path.join(d, f"bayesian_irt_results_binary_ref_{ref_lang}.csv"))
    tau_matrix = pd.read_csv(os.path.join(d, f"tau_matrix_ref_{ref_lang}.csv"), index_col=0)

    return {
        'ref_lang': ref_lang,
        'theta': theta,
        'gamma': gamma,
        'results': results,
        'tau_matrix': tau_matrix,
    }


def top_k_tau_pairs(results_df, k=100, direction="positive"):
    """Get top-k prompt×language pairs by τ magnitude."""
    df = results_df.copy()
    if direction == "positive":
        df = df[df["Safety_Tax"] > 0].nlargest(k, "Safety_Tax")
    elif direction == "negative":
        df = df[df["Safety_Tax"] < 0].nsmallest(k, "Safety_Tax")
    else:  # absolute
        df["abs_tau"] = df["Safety_Tax"].abs()
        df = df.nlargest(k, "abs_tau")
    return set(zip(df["prompt"], df["language"]))


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_theta(data_a, data_b):
    """Compare θ estimates across two reference languages."""
    merged = data_a['theta'].merge(data_b['theta'],
                                    on='test_taker', suffixes=('_a', '_b'))
    rho, p_rho = spearmanr(merged['theta_a'], merged['theta_b'])
    r, p_r = pearsonr(merged['theta_a'], merged['theta_b'])

    # Rank displacement
    merged['rank_a'] = merged['theta_a'].rank(ascending=False)
    merged['rank_b'] = merged['theta_b'].rank(ascending=False)
    merged['rank_shift'] = (merged['rank_a'] - merged['rank_b']).abs()

    return {
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'pearson_r': r,
        'pearson_p': p_r,
        'mean_rank_shift': merged['rank_shift'].mean(),
        'max_rank_shift': merged['rank_shift'].max(),
        'n': len(merged),
    }


def compare_gamma(data_a, data_b):
    """Compare γ estimates (languages present in both)."""
    merged = data_a['gamma'].merge(data_b['gamma'],
                                    on='language', suffixes=('_a', '_b'))
    # Exclude zero-constrained reference languages
    merged = merged[(merged['gamma_L_a'] != 0) | (merged['gamma_L_b'] != 0)]
    if len(merged) < 3:
        return {'note': 'too few shared non-reference languages'}

    rho, p = spearmanr(merged['gamma_L_a'], merged['gamma_L_b'])
    r, p_r = pearsonr(merged['gamma_L_a'], merged['gamma_L_b'])
    return {
        'spearman_rho': rho,
        'spearman_p': p,
        'pearson_r': r,
        'pearson_p': p_r,
        'n': len(merged),
        'details': merged.to_dict('records'),
    }


def compare_tau_overlap(data_a, data_b, k=100):
    """Compare top-k τ overlap across reference languages."""
    top_pos_a = top_k_tau_pairs(data_a['results'], k, "positive")
    top_pos_b = top_k_tau_pairs(data_b['results'], k, "positive")

    top_neg_a = top_k_tau_pairs(data_a['results'], k, "negative")
    top_neg_b = top_k_tau_pairs(data_b['results'], k, "negative")

    top_abs_a = top_k_tau_pairs(data_a['results'], k, "absolute")
    top_abs_b = top_k_tau_pairs(data_b['results'], k, "absolute")

    return {
        'top100_positive_jaccard': jaccard(top_pos_a, top_pos_b),
        'top100_positive_overlap': len(top_pos_a & top_pos_b),
        'top100_negative_jaccard': jaccard(top_neg_a, top_neg_b),
        'top100_negative_overlap': len(top_neg_a & top_neg_b),
        'top100_absolute_jaccard': jaccard(top_abs_a, top_abs_b),
        'top100_absolute_overlap': len(top_abs_a & top_abs_b),
    }


def compare_tau_correlation(data_a, data_b):
    """Correlate τ values across shared prompt×language pairs."""
    merged = data_a['results'].merge(data_b['results'],
                                      on=['prompt', 'language'],
                                      suffixes=('_a', '_b'))
    if len(merged) < 10:
        return {'note': 'insufficient shared pairs'}

    rho, p = spearmanr(merged['Safety_Tax_a'], merged['Safety_Tax_b'])
    r, p_r = pearsonr(merged['Safety_Tax_a'], merged['Safety_Tax_b'])
    return {
        'spearman_rho': rho,
        'pearson_r': r,
        'n_pairs': len(merged),
    }


def count_english_reversal(data):
    """Count model configs where English has highest JSR.
    Note: only meaningful when English is NOT the reference language.
    """
    # This is approximate — uses Safety_Tax as proxy
    # For proper count, need raw JSR data
    return {'ref_lang': data['ref_lang'], 'note': 'requires raw JSR, not τ-based'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["en", "zh", "it", "ar"],
                        help="Reference languages to compare")
    parser.add_argument("--skip-training", action="store_true",
                        help="Use cached results only (don't train)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top τ pairs to compare")
    args = parser.parse_args()

    # ── Optionally train ──
    if not args.skip_training:
        from reflang import train_and_extract
        for lang in args.langs:
            print(f"\n{'='*60}")
            print(f"  Training with reference: {lang}")
            print(f"{'='*60}")
            train_and_extract(ref_lang=lang)

    # ── Load all results ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {' vs '.join(args.langs)}")
    print(f"{'='*60}\n")

    all_data = {}
    for lang in args.langs:
        try:
            all_data[lang] = load_results(lang)
            print(f"  Loaded ref={lang}: {len(all_data[lang]['theta'])} test-takers, "
                  f"{len(all_data[lang]['results'])} prompt×lang pairs")
        except FileNotFoundError as e:
            print(f"  SKIP {lang}: {e}")

    if len(all_data) < 2:
        print("Need at least 2 reference languages. Exiting.")
        return

    # ── Pairwise comparisons ──
    pairs = list(itertools.combinations(all_data.keys(), 2))
    summary_rows = []

    for lang_a, lang_b in pairs:
        print(f"\n--- {lang_a} vs {lang_b} ---")
        da, db = all_data[lang_a], all_data[lang_b]

        # θ comparison
        theta_cmp = compare_theta(da, db)
        print(f"  θ: Spearman ρ = {theta_cmp['spearman_rho']:.4f}, "
              f"Pearson r = {theta_cmp['pearson_r']:.4f}, "
              f"mean rank shift = {theta_cmp['mean_rank_shift']:.2f}, "
              f"max rank shift = {theta_cmp['max_rank_shift']:.0f}")

        # γ comparison
        gamma_cmp = compare_gamma(da, db)
        if 'spearman_rho' in gamma_cmp:
            print(f"  γ: Spearman ρ = {gamma_cmp['spearman_rho']:.4f}, "
                  f"Pearson r = {gamma_cmp['pearson_r']:.4f} "
                  f"(n={gamma_cmp['n']} shared languages)")
        else:
            print(f"  γ: {gamma_cmp.get('note', 'N/A')}")

        # τ correlation
        tau_corr = compare_tau_correlation(da, db)
        if 'spearman_rho' in tau_corr:
            print(f"  τ correlation: Spearman ρ = {tau_corr['spearman_rho']:.4f}, "
                  f"Pearson r = {tau_corr['pearson_r']:.4f} "
                  f"(n={tau_corr['n_pairs']} pairs)")

        # τ top-k overlap
        tau_overlap = compare_tau_overlap(da, db, k=args.top_k)
        print(f"  τ top-{args.top_k} overlap:")
        print(f"    Positive: {tau_overlap['top100_positive_overlap']}/{args.top_k} "
              f"(Jaccard = {tau_overlap['top100_positive_jaccard']:.3f})")
        print(f"    Negative: {tau_overlap['top100_negative_overlap']}/{args.top_k} "
              f"(Jaccard = {tau_overlap['top100_negative_jaccard']:.3f})")
        print(f"    Absolute: {tau_overlap['top100_absolute_overlap']}/{args.top_k} "
              f"(Jaccard = {tau_overlap['top100_absolute_jaccard']:.3f})")

        summary_rows.append({
            'ref_a': lang_a, 'ref_b': lang_b,
            'theta_spearman': theta_cmp['spearman_rho'],
            'theta_pearson': theta_cmp['pearson_r'],
            'theta_mean_rank_shift': theta_cmp['mean_rank_shift'],
            'theta_max_rank_shift': theta_cmp['max_rank_shift'],
            'gamma_spearman': gamma_cmp.get('spearman_rho', np.nan),
            'gamma_pearson': gamma_cmp.get('pearson_r', np.nan),
            'tau_spearman': tau_corr.get('spearman_rho', np.nan),
            'tau_pearson': tau_corr.get('pearson_r', np.nan),
            'tau_top100_pos_overlap': tau_overlap['top100_positive_overlap'],
            'tau_top100_pos_jaccard': tau_overlap['top100_positive_jaccard'],
            'tau_top100_abs_overlap': tau_overlap['top100_absolute_overlap'],
            'tau_top100_abs_jaccard': tau_overlap['top100_absolute_jaccard'],
        })

    # ── Summary table ──
    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(RESULTS_ROOT, "ref_lang_sensitivity_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Summary saved: {out_path}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # ── Interpretation ──
    print(f"\n--- Interpretation Guide ---")
    mean_theta_rho = summary_df['theta_spearman'].mean()
    mean_tau_jaccard = summary_df['tau_top100_pos_jaccard'].mean()
    print(f"  Mean θ Spearman across pairs: {mean_theta_rho:.4f}")
    print(f"  Mean top-100 τ Jaccard:       {mean_tau_jaccard:.3f}")
    if mean_theta_rho > 0.95:
        print("  → θ rankings are robust to reference language choice.")
    elif mean_theta_rho > 0.85:
        print("  → θ rankings are moderately sensitive to reference language.")
    else:
        print("  → θ rankings are sensitive to reference language — investigate further.")
    if mean_tau_jaccard > 0.70:
        print("  → Top τ pairs are largely stable across reference languages.")
    elif mean_tau_jaccard > 0.50:
        print("  → Top τ pairs show moderate stability.")
    else:
        print("  → Top τ pairs are sensitive to reference language — caution warranted.")


if __name__ == "__main__":
    main()
