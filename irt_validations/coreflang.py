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
    python model/coreflang.py --skip-training  # use cached results
"""

import argparse
import ast
import os
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

RESULTS_ROOT = os.path.dirname(os.path.abspath(__file__))


def _clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def load_multijail_categories():
    """Load multijail.csv and return long-form (id, category) DataFrame."""
    candidates = [
        os.path.join(RESULTS_ROOT, "multijail.csv"),
        os.path.join(os.path.dirname(RESULTS_ROOT), "multijail.csv"),
        os.path.join(os.path.dirname(RESULTS_ROOT), "safety-data-mirror", "multijail.csv"),
    ]
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
        candidates.append(os.path.join(d, "multijail.csv"))
    except Exception:
        pass

    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"multijail.csv not found. Tried: {candidates}")

    df = pd.read_csv(path).drop_duplicates(subset=['id'])
    df['id'] = df['id'].apply(_clean_id)
    df['categories'] = df['tags'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith('[') else []
    )
    long = (df[['id', 'categories']]
            .explode('categories')
            .rename(columns={'categories': 'category'})
            .dropna(subset=['category']))
    print(f"  Loaded categories from: {path} "
          f"({len(long)} (id, category) rows, {long['category'].nunique()} unique categories)")
    return long


def compare_per_category_vs_en(all_data, cat_df, top_k=20, min_n_pairs=10):
    """Per-category mean τ (English ref) and per-category top-k |τ| Jaccard vs English."""
    if 'en' not in all_data:
        print("  English reference not loaded; skipping category analysis.")
        return

    data_en = all_data['en']
    other_langs = [l for l in all_data if l != 'en']

    cat_df = cat_df.copy()
    cat_df['id'] = cat_df['id'].astype(str)

    # ── Descriptive: mean τ per category in English-ref fit ──
    en_res = data_en['results'][['prompt', 'language', 'Safety_Tax']].copy()
    en_res['prompt'] = en_res['prompt'].apply(_clean_id)
    en_long = en_res.merge(cat_df, left_on='prompt', right_on='id', how='inner')

    cat_desc = en_long.groupby('category').agg(
        n_pairs=('Safety_Tax', 'size'),
        mean_tau=('Safety_Tax', 'mean'),
        median_tau=('Safety_Tax', 'median'),
        mean_abs_tau=('Safety_Tax', lambda x: x.abs().mean()),
    ).reset_index().sort_values('mean_abs_tau', ascending=False)

    print(f"\n--- Per-category τ statistics (English reference) ---")
    print(f"  {'category':<55} | {'n_pairs':>7} | {'mean τ':>8} | {'mean |τ|':>9}")
    print(f"  {'-'*55}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}")
    for _, r in cat_desc.iterrows():
        label = (r['category'] or '<missing>')[:55]
        print(f"  {label:<55} | {int(r['n_pairs']):>7} | {r['mean_tau']:>+8.3f} | "
              f"{r['mean_abs_tau']:>9.3f}")

    desc_path = os.path.join(RESULTS_ROOT, "category_mean_tau_en_ref.csv")
    cat_desc.to_csv(desc_path, index=False)
    print(f"  Saved: {desc_path}")

    if not other_langs:
        return

    # ── Sensitivity: per (other_ref, category), top-k |τ| Jaccard vs English ──
    sens_rows = []
    for other in other_langs:
        data_other = all_data[other]
        merged = data_en['results'][['prompt', 'language', 'Safety_Tax']].merge(
            data_other['results'][['prompt', 'language', 'Safety_Tax']],
            on=['prompt', 'language'], suffixes=('_en', '_other'))
        merged['prompt'] = merged['prompt'].apply(_clean_id)
        merged_cat = merged.merge(cat_df, left_on='prompt', right_on='id', how='inner')
        merged_cat['abs_en'] = merged_cat['Safety_Tax_en'].abs()
        merged_cat['abs_other'] = merged_cat['Safety_Tax_other'].abs()

        for cat, sub in merged_cat.groupby('category'):
            n = len(sub)
            if n < min_n_pairs:
                continue
            k_use = min(top_k, max(2, n // 2))
            top_en_set = set(zip(sub.nlargest(k_use, 'abs_en')['prompt'],
                                  sub.nlargest(k_use, 'abs_en')['language']))
            top_oth_set = set(zip(sub.nlargest(k_use, 'abs_other')['prompt'],
                                   sub.nlargest(k_use, 'abs_other')['language']))
            sens_rows.append({
                'ref_other': other,
                'category': cat,
                'n_pairs': n,
                'k_used': k_use,
                'mean_tau_en': sub['Safety_Tax_en'].mean(),
                'mean_tau_other': sub['Safety_Tax_other'].mean(),
                'jaccard': jaccard(top_en_set, top_oth_set),
                'overlap': len(top_en_set & top_oth_set),
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_path = os.path.join(RESULTS_ROOT, "category_sensitivity_vs_en.csv")
    sens_df.to_csv(sens_path, index=False)

    agg = sens_df.groupby('category').agg(
        n_pairs=('n_pairs', 'first'),
        k_used=('k_used', 'first'),
        mean_jaccard=('jaccard', 'mean'),
        mean_overlap=('overlap', 'mean'),
    ).reset_index().sort_values('mean_jaccard')

    print(f"\n--- Per-category top-k |τ| Jaccard vs English "
          f"(mean across {len(other_langs)} non-en refs) ---")
    print(f"  {'category':<55} | {'n':>5} | {'k':>3} | {'mean Jaccard':>13} | "
          f"{'mean overlap':>13}")
    print(f"  {'-'*55}-+-{'-'*5}-+-{'-'*3}-+-{'-'*13}-+-{'-'*13}")
    for _, r in agg.iterrows():
        label = (r['category'] or '<missing>')[:55]
        print(f"  {label:<55} | {int(r['n_pairs']):>5} | {int(r['k_used']):>3} | "
              f"{r['mean_jaccard']:>13.3f} | {r['mean_overlap']:>13.1f}")
    print(f"\n  Per-(other_ref, category) detail saved: {sens_path}")


def load_results(ref_lang):
    """Load saved results for a given reference language."""
    d = os.path.join(RESULTS_ROOT, f"results_ref_{ref_lang}")
    if not os.path.exists(d):
        raise FileNotFoundError(f"No results for ref={ref_lang}. Run reflang.py --ref-lang {ref_lang} first.")

    theta = pd.read_csv(os.path.join(d, f"theta_person_params_ref_{ref_lang}.csv"))
    gamma = pd.read_csv(os.path.join(d, f"gamma_language_params_ref_{ref_lang}.csv"))
    results = pd.read_csv(os.path.join(d, f"bayesian_irt_results_binary_ref_{ref_lang}.csv"))
    tau_matrix = pd.read_csv(os.path.join(d, f"tau_matrix_ref_{ref_lang}.csv"), index_col=0)

    delta_path = os.path.join(d, f"delta_person_params_ref_{ref_lang}.csv")
    delta = pd.read_csv(delta_path) if os.path.exists(delta_path) else None

    return {
        'ref_lang': ref_lang,
        'theta': theta,
        'gamma': gamma,
        'results': results,
        'tau_matrix': tau_matrix,
        'delta': delta,
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
    """Deprecated stub — superseded by english_reversal_counts(all_data) which uses θ + δ."""
    return {'ref_lang': data['ref_lang'], 'note': 'see english_reversal_counts'}


def compare_category_rankings_vs_en(all_data, cat_df, min_n_pairs=10):
    """Spearman rank correlation of per-category mean τ across reference languages.

    Backs the Appendix K.2 claim that the categorical ordering (theft/weapons high,
    discrimination low) is preserved when the reference language changes.
    """
    if 'en' not in all_data:
        print("  English reference not loaded; skipping category-ranking sensitivity.")
        return

    cat_df = cat_df.copy()
    cat_df['id'] = cat_df['id'].astype(str)
    other_langs = [l for l in all_data if l != 'en']

    def per_cat_mean_tau(ref_lang):
        res = all_data[ref_lang]['results'][['prompt', 'language', 'Safety_Tax']].copy()
        res['prompt'] = res['prompt'].apply(_clean_id)
        long = res.merge(cat_df, left_on='prompt', right_on='id', how='inner')
        agg = long.groupby('category')['Safety_Tax'].agg(['mean', 'size']).reset_index()
        agg = agg[agg['size'] >= min_n_pairs]
        return agg.set_index('category')['mean']

    en_means = per_cat_mean_tau('en')

    sens_rows = []
    rank_table = {'en': en_means.rank(ascending=False)}
    for other in other_langs:
        other_means = per_cat_mean_tau(other)
        common = en_means.index.intersection(other_means.index)
        if len(common) < 5:
            continue
        en_vals = en_means.loc[common].values
        other_vals = other_means.loc[common].values
        rho, p_rho = spearmanr(en_vals, other_vals)
        r, p_r = pearsonr(en_vals, other_vals)
        sens_rows.append({
            'ref_other': other,
            'n_categories': len(common),
            'spearman_rho': float(rho),
            'spearman_p': float(p_rho),
            'pearson_r': float(r),
            'pearson_p': float(p_r),
        })
        rank_table[other] = other_means.loc[common].rank(ascending=False)

    sens_df = pd.DataFrame(sens_rows)
    sens_path = os.path.join(RESULTS_ROOT, "category_ranking_sensitivity_vs_en.csv")
    sens_df.to_csv(sens_path, index=False)

    print(f"\n--- Per-category mean-τ rank correlation vs English ---")
    print(f"  {'ref_other':<12} | {'n_cats':>6} | {'Spearman ρ':>11} | {'p':>10} | {'Pearson r':>10}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}")
    for _, r in sens_df.iterrows():
        print(f"  {r['ref_other']:<12} | {int(r['n_categories']):>6} | "
              f"{r['spearman_rho']:>11.3f} | {r['spearman_p']:>10.2e} | {r['pearson_r']:>10.3f}")
    if len(sens_df):
        print(f"  Range: ρ ∈ [{sens_df['spearman_rho'].min():.3f}, {sens_df['spearman_rho'].max():.3f}], "
              f"mean = {sens_df['spearman_rho'].mean():.3f}")
    print(f"  Saved: {sens_path}")

    # Top-5 / bottom-5 stability table
    rank_df = pd.DataFrame(rank_table).dropna(how='any')
    if not rank_df.empty:
        top5_lists = {ref: set(rank_df[ref].nsmallest(5).index) for ref in rank_df.columns}
        bot5_lists = {ref: set(rank_df[ref].nlargest(5).index) for ref in rank_df.columns}
        en_top5 = top5_lists['en']
        en_bot5 = bot5_lists['en']
        top5_overlap = [len(en_top5 & top5_lists[r]) for r in rank_df.columns if r != 'en']
        bot5_overlap = [len(en_bot5 & bot5_lists[r]) for r in rank_df.columns if r != 'en']
        print(f"  Top-5 categories vs English: mean overlap = {np.mean(top5_overlap):.1f}/5 "
              f"(min {min(top5_overlap)}/5, max {max(top5_overlap)}/5)")
        print(f"  Bottom-5 categories vs English: mean overlap = {np.mean(bot5_overlap):.1f}/5 "
              f"(min {min(bot5_overlap)}/5, max {max(bot5_overlap)}/5)")

    # ── Overall categorical ranking: average per-category mean τ across all refs ──
    mean_tau_per_ref = {'en': en_means}
    for other in other_langs:
        m = per_cat_mean_tau(other)
        if not m.empty:
            mean_tau_per_ref[other] = m
    wide = pd.DataFrame(mean_tau_per_ref).dropna(how='any')
    if not wide.empty:
        overall = wide.mean(axis=1).sort_values(ascending=False)
        overall_path = os.path.join(RESULTS_ROOT, "category_overall_ranking.csv")
        overall_df = pd.DataFrame({
            'category': overall.index,
            'mean_tau_avg_across_refs': overall.values,
            'n_refs_in_average': wide.shape[1],
        })
        overall_df.to_csv(overall_path, index=False)

        print(f"\n  Overall categorical ranking — mean(τ) averaged across {wide.shape[1]} reference fits")
        print(f"  (UNCONDITIONAL: averages ALL prompts per category — broad pattern)")
        print(f"  {'rank':>4} | {'category':<55} | {'avg mean τ':>10}")
        print(f"  {'-'*4}-+-{'-'*55}-+-{'-'*10}")
        for i, (cat, val) in enumerate(overall.items(), start=1):
            label = (cat or '<missing>')[:55]
            print(f"  {i:>4} | {label:<55} | {val:>+10.3f}")
        print(f"  Saved: {overall_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Top-K conditioned categorical ranking — mirrors paper's Appendix K.2
    # For each ref, take top-K |τ| (prompt, language) pairs, explode by category,
    # compute per-category mean(signed τ) and count, then average across refs.
    # ──────────────────────────────────────────────────────────────────────
    topk_for_paper = 100  # matches K.2's "top 100 τ terms"
    per_ref_topk_means = {}
    per_ref_topk_counts = {}
    for ref_lang_loop in [r for r in all_data]:
        res = all_data[ref_lang_loop]['results'][['prompt', 'language', 'Safety_Tax']].copy()
        res['prompt'] = res['prompt'].apply(_clean_id)
        # Paper K.2 uses top-100 by SIGNED τ (highest positive values, target harder than ref)
        topk = res.nlargest(topk_for_paper, 'Safety_Tax')
        topk_long = topk.merge(cat_df, left_on='prompt', right_on='id', how='inner')
        agg = topk_long.groupby('category')['Safety_Tax'].agg(['mean', 'size'])
        per_ref_topk_means[ref_lang_loop] = agg['mean']
        per_ref_topk_counts[ref_lang_loop] = agg['size']

    if per_ref_topk_means:
        means_wide = pd.DataFrame(per_ref_topk_means)
        counts_wide = pd.DataFrame(per_ref_topk_counts).fillna(0)
        avg_mean = means_wide.mean(axis=1, skipna=True)
        avg_count = counts_wide.mean(axis=1)
        n_refs = means_wide.shape[1]

        ranked = pd.DataFrame({
            'category': avg_mean.index,
            'avg_mean_tau_among_topk': avg_mean.values,
            'avg_count_in_topk': avg_count.reindex(avg_mean.index).values,
            'n_refs_appeared_in': means_wide.notna().sum(axis=1).reindex(avg_mean.index).values,
        }).sort_values('avg_mean_tau_among_topk', ascending=False)

        ranked_path = os.path.join(RESULTS_ROOT, f"category_top{topk_for_paper}_ranking.csv")
        ranked.to_csv(ranked_path, index=False)

        print(f"\n  Top-{topk_for_paper} CONDITIONED ranking — paper's K.2 framing")
        print(f"  (For each ref: take top-{topk_for_paper} POSITIVE τ pairs (target harder than ref),")
        print(f"   then per category report mean(τ) and avg count. Mirrors paper Table 18.)")
        print(f"  Averaged across {n_refs} ref fits.")
        print(f"  {'rank':>4} | {'category':<55} | {'avg mean τ':>10} | {'avg count':>9} | {'n_refs':>6}")
        print(f"  {'-'*4}-+-{'-'*55}-+-{'-'*10}-+-{'-'*9}-+-{'-'*6}")
        for i, (_, r) in enumerate(ranked.iterrows(), start=1):
            label = (r['category'] or '<missing>')[:55]
            print(f"  {i:>4} | {label:<55} | {r['avg_mean_tau_among_topk']:>+10.3f} | "
                  f"{r['avg_count_in_topk']:>9.1f} | {int(r['n_refs_appeared_in']):>6}")
        print(f"  Saved: {ranked_path}")


def english_reversal_counts(all_data):
    """For each ref_lang, count model configs where English is the least-safe language
    under the IRT framework, plus a single ref-language-independent JSR cross-check
    that reproduces the paper's 22/61 figure.

    The strict IRT criterion (argmin_L of θ + δ equals English) requires every non-English
    δ_{j,L} to be positive, which is much stricter than the JSR argmax used in the paper.
    """
    print(f"\n--- English-least-safe model count by reference language ---")
    print(f"  argmin_L (θ_j + δ_{{j,L}}) == en  (strict IRT criterion)")
    print()
    print(f"  {'ref_lang':<10} | {'n_models':>8} | {'en_least_safe':>14} | {'% en-worst':>11}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*14}-+-{'-'*11}")
    rows = []
    for ref_lang, data in all_data.items():
        theta = data.get('theta')
        delta = data.get('delta')
        if theta is None or delta is None:
            print(f"  {ref_lang:<10} | (delta or theta missing — skipping)")
            continue
        merged = delta.merge(theta, on='test_taker', how='inner')
        merged['ability'] = merged['theta'] + merged['delta']
        idx_min = merged.groupby('test_taker')['ability'].idxmin()
        worst = merged.loc[idx_min, ['test_taker', 'language']]
        en_count = int((worst['language'] == 'en').sum())
        total = merged['test_taker'].nunique()
        pct = 100.0 * en_count / total if total else 0.0
        rows.append({
            'ref_lang': ref_lang,
            'n_models': total,
            'en_least_safe': en_count,
            'pct_en_least_safe': pct,
        })
        print(f"  {ref_lang:<10} | {total:>8} | {en_count:>14} | {pct:>10.1f}%")

    if rows:
        df = pd.DataFrame(rows)
        print(f"  Range across {len(df)} reference choices: "
              f"{int(df['en_least_safe'].min())}–{int(df['en_least_safe'].max())} "
              f"(mean {df['en_least_safe'].mean():.1f})")

    # JSR cross-check from raw data — this IS the paper's 22/61 measure, not p-hacking
    print()
    try:
        jsr_count = _english_least_safe_via_jsr()
        print(f"  Paper-style JSR cross-check (ref-independent, raw data):")
        print(f"    {jsr_count['en_count']}/{jsr_count['total']} model configs have argmax_L JSR == en")
        print(f"    (reproduces the paper's 22/61; uses empirical jailbreak rate, no IRT smoothing)")
    except Exception as e:
        print(f"  JSR cross-check skipped: {e}")

    out_path = os.path.join(RESULTS_ROOT, "english_reversal_counts_by_ref.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def _english_least_safe_via_jsr():
    """Compute the paper's 22/61-style measure directly from the raw response matrix.

    JSR per (test_taker, language) = fraction of binary-unsafe rows. The model's
    least-safe language is argmax_L JSR. Count models where that argmax is 'en'.
    Reference-language-independent (no IRT involved).
    """
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
        master = os.path.join(d, "processed_data", "Master_Passes0-9_Dataset.csv")
    except Exception:
        master = None

    candidates = [master] if master else []
    candidates += [
        os.path.join(RESULTS_ROOT, "Master_Passes0-9_Dataset.csv"),
        os.path.join(os.path.dirname(RESULTS_ROOT), "Master_Passes0-9_Dataset.csv"),
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Master_Passes0-9_Dataset.csv not found")

    df = pd.read_csv(path, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    # Paper convention: safe = judge_score >= 4; unsafe = 1-3
    df['unsafe'] = (df['judge_score'] < 4).astype(int)
    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    jsr = df.groupby([student_col, 'language'])['unsafe'].mean().reset_index(name='jsr')
    # For each model, find argmax_L jsr
    idx_max = jsr.groupby(student_col)['jsr'].idxmax()
    worst = jsr.loc[idx_max, [student_col, 'language']]
    en_count = int((worst['language'] == 'en').sum())
    return {'en_count': en_count, 'total': len(worst)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+",
                        default=["ar", "bn", "en", "it", "jv", "ko", "sw", "th", "vi", "zh"],
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
    # All Jaccard / sensitivity comparisons are anchored on English.
    if 'en' in all_data:
        pairs = [('en', other) for other in all_data if other != 'en']
        print(f"\n  Anchoring comparisons on 'en' ({len(pairs)} en-vs-other pairs).")
    else:
        pairs = list(itertools.combinations(all_data.keys(), 2))
        print(f"\n  WARNING: English ref not loaded — falling back to all-pairs.")
    # ── Silently collect per-pair sensitivity stats (full table goes to CSV) ──
    summary_rows = []
    for lang_a, lang_b in pairs:
        da, db = all_data[lang_a], all_data[lang_b]
        theta_cmp = compare_theta(da, db)
        gamma_cmp = compare_gamma(da, db)
        tau_corr = compare_tau_correlation(da, db)
        tau_overlap = compare_tau_overlap(da, db, k=args.top_k)
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
            'tau_top100_abs_overlap': tau_overlap['top100_absolute_overlap'],
            'tau_top100_abs_jaccard': tau_overlap['top100_absolute_jaccard'],
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(RESULTS_ROOT, "ref_lang_sensitivity_summary.csv")
    summary_df.to_csv(out_path, index=False)

    def _banner(title):
        print(f"\n{'='*72}")
        print(f"  {title}")
        print(f"{'='*72}")

    def _stats(series):
        return f"min {series.min():.3f}, mean {series.mean():.3f}, max {series.max():.3f}"

    # ──────────────────────────────────────────────────────────────────────
    # 1. HEADLINE: en-anchored sensitivity stats
    # ──────────────────────────────────────────────────────────────────────
    _banner(f"1. HEADLINE — en-anchored sensitivity (n={len(summary_df)} comparisons)")
    print(f"  θ Spearman (model rankings):     {_stats(summary_df['theta_spearman'])}")
    print(f"  τ Spearman (per-pair τ values):  {_stats(summary_df['tau_spearman'])}")
    print(f"  Top-100 |τ| Jaccard:             {_stats(summary_df['tau_top100_abs_jaccard'])}")
    print(f"  Top-100 |τ| raw overlap (of 100): "
          f"min {int(summary_df['tau_top100_abs_overlap'].min())}, "
          f"mean {summary_df['tau_top100_abs_overlap'].mean():.1f}, "
          f"max {int(summary_df['tau_top100_abs_overlap'].max())}")
    print(f"  → per-pair table saved: {out_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Top-k |τ| overlap averaged across all en-vs-other pairs
    # ──────────────────────────────────────────────────────────────────────
    _banner(f"2. TOP-K |τ| OVERLAP vs ENGLISH (averaged across {len(pairs)} pairs)")
    print(f"  {'k':>5} | {'mean Jaccard':>13} | {'mean overlap':>13}")
    print(f"  {'-'*5}-+-{'-'*13}-+-{'-'*13}")
    topk_rows = []
    for k in [10, 20, 50, 100]:
        jaccards, overlaps = [], []
        for lang_a, lang_b in pairs:
            ov = compare_tau_overlap(all_data[lang_a], all_data[lang_b], k=k)
            jaccards.append(ov['top100_absolute_jaccard'])
            overlaps.append(ov['top100_absolute_overlap'])
        row = {'k': k, 'jaccard_mean': np.mean(jaccards), 'overlap_mean': np.mean(overlaps)}
        topk_rows.append(row)
        print(f"  {k:>5} | {row['jaccard_mean']:>13.3f} | {row['overlap_mean']:>13.1f}")
    topk_path = os.path.join(RESULTS_ROOT, "ref_lang_topk_overlap_summary.csv")
    pd.DataFrame(topk_rows).to_csv(topk_path, index=False)
    print(f"  → saved: {topk_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 3, 4, 5. Per-harm-category sections (English anchor)
    # ──────────────────────────────────────────────────────────────────────
    try:
        cat_df = load_multijail_categories()
        _banner("3. PER-CATEGORY MEAN τ (English reference, descriptive)")
        compare_per_category_vs_en(all_data, cat_df, top_k=args.top_k)

        _banner("4. PER-CATEGORY RANK CORRELATION vs ENGLISH (backs Appendix K.2)")
        compare_category_rankings_vs_en(all_data, cat_df)
    except FileNotFoundError as e:
        print(f"\n  Skipping category analysis: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # 6. English-reversal count (validates 22/61 claim under anchor swap)
    # ──────────────────────────────────────────────────────────────────────
    _banner("5. ENGLISH-LEAST-SAFE MODEL COUNT BY REFERENCE LANGUAGE")
    english_reversal_counts(all_data)

    # ──────────────────────────────────────────────────────────────────────
    # 7. One-line bottom-line summary
    # ──────────────────────────────────────────────────────────────────────
    _banner("BOTTOM LINE")
    mean_theta_rho = summary_df['theta_spearman'].mean()
    mean_tau_jaccard = summary_df['tau_top100_abs_jaccard'].mean()
    print(f"  Mean θ Spearman vs English:       {mean_theta_rho:.3f} "
          f"({'robust' if mean_theta_rho > 0.95 else 'moderate' if mean_theta_rho > 0.85 else 'sensitive'})")
    print(f"  Mean top-100 |τ| Jaccard vs en:   {mean_tau_jaccard:.3f} "
          f"({'stable' if mean_tau_jaccard > 0.70 else 'moderate' if mean_tau_jaccard > 0.50 else 'sensitive'})")
    # Random baseline reminder
    print(f"  (Random-chance baseline for top-100 Jaccard with N=2835: ≈ 0.018)")
    print()


if __name__ == "__main__":
    main()
