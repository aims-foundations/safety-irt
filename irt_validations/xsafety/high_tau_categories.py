# -*- coding: utf-8 -*-
"""
Count harm categories among top 100 highest positive-τ prompts — XSafety.
Adapted from irt_validations/high_tau_categories.py:
  - XSafety uses 'category' column (single string, not list) → no explode needed
  - Run high_tau_top100-prompts.py first to generate the input CSV
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_qualitative_inspection")
INPUT = os.path.join(RESULTS_DIR, "top100_high_tau_prompts.csv")


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} high positive-τ prompts")
    print(f"  τ range: [{df['tau'].min():.3f}, {df['tau'].max():.3f}]")
    print(f"  All positive: {(df['tau'] > 0).all()}")

    if 'category' not in df.columns:
        print("No 'category' column — run high_tau_top100-prompts.py first")
        return

    # XSafety: category is already a single string — no explode needed
    cat_df = df.dropna(subset=['category']).copy()
    cat_df = cat_df[cat_df['category'] != '']

    # ── Overall category counts ──────────────────────────────────
    counts = cat_df['category'].value_counts()
    print(f"\n{'=' * 60}")
    print("CATEGORY COUNTS (top 100 highest positive τ)")
    print(f"{'=' * 60}")
    for cat, n in counts.items():
        print(f"  {n:3d}  {cat}")

    counts.to_csv(os.path.join(RESULTS_DIR, "tau_category_counts.csv"),
                  header=['count'])
    print(f"  Saved: tau_category_counts.csv")

    # ── Mean τ by category (all positive, ranked by severity) ────
    print(f"\n{'=' * 60}")
    print("MEAN τ BY CATEGORY (higher = more dangerous in non-English)")
    print(f"{'=' * 60}")

    mean_tau = (cat_df.groupby('category')['tau']
                .agg(['mean', 'std', 'count'])
                .sort_values('mean', ascending=False))
    print(mean_tau.round(3).to_string())
    mean_tau.to_csv(os.path.join(RESULTS_DIR, "tau_mean_by_category.csv"))
    print(f"  Saved: tau_mean_by_category.csv")

    # ── Per-language × category ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CATEGORY × LANGUAGE (count of high positive-τ prompts)")
    print(f"{'=' * 60}")

    lang_cat = pd.crosstab(cat_df['category'], cat_df['language'])
    lang_cat['total'] = lang_cat.sum(axis=1)
    lang_cat = lang_cat.sort_values('total', ascending=False)

    print(lang_cat.to_string())
    lang_cat.to_csv(os.path.join(RESULTS_DIR, "tau_category_by_language.csv"))
    print(f"  Saved: tau_category_by_language.csv")

    # ── Per-language summary ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PER-LANGUAGE SUMMARY")
    print(f"{'=' * 60}")

    lang_stats = (df.groupby('language')['tau']
                  .agg(['count', 'mean', 'max'])
                  .sort_values('mean', ascending=False))
    print(lang_stats.round(3).to_string())

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
