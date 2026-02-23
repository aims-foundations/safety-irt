# -*- coding: utf-8 -*-
"""
Count harm categories among high-|τ| prompts.
Tags are multi-label: "['Theft', 'Fraud']" → explode into separate rows.
"""

import pandas as pd
import numpy as np
import ast
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_qualitative_inspection")
INPUT = os.path.join(RESULTS_DIR, "top100_high_tau_prompts.csv")


def parse_tags(tag_str):
    """Parse "['Tag1', 'Tag2']" string into a Python list."""
    if pd.isna(tag_str):
        return []
    try:
        return ast.literal_eval(tag_str)
    except (ValueError, SyntaxError):
        # Fallback: strip brackets and split
        s = str(tag_str).strip("[]'\"")
        return [t.strip().strip("'\"") for t in s.split(',') if t.strip()]


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} high-τ prompts")

    if 'tags' not in df.columns:
        print("No 'tags' column — run extract_top_tau.py first with multijail")
        return

    # Explode tags
    df['tag_list'] = df['tags'].apply(parse_tags)
    exploded = df.explode('tag_list').dropna(subset=['tag_list'])
    exploded = exploded[exploded['tag_list'] != ''].reset_index(drop=True)

    # ── Overall category counts ──────────────────────────────────
    counts = exploded['tag_list'].value_counts()
    print(f"\n{'=' * 60}")
    print(f"CATEGORY COUNTS (top 100 high-|τ| prompts)")
    print(f"{'=' * 60}")
    for cat, n in counts.items():
        print(f"  {n:3d}  {cat}")

    counts.to_csv(os.path.join(RESULTS_DIR, "tau_category_counts.csv"),
                  header=['count'])
    print(f"\n  Saved: tau_category_counts.csv")

    # ── Positive vs negative τ breakdown ─────────────────────────
    print(f"\n{'=' * 60}")
    print("POSITIVE τ (harder in this language) vs NEGATIVE τ (easier)")
    print(f"{'=' * 60}")

    pos = exploded[exploded['tau'] > 0]['tag_list'].value_counts()
    neg = exploded[exploded['tau'] < 0]['tag_list'].value_counts()

    all_cats = sorted(set(pos.index) | set(neg.index))
    breakdown = pd.DataFrame({
        'positive_tau': pos.reindex(all_cats, fill_value=0),
        'negative_tau': neg.reindex(all_cats, fill_value=0),
    })
    breakdown['total'] = breakdown.sum(axis=1)
    breakdown['pct_positive'] = (breakdown['positive_tau'] /
                                  breakdown['total'] * 100).round(1)
    breakdown = breakdown.sort_values('total', ascending=False)

    print(breakdown.to_string())
    breakdown.to_csv(os.path.join(RESULTS_DIR,
                                  "tau_category_pos_neg.csv"))
    print(f"\n  Saved: tau_category_pos_neg.csv")

    # ── Per-language × category ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CATEGORY × LANGUAGE (count of high-|τ| prompts)")
    print(f"{'=' * 60}")

    lang_cat = pd.crosstab(exploded['tag_list'], exploded['language'])
    lang_cat['total'] = lang_cat.sum(axis=1)
    lang_cat = lang_cat.sort_values('total', ascending=False)

    print(lang_cat.to_string())
    lang_cat.to_csv(os.path.join(RESULTS_DIR,
                                  "tau_category_by_language.csv"))
    print(f"\n  Saved: tau_category_by_language.csv")

    # ── Mean τ by category ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("MEAN τ BY CATEGORY (+ = harder in non-English)")
    print(f"{'=' * 60}")

    mean_tau = (exploded.groupby('tag_list')['tau']
                .agg(['mean', 'std', 'count'])
                .sort_values('mean', ascending=False))
    print(mean_tau.round(3).to_string())
    mean_tau.to_csv(os.path.join(RESULTS_DIR,
                                  "tau_mean_by_category.csv"))
    print(f"\n  Saved: tau_mean_by_category.csv")

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()