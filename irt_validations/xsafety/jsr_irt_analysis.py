# -*- coding: utf-8 -*-
"""
JSR vs IRT Ability Heatmaps — XSafety.
Adapted from irt_validations/jsr_irt_ordering.py:
  - LANG_ORDER updated to XSafety languages
  - No pass-suffix stripping in model names
  - Reads from xsafety/results_jsr_theta_posthoc/ (output of jsr_difficulty.py)
  - Language heatmap skipped gracefully when no per-language theta data

Produces 3 figures:
  1. Dual heatmap: JSR (top) vs θ+δ (bottom) — same color scale
  2. English-focused: JSR_en vs θ (bar chart, sorted by θ)
  3. Rank discrepancy heatmap: JSR_rank − (θ+δ)_rank per language

Run jsr_difficulty.py first to generate the input CSVs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, COLORS_3, CMAP_DIV, CMAP_SEQ,
                           FAM_COLORS as FS_FAM_COLORS, FAM_ORDER as FS_FAM_ORDER,
                           LABELS, LANG_ORDER, FULL_WIDTH, DPI, ASPECT,
                           get_family, get_family_color, add_identity_line)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    print("[WARN] fig_style.py not found - using defaults")

from scipy.stats import spearmanr
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

POSTHOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_jsr_theta_posthoc")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_ability_heatmaps")
os.makedirs(RESULTS_DIR, exist_ok=True)

# XSafety language display order
LANG_ORDER = ['en', 'zh', 'ar', 'bn', 'de', 'fr', 'hi', 'ja', 'ru', 'sp']

IRT_MODEL = '2PL'

# ── Color Palette Configuration ──
_c1 = C_BLUE   if _HAS_FIG_STYLE else '#2471a3'
_c2 = C_RED    if _HAS_FIG_STYLE else '#c0392b'
_c3 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98'


def get_model_family(name):
    name = str(name).lower()
    if any(x in name for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude'   in name: return 'Claude'
    elif 'gemini'   in name: return 'Gemini'
    elif 'grok'     in name: return 'Grok'
    elif 'deepseek' in name: return 'DeepSeek'
    return 'Other'


def shorten_name(name, max_len=28):
    name = str(name)
    if len(name) > max_len:
        name = name[:max_len-2] + '..'
    return name


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    # Overall theta
    overall_path = os.path.join(POSTHOC_DIR,
                                "1_jsr_vs_theta_all_models.csv")
    if not os.path.exists(overall_path):
        raise FileNotFoundError(
            f"Run jsr_difficulty.py first.\nExpected: {overall_path}")
    overall = pd.read_csv(overall_path)
    overall = overall[overall['irt_model'] == IRT_MODEL].copy()
    print(f"Overall: {len(overall)} models")

    # Per-language: JSR_lang and theta+delta
    # NOTE: for XSafety, this file will be empty (no B experiment / no delta)
    lang_path = os.path.join(POSTHOC_DIR,
                             "2_jsr_vs_theta_minus_delta_all_models.csv")
    lang_df = pd.DataFrame()
    if os.path.exists(lang_path):
        raw = pd.read_csv(lang_path)
        if len(raw) > 0 and 'irt_model' in raw.columns:
            lang_df = raw[raw['irt_model'] == IRT_MODEL].copy()
            print(f"Language: {len(lang_df)} rows")
        else:
            print("Language data file is empty — per-language plots skipped.")
    else:
        print("No per-language file — per-language plots skipped.")

    return overall, lang_df


def build_pivots(overall, lang_df):
    """Build model × language pivot tables for JSR and θ+δ."""
    if len(lang_df) == 0:
        return None, None, None, []

    # JSR pivot
    jsr_pivot = lang_df.pivot_table(
        index='test_taker', columns='language',
        values='JSR_lang', aggfunc='first')

    # θ+δ pivot
    ability_pivot = lang_df.pivot_table(
        index='test_taker', columns='language',
        values='theta_minus_delta', aggfunc='first')

    # Reorder columns to match LANG_ORDER
    present_langs = [l for l in LANG_ORDER if l in jsr_pivot.columns]
    jsr_pivot = jsr_pivot[present_langs]
    ability_pivot = ability_pivot[present_langs]

    # Get model families for row coloring
    families = lang_df.drop_duplicates('test_taker').set_index(
        'test_taker')['model_family']

    # Sort rows: by family then by overall theta (safest at top)
    theta_lookup = overall.set_index('test_taker')['theta']
    fam_order_map = {f: i for i, f in enumerate(
        ['Claude', 'DeepSeek', 'Gemini', 'GPT', 'Grok', 'Other'])}

    sort_df = pd.DataFrame({
        'family': families.reindex(jsr_pivot.index),
        'theta':  theta_lookup.reindex(jsr_pivot.index)
    })
    sort_df['fam_rank'] = sort_df['family'].map(fam_order_map).fillna(99)
    sort_df = sort_df.sort_values(['fam_rank', 'theta'], ascending=[True, False])

    jsr_pivot = jsr_pivot.reindex(sort_df.index)
    ability_pivot = ability_pivot.reindex(sort_df.index)
    families = families.reindex(sort_df.index)

    return jsr_pivot, ability_pivot, families, present_langs


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Dual Heatmap: JSR vs θ+δ
# ══════════════════════════════════════════════════════════════════════════

def plot_dual_heatmap(jsr_pivot, ability_pivot, families, present_langs):
    """
    Horizontal layout: languages on y-axis, models on x-axis.
    """
    n_models = len(jsr_pivot)
    n_langs = len(present_langs)

    jsr_t = jsr_pivot.T
    ability_t = ability_pivot.T

    fig_width = max(18, n_models * 0.35 + 2)
    fig_height = n_langs * 1.5 + 3

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                             layout='tight')

    short_labels = [shorten_name(n) for n in jsr_t.columns]

    # Top: JSR (Reds)
    ax = axes[0]
    sns.heatmap(jsr_t, ax=ax, cmap='Reds', vmin=0, vmax=1,
                annot=True, fmt='.2f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'JSR (1=all unsafe)',
                          'orientation': 'horizontal',
                          'shrink': 0.5, 'pad': 0.15},
                annot_kws={'fontsize': 5.5})
    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)
    ax.set_title('Jailbreak Success Rate (higher = less safe)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Language')

    # Bottom: θ+δ (RdBu)
    ax = axes[1]
    vmax_ab = max(abs(ability_t.values.min()), abs(ability_t.values.max()), 2)
    sns.heatmap(ability_t, ax=ax, cmap='RdBu', center=0,
                vmin=-vmax_ab, vmax=vmax_ab,
                annot=True, fmt='.2f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'θ+δ (higher = safer)',
                          'orientation': 'horizontal',
                          'shrink': 0.5, 'pad': 0.15},
                annot_kws={'fontsize': 5.5})
    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)
    ax.set_title('IRT Ability (θ + δ) (higher = safer)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Language')

    fig.suptitle(f'Model Safety: JSR vs IRT Ability by Language ({IRT_MODEL}) — XSafety',
                 fontsize=14, fontweight='bold', y=1.02)

    path = os.path.join(RESULTS_DIR, f"dual_heatmap_{IRT_MODEL}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — English Focus: JSR_en vs θ
# ══════════════════════════════════════════════════════════════════════════

def plot_english_focus(overall):
    """
    Shows JSR and θ side by side for English (overall table).
    For XSafety, JSR_en comes from the overall table directly.
    """
    df = overall.dropna(subset=['theta', 'JSR']).copy()
    df = df.sort_values('theta', ascending=True)
    n = len(df)

    if n == 0:
        print("  No data for English focus plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, n * 0.2)),
                             sharey=True,
                             gridspec_kw={'wspace': 0.05},
                             layout='tight')

    labels = [shorten_name(m) for m in df['test_taker']]
    fam_col = 'model_family' if 'model_family' in df.columns else 'family'
    colors = [_c2 if f == 'Grok' else _c1 for f in df[fam_col]]

    # Left panel: JSR
    ax = axes[0]
    ax.barh(np.arange(n), df['JSR'].values, color=colors,
            edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.set_xlim(1, 0)
    ax.set_xlabel('JSR (← less safe  |  safer →)', fontsize=10)
    ax.set_title('Jailbreak Success Rate', fontweight='bold', fontsize=12)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(axis='x', alpha=0.2)

    # Right panel: θ
    ax = axes[1]
    ax.barh(np.arange(n), df['theta'].values, color=colors,
            edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('θ (← less safe  |  safer →)', fontsize=10)
    ax.set_title('IRT Ability (θ)', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.2)

    rho, p = spearmanr(df['JSR'], df['theta'])
    fig.text(0.5, -0.02,
             f'Spearman ρ(JSR, θ) = {rho:.3f}  (p = {p:.2e})',
             ha='center', fontsize=9, style='italic')

    fig.suptitle(f'Model Safety: JSR vs IRT Ability ({IRT_MODEL}) — XSafety\n'
                 f'(sorted by θ, least safe at bottom)',
                 fontsize=13, fontweight='bold', y=1.04)

    path = os.path.join(RESULTS_DIR, f"english_focus_{IRT_MODEL}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Rank Discrepancy Heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_rank_discrepancy(jsr_pivot, ability_pivot, families, present_langs):
    """
    Per-language rank discrepancy: JSR_rank − (θ+δ)_rank.
    """
    n_models = len(jsr_pivot)

    jsr_ranks = jsr_pivot.rank(ascending=False, method='min')
    ability_ranks = ability_pivot.rank(ascending=True, method='min')
    delta_ranks = jsr_ranks - ability_ranks

    delta_ranks['_sort'] = delta_ranks.abs().mean(axis=1)
    delta_ranks = delta_ranks.sort_values('_sort', ascending=False)
    delta_ranks = delta_ranks.drop(columns='_sort')
    short_labels = [shorten_name(n) for n in delta_ranks.index]

    delta_ranks_t = delta_ranks.T

    fig_width = max(18, n_models * 0.35 + 2)
    fig_height = len(present_langs) * 0.5 + 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout='tight')

    vmax = max(abs(delta_ranks_t.values.min()), abs(delta_ranks_t.values.max()), 5)

    sns.heatmap(delta_ranks_t, ax=ax, cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax,
                annot=True, fmt='.0f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Rank Δ (JSR rank − IRT rank)',
                          'orientation': 'horizontal',
                          'shrink': 0.4, 'pad': 0.25},
                annot_kws={'fontsize': 5.5})

    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)

    ax.set_title(
        f'Per-Language Rank Discrepancy: JSR vs θ+δ ({IRT_MODEL}) — XSafety\n'
        f'Red = JSR overestimates risk  |  Blue = JSR underestimates risk',
        fontsize=11, fontweight='bold')
    ax.set_ylabel('Language', fontsize=11)
    ax.set_xlabel('')

    path = os.path.join(RESULTS_DIR, f"rank_discrepancy_heatmap_{IRT_MODEL}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if _HAS_FIG_STYLE:
        apply_style()
    print("=" * 60)
    print("JSR vs IRT ABILITY HEATMAPS — XSafety")
    print("=" * 60)

    overall, lang_df = load_data()
    jsr_pivot, ability_pivot, families, present_langs = build_pivots(
        overall, lang_df)

    # Figure 2: Overall JSR vs θ (bar chart)
    print("\n[Figure 2] JSR vs θ bar chart")
    plot_english_focus(overall)

    if jsr_pivot is not None and len(present_langs) > 0:
        print(f"\nModels: {len(jsr_pivot)}, Languages: {present_langs}")
        print(f"Families: {families.value_counts().to_dict()}")

        # Figure 1: Dual heatmap
        print("\n[Figure 1] Dual heatmap: JSR vs θ+δ")
        plot_dual_heatmap(jsr_pivot, ability_pivot, families, present_langs)

        # Figure 3: Rank discrepancy
        print("\n[Figure 3] Rank discrepancy heatmap")
        plot_rank_discrepancy(jsr_pivot, ability_pivot, families, present_langs)
    else:
        print("\nNo per-language data — dual heatmap and rank discrepancy skipped.")
        print("(XSafety has no B experiment; per-language θ−δ not available.)")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
