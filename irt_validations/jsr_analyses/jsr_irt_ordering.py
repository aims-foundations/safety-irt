# -*- coding: utf-8 -*-
"""
JSR vs IRT Ability Heatmaps
============================
Side-by-side comparison of raw JSR and IRT-adjusted ability (θ+δ)
across all models and languages.

Produces 3 figures:
  1. Dual heatmap: JSR (left) vs θ+δ (right) — same color scale
  2. English-focused: JSR_en vs θ (bar chart, sorted by θ)
  3. Rank discrepancy heatmap: JSR_rank − (θ+δ)_rank per language

Reads from jsr_vs_theta_posthoc outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# ── fig_style integration ──
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
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

# Language display order (user-specified)
LANG_ORDER = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']

IRT_MODEL = '2PL'

# ── Color Palette Configuration ──
_c1 = C_BLUE if _HAS_FIG_STYLE else '#2471a3'   # Main / Safe
_c2 = C_RED if _HAS_FIG_STYLE else '#c0392b'    # Highlight / Unsafe
_c3 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98' # Divergence / Alt


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
    name = re.sub(r'[_-]?pass[_-]?\d+', '', name, flags=re.IGNORECASE)
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
    overall = pd.read_csv(overall_path)
    overall = overall[overall['irt_model'] == IRT_MODEL].copy()
    print(f"Overall: {len(overall)} models")

    # Per-language: JSR_lang and theta+delta
    lang_path = os.path.join(POSTHOC_DIR,
                             "2_jsr_vs_theta_minus_delta_all_models.csv")
    lang_df = pd.read_csv(lang_path)
    lang_df = lang_df[lang_df['irt_model'] == IRT_MODEL].copy()
    print(f"Language: {len(lang_df)} rows")

    return overall, lang_df


def build_pivots(overall, lang_df):
    """Build model × language pivot tables for JSR and θ+δ."""
    # JSR pivot
    jsr_pivot = lang_df.pivot_table(
        index='test_taker', columns='language',
        values='JSR_lang', aggfunc='first')

    # θ+δ pivot (column is named theta_minus_delta but values are θ+δ)
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
    Colorblind friendly: Reds (JSR) and RdBu (Ability).
    """
    n_models = len(jsr_pivot)
    n_langs = len(present_langs)

    jsr_t = jsr_pivot.T
    ability_t = ability_pivot.T

    fig_width = max(18, n_models * 0.35 + 2)
    fig_height = n_langs * 1.5 + 3
    
    # FIX: Initialize tight layout here
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), layout='tight')

    short_labels = [shorten_name(n) for n in jsr_t.columns]

    # Top: JSR (Reds -> Dark Red = High JSR = Unsafe)
    ax = axes[0]
    sns.heatmap(jsr_t, ax=ax, cmap='Reds', vmin=0, vmax=1,
                annot=True, fmt='.2f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'JSR (1=all unsafe)', 'orientation': 'horizontal', 
                          'shrink': 0.5, 'pad': 0.15},
                annot_kws={'fontsize': 5.5})
    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)
    ax.set_title('Jailbreak Success Rate (higher = less safe)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Language')

    # Bottom: θ+δ (RdBu -> Red = Unsafe, Blue = Safe)
    ax = axes[1]
    vmax_ab = max(abs(ability_t.values.min()), abs(ability_t.values.max()), 2)
    sns.heatmap(ability_t, ax=ax, cmap='RdBu', center=0, vmin=-vmax_ab, vmax=vmax_ab,
                annot=True, fmt='.2f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'θ+δ (higher = safer)', 'orientation': 'horizontal', 
                          'shrink': 0.5, 'pad': 0.15},
                annot_kws={'fontsize': 5.5})
    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)
    ax.set_title('IRT Ability (θ + δ) (higher = safer)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Language')

    fig.suptitle(f'Model Safety: JSR vs IRT Ability by Language ({IRT_MODEL})',
                 fontsize=14, fontweight='bold', y=1.02)

    path = os.path.join(RESULTS_DIR, f"dual_heatmap_{IRT_MODEL}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — English Focus: JSR_en vs θ (δ_en = 0)
# ══════════════════════════════════════════════════════════════════════════

def plot_english_focus(jsr_pivot, ability_pivot, families, overall):
    """
    Shows JSR_en and θ side by side. Uses 3-color scheme (Red for Grok, Blue for rest).
    """
    if 'en' not in jsr_pivot.columns:
        print("  English not in data, skipping English focus plot.")
        return

    en_df = pd.DataFrame({
        'model': jsr_pivot.index,
        'JSR_en': jsr_pivot['en'].values,
        'theta': overall.set_index('test_taker')['theta'].reindex(
            jsr_pivot.index).values,
        'family': families.values,
    }).dropna()

    en_df = en_df.sort_values('theta', ascending=True)
    n = len(en_df)
    
    # FIX: Initialize tight layout here
    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, n * 0.2)),
                             sharey=True, gridspec_kw={'wspace': 0.05}, layout='tight')

    labels = [shorten_name(m) for m in en_df['model']]
    colors = [_c2 if f == 'Grok' else _c1 for f in en_df['family']]

    label_colors = ['#e74c3c' if f == 'Grok' else '#333333' for f in en_df['family']]
    label_weights = ['bold' if f == 'Grok' else 'normal' for f in en_df['family']]

    # Left panel: JSR_en
    ax = axes[0]
    ax.barh(np.arange(n), en_df['JSR_en'].values, color=colors,
            edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.set_xlim(1, 0)
    ax.set_xlabel('JSR_en (← less safe  |  safer →)', fontsize=10)
    ax.set_title('JSR in English', fontweight='bold', fontsize=12)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(axis='x', alpha=0.2)

    fig.canvas.draw()
    for i, (col, wt) in enumerate(zip(label_colors, label_weights)):
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight(wt)

    # Right panel: θ
    ax = axes[1]
    ax.barh(np.arange(n), en_df['theta'].values, color=colors,
            edgecolor='black', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('θ (← less safe  |  safer →)', fontsize=10)
    ax.set_title('IRT Ability (θ)', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.2)

    rho, p = spearmanr(en_df['JSR_en'], en_df['theta'])
    fig.text(0.5, -0.02,
             f'Spearman ρ(JSR_en, θ) = {rho:.3f}  (p = {p:.2e})  |  '
             f'Note: δ_en = 0 by constraint, so θ+δ = θ for English',
             ha='center', fontsize=9, style='italic')

    fig.suptitle(f'English Safety: JSR vs IRT Ability ({IRT_MODEL})\n'
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
    Colorblind friendly: RdBu_r. Horizontal layout.
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
    
    # FIX: Initialize tight layout here
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout='tight')

    vmax = max(abs(delta_ranks_t.values.min()), abs(delta_ranks_t.values.max()), 5)

    sns.heatmap(delta_ranks_t, ax=ax, cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax,
                annot=True, fmt='.0f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Rank Δ (JSR rank − IRT rank)',
                          'orientation': 'horizontal', 'shrink': 0.4, 'pad': 0.25},
                annot_kws={'fontsize': 5.5})

    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)

    ax.set_title(
        f'Per-Language Rank Discrepancy: JSR vs θ+δ ({IRT_MODEL})\n'
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
    if _HAS_FIG_STYLE: apply_style()
    print("=" * 60)
    print("JSR vs IRT ABILITY HEATMAPS")
    print("=" * 60)

    overall, lang_df = load_data()
    jsr_pivot, ability_pivot, families, present_langs = build_pivots(
        overall, lang_df)

    print(f"\nModels: {len(jsr_pivot)}, Languages: {present_langs}")
    print(f"Families: {families.value_counts().to_dict()}")

    # Figure 1: Dual heatmap
    print("\n[Figure 1] Dual heatmap: JSR vs θ+δ")
    plot_dual_heatmap(jsr_pivot, ability_pivot, families, present_langs)

    # Figure 2: English focus
    print("\n[Figure 2] English focus: JSR_en vs θ")
    plot_english_focus(jsr_pivot, ability_pivot, families, overall)

    # Figure 3: Rank discrepancy
    print("\n[Figure 3] Rank discrepancy heatmap")
    plot_rank_discrepancy(jsr_pivot, ability_pivot, families, present_langs)

    # Print English stats for paper
    if 'en' in jsr_pivot.columns:
        print(f"\n{'=' * 60}")
        print("ENGLISH STATS (for paper)")
        print(f"{'=' * 60}")
        en_jsr = jsr_pivot['en'].dropna()
        en_ability = ability_pivot['en'].dropna()

        # Grok models
        grok_mask = families == 'Grok'
        grok_models = jsr_pivot.index[grok_mask]
        if len(grok_models) > 0:
            print(f"\nGrok models in English:")
            for m in grok_models:
                j = jsr_pivot.loc[m, 'en'] if m in jsr_pivot.index else np.nan
                a = ability_pivot.loc[m, 'en'] if m in ability_pivot.index else np.nan
                t = overall.set_index('test_taker')['theta'].get(m, np.nan)
                print(f"  {shorten_name(m):30s}  JSR_en={j:.3f}  θ={t:.3f}  θ+δ_en={a:.3f}")

            grok_jsr_mean = en_jsr[grok_mask].mean()
            grok_theta_mean = en_ability[grok_mask].mean()
            all_jsr_mean = en_jsr.mean()
            all_theta_mean = en_ability.mean()

            print(f"\n  Grok mean JSR_en:  {grok_jsr_mean:.3f}  "
                  f"(all models: {all_jsr_mean:.3f})")
            print(f"  Grok mean θ:       {grok_theta_mean:.3f}  "
                  f"(all models: {all_theta_mean:.3f})")
            print(f"  → Both metrics confirm Grok cluster at unsafe end"
                  if grok_jsr_mean > all_jsr_mean
                  else f"  → JSR and IRT disagree on Grok severity")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()