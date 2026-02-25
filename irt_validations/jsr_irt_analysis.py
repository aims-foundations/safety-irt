# -*- coding: utf-8 -*-
"""
Rank Divergence Analysis v2: JSR vs IRT with Divergence Metrics
================================================================
Cleaner visualizations + proper divergence metrics.

Metrics:
  RMSRD  — Root Mean Squared Rank Displacement = sqrt(mean(Δrank²)) / (N−1)
           Quadratic: penalizes large rank shifts disproportionately.
           Normalized by N−1 so comparable across different-sized sets.
           Range: [0, ~0.58]; 0 = perfect agreement.

  QWK    — Quadratic Weighted Kappa (like Cohen's κ but ordinal)
           1 = perfect, 0 = chance, <0 = worse than chance.

Visualizations (designed to NOT be spaghetti):
  1. Top-K movers bar chart (only biggest divergences, not all models)
  2. Family-level divergence summary (aggregated, clean)
  3. Divergence distribution histogram + metrics card
  4. Per-language divergence metric comparison
  5. Heatmap: family × language mean rank shift
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ── INSERT THIS BLOCK: fig_style integration ──
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
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
# ───────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

POSTHOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_jsr_theta_posthoc")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_rank_divergence")
os.makedirs(RESULTS_DIR, exist_ok=True)

PRIMARY_IRT = '2PL'

FAM_COLORS = FS_FAM_COLORS if _HAS_FIG_STYLE else {
    'Claude':   '#7d3c98', 'GPT':      '#2471a3',
    'Gemini':   '#c0392b', 'Grok':     '#e67e22',
    'DeepSeek': '#27ae60', 'Other':    '#7f8c8d',
}
FAM_ORDER = FS_FAM_ORDER if _HAS_FIG_STYLE else \
    ['GPT', 'Claude', 'Gemini', 'Grok', 'DeepSeek', 'Other']

_save = fs_savefig if _HAS_FIG_STYLE else \
    lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))


def get_model_family(name):
    name = str(name).lower()
    if any(x in name for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude'   in name: return 'Claude'
    elif 'gemini'   in name: return 'Gemini'
    elif 'grok'     in name: return 'Grok'
    elif 'deepseek' in name: return 'DeepSeek'
    return 'Other'


def shorten_name(name, max_len=30):
    name = str(name)
    name = re.sub(r'[_-]?pass[_-]?\d+', '', name, flags=re.IGNORECASE)
    if len(name) > max_len:
        name = name[:max_len-2] + '..'
    return name


# ══════════════════════════════════════════════════════════════════════════
# DIVERGENCE METRICS
# ══════════════════════════════════════════════════════════════════════════

def rmsrd(rank1, rank2, n=None):
    """
    Root Mean Squared Rank Displacement, normalized.
    RMSRD = sqrt(mean(Δ²)) / (N−1)
    Range [0, ~0.58] for uniform random permutation; 0 = perfect agreement.
    Quadratic penalty: shift of 10 costs 4× more than shift of 5.
    """
    r1 = np.asarray(rank1, dtype=float)
    r2 = np.asarray(rank2, dtype=float)
    if n is None:
        n = len(r1)
    delta = r1 - r2
    return np.sqrt(np.mean(delta ** 2)) / max(n - 1, 1)


def quadratic_weighted_kappa(rank1, rank2):
    """
    Quadratic Weighted Kappa for ordinal rank agreement.
    Like Cohen's κ but penalizes larger disagreements quadratically.
    1 = perfect agreement, 0 = chance-level, <0 = worse than chance.
    """
    r1 = np.asarray(rank1, dtype=int)
    r2 = np.asarray(rank2, dtype=int)
    n = len(r1)
    if n < 2:
        return np.nan

    min_r = min(r1.min(), r2.min())
    max_r = max(r1.max(), r2.max())
    num_cats = max_r - min_r + 1

    # Observed agreement matrix
    observed = np.zeros((num_cats, num_cats))
    for a, b in zip(r1 - min_r, r2 - min_r):
        observed[a, b] += 1
    observed /= n

    # Expected agreement (outer product of marginals)
    hist1 = np.bincount(r1 - min_r, minlength=num_cats) / n
    hist2 = np.bincount(r2 - min_r, minlength=num_cats) / n
    expected = np.outer(hist1, hist2)

    # Quadratic weight matrix
    weights = np.zeros((num_cats, num_cats))
    for i in range(num_cats):
        for j in range(num_cats):
            weights[i, j] = (i - j) ** 2 / max((num_cats - 1) ** 2, 1)

    num = np.sum(weights * observed)
    den = np.sum(weights * expected)
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den


def mean_absolute_rank_shift(rank1, rank2):
    """Simple MAD for comparison."""
    return np.mean(np.abs(np.asarray(rank1) - np.asarray(rank2)))


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING & RANKING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    overall_path = os.path.join(POSTHOC_DIR, "1_jsr_vs_theta_all_models.csv")
    if not os.path.exists(overall_path):
        raise FileNotFoundError(
            f"Run jsr_vs_theta_posthoc.py first.\nExpected: {overall_path}")
    overall = pd.read_csv(overall_path)
    print(f"Loaded overall: {len(overall)} rows, "
          f"models: {overall['irt_model'].unique().tolist()}")

    lang_path = os.path.join(
        POSTHOC_DIR, "2_jsr_vs_theta_minus_delta_all_models.csv")
    lang_df = pd.DataFrame()
    if os.path.exists(lang_path):
        lang_df = pd.read_csv(lang_path)
        print(f"Loaded language: {len(lang_df)} rows")
    return overall, lang_df


def compute_overall_ranks(overall, irt_model):
    df = overall[overall['irt_model'] == irt_model].copy()
    df = df.dropna(subset=['JSR', 'theta'])
    # Rank 1 = least safe
    df['JSR_Rank']   = df['JSR'].rank(ascending=False, method='min').astype(int)
    df['Theta_Rank'] = df['theta'].rank(ascending=True, method='min').astype(int)
    df['Rank_Delta'] = df['JSR_Rank'] - df['Theta_Rank']
    return df.sort_values('JSR_Rank')


def compute_lang_ranks(lang_df, irt_model):
    df = lang_df[lang_df['irt_model'] == irt_model].copy()
    df = df.dropna(subset=['JSR_lang', 'theta_minus_delta'])
    all_rows = []
    for lang, grp in df.groupby('language'):
        grp = grp.copy()
        grp['JSR_Rank']        = grp['JSR_lang'].rank(
            ascending=False, method='min').astype(int)
        grp['ThetaDelta_Rank'] = grp['theta_minus_delta'].rank(
            ascending=True, method='min').astype(int)
        grp['Rank_Delta']      = grp['JSR_Rank'] - grp['ThetaDelta_Rank']
        all_rows.append(grp)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Top-K Movers (replaces spaghetti bump chart)
# ══════════════════════════════════════════════════════════════════════════

def plot_top_movers(overall, irt_model, top_k=20):
    """Show only the models with biggest rank divergence."""
    df = compute_overall_ranks(overall, irt_model)
    df['abs_delta'] = df['Rank_Delta'].abs()
    top = df.nlargest(top_k, 'abs_delta').sort_values('Rank_Delta')

    h = max(3.0, len(top) * 0.22)
    if _HAS_FIG_STYLE:
        fig, ax = make_fig(n_panels=1, height_override=h)
        if isinstance(ax, np.ndarray): ax = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(5.5, h))

    labels = [shorten_name(t) for t in top['test_taker']]
    deltas = top['Rank_Delta'].values
    colors = [FAM_COLORS.get(f, '#7f8c8d') for f in top['model_family']]

    ax.barh(range(len(top)), deltas, color=colors,
            edgecolor='black', linewidth=0.3, alpha=0.85)

    for i, (_, row) in enumerate(top.iterrows()):
        ha = 'right' if row['Rank_Delta'] > 0 else 'left'
        offset = -0.3 if row['Rank_Delta'] > 0 else 0.3
        ax.text(deltas[i] + offset, i,
                f"JSR#{int(row['JSR_Rank'])}→θ#{int(row['Theta_Rank'])}",
                va='center', ha=ha, fontsize=4, color='#333')

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(r'Rank $\Delta$ (JSR $-$ $\theta$)')
    ax.set_title(f'Top {len(top)} Rank Divergences ({irt_model})')

    handles = [mpatches.Patch(color=c, label=f) for f, c in FAM_COLORS.items()
               if f in top['model_family'].values]
    ax.legend(handles=handles, fontsize=4, loc='lower right')

    path = os.path.join(RESULTS_DIR, f"top_movers_{irt_model}.png")
    _save(fig, path)
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Divergence Distribution + Metrics Card
# ══════════════════════════════════════════════════════════════════════════

def plot_divergence_distribution(overall, irt_model):
    """Histogram of rank deltas + divergence metrics summary."""
    df = compute_overall_ranks(overall, irt_model)
    deltas = df['Rank_Delta'].values
    n = len(df)

    _c1 = C_BLUE   if _HAS_FIG_STYLE else '#2471a3'
    _c2 = C_RED    if _HAS_FIG_STYLE else '#c0392b'
    _c3 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98'

    rm   = rmsrd(df['JSR_Rank'].values, df['Theta_Rank'].values, n)
    qwk  = quadratic_weighted_kappa(df['JSR_Rank'].values,
                                    df['Theta_Rank'].values)
    mad  = mean_absolute_rank_shift(df['JSR_Rank'].values,
                                    df['Theta_Rank'].values)
    rho, _ = spearmanr(df['JSR_Rank'].values, df['Theta_Rank'].values)

    if _HAS_FIG_STYLE:
        fig, axes = make_fig(n_panels=2, height_override=3.2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Left: histogram
    ax = axes[0]
    bins = np.arange(deltas.min() - 0.5, deltas.max() + 1.5, 1)
    ax.hist(deltas, bins=bins, color=_c1, edgecolor='black',
            linewidth=0.3, alpha=0.8)
    ax.axvline(0, color=_c2, linewidth=0.8, linestyle='--')
    ax.axvline(np.mean(deltas), color=_c3, linewidth=0.8, linestyle='-',
               label=f'Mean={np.mean(deltas):.1f}')
    ax.set_xlabel(r'Rank $\Delta$ (JSR $-$ $\theta$)')
    ax.set_ylabel('Count')
    ax.set_title('Rank Divergence Distribution')
    ax.legend(fontsize=5)

    # Right: metrics card
    ax = axes[1]
    ax.axis('off')
    metrics_text = (
        f"Metrics ({irt_model}, N={n})\n"
        f"{'─' * 28}\n"
        f"RMSRD      = {rm:.3f}\n"
        f"QW κ       = {qwk:.3f}\n"
        f"Mean |Δ|   = {mad:.1f}\n"
        f"Spearman ρ = {rho:.3f}\n"
        f"{'─' * 28}\n"
        f"Max shift: {int(np.max(np.abs(deltas)))}\n"
        f"|Δ|≥5: {(np.abs(deltas) >= 5).sum()}/{n}\n"
        f"|Δ|≥10: {(np.abs(deltas) >= 10).sum()}/{n}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=6, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    path = os.path.join(RESULTS_DIR, f"divergence_summary_{irt_model}.png")
    _save(fig, path)
    print(f"  Saved: {os.path.basename(path)}")

    return {'RMSRD': rm, 'QWK': qwk, 'MAD': mad, 'Spearman_rho': rho, 'N': n}


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Family-Level Divergence (aggregated, not per-model)
# ══════════════════════════════════════════════════════════════════════════

def plot_family_divergence(overall, irt_model):
    """Box/strip plot of rank delta by family + RMS per family."""
    df = compute_overall_ranks(overall, irt_model)
    _c2 = C_RED if _HAS_FIG_STYLE else '#c0392b'

    present_fams = [f for f in FAM_ORDER if f in df['model_family'].values]
    if not present_fams:
        return

    if _HAS_FIG_STYLE:
        fig, axes = make_fig(n_panels=2, height_override=3.2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Left: box + strip
    ax = axes[0]
    box_data = [df[df['model_family'] == f]['Rank_Delta'].values
                for f in present_fams]
    bp = ax.boxplot(box_data, labels=present_fams, patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color='black', lw=0.8))
    for patch, fam in zip(bp['boxes'], present_fams):
        patch.set_facecolor(FAM_COLORS.get(fam, '#7f8c8d'))
        patch.set_alpha(0.6)
        patch.set_linewidth(0.4)

    for i, fam in enumerate(present_fams):
        fam_data = df[df['model_family'] == fam]['Rank_Delta'].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(fam_data))
        ax.scatter(np.full_like(fam_data, i + 1, dtype=float) + jitter,
                   fam_data, color=FAM_COLORS.get(fam, '#7f8c8d'),
                   s=10, alpha=0.7, edgecolors='black', linewidths=0.2,
                   zorder=3)

    ax.axhline(0, color=_c2, linewidth=0.6, linestyle='--')
    ax.set_ylabel(r'Rank $\Delta$')
    ax.set_title('Divergence by Family')
    ax.tick_params(axis='x', rotation=30)

    # Right: RMS rank shift per family
    ax = axes[1]
    fam_metrics = []
    for fam in present_fams:
        sub = df[df['model_family'] == fam]
        if len(sub) >= 1:
            rms = np.sqrt(np.mean(sub['Rank_Delta'].values ** 2))
            fam_metrics.append({
                'family': fam, 'RMS_shift': rms,
                'mean_delta': sub['Rank_Delta'].mean(), 'n': len(sub)
            })

    if fam_metrics:
        fm_df = pd.DataFrame(fam_metrics).sort_values('RMS_shift',
                                                       ascending=True)
        colors = [FAM_COLORS.get(f, '#7f8c8d') for f in fm_df['family']]
        bars = ax.barh(fm_df['family'], fm_df['RMS_shift'], color=colors,
                       edgecolor='black', linewidth=0.3)
        for bar, val, md in zip(bars, fm_df['RMS_shift'],
                                fm_df['mean_delta']):
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f} ({md:+.1f})',
                    va='center', fontsize=4)
        ax.set_xlabel('RMS Rank Shift')
        ax.set_title('Instability by Family')

    path = os.path.join(RESULTS_DIR, f"family_divergence_{irt_model}.png")
    _save(fig, path)
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Per-Language Divergence Metrics
# ══════════════════════════════════════════════════════════════════════════

def plot_language_divergence(lang_df, irt_model):
    """RMSRD and QWK per language — which languages does JSR misrank most."""
    df = compute_lang_ranks(lang_df, irt_model)
    if len(df) == 0:
        return None

    _c1 = C_BLUE   if _HAS_FIG_STYLE else '#2471a3'
    _c2 = C_RED    if _HAS_FIG_STYLE else '#c0392b'
    _c3 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98'

    languages = sorted(df['language'].unique())
    lang_metrics = []

    for lang in languages:
        grp = df[df['language'] == lang]
        if len(grp) < 3:
            continue
        rm  = rmsrd(grp['JSR_Rank'].values,
                     grp['ThetaDelta_Rank'].values, len(grp))
        qwk = quadratic_weighted_kappa(grp['JSR_Rank'].values,
                                        grp['ThetaDelta_Rank'].values)
        rho, _ = spearmanr(grp['JSR_Rank'].values,
                           grp['ThetaDelta_Rank'].values)
        mad = mean_absolute_rank_shift(grp['JSR_Rank'].values,
                                        grp['ThetaDelta_Rank'].values)
        lang_metrics.append({
            'language': lang, 'RMSRD': rm, 'QWK': qwk,
            'Spearman_rho': rho, 'MAD': mad, 'n': len(grp)
        })

    lm_df = pd.DataFrame(lang_metrics).sort_values('RMSRD', ascending=False)

    if _HAS_FIG_STYLE:
        fig, axes = make_fig(n_panels=3, height_override=3.5)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # RMSRD — red/purple/blue thresholds
    ax = axes[0]
    colors = [_c2 if v > 0.15 else _c3 if v > 0.10 else _c1
              for v in lm_df['RMSRD']]
    ax.barh(lm_df['language'], lm_df['RMSRD'], color=colors,
            edgecolor='black', linewidth=0.3)
    for i, (_, row) in enumerate(lm_df.iterrows()):
        ax.text(row['RMSRD'] + 0.002, i, f"{row['RMSRD']:.3f}",
                va='center', fontsize=4)
    ax.set_xlabel('RMSRD')
    ax.set_title('Divergence by Language')

    # QWK — blue/purple/red thresholds (inverted: high=good)
    ax = axes[1]
    colors_qwk = [_c1 if v > 0.8 else _c3 if v > 0.6 else _c2
                  for v in lm_df['QWK']]
    ax.barh(lm_df['language'], lm_df['QWK'], color=colors_qwk,
            edgecolor='black', linewidth=0.3)
    ax.axvline(0.8, color=_c1, linestyle='--', alpha=0.5, lw=0.5,
               label='Excellent')
    ax.axvline(0.6, color=_c3, linestyle='--', alpha=0.5, lw=0.5,
               label='Moderate')
    for i, (_, row) in enumerate(lm_df.iterrows()):
        ax.text(max(0, row['QWK']) + 0.005, i, f"{row['QWK']:.3f}",
                va='center', fontsize=4)
    ax.set_xlabel(r'QW $\kappa$')
    ax.set_title('Agreement by Language')
    ax.set_xlim(0, 1.45)  # <--- Forces empty space on the right side
    ax.legend(fontsize=4, loc='center right')
    ax = axes[2]
    ax.barh(lm_df['language'], lm_df['Spearman_rho'],
            color=_c1, edgecolor='black', linewidth=0.3)
    for i, (_, row) in enumerate(lm_df.iterrows()):
        ax.text(row['Spearman_rho'] + 0.005, i, f"{row['Spearman_rho']:.3f}",
                va='center', fontsize=4)
    ax.set_xlabel(r'Spearman $\rho$')
    ax.set_title('Correlation by Language')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        f"language_divergence_{irt_model}.png")
    _save(fig, path)
    print(f"  Saved: {os.path.basename(path)}")

    lm_df.to_csv(os.path.join(
        RESULTS_DIR, f"language_divergence_metrics_{irt_model}.csv"),
        index=False)
    return lm_df


# ══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Family × Language Heatmap (mean rank shift)
# ══════════════════════════════════════════════════════════════════════════

def plot_family_lang_heatmap(lang_df, irt_model):
    df = compute_lang_ranks(lang_df, irt_model)
    if len(df) == 0:
        return

    pivot = df.pivot_table(index='model_family', columns='language',
                           values='Rank_Delta', aggfunc='mean')
    present = [f for f in FAM_ORDER if f in pivot.index]
    pivot = pivot.reindex(present)

    _cmap = CMAP_DIV if _HAS_FIG_STYLE else 'RdBu_r'
    h = max(2.0, len(pivot) * 0.45)
    if _HAS_FIG_STYLE:
        fig, ax = make_fig(n_panels=1, height_override=h)
        if isinstance(ax, np.ndarray): ax = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(5.5, h))

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 3)
    sns.heatmap(pivot, cmap=_cmap, center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.1f', annot_kws={'size': 5},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': r'Mean Rank $\Delta$', 'shrink': 0.8},
                ax=ax)
    ax.set_title(f'Family × Language Mean Rank Shift ({irt_model})')
    ax.set_ylabel('')
    ax.set_xlabel('Language')

    path = os.path.join(RESULTS_DIR,
                        f"family_lang_heatmap_{irt_model}.png")
    _save(fig, path)
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════

def save_rank_tables(overall, lang_df, irt_model):
    """Save CSV rank tables for overall and per-language."""
    # Overall
    df = compute_overall_ranks(overall, irt_model)
    cols = ['test_taker', 'model_family', 'JSR', 'theta',
            'JSR_Rank', 'Theta_Rank', 'Rank_Delta']
    out = df[cols].copy()
    out['JSR']   = out['JSR'].round(4)
    out['theta'] = out['theta'].round(3)
    path = os.path.join(RESULTS_DIR,
                        f"rank_table_overall_{irt_model}.csv")
    out.to_csv(path, index=False)
    print(f"  Saved: {os.path.basename(path)} ({len(out)} models)")

    # Per-language
    if len(lang_df) > 0 and irt_model in lang_df['irt_model'].values:
        ldf = compute_lang_ranks(lang_df, irt_model)
        if len(ldf) > 0:
            cols_l = ['test_taker', 'language', 'model_family',
                      'JSR_lang', 'theta_minus_delta',
                      'JSR_Rank', 'ThetaDelta_Rank', 'Rank_Delta']
            out_l = ldf[cols_l].copy()
            out_l['JSR_lang']          = out_l['JSR_lang'].round(4)
            out_l['theta_minus_delta'] = out_l['theta_minus_delta'].round(3)
            path_l = os.path.join(
                RESULTS_DIR, f"rank_table_per_language_{irt_model}.csv")
            out_l.to_csv(path_l, index=False)
            print(f"  Saved: {os.path.basename(path_l)} ({len(out_l)} rows)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if _HAS_FIG_STYLE:
        apply_style()
    print("=" * 60)
    print("RANK DIVERGENCE ANALYSIS v2 (with divergence metrics)")
    print("=" * 60)

    overall, lang_df = load_data()
    irt_models = [m for m in [PRIMARY_IRT, '1PL', 'GRM']
                  if m in overall['irt_model'].values]

    all_metrics = []

    for irt_model in irt_models:
        print(f"\n{'─' * 50}")
        print(f"IRT Model: {irt_model}")
        print(f"{'─' * 50}")

        # Tables
        save_rank_tables(overall, lang_df, irt_model)

        # Overall divergence
        print("\n[Overall Divergence]")
        metrics = plot_divergence_distribution(overall, irt_model)
        metrics['irt_model'] = irt_model
        all_metrics.append(metrics)

        # Top movers
        plot_top_movers(overall, irt_model, top_k=20)

        # Family divergence
        plot_family_divergence(overall, irt_model)

        # Per-language
        if len(lang_df) > 0 and irt_model in lang_df['irt_model'].values:
            print("\n[Per-Language Divergence]")
            plot_language_divergence(lang_df, irt_model)
            plot_family_lang_heatmap(lang_df, irt_model)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(RESULTS_DIR,
                                    "divergence_metrics_summary.csv"),
                      index=False)
    for _, row in summary_df.iterrows():
        print(f"  [{row['irt_model']}]  RMSRD={row['RMSRD']:.3f}  "
              f"QWK={row['QWK']:.3f}  MAD={row['MAD']:.1f}  "
              f"ρ={row['Spearman_rho']:.3f}  N={row['N']}")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
