# -*- coding: utf-8 -*-
"""
H1 Direct Test: Isolating δ_jL (Model-Language Aptitude)
=========================================================
δ_jL answers: "How much worse/better is model j in language L
              compared to model j's own baseline θ?"

δ > 0  → model is BETTER in this language than its baseline
δ < 0  → model is WORSE in this language than its baseline
δ = 0  → English (by constraint)

H1 prediction: if models have genuine ability deficits in certain
languages (not just harder prompts), δ should be systematically
negative for low-resource languages.

Reads from:
  results_experiment_A/A4_person_fit_2pl.csv   → θ per test_taker
  results_experiment_B/B5_validation_data.csv   → θ+δ per (test_taker, language)

Produces:
  1. Family × Language heatmap of mean δ
  2. Per-model δ heatmap (all 61 models)
  3. δ distribution violin/box by language
  4. Summary stats + significance tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

from scipy.stats import ttest_1samp, wilcoxon
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

EXP_A_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results_experiment_A")
EXP_B_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results_experiment_B")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_h1_delta")
os.makedirs(RESULTS_DIR, exist_ok=True)

LANG_ORDER = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']

FAM_COLORS = {
    'GPT':      '#3498db',
    'Claude':   '#9b59b6',
    'Gemini':   '#2ecc71',
    'Grok':     '#e74c3c',
    'DeepSeek': '#f39c12',
    'Other':    '#95a5a6',
}
FAM_ORDER = ['Claude', 'DeepSeek', 'Gemini', 'GPT', 'Grok', 'Other']

_c1 = C_BLUE if _HAS_FIG_STYLE else '#2471a3'
_c2 = C_RED if _HAS_FIG_STYLE else '#c0392b'
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


def extract_base_model(name):
    return re.sub(r'[_-]?pass[_-]?\d+', '', str(name),
                  flags=re.IGNORECASE).strip()


def shorten_name(name, max_len=28):
    name = str(name)
    name = re.sub(r'[_-]?pass[_-]?\d+', '', name, flags=re.IGNORECASE)
    if len(name) > max_len:
        name = name[:max_len-2] + '..'
    return name


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load θ and (θ+δ), compute δ
# ══════════════════════════════════════════════════════════════════════════

def load_and_compute_delta():
    """
    θ from A4 person fit (one value per test_taker).
    θ+δ from B5 validation (one value per test_taker × language).
    δ = (θ+δ) − θ.
    """
    # ── Load θ ────────────────────────────────────────────────────
    theta_path = os.path.join(EXP_A_DIR, "A4_person_fit_2pl.csv")
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Need: {theta_path}")
    theta_df = pd.read_csv(theta_path)
    theta_lookup = theta_df.set_index('student')['theta'].to_dict()
    print(f"Loaded θ for {len(theta_lookup)} test_takers from A4")

    # ── Load θ+δ ──────────────────────────────────────────────────
    b5_path = os.path.join(EXP_B_DIR, "B5_validation_data.csv")
    if not os.path.exists(b5_path):
        raise FileNotFoundError(f"Need: {b5_path}")
    b5 = pd.read_csv(b5_path)
    print(f"Loaded B5: {len(b5)} rows")

    # B5 'theta' column = θ+δ (language-adjusted ability)
    # Aggregate to (base_model, language) level
    theta_eff = (b5.groupby(['base_model', 'language'])['theta']
                   .mean().reset_index()
                   .rename(columns={'theta': 'theta_plus_delta'}))

    # ── Match θ to base_model ─────────────────────────────────────
    # A4 uses test_taker (with pass suffix), B5 uses base_model
    # Build base_model → mean θ mapping
    base_theta = {}
    for tt, th in theta_lookup.items():
        bm = extract_base_model(tt)
        if bm not in base_theta:
            base_theta[bm] = []
        base_theta[bm].append(th)
    base_theta = {bm: np.mean(vals) for bm, vals in base_theta.items()}

    # ── Compute δ ─────────────────────────────────────────────────
    theta_eff['theta'] = theta_eff['base_model'].map(base_theta)
    theta_eff = theta_eff.dropna(subset=['theta'])
    theta_eff['delta'] = theta_eff['theta_plus_delta'] - theta_eff['theta']
    theta_eff['family'] = theta_eff['base_model'].apply(get_model_family)

    # Verify: English delta should be ~0
    en_delta = theta_eff[theta_eff['language'] == 'en']['delta']
    print(f"\nSanity check — English δ: mean={en_delta.mean():.4f}, "
          f"std={en_delta.std():.4f}  (should be ~0)")

    print(f"Computed δ for {theta_eff['base_model'].nunique()} models × "
          f"{theta_eff['language'].nunique()} languages = {len(theta_eff)} rows")

    return theta_eff


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Family × Language Mean δ Heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_family_delta_heatmap(df):
    """
    H1 Test rendered as a horizontal row.
    Columns = Language, Rows = Family. Colorbar moved to bottom.
    """
    pivot = df.pivot_table(index='family', columns='language',
                           values='delta', aggfunc='mean')
    present_langs = [l for l in LANG_ORDER if l in pivot.columns]
    present_fams = [f for f in FAM_ORDER if f in pivot.index]
    pivot = pivot.reindex(index=present_fams, columns=present_langs)

    fig_width = len(present_langs) * 1.0 + 2
    fig_height = len(present_fams) * 0.5 + 1.5
    
    # FIX: Initialize tight layout here instead of calling plt.tight_layout() later
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout='tight')

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.1)
    
    sns.heatmap(pivot, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.3f', linewidths=1, linecolor='white',
                cbar_kws={'label': 'Mean δ (model-language aptitude)',
                          'orientation': 'horizontal', 
                          'shrink': 0.6, 'pad': 0.2},
                ax=ax)

    ax.set_title(
        'H1 Test: Mean δ by Family × Language\n'
        'Blue = better than own baseline  |  Red = worse than own baseline',
        fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Family', fontsize=11)
    ax.set_xlabel('Language', fontsize=11)

    # FIX: Removed plt.tight_layout() 
    path = os.path.join(RESULTS_DIR, "family_delta_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")

    return pivot

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-Model δ Heatmap
# ══════════════════════════════════════════════════════════════════════════

def plot_model_delta_heatmap(df):
    """Full model × language δ heatmap, horizontal layout."""
    pivot = df.pivot_table(index='language', columns='base_model',
                           values='delta', aggfunc='first')

    present_langs = [l for l in LANG_ORDER if l in pivot.index]
    pivot = pivot.reindex(present_langs)

    families = df.drop_duplicates('base_model').set_index('base_model')['family']
    thetas = df.drop_duplicates('base_model').set_index('base_model')['theta']
    fam_rank = {f: i for i, f in enumerate(FAM_ORDER)}

    model_order = []
    for model in pivot.columns:
        fam = families.get(model, 'Other')
        fr = fam_rank.get(fam, 99)
        th = thetas.get(model, 0)
        model_order.append((fr, -th, model))
    model_order.sort()
    pivot = pivot[[m for _, _, m in model_order]]

    n_models = len(pivot.columns)
    
    fig_width = max(16, n_models * 0.35 + 2)
    fig_height = len(present_langs) * 0.4 + 2
    
    # FIX: Initialize tight layout here
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout='tight')

    short_labels = [shorten_name(n) for n in pivot.columns]
    vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.1)

    sns.heatmap(pivot, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.2f', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'δ_jL', 'orientation': 'horizontal', 
                          'shrink': 0.3, 'pad': 0.25},
                annot_kws={'fontsize': 6}, ax=ax)

    ax.set_xticklabels(short_labels, fontsize=6, rotation=90, color='black')
    ax.set_yticklabels(present_langs, fontsize=9, rotation=0)

    ax.set_title(
        'δ_jL: Model-Language Aptitude (all models)\n'
        'Blue = better than own baseline  |  Red = worse',
        fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Language', fontsize=11)

    # FIX: Removed plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_delta_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")
# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — δ Distribution by Language (violin + box)
# ══════════════════════════════════════════════════════════════════════════

def plot_delta_violins(df):
    """
    Shows spread of δ across models for each language.
    Strictly uses the 3-color scheme (Blue, Red, Purple).
    """
    present_langs = [l for l in LANG_ORDER if l in df['language'].values]
    plot_df = df[df['language'].isin(present_langs)].copy()

    fig, ax = plt.subplots(figsize=(len(present_langs) * 1.2 + 1, 4))

    # Violin (Blue)
    parts = ax.violinplot(
        [plot_df[plot_df['language'] == l]['delta'].dropna().values
         for l in present_langs],
        positions=range(len(present_langs)),
        showmeans=True, showmedians=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(_c1)  # C_BLUE
        pc.set_alpha(0.4)
    parts['cmeans'].set_color(_c2)     # C_RED
    parts['cmedians'].set_color('black')

    # Overlay individual points (Purple)
    for i, lang in enumerate(present_langs):
        grp = plot_df[plot_df['language'] == lang]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grp))
        
        ax.scatter(
            np.full(len(grp), i) + jitter,
            grp['delta'].values,
            color=_c3,  # C_PURPLE
            s=20, alpha=0.7, edgecolors='black', linewidths=0.3,
            zorder=3)

    # Reference Line (Red)
    ax.axhline(0, color=_c2, linewidth=1, linestyle='--',
               label='δ = 0 (no language effect)')
               
    ax.set_xticks(range(len(present_langs)))
    ax.set_xticklabels(present_langs, fontsize=10)
    ax.set_ylabel('δ_jL', fontsize=11)
    ax.set_xlabel('Language', fontsize=11)
    ax.set_title(
        'Distribution of δ Across Models by Language\n'
        'Below 0 = models worse than own baseline in this language',
        fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "delta_violins.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(df):
    """Test whether δ is systematically ≠ 0 for each language."""
    present_langs = [l for l in LANG_ORDER
                     if l in df['language'].values and l != 'en']

    print(f"\n{'=' * 70}")
    print("H1 STATISTICAL TESTS: Is δ systematically ≠ 0?")
    print(f"{'=' * 70}")
    print(f"{'Language':<8} {'mean δ':>8} {'median':>8} {'std':>8} "
          f"{'t-stat':>8} {'p(t)':>10} {'p(W)':>10} {'n':>4}  Interpretation")
    print("─" * 90)

    rows = []
    for lang in present_langs:
        vals = df[df['language'] == lang]['delta'].dropna().values
        n = len(vals)
        if n < 3:
            continue

        mean_d = np.mean(vals)
        med_d = np.median(vals)
        std_d = np.std(vals, ddof=1)
        t_stat, p_t = ttest_1samp(vals, 0)

        try:
            w_stat, p_w = wilcoxon(vals)
        except ValueError:
            p_w = np.nan

        # Interpretation
        if p_t < 0.01 and mean_d < -0.05:
            interp = "** Models WORSE (H1 supported)"
        elif p_t < 0.05 and mean_d < -0.02:
            interp = "*  Models slightly worse"
        elif p_t < 0.01 and mean_d > 0.05:
            interp = "** Models BETTER"
        elif p_t < 0.05 and mean_d > 0.02:
            interp = "*  Models slightly better"
        else:
            interp = "   No systematic effect"

        print(f"{lang:<8} {mean_d:>8.4f} {med_d:>8.4f} {std_d:>8.4f} "
              f"{t_stat:>8.2f} {p_t:>10.2e} {p_w:>10.2e} {n:>4}  {interp}")

        rows.append({
            'language': lang, 'mean_delta': mean_d, 'median_delta': med_d,
            'std_delta': std_d, 'n': n,
            't_stat': t_stat, 'p_ttest': p_t, 'p_wilcoxon': p_w,
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "h1_delta_stats.csv"),
                    index=False)
    print(f"\n  Saved: h1_delta_stats.csv")

    # ── Per-family breakdown ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PER-FAMILY MEAN δ BY LANGUAGE")
    print(f"{'=' * 70}")
    fam_lang = df[df['language'] != 'en'].pivot_table(
        index='family', columns='language', values='delta', aggfunc='mean')
    present_fams = [f for f in FAM_ORDER if f in fam_lang.index]
    fam_lang = fam_lang.reindex(index=present_fams,
                                 columns=[l for l in LANG_ORDER if l != 'en'])
    print(fam_lang.round(3).to_string())

    fam_lang.to_csv(os.path.join(RESULTS_DIR, "h1_family_lang_delta.csv"))
    print(f"\n  Saved: h1_family_lang_delta.csv")

    # ── Grok breakdown ────────────────────────────────────────────
    grok = df[(df['family'] == 'Grok') & (df['language'] != 'en')]
    if len(grok) > 0:
        print(f"\n{'=' * 70}")
        print("GROK δ DETAIL")
        print(f"{'=' * 70}")
        grok_pivot = grok.pivot_table(
            index='base_model', columns='language', values='delta',
            aggfunc='first')
        grok_pivot = grok_pivot[[l for l in LANG_ORDER
                                  if l in grok_pivot.columns and l != 'en']]
        print(grok_pivot.round(3).to_string())

        # Grok: are non-English δ positive? (reversed pattern)
        grok_vals = grok['delta'].values
        mean_g = np.mean(grok_vals)
        t_g, p_g = ttest_1samp(grok_vals, 0)
        print(f"\nGrok non-English δ: mean={mean_g:.4f}, t={t_g:.2f}, "
              f"p={p_g:.2e}")
        if mean_g > 0 and p_g < 0.05:
            print("  → Grok is systematically BETTER outside English "
                  "(reversed pattern confirmed)")
        elif mean_g < 0 and p_g < 0.05:
            print("  → Grok is worse outside English (standard pattern)")
        else:
            print("  → No systematic direction for Grok δ")

    return stats_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if _HAS_FIG_STYLE: apply_style()
    print("=" * 60)
    print("H1 DIRECT TEST: Isolating δ_jL")
    print("=" * 60)

    df = load_and_compute_delta()

    # Save raw delta table
    df.to_csv(os.path.join(RESULTS_DIR, "delta_all.csv"), index=False)
    print(f"  Saved: delta_all.csv")

    # Figures
    print("\n[Figure 1] Family × Language mean δ")
    pivot = plot_family_delta_heatmap(df)

    print("\n[Figure 2] Per-model δ heatmap")
    plot_model_delta_heatmap(df)

    print("\n[Figure 3] δ distribution violins")
    plot_delta_violins(df)

    # Stats
    stats_df = compute_stats(df)

    # ── Paper-ready summary ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PAPER SUMMARY")
    print(f"{'=' * 60}")

    non_en = df[df['language'] != 'en']
    overall_mean = non_en['delta'].mean()
    overall_t, overall_p = ttest_1samp(non_en['delta'].values, 0)

    print(f"Overall non-English δ: mean={overall_mean:.4f}, "
          f"t={overall_t:.2f}, p={overall_p:.2e}")

    # Per language: which show significant negative δ?
    sig_neg = stats_df[(stats_df['p_ttest'] < 0.05) &
                       (stats_df['mean_delta'] < 0)]
    sig_pos = stats_df[(stats_df['p_ttest'] < 0.05) &
                       (stats_df['mean_delta'] > 0)]

    if len(sig_neg) > 0:
        print(f"\nLanguages with significant NEGATIVE δ (H1 ability deficit):")
        for _, r in sig_neg.iterrows():
            print(f"  {r['language']}: mean δ = {r['mean_delta']:.4f} "
                  f"(p = {r['p_ttest']:.2e})")

    if len(sig_pos) > 0:
        print(f"\nLanguages with significant POSITIVE δ (models better):")
        for _, r in sig_pos.iterrows():
            print(f"  {r['language']}: mean δ = {r['mean_delta']:.4f} "
                  f"(p = {r['p_ttest']:.2e})")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()