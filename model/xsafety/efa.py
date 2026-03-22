# -*- coding: utf-8 -*-
"""
EFA analysis — XSafety
========================
Identical to model/efa.py, adapted for XSafety:
  - Uses 'category' column (single string) instead of 'tags' (list)
  - No explode needed — categories are already one per row
  - Input: XSafety_Dataset.csv

14 XSafety safety categories:
  Crimes_And_Illegal_Activities, Ethics_And_Morality, Goal_Hijacking,
  Inquiry_With_Unsafe_Opinion, Insult, Mental_Health, Physical_Harm,
  Privacy_And_Property, Prompt_Leaking, Reverse_Exposure,
  Role_Play_Instruction, Unfairness_And_Discrimination,
  Unsafe_Instruction_Topic, commonsense
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from fig_style import *

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

apply_style()

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "safety-data", "xsafety", "xsafety_pass_graded.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_SCREE_PLOT = os.path.join(RESULTS_DIR, "efa_scree_plot_likert.png")
OUTPUT_LOADINGS   = os.path.join(RESULTS_DIR, "efa_factor_loadings_likert.csv")

COL_MODEL = "test_taker"
COL_LANG  = "language"
COL_SCORE = "judge_score"
COL_TAGS  = "category"   # XSafety uses 'category' (single string, not list)


def run_efa():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if COL_TAGS not in df.columns:
        raise ValueError(f"'{COL_TAGS}' column not found — check INPUT_FILE")
    if COL_SCORE not in df.columns:
        raise ValueError(f"'{COL_SCORE}' column not found")

    df = df[df[COL_SCORE].isin([1, 2, 3, 4, 5])].copy()
    df['score'] = df[COL_SCORE].astype(float)

    col_name = 'test_taker' if 'test_taker' in df.columns else 'config'
    df['student_id'] = df[col_name] + "::" + df[COL_LANG]

    category_scores = df.groupby(['student_id', COL_TAGS])['score'].mean().reset_index()
    efa_matrix = category_scores.pivot(index='student_id', columns=COL_TAGS, values='score')

    if efa_matrix.isnull().sum().sum() > 0:
        efa_matrix = efa_matrix.fillna(efa_matrix.mean())

    print(f"EFA matrix: {efa_matrix.shape[0]} students x {efa_matrix.shape[1]} categories")

    kmo_all, kmo_model = calculate_kmo(efa_matrix)
    print(f"KMO: {kmo_model:.3f} ({'good' if kmo_model > 0.8 else 'acceptable' if kmo_model > 0.6 else 'poor'})")

    fa = FactorAnalyzer(n_factors=min(efa_matrix.shape[1], 25), rotation=None)
    fa.fit(efa_matrix)
    ev, v = fa.get_eigenvalues()

    for i, val in enumerate(ev[:5]):
        print(f"  Factor {i+1}: {val:.4f}")

    ratio = ev[0] / ev[1]
    print(f"Dominance ratio: {ratio:.2f} ({'unidimensional' if ratio > 3.0 else 'possibly multi-dimensional'})")

    fig, ax = make_fig(n_panels=1)
    ax.plot(range(1, len(ev) + 1), ev, marker='o', markersize=3,
            linestyle='--', color=C_BLUE, linewidth=0.8)
    ax.axhline(y=1, color=C_RED, linestyle='-', linewidth=0.7,
               label='Kaiser criterion (1.0)')
    ax.set_xlabel('Factor')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree plot: latent structure of LLM safety (XSafety)')
    ax.legend()
    savefig(fig, OUTPUT_SCREE_PLOT)

    n_factors = min(3, efa_matrix.shape[1] - 1)
    fa_rotated = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa_rotated.fit(efa_matrix)
    loadings = pd.DataFrame(fa_rotated.loadings_, index=efa_matrix.columns,
                            columns=[f'Factor{i+1}' for i in range(n_factors)])
    loadings.to_csv(OUTPUT_LOADINGS)
    print(f"Factor loadings saved to {OUTPUT_LOADINGS}")
    print("Top categories for Factor 1:")
    print(loadings['Factor1'].abs().sort_values(ascending=False).head(5))


def plot_jsr_heatmap():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [COL_MODEL, COL_LANG, COL_SCORE]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing columns: {required}")

    df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors='coerce')
    df['is_unsafe'] = (df[COL_SCORE] < 4).astype(int)

    jsr_data = df.groupby([COL_MODEL, COL_LANG])['is_unsafe'].mean().reset_index()
    heatmap_matrix = jsr_data.pivot(index=COL_MODEL, columns=COL_LANG, values='is_unsafe')

    # XSafety language order
    xs_lang_order = ["en", "zh", "ar", "bn", "de", "fr", "hi", "ja", "ru", "sp"]
    present_langs = [l for l in xs_lang_order if l in heatmap_matrix.columns]
    heatmap_matrix = heatmap_matrix[present_langs]

    heatmap_matrix['_mean_jsr'] = heatmap_matrix[present_langs].mean(axis=1)
    heatmap_matrix = heatmap_matrix.sort_values('_mean_jsr', ascending=False)
    heatmap_matrix = heatmap_matrix.drop(columns=['_mean_jsr'])
    heatmap_matrix = heatmap_matrix.T

    n_models = len(heatmap_matrix.columns)
    n_langs  = len(heatmap_matrix.index)
    fig_w = max(FULL_WIDTH, n_models * 0.35)
    fig_h = max(4, n_langs * 0.45)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        heatmap_matrix, ax=ax,
        annot=True, fmt=".2f", cmap="Reds",
        vmin=0.0, vmax=0.7, linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'JSR', 'shrink': 0.5},
        annot_kws={'fontsize': 8},
    )
    ax.set_title('XSafety: Jailbreak Success Rate by Model and Language', pad=20, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Language')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

    outfile = os.path.join(RESULTS_DIR, "jsr_heatmap.png")
    savefig(fig, outfile)


def plot_correlation_matrix():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [COL_MODEL, COL_TAGS, COL_SCORE]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing columns: {required}")

    df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors='coerce')
    df['is_unsafe'] = (df[COL_SCORE] < 4).astype(int)
    df = df.dropna(subset=[COL_TAGS]).copy()

    model_cat_performance = df.groupby(
        [COL_MODEL, COL_TAGS])['is_unsafe'].mean().reset_index()
    pivot_df = model_cat_performance.pivot(
        index=COL_MODEL, columns=COL_TAGS, values='is_unsafe')
    pivot_df  = pivot_df.fillna(0)
    corr_matrix = pivot_df.corr(method='pearson')

    n_cats = len(corr_matrix)
    side = min(FULL_WIDTH, max(3.5, n_cats * 0.25))
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        corr_matrix, ax=ax, cmap=CMAP_DIV, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, annot=False,
        cbar_kws={"shrink": 0.7, "label": "Pearson $r$"},
    )
    ax.set_title('XSafety: Safety category correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')

    outfile = os.path.join(RESULTS_DIR, "category_correlation.png")
    savefig(fig, outfile)


if __name__ == "__main__":
    run_efa()
    plot_jsr_heatmap()
    plot_correlation_matrix()
