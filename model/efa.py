# -*- coding: utf-8 -*-
"""
EFA analysis for safety unidimensionality,
plus JSR heatmap and safety category correlation visualizations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fig_style import *

import pandas as pd
import numpy as np
import ast
import re
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

apply_style()

DATA_DIR = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_SCREE_PLOT = os.path.join(RESULTS_DIR, "efa_scree_plot_likert.png")
OUTPUT_LOADINGS = os.path.join(RESULTS_DIR, "efa_factor_loadings_likert.csv")

COL_MODEL = "test_taker"
COL_LANG = "language"
COL_SCORE = "judge_score"
COL_TAGS = "tags"


def run_efa():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df_exploded = df.explode('tags')

    if 'judge_score' not in df_exploded.columns:
        raise ValueError("'judge_score' column not found")

    df_exploded = df_exploded[df_exploded['judge_score'].isin([1, 2, 3, 4, 5])]
    df_exploded['score'] = df_exploded['judge_score'].astype(float)

    col_name = 'test_taker' if 'test_taker' in df_exploded.columns else 'config'
    df_exploded['student_id'] = df_exploded[col_name] + "::" + df_exploded['language']

    category_scores = df_exploded.groupby(['student_id', 'tags'])['score'].mean().reset_index()
    efa_matrix = category_scores.pivot(index='student_id', columns='tags', values='score')

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

    # ── Scree plot ────────────────────────────────────────────────
    fig, ax = make_fig(n_panels=1)
    ax.plot(range(1, len(ev) + 1), ev, marker='o', markersize=3,
            linestyle='--', color=C_BLUE, linewidth=0.8)
    ax.axhline(y=1, color=C_RED, linestyle='-', linewidth=0.7,
               label='Kaiser criterion (1.0)')
    ax.set_xlabel('Factor')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree plot: latent structure of LLM safety')
    ax.legend()
    savefig(fig, OUTPUT_SCREE_PLOT)

    fa_rotated = FactorAnalyzer(n_factors=3, rotation='varimax')
    fa_rotated.fit(efa_matrix)
    loadings = pd.DataFrame(fa_rotated.loadings_, index=efa_matrix.columns,
                            columns=['Factor1', 'Factor2', 'Factor3'])
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

    # Reorder columns to standard language order
    present_langs = [l for l in LANG_ORDER if l in heatmap_matrix.columns]
    heatmap_matrix = heatmap_matrix[present_langs]

    # Sort rows by family then mean JSR
    heatmap_matrix['_family'] = [get_family(m) for m in heatmap_matrix.index]
    fam_rank = {f: i for i, f in enumerate(FAM_ORDER)}
    heatmap_matrix['_fam_rank'] = heatmap_matrix['_family'].map(fam_rank).fillna(99)
    heatmap_matrix['_mean_jsr'] = heatmap_matrix[present_langs].mean(axis=1)
    heatmap_matrix = heatmap_matrix.sort_values(['_fam_rank', '_mean_jsr'],
                                                 ascending=[True, False])
    heatmap_matrix = heatmap_matrix.drop(columns=['_family', '_fam_rank', '_mean_jsr'])

    # Transpose the matrix for horizontal layout
    heatmap_matrix = heatmap_matrix.T

    # Wide figure — significantly increased the multipliers to give cells more room
    n_models = len(heatmap_matrix.columns)
    n_langs = len(heatmap_matrix.index)
    fig_w = max(FULL_WIDTH, n_models * 0.35) # Increased from 0.18 for wider cells
    fig_h = max(4, n_langs * 0.45)           # Increased from 0.25 for taller cells
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        heatmap_matrix, ax=ax,
        annot=True, fmt=".2f", cmap="Reds",
        vmin=0.0, vmax=0.7, linewidths=0.5, linecolor='white', # Thicker cell borders
        cbar_kws={'label': 'JSR', 'shrink': 0.5},
        annot_kws={'fontsize': 8}, # Increased inner text size from 5
    )
    
    ax.set_title('Jailbreak Success Rate by Model and Language', pad=20, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Language')
    
    # Text is now black naturally since we removed the coloring loop, and fonts are slightly larger
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

    df_clean = df.dropna(subset=[COL_TAGS]).copy()
    df_clean[COL_TAGS] = df_clean[COL_TAGS].astype(str)

    def clean_tags(val):
        val = re.sub(r"[\[\]'\" ]+", " ", val)
        val = val.replace(" ,", ",")
        return val.strip()

    df_clean['clean_tags'] = df_clean[COL_TAGS].apply(clean_tags)
    df_exploded = df_clean.assign(
        single_tag=df_clean['clean_tags'].str.split(',')
    ).explode('single_tag')
    df_exploded['single_tag'] = df_exploded['single_tag'].str.strip()
    df_exploded = df_exploded[df_exploded['single_tag'].str.len() > 1]

    model_cat_performance = df_exploded.groupby(
        [COL_MODEL, 'single_tag'])['is_unsafe'].mean().reset_index()
    pivot_df = model_cat_performance.pivot(
        index=COL_MODEL, columns='single_tag', values='is_unsafe')
    pivot_df = pivot_df.fillna(0)
    corr_matrix = pivot_df.corr(method='pearson')

    # Square figure for correlation matrix
    n_cats = len(corr_matrix)
    side = min(FULL_WIDTH, max(3.5, n_cats * 0.25))
    fig, ax = plt.subplots(figsize=(side, side))

    sns.heatmap(
        corr_matrix, ax=ax, cmap=CMAP_DIV, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3, annot=False,
        cbar_kws={"shrink": 0.7, "label": "Pearson $r$"},
    )
    ax.set_title('Safety category correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')

    outfile = os.path.join(RESULTS_DIR, "category_correlation_fixed.png")
    savefig(fig, outfile)


if __name__ == "__main__":
    run_efa()
    plot_jsr_heatmap()
    plot_correlation_matrix()