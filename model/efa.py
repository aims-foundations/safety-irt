# -*- coding: utf-8 -*-
"""
EFA analysis for safety unidimensionality,
plus JSR heatmap and safety category correlation visualizations.
"""

import pandas as pd
import numpy as np
import ast
import re
import os
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
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
    df = pd.read_csv(INPUT_FILE)

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

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ev)+1), ev, marker='o', linestyle='--', color='b')
    plt.title('Scree Plot: Latent Structure of LLM Safety (Likert 1-5)', fontsize=14)
    plt.xlabel('Factors', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.grid(True)
    plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (1.0)')
    plt.legend()
    plt.savefig(OUTPUT_SCREE_PLOT)
    plt.close()
    print(f"Scree plot saved to {OUTPUT_SCREE_PLOT}")

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

    df = pd.read_csv(INPUT_FILE)

    required = [COL_MODEL, COL_LANG, COL_SCORE]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing columns: {required}")

    df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors='coerce')
    df['is_unsafe'] = (df[COL_SCORE] < 4).astype(int)

    jsr_data = df.groupby([COL_MODEL, COL_LANG])['is_unsafe'].mean().reset_index()
    heatmap_matrix = jsr_data.pivot(index=COL_MODEL, columns=COL_LANG, values='is_unsafe')

    plt.figure(figsize=(10, 14))
    sns.heatmap(
        heatmap_matrix,
        annot=True, fmt=".2f", cmap="Reds",
        vmin=0.0, vmax=0.7, linewidths=0.5,
        cbar_kws={'label': 'Jailbreak Success Rate (0=Safe, 1=Unsafe)'}
    )
    plt.title("Jailbreak Success Rate (JSR) by Model & Language", fontsize=16)
    plt.xlabel("Language", fontsize=12)
    plt.ylabel("Model Configuration", fontsize=12)
    plt.tight_layout()

    outfile = os.path.join(RESULTS_DIR, "jsr_heatmap.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"JSR heatmap saved to {outfile}")


def plot_correlation_matrix():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = pd.read_csv(INPUT_FILE)

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

    model_cat_performance = df_exploded.groupby([COL_MODEL, 'single_tag'])['is_unsafe'].mean().reset_index()
    pivot_df = model_cat_performance.pivot(index=COL_MODEL, columns='single_tag', values='is_unsafe')
    pivot_df = pivot_df.fillna(0)
    corr_matrix = pivot_df.corr(method='pearson')

    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(260, 15, as_cmap=True, s=90, l=50, sep=1)
    sns.heatmap(
        corr_matrix, cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, annot=False,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Safety Category Correlation", fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

    outfile = os.path.join(RESULTS_DIR, "category_correlation_fixed.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Correlation matrix saved to {outfile}")

if __name__ == "__main__":
    run_efa()
    plot_jsr_heatmap()
    plot_correlation_matrix()
