# -*- coding: utf-8 -*-
"""
Response Matrix Visualization — XSafety
=========================================
Adapted from model/response_matrix.py for XSafety:
  - Single pass only (no pass column)
  - Generates one response matrix (no mega-matrix / per-pass matrices)
  - Input: XSafety_Dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from huggingface_hub import snapshot_download

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from fig_style import *

apply_style()

REPO_ID  = "safety-irt/safety-data"
FILENAME = "xsafety/xsafety_pass_graded.csv"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "response_matrices")
RAW_RED  = "#FF0000"
RAW_BLUE = "#0000FF"


def load_data_from_hf():
    print(f"--- Downloading XSafety data from HuggingFace: {REPO_ID} ---")
    data_dir  = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                                  allow_patterns=[FILENAME])
    file_path = os.path.join(data_dir, FILENAME)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"   > Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(-1).astype(int).astype(str)
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df['question_key'] = df['id'] + "_" + df['language']

    # Binary Safety (1=Safe, 0=Unsafe, NaN=Missing)
    df['binary'] = np.where(
        (df['judge_score'].isna()) | (df['judge_score'] == 0), np.nan,
        np.where(df['judge_score'] >= 4, 1.0, 0.0)
    )
    return df


def generate_response_matrix(df):
    """Generates a single response matrix (XSafety has one pass)."""
    print("--- Generating XSafety Response Matrix ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'

    pivot = df.groupby([student_col, 'question_key'])['binary'].mean().unstack()
    pivot = pivot.map(lambda x: 1.0 if x >= 0.5 else (0.0 if x < 0.5 else np.nan))

    if pivot.empty:
        print("Error: pivot table is empty — check data.")
        return

    # Sort: safest models top, easiest prompts left
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.mean(axis=0).sort_values(ascending=False).index]

    plot_data = pivot.fillna(-1).values
    fig_w = max(12, plot_data.shape[1] / 150)
    fig_h = max(8, plot_data.shape[0] / 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap   = mcolors.ListedColormap(['white', RAW_RED, RAW_BLUE])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title("XSafety Response Matrix (Sorted by Difficulty)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Models (Safest Top)", fontsize=12)
    ax.set_xlabel("Prompts (Easiest → Hardest)", fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xticks([])

    out_path = os.path.join(RESULTS_DIR, "XSafety_Response_Matrix.png")
    savefig(fig, out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Per-language matrices
    languages = df['language'].unique()
    for lang in sorted(languages):
        lang_df = df[df['language'] == lang]
        lang_ids = lang_df['question_key'].unique()
        lang_pivot = pivot[[c for c in pivot.columns if c in set(lang_ids)]]
        if lang_pivot.empty:
            continue
        lang_data = lang_pivot.fillna(-1).values
        fig2, ax2 = plt.subplots(figsize=(max(8, lang_data.shape[1] / 30),
                                          max(5, lang_data.shape[0] / 3)))
        ax2.imshow(lang_data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax2.set_title(f"XSafety Response Matrix — {lang}", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Models (Safest Top)", fontsize=11)
        ax2.set_xlabel("Prompts (Easiest → Hardest)", fontsize=11)
        ax2.set_yticks(range(len(lang_pivot.index)))
        ax2.set_yticklabels(lang_pivot.index, fontsize=9)
        ax2.set_xticks([])
        lang_path = os.path.join(RESULTS_DIR, f"Matrix_{lang}.png")
        savefig(fig2, lang_path)
        plt.close()
        print(f"  Saved: {lang_path}")


if __name__ == "__main__":
    df = load_data_from_hf()
    generate_response_matrix(df)
