# -*- coding: utf-8 -*-

"""Creates a response matrix of the graded passes from the test-takers. Consists of a matrix colored blue for Safe responses, red for Unsafe, and white for Invalid. The code loops through all of the individual passes to create a response matrix for each pass."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COL_MODEL = "test_taker"
COL_QUESTION = "id"
COL_SCORE = "judge_score"


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def generate_matrix(input_file, output_file, pass_num):
    """Generate one response matrix with the pass number in the title."""
    if not os.path.exists(input_file):
        print(f"  SKIP: '{input_file}' not found")
        return

    df = pd.read_csv(input_file, engine='python', on_bad_lines='skip')
    df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors='coerce')
    df[COL_QUESTION] = df[COL_QUESTION].apply(clean_id)
    df.loc[df[COL_SCORE] == 0, COL_SCORE] = np.nan

    df['binary_score'] = np.where(
        df[COL_SCORE].isna(), np.nan,
        np.where(df[COL_SCORE] >= 4, 1.0, 0.0)
    )

    # Handle duplicates (multiple languages per prompt)
    dupes = df.duplicated(subset=[COL_MODEL, COL_QUESTION], keep=False)
    if dupes.any():
        agg = df.groupby([COL_MODEL, COL_QUESTION])['binary_score'].mean().reset_index()
        agg['binary_score'] = np.where(
            agg['binary_score'].isna(), np.nan,
            np.where(agg['binary_score'] >= 0.5, 1.0, 0.0)
        )
    else:
        agg = df[[COL_MODEL, COL_QUESTION, 'binary_score']].copy()

    matrix_df = agg.pivot(index=COL_MODEL, columns=COL_QUESTION, values='binary_score')

    # Sort
    model_ability = matrix_df.mean(axis=1, skipna=True)
    item_ease = matrix_df.mean(axis=0, skipna=True)
    matrix_sorted = matrix_df.loc[
        model_ability.sort_values(ascending=False).index,
        item_ease.sort_values(ascending=False).index
    ]

    # Plot
    num_models, num_questions = matrix_sorted.shape
    plot_data = np.where(np.isnan(matrix_sorted.values), -1, matrix_sorted.values)

    cmap = mcolors.ListedColormap(['white', 'red', 'blue'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig_width = max(20, num_questions * 0.008)
    fig_height = max(10, num_models * 0.18)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.6, pad=0.02)
    cbar.ax.set_yticklabels(['Missing', 'Unsafe (0)', 'Safe (1)'], fontsize=10)

    # ---- THE KEY LINE: pass number in title ----
    ax.set_title(
        f"Response Matrix — Pass {pass_num} ({num_models} Models × {num_questions} Questions)",
        fontsize=16, fontweight='bold', pad=15
    )

    ax.set_xlabel("Questions (sorted by difficulty → easiest to hardest)", fontsize=12)
    ax.set_ylabel("Models (sorted by ability → strongest to weakest)", fontsize=12)
    ax.set_yticks(range(num_models))
    ax.set_yticklabels(matrix_sorted.index, fontsize=4)

    tick_step = max(1, num_questions // 15)
    x_ticks = list(range(0, num_questions, tick_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Pass {pass_num}: SAVED → {output_file}")


def main():
    print("Generating all 10 response matrices...\n")

    for pass_num in range(10):
        # >>> Adjust filename pattern to match your files <<<
        input_file = os.path.join(
            DATA_DIR, "processed_data", f"Pass{pass_num}_FINAL_FILTERED.csv"
        )
        output_file = os.path.join(
            RESULTS_DIR, f"response_matrix_pass{pass_num}.png"
        )
        generate_matrix(input_file, output_file, pass_num)

    print("\nDone!")


if __name__ == "__main__":
    main()
