# -*- coding: utf-8 -*-
"""
Power Simulation: Pass@K convergence for unidimensional IRT.
Generates correlation heatmaps and scree plot.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ── fig_style integration ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig,
                           make_fig_grid, C_RED, C_BLUE, C_PURPLE,
                           CMAP_SEQ, CMAP_DIV)
    _HAS_FS = True
except ImportError:
    _HAS_FS = False

_save = fs_savefig if _HAS_FS else \
    lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
NUM_MODELS = 65
NUM_PROMPTS = 3150
TRUE_DIMENSIONS = 1
TRIALS_TO_TEST = [1, 10, 30, 50, 70, 100, 500]

# ══════════════════════════════════════════════════════════════════════════
# 2. GROUND TRUTH (Unidimensional)
# ══════════════════════════════════════════════════════════════════════════
print(f"Generating synthetic world: {NUM_MODELS} models × {NUM_PROMPTS} prompts, "
      f"d={TRUE_DIMENSIONS}")
np.random.seed(42)

model_ability_f1 = np.random.uniform(0.1, 0.9, NUM_MODELS)
if TRUE_DIMENSIONS == 2:
    model_ability_f2 = np.random.uniform(0.1, 0.9, NUM_MODELS)

true_probs = np.zeros((NUM_MODELS, NUM_PROMPTS))
for p in range(NUM_PROMPTS):
    if TRUE_DIMENSIONS == 1:
        true_probs[:, p] = model_ability_f1 + np.random.normal(0, 0.05, NUM_MODELS)
    else:
        if p < NUM_PROMPTS // 2:
            true_probs[:, p] = model_ability_f1 + np.random.normal(0, 0.05, NUM_MODELS)
        else:
            true_probs[:, p] = model_ability_f2 + np.random.normal(0, 0.05, NUM_MODELS)

true_probs = np.clip(true_probs, 0.01, 0.99)
true_corr = np.corrcoef(true_probs)

# ══════════════════════════════════════════════════════════════════════════
# 3. POWER CALCULATION
# ══════════════════════════════════════════════════════════════════════════
results = []
eigenvalues_history = []

print("Running power calculation ...")
for k in TRIALS_TO_TEST:
    success_counts = np.random.binomial(n=k, p=true_probs)
    observed_scores = success_counts / k

    obs_corr = np.corrcoef(observed_scores)
    error = np.mean(np.abs(obs_corr - true_corr))

    pca = PCA(n_components=10)
    pca.fit(observed_scores)
    eigenvalues = pca.explained_variance_ratio_

    results.append({"Pass@N": k, "Error": error, "Matrix": obs_corr})
    eigenvalues_history.append(eigenvalues)
    print(f"  Pass@{k}: ε = {error:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
if _HAS_FS:
    apply_style()

_c1 = C_BLUE   if _HAS_FS else '#2471a3'
_c2 = C_PURPLE if _HAS_FS else '#7d3c98'
_c3 = C_RED    if _HAS_FS else '#c0392b'
_cmap = CMAP_SEQ if _HAS_FS else 'viridis'

# ── Figure 1: Correlation heatmap strip ──
n_total = len(TRIALS_TO_TEST) + 1          # ground truth + each Pass@K
ncols = n_total
if _HAS_FS:
    fig, axes = make_fig_grid(1, ncols, height_override=1.0)
else:
    fig, axes = plt.subplots(1, ncols, figsize=(5.5, 1.0))
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

sns.heatmap(true_corr, vmin=0, vmax=1, cbar=False, cmap=_cmap,
            ax=axes_flat[0], xticklabels=False, yticklabels=False)
axes_flat[0].set_title('True')
axes_flat[0].set_aspect('equal')

for i, res in enumerate(results):
    ax = axes_flat[i + 1]
    sns.heatmap(res["Matrix"], vmin=0, vmax=1, cbar=False, cmap=_cmap,
                ax=ax, xticklabels=False, yticklabels=False)
    ax.set_title(f'Pass@{res["Pass@N"]}')
    ax.set_aspect('equal')
    # ε inside bottom-right of panel
    n = res["Matrix"].shape[0]
    ax.text(n * 0.95, n * 0.95,
            f'$\\varepsilon$={res["Error"]:.3f}',
            ha='right', va='bottom', fontsize=3.5,
            color='white', bbox=dict(fc='black', alpha=0.5, pad=0.5, lw=0))

path1 = os.path.join(RESULTS_DIR, "power_correlation_heatmaps.png")
_save(fig, path1)
print(f"  Saved: {os.path.basename(path1)}")

# ── Figure 2: Scree plot ──
if _HAS_FS:
    fig, ax = make_fig(n_panels=1, height_override=2.2)
    if isinstance(ax, np.ndarray):
        ax = ax[0]
else:
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

# 3-color cycling: blue → purple → red, repeating
_colors = [_c1, _c2, _c3]
_styles = ['-', '--', '-.', ':', '-', '--', '-.']
_markers = ['o', 's', '^', 'D', 'v', '<', '>']

for i, ev in enumerate(eigenvalues_history):
    k = TRIALS_TO_TEST[i]
    ax.plot(range(1, 11), ev,
            marker=_markers[i % len(_markers)],
            color=_colors[i % len(_colors)],
            ls=_styles[i % len(_styles)],
            lw=0.8, markersize=2.5,
            label=f'Pass@{k}')

ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance ratio')
ax.set_title(f'Scree plot: convergence to $d={TRUE_DIMENSIONS}$')
ax.legend(fontsize=4, ncol=4, loc='upper right',
          handlelength=1.5, columnspacing=0.8)

path2 = os.path.join(RESULTS_DIR, "power_scree_plot.png")
_save(fig, path2)
print(f"  Saved: {os.path.basename(path2)}")

# ══════════════════════════════════════════════════════════════════════════
# 5. SAVE TABLES
# ══════════════════════════════════════════════════════════════════════════
summary_data = [{"Pass@N": r["Pass@N"], "Reconstruction_Error": r["Error"]}
                for r in results]
pd.DataFrame(summary_data).to_csv(
    os.path.join(RESULTS_DIR, "power_simulation_summary.csv"), index=False)

df_eigen = pd.DataFrame(
    eigenvalues_history,
    index=[f"Pass@{r['Pass@N']}" for r in results],
    columns=[f"PC{i+1}" for i in range(len(eigenvalues_history[0]))])
df_eigen.to_csv(os.path.join(RESULTS_DIR, "power_simulation_eigenvalues.csv"))

matrix_stack = np.array([r["Matrix"] for r in results])
np.savez_compressed(
    os.path.join(RESULTS_DIR, "power_matrices.npz"),
    matrices=matrix_stack, pass_n=TRIALS_TO_TEST, true_matrix=true_corr)

print("Done.")
