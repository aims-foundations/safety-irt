import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import itertools

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
NUM_MODELS = 65        
NUM_PROMPTS = 3150     
TRUE_DIMENSIONS = 1    # CHANGED: Now simulating a Unidimensional world
TRIALS_TO_TEST = [1, 10, 30, 50, 70, 100, 500] # CHANGED: Added 30, 50, 70

# ==============================================================================
# 2. CREATE THE "GROUND TRUTH" (Unidimensional World)
# ==============================================================================
print(f"🌍 Generating Synthetic World with {NUM_MODELS} Models and {NUM_PROMPTS} Prompts...")
print(f"   True Dimensions: {TRUE_DIMENSIONS}")
np.random.seed(42)

# --- A. Define Latent Abilities for Models ---
# Since TRUE_DIMENSIONS = 1, every model just has one "Safety Ability"
# (e.g., General refusal capability)
model_ability_f1 = np.random.uniform(0.1, 0.9, NUM_MODELS)

# (Optional placeholder if you ever switch back to 2)
if TRUE_DIMENSIONS == 2:
    model_ability_f2 = np.random.uniform(0.1, 0.9, NUM_MODELS)

# --- B. Define Prompts ---
true_probs = np.zeros((NUM_MODELS, NUM_PROMPTS))

for p in range(NUM_PROMPTS):
    if TRUE_DIMENSIONS == 1:
        # ALL prompts depend on the same single factor
        true_probs[:, p] = model_ability_f1 + np.random.normal(0, 0.05, NUM_MODELS)
    else:
        # Split prompts between two factors
        if p < NUM_PROMPTS // 2: 
            true_probs[:, p] = model_ability_f1 + np.random.normal(0, 0.05, NUM_MODELS)
        else:
            true_probs[:, p] = model_ability_f2 + np.random.normal(0, 0.05, NUM_MODELS)

# Clip probabilities to valid 0-1 range
true_probs = np.clip(true_probs, 0.01, 0.99)

# Calculate the "True" Correlation Matrix
true_corr = np.corrcoef(true_probs)

# ==============================================================================
# 3. RUN THE "POWER CALCULATION"
# ==============================================================================
results = []
eigenvalues_history = []

print("⚡ Running Power Calculation...")

for k in TRIALS_TO_TEST:
    # --- Step 1: Simulate Data Collection (Pass@K) ---
    success_counts = np.random.binomial(n=k, p=true_probs)
    observed_scores = success_counts / k 

    # --- Step 2: Measure Correlation Recovery ---
    obs_corr = np.corrcoef(observed_scores)
    error = np.mean(np.abs(obs_corr - true_corr))
    
    # --- Step 3: Check Dimensionality (PCA Eigenvalues) ---
    pca = PCA(n_components=10)
    pca.fit(observed_scores)
    eigenvalues = pca.explained_variance_ratio_
    
    results.append({
        "Pass@N": k,
        "Error": error,
        "Matrix": obs_corr
    })
    eigenvalues_history.append(eigenvalues)
    
    print(f"   Pass@{k}: Reconstruction Error = {error:.4f}")

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
# Adjust figure width to fit more plots
plt.figure(figsize=(24, 5)) 

# Plot 1: The "True" Correlation (Target)
plt.subplot(1, len(TRIALS_TO_TEST) + 1, 1)
sns.heatmap(true_corr, vmin=0, vmax=1, cbar=False, cmap="viridis")
plt.title(f"TRUE Ground Truth\n({TRUE_DIMENSIONS} Dimension)")

# Plot 2+: The Observed Correlations
for i, res in enumerate(results):
    plt.subplot(1, len(TRIALS_TO_TEST) + 1, i + 2)
    sns.heatmap(res["Matrix"], vmin=0, vmax=1, cbar=False, cmap="viridis")
    plt.title(f"Pass@{res['Pass@N']}\nErr: {res['Error']:.3f}")

plt.tight_layout()
plt.show()

# Plot 6: Scree Plot (Eigenvalues)
plt.figure(figsize=(10, 5))
# Extended marker list to handle 7 lines
markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])

for i, ev in enumerate(eigenvalues_history):
    k = TRIALS_TO_TEST[i]
    plt.plot(range(1, 11), ev, marker=next(markers), label=f'Pass@{k}', linewidth=2)

plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Convergence to 1 Dimension')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==============================================================================
# 5. SAVE RESULTS
# ==============================================================================
print("\n💾 Saving results to disk...")

summary_data = [{"Pass@N": r["Pass@N"], "Reconstruction_Error": r["Error"]} for r in results]
pd.DataFrame(summary_data).to_csv("power_simulation_summary.csv", index=False)
print("   ✅ Metrics saved to 'power_simulation_summary.csv'")

df_eigen = pd.DataFrame(eigenvalues_history, 
                        index=[f"Pass@{r['Pass@N']}" for r in results],
                        columns=[f"PC{i+1}" for i in range(len(eigenvalues_history[0]))])
df_eigen.to_csv("power_simulation_eigenvalues.csv")
print("   ✅ Eigenvalues saved to 'power_simulation_eigenvalues.csv'")

matrix_stack = np.array([r["Matrix"] for r in results])
np.savez_compressed("power_matrices.npz", 
                    matrices=matrix_stack, 
                    pass_n=TRIALS_TO_TEST, 
                    true_matrix=true_corr)
print("   ✅ Correlation matrices saved to 'power_matrices.npz'")