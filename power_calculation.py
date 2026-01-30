pip install numpy scipy scikit-learn --upgrade


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ==============================================================================
# 1. CONFIGURATION (Based on your actual dataset scale)
# ==============================================================================
NUM_MODELS = 65        # As per your latest update
NUM_PROMPTS = 3150     # Full dataset size
TRUE_DIMENSIONS = 2    # We SIMULATE a world where Safety has 2 distinct factors (e.g., Violent vs. Social)
TRIALS_TO_TEST = [1, 10, 100, 500] # The "Pass@N" levels your advisor asked for

# ==============================================================================
# 2. CREATE THE "GROUND TRUTH" (A Multidimensional World)
# ==============================================================================
print(f"🌍 Generating Synthetic World with {NUM_MODELS} Models and {NUM_PROMPTS} Prompts...")
np.random.seed(42)

# --- A. Define Latent Abilities for Models ---
# Factor 1: "Physical Safety" (e.g., Bombs, Weapons)
# Factor 2: "Social Safety" (e.g., Hate Speech, Bias)
# We make these UNCORRELATED to prove the point (r=0)
model_ability_f1 = np.random.uniform(0.1, 0.9, NUM_MODELS)
model_ability_f2 = np.random.uniform(0.1, 0.9, NUM_MODELS) 

# --- B. Define Prompts ---
# Half the prompts are "Physical", Half are "Social"
true_probs = np.zeros((NUM_MODELS, NUM_PROMPTS))

for p in range(NUM_PROMPTS):
    if p < NUM_PROMPTS // 2: 
        # Type A: Depends ONLY on Factor 1 (Physical)
        # We add some noise to make it realistic
        true_probs[:, p] = model_ability_f1 + np.random.normal(0, 0.05, NUM_MODELS)
    else:
        # Type B: Depends ONLY on Factor 2 (Social)
        true_probs[:, p] = model_ability_f2 + np.random.normal(0, 0.05, NUM_MODELS)

# Clip probabilities to valid 0-1 range
true_probs = np.clip(true_probs, 0.01, 0.99)

# Calculate the "True" Correlation Matrix between Models based on these probabilities
# This is what we WANT to recover.
true_corr = np.corrcoef(true_probs)

# ==============================================================================
# 3. RUN THE "POWER CALCULATION" (Monte Carlo Simulation)
# ==============================================================================
results = []
eigenvalues_history = []

print("⚡ Running Power Calculation...")

for k in TRIALS_TO_TEST:
    # --- Step 1: Simulate Data Collection (Pass@K) ---
    # We flip a biased coin 'k' times for every single cell
    # If k=1 (Pass@1), output is 0 or 1.
    # If k=500 (Pass@500), output is a float (e.g., 0.842)
    success_counts = np.random.binomial(n=k, p=true_probs)
    observed_scores = success_counts / k 

    # --- Step 2: Measure Correlation Recovery ---
    # We calculate the correlation matrix of the OBSERVED data
    obs_corr = np.corrcoef(observed_scores)
    
    # Calculate "Error" (Mean Absolute Difference from Truth)
    error = np.mean(np.abs(obs_corr - true_corr))
    
    # --- Step 3: Check Dimensionality (PCA Eigenvalues) ---
    # If the data is clean, PCA should find exactly 2 big eigenvalues.
    # If noisy (Pass@1), the eigenvalues might smear out.
    pca = PCA(n_components=10)
    pca.fit(observed_scores)
    eigenvalues = pca.explained_variance_ratio_
    
    # "Dominance Ratio" (Advisor mentioned this)
    # Ratio of 1st to 2nd eigenvalue. 
    # In our synthetic world, this should be close to 1.0 (since F1 and F2 are equal size).
    # If noise dominates, this metric goes haywire.
    
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
plt.figure(figsize=(20, 6))

# Plot 1: The "True" Correlation (Target)
plt.subplot(1, len(TRIALS_TO_TEST) + 1, 1)
sns.heatmap(true_corr, vmin=0, vmax=1, cbar=False, cmap="viridis")
plt.title("TRUE Ground Truth\n(2 Distinct Factors)")

# Plot 2-5: The Observed Correlations
for i, res in enumerate(results):
    plt.subplot(1, len(TRIALS_TO_TEST) + 1, i + 2)
    sns.heatmap(res["Matrix"], vmin=0, vmax=1, cbar=False, cmap="viridis")
    plt.title(f"Pass@{res['Pass@N']}\nErr: {res['Error']:.3f}")

plt.tight_layout()
plt.show()

# Plot 6: Scree Plot (Eigenvalues) - The "Unidimensionality" Check
plt.figure(figsize=(10, 5))
markers = ['o', 's', '^', 'D']
for i, ev in enumerate(eigenvalues_history):
    k = TRIALS_TO_TEST[i]
    plt.plot(range(1, 11), ev, marker=markers[i], label=f'Pass@{k}', linewidth=2)

plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Does Noise Hide the Second Dimension?')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==============================================================================
# 5. SAVE RESULTS TO FILE
# ==============================================================================
print("\n💾 Saving results to disk...")

# 1. Save the Main Metrics (For the Line Plot)
# We extract just the scalar values to keep the CSV clean
summary_data = [{"Pass@N": r["Pass@N"], "Reconstruction_Error": r["Error"]} for r in results]
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("power_simulation_summary.csv", index=False)
print("   ✅ Metrics saved to 'power_simulation_summary.csv'")

# 2. Save the Eigenvalues (For the Scree Plot)
# Rows = Pass@N, Columns = Principal Component 1, 2, 3...
df_eigen = pd.DataFrame(eigenvalues_history, 
                        index=[f"Pass@{r['Pass@N']}" for r in results],
                        columns=[f"PC{i+1}" for i in range(len(eigenvalues_history[0]))])
df_eigen.to_csv("power_simulation_eigenvalues.csv")
print("   ✅ Eigenvalues saved to 'power_simulation_eigenvalues.csv'")

# 3. Save the Heatmap Matrices (For the Heatmaps)
# We use NumPy's compressed format (.npz) because saving matrices in CSV is messy
# To load this later: data = np.load('power_matrices.npz'); matrices = data['matrices']
matrix_stack = np.array([r["Matrix"] for r in results])
np.savez_compressed("power_matrices.npz", 
                    matrices=matrix_stack, 
                    pass_n=TRIALS_TO_TEST, 
                    true_matrix=true_corr)
print("   ✅ Correlation matrices saved to 'power_matrices.npz'")


