#this ipynb file is for EFA analysis to determine safety's unidimensionality
#you need FINAL_MERGED_FOR_IRT.csv to run it


#EFA, KMO code block  (Likert 1-5)
import pandas as pd
import numpy as np
import ast
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_FILE = "FINALMERGEDTAGGED.csv" # Ensure this file has 'judge_score' and 'tags'
OUTPUT_SCREE_PLOT = "efa_scree_plot_likert.png"
OUTPUT_LOADINGS = "efa_factor_loadings_likert.csv"

# =========================================================
# 1: Data preparation
# =========================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# 1. Parse Tags (violence, hate, fraud, etc.)
print("Parsing tags...")
# Ensure tags are actually strings of lists before eval
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# 2. Explode Tags
# Each prompt counts toward the statistics of every tag it belongs to.
df_exploded = df.explode('tags')

# --- CHANGED SECTION: LIKERT SCALE ---
# 3. Filter & Define Score
# We use the 1-5 Scale:
# 1 = Jailbreak (Least Safe)
# 5 = Refusal (Most Safe)
# We drop 0 (Invalid) or any NaNs
print("Filtering for valid Likert scores (1-5)...")

if 'judge_score' in df_exploded.columns:
    df_exploded = df_exploded[df_exploded['judge_score'].isin([1, 2, 3, 4, 5])]
    df_exploded['score'] = df_exploded['judge_score'].astype(float)
else:
    print("❌ Error: 'judge_score' column not found. Please check column names.")
    exit()
# -----------------------------------------------

# 4. Create the "Student" ID
# A "Student" is a Model operating in a specific Language.
# Note: Check if your CSV uses 'config' or 'test_taker'. Swapping to 'test_taker' just in case.
col_name = 'test_taker' if 'test_taker' in df_exploded.columns else 'config'
df_exploded['student_id'] = df_exploded[col_name] + "::" + df_exploded['language']

# 5. Create the EFA Matrix
# Rows = Students (Models), Columns = Categories (Tags), Values = Mean Safety Score (1-5)
category_scores = df_exploded.groupby(['student_id', 'tags'])['score'].mean().reset_index()
efa_matrix = category_scores.pivot(index='student_id', columns='tags', values='score')

# 6. Handle Missing Data
# If a model never encountered a specific tag (rare), fill with global mean for that tag
if efa_matrix.isnull().sum().sum() > 0:
    print(f"⚠️ Warning: Filling {efa_matrix.isnull().sum().sum()} missing category scores with column mean.")
    efa_matrix = efa_matrix.fillna(efa_matrix.mean())

print("-" * 40)
print(f"✅ Matrix Ready: {efa_matrix.shape[0]} Students x {efa_matrix.shape[1]} Categories")
print("-" * 40)

# =========================================================
# STEP 2: KMO (Measure of Sampling Adequacy)
# =========================================================

kmo_all, kmo_model = calculate_kmo(efa_matrix)
print(f"KMO Score: {kmo_model:.3f}")
if kmo_model > 0.8:
    print("✅ Data is GREAT for Factor Analysis.")
elif kmo_model > 0.6:
    print("⚠️ Data is ACCEPTABLE for Factor Analysis.")
else:
    print("❌ Data is POOR for Factor Analysis (Variables not correlated enough).")

# =========================================================
# STEP 3: RUN EFA (Eigenvalues)
# =========================================================
fa = FactorAnalyzer(n_factors=min(efa_matrix.shape[1], 25), rotation=None)
fa.fit(efa_matrix)

# Get Eigenvalues
ev, v = fa.get_eigenvalues()

print("\n📊 EIGENVALUES (Top 5):")
for i, val in enumerate(ev[:5]):
    print(f"Factor {i+1}: {val:.4f}")

# Check Unidimensionality Hypothesis
ratio = ev[0] / ev[1]
print("-" * 40)
print(f"🏆 DOMINANCE RATIO (Factor 1 / Factor 2): {ratio:.2f}")
if ratio > 3.0:
    print("✅ CONCLUSION: Safety is strongly UNIDIMENSIONAL.")
    print("   (Models good at 'Violence' are also good at 'Fraud'.)")
else:
    print("⚠️ CONCLUSION: Safety might be MULTI-DIMENSIONAL.")
    print("   (Models might be good at 'Violence' but bad at 'Fraud'.)")
print("-" * 40)

# =========================================================
# STEP 4: VISUALIZATION (Scree Plot)
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev)+1), ev, marker='o', linestyle='--', color='b')
plt.title('Scree Plot: Latent Structure of LLM Safety (Likert 1-5)', fontsize=14)
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.grid(True)
plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (1.0)')
plt.legend()
plt.savefig(OUTPUT_SCREE_PLOT)
print(f"📈 Scree Plot saved to {OUTPUT_SCREE_PLOT}")

# =========================================================
# STEP 5: FACTOR LOADINGS
# =========================================================
# We assume 3 factors to see if any distinct clusters emerge
fa_rotated = FactorAnalyzer(n_factors=3, rotation='varimax')
fa_rotated.fit(efa_matrix)

loadings = pd.DataFrame(fa_rotated.loadings_, index=efa_matrix.columns, columns=['Factor1', 'Factor2', 'Factor3'])
loadings.to_csv(OUTPUT_LOADINGS)
print(f"📄 Factor Loadings saved to {OUTPUT_LOADINGS}")

print("\nTop Categories contributing to Factor 1 (Likely General Safety):")
print(loadings['Factor1'].abs().sort_values(ascending=False).head(5))

#limitations
#Some tags are almost always moving together (which is consistent with a dominant factor)
#Some categories have near-zero variance (e.g., almost always safe or almost always unsafe)
