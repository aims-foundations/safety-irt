#this ipynb file is for EFA analysis to determine safety's unidimensionality
#you need FINAL_MERGED_FOR_IRT.csv to run it

!pip install pandas factor_analyzer  numpy seaborn matplotlib

#EFA, KMO code block
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
INPUT_FILE = "FINAL_MERGED_FOR_IRT.csv"
OUTPUT_SCREE_PLOT = "efa_scree_plot.png"
OUTPUT_LOADINGS = "efa_factor_loadings.csv"

# =========================================================
# 1: Data preparation
# =========================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# 1. Parse Tags (violence, hate, fraud, etc.)
print("Parsing tags...")
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# 2. Explode Tags
# Note: Many multijail prompts have multiple tags, which is why if a prompt has 2 tags, it becomes 2 rows. This ensures the prompt counts for BOTH categories.
df_exploded = df.explode('tags')

# 3. (Safe = 1, Unsafe = 0)
df_exploded = df_exploded[df_exploded['label'].isin(['safe', 'unsafe'])]
df_exploded['score'] = df_exploded['label'].map({'safe': 1, 'unsafe': 0})

#----------
# 4. Create the "Student" ID
# A "Student" is a specific Model Configuration operating in a specific Language.
# Example: "Llama3_Standard_ar" is one student. "Llama3_Standard_en" is another.
#Note to Sang + self: This is because we want to see if safety is different with specific "tags" (fraud, violence), etc right.
#Each jailbreaking non-english prompt has the same tags as its English counterpart, so we can just treat it as another test-taker for determining tag correlations.
df_exploded['student_id'] = df_exploded['config'] + "::" + df_exploded['language']


#df_exploded['student_id'] = df_exploded['config'].
#NOTE: Other option, normalizing within language, leads to a KMO score of 0.942 (basically the same)
#----------

category_scores = df_exploded.groupby(['student_id', 'tags'])['score'].mean().reset_index()
efa_matrix = category_scores.pivot(index='student_id', columns='tags', values='score') #set up efa_matrix

# 7. Handle Missing Data (From filtering out "invalid" responses)
# If a model missed a specific category entirely , fill with column mean
if efa_matrix.isnull().sum().sum() > 0:
    print(f"⚠️ Warning: Filling {efa_matrix.isnull().sum().sum()} missing category scores with mean.")
    efa_matrix = efa_matrix.fillna(efa_matrix.mean())
    #note to self: EFA requires full matrix which is why this step is required.

print("-" * 40)
print(f"✅ Matrix Ready: {efa_matrix.shape[0]} Students x {efa_matrix.shape[1]} Categories")
print("-" * 40)

# =========================================================
# STEP 2: KMO
# =========================================================

kmo_all, kmo_model = calculate_kmo(efa_matrix)
print(f"KMO Score: {kmo_model:.3f}")

# =========================================================
# STEP 3: RUN EFA (Eigenvalues)
# =========================================================
# We run with rotation=None first to check the raw eigenvalues. Note to self: Rotation just makes it cleaner, not too useful to know
fa = FactorAnalyzer(n_factors=efa_matrix.shape[1], rotation=None)
fa.fit(efa_matrix)

# Get Eigenvalues
ev, v = fa.get_eigenvalues()

print("\n📊 EIGENVALUES (Variance Explained):")
for i, val in enumerate(ev[:5]): # Print top 5 [:5]
    print(f"Factor {i+1}: {val:.4f}")

# Check Unidimensionality Hypothesis
ratio = ev[0] / ev[1]
print("-" * 40)
print(f"🏆 DOMINANCE RATIO (Factor 1 / Factor 2): {ratio:.2f}")
if ratio > 3.0:
    print("✅ CONCLUSION: Safety is likely UNIDIMENSIONAL (Strong primary factor).")
else:
    print("⚠️ CONCLUSION: Safety might be MULTI-DIMENSIONAL.")
print("-" * 40)

# =========================================================
# STEP 4: VISUALIZATION (The Scree Plot)
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, efa_matrix.shape[1]+1), ev, marker='o', linestyle='--')
plt.title('Scree Plot: Is Safety One Thing?', fontsize=14)
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.grid(True)
plt.axhline(y=1, color='r', linestyle='-') # Kaiser criterion line >0.9
plt.savefig(OUTPUT_SCREE_PLOT)
print(f"📈 Scree Plot saved to {OUTPUT_SCREE_PLOT}")

# =========================================================
# STEP 5: FACTOR LOADINGS (What are the factors?)
# =========================================================
# If multi-dimensional, what groups together? (Violence? Fraud?)
# We fit again with 3 factors just to see the structure
fa_rotated = FactorAnalyzer(n_factors=3, rotation='varimax')
fa_rotated.fit(efa_matrix)

loadings = pd.DataFrame(fa_rotated.loadings_, index=efa_matrix.columns, columns=['Factor1', 'Factor2', 'Factor3'])
loadings.to_csv(OUTPUT_LOADINGS)
print(f"📄 Factor Loadings saved to {OUTPUT_LOADINGS}")
print("\nTop Categories contributing to Factor 1 (General Safety):")
print(loadings['Factor1'].sort_values(ascending=False).head(5))


#limitations
#Some tags are almost always moving together (which is consistent with a dominant factor)
#Some categories have near-zero variance (e.g., almost always safe or almost always unsafe)
