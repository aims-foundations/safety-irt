#EFA analysis code.

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "FINAL_MERGED_FOR_IRT.csv"
OUTPUT_SCREE_PLOT = "efa_scree_plot.png"
OUTPUT_LOADINGS = "efa_factor_loadings.csv"

def main():
    print("🚀 STARTING EFA ANALYSIS: Testing Unidimensionality of Safety...")

    # =========================================================
    # 1. LOAD & PREPROCESS DATA
    # =========================================================
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   Loaded data: {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ Error: '{INPUT_FILE}' not found. Run curation first.")
        return

    # Parse 'tags' column (string representation of list -> actual list)
    # Example: "['violence', 'hate']" -> ['violence', 'hate']
    print("   Parsing tags...")
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    else:
        print("❌ Error: 'tags' column missing. Cannot perform category-based EFA.")
        return

    # Explode tags so each tag gets its own row
    # This ensures a prompt about "Violence" AND "Hate" counts for both categories.
    df_exploded = df.explode('tags')

    # Convert Labels to Numeric (Safe=1, Unsafe=0)
    # We filter for valid binary labels only
    df_exploded = df_exploded[df_exploded['label'].isin(['safe', 'unsafe'])].copy()
    df_exploded['score'] = df_exploded['label'].map({'safe': 1, 'unsafe': 0})

    # =========================================================
    # 2. CREATE THE STUDENT MATRIX
    # =========================================================
    # A "Student" is defined as a specific Model Config operating in a specific Language.
    # We want to see if "Llama-3-English" behaves consistently across categories compared to "Llama-3-Bengali".
    df_exploded['student_id'] = df_exploded['config'] + "::" + df_exploded['language']

    # Calculate Mean Score per Category for each Student
    category_scores = df_exploded.groupby(['student_id', 'tags'])['score'].mean().reset_index()
    
    # Pivot: Rows = Students, Cols = Categories, Values = Mean Safety Score
    efa_matrix = category_scores.pivot(index='student_id', columns='tags', values='score')

    # Handle Missing Data
    # If a model missed a specific category entirely (e.g., no valid responses for "Harassment"), fill with mean.
    missing_count = efa_matrix.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Warning: Filling {missing_count} missing category scores with column means.")
        efa_matrix = efa_matrix.fillna(efa_matrix.mean())

    print("-" * 40)
    print(f"✅ Matrix Ready: {efa_matrix.shape[0]} Students x {efa_matrix.shape[1]} Categories")
    print("-" * 40)

    # =========================================================
    # 3. KMO TEST (Suitability)
    # =========================================================
    # Kaiser-Meyer-Olkin test checks if data is suitable for Factor Analysis.
    # > 0.6 is good, > 0.8 is great.
    try:
        kmo_all, kmo_model = calculate_kmo(efa_matrix)
        print(f"📊 KMO Score: {kmo_model:.3f}")
        if kmo_model < 0.6:
            print("⚠️  Warning: KMO score is low. Data may not be suitable for EFA.")
    except Exception as e:
        print(f"⚠️  KMO Calculation Failed (Singular Matrix?): {e}")

    # =========================================================
    # 4. RUN EFA (Eigenvalues)
    # =========================================================
    # We run without rotation first to check the raw Eigenvalues.
    # This tells us how much variance the first factor explains vs the second.
    fa = FactorAnalyzer(n_factors=efa_matrix.shape[1], rotation=None)
    fa.fit(efa_matrix)

    ev, v = fa.get_eigenvalues()

    print("\n📉 EIGENVALUES (Variance Explained):")
    for i, val in enumerate(ev[:5]):  # Print top 5
        print(f"   Factor {i+1}: {val:.4f}")

    # Dominance Ratio Test
    ratio = ev[0] / ev[1] if len(ev) > 1 else 0
    print("-" * 40)
    print(f"🏆 DOMINANCE RATIO (Factor 1 / Factor 2): {ratio:.2f}")
    
    if ratio > 3.0:
        print("✅ CONCLUSION: Safety appears UNIDIMENSIONAL (Strong primary factor).")
    else:
        print("⚠️ CONCLUSION: Safety might be MULTI-DIMENSIONAL (No single dominant factor).")
    print("-" * 40)

    # =========================================================
    # 5. VISUALIZATION (Scree Plot)
    # =========================================================
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, efa_matrix.shape[1] + 1), ev, marker='o', linestyle='--', color='b')
    plt.title('Scree Plot: Is Safety One Thing?', fontsize=14)
    plt.xlabel('Factors', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (1.0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_SCREE_PLOT)
    print(f"📈 Scree Plot saved to '{OUTPUT_SCREE_PLOT}'")

    # =========================================================
    # 6. EXTRACT LOADINGS
    # =========================================================
    # We fit again with Varimax rotation to see how categories group together (if multi-dimensional).
    n_factors_extract = 3
    fa_rotated = FactorAnalyzer(n_factors=n_factors_extract, rotation='varimax')
    fa_rotated.fit(efa_matrix)

    loadings = pd.DataFrame(
        fa_rotated.loadings_, 
        index=efa_matrix.columns, 
        columns=[f'Factor{i+1}' for i in range(n_factors_extract)]
    )
    
    loadings.to_csv(OUTPUT_LOADINGS)
    print(f"📄 Factor Loadings saved to '{OUTPUT_LOADINGS}'")
    
    print("\n🔍 Top Categories for Factor 1 (General Safety):")
    print(loadings['Factor1'].sort_values(ascending=False).head(5))

if __name__ == "__main__":
    main()
