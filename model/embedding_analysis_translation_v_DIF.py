# -*- coding: utf-8 -*-
"""
Embedding Analysis: Translation Similarity vs Safety Cost (Tau).
Outputs:
1. CSV of raw embeddings and similarities.
2. CSVs of correlation tables (by Language and Category).
3. Plots.
"""
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
try:
    print("Locating dataset snapshot...")
    DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    DATA_DIR = "."

INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Final_Passes0-9_Merged_Graded_Tagged.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IRT_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF.png")
OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF_Cat.csv")

# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def compute_embeddings():
    """
    Step 1: Compute Embeddings & Similarity.
    Includes 'tags' (Category) in the output.
    """
    print("--- Starting Embedding Analysis ---")

    # 1. Load IRT Results
    if not os.path.exists(IRT_RESULTS_FILE):
        raise FileNotFoundError(f"Results not found at {IRT_RESULTS_FILE}.")

    irt_df = pd.read_csv(IRT_RESULTS_FILE)
    
    # Robust ID Column Detection
    if 'prompt' in irt_df.columns: irt_df.rename(columns={'prompt': 'id'}, inplace=True)
    elif 'prompt_id' in irt_df.columns: irt_df.rename(columns={'prompt_id': 'id'}, inplace=True)
    elif 'item' in irt_df.columns: irt_df.rename(columns={'item': 'id'}, inplace=True)
    
    if 'id' not in irt_df.columns:
        print(f"Warning: 'id' column not found. Using first column '{irt_df.columns[0]}' as ID.")
        irt_df.rename(columns={irt_df.columns[0]: 'id'}, inplace=True)

    irt_df['id'] = irt_df['id'].apply(clean_id)
    print(f"Loaded {len(irt_df)} IRT parameter rows.")

    # 2. Load Raw Text Data (with tags)
    print(f"Loading raw text from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    raw_df['id'] = raw_df['id'].apply(clean_id)
    
    # Create Lookups
    # text_lookup: ID+Lang -> Text
    # cat_lookup: ID -> Tag (Category)
    text_lookup = raw_df[['id', 'language', 'prompt']].drop_duplicates(subset=['id', 'language'])
    
    # Get Categories from English rows (safest bet)
    cat_df = raw_df[raw_df['language'] == 'en'][['id', 'tags']].drop_duplicates(subset=['id'])
    cat_lookup = cat_df.set_index('id')['tags'].to_dict()

    # 3. Build English Reference Dictionary
    eng_df = text_lookup[text_lookup['language'] == 'en'].set_index('id')
    eng_lookup = eng_df['prompt'].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    # 4. Initialize Model
    model_name = 'sentence-transformers/LaBSE'
    print(f"Loading Model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # 5. Build text pairs
    print("Building text pairs...")
    target_rows = irt_df[irt_df['language'] != 'en'].copy()
    text_lookup['key'] = text_lookup['id'] + "_" + text_lookup['language']
    text_map = text_lookup.set_index('key')['prompt'].to_dict()

    en_texts = []
    target_texts = []
    meta_rows = [] 

    for idx, row in target_rows.iterrows():
        p_id = row['id']
        lang = row['language']
        # Handle column naming for 'Safety_Tax' or 'tau'
        tau = row.get('Safety_Tax', row.get('tau', row.get('diff', 0)))
        
        en_text = eng_lookup.get(p_id)
        target_text = text_map.get(f"{p_id}_{lang}")

        if en_text and target_text:
            en_texts.append(en_text)
            target_texts.append(target_text)
            meta_rows.append({
                'language': lang,
                'id': p_id,
                'category': cat_lookup.get(p_id, 'Unknown'),
                'tau': tau,
                'en_text': en_text,
                'target_text': target_text
            })

    if not en_texts:
        print("No matching text pairs found.")
        return None

    # 6. Batch Encode
    BATCH_SIZE = 256
    print(f"Encoding {len(en_texts)} pairs...")
    en_embeddings = model.encode(en_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)
    target_embeddings = model.encode(target_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)

    # 7. Compute Similarity
    similarities = torch.nn.functional.cosine_similarity(en_embeddings, target_embeddings, dim=1).cpu().numpy()

    # 8. Save
    analysis_df = pd.DataFrame(meta_rows)
    analysis_df['similarity'] = similarities
    analysis_df.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"✅ Data processed and saved to {OUTPUT_CSV_DATA}")
    
    return analysis_df

def plot_and_report():
    """
    Step 2: Generate Tables and Plots
    """
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"❌ Output file {OUTPUT_CSV_DATA} not found. Run compute_embeddings() first.")
        return

    print(f"Loading analysis data from {OUTPUT_CSV_DATA}...")
    df = pd.read_csv(OUTPUT_CSV_DATA)
    
    # --- TABLE 1: PER-LANGUAGE CORRELATIONS ---
    print("\n[Per-Language Correlation Analysis]")
    lang_stats = []
    for lang in df['language'].unique():
        sub = df[df['language'] == lang]
        if len(sub) > 10:
            r, p = spearmanr(sub['similarity'], sub['tau'])
            lang_stats.append({
                'Language': lang, 
                'Spearman_Rho': r, 
                'P_Value': p, 
                'Count': len(sub)
            })
    
    lang_df = pd.DataFrame(lang_stats).sort_values('Spearman_Rho')
    
    # PRINT
    print(lang_df.to_string(index=False))
    
    # SAVE
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)
    print(f"✅ Saved language table to {OUTPUT_CSV_LANG}")

    # --- TABLE 2: PER-CATEGORY CORRELATIONS ---
    print("\n[Per-Category Correlation Analysis]")
    cat_stats = []
    for cat in df['category'].unique():
        sub = df[df['category'] == cat]
        if len(sub) > 10:
            r, p = spearmanr(sub['similarity'], sub['tau'])
            cat_stats.append({
                'Category': cat, 
                'Spearman_Rho': r, 
                'P_Value': p, 
                'Count': len(sub)
            })
    
    cat_df = pd.DataFrame(cat_stats).sort_values('Spearman_Rho')
    
    # PRINT
    print(cat_df.to_string(index=False))
    
    # SAVE
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)
    print(f"✅ Saved category table to {OUTPUT_CSV_CAT}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='similarity', y='tau', hue='language', alpha=0.6, palette='tab10')
    
    # Add trend line for visual aid
    try:
        sns.regplot(data=df, x='similarity', y='tau', scatter=False, color='black', line_kws={'linestyle':'--'})
    except: pass

    plt.title(f'Semantic Integrity vs. Safety Cost\n(See {os.path.basename(OUTPUT_CSV_LANG)} for details)')
    plt.xlabel('Embedding Similarity (LaBSE)')
    plt.ylabel('Translation Safety Cost (Tau)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    compute_embeddings()
    plot_and_report()