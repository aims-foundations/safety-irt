# -*- coding: utf-8 -*-
"""
Embedding Analysis: Translation Similarity (with English) vs DIF/Safety.
Computes cosine similarity between English and translated prompts
and correlates it with the Translation Safety Cost (tau) from the IRT model.
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
IRT_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF.png")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF.csv")

# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def compute_embeddings():
    """
    Step 1: The Heavy Lifting
    Loads data, computes embeddings, and calculates similarity scores.
    Returns a DataFrame ready for plotting.
    """
    print("--- Starting Embedding Analysis ---")

    # 1. Load IRT Results
    if not os.path.exists(IRT_RESULTS_FILE):
        raise FileNotFoundError(f"Results not found at {IRT_RESULTS_FILE}. Run IRT training first.")

    irt_df = pd.read_csv(IRT_RESULTS_FILE)
    irt_df.rename(columns={'prompt': 'id'}, inplace=True)
    irt_df['id'] = irt_df['id'].apply(clean_id)
    print(f"Loaded {len(irt_df)} IRT parameter rows.")

    # 2. Load Raw Text Data
    print(f"Loading raw text from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    raw_df['id'] = raw_df['id'].apply(clean_id)
    text_lookup = raw_df[['id', 'language', 'prompt']].drop_duplicates(subset=['id', 'language'])

    # 3. Build English Reference Dictionary
    eng_df = text_lookup[text_lookup['language'] == 'en'].set_index('id')
    eng_lookup = eng_df['prompt'].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    # 4. Initialize Sentence Transformer
    model_name = 'sentence-transformers/LaBSE'
    print(f"Loading Model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    model = SentenceTransformer(model_name, device=device)

    # 5. Build text pairs
    print("Building text pairs for batch encoding...")
    target_rows = irt_df[irt_df['language'] != 'en'].copy()
    text_lookup['key'] = text_lookup['id'] + "_" + text_lookup['language']
    text_map = text_lookup.set_index('key')['prompt'].to_dict()

    en_texts = []
    target_texts = []
    meta_rows = [] 

    for idx, row in target_rows.iterrows():
        p_id = row['id']
        lang = row['language']
        tau = row['Safety_Tax']
        en_text = eng_lookup.get(p_id)
        target_text = text_map.get(f"{p_id}_{lang}")

        if en_text and target_text:
            en_texts.append(en_text)
            target_texts.append(target_text)
            meta_rows.append({
                'language': lang,
                'id': p_id,
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

    # 7. Compute Cosine Similarity
    similarities = torch.nn.functional.cosine_similarity(en_embeddings, target_embeddings, dim=1).cpu().numpy()

    # 8. Save & Return
    analysis_df = pd.DataFrame(meta_rows)
    analysis_df['similarity'] = similarities
    
    analysis_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Data processed and saved to {OUTPUT_CSV}")
    
    return analysis_df

def plot_results():
    """
    Step 2: The Visualization
    Loads the processed CSV (if needed) and generates the plot.
    """
    if not os.path.exists(OUTPUT_CSV):
        print(f"❌ Output file {OUTPUT_CSV} not found. Run compute_embeddings() first.")
        return

    print(f"Loading analysis data from {OUTPUT_CSV}...")
    df = pd.read_csv(OUTPUT_CSV)
    
    # Statistical Analysis
    corr, p_val = spearmanr(df['similarity'], df['tau'])
    print(f"\n--- STATS ---")
    print(f"Global Spearman Correlation: {corr:.4f} (p={p_val:.4e})")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='similarity',
        y='tau',
        hue='language',
        alpha=0.6,
        palette='tab10'
    )

    try:
        sns.regplot(
            data=df,
            x='similarity',
            y='tau',
            scatter=False,
            color='black',
            line_kws={'linestyle':'--'}
        )
    except Exception:
        pass

    plt.title(f'Semantic Integrity vs. Translation Safety Cost\nSpearman Rho: {corr:.3f}')
    plt.xlabel('Multilingual Embedding Similarity (English vs Target)')
    plt.ylabel(r'Translation Safety Cost ($\tau_{iL}$)')
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    compute_embeddings()
    plot_results()