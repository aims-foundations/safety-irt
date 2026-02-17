# -*- coding: utf-8 -*-
"""
Embedding Analysis: Translation Similarity vs Safety Cost (Tau).
1. Computes LaBSE embeddings and cosine similarity.
2. Calculates Spearman's Rho for Categories and Languages.
3. Generates the 'Semantic Hazard vs Filter Benefit' Bar Charts.
"""
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
try:
    print("Locating dataset snapshot...")
    DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    DATA_DIR = "."

INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IRT_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Analysis_Fidelity_vs_SafetyTax.png")
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
    """
    print("--- Starting Embedding Analysis ---")

    if not os.path.exists(IRT_RESULTS_FILE):
        raise FileNotFoundError(f"Results not found at {IRT_RESULTS_FILE}.")

    irt_df = pd.read_csv(IRT_RESULTS_FILE)
    
    # Robust ID Column Detection
    if 'prompt' in irt_df.columns: irt_df.rename(columns={'prompt': 'id'}, inplace=True)
    elif 'prompt_id' in irt_df.columns: irt_df.rename(columns={'prompt_id': 'id'}, inplace=True)
    elif 'item' in irt_df.columns: irt_df.rename(columns={'item': 'id'}, inplace=True)
    
    if 'id' not in irt_df.columns:
        # Fallback
        irt_df.rename(columns={irt_df.columns[0]: 'id'}, inplace=True)

    irt_df['id'] = irt_df['id'].apply(clean_id)
    print(f"Loaded {len(irt_df)} IRT parameter rows.")

    # Load Raw Text
    print(f"Loading raw text from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    raw_df['id'] = raw_df['id'].apply(clean_id)
    
    # Create Lookups
    text_lookup = raw_df[['id', 'language', 'prompt']].drop_duplicates(subset=['id', 'language'])
    
    # Get Categories (Tags)
    cat_df = raw_df[raw_df['language'] == 'en'][['id', 'tags']].drop_duplicates(subset=['id'])
    cat_lookup = cat_df.set_index('id')['tags'].to_dict()

    # English Reference
    eng_df = text_lookup[text_lookup['language'] == 'en'].set_index('id')
    eng_lookup = eng_df['prompt'].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    # Load Model
    model_name = 'sentence-transformers/LaBSE'
    print(f"Loading Model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Build Pairs
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
        # Prioritize 'tau', then 'Safety_Tax', then 'diff'
        tau = row.get('tau', row.get('Safety_Tax', row.get('diff', 0)))
        
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

    # Encode & Compute
    BATCH_SIZE = 256
    print(f"Encoding {len(en_texts)} pairs...")
    en_embeddings = model.encode(en_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)
    target_embeddings = model.encode(target_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)

    similarities = torch.nn.functional.cosine_similarity(en_embeddings, target_embeddings, dim=1).cpu().numpy()

    analysis_df = pd.DataFrame(meta_rows)
    analysis_df['similarity'] = similarities
    analysis_df.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"✅ Data processed and saved to {OUTPUT_CSV_DATA}")
    
    return analysis_df

def plot_and_report():
    """
    Step 2: Generate Tables AND the Paper-Ready Bar Charts.
    """
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"❌ Output file {OUTPUT_CSV_DATA} not found.")
        return

    print(f"Loading analysis data from {OUTPUT_CSV_DATA}...")
    df = pd.read_csv(OUTPUT_CSV_DATA)
    
    # --- CALCULATE SPEARMAN STATS ---
    print("\nCalculating correlations...")
    
    # Language Stats
    lang_stats = []
    for lang in df['language'].unique():
        sub = df[df['language'] == lang]
        if len(sub) > 10:
            r, p = spearmanr(sub['similarity'], sub['tau'])
            lang_stats.append({'Language': lang, 'Spearman_Rho': r, 'P_Value': p, 'Count': len(sub)})
    lang_df = pd.DataFrame(lang_stats).sort_values('Spearman_Rho')
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)

    # Category Stats
    cat_stats = []
    for cat in df['category'].unique():
        sub = df[df['category'] == cat]
        if len(sub) > 10:
            r, p = spearmanr(sub['similarity'], sub['tau'])
            cat_stats.append({'Category': cat, 'Spearman_Rho': r, 'P_Value': p, 'Count': len(sub)})
    cat_df = pd.DataFrame(cat_stats).sort_values('Spearman_Rho')
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)

    # --- PLOTTING LOGIC (The Exact Graph) ---
    print(f"Generating Bar Charts to {OUTPUT_PLOT}...")
    
    # Prepare Data
    df_cat = cat_df.copy()
    df_lang = lang_df.copy()
    
    # Clean Category Names (remove brackets/quotes)
    df_cat['Category'] = df_cat['Category'].astype(str).str.replace(r"^\['|'\]$|'|\[|\]", "", regex=True)
    
    # Significance Flags
    df_cat['Significant'] = df_cat['P_Value'] < 0.05
    df_lang['Significant'] = df_lang['P_Value'] < 0.05

    # Setup Canvas
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={'width_ratios': [2, 1]})
    
    # Plot 1: Categories
    # Logic: High Fidelity -> High Tau (Unsafe) = Positive Rho = Red (Hazard)
    # Logic: High Fidelity -> Low Tau (Safe) = Negative Rho = Blue (Benefit)
    colors_cat = ['#d62728' if x > 0 else '#1f77b4' for x in df_cat['Spearman_Rho']]
    alphas_cat = [1.0 if x else 0.3 for x in df_cat['Significant']]
    
    bars = axes[0].barh(df_cat['Category'], df_cat['Spearman_Rho'], color=colors_cat)
    for bar, alpha in zip(bars, alphas_cat):
        bar.set_alpha(alpha)
        
    axes[0].set_title('Correlation: Translation Fidelity vs. Safety Tax (By Category)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Spearman's Rho\n(Left/Blue: Better Translation = Safer | Right/Red: Better Translation = More Dangerous)", fontsize=12)
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Annotate Significance
    for i, (rho, p) in enumerate(zip(df_cat['Spearman_Rho'], df_cat['P_Value'])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[0].text(rho + offset, i, "*", va='center', fontsize=16, fontweight='bold', color='black')

    # Plot 2: Languages
    colors_lang = ['#d62728' if x > 0 else '#1f77b4' for x in df_lang['Spearman_Rho']]
    alphas_lang = [1.0 if x else 0.3 for x in df_lang['Significant']]
    
    bars_lang = axes[1].barh(df_lang['Language'], df_lang['Spearman_Rho'], color=colors_lang)
    for bar, alpha in zip(bars_lang, alphas_lang):
        bar.set_alpha(alpha)
        
    axes[1].set_title('Correlation: Fidelity vs. Safety Tax (By Language)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Spearman's Rho", fontsize=12)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Annotate Significance
    for i, (rho, p) in enumerate(zip(df_lang['Spearman_Rho'], df_lang['P_Value'])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[1].text(rho + offset, i, "*", va='center', fontsize=16, fontweight='bold', color='black')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Filter Benefit (High Fidelity -> Safer)'),
        Line2D([0], [0], color='#d62728', lw=4, label='Semantic Hazard (High Fidelity -> More Dangerous)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15, label='Statistically Significant (p < 0.05)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Graph Saved: {OUTPUT_PLOT}")

if __name__ == "__main__":
    # 1. Compute (if needed)
    # Check if we already have the embeddings file to save time, else compute
    if not os.path.exists(OUTPUT_CSV_DATA):
        compute_embeddings()
    
    # 2. Plot
    plot_and_report()
