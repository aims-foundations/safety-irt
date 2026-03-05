# -*- coding: utf-8 -*-
"""
LaBSE Gold Standard Validator.
Filters anchor prompts by ensuring strict semantic similarity (>0.75) across all languages.
Automatically downloads input data from Hugging Face.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import sys
from huggingface_hub import snapshot_download

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ID = "MaxZ119/safetyirt"
INPUT_FILENAME = "anchors/anchors.csv"
OUTPUT_FILE = "GOLD_VALIDATED_ANCHORS.csv"
THRESHOLD = 0.75
MODEL_NAME = "sentence-transformers/LaBSE"

# Columns to ignore (Metadata)
IGNORE_COLS = [
    'id', 'en', 'verdicts', 'rationale_gpt', 'rationale', 
    'verdict', 'is_strict_anchor', 'is_majority_anchor', 'is_anchor',
    'verdicts_str', 'rationale_claude', 'rationale_gemini'
]

def filter_anchors():
    print(f"---DOWNLOADING DATA FROM {REPO_ID} ---")
    data_dir = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
    input_path = os.path.join(data_dir, "processed_data", "Master_Passes0-9_Dataset.csv")

    df = pd.read_csv(input_path, on_bad_lines='skip')

    # 2. Identify Language Columns
    target_langs = [c for c in df.columns if c not in IGNORE_COLS and c != 'en']
    print(f"Target Languages to check: {target_langs}")

    if not target_langs or 'en' not in df.columns:
        print("❌ Error: Could not identify language columns or 'en' is missing.")
        return

    # 3. Load Model
    print("Loading LaBSE Model...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Compute Scores & Filter
    print("Validating semantic equivalence...")
    
    valid_rows = []
    dropped_count = 0
    
    for index, row in df.iterrows():
        en_text = str(row['en'])
        en_emb = model.encode(en_text, convert_to_tensor=True)
        
        min_score = 1.0
        worst_lang = ""
        
        passed = True
        for lang in target_langs:
            lang_text = str(row[lang])
            lang_emb = model.encode(lang_text, convert_to_tensor=True)
            
            sim = util.pytorch_cos_sim(en_emb, lang_emb).item()
            
            if sim < min_score:
                min_score = sim
                worst_lang = lang
            
            if sim <= THRESHOLD:
                passed = False
                break 
        
        if passed:
            valid_rows.append(row)
        else:
            dropped_count += 1
            print(f"   ❌ Dropping ID {row['id']}: '{worst_lang}' score is {min_score:.3f}")

    gold_df = pd.DataFrame(valid_rows)
    gold_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*50)
    print("FILTERING COMPLETE")
    print("="*50)
    print(f"Original Candidates:  {len(df)}")
    print(f"Dropped Rows: {dropped_count}")
    print(f"FINAL GOLD ANCHORS: {len(gold_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    filter_anchors()