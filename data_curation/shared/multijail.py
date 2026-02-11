"""Download and reshape the MultiJail dataset from wide to long format."""

import pandas as pd
import io
import requests

# URL for the raw MultiJail dataset
URL = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv"
OUTPUT_FILE = "multijail.csv"

def prepare_multijail_data():
    print(f":arrow_down: Downloading data from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f":x: Error downloading file: {e}")
        return

    # Load raw data
    df_wide = pd.read_csv(io.StringIO(response.text))
    print(f"   - Downloaded {len(df_wide)} rows.")
    print(f"   - Columns found: {df_wide.columns.tolist()}")

    # --- CRITICAL STEP: Define ID Variables ---
    # We MUST include 'tags' here so it stays attached to every language row
    id_vars = ['id']
    if 'tags' in df_wide.columns:
        id_vars.append('tags')
        print("   - :white_check_mark: Found 'tags' column. It will be preserved.")
    else:
        print("   - :warning: Warning: 'tags' column NOT found in source. Output will have no tags.")

    # Define Language Columns to melt
    target_langs = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
    available_langs = [col for col in target_langs if col in df_wide.columns]

    # Reshape from WIDE to LONG
    # This repeats the ID and TAG for every language row created
    print("   - Reshaping data (Wide -> Long)...")
    df_long = df_wide.melt(
        id_vars=id_vars,            # <--- This keeps 'tags' with the ID
        value_vars=available_langs,
        var_name='language',
        value_name='prompt'
    )

    # Clean up
    df_long = df_long.dropna(subset=['prompt'])
    df_long['prompt'] = df_long['prompt'].str.strip()
    
    # Save
    df_long.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f":white_check_mark: Success! Saved to {OUTPUT_FILE}")
    print(f"   - Total Rows: {len(df_long)}")
    print(f"   - Columns: {df_long.columns.tolist()}")
    print("="*40)
    
    # Validation Preview
    print("\nPreview of data with tags:")
    print(df_long[['id', 'language', 'tags']].head())

if __name__ == "__main__":
    prepare_multijail_data()
