"""Download and reshape the MultiJail dataset from wide to long format."""

import pandas as pd
import io
import requests

MULTIJAIL_URL = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv"
LANGUAGES = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']


def prepare_multijail_data(output_path="multijail.csv"):
    """Download MultiJail CSV and reshape from wide (one col per language) to long format.

    Output columns: id, language, prompt
    """
    print(f"Downloading data from {MULTIJAIL_URL}...")
    response = requests.get(MULTIJAIL_URL)
    response.raise_for_status()

    df_wide = pd.read_csv(io.StringIO(response.text))
    print("Original Columns:", df_wide.columns.tolist())

    available_langs = [col for col in LANGUAGES if col in df_wide.columns]

    # Reshape: id | en | zh  -->  id | language | prompt
    df_long = df_wide.melt(
        id_vars=['id'],
        value_vars=available_langs,
        var_name='language',
        value_name='prompt'
    )

    df_long = df_long.dropna(subset=['prompt'])
    df_long['prompt'] = df_long['prompt'].str.strip()

    df_long.to_csv(output_path, index=False)

    print(f"\nSuccess! Converted {len(df_wide)} prompts into {len(df_long)} rows.")
    print(f"Saved to '{output_path}'.")
    print(df_long.head())
    return df_long


if __name__ == "__main__":
    prepare_multijail_data()
