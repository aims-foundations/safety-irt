#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install asyncio pandas openai tqdm')


# In[2]:


#convert MultiJail to usable format
import pandas as pd
import io
import requests


URL = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv"

def prepare_multijail_data():
    print(f"Downloading data from {URL}...")
    response = requests.get(URL)
    response.raise_for_status()
    
    df_wide = pd.read_csv(io.StringIO(response.text))
    
    print("Original Columns:", df_wide.columns.tolist())
    lang_columns = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
    available_langs = [col for col in lang_columns if col in df_wide.columns]
    
    # Reshape from WIDE (Columns) to LONG (Rows)
    # id | en | zh  -->  id | language | prompt
    df_long = df_wide.melt(
        id_vars=['id'], 
        value_vars=available_langs, 
        var_name='language', 
        value_name='prompt'
    )
    
    # Clean up
    df_long = df_long.dropna(subset=['prompt']) 
    df_long['prompt'] = df_long['prompt'].str.strip()
    
    # Save for the async script
    output_filename = "multijail.csv"
    df_long.to_csv(output_filename, index=False)
    
    print(f"\nSuccess! Converted {len(df_wide)} prompts into {len(df_long)} rows.")
    print(f"Saved to '{output_filename}'. You can now run the asyncio scripts.")
    print(df_long.head())

if __name__ == "__main__":
    prepare_multijail_data()


# In[1]:


get_ipython().system('pip install typing_extensions --upgrade')


# In[ ]:

import asyncio
import pandas as pd
import os
from openai import AsyncOpenAI, APIStatusError
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "multijail.csv"
OUTPUT_FILE = "multijail_results_grok4_reasoning.csv"

# [FIX] PASTE YOUR xAI KEY HERE
API_KEY = ""

# --- MODEL LIST ---
MODELS = [
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning"
]

# --- THE 4 VARIANTS ---
VARIANTS = {
    "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
    "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
    "High_Risk":      {"temp": 1.0, "top_p": 0.95},
    "Chaos":          {"temp": 1.3, "top_p": 1.0}
}

# ==============================================================================
# 2. CLIENT SETUP
# ==============================================================================
if not API_KEY or "PASTE" in API_KEY:
    raise ValueError("❌ You forgot to paste your API Key!")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.x.ai/v1"
)

# [NOTE] Reasoning models are heavy/slow. 
# Reduced from 40 to 10 to avoid hitting Rate Limits (429) instantly.
semaphore = asyncio.Semaphore(40) 

result_buffer = []
save_lock = asyncio.Lock()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
async def save_buffer():
    """Appends the current buffer to the CSV and clears memory."""
    async with save_lock:
        if result_buffer:
            df_save = pd.DataFrame(result_buffer)
            file_exists = os.path.exists(OUTPUT_FILE)
            df_save.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False, encoding="utf-8")
            result_buffer.clear()

async def get_grok_response(model, prompt, prompt_id, language, variant_name, settings, pass_index):
    test_taker_name = f"{model}_{variant_name}"
    
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                # Standard OpenAI-compatible call
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings["temp"],
                    top_p=settings["top_p"],
                    max_completion_tokens=2048 # Reasoning needs room
                )
                
                response_text = resp.choices[0].message.content
                finish_reason = resp.choices[0].finish_reason or "stop"
                
                if not response_text: response_text = "[EMPTY_RESPONSE]"

                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "model": model,
                    "variant": variant_name,
                    "test_taker": test_taker_name,
                    "pass": pass_index, # [NEW] Track which pass this is
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "temperature": settings["temp"],
                    "top_p": settings["top_p"]
                }
                
                result_buffer.append(result_data)
                if len(result_buffer) >= 20: await save_buffer()
                return result_data

            # --- API ERROR HANDLING ---
            except APIStatusError as e:
                # 403 = Content Filter
                if e.status_code == 403:
                    filter_data = {
                        "id": prompt_id, "language": language, "model": model, "variant": variant_name,
                        "test_taker": test_taker_name, "pass": pass_index,
                        "response": f"[BLOCKED_BY_API] {str(e)}", 
                        "finish_reason": "content_filter", "temperature": settings["temp"], "top_p": settings["top_p"]
                    }
                    result_buffer.append(filter_data)
                    return filter_data
                
                # 429 = Rate Limit
                elif e.status_code == 429:
                    wait_time = 10 * (attempt + 1) # Wait longer for reasoning models
                    # tqdm.write(f"⏳ Rate limit on {test_taker_name}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                
                else:
                    tqdm.write(f"⚠️ API Error {test_taker_name}: {e}")

            except Exception as e:
                tqdm.write(f"⚠️ Error {test_taker_name}: {e}")
            
            # Retry logic
            if attempt == retries - 1:
                fail_data = {
                    "id": prompt_id, "language": language, "model": model, "variant": variant_name,
                    "test_taker": test_taker_name, "pass": pass_index,
                    "response": "ERROR_FAILED_RETRIES", "finish_reason": "error",
                    "temperature": settings["temp"], "top_p": settings["top_p"]
                }
                result_buffer.append(fail_data)
                return fail_data
            
            await asyncio.sleep(2 ** attempt + 1)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
async def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    df = pd.read_csv(INPUT_FILE)
    completed_keys = set()

    # --- RESUME LOGIC ---
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Checking existing progress in '{OUTPUT_FILE}'...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8", on_bad_lines='skip', engine='python')
            if 'pass' in existing_df.columns:
                for _, row in existing_df.iterrows():
                    # Key now includes pass index
                    key = (row['id'], row['language'], row['test_taker'], row['pass'])
                    completed_keys.add(key)
            else:
                print("⚠️ Existing file missing 'pass' column. Overwriting or starting fresh advised.")
                
            print(f"✅ Resuming... {len(completed_keys)} responses found.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh.")

    # --- GENERATE TASKS ---
    tasks = []
    skipped_count = 0
    
    # [FIXED] Changed * 3 to * 2
    total_expected = len(df) * len(MODELS) * len(VARIANTS) * 2 
    print(f"🎯 Target Total: {total_expected} responses (Pass 2)")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                
                # [FIXED] Changed range(3) to range(2)
                for pass_index in range(2):
                    unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    
                    if unique_key in completed_keys:
                        skipped_count += 1
                        continue
                    
                    tasks.append(asyncio.create_task(
                        get_grok_response(
                            model, row['prompt'], row['id'], row['language'], variant_name, settings, pass_index
                        )
                    ))

    print(f"⏩ Skipped {skipped_count} already completed.")
    if not tasks:
        print("🎉 All tasks are already complete!")
        return

    print(f"⚡ Queueing remaining {len(tasks)} tasks...")

    # --- RUN LOOP ---
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Grok Reasoning Generation", miniters=50)):
        await f
        if i % 100 == 0 and i > 0:
            tqdm.write(f"❤️ Alive: Completed {i} requests...")
    
    await save_buffer()
    print(f"\n🎉 DONE! All requests processed.")

if __name__ == "__main__":
    await main()

# In[2]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "multijail_results_grok3mini.csv"
OUTPUT_FILE = "multijail_results_grok3mini_clean.csv"

def clean_and_analyze():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Fill NaN
    df['response'] = df['response'].fillna("")
    
    # Save Clean
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Report
    print("\n" + "="*40)
    print("        CLEANING REPORT")
    print("="*40)
    print(f"Total Rows: {len(df)}")
    
    empty_count = len(df[df['response'].str.strip() == ""])
    if empty_count > 0:
        print(f"⚠️ Warning: {empty_count} rows have empty responses.")
    else:
        print("✅ No empty responses detected.")
        
    print(f"Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_analyze()


# In[3]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "multijail_results_grok3mini_clean.csv"

def analyze_length_quality():
    df = pd.read_csv(INPUT_FILE)
    df['response'] = df['response'].fillna("")
    
    # Categorize
    empty_mask = df['response'].str.strip() == ""
    short_mask = (df['response'].str.len() < 50) & (~empty_mask)
    valid_mask = df['response'].str.len() >= 50
    
    print("="*40)
    print("      GROK RESPONSE ANALYSIS")
    print("="*40)
    print(f"Total Rows:       {len(df)}")
    print(f"❌ Empty:         {df[empty_mask].shape[0]}")
    print(f"⚠️ Short (<50c):  {df[short_mask].shape[0]}")
    print(f"✅ Long Content:  {df[valid_mask].shape[0]}")
    
    if df[short_mask].shape[0] > 0:
        print("\nExamples of Short Responses:")
        print(df[short_mask][['test_taker', 'response']].head(5))

if __name__ == "__main__":
    analyze_length_quality()


# In[4]:


get_ipython().system('pip install tiktoken')


# In[5]:


import pandas as pd
import os

# --- CONFIGURATION ---
FILES_TO_MERGE = [
    "multijail_results_grok3mini_clean.csv"
]

OUTPUT_FILE = "Grok_3_mini_test_taker.csv"

def merge_datasets():
    dataframes = []
    print(f"Starting merge...")

    for filename in FILES_TO_MERGE:
        if not os.path.exists(filename):
            print(f"⚠️ Warning: File not found: {filename}")
            continue
        df = pd.read_csv(filename)
        print(f"✅ Loaded {filename}: {len(df)} rows")
        dataframes.append(df)

    if not dataframes:
        print("No data found.")
        return

    master_df = pd.concat(dataframes, ignore_index=True)
    if 'id' in master_df.columns:
        master_df = master_df.sort_values(by=['id', 'test_taker'])

    master_df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 Saved to {OUTPUT_FILE} ({len(master_df)} rows)")

if __name__ == "__main__":
    merge_datasets()


# In[6]:


import pandas as pd
import tiktoken

# --- CONFIGURATION ---
INPUT_FILE = "Grok_3_mini_test_taker.csv"
ENCODING_NAME = "cl100k_base" # Grok is compatible with this encoding

def count_tokens(text, encoding):
    if pd.isna(text): return 0
    return len(encoding.encode(str(text)))

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found.")
        return

    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
        print("Calculating tokens...")
        df['token_count'] = df['response'].apply(lambda x: count_tokens(x, encoding))
    except Exception:
        print("Using word count approx.")
        df['token_count'] = df['response'].apply(lambda x: len(str(x).split()) * 1.3)

    print(f"\nGLOBAL AVERAGE: {df['token_count'].mean():.2f} tokens/response")

    if 'test_taker' in df.columns:
        print("\nAverage Tokens per Test Taker:")
        breakdown = df.groupby('test_taker')['token_count'].mean().sort_values(ascending=False)
        print(breakdown.to_string(float_format="%.1f"))

if __name__ == "__main__":
    main()


# In[ ]:




