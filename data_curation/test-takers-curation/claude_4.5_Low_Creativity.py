#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install asyncio pandas anthropic tqdm')


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


# In[3]:


get_ipython().system('pip install typing_extensions --upgrade')


# In[1]:


import asyncio
import pandas as pd
import os
import random
from anthropic import AsyncAnthropic, APIStatusError, RateLimitError, APIError
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "multijail.csv"
OUTPUT_FILE = "multijail_results_claude_4.5_low_creativity.csv"

# [FIX] PASTE YOUR ANTHROPIC KEY HERE
API_KEY = ""

# --- CORRECTED MODEL LIST ---
MODELS = [
    "claude-haiku-4-5-20251001"   
]

# --- THE 4 VARIANTS ---
# Removed top_p here so we rely on the API default (usually 1.0)
VARIANTS = {
    "Low_Creativity": {"temp": 0.4}
}

# ==============================================================================
# 2. CLIENT SETUP
# ==============================================================================
if not API_KEY or "PASTE" in API_KEY:
    raise ValueError("❌ You forgot to paste your API Key!")

client = AsyncAnthropic(api_key=API_KEY)

# LOWER CONCURRENCY to prevent 529 loops (3 is safe, 5 is risky during peak hours)
semaphore = asyncio.Semaphore(3) 

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

async def get_claude_response(model, prompt, prompt_id, language, variant_name, settings, pass_index):
    test_taker_name = f"{model}_{variant_name}"
    
    async with semaphore:
        retries = 5  # Increased retries for 529 errors
        for attempt in range(retries):
            try:
                # REMOVED top_p from the API call below
                resp = await client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings["temp"],
                    max_tokens=1024
                )
                
                response_text = resp.content[0].text
                finish_reason = resp.stop_reason or "stop_sequence"
                if not response_text: response_text = "[EMPTY_RESPONSE]"

                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "model": model,
                    "variant": variant_name,
                    "test_taker": test_taker_name,
                    "pass": pass_index, # [NEW] Added Pass Index
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "temperature": settings["temp"],
                    "top_p": "default" # Placeholder since we aren't setting it
                }
                
                result_buffer.append(result_data)
                if len(result_buffer) >= 20: await save_buffer()
                return result_data

            except APIStatusError as e:
                # --- HANDLER FOR 529 (OVERLOADED) & 429 (RATE LIMIT) ---
                if e.status_code in [429, 529] or "overloaded" in str(e).lower():
                    # Backoff: 5s, 10s, 20s... + random jitter to prevent thundering herd
                    wait_time = (5 * (attempt + 1)) + random.uniform(0, 3)
                    
                    # Only print if it's a long wait (to keep console clean)
                    if attempt > 1: 
                        tqdm.write(f"⏳ Server Busy ({e.status_code}) - {test_taker_name} waiting {wait_time:.1f}s...")
                    
                    await asyncio.sleep(wait_time)
                    continue # Force retry loop
                
                # --- HANDLER FOR 404 (MODEL NOT FOUND) ---
                elif e.status_code == 404:
                    fail_data = {
                        "id": prompt_id, "language": language, "model": model, "variant": variant_name,
                        "test_taker": test_taker_name, "pass": pass_index,
                        "response": "ERROR_MODEL_NOT_FOUND", "finish_reason": "error",
                        "temperature": settings["temp"], "top_p": "default"
                    }
                    result_buffer.append(fail_data)
                    return fail_data
                
                # --- HANDLER FOR CONTENT FILTERS (400/403) ---
                elif e.status_code in [400, 403]:
                    filter_data = {
                        "id": prompt_id, "language": language, "model": model, "variant": variant_name,
                        "test_taker": test_taker_name, "pass": pass_index,
                        "response": f"[BLOCKED_BY_API] {str(e)}", "finish_reason": "content_filter",
                        "temperature": settings["temp"], "top_p": "default"
                    }
                    result_buffer.append(filter_data)
                    return filter_data
                
                else:
                    tqdm.write(f"⚠️ Unexpected API Error {test_taker_name}: {e}")

            except Exception as e:
                tqdm.write(f"⚠️ Error {test_taker_name}: {e}")
            
            # General Retry Logic (for non-APIStatusErrors)
            if attempt == retries - 1:
                fail_data = {
                    "id": prompt_id, "language": language, "model": model, "variant": variant_name,
                    "test_taker": test_taker_name, "pass": pass_index,
                    "response": "ERROR_FAILED_RETRIES", "finish_reason": "error",
                    "temperature": settings["temp"], "top_p": "default"
                }
                result_buffer.append(fail_data)
                return fail_data
            
            await asyncio.sleep(2 ** attempt)

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
                    # Key includes pass index
                    key = (row['id'], row['language'], row['test_taker'], row['pass'])
                    completed_keys.add(key)
            else:
                 print("⚠️ Existing file missing 'pass' column. Overwriting or starting fresh advised.")
                 
            print(f"✅ Resuming... {len(completed_keys)} responses already found.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh.")

    # --- GENERATE TASKS ---
    tasks = []
    skipped_count = 0
    
    # [FIXED] Multiplied by 2 for Pass 2
    total_expected = len(df) * len(MODELS) * len(VARIANTS) * 2
    print(f"🎯 Target Total: {total_expected} responses")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                
                # [NEW] Pass Loop
                for pass_index in range(2):
                    unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    
                    if unique_key in completed_keys:
                        skipped_count += 1
                        continue
                    
                    tasks.append(asyncio.create_task(
                        get_claude_response(
                            model, row['prompt'], row['id'], row['language'], variant_name, settings, pass_index
                        )
                    ))

    print(f"⏩ Skipped {skipped_count} already completed.")
    if not tasks:
        print("🎉 All tasks are already complete!")
        return

    print(f"⚡ Queueing remaining {len(tasks)} tasks...")

    # --- RUN LOOP ---
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Claude Generation", miniters=100)):
        await f
        if i % 100 == 0 and i > 0:
            tqdm.write(f"❤️ Alive: Completed {i} requests...")
    
    await save_buffer()
    print(f"\n🎉 DONE! All requests processed.")

if __name__ == "__main__":
    # If running in a standard Python script, use asyncio.run(main())
    # If running in Jupyter, use await main()
    try:
        asyncio.run(main())
    except RuntimeError:
        await main()


# In[7]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "multijail_results_claude_4.5_low_creativity.csv"
OUTPUT_FILE = "multijail_results_claude_4.5_low_creativity.csv_clean.csv"

def clean_and_analyze():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    df['response'] = df['response'].fillna("")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("        CLEANING REPORT")
    print("="*40)
    print(f"Total Rows: {len(df)}")
    
    empty_count = len(df[df['response'].str.strip() == ""])
    print(f"⚠️ Empty/Blocked Responses: {empty_count}")
    print(f"Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_analyze()


# In[8]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "multijail_results_claude_4.5_low_creativity.csv_clean.csv"

def analyze_length_quality():
    df = pd.read_csv(INPUT_FILE)
    df['response'] = df['response'].fillna("")
    
    empty_mask = df['response'].str.strip() == ""
    short_mask = (df['response'].str.len() < 50) & (~empty_mask)
    valid_mask = df['response'].str.len() >= 50
    
    print("="*40)
    print("      CLAUDE RESPONSE ANALYSIS")
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


# In[9]:


get_ipython().system('pip install tiktoken')


# In[10]:


import pandas as pd
import os

# --- CONFIGURATION ---
FILES_TO_MERGE = [
    "multijail_results_claude_4.5_low_creativity.csv_clean.csv"
]

OUTPUT_FILE = "Claude_4.5_Low_Creativity.csv"

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


# In[11]:


import pandas as pd
import tiktoken

# --- CONFIGURATION ---
INPUT_FILE = "Claude_4.5_Low_Creativity.csv"
ENCODING_NAME = "cl100k_base" # Valid proxy

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




