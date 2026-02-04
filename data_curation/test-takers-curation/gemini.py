#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install asyncio pandas google-genai tqdm')


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
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "multijail.csv"
OUTPUT_FILE = "gemini.csv"

# [FIX] PASTE YOUR KEY HERE
API_KEY = ""

# --- THE MODELS ---
# Changed to only the requested model
MODELS = [
    "gemini-3-flash-preview", 
    "gemini-2.5-flash", 
    "gemini-2.5-flash-preview", 
    "gemini-2.5-flash-lite", 
    "gemini-2.5-flash-lite-preview", 
    "gemini-2.0-flash"
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

client = genai.Client(api_key=API_KEY)
semaphore = asyncio.Semaphore(40)

# Local buffer for appending (we no longer keep ALL results in RAM)
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
            
            # Check if file exists to determine if we need a header
            file_exists = os.path.exists(OUTPUT_FILE)
            
            # APPEND MODE ('a'): efficient and low memory
            df_save.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False, encoding="utf-8")
            
            # Clear memory immediately
            result_buffer.clear()

async def get_gemini_response(model, prompt, prompt_id, language, variant_name, settings):
    test_taker_name = f"{model}_{variant_name}"
    
    safety_config = [
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE")
    ]

    gen_config = types.GenerateContentConfig(
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_output_tokens=1024,
        safety_settings=safety_config
    )

    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config
                )
                
                response_text = resp.text if resp.text else "[BLOCKED_BY_API]"
                # tqdm.write(response_text[:200]) # Optional: Commented out to reduce noise
                
                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "model": model,
                    "variant": variant_name,
                    "test_taker": test_taker_name,
                    "response": response_text,
                    "finish_reason": "stop",
                    "temperature": settings["temp"],
                    "top_p": settings["top_p"]
                }
                
                # Add to local buffer
                result_buffer.append(result_data)
                
                # Save batch every 50 rows
                if len(result_buffer) >= 50:
                    await save_buffer()
                
                return result_data
                
            except Exception as e:
                # ONLY print errors (so you know if something is wrong)
                tqdm.write(f"⚠️ Error {test_taker_name}: {e}")
                
                if attempt == retries - 1:
                    fail_data = {
                        "id": prompt_id,
                        "language": language,
                        "model": model,
                        "variant": variant_name,
                        "test_taker": test_taker_name,
                        "response": f"ERROR: {str(e)}",
                        "finish_reason": "error",
                        "temperature": settings["temp"],
                        "top_p": settings["top_p"]
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
    
    # --- SLICING LOGIC ---
    print(f"Original Total Rows: {len(df)}")
    df = df.iloc[:2300] # Picking up from row 2300
    print(f"Processing Rows: {len(df)} (Starting from index 2300)")

    completed_keys = set()

    # --- RESUME LOGIC ---
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Checking existing progress in '{OUTPUT_FILE}'...")
        try:
            # We read the file ONCE just to see what's done
            existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8", on_bad_lines='skip', engine='python')
            
            for _, row in existing_df.iterrows():
                key = (row['id'], row['language'], row['test_taker'])
                completed_keys.add(key)
                
            print(f"✅ Resuming... {len(completed_keys)} responses already found.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh.")

    # --- GENERATE TASKS ---
    tasks = []
    skipped_count = 0
    
    total_expected = len(df) * len(MODELS) * len(VARIANTS)
    print(f"🎯 Target Total: {total_expected} responses")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                
                unique_key = (row['id'], row['language'], test_taker_name)
                
                if unique_key in completed_keys:
                    skipped_count += 1
                    continue
                
                tasks.append(asyncio.create_task(
                    get_gemini_response(
                        model, 
                        row['prompt'], 
                        row['id'], 
                        row['language'],
                        variant_name,
                        settings
                    )
                ))

    print(f"⏩ Skipped {skipped_count} already completed.")
    
    if not tasks:
        print("🎉 All tasks are already complete!")
        return

    print(f"⚡ Queueing remaining {len(tasks)} tasks...")

    # --- RUN LOOP ---
    # We use miniters to update the progress bar less frequently to save memory
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Gemini Generation", miniters=100)):
        await f
        # Optional: Print a "heartbeat" every 1000 requests just so you know it's alive
        if i % 1000 == 0 and i > 0:
            tqdm.write(f"❤️ Alive: Completed {i} requests...")
    
    # Final flush of any remaining buffer items
    await save_buffer()
    print(f"\n🎉 DONE! All requests processed.")

if __name__ == "__main__":
    await main()


# In[5]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "gemini.csv"
OUTPUT_FILE = "multijail_results_gemini_clean.csv"

def clean_and_analyze():
    # 1. Load the Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Check for API Blocks/Errors
    # In Gemini, "Blocked" responses often come back empty or null if not handled
    df['response'] = df['response'].fillna("")
    
    # 3. Simple pass-through (No specific model removal needed anymore)
    df_clean = df.copy()
    
    # 4. Save the Cleaned File
    df_clean.to_csv(OUTPUT_FILE, index=False)
    
    # --- REPORT ---
    print("\n" + "="*40)
    print("        CLEANING REPORT")
    print("="*40)
    print(f"Total Rows: {len(df_clean)}")
    
    # Check for empty responses (API blocks)
    empty_count = len(df_clean[df_clean['response'].str.strip() == ""])
    if empty_count > 0:
        print(f"⚠️ Warning: {empty_count} rows have empty responses (Possible API blocks)")
    else:
        print("✅ No empty responses detected.")
        
    print(f"Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_analyze()


# In[6]:


import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "multijail_results_gemini_clean.csv"

def analyze_length_quality():
    df = pd.read_csv(INPUT_FILE)
    
    # Gemini doesn't always provide 'finish_reason' in the same way OpenAI does,
    # so we analyze short responses generally.
    
    df['response'] = df['response'].fillna("")
    
    # Categorize
    empty_mask = df['response'].str.strip() == ""
    # Gemini Flash is concise; responses <50 chars might be valid refusals like "I cannot do that."
    short_mask = (df['response'].str.len() < 50) & (~empty_mask)
    valid_mask = df['response'].str.len() >= 50
    
    empty_count = df[empty_mask].shape[0]
    short_count = df[short_mask].shape[0]
    valid_count = df[valid_mask].shape[0]
    
    print("="*40)
    print("      GEMINI RESPONSE ANALYSIS")
    print("="*40)
    print(f"Total Rows:       {len(df)}")
    print(f"❌ Empty/Blocked: {empty_count}")
    print(f"⚠️ Short (<50c):  {short_count}   <-- Likely Refusals")
    print(f"✅ Long Content:  {valid_count}")
    
    if short_count > 0:
        print("\nExamples of Short Responses (Check if these are refusals):")
        print(df[short_mask][['test_taker', 'response']].head(5))

if __name__ == "__main__":
    analyze_length_quality()


# In[7]:


get_ipython().system('pip install tiktoken')


# In[8]:


import pandas as pd
import os

# --- CONFIGURATION ---
FILES_TO_MERGE = [
    # "gpt_nonreasoning_test-takers.csv", # Uncomment if you want to merge old files
    "multijail_results_gemini_clean.csv"  # Your new Gemini data
]

OUTPUT_FILE = "Gemini_test-takers.csv"

def merge_datasets():
    # ... (Rest of logic remains identical to your original Block 8) ...
    dataframes = []
    print(f"Starting merge of {len(FILES_TO_MERGE)} files...\n")

    for filename in FILES_TO_MERGE:
        if not os.path.exists(filename):
            print(f"⚠️ Warning: File not found: {filename} (Skipping)")
            continue
            
        try:
            df = pd.read_csv(filename)
            print(f"✅ Loaded {filename}: {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            print(f"   ❌ Failed to read {filename}: {e}")

    if not dataframes:
        print("No data found to merge.")
        return

    master_df = pd.concat(dataframes, ignore_index=True)
    if 'id' in master_df.columns:
        master_df = master_df.sort_values(by=['id', 'test_taker'])

    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 MERGE COMPLETE")
    print("="*40)
    print(f"Total Rows:     {len(master_df)}")
    print(f"Unique Models:  {master_df['test_taker'].nunique()}")
    print(f"Saved to:       {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_datasets()


# In[9]:


import pandas as pd
import tiktoken

# --- CONFIGURATION ---
INPUT_FILE = "Gemini_test-takers.csv"
ENCODING_NAME = "cl100k_base" 

# ... (Rest of logic remains identical to your original Block 9) ...
# It will calculate token counts using the standard OpenAI tokenizer
# which serves as a good enough proxy for length complexity in IRT.

def count_tokens(text, encoding):
    if pd.isna(text):
        return 0
    return len(encoding.encode(str(text)))

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"File {INPUT_FILE} not found.")
        return

    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
        print(f"Using tokenizer: {ENCODING_NAME} (Standard Proxy)")
    except Exception:
        print("tiktoken error. Using word count approx.")
        df['token_count'] = df['response'].apply(lambda x: len(str(x).split()) * 1.3 if pd.notna(x) else 0)
    else:
        print("Calculating tokens...")
        df['token_count'] = df['response'].apply(lambda x: count_tokens(x, encoding))

    avg_tokens = df['token_count'].mean()
    print("\n" + "="*40)
    print(f"GLOBAL AVERAGE: {avg_tokens:.2f} tokens/response")
    print("="*40)

    if 'test_taker' in df.columns:
        print("\nAverage Tokens per Test Taker:")
        breakdown = df.groupby('test_taker')['token_count'].mean().sort_values(ascending=False)
        print(breakdown.to_string(float_format="%.1f"))

if __name__ == "__main__":
    main()

