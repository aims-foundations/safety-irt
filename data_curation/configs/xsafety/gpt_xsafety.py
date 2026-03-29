import asyncio
import pandas as pd
import os
import random
from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "xsafety.csv"
OUTPUT_FILE = "xsafety-gpt-pass1.csv"

# [FIX] PASTE YOUR OPENAI KEY HERE
API_KEY = "ADD_YOUR_API_KEY"

# --- THE MODELS ---
MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini", 
    "gpt-4.1-nano"
    # Add other models as needed. 
]

# --- THE 4 VARIANTS ---
VARIANTS = {
    "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
    "Standard":       {"temp": 0.7, "top_p": 0.9},
    "High_Risk":      {"temp": 1.0, "top_p": 0.95},
    "Chaos":          {"temp": 1.3, "top_p": 1.0}
}

# ==============================================================================
# 2. CLIENT SETUP
# ==============================================================================
if not API_KEY or "PASTE" in API_KEY:
    raise ValueError("❌ You forgot to paste your API Key!")

client = AsyncOpenAI(api_key=API_KEY)

# Semaphore controls how many concurrent requests fly at once.
semaphore = asyncio.Semaphore(867) 

# Local buffer for appending
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
            # APPEND MODE
            df_save.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False, encoding="utf-8")
            result_buffer.clear()

async def get_gpt_response(model, prompt, prompt_id, language, variant_name, settings, pass_index, tags = ""):
    test_taker_name = f"{model}_{variant_name}"
    
    async with semaphore:
        max_retries_errors = 3
        error_attempts = 0
        
        # Infinite loop for Rate Limits (429) AND Server Errors (5xx)
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings["temp"],
                    top_p=settings["top_p"],
                    max_completion_tokens=1024
                )
                
                response_text = resp.choices[0].message.content
                finish_reason = resp.choices[0].finish_reason
                
                if not response_text: response_text = "[EMPTY_RESPONSE]"
                
                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "prompt": prompt,  # <--- ADDED THIS LINE
                    "model": model,
                    "test_taker": test_taker_name,
                    "pass": pass_index,
                    "temperature": settings["temp"],
                    "top_p": settings["top_p"],
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "tags": tags
                    
                }
                
                result_buffer.append(result_data)
                if len(result_buffer) >= 50:
                    await save_buffer()
                return result_data
                
            except Exception as e:
                err_str = str(e)
                
                # --- 1. HANDLE RETRYABLE ERRORS (Infinite Retry) ---
                is_retryable = (
                    "429" in err_str or 
                    "500" in err_str or 
                    "503" in err_str or
                    "org-" in err_str or 
                    isinstance(e, (APITimeoutError, APIConnectionError))
                )

                if is_retryable:
                    sleep_time = random.uniform(20, 40)
                    await asyncio.sleep(sleep_time)
                    continue 
                
                # --- 2. HANDLE FATAL ERRORS (Max 3 Retries) ---
                tqdm.write(f"⚠️ Error {test_taker_name}: {e}")
                error_attempts += 1
                
                if error_attempts >= max_retries_errors:
                    fail_data = {
                        "id": prompt_id,
                        "language": language,
                        "prompt": prompt,
                        "model": model,
                        "test_taker": test_taker_name,
                        "pass": pass_index,
                        "temperature": settings["temp"],
                        "top_p": settings["top_p"],
                        "response": response_text,
                        "finish_reason": finish_reason
                    }
                    result_buffer.append(fail_data)
                    return fail_data
                
                await asyncio.sleep(2 ** error_attempts + 1)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
async def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    df = pd.read_csv(INPUT_FILE)
    print(f"Processing ALL Rows: {len(df)}")

    completed_keys = set()
    re_queued_count = 0

    # --- SMART RESUME LOGIC ---
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Checking existing progress in '{OUTPUT_FILE}'...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8", on_bad_lines='skip', engine='python')
            
            if 'pass' in existing_df.columns and 'response' in existing_df.columns:
                for _, row in existing_df.iterrows():
                    
                    # Check for previous failures
                    resp_text = str(row.get('response', ''))
                    is_bad_row = (
                        "429" in resp_text or 
                        "500" in resp_text or 
                        "503" in resp_text or
                        "Rate limit" in resp_text or
                        "Connection error" in resp_text
                    )

                    if is_bad_row:
                        re_queued_count += 1
                        continue 

                    key = (row['id'], row['language'], row['test_taker'], row['pass'])
                    completed_keys.add(key)
            else:
                print("⚠️ Existing file columns missing. Starting fresh advised.")
                
            print(f"✅ Valid Resume: {len(completed_keys)} rows.")
            print(f"♻️  Re-queueing: {re_queued_count} rows that had errors.")
            
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}. Starting fresh.")
    else:
        print("🚀 Starting fresh.")

    # --- GENERATE TASKS ---
    tasks = []
    skipped_count = 0
    
    total_expected = len(df) * len(MODELS) * len(VARIANTS) * 1
    print(f"🎯 Target Total: {total_expected} responses")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                
                for pass_index in range(1):
                    unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    
                    if unique_key in completed_keys:
                        skipped_count += 1
                        continue
                    
                    tasks.append(asyncio.create_task(
                        get_gpt_response(
                            model, 
                            row['prompt'], 
                            row['id'], 
                            row['language'], 
                            variant_name, 
                            settings, 
                            pass_index,
                            tags=row.get('category', '')
                        )
                    ))

    print(f"⏩ Skipped {skipped_count} valid rows.")
    
    if not tasks:
        print("🎉 All tasks are already complete!")
        return

    print(f"⚡ Queueing {len(tasks)} tasks (Includes new + retries)...")

    # --- RUN LOOP ---
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT Generation", miniters=100)):
        await f
    
    await save_buffer()
    print(f"\n🎉 DONE! All requests processed.")

if __name__ == "__main__":
    asyncio.run(main())