import asyncio
import pandas as pd
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "multijail.csv"
OUTPUT_FILE = "deepseek_results.csv"

# [FIX] PASTE YOUR DEEPSEEK KEY HERE
API_KEY = "sk-..."  # Replace with your actual DeepSeek API Key

# --- THE MODEL ---
# DeepSeek V3 is accessed via 'deepseek-chat'
MODELS = ["deepseek-chat"]

# --- THE 4 VARIANTS ---
# Adjusted slightly for DeepSeek/OpenAI parameters
VARIANTS = {
    "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
    "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
    "High_Risk":      {"temp": 1.0, "top_p": 0.95},
    "Chaos":          {"temp": 1.3, "top_p": 1.0}
}

# ==============================================================================
# 2. CLIENT SETUP
# ==============================================================================
if not API_KEY or "sk-..." in API_KEY:
    raise ValueError("❌ You forgot to paste your DeepSeek API Key!")

# Initialize OpenAI Client pointing to DeepSeek
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# Concurrency Limit (Adjust based on your tier limits)
semaphore = asyncio.Semaphore(50)

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
            
            # Check if file exists to determine if we need a header
            file_exists = os.path.exists(OUTPUT_FILE)
            
            # APPEND MODE ('a'): efficient and low memory
            df_save.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False, encoding="utf-8")
            
            # Clear memory immediately
            result_buffer.clear()

async def get_deepseek_response(model, prompt, prompt_id, language, variant_name, settings, pass_index):
    test_taker_name = f"{model}_{variant_name}"
    
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                # DeepSeek API Call (OpenAI-compatible)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings["temp"],
                    top_p=settings["top_p"],
                    max_tokens=1024,
                    stream=False
                )
                
                response_text = response.choices[0].message.content
                if not response_text:
                    response_text = "[EMPTY_RESPONSE]"
                
                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "model": model,
                    "variant": variant_name,
                    "test_taker": test_taker_name,
                    "pass": pass_index, 
                    "response": response_text,
                    "finish_reason": response.choices[0].finish_reason,
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
                # Common DeepSeek errors: RateLimitError, APIConnectionError
                tqdm.write(f"⚠️ Error {test_taker_name} (ID {prompt_id}): {e}")
                
                if attempt == retries - 1:
                    fail_data = {
                        "id": prompt_id,
                        "language": language,
                        "model": model,
                        "variant": variant_name,
                        "test_taker": test_taker_name,
                        "pass": pass_index,
                        "response": f"ERROR: {str(e)}",
                        "finish_reason": "error",
                        "temperature": settings["temp"],
                        "top_p": settings["top_p"]
                    }
                    result_buffer.append(fail_data)
                    return fail_data
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt + 1)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
async def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    df = pd.read_csv(INPUT_FILE)
    
    # Optional: Slice for testing
    # df = df.iloc[:10] 
    
    print(f"Processing Rows: {len(df)}")

    completed_keys = set()

    # --- RESUME LOGIC ---
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Checking existing progress in '{OUTPUT_FILE}'...")
        try:
            # We read the file ONCE just to see what's done
            existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8", on_bad_lines='skip', engine='python')
            
            if 'pass' in existing_df.columns:
                for _, row in existing_df.iterrows():
                    # Key now includes pass index
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
    
    total_expected = len(df) * len(MODELS) * len(VARIANTS) * 2
    print(f"🎯 Target Total: {total_expected} responses")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                
                # Pass Loop (0 and 1)
                for pass_index in range(2):
                    unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    
                    if unique_key in completed_keys:
                        skipped_count += 1
                        continue
                    
                    tasks.append(asyncio.create_task(
                        get_deepseek_response(
                            model, 
                            row['prompt'], 
                            row['id'], 
                            row['language'],
                            variant_name,
                            settings,
                            pass_index
                        )
                    ))

    print(f"⏩ Skipped {skipped_count} already completed.")
    
    if not tasks:
        print("🎉 All tasks are already complete!")
        return

    print(f"⚡ Queueing remaining {len(tasks)} tasks...")

    # --- RUN LOOP ---
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="DeepSeek Generation", miniters=100)):
        await f
        if i % 100 == 0 and i > 0:
            tqdm.write(f"❤️ Alive: Completed {i} requests...")
    
    # Final flush of any remaining buffer items
    await save_buffer()
    print(f"\n🎉 DONE! All requests processed.")

if __name__ == "__main__":
    # If running in a standard script
    asyncio.run(main())
    # If running in Jupyter, use: await main()
