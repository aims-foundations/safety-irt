import asyncio
import pandas as pd
import os
import random
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_FILE = "xsafety.csv"
OUTPUT_FILE = "xsafety_gemini_pass1.csv"
API_KEY = "ADD_YOUR_API_KEY"

MODELS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    #"gemini-2.0-flash"
    #"gemini-2.5-flash-preview-09-2025", 
]



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

# [FIX] Start SMALL. 40 is too high for new/preview models. 
# Once you see it moving, you can restart with 10 or 20.
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

async def get_gemini_response(model, prompt, prompt_id, language, variant_name, settings, pass_index, tags=""):
    test_taker_name = f"{model}_{variant_name}"
    
    # [NOTE] New SDK often prefers 'HarmCategory' enums, but strings usually work.
    # If this fails, we catch it in the exception block.
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
        max_retries_errors = 3
        error_attempts = 0
        
        while True:
            try:
                # [DEBUG] Print to verify we are actually sending requests
                # tqdm.write(f"➡️ Sending: {test_taker_name} (ID: {prompt_id})") 

                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config
                )
                
                # Handle cases where safety filters block the .text property
                try:
                    response_text = resp.text
                except Exception:
                    response_text = "[BLOCKED_OR_EMPTY]"

                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "prompt": prompt,
                    "model": model,
                    "test_taker": test_taker_name,
                    "pass": pass_index,
                    "temperature": settings["temp"],
                    "top_p": settings["top_p"],
                    "response": response_text,
                    "finish_reason": "stop",
                    "tags": tags
                }
                
                result_buffer.append(result_data)
                if len(result_buffer) >= 20: # Save more frequently
                    await save_buffer()
                return result_data
                
            except Exception as e:
                err_str = str(e)
                
                is_retryable = (
                    "429" in err_str or 
                    "RESOURCE_EXHAUSTED" in err_str or
                    "503" in err_str or
                    "UNAVAILABLE" in err_str or
                    "overloaded" in err_str or
                    "Quota" in err_str
                )

                if is_retryable:
                    sleep_time = random.uniform(20, 40)
                    # [DEBUG] Visibly warn user about rate limits
                    tqdm.write(f"🛑 429/Busy: {test_taker_name}. Sleeping {int(sleep_time)}s...")
                    await asyncio.sleep(sleep_time)
                    continue 
                
                tqdm.write(f"⚠️ Fatal Error {test_taker_name}: {e}")
                error_attempts += 1
                
                if error_attempts >= max_retries_errors:
                    fail_data = {
                        "id": prompt_id, "language": language, "prompt": prompt,
                        "model": model, "test_taker": test_taker_name, "pass": pass_index,
                        "temperature": settings["temp"], "top_p": settings["top_p"],
                        "response": f"ERROR: {str(e)}", "finish_reason": "error"
                    }
                    result_buffer.append(fail_data)
                    return fail_data
                
                await asyncio.sleep(2 ** error_attempts)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
async def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

    df = pd.read_csv(INPUT_FILE)
    print(f"Processing ALL Rows: {len(df)}")

    completed_keys = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, on_bad_lines='skip', engine='python')
            if 'pass' in existing_df.columns and 'test_taker' in existing_df.columns:
                for _, row in existing_df.iterrows():
                    key = (row['id'], row['language'], row['test_taker'], row['pass'])
                    completed_keys.add(key)
            print(f"✅ Resuming: {len(completed_keys)} completed.")
        except Exception:
            pass

    tasks = []
    
    # Expected tota
    total_expected = len(df) * len(MODELS) * len(VARIANTS) * 1
    print(f"🎯 Target Total: {total_expected}")

    for _, row in df.iterrows():
        for model in MODELS:
            for variant_name, settings in VARIANTS.items():
                test_taker_name = f"{model}_{variant_name}"
                for pass_index in range(1):
                    unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    if unique_key in completed_keys: continue
                    
                    tasks.append(asyncio.create_task(
                        get_gemini_response(
                            model, row['prompt'], row['id'], row['language'],
                            variant_name, settings, pass_index, tags=row.get('category', '')
                        )
                    ))

    if not tasks:
        print("🎉 All tasks complete!")
        return

    print(f"⚡ Queueing {len(tasks)} tasks...")

    # Start the loop
    for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Gemini Generation")):
        await f
        if i % 100 == 0 and i > 0:
            await save_buffer() # Frequent saving
    
    await save_buffer()
    print(f"\n🎉 DONE!")

if __name__ == "__main__":
    asyncio.run(main())