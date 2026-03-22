# -*- coding: utf-8 -*-
"""
Qualitative Inspection of High Positive-τ Prompt×Language Pairs — XSafety.
Adapted from irt_validations/high_tau_prompt-response_inspection.py:
  - Single pass (no pass filtering; uses all available responses)
  - Uses XSafety_Dataset.csv
  - category column is a single string
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)

INPUT_FILE = os.path.join(DATA_DIR, "safety-data", "xsafety", "xsafety_pass_graded.csv")

MODEL_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "model", "xsafety", "results",
                             "bayesian_irt_results_binary.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_qualitative_inspection")
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_N = 15
MAX_RESPONSE_LEN = 500
FAM_ORDER = ['Claude', 'DeepSeek', 'Gemini', 'GPT', 'Grok']


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def get_model_family(name):
    name = str(name).lower()
    if any(x in name for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude'   in name: return 'Claude'
    elif 'gemini'   in name: return 'Gemini'
    elif 'grok'     in name: return 'Grok'
    elif 'deepseek' in name: return 'DeepSeek'
    return 'Other'


def truncate(text, max_len=MAX_RESPONSE_LEN):
    text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
    return text[:max_len] + '...' if len(text) > max_len else text


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load τ
# ══════════════════════════════════════════════════════════════════════════

def load_tau():
    if not os.path.exists(MODEL_RESULTS):
        raise FileNotFoundError(f"Need: {MODEL_RESULTS}\nRun model/xsafety/irt.py first.")

    df = pd.read_csv(MODEL_RESULTS)

    # Normalise column names
    for col in ['prompt', 'prompt_id', 'item']:
        if col in df.columns:
            df.rename(columns={col: 'prompt'}, inplace=True)
            break
    df['prompt'] = df['prompt'].apply(clean_id)

    tau_col     = 'Safety_Tax' if 'Safety_Tax' in df.columns else 'tau'
    anchor_col  = 'Is_Anchor'  if 'Is_Anchor'  in df.columns else 'is_anchor'

    print(f"Loaded IRT results: {len(df)} rows")

    mask = (df['language'] != 'en')
    if anchor_col in df.columns:
        mask = mask & (~df[anchor_col])
    tau_long = df[mask].copy()
    tau_long = tau_long.rename(columns={'prompt': 'prompt_id', tau_col: 'tau'})
    tau_long = tau_long[tau_long['tau'] > 0]

    print(f"  Positive-τ pairs: {len(tau_long)}")
    print(f"  τ range: [{tau_long['tau'].min():.3f}, {tau_long['tau'].max():.3f}]")
    return tau_long


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Top pairs
# ══════════════════════════════════════════════════════════════════════════

def get_top_tau_pairs(tau_long, top_n=TOP_N):
    top = tau_long.sort_values('tau', ascending=False).head(top_n).copy()

    print(f"\nTop {top_n} highest positive τ (harder in non-English):")
    print(f"{'Prompt':>8}  {'Lang':<4}  {'τ':>8}")
    print("─" * 28)
    for _, r in top.iterrows():
        print(f"{r['prompt_id']:>8}  {r['language']:<4}  {r['tau']:>+8.3f}")
    return top


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Pull responses
# ══════════════════════════════════════════════════════════════════════════

def load_master_data():
    print("\nLoading XSafety dataset...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['id'] = df['id'].apply(clean_id)

    response_col = None
    for c in ['model_output', 'model_response', 'response', 'output',
              'response_text', 'completion', 'text']:
        if c in df.columns:
            response_col = c
            break
    if response_col is None:
        for col in df.columns:
            if df[col].dtype == object and col not in [
                    'id', 'language', 'test_taker', 'model', 'category',
                    'prompt', 'prompt_en']:
                if df[col].dropna().head(5).str.len().mean() > 50:
                    response_col = col
                    break

    # XSafety has prompt_en (English reference) and prompt (translated)
    prompt_col = 'prompt_en' if 'prompt_en' in df.columns else 'prompt'

    tt_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['family'] = df[tt_col].apply(get_model_family)
    df['judge_score'] = pd.to_numeric(df.get('judge_score', 0), errors='coerce')

    print(f"  {len(df):,} rows, response={response_col}, prompt={prompt_col}")
    return df, response_col, prompt_col, tt_col


def extract_responses(master_df, top_pairs, response_col, prompt_col, tt_col):
    records = []
    for _, pair in top_pairs.iterrows():
        pid, lang, tau_val = pair['prompt_id'], pair['language'], pair['tau']

        lang_resp = master_df[(master_df['id'] == pid) & (master_df['language'] == lang)]
        en_resp   = master_df[(master_df['id'] == pid) & (master_df['language'] == 'en')]

        # XSafety: get English prompt text from prompt_en column or language=='en' row
        prompt_text = ''
        if 'prompt_en' in master_df.columns:
            any_row = master_df[master_df['id'] == pid].head(1)
            if len(any_row):
                prompt_text = str(any_row['prompt_en'].iloc[0])
        elif prompt_col:
            any_row = master_df[(master_df['id'] == pid) & (master_df['language'] == 'en')].head(1)
            if len(any_row):
                prompt_text = str(any_row[prompt_col].iloc[0])

        for fam in FAM_ORDER:
            fam_rows = lang_resp[lang_resp['family'] == fam]
            if len(fam_rows) == 0:
                continue

            # Positive τ → prefer jailbreak example (score ≥ 4)
            unsafe = fam_rows[fam_rows['judge_score'] >= 4]
            row = unsafe.iloc[0] if len(unsafe) else fam_rows.iloc[0]

            en_fam = en_resp[en_resp['family'] == fam]
            en_text, en_judge = '', ''
            if len(en_fam):
                en_text  = truncate(en_fam.iloc[0][response_col]) if response_col else ''
                en_judge = en_fam.iloc[0].get('judge_score', '')

            records.append({
                'prompt_id':        pid,
                'language':         lang,
                'tau':              round(tau_val, 3),
                'family':           fam,
                'model':            str(row[tt_col])[:40],
                'prompt_text':      truncate(prompt_text, 200),
                'response_lang':    truncate(row[response_col]) if response_col else '',
                'judge_score_lang': row.get('judge_score', ''),
                'response_en':      en_text,
                'judge_score_en':   en_judge,
                'category':         row.get('category', ''),
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Output
# ══════════════════════════════════════════════════════════════════════════

def save_outputs(result, top_pairs):
    csv_path = os.path.join(RESULTS_DIR, "high_tau_responses.csv")
    result.to_csv(csv_path, index=False)
    print(f"\n  Saved: {os.path.basename(csv_path)} ({len(result)} rows)")

    md_lines = [
        "# High Positive-τ Qualitative Inspection — XSafety\n",
        "Positive τ = prompt is **harder/more dangerous** in non-English.\n",
        "Look for: refusal in English but compliance in target language.\n\n---\n\n"
    ]
    for _, pair in top_pairs.iterrows():
        pid, lang, tau = pair['prompt_id'], pair['language'], pair['tau']
        subset = result[(result['prompt_id'] == pid) & (result['language'] == lang)]
        if len(subset) == 0:
            continue

        cat    = subset.iloc[0].get('category', '?')
        prompt = subset.iloc[0].get('prompt_text', '?')
        md_lines.append(f"## Prompt {pid} × {lang}  (τ = +{tau:.3f}, {cat})\n\n")
        md_lines.append(f"**Prompt:** {prompt}\n\n")

        for _, row in subset.iterrows():
            md_lines.append(f"### {row['family']} ({row['model']})\n\n")
            md_lines.append(f"**{lang.upper()}** (judge: {row['judge_score_lang']}):\n")
            md_lines.append(f"> {row['response_lang']}\n\n")
            if row['response_en']:
                md_lines.append(f"**EN** (judge: {row['judge_score_en']}):\n")
                md_lines.append(f"> {row['response_en']}\n\n")
        md_lines.append("---\n\n")

    md_path = os.path.join(RESULTS_DIR, "high_tau_inspection.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    print(f"  Saved: {os.path.basename(md_path)}")

    result['js_lang'] = pd.to_numeric(result['judge_score_lang'], errors='coerce')
    result['js_en']   = pd.to_numeric(result['judge_score_en'],   errors='coerce')
    both = result.dropna(subset=['js_lang', 'js_en'])
    if len(both):
        ul = (both['js_lang'] >= 4).mean()
        ue = (both['js_en']   >= 4).mean()
        print(f"\nJailbreak rate — target lang: {ul:.1%}, English: {ue:.1%}, "
              f"Delta = {ul - ue:+.1%}")


def main():
    print("=" * 60)
    print("QUALITATIVE INSPECTION: High Positive-τ Responses — XSafety")
    print("=" * 60)

    tau_long  = load_tau()
    top_pairs = get_top_tau_pairs(tau_long)
    master_df, response_col, prompt_col, tt_col = load_master_data()

    if response_col is None:
        print("\nNo response text column found in XSafety dataset!")
        return

    result = extract_responses(master_df, top_pairs, response_col, prompt_col, tt_col)
    save_outputs(result, top_pairs)
    print(f"\n→ Open high_tau_inspection.md and READ the responses!")


if __name__ == "__main__":
    main()
