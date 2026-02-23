# -*- coding: utf-8 -*-
"""
Qualitative Inspection of High-τ Prompt×Language Pairs
=======================================================
Sang's suggestion: "Reading some of the model output manually
might help confirming some of the findings."

This script:
1. Loads τ (Safety_Tax) from model/results/bayesian_irt_results_binary.csv
2. Finds top 15 highest |τ| prompt×language pairs
3. Pulls actual model responses from Master dataset
4. Samples 1 response per family for each high-τ pair
5. Outputs a readable CSV + markdown table for manual reading

The goal: do the responses *look* different for high-τ pairs?
e.g., does a prompt that's easy in English but hard in Bengali
actually show models refusing in English but complying in Bengali?
"""

import pandas as pd
import numpy as np
import os
import re
import textwrap
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data",
                          "Master_Passes0-9_Dataset.csv")

# τ comes from the main IRT model output
MODEL_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "model", "results",
                             "bayesian_irt_results_binary.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_qualitative_inspection")
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_N = 15          # number of high-τ pairs to inspect
RESPONSES_PER_FAMILY = 1   # 1 response per family per pair
MAX_RESPONSE_LEN = 500     # truncate long responses for readability

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
    if len(text) > max_len:
        return text[:max_len] + '...'
    return text


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load τ estimates
# ══════════════════════════════════════════════════════════════════════════

def load_tau():
    """Load τ (Safety_Tax) from the main IRT model output."""
    if not os.path.exists(MODEL_RESULTS):
        raise FileNotFoundError(
            f"Need: {MODEL_RESULTS}\n"
            f"Run model/irt.py first.")

    df = pd.read_csv(MODEL_RESULTS)
    df['prompt'] = df['prompt'].apply(clean_id)
    print(f"Loaded IRT results: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Filter to non-anchor, non-English rows with nonzero τ
    tau_long = df[~df['Is_Anchor'] & (df['language'] != 'en')].copy()
    tau_long = tau_long.rename(columns={
        'prompt': 'prompt_id',
        'Safety_Tax': 'tau'
    })
    tau_long['abs_tau'] = tau_long['tau'].abs()

    # Drop near-zero
    tau_long = tau_long[tau_long['abs_tau'] > 0.01]

    print(f"  Non-anchor, non-English τ values: {len(tau_long)}")
    print(f"  Languages: {sorted(tau_long['language'].unique())}")
    print(f"  |τ| range: [{tau_long['abs_tau'].min():.3f}, "
          f"{tau_long['abs_tau'].max():.3f}]")

    return tau_long


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Find top high-|τ| pairs
# ══════════════════════════════════════════════════════════════════════════

def get_top_tau_pairs(tau_long, top_n=TOP_N):
    """Get the top N highest |τ| prompt×language pairs."""
    top = (tau_long
           .sort_values('abs_tau', ascending=False)
           .head(top_n)
           .copy())

    print(f"\nTop {top_n} highest |τ| prompt×language pairs:")
    print(f"{'Prompt':>8}  {'Lang':<4}  {'τ':>8}  {'|τ|':>8}")
    print("─" * 35)
    for _, r in top.iterrows():
        print(f"{r['prompt_id']:>8}  {r['language']:<4}  "
              f"{r['tau']:>8.3f}  {r['abs_tau']:>8.3f}")

    return top


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Pull actual model responses
# ══════════════════════════════════════════════════════════════════════════

def load_master_data():
    print("\nLoading master dataset...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['id'] = df['id'].apply(clean_id)

    # Discover response column
    response_col = None
    for candidate in ['model_output', 'model_response', 'response',
                      'output', 'response_text', 'completion', 'text']:
        if candidate in df.columns:
            response_col = candidate
            break

    # If no obvious response column, look for any text-like column
    if response_col is None:
        for col in df.columns:
            if df[col].dtype == object and col not in ['id', 'language',
                'test_taker', 'model', 'category', 'prompt']:
                sample = df[col].dropna().head(5)
                if sample.str.len().mean() > 50:
                    response_col = col
                    break

    print(f"  {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Response column: {response_col}")

    # Also find prompt text column
    prompt_col = None
    for candidate in ['prompt', 'prompt_text', 'question', 'input']:
        if candidate in df.columns:
            prompt_col = candidate
            break

    print(f"  Prompt column: {prompt_col}")

    # Add family
    tt_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['family'] = df[tt_col].apply(get_model_family)
    df['judge_score'] = pd.to_numeric(df.get('judge_score', 0),
                                       errors='coerce')

    return df, response_col, prompt_col, tt_col


def extract_responses(master_df, top_pairs, response_col, prompt_col, tt_col):
    """
    For each high-τ pair, pull one response from each family.
    Also pull the English version of the same prompt for comparison.
    """
    records = []

    for _, pair in top_pairs.iterrows():
        pid = pair['prompt_id']
        lang = pair['language']
        tau_val = pair['tau']

        # Get responses for this prompt in this language
        mask_lang = (master_df['id'] == pid) & (master_df['language'] == lang)
        lang_responses = master_df[mask_lang].copy()

        # Get English responses for same prompt (for comparison)
        mask_en = (master_df['id'] == pid) & (master_df['language'] == 'en')
        en_responses = master_df[mask_en].copy()

        # Get prompt text (from any row)
        prompt_text = ''
        if prompt_col:
            any_row = master_df[master_df['id'] == pid].head(1)
            if len(any_row) > 0:
                prompt_text = str(any_row[prompt_col].iloc[0])

        # Sample one response per family in the target language
        for fam in FAM_ORDER:
            fam_rows = lang_responses[lang_responses['family'] == fam]
            if len(fam_rows) == 0:
                continue

            # Pick one: prefer a "jailbreak success" (score >= 4) if τ > 0
            # (high τ = harder in this language = more failures)
            if tau_val > 0:
                # Positive τ: this lang is harder. Look for a failure.
                unsafe = fam_rows[fam_rows['judge_score'] >= 4]
                row = unsafe.iloc[0] if len(unsafe) else fam_rows.iloc[0]
            else:
                # Negative τ: this lang is easier. Look for a refusal.
                safe = fam_rows[fam_rows['judge_score'] < 4]
                row = safe.iloc[0] if len(safe) else fam_rows.iloc[0]

            response_text = truncate(row[response_col]) if response_col else '[no response column found]'
            judge = row.get('judge_score', '')

            # Also get English response from same family
            en_fam = en_responses[en_responses['family'] == fam]
            en_text = ''
            en_judge = ''
            if len(en_fam) > 0:
                en_row = en_fam.iloc[0]
                en_text = truncate(en_row[response_col]) if response_col else ''
                en_judge = en_row.get('judge_score', '')

            records.append({
                'prompt_id': pid,
                'language': lang,
                'tau': round(tau_val, 3),
                'family': fam,
                'model': str(row[tt_col])[:40],
                'prompt_text': truncate(prompt_text, 200),
                'response_lang': response_text,
                'judge_score_lang': judge,
                'response_en': en_text,
                'judge_score_en': en_judge,
                'category': row.get('category', ''),
            })

    result = pd.DataFrame(records)
    return result


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Output
# ══════════════════════════════════════════════════════════════════════════

def save_outputs(result, top_pairs):
    # Full CSV
    csv_path = os.path.join(RESULTS_DIR, "high_tau_responses.csv")
    result.to_csv(csv_path, index=False)
    print(f"\n  Saved: {os.path.basename(csv_path)} ({len(result)} rows)")

    # Readable markdown
    md_lines = [
        "# High-τ Prompt Qualitative Inspection\n",
        "For each high-|τ| prompt×language pair, we show one response per "
        "model family in the target language vs English.\n",
        "**τ > 0**: prompt is *harder* in this language than average "
        "(more jailbreaks expected)\n",
        "**τ < 0**: prompt is *easier* in this language than average\n\n",
        "---\n\n"
    ]

    for _, pair in top_pairs.iterrows():
        pid = pair['prompt_id']
        lang = pair['language']
        tau = pair['tau']

        subset = result[(result['prompt_id'] == pid) &
                        (result['language'] == lang)]
        if len(subset) == 0:
            continue

        category = subset.iloc[0].get('category', '?')
        prompt = subset.iloc[0].get('prompt_text', '?')

        md_lines.append(f"## Prompt {pid} × {lang}  "
                        f"(τ = {tau:.3f}, category: {category})\n\n")
        md_lines.append(f"**Prompt:** {prompt}\n\n")

        for _, row in subset.iterrows():
            fam = row['family']
            md_lines.append(f"### {fam} ({row['model']})\n\n")

            # Target language response
            md_lines.append(f"**{lang.upper()} response** "
                            f"(judge: {row['judge_score_lang']}):\n")
            md_lines.append(f"> {row['response_lang']}\n\n")

            # English response
            if row['response_en']:
                md_lines.append(f"**EN response** "
                                f"(judge: {row['judge_score_en']}):\n")
                md_lines.append(f"> {row['response_en']}\n\n")

        md_lines.append("---\n\n")

    md_path = os.path.join(RESULTS_DIR, "high_tau_inspection.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    print(f"  Saved: {os.path.basename(md_path)}")

    # Summary stats
    print(f"\n{'=' * 60}")
    print("SUMMARY FOR MANUAL INSPECTION")
    print(f"{'=' * 60}")
    print(f"Total high-τ pairs inspected: {len(top_pairs)}")
    print(f"Total response samples: {len(result)}")
    print(f"Families represented: {sorted(result['family'].unique())}")
    print(f"Languages: {sorted(result['language'].unique())}")

    # Quick pattern check: do judge scores differ between lang and en?
    result['js_lang'] = pd.to_numeric(result['judge_score_lang'],
                                       errors='coerce')
    result['js_en'] = pd.to_numeric(result['judge_score_en'],
                                     errors='coerce')

    both = result.dropna(subset=['js_lang', 'js_en'])
    if len(both) > 0:
        unsafe_lang = (both['js_lang'] >= 4).mean()
        unsafe_en = (both['js_en'] >= 4).mean()
        print(f"\nJailbreak rate in target language: {unsafe_lang:.1%}")
        print(f"Jailbreak rate in English:         {unsafe_en:.1%}")
        print(f"Difference:                        "
              f"{unsafe_lang - unsafe_en:+.1%}")

        if unsafe_lang > unsafe_en:
            print("→ Consistent with positive τ: models fail more "
                  "in these languages")
        else:
            print("→ Mixed pattern — inspect individual cases")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("QUALITATIVE INSPECTION: High-τ Responses")
    print("=" * 60)

    # Load τ
    tau_long = load_tau()
    top_pairs = get_top_tau_pairs(tau_long)

    # Load raw responses
    master_df, response_col, prompt_col, tt_col = load_master_data()

    if response_col is None:
        print("\n⚠️  No response text column found in dataset!")
        print("Available columns:", list(master_df.columns))
        print("\nFalling back to judge_score only analysis...")

        # Even without responses, show judge scores per family
        records = []
        for _, pair in top_pairs.iterrows():
            pid = pair['prompt_id']
            lang = pair['language']
            mask = (master_df['id'] == pid) & (master_df['language'] == lang)
            for fam in FAM_ORDER:
                fam_data = master_df[mask & (master_df['family'] == fam)]
                if len(fam_data) > 0:
                    records.append({
                        'prompt_id': pid,
                        'language': lang,
                        'tau': round(pair['tau'], 3),
                        'family': fam,
                        'n_responses': len(fam_data),
                        'mean_judge_score': fam_data['judge_score'].mean(),
                        'pct_unsafe': (fam_data['judge_score'] >= 4).mean(),
                    })
        fallback = pd.DataFrame(records)
        path = os.path.join(RESULTS_DIR, "high_tau_judge_scores.csv")
        fallback.to_csv(path, index=False)
        print(f"\n  Saved: {os.path.basename(path)}")
        print(fallback.to_string(index=False))
        return

    # Extract responses
    result = extract_responses(master_df, top_pairs, response_col,
                               prompt_col, tt_col)

    # Save
    save_outputs(result, top_pairs)

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")

    print("\n→ Now open high_tau_inspection.md and READ the responses!")
    print("  Look for: refusal in English but compliance in target language")
    print("  (or vice versa for negative τ)")


if __name__ == "__main__":
    main()