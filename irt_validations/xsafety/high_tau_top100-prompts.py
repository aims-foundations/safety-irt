# -*- coding: utf-8 -*-
"""Extract top 100 highest τ (positive = harder in non-English) pairs — XSafety.
Adapted from irt_validations/high_tau_top100-prompts.py:
  - Uses XSafety_Dataset.csv (no separate multijail.csv needed)
  - category column is a single string already in the dataset
  - prompt_en column available directly for English reference text
"""

import pandas as pd
import os
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)

INPUT_FILE = os.path.join(DATA_DIR, "safety-data", "xsafety", "xsafety_pass_graded.csv")

IRT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "model", "xsafety", "results",
                   "bayesian_irt_results_binary.csv")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_qualitative_inspection")
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


# Load τ
irt = pd.read_csv(IRT)
print(f"IRT columns: {list(irt.columns)}")

# Normalise prompt ID column name
for col in ['prompt', 'prompt_id', 'item']:
    if col in irt.columns:
        irt.rename(columns={col: 'prompt'}, inplace=True)
        break
irt['prompt'] = irt['prompt'].apply(clean_id)

# Detect anchor/language columns (XSafety IRT results may use different names)
is_anchor_col = 'Is_Anchor' if 'Is_Anchor' in irt.columns else 'is_anchor'
tau_col = 'Safety_Tax' if 'Safety_Tax' in irt.columns else 'tau'

mask_non_anchor = ~irt[is_anchor_col] if is_anchor_col in irt.columns else pd.Series(True, index=irt.index)
tau = irt[mask_non_anchor & (irt['language'] != 'en')].copy()
tau = tau[tau[tau_col] > 0]

want_cols = ['prompt', 'language', tau_col, 'Base_Difficulty', 'alpha']
have_cols = [c for c in want_cols if c in tau.columns]
top = tau.nlargest(100, tau_col)[have_cols].copy()

rename_map = {'prompt': 'id', tau_col: 'tau', 'Base_Difficulty': 'beta'}
top = top.rename(columns={k: v for k, v in rename_map.items() if k in top.columns})

# Get prompt text from XSafety dataset
master = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
master['id'] = master['id'].apply(clean_id)

# XSafety has prompt (translated) and prompt_en (English original) columns
has_prompt_en = 'prompt_en' in master.columns

if has_prompt_en:
    en_rows = master.drop_duplicates('id')[['id', 'prompt_en']].copy()
    en_rows = en_rows.rename(columns={'prompt_en': 'prompt_en'})
else:
    en_rows = master[master['language'] == 'en'].drop_duplicates('id')[
        ['id', 'prompt']].copy()
    en_rows = en_rows.rename(columns={'prompt': 'prompt_en'})

trans_rows = master.drop_duplicates(['id', 'language'])[
    ['id', 'language', 'prompt']].copy()
trans_rows = trans_rows.rename(columns={'prompt': 'prompt_translated'})

# Category is directly in XSafety dataset
if 'category' in master.columns:
    cats = master.drop_duplicates('id')[['id', 'category']]
    en_rows = en_rows.merge(cats, on='id', how='left')

top = top.merge(en_rows, on='id', how='left')
top = top.merge(trans_rows, on=['id', 'language'], how='left')
top = top.sort_values('tau', ascending=False).reset_index(drop=True)

path = os.path.join(RESULTS_DIR, "top100_high_tau_prompts.csv")
top.to_csv(path, index=False)

print(f"Saved: {path}")
print(f"\nTop 20 (positive τ = harder/more dangerous in non-English):")
for i, r in top.head(20).iterrows():
    cat = r.get('category', '?')
    txt = str(r.get('prompt_en', ''))[:80]
    beta = f"β={r['beta']:.2f}" if 'beta' in r.index else ""
    alpha = f"α={r['alpha']:.1f}" if 'alpha' in r.index else ""
    print(f"  {r['id']:>4} {r['language']:<3} τ={r['tau']:+.3f}  "
          f"{beta}  {alpha}  [{cat}] {txt}")
