# -*- coding: utf-8 -*-
"""Extract top 100 highest τ (positive = harder in non-English) pairs."""

import pandas as pd
import os
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
MASTER = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
MULTIJAIL = os.path.join(DATA_DIR, "multijail.csv")
IRT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "model", "results", "bayesian_irt_results_binary.csv")

def clean_id(x):
    try: return str(int(float(x)))
    except: return str(x).strip()

# Load τ
irt = pd.read_csv(IRT)
print(f"IRT columns: {list(irt.columns)}")
irt['prompt'] = irt['prompt'].apply(clean_id)
tau = irt[~irt['Is_Anchor'] & (irt['language'] != 'en')].copy()

# ── POSITIVE τ ONLY: harder/more dangerous in non-English ──
tau = tau[tau['Safety_Tax'] > 0]
want_cols = ['prompt', 'language', 'Safety_Tax', 'Base_Difficulty', 'alpha']
have_cols = [c for c in want_cols if c in tau.columns]
top = tau.nlargest(100, 'Safety_Tax')[have_cols].copy()
rename_map = {'prompt': 'id', 'Safety_Tax': 'tau', 'Base_Difficulty': 'beta'}
top = top.rename(columns={k: v for k, v in rename_map.items() if k in top.columns})

# Get prompt text from master (one per id)
master = pd.read_csv(MASTER, engine='python', on_bad_lines='skip')
master['id'] = master['id'].apply(clean_id)

prompt_col = None
for c in ['prompt', 'prompt_text', 'question', 'input']:
    if c in master.columns:
        prompt_col = c
        break

if prompt_col:
    en_rows = master[master['language'] == 'en'].drop_duplicates('id')[
        ['id', prompt_col]].copy()
    en_rows = en_rows.rename(columns={prompt_col: 'prompt_en'})

    trans_rows = master.drop_duplicates(['id', 'language'])[
        ['id', 'language', prompt_col]].copy()
    trans_rows = trans_rows.rename(columns={prompt_col: 'prompt_translated'})

    if 'category' in master.columns:
        cats = master.drop_duplicates('id')[['id', 'category']]
        en_rows = en_rows.merge(cats, on='id', how='left')

    top = top.merge(en_rows, on='id', how='left')
    top = top.merge(trans_rows, on=['id', 'language'], how='left')

# Add tags from multijail
mj_found = False
for mj_path in [MULTIJAIL,
                os.path.join(DATA_DIR, "raw_data", "multijail.csv"),
                os.path.join(DATA_DIR, "processed_data", "multijail.csv")]:
    if os.path.exists(mj_path):
        mj = pd.read_csv(mj_path)
        mj['id'] = mj['id'].apply(clean_id)
        tags = mj[mj['language'] == 'en'].drop_duplicates('id')[
            ['id', 'tags']].copy()
        top = top.merge(tags, on='id', how='left')
        print(f"Added tags from {mj_path}")
        mj_found = True
        break
if not mj_found:
    print("multijail.csv not found — tags column will be missing")

top = top.sort_values('tau', ascending=False).reset_index(drop=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_qualitative_inspection")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "top100_high_tau_prompts.csv")
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
