"""
Phase 0 — extract real generative parameters from the safety-irt HF dataset.
Prints full descriptive stats for theta, delta, beta, gamma, and tau so the
magnitude / mean / SD judgments can be made by hand. No comparison to any
literature constant — just the numbers.

Colab: set your HF token first (cell below handles userdata / env / login).
"""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# ── config ──────────────────────────────────────────────────────────────────
REPO      = "aims-foundations/safety-irt"
REPO_TYPE = "dataset"
BASE      = "results/results"

FILES = {
    "results": f"{BASE}/bayesian_irt_results_binary.csv",  # beta, tau, gamma, alpha, Is_Anchor
    "theta":   f"{BASE}/theta_person_params.csv",
    "delta":   f"{BASE}/delta_person_params.csv",
    "gamma":   f"{BASE}/gamma_language_params.csv",
}

# ── token (optional — dataset resolves publicly, but use token if gated) ─────
TOKEN = None
try:
    from google.colab import userdata
    TOKEN = userdata.get("HF_TOKEN")
except Exception:
    import os
    TOKEN = os.environ.get("HF_TOKEN")
# If still None and the repo is gated for you, run: from huggingface_hub import login; login()


def load(key):
    path = hf_hub_download(repo_id=REPO, repo_type=REPO_TYPE,
                           filename=FILES[key], token=TOKEN)
    return pd.read_csv(path)


def describe(x, label):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    print(f"\n{label}  (n={len(x)})")
    print(f"  mean   {x.mean():+.4f}")
    print(f"  std    {x.std(ddof=1):.4f}")
    print(f"  median {np.median(x):+.4f}")
    print(f"  min    {x.min():+.4f}")
    print(f"  max    {x.max():+.4f}")
    return x


def quantile_block(x, label, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, dtype=float)
    print(f"\n{label} quantiles (n={len(x)})")
    for q in qs:
        print(f"  P{q:<2d}  {np.percentile(x, q):.4f}")


def mad(x):
    x = np.asarray(x, dtype=float)
    return np.median(np.abs(x - np.median(x)))


# ── load ────────────────────────────────────────────────────────────────────
res   = load("results")
theta = load("theta")
delta = load("delta")
gamma = load("gamma")

print("=" * 70)
print("SHAPES")
print(f"  results rows : {len(res)}  | unique prompts: {res['prompt'].nunique()}"
      f"  | languages: {sorted(res['language'].unique())}")
print(f"  theta rows   : {len(theta)}")
print(f"  delta rows   : {len(delta)}")
print(f"  gamma rows   : {len(gamma)}")

# robust Is_Anchor -> bool (handles bool or 'True'/'False' string)
res["_anchor"] = res["Is_Anchor"].astype(str).str.strip().str.lower().eq("true")

# ── THETA  (person ability; need mean + SD for Normal(mean, SD)) ─────────────
print("\n" + "=" * 70)
print("THETA  — draw theta_j ~ Normal(theta_mean, theta_SD)")
describe(theta["theta"], "theta (all 61 configs)")

# ── DELTA  (model x language; need SD for Normal(0, SD), en pinned to 0) ─────
print("\n" + "=" * 70)
print("DELTA  — draw delta_jL ~ Normal(0, delta_SD); en is structurally 0")
delta_non_en = delta[delta["language"] != "en"]["delta"]
describe(delta["delta"],      "delta (ALL rows, incl. en zeros — deflated, do not use)")
describe(delta_non_en,        "delta (non-en — USE THIS SD)")
print("\n  per-language non-en std (for reference, in case pooled is too coarse):")
for lang, g in delta[delta["language"] != "en"].groupby("language"):
    print(f"    {lang:<3} std {g['delta'].std(ddof=1):.4f}  mean {g['delta'].mean():+.4f}")

# ── BETA  (prompt difficulty; preserved — describe for sanity only) ─────────
print("\n" + "=" * 70)
print("BETA  — preserved as-is (Base_Difficulty, deduped per prompt)")
beta = res.drop_duplicates("prompt").set_index("prompt")["Base_Difficulty"]
describe(beta, "beta (315 unique prompts)")

# ── GAMMA  (language difficulty; preserved — print all) ─────────────────────
print("\n" + "=" * 70)
print("GAMMA  — preserved as-is (9 non-en, en=0)")
print(gamma.to_string(index=False))

# ── TAU  (the magnitude reference; free = non-anchor, non-en) ───────────────
print("\n" + "=" * 70)
print("TAU  — free cross-lingual gaps (Is_Anchor==False, language!='en')")
tau_free = res[(~res["_anchor"]) & (res["language"] != "en")]["Safety_Tax"]
tau_free = tau_free.dropna().astype(float)
abs_tau  = tau_free.abs()

describe(tau_free, "signed tau")
print(f"  MAD(signed)   {mad(tau_free):.4f}")
n_pos = (tau_free > 0).sum(); n_neg = (tau_free < 0).sum(); n_tot = len(tau_free)
print(f"  positive      {n_pos} ({100*n_pos/n_tot:.1f}%)")
print(f"  negative      {n_neg} ({100*n_neg/n_tot:.1f}%)")

describe(abs_tau, "|tau|  <-- m candidate is median of this")
print(f"  MAD(|tau|)    {mad(abs_tau):.4f}")
quantile_block(abs_tau, "|tau|")

# exceedance table: lets you read off DIF proportion at any threshold yourself
print("\n|tau| exceedance (fraction of free cells with |tau| >= t):")
for t in np.arange(0.1, 1.55, 0.1):
    frac = (abs_tau >= t).mean()
    print(f"  |tau| >= {t:.1f}   {frac:.4f}  ({int((abs_tau >= t).sum())} cells)")

# ── headline numbers for the decisions ──────────────────────────────────────
print("\n" + "=" * 70)
print("NUMBERS FOR PHASE 1 / PHASE 2 DECISIONS")
print(f"  theta_mean   = {theta['theta'].mean():+.4f}")
print(f"  theta_SD     = {theta['theta'].std(ddof=1):.4f}")
print(f"  delta_SD     = {delta_non_en.std(ddof=1):.4f}   (non-en)")
print(f"  median|tau|  = {abs_tau.median():.4f}   <-- candidate m")
print(f"  mean|tau|    = {abs_tau.mean():.4f}")
print(f"  P90 |tau|    = {np.percentile(abs_tau, 90):.4f}")
print(f"  P99 |tau|    = {np.percentile(abs_tau, 99):.4f}")
print(f"  max |tau|    = {abs_tau.max():.4f}")
print("=" * 70)