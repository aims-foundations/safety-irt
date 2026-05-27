# -*- coding: utf-8 -*-
"""
Anchor Sensitivity Ablation — γ/τ Multicollinearity Only
=========================================================
Refits the 2PL IRT model under seven anchor conditions and reports a single
statistic per condition: Pearson r(γ_L, mean_i τ_iL) across the nine non-English
languages.

Why this statistic? γ_L (global language shift) and τ_{iL} (item-level cross-
lingual safety gap) are both language-indexed. Without good anchors they trade
off — high |r| means the language-level signal is smeared across both
parameters, low |r| means they are cleanly separated. The paper's main fit
reports |r| = 0.081 under Lord's-χ²-selected anchors; this script tests whether
that low |r| is specific to our anchor procedure or a property of any
reasonable anchor set.

Conditions (τ-anchor treatment fixed across all: τ_iL ~ Normal(0, 0.01) for
anchor items across all non-English languages; non-anchor τ retains the
StudentT(1, 0, tau_scale) horseshoe prior; English column hard-zero via mask):

  lords_dif           — top 40 by mean Lord's χ² (= our published anchor set)
  lords_small         — top 20 by mean Lord's χ²            (nested ⊂ lords_dif)
  lords_large         — top 60 by mean Lord's χ²            (lords_dif ⊂)
  random_small        — 20 random prompts (no DIF screening)
  random_matched      — 40 random prompts (no DIF screening)
  category_balanced   — stratified sample from multijail.csv harm tags. For
                        each of 18 MultiJail categories, we take 2 prompts at
                        random; total ≈ 36 anchors (≤ if categories are
                        smaller than 2 prompts). Tests whether *content*-
                        balanced anchors (no DIF info) work as well as our
                        χ²-ranked set.
  iterative_purification — 57 prompts from anchors_lords_deprecated.csv (HF
                        snapshot). This is the result of classical Lord
                        iterative purification run *without* the 5–95% safety-
                        rate filter (with the filter, the candidate pool is
                        empty — see paper Sec. 3, "Traditional anchor
                        selection"). Included to verify that the standard
                        purification procedure produces a degenerate fit with
                        high γ–τ multicollinearity, motivating our heuristic.

Outputs (results_gamma_tau_multicollinearity/):
  anchor_conditions.csv             — anchor IDs per condition (reproducibility)
  gamma_tau_multicollinearity.csv   — one row per condition with Pearson r,
                                      |r|, p-value, n_anchors

Headline expectation: lords_dif (and its nested subsets) yields the lowest |r|;
random and iterative_purification yield much higher |r|, confirming our
selection criterion materially improves γ/τ identification.
"""

import os
import sys
import ast
import warnings
warnings.filterwarnings('ignore')

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from huggingface_hub import snapshot_download


# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="safety-irt/safety-data",
                                repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data",
                           "Master_Passes0-9_Dataset.csv")
MULTIJAIL_FILE = os.path.join(DATA_DIR, "multijail.csv")
ITER_PURIF_FILE = os.path.join(DATA_DIR, "anchors",
                                "anchors_lords_deprecated.csv")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
MODEL_DIR = os.path.join(REPO_ROOT, "model")
ANCHORS_OUT_DIR = os.path.join(MODEL_DIR, "results_dif_stratified")
DIF_SCORES_FILE = os.path.join(ANCHORS_OUT_DIR, "dif_agreement_scores.csv")

RESULTS_DIR = os.path.join(THIS_DIR, "results_gamma_tau_multicollinearity")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── config ───────────────────────────────────────────────────────────────────
MAX_STEPS    = 4000
CONV_WINDOW  = 200
CONV_THRESH  = 1e-4
MIN_STEPS    = 1000
N_SAMPLES    = 500
SEED         = 42
ANCHOR_PRIOR_SIGMA = 0.01

LORDS_DIF_N     = 40
LORDS_SMALL_N   = 20
LORDS_LARGE_N   = 60
RANDOM_SMALL_N  = 20
RANDOM_MATCHED_N = 40
CATBAL_PER_TAG  = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_tags_cell(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [str(t).strip() for t in out if str(t).strip()]
        except Exception:
            pass
    return [s]


def check_convergence(losses, window, threshold, min_steps):
    if len(losses) < min_steps or len(losses) < 2 * window:
        return False
    recent   = np.mean(losses[-window:])
    previous = np.mean(losses[-2*window:-window])
    if previous == 0:
        return True
    return (previous - recent) / abs(previous) < threshold


# ── 2PL model with soft anchor prior ────────────────────────────────────────

def model_2pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None, anchor_mask_tensor=None):
    """
    P(safe) = σ(α_i · ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))

    Anchor items: τ_{i,L} ~ Normal(0, 0.01) across all L (soft constraint).
    Non-anchor items: τ_{i,L} ~ StudentT(1, 0, tau_scale) (horseshoe-like).
    English column: hard zero via tau_mask (identification).
    """
    theta = pyro.sample("theta",
        dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta = pyro.sample("beta",
        dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))
    alpha = pyro.sample("alpha",
        dist.LogNormal(0.5 * torch.ones(num_prompts, device=device),
                       0.5 * torch.ones(num_prompts, device=device)).to_event(1))

    gamma_raw = pyro.sample("gamma_raw",
        dist.Normal(torch.zeros(num_langs, device=device), 1.0).to_event(1))
    gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)

    # τ: dual-sample. Anchor cells use tau_anchor (Normal), non-anchor cells
    # use tau_nonanchor (StudentT). Combine via anchor_mask_tensor.
    tau_anchor = pyro.sample("tau_anchor",
        dist.Normal(torch.zeros(num_prompts, num_langs, device=device),
                    ANCHOR_PRIOR_SIGMA).to_event(2))

    tau_scale = pyro.sample("tau_scale",
        dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_nonanchor = pyro.sample("tau_nonanchor",
        dist.StudentT(1.0,
                      torch.zeros(num_prompts, num_langs, device=device),
                      tau_scale).to_event(2))

    tau_combined = (anchor_mask_tensor * tau_anchor
                    + (1.0 - anchor_mask_tensor) * tau_nonanchor)
    tau = pyro.deterministic("tau", tau_combined * tau_mask)

    delta_raw = pyro.sample("delta_raw",
        dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5)
            .to_event(2))
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    with pyro.plate("data", len(student_idx)):
        ability    = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        logits     = alpha[prompt_idx] * (ability - difficulty)
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


# ── anchor set construction ──────────────────────────────────────────────────

def ensure_dif_scores():
    """Make sure dif_agreement_scores.csv exists; if not, run anchors.py."""
    if os.path.exists(DIF_SCORES_FILE):
        print(f"  Using existing ranked scores: {DIF_SCORES_FILE}")
        return pd.read_csv(DIF_SCORES_FILE)

    print(f"  {DIF_SCORES_FILE} not found — running anchors.py to generate...")
    sys.path.insert(0, MODEL_DIR)
    import anchors as _anchors_module
    _anchors_module.main()
    if not os.path.exists(DIF_SCORES_FILE):
        raise FileNotFoundError(
            f"anchors.py did not produce {DIF_SCORES_FILE}")
    return pd.read_csv(DIF_SCORES_FILE)


def build_anchor_sets(df_raw):
    scores_df = ensure_dif_scores()
    scores_df["prompt_id"] = scores_df["prompt_id"].apply(clean_id)
    scores_sorted = scores_df.sort_values("mean_chi2", ascending=True)

    lords_dif_ids   = set(scores_sorted.head(LORDS_DIF_N)["prompt_id"].tolist())
    lords_small_ids = set(scores_sorted.head(LORDS_SMALL_N)["prompt_id"].tolist())
    lords_large_ids = set(scores_sorted.head(LORDS_LARGE_N)["prompt_id"].tolist())

    # ── Random conditions ────────────────────────────────────────────────────
    all_prompts = sorted(df_raw["id"].unique())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_prompts))
    arr = np.array(all_prompts)

    random_small_ids   = set(arr[perm[:RANDOM_SMALL_N]].tolist())
    random_matched_ids = set(arr[perm[:RANDOM_MATCHED_N]].tolist())

    # ── Category-balanced from multijail.csv ─────────────────────────────────
    if os.path.exists(MULTIJAIL_FILE):
        mj = pd.read_csv(MULTIJAIL_FILE).drop_duplicates(subset=["id"])
        mj["id"] = mj["id"].apply(clean_id)
        mj["tags_parsed"] = mj["tags"].apply(parse_tags_cell)
        mj_long = (mj[["id", "tags_parsed"]].explode("tags_parsed")
                     .rename(columns={"tags_parsed": "tag"})
                     .dropna(subset=["tag"]))
        cat_ids = set()
        cat_rng = np.random.default_rng(SEED + 1)
        for tag, grp in mj_long.groupby("tag"):
            candidates = grp["id"].astype(str).tolist()
            candidates = [c for c in candidates if c in set(all_prompts)]
            if not candidates:
                continue
            n_sample = min(CATBAL_PER_TAG, len(candidates))
            picked = cat_rng.choice(candidates, size=n_sample, replace=False)
            cat_ids.update(picked.tolist())
        category_balanced_ids = cat_ids
        n_tags = mj_long["tag"].nunique()
        print(f"  category_balanced: {len(cat_ids)} anchors "
              f"({CATBAL_PER_TAG}/tag × {n_tags} tags = up to {CATBAL_PER_TAG*n_tags})")
    else:
        print(f"  [WARN] multijail.csv missing — skipping category_balanced")
        category_balanced_ids = set()

    # ── Iterative purification from deprecated CSV ───────────────────────────
    if os.path.exists(ITER_PURIF_FILE):
        iter_df = pd.read_csv(ITER_PURIF_FILE)
        iter_df["id"] = iter_df["id"].apply(clean_id)
        iterative_purification_ids = set(iter_df["id"].astype(str).unique())
        print(f"  iterative_purification: {len(iterative_purification_ids)} "
              f"anchors from anchors_lords_deprecated.csv (no 5-95% filter)")
    else:
        print(f"  [WARN] {ITER_PURIF_FILE} missing — skipping iterative_purification")
        iterative_purification_ids = set()

    conditions = {
        "lords_dif":              lords_dif_ids,
        "lords_small":            lords_small_ids,
        "lords_large":            lords_large_ids,
        "random_small":           random_small_ids,
        "random_matched":         random_matched_ids,
        "category_balanced":      category_balanced_ids,
        "iterative_purification": iterative_purification_ids,
    }

    print("\n  Anchor set sizes:")
    for name, ids in conditions.items():
        print(f"    {name:25s}: {len(ids):4d} prompts")

    return conditions


# ── fitting ──────────────────────────────────────────────────────────────────

def fit_condition(df, anchor_ids, cond_name):
    pyro.clear_param_store()
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)

    sc = "test_taker" if "test_taker" in df.columns else "model"
    students  = sorted(df[sc].unique())
    prompts   = sorted(df["id"].unique())
    languages = sorted(df["language"].unique())

    s_map = {s: i for i, s in enumerate(students)}
    p_map = {p: i for i, p in enumerate(prompts)}
    l_map = {l: i for i, l in enumerate(languages)}
    ns, np_, nl = len(students), len(prompts), len(languages)

    s_idx = torch.tensor(df[sc].map(s_map).values, dtype=torch.long).to(device)
    p_idx = torch.tensor(df["id"].map(p_map).values, dtype=torch.long).to(device)
    l_idx = torch.tensor(df["language"].map(l_map).values, dtype=torch.long).to(device)
    obs   = torch.tensor(df["score"].values, dtype=torch.float32).to(device)

    tau_mask   = torch.ones((np_, nl), device=device)
    gamma_mask = torch.ones(nl, device=device)
    if "en" in l_map:
        ei = l_map["en"]
        tau_mask[:, ei] = 0.0
        gamma_mask[ei]  = 0.0

    # Soft-anchor mask: 1 for anchor items (across all langs), 0 otherwise
    anchor_mask_tensor = torch.zeros((np_, nl), device=device)
    n_applied = 0
    for pid in prompts:
        if pid in anchor_ids:
            anchor_mask_tensor[p_map[pid], :] = 1.0
            n_applied += 1
    print(f"  [{cond_name}] {n_applied}/{len(anchor_ids)} anchors matched in data")

    guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_2pl,
                           hide=["obs", "tau", "gamma", "delta", "alpha"]))
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi = SVI(model_2pl, guide, optimizer, loss=Trace_ELBO())

    losses = []
    pbar = tqdm(range(MAX_STEPS), desc=f"[{cond_name}]", leave=False)
    converged_at = MAX_STEPS
    for step in pbar:
        loss = svi.step(s_idx, p_idx, l_idx, obs,
                        ns, np_, nl,
                        tau_mask, gamma_mask, anchor_mask_tensor)
        losses.append(loss)
        if step % 200 == 0:
            pbar.set_description(f"[{cond_name}] Loss: {loss:.1f}")
        if check_convergence(losses, CONV_WINDOW, CONV_THRESH, MIN_STEPS):
            converged_at = step + 1
            pbar.close()
            break
    print(f"  [{cond_name}] Converged at step {converged_at}")

    pred = Predictive(model_2pl, guide=guide, num_samples=N_SAMPLES,
                      return_sites=["gamma", "tau"])
    samps = pred(s_idx, p_idx, l_idx, None,
                 ns, np_, nl, tau_mask, gamma_mask, anchor_mask_tensor)

    gamma_mean = samps["gamma"].mean(0).detach().cpu().numpy().reshape(nl)
    tau_mean   = samps["tau"].mean(0).detach().cpu().numpy().reshape(np_, nl)

    return {
        "condition":      cond_name,
        "n_anchors":      n_applied,
        "converged_at":   converged_at,
        "gamma":          gamma_mean,
        "tau_mean":       tau_mean,
        "l_map":          l_map,
    }


# ── γ-τ multicollinearity statistic ─────────────────────────────────────────

def gamma_tau_correlation(res):
    """Pearson r(γ_L, mean_i τ_iL) across the 9 non-English languages."""
    l_map = res["l_map"]
    non_en = [(l, i) for l, i in l_map.items() if l != "en"]
    gammas = np.array([res["gamma"][i] for _, i in non_en])
    mean_taus = []
    for _, i in non_en:
        col = res["tau_mean"][:, i]
        col = col[np.abs(col) > 1e-6]  # skip the English-column zeros
        mean_taus.append(np.mean(col) if len(col) else 0.0)
    mean_taus = np.array(mean_taus)
    if len(gammas) < 3:
        return np.nan, np.nan
    r, p = pearsonr(gammas, mean_taus)
    return float(r), float(p)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ANCHOR SENSITIVITY ABLATION — γ/τ MULTICOLLINEARITY")
    print("=" * 70)

    print("\nLoading data...")
    df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["score"] = (df["judge_score"] >= 4).astype(np.float32)
    df["id"]    = df["id"].apply(clean_id)
    sc = "test_taker" if "test_taker" in df.columns else "model"
    print(f"  {len(df):,} rows | {df[sc].nunique()} models | "
          f"{df['id'].nunique()} prompts | {df['language'].nunique()} languages")

    print("\nBuilding anchor conditions...")
    anchor_sets = build_anchor_sets(df)

    # Save anchor IDs per condition
    cond_rows = []
    for cond, ids in anchor_sets.items():
        for pid in sorted(ids):
            cond_rows.append({"condition": cond, "prompt_id": pid})
    pd.DataFrame(cond_rows).to_csv(
        os.path.join(RESULTS_DIR, "anchor_conditions.csv"), index=False)

    # Fit each condition
    results = {}
    for cond_name, ids in anchor_sets.items():
        if not ids:
            print(f"\n[{cond_name}] empty anchor set — skipping")
            continue
        print(f"\n{'─' * 60}\nFitting: {cond_name}  (n={len(ids)})\n{'─' * 60}")
        results[cond_name] = fit_condition(df, ids, cond_name)

    # Compute γ-τ correlation
    print("\nComputing γ-τ correlations...")
    rows = []
    for cond, res in results.items():
        r, p = gamma_tau_correlation(res)
        rows.append({
            "condition":      cond,
            "n_anchors":      res["n_anchors"],
            "converged_at":   res["converged_at"],
            "pearson_r":      r,
            "abs_r":          abs(r),
            "p_value":        p,
        })

    summary = pd.DataFrame(rows).sort_values("abs_r", ascending=True)
    out_path = os.path.join(RESULTS_DIR, "gamma_tau_multicollinearity.csv")
    summary.to_csv(out_path, index=False)

    # ── Print final table ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("γ/τ MULTICOLLINEARITY BY ANCHOR CONDITION")
    print("(lower |r| = better γ/τ identification)")
    print(f"{'=' * 70}")
    print(f"  {'condition':<25} | {'n':>4} | {'r(γ, mean τ)':>13} | "
          f"{'|r|':>6} | {'p':>10}")
    print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*13}-+-{'-'*6}-+-{'-'*10}")
    for _, r in summary.iterrows():
        print(f"  {r['condition']:<25} | {int(r['n_anchors']):>4} | "
              f"{r['pearson_r']:>+13.3f} | {r['abs_r']:>6.3f} | "
              f"{r['p_value']:>10.2e}")
    print(f"\n  Saved: {out_path}")
    print(f"  Anchor IDs:  {os.path.join(RESULTS_DIR, 'anchor_conditions.csv')}")


if __name__ == "__main__":
    main()
