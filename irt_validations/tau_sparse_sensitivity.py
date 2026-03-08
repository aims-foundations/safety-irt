# -*- coding: utf-8 -*-
"""
Horseshoe Sensitivity & γ-τ Multicollinearity Analysis
========================================================
Two questions, one script:

1. HORSESHOE SENSITIVITY
   - Refit with Normal prior on τ (no sparsity) and StudentT(df=5)
   - Compare γ and τ estimates against the original StudentT(df=1) / Horseshoe
   - Does the Horseshoe actually matter? Do γ estimates change?

2. MULTICOLLINEARITY: γ vs τ
   - Compute cor(γ_L, mean_i(τ_iL)) across languages
   - If high: γ and τ are confounded
   - Show the Horseshoe mitigates this by shrinking τ toward 0

Reads:
   - Master_Passes0-9_Dataset.csv (raw data)
   - anchors.csv

Produces:
   - horseshoe_sensitivity.png — γ comparison across priors
   - tau_prior_comparison.png — τ distribution under each prior
   - gamma_tau_collinearity.png — scatter of γ vs mean(τ) per language
   - sensitivity_summary.csv — numerical comparison table
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# ── fig_style integration ──
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, COLORS_3, CMAP_DIV, CMAP_SEQ,
                           FAM_COLORS as FS_FAM_COLORS, FAM_ORDER as FS_FAM_ORDER,
                           LABELS, LANG_ORDER, FULL_WIDTH, DPI, ASPECT,
                           get_family, get_family_color, add_identity_line)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    print("[WARN] fig_style.py not found - using defaults")

# ── Color Palette Configuration ──
_c1 = C_BLUE if _HAS_FIG_STYLE else '#2471a3'   # Normal (Baseline)
_c2 = C_RED if _HAS_FIG_STYLE else '#c0392b'    # Horseshoe (Target)
_c3 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98' # Moderate (Intermediate)

PRIOR_COLORS = {'horseshoe': _c2, 'moderate': _c3, 'normal': _c1}


from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os
import re
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data",
                           "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors_majority.csv")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_horseshoe_sensitivity2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Priors to compare:
#   "horseshoe"  = StudentT(df=1)   ← your current model (Cauchy-like)
#   "moderate"   = StudentT(df=5)   ← less sparse
#   "normal"     = Normal(0, 1)     ← no sparsity at all
PRIOR_CONFIGS = {
    'horseshoe': {'dist': 'studentt', 'df': 1.0},
    'normal':    {'dist': 'normal',   'df': None},
}

MAX_STEPS = 6000
CONV_WINDOW = 200
CONV_THRESH = 1e-4
MIN_STEPS = 1500
N_SAMPLES = 300
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def check_convergence(losses, window, threshold, min_steps):
    if len(losses) < min_steps or len(losses) < 2 * window:
        return False
    recent = np.mean(losses[-window:])
    previous = np.mean(losses[-2*window:-window])
    if previous == 0:
        return True
    return (previous - recent) / abs(previous) < threshold


# ══════════════════════════════════════════════════════════════════════════
# MODEL VARIANTS
# ══════════════════════════════════════════════════════════════════════════

def make_model(tau_prior='studentt', tau_df=1.0):
    """
    Returns a 2PL model function with the specified τ prior.
    Everything else identical to your main model.
    """
    def model_fn(student_idx, prompt_idx, lang_idx, obs=None,
                 num_students=None, num_prompts=None, num_langs=None,
                 tau_mask=None, gamma_mask=None):

        theta = pyro.sample("theta",
            dist.Normal(torch.zeros(num_students, device=device),
                        1.0).to_event(1))
        beta = pyro.sample("beta",
            dist.Normal(torch.zeros(num_prompts, device=device),
                        1.0).to_event(1))
        log_alpha = pyro.sample("log_alpha",
            dist.Normal(torch.zeros(num_prompts, device=device),
                        0.5).to_event(1))
        alpha = pyro.deterministic("alpha", torch.exp(log_alpha))

        gamma_raw = pyro.sample("gamma_raw",
            dist.Normal(torch.zeros(num_langs, device=device),
                        1.0).to_event(1))
        gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)

        # ── τ prior: this is what changes ──
        if tau_prior == 'studentt':
            tau_scale = pyro.sample("tau_scale",
                dist.HalfCauchy(
                    torch.ones(1, device=device)).to_event(1))
            tau_raw = pyro.sample("tau_raw",
                dist.StudentT(
                    tau_df,
                    torch.zeros(num_prompts, num_langs, device=device),
                    tau_scale).to_event(2))
        else:  # normal
            tau_raw = pyro.sample("tau_raw",
                dist.Normal(
                    torch.zeros(num_prompts, num_langs, device=device),
                    1.0).to_event(2))

        tau = pyro.deterministic("tau", tau_raw * tau_mask)

        delta_raw = pyro.sample("delta_raw",
            dist.Normal(torch.zeros(num_students, num_langs, device=device),
                        0.5).to_event(2))
        delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
        delta = pyro.deterministic("delta", delta_raw * delta_mask)

        with pyro.plate("data", len(student_idx)):
            ability = theta[student_idx] + delta[student_idx, lang_idx]
            difficulty = (beta[prompt_idx] + gamma[lang_idx]
                          + tau[prompt_idx, lang_idx])
            logits = alpha[prompt_idx] * (ability - difficulty)
            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)

    return model_fn


# ══════════════════════════════════════════════════════════════════════════
# FITTING
# ══════════════════════════════════════════════════════════════════════════

def fit_variant(df, anchor_ids, prior_name, prior_cfg):
    """Fit one model variant and return extracted parameters."""
    pyro.clear_param_store()

    sc = 'test_taker' if 'test_taker' in df.columns else 'model'
    students  = sorted(df[sc].unique())
    prompts   = sorted(df['id'].unique())
    languages = sorted(df['language'].unique())

    s_map = {s: i for i, s in enumerate(students)}
    p_map = {p: i for i, p in enumerate(prompts)}
    l_map = {l: i for i, l in enumerate(languages)}
    ns, np_, nl = len(students), len(prompts), len(languages)

    s_idx = torch.tensor(df[sc].map(s_map).values,
                         dtype=torch.long).to(device)
    p_idx = torch.tensor(df['id'].map(p_map).values,
                         dtype=torch.long).to(device)
    l_idx = torch.tensor(df['language'].map(l_map).values,
                         dtype=torch.long).to(device)
    obs   = torch.tensor(df['score'].values,
                         dtype=torch.float32).to(device)

    tau_mask   = torch.ones((np_, nl), device=device)
    gamma_mask = torch.ones(nl, device=device)
    if 'en' in l_map:
        ei = l_map['en']
        tau_mask[:, ei]  = 0.0
        gamma_mask[ei]   = 0.0

    if os.path.exists(ANCHOR_FILE):
        adf = pd.read_csv(ANCHOR_FILE)
        adf['id'] = adf['id'].apply(clean_id)
        print(f"Anchors loaded: {len(adf['id'].unique())}")
        print(f"Source: {ANCHOR_FILE}")
        for pid in adf['id'].unique():
            if pid in p_map:
                tau_mask[p_map[pid], :] = 0.0

    model_fn = make_model(prior_cfg['dist'], prior_cfg.get('df', 1.0))

    # Determine which sites to hide from guide
    hide = ["obs", "tau", "gamma", "delta", "alpha"]
    if prior_cfg['dist'] == 'normal':
        # No tau_scale parameter when using Normal prior
        pass
    # tau_scale is sampled inside model but AutoNormal should handle it

    guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_fn, hide=hide))
    optimizer = ClippedAdam({"lr": 0.005, "clip_norm": 10.0})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    losses = []
    pbar = tqdm(range(MAX_STEPS), desc=f"[{prior_name}]", leave=False)
    converged_at = MAX_STEPS

    for step in pbar:
        loss = svi.step(s_idx, p_idx, l_idx, obs,
                        ns, np_, nl, tau_mask, gamma_mask)
        losses.append(loss)
        if step % 200 == 0:
            pbar.set_description(f"[{prior_name}] Loss: {loss:.1f}")
        if check_convergence(losses, CONV_WINDOW, CONV_THRESH, MIN_STEPS):
            converged_at = step + 1
            pbar.close()
            break

    print(f"  [{prior_name}] Converged at step {converged_at}")

    # Extract
    return_sites = ["gamma", "tau", "alpha"]
    pred = Predictive(model_fn, guide=guide, num_samples=N_SAMPLES,
                      return_sites=return_sites)
    samps = pred(s_idx, p_idx, l_idx, None,
                 ns, np_, nl, tau_mask, gamma_mask)

    gamma_mean = (samps['gamma'].detach().cpu().numpy()
                  .mean(0).reshape(nl).astype(np.float64))
    tau_mean   = (samps['tau'].detach().cpu().numpy()
                  .mean(0).reshape(np_, nl).astype(np.float64))
    tau_std    = (samps['tau'].detach().cpu().numpy()
                  .std(0).reshape(np_, nl).astype(np.float64))

    return {
        'gamma': gamma_mean,
        'tau_mean': tau_mean,
        'tau_std': tau_std,
        'lang_map': l_map,
        'prompt_map': p_map,
        'losses': losses,
        'converged_at': converged_at,
    }


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['id'] = df['id'].apply(clean_id)
    print(f"  {len(df):,} rows, {df['language'].nunique()} languages")

    anchor_ids = set()
    if os.path.exists(ANCHOR_FILE):
        adf = pd.read_csv(ANCHOR_FILE)
        adf['id'] = adf['id'].apply(clean_id)
        anchor_ids = set(adf['id'].unique())
    return df, anchor_ids


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def run_analysis(df, anchor_ids):
    results = {}

    for name, cfg in PRIOR_CONFIGS.items():
        print(f"\n{'─' * 50}")
        df_val = cfg.get('df')
        df_str = f", df={df_val}" if df_val else ""
        print(f"Fitting: {name}  (τ prior = {cfg['dist']}{df_str})")
        print(f"{'─' * 50}")
        results[name] = fit_variant(df, anchor_ids, name, cfg)

    return results


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — γ Comparison Across Priors
# ══════════════════════════════════════════════════════════════════════════

def plot_gamma_comparison(results):
    """Do γ estimates change when we remove the sparsity prior?"""
    ref = results['horseshoe']
    l_map = ref['lang_map']
    langs = sorted(l_map.keys(), key=lambda x: l_map[x])
    non_en = [l for l in langs if l != 'en']

    # Increased height (was 14, 7 -> now 14, 8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='tight')

    # Left: grouped bar chart of γ by language
    ax = axes[0]
    x = np.arange(len(non_en))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        gamma_vals = [res['gamma'][l_map[l]] for l in non_en]
        ax.bar(x + i * width, gamma_vals, width, label=name,
               color=PRIOR_COLORS[name], edgecolor='black', linewidth=0.4)

    ax.set_xticks(x + width)
    ax.set_xticklabels(non_en, fontsize=11)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('γ_L (language difficulty shift)', fontsize=11)
    ax.set_title('γ Estimates Under Different τ Priors', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)

    # Right: scatter of horseshoe γ vs normal γ
    ax = axes[1]
    hs_gamma = np.array([ref['gamma'][l_map[l]] for l in non_en])
    nm_gamma = np.array([results['normal']['gamma'][l_map[l]] for l in non_en])

    ax.scatter(hs_gamma, nm_gamma, s=90, color=_c1,
               edgecolors='black', linewidths=0.5, zorder=3)
    for j, lang in enumerate(non_en):
        ax.annotate(lang, (hs_gamma[j], nm_gamma[j]),
                    fontsize=10, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points')

    lims = [min(hs_gamma.min(), nm_gamma.min()) - 0.1,
            max(hs_gamma.max(), nm_gamma.max()) + 0.1]
    ax.plot(lims, lims, color=_c2, linestyle='--', alpha=0.7, label='Identity')

    if len(non_en) >= 3:
        r, p = pearsonr(hs_gamma, nm_gamma)
        ax.set_title(f'γ: Horseshoe vs Normal Prior\n'
                     f'r = {r:.3f}, p = {p:.3e}', fontweight='bold', fontsize=12)
    ax.set_xlabel('γ (Horseshoe / StudentT df=1)', fontsize=11)
    ax.set_ylabel('γ (Normal prior)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    plt.suptitle('Horseshoe Sensitivity: Effect on γ Estimates',
                 fontsize=14, fontweight='bold')
                 
    path = os.path.join(RESULTS_DIR, "horseshoe_gamma_sensitivity.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — τ Distribution Under Each Prior
# ══════════════════════════════════════════════════════════════════════════

def plot_tau_comparison(results):
    """Show that Horseshoe shrinks τ toward 0 more aggressively."""
    # Increased height (was 18, 7 -> now 18, 8.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), layout='tight')

    # Left: overlaid histograms of all τ values
    ax = axes[0]
    for name, res in results.items():
        tau_flat = res['tau_mean'].flatten()
        tau_nonzero = tau_flat[np.abs(tau_flat) > 1e-6]  # exclude masked
        ax.hist(tau_nonzero, bins=80, alpha=0.5, color=PRIOR_COLORS[name],
                label=f'{name} (std={np.std(tau_nonzero):.3f})',
                density=True, edgecolor='none')

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('τ_iL value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('τ Distribution by Prior', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)

    # Middle: % of τ near zero (sparsity check)
    ax = axes[1]
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    for name, res in results.items():
        tau_flat = res['tau_mean'].flatten()
        tau_nonzero = tau_flat[np.abs(tau_flat) > 1e-6]
        pcts = [(np.abs(tau_nonzero) < t).mean() * 100 for t in thresholds]
        ax.plot(thresholds, pcts, 'o-', color=PRIOR_COLORS[name],
                label=name, linewidth=2.5, markersize=7)

    ax.set_xlabel('|τ| threshold', fontsize=11)
    ax.set_ylabel('% of τ values below threshold', fontsize=11)
    ax.set_title('Sparsity: % τ Near Zero', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Right: per-language mean |τ| comparison
    ax = axes[2]
    ref = results['horseshoe']
    l_map = ref['lang_map']
    non_en = [l for l in sorted(l_map.keys(), key=lambda x: l_map[x])
              if l != 'en']

    x = np.arange(len(non_en))
    width = 0.25
    for i, (name, res) in enumerate(results.items()):
        mean_abs_tau = []
        for lang in non_en:
            li = l_map[lang]
            col = res['tau_mean'][:, li]
            col = col[np.abs(col) > 1e-6]
            mean_abs_tau.append(np.mean(np.abs(col)) if len(col) else 0)
        ax.bar(x + i * width, mean_abs_tau, width,
               color=PRIOR_COLORS[name], label=name,
               edgecolor='black', linewidth=0.4)

    ax.set_xticks(x + width)
    ax.set_xticklabels(non_en, fontsize=11)
    ax.set_ylabel('Mean |τ_iL|', fontsize=11)
    ax.set_title('τ Magnitude by Language', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Horseshoe Sensitivity: Effect on τ Estimates',
                 fontsize=14, fontweight='bold')
                 
    path = os.path.join(RESULTS_DIR, "horseshoe_tau_sensitivity.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — γ vs mean(τ) Multicollinearity
# ══════════════════════════════════════════════════════════════════════════

def plot_gamma_tau_collinearity(results):
    """
    If γ_L and mean_i(τ_iL) are highly correlated, they're confounded.
    The Horseshoe should reduce this correlation by shrinking small τ → 0.
    """
    # Decreased height to fit nicely on 1 line (was 6*len, 7 -> now 16, 5.5)
    fig, axes = plt.subplots(1, len(results), figsize=(16, 5.5),
                             squeeze=False, layout='tight')

    for col_idx, (name, res) in enumerate(results.items()):
        ax = axes[0][col_idx]
        l_map = res['lang_map']
        non_en = [l for l in sorted(l_map.keys(), key=lambda x: l_map[x])
                  if l != 'en']

        gammas = []
        mean_taus = []
        for lang in non_en:
            li = l_map[lang]
            gammas.append(res['gamma'][li])
            tau_col = res['tau_mean'][:, li]
            tau_col = tau_col[np.abs(tau_col) > 1e-6]
            mean_taus.append(np.mean(tau_col) if len(tau_col) else 0)

        gammas = np.array(gammas)
        mean_taus = np.array(mean_taus)

        ax.scatter(gammas, mean_taus, s=90, color=PRIOR_COLORS[name],
                   edgecolors='black', linewidths=0.5, zorder=3)

        for j, lang in enumerate(non_en):
            ax.annotate(lang, (gammas[j], mean_taus[j]),
                        fontsize=10, ha='left', va='bottom',
                        xytext=(5, 5), textcoords='offset points')

        if len(non_en) >= 3:
            r, p = pearsonr(gammas, mean_taus)
            ax.set_title(f'{name.capitalize()}\nr(γ, mean τ) = {r:.3f} (p={p:.3f})',
                         fontweight='bold', fontsize=12)
        else:
            ax.set_title(name.capitalize(), fontweight='bold', fontsize=12)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel('γ_L', fontsize=11)
        ax.set_ylabel('mean_i(τ_iL)', fontsize=11)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Multicollinearity Check: γ_L vs mean(τ_iL)\n'
                 'Lower |r| under Horseshoe = better identification',
                 fontsize=14, fontweight='bold', y=1.05)
                 
    path = os.path.join(RESULTS_DIR, "gamma_tau_collinearity.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")
    
# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════

def make_summary(results):
    ref = results['horseshoe']
    l_map = ref['lang_map']
    non_en = [l for l in sorted(l_map.keys(), key=lambda x: l_map[x])
              if l != 'en']

    rows = []
    for name, res in results.items():
        tau_flat = res['tau_mean'].flatten()
        tau_nz = tau_flat[np.abs(tau_flat) > 1e-6]

        # γ-τ correlation
        gammas = np.array([res['gamma'][l_map[l]] for l in non_en])
        mean_taus = []
        for lang in non_en:
            li = l_map[lang]
            tc = res['tau_mean'][:, li]
            tc = tc[np.abs(tc) > 1e-6]
            mean_taus.append(np.mean(tc) if len(tc) else 0)
        mean_taus = np.array(mean_taus)

        r_gt = pearsonr(gammas, mean_taus)[0] if len(non_en) >= 3 else np.nan

        # γ stability: correlation with horseshoe
        if name != 'horseshoe':
            hs_g = np.array([ref['gamma'][l_map[l]] for l in non_en])
            this_g = np.array([res['gamma'][l_map[l]] for l in non_en])
            r_gamma = pearsonr(hs_g, this_g)[0]
        else:
            r_gamma = 1.0

        rows.append({
            'prior': name,
            'tau_mean': np.mean(tau_nz),
            'tau_std': np.std(tau_nz),
            'tau_mean_abs': np.mean(np.abs(tau_nz)),
            'pct_tau_near_zero_01': (np.abs(tau_nz) < 0.01).mean() * 100,
            'pct_tau_near_zero_05': (np.abs(tau_nz) < 0.05).mean() * 100,
            'pct_tau_near_zero_10': (np.abs(tau_nz) < 0.1).mean() * 100,
            'r_gamma_vs_horseshoe': r_gamma,
            'r_gamma_tau': r_gt,
            'converged_at': res['converged_at'],
        })

    summary = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "sensitivity_summary.csv")
    summary.to_csv(path, index=False)

    print(f"\n{'=' * 70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'=' * 70}")
    print(summary.to_string(index=False))
    print(f"\n  Saved: {os.path.basename(path)}")

    # ── Per-language γ table ──────────────────────────────────────
    gamma_table = pd.DataFrame({
        name: {l: res['gamma'][l_map[l]] for l in non_en}
        for name, res in results.items()
    })
    gamma_table.to_csv(os.path.join(RESULTS_DIR, "gamma_by_prior.csv"))
    print(f"\n  γ by prior:")
    print(gamma_table.round(3).to_string())

    return summary


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if _HAS_FIG_STYLE: apply_style()
    print("=" * 60)
    print("HORSESHOE SENSITIVITY & γ-τ MULTICOLLINEARITY")
    print("=" * 60)

    df, anchor_ids = load_data()

    # Fit all three variants
    results = run_analysis(df, anchor_ids)

    # Plots
    print(f"\n{'=' * 60}")
    print("GENERATING FIGURES")
    print(f"{'=' * 60}")

    plot_gamma_comparison(results)
    plot_tau_comparison(results)
    plot_gamma_tau_collinearity(results)

    # Summary
    summary = make_summary(results)

    # ── Key findings for paper ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("KEY FINDINGS (for paper paragraph)")
    print(f"{'=' * 60}")

    hs = summary[summary['prior'] == 'horseshoe'].iloc[0]
    nm = summary[summary['prior'] == 'normal'].iloc[0]

    print(f"\n1. γ STABILITY:")
    print(f"   r(γ_horseshoe, γ_normal) = {nm['r_gamma_vs_horseshoe']:.3f}")
    print(f"   → {'γ robust to prior choice' if nm['r_gamma_vs_horseshoe'] > 0.95 else 'γ SENSITIVE to prior choice'}")

    print(f"\n2. τ SPARSITY:")
    print(f"   Horseshoe: {hs['pct_tau_near_zero_10']:.1f}% of τ < 0.10")
    print(f"   Normal:    {nm['pct_tau_near_zero_10']:.1f}% of τ < 0.10")
    print(f"   → Horseshoe {'does' if hs['pct_tau_near_zero_10'] > nm['pct_tau_near_zero_10'] else 'does NOT'} produce sparser τ")

    print(f"\n3. MULTICOLLINEARITY:")
    print(f"   r(γ, mean τ) under Horseshoe: {hs['r_gamma_tau']:.3f}")
    print(f"   r(γ, mean τ) under Normal:    {nm['r_gamma_tau']:.3f}")
    print(f"   → {'Horseshoe reduces confounding' if abs(hs['r_gamma_tau']) < abs(nm['r_gamma_tau']) else 'Similar confounding under both'}")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()