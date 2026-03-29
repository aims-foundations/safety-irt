# -*- coding: utf-8 -*-
"""
2PL Binary IRT model — XSafety
================================
Identical model to model/irt.py, adapted for XSafety:
  - Single pass (no pass column)
  - 9 focal languages: zh, ar, bn, de, fr, hi, ja, ru, sp
  - Input: XSafety_Dataset.csv
  - Anchors from xsafety/results_dif_stratified/soft_anchor_priors.csv

Implements: P(safe_ijL = 1) = σ(α_i · ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, CMAP_DIV, add_identity_line)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
_save = fs_savefig if _HAS_FIG_STYLE else \
    lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))
from tqdm import tqdm
from huggingface_hub import snapshot_download

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "xsafety", "xsafety_pass_graded.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "xsafety", "xsafety_anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "xsafety_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SAVE_MODEL_FILE   = os.path.join(RESULTS_DIR, "irt_params_binary_2pl.pt")
SAVE_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
SAVE_PLOT_FILE    = os.path.join(RESULTS_DIR, "0_bayesian_irt_plots_binary.png")
SAVE_GAMMA_FILE   = os.path.join(RESULTS_DIR, "gamma_language_params.csv")
TRAINING_STEPS = 4000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_2pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None, anchor_mask_tensor=None):
    if anchor_mask_tensor is None:
        anchor_mask_tensor = torch.zeros(num_prompts, num_langs, device=device)
    """
    Bayesian 2PL IRT Model (Binary/Bernoulli).

    P(safe) = σ(α_i · ((θ_j + δ_jL) − (β_i + γ_L + τ_iL)))
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

    tau_scale = pyro.sample("tau_scale",
        dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_scale_per = torch.where(
        anchor_mask_tensor > 0.5,
        torch.full((num_prompts, num_langs), 0.01, device=device),
        tau_scale.expand(num_prompts, num_langs))
    tau_raw = pyro.sample("tau_raw",
        dist.StudentT(1.0, torch.zeros(num_prompts, num_langs, device=device),
                      tau_scale_per).to_event(2))
    tau = pyro.deterministic("tau", tau_raw * tau_mask)

    delta_raw = pyro.sample("delta_raw",
        dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5).to_event(2))
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    with pyro.plate("data", len(student_idx)):
        ability     = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty  = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        logits      = alpha[prompt_idx] * (ability - difficulty)
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def train_and_extract():
    print(f"Starting XSafety 2PL binary IRT training on {device}...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"'{INPUT_FILE}' not found")

    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    print(f"Loaded {len(df)} raw rows")

    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    print(f"Validated {len(df)} rows (binary: 4-5=Safe, 1-3=Unsafe)")

    df['id'] = df['id'].apply(clean_id)

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    if student_col not in df.columns:
        raise ValueError("Missing 'test_taker' or 'model' column")

    students  = df[student_col].unique()
    prompts   = df['id'].unique()
    languages = df['language'].unique()

    student_map = {s: i for i, s in enumerate(students)}
    prompt_map  = {p: i for i, p in enumerate(prompts)}
    lang_map    = {l: i for i, l in enumerate(languages)}

    student_idx = torch.tensor(df[student_col].map(student_map).values,
                               dtype=torch.long).to(device)
    prompt_idx  = torch.tensor(df['id'].map(prompt_map).values,
                               dtype=torch.long).to(device)
    lang_idx    = torch.tensor(df['language'].map(lang_map).values,
                               dtype=torch.long).to(device)
    score_obs   = torch.tensor(df['score'].values, dtype=torch.float32).to(device)

    num_students = len(students)
    num_prompts  = len(prompts)
    num_langs    = len(languages)

    tau_mask   = torch.ones((num_prompts, num_langs), device=device)
    gamma_mask = torch.ones(num_langs, device=device)

    if 'en' in lang_map:
        en_i = lang_map['en']
        tau_mask[:, en_i]  = 0.0
        gamma_mask[en_i]   = 0.0
        print("English baseline constraint applied (gamma=0, tau=0)")

    anchor_mask_tensor = torch.zeros((num_prompts, num_langs), device=device)
    if os.path.exists(ANCHOR_FILE):
        anchors_df = pd.read_csv(ANCHOR_FILE)
        if 'is_anchor' in anchors_df.columns:
            anchors_df = anchors_df[anchors_df['is_anchor'] == True]
        id_col = 'prompt_id' if 'prompt_id' in anchors_df.columns else 'id'
        anchors_df['id'] = anchors_df[id_col].apply(clean_id)
        anchor_ids = set(anchors_df['id'].unique())
        count = 0
        for pid in prompts:
            if pid in anchor_ids and pid in prompt_map:
                anchor_mask_tensor[prompt_map[pid], :] = 1.0
                count += 1
        print(f"Anchor constraint applied to {count}/{len(anchor_ids)} prompts")
        if count == 0:
            print("WARNING: 0 anchors matched — run anchors.py first")
    else:
        print(f"Warning: '{ANCHOR_FILE}' not found — run anchors.py first")

    guide     = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_2pl, hide=["obs", "tau", "gamma", "delta"]))
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi       = SVI(model_2pl, guide, optimizer, loss=Trace_ELBO())

    if os.path.exists(SAVE_MODEL_FILE):
        print(f"Loading saved model from '{SAVE_MODEL_FILE}'")
        saved_params = torch.load(SAVE_MODEL_FILE, weights_only=False)
        pyro.get_param_store().set_state(saved_params)
    else:
        print(f"Training 2PL model for {TRAINING_STEPS} steps...")
        pbar   = tqdm(range(TRAINING_STEPS))
        losses = []
        for step in pbar:
            loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs,
                            num_students, num_prompts, num_langs,
                            tau_mask, gamma_mask, anchor_mask_tensor)
            losses.append(loss)
            if step % 100 == 0:
                pbar.set_description(f"Loss: {loss:.2f}")

        torch.save(pyro.get_param_store().get_state(), SAVE_MODEL_FILE)
        print(f"Model saved to '{SAVE_MODEL_FILE}'")

        _cr = C_RED if _HAS_FIG_STYLE else '#c0392b'
        plt.figure(figsize=(5.5, 2.5))
        plt.plot(losses, alpha=0.3, label='Raw Loss')
        if len(losses) > 50:
            ma = np.convolve(losses, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(losses)), ma, color=_cr, label='Smoothed')
        plt.title("XSafety 2PL Training Convergence")
        plt.legend(fontsize=6)
        plt.savefig(os.path.join(RESULTS_DIR, "training_convergence.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print("Sampling posterior...")
    predictive = Predictive(model_2pl, guide=guide, num_samples=500,
                            return_sites=["beta", "gamma", "tau", "alpha", "theta", "delta"])
    samples = predictive(student_idx, prompt_idx, lang_idx, None,
                         num_students, num_prompts, num_langs,
                         tau_mask, gamma_mask, anchor_mask_tensor)

    mean_beta  = samples['beta'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_gamma = samples['gamma'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_tau   = samples['tau'].mean(dim=0).detach().cpu().numpy()
    mean_alpha = samples['alpha'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    if mean_tau.ndim > 2:
        mean_tau = mean_tau.squeeze()

    gamma_rows = []
    for l_name, l_idx in lang_map.items():
        if l_idx < len(mean_gamma):
            gamma_rows.append({"language": l_name, "gamma_L": mean_gamma[l_idx]})
    gamma_df = pd.DataFrame(gamma_rows).sort_values("language")
    gamma_df.to_csv(SAVE_GAMMA_FILE, index=False)
    print(f"Gamma saved to '{SAVE_GAMMA_FILE}'")

    print(f"\n--- 2PL Discrimination (α) Summary ---")
    print(f"  Mean α:   {mean_alpha.mean():.3f}")
    print(f"  Median α: {np.median(mean_alpha):.3f}")
    print(f"  Std α:    {mean_alpha.std():.3f}")
    print(f"  Range:    [{mean_alpha.min():.3f}, {mean_alpha.max():.3f}]")

    results = []
    en_idx = lang_map.get('en', -1)
    if en_idx != -1:
        for l_name, l_idx in lang_map.items():
            if l_name == 'en' or l_idx >= len(mean_gamma):
                continue
            for p_idx, p_name in enumerate(prompts):
                if p_idx >= len(mean_beta):
                    break
                results.append({
                    'prompt':          p_name,
                    'language':        l_name,
                    'Base_Difficulty': mean_beta[p_idx],
                    'gamma_L':         mean_gamma[l_idx],
                    'Lang_Difficulty': mean_beta[p_idx] + mean_gamma[l_idx] + mean_tau[p_idx, l_idx],
                    'Safety_Tax':      mean_tau[p_idx, l_idx],
                    'Is_Anchor':       (tau_mask[p_idx, l_idx].item() == 0.0),
                    'alpha':           mean_alpha[p_idx],
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_RESULTS_FILE, index=False)

    mean_theta = samples['theta'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_delta = samples['delta'].mean(dim=0).detach().cpu().numpy()
    if mean_delta.ndim > 2:
        mean_delta = mean_delta.squeeze()

    theta_rows = [{'test_taker': s_name, 'theta': mean_theta[s_idx]}
                  for s_name, s_idx in student_map.items() if s_idx < len(mean_theta)]
    pd.DataFrame(theta_rows).to_csv(os.path.join(RESULTS_DIR, "theta_person_params.csv"),
                                    index=False)

    delta_rows = []
    for s_name, s_idx in student_map.items():
        for l_name, l_idx in lang_map.items():
            if s_idx < mean_delta.shape[0] and l_idx < mean_delta.shape[1]:
                delta_rows.append({'test_taker': s_name, 'language': l_name,
                                   'delta': mean_delta[s_idx, l_idx]})
    pd.DataFrame(delta_rows).to_csv(os.path.join(RESULTS_DIR, "delta_person_params.csv"),
                                    index=False)
    print(f"Results saved to '{SAVE_RESULTS_FILE}' ({len(res_df)} rows)")


def plot_results():
    if _HAS_FIG_STYLE:
        apply_style()
    if not os.path.exists(SAVE_RESULTS_FILE):
        raise FileNotFoundError("Results file not found — run training first")

    res_df = pd.read_csv(SAVE_RESULTS_FILE)
    target_langs = sorted(res_df["language"].unique())
    n_langs = len(target_langs)

    _cb = C_BLUE if _HAS_FIG_STYLE else '#5dade2'
    _cr = C_RED  if _HAS_FIG_STYLE else '#c0392b'

    nrows, ncols = 3, 3
    if _HAS_FIG_STYLE:
        fig, axes = make_fig_grid(nrows, ncols, height_override=2.0)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 5.5),
                                 sharex=True, sharey=True)
    axes_flat = axes.flatten()

    min_val = min(res_df["Base_Difficulty"].min(), res_df["Lang_Difficulty"].min())
    max_val = max(res_df["Base_Difficulty"].max(), res_df["Lang_Difficulty"].max())

    for i, lang in enumerate(target_langs):
        if i >= nrows * ncols:
            break
        ax = axes_flat[i]
        lang_data  = res_df[res_df["language"] == lang]
        anchors    = lang_data[lang_data["Is_Anchor"]]
        non_anchors = lang_data[~lang_data["Is_Anchor"]]

        ax.scatter(non_anchors["Base_Difficulty"], non_anchors["Lang_Difficulty"],
                   color=_cb, alpha=0.35, s=8, edgecolors='none', label="Non-anchor")
        if not anchors.empty:
            ax.scatter(anchors["Base_Difficulty"], anchors["Lang_Difficulty"],
                       color="black", marker="*", s=25, label="Anchor", zorder=5)
        ax.plot([min_val, max_val], [min_val, max_val],
                color=_cr, ls='--', lw=0.6, alpha=0.7, label="Equal difficulty")

        taxed_rate = (non_anchors["Lang_Difficulty"] > non_anchors["Base_Difficulty"]).mean()
        ax.set_title(f"{lang} ({taxed_rate:.0%} taxed)")
        ax.set_xlabel(r"$\beta_i$")
        ax.set_ylabel(r"$\beta_i + \gamma_L + \tau_{iL}$")
        if i == min(n_langs - 1, nrows * ncols - 1):
            ax.legend(fontsize=4, loc='upper left')

    for j in range(n_langs, nrows * ncols):
        axes_flat[j].set_visible(False)

    _save(fig, SAVE_PLOT_FILE)
    print(f"Plot saved to '{SAVE_PLOT_FILE}'")


if __name__ == "__main__":
    train_and_extract()
    plot_results()
