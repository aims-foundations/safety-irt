# -*- coding: utf-8 -*-
"""
Binary IRT model with anchoring constraints for multilingual safety analysis.
Uses Pyro for Bayesian inference via SVI.
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
from tqdm import tqdm
import os
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "FINALMERGEDTAGGED.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SAVE_MODEL_FILE = os.path.join(RESULTS_DIR, "irt_params_binary_final.pt")
SAVE_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
SAVE_PLOT_FILE = os.path.join(RESULTS_DIR, "0_bayesian_irt_plots_binary.png")
TRAINING_STEPS = 4000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model(student_idx, prompt_idx, lang_idx, obs=None,
          num_students=None, num_prompts=None, num_langs=None,
          tau_mask=None, gamma_mask=None):
    """Bayesian IRT Model (Binary/Bernoulli)."""
    theta = pyro.sample("theta", dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))

    gamma_raw = pyro.sample("gamma_raw", dist.Normal(torch.zeros(num_langs, device=device), 1.0).to_event(1))
    gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)

    tau_scale = pyro.sample("tau_scale", dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_raw = pyro.sample("tau_raw", dist.StudentT(1.0, torch.zeros(num_prompts, num_langs, device=device), tau_scale).to_event(2))
    tau = pyro.deterministic("tau", tau_raw * tau_mask)

    delta_raw = pyro.sample("delta_raw", dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5).to_event(2))
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    with pyro.plate("data", len(student_idx)):
        ability = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        logits = ability - difficulty
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def train_and_extract():
    print(f"Starting binary IRT training on {device}...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"'{INPUT_FILE}' not found")

    try:
        df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    print(f"Loaded {len(df)} raw rows")

    # Preprocessing: map Likert 1-5 to binary (4-5=Safe, 1-3=Unsafe)
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    print(f"Validated {len(df)} rows (binary: 4-5=Safe, 1-3=Unsafe)")

    df['id'] = df['id'].apply(clean_id)

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    if student_col not in df.columns:
        raise ValueError("Missing 'test_taker' or 'model' column")

    students = df[student_col].unique()
    prompts = df['id'].unique()
    languages = df['language'].unique()

    student_map = {s: i for i, s in enumerate(students)}
    prompt_map = {p: i for i, p in enumerate(prompts)}
    lang_map = {l: i for i, l in enumerate(languages)}

    student_idx = torch.tensor(df[student_col].map(student_map).values, dtype=torch.long).to(device)
    prompt_idx = torch.tensor(df['id'].map(prompt_map).values, dtype=torch.long).to(device)
    lang_idx = torch.tensor(df['language'].map(lang_map).values, dtype=torch.long).to(device)
    score_obs = torch.tensor(df['score'].values, dtype=torch.float32).to(device)

    num_students = len(students)
    num_prompts = len(prompts)
    num_langs = len(languages)

    # Constraints
    tau_mask = torch.ones((num_prompts, num_langs), device=device)
    gamma_mask = torch.ones(num_langs, device=device)

    if 'en' in lang_map:
        en_i = lang_map['en']
        tau_mask[:, en_i] = 0.0
        gamma_mask[en_i] = 0.0
        print("English baseline constraint applied (gamma=0, tau=0)")

    try:
        anchors_df = pd.read_csv(ANCHOR_FILE)
        anchors_df['id'] = anchors_df['id'].apply(clean_id)
        anchor_ids = set(anchors_df['id'].unique())

        count = 0
        for pid in prompts:
            if pid in anchor_ids:
                tau_mask[prompt_map[pid], :] = 0.0
                count += 1

        print(f"Anchor constraint applied to {count}/{len(anchor_ids)} prompts")
        if count == 0:
            print("WARNING: 0 anchors matched -- check CSV IDs")
    except FileNotFoundError:
        print(f"Warning: '{ANCHOR_FILE}' not found, only English constraints applied")

    # Training
    guide = pyro.infer.autoguide.AutoNormal(pyro.poutine.block(model, hide=["obs", "tau", "gamma", "delta"]))
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if os.path.exists(SAVE_MODEL_FILE):
        print(f"Loading saved model from '{SAVE_MODEL_FILE}'")
        saved_params = torch.load(SAVE_MODEL_FILE, weights_only=False)
        pyro.get_param_store().set_state(saved_params)
    else:
        print(f"Training for {TRAINING_STEPS} steps...")
        pbar = tqdm(range(TRAINING_STEPS))
        losses = []

        for step in pbar:
            loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs,
                            num_students, num_prompts, num_langs, tau_mask, gamma_mask)
            losses.append(loss)
            if step % 100 == 0:
                pbar.set_description(f"Loss: {loss:.2f}")

        torch.save(pyro.get_param_store().get_state(), SAVE_MODEL_FILE)
        print(f"Model saved to '{SAVE_MODEL_FILE}'")

        plt.figure(figsize=(10, 4))
        plt.plot(losses, alpha=0.3, label='Raw Loss')
        if len(losses) > 50:
            ma = np.convolve(losses, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(losses)), ma, color='red', label='Smoothed')
        plt.title("Binary IRT Training Convergence")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "training_convergence.png"))

    # Extract results
    print("Sampling posterior...")
    predictive = Predictive(model, guide=guide, num_samples=500, return_sites=["beta", "gamma", "tau"])
    samples = predictive(student_idx, prompt_idx, lang_idx, None,
                         num_students, num_prompts, num_langs, tau_mask, gamma_mask)

    mean_beta = samples['beta'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_gamma = samples['gamma'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_tau = samples['tau'].mean(dim=0).detach().cpu().numpy()
    if mean_tau.ndim > 2: mean_tau = mean_tau.squeeze()

    results = []
    en_idx = lang_map.get('en', -1)

    if en_idx != -1:
        for l_name, l_idx in lang_map.items():
            if l_name == 'en': continue
            if l_idx >= len(mean_gamma): continue

            for p_idx, p_name in enumerate(prompts):
                if p_idx >= len(mean_beta): break

                base_diff = mean_beta[p_idx]
                trans_cost = mean_tau[p_idx, l_idx]
                lang_diff = base_diff + mean_gamma[l_idx] + trans_cost
                is_anchor = (tau_mask[p_idx, l_idx].item() == 0.0)

                results.append({
                    'prompt': p_name,
                    'language': l_name,
                    'Base_Difficulty': base_diff,
                    'Lang_Difficulty': lang_diff,
                    'Safety_Tax': trans_cost,
                    'Is_Anchor': is_anchor
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(SAVE_RESULTS_FILE, index=False)
    print(f"Results saved to '{SAVE_RESULTS_FILE}' ({len(res_df)} rows)")


def plot_results():
    if not os.path.exists(SAVE_RESULTS_FILE):
        raise FileNotFoundError("Results file not found -- run training first")

    res_df = pd.read_csv(SAVE_RESULTS_FILE)
    target_langs = res_df["language"].unique()
    n_langs = len(target_langs)

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    min_val = min(res_df["Base_Difficulty"].min(), res_df["Lang_Difficulty"].min())
    max_val = max(res_df["Base_Difficulty"].max(), res_df["Lang_Difficulty"].max())
    palette = sns.color_palette("tab10")

    for i, lang in enumerate(target_langs):
        if i >= nrows * ncols:
            break
        ax = axes[i]
        lang_data = res_df[res_df["language"] == lang]
        anchors = lang_data[lang_data["Is_Anchor"]]
        non_anchors = lang_data[~lang_data["Is_Anchor"]]

        sns.scatterplot(data=non_anchors, x="Base_Difficulty", y="Lang_Difficulty",
                        ax=ax, alpha=0.5, color=palette[i % 10], label="Normal")

        if not anchors.empty:
            sns.scatterplot(data=anchors, x="Base_Difficulty", y="Lang_Difficulty",
                            ax=ax, color="black", marker="*", s=100, label="Anchor")

        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Equal Difficulty")

        taxed_rate = (non_anchors["Lang_Difficulty"] > non_anchors["Base_Difficulty"]).mean()
        ax.set_title(f"{lang.upper()} (Taxed: {taxed_rate:.1%})", fontsize=14, fontweight="bold")
        ax.set_xlabel(r"English Difficulty ($\beta_i$)")
        ax.set_ylabel("Target Difficulty")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for j in range(n_langs, nrows * ncols):
        axes[j].set_visible(False)

    plt.suptitle("Bayesian Safety Cost (Binary Model)", fontsize=16)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_FILE, dpi=300)
    print(f"Plot saved to '{SAVE_PLOT_FILE}'")


if __name__ == "__main__":
    train_and_extract()
    plot_results()
