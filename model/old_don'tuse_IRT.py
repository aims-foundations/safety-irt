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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "FINAL_MERGED_FOR_IRT.csv"
ANCHOR_FILE = "verified_anchors_gpt4o.csv"
SAVE_MODEL_FILE = "irt_params_proposal_final.pt"
SAVE_RESULTS_FILE = "bayesian_irt_results_proposal.csv"
SAVE_PLOT_FILE = "0_bayesian_irt_plots_strict_anchored.png"
TRAINING_STEPS = 4000

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. MODEL DEFINITION
# ==============================================================================
def model(student_idx, prompt_idx, lang_idx, obs=None, 
          num_students=None, num_prompts=None, num_langs=None,
          tau_mask=None, gamma_mask=None):
    """
    Bayesian IRT Model with Safety Tax AND Model-Language Competence (Delta).
    """
    # 1. Priors
    # theta: Global Student Ability
    theta = pyro.sample("theta", dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    
    # beta: Base Prompt Difficulty
    beta = pyro.sample("beta", dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))
    
    # gamma: Global Language Shift
    gamma_raw = pyro.sample("gamma_raw", dist.Normal(torch.zeros(num_langs, device=device), 1.0).to_event(1))
    gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)
    
    # tau: Safety Tax (The core research variable)
    tau_scale = pyro.sample("tau_scale", dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_raw = pyro.sample("tau_raw", dist.StudentT(1.0, torch.zeros(num_prompts, num_langs, device=device), tau_scale).to_event(2))
    tau = pyro.deterministic("tau", tau_raw * tau_mask)

    # --- NEW: delta (Model-Language Competence) ---
    # This absorbs specific model failures (e.g., Llama failing Swahili) 
    # so they don't corrupt the Safety Tax.
    delta_raw = pyro.sample("delta_raw", dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5).to_event(2))
    
    # Constraint: English Delta -> 0 (Baseline ability is defined in English)
    # We broadcast gamma_mask (which is 0 for En, 1 for others) to match shape
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    # 2. Likelihood (Full Batch)
    with pyro.plate("data", len(student_idx)):
        # Effective Ability = Global Theta + Specific Language Adjustment (Delta)
        ability = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        
        logits = ability - difficulty
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)
# ==============================================================================
# 2. MAIN TRAINING LOGIC
# ==============================================================================
def train_and_extract():
    print(f"🚀 STARTING IRT TRAINING on {device}...")

    # --- A. LOAD DATA ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: '{INPUT_FILE}' not found. Please run the Curation script first.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df[df['label'].isin(['safe', 'unsafe'])].copy()
    df['score'] = df['label'].map({'safe': 1, 'unsafe': 0}).values.astype(np.float32)

    # Mappings
    students = df['config'].unique()
    prompts = df['id'].astype(str).unique()
    languages = df['language'].unique()

    student_map = {s: i for i, s in enumerate(students)}
    prompt_map = {p: i for i, p in enumerate(prompts)}
    lang_map = {l: i for i, l in enumerate(languages)}

    # Tensors
    student_idx = torch.tensor(df['config'].map(student_map).values, dtype=torch.long).to(device)
    prompt_idx = torch.tensor(df['id'].astype(str).map(prompt_map).values, dtype=torch.long).to(device)
    lang_idx = torch.tensor(df['language'].map(lang_map).values, dtype=torch.long).to(device)
    score_obs = torch.tensor(df['score'].values, dtype=torch.float32).to(device)

    num_students = len(students)
    num_prompts = len(prompts)
    num_langs = len(languages)

    # --- B. CONFIGURE CONSTRAINTS (MASKS) ---
    print("⚓ Configuring Constraints...")
    tau_mask = torch.ones((num_prompts, num_langs), device=device)
    gamma_mask = torch.ones(num_langs, device=device)

    # Constraint 1: English is Baseline (0 shift, 0 tax)
    if 'en' in lang_map:
        en_i = lang_map['en']
        tau_mask[:, en_i] = 0.0
        gamma_mask[en_i] = 0.0
        print("   ✅ Constraint Applied: English Gamma & Tau forced to 0.")

    # Constraint 2: Anchors (0 tax)
    try:
        anchors_df = pd.read_csv(ANCHOR_FILE)
        # Use a subset if strictly necessary, otherwise use all valid anchors
        anchors_df = anchors_df.sample(frac=1.0, random_state=42) 
        anchor_ids = set(anchors_df['id'].astype(str).unique())
        
        count = 0
        for pid in prompts:
            if pid in anchor_ids:
                p_i = prompt_map[pid]
                tau_mask[p_i, :] = 0.0
                count += 1
        print(f"   ✅ Anchoring applied to {count} prompts (Tau forced to 0).")
    except FileNotFoundError:
        print(f"⚠️ Warning: '{ANCHOR_FILE}' not found. Only English constraints applied.")

    # --- C. TRAINING LOOP ---
    # AutoNormal Guide (blocks deterministic nodes to prevent crashes)
    guide = pyro.infer.autoguide.AutoNormal(pyro.poutine.block(model, hide=["obs", "tau", "gamma"]))
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    if os.path.exists(SAVE_MODEL_FILE):
        print(f"\n📂 Found saved model '{SAVE_MODEL_FILE}'. Loading...")
        saved_params = torch.load(SAVE_MODEL_FILE, weights_only=False) # Fix for unpickling error
        pyro.get_param_store().set_state(saved_params)
    else:
        print(f"\n🔄 Training for {TRAINING_STEPS} steps...")
        pbar = tqdm(range(TRAINING_STEPS))
        losses = []
        
        for step in pbar:
            # Pass all arguments (including masks) to the step
            loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs,
                            num_students, num_prompts, num_langs, tau_mask, gamma_mask)
            losses.append(loss)
            if step % 100 == 0:
                pbar.set_description(f"Loss: {loss:.2f}")
        
        print(f"💾 Saving model to '{SAVE_MODEL_FILE}'...")
        torch.save(pyro.get_param_store().get_state(), SAVE_MODEL_FILE)
        
        # Plot Convergence
        plt.figure(figsize=(10, 4))
        plt.plot(losses, alpha=0.3, label='Raw Loss')
        if len(losses) > 50:
            ma = np.convolve(losses, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(losses)), ma, color='red', label='Smoothed')
        plt.title("IRT Training Convergence")
        plt.legend()
        plt.savefig("training_convergence.png")

    # --- D. EXTRACT RESULTS ---
    print("\n✅ Sampling Posterior (Extracting Results)...")
    predictive = Predictive(model, guide=guide, num_samples=500, return_sites=["beta", "gamma", "tau"])
    
    # We pass None for obs during prediction
    samples = predictive(student_idx, prompt_idx, lang_idx, None,
                         num_students, num_prompts, num_langs, tau_mask, gamma_mask)

    mean_beta = samples['beta'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_gamma = samples['gamma'].mean(dim=0).detach().cpu().numpy().reshape(-1)
    mean_tau = samples['tau'].mean(dim=0).detach().cpu().numpy()
    if mean_tau.ndim > 2: mean_tau = mean_tau.squeeze()

    print("   Formatting CSV...")
    results = []
    en_idx = lang_map.get('en', -1)

    if en_idx != -1:
        for l_name, l_idx in lang_map.items():
            if l_name == 'en': continue # Skip English (Base)
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
    print(f"   Saved results to '{SAVE_RESULTS_FILE}' ({len(res_df)} rows)")

# ==============================================================================
# 3. VISUALIZATION LOGIC
# ==============================================================================
def plot_results():
    print(f"\n🎨 Generating Visualization '{SAVE_PLOT_FILE}'...")
    
    if not os.path.exists(SAVE_RESULTS_FILE):
        print("❌ Error: Results file not found.")
        return

    res_df = pd.read_csv(SAVE_RESULTS_FILE)
    target_langs = res_df["language"].unique()
    n_langs = len(target_langs)

    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 6), sharex=True, sharey=True)
    if n_langs == 1: axes = [axes]
    
    min_val = min(res_df["Base_Difficulty"].min(), res_df["Lang_Difficulty"].min())
    max_val = max(res_df["Base_Difficulty"].max(), res_df["Lang_Difficulty"].max())
    palette = sns.color_palette("tab10")

    for i, lang in enumerate(target_langs):
        ax = axes[i]
        lang_data = res_df[res_df["language"] == lang]
        anchors = lang_data[lang_data["Is_Anchor"]]
        non_anchors = lang_data[~lang_data["Is_Anchor"]]

        # Plot Normal
        sns.scatterplot(data=non_anchors, x="Base_Difficulty", y="Lang_Difficulty",
                        ax=ax, alpha=0.5, color=palette[i % 10], label="Normal")
        # Plot Anchors
        if not anchors.empty:
            sns.scatterplot(data=anchors, x="Base_Difficulty", y="Lang_Difficulty",
                            ax=ax, color="black", marker="*", s=100, label="Anchor")

        # Identity Line
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Equal Difficulty")

        # Calc Tax Rate
        taxed_rate = (non_anchors["Lang_Difficulty"] > non_anchors["Base_Difficulty"]).mean()
        ax.set_title(f"{lang.upper()} (Taxed: {taxed_rate:.1%})", fontsize=14, fontweight="bold")
        ax.set_xlabel(r"English Difficulty ($\beta_i$)")
        if i == 0: ax.set_ylabel("Target Difficulty")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Bayesian Safety Cost (Strict Anchored)", fontsize=16)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_FILE, dpi=300)
    print(f"✅ Saved plot to '{SAVE_PLOT_FILE}'")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    train_and_extract()
    plot_results()
