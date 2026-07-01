"""
Phase 3 — generate one synthetic dataset and expose it to the selection code.

Forward through the exact irt.py link with alpha = 1:
    p(safe) = sigmoid( theta_j + delta_jL - beta_i - gamma_L - tau_iL )
    safe ~ Bernoulli(p)

judge_score is written so anchors.py's binary collapse (>= 4) reproduces the
Bernoulli draw:  safe -> 5,  unsafe -> 1  (never 0 — both loaders drop score<=0).

Two consumers:
  generate_dataset(...) -> long DataFrame [id, language, test_taker, judge_score, pass]
  df_to_matrices(df)    -> {lang: person x prompt binary matrix}, exactly the
                           structure anchors.load_response_matrices() produces,
                           so anchors.variance_filter / compute_agreement_scores
                           run on it unchanged (no file round-trip needed).

Colab: run after phase_2.
"""

import numpy as np
import pandas as pd

from phase_1 import LANGS, EN_IDX, N_PROMPTS, N_PASSES, draw_person_params
from phase_2 import build_tau


def generate_dataset(beta, gamma, configs, proportion, direction, seed, magnitude=None):
    """Phases 1-3: one full synthetic dataset under the given condition.

    Returns
    -------
    df           : long-format DataFrame, 61 * 315 * 10 langs * 10 passes rows
    true_dif_ids : set[int] of prompt ids carrying injected DIF (ground truth)
    """
    rng = np.random.default_rng(seed)
    n_configs = len(configs)

    theta, delta = draw_person_params(rng, n_configs)                 # Phase 1
    if magnitude is None:
        tau, true_dif_ids = build_tau(rng, proportion, direction)    # Phase 2
    else:
        tau, true_dif_ids = build_tau(rng, proportion, direction, magnitude)

    # Phase 3: vectorized forward pass + Bernoulli sampling, language by language.
    frames = []
    for Li, L in enumerate(LANGS):
        ability    = theta[:, None] + delta[:, Li][:, None]              # (configs, 1)
        difficulty = beta[None, :] + gamma[Li] + tau[:, Li][None, :]     # (1, prompts)
        z = ability - difficulty
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))                   # (configs, prompts)

        # 10 independent Bernoulli passes -> (configs, prompts, passes)
        draws = rng.random((n_configs, N_PROMPTS, N_PASSES)) < p[:, :, None]

        cfg_i, prm_i, pass_i = np.meshgrid(
            np.arange(n_configs), np.arange(N_PROMPTS), np.arange(N_PASSES), indexing="ij")
        frames.append(pd.DataFrame({
            "id":          prm_i.ravel(),
            "language":    L,
            "test_taker":  configs[cfg_i.ravel()],
            "judge_score": np.where(draws.ravel(), 5, 1),   # 5=safe, 1=unsafe (never 0)
            "pass":        pass_i.ravel(),
        }))

    df = pd.concat(frames, ignore_index=True)
    return df, true_dif_ids


def df_to_matrices(df):
    """Long df -> {lang: person_key x prompt_id binary matrix}.

    Mirrors anchors.load_response_matrices() so the selection code consumes the
    synthetic data with no changes and no CSV round-trip.
    """
    d = df.copy()
    d["judge_score"] = pd.to_numeric(d["judge_score"], errors="coerce")
    d = d[d["judge_score"] > 0].dropna(subset=["judge_score"])
    d["binary"]     = (d["judge_score"] >= 4).astype(float)
    d["person_key"] = d["test_taker"].astype(str) + "_p" + d["pass"].astype(str)
    d["id"]         = d["id"].astype(str)

    matrices = {}
    for lang in LANGS:
        sub = d[d["language"] == lang]
        if sub.empty:
            continue
        matrices[lang] = sub.pivot_table(
            index="person_key", columns="id", values="binary", aggfunc="first")
    return matrices


if __name__ == "__main__":
    from phase_1 import load_preserved
    beta, gamma, configs = load_preserved()
    df, true_dif = generate_dataset(beta, gamma, configs,
                                    proportion=0.40, direction="unbalanced", seed=0)
    exp = len(configs) * N_PROMPTS * len(LANGS) * N_PASSES
    print(f"rows: {len(df):,}  (expect {exp:,})")
    print(f"languages: {sorted(df['language'].unique())}")
    print(f"configs: {df['test_taker'].nunique()}  prompts: {df['id'].nunique()}")
    print(f"true DIF prompts: {len(true_dif)}  (expect {round(0.40*N_PROMPTS)})")
    print(f"judge_score values: {sorted(df['judge_score'].unique())}")
    print(f"overall safe rate: {(df['judge_score'] >= 4).mean():.3f}")
    mats = df_to_matrices(df)
    print("matrices:", {L: mats[L].shape for L in mats})
