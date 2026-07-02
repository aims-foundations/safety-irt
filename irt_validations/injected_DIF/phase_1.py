"""
Phase 1 — set generative parameters for the injected-DIF validation.

Preserved from the real fit (structure matters):
  beta_i  : 315 fitted prompt difficulties        (Base_Difficulty)
  gamma_L : 9 fitted language shifts, en = 0       (gamma_language_params)

Randomized per replication (exchangeable; severs the real-tau imprint):
  theta_j  ~ Normal(THETA_MEAN, THETA_SD)
  delta_jL ~ Normal(DELTA_MEAN, DELTA_SD), en column pinned to 0

theta and delta keep their FITTED means (not 0). Only (theta-beta) and
(delta_mean-gamma) are identified, so preserving beta/gamma forces us to keep
the fitted person-side means or the synthetic safe-rates drift off reality.

Colab: run this cell after phase_0. It defines load_preserved() and
draw_person_params(), used by phase_3.
"""

import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# ── fixed structure ──────────────────────────────────────────────────────────
N_PROMPTS = 315
N_PASSES  = 10
# en first, then anchors.ALL_LANGS order (keeps column/lang alignment with anchors.py)
LANGS  = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
EN_IDX = LANGS.index("en")

# ── Phase 0 harvested constants (from phase_0.py on the HF anchored fit) ──────
THETA_MEAN, THETA_SD = 1.30, 0.70
DELTA_MEAN, DELTA_SD = -0.26, 0.38    # non-en; en is structurally 0
M = 0.46                              # injected DIF magnitude = median|tau_hat|

# ── HF source (same repo/path as phase_0.py) ─────────────────────────────────
REPO      = "aims-foundations/safety-irt"
REPO_TYPE = "dataset"
BASE      = "results/results"


def _hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def load_preserved(token=None):
    """Load the frozen beta (315) and gamma (10, en=0) vectors + config names.

    Returns
    -------
    beta    : (315,) array indexed by prompt id 0..314 (contiguous ints in the fit)
    gamma   : (10,)  array aligned to LANGS order, gamma[EN_IDX] == 0
    configs : (61,)  test_taker names for the output rows
    """
    if token is None:
        token = _hf_token()

    res = pd.read_csv(hf_hub_download(REPO, f"{BASE}/bayesian_irt_results_binary.csv",
                                      repo_type=REPO_TYPE, token=token))
    gam = pd.read_csv(hf_hub_download(REPO, f"{BASE}/gamma_language_params.csv",
                                      repo_type=REPO_TYPE, token=token))
    th  = pd.read_csv(hf_hub_download(REPO, f"{BASE}/theta_person_params.csv",
                                      repo_type=REPO_TYPE, token=token))

    # beta: dedupe per prompt, align to contiguous 0..314 ids used throughout
    beta = (res.drop_duplicates("prompt")
               .set_index("prompt")["Base_Difficulty"]
               .reindex(range(N_PROMPTS))
               .to_numpy())
    assert not np.isnan(beta).any(), (
        "beta failed to align to 0..314 — check prompt id format in the results file")

    gmap  = dict(zip(gam["language"], gam["gamma_L"]))
    gamma = np.array([gmap.get(L, 0.0) for L in LANGS], dtype=float)
    gamma[EN_IDX] = 0.0

    configs = th["test_taker"].to_numpy()
    assert len(configs) == 61, f"expected 61 configs, got {len(configs)}"
    return beta, gamma, configs


def load_tau_pool(token=None):
    """Empirical signed tau (Safety_Tax) over FREE cells: non-anchor, non-English.

    Used by phase_2's 'realistic' direction to resample real-like DIF (reproduces
    the 59/41 sign split, heavy tails, and net mean ~+0.21 of the real data).
    """
    if token is None:
        token = _hf_token()
    res = pd.read_csv(hf_hub_download(REPO, f"{BASE}/bayesian_irt_results_binary.csv",
                                      repo_type=REPO_TYPE, token=token))
    anchor = res["Is_Anchor"].astype(str).str.strip().str.lower().eq("true")
    pool = res.loc[(~anchor) & (res["language"] != "en"), "Safety_Tax"]
    return pool.dropna().to_numpy(dtype=float)


def draw_person_params(rng, n_configs):
    """Fresh theta (n_configs,) and delta (n_configs, 10) for ONE replication.

    theta ~ Normal(THETA_MEAN, THETA_SD)
    delta ~ Normal(DELTA_MEAN, DELTA_SD), en column pinned to 0
    Both keep their fitted means for location consistency with preserved beta/gamma.
    """
    theta = rng.normal(THETA_MEAN, THETA_SD, size=n_configs)
    delta = rng.normal(DELTA_MEAN, DELTA_SD, size=(n_configs, len(LANGS)))
    delta[:, EN_IDX] = 0.0
    return theta, delta


if __name__ == "__main__":
    beta, gamma, configs = load_preserved()
    print(f"beta:  n={len(beta)}  mean={beta.mean():+.3f}  sd={beta.std():.3f}  "
          f"range=[{beta.min():+.2f}, {beta.max():+.2f}]")
    print("gamma:", {L: round(float(g), 3) for L, g in zip(LANGS, gamma)})
    print(f"configs: {len(configs)}")
    rng = np.random.default_rng(0)
    th, dl = draw_person_params(rng, len(configs))
    print(f"theta draw: mean={th.mean():+.3f} sd={th.std():.3f}")
    print(f"delta draw (non-en): mean={dl[:, 1:].mean():+.3f} sd={dl[:, 1:].std():.3f}  "
          f"en col all-zero: {np.allclose(dl[:, EN_IDX], 0)}")
