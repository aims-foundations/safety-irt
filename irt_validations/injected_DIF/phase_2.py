"""
Phase 2 — build the tau (DIF) matrix for one (proportion, direction) condition.

Knob A  DIF proportion : fraction of the 315 prompts that carry injected DIF.
Knob B  DIF direction:
    'balanced'   = per-prompt +/-m coin flip           (control; gaps cancel)
    'unbalanced' = all DIF prompts +m                   (adversarial worst case)
    'realistic'  = per-prompt tau RESAMPLED from the empirical signed-tau
                   distribution (non-anchor, non-en Safety_Tax). Reproduces the
                   real 59/41 sign split, heavy tails, and net mean (~+0.21) at
                   once — i.e. the data's ACTUAL operating point, which sits
                   between the balanced and unbalanced extremes.
Knob C  N_ANCHORS = 40 (method property; used in phase_4, not here)
Magnitude m = 0.46 (Phase 0 median|tau|), fixed for balanced/unbalanced.

DIF-free prompts have tau = 0 in every language; the en column is always 0.
The per-prompt sign/magnitude is drawn ONCE per prompt and applied to every
non-en language (a prompt that is hard in translation is hard across languages,
matching the balanced/unbalanced design).

Note on 'realistic': because magnitudes are resampled from the real heavy-tailed
distribution, a minority of "DIF" prompts draw near-zero tau and are effectively
invariant — genuine label noise that mildly inflates measured contamination.
That is realistic, not a bug; the two idealized modes remain the clean brackets.

Colab: run after phase_1. Defines build_tau(), used by phase_3.
"""

import numpy as np

try:
    from phase_1 import LANGS, N_PROMPTS, M, load_tau_pool   # when run as a .py module
except ModuleNotFoundError:
    pass  # Colab: phase_1 ran as a cell above; its names are already in globals

PROPORTIONS   = [0.10, 0.25, 0.40, 0.50]
DIRECTIONS    = ["balanced", "unbalanced", "realistic"]
SIGN_POS_FRAC = 0.59   # Phase 0: 59% of real free-cell tau are positive (informational)

_TAU_POOL = None       # cached empirical signed-tau pool (loaded once per process)


def _get_tau_pool():
    """Empirical signed-tau samples for the 'realistic' mode, loaded once and cached."""
    global _TAU_POOL
    if _TAU_POOL is None:
        _TAU_POOL = load_tau_pool()
    return _TAU_POOL


def build_tau(rng, proportion, direction, magnitude=M, tau_pool=None):
    """tau matrix (315, 10) for one condition, plus the ground-truth DIF id set.

    tau_pool : optional array of empirical signed tau for the 'realistic' mode.
               If None, it is loaded (and cached) via phase_1.load_tau_pool().

    Returns
    -------
    tau          : (315, 10) array; column order matches phase_1.LANGS, en col = 0
    true_dif_ids : set[int] of prompt ids (0..314) carrying injected DIF
    """
    n_dif   = round(proportion * N_PROMPTS)
    dif_ids = rng.choice(N_PROMPTS, size=n_dif, replace=False)

    if direction == "unbalanced":
        vals = np.full(n_dif, magnitude)                        # all +m
    elif direction == "balanced":
        vals = rng.choice([-1.0, 1.0], size=n_dif) * magnitude  # +/-m coin flip
    elif direction == "realistic":
        pool = tau_pool if tau_pool is not None else _get_tau_pool()
        vals = rng.choice(np.asarray(pool, dtype=float), size=n_dif)  # empirical signed tau
    else:
        raise ValueError(f"direction must be 'balanced', 'unbalanced', or "
                         f"'realistic', got {direction!r}")

    tau = np.zeros((N_PROMPTS, len(LANGS)))
    for Li, L in enumerate(LANGS):
        if L == "en":
            continue
        tau[dif_ids, Li] = vals            # same per-prompt draw across all non-en langs

    return tau, set(int(x) for x in dif_ids)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_pool = rng.standard_t(1, size=2835) * 0.5   # heavy-tailed stand-in, avoids network
    for direction in DIRECTIONS:
        kw = {"tau_pool": fake_pool} if direction == "realistic" else {}
        tau, dif = build_tau(rng, 0.40, direction, **kw)
        nz = np.abs(tau) > 0
        net = tau[:, 1].sum() / N_PROMPTS      # net mean shift in one non-en language
        print(f"{direction:>10}: DIF={len(dif)} (expect {round(0.40*N_PROMPTS)})  "
              f"nonzero={nz.sum()}  en_zero={np.allclose(tau[:, 0], 0)}  "
              f"net_mean_tau(zh)={net:+.3f}")
        clean = [i for i in range(N_PROMPTS) if i not in dif]
        assert np.allclose(tau[clean], 0.0), "clean prompt carried tau!"
