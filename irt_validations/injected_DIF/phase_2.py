"""
Phase 2 — build the tau (DIF) matrix for one (proportion, direction) condition.

Knob A  DIF proportion : fraction of the 315 prompts that carry injected DIF.
Knob B  DIF direction  : 'balanced'  = per-prompt +/-m coin flip (control)
                         'unbalanced'= all DIF prompts +m       (worst case)
Knob C  N_ANCHORS = 40 (method property; used in phase_4, not here)
Magnitude m = 0.46 (Phase 0 median|tau|), fixed.

DIF-free prompts have tau = 0 in every language; the en column is always 0.
The balanced sign is drawn ONCE per prompt and applied to every non-en language
(literal "per-prompt coin flip"); it cancels within each language either way.

Colab: run after phase_1. Defines build_tau(), used by phase_3.
"""

import numpy as np

try:
    from phase_1 import LANGS, N_PROMPTS, M      # when run as a .py module
except ModuleNotFoundError:
    pass  # Colab: phase_1 ran as a cell above; its names are already in globals

PROPORTIONS = [0.10, 0.25, 0.40, 0.50]
DIRECTIONS  = ["balanced", "unbalanced"]


def build_tau(rng, proportion, direction, magnitude=M):
    """tau matrix (315, 10) for one condition, plus the ground-truth DIF id set.

    Returns
    -------
    tau          : (315, 10) array; column order matches phase_1.LANGS, en col = 0
    true_dif_ids : set[int] of prompt ids (0..314) carrying injected DIF
    """
    n_dif   = round(proportion * N_PROMPTS)
    dif_ids = rng.choice(N_PROMPTS, size=n_dif, replace=False)

    if direction == "unbalanced":
        signs = np.ones(n_dif)
    elif direction == "balanced":
        signs = rng.choice([-1.0, 1.0], size=n_dif)   # one flip per prompt
    else:
        raise ValueError(f"direction must be 'balanced' or 'unbalanced', got {direction!r}")

    tau = np.zeros((N_PROMPTS, len(LANGS)))
    for Li, L in enumerate(LANGS):
        if L == "en":
            continue
        tau[dif_ids, Li] = signs * magnitude

    return tau, set(int(x) for x in dif_ids)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for direction in DIRECTIONS:
        tau, dif = build_tau(rng, 0.40, direction)
        nz = np.abs(tau) > 0
        print(f"{direction:>10}: DIF prompts={len(dif)} (expect {round(0.40*N_PROMPTS)})  "
              f"nonzero cells={nz.sum()}  en col zero={np.allclose(tau[:, 0], 0)}  "
              f"unique |tau|={sorted(set(np.round(np.abs(tau[nz]), 3)))}")
        # clean prompts must be zero in every language
        clean = [i for i in range(N_PROMPTS) if i not in dif]
        assert np.allclose(tau[clean], 0.0), "clean prompt carried tau!"
