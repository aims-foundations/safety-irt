"""
Phase 4 — anchor selection on synthetic data (the test).

Drives the REAL selection code (model/anchors.py, imported — not a copy) on each
synthetic dataset, over the 4 x 2 = 8-cell grid, N_REPS replications per cell.

Per replication, three strategies are scored on the SAME candidate pool (the
variance-filter survivors), so Method / Floor / Oracle differ only in how they
pick 40 anchors from that pool:
  Floor  : 40 random candidates
  Method : the 40 candidates with the LOWEST mean Lord's chi2 (the deployed rule)
  Oracle : 40 random clean (truly DIF-free) candidates  -> contamination == 0

Metrics per cell (mean +/- SD over replications):
  1. contamination = fraction of the 40 selected that are truly DIF
  2. recovery      = 1 - contamination
  3. rank AUC      = does high mean-chi2 rank the truly-DIF prompts above clean?
  4. DIF detection = hit rate / false-alarm rate under the flag rule
                     (chi2 > chi2(.95,2) and |db| > 0.5) using Method's 40 anchors

NOTE (Colab): importing anchors triggers a one-time snapshot_download of the
safety-data dataset (module-level in anchors.py). Harmless; happens once.
Set N_REPS small first for a smoke test, then 300 for the real run.
"""

import os
import sys
import io
import warnings
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

warnings.filterwarnings("ignore")

from phase_1 import load_preserved, LANGS
from phase_2 import PROPORTIONS, DIRECTIONS
from phase_3 import generate_dataset, df_to_matrices


# ── locate and import the deployed selection code (model/anchors.py) ─────────
def _import_anchors():
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(here, "..", "..", "model")))
    except NameError:
        pass                       # __file__ undefined when pasted as a raw cell
    candidates += [
        os.environ.get("SAFETY_IRT_MODEL_DIR", ""),
        os.getcwd(),                                          # %%writefile anchors.py in cwd
        os.path.abspath(os.path.join(os.getcwd(), "model")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "..", "model")),
        "/content/safety-irt/model",                         # typical Colab git-clone path
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "anchors.py")):
            sys.path.insert(0, c)
            import anchors as _a
            return _a
    raise ImportError(
        "Could not find model/anchors.py. Set SAFETY_IRT_MODEL_DIR to the repo's "
        "model/ directory, e.g. os.environ['SAFETY_IRT_MODEL_DIR']='/content/safety-irt/model'")


anchors = _import_anchors()
N_ANCHORS      = anchors.N_ANCHORS
REFERENCE_LANG = anchors.REFERENCE_LANG
ALL_LANGS      = anchors.ALL_LANGS
CHI2_THRESH    = chi2.ppf(0.95, df=2)

# ── run config ───────────────────────────────────────────────────────────────
N_REPS           = 300     # set to e.g. 3 for a smoke test
COMPUTE_DETECTION = True    # metric 4; ~doubles per-rep cost — set False to skip
BASE_SEED        = 20240601


@contextlib.contextmanager
def _quiet():
    """Silence anchors.py's verbose prints during the 2,400-run loop."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── metric helpers ───────────────────────────────────────────────────────────
def _auc(scores, labels):
    """AUC that `scores` (higher = more DIF) ranks positives above negatives."""
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or (~labels).all():
        return np.nan
    pos = np.asarray(scores)[labels]
    neg = np.asarray(scores)[~labels]
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _detection(matrices, anchor_set, true_dif):
    """Hit / false-alarm under the flag rule, equating on `anchor_set` (40 items).

    Flags per (prompt, language) cell for non-anchor prompts; pools across langs.
    """
    mat_ref = matrices[REFERENCE_LANG]
    n_hit = n_dif = n_fa = n_clean = 0
    for lang in ALL_LANGS:
        if lang not in matrices:
            continue
        mat_foc = matrices[lang]
        if mat_foc.shape[0] < anchors.MIN_PERSONS:
            continue
        common = sorted(set(mat_ref.columns) & set(mat_foc.columns))
        anch_mask = np.array([pid in anchor_set for pid in common])
        if anch_mask.sum() < 1:
            continue

        a_r, b_r, se_a_r, se_b_r, cov_r = anchors.fit_2pl(mat_ref[common].values.astype(float))
        a_f, b_f, se_a_f, se_b_f, cov_f = anchors.fit_2pl(mat_foc[common].values.astype(float))
        A, B = anchors.mean_equating(b_r, b_f, anch_mask)
        b_f_l    = A * b_f + B
        se_b_f_l = abs(A) * se_b_f
        se_a_f_l = se_a_f / abs(A)
        stat, _ = anchors.lords_chi_square(
            a_r, b_r, se_a_r, se_b_r, cov_r, a_f, b_f_l, se_a_f_l, se_b_f_l, cov_f)
        flagged = (stat > CHI2_THRESH) & (np.abs(b_r - b_f_l) > anchors.MIN_DELTA_B)

        for idx, pid in enumerate(common):
            if pid in anchor_set:
                continue                        # anchors constrained -> excluded
            is_dif = pid in true_dif
            if is_dif:
                n_dif += 1; n_hit += int(flagged[idx])
            else:
                n_clean += 1; n_fa += int(flagged[idx])
    hit = n_hit / n_dif if n_dif else np.nan
    fa  = n_fa / n_clean if n_clean else np.nan
    return hit, fa


# ── one replication ──────────────────────────────────────────────────────────
def score_replication(beta, gamma, configs, proportion, direction, seed):
    df, true_dif_int = generate_dataset(beta, gamma, configs, proportion, direction, seed)
    true_dif = {str(i) for i in true_dif_int}          # anchors.py uses str ids
    matrices = df_to_matrices(df)
    mat_ref  = matrices[REFERENCE_LANG]

    with _quiet():
        candidate_ids, _stats, ref_b, ref_a = anchors.variance_filter(mat_ref)
        scores = anchors.compute_agreement_scores(mat_ref, matrices, candidate_ids, ref_b, ref_a)

    cand       = {str(c) for c in candidate_ids}
    clean_cand = sorted(cand - true_dif)
    # guard: the plan's "clean pool > budget" must hold WITHIN the candidate pool
    assert len(clean_cand) >= N_ANCHORS, (
        f"clean candidates {len(clean_cand)} < {N_ANCHORS} at proportion={proportion}; "
        f"variance-filter pool too depleted for a fair test")

    rng = np.random.default_rng(seed + 10_000)
    cand_list = sorted(cand)

    method_sel = set(scores.loc[scores["selected"], "prompt_id"].astype(str))
    floor_sel  = set(rng.choice(cand_list,  size=N_ANCHORS, replace=False))
    oracle_sel = set(rng.choice(clean_cand, size=N_ANCHORS, replace=False))

    def contam(sel):
        return len(sel & true_dif) / len(sel)

    rec = {
        "contam_method": contam(method_sel),
        "contam_floor":  contam(floor_sel),
        "contam_oracle": contam(oracle_sel),
        "rank_auc":      _auc(scores["mean_chi2"].to_numpy(),
                              scores["prompt_id"].astype(str).isin(true_dif).to_numpy()),
    }
    if COMPUTE_DETECTION:
        with _quiet():
            hit, fa = _detection(matrices, method_sel, true_dif)
        rec["hit_rate"], rec["false_alarm"] = hit, fa
    return rec


# ── one cell (aggregate over replications) ───────────────────────────────────
def run_cell(beta, gamma, configs, proportion, direction, n_reps=N_REPS):
    # Deterministic, session-independent seed per (proportion, direction, rep).
    # Do NOT use hash() — CPython salts string hashing with PYTHONHASHSEED, so a
    # tuple containing `direction` reseeds differently every interpreter restart.
    prop_idx = PROPORTIONS.index(proportion)
    dir_idx  = DIRECTIONS.index(direction)
    cell_base = BASE_SEED + (prop_idx * len(DIRECTIONS) + dir_idx) * 100_000
    recs = []
    for r in range(n_reps):
        seed = cell_base + r                      # unique per rep (r < 100_000)
        rec = score_replication(beta, gamma, configs, proportion, direction, seed)
        rec["seed"] = seed                        # "recorded", per the plan
        recs.append(rec)
    d = pd.DataFrame(recs)
    # seeds are contiguous (cell_base .. cell_base+n_reps-1); record the range
    # rather than a meaningless mean/sd, so every rep is exactly reproducible.
    out = {"proportion": proportion, "direction": direction, "n_reps": n_reps,
           "seed_base": cell_base, "seed_lo": cell_base, "seed_hi": cell_base + n_reps - 1}
    for col in d.columns:
        if col == "seed":
            continue
        out[f"{col}_mean"] = d[col].mean()
        out[f"{col}_sd"]   = d[col].std(ddof=1)
    return out


def run_grid(n_reps=N_REPS, token=None):
    beta, gamma, configs = load_preserved(token=token)
    rows = []
    for direction in DIRECTIONS:
        for proportion in PROPORTIONS:
            print(f"cell: proportion={proportion:>4}  direction={direction:<10} "
                  f"({n_reps} reps) ...", flush=True)
            rows.append(run_cell(beta, gamma, configs, proportion, direction, n_reps))
    res = pd.DataFrame(rows)
    cols = ["proportion", "direction",
            "contam_method_mean", "contam_method_sd",
            "contam_floor_mean", "contam_oracle_mean",
            "rank_auc_mean"]
    if COMPUTE_DETECTION:
        cols += ["hit_rate_mean", "false_alarm_mean"]
    print("\n=== RESULTS ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(res[cols].to_string(index=False))
    return res


if __name__ == "__main__":
    # smoke test: 3 reps/cell. Set N_REPS=300 (module const) for the real run.
    res = run_grid(n_reps=3)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_4_results.csv")
    res.to_csv(out, index=False)
    print(f"\nsaved: {out}")
