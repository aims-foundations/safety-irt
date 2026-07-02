"""
Phase 4 — anchor selection on synthetic data (the test).  [cached / fast path]

Drives the REAL selection statistic (model/anchors.py's fit_2pl / mean_equating /
lords_chi_square — imported, not reimplemented) on each synthetic dataset, over
the 4 x 2 = 8-cell grid, N_REPS replications per cell.

WHY THIS IS FAST *AND* FAITHFUL
-------------------------------
The deployed anchors.py refits English inside its per-language loop (9x, on the
identical matrix) and the detection metric refit both English and every focal
language again — ~37 fit_2pl calls per replication, ~15s each. Those refits are
pure redundancy: fit_2pl is deterministic, and in synthetic data every language
holds all 315 prompts, so `common` is always the full set. We therefore fit each
language ONCE (English + 9 focal = 10 fits) and derive BOTH the agreement scores
and the detection flags from the cached fits. Equating is a post-fit linear
transform (b_lnk = A*b + B), so computing agreement (equate on all candidates)
and detection (equate on the selected 40) from one fit is exactly how anchors.py
is structured — no numbers change.

verify_equivalence() proves it: it runs one replication through the real
anchors.variance_filter + compute_agreement_scores path AND the cached path and
asserts the SELECTED-40 sets are identical (the output that determines anchors).
Per-prompt chi2 can differ slightly — deployed anchors.py mixes a/b from the full
English fit with SEs from a separate `common` refit, whereas the cached path uses
one consistent English fit — but the ranked selection is unchanged. Run it once
before trusting the sweep.

Only EM_OUTER reduction (below) is an APPROXIMATION, not caching — left at the
deployed value by default. Lower it only after verify_equivalence still passes.

NOTE (Colab): importing anchors triggers a one-time snapshot_download; the T4 GPU
is unused (pure NumPy/SciPy) — prefer a high-CPU runtime.
"""

import os
# Pin BLAS to 1 thread BEFORE importing numpy, so the 8 joblib workers each get a
# core without BLAS oversubscription (the 610x315 fits get no benefit from BLAS
# threads anyway). Must run before numpy/scipy are imported anywhere.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Silence the per-worker "Fetching 483 files" HF progress spam (workers re-import anchors).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
import io
import warnings
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

warnings.filterwarnings("ignore")

try:
    from phase_1 import load_preserved, LANGS
    from phase_2 import PROPORTIONS, DIRECTIONS
    from phase_3 import generate_dataset, df_to_matrices
except ModuleNotFoundError:
    pass  # Colab: phases 1-3 ran as cells above; their names are in globals


# ── locate and import the deployed selection code (anchors.py) ───────────────
def _import_anchors():
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(here, "..", "..", "model")))
    except NameError:
        pass                       # __file__ undefined when pasted as a raw cell
    candidates += [
        os.environ.get("SAFETY_IRT_MODEL_DIR", ""),
        "/content",                                          # Colab: %%writefile anchors.py
        os.getcwd(),
        os.path.abspath(os.path.join(os.getcwd(), "model")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "..", "model")),
        "/content/safety-irt/model",
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "anchors.py")):
            sys.path.insert(0, c)
            import anchors as _a
            return _a
    raise ImportError(
        "Could not find anchors.py. Set SAFETY_IRT_MODEL_DIR or place anchors.py in /content.")


anchors = _import_anchors()
N_ANCHORS      = anchors.N_ANCHORS
REFERENCE_LANG = anchors.REFERENCE_LANG
ALL_LANGS      = anchors.ALL_LANGS
VARIANCE_LO    = anchors.VARIANCE_LO
VARIANCE_HI    = anchors.VARIANCE_HI
MIN_DELTA_B    = anchors.MIN_DELTA_B
MIN_PERSONS    = anchors.MIN_PERSONS
CHI2_THRESH    = chi2.ppf(0.95, df=2)

# ── run config ───────────────────────────────────────────────────────────────
N_REPS            = 50     # ~3 h on an 8-core CPU with the settings below
COMPUTE_DETECTION = True    # now nearly free (reuses cached fits)
BASE_SEED         = 20240601
N_JOBS            = 8       # parallel workers — set to your PHYSICAL core count
PARALLEL_BACKEND  = "loky"  # process-based (~7x). Threads give only ~1.4x here (GIL-bound).

# Number of outer EM iterations in the anchor fit. anchors.py ships with 10 — the
# deployed setting — kept here for fidelity. EM_OUTER = 3 is ~3x faster but is an
# APPROXIMATION: in offline tests it selected 0-4 of the 40 anchors differently from
# EM_OUTER=10, so use it only for quick/preliminary runs, not the reported result.
EM_OUTER = 10
anchors.EM_OUTER = EM_OUTER


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── cached fits: one fit_2pl per language on the shared prompt set ───────────
def _fit_cache(matrices):
    """Fit English + every focal language ONCE, aligned to a shared prompt order.

    Returns (common, ref, foc) where `common` is the sorted shared prompt list,
    ref/foc are dicts of arrays (a, b, se_a, se_b, cov) aligned to `common`.
    Uses anchors.fit_2pl — the real estimator — with no redundant refits.
    """
    mat_ref = matrices[REFERENCE_LANG]
    langs   = [l for l in ALL_LANGS if l in matrices and matrices[l].shape[0] >= MIN_PERSONS]

    common = sorted(set(mat_ref.columns).intersection(
        *[set(matrices[l].columns) for l in langs])) if langs else sorted(mat_ref.columns)

    def fit(mat):
        a, b, se_a, se_b, cov = anchors.fit_2pl(mat[common].values.astype(float))
        return dict(a=a, b=b, se_a=se_a, se_b=se_b, cov=cov)

    ref = fit(mat_ref)
    foc = {l: fit(matrices[l]) for l in langs}
    return common, ref, foc


def _candidates_from_ref(common, ref):
    """Reproduce anchors.variance_filter: keep prompts with sigma(-b) in the window."""
    p_safe = 1.0 / (1.0 + np.exp(np.clip(ref["b"], -30, 30)))   # sigma(-b)
    return {common[i] for i in range(len(common))
            if VARIANCE_LO < p_safe[i] < VARIANCE_HI}


def _lords_over_langs(common, ref, foc, anchor_mask):
    """Per-(prompt, language) Lord's chi2 and linked |db|, equating on anchor_mask.

    anchor_mask : bool over `common` — the provisional anchor set for equating.
    Returns dict lang -> (stat[common], abs_db[common]) using cached fits only.
    """
    out = {}
    a_r, b_r = ref["a"], ref["b"]
    se_a_r, se_b_r, cov_r = ref["se_a"], ref["se_b"], ref["cov"]
    for lang, f in foc.items():
        A, B = anchors.mean_equating(b_r, f["b"], anchor_mask)
        b_f_l    = A * f["b"] + B
        se_b_f_l = abs(A) * f["se_b"]
        se_a_f_l = f["se_a"] / abs(A)
        stat, _ = anchors.lords_chi_square(
            a_r, b_r, se_a_r, se_b_r, cov_r,
            f["a"], b_f_l, se_a_f_l, se_b_f_l, f["cov"])
        out[lang] = (stat, np.abs(b_r - b_f_l))
    return out


def _agreement_scores(common, ref, foc, candidate_ids):
    """Reproduce anchors.compute_agreement_scores from cached fits -> scores_df."""
    cand_mask = np.array([pid in candidate_ids for pid in common])
    per_lang  = _lords_over_langs(common, ref, foc, cand_mask)   # equate on all candidates

    rows = []
    for i, pid in enumerate(common):
        if pid not in candidate_ids:
            continue
        vals = [per_lang[l][0][i] for l in per_lang]
        rows.append({"prompt_id": pid, "n_languages": len(vals),
                     "mean_chi2": float(np.mean(vals))})
    scores = pd.DataFrame(rows).sort_values("mean_chi2").reset_index(drop=True)
    scores["rank"]     = scores.index + 1
    scores["selected"] = scores["rank"] <= N_ANCHORS
    return scores


def _detection(common, ref, foc, anchor_set, true_dif):
    """Hit / false-alarm under the flag rule, equating on `anchor_set` (the 40)."""
    anchor_mask = np.array([pid in anchor_set for pid in common])
    per_lang    = _lords_over_langs(common, ref, foc, anchor_mask)
    n_hit = n_dif = n_fa = n_clean = 0
    for lang, (stat, abs_db) in per_lang.items():
        flagged = (stat > CHI2_THRESH) & (abs_db > MIN_DELTA_B)
        for i, pid in enumerate(common):
            if pid in anchor_set:
                continue
            if pid in true_dif:
                n_dif += 1; n_hit += int(flagged[i])
            else:
                n_clean += 1; n_fa += int(flagged[i])
    hit = n_hit / n_dif if n_dif else np.nan
    fa  = n_fa / n_clean if n_clean else np.nan
    return hit, fa


# ── one replication (cached) ─────────────────────────────────────────────────
def score_replication(beta, gamma, configs, proportion, direction, seed):
    df, true_dif_int = generate_dataset(beta, gamma, configs, proportion, direction, seed)
    true_dif = {str(i) for i in true_dif_int}
    matrices = df_to_matrices(df)

    common, ref, foc = _fit_cache(matrices)          # 10 fits total
    candidate_ids    = _candidates_from_ref(common, ref)
    scores           = _agreement_scores(common, ref, foc, candidate_ids)

    clean_cand = sorted(candidate_ids - true_dif)
    assert len(clean_cand) >= N_ANCHORS, (
        f"clean candidates {len(clean_cand)} < {N_ANCHORS} at proportion={proportion}; "
        f"variance-filter pool too depleted for a fair test")

    rng = np.random.default_rng(seed + 10_000)
    cand_list  = sorted(candidate_ids)
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
        hit, fa = _detection(common, ref, foc, method_sel, true_dif)
        rec["hit_rate"], rec["false_alarm"] = hit, fa
    return rec


def _auc(scores, labels):
    """AUC that `scores` (higher = more DIF) ranks positives above negatives."""
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or (~labels).all():
        return np.nan
    pos = np.asarray(scores)[labels]
    neg = np.asarray(scores)[~labels]
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ── equivalence check: cached path vs the deployed anchors.py path ───────────
def verify_equivalence(proportion=0.25, direction="unbalanced", seed=BASE_SEED):
    """Prove caching selects the SAME anchors as the deployed anchors.py path.

    The criterion is the selected-40 set — the actual output that determines the
    anchors. Per-prompt chi2 is reported for transparency but NOT asserted to
    machine precision: deployed anchors.py takes a/b from the full English fit
    but the SEs from a separate refit on `common`; the cached path uses one
    consistent English fit, so chi2 can differ slightly (more so when the fit is
    under-converged) without ever changing the ranked selection.
    """
    beta, gamma, configs = load_preserved()
    df, _ = generate_dataset(beta, gamma, configs, proportion, direction, seed)
    matrices = df_to_matrices(df)

    with _quiet():                                    # deployed path
        cand, _s, ref_b, ref_a = anchors.variance_filter(matrices[REFERENCE_LANG])
        sc_dep = anchors.compute_agreement_scores(matrices[REFERENCE_LANG], matrices,
                                                  cand, ref_b, ref_a)
    sel_dep = set(sc_dep.loc[sc_dep["selected"], "prompt_id"].astype(str))

    common, ref, foc = _fit_cache(matrices)           # cached path
    sc_cac  = _agreement_scores(common, ref, foc, _candidates_from_ref(common, ref))
    sel_cac = set(sc_cac.loc[sc_cac["selected"], "prompt_id"].astype(str))

    m = (sc_dep.set_index("prompt_id")["mean_chi2"]
         .reindex(sc_cac["prompt_id"]).to_numpy())
    max_diff = np.nanmax(np.abs(m - sc_cac["mean_chi2"].to_numpy()))
    same = sel_dep == sel_cac
    print(f"selected-40 identical: {same}  (differ by {len(sel_dep ^ sel_cac)})  "
          f"|  max |Δ mean_chi2| = {max_diff:.2e} (diagnostic only)")
    assert same, (f"selection DIFFERS by {len(sel_dep ^ sel_cac)} anchors — investigate "
                  f"before trusting the cached path (did EM_OUTER change convergence?)")
    print("OK: cached path selects the same anchors as deployed anchors.py.")
    return True


# ── one cell (deterministic seeds; optional threaded parallelism) ────────────
def run_cell(beta, gamma, configs, proportion, direction, n_reps=N_REPS, n_jobs=N_JOBS):
    prop_idx  = PROPORTIONS.index(proportion)
    dir_idx   = DIRECTIONS.index(direction)
    cell_base = BASE_SEED + (prop_idx * len(DIRECTIONS) + dir_idx) * 100_000
    seeds     = [cell_base + r for r in range(n_reps)]

    def one(seed):
        rec = score_replication(beta, gamma, configs, proportion, direction, seed)
        rec["seed"] = seed
        return rec

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        # verbose=10 prints a live "Done N tasks" line as reps finish within the cell
        recs = Parallel(n_jobs=n_jobs, backend=PARALLEL_BACKEND, verbose=10)(
            delayed(one)(s) for s in seeds)
    else:
        recs = [one(s) for s in seeds]

    d = pd.DataFrame(recs)
    out = {"proportion": proportion, "direction": direction, "n_reps": n_reps,
           "seed_base": cell_base, "seed_lo": cell_base, "seed_hi": cell_base + n_reps - 1}
    for col in d.columns:
        if col == "seed":
            continue
        out[f"{col}_mean"] = d[col].mean()
        out[f"{col}_sd"]   = d[col].std(ddof=1)
    return out


def _results_path():
    try:
        d = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        d = os.getcwd()                        # Colab cell: __file__ undefined
    return os.path.join(d, "phase_4_results.csv")


def run_grid(n_reps=N_REPS, n_jobs=N_JOBS, token=None, results_path=None, resume=True):
    """Run the 8-cell grid, checkpointing to CSV after EACH cell.

    Incremental save survives a Colab disconnect: rerun with resume=True (default)
    and completed cells are skipped, picking up where it left off. Pass resume=False
    (or delete the CSV) to start fresh.
    """
    if results_path is None:
        results_path = _results_path()
    beta, gamma, configs = load_preserved(token=token)

    rows, done = [], set()
    if resume and os.path.exists(results_path):
        rows = pd.read_csv(results_path).to_dict("records")
        done = {(round(float(r["proportion"]), 3), str(r["direction"])) for r in rows}
        if done:
            print(f"resuming: {len(done)}/{len(PROPORTIONS)*len(DIRECTIONS)} cells already "
                  f"done -> {sorted(done)}")

    for direction in DIRECTIONS:
        for proportion in PROPORTIONS:
            if (round(proportion, 3), direction) in done:
                print(f"skip (done): proportion={proportion:>4}  direction={direction}")
                continue
            print(f"cell: proportion={proportion:>4}  direction={direction:<10} "
                  f"({n_reps} reps, n_jobs={n_jobs}) ...", flush=True)
            try:
                rows.append(run_cell(beta, gamma, configs, proportion, direction, n_reps, n_jobs))
            except Exception as e:
                # e.g. clean-pool guard tripping at high proportion — record and go on,
                # so one bad cell doesn't discard hours of completed work.
                print(f"  !! cell FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append({"proportion": proportion, "direction": direction,
                             "n_reps": n_reps, "error": str(e)})
            pd.DataFrame(rows).to_csv(results_path, index=False)     # checkpoint
            print(f"  checkpointed {len(rows)}/{len(PROPORTIONS)*len(DIRECTIONS)} "
                  f"-> {results_path}", flush=True)

    res = (pd.DataFrame(rows)
           .sort_values(["direction", "proportion"]).reset_index(drop=True))
    cols = ["proportion", "direction",
            "contam_method_mean", "contam_method_sd",
            "contam_floor_mean", "contam_oracle_mean", "rank_auc_mean",
            "hit_rate_mean", "false_alarm_mean", "error"]
    cols = [c for c in cols if c in res.columns]   # tolerate errored/partial cells
    print("\n=== RESULTS ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(res[cols].to_string(index=False))
    return res


if __name__ == "__main__":
    # 1) prove the cached path selects the same anchors as the deployed anchors.py
    verify_equivalence()
    # 2) full run: N_REPS x 8 cells, checkpointed after each cell.
    #    Interrupted? Just run this file again — it resumes from the last checkpoint.
    res = run_grid(n_reps=N_REPS, n_jobs=N_JOBS)   # resume=True by default
    print(f"\nsaved: {_results_path()}")
