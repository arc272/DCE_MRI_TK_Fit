from __future__ import annotations
#!/usr/bin/env python3
"""
stage2_hierarchical_fit.py
==========================
Lunn et al. (2013) two-stage hierarchical fit -- STAGE 2 -- on the clean
subset of a completed stage-1 run:

    TCXM + fixed delay + SSM3 (tau_i free, tau_b fixed at 5 ms)
    theta_i = (logit ve, logit vp, log fp, log ps, log tau_i)
    theta_i ~ N(X_i B, Sigma),   X_i = [1, malignant_i (, size_i)]
    B ~ N(0, TAU0^2),  Sigma ~ InverseWishart(NU0, PSI0)

so B[0] is the benign population mean and B[1] is malignant - benign.

Steps -- each prints what it did, so the log is the record of the run:

  1. INCLUSION RULE   the clean set, with the attrition at every filter and
                      the reason each lesion was dropped (inclusion.csv)
  2. STAGE-1 DRAWS    population scale, evenly thinned to K_LOAD per lesion,
                      plus log p_1 at every draw. Every K used below is an
                      even subsample of that one set, so the K grid is
                      NESTED and differences across K are not sampling noise
                      in which draws were kept.
  3. PRE-FLIGHT       log p_1 against the package's own prior objects, at
                      REAL saved draws. Stops the run if they disagree:
                      a wrong p_1 is the one silent failure stage 2 has.
  4. DESIGN           intercept + malignant indicator (+ optional log size)
  5. STAGE 2          hierarchical_lunn.fit_stage2, several seeds
  6. DIAGNOSTICS      per-lesion acceptance and distinct draws; Kish weight
                      ESS at the FITTED population (the honest version of
                      check_20); seed-to-seed agreement; R-hat across seeds
  7. RESULTS          benign mean, malignant - benign, population SDs and
                      correlations, on the population scale and in natural
                      units; per-lesion stage-1 vs stage-2 (shrunken) medians
  8. K-SENSITIVITY    the whole fit repeated over K_GRID at cheaper chain
                      settings, with the Monte-Carlo error of each number, so
                      a drift in a population SD can be told from noise.
                      This is the test of the finite-K approximation: stage 2
                      is exact only as K -> infinity.
  9. DEGENERATE SET   refit with the lesions whose Kish ESS < KISH_MIN
                      dropped, against the same fit with them in. Tells you
                      whether the weight-starved jars drive any conclusion.
 11. PPC              posterior predictive check of the STAGE-2 model only:
                      are the theta_i Gaussian around X_i B with covariance
                      Sigma? Statistics live on z = chol(Sigma)^-1
                      (theta - X B), which is iid N(0, I) if the population
                      model holds, so the reference is exact and nothing has
                      to be simulated from the fitted population. No lesion
                      curve is read and no forward model is run: this says
                      nothing about whether stage 1's TCXM + SSM3 describes
                      any lesion. Repeated on the Kish >= KISH_MIN subset,
                      because a weight-starved lesion cannot move among its
                      saved draws and so shrinks sd(z) for a reason that has
                      nothing to do with the population model.

 10. DERIVED          Ktrans and the extraction fraction E = 1 - exp(-PS/fp),
                      computed PER DRAW so the fp-PS posterior correlation is
                      carried through, plus the median WITHIN-lesion
                      correlation matrix against the fitted POPULATION one.
                      If the population correlations merely reproduce the
                      within-lesion TCXM ridge, they are not between-lesion
                      biology.

Plots (PLOTS = True; each ends in plt.show()):
    fig 1  posterior of the group difference, per parameter, + the tau_i
           population against the stage-1 ceiling
    fig 2  K-sensitivity: M - B and population SD against K
    fig 3  weight diagnostics and the degenerate-set forest plot
    fig 4  Ktrans / E contrasts and within-lesion vs population correlation
    fig 5  shrinkage: stage-1 median against stage-2 median, per lesion
    fig 6  benign vs malignant populations as a density forest plot: the
           posterior of each group's population median (upper), and the
           fitted lesion-to-lesion distribution (lower)
    fig 7  posterior predictive check of the stage-2 population model:
           per-coordinate QQ plots of z against the normal it should follow,
           the joint Mahalanobis check, every statistic standardised, the
           p-values, and per-lesion RMS |z| against Kish ESS

UNITS.  Every plot axis for ve, vp, fp, ps and tau_i is LINEAR and in natural
units -- ve and vp dimensionless, fp and ps in 1/min, tau_i in s (or k_io in
1/s if TAU_I_AS_RATE). The FIT is untouched: it still runs on
(logit ve, logit vp, log fp, log ps, log tau_i), which is where the Gaussian
population model lives. Only the display changes, and every natural-unit
number is formed by transforming each posterior draw and then summarising --
never by transforming a summary, which would be a different quantity.
Because the transform is nonlinear, these are population MEDIANS, not means.

Decide the inclusion rule in the SETTINGS block BEFORE looking at the
stage-2 output, and do not tune it afterwards. Dropping lesions whose fp
mixes badly is not dropping at random -- it plausibly removes one region of
parameter space -- so the honest report is this rule, plus a rerun with the
excluded lesions put back (set POP_ESS_MIN = 0, REQUIRE_QC_PASS = False).

Note on the degeneracy flag: the old rule (n_distinct < 20) is RUN-LENGTH
DEPENDENT -- at 100k draws no lesion can fail it however starved its weights
are. Kish ESS is the length-independent measure and is what KISH_MIN uses.

Known library issue handled here: fit_stage2 does not pass `names` into
Stage2Result, which defaults to the four 4-D names, so summary() silently
drops tau_i. This script sets res.names after every fit.
"""


import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from dce_mri import hierarchical_lunn as hl

# =============================================================================
# SETTINGS
# =============================================================================
RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
           "run_20261021_160427")
OUT_DIR = None                     # None -> RUN_DIR/stage2_<timestamp>

PARAMS = ("ve", "vp", "fp", "ps", "tau_i")
THETA_NAMES = ("logit_ve", "logit_vp", "log_fp", "log_ps", "log_tau_i")

# --- MUST match the stage-1 run (fit_lesions_bayes_wx_fixedtau.py) ----------
PRIOR_CHOICES = {"rates": "uniform", "volfrac": "flat_masked", "tau": "flat"}
TAU_I_BOUNDS = (0.01, 1.5)         # s, relax_cfg.tau_i_lb / tau_i_ub
TAU_B_FIXED = 0.005                # s, only used to rebuild the prior objects

# --- inclusion rule: fix this BEFORE looking at stage-2 results --------------
REQUIRE_QC_PASS = True             # the stage-1 run's own qc_pass flag
POP_ESS_MIN = 400                  # min bulk ESS over PARAMS, POPULATION scale
EXCLUDE = ()                       # manual exclusions, by subject_id, with a
                                   # reason recorded in EXCLUDE_REASON
EXCLUDE_REASON = "manual"

# --- stage 2 (the reference fit) ---------------------------------------------
K = 3000                           # draws per lesion handed to stage 2
ADD_SIZE_COVARIATE = False         # adds standardised log(lesion_nvox) to X
N_TUNE, N_DRAWS = 50000, 50000
THETA_SWEEPS = 16                  # MH passes over lesions per Gibbs sweep
SEEDS = (1, 2, 3)                  # >= 2 for R-hat across seeds
TAU0 = 10.0                        # prior SD of every B coefficient
NU0 = None                         # None -> D + 2
PSI0 = None                        # None -> identity

# --- step 8: K-sensitivity ---------------------------------------------------
# Run at CHEAPER chain settings than the reference fit: the point is the
# difference between K values, and all K values share the run length, so a
# shorter chain costs precision equally everywhere. Cost is roughly
# len(K_GRID) * len(SENS_SEEDS) * (SENS_N_TUNE + SENS_N_DRAWS) / (N_TUNE +
# N_DRAWS) times one reference seed.
RUN_K_SENSITIVITY = True
K_GRID = (1500, 3000, 10000)       # K must appear here for step 9 to reuse it
SENS_N_TUNE, SENS_N_DRAWS = 20000, 40000
SENS_SEEDS = (1, 2, 3)

# --- step 9: degenerate-lesion sensitivity -----------------------------------
RUN_DEGENERATE_SENSITIVITY = True
KISH_MIN = 30.0                    # Kish ESS below this = weight-starved jar

# --- step 10: derived quantities ---------------------------------------------
RUN_DERIVED = True
DERIVED_THIN = 5000                # draws used for the per-lesion Ktrans pass

# --- step 11: posterior predictive check -------------------------------------
# Checks the STAGE-2 model only -- the Gaussian population
# theta_i ~ N(X_i B, Sigma) -- and never re-enters stage 1: no lesion curve is
# read, no forward model is run. Everything comes from the stage-2 draws
# already in memory, so it costs a second or two.
RUN_PPC = True
PPC_DRAWS = 2000                   # stage-2 draws used; evenly thinned
PPC_SEED = 20261027

# --- plots -------------------------------------------------------------------
PLOTS = True
SAVE_PLOTS = True                  # also write PNGs into OUT_DIR

PREFLIGHT = True                   # step 3; do not switch off for real runs
PREFLIGHT_LESIONS = 10
PREFLIGHT_DRAWS = 300

RATE_TO_PER_MIN = 60.0             # fp, ps are stored in s^-1
WORST_N = 12

# tau_i is stored as a residence TIME in seconds (stage-1 bounds 0.01-1.5 s).
# False -> plots and the natural-units table show tau_i [s].
# True  -> they show the exchange RATE k_io = 1 / tau_i [1/s] instead, which
#          is the convention Li et al. use. Nothing about the FIT changes:
#          every number is still computed on the log tau_i modelling scale and
#          only the display transform differs, so the stage-1 ceiling of 1.5 s
#          simply becomes a floor of 1/1.5 = 0.67 s^-1.
TAU_I_AS_RATE = False

# =============================================================================
D = len(PARAMS)
SCALE = ("logit", "logit", "log", "log", "log")
K_LOAD = max((K,) + tuple(K_GRID)) if RUN_K_SENSITIVITY else K


def _bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def to_theta5(X):
    """(..., 5) constrained (ve, vp, fp, ps, tau_i) -> population scale."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for j, s in enumerate(SCALE):
        x = X[..., j]
        out[..., j] = (np.log(x) - np.log1p(-x)) if s == "logit" else np.log(x)
    return out


def natural(theta, j):
    """
    Population-scale value -> natural units, elementwise and monotone:
        ve, vp   sigmoid           (volume fraction, dimensionless)
        fp, ps   exp x 60          [1/min]   (stored in s^-1)
        tau_i    exp               [s]       -- or 1/exp = k_io [1/s] when
                                                TAU_I_AS_RATE
    Monotone (decreasing for the k_io case), so quantiles map through it
    directly; a mean does not, which is why everything below transforms
    PER DRAW and then takes the median.
    """
    theta = np.asarray(theta, float)
    if SCALE[j] == "logit":
        return 1.0 / (1.0 + np.exp(-theta))
    if PARAMS[j] in ("fp", "ps"):
        return np.exp(theta) * RATE_TO_PER_MIN
    if PARAMS[j] == "tau_i" and TAU_I_AS_RATE:
        return np.exp(-theta)
    return np.exp(theta)


NAT_LABEL = ("ve", "vp", "fp [1/min]", "ps [1/min]",
             "k_io = 1/tau_i [1/s]" if TAU_I_AS_RATE else "tau_i [s]")
NAT_SHORT = ("ve", "vp", "fp", "ps", "k_io" if TAU_I_AS_RATE else "tau_i")


def nat_group_draws(B, j):
    """
    Per-draw benign and malignant POPULATION MEDIANS in natural units.

    natural() is monotone, so natural(population mean on the modelling scale)
    is the population MEDIAN in natural units -- not its mean. Said that way
    in every label, because for a log-normal the two differ by exp(sd^2/2).
    """
    return natural(B[:, 0, j], j), natural(B[:, 0, j] + B[:, 1, j], j)


def nat_diff_draws(B, j):
    """Per-draw malignant - benign difference of population medians, natural units."""
    b, m = nat_group_draws(B, j)
    return m - b


def nat_spread_draws(B, SD, j, q=1.645):
    """
    Per-draw lesion-to-lesion 5-95% interval of each group, natural units:
    the modelling-scale interval mean +/- 1.645 sd pushed through natural().
    For k_io the transform is decreasing, so the ends are sorted back.
    """
    lo_b, hi_b = (natural(B[:, 0, j] - q * SD[:, j], j),
                  natural(B[:, 0, j] + q * SD[:, j], j))
    mu_m = B[:, 0, j] + B[:, 1, j]
    lo_m, hi_m = (natural(mu_m - q * SD[:, j], j),
                  natural(mu_m + q * SD[:, j], j))
    return (np.minimum(lo_b, hi_b), np.maximum(lo_b, hi_b),
            np.minimum(lo_m, hi_m), np.maximum(lo_m, hi_m))


def make_log_p1(choices, lb4, ub4, tau_bounds):
    """
    Stage-1 prior density on the 5-D population scale, from constrained draws.
    First four coordinates: hl.log_p1_theta exactly. tau_i under "flat"
    (log-uniform): p_x(tau) ~ 1/tau, times the Jacobian dtau/dlog tau = tau,
    is constant -- so it adds nothing, but out-of-bounds draws still get -inf.
    Same algebra as check_17's log_p1_theta5, which T0 verified to 1e-14.
    """
    if choices.get("tau", "flat") != "flat":
        raise ValueError("only the 'flat' (log-uniform) tau prior is handled; "
                         f"got {choices.get('tau')!r}")
    base4 = hl.log_p1_theta(choices, lb4, ub4)
    tlo, thi = tau_bounds

    def log_p1(X):
        X = np.asarray(X, float)
        out = base4(X[..., :4])
        t = X[..., 4]
        return np.where((t >= tlo) & (t <= thi), out, -np.inf)

    return log_p1


def ess_bulk_tail(x):
    """x: (n_chain, n_draw) -> (bulk, tail) ESS."""
    import arviz as az
    bulk = float(az.ess(x, method="bulk"))
    try:                                   # arviz < 1.0
        tail = float(az.ess(x, method="tail"))
    except TypeError:                      # arviz >= 1.0 requires prob
        tail = float(az.ess(x, method="tail", prob=(0.05, 0.95)))
    return bulk, tail


def kish(logw):
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def even_idx(n_from, n_to):
    """Even-stride indices selecting n_to of n_from, endpoints included."""
    if n_to >= n_from:
        return np.arange(n_from)
    return np.linspace(0, n_from - 1, n_to).round().astype(int)


def subsample_s1(s1, k):
    """Nested even subsample of a Stage1Samples down to k draws per lesion."""
    if k >= s1.draws.shape[1]:
        return s1
    idx = even_idx(s1.draws.shape[1], k)
    return hl.Stage1Samples(draws=s1.draws[:, idx, :],
                            log_p1=s1.log_p1[:, idx], sids=s1.sids)


def subset_s1(s1, mask):
    """Keep the lesions where mask is True."""
    mask = np.asarray(mask, bool)
    return hl.Stage1Samples(draws=s1.draws[mask], log_p1=s1.log_p1[mask],
                            sids=tuple(np.asarray(s1.sids, dtype=object)[mask]))


# =============================================================================
# step 1: inclusion rule
# =============================================================================
def load_table(run_dir):
    """lesion_bayes_results.csv and rows/*.csv, one row per subject_id,
    preferring an error-free row where one exists."""
    frames = []
    p = os.path.join(run_dir, "lesion_bayes_results.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
    rows = sorted(glob.glob(os.path.join(run_dir, "rows", "*.csv")))
    if rows:
        frames.append(pd.concat([pd.read_csv(f) for f in rows], ignore_index=True))
    if not frames:
        raise FileNotFoundError(f"no lesion_bayes_results.csv or rows/*.csv in {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    if "error" not in df:
        df["error"] = ""
    df["error"] = df["error"].fillna("").astype(str).str.strip()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["_ok"] = df["error"] == ""
    df = (df.sort_values("_ok", kind="stable")
            .drop_duplicates("subject_id", keep="last")
            .drop(columns="_ok").reset_index(drop=True))
    return df


def step1_inclusion(run_dir):
    print("=" * 78)
    print("1.  INCLUSION RULE")
    print("=" * 78)
    df = load_table(run_dir)
    log = {s: "" for s in df["subject_id"]}
    print(f"    {len(df):4d} lesions in the results table")

    def drop(mask, reason):
        hit = [s for s in df.loc[mask, "subject_id"] if not log[s]]
        for s in hit:
            log[s] = reason
        kept = [s for s in df["subject_id"] if not log[s]]
        print(f"    {len(kept):4d} after dropping {len(hit):3d}  {reason}")

    drop(df["error"] != "", "stage-1 error")
    dx = df.get("diagnosis", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
    df["diagnosis"] = dx
    drop(~dx.isin(["B", "M"]), "diagnosis not B/M")
    has_npz = df["subject_id"].map(
        lambda s: os.path.exists(os.path.join(run_dir, f"{s}_draws.npz")))
    drop(~has_npz, "no _draws.npz")
    if REQUIRE_QC_PASS:
        if "qc_pass" in df:
            drop(~df["qc_pass"].map(_bool), "stage-1 qc_pass False")
        else:
            print("    !! REQUIRE_QC_PASS set but no qc_pass column -- not applied")

    # population-scale ESS needs the draws: load the survivors
    draws, ess_info = {}, {}
    for s in [s for s in df["subject_id"] if not log[s]]:
        with np.load(os.path.join(run_dir, f"{s}_draws.npz"), allow_pickle=True) as z:
            names = [str(n) for n in z["param_names"]]
            miss = [p for p in PARAMS if p not in names]
            if miss:
                log[s] = f"draws missing {miss}"
                continue
            ch = np.asarray(z["chains"], float)[:, :, [names.index(p) for p in PARAMS]]
            nvox = float(z["lesion_nvox"]) if "lesion_nvox" in z else np.nan
        th = to_theta5(ch)
        if not np.all(np.isfinite(th)):
            log[s] = "non-finite draw on the population scale"
            continue
        eb = [ess_bulk_tail(th[:, :, j]) for j in range(D)]
        ess_info[s] = dict(bulk=[e[0] for e in eb], tail=[e[1] for e in eb])
        # keep only K_LOAD thinned draws: the full arrays for 170+ lesions are
        # ~1 GB. Even stride over the chain-major flattening, so every chain
        # contributes.
        n_raw = ch.shape[0] * ch.shape[1]
        if n_raw < K_LOAD:
            log[s] = f"only {n_raw} draws, K_LOAD = {K_LOAD}"
            continue
        idx = even_idx(n_raw, K_LOAD)
        draws[s] = (ch.reshape(-1, D)[idx], th.reshape(-1, D)[idx], nvox)
        del ch, th
    pop_min = {s: min(v["bulk"]) for s, v in ess_info.items()}
    drop(df["subject_id"].map(lambda s: s in pop_min and pop_min[s] < POP_ESS_MIN),
         f"population-scale min bulk ESS < {POP_ESS_MIN}")
    drop(df["subject_id"].isin(EXCLUDE), f"EXCLUDE ({EXCLUDE_REASON})")

    kept = [s for s in df["subject_id"] if not log[s]]
    dxk = df.set_index("subject_id").loc[kept, "diagnosis"]
    print(f"    clean set: {len(kept)} lesions  "
          f"({int((dxk == 'B').sum())} benign, {int((dxk == 'M').sum())} malignant)")

    inc = df[["subject_id", "diagnosis"]].copy()
    inc["included"] = inc["subject_id"].map(lambda s: not log[s])
    inc["reason"] = inc["subject_id"].map(log)
    for j, p in enumerate(PARAMS):
        inc[f"{p}_pop_ess_bulk"] = inc["subject_id"].map(
            lambda s: ess_info[s]["bulk"][j] if s in ess_info else np.nan)
        inc[f"{p}_pop_ess_tail"] = inc["subject_id"].map(
            lambda s: ess_info[s]["tail"][j] if s in ess_info else np.nan)
    dropped = inc[~inc["included"]]
    if len(dropped):
        print("    dropped:")
        for _, r in dropped.iterrows():
            print(f"      {r['subject_id']:<10} {r['diagnosis']:<2} {r['reason']}")
    return kept, dxk.to_dict(), {s: draws[s] for s in kept}, inc


# =============================================================================
# step 2: stage-1 draws -> Stage1Samples
# =============================================================================
def step2_stage1(kept, draws, log_p1):
    print()
    print("=" * 78)
    print(f"2.  STAGE-1 DRAWS -> population scale, thinned to K_LOAD = {K_LOAD}")
    print("=" * 78)
    L = len(kept)
    TH = np.empty((L, K_LOAD, D))
    LP = np.empty((L, K_LOAD))
    XC = np.empty((L, K_LOAD, D))
    for i, s in enumerate(kept):
        XC[i], TH[i], _ = draws[s]            # already thinned in step 1
        LP[i] = log_p1(XC[i])
    bad = ~np.isfinite(LP)
    if bad.any():
        rows = np.unique(np.where(bad)[0])
        raise ValueError(f"{int(bad.sum())} draws outside the stage-1 bounds in "
                         f"{[kept[r] for r in rows]} -- the bounds or prior "
                         "choices here do not match the stage-1 run")
    print(f"    {L} lesions x {K_LOAD} draws x {D} coordinates")
    print(f"    within-lesion range of log p_1: median "
          f"{np.median(np.ptp(LP, axis=1)):.2f}  max {np.max(np.ptp(LP, axis=1)):.2f}")
    print("    (0 would mean the correction is inert; uniform rates make it live)")
    if K_LOAD != K:
        print(f"    the reference fit uses an even subsample of K = {K} of these,")
        print(f"    and every K in the grid is nested in the same {K_LOAD} draws")
    s1 = hl.Stage1Samples(draws=TH, log_p1=LP, sids=tuple(kept))
    return s1, XC


# =============================================================================
# step 3: pre-flight -- log p_1 against the package's prior objects
# =============================================================================
def step3_preflight(XC, kept, log_p1):
    print()
    print("=" * 78)
    print("3.  PRE-FLIGHT: log p_1 vs the package's own prior, at REAL draws")
    print("=" * 78)
    from dce_mri.bayes import BayesConfig, build_param_layout, NOISE_OPTIONS
    from dce_mri.bayes_priors import assemble_prior
    from dce_mri.synthetic import make_parker_aif
    from dce_mri.kinetic_models import TCXM
    from dce_mri.water_exchange import SSM3
    from dce_mri.config import AcquisitionConfig, RelaxationConfig

    relax = RelaxationConfig(fit_tau_b=False, tau_b_fixed=TAU_B_FIXED,
                             tau_i_lb=TAU_I_BOUNDS[0], tau_i_ub=TAU_I_BOUNDS[1])
    acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
    # the priors depend only on bounds and choices; the AIF/time grid are
    # needed to build a layout and do not enter the prior density
    tt = np.arange(36) * 150 * 8 / 288
    aif = make_parker_aif(tt, t_arrival=35.0, hematocrit=0.4)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    cb = BayesConfig(prior_choices=PRIOR_CHOICES, noise="gauss_marginalized_sigma",
                     delay_mode="fixed", delay_fixed=5.0, delay_bounds=(0.0, 25.0))
    lay = build_param_layout(TCXM, cb, aif=aif, time_var=tt, acq=acq,
                             wx_model=SSM3, relax_cfg=relax)
    pr = assemble_prior(lay.model, relax, wx_model=SSM3, delay_mode="fixed",
                        n_noise=noise.n_params, noise_names=noise.param_names,
                        choices=PRIOR_CHOICES, delay_bounds=(0.0, 25.0))
    names = list(lay.param_names)
    if names[:D] != list(PARAMS):
        raise AssertionError(f"layout samples {names}, expected {list(PARAMS)} first")
    blocks = [b for b in pr.blocks if b.name in ("volfrac", "rates", "tau")]
    print(f"    prior blocks: {[b.name for b in blocks]}   choices {PRIOR_CHOICES}")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(kept), size=min(PREFLIGHT_LESIONS, len(kept)), replace=False)
    worst, n_inf = 0.0, 0
    nd = min(PREFLIGHT_DRAWS, XC.shape[1])
    for i in pick:
        X = XC[i][rng.choice(XC.shape[1], size=nd, replace=False)]
        ref = np.array([sum(b.log_pdf(x[list(b.indices)]) for b in blocks) for x in X])
        jac = (np.log(X[:, 0]) + np.log1p(-X[:, 0]) + np.log(X[:, 1])
               + np.log1p(-X[:, 1]) + np.log(X[:, 2]) + np.log(X[:, 3])
               + np.log(X[:, 4]))
        mine = log_p1(X)
        n_inf += int((~np.isfinite(ref)).sum())
        d = mine - (ref + jac)
        d = d[np.isfinite(d)]
        if len(d):
            worst = max(worst, float(d.max() - d.min()))
    print(f"    {len(pick)} lesions x {nd} real draws")
    print(f"    real draws the PACKAGE prior calls impossible: {n_inf}")
    print(f"    spread of (this script's log p_1) - (package + Jacobian): {worst:.2e}")
    ok = (n_inf == 0) and (worst < 1e-8)
    print(f"    -> {'PASS: same density up to a constant' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("pre-flight failed: PRIOR_CHOICES / bounds here do not "
                         "match the prior the stage-1 run used. Fix before fitting.")


# =============================================================================
# step 4: design
# =============================================================================
def step4_design(kept, dx, draws):
    print()
    print("=" * 78)
    print("4.  DESIGN")
    print("=" * 78)
    y = np.array([1.0 if dx[s] == "M" else 0.0 for s in kept])
    cols, names = [np.ones(len(kept)), y], ["benign", "malignant_minus_benign"]
    if ADD_SIZE_COVARIATE:
        lv = np.log(np.array([draws[s][2] for s in kept], float))
        if not np.all(np.isfinite(lv)):
            raise ValueError("lesion_nvox missing or non-positive for some lesions")
        cols.append((lv - lv.mean()) / lv.std())
        names.append("log_size_z")
        print("    size covariate ON: B[0] is the benign mean at the cohort's "
              "average log size")
    X = np.column_stack(cols)
    print(f"    X: {X.shape}  columns {names}")
    print("    B[0] = benign population mean, B[1] = malignant - benign")
    return X, tuple(names)


# =============================================================================
# step 5: stage 2
# =============================================================================
def fit_many(s1, X, coef_names, n_tune, n_draws, seeds, out_dir=None,
             tag="", verbose=True):
    """Run fit_stage2 once per seed. Saves npz only when out_dir is given."""
    fits = []
    for seed in seeds:
        t0 = time.time()
        r = hl.fit_stage2(s1, design=X, coef_names=coef_names, n_tune=n_tune,
                          n_draws=n_draws, seed=seed, tau0=TAU0, nu0=NU0,
                          Psi0=PSI0, theta_sweeps=THETA_SWEEPS)
        r.names = THETA_NAMES                 # library defect: see header
        el = time.time() - t0
        if verbose:
            print(f"    {tag}seed {seed}: {el:6.1f} s   acceptance "
                  f"{r.accept.min():.3f}-{r.accept.max():.3f} "
                  f"(median {np.median(r.accept):.3f})   "
                  f"distinct {r.n_distinct.min()}-{r.n_distinct.max()}")
        if out_dir is not None:
            np.savez_compressed(os.path.join(out_dir, f"stage2_seed{seed}.npz"),
                                B=r.B, Sigma=r.Sigma, theta_idx=r.theta_idx,
                                accept=r.accept, n_distinct=r.n_distinct,
                                sids=np.array(s1.sids, dtype=object),
                                names=np.array(THETA_NAMES, dtype=object),
                                coef_names=np.array(coef_names, dtype=object),
                                design=X)
        fits.append(r)
    return fits


def step5_fit(s1, X, coef_names, out_dir):
    print()
    print("=" * 78)
    print(f"5.  STAGE 2  ({N_TUNE} tune + {N_DRAWS} draws, theta_sweeps "
          f"{THETA_SWEEPS}, seeds {SEEDS}, K = {s1.draws.shape[1]})")
    print("=" * 78)
    return fit_many(s1, X, coef_names, N_TUNE, N_DRAWS, SEEDS, out_dir=out_dir)


# =============================================================================
# shared summarisers
# =============================================================================
def kish_fitted(fits, s1, X):
    """Kish weight ESS per lesion at the FITTED population (group-specific)."""
    B_med = np.median(np.concatenate([f.B for f in fits]), axis=0)      # (P, D)
    S_med = np.median(np.concatenate([f.Sigma for f in fits]), axis=0)  # (D, D)
    Linv = np.linalg.inv(np.linalg.cholesky(S_med))
    m = X @ B_med
    out = np.empty(s1.draws.shape[0])
    for i in range(s1.draws.shape[0]):
        z = (s1.draws[i] - m[i]) @ Linv.T
        out[i] = kish(-0.5 * np.einsum("kd,kd->k", z, z) - s1.log_p1[i])
    return out


def summarize_fits(fits, label=""):
    """One row per coordinate: benign mean, M - B with 90% CrI and MCSE,
    population SD with MCSE, and P(M>B). MCSE is sd / sqrt(bulk ESS), with the
    ESS computed across seeds, so it is the Monte-Carlo error of the reported
    number -- the yardstick for whether a change between runs is real."""
    import arviz as az
    B = np.concatenate([f.B for f in fits])
    SDp = np.concatenate([f.sd() for f in fits])
    rows = []
    for j, nm in enumerate(THETA_NAMES):
        d = B[:, 1, j]
        d_chains = np.stack([f.B[:, 1, j] for f in fits])
        s_chains = np.stack([f.sd()[:, j] for f in fits])
        ess_d = max(float(az.ess(d_chains, method="bulk")), 1.0)
        ess_s = max(float(az.ess(s_chains, method="bulk")), 1.0)

        # --- the same quantities in natural units, transformed PER DRAW ------
        nb, nm_ = nat_group_draws(B, j)
        nd = nm_ - nb
        nd_chains = np.stack([nat_diff_draws(f.B, j) for f in fits])
        ess_nd = max(float(az.ess(nd_chains, method="bulk")), 1.0)
        plo_b, phi_b, plo_m, phi_m = nat_spread_draws(B, SDp, j)
        # P(M > B) is the SAME event either way for an increasing transform;
        # for k_io it is the opposite one, so report it in natural units too
        p_nat = float(np.mean(nd > 0))

        rows.append(dict(
            label=label, coordinate=nm,
            benign_mean=float(np.median(B[:, 0, j])),
            diff_median=float(np.median(d)),
            diff_q05=float(np.quantile(d, 0.05)),
            diff_q95=float(np.quantile(d, 0.95)),
            diff_mcse=float(np.std(d) / np.sqrt(ess_d)),
            p_malignant_higher=float(np.mean(d > 0)),
            pop_sd_median=float(np.median(SDp[:, j])),
            pop_sd_q05=float(np.quantile(SDp[:, j], 0.05)),
            pop_sd_q95=float(np.quantile(SDp[:, j], 0.95)),
            pop_sd_mcse=float(np.std(SDp[:, j]) / np.sqrt(ess_s)),
            # natural units: population MEDIAN of each group, and the
            # difference of those medians
            natural_param=NAT_LABEL[j],
            nat_benign_median=float(np.median(nb)),
            nat_benign_q05=float(np.quantile(nb, 0.05)),
            nat_benign_q95=float(np.quantile(nb, 0.95)),
            nat_malignant_median=float(np.median(nm_)),
            nat_malignant_q05=float(np.quantile(nm_, 0.05)),
            nat_malignant_q95=float(np.quantile(nm_, 0.95)),
            nat_diff_median=float(np.median(nd)),
            nat_diff_q05=float(np.quantile(nd, 0.05)),
            nat_diff_q95=float(np.quantile(nd, 0.95)),
            nat_diff_mcse=float(np.std(nd) / np.sqrt(ess_nd)),
            nat_p_malignant_higher=p_nat,
            # lesion-to-lesion 5-95% spread of each group, natural units
            nat_pop_q05_benign=float(np.median(plo_b)),
            nat_pop_q95_benign=float(np.median(phi_b)),
            nat_pop_q05_malignant=float(np.median(plo_m)),
            nat_pop_q95_malignant=float(np.median(phi_m))))
    return pd.DataFrame(rows)


# =============================================================================
# step 6: diagnostics
# =============================================================================
def step6_diagnostics(fits, s1, X, coef_names):
    import arviz as az
    print()
    print("=" * 78)
    print("6.  DIAGNOSTICS")
    print("=" * 78)
    L = s1.draws.shape[0]
    sids = list(s1.sids)
    acc = np.mean([f.accept for f in fits], axis=0)
    dist = np.min([f.n_distinct for f in fits], axis=0)
    kish_ess = kish_fitted(fits, s1, X)

    print("    a) per lesion (acceptance averaged over seeds, distinct = min "
          "over seeds, Kish ESS at the fitted population):")
    for lbl, v in (("acceptance", acc), ("distinct", dist), ("Kish ESS", kish_ess)):
        print(f"       {lbl:<11} min {np.min(v):9.3f}   5th pct "
              f"{np.percentile(v, 5):9.3f}   median {np.median(v):9.3f}")
    worst = np.argsort(kish_ess)[:WORST_N]
    print(f"       worst {len(worst)} by Kish ESS (of K = {s1.draws.shape[1]}):")
    print(f"         {'lesion':<10}{'Kish':>8}{'accept':>9}{'distinct':>10}")
    for i in worst:
        print(f"         {sids[i]:<10}{kish_ess[i]:>8.1f}{acc[i]:>9.3f}{dist[i]:>10d}")
    degen_mask = kish_ess < KISH_MIN
    degen = [sids[i] for i in range(L) if degen_mask[i]]
    print(f"       degenerate (Kish ESS < {KISH_MIN:g}, run-length independent): "
          f"{len(degen)}  {degen if degen else ''}")
    old = [sids[i] for i in range(L) if dist[i] < 20 or acc[i] < 0.01]
    print(f"       (the old distinct<20 / accept<0.01 rule would flag "
          f"{len(old)}; that rule weakens as the chain lengthens)")

    # seed-to-seed agreement and R-hat across seeds
    print("    b) across seeds (each seed treated as one chain):")
    rows, worst_shift, worst_rhat = [], 0.0, 1.0
    quants = []
    for p, cn in enumerate(coef_names):
        for j, nm in enumerate(THETA_NAMES):
            quants.append((f"B[{cn}].{nm}", np.stack([f.B[:, p, j] for f in fits])))
    for j, nm in enumerate(THETA_NAMES):
        quants.append((f"popSD.{nm}", np.stack([f.sd()[:, j] for f in fits])))
    for name, arr in quants:
        meds = np.median(arr, axis=1)
        sd = np.std(arr)
        shift = float((meds.max() - meds.min()) / max(sd, 1e-12))
        rh = float(az.rhat(arr)) if len(fits) > 1 else np.nan
        es = float(az.ess(arr, method="bulk"))
        rows.append(dict(quantity=name, rhat=rh, ess_bulk=es, seed_shift_sd=shift))
        worst_shift = max(worst_shift, shift)
        if np.isfinite(rh):
            worst_rhat = max(worst_rhat, rh)
    dg = pd.DataFrame(rows)
    print(f"       worst R-hat {worst_rhat:.4f}   worst seed-to-seed median "
          f"shift {worst_shift:.3f} posterior SD   min ESS "
          f"{dg['ess_bulk'].min():.0f}")
    bad = dg[(dg["rhat"] > 1.01) | (dg["seed_shift_sd"] > 0.2)]
    if len(bad):
        print("       !! not yet agreeing across seeds:")
        for _, r in bad.iterrows():
            print(f"          {r['quantity']:<42} R-hat {r['rhat']:.4f}  "
                  f"shift {r['seed_shift_sd']:.3f} SD  ESS {r['ess_bulk']:.0f}")
    else:
        print("       all quantities: R-hat <= 1.01 and shift <= 0.2 SD")

    lesion = pd.DataFrame(dict(subject_id=sids, accept=acc, n_distinct=dist,
                               kish_ess_fitted=kish_ess,
                               degenerate=degen_mask))
    return lesion, dg


# =============================================================================
# step 7: results
# =============================================================================
def step7_results(fits, s1, X, coef_names, dx, lesion):
    print()
    print("=" * 78)
    print("7.  RESULTS  (all seeds pooled)")
    print("=" * 78)
    B = np.concatenate([f.B for f in fits])                 # (n, P, D)
    SD = np.concatenate([f.sd() for f in fits])             # (n, D)
    C = np.concatenate([f.corr() for f in fits])            # (n, D, D)

    rows = []
    hdr = (f"    {'coordinate':<12}{'benign mean':>13}{'90% CrI':>22}"
           f"{'natural':>11}   {'M - B':>8}{'90% CrI':>22}{'ratio':>8}"
           f"{'P(M>B)':>8}   {'pop SD':>7}{'M-B/SD':>8}")
    print(hdr)
    for j, nm in enumerate(THETA_NAMES):
        b0, d = B[:, 0, j], B[:, 1, j]
        q = lambda v: (np.median(v), np.quantile(v, 0.05), np.quantile(v, 0.95))
        m0, l0, h0 = q(b0)
        md, ld, hd = q(d)
        nat0 = float(natural(m0, j))
        # exp(difference): ratio of population medians for log coordinates,
        # odds ratio for logit coordinates
        ratio = float(np.exp(md))
        p_up = float(np.mean(d > 0))
        ms, ls, hs = q(SD[:, j])
        print(f"    {nm:<12}{m0:>13.3f} [{l0:>8.3f},{h0:>8.3f}]{nat0:>11.4g}   "
              f"{md:>8.3f} [{ld:>8.3f},{hd:>8.3f}]{ratio:>8.3f}{p_up:>8.3f}   "
              f"{ms:>7.3f}{md / ms:>8.2f}")
        rows.append(dict(coordinate=nm, natural_param=NAT_LABEL[j],
                         benign_mean=m0, benign_q05=l0, benign_q95=h0,
                         benign_pop_median_natural=nat0,
                         malignant_pop_median_natural=float(natural(m0 + md, j)),
                         diff_median=md, diff_q05=ld, diff_q95=hd,
                         exp_diff=ratio, exp_diff_q05=float(np.exp(ld)),
                         exp_diff_q95=float(np.exp(hd)), p_malignant_higher=p_up,
                         pop_sd_median=ms, pop_sd_q05=ls, pop_sd_q95=hs,
                         effect_size=md / ms))
    print("    natural = benign population median in natural units "
          "(sigmoid / exp of the mean)")
    print("    ratio   = exp(M - B): ratio of population medians for log "
          "coordinates, odds ratio for logit ones")
    print("    M-B/SD  = group shift in units of the lesion-to-lesion spread")

    # --- the same result in natural units, transformed per draw -------------
    print()
    print("    in NATURAL UNITS (every draw transformed, then summarised --")
    print("    these are population MEDIANS, not means: for a log-normal the")
    print("    two differ by exp(sd^2/2), so do not read these as averages)")
    print(f"    {'parameter':<22}{'benign':>10}{'90% CrI':>21}"
          f"{'malignant':>11}{'90% CrI':>21}{'M - B':>10}{'90% CrI':>21}"
          f"{'P(M>B)':>8}")
    for j in range(D):
        nb, nm_ = nat_group_draws(B, j)
        nd = nm_ - nb
        qq = lambda v: (np.median(v), np.quantile(v, 0.05), np.quantile(v, 0.95))
        mb, lb_, hb = qq(nb)
        mm, lm, hm = qq(nm_)
        mdn, ldn, hdn = qq(nd)
        plo_b, phi_b, plo_m, phi_m = nat_spread_draws(B, SD, j)
        print(f"    {NAT_LABEL[j]:<22}{mb:>10.4g} [{lb_:>8.4g},{hb:>8.4g}]"
              f"{mm:>11.4g} [{lm:>8.4g},{hm:>8.4g}]"
              f"{mdn:>10.4g} [{ldn:>8.4g},{hdn:>8.4g}]"
              f"{np.mean(nd > 0):>8.3f}")
        rows[j].update(
            nat_benign_median=float(mb), nat_benign_q05=float(lb_),
            nat_benign_q95=float(hb), nat_malignant_median=float(mm),
            nat_malignant_q05=float(lm), nat_malignant_q95=float(hm),
            nat_diff_median=float(mdn), nat_diff_q05=float(ldn),
            nat_diff_q95=float(hdn), nat_p_malignant_higher=float(np.mean(nd > 0)),
            nat_pop_q05_benign=float(np.median(plo_b)),
            nat_pop_q95_benign=float(np.median(phi_b)),
            nat_pop_q05_malignant=float(np.median(plo_m)),
            nat_pop_q95_malignant=float(np.median(phi_m)))
    print("    The CrI here is the uncertainty in the GROUP MEDIAN. The "
          "lesion-to-lesion")
    print("    spread of each group is much wider -- see the lower row of fig 6.")
    if TAU_I_AS_RATE:
        print("    tau_i is shown as the exchange rate k_io = 1/tau_i, so its "
              "sign is flipped")
        print("    relative to the log_tau_i row above (TAU_I_AS_RATE = True).")
    for p, cn in enumerate(coef_names[2:], start=2):
        print(f"    {cn}: " + "  ".join(
            f"{nm} {np.median(B[:, p, j]):+.3f}" for j, nm in enumerate(THETA_NAMES)))

    print()
    print("    population correlation (posterior median):")
    Cm = np.median(C, axis=0)
    print("    " + " " * 12 + "".join(f"{n[:10]:>11}" for n in THETA_NAMES))
    for j, nm in enumerate(THETA_NAMES):
        print(f"    {nm:<12}" + "".join(f"{Cm[j, k]:>11.2f}" for k in range(D)))

    # tau_i against the stage-1 ceiling
    from scipy import stats as _st
    j = THETA_NAMES.index("log_tau_i")
    ubl = np.log(TAU_I_BOUNDS[1])
    print()
    print(f"    tau_i against the stage-1 ceiling of {TAU_I_BOUNDS[1]:g} s:")
    for lab, mu in (("benign", np.median(B[:, 0, j])),
                    ("malignant", np.median(B[:, 0, j] + B[:, 1, j]))):
        sdj = float(np.median(SD[:, j]))
        z = (ubl - mu) / sdj
        print(f"      {lab:<10} population median {np.exp(mu):.3f} s   "
              f"mass above the ceiling {_st.norm.sf(z) * 100:5.2f}%  (z {z:.2f})")
    print("      mass above the ceiling has no stage-1 draws to land on, so it")
    print("      biases that group's mean DOWN and compresses the population SD")

    # per-lesion shrinkage: stage-1 median vs stage-2 median
    L = s1.draws.shape[0]
    idx = np.concatenate([f.theta_idx for f in fits])        # (n, L)
    shr = {}
    for j in range(D):
        s1_med = np.median(s1.draws[:, :, j], axis=1)
        s2_med = np.array([np.median(s1.draws[i, idx[:, i], j]) for i in range(L)])
        shr[f"{PARAMS[j]}_stage1_median"] = natural(s1_med, j)
        shr[f"{PARAMS[j]}_stage2_median"] = natural(s2_med, j)
    lesion = lesion.assign(diagnosis=[dx[s] for s in lesion["subject_id"]], **shr)
    return pd.DataFrame(rows), lesion


# =============================================================================
# step 8: K-sensitivity -- the finite-K approximation
# =============================================================================
def step8_k_sensitivity(s1_full, X, coef_names):
    print()
    print("=" * 78)
    print(f"8.  K-SENSITIVITY  ({SENS_N_TUNE} tune + {SENS_N_DRAWS} draws, "
          f"seeds {SENS_SEEDS}, sweeps {THETA_SWEEPS})")
    print("=" * 78)
    print("    Stage 2 is exact only as K -> infinity. All K values are nested")
    print("    subsamples of the same draws and share the chain length, so a")
    print("    systematic trend with K is the finite-K approximation, not noise.")
    cost = (len(K_GRID) * len(SENS_SEEDS) * (SENS_N_TUNE + SENS_N_DRAWS)
            / (N_TUNE + N_DRAWS) / max(len(SEEDS), 1))
    print(f"    expected cost: about {cost:.1f}x the reference fit")

    out, fits_by_K, kish_by_K = [], {}, {}
    for Kv in K_GRID:
        s1k = subsample_s1(s1_full, Kv)
        print(f"    --- K = {Kv} ---")
        fits = fit_many(s1k, X, coef_names, SENS_N_TUNE, SENS_N_DRAWS,
                        SENS_SEEDS, tag=f"K={Kv:<6} ")
        ke = kish_fitted(fits, s1k, X)
        s = summarize_fits(fits, label=f"K={Kv}")
        s["K"] = Kv
        s["kish_median"] = float(np.median(ke))
        s["kish_p05"] = float(np.percentile(ke, 5))
        s["kish_min"] = float(np.min(ke))
        s["n_degenerate"] = int(np.sum(ke < KISH_MIN))
        out.append(s)
        fits_by_K[Kv] = (fits, s1k)
        kish_by_K[Kv] = ke
    ks = pd.concat(out, ignore_index=True)

    print()
    print(f"    {'K':>7}{'coordinate':<12}{'M - B':>9}{'MCSE':>8}"
          f"{'90% CrI':>20}{'pop SD':>9}{'MCSE':>8}{'P(M>B)':>8}")
    for Kv in K_GRID:
        sub = ks[ks["K"] == Kv]
        for _, r in sub.iterrows():
            print(f"    {Kv:>7}{r['coordinate']:<12}{r['diff_median']:>9.3f}"
                  f"{r['diff_mcse']:>8.3f} [{r['diff_q05']:>7.3f},"
                  f"{r['diff_q95']:>7.3f}]{r['pop_sd_median']:>9.3f}"
                  f"{r['pop_sd_mcse']:>8.3f}{r['p_malignant_higher']:>8.3f}")
        print()
    print(f"    {'K':>7}{'Kish min':>10}{'Kish 5th':>10}{'Kish med':>10}"
          f"{'# degenerate':>14}")
    for Kv in K_GRID:
        r = ks[ks["K"] == Kv].iloc[0]
        print(f"    {Kv:>7}{r['kish_min']:>10.1f}{r['kish_p05']:>10.1f}"
              f"{r['kish_median']:>10.1f}{int(r['n_degenerate']):>14d}")

    print()
    print("    drift across K, in units of the quantity's own Monte-Carlo error")
    print("    (> 3 means the change is real and not chain noise):")
    print(f"      {'coordinate':<12}{'M-B drift':>11}{'in MCSE':>9}"
          f"{'popSD drift':>13}{'in MCSE':>9}  verdict")
    verdicts = {}
    for nm in THETA_NAMES:
        sub = ks[ks["coordinate"] == nm].sort_values("K")
        dd = float(sub["diff_median"].max() - sub["diff_median"].min())
        ds = float(sub["pop_sd_median"].max() - sub["pop_sd_median"].min())
        md = float(sub["diff_mcse"].max())
        ms = float(sub["pop_sd_mcse"].max())
        rd, rs = dd / max(md, 1e-12), ds / max(ms, 1e-12)
        v = "FINITE-K TREND" if max(rd, rs) > 3 else "stable"
        verdicts[nm] = v
        print(f"      {nm:<12}{dd:>11.4f}{rd:>9.1f}{ds:>13.4f}{rs:>9.1f}  {v}")
    print("    A trend means the reported value depends on how many stage-1")
    print("    draws you happened to hand over; raise K until it flattens.")
    return ks, fits_by_K, kish_by_K


# =============================================================================
# step 9: drop the weight-starved lesions and refit
# =============================================================================
def step9_degenerate_sensitivity(s1_full, X, coef_names, kish_ref, sids,
                                 fits_by_K):
    print()
    print("=" * 78)
    print(f"9.  DEGENERATE-SET SENSITIVITY  (Kish ESS < {KISH_MIN:g} dropped)")
    print("=" * 78)
    keep = np.asarray(kish_ref) >= KISH_MIN
    dropped = [s for s, k in zip(sids, keep) if not k]
    print(f"    dropping {len(dropped)} of {len(keep)} lesions: {dropped}")
    if not len(dropped):
        print("    nothing below the threshold -- no refit needed")
        return None
    if keep.sum() < 10:
        print("    !! fewer than 10 lesions would remain -- skipping")
        return None

    # baseline: the same K and the same chain length, all lesions. Reuse the
    # K-grid fit when it is there, so only the lesion set differs.
    if K in fits_by_K:
        fits_all, s1_all = fits_by_K[K]
        print(f"    baseline reused from the K = {K} sensitivity fit")
    else:
        s1_all = subsample_s1(s1_full, K)
        print(f"    baseline (all lesions, K = {K}):")
        fits_all = fit_many(s1_all, X, coef_names, SENS_N_TUNE, SENS_N_DRAWS,
                            SENS_SEEDS, tag="all      ")
    s1_sub = subset_s1(subsample_s1(s1_full, K), keep)
    print(f"    refit without them (K = {K}):")
    fits_sub = fit_many(s1_sub, X[keep], coef_names, SENS_N_TUNE, SENS_N_DRAWS,
                        SENS_SEEDS, tag="dropped  ")

    a = summarize_fits(fits_all, label="all lesions")
    b = summarize_fits(fits_sub, label=f"Kish >= {KISH_MIN:g}")
    print()
    print(f"    {'coordinate':<12}{'M-B all':>10}{'M-B kept':>10}{'change':>9}"
          f"{'in MCSE':>9}{'P all':>8}{'P kept':>8}")
    rows = []
    for nm in THETA_NAMES:
        ra = a[a["coordinate"] == nm].iloc[0]
        rb = b[b["coordinate"] == nm].iloc[0]
        ch = rb["diff_median"] - ra["diff_median"]
        mc = max(ra["diff_mcse"], rb["diff_mcse"], 1e-12)
        print(f"    {nm:<12}{ra['diff_median']:>10.3f}{rb['diff_median']:>10.3f}"
              f"{ch:>9.3f}{ch / mc:>9.1f}{ra['p_malignant_higher']:>8.3f}"
              f"{rb['p_malignant_higher']:>8.3f}")
        rows.append(dict(coordinate=nm, diff_all=ra["diff_median"],
                         diff_kept=rb["diff_median"], change=ch,
                         change_in_mcse=ch / mc,
                         p_all=ra["p_malignant_higher"],
                         p_kept=rb["p_malignant_higher"],
                         q05_all=ra["diff_q05"], q95_all=ra["diff_q95"],
                         q05_kept=rb["diff_q05"], q95_kept=rb["diff_q95"],
                         pop_sd_all=ra["pop_sd_median"],
                         pop_sd_kept=rb["pop_sd_median"],
                         natural_param=ra["natural_param"],
                         nat_diff_all=ra["nat_diff_median"],
                         nat_diff_kept=rb["nat_diff_median"],
                         nat_q05_all=ra["nat_diff_q05"],
                         nat_q95_all=ra["nat_diff_q95"],
                         nat_q05_kept=rb["nat_diff_q05"],
                         nat_q95_kept=rb["nat_diff_q95"]))
    out = pd.DataFrame(rows)
    big = out[np.abs(out["change_in_mcse"]) > 3]
    if len(big):
        print("    !! the starved lesions are moving these coordinates:")
        for _, r in big.iterrows():
            print(f"       {r['coordinate']:<12} change {r['change']:+.3f} "
                  f"= {r['change_in_mcse']:+.1f} MCSE")
    else:
        print("    no coordinate moves by more than 3 MCSE: the starved "
              "lesions are not driving the result")
    out.attrs["dropped"] = dropped
    return out


# =============================================================================
# step 10: derived quantities -- Ktrans, E, and the correlation provenance
# =============================================================================
def step10_derived(fits, s1, X, dx, lesion):
    print()
    print("=" * 78)
    print("10. DERIVED QUANTITIES")
    print("=" * 78)
    B = np.concatenate([f.B for f in fits])
    jfp, jps = THETA_NAMES.index("log_fp"), THETA_NAMES.index("log_ps")

    def ktrans_E(lfp, lps):
        fp, ps = np.exp(lfp), np.exp(lps)
        E = -np.expm1(-ps / fp)                 # 1 - exp(-PS/fp), stable
        return fp * E * RATE_TO_PER_MIN, E

    Kt_b, E_b = ktrans_E(B[:, 0, jfp], B[:, 0, jps])
    Kt_m, E_m = ktrans_E(B[:, 0, jfp] + B[:, 1, jfp],
                         B[:, 0, jps] + B[:, 1, jps])
    print("    a) at the population medians, computed PER DRAW so the fp-PS")
    print("       posterior correlation is carried through:")
    print(f"       {'quantity':<16}{'benign':>10}{'90% CrI':>20}"
          f"{'malignant':>11}{'90% CrI':>20}{'log ratio':>11}{'P(M>B)':>8}")
    drv = []
    for lbl, vb, vm in (("Ktrans [1/min]", Kt_b, Kt_m),
                        ("E (extraction)", E_b, E_m)):
        lr = np.log(vm / vb)
        print(f"       {lbl:<16}{np.median(vb):>10.4f} "
              f"[{np.quantile(vb, .05):>8.4f},{np.quantile(vb, .95):>8.4f}]"
              f"{np.median(vm):>11.4f} "
              f"[{np.quantile(vm, .05):>8.4f},{np.quantile(vm, .95):>8.4f}]"
              f"{np.median(lr):>11.3f}{np.mean(lr > 0):>8.3f}")
        drv.append(dict(quantity=lbl, benign_median=float(np.median(vb)),
                        benign_q05=float(np.quantile(vb, .05)),
                        benign_q95=float(np.quantile(vb, .95)),
                        malignant_median=float(np.median(vm)),
                        malignant_q05=float(np.quantile(vm, .05)),
                        malignant_q95=float(np.quantile(vm, .95)),
                        log_ratio_median=float(np.median(lr)),
                        log_ratio_q05=float(np.quantile(lr, .05)),
                        log_ratio_q95=float(np.quantile(lr, .95)),
                        p_malignant_higher=float(np.mean(lr > 0))))
    print("       A Ktrans LOWER in malignant runs against most breast DCE")
    print("       series; treat it as a flag on the fp/PS split, not a finding.")

    # per-lesion Ktrans from the stage-2 draws each lesion actually visited
    idx = np.concatenate([f.theta_idx for f in fits])
    sel = even_idx(idx.shape[0], min(DERIVED_THIN, idx.shape[0]))
    L = s1.draws.shape[0]
    kt_les = np.empty(L)
    for i in range(L):
        th = s1.draws[i, idx[sel, i], :]
        kt, _ = ktrans_E(th[:, jfp], th[:, jps])
        kt_les[i] = float(np.median(kt))

    # within-lesion vs population correlation
    print()
    print("    b) WITHIN-lesion correlation (median over lesions, stage-1 draws)")
    print("       against the fitted POPULATION correlation. If they match, the")
    print("       population correlation may just be the TCXM within-lesion")
    print("       ridge rather than between-lesion biology.")
    # a starved lesion can visit draws with zero spread in a coordinate, which
    # makes corrcoef divide by zero -> nan. Drop those lesions per element.
    with np.errstate(divide="ignore", invalid="ignore"):
        Cw = np.nanmedian(np.stack([np.corrcoef(s1.draws[i], rowvar=False)
                                    for i in range(L)]), axis=0)
        Cs = np.nanmedian(np.stack(
            [np.corrcoef(s1.draws[i, idx[sel, i], :], rowvar=False)
             for i in range(L)]), axis=0)
    Cp = np.median(np.concatenate([f.corr() for f in fits]), axis=0)
    print(f"       {'pair':<24}{'within (s1)':>13}{'within (s2 sel)':>17}"
          f"{'population':>12}{'|pop - within|':>16}")
    pairs = []
    for a in range(D):
        for b in range(a + 1, D):
            nm = f"{THETA_NAMES[a]} ~ {THETA_NAMES[b]}"
            print(f"       {nm:<24}{Cw[a, b]:>13.2f}{Cs[a, b]:>17.2f}"
                  f"{Cp[a, b]:>12.2f}{abs(Cp[a, b] - Cw[a, b]):>16.2f}")
            pairs.append(dict(pair=nm, within_stage1=Cw[a, b],
                              within_stage2_selected=Cs[a, b],
                              population=Cp[a, b],
                              abs_gap=abs(Cp[a, b] - Cw[a, b])))
    pr = pd.DataFrame(pairs)
    susp = pr[(pr["abs_gap"] < 0.15) & (pr["population"].abs() > 0.3)]
    if len(susp):
        print("       !! population correlations that merely reproduce the "
              "within-lesion shape:")
        for _, r in susp.iterrows():
            print(f"          {r['pair']:<24} within {r['within_stage1']:+.2f}"
                  f"  population {r['population']:+.2f}")
    else:
        print("       no strong population correlation simply mirrors the "
              "within-lesion one")
    return pd.DataFrame(drv), pr, dict(Cw=Cw, Cs=Cs, Cp=Cp, kt_les=kt_les,
                                       Kt_b=Kt_b, Kt_m=Kt_m, E_b=E_b, E_m=E_m)


# =============================================================================
# step 11: posterior predictive check of the STAGE-2 (population) model
# =============================================================================
def _bayes_p(rep, obs):
    """P(T(rep) >= T(obs)), paired draw by draw. 0.5 is perfect; the
    two-sided version 2*min(p, 1-p) is what gets judged, because a statistic
    can be too SMALL as well as too large."""
    rep, obs = np.asarray(rep, float), np.asarray(obs, float)
    ok = np.isfinite(rep) & np.isfinite(obs)
    if not ok.any():
        return np.nan
    return float(np.mean(rep[ok] >= obs[ok]))


def ppc_z(fits, s1, X, take):
    """
    Standardised lesion residuals under the stage-2 model, per draw.

    At stage-2 iteration m the model says theta_i ~ N(X_i B, Sigma). With
    Sigma = L L', the standardised residuals

        z_i = L^-1 (theta_i - X_i B)

    are iid N(0, I_D) if that model is right -- for ANY B and Sigma, so the
    reference distribution is exact and needs no simulation from the fitted
    population. Returns (z, zrep), each (M, L, D): the fitted residuals and an
    exact N(0, I) reference of the same shape.
    """
    B = np.concatenate([f.B for f in fits])                 # (n, P, D)
    Sig = np.concatenate([f.Sigma for f in fits])           # (n, D, D)
    idx = np.concatenate([f.theta_idx for f in fits])       # (n, L)
    L = idx.shape[1]
    rng = np.random.default_rng(PPC_SEED)
    z = np.empty((len(take), L, D))
    for a, m in enumerate(take):
        th = s1.draws[np.arange(L), idx[m]]                 # (L, D) selected
        Lc = np.linalg.cholesky(Sig[m])
        z[a] = np.linalg.solve(Lc, (th - X @ B[m]).T).T
    return z, rng.standard_normal(z.shape)


def _row(label, stat, coord, o, r):
    """One table row: the fitted statistic, its exact reference, the paired
    posterior predictive p-value, AND the effect size.

    Both are needed, and they can disagree. The p-value is calibrated but
    weak: it is bounded by how much the fitted statistic's own draw-to-draw
    spread overlaps the reference, and for a noisy statistic (sample kurtosis
    at L ~ 100) that overlap stays large even when the typical fitted value is
    nowhere near the reference. The effect size

        discrepancy_sd = (median fitted - median reference) / sd(reference)

    asks the other question -- is the TYPICAL fitted value where the model
    says it should be -- and is the sensitive one for exactly those
    statistics. See the power table in the step-11 docstring."""
    o, r = np.asarray(o, float), np.asarray(r, float)
    p = _bayes_p(r, o)
    sd = float(np.std(r))
    return dict(subset=label, statistic=stat, coordinate=coord,
                observed=float(np.median(o)), rep_median=float(np.median(r)),
                rep_q05=float(np.quantile(r, 0.05)),
                rep_q95=float(np.quantile(r, 0.95)),
                p_value=p, p_two_sided=2 * min(p, 1 - p),
                discrepancy_sd=((float(np.median(o)) - float(np.median(r))) / sd
                                if sd > 0 else np.nan))


def ppc_table(z, zrep, label):
    """One row per (statistic, coordinate). Statistics live on z, so the group
    shift and the population scale are already removed and every reference is
    the same: iid standard normal."""
    from scipy import stats as st
    rows = []
    stats = (("sd", lambda a: a.std(axis=1, ddof=1)),
             ("skew", lambda a: st.skew(a, axis=1)),
             ("excess kurtosis", lambda a: st.kurtosis(a, axis=1)),
             ("max |z|", lambda a: np.abs(a).max(axis=1)))
    for key, fn in stats:
        o, r = fn(z), fn(zrep)                              # (M, D) each
        for j, nm in enumerate(THETA_NAMES):
            rows.append(_row(label, key, nm, o[:, j], r[:, j]))
    # joint: Mahalanobis d^2 = |z|^2 ~ chi2_D under the model, so these two
    # catch a wrong CORRELATION structure, which no per-coordinate statistic can
    d2o, d2r = (z ** 2).sum(2), (zrep ** 2).sum(2)          # (M, L)
    chi99 = st.chi2.ppf(0.99, D)
    for key, fn in (("d2 95th pct", lambda a: np.percentile(a, 95, axis=1)),
                    (f"n d2 > chi2_99", lambda a: (a > chi99).sum(axis=1))):
        rows.append(_row(label, key, "(joint)",
                         fn(d2o).astype(float), fn(d2r).astype(float)))
    return pd.DataFrame(rows)


def step11_ppc(fits, s1, X, lesion):
    """
    Posterior predictive check of the stage-2 population model.

    WHAT IS BEING CHECKED, AND WHAT IS NOT
      Stage 2's model is theta_i ~ N(X_i B, Sigma). This asks whether the
      fitted theta_i look like draws from it: right spread, symmetric, the
      right tails, the right joint shape. It says NOTHING about whether any
      lesion's own TCXM + SSM3 fit describes its curve -- that is stage 1's
      model, and no curve is read here.

    WHY A POSTERIOR DRAW OF theta AND NOT ITS MEAN
      The theta_i are latent, so there is no observed vector to compare
      against. But a posterior DRAW of theta_i has the right marginal spread
      under the model (Var(theta) = Var(E[theta|y]) + E[Var(theta|y)] = Sigma),
      while the posterior MEAN is shrunk and would fail the check even for a
      perfect model. So each stage-2 iteration contributes one draw, and the
      statistics are computed per iteration and compared per iteration.

    TWO REASONS A p-VALUE HERE CAN MISLEAD, BOTH REPORTED
      1. CONSERVATIVE. theta_i is drawn under the very model being checked, so
         a lesion whose own data says little is pulled onto the population and
         contributes a well-behaved z no matter what. Real misfit therefore
         shows up WEAKER here than it would with observed theta; a p-value
         near 0.5 is weaker evidence of fit than it looks.
      2. FINITE K. theta_i can only ever be one of that lesion's K saved
         stage-1 draws, so a weight-starved lesion has almost no distinct
         draws to move among and its z is nearly frozen across stage-2
         iterations -- while the reference assumes a fresh normal draw every
         iteration. Frozen WHERE matters: a lesion is starved largely BECAUSE
         its own posterior sits far from the fitted population, so it is
         pinned in the TAIL. The effect on this check is therefore to inflate
         max|z|, kurtosis and the d2 tail -- a misfit signature manufactured
         by finite K, not by a non-Gaussian population.

         Two things separate the explanations, both reported: the whole table
         is repeated on the Kish >= KISH_MIN subset, and fig 7 plots
         per-lesion RMS |z| against Kish ESS. A NEGATIVE correlation there is
         expected and benign -- it is the selection effect above, not a
         verdict. What matters is whether the effect sizes FALL on the
         Kish >= KISH_MIN subset: if they do, the apparent misfit is carried
         by lesions whose theta stage 2 cannot represent.

    WHAT THIS CHECK CAN AND CANNOT SEE
      Measured by replacing z with residuals from a known violation, 500
      draws, flagging |effect| > 2 (and, in brackets, two-sided p < 0.05).
      Counts are out of the 22 statistics in the table.

        violation                 L = 80        L = 158
        none (calibration)        0  (0)        0  (0)
        sd x0.85                  1  (0)        6  (6)
        sd x1.20                  7  (1)        7  (7)
        heavy tails, t_4          9  (0)       12  (5)
        skewed (lognormal)       17 (10)       17 (11)
        5% of lesions at 4x      17  (0)       17 (12)
        coordinate corr 0.6       2  (0)        2  (1)

      Three things to read off it. The check never fires on data that does
      satisfy the model. A wrong CORRELATION structure is visible only in the
      two joint d2 statistics, which is why they are there -- no
      per-coordinate statistic can see it. And the p-value column is much the
      weaker of the two readouts for heavy tails and outliers, which is the
      whole reason the effect size is reported beside it.
    """
    print()
    print("=" * 78)
    print("11. POSTERIOR PREDICTIVE CHECK  (stage-2 population model)")
    print("=" * 78)
    n_tot = sum(f.B.shape[0] for f in fits)
    take = even_idx(n_tot, min(PPC_DRAWS, n_tot))
    z, zrep = ppc_z(fits, s1, X, take)
    L = z.shape[1]
    print(f"    model checked : theta_i ~ N(X_i B, Sigma),  i = 1..{L}")
    print(f"    statistics on : z_i = chol(Sigma)^-1 (theta_i - X_i B),"
          f" iid N(0, I_{D}) if it holds")
    print(f"    draws used    : {len(take)} of {n_tot}, evenly thinned")
    print("    NOT checked   : whether any lesion's TCXM + SSM3 fit describes")
    print("                    its own curve. That is stage 1's model; no")
    print("                    lesion curve is read here.")

    kish = lesion["kish_ess_fitted"].to_numpy(float)
    keep = kish >= KISH_MIN
    tabs = [ppc_table(z, zrep, "all lesions")]
    if keep.sum() >= 10 and (~keep).any():
        tabs.append(ppc_table(z[:, keep], zrep[:, keep],
                              f"Kish >= {KISH_MIN:g}"))
    ppc = pd.concat(tabs, ignore_index=True)

    for lab in ppc["subset"].unique():
        sub = ppc[ppc["subset"] == lab]
        n = L if lab == "all lesions" else int(keep.sum())
        print()
        print(f"    {lab}  ({n} lesions)")
        print(f"      {'statistic':<16}{'coordinate':<12}{'fitted':>9}"
              f"{'reference':>11}{'90% ref':>19}{'2-sided p':>11}"
              f"{'effect':>8}")
        for _, r in sub.iterrows():
            flag = ("  <--" if (np.isfinite(r["p_two_sided"])
                                and r["p_two_sided"] < 0.05)
                    or (np.isfinite(r["discrepancy_sd"])
                        and abs(r["discrepancy_sd"]) > 2) else "")
            print(f"      {r['statistic']:<16}{r['coordinate']:<12}"
                  f"{r['observed']:>9.3f}{r['rep_median']:>11.3f}"
                  f" [{r['rep_q05']:>7.3f},{r['rep_q95']:>7.3f}]"
                  f"{r['p_two_sided']:>11.3f}{r['discrepancy_sd']:>8.2f}{flag}")
        nbad = int((sub["p_two_sided"] < 0.05).sum())
        nbig = int((sub["discrepancy_sd"].abs() > 2).sum())
        print(f"      {nbad} of {len(sub)} statistics have two-sided p < 0.05;"
              f"  {nbig} have |effect| > 2")
        print("      effect = (fitted - reference) / sd(reference). Read it "
              "WITH the p-value:")
        print("      p is calibrated but weak for a noisy statistic, so a big "
              "effect at p ~ 0.2")
        print("      is a real departure the p-value cannot resolve, not a "
              "clean bill of health.")

    # per-lesion: how far each lesion sits from its group, and does that track
    # Kish ESS rather than anything about the population model?
    rms = np.sqrt((z ** 2).sum(2).mean(0) / D)         # (L,) RMS |z| per lesion
    d2_med = np.median((z ** 2).sum(2), axis=0)
    les = pd.DataFrame(dict(subject_id=lesion["subject_id"].to_numpy(),
                            diagnosis=lesion["diagnosis"].to_numpy(),
                            kish_ess_fitted=kish, ppc_rms_z=rms,
                            ppc_d2_median=d2_med))
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(kish) & (kish > 0)
        rho = (float(np.corrcoef(np.log(kish[ok]), rms[ok])[0, 1])
               if ok.sum() > 3 else np.nan)
    print()
    print(f"    per-lesion RMS |z| vs log Kish ESS: r = {rho:+.2f}")
    print("      A NEGATIVE r is expected and benign: a lesion is starved")
    print("      largely BECAUSE its posterior sits far from the population,")
    print("      so the starved ones are the far ones. It is not a verdict on")
    print("      the population model. What decides that is whether the effect")
    print("      sizes above FALL on the Kish >= KISH_MIN subset -- if they do,")
    print("      the misfit is carried by lesions stage 2 cannot represent.")
    print(f"\n    {'lesion':<10}{'dx':>3}{'Kish':>9}{'RMS |z|':>9}"
          f"{'median d2':>11}   (largest RMS |z| first)")
    for _, r in les.nlargest(min(WORST_N, len(les)), "ppc_rms_z").iterrows():
        print(f"    {r['subject_id']:<10}{r['diagnosis']:>3}"
              f"{r['kish_ess_fitted']:>9.1f}{r['ppc_rms_z']:>9.2f}"
              f"{r['ppc_d2_median']:>11.2f}")
    return ppc, les, dict(z=z, zrep=zrep, keep=keep)


# =============================================================================
# plots
# =============================================================================
def _finish(fig, name, out_dir):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if SAVE_PLOTS and out_dir:
        fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches="tight")
    plt.show()


def plot_fig1_differences(fits, summ, out_dir):
    """
    Posterior of M - B per parameter IN NATURAL UNITS, plus tau_i against the
    stage-1 ceiling. The difference is formed per draw as

        natural(benign mean + difference) - natural(benign mean)

    i.e. the difference of the two population MEDIANS in natural units. The
    modelling-scale difference is kept in the title (its effect size in
    population SDs is the scale-free number).
    """
    import matplotlib.pyplot as plt
    from scipy import stats as st
    B = np.concatenate([f.B for f in fits])
    SD = np.concatenate([f.sd() for f in fits])
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = ax.ravel()
    for j, nm in enumerate(THETA_NAMES):
        a = axes[j]
        r = summ[summ["coordinate"] == nm].iloc[0]
        nd = nat_diff_draws(B, j)
        md, lo, hi = (np.median(nd), np.quantile(nd, 0.05),
                      np.quantile(nd, 0.95))
        a.hist(nd, bins=80, color="#4878a8", edgecolor="none", density=True)
        a.axvline(0, color="k", ls="--", lw=1.2)
        a.axvspan(lo, hi, color="#f0a860", alpha=0.25, label="90% CrI")
        a.axvline(md, color="#c0392b", lw=1.5, label="median")
        a.set_title(f"{NAT_LABEL[j]}\nM-B {md:+.4g}  "
                    f"P(M>B) {np.mean(nd > 0):.3f}  "
                    f"effect {r['diff_median'] / r['pop_sd_median']:+.2f} SD "
                    f"({nm})", fontsize=9)
        a.set_xlabel(f"malignant - benign, {NAT_LABEL[j]}")
        a.ticklabel_format(axis="x", style="plain", useOffset=False)
        a.set_yticks([])
        if j == 0:
            a.legend(fontsize=7, frameon=False)

    # tau_i population against the stage-1 ceiling. The densities are Gaussian
    # on the log tau_i scale -- which is where the model lives -- but the axis
    # is drawn in natural units, so the curve is the transformed density.
    a = axes[5]
    j = THETA_NAMES.index("log_tau_i")
    mu_b = float(np.median(B[:, 0, j]))
    mu_m = float(np.median(B[:, 0, j] + B[:, 1, j]))
    sdj = float(np.median(SD[:, j]))
    ubl = np.log(TAU_I_BOUNDS[1])                       # the bound, log scale
    zs = np.linspace(min(mu_b, mu_m) - 4 * sdj, max(ubl, mu_m + 4 * sdj), 600)
    xs = natural(zs, j)
    for mu, lab, c in ((mu_b, "benign", "#4878a8"), (mu_m, "malignant", "#c0392b")):
        pdf = st.norm.pdf(zs, mu, sdj)
        # density in the displayed variable: divide by |dx/dz| = |x| here
        pdf_x = pdf / np.abs(xs)
        a.plot(xs, pdf_x, color=c, lw=1.8,
               label=f"{lab}: {natural(mu, j):.3g}, "
                     f"{st.norm.sf((ubl - mu) / sdj) * 100:.1f}% beyond the bound")
        a.fill_between(xs, 0, pdf_x, where=zs > ubl, color=c, alpha=0.3)
    xb = float(natural(ubl, j))
    a.axvline(xb, color="k", lw=1.5)
    a.text(xb, 0.97, f" stage-1 bound {xb:.3g} ", transform=a.get_xaxis_transform(),
           ha="right", va="top", rotation=90, fontsize=7.5)
    a.set_xlim(float(np.min(xs)), float(np.max(xs)))
    a.set_title(f"{NAT_SHORT[4]} population vs the stage-1 bound\n"
                "shaded mass has no stage-1 draws to land on", fontsize=9)
    a.set_xlabel(NAT_LABEL[4])
    a.set_yticks([])
    a.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("fig 1  posterior of the group difference, natural units",
                 fontsize=11)
    _finish(fig, "fig1_group_difference.png", out_dir)


def plot_fig6_forest(fits, summ, lesion, out_dir):
    """
    Benign vs malignant, per parameter, in natural units -- the figure that
    answers "what do the two groups look like", rather than "how big is the
    difference".

    Upper row   POSTERIOR of each group's population MEDIAN. This is what the
                data pin down about the group, and it narrows as the cohort
                grows. Two filled densities per panel, with the median and the
                90% credible interval drawn underneath as a forest-plot row.
    Lower row   FITTED LESION-TO-LESION distribution: a posterior predictive
                for one new lesion, drawn by sampling theta ~ N(mu_g, Sigma)
                at every posterior draw. This is what the lesions themselves
                look like, and it does NOT narrow with cohort size. The dots
                are the per-lesion stage-2 (shrunken) medians, so you can see
                the fitted curve against the lesions it came from.

    The two rows differ by the population SD, and the upper one is always much
    the narrower: that gap is why a parameter can separate the GROUP MEANS
    convincingly while the two groups of LESIONS still overlap almost entirely.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as st
    B = np.concatenate([f.B for f in fits])
    SD = np.concatenate([f.sd() for f in fits])
    rng = np.random.default_rng(0)
    COL = {"B": "#4878a8", "M": "#c0392b"}
    fig, ax = plt.subplots(2, D, figsize=(3.6 * D, 7.4))

    def ridge(a, v, colour, row, label, span=(0.0, 1.0)):
        """
        One filled density + a median / 90% CrI bar below it. `span` trims the
        drawing grid to a quantile range: the predictive of a log-normal has a
        tail long enough to squash the bulk of both groups into one bin.
        """
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        lo, md, hi = (np.quantile(v, 0.05), np.median(v), np.quantile(v, 0.95))
        g0 = v.min() if span[0] <= 0 else float(np.quantile(v, span[0]))
        g1 = v.max() if span[1] >= 1 else float(np.quantile(v, span[1]))
        grid = np.linspace(min(g0, lo), max(g1, hi), 400)
        try:
            dens = st.gaussian_kde(v)(grid)
        except Exception:                       # degenerate (all equal) draws
            return md, lo, hi, grid
        dens = dens / dens.max()
        base = -row                             # row 0 on top, row 1 below it
        a.fill_between(grid, base, base + 0.78 * dens, color=colour, alpha=0.55,
                       lw=0)
        a.plot(grid, base + 0.78 * dens, color=colour, lw=1.2)
        a.hlines(base - 0.08, lo, hi, color=colour, lw=2.2)
        a.plot([md], [base - 0.08], "o", color=colour, ms=5,
               markeredgecolor="w", markeredgewidth=0.7, label=label)
        return md, lo, hi, grid

    for j in range(D):
        # ---- upper: posterior of the group population median ---------------
        a = ax[0, j]
        nb, nm_ = nat_group_draws(B, j)
        mb, lb_, hb, _ = ridge(a, nb, COL["B"], 0, "benign")
        mm, lm, hm, _ = ridge(a, nm_, COL["M"], 1, "malignant")
        nd = nm_ - nb
        a.set_ylim(-1.35, 0.95)
        a.set_yticks([0, -1])
        a.set_yticklabels(["benign", "malignant"], fontsize=8)
        a.set_xlabel(NAT_LABEL[j], fontsize=9)
        a.ticklabel_format(axis="x", style="plain", useOffset=False)
        a.set_title(f"{NAT_SHORT[j]}:  {mb:.4g}  vs  {mm:.4g}\n"
                    f"M-B {np.median(nd):+.4g} "
                    f"[{np.quantile(nd, .05):+.3g}, {np.quantile(nd, .95):+.3g}]"
                    f"   P(M>B) {np.mean(nd > 0):.3f}", fontsize=8.5)
        if j == 0:
            a.set_ylabel("posterior of the\npopulation median", fontsize=9)

        # ---- lower: the fitted lesion-to-lesion distribution ---------------
        a = ax[1, j]
        n = min(len(B), 40000)
        sel = even_idx(len(B), n)
        z = rng.standard_normal(n) * SD[sel, j]
        pb = natural(B[sel, 0, j] + z, j)
        pm = natural(B[sel, 0, j] + B[sel, 1, j] + z, j)
        ridge(a, pb, COL["B"], 0, "benign", span=(0.002, 0.985))
        ridge(a, pm, COL["M"], 1, "malignant", span=(0.002, 0.985))
        # the lesions themselves, as they came out of stage 2
        # already in display units: step 7 builds these with natural()
        x_les = lesion[f"{PARAMS[j]}_stage2_median"].to_numpy(float)
        dxv = lesion["diagnosis"].to_numpy()
        for row, g in ((0, "B"), (1, "M")):
            v = x_les[(dxv == g) & np.isfinite(x_les)]
            a.plot(v, np.full(v.size, -row - 0.26)
                   + rng.uniform(-0.05, 0.05, v.size), "|", color=COL[g],
                   ms=6, alpha=0.75, mew=1.0)
        # P(a random malignant lesion exceeds a random benign one)
        p_les = float(np.mean(pm > rng.permutation(pb)))
        a.set_ylim(-1.45, 0.95)
        a.set_yticks([0, -1])
        a.set_yticklabels(["benign", "malignant"], fontsize=8)
        a.set_xlabel(NAT_LABEL[j], fontsize=9)
        a.ticklabel_format(axis="x", style="plain", useOffset=False)
        a.set_title(f"one new lesion:  P(M > B) = {p_les:.3f}\n"
                    f"ticks = the {len(x_les)} fitted lesions", fontsize=8.5)
        if j == 0:
            a.set_ylabel("fitted lesion-to-lesion\ndistribution", fontsize=9)
    fig.suptitle("fig 6  benign vs malignant populations, natural units  --  "
                 "upper: what the GROUP MEDIAN is (narrows with cohort size)  "
                 "|  lower: what a LESION looks like (does not)", fontsize=10)
    _finish(fig, "fig6_population_forest.png", out_dir)


def plot_fig7_ppc(ppc, les, raw, out_dir):
    """
    fig 7  posterior predictive check of the stage-2 population model.

    Upper row  one panel per coordinate: a QQ plot of the fitted standardised
               residuals z against the standard normal they should follow,
               inside a grey band built from the exact N(0, I) reference. A
               point outside the band at the ends is a tail the Gaussian
               population does not have; a slope below 1 is under-dispersion.
    Lower row  the joint check (Mahalanobis d2 against chi2_D), every
               statistic as a standardised discrepancy, the two-sided
               p-values, the pooled z against N(0,1), and the finite-K
               control: per-lesion RMS |z| against Kish ESS.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as st
    z, zrep, keep = raw["z"], raw["zrep"], raw["keep"]
    M, L, _ = z.shape
    fig, ax = plt.subplots(2, max(D, 5), figsize=(3.5 * max(D, 5), 7.6))
    q = (np.arange(1, L + 1) - 0.5) / L
    nq = st.norm.ppf(q)

    # ---- upper: per-coordinate QQ, with the reference band ------------------
    for j, nm in enumerate(THETA_NAMES):
        a = ax[0, j]
        so = np.sort(z[:, :, j], axis=1)                     # (M, L)
        sr = np.sort(zrep[:, :, j], axis=1)
        a.fill_between(nq, np.percentile(sr, 2.5, axis=0),
                       np.percentile(sr, 97.5, axis=0), color="0.80",
                       lw=0, label="95% reference")
        a.plot(nq, np.median(sr, axis=0), color="0.45", lw=1, ls="--")
        a.plot(nq, np.median(so, axis=0), color="#c0392b", lw=1.8,
               label="fitted z")
        lim = [min(nq[0], -3), max(nq[-1], 3)]
        a.plot(lim, lim, "k:", lw=0.8)
        a.set_xlim(*lim)
        a.set_title(f"{NAT_SHORT[j]}  ({nm})", fontsize=9)
        a.set_xlabel("normal quantile")
        if j == 0:
            a.set_ylabel("standardised residual z")
            a.legend(fontsize=7, frameon=False, loc="upper left")
    for j in range(D, ax.shape[1]):
        ax[0, j].axis("off")

    # ---- lower 0: joint check, Mahalanobis against chi2_D -------------------
    a = ax[1, 0]
    d2o, d2r = np.sort((z ** 2).sum(2), axis=1), np.sort((zrep ** 2).sum(2), axis=1)
    cq = st.chi2.ppf(q, D)
    a.fill_between(cq, np.percentile(d2r, 2.5, axis=0),
                   np.percentile(d2r, 97.5, axis=0), color="0.80", lw=0)
    a.plot(cq, np.median(d2o, axis=0), color="#c0392b", lw=1.8)
    a.plot([0, cq[-1]], [0, cq[-1]], "k:", lw=0.8)
    a.set_xlabel(f"chi2_{D} quantile")
    a.set_ylabel("Mahalanobis d2")
    a.set_title("joint shape: d2 = |z|^2\n(catches a wrong correlation structure)",
                fontsize=8.5)

    # ---- lower 1: every statistic as a standardised discrepancy -------------
    a = ax[1, 1]
    sub = ppc[ppc["subset"] == "all lesions"]
    lab, val = [], []
    for _, r in sub.iterrows():
        sd = (r["rep_q95"] - r["rep_q05"]) / (2 * 1.645)
        val.append((r["observed"] - r["rep_median"]) / sd if sd > 0 else np.nan)
        short = r["coordinate"].replace("logit_", "").replace("log_", "")
        lab.append(f"{r['statistic']}  {short}")
    y = np.arange(len(val))
    col = ["#c0392b" if abs(v) > 1.96 else "#4878a8" for v in val]
    a.barh(y, val, color=col, height=0.7)
    a.axvline(0, color="k", lw=1)
    for s in (-1.96, 1.96):
        a.axvline(s, color="0.5", ls="--", lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels(lab, fontsize=5.5)
    a.invert_yaxis()
    a.set_xlabel("(fitted - reference) / reference SD")
    a.set_title("EFFECT SIZE -- the sensitive readout\n"
                "(red = outside +/- 1.96)", fontsize=8.5)

    # ---- lower 2: two-sided p-values, both subsets --------------------------
    a = ax[1, 2]
    subsets = list(ppc["subset"].unique())
    for k, lb in enumerate(subsets):
        s = ppc[ppc["subset"] == lb]
        a.plot(s["p_two_sided"].to_numpy(float), np.arange(len(s)) + 0.16 * k,
               "o", ms=4, color=("#4878a8" if k == 0 else "#2e8b57"), label=lb)
    a.axvline(0.05, color="#c0392b", ls="--", lw=1.2, label="0.05")
    a.set_xscale("log")
    a.set_yticks(np.arange(len(sub)))
    a.set_yticklabels([l.split("  ")[0] + " " + l.split("  ")[-1]
                       for l in lab], fontsize=5.5)
    a.invert_yaxis()
    a.set_xlabel("two-sided posterior predictive p")
    a.set_title("p-values -- calibrated but weak;\nread with the panel to the "
                "left", fontsize=8.5)
    a.legend(fontsize=6, frameon=False, loc="lower right")

    # ---- lower 3: pooled z against the standard normal ----------------------
    a = ax[1, 3]
    a.hist(z.ravel(), bins=90, density=True, color="#4878a8", alpha=0.65,
           label=f"fitted z  (sd {z.std():.2f})")
    xs = np.linspace(-5, 5, 400)
    a.plot(xs, st.norm.pdf(xs), "k-", lw=1.5, label="N(0, 1)")
    a.set_xlim(-5, 5)
    a.set_xlabel("z, pooled over lesions and draws")
    a.set_yticks([])
    a.set_title("pooled residuals\n(the marginal the model assumes)",
                fontsize=8.5)
    a.legend(fontsize=7, frameon=False)

    # ---- lower 4: the finite-K control --------------------------------------
    a = ax[1, 4]
    kish = les["kish_ess_fitted"].to_numpy(float)
    rms = les["ppc_rms_z"].to_numpy(float)
    dxv = les["diagnosis"].to_numpy()
    for g, c in (("B", "#4878a8"), ("M", "#c0392b")):
        m = dxv == g
        a.scatter(kish[m], rms[m], s=16, color=c, alpha=0.75, label=g)
    a.axvline(KISH_MIN, color="#c0392b", ls="--", lw=1.2,
              label=f"KISH_MIN = {KISH_MIN:g}")
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set_xscale("log")
    ok = np.isfinite(kish) & (kish > 0) & np.isfinite(rms)
    rho = (float(np.corrcoef(np.log(kish[ok]), rms[ok])[0, 1])
           if ok.sum() > 3 else np.nan)
    a.set_xlabel("Kish weight ESS")
    a.set_ylabel("per-lesion RMS |z|")
    a.set_title(f"finite-K control:  r = {rho:+.2f}\n"
                "negative is expected (starved = far);\nwhat matters is the "
                "Kish subset above", fontsize=8.5)
    a.legend(fontsize=6.5, frameon=False)
    for j in range(5, ax.shape[1]):
        ax[1, j].axis("off")

    fig.suptitle("fig 7  posterior predictive check of the STAGE-2 population "
                 "model  --  z should be iid standard normal if "
                 "theta_i ~ N(X_i B, Sigma) holds", fontsize=10)
    _finish(fig, "fig7_ppc.png", out_dir)


def plot_fig2_ksensitivity(ks, out_dir):
    """
    The upper row is the group difference in natural units; the lower row is
    the population SD, which stays on the MODELLING scale because a Gaussian
    SD has no linear-scale counterpart -- on the natural axis the same SD maps
    to a different width at every mean. K keeps its log axis: it is a sample
    size, not one of the parameters.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, D, figsize=(16, 6.5), sharex=True)
    Kv = sorted(ks["K"].unique())
    for j, nm in enumerate(THETA_NAMES):
        sub = ks[ks["coordinate"] == nm].sort_values("K")
        a = ax[0, j]
        a.errorbar(sub["K"], sub["nat_diff_median"],
                   yerr=[sub["nat_diff_median"] - sub["nat_diff_q05"],
                         sub["nat_diff_q95"] - sub["nat_diff_median"]],
                   fmt="o-", color="#4878a8", capsize=3, lw=1.3, ms=4,
                   label="90% CrI")
        a.errorbar(sub["K"], sub["nat_diff_median"], yerr=3 * sub["nat_diff_mcse"],
                   fmt="none", ecolor="#c0392b", elinewidth=3, capsize=0,
                   label="3 x MCSE")
        a.axhline(0, color="k", ls="--", lw=1)
        a.set_xscale("log")
        a.set_xticks(Kv)
        a.set_xticklabels([str(k) for k in Kv])
        a.set_title(NAT_LABEL[j], fontsize=9)
        if j == 0:
            a.set_ylabel("M - B, natural units")
            a.legend(fontsize=7, frameon=False)
        a2 = ax[1, j]
        a2.errorbar(sub["K"], sub["pop_sd_median"],
                    yerr=[sub["pop_sd_median"] - sub["pop_sd_q05"],
                          sub["pop_sd_q95"] - sub["pop_sd_median"]],
                    fmt="s-", color="#2e8b57", capsize=3, lw=1.3, ms=4)
        a2.errorbar(sub["K"], sub["pop_sd_median"], yerr=3 * sub["pop_sd_mcse"],
                    fmt="none", ecolor="#c0392b", elinewidth=3, capsize=0)
        a2.set_xscale("log")
        a2.set_xticks(Kv)
        a2.set_xticklabels([str(k) for k in Kv])
        a2.set_xlabel("K")
        if j == 0:
            a2.set_ylabel("population SD\n(modelling scale)")
    fig.suptitle("fig 2  K-sensitivity: a slope outside the red Monte-Carlo "
                 "bars is the finite-K approximation", fontsize=11)
    _finish(fig, "fig2_k_sensitivity.png", out_dir)


def plot_fig3_weights(lesion, degen_out, out_dir):
    import matplotlib.pyplot as plt
    n = 3 if degen_out is None else 4
    fig, ax = plt.subplots(1, n, figsize=(4.4 * n, 4.2))
    ke = lesion["kish_ess_fitted"].to_numpy()
    acc = lesion["accept"].to_numpy()
    sids = lesion["subject_id"].to_list()

    a = ax[0]
    o = np.argsort(ke)
    a.plot(np.arange(len(ke)), ke[o], "o-", ms=3, lw=1, color="#4878a8")
    a.axhline(KISH_MIN, color="#c0392b", ls="--", lw=1.3,
              label=f"KISH_MIN = {KISH_MIN:g}")
    a.set_yscale("log")
    a.set_xlabel("lesion, sorted")
    a.set_ylabel("Kish weight ESS")
    a.set_title(f"effective slips per jar\n{int((ke < KISH_MIN).sum())} of "
                f"{len(ke)} below the line", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    for rank in range(min(6, len(o))):           # the six worst, by sorted rank
        i = o[rank]
        a.annotate(sids[i], (rank, ke[i]), fontsize=6,
                   xytext=(6, 9 if rank % 2 else -11),   # stagger: names collide
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))

    a = ax[1]
    bad = ke < KISH_MIN
    a.scatter(acc[~bad], ke[~bad], s=16, color="#4878a8", label="kept")
    a.scatter(acc[bad], ke[bad], s=26, color="#c0392b", label="degenerate")
    a.set_xscale("log")
    a.set_yscale("log")
    a.axhline(KISH_MIN, color="#c0392b", ls="--", lw=1)
    a.set_xlabel("MH acceptance")
    a.set_ylabel("Kish weight ESS")
    a.set_title("acceptance and weight starvation\nare different failures",
                fontsize=9)
    a.legend(fontsize=7, frameon=False)

    a = ax[2]
    a.scatter(lesion["n_distinct"], ke, s=16, color="#4878a8")
    a.axhline(KISH_MIN, color="#c0392b", ls="--", lw=1)
    a.axvline(20, color="k", ls=":", lw=1, label="old distinct<20 rule")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("distinct draws visited")
    a.set_ylabel("Kish weight ESS")
    a.set_title("distinct draws grow with run length;\nKish ESS does not",
                fontsize=9)
    a.legend(fontsize=7, frameon=False)

    if degen_out is not None:
        # natural units, so the five rows have five different units: each is
        # drawn as a fraction of its own "all lesions" interval width, with
        # the numbers in the tick labels
        a = ax[3]
        y = np.arange(len(THETA_NAMES))
        w = np.maximum(degen_out["nat_q95_all"].to_numpy()
                       - degen_out["nat_q05_all"].to_numpy(), 1e-300)
        for off, col, lab, qa, qb, md in (
                (-0.15, "#4878a8", "all lesions", "nat_q05_all", "nat_q95_all",
                 "nat_diff_all"),
                (0.15, "#c0392b", f"Kish >= {KISH_MIN:g}", "nat_q05_kept",
                 "nat_q95_kept", "nat_diff_kept")):
            c = degen_out[md].to_numpy() / w
            a.errorbar(c, y + off,
                       xerr=[c - degen_out[qa].to_numpy() / w,
                             degen_out[qb].to_numpy() / w - c],
                       fmt="o", color=col, capsize=3, ms=5, label=lab)
        a.axvline(0, color="k", ls="--", lw=1)
        a.set_yticks(y)
        a.set_yticklabels(
            [f"{NAT_SHORT[j]}\n{degen_out['nat_diff_all'].iloc[j]:+.3g} -> "
             f"{degen_out['nat_diff_kept'].iloc[j]:+.3g}" for j in range(D)],
            fontsize=7)
        a.invert_yaxis()
        a.set_xlabel("M - B, natural units,\nscaled by each row's own CrI width")
        a.set_title("dropping the starved jars", fontsize=9)
        a.legend(fontsize=7, frameon=False)
    fig.suptitle("fig 3  weight diagnostics", fontsize=11)
    _finish(fig, "fig3_weights.png", out_dir)


def plot_fig4_derived(dv, pr, lesion, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.2))

    for a, (vb, vm, lab) in zip(ax[:2],
                                ((dv["Kt_b"], dv["Kt_m"], "Ktrans [1/min]"),
                                 (dv["E_b"], dv["E_m"], "extraction fraction E"))):
        a.hist(vb, bins=70, density=True, alpha=0.6, color="#4878a8",
               label=f"benign  {np.median(vb):.4g}")
        a.hist(vm, bins=70, density=True, alpha=0.6, color="#c0392b",
               label=f"malignant {np.median(vm):.4g}")
        a.set_xlabel(lab)
        a.set_yticks([])
        a.set_title(f"{lab} at the population median\n"
                    f"P(M>B) = {np.mean(np.log(vm / vb) > 0):.3f}", fontsize=9)
        a.legend(fontsize=7, frameon=False)

    a = ax[2]
    kt = lesion["ktrans_stage2_median_per_min"].to_numpy()
    dxv = lesion["diagnosis"].to_numpy()
    rng = np.random.default_rng(0)
    for k, (g, c) in enumerate((("B", "#4878a8"), ("M", "#c0392b"))):
        v = kt[dxv == g]
        a.scatter(np.full(v.size, k) + rng.uniform(-0.12, 0.12, v.size), v,
                  s=14, color=c, alpha=0.7)
        a.hlines(np.median(v), k - 0.25, k + 0.25, color="k", lw=2)
    a.set_xticks([0, 1])
    a.set_xticklabels(["benign", "malignant"])
    a.set_ylim(bottom=0)
    a.set_ylabel("per-lesion Ktrans [1/min]")
    a.set_title("per-lesion Ktrans from the stage-2 draws\n(bar = group median)",
                fontsize=9)

    a = ax[3]
    a.plot([-1, 1], [-1, 1], "k--", lw=1)
    a.scatter(pr["within_stage1"], pr["population"], s=40, color="#4878a8")
    for _, r in pr.iterrows():
        if abs(r["population"]) > 0.25 or abs(r["abs_gap"]) < 0.15:
            a.annotate(r["pair"].replace("log_", "").replace("logit_", ""),
                       (r["within_stage1"], r["population"]), fontsize=6,
                       xytext=(4, 3), textcoords="offset points")
    a.axhline(0, color="k", lw=0.6)
    a.axvline(0, color="k", lw=0.6)
    a.set_xlim(-1, 1)
    a.set_ylim(-1, 1)
    a.set_xlabel("median WITHIN-lesion correlation (stage 1)")
    a.set_ylabel("fitted POPULATION correlation")
    a.set_title("points near the diagonal may be the TCXM ridge,\n"
                "not between-lesion biology", fontsize=9)
    fig.suptitle("fig 4  derived quantities and correlation provenance",
                 fontsize=11)
    _finish(fig, "fig4_derived.png", out_dir)


def plot_fig5_shrinkage(lesion, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, D, figsize=(3.4 * D, 3.6))
    for j, p in enumerate(PARAMS):
        a = ax[j]
        # already in display units: step 7 builds these with natural()
        x = lesion[f"{p}_stage1_median"].to_numpy(float)
        y = lesion[f"{p}_stage2_median"].to_numpy(float)
        dxv = lesion["diagnosis"].to_numpy()
        for g, c in (("B", "#4878a8"), ("M", "#c0392b")):
            m = dxv == g
            a.scatter(x[m], y[m], s=14, color=c, alpha=0.75, label=g)
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.ticklabel_format(style="plain", useOffset=False)
        a.set_xlabel(f"stage-1 median {NAT_LABEL[j]}")
        if j == 0:
            a.set_ylabel("stage-2 (shrunken) median")
            a.legend(fontsize=7, frameon=False)
        a.set_title(NAT_LABEL[j], fontsize=9)
    fig.suptitle("fig 5  partial pooling: distance below the dashed line is "
                 "how far each lesion was pulled toward its group",
                 fontsize=11)
    _finish(fig, "fig5_shrinkage.png", out_dir)


# =============================================================================
def main():
    t_start = time.time()
    out_dir = OUT_DIR or os.path.join(
        RUN_DIR, "stage2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"stage-1 run : {RUN_DIR}")
    print(f"output      : {out_dir}")

    from dce_mri.kinetic_models import TCXM
    lb4, ub4 = np.asarray(TCXM.lb, float), np.asarray(TCXM.ub, float)
    print(f"TCXM bounds : lb {lb4}  ub {ub4}   tau_i {TAU_I_BOUNDS}")
    print("              (compare with 'TCXM bounds at run time' in the stage-1 log)")
    log_p1 = make_log_p1(PRIOR_CHOICES, lb4, ub4, TAU_I_BOUNDS)

    kept, dx, draws, inc = step1_inclusion(RUN_DIR)
    inc.to_csv(os.path.join(out_dir, "inclusion.csv"), index=False)
    s1_full, XC = step2_stage1(kept, draws, log_p1)
    if PREFLIGHT:
        step3_preflight(XC, kept, log_p1)
    else:
        print("\n3.  PRE-FLIGHT skipped (PREFLIGHT = False)")
    del XC
    X, coef_names = step4_design(kept, dx, draws)
    del draws                      # the draws now live in s1_full
    s1 = subsample_s1(s1_full, K)
    fits = step5_fit(s1, X, coef_names, out_dir)
    lesion, dg = step6_diagnostics(fits, s1, X, coef_names)
    summ, lesion = step7_results(fits, s1, X, coef_names, dx, lesion)

    ks, fits_by_K, kish_by_K = (None, {}, {})
    if RUN_K_SENSITIVITY:
        ks, fits_by_K, kish_by_K = step8_k_sensitivity(s1_full, X, coef_names)
        ks.to_csv(os.path.join(out_dir, "stage2_k_sensitivity.csv"), index=False)

    degen_out = None
    if RUN_DEGENERATE_SENSITIVITY:
        degen_out = step9_degenerate_sensitivity(
            s1_full, X, coef_names, lesion["kish_ess_fitted"].to_numpy(),
            list(lesion["subject_id"]), fits_by_K)
        if degen_out is not None:
            degen_out.to_csv(os.path.join(out_dir, "stage2_degenerate_sens.csv"),
                             index=False)

    drv = pr = dv = None
    if RUN_DERIVED:
        drv, pr, dv = step10_derived(fits, s1, X, dx, lesion)
        drv.to_csv(os.path.join(out_dir, "stage2_derived.csv"), index=False)
        pr.to_csv(os.path.join(out_dir, "stage2_correlation_provenance.csv"),
                  index=False)
        lesion = lesion.assign(
            ktrans_stage2_median_per_min=dv["kt_les"])

    ppc = ppc_les = ppc_raw = None
    if RUN_PPC:
        ppc, ppc_les, ppc_raw = step11_ppc(fits, s1, X, lesion)
        ppc.to_csv(os.path.join(out_dir, "stage2_ppc.csv"), index=False)
        ppc_les.to_csv(os.path.join(out_dir, "stage2_ppc_lesions.csv"),
                       index=False)

    summ.to_csv(os.path.join(out_dir, "stage2_summary.csv"), index=False)
    lesion.to_csv(os.path.join(out_dir, "stage2_lesions.csv"), index=False)
    dg.to_csv(os.path.join(out_dir, "stage2_convergence.csv"), index=False)
    settings = dict(RUN_DIR=RUN_DIR, PARAMS=PARAMS, THETA_NAMES=THETA_NAMES,
                    PRIOR_CHOICES=PRIOR_CHOICES, TAU_I_BOUNDS=TAU_I_BOUNDS,
                    REQUIRE_QC_PASS=REQUIRE_QC_PASS, POP_ESS_MIN=POP_ESS_MIN,
                    EXCLUDE=list(EXCLUDE), K=K, K_LOAD=K_LOAD,
                    ADD_SIZE_COVARIATE=ADD_SIZE_COVARIATE, N_TUNE=N_TUNE,
                    N_DRAWS=N_DRAWS, THETA_SWEEPS=THETA_SWEEPS, SEEDS=SEEDS,
                    TAU0=TAU0, NU0=NU0, PSI0=PSI0, n_lesions=len(kept),
                    coef_names=coef_names, KISH_MIN=KISH_MIN,
                    RUN_K_SENSITIVITY=RUN_K_SENSITIVITY, K_GRID=list(K_GRID),
                    SENS_N_TUNE=SENS_N_TUNE, SENS_N_DRAWS=SENS_N_DRAWS,
                    SENS_SEEDS=list(SENS_SEEDS),
                    TAU_I_AS_RATE=TAU_I_AS_RATE,
                    NAT_LABEL=list(NAT_LABEL),
                    RUN_PPC=RUN_PPC, PPC_DRAWS=PPC_DRAWS, PPC_SEED=PPC_SEED)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2, default=str)

    if PLOTS:
        print()
        print("=" * 78)
        print("PLOTS")
        print("=" * 78)
        plot_fig1_differences(fits, summ, out_dir)
        if ks is not None:
            plot_fig2_ksensitivity(ks, out_dir)
        plot_fig3_weights(lesion, degen_out, out_dir)
        if dv is not None:
            plot_fig4_derived(dv, pr, lesion, out_dir)
        plot_fig5_shrinkage(lesion, out_dir)
        plot_fig6_forest(fits, summ, lesion, out_dir)
        if ppc is not None:
            plot_fig7_ppc(ppc, ppc_les, ppc_raw, out_dir)

    print()
    print("wrote inclusion.csv, stage2_summary.csv, stage2_lesions.csv, "
          "stage2_convergence.csv, settings.json, stage2_seed*.npz")
    if ks is not None:
        print("      stage2_k_sensitivity.csv")
    if degen_out is not None:
        print("      stage2_degenerate_sens.csv")
    if drv is not None:
        print("      stage2_derived.csv, stage2_correlation_provenance.csv")
    if ppc is not None:
        print("      stage2_ppc.csv, stage2_ppc_lesions.csv")
    if PLOTS and SAVE_PLOTS:
        print("      fig1..fig7 PNGs")
    print(f"total {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()