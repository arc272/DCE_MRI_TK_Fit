#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:46:21 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_15_batched_vs_sequential_posterior.py
===========================================
Does a lesion fitted inside a BATCH get the same posterior as the same lesion
fitted ALONE by the production code path, at production chain length?

WHY THIS IS NOT check_13b / check_13c
-------------------------------------
Those showed the two paths follow the SAME random stream: with one seed the
draws agreed to ~1e-12 over 40 draws. That cannot survive 3000 draws, and a
failure at 3000 would mean nothing. vmap changes the ORDER of floating-point
reductions in the gradient, so the two trajectories differ by ~1e-16 at step 1;
Hamiltonian dynamics are chaotic, so that difference doubles every few steps
until the two trajectories are unrelated point by point. A bitwise test at
production length is guaranteed to fail for a reason that does not matter.

At production length the question is therefore DISTRIBUTIONAL: do the two
paths sample the same posterior? Any such test must be CALIBRATED, because two
runs of the SAME method with different seeds also differ, through MCMC noise
alone. Hence:

    cross pairs : batched run  vs  sequential run      <- what is tested
    null  pairs : batched vs batched, seq vs seq       <- the yardstick

A difference counts only if the cross pairs differ by MORE than the null pairs
do. Every run uses a DIFFERENT seed, so all pairs are independent-seed pairs
and the two groups are comparable. (Giving a batched and a sequential run the
same seed would put them on one random stream -- the check_13c setup -- which
is the wrong comparison here.)

STAGES
------
0  PRE-FLIGHT (~2 min, optional)
   One seed, 10 tune + 30 draws, batched vs fit_curve_bayes on these real
   lesions. Confirms the two paths are wired identically -- same priors, same
   anchor, same AIF handling -- BEFORE spending hours. A config mismatch shows
   up here as a large gap, not as a mysterious Stage-2 failure.

1  LONG RUNS
   N_BATCHED batched runs (all SIDS together in one batch) and N_SEQ
   sequential runs (fit_curve_bayes, one lesion at a time), at
   N_TUNE + N_DRAWS with N_CHAINS chains. Each run is written to its own .npz
   the moment it finishes and an existing .npz is reused, so the script is
   resumable, survives an interrupted session, and the analysis can be redone
   without refitting (ANALYZE_ONLY = True).

2  ANALYSIS, per lesion and per parameter
   GATE    within-run rank-normalised split R-hat, ESS, divergences and
           tie_frac (the fraction of a chain's draws that repeat a value it
           already visited -- large means a STUCK chain, whose draws are not a
           sample from anything) for every run. Nothing below is interpretable
           if the runs have not converged.
   T_RHAT  rank-normalised split R-hat on the 2*N_CHAINS POOLED chains of a
           pair. Two runs of one posterior pool to R-hat ~ 1; two runs of
           different posteriors do not, because the between-chain variance
           then picks up the difference. This is the sharpest of the three.
   T_Z     (median_1 - median_2) / sqrt(mcse_median_1^2 + mcse_median_2^2).
   T_W1    Wasserstein-1 distance / pooled SD. Catches a change of shape or
           spread that leaves the median alone.
   POWER   the same statistics applied to a COPY of one run shifted by k x SD,
           k = SHIFT_SDS, compared against its own null partner. This is what
           licenses the conclusion: a pass means "no difference larger than
           about k_min x SD", never "no difference".

   A Kolmogorov-Smirnov p-value is deliberately NOT reported. Its null
   distribution assumes independent draws; MCMC draws are autocorrelated, so
   with 48000 draws of ESS ~ 3000 the p-value would be wrong by orders of
   magnitude. The statistics above are all compared against an empirical null
   instead of an assumed one.

   The three PASS thresholds (see THRESHOLDS below) are judgment calls, not
   derived quantities. The printed cross-vs-null numbers are the evidence;
   the verdict is a convenience.

COST
----
From the same-lesion benchmark on this GPU (FMB001, FXL, 16 chains, GPU not
shared): batched B=4 at 3000+3000 ~ 22 min per run; sequential over 4 lesions
~ 46 min per run. The default 2 + 2 is therefore ~2.3 h. The script prints its
own estimate up front and the per-run times it actually measures.

Run on ONE GPU with nothing else on it -- sharing a GPU inflated every timing
in the benchmark by 2.6x:
    JAX_ENABLE_X64=1 PYTHONPATH=src python check_15_batched_vs_sequential_posterior.py
"""
import os
import glob
import time
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "1"      # set BEFORE jax is imported
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, wasserstein_distance

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            AifConfig, RelaxationConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.bayes import BayesConfig, fit_curve_bayes
from dce_mri.bayes_blackjax import _diagnostics
from dce_mri import bayes_batched as bb

# =============================================================================
# settings
# =============================================================================
RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
           "run_grasp_8spf_2026100_beta&logflat")       # has <sid>_draws.npz
SIDS = ("FMB001", "FMB003", "FMB005", "FMB006")

MODE = "FXL"                    # "FXL" (joint delay, D=5) | "NXL" (delay fixed
                                # at each lesion's FXL tau_delay median, D=4)

N_TUNE = 3000
N_DRAWS = 3000
N_CHAINS = 16

BATCHED_SEEDS = (1001, 1002)    # one batched run per seed (all SIDS together)
SEQ_SEEDS = (2001, 2002)        # one sequential sweep per seed
# All four seeds differ, on purpose: see the header.

DO_PREFLIGHT = True
PREFLIGHT_SEED = 7
PREFLIGHT_TUNE, PREFLIGHT_DRAWS = 10, 30

ANALYZE_ONLY = False            # True: fit nothing, just re-read the .npz files
OUT_DIR = os.path.expanduser(f"~/check15_{MODE}_{N_TUNE}x{N_DRAWS}")

SHIFT_SDS = (0.01, 0.02, 0.05, 0.10, 0.20)      # power calibration

# gate and verdict thresholds
RHAT_GATE, ESS_GATE, TIE_GATE = 1.01, 400.0, 0.02
# The cross and null groups hold DIFFERENT NUMBERS of comparisons (4 pairs vs 2
# with the default seeds), and the maximum of a larger sample is larger even
# when both are drawn from the same distribution. So the tests compare the 95th
# PERCENTILE, which does not grow with the group size; the maxima are printed
# for information but not used as the verdict.
T_QUANTILE = 0.95
T_RHAT_FLOOR, T_RHAT_SLACK = 1.01, 1.5
T_Z_FLOOR, T_Z_SLACK = 3.0, 1.5
T_W1_SLACK = 1.5

# acquisition / subject: the same fixed values fit_lesions_bayes.py uses for
# every lesion. They MUST be shared -- fit_curve_bayes_batched refuses a batch
# whose lesions differ in TR/FA/T10/B1/r1/blood-T1/hematocrit, because one
# compiled forward model is shared across the batch.
NBASE = slice(0, 7)
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
subject = SubjectConfig(subject_id="check15", T10=1530.0, B1=1.0)
cfg_fit = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)          # NXL only
relax_cfg = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005)   # NXL only
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS = (0.0, 25.0)

# per-run cost used only for the up-front estimate (minutes, from the benchmark)
EST_MIN_BATCHED = 22.0
EST_MIN_SEQ_PER_LESION = 11.4


# =============================================================================
# diagnostics written out here rather than taken from arviz
# =============================================================================
def _split_chains(ch):
    """(C, N) -> (2C, N//2): the 'split' in split-R-hat."""
    ch = np.asarray(ch, float)
    C, N = ch.shape
    h = N // 2
    return np.concatenate([ch[:, :h], ch[:, h:2 * h]], axis=0)


def _rank_normalise(x):
    """Pooled fractional ranks pushed through the normal quantile function."""
    r = rankdata(x)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _plain_rhat(s):
    """Classic R-hat on an already split (and here rank-normalised) array."""
    C, N = s.shape
    W = s.var(axis=1, ddof=1).mean()
    B = N * s.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.inf
    return float(np.sqrt(((N - 1) / N * W + B / N) / W))


def rank_split_rhat(ch):
    """
    Rank-normalised split R-hat (Vehtari et al. 2021) for (C, N) draws.

    A second opinion only: az.rhat below is what the numbers are taken from,
    because it is what the rest of the pipeline reports. The two agree to
    ~1e-4 on chains that move, and diverge when draws REPEAT -- a stuck chain
    makes many draws exactly equal, and the two differ in how they rank ties.
    So a large gap between them is a symptom of stuck chains, which tie_frac
    in the gate reports directly; it is not a disagreement about R-hat.
    """
    s = _split_chains(ch)
    return _plain_rhat(_rank_normalise(s.ravel()).reshape(s.shape))


def az_rhat(ch):
    """
    Rank-normalised split R-hat from arviz for (C, N) draws -- the same call
    the pipeline's _diagnostics makes, so it behaves identically here.

    Pooling two runs needs nothing special: stack their chains into one
    (2C, N) array and arviz rank-normalises over all of them together, which
    is exactly what makes the statistic notice one group sitting off to one
    side.
    """
    import arviz as az
    return float(np.asarray(az.rhat(np.asarray(ch, float))))


def tie_frac(ch):
    """
    Largest fraction of a chain's draws that repeat a value already seen in
    that chain. ~0 for a chain that is moving; large for one that is stuck and
    rejecting nearly every proposal. A stuck chain breaks every comparison
    below, so this is a gate quantity, not a curiosity.
    """
    ch = np.asarray(ch, float)
    return float(max(1.0 - np.unique(c).size / c.size for c in ch))


# =============================================================================
# lesions
# =============================================================================
def load_lesions(run_dir, sids):
    """
    Pull St, aif, t -- and, for NXL, the FXL tau_delay posterior median --
    out of the <sid>_draws.npz files of an existing fit_lesions_bayes run.
    """
    out = []
    for sid in sids:
        f = os.path.join(run_dir, f"{sid}_draws.npz")
        if not os.path.exists(f):
            have = sorted(os.path.basename(p).split("_draws")[0]
                          for p in glob.glob(os.path.join(run_dir, "*_draws.npz")))
            raise FileNotFoundError(
                f"{f} not found. {run_dir} has: {have}")
        z = np.load(f, allow_pickle=True)
        names = [str(n) for n in z["param_names"]]
        delay = None
        if "tau_delay" in names:
            delay = float(np.median(z["chains"][..., names.index("tau_delay")]))
        out.append(dict(sid=sid, St=np.asarray(z["St"], float),
                        aif=np.asarray(z["aif"], float),
                        t=np.asarray(z["t"], float), delay=delay))
    t0 = out[0]["t"]
    for L in out[1:]:
        if not np.allclose(L["t"], t0):
            raise ValueError(f"{L['sid']} has a different time grid from {out[0]['sid']}")
        if L["St"].shape != out[0]["St"].shape:
            raise ValueError(f"{L['sid']} has {L['St'].size} frames vs "
                             f"{out[0]['St'].size}")
    if MODE == "NXL" and any(L["delay"] is None for L in out):
        bad = [L["sid"] for L in out if L["delay"] is None]
        raise ValueError(f"MODE='NXL' needs an FXL tau_delay posterior to fix the "
                         f"delay, but these have no tau_delay column: {bad}")
    return out


def make_cfg(n_tune, n_draws, seed, delay_fixed=None):
    return BayesConfig(
        prior_choices=PRIOR_CHOICES, n_draws=n_draws, n_tune=n_tune,
        n_chains=N_CHAINS, noise="gauss_marginalized_sigma", seed=int(seed),
        backend="blackjax", step="nuts", dense_mass_matrix=True,
        delay_mode=("joint" if MODE == "FXL" else "fixed"),
        delay_fixed=(0.0 if delay_fixed is None else float(delay_fixed)),
        delay_bounds=DELAY_BOUNDS, progressbar=False)


WX_KW = dict(wx_model=NXL, aif_cfg=aif_cfg, relax_cfg=relax_cfg) if MODE == "NXL" else {}


# =============================================================================
# the two fitting paths
# =============================================================================
def fit_batched(lesions, seed, n_tune, n_draws):
    """All lesions in one batch -> (L, C, N, D), param_names, divergences."""
    cb = make_cfg(n_tune, n_draws, seed)
    res = bb.fit_curve_bayes_batched(
        [L["St"] for L in lesions], [L["aif"] for L in lesions],
        lesions[0]["t"], TCXM, acq, subject, cfg_fit, cb,
        delay_fixed=([L["delay"] for L in lesions] if MODE == "NXL" else None),
        seeds=[seed] * len(lesions), verbose=True, **WX_KW)
    return (np.stack([r.chains for r in res]), list(res[0].param_names),
            np.array([r.n_divergences for r in res]))


def fit_sequential(lesions, seed, n_tune, n_draws):
    """One lesion at a time through the production fit_curve_bayes."""
    chains, ndiv, names = [], [], None
    for L in lesions:
        cb = make_cfg(n_tune, n_draws, seed, delay_fixed=L["delay"])
        t0 = time.time()
        r = fit_curve_bayes(L["St"], L["aif"], L["t"], TCXM, acq, subject,
                            cfg_fit, cb, **WX_KW)
        print(f"    [seq] {L['sid']}: {time.time() - t0:.0f} s", flush=True)
        chains.append(r.chains)
        ndiv.append(r.n_divergences)
        names = list(r.param_names)
    return np.stack(chains), names, np.array(ndiv)


def run_path(path, kind, seed, lesions):
    """Fit if path is missing, else reuse. Returns the loaded dict."""
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        print(f"  reuse {os.path.basename(path)}  "
              f"({float(z['wall_s']) / 60:.1f} min when it was run)", flush=True)
        return dict(chains=z["chains"], param_names=[str(n) for n in z["param_names"]],
                    n_div=z["n_div"], wall_s=float(z["wall_s"]))
    if ANALYZE_ONLY:
        raise FileNotFoundError(f"ANALYZE_ONLY=True but {path} does not exist")
    print(f"  FIT {kind} seed {seed}  ->  {os.path.basename(path)}", flush=True)
    t0 = time.time()
    fn = fit_batched if kind == "batched" else fit_sequential
    ch, names, ndiv = fn(lesions, seed, N_TUNE, N_DRAWS)
    wall = time.time() - t0
    np.savez_compressed(path, chains=ch, param_names=np.array(names, dtype=object),
                        n_div=ndiv, wall_s=wall,
                        sids=np.array([L["sid"] for L in lesions], dtype=object))
    print(f"  done in {wall / 60:.1f} min  (divergences per lesion: {list(ndiv)})",
          flush=True)
    return dict(chains=ch, param_names=names, n_div=ndiv, wall_s=wall)


# =============================================================================
# Stage 0 -- pre-flight
# =============================================================================
def preflight(lesions):
    print("\n" + "=" * 100)
    print(f"STAGE 0  pre-flight: same seed ({PREFLIGHT_SEED}), "
          f"{PREFLIGHT_TUNE} tune + {PREFLIGHT_DRAWS} draws, batched vs fit_curve_bayes")
    print("=" * 100, flush=True)
    t0 = time.time()
    bat, names, _ = fit_batched(lesions, PREFLIGHT_SEED, PREFLIGHT_TUNE, PREFLIGHT_DRAWS)
    seq, names2, _ = fit_sequential(lesions, PREFLIGHT_SEED, PREFLIGHT_TUNE, PREFLIGHT_DRAWS)
    if names != names2:
        raise ValueError(f"parameter names differ between paths: {names} vs {names2}")
    rows = []
    for i, L in enumerate(lesions):
        sd = seq[i].reshape(-1, seq[i].shape[-1]).std(0)
        gap = np.max(np.abs(seq[i] - bat[i]) / sd, axis=(0, 2))   # per draw
        rows.append(dict(sid=L["sid"], draw0=gap[0], draw5=gap[5],
                         draw_mid=gap[len(gap) // 2], draw_last=gap[-1],
                         worst=gap.max()))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print(f"(max |batched - sequential| / posterior SD, per draw index; "
          f"{time.time() - t0:.0f} s)")
    worst = df["worst"].max()
    if worst < 1e-6:
        print(f"PRE-FLIGHT PASS: the two paths share a random stream "
              f"(worst {worst:.1e} posterior SD). Configurations match.")
    else:
        print(f"PRE-FLIGHT WARNING: worst gap {worst:.1e} posterior SD. Over 30 draws "
              f"the streams should still agree to ~1e-10.\n"
              f"  Large early gaps (draw0 already big) mean the two paths are NOT "
              f"configured the same -- prior, NLLS anchor, AIF shift or delay mode.\n"
              f"  Gaps that grow with draw index are float-order chaos and are "
              f"expected; that is exactly why Stage 2 is distributional.\n"
              f"  Fix a draw0 mismatch before spending hours on Stage 1.")
    return worst


# =============================================================================
# Stage 2 -- analysis
# =============================================================================
def per_run_summaries(runs, lesions, param_names):
    """Within-run R-hat / ESS / MCSE for every run, lesion and parameter."""
    gate, cache = [], {}
    for key, r in runs.items():
        for i, L in enumerate(lesions):
            ch = r["chains"][i]                       # (C, N, D)
            rhat_az, ess, mcse_mean, mcse_med = _diagnostics(ch, param_names)
            for j, p in enumerate(param_names):
                d = ch[:, :, j]
                cache[(key, L["sid"], p)] = dict(
                    draws=d, med=float(np.median(d)), mean=float(d.mean()),
                    sd=float(d.std()), mcse_med=mcse_med[p], mcse_mean=mcse_mean[p])
                gate.append(dict(run=f"{key[0]}-{key[1]}", sid=L["sid"], param=p,
                                 rhat=rhat_az[p], rhat_own=rank_split_rhat(d),
                                 ess=ess[p], n_div=int(r["n_div"][i]),
                                 tie_frac=tie_frac(d),
                                 mcse_med_over_sd=mcse_med[p] / max(float(d.std()), 1e-300)))
    return pd.DataFrame(gate), cache


def pair_stats(a, b):
    """The three comparison statistics for two runs' draws of one parameter."""
    pooled = np.concatenate([a["draws"], b["draws"]], axis=0)      # (2C, N)
    sd_pool = float(np.concatenate([a["draws"].ravel(), b["draws"].ravel()]).std())
    z_med = (a["med"] - b["med"]) / np.sqrt(a["mcse_med"] ** 2 + b["mcse_med"] ** 2)
    z_mean = (a["mean"] - b["mean"]) / np.sqrt(a["mcse_mean"] ** 2 + b["mcse_mean"] ** 2)
    w1 = wasserstein_distance(a["draws"].ravel(), b["draws"].ravel()) / max(sd_pool, 1e-300)
    return dict(rhat_pool=az_rhat(pooled), z_med=z_med, z_mean=z_mean,
                w1_sd=w1, sd_ratio=a["sd"] / max(b["sd"], 1e-300))


def analyse(runs, lesions, param_names):
    gate, cache = per_run_summaries(runs, lesions, param_names)

    print("\n" + "=" * 100)
    print("GATE  within-run convergence (nothing below is interpretable without this)")
    print("=" * 100)
    worst = (gate.groupby(["sid", "param"])
                 .agg(rhat_max=("rhat", "max"), ess_min=("ess", "min"),
                      n_div_max=("n_div", "max"), tie_frac_max=("tie_frac", "max"),
                      mcse_med_over_sd=("mcse_med_over_sd", "max"))
                 .reset_index())
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    gate_ok = bool((worst.rhat_max <= RHAT_GATE).all()
                   and (worst.ess_min >= ESS_GATE).all()
                   and (worst.tie_frac_max <= TIE_GATE).all())
    print(f"GATE: {'PASS' if gate_ok else 'FAIL'}  (R-hat <= {RHAT_GATE}, "
          f"ESS >= {ESS_GATE:.0f}, tie_frac <= {TIE_GATE} for every run)")
    if not gate_ok:
        print("  Runs that have not converged cannot be compared. Raise N_TUNE/N_DRAWS,\n"
              "  or treat everything below as INCONCLUSIVE for the rows that failed.")
    if (worst.tie_frac_max > TIE_GATE).any():
        print("  tie_frac is the fraction of a chain's draws that repeat a value it has\n"
              "  already visited: large means a chain is STUCK, rejecting nearly every\n"
              "  proposal, so its draws are not a sample from anything.")
    d = float(np.abs(gate["rhat"] - gate["rhat_own"]).max())
    print(f"\naz.rhat vs the independent implementation here, over all {len(gate)} "
          f"run/lesion/parameter combinations: max |difference| = {d:.2e}")
    if d > 1e-3:
        print("  The two differ only in how they rank TIED draws, so a gap this size "
              "means stuck\n  chains (see tie_frac), not a disagreement about R-hat. "
              "On chains that move they\n  agree to ~1e-4. All numbers below come from "
              "az.rhat, as the rest of the pipeline does.")
    print(f"\nResolution: the median MCSE is {worst.mcse_med_over_sd.median():.4f} "
          f"posterior SD (worst {worst.mcse_med_over_sd.max():.4f}).\n"
          f"A shift much smaller than that cannot be seen however the test is framed.")

    # ---- pairs -------------------------------------------------------------
    rows = []
    for (ka, kb) in itertools.combinations(list(runs), 2):
        kind = "cross" if ka[0] != kb[0] else f"null-{ka[0]}"
        for L in lesions:
            for p in param_names:
                s = pair_stats(cache[(ka, L["sid"], p)], cache[(kb, L["sid"], p)])
                rows.append(dict(pair=f"{ka[0]}{ka[1]}|{kb[0]}{kb[1]}", kind=kind,
                                 group=("cross" if kind == "cross" else "null"),
                                 sid=L["sid"], param=p, **s))
    pairs = pd.DataFrame(rows)
    pairs["abs_z_med"] = pairs.z_med.abs()
    pairs["abs_z_mean"] = pairs.z_mean.abs()
    pairs.to_csv(os.path.join(OUT_DIR, "pair_statistics.csv"), index=False)

    print("\n" + "=" * 100)
    print("PAIR STATISTICS  cross = batched vs sequential;  null = same method, "
          "different seed")
    print("=" * 100)
    q = lambda s: float(np.quantile(s, T_QUANTILE))
    agg = (pairs.groupby("group")
                .agg(n=("rhat_pool", "size"),
                     rhat_pool_q=("rhat_pool", q), rhat_pool_max=("rhat_pool", "max"),
                     abs_z_med_q=("abs_z_med", q), abs_z_med_max=("abs_z_med", "max"),
                     abs_z_med_mean=("abs_z_med", "mean"),
                     w1_sd_mean=("w1_sd", "mean"), w1_sd_max=("w1_sd", "max"))
                .reset_index())
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nby pair:")
    bypair = (pairs.groupby(["kind", "pair"])
                   .agg(rhat_pool_max=("rhat_pool", "max"),
                        abs_z_med_max=("abs_z_med", "max"),
                        w1_sd_mean=("w1_sd", "mean")).reset_index())
    print(bypair.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    c, n = pairs[pairs.group == "cross"], pairs[pairs.group == "null"]
    if len(n) == 0:
        print("\nNo null pairs: give at least two seeds to one of the two methods, "
              "or nothing below can be calibrated.")
        return pairs, gate_ok, None

    t_rhat_lim = max(T_RHAT_FLOOR, 1.0 + T_RHAT_SLACK * (q(n.rhat_pool) - 1.0))
    t_z_lim = max(T_Z_FLOOR, T_Z_SLACK * q(n.abs_z_med))
    t_w1_lim = T_W1_SLACK * n.w1_sd.mean()
    tests = [("T_RHAT", q(c.rhat_pool), t_rhat_lim),
             ("T_Z", q(c.abs_z_med), t_z_lim),
             ("T_W1", float(c.w1_sd.mean()), t_w1_lim)]
    print("\n" + "=" * 100)
    print(f"VERDICT  on the {T_QUANTILE:.0%} percentile (T_RHAT, T_Z) and the mean (T_W1);"
          f"\nthresholds are judgment calls -- the cross-vs-null numbers above are the "
          f"evidence")
    print("=" * 100)
    for nm, got, lim in tests:
        print(f"  {nm:7s} cross {got:.4g}  vs limit {lim:.4g} from the null  "
              f"-> {'PASS' if got <= lim else 'FAIL'}")

    # worst rows, whatever the verdict
    print("\nworst 6 cross pairs by pooled R-hat:")
    print(c.nlargest(6, "rhat_pool")[["pair", "sid", "param", "rhat_pool",
                                      "abs_z_med", "w1_sd", "sd_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---- power -------------------------------------------------------------
    nulls = [k for k in itertools.combinations(list(runs), 2) if k[0][0] == k[1][0]]
    power = None
    if nulls:
        ka, kb = nulls[0]
        prows = []
        for k in (0.0,) + tuple(SHIFT_SDS):
            for L in lesions:
                for p in param_names:
                    a = cache[(ka, L["sid"], p)]
                    b0 = cache[(kb, L["sid"], p)]
                    b = dict(b0, draws=b0["draws"] + k * b0["sd"],
                             med=b0["med"] + k * b0["sd"],
                             mean=b0["mean"] + k * b0["sd"])
                    s = pair_stats(a, b)
                    prows.append(dict(shift_sd=k, sid=L["sid"], param=p, **s))
        pw = pd.DataFrame(prows)
        pw["abs_z_med"] = pw.z_med.abs()
        power = (pw.groupby("shift_sd")
                   .agg(rhat_pool_q=("rhat_pool", q), abs_z_med_q=("abs_z_med", q),
                        w1_sd_mean=("w1_sd", "mean")).reset_index())
        power["flagged_by"] = [
            ",".join([nm for nm, v, lim in
                      (("T_RHAT", r.rhat_pool_q, t_rhat_lim),
                       ("T_Z", r.abs_z_med_q, t_z_lim),
                       ("T_W1", r.w1_sd_mean, t_w1_lim)) if v > lim]) or "-"
            for r in power.itertuples()]
        print("\n" + "=" * 100)
        print(f"POWER  one null run ({kb[0]}-{kb[1]}) shifted by k x SD and re-compared "
              f"against {ka[0]}-{ka[1]}")
        print("=" * 100)
        print(power.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        pw.to_csv(os.path.join(OUT_DIR, "power_calibration.csv"), index=False)
        det = power[(power.shift_sd > 0) & (power.flagged_by != "-")]
        if len(det):
            print(f"\nSmallest shift these tests would have caught: "
                  f"{det.shift_sd.min():.3f} posterior SD.\n"
                  f"A PASS therefore means: no batched-vs-sequential difference larger "
                  f"than about that, for these lesions and parameters.")
        else:
            print(f"\nNo tested shift up to {max(SHIFT_SDS):.2f} SD was flagged -- "
                  f"the test has little power here.\nA PASS would say almost nothing; "
                  f"raise N_DRAWS or add SHIFT_SDS values before trusting it.")

    ok = all(got <= lim for _, got, lim in tests)
    print("\n" + "=" * 100)
    print(f"OVERALL: {'PASS' if (ok and gate_ok) else ('FAIL' if gate_ok else 'INCONCLUSIVE')}"
          f"  -- batched and sequential posteriors are "
          f"{'indistinguishable at this resolution' if (ok and gate_ok) else 'NOT shown to agree'}")
    print("=" * 100)
    return pairs, gate_ok, power


# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import jax
    print(f"devices: {jax.devices()}   x64: {jax.config.jax_enable_x64}")
    print(f"MODE={MODE}  {N_CHAINS} chains  {N_TUNE} tune + {N_DRAWS} draws")
    print(f"lesions: {list(SIDS)}")
    print(f"out: {OUT_DIR}")

    lesions = load_lesions(RUN_DIR, SIDS)
    if MODE == "NXL":
        print("fixed delays from the FXL posteriors: "
              + ", ".join(f"{L['sid']} {L['delay']:.2f}s" for L in lesions))

    est = (len(BATCHED_SEEDS) * EST_MIN_BATCHED
           + len(SEQ_SEEDS) * EST_MIN_SEQ_PER_LESION * len(lesions))
    print(f"\nestimated fitting time: {est:.0f} min ({est / 60:.1f} h) for "
          f"{len(BATCHED_SEEDS)} batched + {len(SEQ_SEEDS)} sequential runs"
          + ("  [ANALYZE_ONLY: nothing will be fitted]" if ANALYZE_ONLY else ""))

    if DO_PREFLIGHT and not ANALYZE_ONLY:
        preflight(lesions)

    print("\n" + "=" * 100)
    print("STAGE 1  long runs")
    print("=" * 100, flush=True)
    runs, names = {}, None
    for kind, seeds in (("batched", BATCHED_SEEDS), ("seq", SEQ_SEEDS)):
        for s in seeds:
            path = os.path.join(OUT_DIR, f"run_{kind}_{s}.npz")
            r = run_path(path, kind, s, lesions)
            if names is None:
                names = r["param_names"]
            elif r["param_names"] != names:
                raise ValueError(f"{path} has parameters {r['param_names']} "
                                 f"but earlier runs have {names}")
            runs[(kind, s)] = r

    tot = sum(r["wall_s"] for r in runs.values()) / 60
    print(f"\nall runs present ({tot:.1f} min of fitting in total). parameters: {names}")
    analyse(runs, lesions, names)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()