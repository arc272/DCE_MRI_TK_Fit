#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Mon Oct 19 17:12:32 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_wx_fixedtau_synthetic.py
=================================
Synthetic validation of the model fit_lesions_bayes_wx_fixedtau.py runs:
TCXM + FIXED tau_delay + SSM3 water exchange (tau_i free), sigma marginalized,
5 sampled parameters [ve, vp, fp, ps, tau_i].

Runs on GPU 1 (CUDA_VISIBLE_DEVICES set BEFORE jax is imported, see below) so it
can run alongside fit_lesions_bayes_wx_fixedtau.py on GPU 0.

Every fit function here is a COPY of the one in fit_lesions_bayes_wx_fixedtau.py,
not an import: this script must keep testing the model that script ran even if
that script is later edited. Where a copy had to change, the change is marked
"[copy changed]" in a comment. The copies are:
    derived_draws, _iqr, chain_disagree, _diagnostics, summarise (-> summarise_fit),
    nlls_params, nlls_cost, at_bound, plot_prior_vs_posterior (-> plot_prior_post),
    and the whole acq / cfg_fit / relax_cfg / aif_cfg / PRIOR_CHOICES config block.
Nothing here reads a lesion mask or a NIfTI volume: the curves are simulated, so
_aggregate_roi_signal is not involved. What that means for the tests is in
"What these tests cannot tell you" below.

The five tests
--------------
T1  DELAY HAND-OFF (the one that is specific to the fixed-tau script)
    The delay fit_lesions_bayes_wx_fixedtau.py fixes came from a TCXM + joint
    delay fit with NO water exchange. If the tissue really does exchange slowly,
    that earlier fit can absorb part of the exchange effect into tau_delay, and
    fixing the delay then passes that error into the WX fit.
    Simulate WX data at a known delay and a range of true tau_i -> fit stage 1
    exactly as the source run did (no WX, joint delay) -> fix the ESTIMATED
    delay and fit WX -> repeat fixing the TRUE delay. The gap between the two
    arms is the cost of the hand-off; the trend of the stage-1 delay error
    against tau_i_true says whether that cost is a tau_i-dependent artifact
    (which would contaminate a benign-vs-malignant tau_i comparison) or just
    noise.

T2  DELAY ERROR TOLERANCE
    Same data, delay fixed at tau_true + offset for each offset in
    TOL_OFFSETS_S, same noise realization throughout, so every difference is
    the delay and not the noise. Gives a bias-per-second-of-delay-error table.
    Compare the offsets you consider realistic against the source run's
    tau_delay 95% CrI width (the tau_delay_src_ci_width column).

T3  PRIOR PREDICTIVE CHECK  (no fitting)
    Draw from the priors only (PRIOR_CHOICES, rates uniform / volfrac
    flat_masked / tau log-uniform), simulate curves, and look at what the model
    believes before it sees data: peak enhancement distribution, how many draws
    sit on a bound, and what the flat volume-fraction prior implies for
    vi = 1 - ve - vp, since vi -> 0 leaves tau_i with no pool to describe.

T4  RECOVERY, CALIBRATION, AND tau_i IDENTIFIABILITY
    (a) recovery: truths on an LHS grid over RECOVERY_BOX, delay fixed at the
        TRUE value, so this is the best case the fixed-tau model can do. Bias,
        z-score and 50/90/95% interval coverage per parameter.
    (b) SBC: truths drawn FROM THE PRIOR, rank of truth within the posterior
        draws. Uniform ranks = the sampler and the model are self-consistent.
        Deviation from strict SBC: sigma is marginalized under an improper
        Jeffreys prior, so there is no sigma to draw a truth from. Every
        dataset is generated at NOISE_SD, which makes this SBC *conditional on
        sigma*. Ranks from fits that did not converge are excluded and counted.
    (c) contraction: posterior IQR / prior IQR per parameter, and for tau_i in
        particular, since the prior-dominated regime is where a reported tau_i
        is really just the prior.

T5  MISSPECIFICATION, BOTH DIRECTIONS
    (a) FALSE-POSITIVE EXCHANGE: data generated with NO water exchange, fit
        WITH the WX model. tau_i should run to the fast-exchange edge or stay
        prior-dominated, not come back as a confident finite value. This is
        what keeps "WX fits better" from meaning only "WX has more parameters".
        "No water exchange" turns out to be TWO different datasets here, so
        T5A_NULLS runs both. fastlimit_diagnostic() measures why: in this
        package TCXM+SSM3 as tau_i -> 0 does NOT converge to the plain TCXM
        forward model. On the default settings the two differ by ~27x the noise
        sd, and almost all of it is the (1 - hematocrit) factor
        water_exchange._ssm3_to_signal applies to the vascular term, which
        signal_models.concentration_to_signal does not. The two models are
        therefore not nested, which matters beyond this test: a LOO/WAIC
        comparison of TCXM against TCXM+SSM3 would be measuring that gap as
        well as water exchange. The run prints the decomposition every time.
    (b) data generated WITH exchange, fit WITHOUT it (delay still fixed at the
        true value), to see how ve / vp / fp / ps absorb the exchange effect --
        subject to the same nesting caveat.

What these tests cannot tell you
--------------------------------
  * Nothing here touches the lesion ROI step. The curves are simulated directly
    at one "lesion", so an error in masks, in which reconstruction is read, or
    in averaging over voxels cannot show up. Those are checked by running
    fit_lesions_bayes_wx_fixedtau.py on one real subject (ONLY=[...]).
  * The generative model IS the fitted model (apart from T5). Real breast tissue
    is not TCXM+SSM3, so a parameter that recovers perfectly here can still be
    biased on real data.
  * Coverage from N fits per cell is a binomial estimate: the printed Wilson
    95% interval is the honest width. At the default N it is a smoke test. Raise
    N_RECOVERY / N_SBC / N_MISSPEC for a number you would put in a paper.
  * T4b fits truths drawn from the prior, which includes near-unidentifiable
    corners (fp at its bound, vi ~ 0). Low ESS and divergences there are
    information about the prior, not necessarily a sampler bug.

Units
-----
Fits are in the pipeline's own units: fp, ps, Ktrans in s^-1, tau_i / tau_delay
/ Tc / Te in s. EVERY PLOT converts the rates to min^-1 (x60) and uses LINEAR
axes throughout -- including fp, ps and tau_i, which fit_lesions_bayes_wx.py
plots in log10. The CSVs keep s^-1 so they stay directly comparable with the
real-run CSVs, and add *_per_min copies for fp / ps / Ktrans.
The prior_dominated QC metric is still computed in log10 for the LOG_SCALE
quantities, exactly as in fit_lesions_bayes_wx_fixedtau.py: that is a QC metric,
not a display choice, and changing it would make the flag mean something else.

Outputs (OUT_ROOT/run_synthval_<timestamp>/)
    rows/<tag>.csv                 one row per fit, written as it finishes
                                   (a rerun skips tags that already have a row)
    test1_handoff/ ... test5_misspec/
        <tag>_prior_post.png       prior vs posterior vs TRUTH, every sampled
                                   parameter and every derived quantity
        <tag>_fit.png              data, truth curve, posterior-median curve,
                                   residuals
    test1_stage1_sweep.csv         the NLLS-only delay-error-vs-tau_i preview
                                   (T1_STAGE1_SWEEP), printed before T1's fits
    test1_handoff_summary.csv / .png ... per test
    prior_predictive.png, prior_predictive_summary.csv
    all_fits.csv                   every row from every test
    run_settings.csv               every setting above, for provenance

Run:
    PYTHONPATH=src python scripts/validate_wx_fixedtau_synthetic.py
"""



import os

# =============================================================================
# GPU -- must be set BEFORE anything imports jax (dce_mri.bayes does)
# =============================================================================
# fit_lesions_bayes_wx_fixedtau.py runs on GPU 0; this validation runs on GPU 1.
# With CUDA_VISIBLE_DEVICES="1" jax sees that card as its device 0 -- that is
# expected, and the printed device line below is the check that it is the right
# physical card (compare with nvidia-smi).
GPU_ID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import glob
import time
import warnings
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            RoiFittingConfig, AifConfig, RelaxationConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.signal_models import compute_S0, shift_aif
from dce_mri.synthetic import simulate_curve, add_noise, make_parker_aif
from dce_mri.bayes import (BayesConfig, fit_curve_bayes, build_param_layout,
                           NOISE_OPTIONS)
from dce_mri.bayes_priors import assemble_prior
from dce_mri.fitting import fit_with_delay_continuous

# =============================================================================
# settings
# =============================================================================

OUT_ROOT = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_validation"
RESUME_DIR = None        # path of an earlier run dir to continue (skips done fits)

# Which tests to run, in this order. Cheapest first: T3 needs no fitting, T4
# is the baseline every other test is read against, T1/T2/T5 build on it.
TESTS = [3, 4, 1, 2, 5]

# --- AIF --------------------------------------------------------------------
# "real"  : the AIF CSVs extract_aif_roi.py wrote, so the tests run on the real
#           bolus shape, the real arrival frame and the real time grid. This is
#           the default because the delay tests are about AIF timing.
# "parker": Parker population AIF on a synthetic grid (no AIF_DIR needed).
AIF_SOURCE     = "real"
AIF_DIR        = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"
AIF_SUBJECTS   = []      # [] = the first AIF_N_SUBJECTS CSVs found, sorted
AIF_N_SUBJECTS = 3       # truths are cycled across these AIFs
# "parker" only:
N_FRAMES       = 36
PARKER_ARRIVAL = 35.0    # s. Must land AFTER the baseline window: with
                         # dt = 4.1667 s, frame 6 sits at 25.0 s, so 35 s keeps
                         # mean(aif[nbase]) exactly 0 -- which is what shift_aif
                         # pads the pre-bolus frames with. main() prints that
                         # mean for every AIF and flags it when it is not 0.

# --- acquisition / tissue: COPIED from fit_lesions_bayes_wx_fixedtau.py ------
NBASE          = slice(0, 7)     # the fixed window that script uses
acq            = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
T10_TISSUE, B1 = 1530.0, 1.0
cfg_fit        = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
roi_cfg        = RoiFittingConfig(average_space="signal")   # unused (no ROI here)
SPOKES_PER_FRAME, NSPOKE, SCANTIME = 8, 288, 150
S0_TRUE        = 1000.0          # simulate_curve's S0; the fits recover their
                                 # own S0 from the noisy baseline, as on real data

# --- water exchange: COPIED ---------------------------------------------------
WX_MODEL  = SSM3
relax_cfg = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005,
                             tau_i_lb=0.01, tau_i_ub=1.5)
aif_cfg   = AifConfig(T10=1779.0, hematocrit=0.4)

# --- Bayesian model: COPIED, except the sampler size -------------------------
PRIOR_CHOICES = {"rates": "uniform", "volfrac": "flat_masked", "tau": "flat"}
DELAY_BOUNDS  = (0, 25.0)
BACKEND, DENSE_MASS = "blackjax", True
# [copy changed] fit_lesions_bayes_wx_fixedtau.py uses 4000/4000 x 16 chains.
# These tests are dozens of fits, so the defaults here are smaller. Raise them
# to the production numbers before reading a coverage result as final -- a fit
# that fails QC only because the chains were short is not evidence about the
# model. Every row carries rhat_max / ess_min / n_divergences so you can see
# which it was.
N_DRAWS, N_TUNE, N_CHAINS = 2000, 2000, 8
SEED                 = 1       # base seed; each fit gets SEED + a stable offset
N_PRIOR_SAMPLES      = 20000
PRIOR_SEED           = 999

# --- noise -------------------------------------------------------------------
# "from_run": take median(baseline_sd / S0) from a real run's CSV and apply it
#             to S0_TRUE, so the synthetic SNR matches the cohort's. This is the
#             only transfer that is dimensionless and therefore meaningful.
# "rel_s0"  : NOISE_REL_S0 * S0_TRUE.
NOISE_SD_MODE  = "from_run"
NOISE_FROM_RUN = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
                  "run_grasp_8spf_20261013_180711_allflatprior")
NOISE_REL_S0   = 0.01
NOISE_KIND     = "gaussian"      # matches gauss_marginalized_sigma
NOISE_SEED     = 20261019

# --- truths ------------------------------------------------------------------
# One central operating point, used by T1 / T2 / T5 and as the LHS centre.
# fp = 0.015 s^-1 = 0.90 min^-1, ps = 0.004 s^-1 = 0.24 min^-1 -> E = 0.21,
# vi = 1 - ve - vp = 0.71.
TRUTH_BASE = dict(ve=0.25, vp=0.04, fp=0.015, ps=0.004, tau_i=0.30)

# T1 delay hand-off
T1_TAU_I_TRUE  = [0.05, 0.20, 0.50, 1.00]   # s; spans fast -> slow exchange
T1_TAU_TRUE_S  = [3.0, 6.0]                 # s; true delays
T1_STAGE1      = "bayes"   # "bayes": TCXM + joint delay, no WX, Bayesian --
                           #   the same fit the source run's tau_delay_median
                           #   came from. Use this to test the real hand-off.
                           # "nlls": fit_with_delay_continuous. ~100x cheaper,
                           #   but it tests a delay you are not actually using
                           #   unless TAU_FIXED_COL = "nlls_tau_delay".
T1_N_NOISE     = 1         # noise realizations per (tau_i, tau) cell
# Before any MCMC, run the stage-1 delay fit ALONE (NLLS, no sampling) over
# T1_TAU_I_TRUE x T1_SWEEP_SEEDS and print delay error vs true tau_i. Under a
# minute, and it is the cheapest way to see whether the no-WX fit absorbs
# exchange into the delay -- the failure T1 exists to catch. It fits the same
# curves T1 then fits, so the two agree by construction.
T1_STAGE1_SWEEP = True
T1_SWEEP_SEEDS  = [11, 12, 13, 14, 15]

# T2 delay error tolerance
T2_TAU_TRUE_S  = 6.0
T2_OFFSETS_S   = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

# T3 prior predictive
T3_N_DRAWS     = 2000      # prior draws pushed through the forward model
T3_N_CURVES    = 60        # curves drawn on the figure

# T4 recovery + SBC
N_RECOVERY     = 12
RECOVERY_BOX   = dict(ve=(0.05, 0.45), vp=(0.005, 0.10),
                      fp=(0.003, 0.035), ps=(0.0005, 0.02),
                      tau_i=(0.03, 1.20))
N_SBC          = 16
SBC_N_BINS     = 8
SBC_THIN_TO    = 100       # draws kept per fit for the rank statistic, evenly
                           # spaced, so autocorrelated draws do not inflate the
                           # effective resolution of the rank

# T5 misspecification
N_MISSPEC      = 6         # truths per null (LHS over RECOVERY_BOX)
# Which "no exchange" data direction (a) is generated from. BOTH are run by
# default because they are not the same null -- see fastlimit_diagnostic():
#   "ssm3_fast": SSM3 at tau_i = T5A_TAU_I_FAST, i.e. fast exchange INSIDE the
#                fitted model family. This is the proper false-positive test:
#                the fitted model can represent this truth exactly.
#   "no_wx"    : the plain TCXM forward model, i.e. the model the fixed-tau WX
#                fit would be compared AGAINST. In this package the two are not
#                nested (the vascular term differs by the hematocrit factor and
#                the blood-vs-tissue baseline T1), so this arm measures that gap
#                as well as any false positive. Run it to see what the WX fit
#                does to data from the comparison model, but do not read a
#                finite tau_i here as "false-positive exchange" on its own.
T5A_NULLS      = ["ssm3_fast", "no_wx"]
# s. Just INSIDE the prior's fast edge rather than exactly on it: a truth
# sitting on a bound cannot be covered by a central credible interval (half of
# it is outside the support), and the logit transform stretches the boundary,
# so a fit there is slow and its coverage number means something else. The
# bound_gap line of fastlimit_diagnostic() reports how close this value still
# is to true fast exchange -- on the defaults, within about one noise sd.
T5A_TAU_I_FAST = 2.0 * relax_cfg.tau_i_lb
# "tau_i came back at the fast-exchange edge" = median below this many times
# the prior's lower bound. Must be larger than T5A_TAU_I_FAST/tau_i_lb or the
# criterion would just be restating where the data were generated.
FAST_EDGE_FACTOR = 4.0
T5_TAU_I_TRUE  = [0.05, 0.30, 1.00]    # direction (b): WX data, fitted without WX

PLOT_FITS      = True      # per-fit data/truth/prediction figure
PLOT_DERIVED   = True      # include vtot / vi / E / Ktrans / Tc / Te panels in
                           # the per-fit prior-vs-posterior figure
# x-range of a SAMPLED parameter's panel:
#   "prior"     the full prior support, so prior dominance is visible as a
#               shape -- but a posterior 50x narrower than its prior is then a
#               single invisible spike (vp usually is).
#   "posterior" zoom on the posterior; the prior is still drawn, and the
#               IQR ratio in the panel title still states the global contraction.
#   "auto"      the prior support, unless the posterior spans less than
#               AUTO_ZOOM_FRAC of it, in which case zoom and say so in the panel.
XLIM_SAMPLED   = "auto"
AUTO_ZOOM_FRAC = 0.15

# --- QC thresholds: COPIED ----------------------------------------------------
RHAT_MAX       = 1.01
ESS_MIN        = 400
IQR_RATIO_FLAT = 0.8
E_FLOW_LIMITED = 0.4
CHAIN_SPLIT    = 0.5
MCSE_REL_MAX   = 0.05
VI_SMALL       = 0.05
AT_BOUND_TOL   = 1e-3

DERIVED   = ["vtot", "vi", "E", "Ktrans", "Tc", "Te"]
# COPIED: the IQR-ratio metric is computed in log10 for these, regardless of how
# they are plotted here.
LOG_SCALE = {"fp", "ps", "Ktrans", "Tc", "Te", "tau_i"}

# --- display: linear axes everywhere, rates in min^-1 -------------------------
RATE_SCALE = {"fp": 60.0, "ps": 60.0, "Ktrans": 60.0}
DISP_UNITS = {"fp": "min$^{-1}$", "ps": "min$^{-1}$", "Ktrans": "min$^{-1}$",
              "tau_i": "s", "tau_delay": "s", "Tc": "s", "Te": "s"}

C_PRIOR, C_POST, C_MEDIAN, C_TRUTH, C_NLLS = "0.75", "#B22234", "#E67E22", "#1B4F72", "#1E8449"


def disp(name, x):
    """Native pipeline units -> display units (s^-1 -> min^-1 for the rates)."""
    return np.asarray(x, float) * RATE_SCALE.get(name, 1.0)


def disp_label(name):
    u = DISP_UNITS.get(name, "")
    return f"{name} ({u})" if u else name


# =============================================================================
# COPIED from fit_lesions_bayes_wx_fixedtau.py -- posterior summaries
# =============================================================================

def derived_draws(d):
    """COPIED verbatim. vi = 1 - ve - vp is the pool SSM3's tau_i describes."""
    ve, vp, fp, ps = d["ve"], d["vp"], d["fp"], d["ps"]
    return dict(vtot=ve + vp, vi=1.0 - ve - vp,
                E=ps / (ps + fp), Ktrans=ps * fp / (ps + fp),
                Tc=vp / fp, Te=ve / ps)


def _iqr(x, log=False):
    """COPIED verbatim."""
    x = np.asarray(x, float)
    if log:
        x = np.log10(x[x > 0])
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def chain_disagree(chains_1d):
    """COPIED verbatim."""
    meds = np.median(chains_1d, axis=1)
    within = np.mean(np.percentile(chains_1d, 75, axis=1)
                     - np.percentile(chains_1d, 25, axis=1))
    spread = float(meds.max() - meds.min())
    return spread > CHAIN_SPLIT * within, spread, float(within)


def _diagnostics(chains_1d):
    """COPIED verbatim: R-hat / ESS / MCSE(mean) / MCSE(median) via ArviZ."""
    try:
        import arviz as az
        x = np.asarray(chains_1d, float)
        return dict(
            rhat=float(np.asarray(az.rhat(x))),
            ess=float(np.asarray(az.ess(x))),
            mcse_mean=float(np.asarray(az.mcse(x, method="mean"))),
            mcse_median=float(np.asarray(az.mcse(x, method="quantile", prob=0.5))),
        )
    except Exception:
        return dict(rhat=np.nan, ess=np.nan, mcse_mean=np.nan, mcse_median=np.nan)


def summarise_fit(res, prior_d, params, derived=DERIVED):
    """COPIED from summarise().

    [copy changed] PARAMS is an ARGUMENT, not a module global: this script fits
    three different parameter sets -- [ve,vp,fp,ps,tau_i] (the fixed-tau WX
    model), [ve,vp,fp,ps,tau_delay] (T1 stage 1, no WX, joint delay) and
    [ve,vp,fp,ps] (T5b, no WX, fixed delay) -- and a global would silently
    summarise the wrong one.
    """
    names = list(res.param_names)
    post = {p: res.flat()[:, names.index(p)] for p in params}
    post.update(derived_draws(post))
    ch = {p: res.chains[:, :, names.index(p)] for p in params}
    ch.update(derived_draws(ch))

    out = {}
    for p in list(params) + list(derived):
        x, log = post[p], p in LOG_SCALE
        q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
        out.update({f"{p}_median": q[2], f"{p}_q25": q[1], f"{p}_q75": q[3],
                    f"{p}_ci2.5": q[0], f"{p}_ci97.5": q[4],
                    f"{p}_mean": float(np.mean(x)),
                    f"{p}_sd": float(np.std(x, ddof=1)),
                    f"{p}_iqr_ratio": _iqr(x, log) / _iqr(prior_d[p], log)})
        out[f"{p}_prior_dominated"] = bool(out[f"{p}_iqr_ratio"] > IQR_RATIO_FLAT)

        d = _diagnostics(ch[p])
        if p in params:
            d["rhat"] = float(res.rhat.get(p, d["rhat"]))
            d["ess"] = float(res.ess.get(p, d["ess"]))
            d["mcse_mean"] = float(getattr(res, "mcse_mean", {}).get(p, d["mcse_mean"]))
            d["mcse_median"] = float(
                getattr(res, "mcse_median", {}).get(p, d["mcse_median"]))
            dis, spread, within = chain_disagree(ch[p])
            out[f"{p}_chains_disagree"] = bool(dis)
        out[f"{p}_rhat"] = d["rhat"]
        out[f"{p}_ess"] = d["ess"]
        out[f"{p}_mcse_mean"] = d["mcse_mean"]
        out[f"{p}_mcse_median"] = d["mcse_median"]
        width = float(q[4] - q[0])
        rel = d["mcse_median"] / width if width > 0 else np.nan
        out[f"{p}_mcse_median_rel_ci"] = float(rel)
        out[f"{p}_mcse_rel_flag"] = bool(np.isfinite(rel) and rel > MCSE_REL_MAX)
    return out, post


def nlls_params(nres):
    """COPIED verbatim."""
    v = getattr(nres, "best_params", None)
    return np.asarray(v if v is not None else nres.params, float)


def nlls_cost(nres):
    """COPIED verbatim."""
    v = getattr(nres, "best_cost", None)
    return float(v if v is not None else nres.cost)


def at_bound(x, lo, hi, log):
    """COPIED verbatim."""
    if log:
        x, lo, hi = np.log(x), np.log(lo), np.log(hi)
    tol = AT_BOUND_TOL * (hi - lo)
    return bool(x - lo <= tol or hi - x <= tol)


# =============================================================================
# truth bookkeeping
# =============================================================================

def truth_with_derived(truth):
    """{param: value} -> the same plus every DERIVED quantity, so coverage can
    be checked for Ktrans / E / vi as well as the sampled parameters."""
    out = dict(truth)
    out.update({k: float(v) for k, v in derived_draws(
        {k: float(out[k]) for k in ("ve", "vp", "fp", "ps")}).items()})
    return out


def theta_for(truth, wx):
    """Truth dict -> simulate_curve's theta, in ITS layout:
    [*theta_kin, tau_i?]. The delay is NOT in theta here: it is applied by
    shifting the AIF before simulating, which is exactly what the package's
    delay-augmented spec does internally (fitting.build_delay_augmented_spec
    wraps model.fn with shift_aif), so the two routes are the same model."""
    th = [float(truth[p]) for p in TCXM.param_names]
    if wx:
        th.append(float(truth["tau_i"]))
    return np.asarray(th, float)


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a coverage estimate k/n. Printed with every
    coverage number so the sample size is visible rather than implied."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def lhs(box, n, rng):
    """Latin hypercube over {name: (lo, hi)}. One stratified sample per
    dimension, independently permuted -- so n fits cover each parameter's range
    evenly instead of clustering as plain uniform draws would."""
    names = list(box)
    out = {}
    for nm in names:
        lo, hi = box[nm]
        edges = (np.arange(n) + rng.random(n)) / n
        out[nm] = lo + (hi - lo) * rng.permutation(edges)
    return [{nm: float(out[nm][i]) for nm in names} for i in range(n)]


# =============================================================================
# inputs: AIF + time grid + noise level
# =============================================================================

def load_aifs():
    """[(label, t, aif), ...]. Real AIF CSVs, or the Parker population AIF."""
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    if AIF_SOURCE == "parker":
        t = np.arange(N_FRAMES) * dt
        aif = make_parker_aif(t, t_arrival=PARKER_ARRIVAL,
                              hematocrit=aif_cfg.hematocrit)
        return [("parker", t, aif)]
    if AIF_SOURCE != "real":
        raise ValueError(f"AIF_SOURCE must be 'real' or 'parker', got {AIF_SOURCE!r}")

    paths = ([os.path.join(AIF_DIR, f"{s}_AIF_plasma.csv") for s in AIF_SUBJECTS]
             if AIF_SUBJECTS else
             sorted(glob.glob(os.path.join(AIF_DIR, "*_AIF_plasma.csv")))[:AIF_N_SUBJECTS])
    out = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no AIF CSV at {p}")
        df = pd.read_csv(p)
        aif = df["aif_plasma_mM"].to_numpy(float)
        t = (df["time_s"].to_numpy(float) if "time_s" in df
             else np.arange(aif.size) * dt)
        out.append((os.path.basename(p).replace("_AIF_plasma.csv", ""), t, aif))
    if not out:
        raise FileNotFoundError(
            f"AIF_SOURCE='real' but no *_AIF_plasma.csv in {AIF_DIR}. Run "
            f"extract_aif_roi.py first, or set AIF_SOURCE='parker' to validate "
            f"on the Parker population AIF instead.")
    return out


def noise_sd():
    """(sd in signal units of S0_TRUE, how it was obtained).

    'from_run' transfers the cohort's SNR, not its absolute noise: the real
    run's S0 is in scanner units and has no relation to S0_TRUE, so the only
    transferable quantity is the ratio baseline_sd / S0.
    """
    if NOISE_SD_MODE == "rel_s0":
        return NOISE_REL_S0 * S0_TRUE, f"rel_s0 ({NOISE_REL_S0:.4g} x S0)"
    path = os.path.join(NOISE_FROM_RUN, "lesion_bayes_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NOISE_SD_MODE='from_run' but no CSV at {path}. Point "
            f"NOISE_FROM_RUN at a finished run, or set "
            f"NOISE_SD_MODE='rel_s0'.")
    df = pd.read_csv(path)
    for c in ("baseline_sd", "S0"):
        if c not in df.columns:
            raise KeyError(f"{path} has no {c!r} column, so the cohort noise "
                           f"level cannot be read; set NOISE_SD_MODE='rel_s0'")
    r = (df["baseline_sd"].astype(float) / df["S0"].astype(float)).dropna()
    r = r[np.isfinite(r) & (r > 0)]
    if r.empty:
        raise ValueError(f"{path}: no finite baseline_sd/S0 to take a median of")
    frac = float(r.median())
    return frac * S0_TRUE, (f"from_run median baseline_sd/S0 = {frac:.5f} "
                            f"over {len(r)} lesions ({os.path.basename(NOISE_FROM_RUN)})")


def make_dataset(truth, aif, t, tau_true, wx, sd, seed):
    """(St_noisy, St_clean). The delay is applied by shifting the AIF, so the
    data carry exactly the delay the fixed-tau fit will be told about."""
    aif_sh = shift_aif(np.asarray(aif, float), float(tau_true), t, acq)
    clean = simulate_curve(theta_for(truth, wx), aif_sh, t, TCXM, acq,
                           SubjectConfig(subject_id="SYN", T10=T10_TISSUE, B1=B1),
                           cfg_fit, wx_model=(WX_MODEL if wx else None),
                           aif_cfg=(aif_cfg if wx else None),
                           relax_cfg=(relax_cfg if wx else None), S0=S0_TRUE)
    noisy = add_noise(clean, sd, kind=NOISE_KIND,
                      rng=np.random.default_rng(seed))
    return np.asarray(noisy, float), np.asarray(clean, float)


# =============================================================================
# figures
# =============================================================================

def plot_prior_post(res, layout, prior_samples, post, prior_d, summ, truth_all,
                    tag, title, save_path, nlls_vals=None):
    """COPIED from plot_prior_vs_posterior().

    [copy changed] three things, all requested:
      * LINEAR axes for every panel (the original bins fp / ps / tau_i in
        log10). A uniform rates prior is flat on a linear axis, so this is also
        the axis on which "prior-dominated" looks like what it is.
      * rates converted to min^-1 (x60) for display only; everything stored is
        still s^-1.
      * a TRUTH line, which the original cannot have (real data, no truth).
    Derived quantities get panels too (PLOT_DERIVED), with the x-range clipped
    to the posterior+truth so an unbounded prior (Tc, Te) cannot flatten the
    panel; the clipped prior range is annotated instead of silently cropped.
    """
    names_fit = list(layout.param_names[:layout.n_model])
    lb = np.asarray(layout.lb, float)[:layout.n_model]
    ub = np.asarray(layout.ub, float)[:layout.n_model]
    panels = list(names_fit) + (list(DERIVED) if PLOT_DERIVED else [])

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, p in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        pk = disp(p, post[p])
        pri = disp(p, prior_d[p])
        sampled = p in names_fit

        zoomed = False
        if sampled:
            j = names_fit.index(p)
            lo, hi = float(disp(p, lb[j])), float(disp(p, ub[j]))
            q = np.percentile(pk, [0.5, 99.5])
            narrow = (q[1] - q[0]) < AUTO_ZOOM_FRAC * (hi - lo)
            if XLIM_SAMPLED == "posterior" or (XLIM_SAMPLED == "auto" and narrow):
                tv0 = truth_all.get(p, np.nan)
                cand = [q[0], q[1]] + ([float(disp(p, tv0))]
                                       if np.isfinite(tv0) else [])
                z0, z1 = float(np.min(cand)), float(np.max(cand))
                pad = 0.35 * (z1 - z0 if z1 > z0 else abs(z1) or 1.0)
                lo, hi = max(lo, z0 - pad), min(hi, z1 + pad)
                zoomed = True
        else:
            # derived: linear axis over the posterior + truth, widened 10%
            q = np.percentile(pk, [0.5, 99.5])
            tv = truth_all.get(p, np.nan)
            cand = [q[0], q[1]] + ([disp(p, tv)] if np.isfinite(tv) else [])
            lo, hi = float(np.min(cand)), float(np.max(cand))
            pad = 0.10 * (hi - lo if hi > lo else abs(hi) or 1.0)
            lo, hi = lo - pad, hi + pad

        bins = np.linspace(lo, hi, 60)
        ax.hist(pri, bins=bins, density=True, color=C_PRIOR, alpha=0.9,
                label="prior", zorder=1)
        ax.hist(pk, bins=bins, density=True, color=C_POST, alpha=0.75,
                label="posterior", zorder=2)
        med = float(np.median(pk))
        ax.axvline(med, color=C_MEDIAN, lw=1.8, zorder=3,
                   label=f"post. median\n{med:.4g}")
        tv = truth_all.get(p, np.nan)
        if np.isfinite(tv):
            tvd = float(disp(p, tv))
            inside = bool(summ[f"{p}_ci2.5"] <= tv <= summ[f"{p}_ci97.5"])
            ax.axvline(tvd, color=C_TRUTH, lw=2.2, zorder=5,
                       label=f"TRUTH {tvd:.4g}\n{'in' if inside else 'OUTSIDE'} 95% CrI")
        if nlls_vals is not None and p in nlls_vals:
            ax.axvline(float(disp(p, nlls_vals[p])), color=C_NLLS, lw=1.5,
                       ls="--", zorder=4, label=f"NLLS {disp(p, nlls_vals[p]):.4g}")
        ax.set_xlim(lo, hi)
        plo, phi = float(np.min(pri)), float(np.max(pri))
        if plo < lo or phi > hi:
            ax.text(0.98, 0.70,
                    ("ZOOMED\n" if zoomed else "") + f"prior spans\n"
                    f"[{plo:.3g}, {phi:.3g}]",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    color=("#B22234" if zoomed else "0.35"))
        ttl = (f"{p}   R-hat={summ[f'{p}_rhat']:.3f}   "
               f"ESS={summ[f'{p}_ess']:.0f}   "
               f"IQR post/prior={summ[f'{p}_iqr_ratio']:.2f}"
               f"{'*' if summ[f'{p}_prior_dominated'] else ''}")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(disp_label(p))
        ax.set_ylabel("density")
        ax.legend(fontsize=6.0, loc="best")
        ax.grid(alpha=0.2)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{tag}   {title}\n"
                 f"linear axes; rates in min$^{{-1}}$; "
                 f"* = prior-dominated (IQR ratio > {IQR_RATIO_FLAT}, "
                 f"computed in log10 for {sorted(LOG_SCALE)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fit_curve(t, St, clean, pred, tag, title, save_path):
    """Data vs the TRUE noiseless curve vs the posterior-median prediction."""
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, St, "o", ms=4, color="k", label="synthetic data")
    ax[0].plot(t, clean, "-", color=C_TRUTH, lw=2, label="truth (noiseless)")
    if pred is not None:
        ax[0].plot(t, pred, "--", color=C_POST, lw=2,
                   label=f"posterior-median prediction (RMSE {rms(St - pred):.4g})")
    ax[0].axvspan(t[NBASE.start], t[NBASE.stop - 1], color="0.6", alpha=0.25,
                  label=f"baseline frames {NBASE.start}-{NBASE.stop - 1}")
    ax[0].set_ylabel("Signal (a.u.)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="0.4", lw=1)
    if pred is not None:
        ax[1].plot(t, St - pred, "o-", ms=3, color=C_POST, label="data - prediction")
    ax[1].plot(t, St - clean, "s--", ms=3, color=C_TRUTH, alpha=0.7,
               label="data - truth (= noise)")
    ax[1].legend(fontsize=7)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("residual")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{tag}   {title}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# one fit
# =============================================================================

def seed_for(tag):
    """A stable per-fit seed. Derived from the tag rather than a running
    counter so that resuming a run, or running one test on its own, gives the
    same chains as the full run did."""
    import zlib
    return int(SEED + zlib.crc32(tag.encode()) % 1_000_000)


def run_one_fit(St, clean, aif, t, truth, tau_fixed, wx, delay_mode, tag,
                title, outdir, rowdir, extra=None, seed=None):
    """One Bayesian fit of one synthetic curve + its row and figures.

    This is fit_lesions_bayes_wx_fixedtau.fit_one with the data-loading half
    removed (the curve is simulated, not aggregated over lesion voxels) and the
    truth comparison added. The MODEL half -- BayesConfig, fit_curve_bayes,
    build_param_layout, assemble_prior, the posterior-median prediction -- is
    the same code path, called the same way.

    delay_mode : "fixed"  tau_fixed is held (the model under test)
                 "joint"  tau_delay is sampled (T1 stage 1)
    wx         : True  -> SSM3 with tau_i sampled
                 False -> plain TCXM
    """
    row_path = os.path.join(rowdir, f"{tag}.csv")
    if os.path.exists(row_path):
        print(f"  {tag}: row exists, skipping")
        # A resumed row comes back through the CSV, where a flag can arrive as
        # the STRING "False" -- and bool("False") is True, the trap
        # fit_lesions_bayes_wx_fixedtau.py's main() guards against when it
        # reassembles the cohort table. Coerce the flags back here instead.
        r = pd.read_csv(row_path).iloc[0].to_dict()
        return {k: (str(v).strip().lower() == "true"
                    if isinstance(v, str)
                    and str(v).strip().lower() in ("true", "false") else v)
                for k, v in r.items()}

    seed = seed_for(tag) if seed is None else int(seed)
    wx_model = WX_MODEL if wx else None
    subject = SubjectConfig(subject_id=tag, T10=T10_TISSUE, B1=B1)

    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS,
                        n_tune=N_TUNE, n_chains=N_CHAINS,
                        noise="gauss_marginalized_sigma", seed=seed,
                        backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS,
                        delay_mode=delay_mode,
                        delay_fixed=float(tau_fixed if tau_fixed is not None else 0.0),
                        delay_bounds=DELAY_BOUNDS)

    t0 = time.time()
    res = fit_curve_bayes(St, aif, t, TCXM, acq, subject, cfg_fit, cfg_b,
                          wx_model=wx_model,
                          aif_cfg=(aif_cfg if wx else None),
                          relax_cfg=(relax_cfg if wx else None))
    fit_s = time.time() - t0

    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq,
                                wx_model=wx_model,
                                relax_cfg=(relax_cfg if wx else None))
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, (relax_cfg if wx else None),
                           wx_model=wx_model, delay_mode=delay_mode,
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]
    names_fit = list(layout.param_names[:layout.n_model])
    lb_fit = np.asarray(layout.lb, float)[:layout.n_model]
    ub_fit = np.asarray(layout.ub, float)[:layout.n_model]
    prior_d = {p: prior_samples[:, names_fit.index(p)] for p in names_fit}
    prior_d.update(derived_draws(prior_d))

    # the sampled vector must be what this test thinks it is
    if tuple(res.param_names[:layout.n_model]) != tuple(names_fit):
        raise ValueError(f"{tag}: sampler returned {res.param_names}, layout "
                         f"expected {tuple(names_fit)}")

    summ, post = summarise_fit(res, prior_d, names_fit, DERIVED)

    # --- truth comparison -----------------------------------------------------
    truth_all = truth_with_derived(truth)
    if delay_mode == "joint":
        truth_all["tau_delay"] = float(truth.get("tau_delay", np.nan))
    cov = {}
    for p in names_fit + DERIVED:
        tv = truth_all.get(p, np.nan)
        cov[f"{p}_truth"] = tv
        if not np.isfinite(tv):
            continue
        med, sd = summ[f"{p}_median"], summ[f"{p}_sd"]
        cov[f"{p}_bias"] = med - tv
        cov[f"{p}_rel_bias"] = (med - tv) / tv if tv != 0 else np.nan
        cov[f"{p}_z"] = (med - tv) / sd if sd > 0 else np.nan
        x = post[p]
        for lvl, (a, b) in ((50, (25.0, 75.0)), (90, (5.0, 95.0)),
                            (95, (2.5, 97.5))):
            q = np.percentile(x, [a, b])
            cov[f"{p}_in_ci{lvl}"] = bool(q[0] <= tv <= q[1])
        # SBC rank of the truth among thinned draws (see docstring T4b)
        ch = res.chains[:, :, list(res.param_names).index(p)] if p in names_fit else None
        if ch is not None:
            flatd = ch.reshape(-1)
            idx = np.linspace(0, flatd.size - 1, min(SBC_THIN_TO, flatd.size))
            thin = flatd[np.round(idx).astype(int)]
            cov[f"{p}_sbc_rank"] = int(np.sum(thin < tv))
            cov[f"{p}_sbc_n"] = int(thin.size)

    # --- posterior-median prediction -----------------------------------------
    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(compute_S0(St[None, :], acq)[0])
    # delay_mode="fixed": the delay lives in the AIF, not in theta, so simulate
    # from the SHIFTED AIF -- the same reasoning as the real script.
    # delay_mode="joint": layout.model is the delay-augmented spec, which
    # closes over the unshifted AIF and applies theta's tau_delay itself.
    aif_sim = (shift_aif(aif, float(tau_fixed), t, acq)
               if delay_mode == "fixed" else aif)
    pred = simulate_curve(theta_med, aif_sim, t, layout.model, acq, subject,
                          cfg_fit, wx_model=wx_model,
                          aif_cfg=(aif_cfg if wx else None),
                          relax_cfg=(relax_cfg if wx else None), S0=S0)
    resid = St - pred

    # --- figures --------------------------------------------------------------
    nlls_vals = None
    if res.point_estimate is not None:
        pe = np.asarray(res.point_estimate, float)[:len(names_fit)]
        nlls_vals = {p: float(pe[j]) for j, p in enumerate(names_fit)}
    plot_prior_post(res, layout, prior_samples, post, prior_d, summ, truth_all,
                    tag, title, os.path.join(outdir, f"{tag}_prior_post.png"),
                    nlls_vals=nlls_vals)
    if PLOT_FITS:
        plot_fit_curve(t, St, clean, pred, tag, title,
                       os.path.join(outdir, f"{tag}_fit.png"))

    # --- QC: the same gate fit_lesions_bayes_wx_fixedtau.py applies -----------
    rhat_max = max(summ[f"{p}_rhat"] for p in names_fit)
    ess_min = min(summ[f"{p}_ess"] for p in names_fit)
    mcse_rel = [summ[f"{p}_mcse_median_rel_ci"] for p in names_fit]
    mcse_rel_max = float(np.nanmax(mcse_rel)) if len(mcse_rel) else np.nan
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    any_disagree = any(summ[f"{p}_chains_disagree"] for p in names_fit)
    mcse_ok = not (np.isfinite(mcse_rel_max) and mcse_rel_max > MCSE_REL_MAX)
    qc_pass = bool(rhat_max <= RHAT_MAX and ess_min >= ESS_MIN and n_div == 0
                   and not any_disagree and mcse_ok)

    row = dict(tag=tag, sampled=",".join(names_fit), n_sampled=len(names_fit),
               wx=bool(wx), delay_mode=delay_mode,
               tau_delay_fixed=(float(tau_fixed) if tau_fixed is not None else np.nan),
               noise_sd=float(np.std(St - clean, ddof=1)),
               peak_enhancement=float(St.max() / np.mean(St[NBASE])),
               S0=S0, rmse=float(np.sqrt(np.mean(resid ** 2))),
               rmse_truth=float(np.sqrt(np.mean((St - clean) ** 2))),
               rhat_max=rhat_max, ess_min=ess_min,
               mcse_median_rel_ci_max=mcse_rel_max,
               n_divergences=n_div, any_chains_disagree=any_disagree,
               qc_pass=qc_pass,
               at_bound=any(at_bound(summ[f"{p}_median"], lb_fit[j], ub_fit[j],
                                     p in LOG_SCALE)
                            for j, p in enumerate(names_fit)),
               vi_small_flag=bool(wx and summ["vi_median"] < VI_SMALL),
               seed=seed, fit_seconds=fit_s, error="")
    row.update(extra or {})
    row.update(cov)
    row.update(summ)
    for p in ("fp", "ps", "Ktrans"):
        row[f"{p}_median_per_min"] = summ[f"{p}_median"] * 60.0
        if np.isfinite(truth_all.get(p, np.nan)):
            row[f"{p}_truth_per_min"] = float(truth_all[p]) * 60.0

    pd.DataFrame([row]).to_csv(row_path, index=False)
    bits = []
    for p in names_fit:
        tv = truth_all.get(p, np.nan)
        s = f"{p}={float(disp(p, summ[f'{p}_median'])):.4g}"
        if np.isfinite(tv):
            s += f"(T {float(disp(p, tv)):.4g})"
        bits.append(s)
    print(f"  {tag}: " + "  ".join(bits)
          + f"   R-hat {rhat_max:.3f}  ESS {ess_min:.0f}  div {n_div}"
            f"  QC {'PASS' if qc_pass else 'FAIL'}  ({fit_s:.0f}s)")
    return row


# =============================================================================
# T3 -- prior predictive check (no fitting)
# =============================================================================

def test3_prior(t, aif, outdir, sd):
    print(f'\n{"=" * 80}\nT3  prior predictive check (no fitting)\n{"=" * 80}')
    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, delay_mode="fixed",
                        delay_fixed=0.0, delay_bounds=DELAY_BOUNDS,
                        noise="gauss_marginalized_sigma")
    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq,
                                wx_model=WX_MODEL, relax_cfg=relax_cfg)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, relax_cfg, wx_model=WX_MODEL,
                           delay_mode="fixed", n_noise=noise.n_params,
                           noise_names=noise.param_names, choices=PRIOR_CHOICES,
                           delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    names = list(layout.param_names[:layout.n_model])
    lb = np.asarray(layout.lb, float)[:layout.n_model]
    ub = np.asarray(layout.ub, float)[:layout.n_model]
    draws = prior.sample(np.random.default_rng(PRIOR_SEED),
                         T3_N_DRAWS)[:, :layout.n_model]

    subject = SubjectConfig(subject_id="PRIOR", T10=T10_TISSUE, B1=B1)
    curves, keep = [], []
    for th in draws:
        try:
            c = simulate_curve(np.asarray(th, float), aif, t, TCXM, acq, subject,
                               cfg_fit, wx_model=WX_MODEL, aif_cfg=aif_cfg,
                               relax_cfg=relax_cfg, S0=S0_TRUE)
            ok = np.all(np.isfinite(c))
        except Exception:
            c, ok = np.full(t.size, np.nan), False
        curves.append(c)
        keep.append(ok)
    curves = np.asarray(curves, float)
    keep = np.asarray(keep, bool)

    base = np.nanmean(curves[:, NBASE], axis=1)
    peak = np.nanmax(curves, axis=1)
    enh = peak / base
    d = {n: draws[:, j] for j, n in enumerate(names)}
    vi = 1.0 - d["ve"] - d["vp"]

    rows = []
    for j, n in enumerate(names):
        x = draws[:, j]
        onb = np.array([at_bound(v, lb[j], ub[j], n in LOG_SCALE) for v in x])
        rows.append(dict(quantity=n, unit=DISP_UNITS.get(n, ""),
                         median=float(np.median(disp(n, x))),
                         q2_5=float(np.percentile(disp(n, x), 2.5)),
                         q97_5=float(np.percentile(disp(n, x), 97.5)),
                         frac_within_tol_of_bound=float(onb.mean())))
    for n, x in (("vi", vi), ("peak_enhancement", enh),
                 ("E", d["ps"] / (d["ps"] + d["fp"])),
                 ("Ktrans", disp("Ktrans", d["ps"] * d["fp"] / (d["ps"] + d["fp"])))):
        rows.append(dict(quantity=n, unit=DISP_UNITS.get(n, ""),
                         median=float(np.nanmedian(x)),
                         q2_5=float(np.nanpercentile(x, 2.5)),
                         q97_5=float(np.nanpercentile(x, 97.5)),
                         frac_within_tol_of_bound=np.nan))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(os.path.dirname(outdir),
                                "prior_predictive_summary.csv"), index=False)

    frac_vi_small = float(np.mean(vi < VI_SMALL))
    frac_flat = float(np.mean(enh[keep] < 1.05))
    frac_huge = float(np.mean(enh[keep] > 10.0))
    print(f"  {int(keep.sum())}/{len(keep)} prior draws simulated finite curves")
    print(f"  peak enhancement: median {np.nanmedian(enh[keep]):.2f}x, "
          f"2.5-97.5% {np.nanpercentile(enh[keep], 2.5):.2f}-"
          f"{np.nanpercentile(enh[keep], 97.5):.2f}x; "
          f"{100 * frac_flat:.1f}% below 1.05x, {100 * frac_huge:.1f}% above 10x")
    print(f"  vi = 1-ve-vp: median {np.median(vi):.3f}; "
          f"{100 * frac_vi_small:.1f}% of draws below VI_SMALL={VI_SMALL} "
          f"(tau_i has almost no pool to describe there)")
    print(f"  tau_i prior: median {np.median(d['tau_i']):.3f} s, "
          f"2.5-97.5% {np.percentile(d['tau_i'], 2.5):.3f}-"
          f"{np.percentile(d['tau_i'], 97.5):.3f} s (log-uniform over "
          f"[{relax_cfg.tau_i_lb}, {relax_cfg.tau_i_ub}])")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- figure: curves + every prior marginal, linear axes, rates in 1/min ---
    panels = ["curves"] + names + ["vi", "peak_enhancement"]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for k, p in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        if p == "curves":
            sel = rng.choice(np.flatnonzero(keep),
                             size=min(T3_N_CURVES, int(keep.sum())), replace=False)
            for i in sel:
                ax.plot(t, curves[i], "-", color=C_PRIOR, lw=0.8, alpha=0.7)
            lo = np.nanpercentile(curves[keep], 2.5, axis=0)
            hi = np.nanpercentile(curves[keep], 97.5, axis=0)
            ax.fill_between(t, lo, hi, color=C_POST, alpha=0.18,
                            label="2.5-97.5% of prior draws")
            ax.plot(t, np.nanmedian(curves[keep], axis=0), "-", color=C_POST,
                    lw=2, label="prior median curve")
            ax.axvspan(t[NBASE.start], t[NBASE.stop - 1], color="0.6", alpha=0.25,
                       label="baseline")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_title(f"prior predictive curves (S0={S0_TRUE:g}, "
                         f"noise sd={sd:.3g})", fontsize=9)
            ax.legend(fontsize=7)
        else:
            x = vi if p == "vi" else (enh[keep] if p == "peak_enhancement"
                                      else disp(p, d[p]))
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            if p == "peak_enhancement":
                x = np.clip(x, 0, np.percentile(x, 99))
            ax.hist(x, bins=60, density=True, color=C_PRIOR, alpha=0.95)
            ax.axvline(float(np.median(x)), color=C_MEDIAN, lw=1.8,
                       label=f"median {np.median(x):.4g}")
            if p in TRUTH_BASE:
                ax.axvline(float(disp(p, TRUTH_BASE[p])), color=C_TRUTH, lw=2.0,
                           label=f"TRUTH_BASE {disp(p, TRUTH_BASE[p]):.4g}")
            if p == "vi":
                ax.axvline(VI_SMALL, color="k", ls=":", lw=1.5,
                           label=f"VI_SMALL {VI_SMALL}")
            ax.set_xlabel(disp_label(p) if p not in ("vi", "peak_enhancement") else p)
            ax.set_ylabel("density")
            ax.set_title(p, fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"T3 prior predictive: {PRIOR_CHOICES}, {T3_N_DRAWS} draws\n"
                 f"linear axes; rates in min$^{{-1}}$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(os.path.dirname(outdir), "prior_predictive.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


# =============================================================================
# T4 -- recovery, SBC, contraction
# =============================================================================

def test4_recovery(aifs, outdir, rowdir, sd):
    print(f'\n{"=" * 80}\nT4  recovery / calibration / tau_i identifiability\n{"=" * 80}')
    rng = np.random.default_rng(4040)
    rows = []

    # (a) recovery on an LHS grid, delay fixed at the TRUE value
    truths = lhs(RECOVERY_BOX, N_RECOVERY, rng)
    for i, tr in enumerate(truths):
        label, t, aif = aifs[i % len(aifs)]
        tau_true = float(TRUTH_BASE.get("tau_delay", 6.0))
        St, clean = make_dataset(tr, aif, t, tau_true, True, sd,
                                 NOISE_SEED + 1000 + i)
        rows.append(run_one_fit(
            St, clean, aif, t, tr, tau_true, True, "fixed",
            f"T4a_rec{i:02d}", f"T4a recovery, AIF {label}, true delay fixed",
            outdir, rowdir,
            extra=dict(test="T4a_recovery", aif_label=label, case=i,
                       tau_delay_true=tau_true, arm="true_delay")))

    # (b) SBC: truths drawn from the prior (see docstring for the sigma caveat)
    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, delay_mode="fixed",
                        delay_fixed=0.0, delay_bounds=DELAY_BOUNDS,
                        noise="gauss_marginalized_sigma")
    layout = build_param_layout(TCXM, cfg_b, aif=aifs[0][2], time_var=aifs[0][1],
                                acq=acq, wx_model=WX_MODEL, relax_cfg=relax_cfg)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, relax_cfg, wx_model=WX_MODEL,
                           delay_mode="fixed", n_noise=noise.n_params,
                           noise_names=noise.param_names, choices=PRIOR_CHOICES,
                           delay_bounds=DELAY_BOUNDS)
    names = list(layout.param_names[:layout.n_model])
    pdraws = prior.sample(np.random.default_rng(PRIOR_SEED + 7),
                          N_SBC)[:, :layout.n_model]
    for i, th in enumerate(pdraws):
        tr = {n: float(th[j]) for j, n in enumerate(names)}
        label, t, aif = aifs[i % len(aifs)]
        tau_true = float(TRUTH_BASE.get("tau_delay", 6.0))
        St, clean = make_dataset(tr, aif, t, tau_true, True, sd,
                                 NOISE_SEED + 2000 + i)
        rows.append(run_one_fit(
            St, clean, aif, t, tr, tau_true, True, "fixed",
            f"T4b_sbc{i:02d}", f"T4b SBC (truth from prior), AIF {label}",
            outdir, rowdir,
            extra=dict(test="T4b_sbc", aif_label=label, case=i,
                       tau_delay_true=tau_true, arm="true_delay")))

    df = pd.DataFrame(rows)
    params = ["ve", "vp", "fp", "ps", "tau_i"]

    # --- recovery / coverage table -------------------------------------------
    # Coverage is reported on ALL fits and on QC-passing fits separately: a fit
    # whose chains did not converge has an interval the sampler did not
    # actually resolve, so including it measures the sampler, not the model --
    # but dropping it silently would hide how many were dropped.
    out = []
    for test in ("T4a_recovery", "T4b_sbc"):
        for subset, d in (("all", df[df["test"] == test]),
                          ("qc_pass", df[(df["test"] == test)
                                         & df["qc_pass"].astype(bool)])):
            if d.empty:
                continue
            for p in params + DERIVED:
                if f"{p}_bias" not in d.columns:
                    continue
                b = d[f"{p}_bias"].astype(float)
                z = d[f"{p}_z"].astype(float)
                r = dict(test=test, subset=subset, quantity=p,
                         unit=DISP_UNITS.get(p, ""), n=len(d),
                         median_bias=float(np.nanmedian(disp(p, b))),
                         median_rel_bias=float(np.nanmedian(
                             d[f"{p}_rel_bias"].astype(float))),
                         median_abs_z=float(np.nanmedian(np.abs(z))),
                         median_iqr_ratio=float(np.nanmedian(
                             d[f"{p}_iqr_ratio"].astype(float))),
                         frac_prior_dominated=float(np.mean(
                             d[f"{p}_prior_dominated"].astype(bool))))
                for lvl in (50, 90, 95):
                    c = d[f"{p}_in_ci{lvl}"].astype(bool)
                    lo, hi = wilson(int(c.sum()), len(c))
                    r[f"cov{lvl}"] = float(c.mean())
                    r[f"cov{lvl}_lo"] = lo
                    r[f"cov{lvl}_hi"] = hi
                out.append(r)
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(os.path.dirname(outdir),
                                "test4_recovery_summary.csv"), index=False)
    nqc = int(df["qc_pass"].astype(bool).sum())
    print(f"\n  {nqc}/{len(df)} fits pass QC "
          f"(R-hat<={RHAT_MAX}, ESS>={ESS_MIN}, 0 divergences, chains agree, "
          f"MCSE(med)/CrI<={MCSE_REL_MAX})")
    if len(summary):
        print(summary[summary.quantity.isin(params)].to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))

    # --- figure: SBC ranks, coverage, contraction ----------------------------
    qc = df[df["qc_pass"].astype(bool)]
    sbc = qc[qc["test"] == "T4b_sbc"]
    fig, axes = plt.subplots(3, len(params), figsize=(3.6 * len(params), 10.0),
                             squeeze=False)
    for j, p in enumerate(params):
        ax = axes[0][j]
        if len(sbc) and f"{p}_sbc_rank" in sbc:
            rk = sbc[f"{p}_sbc_rank"].astype(float).to_numpy()
            n = float(sbc[f"{p}_sbc_n"].astype(float).iloc[0])
            ax.hist(rk / n, bins=SBC_N_BINS, range=(0, 1), color=C_POST, alpha=0.8)
            ax.axhline(len(rk) / SBC_N_BINS, color="k", ls="--", lw=1.2,
                       label="uniform")
            ax.legend(fontsize=7)
        ax.set_title(f"SBC rank {p}\n(n={len(sbc)} QC-passing)", fontsize=9)
        ax.set_xlabel("normalized rank of truth")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)

        ax = axes[1][j]
        d = df[df["test"] == "T4a_recovery"]
        if len(d):
            x = disp(p, d[f"{p}_truth"].astype(float))
            y = disp(p, d[f"{p}_median"].astype(float))
            lo = disp(p, d[f"{p}_ci2.5"].astype(float))
            hi = disp(p, d[f"{p}_ci97.5"].astype(float))
            bad = ~d["qc_pass"].astype(bool).to_numpy()
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="0.6",
                        capsize=3, lw=1)
            ax.scatter(x[~bad], y[~bad], s=34, color=C_POST, zorder=3)
            ax.scatter(x[bad], y[bad], s=34, facecolors="none",
                       edgecolors=C_POST, zorder=3)
            lim = [min(x.min(), lo.min()), max(x.max(), hi.max())]
            ax.plot(lim, lim, "k--", lw=0.9)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_title(f"recovery {p}", fontsize=9)
        ax.set_xlabel(f"truth, {disp_label(p)}")
        ax.set_ylabel("posterior median (95% CrI)")
        ax.grid(alpha=0.2)

        ax = axes[2][j]
        for test, c in (("T4a_recovery", C_POST), ("T4b_sbc", C_TRUTH)):
            d = df[df["test"] == test]
            if len(d):
                xx = disp(p, d[f"{p}_truth"].astype(float))
                yy = d[f"{p}_iqr_ratio"].astype(float).to_numpy()
                bad = ~d["qc_pass"].astype(bool).to_numpy()
                ax.scatter(xx[~bad], yy[~bad], s=22, color=c, alpha=0.8,
                           label=test)
                ax.scatter(xx[bad], yy[bad], s=22, facecolors="none",
                           edgecolors=c, alpha=0.8)
        ax.axhline(IQR_RATIO_FLAT, color="k", ls=":", lw=1.5,
                   label=f"prior-dominated > {IQR_RATIO_FLAT}")
        ax.set_ylim(0, 1.3)
        ax.set_title(f"contraction {p}", fontsize=9)
        ax.set_xlabel(f"truth, {disp_label(p)}")
        ax.set_ylabel("posterior IQR / prior IQR")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.2)
    fig.suptitle("T4 recovery (delay fixed at the TRUE value), SBC ranks, and "
                 "prior contraction\nlinear axes; rates in min$^{-1}$; "
                 "hollow = failed QC. SBC ranks use QC-PASSING fits only: a "
                 "rank from chains that did not converge says nothing about "
                 "calibration", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(os.path.dirname(outdir), "test4_recovery.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return df, summary


# =============================================================================
# T1 -- delay hand-off
# =============================================================================

def stage1_delay(St, clean, aif, t, truth, tau_true, tag, title, outdir, rowdir,
                 extra):
    """Estimate tau_delay the way the SOURCE run did: TCXM + joint delay, NO
    water exchange. Returns (tau_hat, row or None)."""
    if T1_STAGE1 == "nlls":
        subject = SubjectConfig(subject_id=tag, T10=T10_TISSUE, B1=B1)
        t0 = time.time()
        r = fit_with_delay_continuous(St, aif, t, TCXM, acq, subject, cfg_fit,
                                      tau_lb=DELAY_BOUNDS[0], tau_ub=DELAY_BOUNDS[1],
                                      tau_p0=0.0, seed=SEED, interp_method="linear")
        row = dict(extra)
        row.update(tag=tag, arm="stage1_nlls", tau_delay_true=float(tau_true),
                   tau_delay_hat=float(r.best_tau),
                   tau_delay_hat_err=float(r.best_tau) - float(tau_true),
                   nlls_cost=nlls_cost(r), nlls_success=bool(r.success),
                   qc_pass=False,   # an NLLS point estimate has no MCMC QC
                   fit_seconds=time.time() - t0)
        pd.DataFrame([row]).to_csv(os.path.join(rowdir, f"{tag}.csv"), index=False)
        print(f"  {tag}: NLLS tau_hat={r.best_tau:.3f} s "
              f"(true {tau_true:.3f}, err {float(r.best_tau) - tau_true:+.3f})")
        return float(r.best_tau), row
    # "bayes": the fit the source run's tau_delay_median actually came from.
    # The data have water exchange, the model does not -- that mismatch is the
    # whole point of this stage.
    row = run_one_fit(St, clean, aif, t,
                      dict(truth, tau_delay=tau_true), None, False, "joint",
                      tag, title, outdir, rowdir,
                      extra=dict(extra, arm="stage1_bayes",
                                 tau_delay_true=tau_true))
    tau_hat = float(row["tau_delay_median"])
    # the same two columns the NLLS branch writes, so the T1 summary and figure
    # read one column whichever stage-1 method ran. The row CSV is rewritten so
    # a resumed run has them too.
    row["tau_delay_hat"] = tau_hat
    row["tau_delay_hat_err"] = tau_hat - float(tau_true)
    row["tau_delay_src_ci_width"] = float(row["tau_delay_ci97.5"]
                                          - row["tau_delay_ci2.5"])
    pd.DataFrame([row]).to_csv(os.path.join(rowdir, f"{tag}.csv"), index=False)
    print(f"  {tag}: Bayes tau_hat={tau_hat:.3f} s (true {tau_true:.3f}, "
          f"err {tau_hat - tau_true:+.3f}, "
          f"95% CrI {row['tau_delay_ci2.5']:.2f}-{row['tau_delay_ci97.5']:.2f}"
          f" -> this is the width the fixed-tau fit throws away)")
    return tau_hat, row


def stage1_sweep(aifs, sd):
    """NLLS-only preview of T1 stage 1: delay error vs true tau_i, over several
    noise realizations. No sampling, so it costs seconds, and it answers the
    question T1 is really asking -- does a no-water-exchange fit put the
    exchange into the delay? A delay error that MOVES with tau_i is the
    dangerous case, because it transfers a tau_i difference between two groups
    into a delay difference, and from there into the fixed-tau WX fit."""
    label, t, aif = aifs[0]
    subject = SubjectConfig(subject_id="T1SWEEP", T10=T10_TISSUE, B1=B1)
    tau_true = float(T1_TAU_TRUE_S[0])
    print(f"  stage-1 NLLS sweep (no MCMC): true delay {tau_true:g} s, AIF "
          f"{label}, {len(T1_SWEEP_SEEDS)} noise realizations per row")
    print(f"    {'tau_i true':>10} {'median tau_hat':>15} {'median err':>11} "
          f"{'min':>8} {'max':>8}")
    rows = []
    for tau_i in T1_TAU_I_TRUE:
        tr = dict(TRUTH_BASE, tau_i=float(tau_i))
        hats = []
        for s in T1_SWEEP_SEEDS:
            St, _ = make_dataset(tr, aif, t, tau_true, True, sd, int(s))
            r = fit_with_delay_continuous(St, aif, t, TCXM, acq, subject,
                                          cfg_fit, tau_lb=DELAY_BOUNDS[0],
                                          tau_ub=DELAY_BOUNDS[1], tau_p0=0.0,
                                          seed=SEED, interp_method="linear")
            hats.append(float(r.best_tau))
        e = np.asarray(hats) - tau_true
        print(f"    {tau_i:10.3f} {np.median(hats):15.3f} "
              f"{np.median(e):+11.3f} {e.min():+8.3f} {e.max():+8.3f}")
        rows.append(dict(tau_i_true=float(tau_i), tau_delay_true=tau_true,
                         median_tau_hat=float(np.median(hats)),
                         median_err=float(np.median(e)), min_err=float(e.min()),
                         max_err=float(e.max()), n_seeds=len(T1_SWEEP_SEEDS),
                         aif_label=label))
    d = pd.DataFrame(rows)
    span = float(d["median_err"].max() - d["median_err"].min())
    print(f"    delay error spans {span:.3f} s across tau_i in "
          f"[{min(T1_TAU_I_TRUE):g}, {max(T1_TAU_I_TRUE):g}] s. A span much "
          f"larger than 0 means the delay carries tau_i information, so fixing "
          f"it imports a tau_i-dependent error into the WX fit; T2 says what "
          f"that much delay error does to each parameter.")
    return d


def test1_handoff(aifs, outdir, rowdir, sd):
    print(f'\n{"=" * 80}\nT1  delay hand-off: no-WX joint-delay fit -> fixed-tau WX fit'
          f'\n{"=" * 80}')
    if T1_STAGE1_SWEEP:
        sw = stage1_sweep(aifs, sd)
        sw.to_csv(os.path.join(os.path.dirname(outdir),
                               "test1_stage1_sweep.csv"), index=False)
    rows, k = [], 0
    for tau_true in T1_TAU_TRUE_S:
        for tau_i in T1_TAU_I_TRUE:
            for rep in range(T1_N_NOISE):
                label, t, aif = aifs[k % len(aifs)]
                tr = dict(TRUTH_BASE, tau_i=float(tau_i))
                St, clean = make_dataset(tr, aif, t, tau_true, True, sd,
                                         NOISE_SEED + 3000 + k)
                base = f"T1_tau{tau_true:g}_taui{tau_i:g}_r{rep}"
                ex = dict(test="T1_handoff", aif_label=label, case=k,
                          tau_i_true=float(tau_i), tau_delay_true=float(tau_true),
                          rep=rep)
                # stage 1: no WX, joint delay -- the source run's model
                tau_hat, r1 = stage1_delay(
                    St, clean, aif, t, tr, tau_true, base + "_s1",
                    f"T1 stage 1: TCXM + joint delay, NO water exchange "
                    f"(tau_i true {tau_i:g} s)", outdir, rowdir, ex)
                if r1 is not None:
                    rows.append(r1)
                # stage 2a: WX, delay fixed at the ESTIMATED value
                rows.append(run_one_fit(
                    St, clean, aif, t, tr, tau_hat, True, "fixed",
                    base + "_s2est",
                    f"T1 stage 2: WX, delay FIXED at the ESTIMATED "
                    f"{tau_hat:.2f} s (true {tau_true:g} s)",
                    outdir, rowdir,
                    extra=dict(ex, arm="estimated_delay", tau_delay_hat=tau_hat,
                               tau_delay_hat_err=tau_hat - tau_true)))
                # stage 2b: the same data with the TRUE delay fixed
                rows.append(run_one_fit(
                    St, clean, aif, t, tr, tau_true, True, "fixed",
                    base + "_s2true",
                    f"T1 stage 2 reference: WX, delay FIXED at the TRUE "
                    f"{tau_true:g} s", outdir, rowdir,
                    extra=dict(ex, arm="true_delay", tau_delay_hat=tau_true,
                               tau_delay_hat_err=0.0)))
                k += 1

    df = pd.DataFrame(rows)
    params = ["ve", "vp", "fp", "ps", "tau_i"]
    s2 = df[df["arm"].isin(["estimated_delay", "true_delay"])]
    out = []
    for arm in ("true_delay", "estimated_delay"):
        d = s2[s2["arm"] == arm]
        for p in params:
            if d.empty or f"{p}_bias" not in d.columns:
                continue
            b = d[f"{p}_bias"].astype(float)
            r = dict(arm=arm, quantity=p, unit=DISP_UNITS.get(p, ""), n=len(d),
                     median_bias=float(np.nanmedian(disp(p, b))),
                     median_rel_bias=float(np.nanmedian(
                         d[f"{p}_rel_bias"].astype(float))),
                     median_abs_z=float(np.nanmedian(np.abs(
                         d[f"{p}_z"].astype(float)))))
            c = d[f"{p}_in_ci95"].astype(bool)
            lo, hi = wilson(int(c.sum()), len(c))
            r.update(cov95=float(c.mean()), cov95_lo=lo, cov95_hi=hi)
            out.append(r)
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(os.path.dirname(outdir),
                                "test1_handoff_summary.csv"), index=False)

    s1 = df[df["arm"].isin(["stage1_bayes", "stage1_nlls"])]
    if len(s1):
        print("\n  stage-1 delay error vs true tau_i "
              "(is the delay error tau_i-dependent?):")
        g = (s1.groupby("tau_i_true")["tau_delay_hat_err"]
             .agg(["median", "min", "max", "count"]))
        print(g.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n  stage-2 bias and 95% coverage, estimated vs true delay:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(2, max(3, len(params)),
                             figsize=(3.6 * max(3, len(params)), 7.2), squeeze=False)
    ax = axes[0][0]
    if len(s1):
        for tau_true in T1_TAU_TRUE_S:
            d = s1[s1["tau_delay_true"] == tau_true]
            if len(d):
                ax.plot(d["tau_i_true"].astype(float),
                        d["tau_delay_hat_err"].astype(float), "o-",
                        label=f"true delay {tau_true:g} s")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("true tau_i (s)")
    ax.set_ylabel("stage-1 delay error (s)")
    ax.set_title("stage 1 (no WX, joint delay):\ndelay error vs true tau_i",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    for j, p in enumerate(params):
        ax = axes[1][j]
        for arm, c, m in (("true_delay", C_TRUTH, "o"),
                          ("estimated_delay", C_POST, "s")):
            d = s2[s2["arm"] == arm]
            if len(d):
                ax.plot(d["tau_i_true"].astype(float),
                        disp(p, d[f"{p}_bias"].astype(float)), m, ms=6,
                        color=c, alpha=0.85, label=arm)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("true tau_i (s)")
        ax.set_ylabel(f"bias, {disp_label(p)}")
        ax.set_title(f"{p} bias", fontsize=9)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.25)
    ax = axes[0][1]
    if len(s2):
        for arm, c, m in (("true_delay", C_TRUTH, "o"),
                          ("estimated_delay", C_POST, "s")):
            d = s2[s2["arm"] == arm]
            if len(d):
                ax.plot(d["tau_i_true"].astype(float),
                        d["tau_i_median"].astype(float), m, ms=7, color=c,
                        alpha=0.85, label=arm)
        lim = [0, max(T1_TAU_I_TRUE) * 1.3]
        ax.plot(lim, lim, "k--", lw=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("true tau_i (s)")
    ax.set_ylabel("tau_i posterior median (s)")
    ax.set_title("tau_i recovery: does the hand-off\nmove it?", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    for j in range(2, axes.shape[1]):
        axes[0][j].axis("off")
    fig.suptitle("T1 delay hand-off: a delay estimated by a NO-water-exchange "
                 "fit, then fixed in the WX fit\nlinear axes; rates in min$^{-1}$",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(os.path.dirname(outdir), "test1_handoff.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return df, summary


# =============================================================================
# T2 -- delay error tolerance
# =============================================================================

def test2_tolerance(aifs, outdir, rowdir, sd):
    print(f'\n{"=" * 80}\nT2  delay error tolerance (same data, same noise, '
          f'delay offset)\n{"=" * 80}')
    label, t, aif = aifs[0]
    tr = dict(TRUTH_BASE)
    # ONE dataset, ONE noise realization: every difference below is the delay.
    St, clean = make_dataset(tr, aif, t, T2_TAU_TRUE_S, True, sd,
                             NOISE_SEED + 4000)
    rows = []
    for off in T2_OFFSETS_S:
        tau_fix = float(T2_TAU_TRUE_S + off)
        if not (DELAY_BOUNDS[0] <= tau_fix <= DELAY_BOUNDS[1]):
            print(f"  offset {off:+g} s -> tau {tau_fix:g} s is outside "
                  f"DELAY_BOUNDS {DELAY_BOUNDS}, skipped")
            continue
        rows.append(run_one_fit(
            St, clean, aif, t, tr, tau_fix, True, "fixed",
            f"T2_off{off:+g}".replace("+", "p").replace("-", "m").replace(".", "_"),
            f"T2 delay fixed at {tau_fix:g} s = true {T2_TAU_TRUE_S:g} "
            f"{off:+g} s", outdir, rowdir,
            extra=dict(test="T2_tolerance", aif_label=label,
                       tau_delay_true=float(T2_TAU_TRUE_S), delay_offset_s=float(off))))
    df = pd.DataFrame(rows)
    params = ["ve", "vp", "fp", "ps", "tau_i"]
    cols = ["delay_offset_s", "rmse", "rmse_truth", "qc_pass"] + \
           [f"{p}_bias" for p in params] + [f"{p}_in_ci95" for p in params]
    summary = df[[c for c in cols if c in df.columns]].copy()
    for p in params:
        if f"{p}_bias" in summary:
            summary[f"{p}_bias"] = disp(p, summary[f"{p}_bias"].astype(float))
    summary.to_csv(os.path.join(os.path.dirname(outdir),
                                "test2_tolerance_summary.csv"), index=False)
    print("\n  bias per parameter vs delay offset "
          "(fp/ps in min^-1, tau_i in s):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), squeeze=False)
    flat = axes.ravel()
    for j, p in enumerate(params):
        ax = flat[j]
        x = df["delay_offset_s"].astype(float)
        ax.plot(x, disp(p, df[f"{p}_bias"].astype(float)), "o-", color=C_POST,
                ms=6)
        lo = disp(p, (df[f"{p}_ci2.5"] - df[f"{p}_truth"]).astype(float))
        hi = disp(p, (df[f"{p}_ci97.5"] - df[f"{p}_truth"]).astype(float))
        ax.fill_between(x, lo, hi, color=C_POST, alpha=0.15,
                        label="95% CrI, centred on truth")
        ax.axhline(0, color=C_TRUTH, lw=1.8, label="truth")
        ax.axvline(0, color="k", ls=":", lw=1.2, label="correct delay")
        ax.set_xlabel("delay fixed - true delay (s)")
        ax.set_ylabel(f"bias, {disp_label(p)}")
        ax.set_title(p, fontsize=10)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.25)
    ax = flat[len(params)]
    ax.plot(df["delay_offset_s"].astype(float), df["rmse"].astype(float), "o-",
            color=C_POST, label="fit RMSE")
    ax.axhline(float(df["rmse_truth"].astype(float).median()), color=C_TRUTH,
               lw=1.8, label="noise floor (RMSE of the truth)")
    ax.set_xlabel("delay fixed - true delay (s)")
    ax.set_ylabel("RMSE (signal a.u.)")
    ax.set_title("fit quality", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.suptitle(f"T2 delay error tolerance: one dataset (true delay "
                 f"{T2_TAU_TRUE_S:g} s, tau_i {TRUTH_BASE['tau_i']:g} s), one "
                 f"noise realization, delay fixed wrong on purpose\n"
                 f"linear axes; rates in min$^{{-1}}$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(os.path.dirname(outdir), "test2_tolerance.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return df, summary


# =============================================================================
# T5 -- misspecification both directions
# =============================================================================

def fastlimit_diagnostic(aifs, sd):
    """Do the WX model and the no-WX model agree when water exchange is fast?

    They should, if the WX model nests the no-WX one -- and the whole logic of
    "does adding water exchange improve the fit" assumes they do. They do NOT
    in this package, and this function measures by how much and says where it
    comes from, at the TRUTH_BASE operating point:

      bound_gap  SSM3(T5A_TAU_I_FAST) vs SSM3(tau_i -> 0): how far the null
                 T5a actually generates still is from true fast exchange.
                 Should be small against the noise; if it is not, lower
                 relax_cfg.tau_i_lb (and with it T5A_TAU_I_FAST).
      nest_gap   SSM3(tau_i -> 0) vs the no-WX forward model: the part the
                 bound CANNOT fix.
      hct_gap    the same comparison after applying (1 - hematocrit) to the
                 no-WX vascular term. water_exchange._ssm3_to_signal uses
                 Cv_density = (Cv/vb)*(1-hct) with vb := vp, while
                 signal_models.concentration_to_signal uses Ct = Cv + Ce with
                 no such factor. If hct_gap is much smaller than nest_gap, that
                 factor is the whole story, and it is a question about what vp
                 means -- PLASMA volume (then Cv/vb is already the plasma
                 concentration and the factor double-counts the hematocrit) or
                 BLOOD volume (then the factor is right and the no-WX model's
                 vascular term is the one that is off). Resolve it against the
                 MATLAB reference before comparing TCXM with TCXM+SSM3 by
                 LOO/WAIC: an elpd difference between two models that disagree
                 at the fast-exchange limit is not measuring water exchange
                 alone.

    Returns a dict of the three gaps, in signal units and in noise sd.
    """
    label, t, aif = aifs[0]
    tr = dict(TRUTH_BASE)
    aif_sh = shift_aif(np.asarray(aif, float), 6.0, t, acq)
    subject = SubjectConfig(subject_id="FASTLIM", T10=T10_TISSUE, B1=B1)

    def ssm3(tau_i):
        rc = replace(relax_cfg, tau_i_lb=min(relax_cfg.tau_i_lb, tau_i))
        return simulate_curve(theta_for(dict(tr, tau_i=tau_i), True), aif_sh, t,
                              TCXM, acq, subject, cfg_fit, wx_model=WX_MODEL,
                              aif_cfg=aif_cfg, relax_cfg=rc, S0=S0_TRUE)

    c_fxl = simulate_curve(theta_for(tr, False), aif_sh, t, TCXM, acq, subject,
                           cfg_fit, S0=S0_TRUE)
    c_lb, c_0 = ssm3(float(T5A_TAU_I_FAST)), ssm3(1e-6)
    Cv, Ce = TCXM.fn_compartments(theta_for(tr, False), t, aif_sh)
    from dce_mri.signal_models import concentration_to_signal
    c_fxl_hct = concentration_to_signal(
        Cv * (1.0 - aif_cfg.hematocrit) + Ce, acq, subject.T10, B1, S0_TRUE)

    g = dict(bound_gap=float(np.max(np.abs(c_lb - c_0))),
             nest_gap=float(np.max(np.abs(c_0 - c_fxl))),
             hct_gap=float(np.max(np.abs(c_0 - c_fxl_hct))))
    print(f"  fast-exchange limit, at TRUTH_BASE (noise sd {sd:.4g}):")
    for lbl, key in ((f"SSM3(tau_i={T5A_TAU_I_FAST:g}, the T5a null) vs "
                      f"SSM3(tau_i->0)", "bound_gap"),
                     ("SSM3(tau_i->0) vs the no-WX model", "nest_gap"),
                     ("... with (1-hct) on the no-WX Cv term", "hct_gap")):
        print(f"    {lbl:<50s} {g[key]:9.3f}  = {g[key] / sd:7.3f} x noise sd")
    if g["nest_gap"] > sd:
        print(f"  !! the two models do NOT agree at fast exchange, so they are "
              f"NOT nested.")
        if g["hct_gap"] < 0.3 * g["nest_gap"]:
            print(f"     The (1-hematocrit) factor on the vascular term "
                  f"accounts for "
                  f"{100 * (1 - g['hct_gap'] / g['nest_gap']):.0f}% of it "
                  f"(_ssm3_to_signal uses Cv_density=(Cv/vb)*(1-hct) with "
                  f"vb:=vp; concentration_to_signal uses Ct=Cv+Ce). Which is "
                  f"right depends on whether vp is plasma or blood volume.")
        print(f"     The 'no_wx' arm of T5a therefore measures this gap as "
              f"well as any false-positive exchange; the 'ssm3_fast' arm is "
              f"the clean false-positive test.")
    return g


def test5_misspec(aifs, outdir, rowdir, sd):
    print(f'\n{"=" * 80}\nT5  misspecification: false-positive exchange, and '
          f'exchange fitted away\n{"=" * 80}')
    g = fastlimit_diagnostic(aifs, sd)
    rng = np.random.default_rng(5050)
    truths = lhs(RECOVERY_BOX, N_MISSPEC, rng)
    tau_true = 6.0
    rows = []

    # (a) data with NO exchange, fitted WITH the WX model -- two different
    #     nulls, see T5A_NULLS
    for null in T5A_NULLS:
        for i, tr in enumerate(truths):
            label, t, aif = aifs[i % len(aifs)]
            tr0 = {k: v for k, v in tr.items() if k != "tau_i"}
            if null == "ssm3_fast":
                gen = dict(tr0, tau_i=float(T5A_TAU_I_FAST))
                St, clean = make_dataset(gen, aif, t, tau_true, True, sd,
                                         NOISE_SEED + 5000 + i)
                what = (f"SSM3 at tau_i={T5A_TAU_I_FAST:g} s (fast exchange "
                        f"inside the fitted model)")
            else:
                gen = dict(tr0, tau_i=float(T5A_TAU_I_FAST))
                St, clean = make_dataset(tr0, aif, t, tau_true, False, sd,
                                         NOISE_SEED + 5500 + i)
                what = "the plain TCXM forward model (NOT nested, see above)"
            # there is no true tau_i in either null: the truth is "as fast as
            # the model allows", recorded as the fast edge so the figure can
            # show where the posterior ought to pile up.
            rows.append(run_one_fit(
                St, clean, aif, t, gen, tau_true, True, "fixed",
                f"T5a_{null}{i:02d}",
                f"T5a null = {what}, fitted WITH SSM3, AIF {label}",
                outdir, rowdir,
                extra=dict(test=f"T5a_null_{null}", aif_label=label, case=i,
                           null=null, tau_delay_true=tau_true,
                           arm="wx_fit_no_exchange_data",
                           fastlimit_bound_gap=g["bound_gap"],
                           fastlimit_nest_gap=g["nest_gap"],
                           fastlimit_hct_gap=g["hct_gap"],
                           fastlimit_nest_gap_over_noise=g["nest_gap"] / sd)))

    # (b) data WITH exchange, fitted WITHOUT it
    for i, tau_i in enumerate(T5_TAU_I_TRUE):
        for j, tr0 in enumerate(truths[:max(1, N_MISSPEC // len(T5_TAU_I_TRUE))]):
            tr = dict(tr0, tau_i=float(tau_i))
            label, t, aif = aifs[(i + j) % len(aifs)]
            St, clean = make_dataset(tr, aif, t, tau_true, True, sd,
                                     NOISE_SEED + 6000 + 10 * i + j)
            rows.append(run_one_fit(
                St, clean, aif, t, tr, tau_true, False, "fixed",
                f"T5b_taui{tau_i:g}_{j:02d}".replace(".", "_"),
                f"T5b data HAS exchange (tau_i {tau_i:g} s), fitted WITHOUT "
                f"it, AIF {label}", outdir, rowdir,
                extra=dict(test="T5b_wx_ignored", aif_label=label, case=10 * i + j,
                           tau_i_true=float(tau_i), tau_delay_true=tau_true,
                           arm="nowx_fit_wx_data")))

    df = pd.DataFrame(rows)
    b = df[df["test"] == "T5b_wx_ignored"]
    out = []
    for null in T5A_NULLS:
        a = df[df["test"] == f"T5a_null_{null}"]
        if not len(a):
            continue
        edge = (a["tau_i_median"].astype(float)
                < FAST_EDGE_FACTOR * relax_cfg.tau_i_lb)
        out.append(dict(test=f"T5a_null_{null}", quantity="tau_i", n=len(a),
                        median=float(a["tau_i_median"].astype(float).median()),
                        frac_at_fast_edge=float(edge.mean()),
                        frac_prior_dominated=float(
                            a["tau_i_prior_dominated"].astype(bool).mean()),
                        median_iqr_ratio=float(
                            a["tau_i_iqr_ratio"].astype(float).median())))
        print(f"\n  T5a [{null}]: no-exchange data fitted with WX -- tau_i "
              f"median {a['tau_i_median'].astype(float).median():.4g} s; "
              f"{100 * edge.mean():.0f}% within {FAST_EDGE_FACTOR:g}x the "
              f"fast-exchange bound "
              f"({relax_cfg.tau_i_lb} s); "
              f"{100 * a['tau_i_prior_dominated'].astype(bool).mean():.0f}% "
              f"prior-dominated")
        if null == "ssm3_fast":
            print(f"       a confident FINITE tau_i here IS a false positive: "
                  f"the data have no exchange and the model could have said so")
        else:
            print(f"       read with the nesting gap above in mind: the fitted "
                  f"model cannot reproduce these curves at ANY tau_i")
    for p in ("ve", "vp", "fp", "ps"):
        if len(b) and f"{p}_bias" in b.columns:
            bb = b[f"{p}_bias"].astype(float)
            c = b[f"{p}_in_ci95"].astype(bool)
            lo, hi = wilson(int(c.sum()), len(c))
            out.append(dict(test="T5b_wx_ignored", quantity=p, n=len(b),
                            unit=DISP_UNITS.get(p, ""),
                            median_bias=float(np.nanmedian(disp(p, bb))),
                            median_rel_bias=float(np.nanmedian(
                                b[f"{p}_rel_bias"].astype(float))),
                            cov95=float(c.mean()), cov95_lo=lo, cov95_hi=hi))
    summary = pd.DataFrame(out)
    summary.to_csv(os.path.join(os.path.dirname(outdir),
                                "test5_misspec_summary.csv"), index=False)
    if len(b):
        print("\n  T5b: WX data fitted WITHOUT exchange -- where the exchange "
              "effect goes:")
        if g["nest_gap"] > sd:
            print(f"       (the two models also differ by "
                  f"{g['nest_gap'] / sd:.1f}x the noise at fast exchange, so "
                  f"part of this bias is that gap, not the exchange itself)")
        # drop the T5a-only columns, which are all NaN on these rows
        print(summary[summary.test == "T5b_wx_ignored"]
              .dropna(axis=1, how="all")
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, axes = plt.subplots(1, 5, figsize=(22.0, 4.2), squeeze=False)
    nulls = {n: df[df["test"] == f"T5a_null_{n}"] for n in T5A_NULLS}
    cols = {"ssm3_fast": C_POST, "no_wx": "#7D3C98"}
    ax = axes[0][0]
    for n, d in nulls.items():
        if len(d):
            ax.hist(d["tau_i_median"].astype(float), bins=20,
                    range=(relax_cfg.tau_i_lb, relax_cfg.tau_i_ub),
                    color=cols.get(n, C_POST), alpha=0.6, label=f"null = {n}")
    ax.axvline(relax_cfg.tau_i_lb, color=C_TRUTH, lw=2.2,
               label=f"fast-exchange edge ({relax_cfg.tau_i_lb} s)")
    ax.set_xlabel("tau_i (s)")
    ax.set_ylabel("count")
    ax.set_title("T5a: tau_i from data with NO exchange\n"
                 "(should sit at the fast edge)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax = axes[0][1]
    for n, d in nulls.items():
        if len(d):
            ax.scatter(d["tau_i_iqr_ratio"].astype(float),
                       d["tau_i_median"].astype(float), s=30,
                       color=cols.get(n, C_POST), label=f"null = {n}")
    ax.axvline(IQR_RATIO_FLAT, color="k", ls=":", lw=1.5,
               label=f"prior-dominated > {IQR_RATIO_FLAT}")
    ax.axhline(relax_cfg.tau_i_lb, color=C_TRUTH, lw=2.0, label="fast edge")
    ax.set_xlabel("tau_i posterior IQR / prior IQR")
    ax.set_ylabel("tau_i median (s)")
    ax.set_title("T5a: is a finite tau_i learned,\nor just the prior?", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    for j, p in enumerate(("ve", "vp", "fp")):
        ax = axes[0][2 + j]
        if len(b):
            ax.scatter(b["tau_i_true"].astype(float),
                       disp(p, b[f"{p}_bias"].astype(float)), s=34,
                       color=C_POST)
            ax.axhline(0, color=C_TRUTH, lw=1.8, label="unbiased")
        ax.set_xlabel("true tau_i of the data (s)")
        ax.set_ylabel(f"bias, {disp_label(p)}")
        ax.set_title(f"T5b: {p} when exchange is ignored", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("T5 misspecification: (a) WX model on no-exchange data, "
                 "(b) no-exchange model on WX data\nlinear axes; rates in "
                 "min$^{-1}$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(os.path.dirname(outdir), "test5_misspec.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return df, summary


# =============================================================================
# main
# =============================================================================

def main():
    import jax
    outdir = RESUME_DIR or os.path.join(
        OUT_ROOT, "run_synthval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rowdir = os.path.join(outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)

    print(f"results -> {outdir}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"(GPU {GPU_ID}); jax devices: {jax.devices()}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        print("  !! no GPU visible to jax -- this will run on CPU and be slow. "
              "Check that GPU_ID exists (nvidia-smi).")
    print(f"model under test: TCXM + fixed delay + {WX_MODEL.name} (tau_i), "
          f"sigma marginalized, sampled = [ve, vp, fp, ps, tau_i]")
    print(f"sampler: {BACKEND} NUTS, {N_CHAINS} chains x {N_DRAWS} draws "
          f"({N_TUNE} tune), dense mass = {DENSE_MASS}")
    print(f"priors: {PRIOR_CHOICES}, tau_i in "
          f"[{relax_cfg.tau_i_lb}, {relax_cfg.tau_i_ub}] s, "
          f"tau_b fixed {relax_cfg.tau_b_fixed} s")
    print(f"TCXM bounds at run time: lb={TCXM.lb}, ub={TCXM.ub}")

    aifs = load_aifs()
    sd, sd_how = noise_sd()
    print(f"AIF: {AIF_SOURCE} -- " + ", ".join(
        f"{lab} ({len(a)} frames, dt={np.median(np.diff(tt)):.3f} s, "
        f"peak {a.max():.2f} mM)" for lab, tt, a in aifs))
    print(f"noise sd: {sd:.4g} signal units on S0={S0_TRUE:g} ({sd_how})")

    # The pad shift_aif writes into the pre-bolus frames. With the baseline
    # window at frames 0-6 and an AIF that arrives later this is 0; if it is
    # not, the delay tests are also testing a contaminated AIF, which is the
    # defect the real script's aif_pad_mM column watches for.
    for lab, tt, a in aifs:
        pad = float(np.mean(np.asarray(a)[NBASE]))
        flag = "  <-- NOT ZERO: shift_aif will pad the pre-bolus frames with this"
        print(f"  mean(aif[{NBASE.start}:{NBASE.stop}]) for {lab}: "
              f"{pad:.5f} mM{flag if abs(pad) > 1e-6 else ''}")

    n_fits = 0
    if 4 in TESTS:
        n_fits += N_RECOVERY + N_SBC
    if 1 in TESTS:
        cells = len(T1_TAU_TRUE_S) * len(T1_TAU_I_TRUE) * T1_N_NOISE
        n_fits += cells * (3 if T1_STAGE1 == "bayes" else 2)
    if 2 in TESTS:
        n_fits += len(T2_OFFSETS_S)
    if 5 in TESTS:
        n_fits += (N_MISSPEC * len(T5A_NULLS)
                   + len(T5_TAU_I_TRUE) * max(1, N_MISSPEC // len(T5_TAU_I_TRUE)))
    print(f"\ntests {TESTS}: about {n_fits} Bayesian fits "
          f"(T3 needs none). Time one fit first if that matters.")

    all_rows, summaries = [], {}
    for which in TESTS:
        sub = {3: "test3_prior", 4: "test4_recovery", 1: "test1_handoff",
               2: "test2_tolerance", 5: "test5_misspec"}[which]
        sdir = os.path.join(outdir, sub)
        os.makedirs(sdir, exist_ok=True)
        t0 = time.time()
        if which == 3:
            summaries["T3"] = test3_prior(aifs[0][1], aifs[0][2], sdir, sd)
        elif which == 4:
            d, s = test4_recovery(aifs, sdir, rowdir, sd)
            all_rows.append(d)
            summaries["T4"] = s
        elif which == 1:
            d, s = test1_handoff(aifs, sdir, rowdir, sd)
            all_rows.append(d)
            summaries["T1"] = s
        elif which == 2:
            d, s = test2_tolerance(aifs, sdir, rowdir, sd)
            all_rows.append(d)
            summaries["T2"] = s
        elif which == 5:
            d, s = test5_misspec(aifs, sdir, rowdir, sd)
            all_rows.append(d)
            summaries["T5"] = s
        print(f"  [T{which} took {(time.time() - t0) / 60:.1f} min]")

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df.to_csv(os.path.join(outdir, "all_fits.csv"), index=False)
        nqc = int(df["qc_pass"].astype(bool).sum()) if "qc_pass" in df else 0
        print(f"\n{len(df)} fits, {nqc} pass QC")

    pd.DataFrame([dict(
        tests=str(TESTS), aif_source=AIF_SOURCE, aif_dir=AIF_DIR,
        n_aifs=len(aifs), noise_sd=sd, noise_sd_how=sd_how,
        noise_kind=NOISE_KIND, S0_TRUE=S0_TRUE, nbase=str(NBASE),
        n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS, backend=BACKEND,
        dense_mass=DENSE_MASS, seed=SEED, prior_choices=str(PRIOR_CHOICES),
        delay_bounds=str(DELAY_BOUNDS), tau_i_lb=relax_cfg.tau_i_lb,
        tau_i_ub=relax_cfg.tau_i_ub, tau_b_fixed=relax_cfg.tau_b_fixed,
        wx_model=WX_MODEL.name, truth_base=str(TRUTH_BASE),
        t1_stage1=T1_STAGE1, t1_tau_i=str(T1_TAU_I_TRUE),
        t1_tau=str(T1_TAU_TRUE_S), t2_offsets=str(T2_OFFSETS_S),
        n_recovery=N_RECOVERY, n_sbc=N_SBC, n_misspec=N_MISSPEC,
        recovery_box=str(RECOVERY_BOX), gpu=GPU_ID)]).to_csv(
        os.path.join(outdir, "run_settings.csv"), index=False)

    print(f"\nresults written to: {outdir}")


if __name__ == "__main__":
    main()