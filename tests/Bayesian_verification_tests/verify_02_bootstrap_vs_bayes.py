#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_02_bootstrap_vs_bayes.py
===============================
VERIFICATION 2 -- frequentist residual bootstrap vs. Bayesian posterior,
across a sweep of simulated noise levels.

Scope: TCXM, no delay, no water exchange (ROW 1 only). Deliberately the
simplest configuration -- rows 1-3 are the validation vehicle, because a
disagreement here means the METHODS differ, not that tau_i is hard.

What is being compared
----------------------
    residual bootstrap : refit NLLS many times on  fitted_curve + resampled
                         residuals.  Approximates the SAMPLING distribution
                         of theta_hat: "if I repeated the experiment, how
                         much would my estimate move?"

    Bayesian posterior : p(theta | this data).  "Given the data I actually
                         have, which theta values are plausible?"

These are different questions. Under FLAT priors and a well-behaved
likelihood they often give similar spreads -- and that near-agreement is
exactly what this script checks. Where they diverge (skewed or
ridge-shaped surfaces, or near a bound) the bootstrap's implicit
Gaussian/asymptotic assumptions are the more suspect of the two.

Run:
    PYTHONPATH=src python scripts/verify_02_bootstrap_vs_bayes.py
"""
from __future__ import annotations

import os
import time
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # comment out to view interactively in Spyder

warnings.filterwarnings("ignore")

from dce_mri.synthetic import default_test_setup, simulate_curve, add_noise
from dce_mri.kinetic_models import TCXM
from dce_mri.bayes import BayesConfig, fit_curve_bayes
from dce_mri.bayes_validation import residual_bootstrap_curve
from dce_mri.bayes_figures import plot_bayes_vs_bootstrap, plot_width_vs_noise

# ----------------------------------------------------------------- settings
NOISE_LEVELS = [5.0, 20.0, 50.0]      # signal peak is ~2950, S0=1000
N_REP        = 200                    # bootstrap replicates
N_DRAWS      = 2000
N_TUNE       = 2000
N_CHAINS     = 4
SEED         = 1
PARAMS       = ["ve", "vp", "fp", "ps"]

OUTDIR = os.path.join(
    os.path.expanduser("~"),
    "dce_bayes_results",
    "verify02_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"figures -> {OUTDIR}")

    setup = default_test_setup()          # Parker AIF, 0-150 s
    acq, subject, cfg_fit = setup["acq"], setup["subject"], setup["cfg"]
    aif, tv = setup["aif"], setup["time_var"]

    truth_vec = np.asarray(TCXM.p0, float)
    truth = {nm: float(truth_vec[j]) for j, nm in enumerate(TCXM.param_names)}
    print("truth:", {k: round(v, 5) for k, v in truth.items()})

    S_clean = simulate_curve(truth_vec, aif, tv, TCXM, acq, subject, cfg_fit)

    bayes_by_noise, boot_by_noise, rows = {}, {}, []

    for sg in NOISE_LEVELS:
        print(f"\n{'='*78}\nNOISE sigma = {sg:g}   "
              f"(peak SNR ~ {S_clean.max()/sg:.0f})")
        sig = add_noise(S_clean, sg, "gaussian", np.random.default_rng(SEED))

        # ---------- frequentist: residual bootstrap ----------------------
        t0 = time.time()
        boot = residual_bootstrap_curve(
            sig, aif, tv, TCXM, acq, subject, cfg_fit,
            n_rep=N_REP, seed=SEED, verbose=False)
        t_boot = time.time() - t0
        print(f"  residual bootstrap: {N_REP} refits in {t_boot:.0f}s "
              f"({boot['n_failed']} failed)")

        # ---------- Bayesian posterior -----------------------------------
        cfg_b = BayesConfig(n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
                            noise="gauss_free_jeffreys", seed=SEED)
        t0 = time.time()
        res = fit_curve_bayes(sig, aif, tv, TCXM, acq, subject, cfg_fit, cfg_b)
        t_bayes = time.time() - t0
        maxr = max(res.rhat.values()) if res.rhat else np.nan
        print(f"  Bayesian MCMC:      {N_DRAWS}x{N_CHAINS} draws in "
              f"{t_bayes:.0f}s (max R-hat {maxr:.3f})")

        # ---------- side-by-side table -----------------------------------
        print(f"\n  {'param':<8}{'truth':>10}"
              f"{'boot med':>11}{'boot IQR':>11}"
              f"{'bayes med':>11}{'bayes IQR':>11}{'IQR ratio':>11}")
        b_dict, f_dict = {}, {}
        for nm in PARAMS:
            bs = np.asarray(boot[nm], float)
            bs = bs[np.isfinite(bs)]
            by = res.flat()[:, res.param_names.index(nm)]
            b_dict[nm], f_dict[nm] = by, bs

            b_med = float(np.median(by))
            b_iqr = float(np.subtract(*np.percentile(by, [75, 25])))
            f_med = float(np.median(bs)) if bs.size else np.nan
            f_iqr = float(np.subtract(*np.percentile(bs, [75, 25]))) if bs.size else np.nan
            ratio = b_iqr / f_iqr if f_iqr and np.isfinite(f_iqr) else np.nan

            print(f"  {nm:<8}{truth[nm]:10.5f}"
                  f"{f_med:11.5f}{f_iqr:11.5f}"
                  f"{b_med:11.5f}{b_iqr:11.5f}{ratio:11.2f}")
            rows.append(dict(sigma=sg, param=nm,
                             bayes_iqr=b_iqr, boot_iqr=f_iqr,
                             bayes_med=b_med, boot_med=f_med,
                             truth=truth[nm], iqr_ratio=ratio))

        print("    IQR ratio = bayes/boot;  ~1 means the two methods agree "
              "on the width")
        bayes_by_noise[sg], boot_by_noise[sg] = b_dict, f_dict

    # ---------- figures ---------------------------------------------------
    plot_bayes_vs_bootstrap(
        bayes_by_noise, boot_by_noise, truth, PARAMS,
        title="TCXM (no delay, no wx): Bayesian posterior vs. residual "
              "bootstrap\n(all densities normalised to unit area)",
        save_path=os.path.join(OUTDIR, "bayes_vs_bootstrap.png"))

    plot_width_vs_noise(
        rows, PARAMS,
        title="Interval width vs. simulated noise",
        save_path=os.path.join(OUTDIR, "width_vs_noise.png"))

    # ---------- compact summary ------------------------------------------
    print(f"\n{'='*78}\nSUMMARY  (IQR ratio = Bayesian / bootstrap)")
    print(f"  {'sigma':>8}" + "".join(f"{p:>12}" for p in PARAMS))
    for sg in NOISE_LEVELS:
        vals = [r["iqr_ratio"] for p in PARAMS
                for r in rows if r["sigma"] == sg and r["param"] == p]
        print(f"  {sg:8g}" + "".join(f"{v:12.2f}" for v in vals))
    print("\n  ratio ~1   : methods agree (expected under flat priors, good SNR)")
    print("  ratio >1   : Bayesian reports MORE uncertainty than the bootstrap")
    print("  ratio <1   : bootstrap wider -- check for failed/railed refits")
    print(f"\nfigures written to: {OUTDIR}")


if __name__ == "__main__":
    main()
