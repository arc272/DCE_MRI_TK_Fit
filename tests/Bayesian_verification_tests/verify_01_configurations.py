#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_01_configurations.py
===========================
VERIFICATION 1 -- does the machinery run, and does it land near truth?

Runs all eight model configurations, and for each one reports:
  * posterior median / 95% CI / R-hat / ESS vs. the known truth
  * WHERE EACH CHAIN ENDED UP (per-chain medians + between/within spread)
  * posterior/prior IQR ratio  (the identifiability read-out)
  * a posterior-vs-prior figure and a trace plot, saved to disk

Simulated data uses a PARKER population AIF on a 0-150 s grid.

Reading the output
------------------
Two very different failure modes both print "CHECK":

  wide posterior, R-hat ~1.0, chains AGREE
      -> honest non-identifiability. Trustworthy result.

  narrow posterior far from truth, R-hat > 1.1, chains DISAGREE
      -> stuck sampler. NOT an identifiability finding. Raise n_tune
         before interpreting anything.

Run:
    PYTHONPATH=src python scripts/verify_01_configurations.py
    PYTHONPATH=src python scripts/verify_01_configurations.py "ROW 1" "ROW 4"
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # comment out to view interactively in Spyder
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from dce_mri.synthetic import default_test_setup, simulate_curve, add_noise
from dce_mri.kinetic_models import TCXM, GKM
from dce_mri.water_exchange import SSM3
from dce_mri.config import RelaxationConfig
from dce_mri.bayes import (
    BayesConfig, fit_curve_bayes, build_param_layout,
    posterior_vs_prior_width, NOISE_OPTIONS,
)
from dce_mri.bayes_priors import assemble_prior
from dce_mri.bayes_validation import convergence_report
from dce_mri.bayes_figures import (
    plot_posterior_vs_prior, plot_trace, per_chain_summary, plot_aif,
)

# ----------------------------------------------------------------- settings
SIGMA_TRUE     = 20.0
TAU_I_TRUE     = 0.5
TAU_B_TRUE     = 0.005
TAU_DELAY_TRUE = 8.0

N_DRAWS, N_TUNE, N_CHAINS = 3000, 3000, 8

BACKEND = "blackjax"      # "pymc" | "blackjax"
DENSE_MASS = True         # blackjax only; False = diagonal control

OUTDIR = os.path.join(
    os.path.expanduser("~"),
    "dce_bayes_results",
    "verify01_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
)


def build_truth(layout, base_model, wx_model, relax_cfg, delay_mode):
    th = list(np.asarray(base_model.p0, float))
    if delay_mode == "joint":
        th.append(TAU_DELAY_TRUE)
    if wx_model is not None:
        th += ([TAU_B_TRUE, TAU_I_TRUE] if relax_cfg.fit_tau_b else [TAU_I_TRUE])
    return np.array(th[:layout.n_model], float)

def _bimodality_coeff(x):
    """
    Sarle's bimodality coefficient, BC = (skew^2 + 1) / kurtosis.
    Uniform gives 0.555; above that is the conventional "possibly
    bimodal" flag, below is "unimodal".

    CAUTION, verified numerically before use: BC is fooled by SKEW.
    A ratio of uniform-over-loguniform -- exactly Tc's and Te's shape --
    scores 0.70 on the raw scale while being unimodal by construction,
    and a plain lognormal scores 0.47. So this is only trustworthy on
    the LOG scale for a positive ratio, which is how it's used below.
    Treat it as a hint that directs your eye to the histogram, never as
    the answer on its own.
    """
    from scipy.stats import skew, kurtosis
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.nan
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)
    return float((g ** 2 + 1.0) / (k + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))))


def tc_te_diagnostic(res, tag, outdir, truth_map=None):
    """
    Is the vp/ps bimodality a COORDINATE artefact or a real second mode?

    Tc = vp/fp and Te = ve/ps are the two mean transit times the TCXM
    impulse response is actually built from (jax_forward.tcxm_compartments
    computes them on every call, lines ~225-227). The measured curve
    depends on ve,vp,fp,ps only through these ratios plus the vp/ve
    prefactors -- so a CURVED ridge in (vp,fp) shows up as two humps in
    the vp marginal while being one clean hump in Tc.

        Tc/Te unimodal  -> the humps were a coordinate artefact;
                           reparameterising to (Tc,Te,vp,ve) should help.
        Tc/Te bimodal   -> two genuinely separate solutions. NO bijective
                           reparameterisation can merge them; next step is
                           a multi-start NLLS check, not new coordinates.

    Returns the dict it prints, or None if the row has no TCXM block.
    """
    names = list(res.param_names)
    need = ("ve", "vp", "fp", "ps")
    if not all(nm in names for nm in need):
        print(f"  Tc/Te diagnostic: skipped (needs {need}; row has {tuple(names)})")
        return None

    flat = res.flat()
    ve, vp = flat[:, names.index("ve")], flat[:, names.index("vp")]
    fp, ps = flat[:, names.index("fp")], flat[:, names.index("ps")]
    Tc, Te = vp / fp, ve / ps

    tc_true = te_true = None
    if truth_map:
        try:
            tc_true = float(truth_map["vp"]) / float(truth_map["fp"])
            te_true = float(truth_map["ve"]) / float(truth_map["ps"])
        except (KeyError, TypeError, ZeroDivisionError):
            pass

    out = {}
    print(f"\n  Tc/Te reparameterisation check   (BC on LOG scale; "
          f"uniform=0.555, higher => possibly bimodal)")
    print(f"    {'quantity':<10}{'median':>12}{'IQR':>12}{'BC(log)':>10}"
          f"{'truth':>10}   reading")
    for nm, v, tv in (("Tc", Tc, tc_true), ("Te", Te, te_true),
                      ("vp", vp, truth_map.get("vp") if truth_map else None),
                      ("ps", ps, truth_map.get("ps") if truth_map else None)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        bc = _bimodality_coeff(np.log(v))
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        flag = "possibly bimodal" if bc > 0.555 else "unimodal"
        tvs = f"{float(tv):10.4g}" if tv is not None else f"{'-':>10}"
        print(f"    {nm:<10}{med:12.4g}{iqr:12.4g}{bc:10.3f}{tvs}   {flag}")
        out[nm] = dict(median=med, iqr=iqr, bc_log=bc,
                       truth=(float(tv) if tv is not None else None))

    # per-chain medians of Tc/Te -- do the chains agree in THESE coordinates?
    ch = res.chains
    cTc = ch[:, :, names.index("vp")] / ch[:, :, names.index("fp")]
    cTe = ch[:, :, names.index("ve")] / ch[:, :, names.index("ps")]
    for nm, arr in (("Tc", cTc), ("Te", cTe)):
        meds = np.median(arr, axis=1)
        within = float(np.mean(np.percentile(arr, 75, axis=1)
                               - np.percentile(arr, 25, axis=1)))
        spread = float(meds.max() - meds.min())
        ok = "OK" if spread <= 0.5 * within else "DISAGREE"
        print(f"    per-chain {nm}: spread={spread:.4g}  within={within:.4g}  {ok}")
        out[nm]["chain_spread"] = spread
        out[nm]["chain_within"] = within

    # ---- figure: marginals in both coordinate systems + the ridge --------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (v, nm, tv) in zip(ax[0],
                              ((vp, "vp", truth_map.get("vp") if truth_map else None),
                               (ps, "ps", truth_map.get("ps") if truth_map else None),
                               (fp, "fp", truth_map.get("fp") if truth_map else None))):
        a.hist(v, bins=90, color="#B22234", alpha=0.85)
        if tv is not None:
            a.axvline(float(tv), ls="--", color="#1B4F72", lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm}  (sampled coordinate)")

    for a, (v, nm, tv) in zip(ax[1], ((Tc, "Tc = vp/fp  (s)", tc_true),
                                      (Te, "Te = ve/ps  (s)", te_true))):
        vv = v[np.isfinite(v) & (v > 0)]
        a.hist(vv, bins=np.geomspace(vv.min(), vv.max(), 90),
               color="#2E7D32", alpha=0.85)
        a.set_xscale("log")
        if tv is not None:
            a.axvline(float(tv), ls="--", color="#1B4F72", lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm.split()[0]}  (derived -- log axis)")

    a = ax[1][2]
    a.scatter(vp, fp, s=2, alpha=0.12, color="#444444")
    if truth_map and truth_map.get("vp") is not None:
        a.plot(float(truth_map["vp"]), float(truth_map["fp"]), "*",
               ms=16, color="#1B4F72", label="truth")
        a.legend(fontsize=8)
    a.set_yscale("log"); a.set_xlabel("vp"); a.set_ylabel("fp")
    a.set_title("the ridge: curved streak => coordinate artefact,\n"
                "two separate clouds => real second mode")

    fig.suptitle(f"{tag}: are the vp/ps humps a coordinate artefact?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, f"{tag}_tc_te_check.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return out

def run_row(label, tag, base_model, delay_mode, wx_model, relax_cfg, setup,
            seed=1, make_figs=True):
    acq, subject, cfg_fit = setup["acq"], setup["subject"], setup["cfg"]
    aif_cfg = setup["aif_cfg"]
    tau_grid = np.arange(-4.0, 21.0, 4.0)

    cfg_b = BayesConfig(n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
        noise="gauss_marginalized_sigma", delay_mode=delay_mode,
        delay_grid=tau_grid if delay_mode == "marginalized" else None,
        seed=seed, backend=BACKEND, step="nuts",
        dense_mass_matrix=DENSE_MASS,
    )

    layout = build_param_layout(
        base_model, cfg_b, aif=setup["aif"], time_var=setup["time_var"],
        acq=acq, wx_model=wx_model, relax_cfg=relax_cfg)

    truth_vec = build_truth(layout, base_model, wx_model, relax_cfg, delay_mode)

    # data carries a real delay even when the model marginalises it away
    aif_sim = setup["aif"]
    if delay_mode == "marginalized":
        from dce_mri.signal_models import shift_aif
        aif_sim = shift_aif(setup["aif"], TAU_DELAY_TRUE, setup["time_var"], acq)

    S = simulate_curve(truth_vec, aif_sim, setup["time_var"], layout.model,
                       acq, subject, cfg_fit, wx_model, aif_cfg, relax_cfg,
                       S0=setup["S0"])
    sig = add_noise(S, SIGMA_TRUE, "gaussian", np.random.default_rng(seed))

    t0 = time.time()
    res = fit_curve_bayes(sig, setup["aif"], setup["time_var"], base_model,
                          acq, subject, cfg_fit, cfg_b, wx_model=wx_model,
                          aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    dt = time.time() - t0

    # ---- truth dict keyed by name (for tables + figures) -----------------
    truth_map = {nm: (truth_vec[j] if j < len(truth_vec) else None)
                 for j, nm in enumerate(res.param_names)}
    if "sigma" in truth_map:
        truth_map["sigma"] = SIGMA_TRUE

    print(f"\n{'='*78}\n{label}   D={layout.n_total}   {dt:.0f}s")
    print(f"{'param':<11}{'truth':>11}{'post med':>11}{'ci_lo':>11}"
          f"{'ci_hi':>11}{'R-hat':>8}{'ESS':>7}  in95")
    n_in = 0
    for nm in res.param_names:
        d = res.summary(nm)
        tv = truth_map.get(nm, np.nan)
        tv = np.nan if tv is None else float(tv)
        inside = np.isfinite(tv) and d["ci_lo"] <= tv <= d["ci_hi"]
        n_in += int(inside)
        print(f"{nm:<11}{tv:11.5f}{d['median']:11.5f}{d['ci_lo']:11.5f}"
              f"{d['ci_hi']:11.5f}{d['rhat']:8.3f}{d['ess']:7.0f}"
              f"  {'YES' if inside else 'no'}")

    rep = convergence_report(res, verbose=False)
    print(f"  converged={rep['passed']}   truth inside 95% CI: "
          f"{n_in}/{len(res.param_names)}")
    if res.backend == "blackjax":
        bj = res.meta["blackjax"]
        imm = np.shape(bj["inverse_mass_matrix"])
        print(f"  blackjax: step_size mean={bj['step_size_mean']:.3e} "
              f"spread={bj['step_size_spread']:.3e}  imm{imm}  "
              f"dense={bj['dense_mass_matrix']}  div={res.n_divergences}")
   
    # ---- (b) WHERE DID EACH CHAIN END UP --------------------------------
    per_chain_summary(res, truth=truth_map, verbose=True)

    # ---- identifiability read-out ---------------------------------------
    noise = NOISE_OPTIONS[cfg_b.noise]
    prior = assemble_prior(layout.model, relax_cfg, wx_model=wx_model,
                           delay_mode=delay_mode, n_noise=noise.n_params,
                           noise_names=noise.param_names,
                           choices=cfg_b.prior_choices)
    w = posterior_vs_prior_width(res, prior)
    print("  posterior/prior IQR ratio:",
          {k: round(v["ratio"], 3) for k, v in w.items()})
    print("    (ratio near 1 = posterior collapsed onto prior = not identifiable)")
    # ---- (a)+(c) FIGURES -------------------------------------------------
    if make_figs:
        plot_posterior_vs_prior(
            res, prior, truth=truth_map,
            title=f"{label}   (densities normalised to unit area)",
            save_path=os.path.join(OUTDIR, f"{tag}_posterior_vs_prior.png"))
        plot_trace(
            res, title=f"{label} - chain traces",
            save_path=os.path.join(OUTDIR, f"{tag}_traces.png"))
    
    # ---- SAVE THE POSTERIOR ---------------------------------------------
    npz_path = os.path.join(OUTDIR, f"{tag}_posterior.npz")
    np.savez_compressed(
        npz_path,
        chains=res.chains,                                  # (n_chains, n_draws, D)
        param_names=np.array(res.param_names, dtype=object),
        truth=truth_vec,
        rhat=np.array([res.rhat.get(n, np.nan) for n in res.param_names]),
        ess=np.array([res.ess.get(n, np.nan) for n in res.param_names]),
        n_divergences=np.array(res.n_divergences
                               if res.n_divergences is not None else -1),
        backend=np.array(res.backend), elapsed_s=np.array(res.elapsed_s),
        label=np.array(label), seed=np.array(seed),
    )
    print(f"  saved posterior: {npz_path}  "
          f"({res.chains.nbytes/1e6:.1f} MB raw, {res.chains.shape})")
    
    # ---- Tc/Te reparameterisation check ---------------------------------
    tc_te_diagnostic(res, tag, OUTDIR, truth_map=truth_map)
    
    return res, rep, n_in
    


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"figures -> {OUTDIR}")

    setup = default_test_setup(n_frames=36, dt=4.17, t_arrival=32.0)          # Parker AIF, 0-150 s
    tv = setup["time_var"]
    print(f"acquisition: {tv[0]:.0f}-{tv[-1]:.0f} s, {len(tv)} frames, "
          f"dt={tv[1]-tv[0]:.1f} s;  AIF peak {setup['aif'].max():.2f} mM "
          f"at t={tv[setup['aif'].argmax()]:.0f} s")
    plot_aif(tv, setup["aif"], title="Parker population AIF (0-150 s)",
             save_path=os.path.join(OUTDIR, "aif.png"))

    relax_fix  = RelaxationConfig(fit_tau_b=False)
    relax_both = RelaxationConfig(fit_tau_b=True)

    rows = [
         ("ROW 1  TCXM | no delay      | no wx",              "row1", TCXM, "none",         None, None),
        ("ROW 2  TCXM | JOINT delay   | no wx",              "row2", TCXM, "joint",        None, None),
       # ("ROW 3  TCXM | MARGIN delay  | no wx  [Orton2007]", "row3", TCXM, "marginalized", None, None),
         ("ROW 4  TCXM | no delay      | SSM3 tau_i",         "row4", TCXM, "none",         SSM3, relax_fix),
       # ("ROW 5  TCXM | no delay      | SSM3 tau_b+tau_i",   "row5", TCXM, "none",         SSM3, relax_both),
       # ("ROW 6  TCXM | JOINT delay   | SSM3 tau_i",         "row6", TCXM, "joint",        SSM3, relax_fix),
       # ("ROW 7  TCXM | MARGIN delay  | SSM3 tau_i [Orton]", "row7", TCXM, "marginalized", SSM3, relax_fix),
       # ("ROW 8  GKM  | no delay      | no wx  (baseline)",  "row8", GKM,  "none",         None, None),
    ]

    only = sys.argv[1:] or None
    summary = []
    for label, tag, m, dm, wx, rc in rows:
        if only and not any(o in label for o in only):
            continue
        try:
            res, rep, n_in = run_row(label, tag, m, dm, wx, rc, setup)
            summary.append((label, rep["passed"], n_in, len(res.param_names)))
            print(res.chains.std(axis=1))
        except Exception as exc:
            print(f"\n{label}\n  FAILED: {type(exc).__name__}: {exc}")
            summary.append((label, False, 0, 0))
            
    print(f"\n{'='*78}\nSUMMARY")
    for label, ok, n_in, n_tot in summary:
        print(f"  {'PASS' if ok else 'CHECK':<6}{label:<52} {n_in}/{n_tot} in 95% CI")
    print(f"\nfigures written to: {OUTDIR}")
    print("\nNOTE: 'CHECK' = R-hat/ESS thresholds not met at this draw count.")
    print("      Look at the per-chain table: if the chains DISAGREE, the")
    print("      posterior is a sampler artefact, not an identifiability result.")


if __name__ == "__main__":
    main()
