#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_group_stats.py
=====================
Verification for src/dce_mri/group_stats.py, scripts/group_stats_analysis.py
and scripts/subsample_lesions.py.

  S  statistics      AUC == sklearn; DeLong CI coverage; IRLS == sklearn;
                     size adjustment holds its 5% level under a pure size
                     confound (where the unadjusted test does not), keeps
                     power for a real effect; block-test level; Holm / BH
  P  prior sens.     base factors == package prior; new priors normalised;
                     identity reweighting; reweighted posterior == analytic
  X  artifact vs     synthetic cohort with a SHRINKAGE ARTIFACT (ps: no true
     biology         group difference, benign prior-dominated) and a REAL
                     effect (fp: true difference, well measured in both).
                     group_stats_analysis.main() must call ps prior-driven
                     and fp stable, and the subsample section must show ps
                     collapsing toward AUC 0.5 at benign size but not fp
  U  subsampling     exact size, subset, reproducibility; target rules;
                     subset mean unbiased; variance == finite-population
                     formula; subsample_lesions.main() end to end on
                     synthetic NIfTIs (NLLS; Bayes with --bayes)

Run (repo root):
    PYTHONPATH=src python tests/group_stats_tests/verify_group_stats.py
    PYTHONPATH=src python tests/group_stats_tests/verify_group_stats.py --bayes
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from dce_mri.config import AcquisitionConfig
from dce_mri import group_stats as G
from dce_mri import hierarchical as H

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(REPO, "scripts")
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS = (-10.0, 30.0)
ACQ = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
T = np.arange(36) * 150 * 8 / 288
AIF = 5.0 * np.exp(-((T - 50.0) / 15.0) ** 2)


class TestLog:
    def __init__(self, title):
        self.title, self.rows = title, []

    def record(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f"  --  {detail}" if detail else ""))

    def summary(self):
        n_fail = sum(1 for _, p, _ in self.rows if not p)
        print("\n" + "=" * 72)
        print(f"  {self.title}: {len(self.rows) - n_fail} passed, {n_fail} failed")
        for n, p, d in self.rows:
            if not p:
                print(f"    - {n}  ({d})")
        print("=" * 72)
        return n_fail == 0


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# =============================================================================
# S. statistics
# =============================================================================

def test_S(log):
    print("\nS. discrimination statistics")
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)

    xB = np.round(rng.normal(0, 1, 60), 1)
    xM = np.round(rng.normal(0.4, 1, 75), 1)          # rounding -> ties
    a = G.auc_mw(xB, xM)[0]
    b = roc_auc_score(np.r_[np.zeros(60), np.ones(75)], np.r_[xB, xM])
    log.record("S1 AUC (with ties) == sklearn roc_auc_score", abs(a - b) < 1e-12,
               f"{a:.6f} vs {b:.6f}")

    true_auc = __import__("scipy").stats.norm.cdf(0.6 / np.sqrt(2))
    cov = np.mean([lo <= true_auc <= hi for lo, hi, _ in
                   (G.delong_ci(rng.normal(0, 1, 40), rng.normal(0.6, 1, 50))
                    for _ in range(600))])
    log.record("S2 DeLong 95% CI coverage in [0.925, 0.97]", 0.925 <= cov <= 0.97,
               f"{cov:.3f} over 600 simulations")

    X = rng.normal(size=(250, 3))
    y = (rng.uniform(size=250) < 1 / (1 + np.exp(-(0.3 + X @ [1, -0.5, 0.2])))).astype(float)
    f = G.logit_fit(X, y)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-12).fit(X, y)
    d = np.max(np.abs(f.beta - np.r_[sk.intercept_, sk.coef_[0]]))
    log.record("S3 IRLS logistic == sklearn (unpenalised)", d < 1e-5, f"max diff {d:.1e}")

    # size confound: benign smaller; parameter depends ONLY on size
    def sim(effect):
        n = 170
        yy = np.r_[np.zeros(85), np.ones(85)]
        ls = np.where(yy == 0, rng.normal(3.5, 0.6, n), rng.normal(5.0, 0.6, n))
        x = 0.8 * ls + effect * yy + rng.normal(0, 0.6, n)
        return yy, ls, x
    fp_raw, fp_adj, pw = [], [], []
    for _ in range(400):
        yy, ls, x = sim(0.0)
        fp_raw.append(G.auc_mw(x[yy == 0], x[yy == 1])[1] < 0.05)
        fp_adj.append(G.size_adjusted(yy, ls, x)["lr_p"] < 0.05)
        yy, ls, x = sim(0.6)
        pw.append(G.size_adjusted(yy, ls, x)["lr_p"] < 0.05)
    log.record("S4 pure size confound: UNADJUSTED test rejects (> 50%)",
               np.mean(fp_raw) > 0.5, f"{100 * np.mean(fp_raw):.0f}% rejections")
    log.record("S5 pure size confound: size-adjusted test holds 5% (2-9%)",
               0.02 <= np.mean(fp_adj) <= 0.09, f"{100 * np.mean(fp_adj):.1f}% rejections")
    log.record("S6 real effect beyond size: size-adjusted power > 80%",
               np.mean(pw) > 0.8, f"{100 * np.mean(pw):.0f}% rejections")

    rej = []
    for _ in range(300):
        yy, ls, _ = sim(0.0)
        Xb = np.column_stack([0.5 * ls + rng.normal(0, 1, ls.size) for _ in range(3)])
        rej.append(G.block_test(yy, ls, Xb, n_repeats=2)["lr_p"] < 0.05)
    log.record("S7 block test (k=3) holds 5% under the size confound (2-9%)",
               0.02 <= np.mean(rej) <= 0.09, f"{100 * np.mean(rej):.1f}%")

    p = np.array([0.01, 0.04, 0.03, 0.005])
    h, b_ = G.holm(p), G.bh(p)
    log.record("S8 Holm and BH == hand-computed",
               np.allclose(h, [0.03, 0.06, 0.06, 0.02]) and
               np.allclose(b_, [0.02, 0.04, 0.04, 0.02]),
               f"Holm {h}, BH {b_}")


# =============================================================================
# P. prior sensitivity machinery
# =============================================================================

def test_P(log, prior, tr):
    print("\nP. prior-sensitivity reweighting")
    base = G.base_factors(PRIOR_CHOICES, tr)
    sd, mean = G.check_base_factors(prior, base, tr)
    log.record("P1 base factors == package PriorSpec.log_pdf", sd < 1e-10,
               f"SD of difference {sd:.1e}")

    settings, skipped = G.default_settings(prior, tr, ("ve", "vp", "fp", "ps"),
                                           {"fp": (0.5, 2.0), "ps": (0.5, 2.0),
                                            "vp": (0.5, 1.5), "ve": (0.5, 1.5)})
    log.record("P2 every requested setting is inside the box (ve/vp use x1.5, "
               "since x2 would put their median on the bound)",
               len(skipped) == 0, f"{len(settings)} settings, skipped {skipped}")
    # sample from the WIDER of the two densities so the ratio is bounded
    # (sampling a narrow prior and averaging p_base/p_new is heavy-tailed and
    # its SE is underestimated)
    rng = np.random.default_rng(1)
    worst = 0.0
    names = list(tr.names)
    Xb = prior.sample(rng, 400_000)
    for st in settings[1:]:
        for p, smp in st.samplers.items():
            j = names.index(p)
            if st.kind == "width" and st.name.endswith("x0.5"):
                x = Xb[:, j]
                r = np.exp(st.factors[p](x) - base[p](x))   # E_base[new/base]
            else:
                x = smp(rng, 400_000)
                r = np.exp(base[p](x) - st.factors[p](x))   # E_new[base/new]
            worst = max(worst, abs(r.mean() - 1) / (r.std() / np.sqrt(r.size)))
    log.record("P3 every new prior is normalised on the box "
               "(importance identity = 1 within 4 SE)", worst < 4,
               f"worst |dev| {worst:.2f} SE")

    draws = prior.sample(rng, 5000)
    lw = G.lesion_log_weights(draws, names, settings[0], base)
    log.record("P4 base setting: all weights equal", np.ptp(lw) == 0)

    # posterior under base prior, Gaussian likelihood on z, SIR; reweight to
    # a z-Normal prior -> posterior is Gaussian on z with known mean
    j = names.index("ps")
    st = next(s for s in settings if s.name == "ps:loc x2")
    mu, sdp = st.meta["mu"], st.meta["sd"]
    errs = []
    for s_lik in (0.2, 0.6, 1.5, 3.0):
        for y_ in (-1.0, 0.5):
            zp = y_ + s_lik * rng.standard_normal(200_000)
            X = prior.sample(rng, 200_000)
            X[:, j] = tr.from_z_col(zp, j)
            lw0 = base["ps"](X[:, j]) - G._ljac_col(tr, j, X[:, j])  # p0 on z
            w0 = np.exp(lw0 - lw0.max())
            idx = rng.choice(X.shape[0], 8000, p=w0 / w0.sum())
            dr = X[idx]
            lwn = G.lesion_log_weights(dr, names, st, base)
            w = np.exp(lwn - lwn.max())
            w /= w.sum()
            z = G._zcol(tr, j, dr[:, j])[0]
            prec = 1 / s_lik ** 2 + 1 / sdp ** 2
            exact = (y_ / s_lik ** 2 + mu / sdp ** 2) / prec
            errs.append(abs(np.sum(w * z) - exact) * np.sqrt(prec))
    log.record("P5 reweighted posterior mean == analytic (max < 0.1 posterior SD)",
               max(errs) < 0.1, f"max {max(errs):.3f} SD over 8 lesion types")


# =============================================================================
# X. artifact vs biology, end to end through group_stats_analysis.main()
# =============================================================================

def _sir_draws(rng, prior, tr, y_z, s_z, S=3000, n_prop=30_000):
    """Posterior draws under the BASE prior for a Gaussian-on-z likelihood."""
    P = tr.P
    zp = y_z + s_z * rng.standard_normal((n_prop, P))
    X = tr.from_z(zp)
    base = G.base_factors(PRIOR_CHOICES, tr)        # == prior.log_pdf (P1)
    lp = sum(base[p](X[:, j]) - G._ljac_col(tr, j, X[:, j])
             for j, p in enumerate(tr.names))
    w = np.exp(lp - lp.max())
    return X[rng.choice(n_prop, S, p=w / w.sum())]


def build_artifact_cohort(tmp, prior, tr, rng, nB=40, nM=40):
    """
    ps : NO true group difference (z ~ N(1.5, 0.3) in both); benign ROIs
         small -> s_z ~ 3 (prior-dominated), malignant large -> s_z ~ 0.15
    fp : REAL difference (z ~ N(-0.6, 0.3) B vs N(+0.6, 0.3) M), s_z = 0.12
         in both groups (well measured)
    ve, vp, tau_delay : same in both groups, moderate s
    """
    names = list(tr.names)
    rows, draws_all = [], {}
    y = np.r_[np.zeros(nB), np.ones(nM)]
    nvox = np.where(y == 0, rng.integers(15, 120, y.size), rng.integers(80, 600, y.size))
    base = G.base_factors(PRIOR_CHOICES, tr)
    for i in range(y.size):
        sid = f"FMB{700 + i}"
        zt = np.array([rng.normal(0.0, 0.3), rng.normal(-0.8, 0.3),
                       rng.normal(-0.6 if y[i] == 0 else 0.6, 0.3),
                       rng.normal(1.5, 0.3), rng.normal(-0.5, 0.3)])
        s = np.array([0.4, 0.4, 0.12, 3.0 if y[i] == 0 else 0.15, 0.3])
        yz = zt + s * rng.standard_normal(5)
        dr = _sir_draws(rng, prior, tr, yz, s)
        np.savez_compressed(os.path.join(tmp, f"{sid}_draws.npz"),
                            chains=dr.reshape(2, -1, 5),
                            param_names=np.array(names, dtype=object), t=T, aif=AIF,
                            St=np.zeros(T.size), lesion_nvox=int(nvox[i]))
        draws_all[sid] = dr
        q = G.derived_from_draws(dr, names)
        r = dict(subject_id=sid, diagnosis="M" if y[i] else "B", qc_pass=True,
                 error="", n_vox=int(nvox[i]),
                 baseline_cv=float(0.02 * np.sqrt(300 / nvox[i])),
                 ps_flow_limited=bool(np.median(q["E"]) > 0.4))
        for k, v in q.items():
            r[f"{k}_median"] = float(np.median(v))
        rows.append(r)
    df = pd.DataFrame(rows)
    # prior-dominated flags exactly as the fit script defines them
    st0 = G.PriorSetting("base", "base", "none", {}, {})
    piqr = G.prior_iqr(st0, prior, tr, ["ve", "vp", "fp", "ps", "tau_delay",
                                        "vtot", "E", "Ktrans"],
                       log_iqr=("fp", "ps", "Ktrans"))
    for k in piqr:
        lg = k in ("fp", "ps", "Ktrans")
        iq = [(np.log10(np.percentile(G.derived_from_draws(draws_all[s], names)[k], 75))
               - np.log10(np.percentile(G.derived_from_draws(draws_all[s], names)[k], 25)))
              if lg else np.subtract(*np.percentile(
                  G.derived_from_draws(draws_all[s], names)[k], [75, 25]))
              for s in df["subject_id"]]
        df[f"{k}_prior_dominated"] = np.array(iq) / piqr[k] > 0.8
    df.to_csv(os.path.join(tmp, "lesion_bayes_results.csv"), index=False)

    # subsample run: the SAME malignant lesions refitted at benign size ->
    # their ps likelihood widens to the benign s_z; fp stays well measured
    sub = []
    for i in np.flatnonzero(y == 1):
        sid = df["subject_id"].iloc[i]
        dr0 = draws_all[sid]
        zc = np.array([np.median(G._zcol(tr, j, dr0[:, j])[0]) for j in range(5)])
        # same lesion re-observed at benign SNR: ps likelihood as wide as a
        # benign lesion's; fp still well measured
        s = np.array([0.6, 0.6, 0.12, 3.0, 0.4])
        dr = _sir_draws(rng, prior, tr, zc + s * rng.standard_normal(5), s)
        q = G.derived_from_draws(dr, names)
        nv = int(rng.integers(15, 120))
        r = dict(subject_id=sid, diagnosis="M", replicate=0, qc_pass=True, error="",
                 n_vox=nv, baseline_cv=float(0.02 * np.sqrt(300 / nv)))
        for k, v in q.items():
            r[f"{k}_median"] = float(np.median(v))
        for k in piqr:
            r[f"{k}_prior_dominated"] = bool(k == "ps")
        sub.append(r)
    sdir = os.path.join(tmp, "sub")
    os.makedirs(sdir)
    pd.DataFrame(sub).to_csv(os.path.join(sdir, "subsample_results.csv"), index=False)
    return sdir


def test_X(log, prior, tr):
    print("\nX. artifact-vs-biology cohort through group_stats_analysis.main()")
    tmp = tempfile.mkdtemp(prefix="gstats_")
    try:
        rng = np.random.default_rng(11)
        t0 = time.time()
        sdir = build_artifact_cohort(tmp, prior, tr, rng)
        print(f"   synthetic cohort built ({time.time() - t0:.0f} s)")
        run = load_script("group_stats_analysis")
        run.RUN_DIR, run.SUBSAMPLE_RUN_DIR = tmp, sdir
        run.OUT_DIR = os.path.join(tmp, "out")
        run.SUBSET, run.CV_REPEATS, run.N_BOOT = "all", 3, 500
        res = run.main()

        want = ["1_size_check.csv", "2_discrimination_bayes.csv", "3_block_tests.csv",
                "4_identifiability.csv", "5_prior_sens_lesions.csv",
                "5_prior_sens_auc.csv", "5_prior_sens_summary.csv",
                "5_prior_sens_auc.png", "6_subsample_comparison.csv",
                "6_subsample_snr_check.csv", "6_subsample_values.png", "config.json"]
        miss = [f for f in want if not os.path.exists(os.path.join(tmp, "out", f))]
        log.record("X1 all section outputs written", not miss, f"missing {miss}")

        s = res["prior_summary"].set_index("param")
        log.record("X2 artifact (ps): AUC moves with ps's prior location (range > 0.15)",
                   s.loc["ps", "auc_range_own_location"] > 0.15,
                   f"range {s.loc['ps', 'auc_range_own_location']:.3f}")
        log.record("X3 real effect (fp): AUC stable under fp's prior location (< 0.05)",
                   s.loc["fp", "auc_range_own_location"] < 0.05,
                   f"range {s.loc['fp', 'auc_range_own_location']:.3f}")
        tB, tM = s.loc["ps", "transmitted_B_median"], s.loc["ps", "transmitted_M_median"]
        log.record("X4 ps: benign transmit the prior shift, malignant do not "
                   "(B - M > 0.4, p < 0.001)",
                   tB - tM > 0.4 and s.loc["ps", "transmitted_B_vs_M_p"] < 1e-3,
                   f"B {tB:.2f}, M {tM:.2f}, p {s.loc['ps', 'transmitted_B_vs_M_p']:.1e}")
        log.record("X5 fp: neither group transmits the prior shift (< 0.1)",
                   max(s.loc["fp", "transmitted_B_median"],
                       s.loc["fp", "transmitted_M_median"]) < 0.1,
                   f"B {s.loc['fp', 'transmitted_B_median']:.3f}, "
                   f"M {s.loc['fp', 'transmitted_M_median']:.3f}")
        sub = res["subsample"]
        sb = sub[sub["estimate"] == "bayes"].set_index("param")
        log.record("X6 subsample: ps |AUC-0.5| shrinks, CI excludes 0",
                   sb.loc["ps", "change_ci97_5"] < 0,
                   f"{sb.loc['ps', 'auc_B_vs_Mfull']:.3f} -> "
                   f"{sb.loc['ps', 'auc_B_vs_Msub']:.3f}, change CI "
                   f"[{sb.loc['ps', 'change_ci2_5']:.3f}, {sb.loc['ps', 'change_ci97_5']:.3f}]")
        log.record("X7 subsample: fp separation survives (|change| < 0.1)",
                   abs(sb.loc["fp", "change_in_abs_auc_minus_half"]) < 0.1,
                   f"{sb.loc['fp', 'auc_B_vs_Mfull']:.3f} -> "
                   f"{sb.loc['fp', 'auc_B_vs_Msub']:.3f}")
        log.record("X8 SNR check reports matched baseline CV (p > 0.05)",
                   res["snr"]["snr_match_p_B_vs_Msub"] > 0.05,
                   f"p {res['snr']['snr_match_p_B_vs_Msub']:.2g}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# U. voxel subsampling
# =============================================================================

def test_U(log, with_bayes):
    print("\nU. voxel subsampling")
    S = load_script("subsample_lesions")
    rng = np.random.default_rng(5)
    mask = np.zeros((20, 20, 6), bool)
    mask[4:14, 5:15, 2:4] = True                       # 200 voxels
    a, ia = S.subsample_mask(mask, 37, S.lesion_rng("FMB001", 0))
    b, ib = S.subsample_mask(mask, 37, S.lesion_rng("FMB001", 0))
    c, ic = S.subsample_mask(mask, 37, S.lesion_rng("FMB001", 1))
    log.record("U1 subset: exact size, inside the lesion, reproducible, "
               "new per replicate",
               a.sum() == 37 and not (a & ~mask).any() and np.array_equal(ia, ib)
               and not np.array_equal(ia, ic))

    benign = np.array([20, 35, 50, 80, 120])
    ks = [S.choose_target(np.random.default_rng(i), 60, benign, "benign_sample",
                          min_voxels=5)[0] for i in range(200)]
    k_small, note = S.choose_target(rng, 15, benign, "benign_sample", min_voxels=5)
    kq = S.choose_target(rng, 500, benign, "benign_quantile", quantile=0.5, min_voxels=5)[0]
    log.record("U2 target rules: sampled sizes are benign sizes < own size; "
               "already-small lesion kept whole; quantile rule",
               set(ks) == {20, 35, 50} and k_small == 15 and kq == 50,
               f"sampled {sorted(set(ks))}, small -> {k_small} ({note}), median -> {kq}")

    n, Tn = int(mask.sum()), 30
    sig = rng.normal(0, 1, (20, 20, 6, Tn)) * 20 + 1000 \
        + rng.normal(0, 30, (20, 20, 6, 1))            # voxel heterogeneity
    full = sig[mask].mean(0)
    k = 40
    means = np.array([sig[S.subsample_mask(mask, k, np.random.default_rng(r))[0]].mean(0)
                      for r in range(600)])
    se = means.std(0) / np.sqrt(600)
    z = np.abs(means.mean(0) - full) / se
    log.record("U3 subset mean curve unbiased for the full-ROI curve (< 4 SE)",
               z.max() < 4, f"max {z.max():.2f} SE over {Tn} frames")
    S2 = sig[mask].var(0, ddof=1)
    theory = S2 / k * (1 - k / n)
    ratio = means.var(0, ddof=1) / theory
    log.record("U4 subset-mean variance == finite-population formula "
               "(ratio within [0.85, 1.15])", 0.85 < np.median(ratio) < 1.15,
               f"median ratio {np.median(ratio):.3f}")

    _subsample_end_to_end(log, S, with_bayes)


def _subsample_end_to_end(log, S, with_bayes):
    import nibabel as nib
    from dce_mri.kinetic_models import TCXM
    from dce_mri.config import SubjectConfig, FittingConfig
    from dce_mri.fitting import build_delay_augmented_spec
    from dce_mri.synthetic import simulate_curve
    tmp = tempfile.mkdtemp(prefix="subs_")
    try:
        root, aifd, base = (os.path.join(tmp, d) for d in ("canonical", "aif", "base"))
        for d in (root, aifd, base):
            os.makedirs(d)
        X_, Y_, Z_, Tn = 32, 32, 6, 36
        dt = 150 * 8 / 288
        t = np.arange(Tn) * dt
        tt = np.clip(t - 8 * dt, 0, None)
        aif = 6 * (tt / 12) ** 2 * np.exp(-(tt - 12) / 6) * (tt > 0)
        aif = aif / aif.max() * 5 + 1.2 * (1 - np.exp(-tt / 20))
        aug = build_delay_augmented_spec(TCXM, aif, t, ACQ, tau_lb=-10, tau_ub=30)
        cfg = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
        rng = np.random.default_rng(2)
        rows = []
        for sid, dx, r in (("FMB801", "B", 2), ("FMB802", "B", 3),
                           ("FMB803", "M", 6), ("FMB804", "M", 7)):
            d = os.path.join(root, sid)
            os.makedirs(d)
            subj = SubjectConfig(subject_id=sid, T10=1530.0, B1=1.0)
            St = simulate_curve(np.array([0.3, 0.05, 0.01, 0.003, 4.0]), aif, t, aug,
                                ACQ, subj, cfg, S0=1000.0)
            m = np.zeros((X_, Y_, Z_), bool)
            yy, xx = np.meshgrid(np.arange(Y_), np.arange(X_))
            for z in (2, 3):
                m[:, :, z] = (xx - 12) ** 2 + (yy - 12) ** 2 <= r ** 2
            sig = 1000 + rng.normal(0, 15, (X_, Y_, Z_, Tn))
            sig[m] = St[None] + rng.normal(0, 15, (int(m.sum()), Tn))
            nib.save(nib.Nifti1Image(sig.astype(np.float32), np.eye(4)),
                     os.path.join(d, f"{sid}_tv1em05c_ttgv3em04z.nii.gz"))
            nib.save(nib.Nifti1Image(m.astype(np.float32), np.eye(4)),
                     os.path.join(d, f"{sid[-3:]}_Lesion 1.nii"))
            pd.DataFrame(dict(frame=np.arange(Tn), time_s=t, aif_signal_au=0.0,
                              aif_plasma_mM=aif, aif_source="idif")).to_csv(
                os.path.join(aifd, f"{sid}_AIF_plasma.csv"), index=False)
            rows.append(dict(subject_id=sid, diagnosis=dx, n_vox=int(m.sum()),
                             lesion_file=f"{sid[-3:]}_Lesion 1.nii", qc_pass=True,
                             error=""))
        pd.DataFrame(rows).to_csv(os.path.join(base, "lesion_bayes_results.csv"),
                                  index=False)
        S.BASE_RUN_DIR, S.RESULTS_ROOT = base, os.path.join(tmp, "out")
        S.RESUME_DIR = os.path.join(tmp, "out", "run_test")
        S.FIT_OVERRIDES = {"DATA_ROOT": root, "AIF_DIR": aifd}
        S.N_REPLICATES = 2
        S.FITS = ("nlls",)
        out = S.main()
        r = pd.read_csv(os.path.join(out, "subsample_results.csv"))
        ok = (r["error"].fillna("") == "").all()
        log.record("U5 end to end (NLLS): 2 malignant x 2 replicates, no errors",
                   ok and len(r) == 4, f"{len(r)} rows")
        bsz = {int(v) for v in pd.DataFrame(rows).query("diagnosis == 'B'")["n_vox"]}
        log.record("U6 every subset has a benign size and is flagged subsampled",
                   set(r["n_vox"]) <= bsz and r["subsampled"].all(),
                   f"sizes {sorted(set(r['n_vox']))}, benign {sorted(bsz)}")
        crc = r.groupby("subject_id")["voxel_crc"].nunique()
        log.record("U7 replicates use different voxel subsets", (crc == 2).all())
        log.record("U8 NLLS columns written (same names as the base results)",
                   all(f"nlls_{p}" in r for p in ("ve", "vp", "fp", "ps", "tau_delay")))
        if with_bayes:
            S.RESUME_DIR = os.path.join(tmp, "out", "run_bayes")
            S.FITS, S.N_REPLICATES, S.ONLY = ("bayes", "nlls"), 1, ["FMB804"]
            S.FIT_OVERRIDES = {"DATA_ROOT": root, "AIF_DIR": aifd, "N_DRAWS": 300,
                               "N_TUNE": 300, "N_CHAINS": 4}
            out = S.main()
            rb = pd.read_csv(os.path.join(out, "subsample_results.csv"))
            have = all(c in rb for c in ("ve_median", "ps_prior_dominated",
                                         "qc_pass", "nlls_ve", "fp_nlls_in_ci95"))
            log.record("U9 end to end (Bayes + NLLS): row has the base-run columns",
                       (rb["error"].fillna("") == "").all() and have,
                       f"R-hat max {rb['rhat_max'].iloc[0]:.3f}, "
                       f"{rb['fit_seconds'].iloc[0]:.0f} s")
            log.record("U10 draws saved for prior sensitivity",
                       os.path.exists(os.path.join(out, "FMB804_r0_draws.npz")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="SPXU")
    ap.add_argument("--bayes", action="store_true",
                    help="also run one Bayesian subsample fit (slow on CPU)")
    a = ap.parse_args()
    log = TestLog("group statistics verification")
    prior, tr = H.tcxm_joint_delay_prior(PRIOR_CHOICES, DELAY_BOUNDS, ACQ, T, AIF)
    t0 = time.time()
    if "S" in a.only:
        test_S(log)
    if "P" in a.only:
        test_P(log, prior, tr)
    if "X" in a.only:
        test_X(log, prior, tr)
    if "U" in a.only:
        test_U(log, a.bayes)
    print(f"\ntotal {time.time() - t0:.0f} s")
    sys.exit(0 if log.summary() else 1)


if __name__ == "__main__":
    main()
