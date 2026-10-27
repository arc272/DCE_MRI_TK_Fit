#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_hierarchical.py
======================
Verification ladder for src/dce_mri/hierarchical.py and
scripts/hierarchical_bvsm.py.

Design: the hierarchical step only ever sees per-tumor POSTERIOR DRAWS. The
synthetic cohorts here give each tumor a likelihood that is Gaussian on the
population (z) scale,  L_i(z) = N(y_i; z, s_i^2),  while the draws are taken
from  L_i(z) p0(z)  under the REAL TCXM prior (Beta(2,2) volume fractions,
log-uniform rates, uniform delay). Then the exact marginal likelihood is

    p(y_i | phi) = N(y_i; m_i(phi), sigma_i(phi)^2 + s_i^2),

so every Monte Carlo quantity the code produces has a closed-form target.

  A  transform        round trip; analytic Jacobian vs finite differences
  B  prior            B0 the generator's vectorised p0 == the package's
                         PriorSpec.log_pdf
                      B1 p0 is a normalised density on the box
                      B2 zero-information tumors (draws = prior): mean
                         importance weight = 1 for any phi  (tests Jacobian,
                         sign of c, population normalisation, together)
  C  PSIS k-hat       recovers known generalized-Pareto tail shapes
  D  MC likelihood    tracks exact log p(y | phi) across phi; its error is
                      Monte Carlo noise (shrinks with more draws), not bias,
                      and k-hat > 0.7 tumors carry most of it
  N  Gaussian lik.    algebra (closed form, flat directions), precision
                      recovery, separability of the real prior, likelihood
                      and hyperparameter posterior vs exact
  E  MC posterior     hyperparameter posterior from the MC likelihood ==
                      posterior from the exact likelihood; truth recovered
                      (slow on CPU: --skip-mc skips E and F)
  F  tumor level      reweighted per-tumor posterior means == analytic
                      conditional means
  G  calibration      (--coverage R) 95% CI coverage of delta over R
                      replicate cohorts, and false-positive rate at delta=0
                      (Gaussian likelihood; --coverage-method mc for MC)
  H  runner I/O       synthetic fit_lesions_bayes run directory ->
                      hierarchical_bvsm.main() with both likelihoods

Run (from the repo root):
    PYTHONPATH=src python tests/hierarchical_verification_tests/verify_hierarchical.py
    PYTHONPATH=src python tests/hierarchical_verification_tests/verify_hierarchical.py --coverage 30
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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dce_mri.config import AcquisitionConfig
from dce_mri import hierarchical as H

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNNER = os.path.join(REPO, "scripts", "hierarchical_bvsm.py")

PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS = (-10.0, 30.0)
ACQ = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
T = np.arange(36) * 150 * 8 / 288
AIF = 5.0 * np.exp(-((T - 50.0) / 15.0) ** 2)

# synthetic truth on the z-scale, order (ve, vp, fp, ps, tau_delay)
TRUTH = dict(mu_B=np.array([-0.3, -1.0, -0.5, -0.8, -0.5]),
             delta=np.array([0.6, 0.8, 0.7, 0.5, 0.0]),
             sigma_B=np.array([0.5, 0.6, 0.5, 0.6, 0.4]),
             sigma_M=np.array([0.7, 0.6, 0.6, 0.8, 0.4]))
GROUP_EFFECT = (True, True, True, True, False)
SEPARATE_SIGMA = (True, True, True, True, False)


# =============================================================================
# test log (same API as tests/wx_verification_tests/wx_verify_common.TestLog)
# =============================================================================

class TestLog:
    def __init__(self, title):
        self.title, self.rows = title, []

    def record(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f"  --  {detail}" if detail else ""))
        return passed

    def summary(self):
        n_fail = sum(1 for _, p, _ in self.rows if not p)
        print("\n" + "=" * 72)
        print(f"  {self.title}: {len(self.rows) - n_fail} passed, {n_fail} failed")
        for name, p, d in self.rows:
            if not p:
                print(f"    - {name}  ({d})")
        print("=" * 72)
        return n_fail == 0


# =============================================================================
# shared fixtures
# =============================================================================

def real_prior():
    return H.tcxm_joint_delay_prior(PRIOR_CHOICES, DELAY_BOUNDS, ACQ, T, AIF)


def vec_logp0(tr):
    """Vectorised copy of the TCXM prior for FAST test-data generation only.
    B0 checks it against the package's PriorSpec.log_pdf; the code under
    test always uses the package's own log_pdf (or this, once B0 passed)."""
    lb, ub = tr.lb, tr.ub
    w = ub - lb

    def f(X):
        X = np.asarray(X, float)
        out = np.zeros(X.shape[:-1])
        inside = np.all((X > lb) & (X < ub), axis=-1)
        for j in (0, 1):                      # Beta(2,2) on [lb, ub]
            u = (X[..., j] - lb[j]) / w[j]
            out += np.log(6.0) + np.log(np.clip(u, 1e-300, None)) \
                + np.log(np.clip(1 - u, 1e-300, None)) - np.log(w[j])
        for j in (2, 3):                      # log-uniform
            out += -np.log(np.clip(X[..., j], 1e-300, None)) \
                - np.log(np.log(ub[j]) - np.log(lb[j]))
        out += -np.log(w[4])                  # uniform delay
        return np.where(inside, out, -np.inf)
    return f


def make_gaussian_cohort(rng, tr, logp0, n_B, n_M, S, truth=TRUTH,
                         s_lo=0.1, s_hi=2.5, prop_factor=10):
    """
    Per tumor: z_true ~ population; s_ij log-uniform in [s_lo, s_hi]
    (heterogeneous, the upper end ~ prior-dominated); y = z_true + s*eps.
    Draws ~ N(y, s^2) * p0_z(z) by sampling-importance-resampling from
    N(y, s^2), weights = p0_z = exp(log p0(x) - log|dz/dx|).
    """
    P = tr.P
    M = np.r_[np.zeros(n_B), np.ones(n_M)]
    N = M.size
    mu = truth["mu_B"][None] + M[:, None] * truth["delta"][None]
    sig = np.where(M[:, None] > 0.5, truth["sigma_M"][None], truth["sigma_B"][None])
    z_true = mu + sig * rng.standard_normal((N, P))
    s = np.exp(rng.uniform(np.log(s_lo), np.log(s_hi), (N, P)))
    y = z_true + s * rng.standard_normal((N, P))
    draws = np.empty((N, S, P))
    n_prop = prop_factor * S
    for i in range(N):
        zp = y[i] + s[i] * rng.standard_normal((n_prop, P))
        xp = tr.from_z(zp)
        lw = logp0(xp) - tr.log_abs_jac(xp)
        w = np.exp(lw - lw.max())
        draws[i] = xp[rng.choice(n_prop, S, replace=True, p=w / w.sum())]
    return dict(M=M, y=y, s=s, z_true=z_true, draws=draws)


def make_spec(cov_names=()):
    return H.HierSpec(param_names=("ve", "vp", "fp", "ps", "tau_delay"),
                      group_effect=GROUP_EFFECT, separate_sigma=SEPARATE_SIGMA,
                      cov_names=cov_names)


def exact_loglik_fn(layout, y, s):
    y, s = jnp.asarray(y), jnp.asarray(s)

    def ll(phi, data):
        m, sig = H.means_and_sigmas(phi, data["M"], data["X"], layout)
        v = sig ** 2 + s ** 2
        return jnp.sum(-0.5 * (y - m) ** 2 / v - 0.5 * jnp.log(v) - 0.5 * H.LOG2PI)
    return ll


def truth_phi(layout):
    phi = np.zeros(layout.dim)
    phi[layout.sl["a"]] = TRUTH["mu_B"]
    phi[layout.sl["d"]] = TRUTH["delta"][layout.ge]
    phi[layout.sl["ls"]] = np.log(TRUTH["sigma_B"])
    phi[layout.sl["lsm"]] = np.log(TRUTH["sigma_M"][layout.ss])
    return phi


# =============================================================================
# A. transform
# =============================================================================

def test_A(log, prior, tr):
    print("\nA. population-scale transform")
    x = prior.sample(np.random.default_rng(0), 5000)
    log.record("A1 round trip x -> z -> x",
               np.max(np.abs(tr.from_z(tr.to_z(x)) - x) / np.abs(x)) < 1e-10,
               f"max rel err {np.max(np.abs(tr.from_z(tr.to_z(x)) - x) / np.abs(x)):.1e}")

    xs = x[:200]
    num = np.zeros(len(xs))
    for j in range(tr.P):
        h = 1e-6 * np.maximum(np.abs(xs[:, j]), 1e-3)
        xp, xm = xs.copy(), xs.copy()
        xp[:, j] += h
        xm[:, j] -= h
        num += np.log(np.abs((tr.to_z(xp)[:, j] - tr.to_z(xm)[:, j]) / (2 * h)))
    err = np.max(np.abs(num - tr.log_abs_jac(xs)))
    log.record("A2 analytic log|dz/dx| == finite differences", err < 1e-5,
               f"max abs err {err:.1e}")

    _, n_clip = tr.to_z(np.vstack([tr.lb, tr.ub]), return_n_clipped=True)
    log.record("A3 exact bounds are clipped, not +/-inf", n_clip == 2 * tr.P,
               f"{n_clip} clipped")


# =============================================================================
# B. prior consistency
# =============================================================================

def test_B(log, prior, tr):
    print("\nB. prior density used in the importance ratio")
    rng = np.random.default_rng(1)
    x = prior.sample(rng, 5000)
    pkg = H.prior_logpdf_rows(prior)(x)
    vec = vec_logp0(tr)(x)
    err = np.max(np.abs(pkg - vec))
    log.record("B0 test generator's p0 == package PriorSpec.log_pdf",
               err < 1e-10, f"max abs diff {err:.1e} over 5000 draws")

    # E_{x~p0}[ 1 / (V p0(x)) ] = 1 exactly iff p0 is normalised on the box
    xb = prior.sample(rng, 400_000)
    V = np.prod(tr.ub - tr.lb)
    r = np.exp(-vec_logp0(tr)(xb) - np.log(V))
    m, se = r.mean(), r.std() / np.sqrt(r.size)
    log.record("B1 p0 integrates to 1 over the box", abs(m - 1) < 4 * se,
               f"{m:.4f} +/- {se:.4f}")

    # zero-information tumors: posterior = prior, so p(y|phi) = 1 for all phi
    S = 20_000
    draws = prior.sample(rng, S)[None]
    co = H.build_cohort(["z0"], [0.0], draws, tr, H.prior_logpdf_rows(prior))
    spec = make_spec()
    layout = H.PhiLayout(spec)
    _, lw_fn = H.make_two_stage_model(spec, layout)
    data = co.data()
    for k, (mu, lsig) in enumerate(((0.0, 0.0), (0.5, np.log(0.7)),
                                    (-1.0, np.log(1.5)))):
        phi = np.zeros(layout.dim)
        phi[layout.sl["a"]] = mu
        phi[layout.sl["ls"]] = lsig
        phi[layout.sl["lsm"]] = lsig
        w = np.exp(np.asarray(lw_fn(jnp.asarray(phi), data))[0])
        m, se = w.mean(), w.std() / np.sqrt(S)
        kh = H.pareto_khat(np.log(w))
        log.record(f"B2.{k} zero-info tumor: mean weight = 1 "
                   f"(mu={mu}, sigma={np.exp(lsig):.2f})",
                   abs(m - 1) < 4 * se and kh < 0.7,
                   f"{m:.4f} +/- {se:.4f}, k-hat {kh:.2f}")


# =============================================================================
# C. PSIS k-hat
# =============================================================================

def test_C(log):
    print("\nC. PSIS Pareto k-hat")
    rng = np.random.default_rng(2)
    for xi in (0.2, 0.5, 0.8):
        ks = [H.pareto_khat(np.log((rng.uniform(size=20_000) ** -xi - 1) / xi + 1e-300))
              for _ in range(20)]
        log.record(f"C1 GPD tail xi={xi}: mean k-hat within 0.08",
                   abs(np.mean(ks) - xi) < 0.08, f"mean {np.mean(ks):.3f}")
    ks = [H.pareto_khat(rng.normal(0, 0.3, 4000)) for _ in range(20)]
    log.record("C2 low-variance weights: k-hat < 0.3", np.max(ks) < 0.3,
               f"max {np.max(ks):.3f}")


# =============================================================================
# shared synthetic cohort for D, N, E, F
# =============================================================================

def gaussian_fixture(tr, quick, S=None):
    n_per = 30 if quick else 60
    S = S or (2000 if quick else 4000)
    g = make_gaussian_cohort(np.random.default_rng(3), tr, vec_logp0(tr),
                             n_per, n_per, S)
    co = H.build_cohort([f"s{i}" for i in range(2 * n_per)], g["M"], g["draws"],
                        tr, vec_logp0(tr))
    return g, co


def phi_grid(layout, n=40):
    return truth_phi(layout)[None] + 0.25 * np.random.default_rng(99).standard_normal(
        (n, layout.dim))


# =============================================================================
# D. Monte Carlo likelihood
# =============================================================================

def _mc_vs_exact(co, g, spec, layout):
    """per-phi, per-tumor MC and exact log-likelihoods, and k-hat."""
    _, lw_fn = H.make_two_stage_model(spec, layout)
    lw_j = jax.jit(lw_fn)
    ll_ex = exact_loglik_fn(layout, g["y"], g["s"])
    data = co.data()
    mc, ex, kh = [], [], []
    for p in phi_grid(layout):
        lw = np.asarray(lw_j(jnp.asarray(p), data))
        m = lw.max(1, keepdims=True)
        mc.append(m[:, 0] + np.log(np.exp(lw - m).mean(1)))
        mm, sg = H.means_and_sigmas(jnp.asarray(p), data["M"], data["X"], layout)
        v = np.asarray(sg) ** 2 + g["s"] ** 2
        ex.append(np.sum(-0.5 * (g["y"] - np.asarray(mm)) ** 2 / v
                         - 0.5 * np.log(v) - 0.5 * H.LOG2PI, axis=1))
        kh.append([H.pareto_khat(lw[i]) for i in range(lw.shape[0])])
    return np.array(mc), np.array(ex), np.array(kh)


def test_D(log, tr, quick):
    print("\nD. Monte Carlo (importance-reweighting) likelihood")
    spec = make_spec()
    layout = H.PhiLayout(spec)
    g, co = gaussian_fixture(tr, quick)
    mc, ex, kh = _mc_vs_exact(co, g, spec, layout)
    tot_mc, tot_ex = mc.sum(1), ex.sum(1)
    corr = np.corrcoef(tot_mc, tot_ex)[0, 1]
    slope = np.polyfit(tot_ex, tot_mc, 1)[0]
    log.record("D1 MC vs exact log p(y|phi): correlation > 0.999", corr > 0.999,
               f"r = {corr:.6f}")
    log.record("D2 slope within 3% of 1", abs(slope - 1) < 0.03, f"slope {slope:.4f}")

    # the phi-dependent error must be NOISE: 4x the draws -> clearly smaller
    S0 = co.S
    g4, co4 = gaussian_fixture(tr, quick, S=4 * S0)
    mc4, ex4, _ = _mc_vs_exact(co4, g4, spec, layout)
    sd1 = (tot_mc - tot_ex).std()
    sd4 = (mc4.sum(1) - ex4.sum(1)).std()
    log.record("D3 MC error shrinks with more draws (4x draws -> < 0.85x error)",
               sd4 < 0.85 * sd1,
               f"SD across phi {sd1:.3f} -> {sd4:.3f} nats ({S0} -> {4 * S0} draws)")

    # ... and k-hat must point at where it comes from
    per_sd = (mc - ex).std(0)
    flagged = kh.max(0) > 0.7
    share = (per_sd[flagged] ** 2).sum() / (per_sd ** 2).sum()
    log.record("D4 tumors with k-hat > 0.7 carry most of the error (share > 0.5)",
               share > 0.5,
               f"{int(flagged.sum())}/{len(flagged)} tumors flagged, "
               f"{100 * share:.0f}% of per-tumor error variance")


# =============================================================================
# N. Gaussian-approximation likelihood
# =============================================================================

def test_N(log, prior, tr, quick):
    print("\nN. Gaussian-approximation likelihood")
    spec = make_spec()
    layout = H.PhiLayout(spec)
    _, tl_fn, _ = H.make_gaussian_model(spec, layout)
    rng = np.random.default_rng(7)

    # N1 algebra: full-rank Q -> exactly N(l; m, Q^-1 + D) up to a phi-free const
    N, P = 6, 5
    Ls = rng.standard_normal((N, P, P))
    Q = np.einsum("nij,nkj->nik", Ls, Ls) + 0.5 * np.eye(P)
    ell = rng.standard_normal((N, P))
    h = np.einsum("nij,nj->ni", Q, ell)
    Mv = np.r_[np.zeros(3), np.ones(3)]
    dd = dict(Q=jnp.asarray(Q), h=jnp.asarray(h), M=jnp.asarray(Mv),
              X=jnp.zeros((N, 0)))
    diffs = []
    for _ in range(5):
        phi = 0.5 * rng.standard_normal(layout.dim)
        m, sg = (np.asarray(a) for a in H.means_and_sigmas(
            jnp.asarray(phi), dd["M"], dd["X"], layout))
        ref = []
        for i in range(N):
            Cm = np.linalg.inv(Q[i]) + np.diag(sg[i] ** 2)
            r = ell[i] - m[i]
            ref.append(-0.5 * r @ np.linalg.solve(Cm, r)
                       - 0.5 * np.linalg.slogdet(Cm)[1])
        diffs.append(np.asarray(tl_fn(jnp.asarray(phi), dd)) - np.array(ref))
    diffs = np.array(diffs)
    log.record("N1 closed form == N(l; m, Q^-1 + D) (difference constant in phi)",
               np.max(diffs.std(0)) < 1e-8, f"max SD over phi {np.max(diffs.std(0)):.1e}")

    # N2 a tumor with Q = 0, h = 0 contributes exactly 0 for every phi
    dz = dict(Q=jnp.zeros((1, P, P)), h=jnp.zeros((1, P)), M=jnp.zeros(1),
              X=jnp.zeros((1, 0)))
    vals = [float(tl_fn(jnp.asarray(0.7 * rng.standard_normal(layout.dim)), dz)[0])
            for _ in range(5)]
    log.record("N2 uninformative tumor contributes 0 to log p(y|phi)",
               np.max(np.abs(vals)) < 1e-10, f"max |value| {np.max(np.abs(vals)):.1e}")

    # N3 the real prior is separable on z (diagonal curvature is exact)
    lp_rows = H.prior_logpdf_rows(prior)
    sep = max(H.check_prior_separable(lp_rows, tr, z0)
              for z0 in rng.normal(0, 1.5, (5, P)))
    log.record("N3 real TCXM prior separable on z (mixed 2nd derivative ~ 0)",
               sep < 1e-6, f"max {sep:.1e}")

    g, co = gaussian_fixture(tr, quick)
    glik = H.gaussian_likelihood(co, lp_rows)
    small = g["s"] < 0.3
    rel = np.abs(1 / np.sqrt(np.einsum("nii->ni", glik.Q)[small]) - g["s"][small]) \
        / g["s"][small]
    log.record("N4 likelihood-dominated coordinates: recovered SD == true s "
               "(median err < 5%)", np.median(rel) < 0.05,
               f"median {100 * np.median(rel):.1f}%, n = {int(small.sum())}")

    data = glik.data(co)
    ll_g = jax.jit(lambda p, d: tl_fn(p, d).sum())
    ll_ex = jax.jit(exact_loglik_fn(layout, g["y"], g["s"]))
    ga = np.array([float(ll_g(jnp.asarray(p), data)) for p in phi_grid(layout)])
    ex = np.array([float(ll_ex(jnp.asarray(p), co.design())) for p in phi_grid(layout)])
    corr, slope = np.corrcoef(ga, ex)[0, 1], np.polyfit(ex, ga, 1)[0]
    log.record("N5 Gaussian vs exact log p(y|phi): r > 0.99, slope within 3%",
               corr > 0.99 and abs(slope - 1) < 0.03,
               f"r = {corr:.5f}, slope {slope:.4f}")

    glog, _, _ = H.make_gaussian_model(spec, layout)
    phi0 = H.init_phi(co, layout)
    nw = 1000 if quick else 2000
    ch_g, div_g, _, _ = H.sample_hyper(glog, data, phi0, n_warmup=nw, n_draws=nw,
                                       n_chains=4, seed=21)
    exact = lambda p, d: H.log_hyperprior(p, spec, layout) + \
        exact_loglik_fn(layout, g["y"], g["s"])(p, d)
    ch_e, div_e, _, _ = H.sample_hyper(exact, co.design(), phi0, n_warmup=nw,
                                       n_draws=nw, n_chains=4, seed=22)
    rh, es = H.hyper_diagnostics(ch_g, layout.names)
    log.record("N6 Gaussian-likelihood sampler converged (R-hat<=1.01, 0 div)",
               max(rh.values()) <= 1.01 and div_g == 0,
               f"R-hat max {max(rh.values()):.4f}, ESS min {min(es.values()):.0f}, "
               f"div {div_g}")
    cmp = H.compare_methods(ch_g, ch_e, layout)
    dl = cmp[cmp["name"].str.startswith("delta")]
    log.record("N7 group effects (delta) == exact posterior (< 0.25 SD)",
               dl["mean_diff_in_sd"].max() < 0.25,
               f"max {dl['mean_diff_in_sd'].max():.3f} SD")
    w = cmp.loc[cmp["mean_diff_in_sd"].idxmax()]
    log.record("N8 every hyperparameter within 0.5 SD of exact "
               "(approximation; looser by design)",
               cmp["mean_diff_in_sd"].max() < 0.5,
               f"worst {w['name']} {w['mean_diff_in_sd']:.3f} SD; "
               f"{int((cmp['mean_diff_in_sd'] >= 0.25).sum())} component(s) >= 0.25 SD")
    sdr = cmp["sd_ratio_a_over_b"]
    log.record("N9 posterior SDs within [0.8, 1.25] of exact",
               sdr.min() > 0.8 and sdr.max() < 1.25,
               f"range [{sdr.min():.3f}, {sdr.max():.3f}]")
    return ch_e


# =============================================================================
# E-F. MC-likelihood posterior and tumor-level reweighting
# =============================================================================

def test_EF(log, tr, quick, ch_e=None):
    print("\nE-F. MC-likelihood posterior vs exact (slow on CPU)")
    spec = make_spec()
    layout = H.PhiLayout(spec)
    logdensity, lw_fn = H.make_two_stage_model(spec, layout)
    g, co = gaussian_fixture(tr, quick)
    data = co.data()
    phi0 = H.init_phi(co, layout)
    nw = 600 if quick else 2000
    t1 = time.time()
    ch_mc, div_mc, _, _ = H.sample_hyper(logdensity, data, phi0, n_warmup=nw,
                                         n_draws=nw, n_chains=4, seed=11)
    print(f"   MC-likelihood sampler {time.time() - t1:.0f} s")
    if ch_e is None:
        exact = lambda p, d: H.log_hyperprior(p, spec, layout) + \
            exact_loglik_fn(layout, g["y"], g["s"])(p, d)
        ch_e, _, _, _ = H.sample_hyper(exact, co.design(), phi0, n_warmup=1000,
                                       n_draws=1000, n_chains=4, seed=22)
    rh, es = H.hyper_diagnostics(ch_mc, layout.names)
    log.record("E1 MC-likelihood sampler converged (R-hat<=1.01, 0 div)",
               max(rh.values()) <= 1.01 and div_mc == 0,
               f"R-hat max {max(rh.values()):.4f}, ESS min {min(es.values()):.0f}, "
               f"div {div_mc}")
    cmp = H.compare_methods(ch_mc, ch_e, layout)
    w = cmp.loc[cmp["mean_diff_in_sd"].idxmax()]
    log.record("E2 MC posterior means == exact (< 0.25 SD, every component)",
               cmp["mean_diff_in_sd"].max() < 0.25,
               f"worst {w['name']} {w['mean_diff_in_sd']:.3f} SD")
    sdr = cmp["sd_ratio_a_over_b"]
    log.record("E3 MC posterior SDs within [0.8, 1.25] of exact",
               sdr.min() > 0.8 and sdr.max() < 1.25,
               f"range [{sdr.min():.3f}, {sdr.max():.3f}]")
    fe = ch_e.reshape(-1, layout.dim)
    zt = np.abs(fe.mean(0) - truth_phi(layout)) / fe.std(0)
    dz = zt[layout.sl["d"]]
    log.record("E4 group effects recovered by the exact posterior (< 3 SD)",
               dz.max() < 3, "z = " + ", ".join(f"{v:.2f}" for v in dz))

    pick = fe[np.random.default_rng(4).choice(fe.shape[0], 30, replace=False)]
    W, khat, _ = H.tumor_reweighting(pick, lw_fn, data)
    zw = np.einsum("ns,nsp->np", W, co.z)
    cm, cv = [], []
    for p in pick:
        m, sig = (np.asarray(a) for a in H.means_and_sigmas(
            jnp.asarray(p), data["M"], data["X"], layout))
        prec = 1 / g["s"] ** 2 + 1 / sig ** 2
        cm.append((g["y"] / g["s"] ** 2 + m / sig ** 2) / prec)
        cv.append(1 / prec)
    cm, cv = np.array(cm), np.array(cv)
    err = np.abs(zw - cm.mean(0)) / np.sqrt(cv.mean(0) + cm.var(0))
    ok = khat.max(axis=1) <= 0.7
    log.record("F1 reweighted tumor means == analytic (k-hat<=0.7 tumors, "
               "median err < 0.05 SD)", np.median(err[ok]) < 0.05,
               f"median {np.median(err[ok]):.3f} SD over {int(ok.sum())} tumors")
    log.record("F2 ... 99th percentile err < 0.3 SD",
               np.percentile(err[ok], 99) < 0.3,
               f"P99 {np.percentile(err[ok], 99):.3f} SD")
    log.record("F3 k-hat > 0.7 tumors are where reweighting is worse "
               "(median err flagged > unflagged, or none flagged)",
               ok.all() or np.median(err[~ok]) > np.median(err[ok]),
               f"flagged median {np.median(err[~ok]) if (~ok).any() else float('nan'):.3f} SD"
               f" over {int((~ok).sum())} tumors")


# =============================================================================
# G. calibration over replicate cohorts
# =============================================================================

def test_G(log, prior, tr, R, method="gaussian"):
    print(f"\nG. calibration ({method} likelihood): {R} replicate cohorts "
          f"(half with delta = 0)")
    spec = make_spec()
    layout = H.PhiLayout(spec)
    if method == "gaussian":
        logdensity, _, _ = H.make_gaussian_model(spec, layout)
    else:
        logdensity, _ = H.make_two_stage_model(spec, layout)
    lp_rows = H.prior_logpdf_rows(prior)
    cover, fp, n_null = [], 0, 0
    t0 = time.time()
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        null = r % 2 == 0
        truth = dict(TRUTH)
        truth["delta"] = (np.zeros(5) if null else
                          np.r_[rng.uniform(-1, 1, 4), 0.0])
        g = make_gaussian_cohort(rng, tr, vec_logp0(tr), 20, 20, 1000, truth=truth)
        co = H.build_cohort([f"s{i}" for i in range(40)], g["M"], g["draws"],
                            tr, vec_logp0(tr) if method == "mc" else None)
        data = (H.gaussian_likelihood(co, lp_rows).data(co) if method == "gaussian"
                else co.data())
        ch, ndiv, _, _ = H.sample_hyper(logdensity, data, H.init_phi(co, layout),
                                        n_warmup=500, n_draws=500, n_chains=4,
                                        seed=2000 + r)
        d = ch.reshape(-1, layout.dim)[:, layout.sl["d"]]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        tr_d = truth["delta"][layout.ge]
        cover.extend(((lo <= tr_d) & (tr_d <= hi)).tolist())
        if null:
            n_null += len(tr_d)
            fp += int(np.sum((lo > 0) | (hi < 0)))
        if (r + 1) % 5 == 0:
            print(f"   {r + 1}/{R} replicates ({time.time() - t0:.0f} s)")
    cov = np.mean(cover)
    log.record("G1 95% CI coverage of delta in [0.88, 0.99]", 0.88 <= cov <= 0.99,
               f"{cov:.3f} over {len(cover)} intervals")
    log.record("G2 false-positive rate at delta = 0 below 0.12",
               fp / max(n_null, 1) < 0.12, f"{fp}/{n_null}")


# =============================================================================
# H. runner end-to-end on a synthetic run directory
# =============================================================================

def test_H(log, prior, tr):
    print("\nH. hierarchical_bvsm.py end-to-end on a synthetic run directory")
    tmp = tempfile.mkdtemp(prefix="hier_io_")
    try:
        rng = np.random.default_rng(5)
        n_per, S = 20, 1200
        g = make_gaussian_cohort(rng, tr, vec_logp0(tr), n_per, n_per, S)
        names = np.array(tr.names, dtype=object)
        rows = []
        for i in range(2 * n_per):
            sid = f"FMB{900 + i}"
            np.savez_compressed(os.path.join(tmp, f"{sid}_draws.npz"),
                                chains=g["draws"][i].reshape(4, S // 4, tr.P),
                                param_names=names, St=np.zeros(T.size), aif=AIF,
                                t=T, lesion_nvox=50)
            rows.append(dict(subject_id=sid, diagnosis="M" if g["M"][i] else "B",
                             qc_pass=True, error=""))
        # one QC failure (has draws) and one errored fit (no draws): both skipped
        np.savez_compressed(os.path.join(tmp, "FMB998_draws.npz"),
                            chains=g["draws"][0].reshape(4, S // 4, tr.P),
                            param_names=names, St=np.zeros(T.size), aif=AIF, t=T,
                            lesion_nvox=50)
        rows.append(dict(subject_id="FMB998", diagnosis="B", qc_pass=False, error=""))
        rows.append(dict(subject_id="FMB999", diagnosis="M", error="ValueError('x')"))
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "lesion_bayes_results.csv"),
                                  index=False)

        spec_ = importlib.util.spec_from_file_location("hier_runner", RUNNER)
        run = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(run)
        run.RUN_DIR, run.OUT_DIR = tmp, os.path.join(tmp, "out")
        run.N_DRAWS_PER_TUMOR, run.PRIOR_Z_SAMPLES = 1000, 50_000
        run.N_WARMUP, run.N_DRAWS, run.K_PHI_REWEIGHT = 600, 600, 10
        run.LIKELIHOOD = "both"

        run.RUN_HIERARCHICAL = False
        r0 = run.main()
        log.record("H1 diagnostic-only mode stops after stage 1",
                   "summary" not in r0 and os.path.exists(
                       os.path.join(tmp, "out", "stage1_within_between.csv")))

        run.RUN_HIERARCHICAL = True
        r = run.main()
        log.record("H2 QC-fail and errored rows skipped",
                   r["n_tumors"] == 2 * n_per, f"{r['n_tumors']} tumors loaded")
        want = ["stage1_within_between.csv", "stage1_within_between.png",
                "hier_khat.png", "hier_config.json", "hier_method_comparison.csv"]
        want += [f"{b}_{m}.{e}" for m in ("gaussian", "mc") for b, e in (
            ("hier_group_summary", "csv"), ("hier_hyper_draws", "npz"),
            ("hier_hyper_diagnostics", "csv"), ("hier_tumor_estimates", "csv"),
            ("hier_group_effects", "png"), ("hier_shrinkage", "png"))]
        missing = [f for f in want if not os.path.exists(os.path.join(tmp, "out", f))]
        log.record("H3 all outputs written", not missing, f"missing {missing}")
        layout = r["layout"]
        for m in ("gaussian", "mc"):
            rm = r["methods"][m]
            log.record(f"H4 {m}: sampler converged on synthetic run", rm["converged"],
                       f"R-hat max {rm['rhat_max']:.4f}, ESS min {rm['ess_min']:.0f}, "
                       f"div {rm['n_div']}")
            f = rm["chains"].reshape(-1, layout.dim)[:, layout.sl["d"]]
            z = np.abs(f.mean(0) - TRUTH["delta"][layout.ge]) / f.std(0)
            log.record(f"H5 {m}: true delta recovered (< 3 SD; truth 0.6/0.8/0.7/0.5)",
                       z.max() < 3, "z = " + ", ".join(f"{v:.2f}" for v in z))
            log.record(f"H6 {m}: tumor table has one row per tumor",
                       len(rm["tumors"]) == 2 * n_per)
        dl = r["comparison"][r["comparison"]["name"].str.startswith("delta")]
        log.record("H7 runner's gaussian-vs-mc comparison: deltas agree (< 0.25 SD)",
                   dl["agree"].all(),
                   f"max {dl['mean_diff_in_sd'].max():.3f} SD")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=int, default=0,
                    help="replicate cohorts for the calibration test G (0 = skip)")
    ap.add_argument("--coverage-method", default="gaussian", choices=("gaussian", "mc"))
    ap.add_argument("--full", action="store_true",
                    help="larger cohort and longer chains")
    ap.add_argument("--skip-mc", action="store_true",
                    help="skip E-F (the MC-likelihood sampler is slow on CPU)")
    ap.add_argument("--only", default="ABCDNEFH",
                    help="subset of test letters to run, e.g. 'ABC'")
    a = ap.parse_args()
    quick = not a.full
    log = TestLog("hierarchical verification")
    prior, tr = real_prior()
    t0 = time.time()
    if "A" in a.only:
        test_A(log, prior, tr)
    if "B" in a.only:
        test_B(log, prior, tr)
    if "C" in a.only:
        test_C(log)
    if "D" in a.only:
        test_D(log, tr, quick)
    ch_e = None
    if "N" in a.only:
        ch_e = test_N(log, prior, tr, quick)
    if ("E" in a.only or "F" in a.only) and not a.skip_mc:
        test_EF(log, tr, quick, ch_e)
    if a.coverage:
        test_G(log, prior, tr, a.coverage, a.coverage_method)
    if "H" in a.only:
        test_H(log, prior, tr)
    print(f"\ntotal {time.time() - t0:.0f} s")
    sys.exit(0 if log.summary() else 1)


if __name__ == "__main__":
    main()
