#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:19:47 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
test_bayes_mfa.py
=================
Verification tests for dce_mri.bayes_mfa (refactor Phase 5.1).

The reference throughout is BRUTE-FORCE NUMERICAL INTEGRATION over
(M0, sigma^2), which shares no algebra with the closed form under test.
The derivation is only trusted because it reproduces that integral --
absolutely, not just up to a constant.

Run:
    python tests/mfa_verification_tests/test_bayes_mfa.py
"""

import numpy as np
from scipy.special import logsumexp
from scipy import stats

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dce_mri.ace_protocol import ACEProtocol
from dce_mri.bayes_mfa import (m0_sigma_marginal_loglik, m0_conditional,
                               make_log_lik_mfa_jax, check_mask)
from dce_mri.config import AcquisitionConfig, AifConfig, SubjectConfig
from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa
from dce_mri.synthetic import make_parker_aif


# =============================================================================
# Brute-force reference
# =============================================================================

def _brute_loglik(g, y, w=None, nM=2001, nU=2001, span=12.0):
    """
    log of  ∫∫ N(y | M0 g, sigma^2 I) · 1 · (1/sigma^2) dM0 dsigma^2,
    by quadrature. Substituting u = log sigma^2 makes the Jeffreys prior
    FLAT in u, so the integrand is just the Gaussian likelihood on a
    uniform (M0, u) grid. No closed-form step appears anywhere.
    """
    if w is not None:
        g, y = g[w == 1], y[w == 1]
    N = g.size
    m0h = (g @ y) / (g @ g)
    s2h = ((y - m0h * g) ** 2).sum() / N
    sd = np.sqrt(s2h / (g @ g))
    M = np.linspace(m0h - 3 * span * sd, m0h + 3 * span * sd, nM)
    U = np.linspace(np.log(s2h) - span, np.log(s2h) + span, nU)
    chi2 = ((y[None, :] - M[:, None] * g[None, :]) ** 2).sum(1)
    logL = (-0.5 * N * (np.log(2 * np.pi) + U[None, :])
            - chi2[:, None] / (2 * np.exp(U)[None, :]))
    return logsumexp(logL) + np.log(M[1] - M[0]) + np.log(U[1] - U[0])


def _toy(rng, N=40, M0=700.0, noise=15.0):
    g = rng.uniform(0.2, 2.0, N)
    return g, M0 * g + rng.normal(0, noise, N)


# Real-model fixture: Zhang 2019 kinetics, DFA schedule, T10/B1 known (M3)
T    = np.arange(0, 530, 5.4)
AIF  = make_parker_aif(T, t_arrival=25.0)
KIN  = np.array([0.45, 0.06, 0.8 / 60, 0.12 / 60])
TAU_I, M0_TRUE = 0.2, 1000.0
SUBJ = SubjectConfig(subject_id="x", T10=1530.0, B1=1.0)
AIFC = AifConfig()


def _predict(proto):
    acq = AcquisitionConfig(TR=3.64, FA=10.0, r1=3.8, protocol=proto)
    ctx = static_ctx_from_configs(acq, SUBJ, AIFC, T)
    return make_predict_mfa("TCXM", T, AIF, ctx, wx=True, tau_b_fixed=0.005)


def _dfa():
    return ACEProtocol.zhang2019_dfa(t_max=T[-1] + 1.0, n_settle=0)


# =============================================================================
# 1. The closed form against brute-force integration
# =============================================================================

def test_matches_brute_force_integral_absolutely():
    """Absolute values, not differences -- so the constants are checked
    too. Varied N, M0 and noise so no special case hides an error."""
    rng = np.random.default_rng(0)
    for _ in range(6):
        N = int(rng.integers(8, 60))
        g, y = _toy(rng, N=N, M0=rng.uniform(10, 5000),
                    noise=rng.uniform(0.5, 60))
        a = m0_sigma_marginal_loglik(g, y)
        b = _brute_loglik(g, y)
        assert abs(a - b) < 1e-8, f"N={N}: analytic {a} vs brute {b}"


def test_masked_frames_match_brute_force_on_kept_frames():
    rng = np.random.default_rng(1)
    g, y = _toy(rng, N=50)
    w = np.ones(50)
    w[[3, 4, 20, 21, 22, 40]] = 0
    a = m0_sigma_marginal_loglik(g, y, w)
    assert abs(a - _brute_loglik(g, y, w)) < 1e-8
    # and a masked frame's value is irrelevant
    y2 = y.copy()
    y2[w == 0] *= 1e6
    assert m0_sigma_marginal_loglik(g, y2, w) == a


# =============================================================================
# 2. The real model: posterior over tau_i
# =============================================================================

def _tau_grid_posteriors():
    """Posterior over a tau_i grid, other parameters at truth (M3),
    computed three ways: analytic marginal, brute-force marginal, and the
    naive profile plug-in."""
    predict = _predict(_dfa())
    g_true = np.asarray(predict(jnp.asarray(np.r_[KIN, TAU_I])))
    rng = np.random.default_rng(5)
    y = M0_TRUE * g_true + rng.normal(0, M0_TRUE * g_true[0] / 20.0, T.size)

    taus = np.exp(np.linspace(np.log(0.02), np.log(3.0), 61))
    ana, bru, prof = [], [], []
    N = T.size
    for t in taus:
        g = np.asarray(predict(jnp.asarray(np.r_[KIN, t])))
        ana.append(m0_sigma_marginal_loglik(g, y))
        bru.append(_brute_loglik(g, y, nM=801, nU=801))
        # naive: plug M0_hat into the sigma-only Orton likelihood
        m0h = (g @ y) / (g @ g)
        chi2 = ((y - m0h * g) ** 2).sum()
        prof.append(-0.5 * N * np.log(chi2))
    norm = lambda a: np.exp(np.array(a) - logsumexp(a))
    return taus, norm(ana), norm(bru), norm(prof)


def test_real_model_posterior_matches_brute_force():
    """
    The gold standard: posterior over tau_i under the real TCXM + SSM3
    forward model on the DFA schedule, analytic marginal vs brute-force
    3-D integration (tau_i grid x M0 x sigma^2).
    """
    taus, ana, bru, _ = _tau_grid_posteriors()
    diff = np.max(np.abs(ana - bru))
    print(f"    max |posterior diff| analytic vs brute force: {diff:.2e}  "
          f"(peak posterior mass {ana.max():.3f})")
    assert diff < 1e-6


def test_marginal_and_profile_converge_as_O_1_over_N():
    """
    The -1/2 log S_gg Occam term depends on theta, so the analytic
    marginal and the naive profile plug-in (M0_hat substituted into the
    sigma-only Orton likelihood) are different posteriors -- but the
    difference is an O(1/N) correction, NOT a practical bias at typical
    frame counts. Measured on a B1 grid, where S_gg varies 1.5x:

        N =  6  TV 0.064     N*TV 0.39
        N = 12  TV 0.025     N*TV 0.30
        N = 99  TV 0.003     N*TV 0.28

    Asserts the TRUE property: non-zero, and shrinking like 1/N. (An
    earlier version of this test asserted the profile plug-in was
    biased at N = 99; the data refuted it.)
    """
    acq = AcquisitionConfig(TR=3.64, FA=10.0, r1=3.8, protocol=_dfa())
    ctx = static_ctx_from_configs(acq, SUBJ, AIFC, T)
    pr = make_predict_mfa("TCXM", T, AIF, ctx, wx=True, tau_b_fixed=0.005,
                          fit_T10=True, fit_B1=True)
    g0 = np.asarray(pr(jnp.asarray(np.r_[KIN, TAU_I, 1.53, 1.0])))
    y = M0_TRUE * g0 + np.random.default_rng(5).normal(
        0, M0_TRUE * g0[0] / 20.0, T.size)
    B1s = np.linspace(0.6, 1.4, 161)
    G = np.array([np.asarray(pr(jnp.asarray(np.r_[KIN, TAU_I, 1.53, b])))
                  for b in B1s])
    norm = lambda a: np.exp(np.array(a) - logsumexp(a))

    tvs = {}
    for n_keep in (6, 12, T.size):
        sel = np.linspace(0, T.size - 1, n_keep).astype(int)
        ana, prof = [], []
        for g_full in G:
            g, yy = g_full[sel], y[sel]
            ana.append(m0_sigma_marginal_loglik(g, yy))
            m0h = (g @ yy) / (g @ g)
            prof.append(-0.5 * g.size * np.log(((yy - m0h * g) ** 2).sum()))
        tvs[n_keep] = 0.5 * np.abs(norm(ana) - norm(prof)).sum()

    print("    TV(marginal, profile): " +
          "  ".join(f"N={n}: {tv:.4f} (N*TV {n*tv:.2f})" for n, tv in tvs.items()))
    ns = sorted(tvs)
    assert tvs[ns[0]] > 0.02, "Occam term should be visible at small N"
    assert all(tvs[a] > tvs[b] for a, b in zip(ns, ns[1:])), "not shrinking with N"
    ntv = [n * tvs[n] for n in ns]
    assert max(ntv) / min(ntv) < 2.0, f"not O(1/N): N*TV = {ntv}"


# =============================================================================
# 3. The M0 conditional
# =============================================================================

def test_m0_conditional_matches_brute_force_moments():
    """Mean and SD of M0 | y under the same priors, by quadrature."""
    rng = np.random.default_rng(2)
    g, y = _toy(rng, N=25, M0=900.0, noise=25.0)
    loc, scale, dof = m0_conditional(g, y)

    N = g.size
    m0h = (g @ y) / (g @ g)
    s2h = ((y - m0h * g) ** 2).sum() / N
    sd = np.sqrt(s2h / (g @ g))
    M = np.linspace(m0h - 40 * sd, m0h + 40 * sd, 4001)
    U = np.linspace(np.log(s2h) - 12, np.log(s2h) + 12, 2001)
    chi2 = ((y[None, :] - M[:, None] * g[None, :]) ** 2).sum(1)
    logL = (-0.5 * N * U[None, :] - chi2[:, None] / (2 * np.exp(U)[None, :]))
    pM = np.exp(logsumexp(logL, axis=1) - logsumexp(logL))
    mean_b = np.sum(pM * M)
    sd_b = np.sqrt(np.sum(pM * (M - mean_b) ** 2))

    t = stats.t(df=dof, loc=loc, scale=scale)
    assert abs(t.mean() - mean_b) / sd_b < 1e-6
    assert abs(t.std() / sd_b - 1) < 1e-5


# =============================================================================
# 4. JAX path
# =============================================================================

def test_jax_and_numpy_agree():
    rng = np.random.default_rng(3)
    g, y = _toy(rng)
    w = np.ones(g.size)
    w[:4] = 0
    a = m0_sigma_marginal_loglik(g, y, w)
    b = float(m0_sigma_marginal_loglik(jnp.asarray(g), jnp.asarray(y),
                                       jnp.asarray(w)))
    assert abs(a - b) < 1e-10


def test_gradient_matches_finite_differences():
    """The sampler needs this gradient; check it through M0_hat."""
    proto = _dfa()
    predict = _predict(proto)
    y = M0_TRUE * np.asarray(predict(jnp.asarray(np.r_[KIN, TAU_I])))
    y = y + np.random.default_rng(4).normal(0, 1.0, y.size)
    log_lik = jax.jit(make_log_lik_mfa_jax(predict, n_frames=T.size))
    f = lambda p: float(log_lik(jnp.asarray(p), jnp.asarray(y)))

    p = np.r_[KIN, TAU_I]
    g_ad = np.asarray(jax.grad(lambda pp: log_lik(pp, jnp.asarray(y)))(jnp.asarray(p)))
    g_fd = np.empty_like(p)
    for k in range(p.size):
        h = 1e-6 * p[k]
        e = np.zeros_like(p)
        e[k] = h
        g_fd[k] = (f(p + e) - f(p - e)) / (2 * h)
    rel = np.max(np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-8))
    assert rel < 1e-5, f"autodiff vs finite differences: {rel:.2e}"


def test_one_compilation_serves_many_targets():
    """y is an argument, so retargeting must not change the compiled
    function -- the lesson from fitting_mfa.with_target."""
    predict = _predict(_dfa())
    log_lik = jax.jit(make_log_lik_mfa_jax(predict, n_frames=T.size))
    p = jnp.asarray(np.r_[KIN, TAU_I])
    y0 = np.asarray(predict(p)) * M0_TRUE
    log_lik(p, jnp.asarray(y0))
    n0 = log_lik._cache_size()
    for s in range(5):
        log_lik(p, jnp.asarray(y0 + s))
    assert log_lik._cache_size() == n0, "retargeting recompiled"


# =============================================================================
# 5. Invariances and guards
# =============================================================================

def test_rescaling_g_shifts_loglik_by_a_constant_only():
    """g is only defined up to the M0 scale; rescaling it must not change
    the POSTERIOR, i.e. only add a theta-independent constant."""
    rng = np.random.default_rng(6)
    g1, y = _toy(rng)
    g2 = g1 * rng.uniform(0.8, 1.2, g1.size)
    c = 37.5
    d1 = m0_sigma_marginal_loglik(c * g1, y) - m0_sigma_marginal_loglik(g1, y)
    d2 = m0_sigma_marginal_loglik(c * g2, y) - m0_sigma_marginal_loglik(g2, y)
    assert abs(d1 - d2) < 1e-10
    assert abs(d1 + np.log(c)) < 1e-10


def test_chi2_min_is_numerically_stable_at_high_snr():
    """
    The naive S_yy - S_gy^2/S_gg subtracts two nearly equal numbers.
    Reference computed from the NOISE vector directly, where no large
    numbers ever meet -- independent of both formulas.
    """
    rng = np.random.default_rng(7)
    g = rng.uniform(0.2, 2.0, 60)
    eps = rng.normal(0, 1e-4, 60)
    y = 3000.0 * g + eps
    exact = eps @ eps - (g @ eps) ** 2 / (g @ g)

    from dce_mri.bayes_mfa import _sums
    _, _, stable = _sums(g, y, np.ones(60), np)
    naive = y @ y - (g @ y) ** 2 / (g @ g)
    rel_stable = abs(stable / exact - 1)
    rel_naive = abs(naive / exact - 1)
    print(f"    chi2_min rel error: stable {rel_stable:.1e}   naive {rel_naive:.1e}")
    assert rel_stable < 1e-6
    assert rel_naive > 100 * rel_stable


def test_rejects_bad_masks():
    for bad in (np.full(10, 0.5), np.r_[np.ones(9), 2.0],
                np.r_[np.ones(2), np.zeros(8)]):
        try:
            check_mask(bad, 10)
        except ValueError:
            continue
        raise AssertionError(f"mask {bad} should be refused")
    try:
        check_mask(np.ones(9), 10)
    except ValueError:
        return
    raise AssertionError("wrong-length mask should be refused")


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")