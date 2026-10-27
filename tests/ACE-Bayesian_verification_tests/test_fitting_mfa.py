#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:52:35 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
test_fitting_mfa.py
===================
Verification tests for dce_mri.fitting_mfa (refactor Phase 4).

References independent of the code under test:
  * a hand-derived closed form for the projected M0
  * a central-difference jacobian of the projected residual, which is the
    only way to check that jacfwd carries the Golub-Pereyra term through
    the projection rather than treating M0_hat as constant
  * information.crlb_percent -- the fitter's empirical spread must match
    the Cramer-Rao bound, which was itself validated against closed-form
    linear-Gaussian results and Monte Carlo

Run:
    python tests/mfa_verification_tests/test_fitting_mfa.py
"""

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dce_mri.ace_protocol import ACEProtocol
from dce_mri.config import (AcquisitionConfig, AifConfig, FittingConfig,
                            NuisanceConfig, RelaxationConfig, SubjectConfig)
from dce_mri.fitting_mfa import (profile_m0, build_mfa_residual,
                                 fit_curve_mfa, MfaResidualBundle)
from dce_mri.information import crlb_percent, param_names_mfa
from dce_mri.kinetic_models import TCXM
from dce_mri.synthetic import make_parker_aif, simulate_curve_mfa
from dce_mri.water_exchange import SSM3


T     = np.arange(0, 530, 5.4)
AIF   = make_parker_aif(T, t_arrival=25.0)
KIN   = np.array([0.45, 0.06, 0.8 / 60, 0.12 / 60])
TAU_I, TAU_B = 0.2, 0.005
T10_MS, B1_REF, M0_REF, R1REL = 1530.0, 1.0, 1000.0, 3.8

SUBJ  = SubjectConfig(subject_id="x", T10=T10_MS, B1=B1_REF)
AIFC  = AifConfig()
RELAX = RelaxationConfig()
CFG   = FittingConfig(n_lhs=200, n_top=5, jacobian="jax")


def _ace(n_settle=0):
    return ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0, n_settle=n_settle)


def _acq(protocol=None):
    return AcquisitionConfig(TR=3.64, FA=10.0, r1=R1REL, protocol=protocol)


def _truth(wx=True, fit_T10=False, fit_B1=False, B1=B1_REF, T10_ms=T10_MS):
    p = list(KIN)
    if wx:
        p.append(TAU_I)
    if fit_T10:
        p.append(T10_ms / 1000.0)
    if fit_B1:
        p.append(B1)
    return np.asarray(p)


def _data(proto, wx=True, B1=B1_REF, T10_ms=T10_MS, M0=M0_REF):
    acq = _acq(proto)
    theta = np.r_[KIN, TAU_I] if wx else KIN
    return simulate_curve_mfa(
        theta, AIF, T, TCXM, acq, SUBJ, CFG,
        wx_model=(SSM3 if wx else None), aif_cfg=AIFC, relax_cfg=RELAX,
        M0=M0, T10_ms=T10_ms, B1=B1)


def _bundle(proto, nu=None, wx=True, sig=None, **kw):
    acq = _acq(proto)
    nu = nu or NuisanceConfig()
    return build_mfa_residual(
        sig if sig is not None else _data(proto, wx=wx),
        AIF, T, TCXM, acq, SUBJ, CFG, nu,
        aif_cfg=AIFC, relax_cfg=RELAX, wx=wx, **kw)


# =============================================================================
# 1. Variable projection -- the closed form
# =============================================================================

def test_profile_m0_matches_hand_derived_closed_form():
    """M0_hat = sum(w g y)/sum(w g^2), derived by hand, not from the code."""
    rng = np.random.default_rng(0)
    g = rng.uniform(0.5, 2.0, 40)
    y = rng.uniform(0.5, 2.0, 40)
    w = rng.choice([0.0, 1.0], 40)
    assert abs(profile_m0(g, y) - np.sum(g * y) / np.sum(g * g)) < 1e-14
    assert abs(profile_m0(g, y, w) - np.sum(w * g * y) / np.sum(w * g * g)) < 1e-14


def test_profile_m0_is_the_actual_minimiser():
    """Independent check: brute-force scan of the weighted SSE must have
    its minimum at the closed form."""
    rng = np.random.default_rng(1)
    g = rng.uniform(0.5, 2.0, 30)
    y = 3.7 * g + rng.normal(0, 0.05, 30)
    m0 = profile_m0(g, y)
    grid = np.linspace(0.5 * m0, 1.5 * m0, 20001)
    sse = ((grid[:, None] * g[None, :] - y[None, :]) ** 2).sum(axis=1)
    assert abs(grid[np.argmin(sse)] - m0) < (grid[1] - grid[0]) * 2


def test_projected_m0_recovers_the_true_M0():
    for m0_true in (1.0, 1000.0, 5e4):
        sig = _data(_ace(), M0=m0_true)
        b = _bundle(_ace(), sig=sig)
        got = b.m0_fn(np.log(_truth()))
        assert abs(got / m0_true - 1) < 1e-9, f"M0 {got} vs {m0_true}"


def test_M0_is_not_among_the_fitted_parameters():
    """The whole point of projection: it is not a search dimension."""
    b = _bundle(_ace())
    assert "M0" not in b.param_names
    assert b.log_lb.size == len(b.param_names) == 5


# =============================================================================
# 2. The Golub-Pereyra jacobian
# =============================================================================

def test_varpro_jacobian_matches_finite_differences():
    """
    THE test for variable projection. M0_hat depends on q, so the correct
    jacobian carries a term through the projection. If jacfwd were
    differentiating with M0_hat held constant, this would fail -- and no
    other test here would notice, because the residual VALUE is identical
    either way.
    """
    b = _bundle(_ace())
    q = np.log(_truth())
    J = b.jac_fn(q)

    fd = np.empty_like(J)
    for k in range(q.size):
        h = 1e-6 * max(abs(q[k]), 1.0)
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        fd[:, k] = (b.residual_fn(qp) - b.residual_fn(qm)) / (2 * h)

    scale = np.max(np.abs(fd))
    rel = np.max(np.abs(J - fd)) / scale
    assert rel < 1e-6, f"varpro jacobian vs finite differences: {rel:.3e}"


def test_jacobian_of_a_constant_M0_residual_would_differ():
    """
    Guards the test above from being vacuous: show that holding M0 fixed
    really does change the jacobian, so the agreement found there is
    informative rather than automatic.
    """
    b = _bundle(_ace())
    q = np.log(_truth())
    m0_fixed = b.m0_fn(q)
    sig = b.target
    w = b.weights

    # rebuild the residual with M0 pinned, differentiate that
    from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa
    ctx = static_ctx_from_configs(_acq(_ace()), SUBJ, AIFC, T)
    pred = make_predict_mfa("TCXM", T, AIF, ctx, wx=True, tau_b_fixed=TAU_B)
    f = lambda qq: jnp.asarray(w) * (m0_fixed * pred(jnp.exp(qq)) - jnp.asarray(sig))
    J_fixed = np.asarray(jax.jacfwd(f)(jnp.asarray(q)))

    J_var = b.jac_fn(q)
    rel = np.max(np.abs(J_var - J_fixed)) / np.max(np.abs(J_var))
    assert rel > 1e-3, (
        "fixed-M0 and projected jacobians are indistinguishable here, so "
        f"the varpro test proves nothing (rel diff {rel:.2e})")


# =============================================================================
# 3. Noiseless recovery
# =============================================================================

def test_recovers_truth_exactly_without_noise():
    for wx in (False, True):
        b = _bundle(_ace(), wx=wx)
        truth = _truth(wx=wx)
        r = fit_curve_mfa(b, CFG, seed=0, p0=truth)
        assert r["success"]
        np.testing.assert_allclose(r["params"], truth, rtol=1e-6)
        assert abs(r["M0"] / M0_REF - 1) < 1e-8


def test_recovers_free_T10_and_B1_without_noise():
    """M4: the configuration the guardrail permits only on an MFA protocol."""
    proto = _ace()
    B1_true, T10_true = 0.85, 1400.0
    sig = _data(proto, B1=B1_true, T10_ms=T10_true)
    nu = NuisanceConfig(fit_T10=True, fit_B1=True)
    b = _bundle(proto, nu=nu, sig=sig)
    truth = _truth(fit_T10=True, fit_B1=True, B1=B1_true, T10_ms=T10_true)
    r = fit_curve_mfa(b, CFG, seed=0, p0=truth)
    assert r["success"]
    np.testing.assert_allclose(r["params"], truth, rtol=1e-5)


def test_reaches_the_truth_from_lhs_starts_alone():
    """Without p0, so the LHS + TRF stages must find it unaided."""
    b = _bundle(_ace())
    r = fit_curve_mfa(b, FittingConfig(n_lhs=400, n_top=8, jacobian="jax"),
                      seed=3)
    assert r["success"]
    np.testing.assert_allclose(r["params"], _truth(), rtol=1e-3)


# =============================================================================
# 4. The settle mask
# =============================================================================

def test_settle_mask_is_picked_up_from_the_protocol():
    b = _bundle(_ace(n_settle=1))
    keep = _ace(n_settle=1).frame_arrays(T).keep
    np.testing.assert_array_equal(b.weights, keep.astype(float))
    assert (1 - b.weights).sum() == 6


def test_zero_weighted_frames_cannot_affect_the_fit():
    """Corrupt the dropped frames beyond recognition; the answer must not
    move at all."""
    proto = _ace(n_settle=1)
    sig = _data(proto)
    keep = proto.frame_arrays(T).keep

    b_clean = _bundle(proto, sig=sig)
    sig_bad = sig.copy()
    sig_bad[~keep] *= 50.0
    b_bad = _bundle(proto, sig=sig_bad)

    q = np.log(_truth())
    np.testing.assert_allclose(b_clean.residual_fn(q), b_bad.residual_fn(q),
                               rtol=0, atol=0)
    r1 = fit_curve_mfa(b_clean, CFG, seed=0, p0=_truth())
    r2 = fit_curve_mfa(b_bad, CFG, seed=0, p0=_truth())
    np.testing.assert_allclose(r1["params"], r2["params"], rtol=1e-9)


def test_all_zero_weights_is_refused():
    try:
        _bundle(_ace(), weights=np.zeros(T.size))
    except ValueError:
        return
    raise AssertionError("all-zero weights should raise")


# =============================================================================
# 5. Guardrails
# =============================================================================

def test_refuses_free_T10_B1_without_a_protocol():
    try:
        _bundle(None, nu=NuisanceConfig(fit_T10=True, fit_B1=True))
    except ValueError as exc:
        assert "identifiable" in str(exc)
        return
    raise AssertionError("single-FA + free T10/B1 must be refused")


def test_refuses_non_profile_m0_modes():
    for mode in ("marginal", "baseline"):
        try:
            _bundle(None, nu=NuisanceConfig(m0_mode=mode))
        except ValueError:
            continue
        raise AssertionError(f"m0_mode={mode!r} should be refused here")


def test_refuses_wx_without_configs():
    acq = _acq(_ace())
    try:
        build_mfa_residual(_data(_ace()), AIF, T, TCXM, acq, SUBJ, CFG,
                           NuisanceConfig(), wx=True)
    except ValueError:
        return
    raise AssertionError("wx without aif_cfg/relax_cfg should raise")


def test_refuses_mismatched_weight_length():
    try:
        _bundle(_ace(), weights=np.ones(7))
    except ValueError:
        return
    raise AssertionError("wrong-length weights should raise")


# =============================================================================
# 6. Priors enter as penalty rows
# =============================================================================

def test_prior_rows_are_appended_and_correct():
    proto = _ace()
    nu = NuisanceConfig(fit_T10=True, fit_B1=True,
                        B1_prior=(0.9, 0.08), T10_prior=(1400.0, 0.25))
    b = _bundle(proto, nu=nu)
    assert b.n_prior_rows == 2

    q = np.log(_truth(fit_T10=True, fit_B1=True))
    r = b.residual_fn(q)
    assert r.size == T.size + 2

    iT, iB = b.param_names.index("T10"), b.param_names.index("B1")
    np.testing.assert_allclose(r[-2], (q[iT] - np.log(1.400)) / 0.25, rtol=1e-12)
    np.testing.assert_allclose(r[-1], (q[iB] - np.log(0.9)) / 0.08, rtol=1e-12)


def test_no_prior_rows_when_priors_are_none():
    b = _bundle(_ace(), nu=NuisanceConfig(fit_T10=True, fit_B1=True))
    assert b.n_prior_rows == 0
    assert b.residual_fn(np.log(_truth(fit_T10=True, fit_B1=True))).size == T.size


def test_a_centred_prior_does_not_move_a_noiseless_fit():
    """Sanity: a prior centred on the truth must be inert on clean data."""
    proto = _ace()
    nu = NuisanceConfig(fit_T10=True, fit_B1=True,
                        B1_prior=(B1_REF, 0.08), T10_prior=(T10_MS, 0.25))
    b = _bundle(proto, nu=nu)
    truth = _truth(fit_T10=True, fit_B1=True)
    r = fit_curve_mfa(b, CFG, seed=0, p0=truth)
    np.testing.assert_allclose(r["params"], truth, rtol=1e-5)


# =============================================================================
# 7. THE GATE: empirical spread vs the Cramer-Rao bound
# =============================================================================

def test_recovery_spread_matches_the_crlb():
    """
    The strong gate. The CRLB is a property of the experiment; a fitter
    that is efficient attains it. Substantially WIDER means an optimiser
    problem (local minima, poor starts, loose tolerances). Substantially
    NARROWER at these SNRs would mean a bug, since no unbiased estimator
    can beat it.

    Uses a high SNR so the estimator is in its asymptotically efficient
    regime -- at low SNR nonlinear bias makes the bound a soft reference
    (measured earlier: fp +3.8% bias at SNR 20).
    """
    proto = _ace()
    acq = _acq(proto)
    ctx_truth = _truth()
    sig = _data(proto)
    SNR, N = 200.0, 60
    sigma = sig[0] / SNR

    names = param_names_mfa("TCXM", wx=True)
    bound, _ = crlb_percent(
        __import__("dce_mri.jax_forward", fromlist=["x"]).make_predict_mfa(
            "TCXM", T, AIF,
            __import__("dce_mri.jax_forward", fromlist=["x"])
            .static_ctx_from_configs(acq, SUBJ, AIFC, T),
            wx=True, tau_b_fixed=TAU_B),
        ctx_truth, M0_REF, SNR, names=names)

    rng = np.random.default_rng(7)
    base = _bundle(proto, sig=sig)          # compile ONCE
    cfg_fast = FittingConfig(n_lhs=1, n_top=1, jacobian="jax")
    ests = []
    for _ in range(N):
        y = sig + rng.normal(0, sigma, sig.size)
        r = fit_curve_mfa(base.with_target(y), cfg_fast, seed=0, p0=ctx_truth)
        if r["success"]:
            ests.append(r["params"])
    ests = np.array(ests)
    assert len(ests) >= 0.9 * N, f"only {len(ests)}/{N} fits converged"

    emp = 100 * ests.std(axis=0, ddof=1) / ctx_truth
    se = 1.0 / np.sqrt(2 * (len(ests) - 1))      # MC error on an SD ratio
    print(f"    N={len(ests)} fits, SNR={SNR:g}, MC SE on each ratio ~{se:.3f}")
    for k, n in enumerate(base.param_names):
        ratio = emp[k] / bound[n]
        print(f"      {n:6s} CRLB {bound[n]:7.3f}%   MC {emp[k]:7.3f}%   "
              f"ratio {ratio:5.3f}")
        assert 0.75 < ratio < 1.35, (
            f"{n}: empirical/CRLB = {ratio:.3f}; the fitter is not "
            "attaining the bound")


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