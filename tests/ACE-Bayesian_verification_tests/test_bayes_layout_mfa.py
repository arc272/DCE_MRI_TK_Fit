#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:15:42 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
test_bayes_layout_mfa.py
========================
Verification tests for refactor Phase 5.2: free T10/B1 in the Bayesian
parameter layout, their prior block, and the blackjax transform.

The layout, the prior, the JAX forward model and the NLLS fitter are four
independent descriptions of the same parameter vector. Most tests here
check that they AGREE -- on order, on units, and on density -- because a
disagreement between them would produce a sampler that runs, converges,
and reports the wrong parameter under the wrong name.

Run:
    python tests/ACE-Bayesian_verification_tests/test_bayes_layout_mfa.py
"""

import numpy as np
from scipy import stats

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dce_mri.ace_protocol import ACEProtocol
from dce_mri.bayes import BayesConfig, build_param_layout
from dce_mri.bayes_blackjax import build_transform, make_unconstrained_logdensity
from dce_mri.bayes_mfa import make_log_lik_mfa_jax
from dce_mri.bayes_priors import (assemble_prior, make_nuisance_loguniform,
                                  NUISANCE_FLAT_SOURCE)
from dce_mri.config import (AcquisitionConfig, AifConfig, NuisanceConfig,
                            RelaxationConfig, SubjectConfig)
from dce_mri.information import param_names_mfa
from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa
from dce_mri.kinetic_models import TCXM
from dce_mri.synthetic import make_parker_aif
from dce_mri.water_exchange import SSM3


T     = np.arange(0, 530, 5.4)
AIF   = make_parker_aif(T, t_arrival=25.0)
KIN   = np.array([0.45, 0.06, 0.8 / 60, 0.12 / 60])
RELAX = RelaxationConfig()
CB    = BayesConfig(noise="gauss_marginalized_sigma")
BOTH  = NuisanceConfig(fit_T10=True, fit_B1=True)


def _acq(protocol=None):
    return AcquisitionConfig(TR=3.64, FA=10.0, r1=3.8, protocol=protocol)


def _ace():
    return ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0)


def _layout(nuisance=None, wx=True, acq=None):
    return build_param_layout(
        TCXM, CB, acq=acq if acq is not None else _acq(_ace()),
        wx_model=SSM3 if wx else None, relax_cfg=RELAX, nuisance=nuisance)


def _prior(nuisance=None, wx=True):
    return assemble_prior(TCXM, RELAX, wx_model=SSM3 if wx else None,
                          nuisance=nuisance)


# =============================================================================
# 1. Order and units agree across modules
# =============================================================================

def test_layout_order_matches_make_predict_mfa():
    """information.param_names_mfa is the guarded description of
    make_predict_mfa's p_aug layout; the Bayes layout must equal it
    (minus M0, which is marginalized, never a parameter)."""
    for nu, kw in ((None, {}),
                   (NuisanceConfig(fit_T10=True), dict(fit_T10=True)),
                   (NuisanceConfig(fit_B1=True), dict(fit_B1=True)),
                   (BOTH, dict(fit_T10=True, fit_B1=True))):
        L = _layout(nu)
        want = tuple(param_names_mfa("TCXM", wx=True, include_M0=False, **kw))
        assert L.param_names == want, f"{L.param_names} != {want}"


def test_T10_is_in_seconds_not_milliseconds():
    """NuisanceConfig stores ms; make_predict_mfa reads seconds. This is
    the unit trap the Phase 4 mutation test caught in the NLLS fitter."""
    nu = NuisanceConfig(fit_T10=True, T10_bounds_ms=(300.0, 4000.0))
    L = _layout(nu)
    i = L.param_names.index("T10")
    assert L.lb[i] == 0.3 and L.ub[i] == 4.0, (L.lb[i], L.ub[i])

    P = _prior(nu)
    blk = [b for b in P.blocks if b.name == "nuisance"][0]
    assert np.isfinite(blk.log_pdf(np.array([1.53])))       # 1530 ms in s
    assert not np.isfinite(blk.log_pdf(np.array([1530.0])))  # ms: rejected


def test_prior_validates_against_layout():
    """PriorSpec.validate_against is the existing off-by-one guard; it
    must accept every nuisance configuration."""
    for nu in (None, NuisanceConfig(fit_T10=True),
               NuisanceConfig(fit_B1=True), BOTH):
        _prior(nu).validate_against(_layout(nu).param_names)


def test_nuisances_sit_inside_the_model_block():
    """n_model and lb/ub span everything before noise; every consumer in
    bayes.py and bayes_validation.py relies on that."""
    L = _layout(BOTH)
    assert L.n_nuis == 2
    assert L.n_model == L.n_kin + L.n_wx + 2 == 7
    assert L.lb.size == L.ub.size == L.n_model
    assert L.noise_slice == slice(7, 7)          # sigma marginalized


# =============================================================================
# 2. Backward compatibility
# =============================================================================

def test_no_nuisance_leaves_the_layout_unchanged():
    for nu in (None, NuisanceConfig()):
        L = _layout(nu)
        assert L.n_nuis == 0
        assert L.n_model == L.n_kin + L.n_wx
        assert L.param_names == ("ve", "vp", "fp", "ps", "tau_i")


def test_existing_priors_are_unchanged_without_nuisance():
    P = _prior(None)
    assert all(b.name != "nuisance" for b in P.blocks)
    assert P.param_names == ("ve", "vp", "fp", "ps", "tau_i")


# =============================================================================
# 3. The identifiability guardrail reaches the Bayes path
# =============================================================================

def test_free_nuisance_without_acq_is_refused():
    try:
        build_param_layout(TCXM, CB, wx_model=SSM3, relax_cfg=RELAX,
                           nuisance=BOTH)
    except ValueError as exc:
        assert "acq" in str(exc)
        return
    raise AssertionError("free T10/B1 with no acq should raise")


def test_free_nuisance_on_single_flip_angle_is_refused():
    for acq in (_acq(None),
                _acq(ACEProtocol.single_fa(alpha=10.0, TR_ms=3.64,
                                           t_max=T[-1] + 1.0))):
        try:
            _layout(BOTH, acq=acq)
        except ValueError as exc:
            assert "identifiable" in str(exc)
            continue
        raise AssertionError("single-FA + free T10/B1 should be refused")


def test_informative_nuisance_prior_is_refused_not_ignored():
    """The NLLS fitter honours T10_prior/B1_prior. A Bayes path that
    silently substituted a flat prior would disagree with it unannounced."""
    for nu in (NuisanceConfig(fit_B1=True, B1_prior=(0.824, 0.08)),
               NuisanceConfig(fit_T10=True, T10_prior=(1530.0, 0.25))):
        try:
            _prior(nu)
        except NotImplementedError:
            continue
        raise AssertionError(f"{nu} should raise, not fall back to flat")


# =============================================================================
# 4. The prior block's density
# =============================================================================

def test_loguniform_density_matches_hand_formula():
    lb, ub = np.array([0.3, 0.5]), np.array([4.0, 1.5])
    blk = make_nuisance_loguniform([0, 1], ["T10", "B1"], lb, ub)
    x = np.array([1.53, 0.824])
    hand = -np.sum(np.log(x)) - np.sum(np.log(np.log(ub) - np.log(lb)))
    assert abs(blk.log_pdf(x) - hand) < 1e-12
    for bad in (np.array([0.29, 1.0]), np.array([1.0, 1.51])):
        assert not np.isfinite(blk.log_pdf(bad))


def test_loguniform_samples_are_uniform_in_log():
    lb, ub = np.array([0.3, 0.5]), np.array([4.0, 1.5])
    blk = make_nuisance_loguniform([0, 1], ["T10", "B1"], lb, ub)
    s = blk.sample(np.random.default_rng(0), 20000)
    for j in range(2):
        u = (np.log(s[:, j]) - np.log(lb[j])) / (np.log(ub[j]) - np.log(lb[j]))
        assert stats.kstest(u, "uniform").pvalue > 1e-3


# =============================================================================
# 5. The blackjax transform
# =============================================================================

def test_transform_is_registered_for_the_nuisance_block():
    """build_transform dispatches on (name, source) and raises for any
    unregistered pair -- this fails loudly if the tag ever drifts."""
    P = _prior(BOTH)
    L = _layout(BOTH)
    tr = build_transform(P, L.n_total)
    nu_blocks = [b for b in tr.blocks if set(b.indices) == {5, 6}]
    assert len(nu_blocks) == 1 and nu_blocks[0].log_scale


def test_transform_density_equals_numpy_prior():
    """
    The real integration test. transform.log_prior(z) is the prior density
    IN x evaluated from z; it must equal PriorSpec.log_pdf(x) at
    x = constrain(z). A block registered under the wrong family (linear
    instead of log) would pass the dispatch test above and fail here.
    """
    P = _prior(BOTH)
    L = _layout(BOTH)
    tr = build_transform(P, L.n_total)
    x_samples = P.sample(np.random.default_rng(1), 200)
    worst = 0.0
    for x in x_samples:
        z = tr.unconstrain(jnp.asarray(x))
        worst = max(worst, abs(float(tr.log_prior(z)) - P.log_pdf(x)))
    assert worst < 1e-8, f"transform vs numpy prior: {worst:.2e}"


def test_transform_round_trip_and_jacobian():
    P = _prior(BOTH)
    L = _layout(BOTH)
    tr = build_transform(P, L.n_total)
    x = P.sample(np.random.default_rng(2), 1)[0]
    z = tr.unconstrain(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(tr.constrain(z)), x, rtol=1e-10)

    J = np.asarray(jax.jacfwd(tr.constrain)(z))
    ld = float(np.log(np.abs(np.linalg.det(J))))
    assert abs(ld - float(tr.log_det_jacobian(z))) < 1e-8


# =============================================================================
# 6. End to end with the Phase 5.1 likelihood
# =============================================================================

def test_unconstrained_logdensity_is_finite_with_finite_gradient():
    """Layout + prior + transform + bayes_mfa likelihood + make_predict_mfa,
    composed exactly as the sampler will compose them."""
    from functools import partial
    proto = _ace()
    acq = _acq(proto)
    ctx = static_ctx_from_configs(acq, SubjectConfig(subject_id="x"),
                                  AifConfig(), T)
    predict = make_predict_mfa("TCXM", T, AIF, ctx, wx=True,
                               tau_b_fixed=RELAX.tau_b_fixed,
                               fit_T10=True, fit_B1=True)
    p_true = np.r_[KIN, 0.2, 1.53, 1.0]
    y = 1000.0 * np.asarray(predict(jnp.asarray(p_true)))
    y = y + np.random.default_rng(3).normal(0, y[0] / 50.0, y.size)

    L = _layout(BOTH)
    tr = build_transform(_prior(BOTH), L.n_total)
    ll = partial(make_log_lik_mfa_jax(predict, n_frames=T.size),
                 y=jnp.asarray(y))
    logdens = jax.jit(make_unconstrained_logdensity(ll, tr))

    z = tr.unconstrain(jnp.asarray(p_true))
    v = float(logdens(z))
    g = np.asarray(jax.grad(logdens)(z))
    assert np.isfinite(v), v
    assert np.all(np.isfinite(g)), g


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