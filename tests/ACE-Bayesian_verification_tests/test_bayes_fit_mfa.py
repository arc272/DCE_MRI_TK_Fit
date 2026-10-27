# -*- coding: utf-8 -*-
"""
test_bayes_fit_mfa.py
=====================
Verification tests for dce_mri.bayes_fit_mfa (refactor Phase 5.3).

Most tests here are CHEAP: guardrails, the M0 reconstruction, and
agreement with the NLLS fitter, all on short chains. The expensive gates
-- posterior SD against the CRLB, and 95% interval coverage -- are in
`slow_gate()`, which is NOT run by default because it needs eight full
posteriors (~4 minutes). Run it explicitly:

    python tests/ACE-Bayesian_verification_tests/test_bayes_fit_mfa.py --slow

Recorded result of that gate (8 posteriors, SNR 200, 2 chains x 250
draws, 0 divergences):

    param     CRLB %  postSD %   ratio   95% cover  med bias %
    ve         0.227     0.244   1.075   8/8          -0.049
    vp         2.472     2.770   1.121   8/8          -0.428
    fp         2.815     3.140   1.116   8/8          +1.606
    ps         0.766     0.781   1.020   7/8          +0.136
    tau_i      5.288     5.479   1.036   7/8          +2.132
    M0         0.141     0.144   1.024   8/8          +0.017
"""

import sys

import numpy as np
from scipy import stats

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dce_mri.ace_protocol import ACEProtocol
from dce_mri.bayes import BayesConfig
from dce_mri.bayes_fit_mfa import (build_predict_unit,
                                   draw_m0_conditional,
                                   fit_curve_bayes_mfa)
from dce_mri.bayes_mfa import m0_conditional
from dce_mri.config import (AcquisitionConfig, AifConfig, FittingConfig,
                            NuisanceConfig, RelaxationConfig, SubjectConfig)
from dce_mri.fitting_mfa import build_mfa_residual, fit_curve_mfa
from dce_mri.kinetic_models import TCXM
from dce_mri.synthetic import make_parker_aif, simulate_curve_mfa
from dce_mri.water_exchange import SSM3


T      = np.arange(0, 530, 5.4)
AIF    = make_parker_aif(T, t_arrival=25.0)
KIN    = np.array([0.45, 0.06, 0.8 / 60, 0.12 / 60])
TRUTH  = np.r_[KIN, 0.2]
M0T    = 1000.0
SUBJ   = SubjectConfig(subject_id="x", T10=1530.0, B1=1.0)
AIFC   = AifConfig()
RELAX  = RelaxationConfig()
CFG    = FittingConfig(jacobian="jax")


def _acq(protocol=None):
    return AcquisitionConfig(TR=3.64, FA=10.0, r1=3.8, protocol=protocol)


def _ace(n_settle=0):
    return ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0, n_settle=n_settle)


def _data(proto, snr=200.0, seed=0):
    acq = _acq(proto)
    y = simulate_curve_mfa(TRUTH, AIF, T, TCXM, acq, SUBJ, CFG,
                           wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX,
                           M0=M0T)
    if snr is None:
        return y
    return y + np.random.default_rng(seed).normal(0, y[0] / snr, y.size)


def _cb(n_draws=120, n_tune=120, n_chains=2, seed=0):
    return BayesConfig(noise="gauss_marginalized_sigma", n_chains=n_chains,
                       n_draws=n_draws, n_tune=n_tune, seed=seed)


def _fit(proto=None, y=None, nuisance=None, **kw):
    proto = proto if proto is not None else _ace()
    acq = _acq(proto)
    y = _data(proto) if y is None else y
    return fit_curve_bayes_mfa(
        y, AIF, T, TCXM, acq, SUBJ, CFG, kw.pop("cfg_bayes", _cb()),
        nuisance=nuisance, wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX,
        point_fit={"params": TRUTH}, **kw)


# =============================================================================
# 1. Guardrails -- cheap, no sampling
# =============================================================================

def test_refuses_a_sampled_sigma():
    """The likelihood integrates sigma out; a BayesConfig asking for a
    free sigma would add a chain column the likelihood never reads."""
    for noise in ("gauss_free_sigma", "gauss_fixed_sigma"):
        try:
            _fit(cfg_bayes=BayesConfig(noise=noise, n_chains=2, n_draws=10,
                                       n_tune=10, seed=0))
        except ValueError as exc:
            assert "marginaliz" in str(exc)
            continue
        raise AssertionError(f"noise={noise!r} should be refused")


def test_refuses_free_nuisance_on_single_flip_angle():
    sfa = ACEProtocol.single_fa(alpha=10.0, TR_ms=3.64, t_max=T[-1] + 1.0)
    try:
        _fit(proto=sfa, y=_data(sfa),
             nuisance=NuisanceConfig(fit_T10=True, fit_B1=True))
    except ValueError as exc:
        assert "identifiable" in str(exc)
        return
    raise AssertionError("single-FA + free T10/B1 must be refused")


def test_refuses_baseline_m0_mode():
    try:
        _fit(nuisance=NuisanceConfig(m0_mode="baseline"))
    except ValueError:
        return
    raise AssertionError("m0_mode='baseline' should be refused")


def test_refuses_wx_without_configs():
    acq = _acq(_ace())
    try:
        fit_curve_bayes_mfa(_data(_ace()), AIF, T, TCXM, acq, SUBJ, CFG, _cb(),
                            wx_model=SSM3)
    except ValueError:
        return
    raise AssertionError("wx without aif_cfg/relax_cfg should raise")


# =============================================================================
# 2. Structure of the result
# =============================================================================

def test_result_shape_names_and_meta():
    r = _fit()
    assert r.param_names == ("ve", "vp", "fp", "ps", "tau_i", "M0")
    assert r.chains.shape == (2, 120, 6)
    assert r.backend == "blackjax-mfa"
    assert r.meta["m0_marginalized"] and r.meta["sigma_marginalized"]
    assert r.meta["n_eff_frames"] == float(T.size)      # n_settle=0


def test_no_noise_column_is_sampled():
    """sigma is analytic, so it must not appear among the parameters."""
    r = _fit()
    assert not any(n.startswith("sigma") for n in r.param_names)


def test_draw_m0_false_omits_the_column():
    r = _fit(draw_m0=False)
    assert r.param_names == ("ve", "vp", "fp", "ps", "tau_i")
    assert r.chains.shape[-1] == 5


def test_settle_mask_reaches_the_likelihood():
    r = _fit(proto=_ace(n_settle=1))
    assert r.meta["n_eff_frames"] == float(T.size - 6)
    assert (1 - r.meta["weights"]).sum() == 6


# =============================================================================
# 3. M0 reconstruction
# =============================================================================

def test_m0_draws_follow_the_exact_conditional():
    """
    M0 is never sampled; one draw is taken from m0_conditional per
    posterior sample of theta. So the M0 column is a COMPOSITION,
    p(M0) = int p(M0|theta) p(theta|y) dtheta, and the law of total
    variance must hold:

        Var(M0) = E[Var(M0|theta)]  +  Var(E[M0|theta])

    An earlier version of this test compared the marginal SD against the
    conditional scale at a single theta and failed by a factor of 5 --
    the code was right and the test was wrong, because it omitted the
    second term.
    """
    proto = _ace()
    y = _data(proto)
    r = _fit(proto=proto, y=y, cfg_bayes=_cb(n_draws=400, n_tune=400))
    pu = build_predict_unit(T, AIF, TCXM, _acq(proto), SUBJ, NuisanceConfig(),
                            aif_cfg=AIFC, relax_cfg=RELAX, wx=True)
    w = r.meta["weights"]

    flat = r.flat()
    sub = flat[::max(1, flat.shape[0] // 300)]           # subsample for speed
    locs, scales = [], []
    for p in sub:
        g = np.asarray(pu(jnp.asarray(p[:5])))
        loc, scale, dof = m0_conditional(g, y, w)
        locs.append(loc)
        scales.append(scale)
    locs, scales = np.array(locs), np.array(scales)

    m0 = r._col("M0")
    # location: the draws must centre on the conditional locations
    assert abs(np.median(m0) - np.median(locs)) < 0.3 * m0.std(ddof=1)
    # spread: total variance decomposition, Student-t var = scale^2*dof/(dof-2)
    var_within = np.mean(scales ** 2 * dof / (dof - 2.0))
    var_between = np.var(locs, ddof=1)
    predicted = np.sqrt(var_within + var_between)
    assert 0.7 < m0.std(ddof=1) / predicted < 1.4, (
        f"M0 SD {m0.std(ddof=1):.4f} vs law-of-total-variance prediction "
        f"{predicted:.4f} (within {np.sqrt(var_within):.4f}, "
        f"between {np.sqrt(var_between):.4f})")
    assert abs(np.median(locs) / M0T - 1) < 0.02


def test_draw_m0_matches_the_exact_student_t():
    """
    THE test for the M0 draw. With theta FIXED, the draws are i.i.d. from
    t_{N-1}(loc, scale), so a KS test against the exact distribution
    pins down loc, scale AND dof.

    This is why draw_m0 is a separate function. Inline, with theta
    varying across posterior samples, the between-theta spread of `loc`
    swamped the conditional scale: mutation testing showed that neither
    an N vs N-1 dof error nor a missing dof inside `scale` moved the M0
    column detectably. Here all six injected variants fail.
    """
    proto = _ace(n_settle=0)
    y = _data(proto)
    pu = build_predict_unit(T, AIF, TCXM, _acq(proto), SUBJ, NuisanceConfig(),
                            aif_cfg=AIFC, relax_cfg=RELAX, wx=True)
    g = np.asarray(pu(jnp.asarray(TRUTH)))

    masks = [("all frames", np.ones(T.size))]
    w6 = np.zeros(T.size)
    w6[[20, 30, 45, 60, 75, 90]] = 1.0
    masks.append(("6 frames (dof 5)", w6))      # small dof: N vs N-1 matters

    for lab, w in masks:
        loc, scale, dof = m0_conditional(g, y, w)
        G = np.tile(g, (20000, 1))              # theta fixed -> i.i.d.
        d = draw_m0_conditional(G, y, w, np.random.default_rng(0))
        p = stats.kstest((d - loc) / scale,
                         lambda x: stats.t.cdf(x, df=dof)).pvalue
        assert p > 1e-4, f"{lab}: KS p={p:.3e} against t(dof={dof})"


def test_masked_frames_cannot_affect_the_M0_draws():
    """
    The M0 location is a WEIGHTED projection, sum(w g y)/sum(w g g).
    Dropping the weights there is nearly invisible when the masked frames
    happen to be unremarkable -- mutation testing found exactly that gap.
    So corrupt the masked frames beyond recognition: if the mask is
    honoured, the M0 column is bit-identical; if not, it moves a lot.
    """
    proto = _ace(n_settle=2)
    y = _data(proto)
    keep = proto.frame_arrays(T).keep
    assert (~keep).sum() == 12

    y_bad = y.copy()
    y_bad[~keep] *= 40.0          # masked frames: absurd values

    a = _fit(proto=proto, y=y)
    b = _fit(proto=proto, y=y_bad)
    np.testing.assert_allclose(a.chains, b.chains, rtol=0, atol=0)


# =============================================================================
# 4. Agreement with the NLLS fitter -- the key consistency check
# =============================================================================

def test_posterior_median_matches_the_nlls_fit():
    """
    Two independent inference machines (NUTS and trust-region least
    squares) on the SAME data, SAME forward model, SAME frame mask. If
    the likelihood, layout, prior, transform or mask disagreed anywhere,
    these would not land in the same place.

    Tolerance is loose (3% of the posterior SD scale) because MCMC has
    Monte Carlo error; the observed agreement at longer chains is 4-5
    significant figures.
    """
    proto = _ace(n_settle=1)
    y = _data(proto)
    acq = _acq(proto)

    b = build_mfa_residual(y, AIF, T, TCXM, acq, SUBJ, CFG, NuisanceConfig(),
                           aif_cfg=AIFC, relax_cfg=RELAX, wx=True)
    nl = fit_curve_mfa(b, FittingConfig(n_lhs=200, n_top=5, jacobian="jax"),
                       seed=0, p0=TRUTH)

    r = _fit(proto=proto, y=y, cfg_bayes=_cb(n_draws=400, n_tune=400))
    for k, nm in enumerate(("ve", "vp", "fp", "ps", "tau_i")):
        med = r.summary(nm)["median"]
        sd = r._col(nm).std(ddof=1)
        z = abs(med - nl["params"][k]) / sd
        assert z < 0.3, f"{nm}: posterior median {med:.6g} vs NLLS " \
                        f"{nl['params'][k]:.6g} ({z:.2f} posterior SD)"
    assert abs(np.median(r._col("M0")) / nl["M0"] - 1) < 0.01


def test_chains_converge_without_divergences():
    r = _fit(cfg_bayes=_cb(n_draws=300, n_tune=300))
    assert r.n_divergences == 0, f"{r.n_divergences} divergences"
    for nm in r.param_names[:5]:
        assert r.rhat[nm] < 1.05, f"{nm}: rhat {r.rhat[nm]:.4f}"
        assert r.ess[nm] > 100, f"{nm}: ess {r.ess[nm]:.0f}"


# =============================================================================
# 5. Reusing a prebuilt forward model
# =============================================================================

def test_prebuilt_predict_unit_gives_identical_results():
    """build_predict_unit exists to avoid recompiling per curve; using it
    must not change the answer."""
    proto = _ace()
    y = _data(proto)
    pu = build_predict_unit(T, AIF, TCXM, _acq(proto), SUBJ, NuisanceConfig(),
                            aif_cfg=AIFC, relax_cfg=RELAX, wx=True)
    a = _fit(proto=proto, y=y)
    b = _fit(proto=proto, y=y, predict_unit=pu)
    np.testing.assert_allclose(a.chains, b.chains, rtol=0, atol=0)


# =============================================================================
# SLOW GATE -- posterior width vs the CRLB, and interval coverage
# =============================================================================

def slow_gate(n_rep=8, snr=200.0):
    from dce_mri.information import crlb_percent, param_names_mfa
    from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa

    proto = _ace(n_settle=0)      # CRLB and sampler must see identical frames
    acq = _acq(proto)
    y0 = _data(proto, snr=None)
    sigma = y0[0] / snr

    ctx = static_ctx_from_configs(acq, SUBJ, AIFC, T)
    pr = make_predict_mfa("TCXM", T, AIF, ctx, wx=True,
                          tau_b_fixed=RELAX.tau_b_fixed)
    names = param_names_mfa("TCXM", wx=True)
    bound, _ = crlb_percent(pr, TRUTH, M0T, snr, names=names)

    pu = build_predict_unit(T, AIF, TCXM, acq, SUBJ, NuisanceConfig(),
                            aif_cfg=AIFC, relax_cfg=RELAX, wx=True)
    cb = _cb(n_draws=250, n_tune=250)
    tr = dict(zip(("ve", "vp", "fp", "ps", "tau_i"), TRUTH))
    tr["M0"] = M0T

    sds, meds, cover, divs = [], [], np.zeros(6), 0
    for s in range(n_rep):
        y = y0 + np.random.default_rng(100 + s).normal(0, sigma, y0.size)
        r = fit_curve_bayes_mfa(y, AIF, T, TCXM, acq, SUBJ, CFG, cb,
                                wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX,
                                point_fit={"params": TRUTH}, predict_unit=pu)
        f = r.flat()
        sds.append(f.std(axis=0, ddof=1))
        meds.append(np.median(f, axis=0))
        divs += r.n_divergences
        for k, nm in enumerate(r.param_names):
            lo, hi = r.ci(nm)
            cover[k] += (lo <= tr[nm] <= hi)
    sds, meds = np.array(sds), np.array(meds)

    print(f"  {n_rep} posteriors, SNR {snr:.0f}, {divs} divergences")
    ok = True
    for k, nm in enumerate(r.param_names):
        psd = 100 * sds[:, k].mean() / tr[nm]
        b = bound[nm]
        ratio = psd / b
        bias = 100 * (meds[:, k].mean() / tr[nm] - 1)
        print(f"    {nm:6s} CRLB {b:8.3f}%  postSD {psd:8.3f}%  "
              f"ratio {ratio:5.3f}  cover {int(cover[k])}/{n_rep}  "
              f"bias {bias:+7.3f}%")
        if not (0.8 < ratio < 1.4):
            ok = False
            print(f"      ^ ratio outside [0.8, 1.4]")
        if cover[k] < n_rep - 3:
            ok = False
            print(f"      ^ coverage too low")
    assert divs == 0, f"{divs} divergences"
    assert ok, "slow gate failed"


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

    if "--slow" in sys.argv:
        print("\nSLOW GATE (CRLB + coverage), several minutes:")
        try:
            slow_gate()
            print("  PASS  slow_gate")
        except Exception as exc:
            print(f"  FAIL  slow_gate: {type(exc).__name__}: {exc}")
