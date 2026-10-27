# -*- coding: utf-8 -*-
"""
bayes_fit_mfa.py
================
Posterior estimation for multi-flip-angle / ACE curves (refactor
Phase 5.3).

The multi-flip-angle counterpart of bayes.fit_curve_bayes. It composes
pieces that already exist and are separately verified:

    bayes.build_param_layout(..., nuisance=)   Phase 5.2 layout
    bayes_priors.assemble_prior(..., nuisance=) Phase 5.2 priors
    bayes_blackjax.build_transform / sample_blackjax
    bayes_mfa.make_log_lik_mfa_jax             Phase 5.1 M0+sigma marginal
    jax_forward.make_predict_mfa               Phase 3.1 unit-M0 model

and returns a bayes.BayesResult, so every existing consumer of a
posterior (percentile / ci / iqr / summary, the BC1/BC3 figure code)
works unchanged.

Three things differ from fit_curve_bayes
----------------------------------------
1. NO NOISE DIMENSION, EVER. The likelihood integrates sigma out
   analytically along with M0, so cfg_bayes.noise is ignored and the
   sampled vector is the model block alone. A BayesConfig asking for a
   free sigma is refused rather than silently honoured, because the
   resulting chain would have a sigma column the likelihood never used.

2. M0 IS RECONSTRUCTED, NOT SAMPLED. It is marginalized out, so there is
   no M0 column to sample. Its posterior conditional on theta is an
   exact Student-t (bayes_mfa.m0_conditional), so M0 draws are made
   afterwards: one per posterior sample of theta. This is exact
   composition, p(M0) = int p(M0|theta) p(theta|y) dtheta, not an
   approximation -- and it costs one forward evaluation per draw rather
   than an extra sampled dimension.

3. THE FRAME MASK IS LOAD-BEARING. Frames after a flip-angle switch are
   dropped by weight, exactly as in fitting_mfa, so the NLLS fit and the
   posterior see the same data. Without that they are not comparable.

Public API
----------
    fit_curve_bayes_mfa(...)  -> BayesResult  (M0 column appended)
"""

from __future__ import annotations

import time
from functools import partial
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from scipy import stats

from dce_mri.bayes import BayesResult, build_param_layout
from dce_mri.bayes_mfa import m0_conditional, make_log_lik_mfa_jax, check_mask
from dce_mri.bayes_priors import assemble_prior
from dce_mri.config import (AcquisitionConfig, AifConfig, FittingConfig,
                            NuisanceConfig, RelaxationConfig, SubjectConfig,
                            validate_nuisance_compatibility)
from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa


__all__ = ["fit_curve_bayes_mfa", "build_predict_unit", "draw_m0_conditional"]


def draw_m0_conditional(G, y, w, rng):
    """
    One M0 draw per row of G, from the exact conditional
    bayes_mfa.m0_conditional describes:

        M0 | theta ~ t_{N-1}( loc, scale ),
        loc = sum(w g y)/S_gg,  scale = sqrt(chi2_min / ((N-1) S_gg))

    Vectorised over posterior samples (G is (n_samples, T)); the scalar
    m0_conditional is the reference it must reproduce.

    Named *_conditional, not draw_m0: fit_curve_bayes_mfa has a
    `draw_m0: bool` keyword, which shadowed the bare name inside that
    function and turned every call into "'bool' object is not
    callable". Factored out so it can be tested on its own.
    Inline, its dof and scale conventions were UNTESTABLE: with G varying
    across posterior samples, the between-theta spread of `loc` dominates
    the M0 column, and mutation testing confirmed that neither an N vs
    N-1 error nor a missing dof in `scale` changed the column enough to
    detect. With G fixed, the draws are i.i.d. and a KS test against the
    exact Student-t catches both.
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    S_gg = (w * G * G).sum(axis=1)
    loc = (w * G * y).sum(axis=1) / S_gg
    chi2 = (w * (y[None, :] - loc[:, None] * G) ** 2).sum(axis=1)
    dof = float(w.sum() - 1.0)
    scale = np.sqrt(chi2 / (dof * S_gg))
    return loc + scale * rng.standard_t(dof, size=loc.size)


def build_predict_unit(time_var, aif, model, acq, subject, nuisance,
                       *, aif_cfg=None, relax_cfg=None, wx=False):
    """
    The unit-M0 forward model, built once so its compilation can be
    REUSED across curves.

    fit_curve_bayes_mfa builds one internally when not given one, which
    means a fresh jit trace per call. That is the same trap
    fitting_mfa.MfaResidualBundle.with_target exists to avoid: a loop
    over voxels or noise replicates accumulates one compiled binary per
    iteration and runs out of memory within a few dozen.
    """
    fit_tau_b = bool(relax_cfg.fit_tau_b) if (wx and relax_cfg is not None) else False
    ctx = static_ctx_from_configs(acq, subject, aif_cfg, time_var)
    return make_predict_mfa(
        model.name, time_var, aif, ctx,
        wx=wx, fit_tau_b=fit_tau_b,
        tau_b_fixed=(relax_cfg.tau_b_fixed if relax_cfg is not None else 0.005),
        fit_T10=nuisance.fit_T10, fit_B1=nuisance.fit_B1,
        upsample=getattr(acq, "upsample", 50),
    )


def fit_curve_bayes_mfa(
    sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
    *,
    nuisance:   Optional[NuisanceConfig] = None,
    wx_model=None, aif_cfg=None, relax_cfg=None,
    weights:    Optional[np.ndarray] = None,
    use_keep:   bool = True,
    point_fit=None,
    seed:       Optional[int] = None,
    draw_m0:    bool = True,
    predict_unit=None,
) -> BayesResult:
    """
    Posterior for one multi-flip-angle curve, with M0 and sigma
    integrated out analytically.

    Parameters mirror fit_curve_bayes. The additions:

    nuisance : NuisanceConfig. fit_T10/fit_B1 add trailing parameters, in
               that order, matching make_predict_mfa. Guarded by
               validate_nuisance_compatibility, so single-flip-angle data
               cannot free them.
    weights  : explicit frame mask; defaults to
               acq.frame_arrays(time_var).keep when a protocol is set.
    point_fit: an NLLS result (fitting_mfa.fit_curve_mfa's dict) used to
               initialise the chains. Only its "params" entry is read.
    draw_m0  : append an M0 column drawn from its exact conditional. Set
               False to keep the chains purely in the sampled parameters.
    predict_unit : a prebuilt make_predict_mfa callable, to REUSE ITS
               COMPILATION across curves. Built here when None, which
               costs a fresh jit trace per call. Build it once with
               build_predict_unit() and pass it in for any loop over
               curves. The caller is then responsible for it matching
               `nuisance`, `wx_model` and `relax_cfg`; the layout built
               here is checked against it.

               MEASURED, not assumed: sample_blackjax still recompiles
               its NUTS kernel per call, because this function binds y
               into the likelihood closure (bayes_mfa's log_lik takes y
               as an argument precisely to avoid that, and partial()
               here gives it up). The cost is TIME, not memory -- peak
               RSS grows ~20 MB per curve over 6 curves, which is
               retained-cache growth, not a leak. Reusing predict_unit
               cuts per-curve cost from ~77 s to ~56 s at 4x400 draws.
               Threading y through sample_blackjax as a traced argument
               would remove the rest, at the cost of changing a function
               the existing verified suites depend on.

    Returns
    -------
    BayesResult whose chains are (n_chains, n_draws, D) in CONSTRAINED
    space, with an M0 column appended last when draw_m0. meta carries
    the frame mask, the effective N and the backend details.
    """
    t0 = time.time()

    nuisance = nuisance or NuisanceConfig()
    validate_nuisance_compatibility(nuisance, acq)
    if nuisance.m0_mode not in ("profile", "marginal"):
        raise ValueError(
            f"m0_mode={nuisance.m0_mode!r}: this sampler always marginalizes "
            "M0 analytically; 'baseline' is the legacy single-FA convention")

    noise_name = getattr(cfg_bayes, "noise", "gauss_marginalized_sigma")
    if noise_name != "gauss_marginalized_sigma":
        raise ValueError(
            f"cfg_bayes.noise={noise_name!r} is incompatible: this likelihood "
            "integrates sigma out analytically, so a sampled or fixed sigma "
            "would be a chain column the likelihood never reads. Use "
            "'gauss_marginalized_sigma'.")
    if wx_model is not None and (aif_cfg is None or relax_cfg is None):
        raise ValueError("water exchange needs aif_cfg and relax_cfg")
    if wx_model is not None and wx_model.name != "SSM3":
        # make_predict_mfa only knows the SSM3 matrix; passing wx=True for
        # any other model would silently fit SSM3 instead.
        raise NotImplementedError(
            f"multi-flip-angle Bayes path has no forward model for "
            f"wx_model={wx_model.name!r} (have: SSM3). NXL is wired through "
            f"the single-FA path (bayes.fit_curve_bayes) only.")

    if seed is not None:
        from dce_mri.bayes import BayesConfig
        cfg_bayes = BayesConfig(**{**cfg_bayes.__dict__, "seed": seed})

    sig = np.asarray(sig, dtype=float)
    time_var = np.asarray(time_var, dtype=float)

    # --- frame mask, identical rule to fitting_mfa ----------------------
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
    elif use_keep and getattr(acq, "protocol", None) is not None:
        w = acq.frame_arrays(time_var).keep.astype(float)
    else:
        w = np.ones(sig.size)
    w = check_mask(w, sig.size)

    # --- forward model, layout, prior -----------------------------------
    wx = wx_model is not None
    fit_tau_b = bool(relax_cfg.fit_tau_b) if (wx and relax_cfg is not None) else False
    if predict_unit is None:
        predict_unit = build_predict_unit(
            time_var, aif, model, acq, subject, nuisance,
            aif_cfg=aif_cfg, relax_cfg=relax_cfg, wx=wx)

    layout = build_param_layout(
        model, cfg_bayes, aif=aif, time_var=time_var, acq=acq,
        wx_model=wx_model, relax_cfg=relax_cfg, nuisance=nuisance)
    prior = assemble_prior(
        model, relax_cfg, wx_model=wx_model,
        delay_mode=getattr(cfg_bayes, "delay_mode", "none"),
        n_noise=0, nuisance=nuisance)
    prior.validate_against(layout.param_names)

    if layout.n_noise != 0:
        raise ValueError(
            f"layout has {layout.n_noise} noise parameters; expected 0 "
            "(sigma is marginalized analytically)")

    # --- likelihood, with y bound for this one curve --------------------
    log_lik = partial(
        make_log_lik_mfa_jax(predict_unit, weights=w, n_frames=sig.size),
        y=jnp.asarray(sig))

    # --- initial point ---------------------------------------------------
    init_point = None
    if point_fit is not None:
        pe = point_fit["params"] if isinstance(point_fit, dict) \
            else np.asarray(point_fit.params, float)
        pe = np.asarray(pe, dtype=float)
        if pe.size == layout.n_total and np.all(np.isfinite(pe)):
            init_point = pe

    # --- sample -----------------------------------------------------------
    from dce_mri.bayes_blackjax import sample_blackjax
    chains, rhat, ess, mcse_mean, mcse_median, n_div, info = sample_blackjax(
        log_lik, layout, prior, cfg_bayes, init_point=init_point)

    names = tuple(layout.param_names)

    # --- M0: exact conditional, one draw per posterior sample -------------
    if draw_m0:
        n_c, n_d, D = chains.shape
        flat = chains.reshape(-1, D)
        # vmap, not a Python loop: one forward evaluation per posterior
        # sample is thousands of calls, and looping them in Python
        # dominated the whole fit.
        G = np.asarray(jax.vmap(predict_unit)(jnp.asarray(flat)), dtype=float)
        m0 = draw_m0_conditional(G, sig, w, np.random.default_rng(cfg_bayes.seed))
        chains = np.concatenate([chains, m0.reshape(n_c, n_d, 1)], axis=2)
        names = names + ("M0",)

    return BayesResult(
        param_names=names,
        chains=chains,
        point_estimate=init_point,
        rhat=rhat, ess=ess, mcse_mean=mcse_mean, mcse_median=mcse_median,
        n_divergences=int(n_div),
        backend="blackjax-mfa",
        elapsed_s=time.time() - t0,
        meta=dict(info or {}, weights=w, n_eff_frames=float(w.sum()),
                  m0_marginalized=True, sigma_marginalized=True,
                  m0_drawn=bool(draw_m0)),
    )
