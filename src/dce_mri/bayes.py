# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
bayes.py
========
Bayesian posterior estimation for kinetic + water-exchange models.

Primary backend is PyMC (gradient-free steppers), because the forward
model (scipy expm / FFT convolution inside the kinetic models) is not
written in a form PyTensor can autodifferentiate. The forward model is
wrapped in a black-box Op and sampled with DEMetropolisZ / DEMetropolis
/ Metropolis / Slice.

ONE module covers all model configurations. That is not new design work
-- it follows the layout convention fit_single_voxel already uses:

    p_aug = [ *theta_kin , (tau_delay?) , (tau_b?) , tau_i? , (noise?) ]

Delay-on/off and wx-on/off are therefore CONFIGURATION, not code paths.
The sampler never branches on model identity; D ranges from 3 (GKM, no
delay, no wx) to 8 (TCXM + delay + both taus + Student-t noise).

Public API
----------
    NoiseModelSpec + the six named noise options
    BayesConfig, BayesResult
    build_param_layout(...)      -- single source of truth for ordering
    make_log_posterior(...)      -- numpy log-posterior (any backend)
    fit_curve_bayes(...)         -- single-curve entry point
    fit_curve_bayes_delay_grid(...) -- Orton-style delay marginalisation
    posterior_vs_prior_width(...)

Delay handling (see DELAY MODES below)
    'none'          no delay term
    'joint'         tau_delay sampled as a free parameter        (D+1)
    'marginalized'  logsumexp over a tau grid, tau integrated out (D)
                    [Orton2007 Eq. 10]
    'fixed'         tau held at a supplied value                 (D)
"""


import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.special import logsumexp, gammaln

from dce_mri.config import (
    AcquisitionConfig, SubjectConfig, FittingConfig,
    AifConfig, RelaxationConfig,
)
from dce_mri.kinetic_models import KineticModelSpec
from dce_mri.water_exchange import WaterExchangeModelSpec
from dce_mri.fitting import (
    build_residual, build_delay_augmented_spec, fit_single_voxel,
    _validate_wx_compatibility,
)

from dce_mri.bayes_priors import PriorSpec, assemble_prior


# =============================================================================
# Noise / likelihood layer
# =============================================================================

@dataclass(frozen=True)
class NoiseModelSpec:
    """
    How a residual vector becomes a log-likelihood number.

    n_params    : extra dimensions the sampler must explore
                  0 = sigma fixed, 1 = sigma free, 2 = Student-t
    param_names : names of those extra parameters
    log_lik     : (resid, pred, noise_params, ctx) -> float
                  `pred` is passed as well as `resid` because the
                  signal-scaled variant needs the predicted curve.
    prior_key   : which noise prior assemble_prior should build
    """
    name:        str
    source:      str
    n_params:    int
    param_names: tuple
    log_lik:     Callable
    prior_key:   Optional[str] = None
    marginalized: bool = False


_LOG_2PI = float(np.log(2.0 * np.pi))


def _ll_gauss_fixed(resid, pred, noise_params, ctx):
    """
    Constant sigma, supplied in ctx['sigma'] and held fixed.

    log L = -N/2 log(2 pi sigma^2) - (1/2 sigma^2) * sum(resid^2)

    Cheapest option: sigma never enters p_aug. Risk is mechanical, not
    philosophical -- a sigma that is too small makes the likelihood
    falsely strict and the posterior artificially narrow, and nothing
    inside the sampler can notice or correct for it.
    """
    s = ctx["sigma"]
    n = resid.size
    return -0.5 * n * (_LOG_2PI + 2.0 * np.log(s)) - 0.5 * np.sum(resid ** 2) / (s * s)


def _ll_gauss_free(resid, pred, noise_params, ctx):
    """Same form, but sigma is sampled. Its prior is a separate block."""
    s = float(noise_params[0])
    if s <= 0:
        return -np.inf
    n = resid.size
    return -0.5 * n * (_LOG_2PI + 2.0 * np.log(s)) - 0.5 * np.sum(resid ** 2) / (s * s)


def _ll_gauss_signal_scaled(resid, pred, noise_params, ctx):
    """
    Per-timepoint variance proportional to the predicted signal:
        Var_t = sigma^2 * |pred_t| / mean(|pred|)

    [SOURCE: Sitek2016 (Var ∝ C_i/(T_i d_i)), Zhou2013]

    Not a prior on an extra unknown -- a change to the SHAPE of the
    likelihood. This is the principled answer when diagnose_residuals()
    shows variance tracking the bolus: a single flat sigma would let
    high-signal timepoints dominate the sum of squares far more than
    their true noise level justifies.

    Normalised by mean(|pred|) so sigma keeps the same interpretation
    (average noise sd) as in the constant-variance options.
    """
    s = float(noise_params[0]) if len(noise_params) else ctx["sigma"]
    if s <= 0:
        return -np.inf
    a = np.abs(pred)
    m = np.mean(a)
    if not np.isfinite(m) or m <= 0:
        return -np.inf
    var = (s * s) * np.maximum(a / m, 1e-6)
    return float(-0.5 * np.sum(_LOG_2PI + np.log(var) + resid ** 2 / var))


def _ll_student_t(resid, pred, noise_params, ctx):
    """
    Student-t residuals, noise_params = (sigma, inv_nu), nu = 1/inv_nu.

    [SOURCE: Zhou2013]

    Squaring in the Gaussian log-likelihood means one badly-off frame
    (motion, susceptibility) contributes its squared deviation and can
    hijack the whole fit. The t density still penalises a large residual
    but far less aggressively, so a single corrupted frame does not
    dominate. Zhou found normality rejected in the MAJORITY of real
    voxels, traced to a few low-signal frames.
    """
    s = float(noise_params[0])
    inv_nu = float(noise_params[1])
    if s <= 0 or inv_nu <= 0:
        return -np.inf
    nu = 1.0 / inv_nu
    n = resid.size
    z2 = (resid / s) ** 2
    return float(
        n * (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
             - 0.5 * np.log(np.pi * nu) - np.log(s))
        - 0.5 * (nu + 1.0) * np.sum(np.log1p(z2 / nu))
    )


def _ll_gauss_marginalized(resid, pred, noise_params, ctx):
    """
    Sigma integrated out ANALYTICALLY under the Jeffreys prior
    p(sigma^2) ∝ 1/sigma^2.

    [SOURCE: Orton2007 Eq. 8]

        int (2 pi s^2)^(-N/2) exp(-chi2 / 2 s^2) (1/s^2) d(s^2)
            ∝ Gamma(N/2) * (chi2/2)^(-N/2)

    so  log L = const - (N/2) * log(chi2).

    Two things worth noting. First, the sampler dimension stays exactly
    where it was under fixed-sigma while STILL propagating uncertainty
    about the noise scale into every other parameter. Second, the
    resulting chi2^(-N/2) form behaves like a Student-t likelihood in
    theta -- heavy-tailed robustness falls out for free, with no
    Student-t assumption ever made explicitly.
    """
    n = resid.size
    chi2 = float(np.sum(resid ** 2))
    if not np.isfinite(chi2) or chi2 <= 0:
        return -np.inf
    return float(gammaln(0.5 * n) - 0.5 * n * np.log(0.5 * chi2)
                 - 0.5 * n * _LOG_2PI)


GAUSS_FIXED_SIGMA = NoiseModelSpec(
    "gauss_fixed_sigma", "-", 0, (), _ll_gauss_fixed, None)

GAUSS_FREE_JEFFREYS = NoiseModelSpec(
    "gauss_free_jeffreys", "Orton2007", 1, ("sigma",),
    _ll_gauss_free, "jeffreys")

GAUSS_FREE_INVGAMMA = NoiseModelSpec(
    "gauss_free_invgamma", "Scannell2019/Zhou2013", 1, ("sigma",),
    _ll_gauss_free, "invgamma")

GAUSS_FREE_LOGNORMAL = NoiseModelSpec(
    "gauss_free_lognormal", "Mittermeier2019", 1, ("sigma",),
    _ll_gauss_free, "lognormal")

GAUSS_SIGNAL_SCALED = NoiseModelSpec(
    "gauss_signal_scaled", "Sitek2016/Zhou2013", 1, ("sigma",),
    _ll_gauss_signal_scaled, "jeffreys")

STUDENT_T = NoiseModelSpec(
    "student_t", "Zhou2013", 2, ("sigma", "inv_nu"),
    _ll_student_t, None)

GAUSS_MARGINALIZED_SIGMA = NoiseModelSpec(
    "gauss_marginalized_sigma", "Orton2007", 0, (),
    _ll_gauss_marginalized, None, marginalized=True)


NOISE_OPTIONS = {
    m.name: m for m in (
        GAUSS_FIXED_SIGMA, GAUSS_FREE_JEFFREYS, GAUSS_FREE_INVGAMMA,
        GAUSS_FREE_LOGNORMAL, GAUSS_SIGNAL_SCALED, STUDENT_T,
        GAUSS_MARGINALIZED_SIGMA,
    )
}


# =============================================================================
# Config / result
# =============================================================================

@dataclass(frozen=True)
class BayesConfig:
    """
    backend      : 'pymc' (default) | 'emcee' | 'mh_reflect'
    step         : PyMC stepper -- 'demetropolisz' (recommended
                   gradient-free default) | 'demetropolis'
                   | 'metropolis' | 'slice'
                   | 'nuts' | 'hmc'  (GRADIENT-BASED -- see cost warning
                     on _make_pymc_op_with_grad before using either. The
                     forward model has no analytic gradient, so these use
                     finite-difference gradients: 2*D forward-model calls
                     per gradient, and NUTS/HMC need one gradient per
                     leapfrog step, typically dozens per draw. Routinely
                     100-300x more forward-model evaluations per draw
                     than DEMetropolisZ for D=6-7. Measure on one row
                     before adopting for a sweep.)
    delay_mode   : 'none' | 'joint' | 'marginalized' | 'fixed'
    delay_grid   : tau values for 'marginalized'  [Orton2007]
    delay_fixed  : tau value for 'fixed'
    n_draws      : kept draws per chain (post burn-in)
    n_tune       : burn-in / adaptation draws, discarded
    n_chains     : independent chains (needed for R-hat)
    target_accept: 0.234 for random-walk Metropolis  [Roberts1997]
    init_from_nlls: seed chains near the NLLS point estimate. Not
                   required for correctness -- burn-in exists precisely
                   so the start is forgotten -- but avoids wasting early
                   iterations wandering toward the right neighbourhood.
    """
    prior_choices: dict  = field(default_factory=dict)
    noise:         str   = "gauss_free_jeffreys"
    sigma_fixed:   Optional[float] = None
    backend:       str   = "pymc"
    step:          str   = "demetropolisz"
    delay_mode:    str   = "none"
    delay_grid:    Optional[np.ndarray] = None
    delay_fixed:   float = 0.0
    delay_bounds:  tuple = (-10.0, 30.0)
    n_draws:       int   = 2000
    n_tune:        int   = 2000
    n_chains:      int   = 4
    target_accept: float = 0.234
    init_from_nlls: bool = True
    n_walkers:     Optional[int] = None     # emcee only
    seed:          int   = 0
    progressbar:   bool  = True
    dense_mass_matrix: bool = True   # False = diagonal (BlackJAX default,
                                      # == PyMC init='adapt_diag'); True is
                                      # the equivalent of 'jitter+adapt_full'


@dataclass
class BayesResult:
    """
    Deliberately mirrors BootstrapWxResult's accessor names
    (percentile / ci / iqr / summary) so existing BC1/BC3 figure code
    can consume posteriors and bootstrap replicates interchangeably.
    """
    param_names:    tuple
    chains:         np.ndarray            # (n_chains, n_draws, D)
    point_estimate: Optional[np.ndarray]  # NLLS anchor, may be None
    rhat:           dict = field(default_factory=dict)
    ess:            dict = field(default_factory=dict)
    mcse_mean:      dict = field(default_factory=dict)
    mcse_median:    dict = field(default_factory=dict)
    accept_frac:    Optional[float] = None
    n_divergences:  Optional[int]   = None
    backend:        str  = "pymc"
    elapsed_s:      float = 0.0
    meta:           dict = field(default_factory=dict)

    # -- interface shared with BootstrapWxResult ----------------------
    def flat(self) -> np.ndarray:
        return self.chains.reshape(-1, self.chains.shape[-1])

    def _col(self, param):
        j = (self.param_names.index(param) if isinstance(param, str) else param)
        return self.flat()[:, j]

    def percentile(self, q, param=None):
        if param is None:
            return np.percentile(self.flat(), q, axis=0)
        return float(np.percentile(self._col(param), q))

    def ci(self, param, lo=2.5, hi=97.5):
        return (self.percentile(lo, param), self.percentile(hi, param))

    def iqr(self, param):
        return self.percentile(75, param) - self.percentile(25, param)

    def summary(self, param=None):
        if param is None:
            return {p: self.summary(p) for p in self.param_names}
        j = self.param_names.index(param)
        lo, hi = self.ci(param)
        d = dict(median=self.percentile(50, param),
                 ci_lo=lo, ci_hi=hi, iqr=self.iqr(param),
                 mean=float(np.mean(self._col(param))),
                 sd=float(np.std(self._col(param))),
                 rhat=self.rhat.get(param, np.nan),
                 ess=self.ess.get(param, np.nan),
                 mcse_mean=self.mcse_mean.get(param, np.nan),
                 mcse_median=self.mcse_median.get(param, np.nan))
        if self.point_estimate is not None and j < len(self.point_estimate):
            d["point"] = float(self.point_estimate[j])
        return d

    def converged(self, rhat_max=1.01, ess_min=400) -> bool:
        if not self.rhat:
            return False
        ok_r = all(np.isfinite(v) and v <= rhat_max for v in self.rhat.values())
        ok_e = all(np.isfinite(v) and v >= ess_min for v in self.ess.values())
        return bool(ok_r and ok_e)


# =============================================================================
# Parameter layout -- SINGLE source of truth for p_aug ordering
# =============================================================================

@dataclass(frozen=True)
class ParamLayout:
    model:        KineticModelSpec   # possibly delay-augmented
    param_names:  tuple              # full p_aug, incl. noise
    lb:           np.ndarray         # model block only (no noise)
    ub:           np.ndarray
    n_kin:        int
    n_wx:         int
    n_noise:      int
    noise_slice:  slice
    n_nuis:       int = 0            # free T10/B1, after tau, before noise

    @property
    def n_model(self) -> int:
        """Everything before the noise block: kinetics, tau, T10/B1."""
        return self.n_kin + self.n_wx + self.n_nuis

    @property
    def n_total(self) -> int:
        return self.n_model + self.n_noise


def build_param_layout(
    model:     KineticModelSpec,
    cfg_bayes: BayesConfig,
    *,
    aif=None, time_var=None, acq=None,
    wx_model=None, relax_cfg=None,
    nuisance=None,
) -> ParamLayout:
    """
    Assemble the augmented model spec and the full parameter ordering.

    For delay_mode='joint' this calls fitting.build_delay_augmented_spec,
    so the Bayesian layout is guaranteed identical to what
    fit_with_delay_continuous produces rather than a re-derivation.
    """
    noise = NOISE_OPTIONS[cfg_bayes.noise]

    if cfg_bayes.delay_mode == "joint":
        if aif is None or time_var is None or acq is None:
            raise ValueError("delay_mode='joint' needs aif, time_var, acq")
        model_use = build_delay_augmented_spec(
            model, aif, time_var, acq,
            tau_lb=cfg_bayes.delay_bounds[0],
            tau_ub=cfg_bayes.delay_bounds[1],
            wx_model=wx_model,
        )
    else:
        model_use = model

    n_kin = len(model_use.p0)
    names = list(model_use.param_names)
    lb = list(np.asarray(model_use.lb, float))
    ub = list(np.asarray(model_use.ub, float))

    n_wx = 0
    if wx_model is not None and wx_model.n_params > 0:   # NXL adds nothing
        if relax_cfg is None:
            raise ValueError("wx_model requires relax_cfg")
        if relax_cfg.fit_tau_b:
            names += ["tau_b", "tau_i"]
            lb += [relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]
            ub += [relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]
            n_wx = 2
        else:
            names += ["tau_i"]
            lb += [relax_cfg.tau_i_lb]
            ub += [relax_cfg.tau_i_ub]
            n_wx = 1

    # --- free nuisances (T10, B1), multi-flip-angle only ----------------
    # Order and units come from the same helper assemble_prior uses, so
    # the layout and the prior cannot disagree. T10 is in SECONDS here.
    n_nuis = 0
    if nuisance is not None and (nuisance.fit_T10 or nuisance.fit_B1):
        if acq is None:
            raise ValueError(
                "free T10/B1 need acq so the identifiability guardrail can "
                "check the protocol (single flip angle cannot identify them)")
        from dce_mri.config import validate_nuisance_compatibility
        from dce_mri.bayes_priors import _nuisance_block_spec
        validate_nuisance_compatibility(nuisance, acq)
        nu_names, nu_lb, nu_ub = _nuisance_block_spec(nuisance)
        names += nu_names
        lb += list(nu_lb)
        ub += list(nu_ub)
        n_nuis = len(nu_names)

    n_model = n_kin + n_wx + n_nuis
    names += list(noise.param_names)

    return ParamLayout(
        model=model_use, param_names=tuple(names),
        lb=np.array(lb, float), ub=np.array(ub, float),
        n_kin=n_kin, n_wx=n_wx, n_noise=noise.n_params,
        noise_slice=slice(n_model, n_model + noise.n_params),
        n_nuis=n_nuis,
    )


# =============================================================================
# Log-posterior
# =============================================================================

def make_log_posterior(
    sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
    *, wx_model=None, aif_cfg=None, relax_cfg=None,
    layout=None, prior=None, sigma_fixed=None,
):
    """
    Returns (log_post, layout, prior, ctx).

    log_post(p_aug) -> float, evaluated as:
        1. prior.log_pdf  -- cheap; early -inf return means the
           expensive forward model is never called for an out-of-bounds
           proposal
        2. build_residual(..., on_failure='nan')
        3. noise.log_lik

    DELAY MODES
    -----------
    'marginalized' integrates tau out over cfg_bayes.delay_grid rather
    than sampling it  [Orton2007 Eq. 10]:

        log L(theta) = logsumexp_tau[ log L(theta, tau) ] - log(n_grid)

    which is the uniform-weighted average over the grid. Orton's Fig. 1
    shows the joint (theta, tau0) surface is genuinely bimodal at
    realistic noise with the WRONG minimum often winning; marginalising
    restores unimodality in the parameters of interest. The trade-off is
    that no per-voxel delay estimate is produced.
    """
    noise = NOISE_OPTIONS[cfg_bayes.noise]

    if layout is None:
        layout = build_param_layout(
            model, cfg_bayes, aif=aif, time_var=time_var, acq=acq,
            wx_model=wx_model, relax_cfg=relax_cfg)

    if prior is None:
        # noise.prior_key names which prior SHAPE this noise model implies
        # (e.g. GAUSS_FREE_INVGAMMA -> "invgamma"). Without this, choosing
        # that noise model silently built a Jeffreys-shaped prior instead,
        # since assemble_prior's own default is "jeffreys" regardless of
        # which noise model was selected. An explicit prior_choices["noise"]
        # still wins, so this only fills in what wasn't already set.
        prior_choices = dict(cfg_bayes.prior_choices)
        if noise.prior_key is not None and "noise" not in prior_choices:
            prior_choices["noise"] = noise.prior_key

        prior = assemble_prior(
            layout.model, relax_cfg, wx_model=wx_model,
            delay_mode=cfg_bayes.delay_mode,
            n_noise=noise.n_params, noise_names=noise.param_names,
            choices=prior_choices,
            delay_bounds=cfg_bayes.delay_bounds,
        )
        prior.validate_against(layout.param_names)

    ctx = {}
    if noise.n_params == 0 and not noise.marginalized:
        if sigma_fixed is None:
            raise ValueError(
                f"noise={noise.name!r} needs a fixed sigma; pass "
                f"cfg_bayes.sigma_fixed or use estimate_sigma_from_fit()."
            )
        ctx["sigma"] = float(sigma_fixed)

    from dce_mri.signal_models import shift_aif

    # --- build the residual evaluator(s) -----------------------------
    if cfg_bayes.delay_mode == "marginalized":
        grid = cfg_bayes.delay_grid
        if grid is None:
            raise ValueError("delay_mode='marginalized' requires delay_grid")
        bundles = []
        for tau in np.asarray(grid, float):
            aif_s = shift_aif(aif, float(tau), time_var, acq)
            bundles.append(build_residual(
                sig, aif_s, time_var, layout.model, acq, subject, cfg_fit,
                wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
                on_failure="nan"))
        n_grid = len(bundles)

        def _loglik(p_model, p_noise):
            terms = np.empty(n_grid)
            for k, b in enumerate(bundles):
                r = b.residual_fn(p_model)
                if not np.all(np.isfinite(r)):
                    terms[k] = -np.inf
                    continue
                terms[k] = noise.log_lik(r, r + b.target, p_noise, ctx)
            if not np.any(np.isfinite(terms)):
                return -np.inf
            return float(logsumexp(terms) - np.log(n_grid))

    else:
        aif_use = aif
        if cfg_bayes.delay_mode == "fixed":
            aif_use = shift_aif(aif, float(cfg_bayes.delay_fixed), time_var, acq)
        bundle = build_residual(
            sig, aif_use, time_var, layout.model, acq, subject, cfg_fit,
            wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
            on_failure="nan")

        def _loglik(p_model, p_noise):
            r = bundle.residual_fn(p_model)
            if not np.all(np.isfinite(r)):
                return -np.inf
            return noise.log_lik(r, r + bundle.target, p_noise, ctx)

    n_model = layout.n_model

    def log_post(p_aug):
        p_aug = np.asarray(p_aug, float)
        lp = prior.log_pdf(p_aug)
        if not np.isfinite(lp):
            return -np.inf
        ll = _loglik(p_aug[:n_model], p_aug[layout.noise_slice])
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)

    return log_post, layout, prior, ctx


def estimate_sigma_from_fit(result, dof: Optional[int] = None) -> float:
    """
    Residual sd from an existing NLLS fit -- the sigma for
    GAUSS_FIXED_SIGMA. dof defaults to N - n_params if supplied.
    """
    resid = np.asarray(result.S_meas) - np.asarray(result.S_pred)
    n = resid.size
    d = n if dof is None else max(n - dof, 1)
    return float(np.sqrt(np.sum(resid ** 2) / d))


# =============================================================================
# Shared diagnostics helper (used by all three backends)
# =============================================================================

def _rhat_ess_mcse(idata, param_names):
    """
    R-hat, ESS, and MCSE (of the mean AND of the median) via ArviZ, for
    whichever param_names are actually present in idata.posterior.

    mcse_median is the more relevant number day-to-day in this project --
    point estimates are reported as medians (BC1/BC3 convention, robust
    to the skew tau_i/tau_b posteriors can have), so that is the MCSE
    that describes the precision of the number actually being reported.
    mcse_mean is the textbook-default quantity, kept alongside it.
    """
    import arviz as az
    present = [nm for nm in param_names if nm in idata.posterior]
    rhat = {nm: float(np.asarray(az.rhat(idata, var_names=[nm])[nm])) for nm in present}
    ess = {nm: float(np.asarray(az.ess(idata, var_names=[nm])[nm])) for nm in present}
    mcse_mean = {nm: float(np.asarray(az.mcse(idata, var_names=[nm], method="mean")[nm]))
                for nm in present}
    mcse_median = {nm: float(np.asarray(
                    az.mcse(idata, var_names=[nm], method="quantile", prob=0.5)[nm]))
                  for nm in present}
    return rhat, ess, mcse_mean, mcse_median


# =============================================================================
# PyMC backend (PRIMARY)
# =============================================================================

def _make_pymc_op(log_post, n_params):
    """
    Wrap the numpy log-posterior in a PyTensor Op -- NO gradient.

    The forward model uses scipy expm / FFT convolution, which PyTensor
    cannot differentiate through, so no grad() is defined and only
    gradient-free steppers may be used. This is the standard black-box
    pattern from PyMC's own ODE case study.
    """
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
    from pytensor.graph.basic import Apply

    class _LogPostOp(Op):
        itypes = [pt.dvector]
        otypes = [pt.dscalar]

        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            return Apply(self, [x], [pt.dscalar()])

        def perform(self, node, inputs, outputs):
            (theta,) = inputs
            val = log_post(np.asarray(theta, dtype=float))
            if not np.isfinite(val):
                val = -1e30      # PyMC needs a float, not -inf
            outputs[0][0] = np.array(float(val))

    return _LogPostOp()


def _make_pymc_op_with_grad(log_post, n_params, eps=1e-5):
    """
    Same black-box wrapping, but WITH a gradient -- via central finite
    differences -- so NUTS/HMC can be used.

    COST WARNING, read before reaching for step='nuts':
    each gradient evaluation costs 2*n_params calls to log_post (central
    differences: one +eps and one -eps per dimension). NUTS/HMC need a
    gradient at EVERY leapfrog step within a trajectory, and a single
    NUTS draw is typically many leapfrog steps (often a few dozen,
    sometimes more, especially early in adaptation). So one posterior
    DRAW here can cost on the order of  (leapfrog steps) x 2 x n_params
    calls to the forward model, versus ~1 call per draw for
    DEMetropolisZ. For D=6-7 this is routinely 100-300x more forward-
    model evaluations per draw. Measure it on one row before switching
    a whole verification sweep over -- see the cost comparison this was
    built to produce.

    The pattern is PyMC's own documented recipe for a black-box
    likelihood with a numerical gradient (two Ops: the log-posterior
    itself, and a second Op computing its gradient, wired together via
    .grad()).
    """
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
    from pytensor.graph.basic import Apply

    class _LogPostGradOp(Op):
        """Returns d(log_post)/d(theta), one call = 2*n_params forward evals."""
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            (theta,) = inputs
            theta = np.asarray(theta, dtype=float)
            grad = np.empty(n_params)
            for i in range(n_params):
                tp, tm = theta.copy(), theta.copy()
                tp[i] += eps
                tm[i] -= eps
                fp = log_post(tp)
                fm = log_post(tm)
                fp = fp if np.isfinite(fp) else -1e30
                fm = fm if np.isfinite(fm) else -1e30
                grad[i] = (fp - fm) / (2.0 * eps)
            outputs[0][0] = grad

    class _LogPostOpGrad(Op):
        itypes = [pt.dvector]
        otypes = [pt.dscalar]

        def __init__(self):
            self._grad_op = _LogPostGradOp()

        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            return Apply(self, [x], [pt.dscalar()])

        def perform(self, node, inputs, outputs):
            (theta,) = inputs
            val = log_post(np.asarray(theta, dtype=float))
            if not np.isfinite(val):
                val = -1e30
            outputs[0][0] = np.array(float(val))

        def grad(self, inputs, output_grads):
            (theta,) = inputs
            (g,) = output_grads
            return [g * self._grad_op(theta)]

    return _LogPostOpGrad()


def _sample_pymc(log_post, layout, prior, cfg_bayes, init_point=None):
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    step_name = cfg_bayes.step.lower()
    GRADIENT_STEPS = {"nuts", "hmc"}
    needs_grad = step_name in GRADIENT_STEPS
    op = (_make_pymc_op_with_grad(log_post, layout.n_total)
          if needs_grad else _make_pymc_op(log_post, layout.n_total))

    with pm.Model() as pmodel:
        rvs = []
        for blk in sorted(prior.blocks, key=lambda b: min(b.indices)):
            rvs.extend(blk.build_pymc(blk.param_names, pmodel))
        theta = pt.stack(rvs)
        pm.Potential("loglik", op(theta))

        # 0.234 (BayesConfig's default) is the Roberts/Gelman/Gilks
        # optimum for RANDOM-WALK Metropolis -- it does not apply to
        # NUTS/HMC, whose own well-established target is ~0.8 (Betancourt
        # 2017; Neal 2011). Silently reusing 0.234 for NUTS would tune it
        # toward far-too-small step sizes. Only override if the config is
        # still sitting on the random-walk default -- an explicit,
        # deliberately-chosen target_accept is always respected.
        target_accept = cfg_bayes.target_accept
        if needs_grad and target_accept == 0.234:
            target_accept = 0.8

        if step_name == "demetropolisz":
            step = pm.DEMetropolisZ(tune_drop_fraction=0.9)
        elif step_name == "demetropolis":
            step = pm.DEMetropolis()
        elif step_name == "metropolis":
            step = pm.Metropolis()
        elif step_name == "slice":
            step = pm.Slice()
        elif step_name == "nuts":
            step = pm.NUTS(target_accept=target_accept)
        elif step_name == "hmc":
            step = pm.HamiltonianMC(target_accept=target_accept)
        else:
            raise ValueError(f"unknown PyMC step {cfg_bayes.step!r}")

        # Per-chain DISPERSED starts. PyMC accepts a list of dicts, one
        # per chain; a single dict would start every chain at the same
        # point, which makes R-hat meaningless (it compares between- to
        # within-chain variance) and starves DEMetropolisZ of the
        # spread its history-based proposals rely on. Dispersed starts
        # are what Sitek2016 and Zhou2013 both use to assess convergence.
        initvals = None
        if init_point is not None:
            rng_i = np.random.default_rng(cfg_bayes.seed)
            initvals = [
                _pymc_initvals(prior, layout, init_point,
                               rng=(None if c == 0 else rng_i))
                for c in range(cfg_bayes.n_chains)
            ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=cfg_bayes.n_draws,
                tune=cfg_bayes.n_tune,
                chains=cfg_bayes.n_chains,
                cores=1,                      # black-box Op is not fork-safe
                step=step,
                initvals=initvals,
                random_seed=cfg_bayes.seed,
                progressbar=cfg_bayes.progressbar,
                compute_convergence_checks=False,
            )

    # --- pull the DETERMINISTIC named params in p_aug order ----------
    post = idata.posterior
    cols = []
    for nm in layout.param_names:
        if nm in post:
            cols.append(np.asarray(post[nm].values))
        else:
            raise KeyError(
                f"parameter {nm!r} missing from PyMC posterior; available: "
                f"{list(post.data_vars)}"
            )
    chains = np.stack(cols, axis=-1)          # (chain, draw, D)

    rhat, ess, mcse_mean, mcse_median = _rhat_ess_mcse(idata, layout.param_names)

    n_div = None
    if "diverging" in idata.sample_stats:
        n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())

    return chains, rhat, ess, mcse_mean, mcse_median, n_div, idata


def _pymc_initvals(prior, layout, init_point, rng=None, jitter=0.15):
    """
    Translate a p_aug point estimate into a PyMC initval dict, honouring
    the reparameterisations the prior blocks introduce (log_x for
    log-uniform, x_raw for rescaled Beta, and so on).

    rng not None -> apply multiplicative jitter and pull the value
    strictly INSIDE its bounds. The pull-inside matters: an NLLS anchor
    that railed against a bound (tau_b and a delay-mismatched ve both do
    this routinely) would otherwise start a chain exactly at the edge,
    where a log-uniform or Beta prior has zero or near-zero density.
    """
    iv = {}
    for blk in prior.blocks:
        for k, idx in enumerate(blk.indices):
            nm = blk.param_names[k]
            val = float(init_point[idx])
            if rng is not None:
                val = val * (1.0 + jitter * rng.standard_normal())
            if idx < len(layout.lb):
                lo, hi = float(layout.lb[idx]), float(layout.ub[idx])
                pad = 0.02 * (hi - lo)
                val = float(np.clip(val, lo + pad, hi - pad))
            elif val <= 0:
                val = abs(val) + 1e-6
            if blk.name in ("rates", "tau") and blk.source in (
                    "log_dims convention", "proposal (flat)"):
                iv[f"log_{nm}"] = float(np.log(max(val, 1e-12)))
            elif blk.name == "volfrac" and blk.source == "Mittermeier2019":
                lo, hi = layout.lb[idx], layout.ub[idx]
                iv[f"{nm}_raw"] = float(np.clip((val - lo) / (hi - lo), 1e-4, 1 - 1e-4))
            elif blk.name == "noise" and blk.source == "Orton2007":
                iv[f"log_{nm}"] = float(np.log(max(val, 1e-12)))
            else:
                iv[nm] = val
    return iv


# =============================================================================
# emcee backend (fallback / cross-check)
# =============================================================================

def _sample_emcee(log_post, layout, prior, cfg_bayes, init_point=None):
    import emcee

    D = layout.n_total
    n_walkers = cfg_bayes.n_walkers or max(2 * D + 2, 16)
    rng = np.random.default_rng(cfg_bayes.seed)

    p0 = prior.sample(rng, n_walkers)
    if init_point is not None:
        jitter = 1.0 + 0.05 * rng.standard_normal((n_walkers, D))
        cand = np.asarray(init_point, float)[None, :] * jitter
        ok = np.array([np.isfinite(prior.log_pdf(c)) for c in cand])
        p0[ok] = cand[ok]

    sampler = emcee.EnsembleSampler(n_walkers, D, log_post)
    sampler.run_mcmc(p0, cfg_bayes.n_tune + cfg_bayes.n_draws, progress=False)
    ch = sampler.get_chain(discard=cfg_bayes.n_tune)      # (draw, walker, D)
    chains = np.transpose(ch, (1, 0, 2))                  # (walker, draw, D)

    rhat, ess, mcse_mean, mcse_median = {}, {}, {}, {}
    try:
        import arviz as az
        idata = az.from_dict(
            posterior={nm: chains[..., j]
                       for j, nm in enumerate(layout.param_names)})
        rhat, ess, mcse_mean, mcse_median = _rhat_ess_mcse(idata, layout.param_names)
    except Exception:
        pass

    return chains, rhat, ess, mcse_mean, mcse_median, float(np.mean(sampler.acceptance_fraction))


# =============================================================================
# mh_reflect backend -- random-walk Metropolis with bound reflection
# =============================================================================

def _sample_mh_reflect(log_post, layout, prior, cfg_bayes, init_point=None):
    """
    Hand-rolled random-walk Metropolis with REFLECTION at bounds.

    [SOURCE: Sitek2016 Appendix A]  If a proposal lands outside
    [a, b], fold it back:  theta_new = 2a - theta_prop (or 2b - ...).
    This preserves detailed balance and keeps the chain moving when a
    parameter sits hard against a bound -- which matters here because
    tau_b rails, and under plain rejection a pinned chain wastes most of
    its proposals.

    Step sizes are tuned during burn-in toward cfg_bayes.target_accept
    (0.234)  [Roberts/Gelman/Gilks 1997].
    """
    D = layout.n_total
    lb = np.concatenate([layout.lb, np.full(layout.n_noise, -np.inf)])
    ub = np.concatenate([layout.ub, np.full(layout.n_noise, np.inf)])
    rng = np.random.default_rng(cfg_bayes.seed)

    all_chains, accepts = [], []
    for c in range(cfg_bayes.n_chains):
        x = (np.asarray(init_point, float) if init_point is not None
             else prior.sample(rng, 1)[0])
        if init_point is not None and c > 0:
            x = x * (1.0 + 0.05 * rng.standard_normal(D))
        lpx = log_post(x)
        tries = 0
        while not np.isfinite(lpx) and tries < 500:
            x = prior.sample(rng, 1)[0]
            lpx = log_post(x)
            tries += 1

        step = np.where(np.isfinite(ub - lb), (ub - lb) / 20.0,
                        np.maximum(np.abs(x), 1e-3) / 10.0)
        kept, n_acc, n_tot = [], 0, 0

        for it in range(cfg_bayes.n_tune + cfg_bayes.n_draws):
            prop = x + step * rng.standard_normal(D)
            # reflection at finite bounds
            for d in range(D):
                if np.isfinite(lb[d]) and prop[d] < lb[d]:
                    prop[d] = 2 * lb[d] - prop[d]
                if np.isfinite(ub[d]) and prop[d] > ub[d]:
                    prop[d] = 2 * ub[d] - prop[d]
                prop[d] = min(max(prop[d], lb[d]), ub[d])

            lpp = log_post(prop)
            n_tot += 1
            if np.isfinite(lpp) and np.log(rng.uniform()) < (lpp - lpx):
                x, lpx = prop, lpp
                n_acc += 1

            if it < cfg_bayes.n_tune:
                if (it + 1) % 100 == 0:
                    rate = n_acc / max(n_tot, 1)
                    if rate < cfg_bayes.target_accept * 0.85:
                        step *= 0.9
                    elif rate > cfg_bayes.target_accept * 1.15:
                        step *= 1.1
                    n_acc, n_tot = 0, 0
            else:
                kept.append(x.copy())

        all_chains.append(np.array(kept))
        accepts.append(n_acc / max(n_tot, 1))

    chains = np.stack(all_chains, axis=0)
    rhat, ess, mcse_mean, mcse_median = {}, {}, {}, {}
    try:
        import arviz as az
        idata = az.from_dict(posterior={nm: chains[..., j]
                                        for j, nm in enumerate(layout.param_names)})
        rhat, ess, mcse_mean, mcse_median = _rhat_ess_mcse(idata, layout.param_names)
    except Exception:
        pass
    return chains, rhat, ess, mcse_mean, mcse_median, float(np.mean(accepts))


# =============================================================================
# Public entry point
# =============================================================================

def fit_curve_bayes(
    sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
    *, wx_model=None, aif_cfg=None, relax_cfg=None,
    point_fit=None, seed=None,
) -> BayesResult:
    """
    Posterior estimation for one curve. Mirrors bootstrap_curve's
    signature so the two can be swapped in existing scripts.

    Covers, by configuration alone:
        TCXM / GKM,  +/- delay (joint | marginalized | fixed),
        +/- water exchange (tau_i only, or tau_b + tau_i)
    """
    t0 = time.time()
    _validate_wx_compatibility(wx_model, cfg_fit.fit_space, model,
                               time_var, aif, aif_cfg, relax_cfg)
    if seed is not None:
        cfg_bayes = BayesConfig(**{**cfg_bayes.__dict__, "seed": seed})

    noise = NOISE_OPTIONS[cfg_bayes.noise]

    layout = build_param_layout(
        model, cfg_bayes, aif=aif, time_var=time_var, acq=acq,
        wx_model=wx_model, relax_cfg=relax_cfg)

    # --- NLLS anchor (for sigma, init, and comparison) ---------------
    sigma_fixed = cfg_bayes.sigma_fixed
    init_point = None
    if point_fit is None and (cfg_bayes.init_from_nlls or
                              (noise.n_params == 0 and not noise.marginalized
                               and sigma_fixed is None)):
        from dce_mri.signal_models import shift_aif
        aif_pf = aif
        if cfg_bayes.delay_mode == "fixed":
            aif_pf = shift_aif(aif, float(cfg_bayes.delay_fixed), time_var, acq)
        elif cfg_bayes.delay_mode == "marginalized":
            # The data may carry a real delay that the model integrates
            # out. Anchoring on the UNSHIFTED aif would fit delayed data
            # with a zero-delay model, which rails parameters against
            # their bounds and hands every chain the same garbage start.
            # Use the best-cost tau on the grid instead -- i.e. exactly
            # what fit_with_delay_grid does for its point estimate.
            best_cost, best_pf = np.inf, None
            for tau in np.asarray(cfg_bayes.delay_grid, float):
                pf_t = fit_single_voxel(
                    sig, shift_aif(aif, float(tau), time_var, acq), time_var,
                    layout.model, acq, subject, cfg_fit, seed=cfg_bayes.seed,
                    wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
                if pf_t.cost < best_cost:
                    best_cost, best_pf = pf_t.cost, pf_t
            point_fit = best_pf
        if point_fit is None:
            point_fit = fit_single_voxel(
                sig, aif_pf, time_var, layout.model, acq, subject, cfg_fit,
                seed=cfg_bayes.seed, wx_model=wx_model,
                aif_cfg=aif_cfg, relax_cfg=relax_cfg)

    if point_fit is not None:
        pe = np.asarray(point_fit.params, float)
        if point_fit.wx_params is not None:
            pe = np.concatenate([pe, np.asarray(point_fit.wx_params, float)])
        if sigma_fixed is None:
            sigma_fixed = estimate_sigma_from_fit(point_fit, dof=layout.n_model)
        if cfg_bayes.init_from_nlls:
            init_point = np.concatenate([
                pe, np.array([sigma_fixed] * layout.n_noise)[:layout.n_noise]
            ]) if layout.n_noise else pe
            if layout.n_noise == 2:      # student-t: (sigma, inv_nu)
                init_point = np.concatenate([pe, [sigma_fixed, 0.2]])
    else:
        pe = None

    log_post, layout, prior, _ = make_log_posterior(
        sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
        wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        layout=layout, sigma_fixed=sigma_fixed)

    if init_point is not None and not np.isfinite(prior.log_pdf(init_point)):
        init_point = None       # NLLS optimum sat on a bound; let prior init

    backend = cfg_bayes.backend.lower()
    accept, n_div = None, None
    if backend == "pymc":
        chains, rhat, ess, mcse_mean, mcse_median, n_div, idata = _sample_pymc(
            log_post, layout, prior, cfg_bayes, init_point)
        meta = {"idata": idata}
    elif backend == "emcee":
        chains, rhat, ess, mcse_mean, mcse_median, accept = _sample_emcee(
            log_post, layout, prior, cfg_bayes, init_point)
        meta = {}
    elif backend == "mh_reflect":
        chains, rhat, ess, mcse_mean, mcse_median, accept = _sample_mh_reflect(
            log_post, layout, prior, cfg_bayes, init_point)
        meta = {}
    elif backend == "blackjax":
        if cfg_bayes.delay_mode == "marginalized":
            raise NotImplementedError(
                "backend='blackjax' does not handle delay_mode='marginalized' "
                "-- the logsumexp-over-grid likelihood is not wired through "
                "make_log_lik_jax. Use backend='pymc' for that mode."
            )

        from dce_mri.signal_models import compute_S0, shift_aif
        from dce_mri.jax_forward import (static_ctx_from_configs,
                                         make_predict, make_residual)
        from dce_mri.bayes_blackjax import sample_blackjax, make_log_lik_jax

        ctx = static_ctx_from_configs(acq, subject, aif_cfg)
        S0  = float(compute_S0(sig[None, :], acq)[0])
        
        aif_use = aif
        if cfg_bayes.delay_mode == "fixed":
            aif_use = shift_aif(aif, float(cfg_bayes.delay_fixed), time_var, acq)

        # wx models with free taus need relax_cfg; NXL (n_params=0) does not,
        # and relax_cfg may legitimately be None for it.
        has_tau = wx_model is not None and wx_model.n_params > 0
        predict = make_predict(
            model.name, time_var, aif_use, ctx, S0,
            wx          = wx_model is not None,
            wx_name     = wx_model.name if wx_model is not None else "SSM3",
            fit_tau_b   = relax_cfg.fit_tau_b if has_tau else False,
            tau_b_fixed = relax_cfg.tau_b_fixed if has_tau else 0.0,
            delay        = (cfg_bayes.delay_mode == "joint"),
            baseline_val = float(np.mean(np.asarray(aif_use)[acq.nbase])),
            upsample    = acq.upsample,
        )
        residual    = make_residual(predict, sig)
        
        _NOISE_MAP = {
            "gauss_marginalized_sigma": "marginalized",
            "gauss_free_jeffreys":      "free",
            "gauss_free_invgamma":      "free",
            "gauss_free_lognormal":     "free",
        }
        if cfg_bayes.noise not in _NOISE_MAP:
            raise NotImplementedError(
                f"backend='blackjax' does not yet support noise={cfg_bayes.noise!r} "
                "-- only the marginalized-sigma and free-Gaussian-sigma noise "
                "models are wired through make_log_lik_jax. Use backend='pymc' "
                "for gauss_signal_scaled, student_t, or gauss_fixed_sigma."
            )
        log_lik_jax = make_log_lik_jax(residual, noise=_NOISE_MAP[cfg_bayes.noise])


        chains, rhat, ess, mcse_mean, mcse_median, n_div, bj_info = \
            sample_blackjax(log_lik_jax, layout, prior, cfg_bayes, init_point)
        meta = {"blackjax": bj_info}
    else:
        raise ValueError(f"unknown backend {cfg_bayes.backend!r}")

    meta.update(dict(noise=noise.name, delay_mode=cfg_bayes.delay_mode,
                     prior_sources=prior.sources,
                     prior_is_flat=prior.is_flat,
                     sigma_fixed=sigma_fixed))

    return BayesResult(
        param_names=layout.param_names, chains=chains, point_estimate=pe,
        rhat=rhat, ess=ess, mcse_mean=mcse_mean, mcse_median=mcse_median,
        accept_frac=accept, n_divergences=n_div, backend=backend,
        elapsed_s=time.time() - t0, meta=meta)


def fit_curve_bayes_delay_grid(
    sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
    tau_values, *, wx_model=None, aif_cfg=None, relax_cfg=None, **kw
) -> BayesResult:
    """
    Convenience wrapper for Orton-style delay marginalisation over an
    explicit tau grid -- the Bayesian counterpart of
    fit_with_delay_grid, which instead takes the single best-cost tau.

        fit_with_delay_grid   : argmin over tau        (point estimate)
        this function         : logsumexp over tau     (integrate out)
    """
    cfg2 = BayesConfig(**{**cfg_bayes.__dict__,
                          "delay_mode": "marginalized",
                          "delay_grid": np.asarray(tau_values, float)})
    return fit_curve_bayes(sig, aif, time_var, model, acq, subject,
                           cfg_fit, cfg2, wx_model=wx_model,
                           aif_cfg=aif_cfg, relax_cfg=relax_cfg, **kw)


# =============================================================================
# Identifiability read-out
# =============================================================================

def posterior_vs_prior_width(result: BayesResult, prior: PriorSpec,
                             rng=None, n=20000) -> dict:
    """
    Ratio of posterior IQR to prior IQR, per parameter.

    The direct read on how much the data actually constrained each
    parameter. Ratio near 1 means the posterior collapsed back onto the
    prior -- the honest signature of a non-identifiable parameter
    (expected for tau_b). Small ratio means the data genuinely
    constrained it.

    This is the identifiability test; SBC is NOT (a correctly
    implemented sampler on a non-identifiable parameter returns ~the
    prior and PASSES calibration).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    draws = prior.sample(rng, n)
    out = {}
    for j, nm in enumerate(result.param_names):
        pri = np.subtract(*np.percentile(draws[:, j], [75, 25]))
        pos = result.iqr(nm)
        out[nm] = dict(prior_iqr=float(pri), posterior_iqr=float(pos),
                       ratio=float(pos / pri) if pri > 0 else np.nan)
    return out