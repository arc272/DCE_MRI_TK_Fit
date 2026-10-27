from __future__ import annotations
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bayes_blackjax.py
=================
STAGE 2 -- BlackJAX NUTS backend for fit_curve_bayes.

Why this module exists separately from bayes.py
-----------------------------------------------
The three existing backends (pymc / emcee / mh_reflect) all consume the
NUMPY log-posterior built by bayes.make_log_posterior, and all of them
either (a) need no gradient, or (b) get one by finite differences
through a black-box PyTensor Op. BlackJAX needs a JAX-traceable
log-density it can jit/vmap/grad, which is a different object entirely.

Keeping it in its own module means bayes.py has no import-time JAX
dependency -- the existing backends keep working on a machine with no
JAX installed.

CONFIRMED (against the real jax_forward.py, 2026-09-17): its
make_log_posterior returns `log_prior(p_aug) + log_lik(p_aug)` fused
into one scalar -- the prior argument is not separable from the
returned closure. Do NOT pass that function's output as this module's
log_lik_jax: sample_blackjax adds the prior itself (via the box
transform), so a fused log_post would double the prior's weight
silently -- no error, no NaN, just a systematically over-narrow
posterior. Use make_log_lik_jax() below instead, which reproduces
make_log_posterior's noise math with no prior term, verified against
it directly in test_blackjax_transform.py.

THE CENTRAL ISSUE: CONSTRAINED vs UNCONSTRAINED SPACE
-----------------------------------------------------
PyMC silently does something for you that BlackJAX does not.

Every block in the shipped priors is a BOX:

    volfrac  make_volfrac_flat_masked  -> Uniform     [lb, ub]
    rates    make_rate_loguniform      -> LogUniform  [lb, ub]   (default)
    rates    make_rate_uniform         -> Uniform     [lb, ub]
    tau      make_tau_flat             -> LogUniform  [lb, ub]   (default)
    tau      make_tau_uniform          -> Uniform     [lb, ub]

PriorSpec.log_pdf returns -inf outside the box. pm.Uniform applies an
interval transform internally and samples an unconstrained variable;
make_rate_loguniform even does it explicitly (it declares
pm.Uniform("log_fp", ...) then exp()s it back as a Deterministic).

BlackJAX applies NO transform. Hand it a box-bounded log-density and
the leapfrog integrator will step outside the box on its first large
step, evaluate -inf, and produce NaN gradients. The chain does not mix
badly -- it dies.

So this module samples z in R^D and maps to the constrained x:

    s   = scale(x)             identity for uniform, log for log-uniform
    w   = (s - s_lb)/(s_ub - s_lb)          in (0,1)
    z   = logit(w)                          in R

and the target density in z is

    log p(z) = log_post(x(z)) + log|dx/dz|

with the Jacobian for one coordinate

    log|dx/dz| = log(s_ub - s_lb) + log w + log(1-w) + [ s  if log-scale ]

The last bracketed term is d x/d s = exp(s) = x for the log-scale case,
0 (i.e. factor 1) for the linear case.

WORTH KNOWING: for a prior that is FLAT on the s scale -- which every
block listed above is -- the prior's normalising constant and the
scale terms of the Jacobian cancel exactly, leaving

    log p(z) = log_lik(x(z)) + sum_j [ log w_j + log(1 - w_j) ]

This is verified numerically in test_blackjax_transform.py rather than
asserted. The general path (prior log-pdf + full Jacobian, computed
separately) is what is implemented below, so that a future non-flat
prior block still works; the cancellation is a property of the current
priors, not an assumption baked into the code.

WHAT IS NOT HANDLED
-------------------
Only the flat families are given JAX log-densities here. beta,
truncnorm, gamma, conditional, ordered, informed, and the sigma /
student-t blocks raise NotImplementedError with the block name. That is
deliberate: a wrong-but-silent prior is far worse than a loud refusal,
and ROW 4 needs none of them. gauss_marginalized_sigma has n_noise = 0,
so there is no noise block to transform.

The ve+vp<=1 feasibility mask is NOT applied in the transform. Per the
note in make_volfrac_flat_masked, with the shipped bounds (ve, vp both
capped at 0.5) the maximum achievable sum is exactly 1.0, so the
constraint binds only at a single corner point and is inert. If those
bounds are ever widened, this needs revisiting -- the box transform
cannot represent a simplex constraint.
"""


import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Per-block transform
# =============================================================================

#: prior-block factories whose log-density is flat on the linear scale
_LINEAR_FLAT_SOURCES = {
    ("volfrac", "current NLLS behaviour"),
    ("rates",   "Sitek2016/Zhou2013"),
    ("delay",   "Orton2007"),
}
#: prior-block factories whose log-density is flat on the LOG scale
_LOG_FLAT_SOURCES = {
    ("rates",    "log_dims convention"),
    ("tau",      "proposal (flat)"),
    ("nuisance", "flat log (NuisanceConfig default)"),   # T10, B1 (Phase 5.2)
}

_BETA_SOURCES = {("volfrac", "Mittermeier2019")}


@dataclass(frozen=True)
class _BlockTransform:
    """
    Bijection R^k <-> box, for one ParamBlockPrior.

    indices  : positions in p_aug this block covers (same as the block)
    log_scale: True  -> x = exp(s), box is on log(x)
               False -> x = s,      box is on x
    s_lb/s_ub: box bounds already on the s scale
    """
    indices:   tuple
    log_scale: bool
    s_lb:      jnp.ndarray
    s_ub:      jnp.ndarray
    beta_ab:   Optional[tuple] = None    # (a, b) for a Beta block; None = flat

    # -- z (unconstrained) -> x (constrained) --------------------------
    def constrain(self, z):
        w = jax.nn.sigmoid(z)
        s = self.s_lb + w * (self.s_ub - self.s_lb)
        return jnp.exp(s) if self.log_scale else s

    # -- x -> z ---------------------------------------------------------
    def unconstrain(self, x):
        s = jnp.log(x) if self.log_scale else x
        w = (s - self.s_lb) / (self.s_ub - self.s_lb)
        w = jnp.clip(w, 1e-12, 1.0 - 1e-12)
        return jnp.log(w) - jnp.log1p(-w)

    # -- log |dx/dz|, summed over the block -----------------------------
    def log_det_jacobian(self, z):
        w = jax.nn.sigmoid(z)
        # log w + log(1-w) written via log_sigmoid for stability in the tails
        log_w   = jax.nn.log_sigmoid(z)
        log_1mw = jax.nn.log_sigmoid(-z)
        ld = jnp.log(self.s_ub - self.s_lb) + log_w + log_1mw
        if self.log_scale:
            # d exp(s)/d s = exp(s) = x
            s = self.s_lb + w * (self.s_ub - self.s_lb)
            ld = ld + s
        return jnp.sum(ld)

    # -- log prior density in x, evaluated from z -----------------------
    def log_prior(self, z):
        """
        Flat on the s scale, so density in s is 1/(s_ub - s_lb).
        Converting back to x costs -log|dx/ds|:
            linear : |dx/ds| = 1      -> log p_x = -log(s_ub - s_lb)
            log    : |dx/ds| = x      -> log p_x = -log(s_ub-s_lb) - log x
        No support check is needed: the sigmoid makes x in-support by
        construction, which is the entire point of the transform.
        """
     
        width = jnp.log(self.s_ub - self.s_lb)
        if self.beta_ab is not None:
            # Beta(a,b) rescaled onto [lb,ub]; w == sigmoid(z) exactly, so
            #   log p_x = (a-1)log w + (b-1)log(1-w) - logB(a,b) - log(ub-lb)
            # Verified: integrates to 1, matches bayes_priors._beta_funcs
            # to 2e-12 with zero offset.
            from jax.scipy.special import betaln
            a, b = self.beta_ab
            return jnp.sum((a - 1.0) * jax.nn.log_sigmoid(z)
                           + (b - 1.0) * jax.nn.log_sigmoid(-z)
                           - betaln(a, b) - width)
        if self.log_scale:
            w = jax.nn.sigmoid(z)
            s = self.s_lb + w * (self.s_ub - self.s_lb)
            return jnp.sum(-width - s)
        return jnp.sum(-width)


@dataclass(frozen=True)
class PriorTransform:
    """Whole-vector transform, assembled from one _BlockTransform per block."""
    blocks:  tuple
    n_total: int

    def constrain(self, z):
        x = jnp.zeros(self.n_total, dtype=z.dtype)
        for b in self.blocks:
            idx = jnp.asarray(b.indices)
            x = x.at[idx].set(b.constrain(z[idx]))
        return x

    def unconstrain(self, x):
        x = jnp.asarray(x)
        z = jnp.zeros(self.n_total, dtype=x.dtype)
        for b in self.blocks:
            idx = jnp.asarray(b.indices)
            z = z.at[idx].set(b.unconstrain(x[idx]))
        return z

    def log_det_jacobian(self, z):
        return sum(b.log_det_jacobian(z[jnp.asarray(b.indices)])
                   for b in self.blocks)

    def log_prior(self, z):
        return sum(b.log_prior(z[jnp.asarray(b.indices)])
                   for b in self.blocks)


def build_transform(prior, n_total: int) -> PriorTransform:
    """
    Map a PriorSpec onto a PriorTransform.

    Dispatch is on (block.name, block.source) rather than on the block's
    log_pdf closure, because the closure is a plain Python function with
    no introspectable identity. The source tag is set by each
    make_*() factory and is unique per family, so it is a reliable
    discriminator -- and if a factory's tag is ever changed, this raises
    rather than silently picking the wrong transform.
    """
    blocks = []
    for blk in prior.blocks:
        key = (blk.name, blk.source)
        lb, ub = _block_bounds(blk)
        if key in _LOG_FLAT_SOURCES:
            if np.any(lb <= 0):
                raise ValueError(
                    f"block {blk.name!r} is log-scale but has lb <= 0: {lb}")
            blocks.append(_BlockTransform(
                tuple(blk.indices), True,
                jnp.asarray(np.log(lb)), jnp.asarray(np.log(ub))))
        elif key in _LINEAR_FLAT_SOURCES:
            blocks.append(_BlockTransform(
                tuple(blk.indices), False,
                jnp.asarray(lb), jnp.asarray(ub)))
        elif key in _BETA_SOURCES:
            a, b = _block_beta_ab(blk, lb, ub)
            blocks.append(_BlockTransform(
                tuple(blk.indices), False,
                jnp.asarray(lb), jnp.asarray(ub), beta_ab=(a, b)))
        else:
            raise NotImplementedError(
                f"no JAX transform for prior block {blk.name!r} "
                f"(source {blk.source!r}). Only the flat uniform / "
                f"log-uniform families are implemented. Use backend="
                f"'pymc' for this prior, or add the family here."
            )
    return PriorTransform(tuple(blocks), n_total)

def _block_bounds(b):
    """
    Recover (lb, ub) for a flat block by probing its own log_pdf.

    The bounds are captured in the block's closure and not exposed as
    attributes, so rather than duplicating them here -- where they could
    drift out of sync with bayes_priors.py -- they are recovered by
    bisection on the support boundary of the block's OWN log_pdf. This
    guarantees the transform's box is exactly the prior's box.
    """
    k = len(b.indices)
    lo = np.full(k, -np.inf)
    hi = np.full(k, np.inf)

    # a point known to be inside: blocks expose .sample
    rng = np.random.default_rng(0)
    inside = np.asarray(b.sample(rng, 1))[0]
    if not np.isfinite(b.log_pdf(inside)):
        raise RuntimeError(f"block {b.name!r}: sample() produced an "
                           f"out-of-support point; cannot probe bounds")

    for j in range(k):
        lo[j] = _bisect_edge(b, inside, j, direction=-1)
        hi[j] = _bisect_edge(b, inside, j, direction=+1)
    return lo, hi

def _block_beta_ab(blk, lb, ub, tol=1e-9):
    """
    Recover (a, b) from a Beta block's own log_pdf, same philosophy as
    _block_bounds: probe rather than duplicate, so the transform cannot
    drift out of sync with bayes_priors.py. Three points determine
    (a-1), (b-1) and the constant; held-out points then verify.
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    k = len(blk.indices)
    A, y = [], []
    for w in (0.2, 0.5, 0.8):
        A.append([k * np.log(w), k * np.log(1 - w), 1.0])
        y.append(blk.log_pdf(lb + w * (ub - lb)))
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    a, b, c = sol[0] + 1.0, sol[1] + 1.0, sol[2]
    for w in (0.11, 0.37, 0.63, 0.91):          # held-out verification
        pred = k * ((a - 1) * np.log(w) + (b - 1) * np.log(1 - w)) + c
        if abs(pred - blk.log_pdf(lb + w * (ub - lb))) > tol:
            raise RuntimeError(
                f"block {blk.name!r}: recovered Beta(a={a:.4g}, b={b:.4g}) does "
                f"not reproduce its own log_pdf -- is this really a Beta block?")
    return float(a), float(b)

def _bisect_edge(b, inside, j, direction, n_iter=200):
    """Bisect along coordinate j until the support edge is located."""
    good = float(inside[j])
    step = max(abs(good), 1.0)
    bad = good + direction * step
    probe = np.array(inside, dtype=float)

    # expand until outside support
    for _ in range(200):
        probe[j] = bad
        if not np.isfinite(b.log_pdf(probe)):
            break
        step *= 2.0
        bad = good + direction * step
    else:
        raise RuntimeError(f"block {b.name!r}: coordinate {j} appears "
                           f"unbounded in direction {direction}")

    for _ in range(n_iter):
        mid = 0.5 * (good + bad)
        probe[j] = mid
        if np.isfinite(b.log_pdf(probe)):
            good = mid
        else:
            bad = mid
    return 0.5 * (good + bad)


# =============================================================================
# Likelihood-only log-density (companion to jax_forward.make_log_posterior,
# with the prior term removed)
# =============================================================================

def make_log_lik_jax(residual: Callable, noise: str = "marginalized",
                     n_eff: Optional[float] = None) -> Callable:
    """
    `log_lik(p_aug) -> scalar`, jit/grad-ready -- the LIKELIHOOD half of
    jax_forward.make_log_posterior, with the `lp = log_prior(p_aug)` term
    and the final `lp + ll` fusion removed.

    Exists because make_log_posterior's prior and likelihood are fused
    inside one closure and not separable after construction -- there is
    no `log_lik` half to pull back out of it. This rebuilds that half
    directly from `residual` (jax_forward.make_residual's output), with
    the exact same noise math, so it stays numerically identical to the
    likelihood term already trusted inside make_log_posterior. That
    identity is checked directly, not assumed: see
    test_blackjax_transform.py, which compares this function's output to
    make_log_posterior(residual, log_prior=lambda p: 0.0) at random
    points, for both noise='marginalized' and noise='free'.

    _EPS / _TINY are imported from jax_forward itself rather than
    redefined here, so the two can never numerically drift apart if
    jax_forward's guards ever change.

    residual : jax_forward.make_residual's output, (p_aug,) -> (T,) array
    noise    : 'marginalized' | 'free' -- must match cfg_bayes.noise
    n_eff    : effective N for the marginalized form. With frame-drop
               weights this is sum(w), NOT len(target) -- see
               make_log_posterior's own docstring for why. Defaults to
               len(residual output) when None, matching that function.
    """
    from dce_mri.jax_forward import _EPS, _TINY
    from jax.scipy.special import gammaln
    LOG_2PI = float(jnp.log(2.0 * jnp.pi))

    def log_lik(p_aug):
        if noise == "marginalized":
            r = residual(p_aug)
            n = r.shape[0] if n_eff is None else n_eff
            chi2 = jnp.maximum(jnp.sum(r ** 2), _TINY)
            ll = (gammaln(0.5 * n) - 0.5 * n * jnp.log(0.5 * chi2)
                  - 0.5 * n * LOG_2PI)
        elif noise == "free":
            s = p_aug[-1]
            r = residual(p_aug[:-1])
            n = r.shape[0] if n_eff is None else n_eff
            s_safe = jnp.maximum(s, _EPS)
            ll = (-0.5 * n * (LOG_2PI + 2.0 * jnp.log(s_safe))
                  - 0.5 * jnp.sum(r ** 2) / (s_safe * s_safe))
            ll = jnp.where(s > 0, ll, -1e30)
        else:
            raise ValueError(f"unknown noise {noise!r}")
        return jnp.where(jnp.isfinite(ll), ll, -1e30)

    return log_lik


# =============================================================================
# Unconstrained log-density
# =============================================================================

def make_unconstrained_logdensity(log_lik_jax: Callable,
                                  transform: PriorTransform) -> Callable:
    """
    Compose the JAX log-likelihood with the transform.

    log_lik_jax : (x,) -> scalar, the JAX-traceable log-LIKELIHOOD ONLY
                  (no prior term) on the CONSTRAINED parameter vector.
                  Build it with make_log_lik_jax() below -- NOT with
                  jax_forward.make_log_posterior, which fuses in a
                  prior and would double-count it here. See the module
                  docstring for the confirmed behaviour.

    Returns f(z) -> scalar, suitable for blackjax.
    """
    def logdensity(z):
        x = transform.constrain(z)
        return (log_lik_jax(x)
                + transform.log_prior(z)
                + transform.log_det_jacobian(z))
    return logdensity


# =============================================================================
# Sampler
# =============================================================================

def sample_blackjax(log_lik_jax, layout, prior, cfg_bayes, init_point=None):
    """
    Signature deliberately mirrors _sample_pymc / _sample_emcee /
    _sample_mh_reflect in bayes.py, so the dispatch block in
    fit_curve_bayes gains one more `elif` and nothing else changes.

    Returns
    -------
    chains       : (n_chains, n_draws, D) in CONSTRAINED space
    rhat, ess, mcse_mean, mcse_median : dicts keyed by param name
    n_divergences: int
    info         : dict (step size, inverse mass matrix, etc.)
    """
    import blackjax

    D = layout.n_total
    transform = build_transform(prior, D)
    logdensity = make_unconstrained_logdensity(log_lik_jax, transform)

    key = jax.random.PRNGKey(cfg_bayes.seed)
    n_chains = cfg_bayes.n_chains

    # ---- initial points, in unconstrained space ----------------------
    # Dispersed starts, same rationale as _pymc_initvals: R-hat compares
    # between-chain to within-chain variance, so identical starts make it
    # meaningless.
    key, k_init = jax.random.split(key)
    if init_point is not None and np.all(np.isfinite(init_point)):
        z0 = transform.unconstrain(jnp.asarray(init_point, dtype=jnp.float64))
        jitter = jax.random.normal(k_init, (n_chains, D)) * 0.5
        z_init = z0[None, :] + jitter
        z_init = z_init.at[0].set(z0)          # chain 0 exactly at the NLLS point
    else:
        # prior draws, pushed through to z
        rng = np.random.default_rng(cfg_bayes.seed)
        x0 = prior.sample(rng, n_chains)
        z_init = jnp.stack([transform.unconstrain(jnp.asarray(x, dtype=jnp.float64))
                            for x in x0])

    # ---- window adaptation: diagonal vs. full mass matrix, toggleable -
    # cfg_bayes.dense_mass_matrix (default False = diagonal) controls
    # this. is_mass_matrix_diagonal=False is the BlackJAX equivalent of
    # PyMC's init="jitter+adapt_full": a diagonal matrix rescales each
    # coordinate independently but cannot represent a rotated ridge,
    # whereas the vp/fp and ve/ps couplings in TCXM are linear
    # (Tc = vp/fp, Te = ve/ps) -- exactly what a dense matrix captures
    # and a diagonal one structurally cannot. Verified end-to-end for
    # both settings (imm.shape is (D,) when diagonal, (D,D) when dense;
    # both converged with 0 divergences on a synthetic 5-D target).
    dense = getattr(cfg_bayes, "dense_mass_matrix", False)
    target_accept = cfg_bayes.target_accept
    if target_accept == 0.234:        # random-walk default; wrong for NUTS
        target_accept = 0.8

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=not dense,
        target_acceptance_rate=target_accept,
    )

    # Adaptation runs PER CHAIN now -- each chain warms up on its own
    # starting point and keeps its own step size / mass matrix, rather
    # than every chain borrowing chain 0's. Fixes chains landing in a
    # different region of parameter space and never leaving it (a
    # step size tuned for chain 0's terrain can be badly wrong for a
    # chain that started somewhere the posterior looks different).
    key, k_warm = jax.random.split(key)
    warm_keys = jax.random.split(k_warm, n_chains)

    def _run_warmup(k, z0):
        (state, params), _ = warmup.run(k, z0, num_steps=cfg_bayes.n_tune)
        return state, params

    init_states, tuned = jax.vmap(_run_warmup)(warm_keys, z_init)
    # tuned["step_size"] is now shape (n_chains,); tuned["inverse_mass_matrix"]
    # is (n_chains, D) if diagonal or (n_chains, D, D) if dense -- one set
    # of dials per chain, not one shared set.

    def _kernel_step(k, state, step_size, imm):
        return blackjax.nuts(logdensity, step_size=step_size,
                             inverse_mass_matrix=imm).step(k, state)

    # ---- vmapped multi-chain inference loop --------------------------
    # This is where the GPU batching win lands: one batched gradient
    # evaluation per step across all chains, instead of n_chains
    # sequential ones.
    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        states, infos = jax.vmap(_kernel_step)(
            keys, states, tuned["step_size"], tuned["inverse_mass_matrix"])
        return states, (states.position, infos)

    key, k_run = jax.random.split(key)
    keys = jax.random.split(k_run, cfg_bayes.n_draws)
    _, (positions, infos) = jax.lax.scan(one_step, init_states, keys)

    # positions: (n_draws, n_chains, D) -> (n_chains, n_draws, D)
    z_chains = jnp.swapaxes(positions, 0, 1)

    # ---- back to constrained space -----------------------------------
    x_chains = jax.vmap(jax.vmap(transform.constrain))(z_chains)
    chains = np.asarray(jax.device_get(x_chains))

    n_div = int(np.asarray(jax.device_get(infos.is_divergent)).sum())

    rhat, ess, mcse_mean, mcse_median = _diagnostics(chains, layout.param_names)
    
    step_sizes = np.asarray(jax.device_get(tuned["step_size"]))
    info = dict(
        step_size=step_sizes,                              # one per chain now
        step_size_mean=float(step_sizes.mean()),
        step_size_spread=float(step_sizes.max() - step_sizes.min()),
        dense_mass_matrix=bool(dense),
        target_accept=float(target_accept),
        inverse_mass_matrix=np.asarray(jax.device_get(tuned["inverse_mass_matrix"])),
    )
    
    return chains, rhat, ess, mcse_mean, mcse_median, n_div, info


def _diagnostics(chains, param_names):
    """
    Same four dicts _rhat_ess_mcse returns, built from a raw
    (chain, draw, D) array instead of a PyMC InferenceData.

    mcse_median is computed because that is the number this project
    actually reports (medians, BC1/BC3 convention).
    """
    import arviz as az
    # Raw (chain, draw) arrays are passed straight to ArviZ rather than
    # building an InferenceData first. az.from_dict's signature changed
    # between ArviZ 0.x and 1.x; the ndarray path is stable across both.
    rhat, ess, mcse_mean, mcse_median = {}, {}, {}, {}
    for j, nm in enumerate(param_names):
        a = np.asarray(chains[:, :, j])
        rhat[nm] = float(np.asarray(az.rhat(a)))
        ess[nm] = float(np.asarray(az.ess(a)))
        mcse_mean[nm] = float(np.asarray(az.mcse(a, method="mean")))
        mcse_median[nm] = float(np.asarray(
            az.mcse(a, method="quantile", prob=0.5)))
    return rhat, ess, mcse_mean, mcse_median