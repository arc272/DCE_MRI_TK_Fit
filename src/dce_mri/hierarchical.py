#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierarchical.py
===============
Two-stage hierarchical Bayesian group comparison (benign vs malignant) built
on the per-tumor posterior draws written by scripts/fit_lesions_bayes.py.

Why two-stage
-------------
Stage 1 (already done, one fit per tumor) gives draws

    theta_i^(s) ~ p(theta | y_i)  proportional to  L_i(theta) p0(theta)

where p0 is the FIXED prior the per-tumor sampler used. A hierarchical model
replaces p0 with a population distribution p(theta | phi) whose
hyperparameters phi (group means, spreads, B-vs-M effect, covariate effects)
are learned across the cohort. Because

    p(y_i | phi) = integral L_i(theta) p(theta | phi) dtheta
                 = Z_i * E_{theta ~ p(.|y_i)} [ p(theta | phi) / p0(theta) ],

with Z_i = integral L_i p0 a constant in phi, the stage-1 draws can be
REWEIGHTED instead of refitting every tumor (Lunn, Barrett, Sweeting &
Thompson 2013, "Fully Bayesian hierarchical modelling in two stages, with
application to meta-analysis"). Only phi (~20 dims) is sampled. Within-tumor
parameter correlations are kept exactly, because whole joint draws are
reweighted -- only the POPULATION model is diagonal.

Population scale
----------------
The population model lives on

    z = logit(u),   u = (h(x) - h(lb)) / (h(ub) - h(lb)),

with h = log for log-sampled parameters (fp, ps) and identity otherwise. z
spans the whole real line, so a Normal / Student-t population needs no
truncation constant. (A population truncated to [lb, ub] on the natural
scale has a phi-dependent normaliser that would otherwise have to enter the
likelihood.) Both densities in the importance ratio are expressed in x-space:

    log w_is(phi) = log f(z_is | phi) + log|dz/dx|_is - log p0(x_is)
                    '---------------'   '----------------------------'
                     depends on phi        c_is, precomputed once

The population median of parameter j in group g is g^-1(mu_gj) in natural
units (exact: g is monotone and the population is symmetric on z).

Two likelihoods for p(y_i | phi)
--------------------------------
"mc"       the importance average above, over the raw draws. Exact as the
           number of draws grows, but NOISY for prior-dominated tumors:
           reweighting a broad, prior-like posterior onto a narrow
           population in five dimensions leaves few effective draws.
           Per-tumor Pareto k-hat (PSIS, Vehtari et al.) is reported over
           posterior phi draws; k-hat > 0.7 marks an unreliable tumor.
           Each evaluation sums over N x S x P values -- expensive.
"gaussian" each tumor's draws summarised by mean and covariance on z, the
           prior divided out analytically (section 3b), the population
           integrated out in closed form. No Monte Carlo noise; a
           prior-dominated direction contributes exactly nothing; cheap.
           Approximate where stage-1 posteriors are non-Gaussian on z.
Running both and checking agreement is the recommended use: where they
agree, neither approximation is driving the answer.

Verification (tests/hierarchical_verification_tests/verify_hierarchical.py)
on synthetic cohorts with known exact answers: group effects (delta) from
the gaussian likelihood match the exact posterior within 0.09 posterior SD;
two vp nuisance components were off by ~0.3 SD. The mc likelihood's error
falls with more draws and 89% of it sat in k-hat > 0.7 tumors.

The ratio is only correct if p0 is EXACTLY the prior the per-tumor sampler
used. Draws outside p0's support raise immediately; a different prior with
the same support (e.g. 'flat_masked' vs 'beta' volume fractions) cannot be
detected from the draws -- PRIOR_CHOICES / DELAY_BOUNDS must match the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import t as _jt

LOG2PI = float(np.log(2.0 * np.pi))


# =============================================================================
# 1. Population-scale transform
# =============================================================================

@dataclass(frozen=True)
class BoxLogitTransform:
    """
    x (natural, box [lb, ub]) <-> z (real line), per parameter.
    log[j] True -> the box is taken on log(x) (log-sampled parameters).
    """
    names: tuple
    lb: np.ndarray
    ub: np.ndarray
    log: np.ndarray
    eps: float = 1e-12

    def __post_init__(self):
        lb = np.asarray(self.lb, float)
        ub = np.asarray(self.ub, float)
        log = np.asarray(self.log, bool)
        if not (lb.shape == ub.shape == log.shape == (len(self.names),)):
            raise ValueError("names / lb / ub / log must all have length P")
        if np.any(ub <= lb):
            raise ValueError("every ub must exceed its lb")
        if np.any(lb[log] <= 0):
            raise ValueError("log-scale parameters need lb > 0")
        object.__setattr__(self, "lb", lb)
        object.__setattr__(self, "ub", ub)
        object.__setattr__(self, "log", log)

    @property
    def P(self) -> int:
        return len(self.names)

    def _h(self, x):
        x = np.array(x, float, copy=True)
        x[..., self.log] = np.log(x[..., self.log])
        return x

    @property
    def _hl(self):
        return self._h(self.lb)

    @property
    def _hu(self):
        return self._h(self.ub)

    def to_u(self, x):
        return (self._h(x) - self._hl) / (self._hu - self._hl)

    def to_z(self, x, return_n_clipped: bool = False):
        u = self.to_u(x)
        n_clip = int(np.sum((u < self.eps) | (u > 1.0 - self.eps)))
        u = np.clip(u, self.eps, 1.0 - self.eps)
        z = np.log(u) - np.log1p(-u)
        return (z, n_clip) if return_n_clipped else z

    def log_abs_jac(self, x):
        """sum_j log|dz_j/dx_j|, shape x.shape[:-1]."""
        x = np.asarray(x, float)
        u = np.clip(self.to_u(x), self.eps, 1.0 - self.eps)
        lj = -np.log(u) - np.log1p(-u) - np.log(self._hu - self._hl)
        lj = lj + np.where(self.log, -np.log(np.where(self.log, x, 1.0)), 0.0)
        return lj.sum(-1)

    def from_z(self, z):
        z = np.asarray(z, float)
        u = 1.0 / (1.0 + np.exp(-z))
        hx = self._hl + u * (self._hu - self._hl)
        return np.where(self.log, np.exp(hx), hx)

    def from_z_col(self, z, j: int):
        """Inverse for parameter j only (z any shape)."""
        z = np.asarray(z, float)
        u = 1.0 / (1.0 + np.exp(-z))
        hl, hu = self._hl[j], self._hu[j]
        hx = hl + u * (hu - hl)
        return np.exp(hx) if self.log[j] else hx


# =============================================================================
# 2. Cohort: stage-1 draws, transformed, with the importance constant c
# =============================================================================

@dataclass
class Cohort:
    subject_ids: list
    malignant: np.ndarray        # (N,) 0 = benign, 1 = malignant
    x: np.ndarray                # (N, S, P) natural-scale stage-1 draws
    z: np.ndarray                # (N, S, P) population-scale draws
    c: Optional[np.ndarray]      # (N, S)  log|dz/dx| - log p0(x); None if
                                 #         only the Gaussian method is used
    transform: BoxLogitTransform
    X: np.ndarray = None         # (N, C) covariates
    cov_names: tuple = ()

    @property
    def N(self):
        return self.x.shape[0]

    @property
    def S(self):
        return self.x.shape[1]

    @property
    def param_names(self):
        return self.transform.names

    def data(self) -> dict:
        """JAX arrays for the log-density (passed as jit ARGUMENTS, not
        closed over, so large draw arrays are not baked into the HLO)."""
        if self.c is None:
            raise ValueError("cohort was built without log_prior_fn; the Monte "
                             "Carlo likelihood needs c")
        return dict(z=jnp.asarray(self.z), c=jnp.asarray(self.c),
                    **self.design())

    def design(self) -> dict:
        X = self.X if self.X is not None else np.zeros((self.N, 0))
        return dict(M=jnp.asarray(self.malignant, dtype=jnp.float64),
                    X=jnp.asarray(X, dtype=jnp.float64))


def build_cohort(subject_ids, malignant, draws, transform: BoxLogitTransform,
                 log_prior_fn: Callable, X=None, cov_names=()) -> Cohort:
    """
    draws        : (N, S, P) natural-scale stage-1 posterior draws
    log_prior_fn : (S, P) -> (S,) log p0(x) in x-space (the per-tumor prior),
                   or None to skip c (Gaussian method only -- much faster,
                   since evaluating p0 at every draw is the slow step)
    """
    draws = np.asarray(draws, float)
    N, S, P = draws.shape
    if P != transform.P:
        raise ValueError(f"draws have {P} parameters, transform has {transform.P}")
    malignant = np.asarray(malignant, float)
    if not np.all(np.isin(malignant, (0.0, 1.0))):
        raise ValueError("malignant must be 0/1")

    z, n_clip = transform.to_z(draws, return_n_clipped=True)
    if n_clip:
        print(f"  ! {n_clip} draw coordinate(s) within {transform.eps:g} of a "
              f"bound were clipped for the logit")
    c = None
    if log_prior_fn is not None:
        lp0 = np.stack([np.asarray(log_prior_fn(draws[i]), float)
                        for i in range(N)])
        bad = ~np.isfinite(lp0)
        if bad.any():
            i = int(np.argwhere(bad)[0, 0])
            raise ValueError(
                f"{int(bad.sum())} draw(s) fall outside the prior's support "
                f"(first: subject {subject_ids[i]}). The prior rebuilt here "
                f"does not match the one the per-tumor sampler used -- check "
                f"PRIOR_CHOICES / DELAY_BOUNDS against the fit run.")
        c = transform.log_abs_jac(draws) - lp0

    if X is not None:
        X = np.asarray(X, float)
        if X.shape != (N, len(cov_names)):
            raise ValueError(f"X must be (N, {len(cov_names)}), got {X.shape}")
    return Cohort(list(subject_ids), malignant, draws, z, c, transform,
                  X, tuple(cov_names))


def prior_logpdf_rows(prior) -> Callable:
    """Wrap a PriorSpec (per-vector log_pdf) as (S, P) -> (S,).
    Uses the prior's OWN log_pdf, row by row, so there is no second
    implementation of p0 that could drift from the sampler's."""
    def f(X):
        return np.fromiter((prior.log_pdf(r) for r in np.asarray(X, float)),
                           float, count=len(X))
    return f


# =============================================================================
# 3. Hierarchical specification and hyperparameter layout
# =============================================================================

@dataclass(frozen=True)
class HierSpec:
    """
    param_names    : population-modelled parameters, same order as the cohort
    group_effect   : per parameter -- separate B and M means (True) or one
                     shared mean (False, e.g. the nuisance tau_delay)
    separate_sigma : per parameter -- separate B and M population spreads
    family         : 'normal' or 'student_t' (population on the z-scale)
    nu             : Student-t degrees of freedom (fixed)
    mu_sd          : Normal(0, mu_sd) on each reference-group mean (z-scale)
    delta_sd       : Normal(0, delta_sd) on each M - B difference (z-scale)
    sigma_scale    : HalfNormal(sigma_scale) on each population SD (z-scale)
    beta_sd        : Normal(0, beta_sd) on each covariate effect (z-scale)
    cov_names      : covariates; they act on the group-effect parameters
    """
    param_names: tuple
    group_effect: tuple
    separate_sigma: tuple
    family: str = "normal"
    nu: float = 5.0
    mu_sd: float = 2.0
    delta_sd: float = 1.0
    sigma_scale: float = 1.0
    beta_sd: float = 1.0
    cov_names: tuple = ()

    def __post_init__(self):
        P = len(self.param_names)
        if len(self.group_effect) != P or len(self.separate_sigma) != P:
            raise ValueError("group_effect / separate_sigma need one entry "
                             "per parameter")
        if self.family not in ("normal", "student_t"):
            raise ValueError(f"family must be 'normal' or 'student_t'")


class PhiLayout:
    """Index map for the unconstrained hyperparameter vector phi."""

    def __init__(self, spec: HierSpec):
        pn = list(spec.param_names)
        self.P = len(pn)
        self.ge = np.flatnonzero(np.asarray(spec.group_effect, bool))
        self.ss = np.flatnonzero(np.asarray(spec.separate_sigma, bool))
        self.C = len(spec.cov_names)
        names, self.sl, k = [], {}, 0

        def block(key, labels):
            nonlocal k
            self.sl[key] = slice(k, k + len(labels))
            names.extend(labels)
            k += len(labels)

        block("a", [f"mu_B[{p}]" if j in self.ge else f"mu[{p}]"
                    for j, p in enumerate(pn)])
        block("d", [f"delta[{pn[j]}]" for j in self.ge])
        block("b", [f"beta[{c}][{pn[j]}]" for c in spec.cov_names for j in self.ge])
        block("ls", [f"log_sigma_B[{p}]" if j in self.ss else f"log_sigma[{p}]"
                     for j, p in enumerate(pn)])
        block("lsm", [f"log_sigma_M[{pn[j]}]" for j in self.ss])
        self.names = tuple(names)
        self.dim = k

    def unpack(self, phi):
        """Works for a single phi (dim,) or a batch (..., dim), numpy or jax."""
        a = phi[..., self.sl["a"]]
        d = phi[..., self.sl["d"]]
        b = phi[..., self.sl["b"]]
        b = b.reshape(b.shape[:-1] + (self.C, len(self.ge)))
        ls = phi[..., self.sl["ls"]]
        lsm = phi[..., self.sl["lsm"]]
        return a, d, b, ls, lsm


def _norm_logpdf(x, sd):
    return -0.5 * (x / sd) ** 2 - jnp.log(sd) - 0.5 * LOG2PI


def log_hyperprior(phi, spec: HierSpec, layout: PhiLayout):
    a, d, b, ls, lsm = layout.unpack(phi)
    lp = (_norm_logpdf(a, spec.mu_sd).sum()
          + _norm_logpdf(d, spec.delta_sd).sum()
          + _norm_logpdf(b, spec.beta_sd).sum())

    def halfnormal_on_log(l):
        s = jnp.exp(l)
        # HalfNormal(scale) density of s, plus log-Jacobian ds/dl = s
        return (0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(spec.sigma_scale)
                - 0.5 * (s / spec.sigma_scale) ** 2 + l).sum()

    return lp + halfnormal_on_log(ls) + halfnormal_on_log(lsm)


def means_and_sigmas(phi, M, X, layout: PhiLayout):
    """Per-tumor population mean m (N, P) and SD sig (N, P) on the z-scale."""
    a, d, b, ls, lsm = layout.unpack(phi)
    P = layout.P
    m = jnp.broadcast_to(a, (M.shape[0], P))
    if len(layout.ge):
        d_full = jnp.zeros(P).at[layout.ge].set(d)
        m = m + M[:, None] * d_full[None, :]
        if layout.C:
            B_full = jnp.zeros((layout.C, P)).at[:, layout.ge].set(b)
            m = m + X @ B_full
    sB = jnp.exp(ls)
    sM = sB.at[layout.ss].set(jnp.exp(lsm)) if len(layout.ss) else sB
    sig = jnp.where(M[:, None] > 0.5, sM[None, :], sB[None, :])
    return m, sig


def _logpdf_std(u, family, nu):
    if family == "normal":
        return -0.5 * u * u - 0.5 * LOG2PI
    return _jt.logpdf(u, df=nu)


def make_two_stage_model(spec: HierSpec, layout: PhiLayout):
    """
    Returns (logdensity(phi, data), log_weights(phi, data)).

    log_weights : (N, S) unnormalised log importance weights
                  log f(z | phi) + c  -- the per-tumor quantity whose
                  log-mean-exp is log p(y_i | phi) up to a phi-free constant.
    logdensity  : log hyperprior + sum_i log-mean-exp_s log_weights
    """
    fam, nu = spec.family, spec.nu

    def log_weights(phi, data):
        m, sig = means_and_sigmas(phi, data["M"], data["X"], layout)
        u = (data["z"] - m[:, None, :]) / sig[:, None, :]
        lf = _logpdf_std(u, fam, nu) - jnp.log(sig)[:, None, :]
        return lf.sum(-1) + data["c"]

    def logdensity(phi, data):
        lw = log_weights(phi, data)
        ll = logsumexp(lw, axis=1) - jnp.log(lw.shape[1])
        return log_hyperprior(phi, spec, layout) + ll.sum()

    return logdensity, log_weights


def init_phi(cohort: Cohort, layout: PhiLayout) -> np.ndarray:
    """Method-of-moments start from per-tumor posterior means on z."""
    zbar = cohort.z.mean(1)
    M = cohort.malignant > 0.5
    phi = np.zeros(layout.dim)
    ref = zbar[~M] if (~M).any() else zbar
    a = ref.mean(0)
    a[[j for j in range(layout.P) if j not in layout.ge]] = \
        zbar.mean(0)[[j for j in range(layout.P) if j not in layout.ge]]
    phi[layout.sl["a"]] = np.clip(a, -4, 4)
    if len(layout.ge) and M.any() and (~M).any():
        phi[layout.sl["d"]] = np.clip(zbar[M].mean(0)[layout.ge]
                                      - zbar[~M].mean(0)[layout.ge], -3, 3)
    sd_ref = np.maximum(ref.std(0), 0.2)
    phi[layout.sl["ls"]] = np.log(sd_ref)
    if len(layout.ss):
        sd_m = np.maximum(zbar[M].std(0) if M.sum() > 1 else sd_ref, 0.2)
        phi[layout.sl["lsm"]] = np.log(sd_m[layout.ss])
    return phi


# =============================================================================
# 3b. Gaussian-approximation likelihood (no Monte Carlo noise)
# =============================================================================
#
# Each tumor's stage-1 posterior on z is summarised by its mean mu_i and
# covariance C_i. The per-tumor prior is divided out LOCALLY (Laplace-style):
# expanding log p0_z around mu_i with gradient g0 and curvature H0,
#
#     log L_i(z) ~ log post_i(z) - log p0_z(z)
#                ~ -1/2 z' Q_i z + h_i' z + const,
#     Q_i = C_i^-1 - H0(mu_i),    h_i = Q_i mu_i - g0(mu_i)
#
# (information form, so Q_i may be singular). Directions where Q_i has an
# eigenvalue <= min_info carry no usable information and are DROPPED -- the
# likelihood is exactly flat there, so a prior-dominated tumor contributes
# nothing instead of noise. Integrating the population N(z; m, D) out:
#
#     log p(y_i | phi) = -1/2 log|D| - 1/2 log|A| + 1/2 b'A^-1 b
#                        - 1/2 m'D^-1 m + const,
#     A = Q_i + D^-1,   b = h_i + D^-1 m.
#
# Exact when the stage-1 posterior is Gaussian on z and p0_z is locally
# quadratic; approximate otherwise. Per-tumor skewness of the z-draws is
# reported as a flag for where the Gaussian summary is questionable.

@dataclass
class GaussianLik:
    Q: np.ndarray          # (N, P, P) likelihood precision on z (PSD)
    h: np.ndarray          # (N, P)    information vector
    post_mean: np.ndarray  # (N, P)
    post_cov: np.ndarray   # (N, P, P)
    n_dropped: np.ndarray  # (N,) directions with no usable information
    max_abs_skew: np.ndarray  # (N,) worst |skewness| of the z-draws
    min_info: float

    def data(self, cohort: "Cohort") -> dict:
        return dict(Q=jnp.asarray(self.Q), h=jnp.asarray(self.h),
                    **cohort.design())


def _log_p0_z(log_prior_fn, transform, Z):
    """log p0 on the z-scale: log p0(x(z)) - log|dz/dx|(x(z))."""
    X = transform.from_z(Z)
    return np.asarray(log_prior_fn(X), float) - transform.log_abs_jac(X)


def prior_z_derivatives(log_prior_fn, transform, zc, step=1e-3):
    """Diagonal gradient g0 and curvature H0 = -d2/dz2 of log p0_z at each
    row of zc (N, P), by central differences. Diagonal is exact for a prior
    that is a product of 1-D factors (true for the TCXM choices); see
    check_prior_separable."""
    zc = np.asarray(zc, float)
    f0 = _log_p0_z(log_prior_fn, transform, zc)
    g = np.zeros_like(zc)
    H = np.zeros_like(zc)
    for j in range(zc.shape[1]):
        zp, zm = zc.copy(), zc.copy()
        zp[:, j] += step
        zm[:, j] -= step
        fp = _log_p0_z(log_prior_fn, transform, zp)
        fm = _log_p0_z(log_prior_fn, transform, zm)
        g[:, j] = (fp - fm) / (2 * step)
        H[:, j] = -(fp - 2 * f0 + fm) / step ** 2
    return g, H


def check_prior_separable(log_prior_fn, transform, z0, step=1e-3):
    """Largest |mixed second derivative| of log p0_z at z0 (P,).
    ~0 means the diagonal curvature in prior_z_derivatives is exact."""
    z0 = np.asarray(z0, float)
    P = z0.size
    worst = 0.0
    for j in range(P):
        for k in range(j + 1, P):
            pts = []
            for sj, sk in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                z = z0.copy()
                z[j] += sj * step
                z[k] += sk * step
                pts.append(z)
            f = _log_p0_z(log_prior_fn, transform, np.array(pts))
            worst = max(worst, abs((f[0] - f[1] - f[2] + f[3]) / (4 * step ** 2)))
    return worst


def gaussian_likelihood(cohort: "Cohort", log_prior_fn, min_info=0.05):
    """
    Build the per-tumor Gaussian likelihood (Q_i, h_i) from the stage-1
    draws. min_info: smallest likelihood precision (z-scale) treated as
    information; 0.05 = an SD of ~4.5 on z, wider than the prior itself.
    """
    z = cohort.z
    N, S, P = z.shape
    mu = z.mean(1)
    d = z - mu[:, None, :]
    C = np.einsum("nsi,nsj->nij", d, d) / (S - 1)
    sd = np.sqrt(np.einsum("nii->ni", C))
    skew = np.abs(np.mean(d ** 3, axis=1) / sd ** 3).max(axis=1)
    g0, H0 = prior_z_derivatives(log_prior_fn, cohort.transform, mu)
    Q = np.linalg.inv(C) - H0[:, :, None] * np.eye(P)[None]
    Q = 0.5 * (Q + np.swapaxes(Q, 1, 2))
    lam, U = np.linalg.eigh(Q)
    keep = lam > min_info
    Qc = np.einsum("nik,nk,njk->nij", U, np.where(keep, lam, 0.0), U)
    Pp = np.einsum("nik,nk,njk->nij", U, keep.astype(float), U)
    h = np.einsum("nij,nj->ni", Qc, mu) - np.einsum("nij,nj->ni", Pp, g0)
    return GaussianLik(Qc, h, mu, C, (~keep).sum(1), skew, float(min_info))


def make_gaussian_model(spec: HierSpec, layout: PhiLayout):
    """
    Returns (logdensity(phi, data), tumor_loglik(phi, data) -> (N,),
             cond_posterior(phi, data) -> (mean (N,P), cov (N,P,P)) of z_i | phi).
    data = GaussianLik.data(cohort).
    """
    def _A_b(phi, data):
        m, sig = means_and_sigmas(phi, data["M"], data["X"], layout)
        Dinv = sig ** -2
        A = data["Q"] + jax.vmap(jnp.diag)(Dinv)
        return A, data["h"] + Dinv * m, m, sig, Dinv

    def tumor_loglik(phi, data):
        A, b, m, sig, Dinv = _A_b(phi, data)
        L = jnp.linalg.cholesky(A)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)), axis=1)
        y = jax.vmap(lambda L_, b_: jax.scipy.linalg.solve_triangular(
            L_, b_, lower=True))(L, b)
        return (-jnp.sum(jnp.log(sig), axis=1) - 0.5 * logdetA
                + 0.5 * jnp.sum(y ** 2, axis=1) - 0.5 * jnp.sum(Dinv * m ** 2, axis=1))

    def logdensity(phi, data):
        return log_hyperprior(phi, spec, layout) + tumor_loglik(phi, data).sum()

    def cond_posterior(phi, data):
        A, b, _, _, _ = _A_b(phi, data)
        cov = jnp.linalg.inv(A)
        return jnp.einsum("nij,nj->ni", cov, b), cov

    return logdensity, tumor_loglik, cond_posterior


def gaussian_tumor_samples(phi_draws, cond_posterior_fn, data, n_per=20, seed=0):
    """Samples from each tumor's hierarchical posterior on z (mixture over
    phi draws of the closed-form conditionals). Returns (N, K*n_per, P)."""
    cp = jax.jit(cond_posterior_fn)
    rng = np.random.default_rng(seed)
    out = []
    for phi in np.atleast_2d(phi_draws):
        mean, cov = (np.asarray(a) for a in cp(jnp.asarray(phi), data))
        Lc = np.linalg.cholesky(cov)
        e = rng.standard_normal((mean.shape[0], n_per, mean.shape[1]))
        out.append(mean[:, None, :] + np.einsum("nij,nkj->nki", Lc, e))
    return np.concatenate(out, axis=1)


# =============================================================================
# 4. Hyperparameter sampler (same BlackJAX pattern as bayes_blackjax.py)
# =============================================================================

def sample_hyper(logdensity_fn, data, phi0, *, n_warmup=2000, n_draws=2000,
                 n_chains=4, seed=1, dense=True, target_accept=0.9,
                 init_jitter=0.1):
    """
    NUTS over phi with per-chain window adaptation, chains vmapped -- the
    same calls bayes_blackjax.sample_blackjax uses. The whole run is one jit
    that takes `data` as an ARGUMENT.

    Returns (chains (n_chains, n_draws, dim), n_divergent, mean_accept,
             step_sizes (n_chains,)).
    """
    import blackjax

    key = jax.random.PRNGKey(seed)
    key, kj = jax.random.split(key)
    phi0 = jnp.asarray(phi0, jnp.float64)
    z0 = phi0[None, :] + init_jitter * jax.random.normal(
        kj, (n_chains, phi0.shape[0]), dtype=jnp.float64)

    @jax.jit
    def run(data, z0, key):
        logdensity = lambda p: logdensity_fn(p, data)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity,
            is_mass_matrix_diagonal=not dense,
            target_acceptance_rate=target_accept)
        k_warm, k_run = jax.random.split(key)
        warm_keys = jax.random.split(k_warm, n_chains)

        def _run_warmup(k, p0):
            (state, params), _ = warmup.run(k, p0, num_steps=n_warmup)
            return state, params

        states, tuned = jax.vmap(_run_warmup)(warm_keys, z0)

        def _kernel_step(k, state, step_size, imm):
            return blackjax.nuts(logdensity, step_size=step_size,
                                 inverse_mass_matrix=imm).step(k, state)

        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, n_chains)
            states, infos = jax.vmap(_kernel_step)(
                keys, states, tuned["step_size"], tuned["inverse_mass_matrix"])
            return states, (states.position, infos.is_divergent,
                            infos.acceptance_rate)

        keys = jax.random.split(k_run, n_draws)
        _, (pos, div, acc) = jax.lax.scan(one_step, states, keys)
        return jnp.swapaxes(pos, 0, 1), div, acc, tuned["step_size"]

    pos, div, acc, steps = run(data, z0, key)
    return (np.asarray(pos), int(np.asarray(div).sum()),
            float(np.asarray(acc).mean()), np.asarray(steps))


def hyper_diagnostics(chains, names):
    """R-hat / ESS through the SAME helper the per-tumor fits use."""
    from dce_mri.bayes_blackjax import _diagnostics
    rhat, ess, _, _ = _diagnostics(chains, list(names))
    return rhat, ess


# =============================================================================
# 5. Importance diagnostics, PSIS k-hat, per-tumor shrunken posteriors
# =============================================================================

def _gpdfit(x):
    """Zhang & Stephens (2009) generalized-Pareto fit with the weakly
    informative prior on k used in PSIS (Vehtari et al.).
    x: ascending, positive exceedances."""
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(np.sqrt(n))
    b_ary = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b_ary += 1.0 / x[-1]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        k_ary = np.log1p(-b_ary[:, None] * x).mean(axis=1)
        len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1.0)
        w = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b_ary = w[keep], b_ary[keep]
    w /= w.sum()
    b_post = np.sum(b_ary * w)
    k_post = np.log1p(-b_post * x).mean()
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def pareto_khat(log_w) -> float:
    """PSIS Pareto k-hat of a vector of (unnormalised) log importance
    weights. > 0.7: the importance estimate is unreliable."""
    lw = np.sort(np.asarray(log_w, float))
    S = lw.size
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    if M < 5:
        return np.nan
    lmax = lw[-1]
    x = np.exp(lw[-M:] - lmax) - np.exp(lw[-M - 1] - lmax)
    if not np.any(x > 0):
        return -np.inf
    return _gpdfit(np.maximum(x, np.finfo(float).tiny))


def _np_logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)))


def tumor_reweighting(phi_draws, log_weights_fn, data):
    """
    For K posterior phi draws: per-tumor normalised weights, Pareto k-hat and
    weight ESS. The per-tumor hierarchical posterior is the mixture over phi,
    so its weights are the average of the K normalised weight vectors.

    Returns W (N, S), khat (N, K), ess (N, K).
    """
    lw_fn = jax.jit(log_weights_fn)
    phi_draws = np.atleast_2d(phi_draws)
    K = phi_draws.shape[0]
    W, khat, ess = None, None, None
    for k in range(K):
        lw = np.asarray(lw_fn(jnp.asarray(phi_draws[k]), data))
        if W is None:
            N, S = lw.shape
            W = np.zeros((N, S))
            khat = np.zeros((N, K))
            ess = np.zeros((N, K))
        w = np.exp(lw - _np_logsumexp(lw, axis=1))
        W += w
        ess[:, k] = 1.0 / np.sum(w ** 2, axis=1)
        khat[:, k] = [pareto_khat(lw[i]) for i in range(lw.shape[0])]
    return W / K, khat, ess


def weighted_quantile(values, weights, qs):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(qs), cw, v)


# =============================================================================
# 6. Summaries
# =============================================================================

def summarize_hyper(chains, layout: PhiLayout, spec: HierSpec,
                    transform: BoxLogitTransform, rhat: dict, ess: dict):
    """
    One row per parameter: population medians (natural units) for B and M,
    their difference / ratio, the z-scale group effect with P(delta > 0),
    population SDs, and the worst R-hat / ESS among that parameter's phi
    components. 'B' / 'M' means are at covariate value 0.
    """
    flat = chains.reshape(-1, chains.shape[-1])
    a, d, b, ls, lsm = layout.unpack(flat)
    rows = []
    q = lambda v: np.percentile(v, [2.5, 50, 97.5])
    for j, p in enumerate(spec.param_names):
        mu_B = a[:, j]
        if j in layout.ge:
            k = int(np.flatnonzero(layout.ge == j)[0])
            delta = d[:, k]
        else:
            delta = np.zeros_like(mu_B)
        mu_M = mu_B + delta
        med_B = transform.from_z_col(mu_B, j)
        med_M = transform.from_z_col(mu_M, j)
        sB = np.exp(ls[:, j])
        sM = (np.exp(lsm[:, int(np.flatnonzero(layout.ss == j)[0])])
              if j in layout.ss else sB)
        own = [n for n in layout.names if n.endswith(f"[{p}]")]
        row = dict(param=p, group_effect=bool(j in layout.ge))
        for nm, v in (("popmedian_B", med_B), ("popmedian_M", med_M),
                      ("diff_M_minus_B", med_M - med_B),
                      ("delta_z", delta), ("sigma_B_z", sB), ("sigma_M_z", sM)):
            lo, md, hi = q(v)
            row.update({f"{nm}_median": md, f"{nm}_ci2.5": lo, f"{nm}_ci97.5": hi})
        if np.all(med_B > 0):
            lo, md, hi = q(med_M / med_B)
            row.update(ratio_M_over_B_median=md, **{"ratio_M_over_B_ci2.5": lo,
                                                    "ratio_M_over_B_ci97.5": hi})
        row["P_delta_gt_0"] = (float(np.mean(delta > 0))
                               if j in layout.ge else np.nan)
        row["ci95_excludes_0"] = bool(j in layout.ge and
                                      (q(delta)[0] > 0 or q(delta)[2] < 0))
        row["rhat_max"] = max(rhat[n] for n in own)
        row["ess_min"] = min(ess[n] for n in own)
        rows.append(row)
    return pd.DataFrame(rows)


def tumor_table(cohort: Cohort, W, khat, ess):
    """Stage-1 vs hierarchical per-tumor estimates + importance diagnostics."""
    rows = []
    for i, sid in enumerate(cohort.subject_ids):
        r = dict(subject_id=sid, diagnosis="M" if cohort.malignant[i] else "B",
                 khat_max=float(np.nanmax(khat[i])),
                 khat_median=float(np.nanmedian(khat[i])),
                 weight_ess_min=float(ess[i].min()),
                 weight_ess_median=float(np.median(ess[i])))
        for j, p in enumerate(cohort.param_names):
            v = cohort.x[i, :, j]
            r[f"{p}_stage1_median"] = float(np.median(v))
            lo, md, hi = weighted_quantile(v, W[i], [0.025, 0.5, 0.975])
            r.update({f"{p}_hier_median": md, f"{p}_hier_ci2.5": lo,
                      f"{p}_hier_ci97.5": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def tumor_table_gaussian(cohort: Cohort, z_samples, glik: GaussianLik):
    """Stage-1 vs hierarchical per-tumor estimates for the Gaussian method,
    from z-samples (N, M, P), plus its per-tumor flags."""
    x_s = cohort.transform.from_z(z_samples)
    rows = []
    for i, sid in enumerate(cohort.subject_ids):
        r = dict(subject_id=sid, diagnosis="M" if cohort.malignant[i] else "B",
                 n_dropped_directions=int(glik.n_dropped[i]),
                 max_abs_skew_z=float(glik.max_abs_skew[i]))
        for j, p in enumerate(cohort.param_names):
            r[f"{p}_stage1_median"] = float(np.median(cohort.x[i, :, j]))
            lo, md, hi = np.percentile(x_s[i, :, j], [2.5, 50, 97.5])
            r.update({f"{p}_hier_median": md, f"{p}_hier_ci2.5": lo,
                      f"{p}_hier_ci97.5": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def compare_methods(chains_a, chains_b, layout: PhiLayout, tol_sd=0.25):
    """
    Agreement of two hyperparameter posteriors (e.g. Gaussian vs Monte
    Carlo likelihood), per phi component: |mean_a - mean_b| in units of the
    larger posterior SD, and the SD ratio. 'agree' uses tol_sd -- the same
    threshold the verification suite uses against the exact answer.
    """
    fa = chains_a.reshape(-1, chains_a.shape[-1])
    fb = chains_b.reshape(-1, chains_b.shape[-1])
    sd = np.maximum(fa.std(0), fb.std(0))
    dm = np.abs(fa.mean(0) - fb.mean(0)) / sd
    return pd.DataFrame(dict(name=layout.names, mean_a=fa.mean(0), mean_b=fb.mean(0),
                             sd_a=fa.std(0), sd_b=fb.std(0), mean_diff_in_sd=dm,
                             sd_ratio_a_over_b=fa.std(0) / fb.std(0),
                             agree=dm < tol_sd))


# =============================================================================
# 7. Stage-1 diagnostic: is pooling worth it?
# =============================================================================

def prior_z_moments(prior_sample_fn, transform: BoxLogitTransform,
                    n=200_000, seed=0):
    """Mean / SD of each parameter's PRIOR on the z-scale."""
    x = prior_sample_fn(np.random.default_rng(seed), n)
    z = transform.to_z(x)
    return z.mean(0), z.std(0)


def stage1_within_between(cohort: Cohort, prior_mean_z, prior_sd_z,
                          informative_frac=0.9):
    """
    Normal approximation on the z-scale, per parameter and group.

    The stage-1 posterior already contains the prior, so its SD is NOT the
    measurement uncertainty. The prior is divided out by precision
    subtraction:  1/s_L^2 = 1/s_post^2 - 1/s_prior^2 . A tumor whose
    posterior variance is not below informative_frac x the prior variance is
    counted as uninformative for that parameter (the data say ~nothing).

    Among informative tumors:
      within_sd         median likelihood SD s_L
      between_sd        method-of-moments tau = sqrt(max(0, Var(m_L) - mean(s_L^2)))
      heterogeneity     P90(s_L) / P10(s_L)  -- unevenness drives the gain
      pred_sd_ratio     SD of a precision-weighted group mean / SD of an
                        equal-weight mean of the same tumors. < 1 = pooling
                        helps. Captures ONLY the precision-weighting gain,
                        not the removal of prior attenuation from
                        prior-dominated tumors (reported via frac_uninformative).
    """
    rows = []
    zbar = cohort.z.mean(1)
    zsd = cohort.z.std(1)
    for j, p in enumerate(cohort.param_names):
        for g, gname in ((0.0, "B"), (1.0, "M")):
            sel = cohort.malignant == g
            if not sel.any():
                continue
            m_post, s_post = zbar[sel, j], zsd[sel, j]
            v_prior = prior_sd_z[j] ** 2
            informative = s_post ** 2 < informative_frac * v_prior
            prec_L = 1.0 / s_post[informative] ** 2 - 1.0 / v_prior
            s_L = 1.0 / np.sqrt(prec_L)
            m_L = (m_post[informative] / s_post[informative] ** 2
                   - prior_mean_z[j] / v_prior) / prec_L
            n_inf = int(informative.sum())
            row = dict(param=p, group=gname, n=int(sel.sum()),
                       n_informative=n_inf,
                       frac_uninformative=1.0 - n_inf / sel.sum(),
                       stage1_post_sd_median=float(np.median(s_post)),
                       stage1_between_sd_of_means=float(np.std(m_post, ddof=1))
                       if sel.sum() > 1 else np.nan)
            if n_inf >= 3:
                tau2 = max(0.0, float(np.var(m_L, ddof=1) - np.mean(s_L ** 2)))
                naive = np.sqrt(np.sum(tau2 + s_L ** 2)) / n_inf
                hier = np.sqrt(1.0 / np.sum(1.0 / (tau2 + s_L ** 2)))
                row.update(within_sd=float(np.median(s_L)),
                           between_sd=float(np.sqrt(tau2)),
                           within_over_between=(float(np.median(s_L) / np.sqrt(tau2))
                                                if tau2 > 0 else np.inf),
                           heterogeneity_P90_P10=float(np.percentile(s_L, 90)
                                                       / np.percentile(s_L, 10)),
                           pred_sd_ratio=float(hier / naive))
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 8. Loading a fit_lesions_bayes.py run + rebuilding its exact prior
# =============================================================================

def _as_bool(v):
    return str(v).strip().lower() == "true"


def load_run_draws(run_dir, param_names, n_draws, *, subset="qc_pass", seed=0,
                   results_csv=None):
    """
    Read lesion_bayes_results.csv + {sid}_draws.npz from a fit run.
    Returns (subject_ids, malignant (N,), draws (N, n_draws, P), info DataFrame).
    Draws are a seeded random subset (without replacement) of the pooled
    chains, so every tumor contributes the same number S.
    """
    path = results_csv or os.path.join(run_dir, "lesion_bayes_results.csv")
    res = pd.read_csv(path)
    err = res["error"] if "error" in res else pd.Series("", index=res.index)
    ok = res[err.isna() | (err.astype(str).str.strip() == "")].copy()
    if subset == "qc_pass":
        ok = ok[ok["qc_pass"].map(_as_bool)]
    elif subset != "all":
        raise ValueError("subset must be 'qc_pass' or 'all'")

    sids, mal, draws, skipped = [], [], [], []
    for n_i, (_, r) in enumerate(ok.iterrows()):
        sid = str(r["subject_id"]).strip()
        dx = str(r["diagnosis"]).strip().upper()
        f = os.path.join(run_dir, f"{sid}_draws.npz")
        if dx not in ("B", "M"):
            skipped.append((sid, f"diagnosis {dx!r}"))
            continue
        if not os.path.exists(f):
            skipped.append((sid, "no draws.npz"))
            continue
        z = np.load(f, allow_pickle=True)
        names = [str(s) for s in z["param_names"]]
        missing = [p for p in param_names if p not in names]
        if missing:
            skipped.append((sid, f"draws lack {missing}"))
            continue
        ch = np.asarray(z["chains"], float)
        flat = ch.reshape(-1, ch.shape[-1])[:, [names.index(p) for p in param_names]]
        if n_draws is None:
            draws.append(flat)
        elif flat.shape[0] < n_draws:
            skipped.append((sid, f"only {flat.shape[0]} draws < {n_draws}"))
            continue
        else:
            rng = np.random.default_rng([seed, n_i])
            draws.append(flat[rng.choice(flat.shape[0], n_draws, replace=False)])
        sids.append(sid)
        mal.append(1.0 if dx == "M" else 0.0)
    info = pd.DataFrame(skipped, columns=["subject_id", "reason"])
    if not draws:
        raise ValueError(f"no usable tumors in {run_dir}")
    if n_draws is None:
        s_min = min(d.shape[0] for d in draws)
        rng = np.random.default_rng(seed)
        draws = [d if d.shape[0] == s_min
                 else d[rng.choice(d.shape[0], s_min, replace=False)] for d in draws]
    return sids, np.asarray(mal), np.stack(draws), info


def tcxm_joint_delay_prior(prior_choices, delay_bounds, acq, t, aif):
    """
    Rebuild the EXACT per-tumor prior fit_lesions_bayes.py used
    (TCXM + joint delay, marginalised-sigma noise) and the matching
    population transform. Returns (prior, transform).
    """
    from dce_mri.kinetic_models import TCXM
    from dce_mri.bayes import BayesConfig, build_param_layout, NOISE_OPTIONS
    from dce_mri.bayes_priors import assemble_prior

    cfg_b = BayesConfig(prior_choices=prior_choices,
                        noise="gauss_marginalized_sigma",
                        delay_mode="joint", delay_bounds=delay_bounds)
    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, None, delay_mode="joint",
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=prior_choices, delay_bounds=delay_bounds)
    prior.validate_against(layout.param_names)
    names = tuple(layout.model.param_names)
    if prior.n_params != len(names):
        raise ValueError(f"prior has {prior.n_params} params, model {len(names)}"
                         f" -- noise parameters are not supported here")
    lb = np.asarray(layout.model.lb, float)
    ub = np.asarray(layout.model.ub, float)
    if "ve" in names and "vp" in names:
        # the z-scale population lives on the box; it would put mass on
        # ve + vp > 1 if the box allowed it, breaking normalisation
        if ub[names.index("ve")] + ub[names.index("vp")] > 1.0:
            raise ValueError("ve_ub + vp_ub > 1: the ve+vp<=1 constraint is "
                             "active inside the box and the population model "
                             "would not be normalised on the feasible set")
    log = np.zeros(len(names), bool)
    log[list(layout.model.log_dims or ())] = True
    return prior, BoxLogitTransform(names, lb, ub, log)
