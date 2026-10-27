# -*- coding: utf-8 -*-
from __future__ import annotations
"""
bayes_priors.py
===============
Prior specifications for Bayesian kinetic / water-exchange fitting.

Pure specs: no sampling, no forward-model evaluation, no I/O. Importable
without touching fitting.py.

Every option carries a `source` tag naming the paper it comes from, so
the provenance of any default is recoverable from this file alone.

Design
------
A prior is a list of ParamBlockPrior objects, each covering one or more
indices of the full augmented parameter vector

    p_aug = [ *theta_kin , (tau_delay?) , (tau_b?) , tau_i? ]

Blocks carry explicit `indices` rather than assuming contiguity, because
the volume fractions are NOT contiguous in every model:
    TCXM param_names = ('ve','vp','fp','ps')   -> volfrac at 0,1
    GKM  param_names = ('Ktrans','ve','vp')    -> volfrac at 1,2

Each block provides three things:
    build_pymc(model_ctx)  -> list of PyMC RVs        (for sampling)
    sample(rng, n)         -> (n, k) numpy draws      (REQUIRED for SBC)
    log_pdf(x)             -> float                   (numpy cross-check)

The `sample` method is not decoration: simulation-based calibration
draws theta_true from the prior, so a prior without a sampler cannot be
calibration-checked.

Public API
----------
    ParamBlockPrior, PriorSpec
    make_* block constructors (see PRIOR OPTIONS below)
    assemble_prior(...)  -- the single place block ORDER is decided
"""



from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats


# =============================================================================
# Core spec objects
# =============================================================================

@dataclass(frozen=True)
class ParamBlockPrior:
    """
    Prior over one group of parameters within p_aug.

    name        : short label, e.g. 'volfrac', 'rates', 'tau'
    source      : citation tag, e.g. 'Orton2007', 'Mittermeier2019', 'none'
    indices     : positions in p_aug this block covers
    param_names : names, same length/order as indices
    is_flat     : True if uniform (or log-uniform) over its support.
                  Used to flag when the posterior stays directly
                  comparable to bootstrap replicates.
    build_pymc  : (names, ctx) -> list of PyMC RVs, one per index
    sample      : (rng, n) -> (n, k) array
    log_pdf     : (x,) -> float, -inf outside support
    """
    name:        str
    source:      str
    indices:     tuple
    param_names: tuple
    is_flat:     bool
    build_pymc:  Callable
    sample:      Callable
    log_pdf:     Callable


@dataclass(frozen=True)
class PriorSpec:
    """Assembled prior over the full p_aug, as an ordered list of blocks."""
    blocks: tuple

    @property
    def n_params(self) -> int:
        return sum(len(b.indices) for b in self.blocks)

    @property
    def param_names(self) -> tuple:
        """Names ordered by p_aug index, not by block order."""
        pairs = []
        for b in self.blocks:
            pairs.extend(zip(b.indices, b.param_names))
        return tuple(nm for _, nm in sorted(pairs))

    @property
    def is_flat(self) -> bool:
        return all(b.is_flat for b in self.blocks)

    @property
    def sources(self) -> dict:
        return {b.name: b.source for b in self.blocks}

    def log_pdf(self, p_aug) -> float:
        """Sum of block log-densities. Short-circuits on -inf."""
        p_aug = np.asarray(p_aug, dtype=float)
        total = 0.0
        for b in self.blocks:
            lp = b.log_pdf(p_aug[list(b.indices)])
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return float(total)

    def sample(self, rng, n: int = 1) -> np.ndarray:
        """(n, D) draws from the prior. Used by SBC."""
        out = np.empty((n, self.n_params), dtype=float)
        for b in self.blocks:
            out[:, list(b.indices)] = b.sample(rng, n)
        return out

    def validate_against(self, param_names: Sequence[str]) -> None:
        """
        Guard against silent off-by-one in block ordering. An index
        mismatch here would be invisible in the output and catastrophic
        in interpretation, so it is checked explicitly.
        """
        got, want = self.param_names, tuple(param_names)
        if got != want:
            raise ValueError(
                "PriorSpec parameter layout does not match the model's:\n"
                f"  prior : {got}\n  model : {want}"
            )


# =============================================================================
# Generic 1-D building blocks (used by the named options below)
# =============================================================================

def _uniform_funcs(lb, ub):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    def sample(rng, n):
        return rng.uniform(lb, ub, size=(n, len(lb)))

    def log_pdf(x):
        x = np.asarray(x, float)
        if np.any(x < lb) or np.any(x > ub):
            return -np.inf
        return float(-np.sum(np.log(ub - lb)))

    return sample, log_pdf


def _loguniform_funcs(lb, ub):
    """Uniform on log scale. p(x) = 1 / (x * log(ub/lb))."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    if np.any(lb <= 0):
        raise ValueError("log-uniform prior requires strictly positive lb")
    llb, lub = np.log(lb), np.log(ub)

    def sample(rng, n):
        return np.exp(rng.uniform(llb, lub, size=(n, len(lb))))

    def log_pdf(x):
        x = np.asarray(x, float)
        if np.any(x < lb) or np.any(x > ub):
            return -np.inf
        return float(-np.sum(np.log(x)) - np.sum(np.log(lub - llb)))

    return sample, log_pdf


def _beta_funcs(lb, ub, a, b):
    """Beta(a,b) rescaled onto [lb, ub]."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    width = ub - lb

    def sample(rng, n):
        u = rng.beta(a, b, size=(n, len(lb)))
        return lb + u * width

    def log_pdf(x):
        x = np.asarray(x, float)
        if np.any(x <= lb) or np.any(x >= ub):
            return -np.inf
        u = (x - lb) / width
        return float(np.sum(stats.beta.logpdf(u, a, b) - np.log(width)))

    return sample, log_pdf


def _truncnorm_funcs(lb, ub, mu, sd):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    a, b = (lb - mu) / sd, (ub - mu) / sd

    def sample(rng, n):
        return stats.truncnorm.rvs(a, b, loc=mu, scale=sd,
                                   size=(n, len(lb)), random_state=rng)

    def log_pdf(x):
        x = np.asarray(x, float)
        if np.any(x < lb) or np.any(x > ub):
            return -np.inf
        return float(np.sum(stats.truncnorm.logpdf(x, a, b, loc=mu, scale=sd)))

    return sample, log_pdf


def _gamma_funcs(lb, ub, shape, scale):
    """Gamma(shape, scale) truncated to [lb, ub]."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    lo = stats.gamma.cdf(lb, shape, scale=scale)
    hi = stats.gamma.cdf(ub, shape, scale=scale)

    def sample(rng, n):
        u = rng.uniform(lo, hi, size=(n, len(lb)))
        return stats.gamma.ppf(u, shape, scale=scale)

    def log_pdf(x):
        x = np.asarray(x, float)
        if np.any(x < lb) or np.any(x > ub):
            return -np.inf
        return float(np.sum(stats.gamma.logpdf(x, shape, scale=scale)
                            - np.log(hi - lo)))

    return sample, log_pdf


# =============================================================================
# PRIOR OPTIONS -- volume fractions (ve, vp)
# =============================================================================

def make_volfrac_flat_masked(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Independent Uniform on each bound, with the ve+vp<=1 feasibility
    mask applied separately by the model's own `constraints` callable.

    [SOURCE: current NLLS behaviour]  Reinterprets exactly what the
    existing optimiser already does. Because it is flat, the posterior
    stays directly comparable to bootstrap replicates -- which is what
    makes it the right choice for the row-1 validation step, where any
    Bayesian-vs-NLLS difference must be attributable to the estimation
    method and not to a new belief smuggled in via the prior.

    NOTE on the constraint: with the shipped bounds (ve,vp both capped
    at 0.5) the maximum achievable sum is exactly 1.0, so ve+vp<=1 only
    binds at a single corner point and is effectively inert.
    """
    sample, log_pdf = _uniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Uniform(nm, lower=float(l), upper=float(u))
                for nm, l, u in zip(names_, lb, ub)]

    return ParamBlockPrior("volfrac", "current NLLS behaviour",
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def make_volfrac_beta(indices, names, lb, ub, a=2.0, b=2.0) -> ParamBlockPrior:
    """
    Beta(2,2) rescaled onto each parameter's own [lb, ub].

    [SOURCE: Mittermeier2019]  They state only "weakly informative" and
    validate it via a prior predictive check (their Appendix A), not via
    any argument about boundary behaviour.

    Mechanically the rescaled density is proportional to
    (x-lb)(ub-x), which is EXACTLY zero at both edges, so it applies
    smooth downward pressure approaching either bound. Beta(2,2) is
    worth ~2 pseudo-observations (1 per side) -- easily overwhelmed by
    30+ informative timepoints, which is what makes it usable as a
    *differential* probe of how much the likelihood really constrains a
    parameter near its wall, rather than as a hard barrier.
    """
    sample, log_pdf = _beta_funcs(lb, ub, a, b)

    def build_pymc(names_, ctx):
        import pymc as pm
        out = []
        for nm, l, u in zip(names_, lb, ub):
            raw = pm.Beta(f"{nm}_raw", alpha=a, beta=b)
            out.append(pm.Deterministic(nm, float(l) + raw * float(u - l)))
        return out

    return ParamBlockPrior("volfrac", "Mittermeier2019",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


def make_volfrac_conditional(indices, names, lb, ub, total=1.0) -> ParamBlockPrior:
    """
    Two-step construction: the FIRST listed parameter is drawn flat over
    its own bounds; the SECOND is drawn flat over whatever room the
    constraint leaves.

        vp ~ U[lb, ub]
        ve | vp ~ U[lb, min(ub, total - vp)]

    [SOURCE: Orton2007]  His K^trans | k_ep ~ U[0, k_ep] enforces
    ve = K^trans/k_ep < 1 by construction, so a proposal can never land
    outside the feasible region and be wasted.

    IMPORTANT: this is NOT the same joint distribution as
    make_volfrac_flat_masked. Multiplying the two factors gives a joint
    density proportional to 1/(available room), which is not constant --
    it favours the corner where the first parameter is large. "Uniform
    on the joint" does not imply "uniform on either marginal"; for the
    flat-masked joint the first parameter's marginal would have to be
    triangular, not flat. Both are valid priors, they are just different.

    With the shipped bounds this block is near-inert (see the note in
    make_volfrac_flat_masked) and is provided mainly for completeness
    and for use with widened bounds.
    """
    if len(indices) != 2:
        raise ValueError("conditional volfrac prior expects exactly 2 params")
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    def sample(rng, n):
        first = rng.uniform(lb[0], ub[0], size=n)
        hi2 = np.minimum(ub[1], total - first)
        hi2 = np.maximum(hi2, lb[1] + 1e-12)
        second = rng.uniform(lb[1], hi2)
        return np.column_stack([first, second])

    def log_pdf(x):
        x = np.asarray(x, float)
        if x[0] < lb[0] or x[0] > ub[0]:
            return -np.inf
        hi2 = min(ub[1], total - x[0])
        if x[1] < lb[1] or x[1] > hi2 or hi2 <= lb[1]:
            return -np.inf
        return float(-np.log(ub[0] - lb[0]) - np.log(hi2 - lb[1]))

    def build_pymc(names_, ctx):
        import pymc as pm
        import pytensor.tensor as pt
        first = pm.Uniform(names_[0], lower=float(lb[0]), upper=float(ub[0]))
        hi2 = pt.minimum(float(ub[1]), total - first)
        second = pm.Uniform(names_[1], lower=float(lb[1]), upper=hi2)
        return [first, second]

    return ParamBlockPrior("volfrac", "Orton2007",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


# =============================================================================
# PRIOR OPTIONS -- rate parameters (fp, ps; Ktrans)
# =============================================================================

def make_rate_loguniform(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Uniform on log scale.

    [SOURCE: consistent with existing log_dims]  This is the same belief
    the pipeline's LHS search already encodes via
    KineticModelSpec.log_dims -- carried through to the Bayesian step
    instead of being abandoned at the boundary between search and
    inference. Requires no new numbers to be chosen.

    Same mathematical form as the Jeffreys prior used for a noise scale:
    both are "flat in log", appropriate for a positive quantity that
    could plausibly span decades.
    """
    sample, log_pdf = _loguniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        out = []
        for nm, l, u in zip(names_, lb, ub):
            lg = pm.Uniform(f"log_{nm}", lower=float(np.log(l)),
                            upper=float(np.log(u)))
            out.append(pm.Deterministic(nm, pm.math.exp(lg)))
        return out

    return ParamBlockPrior("rates", "log_dims convention",
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def make_rate_uniform(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Uniform on the LINEAR scale.

    [SOURCE: Sitek2016, Zhou2013 (vague priors)]

    Caveat worth stating: over a range spanning decades this is not the
    "no preference" option it appears to be. For [0.001, 1], over 90% of
    the mass sits in the top decade. Sitek and Zhou got away with it
    because their PET rate constants had literature bounds spanning less
    than one decade (e.g. K1 in [0.1, 0.5]), where the linear/log
    distinction barely matters. fp/ps span more than that.
    """
    sample, log_pdf = _uniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Uniform(nm, lower=float(l), upper=float(u))
                for nm, l, u in zip(names_, lb, ub)]

    return ParamBlockPrior("rates", "Sitek2016/Zhou2013",
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def make_rate_gamma(indices, names, lb, ub, shape=1.1, scale=None) -> ParamBlockPrior:
    """
    Gamma(shape, scale) truncated to the model bounds.

    [SOURCE: Mittermeier2019]  Their Gamma(1.1, rate=500) was calibrated
    to K^trans in min^-1 over the QIBA phantom range. It is NOT a
    drop-in here: fp/ps are in s^-1 over different bounds, so `scale`
    must be recalibrated. Default below sets the mean to the midpoint of
    each parameter's own bounds, which is a reasonable neutral starting
    point but is a CHOICE, not an inherited literature value.

    With shape just above 1 the density falls off nearly exponentially
    from a small positive mode: it actively suppresses implausibly large
    values, unlike log-uniform which gives every decade equal weight all
    the way to the box bound.
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    if scale is None:
        scale = ((lb + ub) / 2.0) / shape
    scale = np.asarray(scale, float)
    sample, log_pdf = _gamma_funcs(lb, ub, shape, scale)

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Truncated(f"{nm}_t",
                             pm.Gamma.dist(alpha=shape, beta=1.0 / float(sc)),
                             lower=float(l), upper=float(u))
                for nm, l, u, sc in zip(names_, lb, ub, np.atleast_1d(scale))]

    return ParamBlockPrior("rates", "Mittermeier2019 (recalibrated)",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


def make_rate_truncnorm(indices, names, lb, ub, mu, sd) -> ParamBlockPrior:
    """
    Truncated normal centred on a literature value.

    [SOURCE: Scannell2019 (fixed-mean variant), Zhou2013 (informed)]

    The most opinionated option, and the one most in tension with the
    scientific question. If a parameter is weakly identified in a given
    voxel this hands back something close to the chosen literature
    value, which can read as "the data confirm the textbook number" when
    the prior did most of the work. Run it as a clearly-labelled second
    experiment ALONGSIDE a flat default, never as the default.
    """
    sample, log_pdf = _truncnorm_funcs(lb, ub, mu, sd)
    mu_a, sd_a = np.atleast_1d(mu), np.atleast_1d(sd)

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Truncated(f"{nm}_t",
                             pm.Normal.dist(mu=float(m), sigma=float(s)),
                             lower=float(l), upper=float(u))
                for nm, l, u, m, s in zip(names_, lb, ub, mu_a, sd_a)]

    return ParamBlockPrior("rates", "Scannell2019/Zhou2013",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


# =============================================================================
# PRIOR OPTIONS -- delay
# =============================================================================

def make_delay_uniform(index, lb=-10.0, ub=30.0) -> ParamBlockPrior:
    """
    Uniform over the same bounds fit_with_delay_continuous already
    searches.

    [SOURCE: Orton2007]  He used flat on [0, T_U]. His contribution was
    the TREATMENT of the delay (marginalise it out rather than estimate
    it), not the prior's shape -- see bayes.DELAY_MARGINALIZED.

    A zero-centred Gaussian would only be justified by a real belief
    that most voxels cluster near zero delay relative to the AIF
    sampling site; absent that, do not manufacture informativeness.
    """
    sample, log_pdf = _uniform_funcs([lb], [ub])

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Uniform(names_[0], lower=float(lb), upper=float(ub))]

    return ParamBlockPrior("delay", "Orton2007",
                           (index,), ("tau_delay",), True,
                           build_pymc, sample, log_pdf)


# =============================================================================
# PRIOR OPTIONS -- water exchange (tau_b, tau_i)
# =============================================================================

def make_tau_uniform(indices, names, lb, ub) -> ParamBlockPrior:
    """
    LINEAR-uniform over the relax_cfg bounds.

    [SOURCE: no literature precedent -- proposal]

    NOT the default -- make_tau_flat (log-uniform) is. This exists as a
    deliberate alternative, and empirically it samples BETTER for tau_i.

    Why it helps, mechanically. As tau_i -> 0 the SSM3 three-pool model
    collapses to the fast-exchange limit, where the signal stops
    depending on tau_i at all -- so the likelihood is genuinely FLAT in
    that region. Under a log-uniform prior the sampler works in
    log(tau_i) and spends equal time per DECADE, which over the shipped
    [1e-3, 2.0] bounds puts roughly half its effort in that degenerate
    flat zone below ~0.03 s. That flat-at-one-end / curved-at-the-other
    geometry is a funnel, and it is what drives NUTS divergences here
    (646/800 observed on row 4 with the log-uniform prior).

    Linear-uniform makes [1e-3, 0.03] only ~1.4% of the prior range, so
    the sampler barely visits the funnel throat. Pair it with a raised
    tau_i_lb (~0.03, the floor of reported literature values) to remove
    the degenerate region outright rather than merely down-weighting it.

    NOTE the deliberate trade-off: linear-uniform over a multi-decade
    range is NOT the "no preference" option -- over [1e-3, 2.0] more
    than half its mass sits above tau_i = 1.0 s, above most reported
    values. Run it ALONGSIDE make_tau_flat so prior-driven behaviour
    stays separable from data-driven behaviour.
    """
    sample, log_pdf = _uniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        # Sample tau DIRECTLY -- no log transform. Must match the
        # linear _uniform_funcs above; a log-transformed build_pymc here
        # (copied from make_tau_flat) would have PyMC sampling
        # log-uniformly while log_pdf scored linear-uniform, i.e. the two
        # halves of the same prior silently disagreeing.
        return [pm.Uniform(nm, lower=float(l), upper=float(u))
                for nm, l, u in zip(names_, lb, ub)]

    # Source string must NOT be "proposal (flat)": _pymc_initvals keys
    # off that exact string to decide whether to write iv["log_tau_i"]
    # or iv["tau_i"]. This block creates an RV named tau_i, so it needs
    # to fall through to the plain-name branch.
    return ParamBlockPrior("tau", "proposal (linear-uniform)",
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def make_tau_flat(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Log-uniform over the relax_cfg bounds.

    [SOURCE: no literature precedent -- proposal]

    DEFAULT, for two reasons that are worth keeping separate:
      1. It will not mask non-identifiability. This is the essential
         one, and it has nothing to do with the bootstrap: a flat prior
         lets a posterior that collapses back onto the prior be
         *visible* as such. Under an informative prior the same
         "learned nothing" collapse looks like a confident,
         literature-consistent answer.
      2. It keeps the posterior directly comparable to bootstrap
         replicates. This is a convenience for the validation phase
         only, NOT a general requirement -- the bootstrap is itself
         approximate, and the approximation is weakest exactly on the
         ridge-shaped surfaces (tau_b, possibly tau_i) where the
         comparison would matter most.

    Log-uniform rather than linear-uniform because tau_i spans three
    decades (1e-3 to 2.0 s), the same reasoning as fp/ps.
    """
    sample, log_pdf = _loguniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        out = []
        for nm, l, u in zip(names_, lb, ub):
            lg = pm.Uniform(f"log_{nm}", lower=float(np.log(l)),
                            upper=float(np.log(u)))
            out.append(pm.Deterministic(nm, pm.math.exp(lg)))
        return out

    return ParamBlockPrior("tau", "proposal (flat)",
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def make_tau_ordered(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Enforces tau_b < tau_i via nested truncation:
        tau_b ~ LogU[tau_b_lb, tau_b_ub]
        tau_i | tau_b ~ LogU[max(tau_i_lb, tau_b), tau_i_ub]

    [SOURCE: Zhou2013 chained-truncation pattern]  They enforced
    theta_1 <= theta_2 <= theta_3 the same way.

    Unlike the ve+vp constraint, this one is NOT inert: the shipped
    bounds overlap (tau_b in [1e-4, 0.01], tau_i in [1e-3, 2.0]), so
    tau_b > tau_i IS reachable and ruling it out changes the prior.

    ONLY use if the physiology genuinely supports the ordering across
    all tissue in the cohort. If it holds only usually, this zeroes out
    posterior mass in a region that may contain the truth for some
    voxels. Requires exactly 2 params (fit_tau_b=True).
    """
    if len(indices) != 2:
        raise ValueError("ordered tau prior requires fit_tau_b=True (2 params)")
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    def sample(rng, n):
        tb = np.exp(rng.uniform(np.log(lb[0]), np.log(ub[0]), size=n))
        lo_i = np.maximum(lb[1], tb)
        ti = np.exp(rng.uniform(np.log(lo_i), np.log(ub[1])))
        return np.column_stack([tb, ti])

    def log_pdf(x):
        x = np.asarray(x, float)
        tb, ti = x[0], x[1]
        if tb < lb[0] or tb > ub[0]:
            return -np.inf
        lo_i = max(lb[1], tb)
        if ti < lo_i or ti > ub[1]:
            return -np.inf
        return float(-np.log(tb) - (np.log(ub[0]) - np.log(lb[0]))
                     - np.log(ti) - (np.log(ub[1]) - np.log(lo_i)))

    def build_pymc(names_, ctx):
        import pymc as pm
        import pytensor.tensor as pt
        lg_b = pm.Uniform(f"log_{names_[0]}", lower=float(np.log(lb[0])),
                          upper=float(np.log(ub[0])))
        tau_b = pm.Deterministic(names_[0], pm.math.exp(lg_b))
        lo_i = pt.maximum(float(np.log(lb[1])), lg_b)
        lg_i = pm.Uniform(f"log_{names_[1]}", lower=lo_i,
                          upper=float(np.log(ub[1])))
        tau_i = pm.Deterministic(names_[1], pm.math.exp(lg_i))
        return [tau_b, tau_i]

    return ParamBlockPrior("tau", "Zhou2013 pattern",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


def make_tau_i_informed(indices, names, lb, ub, mu, sd) -> ParamBlockPrior:
    """
    Truncated normal on tau_i centred on a published breast value.

    [SOURCE: Scannell2019 pattern -- no tau_i precedent exists]

    MUST NOT BE THE DEFAULT. Under this prior a non-identifiable tau_i
    produces a posterior that looks like a confident, literature-
    consistent answer rather than an obvious "learned nothing" flat
    band. It asks a different, narrower question than the flat default
    ("what does this voxel say, pulled toward the population") and
    should be run as a labelled second experiment.
    """
    sample, log_pdf = _truncnorm_funcs(lb, ub, mu, sd)
    mu_a, sd_a = np.atleast_1d(mu), np.atleast_1d(sd)

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.Truncated(f"{nm}_t",
                             pm.Normal.dist(mu=float(m), sigma=float(s)),
                             lower=float(l), upper=float(u))
                for nm, l, u, m, s in zip(names_, lb, ub, mu_a, sd_a)]

    return ParamBlockPrior("tau", "Scannell2019 pattern (informed)",
                           tuple(indices), tuple(names), False,
                           build_pymc, sample, log_pdf)


# =============================================================================
# PRIOR OPTIONS -- noise parameters
# =============================================================================

def make_sigma_jeffreys(index, lb=1e-6, ub=1e6) -> ParamBlockPrior:
    """
    Jeffreys prior on the noise scale: p(sigma^2) proportional to
    1/sigma^2, which in terms of sigma is p(sigma) proportional to
    1/sigma -- i.e. flat in log(sigma).

    [SOURCE: Orton2007 section 2.3.2]

    Improper in principle; made proper here by truncating to [lb, ub],
    which is also what lets it be sampled directly. Orton instead
    integrates sigma out ANALYTICALLY (his Eq. 8) -- the gamma-function
    integral turns exp(-chi2/2sigma^2) into chi2^(-N/2), which behaves
    like a Student-t likelihood in theta without ever assuming one.
    See bayes.GAUSS_MARGINALIZED_SIGMA for that route.
    """
    sample, log_pdf = _loguniform_funcs([lb], [ub])

    def build_pymc(names_, ctx):
        import pymc as pm
        lg = pm.Uniform(f"log_{names_[0]}", lower=float(np.log(lb)),
                        upper=float(np.log(ub)))
        return [pm.Deterministic(names_[0], pm.math.exp(lg))]

    return ParamBlockPrior("noise", "Orton2007",
                           (index,), ("sigma",), True,
                           build_pymc, sample, log_pdf)


def make_sigma_invgamma(index, alpha=1e-3, beta=1e-3) -> ParamBlockPrior:
    """
    sigma^2 ~ InverseGamma(1e-3, 1e-3).

    [SOURCE: Scannell2019, Zhou2013]

    A PROPER approximation to Jeffreys. The distinction matters only if
    marginal likelihoods (Bayes factors, e.g. SSM3 vs FXL) are wanted
    later: those are ill-defined when any prior is improper.
    """
    def sample(rng, n):
        var = stats.invgamma.rvs(alpha, scale=beta, size=n, random_state=rng)
        return np.sqrt(var).reshape(n, 1)

    def log_pdf(x):
        s = float(np.asarray(x).ravel()[0])
        if s <= 0:
            return -np.inf
        # change of variables sigma^2 -> sigma contributes log(2*sigma)
        return float(stats.invgamma.logpdf(s ** 2, alpha, scale=beta)
                     + np.log(2.0 * s))

    def build_pymc(names_, ctx):
        import pymc as pm
        var = pm.InverseGamma(f"{names_[0]}_sq", alpha=alpha, beta=beta)
        return [pm.Deterministic(names_[0], pm.math.sqrt(var))]

    return ParamBlockPrior("noise", "Scannell2019/Zhou2013",
                           (index,), ("sigma",), False,
                           build_pymc, sample, log_pdf)


def make_sigma_lognormal(index, mu=0.0, sd=1.0) -> ParamBlockPrior:
    """
    sigma ~ LogNormal(mu, sd).

    [SOURCE: Mittermeier2019]

    Genuinely weakly informative rather than maximally uninformative:
    the median sits at exp(mu) IN THE UNITS OF THE DATA. Mittermeier's
    mu=0 means a median sigma of 1 mmol/L, which is meaningful only in
    their concentration units. If used on signal-space data here, mu
    MUST be recalibrated to a plausible signal-noise magnitude.
    """
    def sample(rng, n):
        return rng.lognormal(mu, sd, size=(n, 1))

    def log_pdf(x):
        s = float(np.asarray(x).ravel()[0])
        if s <= 0:
            return -np.inf
        return float(stats.lognorm.logpdf(s, sd, scale=np.exp(mu)))

    def build_pymc(names_, ctx):
        import pymc as pm
        return [pm.LogNormal(names_[0], mu=mu, sigma=sd)]

    return ParamBlockPrior("noise", "Mittermeier2019",
                           (index,), ("sigma",), False,
                           build_pymc, sample, log_pdf)


def make_student_t_params(indices, sigma_lb=1e-6, sigma_ub=1e6) -> ParamBlockPrior:
    """
    Student-t noise parameters: (sigma, inv_nu).

    [SOURCE: Zhou2013]

    inv_nu = 1/nu with a Uniform[0, 0.5) prior. The reparameterisation
    matters: nu itself lives on (0, inf) with no natural upper cutoff,
    whereas 1/nu compactifies it so that the GAUSSIAN LIMIT sits exactly
    at the boundary inv_nu = 0. The sampler can then explore all the way
    from "indistinguishable from Gaussian" to "quite heavy-tailed"
    without an arbitrary cap on tail weight.

    Implemented with inv_nu in [1/50, 0.5] rather than [0, 0.5] so that
    nu stays finite and the density is well defined at the edge.
    """
    sig_s, sig_lp = _loguniform_funcs([sigma_lb], [sigma_ub])
    NU_MAX = 50.0
    inv_lb, inv_ub = 1.0 / NU_MAX, 0.5

    def sample(rng, n):
        s = sig_s(rng, n)
        iv = rng.uniform(inv_lb, inv_ub, size=(n, 1))
        return np.column_stack([s[:, 0], iv[:, 0]])

    def log_pdf(x):
        x = np.asarray(x, float)
        lp = sig_lp([x[0]])
        if not np.isfinite(lp):
            return -np.inf
        if x[1] < inv_lb or x[1] > inv_ub:
            return -np.inf
        return float(lp - np.log(inv_ub - inv_lb))

    def build_pymc(names_, ctx):
        import pymc as pm
        lg = pm.Uniform("log_sigma", lower=float(np.log(sigma_lb)),
                        upper=float(np.log(sigma_ub)))
        sigma = pm.Deterministic("sigma", pm.math.exp(lg))
        inv_nu = pm.Uniform("inv_nu", lower=inv_lb, upper=inv_ub)
        return [sigma, inv_nu]

    return ParamBlockPrior("noise", "Zhou2013",
                           tuple(indices), ("sigma", "inv_nu"), False,
                           build_pymc, sample, log_pdf)


# =============================================================================
# The assembler -- SINGLE source of truth for block ordering
# =============================================================================

_VOLFRAC_NAMES = ("ve", "vp")

NUISANCE_FLAT_SOURCE = "flat log (NuisanceConfig default)"


def make_nuisance_loguniform(indices, names, lb, ub) -> ParamBlockPrior:
    """
    Flat-in-log prior on the instrument nuisances T10 and B1.

    [SOURCE: refactor Phase 5.2]  The default because the CRLB analysis
    showed an INFORMATIVE prior on B1 is a poor trade when its centre is
    uncertain: at ROI SNR on the ACE schedule it tightened the tau_i bound
    by 0.5 percentage points while a centre wrong by 0.18 (Sung 2013
    right breast 0.82 vs a naive 1.0) added +31% tau_i bias.

    "Flat" is not coordinate-free. Flat in log B1 over (0.5, 1.5) puts
    3x more density at 0.5 than at 1.5. Log is chosen because the NLLS
    fitter, information.py's CRLB and the existing rate priors all work
    in log coordinates, so all four agree on what "uninformative" means.

    T10 must arrive in SECONDS, matching jax_forward.make_predict_mfa --
    not milliseconds like NuisanceConfig. assemble_prior converts.
    """
    sample, log_pdf = _loguniform_funcs(lb, ub)

    def build_pymc(names_, ctx):
        import pymc as pm
        out = []
        for nm, l, u in zip(names_, lb, ub):
            lg = pm.Uniform(f"log_{nm}", lower=float(np.log(l)),
                            upper=float(np.log(u)))
            out.append(pm.Deterministic(nm, pm.math.exp(lg)))
        return out

    return ParamBlockPrior("nuisance", NUISANCE_FLAT_SOURCE,
                           tuple(indices), tuple(names), True,
                           build_pymc, sample, log_pdf)


def _nuisance_block_spec(nuisance):
    """
    (names, lb, ub) for the free nuisances, in make_predict_mfa's order
    (T10 before B1), T10 converted ms -> s. Shared by assemble_prior and
    bayes.build_param_layout so the two cannot disagree on order or units.
    """
    names, lb, ub = [], [], []
    if nuisance is None:
        return names, np.array(lb), np.array(ub)
    if nuisance.fit_T10:
        names.append("T10")
        lb.append(nuisance.T10_bounds_ms[0] / 1000.0)
        ub.append(nuisance.T10_bounds_ms[1] / 1000.0)
    if nuisance.fit_B1:
        names.append("B1")
        lb.append(nuisance.B1_bounds[0])
        ub.append(nuisance.B1_bounds[1])
    return names, np.array(lb, float), np.array(ub, float)


def assemble_prior(
    model,
    relax_cfg = None,
    *,
    wx_model     = None,
    delay_mode:  str = "none",
    n_noise:     int = 0,
    noise_names: tuple = (),
    choices:     Optional[dict] = None,
    delay_bounds: tuple = (-10.0, 30.0),
    nuisance     = None,
) -> PriorSpec:
    """
    Build a PriorSpec matching the p_aug layout that build_residual
    produces:

        [ *theta_kin , (tau_delay?) , (tau_b?) , tau_i? ,
          (T10?) , (B1?) , (noise?) ]

    nuisance : NuisanceConfig or None. Free T10/B1 get a flat-in-log block
               (make_nuisance_loguniform), T10 in SECONDS. An informative
               T10_prior/B1_prior is REFUSED rather than silently replaced
               by a flat one: the NLLS fitter honours those priors, and a
               Bayes path that ignored them would disagree with it
               without any warning.

    `model` must ALREADY be the delay-augmented spec when
    delay_mode == 'joint' (use fitting.build_delay_augmented_spec), so
    that tau_delay is the last kinetic parameter.

    choices : dict overriding per-block defaults, e.g.
              {'volfrac': 'beta', 'rates': 'loguniform', 'tau': 'flat',
               'noise': 'jeffreys'}
    """
    choices = dict(choices or {})
    blocks = []

    kin_names = list(model.param_names)
    n_kin = len(kin_names)

    # --- kinetic block: split into volume fractions vs rates ---------
    vf_idx = [i for i, nm in enumerate(kin_names) if nm in _VOLFRAC_NAMES]
    delay_idx = [i for i, nm in enumerate(kin_names) if nm == "tau_delay"]
    rate_idx = [i for i in range(n_kin)
                if i not in vf_idx and i not in delay_idx]

    if vf_idx:
        vf_names = [kin_names[i] for i in vf_idx]
        vf_lb = model.lb[vf_idx]
        vf_ub = model.ub[vf_idx]
        which = choices.get("volfrac", "flat_masked")
        if which == "flat_masked":
            blocks.append(make_volfrac_flat_masked(vf_idx, vf_names, vf_lb, vf_ub))
        elif which == "beta":
            blocks.append(make_volfrac_beta(vf_idx, vf_names, vf_lb, vf_ub))
        elif which == "conditional":
            blocks.append(make_volfrac_conditional(vf_idx, vf_names, vf_lb, vf_ub))
        else:
            raise ValueError(f"unknown volfrac choice {which!r}")

    if rate_idx:
        r_names = [kin_names[i] for i in rate_idx]
        r_lb = model.lb[rate_idx]
        r_ub = model.ub[rate_idx]
        which = choices.get("rates", "loguniform")
        if which == "loguniform":
            blocks.append(make_rate_loguniform(rate_idx, r_names, r_lb, r_ub))
        elif which == "uniform":
            blocks.append(make_rate_uniform(rate_idx, r_names, r_lb, r_ub))
        elif which == "gamma":
            blocks.append(make_rate_gamma(rate_idx, r_names, r_lb, r_ub))
        elif which == "truncnorm":
            mu = choices.get("rates_mu")
            sd = choices.get("rates_sd")
            if mu is None or sd is None:
                raise ValueError(
                    "rates='truncnorm' requires 'rates_mu' and 'rates_sd' "
                    "in choices -- no literature values for fp/ps are "
                    "hard-coded here deliberately."
                )
            blocks.append(make_rate_truncnorm(rate_idx, r_names, r_lb, r_ub, mu, sd))
        else:
            raise ValueError(f"unknown rates choice {which!r}")

    if delay_idx:
        i = delay_idx[0]
        blocks.append(make_delay_uniform(i, float(model.lb[i]), float(model.ub[i])))

    # --- water-exchange block ----------------------------------------
    cursor = n_kin
    # wx models with no free exchange parameters (NXL) contribute no
    # block, and therefore need no relax_cfg either.
    if wx_model is not None and wx_model.n_params > 0:
        if relax_cfg is None:
            raise ValueError("wx_model requires relax_cfg")
        if relax_cfg.fit_tau_b:
            tau_idx = [cursor, cursor + 1]
            tau_names = ["tau_b", "tau_i"]
            tau_lb = np.array([relax_cfg.tau_b_lb, relax_cfg.tau_i_lb])
            tau_ub = np.array([relax_cfg.tau_b_ub, relax_cfg.tau_i_ub])
            cursor += 2
        else:
            tau_idx = [cursor]
            tau_names = ["tau_i"]
            tau_lb = np.array([relax_cfg.tau_i_lb])
            tau_ub = np.array([relax_cfg.tau_i_ub])
            cursor += 1

        which = choices.get("tau", "flat")
        if which == "flat":
            blocks.append(make_tau_flat(tau_idx, tau_names, tau_lb, tau_ub))
        elif which == "uniform":
            blocks.append(make_tau_uniform(tau_idx, tau_names, tau_lb, tau_ub))
        elif which == "ordered":
            blocks.append(make_tau_ordered(tau_idx, tau_names, tau_lb, tau_ub))
        elif which == "informed":
            mu = choices.get("tau_i_mu")
            sd = choices.get("tau_i_sd")
            if mu is None or sd is None:
                raise ValueError(
                    "tau='informed' requires 'tau_i_mu' and 'tau_i_sd' "
                    "in choices -- deliberately not defaulted."
                )
            if relax_cfg.fit_tau_b:
                raise ValueError(
                    "tau='informed' currently covers tau_i only; use "
                    "fit_tau_b=False or extend make_tau_i_informed."
                )
            blocks.append(make_tau_i_informed(tau_idx, tau_names,
                                              tau_lb, tau_ub, mu, sd))
        else:
            raise ValueError(f"unknown tau choice {which!r}")

    # --- nuisance block (T10, B1) ---------------------------------------
    nu_names, nu_lb, nu_ub = _nuisance_block_spec(nuisance)
    if nu_names:
        if nuisance.T10_prior is not None or nuisance.B1_prior is not None:
            raise NotImplementedError(
                "informative T10/B1 priors are not yet supported in the "
                "Bayes path: bayes_blackjax has no truncated-normal or "
                "lognormal transform family. Leave T10_prior/B1_prior as "
                "None (flat), or see refactor Phase 5.4.")
        which = choices.get("nuisance", "loguniform")
        if which != "loguniform":
            raise ValueError(f"unknown nuisance choice {which!r}")
        nu_idx = list(range(cursor, cursor + len(nu_names)))
        blocks.append(make_nuisance_loguniform(nu_idx, nu_names, nu_lb, nu_ub))
        cursor += len(nu_names)

    # --- noise block --------------------------------------------------
    if n_noise > 0:
        which = choices.get("noise", "jeffreys")
        idxs = list(range(cursor, cursor + n_noise))
        if n_noise == 1:
            if which == "jeffreys":
                blocks.append(make_sigma_jeffreys(idxs[0]))
            elif which == "invgamma":
                blocks.append(make_sigma_invgamma(idxs[0]))
            elif which == "lognormal":
                mu = choices.get("sigma_mu", 0.0)
                sd = choices.get("sigma_sd", 1.0)
                blocks.append(make_sigma_lognormal(idxs[0], mu, sd))
            else:
                raise ValueError(f"unknown noise choice {which!r}")
        elif n_noise == 2:
            blocks.append(make_student_t_params(idxs))
        else:
            raise ValueError(f"unsupported n_noise={n_noise}")

    return PriorSpec(tuple(blocks))