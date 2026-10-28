
"""
hierarchical_lunn.py
====================
Lunn et al. (2013) two-stage hierarchical fit, stage 2.

Stage 1 (already done per lesion, by fit_curve_bayes / fit_curve_bayes_batched)
gives draws from
        p1_post(theta_i) proportional to  L_i(theta_i) * p1(theta_i)
Stage 2 targets the hierarchical posterior
        theta_i ~ N(m_i, Sigma),   m_i = X_i B,   B, Sigma ~ hyperprior
by proposing theta_i* uniformly from lesion i's SAVED stage-1 draws
(an independence sampler whose proposal density IS p1_post). The
likelihood then cancels from the Metropolis-Hastings ratio, leaving

        r = [ N(theta* | m_i, Sigma) / p1(theta*) ]
            -------------------------------------------
            [ N(theta  | m_i, Sigma) / p1(theta ) ]                 (Lunn Eq. 7)

so the ONLY thing stage 2 needs from stage 1, besides the draws, is p1
evaluated at those draws -- on the POPULATION scale
theta = (logit ve, logit vp, log fp, log ps), including the change-of-
variables Jacobian. That is what Stage1Samples.log_p1 holds.

Two consequences worth stating, because they decide what stage 2 can do:

* Lesions fitted under DIFFERENT stage-1 priors can be pooled. Each
  lesion's ratio divides by its own p1_i; nothing requires p1_i = p1_j.
  A stage-1 prior that is flat on the population scale (log-uniform
  rates, say) contributes log_p1 = 0 and needs no correction at all.
* theta_i can only ever take one of its K saved values. Stage 2 is exact
  only as K -> infinity, and it degenerates when the population model
  concentrates where few stage-1 draws live -- e.g. dividing by a
  Beta(2,2) prior that vanishes at a bound a lesion is pinned against.
  fit_stage2 therefore reports, per lesion, the acceptance rate and the
  number of DISTINCT draws visited. Low numbers there invalidate that
  lesion's contribution no matter how healthy the hyperparameter chains
  look.

The log_p1_theta factory below writes down the two prior families this
project uses. Their algebra is the form checked numerically in
verify/check_09_stage1_prior_reuse.py Part 1 -- including that
bayes_priors.make_volfrac_beta is Beta(2,2) rescaled onto [lb, ub], NOT
Beta(2,2) on [0, 1]. Before using it on REAL draws, confirm it against
the prior object the stage-1 run actually used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, solve_triangular

THETA_NAMES = ("logit_ve", "logit_vp", "log_fp", "log_ps")


# =============================================================================
# the population-scale transform and the stage-1 prior density on it
# =============================================================================

def to_theta(X: np.ndarray) -> np.ndarray:
    """(..., 4) of (ve, vp, fp, ps) -> (logit ve, logit vp, log fp, log ps)."""
    X = np.asarray(X, float)
    ve, vp, fp, ps = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
    return np.stack([np.log(ve / (1.0 - ve)), np.log(vp / (1.0 - vp)),
                     np.log(fp), np.log(ps)], axis=-1)


def from_theta(TH: np.ndarray) -> np.ndarray:
    """Inverse of to_theta."""
    TH = np.asarray(TH, float)
    a, b, c, d = TH[..., 0], TH[..., 1], TH[..., 2], TH[..., 3]
    sig = lambda s: 1.0 / (1.0 + np.exp(-s))
    return np.stack([sig(a), sig(b), np.exp(c), np.exp(d)], axis=-1)


def log_p1_theta(choices: dict, lb, ub, *, beta_ab=(2.0, 2.0)):
    """
    Build `log_p1(X) -> (...,)`, the stage-1 prior density ON THE POPULATION
    SCALE, evaluated from CONSTRAINED draws X = (ve, vp, fp, ps).

    p1(theta) = p1_x(x(theta)) * |dx/dtheta|, with
        |dx/dtheta| = ve(1-ve) * vp(1-vp) * fp * ps

    choices : {'volfrac': 'flat_masked' | 'beta', 'rates': 'uniform' | 'loguniform'}
              -- the same keys BayesConfig.prior_choices uses.
    lb, ub  : length-4 model bounds, in (ve, vp, fp, ps) order
              (np.asarray(TCXM.lb), np.asarray(TCXM.ub)).

    Normalising constants are dropped: stage 2 only ever uses differences
    of log_p1 within one lesion, where any constant cancels. Out-of-bounds
    draws get -inf, which makes them unproposable.
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    vol, rat = choices.get("volfrac", "flat_masked"), choices.get("rates", "uniform")
    if vol not in ("flat_masked", "beta"):
        raise ValueError(f"volfrac {vol!r}: expected 'flat_masked' or 'beta'")
    if rat not in ("uniform", "loguniform"):
        raise ValueError(f"rates {rat!r}: expected 'uniform' or 'loguniform'")
    a_, b_ = beta_ab

    def log_p1(X):
        X = np.asarray(X, float)
        ve, vp, fp, ps = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
        inside = np.all((X >= lb) & (X <= ub), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Jacobian of the logit / log transform
            out = (np.log(ve) + np.log1p(-ve) + np.log(vp) + np.log1p(-vp)
                   + np.log(fp) + np.log(ps))
            if vol == "beta":
                # Beta(a,b) RESCALED onto [lb, ub]: density proportional to
                # (v - lb)^(a-1) (ub - v)^(b-1)
                for j, v in ((0, ve), (1, vp)):
                    out = out + ((a_ - 1.0) * np.log(v - lb[j])
                                 + (b_ - 1.0) * np.log(ub[j] - v))
            if rat == "loguniform":
                # p1_x(rate) proportional to 1/rate, which exactly cancels the
                # Jacobian above: the density on log(rate) is FLAT. (This is
                # the 0.00 "non-flatness" check_09 Part 1 reports for run C.)
                out = out - np.log(fp) - np.log(ps)
            # 'uniform' rates have a constant p1_x, so they add nothing and the
            # density on log(rate) keeps the Jacobian factor, proportional to
            # the rate itself (check_09's 0.99 non-flatness for run B).
            # 'flat_masked' volfracs likewise add nothing beyond the Jacobian.
        return np.where(inside, out, -np.inf)

    return log_p1


# =============================================================================
# stage-1 input
# =============================================================================

@dataclass
class Stage1Samples:
    """
    draws  : (L, K, D) stage-1 posterior draws ON THE POPULATION SCALE theta.
             Thin each lesion's chains down to K roughly independent draws;
             correlated draws cost acceptance, not correctness.
    log_p1 : (L, K) stage-1 prior log-density on theta at those draws. Zeros
             for a prior already flat on the population scale.
    sids   : lesion ids, for reporting.
    """
    draws:  np.ndarray
    log_p1: np.ndarray
    sids:   tuple = ()

    def __post_init__(self):
        self.draws = np.asarray(self.draws, float)
        self.log_p1 = np.asarray(self.log_p1, float)
        if self.draws.ndim != 3:
            raise ValueError(f"draws must be (L, K, D), got {self.draws.shape}")
        if self.log_p1.shape != self.draws.shape[:2]:
            raise ValueError(f"log_p1 {self.log_p1.shape} does not match "
                             f"draws {self.draws.shape[:2]}")
        if not np.all(np.isfinite(self.draws)):
            raise ValueError("draws contain non-finite values")
        bad = ~np.isfinite(self.log_p1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} log_p1 entries are non-finite; "
                             "drop those draws before building Stage1Samples")
        if not self.sids:
            self.sids = tuple(f"L{i:03d}" for i in range(self.draws.shape[0]))

    @property
    def shape(self):
        return self.draws.shape

    @classmethod
    def from_constrained(cls, draws_x: Sequence[np.ndarray], log_p1_fn,
                         sids: Sequence[str] = ()):
        """
        Build from per-lesion CONSTRAINED draws (ve, vp, fp, ps), applying
        log_p1_fn (from log_p1_theta) and the transform. Accepts a list so
        lesions may be thinned to the same K separately.
        """
        X = np.stack([np.asarray(d, float) for d in draws_x])
        return cls(draws=to_theta(X), log_p1=log_p1_fn(X), sids=tuple(sids))


# =============================================================================
# stage 2
# =============================================================================

@dataclass
class Stage2Result:
    B:          np.ndarray          # (n_draws, P, D) regression coefficients
    Sigma:      np.ndarray          # (n_draws, D, D)
    theta_idx:  np.ndarray          # (n_draws, L) which stage-1 draw each lesion holds
    accept:     np.ndarray          # (L,) per-lesion acceptance rate
    n_distinct: np.ndarray          # (L,) distinct stage-1 draws visited
    names:      tuple = THETA_NAMES
    coef_names: tuple = ("intercept",)
    meta:       dict = field(default_factory=dict)

    def mu(self, row=0):
        """Posterior draws of one coefficient row, (n_draws, D)."""
        return self.B[:, row, :]

    def sd(self):
        """Posterior draws of the population SDs, (n_draws, D)."""
        return np.sqrt(np.einsum("nii->ni", self.Sigma))

    def corr(self):
        s = self.sd()
        return self.Sigma / (s[:, :, None] * s[:, None, :])

    def summary(self, row=0):
        """Median and 90% interval for that coefficient row and the SDs."""
        out = {}
        for j, nm in enumerate(self.names):
            b = self.B[:, row, j]
            out[nm] = dict(median=float(np.median(b)),
                           q05=float(np.quantile(b, 0.05)),
                           q95=float(np.quantile(b, 0.95)),
                           sd=float(np.std(b)))
            s = self.sd()[:, j]
            out[nm]["pop_sd_median"] = float(np.median(s))
            out[nm]["pop_sd_q05"] = float(np.quantile(s, 0.05))
            out[nm]["pop_sd_q95"] = float(np.quantile(s, 0.95))
        return out

    def degenerate(self, min_distinct=20, min_accept=0.01):
        """Lesions whose theta barely moved -- their contribution is unreliable."""
        bad = (self.n_distinct < min_distinct) | (self.accept < min_accept)
        return [s for s, b in zip(self.meta.get("sids", ()), bad) if b]


def _log_mvn_quad(resid, chol_upper):
    """(theta - m)' Sigma^-1 (theta - m), rowwise, given chol(Sigma) upper."""
    z = solve_triangular(chol_upper, resid.T, lower=False, trans="T")
    return np.einsum("ij,ij->j", z, z)


def fit_stage2(
    s1: Stage1Samples,
    *,
    design: Optional[np.ndarray] = None,
    coef_names: Sequence[str] = ("intercept",),
    n_draws: int = 4000,
    n_tune: int = 1000,
    thin: int = 1,
    seed: int = 0,
    tau0: float = 10.0,
    nu0: Optional[int] = None,
    Psi0: Optional[np.ndarray] = None,
    correct_p1: bool = True,
    theta_sweeps: int = 1,
    names: Optional[Sequence[str]] = None,
) -> Stage2Result:
    """
    Gibbs / Metropolis-within-Gibbs for

        theta_i ~ N(X_i B, Sigma)          i = 1..L
        B[p, :] ~ N(0, tau0^2 I)           independent across p
        Sigma   ~ InverseWishart(nu0, Psi0)

    design     : (L, P) covariate matrix; default a single intercept column.
                 Add a 0/1 column for benign/malignant to get a group
                 difference in B[1, :].
    correct_p1 : divide out the stage-1 prior (Lunn Eq. 7). False is the
                 NEGATIVE CONTROL -- it targets the wrong posterior and
                 exists only to show a test has power to detect that.
    theta_sweeps: independence-MH passes over all lesions per Gibbs sweep.
                  Raise it when acceptance is low.
    names      : names for the D coordinates of theta, used by summary(). The
                 default is THETA_NAMES at D = 4 and theta0..theta{D-1}
                 otherwise -- never a 4-element tuple at D != 4, which would
                 make summary() drop coordinates without saying so.

    Draws are Gibbs-exact for B and Sigma given theta, and Metropolis for
    theta given B, Sigma -- so n_tune still matters and R-hat should be
    checked across seeds.
    """
    L, K, D = s1.draws.shape
    rng = np.random.default_rng(seed)
    logp1 = s1.log_p1 if correct_p1 else np.zeros((L, K))

    X = np.ones((L, 1)) if design is None else np.asarray(design, float)
    if X.shape[0] != L:
        raise ValueError(f"design has {X.shape[0]} rows, expected {L}")
    P = X.shape[1]
    if len(coef_names) != P:
        coef_names = tuple([*coef_names] + [f"coef{p}" for p in range(len(coef_names), P)])
    if names is None:
        names = (THETA_NAMES if D == len(THETA_NAMES)
                 else tuple(f"theta{j}" for j in range(D)))
    elif len(names) != D:
        raise ValueError(f"names has {len(names)} entries, expected D = {D}")
    nu0 = (D + 2) if nu0 is None else int(nu0)
    Psi0 = np.eye(D) if Psi0 is None else np.asarray(Psi0, float)

    # --- initial state: a random draw per lesion, moments from it ------------
    idx = rng.integers(0, K, L)
    theta = s1.draws[np.arange(L), idx]
    B = np.zeros((P, D))
    B[0] = theta.mean(0)
    Sigma = np.cov(theta.T) + 1e-8 * np.eye(D)

    XtX = X.T @ X
    I_PD = np.eye(P * D)
    n_keep = n_draws // thin
    out_B = np.empty((n_keep, P, D))
    out_S = np.empty((n_keep, D, D))
    out_i = np.empty((n_keep, L), dtype=np.int32)
    n_acc = np.zeros(L)
    n_prop = 0
    visited = [set() for _ in range(L)]

    for it in range(n_tune + n_draws):
        chol = cholesky(Sigma, lower=False)
        Sinv = solve_triangular(chol, solve_triangular(chol, np.eye(D), lower=False,
                                                       trans="T"), lower=False)

        # ---- theta_i : independence MH over lesion i's stage-1 draws -------
        m = X @ B                                              # (L, D)
        for _ in range(theta_sweeps):
            cur_q = _log_mvn_quad(theta - m, chol)
            cur = -0.5 * cur_q - logp1[np.arange(L), idx]
            k_new = rng.integers(0, K, L)
            th_new = s1.draws[np.arange(L), k_new]
            new = (-0.5 * _log_mvn_quad(th_new - m, chol)
                   - logp1[np.arange(L), k_new])
            take = np.log(rng.random(L)) < (new - cur)
            idx = np.where(take, k_new, idx)
            theta = np.where(take[:, None], th_new, theta)
            if it >= n_tune:
                n_acc += take
                n_prop += 1
        if it >= n_tune:
            for i in range(L):
                visited[i].add(int(idx[i]))

        # ---- B | Sigma, theta : Gaussian ----------------------------------
        A = np.kron(XtX, Sinv) + I_PD / (tau0 ** 2)
        rhs = np.concatenate([Sinv @ (X[:, p] @ theta) for p in range(P)])
        cA = cholesky(A, lower=False)
        mean_b = solve_triangular(cA, solve_triangular(cA, rhs, lower=False,
                                                       trans="T"), lower=False)
        b = mean_b + solve_triangular(cA, rng.standard_normal(P * D), lower=False)
        B = b.reshape(P, D)

        # ---- Sigma | B, theta : inverse Wishart ---------------------------
        R = theta - X @ B
        Sigma = stats.invwishart.rvs(df=nu0 + L, scale=Psi0 + R.T @ R,
                                     random_state=rng)
        Sigma = np.atleast_2d(Sigma)

        if it >= n_tune and (it - n_tune) % thin == 0:
            j = (it - n_tune) // thin
            out_B[j], out_S[j], out_i[j] = B, Sigma, idx

    return Stage2Result(
        B=out_B, Sigma=out_S, theta_idx=out_i,
        # n_prop counts proposals PER LESION (one per lesion per sweep)
        accept=n_acc / max(n_prop, 1),
        n_distinct=np.array([len(v) for v in visited]),
        names=tuple(names),
        coef_names=tuple(coef_names),
        meta=dict(sids=s1.sids, L=L, K=K, D=D, P=P, correct_p1=bool(correct_p1),
                  n_draws=n_keep, n_tune=n_tune, thin=thin, seed=seed,
                  tau0=tau0, nu0=nu0, theta_sweeps=theta_sweeps),
    )