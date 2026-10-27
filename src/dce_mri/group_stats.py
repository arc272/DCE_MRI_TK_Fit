#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
group_stats.py
==============
Benign-vs-malignant statistics for per-lesion parameter estimates, and
prior-sensitivity analysis by importance reweighting of stored posterior
draws ({sid}_draws.npz from fit_lesions_bayes.py).

Conventions
-----------
AUC         P(X_M > X_B) + 0.5 P(X_M = X_B)  (Mann-Whitney / c-statistic).
            > 0.5 means malignant higher. 95% CI from DeLong's variance.
Cliff's d   2*AUC - 1  (rank effect size in [-1, 1]).
HL shift    Hodges-Lehmann: median of all M - B pairwise differences
            (ratio M / B on the log scale for log-scale parameters).
Size adjustment
            logistic regression of diagnosis (1 = M) on standardised
            log(lesion voxels); the parameter (log-transformed if in
            log_params, then standardised) is added and tested by the
            likelihood-ratio test, 1 df. Reported odds ratio is per SD of
            the parameter, holding size fixed.
Block test  the same, adding k parameters at once (k df), plus repeated
            stratified cross-validated AUC of size-only vs size + block.
Multiplicity
            Holm (family-wise) and Benjamini-Hochberg (FDR) within a family.

Prior sensitivity (reweighting)
-------------------------------
Draws theta_s were sampled under the base prior p_old. Under a new prior,
self-normalised weights  w_s ~ p_new(theta_s) / p_old(theta_s)  give the
new posterior. Only replaced factors enter the ratio (the TCXM priors are
products of 1-D factors), and normalisation constants cancel. Valid only
while p_new is covered by the draws: per-lesion weight ESS and Pareto
k-hat are reported, and unreliable lesions are listed for refitting.

Alternative priors are Normal (or Student-t) on the logit-box z-scale of
dce_mri.hierarchical (log box for fp, ps), so a 'location shift by factor f'
is an exact shift of the prior median in natural units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit


# =============================================================================
# 1. Univariate discrimination
# =============================================================================

def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def auc_mw(xB, xM):
    """AUC = P(M > B) + 0.5 P(tie), and the two-sided Mann-Whitney p."""
    xB, xM = _clean(xB), _clean(xM)
    if xB.size < 2 or xM.size < 2:
        return np.nan, np.nan
    r = stats.mannwhitneyu(xM, xB, alternative="two-sided")
    return float(r.statistic / (xB.size * xM.size)), float(r.pvalue)


def delong_ci(xB, xM, alpha=0.05):
    """DeLong (1988) variance of the empirical AUC; normal CI clipped to [0,1]."""
    xB, xM = _clean(xB), _clean(xM)
    nB, nM = xB.size, xM.size
    psi = (xM[:, None] > xB[None, :]) + 0.5 * (xM[:, None] == xB[None, :])
    auc = psi.mean()
    var = psi.mean(1).var(ddof=1) / nM + psi.mean(0).var(ddof=1) / nB
    zq = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var)
    return float(max(0.0, auc - zq * se)), float(min(1.0, auc + zq * se)), float(se)


def hodges_lehmann(xB, xM, log=False):
    """Median pairwise M - B difference; for log=True the median M/B ratio."""
    xB, xM = _clean(xB), _clean(xM)
    if log:
        xB, xM = np.log(xB[xB > 0]), np.log(xM[xM > 0])
    d = np.median((xM[:, None] - xB[None, :]).ravel())
    return float(np.exp(d)) if log else float(d)


# =============================================================================
# 2. Logistic regression (IRLS), size adjustment, block tests
# =============================================================================

@dataclass
class LogitFit:
    beta: np.ndarray      # (k+1,) intercept first
    se: np.ndarray
    loglik: float
    converged: bool
    separation: bool


def logit_fit(X, y, max_iter=100, tol=1e-10):
    """Unpenalised logistic regression by Newton-Raphson with step halving.
    X: (n, k) WITHOUT intercept (added here). y: 0/1."""
    y = np.asarray(y, float)
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != y.size:
        X = X.T
    Xd = np.column_stack([np.ones(y.size), X])
    beta = np.zeros(Xd.shape[1])

    def ll(b):
        eta = Xd @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    cur = ll(beta)
    converged = False
    for _ in range(max_iter):
        p = expit(Xd @ beta)
        W = p * (1 - p)
        H = Xd.T @ (Xd * W[:, None])
        g = Xd.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-8:
            nb = beta + t * step
            nl = ll(nb)
            if nl >= cur - 1e-12:
                break
            t *= 0.5
        beta, cur = nb, nl
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    p = expit(Xd @ beta)
    H = Xd.T @ (Xd * (p * (1 - p))[:, None])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(beta.size, np.nan)
    sep = bool(np.max(np.abs(beta[1:])) > 15 or not np.all(np.isfinite(se)))
    return LogitFit(beta, se, cur, converged, sep)


def _std(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def size_adjusted(y, log_size, x, log=False):
    """
    LR test of a parameter beyond lesion size. Returns OR per SD of the
    (log-)parameter at fixed size, its Wald 95% CI, the LR p, and flags.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if log:
        x = np.log(np.where(x > 0, x, np.nan))
    ok = np.isfinite(x) & np.isfinite(log_size)
    y, s, x = y[ok], _std(np.asarray(log_size)[ok]), _std(x[ok])
    m0 = logit_fit(s[:, None], y)
    m1 = logit_fit(np.column_stack([s, x]), y)
    lr = max(0.0, 2 * (m1.loglik - m0.loglik))
    b, se = m1.beta[2], m1.se[2]
    sep = m1.separation or m0.separation
    # (quasi-)complete separation: the ML odds ratio is infinite, so it is
    # not reported; the LR statistic still has a finite limit
    nan = sep or not np.isfinite(se)
    return dict(or_per_sd=np.nan if nan else float(np.exp(b)),
                or_ci2_5=np.nan if nan else float(np.exp(b - 1.96 * se)),
                or_ci97_5=np.nan if nan else float(np.exp(b + 1.96 * se)),
                lr_p=float(stats.chi2.sf(lr, 1)),
                size_or_per_sd=float(np.exp(m1.beta[1])),
                separation=sep,
                converged=m1.converged and m0.converged, n=int(ok.sum()))


def cv_auc(y, X, n_splits=5, n_repeats=20, seed=0):
    """Repeated stratified k-fold AUC of an unpenalised logistic model."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    y = np.asarray(y, int)
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != y.size:
        X = X.T
    out = []
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    pred = np.zeros(y.size)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        f = logit_fit((X[tr] - mu) / sd, y[tr])
        pred[te] = np.column_stack([np.ones(te.size), (X[te] - mu) / sd]) @ f.beta
        if (k + 1) % n_splits == 0:
            out.append(auc_mw(pred[y == 0], pred[y == 1])[0])
    return float(np.mean(out)), float(np.std(out))


def block_test(y, log_size, X, log_mask=None, n_repeats=20, seed=0):
    """LR test of k parameters jointly beyond size (k df) + CV AUC."""
    y = np.asarray(y, float)
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != y.size:
        X = X.T
    X = X.copy()
    if log_mask is not None:
        for j, lg in enumerate(log_mask):
            if lg:
                X[:, j] = np.log(np.where(X[:, j] > 0, X[:, j], np.nan))
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(log_size)
    y, s, X = y[ok], np.asarray(log_size)[ok], X[ok]
    Xs = np.column_stack([_std(c) for c in X.T])
    m0 = logit_fit(_std(s)[:, None], y)
    m1 = logit_fit(np.column_stack([_std(s), Xs]), y)
    lr = max(0.0, 2 * (m1.loglik - m0.loglik))
    auc0 = cv_auc(y, s[:, None], n_repeats=n_repeats, seed=seed)
    auc1 = cv_auc(y, np.column_stack([s, X]), n_repeats=n_repeats, seed=seed)
    return dict(k=X.shape[1], n=int(ok.sum()), lr_stat=float(lr),
                lr_p=float(stats.chi2.sf(lr, X.shape[1])),
                cv_auc_size_only=auc0[0], cv_auc_size_plus_block=auc1[0],
                cv_auc_gain=auc1[0] - auc0[0],
                separation=m1.separation or m0.separation)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    o = np.argsort(q)
    m = q.size
    adj = np.maximum.accumulate((m - np.arange(m)) * q[o])
    res = np.empty(m)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    o = np.argsort(q)
    adj = q[o] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def discrimination_table(values: dict, y, log_size, log_params=(),
                         family=None):
    """
    values : {param: per-lesion estimate array}, y : 0 B / 1 M,
    log_size : log(voxels). family : params in the multiplicity family
    (default all). Returns one row per parameter.
    """
    y = np.asarray(y)
    rows = []
    for p, v in values.items():
        v = np.asarray(v, float)
        xB, xM = v[y == 0], v[y == 1]
        auc, pmw = auc_mw(xB, xM)
        lo, hi, se = (delong_ci(xB, xM) if np.isfinite(auc)
                      else (np.nan, np.nan, np.nan))
        lg = p in log_params
        sa = size_adjusted(y, log_size, v, log=lg)
        rows.append(dict(
            param=p, n_B=int(np.isfinite(xB).sum()), n_M=int(np.isfinite(xM).sum()),
            median_B=float(np.nanmedian(xB)), median_M=float(np.nanmedian(xM)),
            auc=auc, auc_ci2_5=lo, auc_ci97_5=hi, mw_p=pmw,
            cliffs_delta=2 * auc - 1,
            hl_shift=hodges_lehmann(xB, xM, log=lg), hl_is_ratio=lg,
            sizeadj_or_per_sd=sa["or_per_sd"], sizeadj_or_ci2_5=sa["or_ci2_5"],
            sizeadj_or_ci97_5=sa["or_ci97_5"], sizeadj_lr_p=sa["lr_p"],
            sizeadj_separation=sa["separation"]))
    df = pd.DataFrame(rows)
    fam = df["param"].isin(family) if family is not None else np.ones(len(df), bool)
    for col in ("mw_p", "sizeadj_lr_p"):
        df[f"{col}_holm"] = np.nan
        df[f"{col}_bh"] = np.nan
        df.loc[fam, f"{col}_holm"] = holm(df.loc[fam, col])
        df.loc[fam, f"{col}_bh"] = bh(df.loc[fam, col])
    df["in_family"] = fam
    return df


# =============================================================================
# 3. Identifiability asymmetry
# =============================================================================

def flag_table(flags: dict, y, log_size):
    """
    flags : {name: per-lesion bool}. Per flag: rate in B and M, Fisher exact
    p, and whether size explains it (logistic flag ~ log size) and whether
    group still matters beyond size (LR p of group added to size).
    """
    y = np.asarray(y)
    s = _std(log_size)
    rows = []
    for name, f in flags.items():
        f = np.asarray(f, float)
        ok = np.isfinite(f)
        fB, fM = f[ok & (y == 0)], f[ok & (y == 1)]
        tab = [[int(fB.sum()), int(fB.size - fB.sum())],
               [int(fM.sum()), int(fM.size - fM.sum())]]
        p_f = stats.fisher_exact(tab)[1] if fB.size and fM.size else np.nan
        row = dict(flag=name, n_B=fB.size, n_M=fM.size,
                   rate_B=float(fB.mean()) if fB.size else np.nan,
                   rate_M=float(fM.mean()) if fM.size else np.nan, fisher_p=p_f)
        if 0 < f[ok].sum() < ok.sum():
            m0 = logit_fit(s[ok][:, None], f[ok])
            m1 = logit_fit(np.column_stack([s[ok], y[ok]]), f[ok])
            row.update(size_or_per_sd=float(np.exp(m0.beta[1])),
                       size_p=float(2 * stats.norm.sf(abs(m0.beta[1] / m0.se[1]))),
                       group_beyond_size_p=float(stats.chi2.sf(
                           max(0.0, 2 * (m1.loglik - m0.loglik)), 1)))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 4. Prior sensitivity by importance reweighting
# =============================================================================

def _zcol(tr, j, x):
    x = np.asarray(x, float)
    h = np.log(x) if tr.log[j] else x
    hl = np.log(tr.lb[j]) if tr.log[j] else tr.lb[j]
    hu = np.log(tr.ub[j]) if tr.log[j] else tr.ub[j]
    u = np.clip((h - hl) / (hu - hl), tr.eps, 1 - tr.eps)
    return np.log(u) - np.log1p(-u), u, hu - hl


def _ljac_col(tr, j, x):
    """log|dz/dx| for column j."""
    x = np.asarray(x, float)
    _, u, width = _zcol(tr, j, x)
    lj = -np.log(u) - np.log1p(-u) - np.log(width)
    return lj - np.log(x) if tr.log[j] else lj


def base_factors(prior_choices, tr):
    """
    The base prior's 1-D log-density factors {param: f(x) -> log p}, for the
    choices fit_lesions_bayes.py uses. Shape-checked against the package's
    PriorSpec.log_pdf by check_base_factors before any reweighting.
    """
    out = {}
    names = list(tr.names)
    vf = prior_choices.get("volfrac", "flat_masked")
    rt = prior_choices.get("rates", "loguniform")
    for j, p in enumerate(names):
        lb, ub = tr.lb[j], tr.ub[j]
        if p in ("ve", "vp"):
            if vf == "beta":
                w = ub - lb
                out[p] = (lambda lb, w: lambda x: (np.log(6.0)
                          + np.log(np.clip((x - lb) / w, 1e-300, None))
                          + np.log(np.clip(1 - (x - lb) / w, 1e-300, None))
                          - np.log(w)))(lb, w)
            elif vf == "flat_masked":
                out[p] = (lambda c: lambda x: np.full(np.shape(x), c))(-np.log(ub - lb))
            else:
                raise NotImplementedError(f"volfrac prior {vf!r} not supported here")
        elif p == "tau_delay":
            out[p] = (lambda c: lambda x: np.full(np.shape(x), c))(-np.log(ub - lb))
        else:
            if rt == "loguniform":
                out[p] = (lambda c: lambda x: -np.log(x) + c)(
                    -np.log(np.log(ub) - np.log(lb)))
            elif rt == "uniform":
                out[p] = (lambda c: lambda x: np.full(np.shape(x), c))(-np.log(ub - lb))
            else:
                raise NotImplementedError(f"rates prior {rt!r} not supported here")
    return out


def check_base_factors(prior, factors, tr, n=2000, seed=0):
    """Sum of factors vs PriorSpec.log_pdf on prior draws. Constant offsets
    cancel in self-normalised weights, so the SD of the difference is the
    criterion. Returns (sd_of_difference, mean_difference)."""
    X = prior.sample(np.random.default_rng(seed), n)
    pkg = np.array([prior.log_pdf(r) for r in X])
    mine = sum(factors[p](X[:, j]) for j, p in enumerate(tr.names))
    d = mine - pkg
    return float(np.std(d)), float(np.mean(d))


@dataclass
class PriorSetting:
    name: str
    kind: str                      # base / reference / location / width / family
    target: str                    # parameter whose prior changes, or 'all'
    factors: dict                  # {param: new log-density f(x)}
    samplers: dict                 # {param: (rng, n) -> x draws from new factor}
    detail: str = ""
    meta: dict = field(default_factory=dict)   # z-scale centre / SD used


def znormal_factor(tr, j, mu, sd):
    def f(x):
        z, _, _ = _zcol(tr, j, x)
        return stats.norm.logpdf(z, mu, sd) + _ljac_col(tr, j, x)

    def samp(rng, n):
        return tr.from_z_col(mu + sd * rng.standard_normal(n), j)
    return f, samp


def zt_factor(tr, j, mu, scale, df):
    def f(x):
        z, _, _ = _zcol(tr, j, x)
        return stats.t.logpdf(z, df, mu, scale) + _ljac_col(tr, j, x)

    def samp(rng, n):
        return tr.from_z_col(mu + scale * rng.standard_t(df, n), j)
    return f, samp


def default_settings(prior, tr, sens_params, loc_factors, width_factors=(0.5, 2.0),
                     t_df=3, n_ref=200_000, seed=0):
    """
    base                    the fit's own prior (all weights 1)
    {p}:ref                 Normal on z matched to the base prior's median / SD
    {p}:loc x f             reference shape, prior MEDIAN multiplied by f
    {p}:width x w           reference location, z-SD multiplied by w
    all:ref, all:t{df}      every sens param Normal vs Student-t (family check)
    A location target outside the box is skipped (reported).
    """
    rng = np.random.default_rng(seed)
    Xp = prior.sample(rng, n_ref)
    names = list(tr.names)
    med_x = {p: float(np.median(Xp[:, names.index(p)])) for p in sens_params}
    zs = {p: _zcol(tr, names.index(p), Xp[:, names.index(p)])[0] for p in sens_params}
    mu0 = {p: float(np.median(zs[p])) for p in sens_params}
    sd0 = {p: float(np.std(zs[p])) for p in sens_params}
    out = [PriorSetting("base", "base", "none", {}, {}, "fit prior")]
    skipped = []
    for p in sens_params:
        j = names.index(p)
        f, s = znormal_factor(tr, j, mu0[p], sd0[p])
        out.append(PriorSetting(f"{p}:ref", "reference", p, {p: f}, {p: s},
                                f"Normal on z, median {med_x[p]:.4g}, z-SD {sd0[p]:.2f}",
                                dict(mu=mu0[p], sd=sd0[p])))
        for fac in loc_factors.get(p, ()):
            tgt = med_x[p] * fac
            if not (tr.lb[j] < tgt < tr.ub[j]):
                skipped.append(f"{p}:loc x{fac} (median {tgt:.4g} outside "
                               f"[{tr.lb[j]:.4g}, {tr.ub[j]:.4g}])")
                continue
            mu = float(_zcol(tr, j, np.array([tgt]))[0][0])
            f, s = znormal_factor(tr, j, mu, sd0[p])
            out.append(PriorSetting(f"{p}:loc x{fac:g}", "location", p, {p: f}, {p: s},
                                    f"median {tgt:.4g}", dict(mu=mu, sd=sd0[p])))
        for w in width_factors:
            f, s = znormal_factor(tr, j, mu0[p], sd0[p] * w)
            out.append(PriorSetting(f"{p}:width x{w:g}", "width", p, {p: f}, {p: s},
                                    f"z-SD {sd0[p] * w:.2f}"))
    fr, sr, ft, st = {}, {}, {}, {}
    for p in sens_params:
        j = names.index(p)
        fr[p], sr[p] = znormal_factor(tr, j, mu0[p], sd0[p])
        ft[p], st[p] = zt_factor(tr, j, mu0[p], sd0[p], t_df)
    out.append(PriorSetting("all:ref", "family", "all", fr, sr, "Normal on z, all"))
    out.append(PriorSetting(f"all:t{t_df}", "family", "all", ft, st,
                            f"Student-t({t_df}) on z, all"))
    return out, skipped


def lesion_log_weights(draws, names, setting, base):
    """(S,) log importance weights for one lesion under one setting."""
    lw = np.zeros(draws.shape[0])
    for p, f in setting.factors.items():
        j = names.index(p)
        lw += f(draws[:, j]) - base[p](draws[:, j])
    return lw


def derived_from_draws(draws, names):
    d = {p: draws[:, names.index(p)] for p in names}
    ve, vp, fp, ps = d["ve"], d["vp"], d["fp"], d["ps"]
    d.update(vtot=ve + vp, E=ps / (ps + fp), Ktrans=ps * fp / (ps + fp),
             Tc=vp / fp, Te=ve / ps)
    return d


def _wq_sorted(v_sorted, w_sorted, qs):
    cw = np.cumsum(w_sorted)
    cw = (cw - 0.5 * w_sorted) / cw[-1]
    return np.interp(qs, cw, v_sorted)


def reweight_cohort(draws_list, names, settings, base, quantities, khat_fn,
                    log_iqr=()):
    """
    Per lesion x setting: weighted median, q25, q75 of every quantity, plus
    weight ESS and k-hat. Returns a long DataFrame.
    """
    rows = []
    for i, dr in enumerate(draws_list):
        q = derived_from_draws(dr, names)
        order = {k: np.argsort(q[k]) for k in quantities}
        for st in settings:
            lw = lesion_log_weights(dr, names, st, base)
            w = np.exp(lw - lw.max())
            w /= w.sum()
            ess = 1.0 / np.sum(w ** 2)
            kh = khat_fn(lw) if st.factors else -np.inf
            r = dict(lesion=i, setting=st.name, ess=ess, khat=kh)
            for k in quantities:
                o = order[k]
                q25, q50, q75 = _wq_sorted(q[k][o], w[o], [0.25, 0.5, 0.75])
                r[f"{k}_median"] = q50
                r[f"{k}_iqr"] = (np.log10(q75) - np.log10(q25) if k in log_iqr
                                 else q75 - q25)
            rows.append(r)
    return pd.DataFrame(rows)


def prior_iqr(setting, prior, tr, quantities, log_iqr=(), n=200_000, seed=1):
    """IQR of each quantity under the setting's prior (base factors where
    not replaced), for the posterior/prior IQR (prior-dominated) ratio."""
    rng = np.random.default_rng(seed)
    X = prior.sample(rng, n)
    names = list(tr.names)
    for p, smp in setting.samplers.items():
        X[:, names.index(p)] = smp(rng, n)
    q = derived_from_draws(X, names)
    out = {}
    for k in quantities:
        q25, q75 = np.percentile(q[k], [25, 75])
        out[k] = (np.log10(q75) - np.log10(q25)) if k in log_iqr else (q75 - q25)
    return out
