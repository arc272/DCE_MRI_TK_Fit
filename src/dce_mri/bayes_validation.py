# -*- coding: utf-8 -*-
"""
bayes_validation.py
===================
The validation ladder for Bayesian estimation. Five rungs, in order.
Each answers a DIFFERENT question; passing one says nothing about the
others.

    Rung 0  noise-model selection      -- which likelihood is justified
    Rung 1  sampler convergence        -- did the chains actually mix
    Rung 2  posterior calibration      -- is the posterior SHAPE honest
    Rung 3  accuracy vs NLLS           -- is it better than what we had
    Rung 4  ablation / sensitivity     -- does the conclusion survive
    Rung 5  real-data comparison       -- sanity, no ground truth

Kept separate from bayes.py because it is run occasionally, imports
heavier dependencies, and should not be a hard dependency of the
fitting path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from dce_mri.bayes import (
    BayesConfig, BayesResult, fit_curve_bayes, NOISE_OPTIONS,
    build_param_layout, GAUSS_FIXED_SIGMA, GAUSS_SIGNAL_SCALED, STUDENT_T,
)
from dce_mri.bayes_priors import assemble_prior
from dce_mri.synthetic import simulate_curve, add_noise


# =============================================================================
# RUNG 0 -- noise-model selection (BEFORE any MCMC)
# =============================================================================

def check_residual_normality(residuals, sigma=None) -> dict:
    """
    Kolmogorov-Smirnov test of standardised residuals against a normal.

    [SOURCE: Zhou2013]  They found normality REJECTED for the great
    majority of real voxels, traced to a few very-low-signal early
    frames behaving like outliers, and switching to a Student-t
    likelihood changed the parameter estimates substantially.

    The KS test is conservative here because sigma is estimated from the
    same residuals, so a rejection is meaningful while a pass is weak
    evidence.
    """
    r = np.asarray(residuals, float)
    s = float(np.std(r, ddof=1)) if sigma is None else float(sigma)
    if s <= 0:
        return dict(ks_stat=np.nan, p_value=np.nan, normal=False,
                    skew=np.nan, kurtosis=np.nan)
    z = (r - np.mean(r)) / s
    ks, p = stats.kstest(z, "norm")
    return dict(ks_stat=float(ks), p_value=float(p), normal=bool(p > 0.05),
                skew=float(stats.skew(r)), kurtosis=float(stats.kurtosis(r)),
                sigma=s)


def check_heteroscedasticity(residuals, pred, n_bins=4) -> dict:
    """
    Does residual variance track the predicted signal?

    Splits timepoints into bins by |pred| and compares residual variance
    across bins (Levene's test, which is robust to non-normality).
    A significant result argues for GAUSS_SIGNAL_SCALED
    [Sitek2016, Zhou2013] over a single flat sigma.
    """
    r, p = np.asarray(residuals, float), np.asarray(pred, float)
    order = np.argsort(np.abs(p))
    groups = [r[g] for g in np.array_split(order, n_bins) if len(g) > 2]
    if len(groups) < 2:
        return dict(p_value=np.nan, heteroscedastic=False, var_ratio=np.nan)
    stat, pv = stats.levene(*groups)
    v = [float(np.var(g, ddof=1)) for g in groups]
    return dict(levene_stat=float(stat), p_value=float(pv),
                heteroscedastic=bool(pv < 0.05),
                var_by_bin=v,
                var_ratio=float(max(v) / max(min(v), 1e-12)))


def check_autocorrelation(residuals, lag=1) -> dict:
    """
    Lag-1 autocorrelation (Durbin-Watson style). Point-wise likelihoods
    treat residuals as independent; meaningful serial correlation means
    the reported uncertainty will be understated regardless of which
    noise SHAPE is chosen.
    """
    r = np.asarray(residuals, float) - np.mean(residuals)
    n = len(r)
    if n <= lag + 2:
        return dict(acf1=np.nan, durbin_watson=np.nan, correlated=False)
    acf = float(np.sum(r[lag:] * r[:-lag]) / np.sum(r ** 2))
    dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
    return dict(acf1=acf, durbin_watson=dw, correlated=bool(abs(acf) > 2 / np.sqrt(n)))


def select_noise_model(residuals, pred, verbose=True) -> dict:
    """
    Combine the three diagnostics into a JUSTIFIED recommendation.

    Selection is empirical, not a matter of taste -- this is the whole
    point of Rung 0. Returns the recommended NoiseModelSpec plus the
    reasoning, so the choice is recorded rather than assumed.
    """
    norm = check_residual_normality(residuals)
    het = check_heteroscedasticity(residuals, pred)
    auto = check_autocorrelation(residuals)

    if not norm["normal"]:
        rec, why = STUDENT_T, ("residuals fail the normality test "
                               f"(KS p={norm['p_value']:.4f}); heavy tails "
                               "or outlier frames [Zhou2013]")
    elif het["heteroscedastic"]:
        rec, why = GAUSS_SIGNAL_SCALED, ("variance tracks the predicted signal "
                                         f"(Levene p={het['p_value']:.4f}) "
                                         "[Sitek2016]")
    else:
        rec, why = GAUSS_FIXED_SIGMA, ("residuals are approximately normal and "
                                       "homoscedastic; a single sigma suffices")

    out = dict(recommended=rec.name, rationale=why,
               normality=norm, heteroscedasticity=het, autocorrelation=auto)
    if auto["correlated"]:
        out["warning"] = (
            f"lag-1 autocorrelation {auto['acf1']:.3f} exceeds the "
            "white-noise band; point-wise likelihoods will UNDERSTATE "
            "uncertainty regardless of noise shape.")
    if verbose:
        print(f"[Rung 0] recommended noise model: {rec.name}\n  because {why}")
        if "warning" in out:
            print("  WARNING:", out["warning"])
    return out


# =============================================================================
# RUNG 1 -- convergence
# =============================================================================

def convergence_report(result: BayesResult, rhat_max=1.01, ess_min=400,
                       verbose=True) -> dict:
    """R-hat / ESS / acceptance summary. [Gelman&Rubin 1992; Roberts1997]"""
    bad_r = {k: v for k, v in result.rhat.items() if not (v <= rhat_max)}
    bad_e = {k: v for k, v in result.ess.items() if not (v >= ess_min)}
    rep = dict(rhat=result.rhat, ess=result.ess,
               accept_frac=result.accept_frac,
               failed_rhat=bad_r, failed_ess=bad_e,
               passed=bool(not bad_r and not bad_e))
    if verbose:
        print(f"[Rung 1] converged={rep['passed']}  "
              f"max R-hat={max(result.rhat.values()) if result.rhat else np.nan:.4f}  "
              f"min ESS={min(result.ess.values()) if result.ess else np.nan:.0f}")
        if bad_r:
            print("  R-hat failures:", {k: round(v, 4) for k, v in bad_r.items()})
        if bad_e:
            print("  ESS failures:", {k: round(v) for k, v in bad_e.items()})
    return rep


def zhou_variance_ratio(chain_1d, tail_frac=0.1) -> float:
    """
    var(final tail of a SINGLE chain) / var(all post-burn-in samples).

    [SOURCE: Zhou2013]  Pass band 0.9-1.1. A cheap per-voxel proxy that
    makes a volume-wide convergence sweep affordable when running 4
    chains per voxel for a true R-hat is not. Zhou applied it across
    1.5 million chains and simply reported the fraction passing (91.2%).
    """
    x = np.asarray(chain_1d, float).ravel()
    k = max(int(len(x) * tail_frac), 10)
    v_all = np.var(x, ddof=1)
    if v_all <= 0:
        return np.nan
    return float(np.var(x[-k:], ddof=1) / v_all)


# =============================================================================
# RUNG 2 -- simulation-based calibration (SBC)
# =============================================================================

@dataclass
class SBCResult:
    param_names: tuple
    ranks:       np.ndarray     # (n_sims, D) integer ranks in [0, n_draws]
    n_draws:     int
    n_sims:      int
    meta:        dict = field(default_factory=dict)

    def chisq(self) -> dict:
        """
        Cook/Gelman/Rubin (2006) recipe, as used by Sitek2016:
        turn each rank into a quantile, push through the inverse normal
        CDF, sum the squares -> chi-square with n_sims d.o.f.

        p near 0   -> posteriors too WIDE (truth too central)
        p near 1   -> posteriors too NARROW (truth in the tails)
        """
        out = {}
        for j, nm in enumerate(self.param_names):
            q = (self.ranks[:, j] + 0.5) / (self.n_draws + 1)
            z = stats.norm.ppf(np.clip(q, 1e-9, 1 - 1e-9))
            chi2 = float(np.sum(z ** 2))
            out[nm] = dict(chi2=chi2, dof=self.n_sims,
                           p_value=float(stats.chi2.cdf(chi2, self.n_sims)))
        return out

    def uniformity(self) -> dict:
        """
        KS test of rank uniformity -- the modern check
        [Talts2018, Sailynoja2022], less prone than a single chi-square
        p-value to hiding WHICH kind of miscalibration is present.
        """
        out = {}
        for j, nm in enumerate(self.param_names):
            q = (self.ranks[:, j] + 0.5) / (self.n_draws + 1)
            ks, p = stats.kstest(q, "uniform")
            out[nm] = dict(ks_stat=float(ks), p_value=float(p),
                           calibrated=bool(p > 0.05))
        return out


def run_sbc(
    model, acq, subject, cfg_fit, cfg_bayes, setup,
    *, wx_model=None, aif_cfg=None, relax_cfg=None,
    n_sims=100, sigma_true=20.0, noise_kind="gaussian",
    seed=0, verbose=True,
) -> SBCResult:
    """
    Simulation-based calibration.

    For l = 1..n_sims:
        1. draw theta_true from THE PRIOR
        2. simulate data from theta_true through the SAME forward model
        3. run inference, collect R posterior draws
        4. rank theta_true among those draws

    If inference is correct, the ranks are UNIFORM -- regardless of the
    prior's shape. Uniformity is a CONCLUSION about ranks, not an
    assumption about the prior, so this works for Beta/Gamma/truncated
    priors exactly as well as for flat ones.

    THE ONE REQUIREMENT: the distribution theta_true is drawn from must
    be EXACTLY the prior the sampler's log-posterior assumes. A mismatch
    produces non-uniform ranks that mean "my SBC setup has a bug", not
    "my sampler is broken".

    IMPORTANT CAVEAT (put in the docstring deliberately): SBC is
    NECESSARY, NOT SUFFICIENT. A correctly implemented sampler on a
    non-identifiable tau_b returns approximately the prior and PASSES,
    because "I learned nothing" is a calibrated answer. What SBC catches
    is a sampler stuck on a ridge returning an artificially NARROW,
    overconfident posterior. It is not an identifiability test -- use
    posterior_vs_prior_width() for that.
    """
    rng = np.random.default_rng(seed)
    layout = build_param_layout(
        model, cfg_bayes, aif=setup["aif"], time_var=setup["time_var"],
        acq=acq, wx_model=wx_model, relax_cfg=relax_cfg)
    noise = NOISE_OPTIONS[cfg_bayes.noise]
    prior = assemble_prior(
        layout.model, relax_cfg, wx_model=wx_model,
        delay_mode=cfg_bayes.delay_mode, n_noise=noise.n_params,
        noise_names=noise.param_names, choices=cfg_bayes.prior_choices,
        delay_bounds=cfg_bayes.delay_bounds)
    prior.validate_against(layout.param_names)

    ranks, n_ok, n_skip = [], 0, 0
    for l in range(n_sims):
        # 1. draw truth FROM THE PRIOR
        for _ in range(100):
            theta_true = prior.sample(rng, 1)[0]
            if layout.model.constraints is None or \
               layout.model.constraints(theta_true[:layout.n_kin]):
                break
        else:
            n_skip += 1
            continue

        # 2. simulate
        try:
            S = simulate_curve(
                theta_true[:layout.n_model], setup["aif"], setup["time_var"],
                layout.model, acq, subject, cfg_fit, wx_model, aif_cfg,
                relax_cfg, S0=setup.get("S0", 1000.0))
            if not np.all(np.isfinite(S)):
                n_skip += 1
                continue
            s_use = (float(theta_true[layout.noise_slice][0])
                     if layout.n_noise >= 1 else sigma_true)
            sig = add_noise(S, s_use, noise_kind, rng)
        except Exception:
            n_skip += 1
            continue

        # 3. infer
        try:
            res = fit_curve_bayes(
                sig, setup["aif"], setup["time_var"], model, acq, subject,
                cfg_fit, BayesConfig(**{**cfg_bayes.__dict__,
                                        "seed": int(rng.integers(1 << 30))}),
                wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
        except Exception:
            n_skip += 1
            continue

        # 4. rank the truth among the posterior draws
        flat = res.flat()
        ranks.append([(flat[:, j] < theta_true[j]).sum()
                      for j in range(layout.n_total)])
        n_ok += 1
        if verbose and (l + 1) % 10 == 0:
            print(f"  SBC {l+1}/{n_sims}  ok={n_ok} skipped={n_skip}")

    ranks = np.array(ranks, dtype=int)
    n_draws = cfg_bayes.n_draws * cfg_bayes.n_chains
    return SBCResult(param_names=layout.param_names, ranks=ranks,
                     n_draws=n_draws, n_sims=len(ranks),
                     meta=dict(n_skipped=n_skip, prior_sources=prior.sources))


# =============================================================================
# RUNG 3 -- accuracy vs NLLS on synthetic ground truth
# =============================================================================

@dataclass
class RecoveryResult:
    param_names: tuple
    truth:       np.ndarray   # (n, D_model)
    bayes_est:   np.ndarray
    nlls_est:    np.ndarray
    nlls_inside_ci: np.ndarray  # (n, D_model) bool, 68% asymptotic CI
    n_nlls_failed:  int
    n_nlls_railed:  int
    meta:        dict = field(default_factory=dict)

    def nmse(self) -> dict:
        """
        Normalised MSE per parameter, plus a Mann-Whitney U test.

        Mann-Whitney rather than a paired t-test because NMSE across
        noise draws is a right-skewed, non-negative distribution --
        nowhere near Gaussian.  [SOURCE: Scannell2019]
        """
        out = {}
        for j, nm in enumerate(self.param_names):
            t = self.truth[:, j]
            denom = np.maximum(t ** 2, 1e-30)
            nb = (self.bayes_est[:, j] - t) ** 2 / denom
            nn = (self.nlls_est[:, j] - t) ** 2 / denom
            ok = np.isfinite(nb) & np.isfinite(nn)
            u, p = (stats.mannwhitneyu(nb[ok], nn[ok], alternative="less")
                    if ok.sum() > 3 else (np.nan, np.nan))
            out[nm] = dict(bayes_nmse=float(np.mean(nb[ok])),
                           nlls_nmse=float(np.mean(nn[ok])),
                           bayes_nmse_sd=float(np.std(nb[ok])),
                           nlls_nmse_sd=float(np.std(nn[ok])),
                           u_stat=float(u), p_value=float(p),
                           bayes_better=bool(np.isfinite(p) and p < 0.05))
        return out

    def nlls_ci_coverage(self, level=0.6827) -> dict:
        """
        How often the truth actually fell inside the NLLS asymptotic
        68% CI, tested against Binomial(n, 0.6827).

        [SOURCE: Sitek2016]  He found 589/1000 for K1 (expected ~683),
        p = 8.9e-16 -- gross undercoverage. Orton2007 independently
        found Hessian-based intervals too narrow for all three of his
        estimators. This is the single most quotable answer to "why not
        just use the covariance matrix".
        """
        out = {}
        n = self.nlls_inside_ci.shape[0]
        for j, nm in enumerate(self.param_names):
            k = int(np.nansum(self.nlls_inside_ci[:, j]))
            p = float(stats.binomtest(k, n, level).pvalue) if n else np.nan
            out[nm] = dict(n_inside=k, n_total=n, expected=level * n,
                           observed_frac=k / n if n else np.nan,
                           p_value=p,
                           miscalibrated=bool(np.isfinite(p) and p < 0.05))
        return out


def _nlls_asymptotic_ci(bundle, params, dof, level=0.6827):
    """
    Asymptotic CI from the Jacobian: sqrt(diag((J'J)^-1 * sigma^2)).
    This is the classical uncertainty estimate Rung 3 is testing.
    """
    p = np.asarray(params, float)
    r0 = bundle.residual_fn(p)
    if not np.all(np.isfinite(r0)):
        return None
    n = r0.size
    J = np.empty((n, len(p)))
    for k in range(len(p)):
        h = max(abs(p[k]) * 1e-6, 1e-10)
        pk = p.copy(); pk[k] += h
        rk = bundle.residual_fn(pk)
        if not np.all(np.isfinite(rk)):
            return None
        J[:, k] = (rk - r0) / h
    try:
        cov = np.linalg.inv(J.T @ J) * (np.sum(r0 ** 2) / max(n - len(p), 1))
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        return None
    z = stats.norm.ppf(0.5 + level / 2)
    return se * z


def synthetic_recovery_study(
    model, acq, subject, cfg_fit, cfg_bayes, setup,
    *, wx_model=None, aif_cfg=None, relax_cfg=None,
    truth=None, n_noise_draws=20, sigma_true=20.0,
    noise_kind="gaussian", seed=0, verbose=True,
) -> RecoveryResult:
    """
    Bayesian vs NLLS on curves with KNOWN parameters.

    n_noise_draws INDEPENDENT noise realisations per operating point --
    a single realisation says almost nothing, because NLLS's answer is
    itself a random variable (Scannell's Fig. 2: same truth, same AIF,
    different noise draw, factor-of-two difference in fitted flow).
    """
    from dce_mri.fitting import fit_single_voxel, build_residual

    rng = np.random.default_rng(seed)
    layout = build_param_layout(
        model, cfg_bayes, aif=setup["aif"], time_var=setup["time_var"],
        acq=acq, wx_model=wx_model, relax_cfg=relax_cfg)

    if truth is None:
        truth = np.asarray(layout.model.p0, float)
        if wx_model is not None and wx_model.n_params > 0:   # NXL: none
            extra = ([relax_cfg.tau_b_fixed, 0.5] if relax_cfg.fit_tau_b
                     else [0.5])
            truth = np.concatenate([truth, extra])
    truth = np.asarray(truth, float)[:layout.n_model]

    S_clean = simulate_curve(truth, setup["aif"], setup["time_var"],
                             layout.model, acq, subject, cfg_fit,
                             wx_model, aif_cfg, relax_cfg,
                             S0=setup.get("S0", 1000.0))

    T, B, N, INS = [], [], [], []
    n_fail = n_rail = 0
    for i in range(n_noise_draws):
        sig = add_noise(S_clean, sigma_true, noise_kind, rng)

        pf = fit_single_voxel(sig, setup["aif"], setup["time_var"],
                              layout.model, acq, subject, cfg_fit,
                              seed=int(rng.integers(1 << 30)),
                              wx_model=wx_model, aif_cfg=aif_cfg,
                              relax_cfg=relax_cfg)
        nl = np.asarray(pf.params, float)
        if pf.wx_params is not None:
            nl = np.concatenate([nl, np.asarray(pf.wx_params, float)])
        if not pf.success:
            n_fail += 1
        if np.any(np.isclose(nl, layout.lb, rtol=1e-3)) or \
           np.any(np.isclose(nl, layout.ub, rtol=1e-3)):
            n_rail += 1

        bundle = build_residual(sig, setup["aif"], setup["time_var"],
                                layout.model, acq, subject, cfg_fit,
                                wx_model=wx_model, aif_cfg=aif_cfg,
                                relax_cfg=relax_cfg, on_failure="nan")
        half = _nlls_asymptotic_ci(bundle, nl, layout.n_model)
        inside = (np.abs(nl - truth) <= half) if half is not None \
            else np.full(layout.n_model, np.nan)

        try:
            br = fit_curve_bayes(
                sig, setup["aif"], setup["time_var"], model, acq, subject,
                cfg_fit, BayesConfig(**{**cfg_bayes.__dict__,
                                        "seed": int(rng.integers(1 << 30))}),
                wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
                point_fit=pf)
            be = np.array([br.percentile(50, nm)
                           for nm in layout.param_names[:layout.n_model]])
        except Exception:
            be = np.full(layout.n_model, np.nan)

        T.append(truth); B.append(be); N.append(nl); INS.append(inside)
        if verbose and (i + 1) % 5 == 0:
            print(f"  recovery {i+1}/{n_noise_draws}")

    return RecoveryResult(
        param_names=layout.param_names[:layout.n_model],
        truth=np.array(T), bayes_est=np.array(B), nlls_est=np.array(N),
        nlls_inside_ci=np.array(INS, dtype=float),
        n_nlls_failed=n_fail, n_nlls_railed=n_rail,
        meta=dict(sigma_true=sigma_true, noise_kind=noise_kind))


# =============================================================================
# RUNG 4 -- ablation / sensitivity
# =============================================================================

def residual_bootstrap_curve(
    sig, aif, time_var, model, acq, subject, cfg_fit,
    *, wx_model=None, aif_cfg=None, relax_cfg=None,
    n_rep=200, seed=0, verbose=False,
):
    """
    Parametric-free residual bootstrap -- the FREQUENTIST comparator.

    1. Fit once with NLLS -> theta_hat, fitted curve, residuals r
    2. Resample r WITH REPLACEMENT -> r*
    3. Build a synthetic dataset  y* = fitted_curve + r*
    4. Refit -> theta*;  repeat n_rep times

    The spread of {theta*} approximates the SAMPLING distribution of the
    estimator ("if I repeated the experiment, how much would theta_hat
    move?"). That is a different question from the Bayesian posterior
    ("given THIS data, which theta are plausible?"), even though under
    flat priors the two often look similar -- which is exactly what the
    comparison in verify_02 is testing.

    Returns dict {param_name: (n_rep,) array}, plus 'point' and 'n_failed'.
    """
    from dce_mri.fitting import fit_single_voxel, build_residual

    rng = np.random.default_rng(seed)

    pf = fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg_fit,
                          seed=seed, wx_model=wx_model, aif_cfg=aif_cfg,
                          relax_cfg=relax_cfg)
    point = np.asarray(pf.params, float)
    if pf.wx_params is not None:
        point = np.concatenate([point, np.asarray(pf.wx_params, float)])

    fitted = np.asarray(pf.S_pred, float)
    resid = np.asarray(pf.S_meas, float) - fitted

    bundle = build_residual(sig, aif, time_var, model, acq, subject, cfg_fit,
                            wx_model=wx_model, aif_cfg=aif_cfg,
                            relax_cfg=relax_cfg, on_failure="nan")
    names = bundle.param_names

    draws, n_failed = [], 0
    for i in range(n_rep):
        r_star = rng.choice(resid, size=resid.size, replace=True)
        y_star = fitted + r_star
        try:
            pf_i = fit_single_voxel(y_star, aif, time_var, model, acq, subject,
                                    cfg_fit, seed=int(rng.integers(1 << 30)),
                                    wx_model=wx_model, aif_cfg=aif_cfg,
                                    relax_cfg=relax_cfg)
            th = np.asarray(pf_i.params, float)
            if pf_i.wx_params is not None:
                th = np.concatenate([th, np.asarray(pf_i.wx_params, float)])
            draws.append(th)
            if not pf_i.success:
                n_failed += 1
        except Exception:
            n_failed += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"    bootstrap {i+1}/{n_rep}")

    arr = np.array(draws) if draws else np.zeros((0, len(names)))
    out = {nm: arr[:, j] for j, nm in enumerate(names)}
    out["point"] = point
    out["n_failed"] = n_failed
    out["param_names"] = names
    return out


def prior_sensitivity(sig, setup, model, acq, subject, cfg_fit,
                      cfg_a, cfg_b, *, wx_model=None, aif_cfg=None,
                      relax_cfg=None, verbose=True) -> dict:
    """
    Same curve, two priors. Does the conclusion survive?

    [SOURCE: Zhou2013]  They deliberately ran vague AND informed priors
    to check that their substantive findings did not depend on the
    choice. Mittermeier2019 lists NOT doing this as a limitation of
    their own paper.

    The concrete use here: run TAU_FLAT vs TAU_I_INFORMED, or
    VOLFRAC_FLAT_MASKED vs VOLFRAC_BETA on a railing ve, and read the
    DIFFERENTIAL -- how far a deliberately weak prior moves the
    posterior tells you how much real constraining information the
    likelihood carries near that boundary.
    """
    ra = fit_curve_bayes(sig, setup["aif"], setup["time_var"], model, acq,
                         subject, cfg_fit, cfg_a, wx_model=wx_model,
                         aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    rb = fit_curve_bayes(sig, setup["aif"], setup["time_var"], model, acq,
                         subject, cfg_fit, cfg_b, wx_model=wx_model,
                         aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    out = {}
    for nm in ra.param_names:
        if nm not in rb.param_names:
            continue
        ma, mb = ra.percentile(50, nm), rb.percentile(50, nm)
        ia, ib = ra.iqr(nm), rb.iqr(nm)
        out[nm] = dict(median_a=ma, median_b=mb,
                       shift=float(mb - ma),
                       rel_shift=float((mb - ma) / ma) if ma else np.nan,
                       iqr_a=ia, iqr_b=ib,
                       shift_in_iqr=float((mb - ma) / ia) if ia else np.nan)
    if verbose:
        print("[Rung 4] prior sensitivity (shift measured in prior-A IQRs):")
        for nm, d in out.items():
            print(f"  {nm:10s} {d['median_a']:.5f} -> {d['median_b']:.5f}  "
                  f"({d['shift_in_iqr']:+.2f} IQR)")
    return dict(per_param=out, result_a=ra, result_b=rb)


def posterior_predictive_check(result: BayesResult, sig, setup, model,
                               acq, subject, cfg_fit, *, wx_model=None,
                               aif_cfg=None, relax_cfg=None,
                               n_draws=200, seed=0) -> dict:
    """
    Simulate replicate curves from posterior draws and compare to the
    observed curve.  [SOURCE: Mittermeier2019]

    Distinguishes "wide posterior because the data are noisy" (fine)
    from "wide posterior despite clean data" (model misspecification).
    """
    rng = np.random.default_rng(seed)
    layout = build_param_layout(model, BayesConfig(), aif=setup["aif"],
                                time_var=setup["time_var"], acq=acq,
                                wx_model=wx_model, relax_cfg=relax_cfg)
    flat = result.flat()
    idx = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    curves = []
    for i in idx:
        try:
            curves.append(simulate_curve(
                flat[i, :layout.n_model], setup["aif"], setup["time_var"],
                layout.model, acq, subject, cfg_fit, wx_model, aif_cfg,
                relax_cfg, S0=setup.get("S0", 1000.0)))
        except Exception:
            continue
    curves = np.array(curves)
    lo, med, hi = np.percentile(curves, [2.5, 50, 97.5], axis=0)
    inside = float(np.mean((sig >= lo) & (sig <= hi)))
    return dict(lo=lo, med=med, hi=hi, curves=curves,
                frac_inside_95=inside,
                well_calibrated=bool(0.85 <= inside <= 1.0))


def prior_predictive_check(prior, model, acq, subject, cfg_fit, setup,
                           *, wx_model=None, aif_cfg=None, relax_cfg=None,
                           n=200, seed=0) -> dict:
    """
    Simulate curves from the PRIOR ALONE, before any data.
    [SOURCE: Mittermeier2019 Appendix A]

    Confirms the prior generates the kind of curves you would actually
    expect to see -- spanning, but not wildly exceeding, plausibility.
    """
    rng = np.random.default_rng(seed)
    draws = prior.sample(rng, n)
    n_model = len(model.p0) + (0 if (wx_model is None or wx_model.n_params == 0)
                               else (2 if relax_cfg.fit_tau_b else 1))
    curves = []
    for d in draws:
        try:
            c = simulate_curve(d[:n_model], setup["aif"], setup["time_var"],
                               model, acq, subject, cfg_fit, wx_model,
                               aif_cfg, relax_cfg, S0=setup.get("S0", 1000.0))
            if np.all(np.isfinite(c)):
                curves.append(c)
        except Exception:
            continue
    curves = np.array(curves)
    return dict(curves=curves, n_ok=len(curves),
                peak_median=float(np.median(curves.max(axis=1))) if len(curves) else np.nan,
                peak_p95=float(np.percentile(curves.max(axis=1), 95)) if len(curves) else np.nan)


# =============================================================================
# RUNG 5 -- comparison against the existing bootstrap
# =============================================================================

def compare_to_bootstrap(bayes_result: BayesResult, bootstrap_result) -> dict:
    """
    Posterior vs bootstrap replicate distribution, per shared parameter.

    Under FLAT priors the two should broadly agree, and this is a useful
    validation-phase sanity check. It is NOT a general requirement: the
    bootstrap is itself approximate, and the flat-prior/bootstrap
    correspondence is weakest exactly on ridge-shaped surfaces -- i.e.
    precisely where tau_b and possibly tau_i live. Treat disagreement as
    informative, not as a failure of the posterior.
    """
    out = {}
    for nm in bayes_result.param_names:
        if nm not in bootstrap_result.param_names:
            continue
        bm, bi = bayes_result.percentile(50, nm), bayes_result.iqr(nm)
        sm, si = bootstrap_result.percentile(50, nm), bootstrap_result.iqr(nm)
        out[nm] = dict(bayes_median=float(bm), boot_median=float(sm),
                       bayes_iqr=float(bi), boot_iqr=float(si),
                       median_diff_in_iqr=float((bm - sm) / bi) if bi else np.nan,
                       iqr_ratio=float(bi / si) if si else np.nan)
    return out
