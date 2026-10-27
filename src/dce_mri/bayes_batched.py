"""
bayes_batched.py
================
Many INDEPENDENT single-lesion BlackJAX NUTS fits, stepped together on one
device.

Nothing is pooled. Each (lesion, chain) lane is its own NUTS sampler with its
own position, step size, mass matrix, random keys and -- the only new thing --
its own lesion data. The posterior of lesion A does not depend on which other
lesions share its batch. What the lanes DO share is the clock: a vmapped NUTS
step lasts as long as the lane whose trajectory is longest, so one hard lesion
slows its batch (but does not change its batch-mates' draws).

Why a separate module
---------------------
bayes.fit_curve_bayes -> jax_forward.make_predict bakes the lesion's signal,
AIF and S0 into the log-density as closure CONSTANTS. That makes every lesion
a different function, so it is recompiled per lesion and cannot be vmapped
across lesions. Here the same quantities are passed as an ARGUMENT
(LesionData), so one compiled function serves every lesion in the batch.

The forward model itself is not re-derived: make_predict_data calls the same
jax_forward building blocks (KINETIC_FNS, shift_aif, concentration_to_signal,
nxl_to_signal, cnxl_to_signal, ssm3_to_signal) in the same order as
make_predict, and make_log_lik_data has the same noise maths as
bayes_blackjax.make_log_lik_jax. Both identities are checked numerically
(verify/check_12_batched_loglik.py), not assumed.

What is per lesion and what is shared
-------------------------------------
per lesion (LesionData) : signal St, AIF (pre-shifted when delay_mode='fixed'),
                          S0 (mean of St over THAT lesion's acq.nbase),
                          AIF baseline mean (used by the joint-delay shift),
                          frame weights, n_eff
shared, checked equal   : time grid, StaticCtx (TR, FA, T10, B1, r1, blood
                          T1, hematocrit), forward-model options, parameter
                          layout, prior (hence the box transform)

Supported: delay_mode 'none' | 'joint' | 'fixed' (a different fixed delay per
lesion is fine), noise 'gauss_marginalized_sigma', wx None | NXL | CNXL | SSM3
-- i.e. what fit_curve_bayes's blackjax branch can actually run. The
free-Gaussian-sigma likelihood is written (make_log_lik_data noise='free'),
but bayes_blackjax.build_transform has no transform for its noise prior yet,
so neither sample_blackjax nor this module can sample it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from dce_mri.jax_forward import (
    KINETIC_FNS, StaticCtx, static_ctx_from_configs, shift_aif as _shift_aif_jax,
    concentration_to_signal, nxl_to_signal,  ssm3_to_signal,
    _TINY,
)
from dce_mri.bayes import (
    BayesConfig, BayesResult, NOISE_OPTIONS, build_param_layout,
    make_log_posterior, estimate_sigma_from_fit, fit_single_voxel,
    _validate_wx_compatibility,
)
from dce_mri.bayes_blackjax import build_transform, _diagnostics


# Same table as fit_curve_bayes's blackjax branch.
_NOISE_MAP = {
    "gauss_marginalized_sigma": "marginalized",
    "gauss_free_jeffreys":      "free",
    "gauss_free_invgamma":      "free",
    "gauss_free_lognormal":     "free",
}


# =============================================================================
# Step 1 -- forward model and likelihood with the lesion as an ARGUMENT
# =============================================================================

class LesionData(NamedTuple):
    """
    Everything that differs between lesions, as a pytree of arrays so it can
    be stacked along a leading lesion axis and vmapped.

    sig          : (T,) lesion signal St
    aif          : (T,) plasma AIF, mM -- ALREADY SHIFTED when
                   delay_mode='fixed' (the same numpy signal_models.shift_aif
                   call fit_curve_bayes makes), unshifted otherwise
    S0           : ()   mean of sig over this lesion's acq.nbase
    baseline_val : ()   mean of aif over this lesion's acq.nbase; the left
                        fill value of the joint-delay shift
    weights      : (T,) frame weights, 1.0 = keep (all ones unless frames
                        are dropped)
    n_eff        : ()   sum(weights) -- the N of the sigma-marginal
    """
    sig:          jnp.ndarray
    aif:          jnp.ndarray
    S0:           jnp.ndarray
    baseline_val: jnp.ndarray
    weights:      jnp.ndarray
    n_eff:        jnp.ndarray


def stack_lesion_data(items: Sequence[LesionData]) -> LesionData:
    """List of per-lesion LesionData -> one LesionData with a leading L axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *items)


def make_predict_data(
    model_name: str,
    t,
    ctx: StaticCtx,
    *,
    wx: bool = False,
    wx_name: str = "SSM3",
    fit_tau_b: bool = False,
    tau_b_fixed: float = 0.05,
    delay: bool = False,
    upsample: int = 50,
) -> Callable:
    """
    `predict(p_aug, data) -> signal curve`.

    Line-for-line the body of jax_forward.make_predict, with aif / S0 /
    baseline_val read from `data` instead of the closure. Same p_aug layout:

        p_aug = [ *theta_kin , (tau_delay?) , (tau_b?) , (tau_i?) ]

    t and ctx stay in the closure: they are identical for every lesion in a
    batch (fit_curve_bayes_batched checks this).
    """
    comp_fn, n_kin, _names, ve_idx, vp_idx = KINETIC_FNS[model_name]
    t = jnp.asarray(t, dtype=jnp.float64)
    if wx and wx_name not in ("SSM3", "NXL"):
        raise ValueError(f"no JAX forward model for wx_name={wx_name!r} "
                         f"(have: 'SSM3', 'NXL')")

    def predict(p_aug, data: LesionData):
        aif, S0 = data.aif, data.S0
        theta_kin = p_aug[:n_kin]
        k = n_kin

        if delay:
            aif_use = _shift_aif_jax(aif, p_aug[k], t, data.baseline_val)
            k += 1
        else:
            aif_use = aif

        Cv, Ce = comp_fn(theta_kin, t, aif_use, upsample)

        if not wx:
            return concentration_to_signal(Cv + Ce, ctx, S0)

        if wx_name == "NXL":
            return nxl_to_signal(
                Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], ctx, S0)

        if fit_tau_b:
            tau_b, tau_i = p_aug[k], p_aug[k + 1]
        else:
            tau_b, tau_i = tau_b_fixed, p_aug[k]

        return ssm3_to_signal(
            Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], tau_b, tau_i, ctx, S0)

    return predict


def make_log_lik_data(predict_data: Callable, noise: str = "marginalized") -> Callable:
    """
    `log_lik(p_aug, data) -> scalar`. Same maths as
    bayes_blackjax.make_log_lik_jax (which composes make_residual and the
    noise term); the only difference is that target, weights and n_eff come
    from `data`.

    noise : 'marginalized' -- sigma integrated out (Orton Eq. 8); no sigma
                              entry in p_aug
            'free'         -- p_aug's LAST entry is sigma
    """
    from jax.scipy.special import gammaln
    LOG_2PI = float(jnp.log(2.0 * jnp.pi))
    from dce_mri.jax_forward import _EPS

    def log_lik(p_aug, data: LesionData):
        n = data.n_eff
        if noise == "marginalized":
            r = (predict_data(p_aug, data) - data.sig) * data.weights
            chi2 = jnp.maximum(jnp.sum(r ** 2), _TINY)
            ll = (gammaln(0.5 * n) - 0.5 * n * jnp.log(0.5 * chi2)
                  - 0.5 * n * LOG_2PI)
        elif noise == "free":
            s = p_aug[-1]
            r = (predict_data(p_aug[:-1], data) - data.sig) * data.weights
            s_safe = jnp.maximum(s, _EPS)
            ll = (-0.5 * n * (LOG_2PI + 2.0 * jnp.log(s_safe))
                  - 0.5 * jnp.sum(r ** 2) / (s_safe * s_safe))
            ll = jnp.where(s > 0, ll, -1e30)
        else:
            raise ValueError(f"unknown noise {noise!r}")
        return jnp.where(jnp.isfinite(ll), ll, -1e30)

    return log_lik


def make_unconstrained_logdensity_data(log_lik_data: Callable, transform) -> Callable:
    """`logdensity(z, data)`; same composition as
    bayes_blackjax.make_unconstrained_logdensity."""
    def logdensity(z, data):
        x = transform.constrain(z)
        return (log_lik_data(x, data)
                + transform.log_prior(z)
                + transform.log_det_jacobian(z))
    return logdensity


# =============================================================================
# Per-lesion preparation (CPU) -- mirrors fit_curve_bayes up to the sampler
# =============================================================================

@dataclass
class PreparedLesion:
    data:           LesionData
    init_point:     Optional[np.ndarray]   # constrained, or None -> prior draws
    point_estimate: Optional[np.ndarray]   # NLLS anchor
    sigma_fixed:    Optional[float]
    layout:         object
    prior:          object
    ctx:            StaticCtx
    fwd_kwargs:     dict
    cfg_bayes:      BayesConfig            # with this lesion's seed / delay_fixed
    prep_s:         float


def _forward_kwargs(model, cfg_bayes, wx_model, relax_cfg, acq) -> dict:
    """The make_predict options fit_curve_bayes's blackjax branch uses."""
    has_tau = wx_model is not None and wx_model.n_params > 0
    uses_tb = has_tau or (wx_model is not None
                          and getattr(wx_model, "uses_tau_b_fixed", False))
    return dict(
        model_name  = model.name,
        wx          = wx_model is not None,
        wx_name     = wx_model.name if wx_model is not None else "SSM3",
        fit_tau_b   = relax_cfg.fit_tau_b if has_tau else False,
        tau_b_fixed = float(relax_cfg.tau_b_fixed) if uses_tb else 0.0,
        delay       = (cfg_bayes.delay_mode == "joint"),
        upsample    = int(acq.upsample),
    )


def prepare_lesion(sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
                   *, wx_model=None, aif_cfg=None, relax_cfg=None,
                   point_fit=None) -> PreparedLesion:
    """
    Everything fit_curve_bayes does BEFORE calling sample_blackjax, for one
    lesion: NLLS anchor -> init point, prior, S0, (shifted) AIF. Copied from
    fit_curve_bayes rather than refactored out of it, so existing code is
    untouched; check_13 confirms the two give the same draws.

    Pure CPU/numpy apart from building LesionData -- safe to run in a worker
    process (with JAX_PLATFORMS=cpu) while the GPU samples another batch.
    """
    from dce_mri.signal_models import compute_S0, shift_aif

    t0 = time.time()
    if cfg_bayes.backend.lower() != "blackjax":
        raise ValueError("bayes_batched is BlackJAX-only (backend='blackjax')")
    if cfg_bayes.delay_mode == "marginalized":
        raise NotImplementedError(
            "delay_mode='marginalized' is not supported by the blackjax path")
    if cfg_bayes.noise not in _NOISE_MAP:
        raise NotImplementedError(
            f"noise={cfg_bayes.noise!r} is not wired through make_log_lik_data")
    _validate_wx_compatibility(wx_model, cfg_fit.fit_space, model,
                               time_var, aif, aif_cfg, relax_cfg)

    noise = NOISE_OPTIONS[cfg_bayes.noise]
    layout = build_param_layout(
        model, cfg_bayes, aif=aif, time_var=time_var, acq=acq,
        wx_model=wx_model, relax_cfg=relax_cfg)

    # --- NLLS anchor: same block as fit_curve_bayes ---------------------------
    sigma_fixed = cfg_bayes.sigma_fixed
    init_point = None
    if point_fit is None and (cfg_bayes.init_from_nlls or
                              (noise.n_params == 0 and not noise.marginalized
                               and sigma_fixed is None)):
        aif_pf = aif
        if cfg_bayes.delay_mode == "fixed":
            aif_pf = shift_aif(aif, float(cfg_bayes.delay_fixed), time_var, acq)
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
            if layout.n_noise == 2:
                init_point = np.concatenate([pe, [sigma_fixed, 0.2]])
    else:
        pe = None

    # the prior exactly as fit_curve_bayes builds it
    _, layout, prior, _ = make_log_posterior(
        sig, aif, time_var, model, acq, subject, cfg_fit, cfg_bayes,
        wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        layout=layout, sigma_fixed=sigma_fixed)
    if init_point is not None and not np.isfinite(prior.log_pdf(init_point)):
        init_point = None

    # --- per-lesion data: same quantities as fit_curve_bayes's blackjax branch
    ctx = static_ctx_from_configs(acq, subject, aif_cfg)
    S0 = float(compute_S0(np.asarray(sig)[None, :], acq)[0])
    aif_use = aif
    if cfg_bayes.delay_mode == "fixed":
        aif_use = shift_aif(aif, float(cfg_bayes.delay_fixed), time_var, acq)
    T = len(sig)
    data = LesionData(
        sig=jnp.asarray(sig, dtype=jnp.float64),
        aif=jnp.asarray(aif_use, dtype=jnp.float64),
        S0=jnp.asarray(S0, dtype=jnp.float64),
        baseline_val=jnp.asarray(float(np.mean(np.asarray(aif_use)[acq.nbase])),
                                 dtype=jnp.float64),
        weights=jnp.ones(T, dtype=jnp.float64),
        n_eff=jnp.asarray(float(T), dtype=jnp.float64),
    )
    return PreparedLesion(
        data=data, init_point=init_point, point_estimate=pe,
        sigma_fixed=sigma_fixed, layout=layout, prior=prior, ctx=ctx,
        fwd_kwargs=_forward_kwargs(model, cfg_bayes, wx_model, relax_cfg, acq),
        cfg_bayes=cfg_bayes, prep_s=time.time() - t0)


# =============================================================================
# Step 2 -- batched sampler
# =============================================================================

def _init_and_keys(transform, prior, init_point, n_chains, D, seed):
    """
    Initial positions and keys, derived EXACTLY as sample_blackjax derives
    them (same splits, same order), so a lesion fitted in a batch gets the
    same random stream as when fitted alone with the same seed.
    Returns z_init (C, D), warm_keys (C, 2), k_run (2,).
    """
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    if init_point is not None and np.all(np.isfinite(init_point)):
        z0 = transform.unconstrain(jnp.asarray(init_point, dtype=jnp.float64))
        jitter = jax.random.normal(k_init, (n_chains, D)) * 0.5
        z_init = z0[None, :] + jitter
        z_init = z_init.at[0].set(z0)
    else:
        rng = np.random.default_rng(seed)
        x0 = prior.sample(rng, n_chains)
        z_init = jnp.stack([transform.unconstrain(jnp.asarray(x, dtype=jnp.float64))
                            for x in x0])
    key, k_warm = jax.random.split(key)
    warm_keys = jax.random.split(k_warm, n_chains)
    key, k_run = jax.random.split(key)
    return z_init, warm_keys, k_run


def make_batched_runner(logdensity_data: Callable, *, n_chains: int, n_tune: int,
                        n_draws: int, dense: bool, target_accept: float) -> Callable:
    """
    Returns run(warm_keys (L,C,2), k_run (L,2), z_init (L,C,D), data[L])
        -> positions (L,C,N,D) unconstrained,
           is_divergent (L,C,N), n_leapfrog (L,C,N),
           step_size (L,C), inverse_mass_matrix (L,C,D[,D])

    Per chain: window adaptation on its own start (as sample_blackjax), then
    n_draws NUTS steps. vmapped over chains, then over lesions, jitted once.

    Kept per step: position, divergence flag, number of leapfrog steps --
    NOT BlackJAX's full NUTSInfo, which also carries momentum and both
    trajectory end states (~8x the size of the draws). The warmup's per-step
    record is dropped the same way (get_filter_adapt_info_fn). Neither
    changes the draws; they only change what is stored.
    """
    import blackjax
    from blackjax.adaptation.base import get_filter_adapt_info_fn

    def one_chain(warm_key, run_keys, z0, data):
        logdensity = lambda z: logdensity_data(z, data)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity,
            is_mass_matrix_diagonal=not dense,
            target_acceptance_rate=target_accept,
            adaptation_info_fn=get_filter_adapt_info_fn())
        (state, params), _ = warmup.run(warm_key, z0, num_steps=n_tune)
        kernel = blackjax.nuts(logdensity, step_size=params["step_size"],
                               inverse_mass_matrix=params["inverse_mass_matrix"])

        def step(st, k):
            st, info = kernel.step(k, st)
            return st, (st.position, info.is_divergent, info.num_integration_steps)

        _, (pos, div, nlf) = jax.lax.scan(step, state, run_keys)
        return pos, div, nlf, params["step_size"], params["inverse_mass_matrix"]

    def one_lesion(warm_keys, k_run, z_init, data):
        # sample_blackjax: keys = split(k_run, n_draws); at draw i, chain c
        # uses split(keys[i], n_chains)[c]. Rebuilt here as (C, N, 2).
        per_draw = jax.random.split(k_run, n_draws)
        run_keys = jnp.swapaxes(
            jax.vmap(lambda k: jax.random.split(k, n_chains))(per_draw), 0, 1)
        return jax.vmap(one_chain, in_axes=(0, 0, 0, None))(
            warm_keys, run_keys, z_init, data)

    return jax.jit(jax.vmap(one_lesion))


def _ctx_signature(ctx: StaticCtx):
    return tuple(np.round(np.atleast_1d(np.asarray(v, float)), 12).tobytes()
                 for v in ctx)


def fit_curve_bayes_batched(
    sigs, aifs, time_var, model, acqs, subjects, cfg_fit, cfg_bayes,
    *, delay_fixed=None, seeds=None,
    wx_model=None, aif_cfg=None, relax_cfg=None, point_fits=None,
    prepared: Optional[Sequence[PreparedLesion]] = None,
    verbose: bool = True,
) -> list:
    """
    Fit L lesions with independent NUTS chains, all stepped in one batch.
    Returns a list of BayesResult, one per lesion, in input order --
    the same object fit_curve_bayes returns, so downstream code
    (summarise, plots, rows) is unchanged.

    sigs, aifs    : length-L sequences of (T,) arrays (same T)
    time_var      : (T,) shared time grid
    acqs/subjects : one config shared by all, or a length-L sequence
                    (per-lesion acq.nbase is fine -- it only enters S0 and
                    the AIF baseline, which are per-lesion data)
    cfg_bayes     : shared settings; per-lesion overrides via
    delay_fixed   : length-L delays for delay_mode='fixed' (else ignored)
    seeds         : length-L seeds; default cfg_bayes.seed for every lesion,
                    which is what the per-lesion drivers do today
    prepared      : already-prepared lesions (skips the CPU preparation)

    Timing in each result's meta['batch']: prep, compile and sampling
    seconds for the WHOLE batch; elapsed_s is the batch wall time / L.
    """
    t_all = time.time()
    L = len(sigs) if prepared is None else len(prepared)

    def _per(x, i):
        return x[i] if isinstance(x, (list, tuple)) else x

    if prepared is None:
        prepared = []
        for i in range(L):
            cb = BayesConfig(**{**cfg_bayes.__dict__,
                                "seed": int(seeds[i]) if seeds is not None else cfg_bayes.seed,
                                "delay_fixed": (float(delay_fixed[i]) if delay_fixed is not None
                                                else cfg_bayes.delay_fixed)})
            prepared.append(prepare_lesion(
                np.asarray(sigs[i], float), np.asarray(aifs[i], float), time_var,
                model, _per(acqs, i), _per(subjects, i), cfg_fit, cb,
                wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
                point_fit=_per(point_fits, i) if point_fits is not None else None))
    t_prep = time.time() - t_all

    # --- everything shared must actually be shared --------------------------
    p0 = prepared[0]
    for i, p in enumerate(prepared[1:], start=1):
        if p.fwd_kwargs != p0.fwd_kwargs:
            raise ValueError(f"lesion {i}: forward-model options differ "
                             f"({p.fwd_kwargs} vs {p0.fwd_kwargs})")
        if _ctx_signature(p.ctx) != _ctx_signature(p0.ctx):
            raise ValueError(f"lesion {i}: TR/FA/T10/B1/r1/blood T1/hematocrit "
                             f"differ from lesion 0; cannot share one compiled model")
        if (tuple(p.layout.param_names) != tuple(p0.layout.param_names)
                or not np.array_equal(p.layout.lb, p0.layout.lb)
                or not np.array_equal(p.layout.ub, p0.layout.ub)
                or p.prior.sources != p0.prior.sources):
            raise ValueError(f"lesion {i}: parameter layout / prior differs "
                             f"from lesion 0; the box transform must be shared")
        if p.data.sig.shape != p0.data.sig.shape:
            raise ValueError(f"lesion {i}: {p.data.sig.shape[0]} frames vs "
                             f"{p0.data.sig.shape[0]}")

    cb0 = p0.cfg_bayes
    layout, prior = p0.layout, p0.prior
    D = layout.n_total
    transform = build_transform(prior, D)
    fk = dict(p0.fwd_kwargs)
    predict = make_predict_data(fk.pop("model_name"), time_var, p0.ctx, **fk)
    log_lik = make_log_lik_data(predict, noise=_NOISE_MAP[cb0.noise])
    logdensity = make_unconstrained_logdensity_data(log_lik, transform)

    target_accept = cb0.target_accept
    if target_accept == 0.234:            # same override as sample_blackjax
        target_accept = 0.8
    dense = getattr(cb0, "dense_mass_matrix", False)

    inits = [_init_and_keys(transform, p.prior, p.init_point, cb0.n_chains, D,
                            p.cfg_bayes.seed) for p in prepared]
    z_init = jnp.stack([a[0] for a in inits])
    warm_keys = jnp.stack([a[1] for a in inits])
    k_run = jnp.stack([a[2] for a in inits])
    data = stack_lesion_data([p.data for p in prepared])

    run = make_batched_runner(logdensity, n_chains=cb0.n_chains, n_tune=cb0.n_tune,
                              n_draws=cb0.n_draws, dense=dense,
                              target_accept=target_accept)
    t0 = time.time()
    compiled = run.lower(warm_keys, k_run, z_init, data).compile()
    t_compile = time.time() - t0
    if verbose:
        print(f"  [batched] {L} lesions x {cb0.n_chains} chains: "
              f"prep {t_prep:.1f} s, compile {t_compile:.1f} s", flush=True)

    t0 = time.time()
    pos, div, nlf, step_size, imm = jax.block_until_ready(
        compiled(warm_keys, k_run, z_init, data))
    t_sample = time.time() - t0

    # back to constrained space, on device, then to host
    to_x = jax.jit(jax.vmap(jax.vmap(jax.vmap(transform.constrain))))
    x = np.asarray(jax.device_get(to_x(pos)))                  # (L, C, N, D)
    div = np.asarray(jax.device_get(div))
    nlf = np.asarray(jax.device_get(nlf))
    step_size = np.asarray(jax.device_get(step_size))
    imm = np.asarray(jax.device_get(imm))
    if verbose:
        print(f"  [batched] sampling {t_sample:.1f} s "
              f"({t_sample / (cb0.n_tune + cb0.n_draws) * 1e3:.1f} ms per NUTS step, "
              f"whole batch)", flush=True)

    # lock-step cost: per step the batch waits for its longest trajectory
    lockstep = dict(
        mean_leapfrog=float(nlf.mean()),
        mean_of_batch_max=float(nlf.max(axis=(0, 1)).mean()),
        mean_of_lesion_max=float(nlf.max(axis=1).mean()),
    )

    t_wall = time.time() - t_all
    results = []
    for i, p in enumerate(prepared):
        chains = x[i]
        rhat, ess, mcse_mean, mcse_median = _diagnostics(chains, layout.param_names)
        info = dict(
            step_size=step_size[i],
            step_size_mean=float(step_size[i].mean()),
            step_size_spread=float(step_size[i].max() - step_size[i].min()),
            dense_mass_matrix=bool(dense),
            target_accept=float(target_accept),
            inverse_mass_matrix=imm[i],
            n_leapfrog_mean=float(nlf[i].mean()),
            n_leapfrog_max=int(nlf[i].max()),
        )
        meta = {"blackjax": info,
                "batch": dict(index=i, size=L, prep_s=t_prep, compile_s=t_compile,
                              sample_s=t_sample, wall_s=t_wall, **lockstep),
                "noise": NOISE_OPTIONS[p.cfg_bayes.noise].name,
                "delay_mode": p.cfg_bayes.delay_mode,
                "delay_fixed": (p.cfg_bayes.delay_fixed
                                if p.cfg_bayes.delay_mode == "fixed" else None),
                "prior_sources": p.prior.sources,
                "prior_is_flat": p.prior.is_flat,
                "sigma_fixed": p.sigma_fixed}
        results.append(BayesResult(
            param_names=layout.param_names, chains=chains,
            point_estimate=p.point_estimate, rhat=rhat, ess=ess,
            mcse_mean=mcse_mean, mcse_median=mcse_median, accept_frac=None,
            n_divergences=int(div[i].sum()), backend="blackjax",
            elapsed_s=t_wall / L, meta=meta))
    return results
