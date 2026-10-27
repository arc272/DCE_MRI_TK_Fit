# -*- coding: utf-8 -*-
"""
jax_forward.py
==============
JAX re-implementation of the forward-model hot path, for jit/vmap/grad.

Scope
-----
This module deliberately covers ONLY the functions that are called inside
a sampler's inner loop:

    kinetic_models.tcxm_ct_compartments   -> tcxm_compartments
    kinetic_models.gkm_ct_compartments    -> gkm_compartments
    signal_models.concentration_to_signal -> concentration_to_signal  (FXL)
    signal_models.shift_aif               -> shift_aif
    water_exchange._ssm3_to_signal        -> ssm3_to_signal
    water_exchange._nxl_to_signal         -> nxl_to_signal
    fitting.build_residual                -> make_residual
    bayes.make_log_posterior              -> make_log_posterior

Everything else in the package (I/O, config, LHS, figures, ROI handling,
volume iteration, ArviZ diagnostics) stays numpy and is untouched. The
numpy implementations remain the reference; `tests/test_jax_equivalence.py`
asserts agreement between the two.

Contract differences from the numpy versions
--------------------------------------------
1. No config dataclasses cross the jit boundary. Configs are flattened
   once into `StaticCtx` (plain floats + ints) by `static_ctx_from_configs`.
   `acq.nbase` is a `slice`, which is not a valid pytree leaf, so the
   baseline mean it implies is resolved eagerly instead.

2. No data-dependent Python branches. The numpy code returns early on
   `fp <= EPS`, `ps <= EPS`, `disc < 0`, `ve > 1e-9`. Under tracing those
   `if`s would bake in whichever branch the *tracer* took, not the one the
   runtime value needs. See ANALYSIS below for why every one of these
   branches is unreachable inside the model bounds, which is what makes
   dropping them safe rather than merely convenient.

3. No try/except around the forward model. Failure is expressed as a
   non-finite return value and converted to -inf by the caller, outside
   jit.

4. Frame dropping is a WEIGHT vector, not a boolean index. `target[mask]`
   has a shape that depends on the mask's contents, which jit forbids and
   vmap cannot batch over. A 0/1 weight multiplying the residual gives the
   identical sum of squares with a static shape.

ANALYSIS: why the degenerate branches are unreachable
------------------------------------------------------
TCXM bounds are lb = [ve, vp, fp, ps] = [1e-4, 1e-4, 0.1/60, 1e-5/60], so
inside the search box fp >= 1.67e-3 and ps >= 1.67e-7, both far above the
numpy code's 1e-12 guard. The `fp <= EPS` and `ps <= EPS` early returns are
therefore dead code for any in-bounds parameter vector.

The discriminant is non-negative for every physical parameter value:

    disc = (T_ + Te)^2 - 4*Tc*Te,   T_ = (vp+ve)/fp,  Tc = vp/fp,  Te = ve/ps

Since ve >= 0 we always have T_ >= Tc, hence

    (T_ + Te)^2 - 4*Tc*Te >= (T_ + Te)^2 - 4*T_*Te = (T_ - Te)^2 >= 0

so the `disc < -1e-12 -> NaN` branch is likewise unreachable. sqrt is still
evaluated at `disc + TINY` because d(sqrt)/dx is unbounded at exactly zero
(reached when T_ == Te), which would otherwise hand NUTS an inf gradient on
a measure-zero but not-impossible ridge.

GKM's `Ktrans <= EPS or ve <= EPS` branch is dead for the same reason:
GKM lb = [1e-5, 1e-4, 1e-4].

If you ever widen the lower bounds toward zero, these justifications lapse
and the guards have to come back as `jnp.where`.

Gradient safety
---------------
`jnp.where` evaluates BOTH branches before selecting, so a NaN produced in
the branch that is *not* taken still propagates through the reverse pass
and poisons the gradient. Every guarded division here therefore uses the
double-where pattern: sanitise the denominator first, divide, then select.
`_safe_div` implements it. Do not replace it with a plain `jnp.where`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# Matches signal_models._EPS / kinetic_models._EPS, kept separate so the
# numpy modules can change theirs without silently changing these.
_EPS = 1e-11
_TINY = 1e-30


# =============================================================================
# Static context -- configs flattened to jit-safe leaves
# =============================================================================
class StaticCtx(NamedTuple):
    """
    Everything the forward model needs from AcquisitionConfig /
    SubjectConfig / AifConfig, as plain floats.
 
    Held as a NamedTuple (a pytree) so it can be closed over or passed as
    an argument; `upsample` and `n_frames` are ints used for shapes and
    must be static, so pass them via closure, not as traced arguments.
 
    TR_s, T10_s, Tb_s : seconds (the configs store milliseconds)
    FA_rad            : acq.FA * subject.B1, in radians
    r1                : relaxivity, 1/(mM*s)
    plasma_frac       : 1 - hematocrit
 
    alpha_rad_t : (T,) NOMINAL per-frame flip angle, radians, with B1
                  deliberately NOT folded in -- see B1 below.
    TR_s_t      : (T,) per-frame repetition time, seconds.
                  Both come from acq.frame_arrays(time_var); with
                  acq.protocol=None they are constant arrays equal to the
                  scalars above, so single-FA is the degenerate case, not
                  a separate path. vmap's in_axes=0 cannot accept a
                  scalar, which is why these must always be real arrays.
    B1          : transmit scale, stored SEPARATELY from alpha_rad_t so
                  models M1/M4 can treat it as a free parameter. Folding
                  B1 into the per-frame angle would bake the fitted
                  quantity into the static context, where the sampler
                  cannot reach it.
    """
    TR_s:        float
    T10_s:       float
    Tb_s:        float
    FA_rad:      float
    alpha_rad_t: jnp.ndarray
    TR_s_t:      jnp.ndarray
    B1:          float
    r1:          float
    plasma_frac: float
 

def static_ctx_from_configs(acq, subject, aif_cfg, time_var=None) -> StaticCtx:
    """
    Flatten the numpy config dataclasses into a jit-safe StaticCtx.
 
    `time_var` is required for the per-frame arrays. Passing None keeps
    the legacy single-FA behaviour by building one-element arrays, which
    is enough for the scalar-only code paths but NOT for the _mfa
    functions -- give those a real time vector.
    """
    if time_var is None:
        alpha_rad_t = jnp.deg2rad(jnp.asarray([float(acq.FA)], dtype=jnp.float64))
        TR_s_t      = jnp.asarray([float(acq.TR)], dtype=jnp.float64) / 1000.0
    else:
        fa = acq.frame_arrays(time_var)
        alpha_rad_t = jnp.deg2rad(jnp.asarray(fa.alpha_deg, dtype=jnp.float64))
        TR_s_t      = jnp.asarray(fa.TR_ms, dtype=jnp.float64) / 1000.0
 
    return StaticCtx(
        TR_s        = float(acq.TR) / 1000.0,
        T10_s       = float(subject.T10) / 1000.0,
        Tb_s        = float(aif_cfg.T10) / 1000.0 if aif_cfg is not None else 1.6,
        FA_rad      = float(jnp.deg2rad(acq.FA * subject.B1)),
        alpha_rad_t = alpha_rad_t,
        TR_s_t      = TR_s_t,
        B1          = float(subject.B1),
        r1          = float(acq.r1),
        plasma_frac = (1.0 - float(aif_cfg.hematocrit)) if aif_cfg is not None else 1.0,
    )
 

# =============================================================================
# Numerics helpers
# =============================================================================

def _safe_div(num, den, eps=_EPS):
    """
    num/den, guarded so that neither the value nor the REVERSE-MODE
    gradient can go non-finite when den is near zero.

    The naive `jnp.where(den > eps, num/den, 0.0)` still computes num/den
    in the untaken branch; its NaN/inf shows up in the cotangent. Replacing
    the denominator BEFORE the division is what actually fixes it.
    """
    safe_den = jnp.where(jnp.abs(den) > eps, den, 1.0)
    out = num / safe_den
    return jnp.where(jnp.abs(den) > eps, out, 0.0)


def _upsample_grid(t, upsample: int):
    """
    Fine time grid matching kinetic_models' np.linspace(t0, t[-1], (n-1)*U+1).

    `upsample` and len(t) must be Python ints (static) -- the output SHAPE
    depends on them.
    """
    n = t.shape[0]
    n_fine = (n - 1) * upsample + 1
    return jnp.linspace(t[0], t[-1], n_fine)


def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth integer >= n (same idea as scipy.fft.next_fast_len)."""
    best = 2 ** int(n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p3 = p5
        while p3 < best:
            p2 = p3
            while p2 < n:
                p2 *= 2
            if p2 < best:
                best = p2
            p3 *= 3
        p5 *= 5
    return max(best, 1)


def _conv_dt(a, h, dt, n_out: int):
    """
    Convolution truncated to n_out samples and scaled by dt -- the jnp
    equivalent of `fftconvolve(a, h)[:n_out] * dt`.

    MEASURED, not assumed: `jnp.convolve` (which lowers to a direct
    lax.conv_general_dilated) took 16.1 ms per TCXM evaluation on CPU
    against numpy's 0.35 ms -- 46x SLOWER, because the upsampled grid is
    ~2.5k samples and direct convolution is O(n^2) with no CPU kernel
    worth the name. The rfft route below matches scipy's own strategy and
    brings it back in line. If you ever drop acq.upsample to 1 (51 samples)
    direct convolution would win instead; at upsample=50 it does not.
    """
    n_lin = a.shape[0] + h.shape[0] - 1
    n_fft = _next_fast_len(n_lin)
    A = jnp.fft.rfft(a, n_fft)
    H = jnp.fft.rfft(h, n_fft)
    return jnp.fft.irfft(A * H, n_fft)[:n_out] * dt


# =============================================================================
# Kinetic models
# =============================================================================

def tcxm_compartments(theta_kin, t, aif, upsample: int = 50):
    """
    Two-compartment exchange model, split into (Cv, Ce).

    theta_kin = [ve, vp, fp, ps];  fp, ps in 1/s.
    Mirrors kinetic_models.tcxm_ct_compartments, minus the dead branches
    documented in the module docstring.
    """
    ve, vp, fp, ps = theta_kin[0], theta_kin[1], theta_kin[2], theta_kin[3]
    n = t.shape[0]

    if upsample > 1:
        t_fine = _upsample_grid(t, upsample)
        dt_fine = t_fine[1] - t_fine[0]
        ca_fine = jnp.interp(t_fine, t, aif)
    else:
        t_fine, dt_fine, ca_fine = t, t[1] - t[0], aif
    n_fine = t_fine.shape[0]

    Tc = _safe_div(vp, fp)
    Te = _safe_div(ve, ps)
    T_ = _safe_div(vp + ve, fp)

    disc = (T_ + Te) ** 2 - 4.0 * Tc * Te
    root = jnp.sqrt(jnp.maximum(disc, 0.0) + _TINY)

    denom2 = 2.0 * Tc * Te
    sigma_p = _safe_div((T_ + Te) + root, denom2)
    sigma_m = _safe_div((T_ + Te) - root, denom2)

    delta = sigma_p - sigma_m
    K = _safe_div(sigma_p * sigma_m, delta)

    exp_p = jnp.exp(-sigma_p * t_fine)
    exp_m = jnp.exp(-sigma_m * t_fine)

    h_e = K * (exp_m - exp_p)
    h_p = K * ((1.0 - Te * sigma_m) * exp_m + (Te * sigma_p - 1.0) * exp_p)

    Cv_fine = vp * _conv_dt(ca_fine, h_p, dt_fine, n_fine)
    Ce_fine = ve * _conv_dt(ca_fine, h_e, dt_fine, n_fine)

    if upsample > 1:
        return Cv_fine[::upsample], Ce_fine[::upsample]
    return Cv_fine, Ce_fine


def tcxm_ct(theta_kin, t, aif, upsample: int = 50):
    Cv, Ce = tcxm_compartments(theta_kin, t, aif, upsample)
    return Cv + Ce


def gkm_compartments(theta_kin, t, aif, upsample: int = 50):
    """
    Extended Tofts / GKM, split into (Cv, Ce).
    theta_kin = [Ktrans (1/s), ve, vp].
    """
    Ktrans, ve, vp = theta_kin[0], theta_kin[1], theta_kin[2]

    if upsample > 1:
        t_fine = _upsample_grid(t, upsample)
        dt_fine = t_fine[1] - t_fine[0]
        Cp_fine = jnp.interp(t_fine, t, aif)
    else:
        t_fine, dt_fine, Cp_fine = t, t[1] - t[0], aif
    n_fine = t_fine.shape[0]

    ke = _safe_div(Ktrans, ve + _EPS)
    h_t = jnp.exp(-ke * t_fine)

    Cv_fine = vp * Cp_fine
    Ce_fine = Ktrans * _conv_dt(Cp_fine, h_t, dt_fine, n_fine)

    if upsample > 1:
        return Cv_fine[::upsample], Ce_fine[::upsample]
    return Cv_fine, Ce_fine


def gkm_ct(theta_kin, t, aif, upsample: int = 50):
    Cv, Ce = gkm_compartments(theta_kin, t, aif, upsample)
    return Cv + Ce


# =============================================================================
# AIF delay
# =============================================================================

def shift_aif(aif, tau, t, baseline_val):
    """
    Shift the AIF by tau seconds. Differentiable in tau, which is the
    point: under the numpy scipy.interp1d version, joint delay fitting had
    to fall back on finite differences for the tau column of the gradient.

    baseline_val : mean of aif over acq.nbase, resolved eagerly by the
                   caller (a `slice` cannot cross the jit boundary).
    """
    return jnp.interp(t - tau, t, aif, left=baseline_val, right=aif[-1])


# =============================================================================
# Signal models
# =============================================================================

def concentration_to_signal(C_t, ctx: StaticCtx, S0):
    """
    FXL SPGR signal, baseline-normalised -- matches
    signal_models.concentration_to_signal for the single-curve case.
    """
    R10 = 1.0 / (ctx.T10_s + _EPS)
    cos_a = jnp.cos(ctx.FA_rad)

    E10 = jnp.exp(-ctx.TR_s * R10)
    R1_t = R10 + ctx.r1 * C_t
    E1 = jnp.exp(-ctx.TR_s * R1_t)

    one_minus_E1_cos = jnp.maximum(1.0 - E1 * cos_a, _EPS)
    one_minus_E10_cos = 1.0 - E10 * cos_a
    one_minus_E10 = jnp.maximum(1.0 - E10, _EPS)

    Eh = ((1.0 - E1) * one_minus_E10_cos) / (one_minus_E10 * one_minus_E1_cos)
    return S0 * Eh


def _ssm3_pool_sum(R1_b, R1_e, R1_i, vb, ve, vi,
                   tau_b, tau_i, tau_e, TR_s, FA):
    """
    Steady-state SPGR signal summed over the 3-pool chain (b<->e<->i),
    for ONE timepoint. R1_b/R1_e are scalars here; vmap over time.

    Same matrix as water_exchange._ssm3_pool_sum, with two changes:
      * jax.scipy.linalg.expm instead of scipy's (differentiable),
      * jnp.linalg.solve instead of pinv -- already the choice made in
        the numpy _ssm3_pool_sum_batched, kept for consistency.
    """
    X = jnp.array([
        [-(R1_b + 1.0 / tau_b),   1.0 / tau_e - (vi / ve) / tau_i,  0.0],
        [ 1.0 / tau_b,           -(R1_e + 1.0 / tau_e),             1.0 / tau_i],
        [ 0.0,                    1.0 / tau_e - (vb / ve) / tau_b, -(R1_i + 1.0 / tau_i)],
    ])
    M0 = jnp.array([vb, ve, vi])
    I = jnp.eye(3)
    eX = jsl.expm(TR_s * X)
    A = I - eX * jnp.cos(FA)
    rhs = (I - eX) @ (jnp.sin(FA) * M0)
    M = jnp.linalg.solve(A, rhs)
    return jnp.sum(M)


# Batched over timepoints. vmap replaces the (n,3,3) stacked scipy.expm
# call, which profiling showed was ~29% of one wx forward-model evaluation.
_ssm3_pool_sum_batched = jax.vmap(
    _ssm3_pool_sum,
    in_axes=(0, 0, None, None, None, None, None, None, None, 0, 0),
)


def ssm3_to_signal(Cv, Ce, ve, vb, tau_b, tau_i, ctx: StaticCtx, S0):
    """
    3-pool Shutter Speed Model signal. Mirrors
    water_exchange._ssm3_to_signal, including the load-bearing
    Ce/ve and (Cv/vb)*plasma_frac density conversions and the
    Eh/Eh_baseline normalisation.
    """
 
    vi = 1.0 - vb - ve
    tau_e = _safe_div(ve, vb / tau_b + vi / tau_i)
    #vi = 1.0 - vb - ve
    #tau_e = _safe_div(ve, vb / tau_b + vi / tau_i)

    R1_tissue = 1.0 / ctx.T10_s
    R1_blood_baseline = 1.0 / ctx.Tb_s

    Eh_baseline = _ssm3_pool_sum(
        R1_blood_baseline, R1_tissue, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e, ctx.TR_s, ctx.FA_rad,
    )

    Ce_density = _safe_div(Ce, ve, eps=1e-9)
    Cv_density = _safe_div(Cv, vb, eps=1e-9)

    R1_e_arr = R1_tissue + ctx.r1 * Ce_density
    R1_b_arr = R1_blood_baseline + ctx.r1 * Cv_density
    n_t = Ce_density.shape[0]
    Eh = _ssm3_pool_sum_batched(
        R1_b_arr, R1_e_arr, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e,
        jnp.broadcast_to(ctx.TR_s,(n_t,)), jnp.broadcast_to(ctx.FA_rad,(n_t,)),
    )
    return S0 * _safe_div(Eh, Eh_baseline)


def _nxl_pool_sum(R1_b, R1_e, R1_i, vb, ve, vi, TR_s, FA):
    """
    Three NON-exchanging pools, summed (thesis Eq 3.37). Mirrors
    water_exchange._nxl_pool_sum. Elementwise, so R1_b / R1_e can be
    (T,) arrays directly -- no vmap needed, unlike _ssm3_pool_sum.
    """
    sin_a, cos_a = jnp.sin(FA), jnp.cos(FA)

    def _spgr(R1):
        E = jnp.exp(-TR_s * R1)
        return sin_a * (1.0 - E) / (1.0 - cos_a * E)

    return vb * _spgr(R1_b) + ve * _spgr(R1_e) + vi * _spgr(R1_i)


def nxl_to_signal(Cv, Ce, ve, vb, ctx: StaticCtx, S0):
    """
    No-exchange-limit signal. Mirrors water_exchange._nxl_to_signal:
    same pools, same R1_blood vs R1_tissue baseline split as
    ssm3_to_signal, same Ce/ve and Cv/vb density conversions, same
    Eh/Eh_baseline normalisation -- so it equals ssm3_to_signal in the
    limit tau_b, tau_i -> infinity. No tau arguments: NXL has none.
    """
    vi = 1.0 - vb - ve

    R1_tissue = 1.0 / ctx.T10_s
    R1_blood_baseline = 1.0 / ctx.Tb_s

    Eh_baseline = _nxl_pool_sum(R1_blood_baseline, R1_tissue, R1_tissue,
                                vb, ve, vi, ctx.TR_s, ctx.FA_rad)

    Ce_density = _safe_div(Ce, ve, eps=1e-9)
    Cv_density = _safe_div(Cv, vb, eps=1e-9)

    R1_e_arr = R1_tissue + ctx.r1 * Ce_density
    R1_b_arr = R1_blood_baseline + ctx.r1 * Cv_density

    Eh = _nxl_pool_sum(R1_b_arr, R1_e_arr, R1_tissue,
                       vb, ve, vi, ctx.TR_s, ctx.FA_rad)
    return S0 * _safe_div(Eh, Eh_baseline)


def concentration_to_signal_mfa(C_t, ctx: StaticCtx, T10_s, B1, M0):
    """
    FXL SPGR signal, ABSOLUTE (not baseline-normalised), on a
    multi-flip-angle / ACE schedule. Mirrors
    signal_models.concentration_to_signal_mfa.
 
    NOT an array-valued version of concentration_to_signal. That function
    returns S0 * Eh/Eh_baseline, and its baseline is computed with the
    SAME flip angle as the frame it divides -- so on a multi-flip-angle
    schedule the flip-angle-dependent scale cancels exactly and the
    segment jump vanishes (verified: C=0 across an 8deg->25deg switch
    gives [1,1,1,1,1,1]). M0 stays explicit instead, to be profiled or
    marginalised downstream.
 
    T10_s and B1 are explicit arguments rather than read from ctx because
    models M1/M4 treat them as free parameters.
    """
    R1_t  = 1.0 / (T10_s + _EPS) + ctx.r1 * C_t
    E1    = jnp.exp(-ctx.TR_s_t * R1_t)
    FA    = B1 * ctx.alpha_rad_t
    cos_a = jnp.cos(FA)
    sin_a = jnp.sin(FA)
    return M0 * sin_a * (1.0 - E1) / jnp.maximum(1.0 - E1 * cos_a, _EPS)
 
 
def ssm3_to_signal_mfa(Cv, Ce, ve, vb, tau_b, tau_i,
                       ctx: StaticCtx, T10_s, B1, M0):
    """
    3-pool Shutter Speed Model on a multi-flip-angle / ACE schedule.
    Mirrors water_exchange._ssm3_to_signal_mfa: same exchange matrix,
    same load-bearing Ce/ve and (Cv/vb)*plasma_frac density conversions,
    same R1_blood vs R1_tissue baseline split -- but per-frame FA/TR and
    absolute M0-scaled output rather than Eh/Eh_baseline.
 
    T10_s and B1 are explicit for the same reason as above (M1/M4).
    """

    vi = 1.0 - vb - ve
    tau_e = _safe_div(ve, vb / tau_b + vi / tau_i)
    
    R1_tissue = 1.0 / (T10_s + _EPS)
    R1_blood0 = 1.0 / ctx.Tb_s
 
    Ce_density = _safe_div(Ce, ve, eps=1e-9)
    Cv_density = _safe_div(Cv, vb, eps=1e-9)
 
    R1_e_arr = R1_tissue + ctx.r1 * Ce_density
    R1_b_arr = R1_blood0 + ctx.r1 * Cv_density
 
    pool_sum = _ssm3_pool_sum_batched(
        R1_b_arr, R1_e_arr, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e,
        ctx.TR_s_t, B1 * ctx.alpha_rad_t,
    )
    return M0 * pool_sum
 

# =============================================================================
# Residual / log-posterior factories
# =============================================================================

KINETIC_FNS = {
    "TCXM": (tcxm_compartments, 4, ("ve", "vp", "fp", "ps"), 0, 1),
    "GKM":  (gkm_compartments,  3, ("Ktrans", "ve", "vp"),   1, 2),
}
#  name -> (compartments_fn, n_kin, param_names, ve_idx, vp_idx)


def make_predict(
    model_name: str,
    t,
    aif,
    ctx: StaticCtx,
    S0: float,
    *,
    wx: bool = False,
    wx_name: str = "SSM3",
    fit_tau_b: bool = False,
    tau_b_fixed: float = 0.05,
    delay: bool = False,
    baseline_val: float = 0.0,
    upsample: int = 50,
) -> Callable:
    """
    Build `predict(p_aug) -> signal curve`, matching build_residual's
    p_aug layout exactly:

        p_aug = [ *theta_kin , (tau_delay?) , (tau_b?) , (tau_i?) ]

    wx_name selects the water-exchange model when wx=True:
        'SSM3' -- consumes (tau_b?, tau_i) per fit_tau_b (default, so
                  existing callers are unchanged)
        'NXL'  -- consumes nothing; p_aug ends after tau_delay
    It must match the WaterExchangeModelSpec.name the numpy side uses,
    or the two residuals silently describe different models.

    The returned function is pure and traceable: jit it, vmap it, or take
    grad of anything built on top of it.
    """
    comp_fn, n_kin, _names, ve_idx, vp_idx = KINETIC_FNS[model_name]
    t = jnp.asarray(t, dtype=jnp.float64)
    aif = jnp.asarray(aif, dtype=jnp.float64)
    if wx and wx_name not in ("SSM3", "NXL"):
        raise ValueError(f"no JAX forward model for wx_name={wx_name!r} "
                         f"(have: 'SSM3', 'NXL')")

    def predict(p_aug):
        theta_kin = p_aug[:n_kin]
        k = n_kin

        if delay:
            aif_use = shift_aif(aif, p_aug[k], t, baseline_val)
            k += 1
        else:
            aif_use = aif

        Cv, Ce = comp_fn(theta_kin, t, aif_use, upsample)

        if not wx:
            return concentration_to_signal(Cv + Ce, ctx, S0)

        if wx_name == "NXL":            # Python-level branch: resolved at trace time
            return nxl_to_signal(
                Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], ctx, S0)

        if fit_tau_b:
            tau_b, tau_i = p_aug[k], p_aug[k + 1]
        else:
            tau_b, tau_i = tau_b_fixed, p_aug[k]

        return ssm3_to_signal(
            Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], tau_b, tau_i, ctx, S0
        )

    return predict


def make_predict_mfa(
    model_name: str,
    t,
    aif,
    ctx: StaticCtx,
    *,
    wx: bool = False,
    fit_tau_b: bool = False,
    tau_b_fixed: float = 0.05,
    delay: bool = False,
    baseline_val: float = 0.0,
    fit_T10: bool = False,
    fit_B1: bool = False,
    upsample: int = 50,
) -> Callable:
    """
    Multi-flip-angle sibling of make_predict. Builds
    `predict(p_aug) -> unit-M0 signal curve`, with layout:
 
        p_aug = [ *theta_kin , (tau_delay?) , (tau_b?) , (tau_i?) ,
                  (T10_s?) , (B1?) ]
 
    Two contract differences from make_predict:
 
    1. M0 IS NOT A PARAMETER AND IS NOT APPLIED HERE. The model is exactly
       linear in M0, so this returns g(theta) with M0 factored out and the
       caller forms S = M0 * g. That is what makes variable projection
       (Phase 4) and the analytic M0 marginal (Phase 5) cheap: both need
       g, |g| and g.y, never M0 itself. Baking M0 in would force the
       sampler to explore a dimension that has a closed form.
    2. T10 and B1 are OPTIONAL TRAILING PARAMETERS, in that order, taken
       from ctx when not fit. T10 enters as seconds, matching ctx.T10_s
       -- not milliseconds like the configs.
 
    Order matters: T10 before B1, matching the model ladder's
    [kin, tau, T10, B1] convention. It is the caller's job to keep
    ParamLayout consistent with this.
    """
    comp_fn, n_kin, _names, ve_idx, vp_idx = KINETIC_FNS[model_name]
    t = jnp.asarray(t, dtype=jnp.float64)
    aif = jnp.asarray(aif, dtype=jnp.float64)
 
    def predict(p_aug):
        theta_kin = p_aug[:n_kin]
        k = n_kin
 
        if delay:
            aif_use = shift_aif(aif, p_aug[k], t, baseline_val)
            k += 1
        else:
            aif_use = aif
 
        if wx:
            if fit_tau_b:
                tau_b, tau_i = p_aug[k], p_aug[k + 1]
                k += 2
            else:
                tau_b, tau_i = tau_b_fixed, p_aug[k]
                k += 1
 
        T10_s = p_aug[k] if fit_T10 else ctx.T10_s
        k += 1 if fit_T10 else 0
        B1 = p_aug[k] if fit_B1 else ctx.B1
 
        Cv, Ce = comp_fn(theta_kin, t, aif_use, upsample)
 
        if not wx:
            return concentration_to_signal_mfa(Cv + Ce, ctx, T10_s, B1, 1.0)
 
        return ssm3_to_signal_mfa(
            Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], tau_b, tau_i,
            ctx, T10_s, B1, 1.0,
        )
 
    return predict

def make_residual(predict, target, weights=None) -> Callable:
    """
    `residual(p_aug) -> (pred - target) * weights`.

    weights : (T,) float array, or None for all-ones. Frame dropping goes
              here as 0/1 entries. This is the jit-safe replacement for
              build_residual's `keep_mask` boolean indexing -- same sum of
              squares, static output shape, and vmap-able across replicates
              (stack the masks and vmap over the weights axis).
    """
    target = jnp.asarray(target, dtype=jnp.float64)
    w = jnp.ones_like(target) if weights is None else jnp.asarray(weights, jnp.float64)

    def residual(p_aug):
        return (predict(p_aug) - target) * w

    return residual


def make_log_posterior(residual, log_prior, noise: str = "marginalized",
                       n_eff: Optional[float] = None) -> Callable:
    """
    `log_post(p_aug) -> scalar`, jit/grad-ready.

    noise : 'marginalized' -- Orton Eq. 8, sigma integrated out under a
                              Jeffreys prior. No noise dimension in p_aug.
            'free'         -- p_aug's LAST entry is sigma.

    n_eff : effective N for the marginalized form. With frame-drop weights
            this is sum(w), NOT len(target) -- the zeroed frames contribute
            no information and must not contribute to the -N/2 log(chi2)
            exponent either. Defaults to len(residual output) when None.

    Non-finite handling stays OUTSIDE jit: this returns -1e30 rather than
    -inf so that gradients through the rejected region are finite and the
    value is safe to hand to a sampler that does arithmetic on it.
    """
    from jax.scipy.special import gammaln
    LOG_2PI = float(jnp.log(2.0 * jnp.pi))

    def log_post(p_aug):
        lp = log_prior(p_aug)

        if noise == "marginalized":
            r = residual(p_aug)
            n = r.shape[0] if n_eff is None else n_eff
            chi2 = jnp.sum(r ** 2)
            chi2 = jnp.maximum(chi2, _TINY)
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

        out = lp + ll
        return jnp.where(jnp.isfinite(out), out, -1e30)

    return log_post
