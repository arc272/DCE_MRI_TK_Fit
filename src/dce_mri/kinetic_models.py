# -*- coding: utf-8 -*-
"""
kinetic_models.py
=================
Forward kinetic models and their self-describing KineticModelSpec definitions.

Design principles
-----------------
Each kinetic model is implemented as a deterministic forward model with the
canonical interface

    Ct = model_fn(theta_kin, time_var, aif)

where

    theta_kin : model-specific kinetic parameters
    time_var  : uniformly sampled time vector (seconds)
    aif       : arterial input function sampled on the same grid

Every model is described by a KineticModelSpec, which bundles together:

    • forward model function
    • parameter names
    • parameter bounds
    • default initial guess
    • feasibility constraints
    • log-scaled optimization dimensions
    • optional compartment-level forward model

This allows fitting, Monte Carlo simulation, optimization, and visualization
code to remain completely model-agnostic.

Models that support water-exchange fitting additionally expose

    fn_compartments(theta, t, aif) -> (Cv, Ce)

which returns separate vascular and extracellular concentration curves instead
of only their summed tissue concentration.

The dispersed-AIF GKM is represented as a family of fixed branch-level model
specifications (DispersedGKM_L1 ... DispersedGKM_L6). Branch level is treated
as a discrete model choice rather than a continuous optimization parameter,
keeping every KineticModelSpec fully callable and multiprocessing-safe.

Public API
----------
Forward model functions

    tcxm_ct_analytic(theta, t, aif)
        Total tissue concentration for the two-compartment exchange model.

    tcxm_ct_compartments(theta, t, aif)
        Returns (Cv, Ce) separately.

    gkm_ct(theta, t, aif)
        Total tissue concentration for the extended Tofts / GKM model.

    gkm_ct_compartments(theta, t, aif)
        Returns (Cv, Ce) separately.

    dispersed_gkm_ct(theta, t, aif, n_level)
        Dispersed-AIF GKM evaluated for one fixed vascular branch level.

Model specifications

    TCXM
        Two-compartment exchange model.

    GKM
        Extended Tofts / generalized kinetic model.

    DISPERSED_GKM
        Base specification for dispersed GKM (metadata only).

    DISPERSED_GKM_LEVELS[n]
        Fully callable KineticModelSpec for branch level n = 1...6.

Utilities

    make_dispersed_gkm_spec(n_level)
        Construct a callable dispersed-GKM specification for one fixed
        branch level.
        """
        
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

_EPS = 1e-12

# Type alias
KineticModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
# KineticModelSpec — self-describing model container
# =============================================================================

@dataclass(frozen=True)
class KineticModelSpec:
    """
    Everything the fitter needs to run and optimise one kinetic model.

    Fields
    ------
    name         : human-readable label used in filenames and plots
    fn           : callable  C_t = fn(theta_kin, time_var, aif)
    param_names  : tuple of parameter name strings (used in plots / outputs)
    lb           : lower bounds array, same length as param_names
    ub           : upper bounds array
    p0           : default initial guess
    constraints  : optional callable returning True if params are feasible
                   e.g. lambda p: p[0] + p[1] <= 1.0
    fn_compartments : optional callable (Cv, Ce) = fn_compartments(
                   theta_kin, time_var, aif) -- same theta_kin as fn,
                   but returns the vascular/extracellular split instead
                   of their sum. None for any model that doesn't
                   support water-exchange fitting yet.
    """
    name:        str
    fn:          KineticModelFn
    param_names: tuple
    lb:          np.ndarray
    ub:          np.ndarray
    p0:          np.ndarray
    constraints: Optional[Callable] = None
    log_dims:    Optional[tuple]    = None  # parameter indices to sample log-uniformly
    fn_compartments: Optional[Callable] = None

# --- replace the three lambdas with these named functions ---

def _tcxm_constraints(p: np.ndarray) -> bool:
    """ve + vp <= 1"""
    return p[0] + p[1] <= 1.0

def _gkm_constraints(p: np.ndarray) -> bool:
    """ve + vp <= 1"""
    return p[1] + p[2] <= 1.0

def _dispersed_gkm_constraints(p: np.ndarray) -> bool:
    """ve + vp <= 1"""
    return p[1] + p[2] <= 1.0
# =============================================================================
# 1. Two-Compartment Exchange Model (TCXM / 2CXM)
# =============================================================================

def tcxm_ct_compartments(
    theta_kin: np.ndarray,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    upsample:  int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-compartment exchange model (2CXM), split into its vascular
    (Cv) and extracellular (Ce) contributions instead of their sum.
    
    Follows Sourbron & Buckley propagator formulation (Table A2/A3).

    Parameters
    ----------
    theta_kin : [ve, vp, fp, ps]  — volume fractions and rate constants (1/s)
    time_var  : uniform 1-D time grid starting at 0  (seconds)
    aif       : arterial input function on the same grid (plasma units)
    upsample  : internal upsampling factor for convolution accuracy

    tcxm_ct_analytic is a thin wrapper around this function (Cv + Ce)
    -- this is the only place the actual convolution math lives.
    """
    ve, vp, fp, ps = (float(x) for x in theta_kin)
    t  = np.asarray(time_var, dtype=float)
    ca = np.asarray(aif,      dtype=float)

    if t.shape[0] < 2:
        return np.zeros_like(t), np.zeros_like(t)
    dt = t[1] - t[0]

    if upsample > 1:
        t_fine  = np.linspace(t[0], t[-1], (len(t) - 1) * upsample + 1)
        dt_fine = t_fine[1] - t_fine[0]
        ca_fine = interp1d(t, ca, kind="linear",
                           bounds_error=False,
                           fill_value=(ca[0], ca[-1]))(t_fine)
    else:
        t_fine, dt_fine, ca_fine = t, dt, ca

    if fp <= _EPS:
        # no flow at all -- nothing enters either compartment
        return np.zeros_like(t), np.zeros_like(t)
    if ps <= _EPS:
        # no permeability -- everything stays vascular, nothing
        # reaches the extracellular compartment. Uses the COARSE
        # ca/dt, not the upsampled grid -- matches the original branch.
        if vp <= _EPS:
            return np.zeros_like(t), np.zeros_like(t)
        k   = fp / max(vp, _EPS)
        h_p = k * np.exp(-k * t)
        Cv  = vp * fftconvolve(ca, h_p)[:len(t)] * dt
        return Cv, np.zeros_like(t)

    Tc   = vp / max(fp, _EPS)
    Te   = ve / max(ps, _EPS)
    T_   = (vp + ve) / max(fp, _EPS)
    disc = (T_ + Te) ** 2 - 4.0 * Tc * Te
    if disc < -1e-12:
        return np.full_like(t, np.nan), np.full_like(t, np.nan)
    disc = max(disc, 0.0)
    root    = np.sqrt(disc)
    denom2  = 2.0 * Tc * Te
    sigma_p = ((T_ + Te) + root) / denom2
    sigma_m = ((T_ + Te) - root) / denom2
    delta = sigma_p - sigma_m
    if abs(delta) < _EPS:
        delta = _EPS

    exp_p = np.exp(-sigma_p * t_fine)
    exp_m = np.exp(-sigma_m * t_fine)
    K     = (sigma_p * sigma_m) / delta
    h_e = K * (exp_m - exp_p)
    h_p = K * ((1.0 - Te * sigma_m) * exp_m + (Te * sigma_p - 1.0) * exp_p)

    cp_fine = fftconvolve(ca_fine, h_p)[:len(t_fine)] * dt_fine
    ce_fine = fftconvolve(ca_fine, h_e)[:len(t_fine)] * dt_fine
    Cv_fine = vp * cp_fine
    Ce_fine = ve * ce_fine
    if upsample > 1:
        return Cv_fine[::upsample], Ce_fine[::upsample]
    return Cv_fine, Ce_fine

def tcxm_ct_analytic(
    theta_kin: np.ndarray,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    upsample:  int = 50,
) -> np.ndarray:
    """
    Two-compartment exchange model (2CXM) closed-form Ct.
    Follows Sourbron & Buckley propagator formulation (Table A2/A3).

    Parameters
    ----------
    theta_kin : [ve, vp, fp, ps]  — volume fractions and rate constants (1/s)
    time_var  : uniform 1-D time grid starting at 0  (seconds)
    aif       : arterial input function on the same grid (plasma units)
    upsample  : internal upsampling factor for convolution accuracy

    Returns
    -------
    Ct : 1-D array, same length as time_var
    """
    Cv, Ce = tcxm_ct_compartments(theta_kin, time_var, aif, upsample)
    return Cv + Ce


TCXM = KineticModelSpec(
    name        = "TCXM",
    fn          = tcxm_ct_analytic,
    param_names = ("ve", "vp", "fp", "ps"),
    lb          = np.array([1e-4,  1e-4,  2.5e-4,  2.5e-4]), # changed lower bounds for fp and ps based on lower limit of detection of simulated data by the bayesian estimation model
    ub          = np.array([0.5,   0.5,   2.5 / 60,  2.5  / 60]),
    p0          = np.array([0.243, 0.196, 0.00635,   0.00515]),
    constraints = _tcxm_constraints, #lambda p: p[0] + p[1] <= 1.0,
    log_dims    = (2, 3),    # fp, ps span orders of magnitude
    fn_compartments = tcxm_ct_compartments, 
)


# =============================================================================
# 2. Generalised Kinetic Model (extended Tofts / GKM)
# =============================================================================
def gkm_ct_compartments(
    theta_kin: np.ndarray,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    upsample:  int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Tofts (GKM), split into vascular (Cv) and extracellular
    (Ce) contributions. gkm_ct is a thin wrapper around this (Cv + Ce).
    
        C(t) = vp·Cp(t) + Ktrans · ∫ Cp(u) · exp[−(Ktrans/ve)(t−u)] du

    theta_kin = [Ktrans (s⁻¹), ve, vp]
    """
    Ktrans, ve, vp = float(theta_kin[0]), float(theta_kin[1]), float(theta_kin[2])
    t  = np.asarray(time_var, dtype=np.float64)
    Cp = np.asarray(aif,      dtype=np.float64)

    if t.shape[0] < 2:
        return np.zeros_like(t), np.zeros_like(t)

    if upsample > 1:
        t_fine  = np.linspace(t[0], t[-1], (len(t) - 1) * upsample + 1)
        dt_fine = t_fine[1] - t_fine[0]
        Cp_fine = interp1d(t, Cp, kind="linear",
                           bounds_error=False,
                           fill_value=(Cp[0], Cp[-1]))(t_fine)
    else:
        t_fine, dt_fine, Cp_fine = t, float(t[1] - t[0]), Cp

    ke    = Ktrans / (ve + _EPS)
    h_t   = np.exp(-ke * t_fine)

    Cv_fine = vp * Cp_fine
    Ce_fine = Ktrans * fftconvolve(Cp_fine, h_t)[:len(t_fine)] * dt_fine

    if upsample > 1:
        return Cv_fine[::upsample], Ce_fine[::upsample]
    return Cv_fine, Ce_fine

def gkm_ct(
    theta_kin: np.ndarray,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    upsample:  int = 50,
) -> np.ndarray:
    """
    Extended Tofts (GKM) tissue concentration.

        C(t) = vp·Cp(t) + Ktrans · ∫ Cp(u) · exp[−(Ktrans/ve)(t−u)] du

    theta_kin = [Ktrans (s⁻¹), ve, vp]
    """
    Cv, Ce = gkm_ct_compartments(theta_kin, time_var, aif, upsample)
    return Cv + Ce


GKM = KineticModelSpec(
    name        = "GKM",
    fn          = gkm_ct,
    param_names = ("Ktrans", "ve", "vp"),
    lb          = np.array([1e-5,        1e-4,  1e-4]),
    ub          = np.array([2.0 / 60.0,  0.5,  0.5]),
    p0          = np.array([0.22 / 60.0, 0.37,  0.01]),
    constraints = _gkm_constraints, #lambda p: p[1] + p[2] <= 1.0,
    log_dims    = (0,),      # Ktrans spans orders of magnitude
    fn_compartments = gkm_ct_compartments, 
)
# =============================================================================
# 3. Dispersed-AIF GKM
# =============================================================================
def dispersed_gkm_ct(
    theta_kin: np.ndarray,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    n_level:   int = 3,
    upsample:  int = 50,
) -> np.ndarray:
    """
    GKM with vascular-dispersed local AIF (Nejad-Davarani et al. 2017).

    NOTE: n_level is a fixed integer argument, NOT part of theta_kin.
    The fitting loop in fitting.py calls this once per integer level
    and selects the best by lowest cost.

    theta_kin = [Ktrans (s⁻¹), ve, vp, t0_vascular (s)]  ← 4 params only

    Parameters
    ----------
    theta_kin : [Ktrans, ve, vp, t0]
    time_var  : (T,) uniform time array  (s)
    aif       : (T,) global AIF  (mM, plasma units)
    n_level   : integer branching level (1-6), fixed externally
    upsample  : internal upsampling factor for convolution accuracy
    """
    from dce_mri.vascular_aif import vascular_tree_h   # ← updated import

    Ktrans = float(theta_kin[0])
    ve     = float(theta_kin[1])
    vp     = float(theta_kin[2])
    t0     = float(theta_kin[3])
    # n_level is now a plain int argument — not extracted from theta_kin

    t  = np.asarray(time_var, dtype=np.float64)
    Cp = np.asarray(aif,      dtype=np.float64)

    if t.shape[0] < 2:
        return np.zeros_like(t)

    dt = float(t[1] - t[0])

    # ── Degenerate / edge cases ───────────────────────────────────────
    if Ktrans <= _EPS or ve <= _EPS:
        if vp <= _EPS:
            return np.zeros_like(t)
        # no GKM convolution needed — just vascular component
        if upsample > 1:
            t_fine  = np.linspace(t[0], t[-1], (len(t) - 1) * upsample + 1)
            dt_fine = t_fine[1] - t_fine[0]
            Cp_fine = interp1d(t, Cp, kind="linear",
                               bounds_error=False,
                               fill_value=(Cp[0], Cp[-1]))(t_fine)
            h_v    = vascular_tree_h(t_fine, t0, n_level)
            h_v    = h_v / (np.sum(h_v) * dt_fine + _EPS)
            Cp_loc = fftconvolve(Cp_fine, h_v)[:len(t_fine)] * dt_fine
            return (vp * Cp_loc)[::upsample]
        else:
            h_v    = vascular_tree_h(t, t0, n_level)
            h_v    = h_v / (np.sum(h_v) * dt + _EPS)
            Cp_loc = fftconvolve(Cp, h_v)[:len(t)] * dt
            return vp * Cp_loc

    # ── Upsample ─────────────────────────────────────────────────────
    if upsample > 1:
        t_fine  = np.linspace(t[0], t[-1], (len(t) - 1) * upsample + 1)
        dt_fine = t_fine[1] - t_fine[0]
        Cp_fine = interp1d(t, Cp, kind="linear",
                           bounds_error=False,
                           fill_value=(Cp[0], Cp[-1]))(t_fine)
    else:
        t_fine, dt_fine, Cp_fine = t, dt, Cp

    # ── Stage 1: vascular dispersion ─────────────────────────────────
    h_vasc = vascular_tree_h(t_fine, t0, n_level)
    h_vasc = h_vasc / (np.sum(h_vasc) * dt_fine + _EPS)
    Cp_local_fine = fftconvolve(Cp_fine, h_vasc)[:len(t_fine)] * dt_fine

    # ── Stage 2: GKM convolution ──────────────────────────────────────
    ke     = Ktrans / (ve + _EPS)
    h_t    = np.exp(-ke * t_fine)
    Ct_fine = (vp * Cp_local_fine
               + Ktrans * fftconvolve(Cp_local_fine, h_t)[:len(t_fine)] * dt_fine)

    return Ct_fine[::upsample] if upsample > 1 else Ct_fine


DISPERSED_GKM = KineticModelSpec(
    name        = "DispersedGKM",
    fn          = None,    # use fit_dispersed_gkm() from fitting.py — not callable directly
    param_names = ("Ktrans", "ve", "vp", "t0_vasc"),
    lb          = np.array([1e-5,        1e-4, 1e-4, 0.01]),
    ub          = np.array([2.0 / 60.0,  0.5, 0.5,  5.0]),
    p0          = np.array([0.25 / 60.0, 0.45, 0.06,  0.5]),
    constraints = _dispersed_gkm_constraints,#lambda p: p[1] + p[2] <= 1.0,
    log_dims    = (0,),      # Ktrans spans orders of magnitude
)


def _dispersed_gkm_fn_level_1(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=1)

def _dispersed_gkm_fn_level_2(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=2)

def _dispersed_gkm_fn_level_3(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=3)

def _dispersed_gkm_fn_level_4(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=4)

def _dispersed_gkm_fn_level_5(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=5)

def _dispersed_gkm_fn_level_6(theta_kin, t, aif):
    return dispersed_gkm_ct(theta_kin, t, aif, n_level=6)


_DISPERSED_GKM_LEVEL_FNS = {
    1: _dispersed_gkm_fn_level_1,
    2: _dispersed_gkm_fn_level_2,
    3: _dispersed_gkm_fn_level_3,
    4: _dispersed_gkm_fn_level_4,
    5: _dispersed_gkm_fn_level_5,
    6: _dispersed_gkm_fn_level_6,
}


def make_dispersed_gkm_spec(n_level: int) -> KineticModelSpec:
    """
    Build a fully callable KineticModelSpec for one fixed branch level.

    Uses a top-level (module-level) function for `fn`, not a closure,
    so the resulting spec — and therefore the whole KineticModelSpec —
    remains picklable for use with multiprocessing.Pool.
    """
    return KineticModelSpec(
        name        = f"DispersedGKM_L{n_level}",
        fn          = _DISPERSED_GKM_LEVEL_FNS[n_level],
        param_names = ("Ktrans", "ve", "vp", "t0_vasc"),
        lb          = DISPERSED_GKM.lb,
        ub          = DISPERSED_GKM.ub,
        p0          = DISPERSED_GKM.p0,
        constraints = DISPERSED_GKM.constraints,
        log_dims    = DISPERSED_GKM.log_dims,
    )


DISPERSED_GKM_LEVELS = {n: make_dispersed_gkm_spec(n) for n in range(1, 7)}