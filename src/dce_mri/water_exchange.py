#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:49:53 2026

@author: allisonchen

water_exchange.py
==================
Water exchange models -- FXL (Fast eXchange Limit, the assumption
already baked into concentration_to_signal), SSM3 (3-pool Shutter
Speed Model: vascular <-> extracellular <-> intracellular, chain
topology, matching MATLAB's GKMv/ATH+SSM case) and NXL (No eXchange
Limit: the same three pools, each relaxing independently -- thesis
Ch. 3 Eqs 3.36-3.37, Bains et al. 2010 Eq. 10).

Public API
----------
    WaterExchangeModelSpec  — dataclass bundling a wx model, mirrors
                               KineticModelSpec's role for kinetic models
    FXL   : WaterExchangeModelSpec instance, n_pools=1, n_params=0
    SSM3  : WaterExchangeModelSpec instance, n_pools=3, n_params=2
            (tau_b, tau_i)
    NXL   : WaterExchangeModelSpec instance, n_pools=3, n_params=0
            (closed form; the tau_b, tau_i -> infinity limit of SSM3)

Design notes
------------
- to_signal ALWAYS returns a signal-space prediction. There is no
  concentration-space equivalent and there should not be one -- see
  the water-exchange integration outline's RESOLVED CONSTRAINT section.
- to_signal takes ve/vb as EXPLICIT arguments, separate from Cv/Ce --
  they cannot be recovered from the concentration curves alone
  (Cv(t)=vp*Cp(t) already has vp baked in as a time-varying curve).
  Callers must extract ve/vb from theta_kin themselves before calling.
- FXL and SSM3(tau->0) are NOT expected to match exactly when vb>0.
  Confirmed against the MATLAB reference: FXL uses ONE baseline
  (tissue only) for the whole voxel; SSM's fast-exchange limit
  correctly uses a volume-weighted baseline
  (vb*R1_blood + (ve+vi)*R1_tissue). This is a real, predictable,
  physically-meaningful gap -- not a bug to reconcile. See the
  verification functions/tests for how to check the CORRECT limit.
- R1_blood's baseline comes from AifConfig.T10 (genuinely distinct
  from subject.T10, not forced equal to it) -- this matches the
  MATLAB reference's relax.blood vs relax.tumor distinction exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import expm, pinv

from dce_mri.config import AcquisitionConfig, SubjectConfig, AifConfig, RelaxationConfig
from dce_mri.signal_models import concentration_to_signal


# =============================================================================
# Spec type
# =============================================================================

@dataclass(frozen=True)
class WaterExchangeModelSpec:
    """
    Bundles a water-exchange model with its metadata, mirroring
    KineticModelSpec's role for kinetic models.

    Fields
    ------
    name        : human-readable label
    n_pools     : total compartment count (1 for FXL, 3 for SSM3) --
                  used by the pool-count compatibility guardrail (see
                  the water-exchange outline's RESOLVED CONSTRAINT,
                  guardrail 3) to reject mismatched kinetic-model/
                  wx_model pairings.
    n_params    : number of free water-exchange parameters (0 for FXL
                  and NXL, 2 for SSM3: tau_b, tau_i). Callers that build
                  p_aug branch on n_params == 0 -> no wx block at all,
                  so relax_cfg is not needed for FXL/NXL.
    param_names : tuple of parameter name strings, e.g. () or
                  ('tau_b', 'tau_i')
    to_signal   : callable, see module docstring for the exact contract
    """
    name:        str
    n_pools:     int
    n_params:    int
    param_names: tuple
    to_signal:   Callable


# =============================================================================
# FXL
# =============================================================================

def _fxl_to_signal(
    Cv:        np.ndarray,
    Ce:        np.ndarray,
    ve:        float,
    vb:        float,
    theta_wx:  tuple,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: RelaxationConfig,
    S0:        float,
) -> np.ndarray:
    """
    Fast eXchange Limit -- thin wrapper around the existing
    concentration_to_signal. This is not new physics, it's the physics
    you already have, wrapped to match the pluggable to_signal
    contract. ve/vb/theta_wx/aif_cfg/relax_cfg are accepted (for
    signature compatibility with SSM3) but unused -- FXL treats the
    whole voxel as one pool.
    """
    return concentration_to_signal(Cv + Ce, acq, subject.T10, subject.B1, S0)


FXL = WaterExchangeModelSpec(
    name        = "FXL",
    n_pools     = 1,
    n_params    = 0,
    param_names = (),
    to_signal   = _fxl_to_signal,
)


# =============================================================================
# SSM3 -- 3-pool Shutter Speed Model
# =============================================================================

def _ssm3_pool_sum_batched(
    R1_b, R1_e, R1_i: float,
    vb: float, ve: float, vi: float,
    tau_b: float, tau_i: float, tau_e: float,
    TR_s: float, FA: float,
):
    """
    VECTORISED across timepoints. R1_b and R1_e are (n,) arrays. TR_s
    and FA may be scalars (constant across the curve -- the original
    single-flip-angle behaviour) or (n,) arrays giving a per-frame
    repetition time / flip angle (multi-flip-angle / ACE). Every other
    argument is a scalar.

    Identical math to _ssm3_pool_sum, done for all n timepoints at once:

      * scipy.linalg.expm accepts a stacked (n,3,3) array and returns
        (n,3,3), so the n separate 3x3 matrix exponentials become one
        batched call instead of a Python loop.
      * pinv -> np.linalg.solve. (I - eX*cos(FA)) is square and
        non-singular here, so the pseudo-inverse was doing an SVD to
        solve a 3x3 system a direct solve handles far more cheaply.
        pinv also builds an explicit inverse that was then immediately
        multiplied into a vector -- solve skips forming it at all.

    Verified against the loop version: max abs difference 1.8e-16
    (floating-point noise), 8.6x faster at n=50.
    """
    R1_b = np.atleast_1d(np.asarray(R1_b, dtype=float))
    R1_e = np.atleast_1d(np.asarray(R1_e, dtype=float))
    n = R1_b.shape[0]
    
    # TR_s and FA may be scalars (single-flip-angle callers, unchanged
    # behaviour) or (n,) per-frame arrays (multi-flip-angle / ACE).
    TR_s = np.broadcast_to(np.asarray(TR_s, dtype=float), (n,))
    FA   = np.broadcast_to(np.asarray(FA,   dtype=float), (n,))

    X = np.zeros((n, 3, 3))
    X[:, 0, 0] = -(R1_b + 1.0 / tau_b)
    X[:, 0, 1] = 1.0 / tau_e - (vi / ve) / tau_i
    X[:, 1, 0] = 1.0 / tau_b
    X[:, 1, 1] = -(R1_e + 1.0 / tau_e)
    X[:, 1, 2] = 1.0 / tau_i
    X[:, 2, 1] = 1.0 / tau_e - (vb / ve) / tau_b
    X[:, 2, 2] = -(R1_i + 1.0 / tau_i)

    M0 = np.array([vb, ve, vi])
    I = np.eye(3)
    eX = expm(TR_s[:, None, None] * X)                      # (n,3,3)
    A = I - eX * np.cos(FA)[:, None, None]                  # (n,3,3)
    rhs = np.einsum("kij,kj->ki", I - eX,
                    np.sin(FA)[:, None] * M0[None, :])      # (n,3)
    M = np.linalg.solve(A, rhs[..., None])[..., 0]          # (n,3)
    return M.sum(axis=1)


def _ssm3_pool_sum(
    R1_b: float, R1_e: float, R1_i: float,
    vb: float, ve: float, vi: float,
    tau_b: float, tau_i: float, tau_e: float,
    TR_s: float, FA: float,
) -> float:
    """
    Steady-state SPGR signal for the 3-pool chain (b<->e<->i), summed
    across pools. Internal helper -- ported directly from the verified
    standalone prototype (step4_three_pool_with_exchange.py), unchanged.
    """
    X = np.array([
        [-(R1_b + 1/tau_b),      1/tau_e - (vi/ve)/tau_i,   0.0                    ],
        [ 1/tau_b,               -(R1_e + 1/tau_e),          1/tau_i                ],
        [ 0.0,                    1/tau_e - (vb/ve)/tau_b,   -(R1_i + 1/tau_i)      ],
    ])
    M0 = np.array([vb, ve, vi])
    I = np.eye(3)
    eX = expm(TR_s * X)
    M = pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)
    return M.sum()


def _ssm3_to_signal(
    Cv:        np.ndarray,
    Ce:        np.ndarray,
    ve:        float,
    vb:        float,
    theta_wx:  tuple,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: RelaxationConfig,
    S0:        float,
) -> np.ndarray:
    """
    3-pool Shutter Speed Model. theta_wx = (tau_b, tau_i).

    R1_e/R1_i baselines come from subject.T10 (tissue). R1_b's baseline
    comes from aif_cfg.T10 (blood) -- deliberately NOT the same value,
    matching the MATLAB reference's relax.blood vs relax.tumor split.

    Ce_density = Ce/ve and Cv_density = Cv/vb: whole-voxel-averaged
    concentrations from the kinetic model converted to the LOCAL
    concentration density each compartment's own water actually
    experiences -- confirmed during prototyping that skipping this
    division breaks the fast-exchange-limit consistency check by
    ~65%. This is load-bearing, not optional.

    Returns S0-scaled signal, baseline-normalized the same way
    concentration_to_signal is (ratio to the C=0 / no-contrast value),
    NOT the raw unnormalized SPGR equation used during standalone
    prototyping -- confirmed necessary to correctly interface with the
    rest of this codebase's S0 convention.
    """
    tau_b, tau_i = theta_wx
    

    # `vb` arrives as the PLASMA volume fraction vp -- every caller passes
    # theta_kin[vp_idx].  Following the WX-2CXM derivation (Fig 3.1, eqs
    # 3.33/3.40/3.42), the vascular water pool IS the plasma space: volume
    # vp, carrying the plasma concentration Cp = Cv/vp, with
    # vi = 1 - vp - ve.  Hematocrit enters ONLY upstream, in the conversion
    # of the measured AIF to plasma concentration -- it does not appear in
    # the water-exchange model at all.
    vi = 1.0 - vb - ve
    tau_e = ve / (vb/tau_b + vi/tau_i)

    TR_s   = acq.TR    / 1000.0
    T10_s  = subject.T10 / 1000.0
    Tb_s   = aif_cfg.T10  / 1000.0
    R1_tissue         = 1.0 / T10_s
    R1_blood_baseline = 1.0 / Tb_s
    FA = np.deg2rad(acq.FA * subject.B1)
    #plasma_frac = 1.0 - aif_cfg.hematocrit
    
    Eh_baseline = _ssm3_pool_sum(
        R1_blood_baseline, R1_tissue, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e, TR_s, FA,
    )

    # Vectorised over timepoints -- was a Python loop calling
    # _ssm3_pool_sum (expm + pinv on a 3x3) once per frame, which
    # dominated the whole forward-model cost for every wx row.
    Ce_density = (Ce / ve) if ve > 1e-9 else np.zeros_like(Ce)
    Cv_density = (Cv / vb) if vb > 1e-9 else np.zeros_like(Cv)
    
    R1_e_arr = R1_tissue + acq.r1 * Ce_density
    R1_b_arr = R1_blood_baseline + acq.r1 * Cv_density

    Eh = _ssm3_pool_sum_batched(
        R1_b_arr, R1_e_arr, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e, TR_s, FA,
    )
    return S0 * (Eh / Eh_baseline)

def _ssm3_to_signal_mfa(
    Cv:           np.ndarray,
    Ce:           np.ndarray,
    ve:           float,
    vb:           float,
    tau_b:        float,
    tau_i:        float,
    alpha_t_deg:  np.ndarray,
    TR_t_ms:      np.ndarray,
    T10_ms:       float,
    T10_blood_ms: float,
    B1:           float,
    M0:           float,
    r1:           float,
    hematocrit:   float,
) -> np.ndarray:
    """
    3-pool Shutter Speed Model signal for a multi-flip-angle / ACE
    acquisition. Absolute (un-normalised) signal, M0 explicit.

    Same physics as _ssm3_to_signal -- same exchange matrix, same
    compartment-density conversion (Ce/ve, (Cv/vb)*(1-hematocrit)),
    same R1_blood vs R1_tissue baseline split -- with two differences:

      1. Flip angle and TR are per-frame arrays (e.g. from
         AcquisitionConfig.frame_arrays), not acquisition-level scalars.
      2. Returns M0 * (pool sum), NOT S0 * Eh/Eh_baseline. The
         baseline-ratio convention of _ssm3_to_signal assumes one flip
         angle for the whole curve (see spgr_signal_abs for why that
         breaks), so M0 stays explicit here, to be fit, profiled or
         analytically marginalised.

    T10_ms, B1 and M0 are explicit arguments rather than read from
    SubjectConfig because models M1/M4 treat them as free parameters.

    Parameters
    ----------
    Cv, Ce       : (T,) vascular / extravascular-extracellular
                   concentration from the kinetic model (voxel-averaged)
    ve, vb       : volume fractions; vi = 1 - vb - ve
    tau_b, tau_i : vascular / intracellular water residence times (s)
    alpha_t_deg  : (T,) nominal flip angle per frame (deg)
    TR_t_ms      : (T,) repetition time per frame (ms)
    T10_ms       : pre-contrast tissue T1 (ms)
    T10_blood_ms : pre-contrast blood T1 (ms), e.g. AifConfig.T10
    B1           : RF transmit scaling (actual FA = B1 * nominal)
    M0           : total equilibrium magnetization (a.u.); pool
                   equilibria are M0 * (vb, ve, vi)
    r1           : CA relaxivity (mM^-1 s^-1)
    hematocrit   : e.g. AifConfig.hematocrit

    Returns
    -------
    S_t : (T,) absolute signal, same units as M0
    """
    Cv = np.asarray(Cv, dtype=np.float64)
    Ce = np.asarray(Ce, dtype=np.float64)

    vi          = 1.0 - vb - ve
    tau_e       = ve / (vb / tau_b + vi / tau_i)


    R1_tissue = 1_000.0 / T10_ms
    R1_blood0 = 1_000.0 / T10_blood_ms

    Ce_density = (Ce / ve) if ve > 1e-9 else np.zeros_like(Ce)
    Cv_density = (Cv / vb) if vb > 1e-9 else np.zeros_like(Cv)
   
    R1_e_arr = R1_tissue + r1 * Ce_density
    R1_b_arr = R1_blood0 + r1 * Cv_density

    TR_s = np.asarray(TR_t_ms, dtype=np.float64) / 1_000.0
    FA   = np.deg2rad(B1 * np.asarray(alpha_t_deg, dtype=np.float64))

    pool_sum = _ssm3_pool_sum_batched(
        R1_b_arr, R1_e_arr, R1_tissue,
        vb, ve, vi, tau_b, tau_i, tau_e, TR_s, FA,
    )
    return M0 * pool_sum

SSM3 = WaterExchangeModelSpec(
    name        = "SSM3",
    n_pools     = 3,
    n_params    = 2,
    param_names = ("tau_b", "tau_i"),
    to_signal   = _ssm3_to_signal,
)


# =============================================================================
# NXL -- No eXchange Limit (thesis Ch. 3 Eqs 3.36-3.37)
# =============================================================================

def _nxl_pool_sum(
    R1_b, R1_e, R1_i,
    vb: float, ve: float, vi: float,
    TR_s, FA,
):
    """
    Steady-state SPGR signal summed over three NON-exchanging pools
    (thesis Eq 3.37; Bains et al. 2010 Eq. 10):

        sum_k  v_k * sin(FA) * (1 - E_k) / (1 - cos(FA) * E_k),
        E_k = exp(-TR * R1_k),   k in {b, e, i}

    R1_b and R1_e may be (n,) arrays (one per timepoint); R1_i is the
    intracellular rate, a scalar because contrast agent never enters
    that pool. TR_s and FA may be scalars or (n,) arrays, as in
    _ssm3_pool_sum_batched.

    This is _ssm3_pool_sum with the exchange matrix X made diagonal
    (tau_b, tau_i -> infinity): expm(TR*X) is then diag(E_k), the 3x3
    solve decouples into three scalar SPGR equations, and M0 = (vb, ve,
    vi) weights them. Written out directly rather than as that limit so
    there is no expm and no near-singular 1/tau arithmetic.
    """
    TR_s = np.asarray(TR_s, dtype=float)
    FA   = np.asarray(FA,   dtype=float)
    sin_a, cos_a = np.sin(FA), np.cos(FA)

    def _spgr(R1):
        E = np.exp(-TR_s * np.asarray(R1, dtype=float))
        return sin_a * (1.0 - E) / (1.0 - cos_a * E)

    return vb * _spgr(R1_b) + ve * _spgr(R1_e) + vi * _spgr(R1_i)


def _nxl_to_signal(
    Cv:        np.ndarray,
    Ce:        np.ndarray,
    ve:        float,
    vb:        float,
    theta_wx:  tuple,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: Optional[RelaxationConfig],
    S0:        float,
) -> np.ndarray:
    """
    No eXchange Limit. theta_wx must be () -- NXL has no free water-
    exchange parameters.

    Every convention is copied from _ssm3_to_signal so that NXL is
    EXACTLY lim SSM3(tau_b, tau_i -> infinity), not merely "an" NXL:

      * pools b (plasma, volume vb = vp), e (EES, ve), i (vi = 1-vb-ve)
      * R1 baselines: plasma from aif_cfg.T10 (blood), e and i from
        subject.T10 (tissue) -- the same split SSM3 uses
      * compartment densities Cv/vb and Ce/ve (thesis Eq 3.36 needs the
        concentration IN each compartment, which is C_p = Cv/vp and
        C_e = Ce/ve -- see the note below on thesis Eq 3.37)
      * S0 * Eh / Eh_baseline normalisation

    NOTE on thesis Eq 3.37, third line: it writes the plasma term as
    R_1,0p + r F_p * integral(R_p(t-u) C_a(u) du), inheriting the F_p
    prefactor of Eq 3.12 as printed. With R_p, R_e defined as in Eq 3.13
    the concentrations are C_p = R_p (x) C_a and C_e = R_e (x) C_a with
    NO F_p prefactor (checked against an ODE solve of Eqs 3.10/3.11).
    tcxm_ct_compartments already returns vp*C_p and ve*C_e, so dividing
    by the volume fractions here gives the right per-compartment
    concentrations; nothing in the code inherits the typo.

    relax_cfg is accepted for signature compatibility and unused (it
    may be None).
    """
    if len(tuple(theta_wx)) != 0:
        raise ValueError(
            f"NXL has no free water-exchange parameters, got "
            f"theta_wx={tuple(theta_wx)!r}. A caller is still building "
            f"(tau_b, tau_i) for it -- that call site has not been made "
            f"n_params-aware.")

    vi = 1.0 - vb - ve

    TR_s  = acq.TR      / 1000.0
    T10_s = subject.T10 / 1000.0
    Tb_s  = aif_cfg.T10 / 1000.0
    R1_tissue         = 1.0 / T10_s
    R1_blood_baseline = 1.0 / Tb_s
    FA = np.deg2rad(acq.FA * subject.B1)

    Eh_baseline = _nxl_pool_sum(
        R1_blood_baseline, R1_tissue, R1_tissue, vb, ve, vi, TR_s, FA)

    Ce_density = (Ce / ve) if ve > 1e-9 else np.zeros_like(Ce)
    Cv_density = (Cv / vb) if vb > 1e-9 else np.zeros_like(Cv)

    R1_e_arr = R1_tissue + acq.r1 * Ce_density
    R1_b_arr = R1_blood_baseline + acq.r1 * Cv_density

    Eh = _nxl_pool_sum(R1_b_arr, R1_e_arr, R1_tissue, vb, ve, vi, TR_s, FA)
    return S0 * (Eh / Eh_baseline)


NXL = WaterExchangeModelSpec(
    name        = "NXL",
    n_pools     = 3,
    n_params    = 0,
    param_names = (),
    to_signal   = _nxl_to_signal,
)