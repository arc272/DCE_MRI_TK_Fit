# -*- coding: utf-8 -*-
"""
signal_models.py
================
SPGR signal ↔ CA concentration conversion.
This is the canonical single copy of these functions for the entire package.
All other modules import from here — no duplicates elsewhere.
"""
from __future__ import annotations
import numpy as np
from dce_mri.config import AcquisitionConfig, AifConfig
from scipy.interpolate import interp1d
_EPS = 1e-11

# =============================================================================
# Public functions
# =============================================================================

def signal_to_concentration(
    sig: np.ndarray,
    acq: AcquisitionConfig,
    T10: float | np.ndarray,   # scalar OR (X, Y, Z) map
    B1:  float | np.ndarray,   # scalar OR (X, Y, Z) map 
) -> np.ndarray:
    squeeze = sig.ndim == 1
    if squeeze:
        sig = sig[None, :]

    S0    = sig[:, acq.nbase].mean(axis=1, keepdims=True)
    Eh    = sig / (S0 + _EPS)

    theta  = np.deg2rad(acq.FA * B1)
    cos_a  = np.cos(theta)
    TR_s   = acq.TR  / 1_000.0
    T10_s  = T10     / 1_000.0
    E10    = np.exp(-TR_s / (T10_s + _EPS))
    K      = (1 - E10 * cos_a) / (1 - E10 + _EPS)

    yprime = Eh / (K + _EPS)
    numer  = 1.0 - yprime
    denom  = np.where(np.abs(1.0 - yprime * cos_a) < _EPS,
                      _EPS, 1.0 - yprime * cos_a)

    E1   = numer / denom
    T1_t = -TR_s / (np.log(E1) + _EPS)
    dR1  = 1.0 / (T1_t + _EPS) - 1.0 / T10_s
    C_t  = dR1 / (acq.r1 + _EPS)

    return C_t[0] if squeeze else C_t


def concentration_to_signal(
    C_t: np.ndarray,
    acq: AcquisitionConfig,
    T10: float | np.ndarray,   # scalar OR (X, Y, Z) map
    B1:  float | np.ndarray,   # scalar OR (X, Y, Z) map
    S0,
) -> np.ndarray:
    C = np.asarray(C_t, dtype=np.float64)
    squeeze = C.ndim == 1
    if squeeze:
        C = C[None, :]

    S0 = np.asarray(S0, dtype=np.float64)
    if S0.ndim == 0:
        S0 = np.full((C.shape[0], 1), float(S0))
    elif S0.ndim == 1:
        S0 = S0[:, None]
    elif S0.ndim == 2 and S0.shape[1] != 1:
        raise ValueError("S0 must be scalar, (N,), or (N,1).")

    TR_s   = acq.TR / 1_000.0
    T10_s  = np.asarray(T10, dtype=np.float64) / 1_000.0
    R10    = 1.0 / (T10_s + _EPS)
    theta  = np.deg2rad(acq.FA * np.asarray(B1, dtype=np.float64))
    cos_a  = np.cos(theta)

    E10  = np.exp(-TR_s * R10)
    R1_t = R10[..., None] + acq.r1 * C
    E1   = np.exp(-TR_s * R1_t)

    one_minus_E1_cos  = np.maximum(1.0 - E1 * cos_a[...,None],     _EPS)
    one_minus_E10_cos = 1.0 - E10 * cos_a[..., None]
    one_minus_E10     = np.maximum(1.0 - E10, _EPS)[..., None]
    one_minus_E1      = 1.0 - E1

    Eh  = (one_minus_E1 * one_minus_E10_cos) / \
          (one_minus_E10 * one_minus_E1_cos)
    sig = S0 * Eh

    return sig[0] if squeeze else sig


def compute_S0(
    sig: np.ndarray,
    acq: AcquisitionConfig,
) -> np.ndarray:
    return sig[:, acq.nbase].mean(axis=1)

# --- Aif Extraction from ROI -----------------------------------------------

def extract_aif(
    sig_4d:   np.ndarray,
    aif_mask: np.ndarray,
    acq:      AcquisitionConfig,
    aif_cfg:  "AifConfig",
    B1:       float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a plasma-concentration AIF curve from a vascular ROI.

    Pipeline:
        1. pull all voxel signals inside aif_mask
        2. optionally keep only the top-enhancing voxels (aif_cfg.top_frac)
        3. average remaining voxel signals → single AIF signal curve
        4. convert to whole-blood concentration via SPGR inversion,
           using aif_cfg.T10 (blood T1) and aif_cfg.nbase (AIF-specific
           baseline window) — NOT the tissue subject.T10/acq.nbase
        5. apply hematocrit correction → plasma concentration

    Parameters
    ----------
    sig_4d   : (X, Y, Z, T) MRI signal volume
    aif_mask : (X, Y, Z) bool — vascular ROI (e.g. aorta)
    acq      : AcquisitionConfig  (TR, FA, r1 — shared sequence params)
    aif_cfg  : AifConfig  (T10, hematocrit, top_frac, nbase — AIF-specific)
    B1       : RF transmit field scaling, scalar or per-voxel

    Returns
    -------
    aif_plasma : (T,) plasma CA concentration curve  (mM)
    aif_signal : (T,) mean AIF signal curve  (a.u.) — for QC plotting
    """
    sig_roi = sig_4d[aif_mask, :]       # (N_vox, T)

    if aif_cfg.top_frac < 1.0:
        baseline = sig_roi[:, aif_cfg.nbase].mean(axis=1)
        peak     = sig_roi.max(axis=1)
        ratio    = peak / (baseline + _EPS)
        k        = max(1, int(np.ceil(aif_cfg.top_frac * len(ratio))))
        top_idx  = np.argpartition(ratio, -k)[-k:]
        sig_roi  = sig_roi[top_idx, :]

    aif_signal = sig_roi.mean(axis=0)    # (T,)

    # build a temporary AcquisitionConfig using the AIF-specific baseline
    # window, while keeping TR/FA/r1/upsample shared with the tissue acq
    from dataclasses import replace
    acq_aif = replace(acq, nbase=aif_cfg.nbase)

    aif_blood  = signal_to_concentration(aif_signal, acq_aif,
                                         T10=aif_cfg.T10, B1=B1)
    aif_plasma = aif_blood / (1.0 - aif_cfg.hematocrit)

    return aif_plasma, aif_signal

# ── signal_models.py addition ─────────────────────────────────────────────────

def shift_aif(
    aif:      np.ndarray,
    tau:      float,
    time_var: np.ndarray,
    acq:      AcquisitionConfig,
    method:   str = "linear",   # "linear" | "pchip"
) -> np.ndarray:
    """
    Shift AIF by tau seconds using interpolation.

    method : 'linear' (default, original behavior) or 'pchip'
             PCHIP is monotonicity-preserving — recommended for sparse
             time grids where linear interpolation between widely-spaced
             points can produce visually distorted shapes after a shift.
    Negative tau — AIF arrives earlier (shifts left):
        beginning truncated naturally
        end filled by repeating last AIF value

    The fill value for the left boundary uses acq.nbase mean of the AIF
    — the true pre-contrast baseline — not aif[0], so that enhancement
    is never propagated into the baseline region regardless of shift size.

    Parameters
    ----------
    aif      : (T,) global AIF  (mM, plasma units)
    tau      : delay in seconds.  Positive = AIF arrives later.
    time_var : (T,) uniform time grid  (s)
    acq      : AcquisitionConfig  (uses acq.nbase)

    Returns
    -------
    aif_shifted : (T,) shifted AIF on the same time grid
    """
    if tau == 0.0:
        return aif.copy()

    baseline_val = float(np.asarray(aif)[acq.nbase].mean())
    tail_val     = float(aif[-1])

    t_query = time_var - tau

    if method == "linear":
        aif_shifted = interp1d(
            time_var, aif,
            kind         = "linear",
            bounds_error = False,
            fill_value   = (baseline_val, tail_val),
        )(t_query)

    elif method == "pchip":
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(time_var, aif, extrapolate=False)
        aif_shifted = pchip(t_query)
        # PCHIP returns NaN outside the original range — fill manually
        aif_shifted = np.where(t_query < time_var[0],  baseline_val, aif_shifted)
        aif_shifted = np.where(t_query > time_var[-1], tail_val,     aif_shifted)

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'linear' or 'pchip'.")

    return aif_shifted

# =============================================================================
# Internal helpers
# =============================================================================

def _spgr_E1(TR_s, R1, cos_a):
    return np.exp(-R1 * TR_s)


def _spgr_signal(E1, cos_a, sin_a, S0):
    return S0 * (1.0 - E1) * sin_a / (1.0 - E1 * cos_a + _EPS)

