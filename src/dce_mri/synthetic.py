# -*- coding: utf-8 -*-
"""
synthetic.py
============
Ground-truth curve generation for validation.

Everything in the Bayesian validation ladder (SBC, synthetic recovery,
NLLS coverage) needs the same thing: simulate a curve from KNOWN
parameters through the SAME forward model the fitter uses, add
realistic noise, and hand it back. That generator lives here so the
validation scripts, the SBC loop and any figure code all share one
implementation.

Public API
----------
    make_synthetic_aif(time_var, ...)     -> (T,) plasma AIF, mM
    simulate_curve(theta, ...)            -> (T,) noiseless signal
    add_noise(S_clean, sigma, kind, rng)  -> (T,) noisy signal
    simulate_curve_mfa(theta, ...)        -> (T,) noiseless absolute signal
    default_test_setup(n_frames, ...)     -> dict of configs + AIF + time
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dce_mri.config import (
    AcquisitionConfig, SubjectConfig, AifConfig,
    FittingConfig, RelaxationConfig,
)
from dce_mri.kinetic_models import KineticModelSpec, TCXM
from dce_mri.water_exchange import WaterExchangeModelSpec


# =============================================================================
# AIF
# =============================================================================

def make_parker_aif(
    time_var:  np.ndarray,
    t_arrival: float = 25.0,
    hematocrit: float = 0.42,
    scale:     float = 1.0,
) -> np.ndarray:
    """
    Parker population AIF (Parker et al., MRM 2006;56:993-1000).

        C_b(t) = SUM_n  A_n/(sd_n*sqrt(2pi)) * exp(-(t-T_n)^2/(2 sd_n^2))
                 + alpha*exp(-beta*t) / (1 + exp(-s*(t - tau)))

    Two Gaussians model the first and second pass; the
    exponential-modulated sigmoid models the washout tail.

    Parker's published parameters are in MINUTES; this pipeline's
    time_var is in SECONDS, so every rate/time constant below is
    converted on the way in.

    Returns PLASMA concentration (mM): Parker's form gives whole-blood
    concentration, so it is divided by (1 - hematocrit), matching the
    convention the kinetic models expect for `aif`.

    t_arrival shifts the bolus so it lands after the baseline window
    (acq.nbase). With the 0-150 s grid used here, 25 s leaves ~7 clean
    baseline frames at dt=3 s while keeping the whole first pass and a
    usable washout tail inside the acquisition.
    """
    # --- Parker Table 1 parameters, converted min -> s ----------------
    A     = np.array([0.809, 0.330])      # mmol.min  (mM * min)
    T     = np.array([0.17046, 0.365]) * 60.0     # min -> s
    sd    = np.array([0.0563, 0.132]) * 60.0      # min -> s
    alpha = 1.050                                  # mM
    beta  = 0.1685 / 60.0                          # 1/min -> 1/s
    s_sig = 38.078 / 60.0                          # 1/min -> 1/s
    tau   = 0.483 * 60.0                           # min -> s
    A_s   = A * 60.0                               # mM*min -> mM*s

    t = np.asarray(time_var, dtype=float) - float(t_arrival)

    cb = np.zeros_like(t)
    for n in range(2):
        cb += A_s[n] / (sd[n] * np.sqrt(2.0 * np.pi)) * \
              np.exp(-((t - T[n]) ** 2) / (2.0 * sd[n] ** 2))
    # exp(-beta t) can overflow for very negative t; clip pre-arrival
    t_pos = np.maximum(t, 0.0)
    cb += alpha * np.exp(-beta * t_pos) / (1.0 + np.exp(-s_sig * (t - tau)))

    cb[t < 0.0] = 0.0                      # nothing before arrival
    cp = cb / (1.0 - float(hematocrit))    # blood -> plasma
    return scale * cp


def make_synthetic_aif(
    time_var:   np.ndarray,
    # Default arrival must sit AFTER acq.nbase (default slice(0,7), i.e.
    # 7 baseline frames at dt=5s -> 35 s) or compute_S0 averages enhanced
    # frames into the baseline and recovers the wrong S0.
    t_arrival:  float = 45.0,
    peak:       float = 4.0,
    a1:         float = 3.99,
    m1:         float = 0.144 / 60.0,
    a2:         float = 4.78,
    m2:         float = 0.0111 / 60.0,
) -> np.ndarray:
    """
    Biexponential plasma AIF with a smooth ramped bolus arrival.

    Functional form follows Weinmann/Orton (Orton et al. 2007 Eq. 1):
        c_p(t) = D * (a1*exp(-m1*t) + a2*exp(-m2*t))
    with rate constants converted from min^-1 to s^-1 because this
    pipeline's time_var is in seconds.
    """
    t_rel = np.maximum(time_var - t_arrival, 0.0)
    c = a1 * np.exp(-m1 * t_rel) + a2 * np.exp(-m2 * t_rel)
    c[time_var < t_arrival] = 0.0

    # smooth the discontinuity at arrival over ~2 frames so the
    # convolution does not see a hard step
    dt = float(np.median(np.diff(time_var)))
    ramp = np.clip((time_var - t_arrival) / (2.0 * dt), 0.0, 1.0)
    c = c * ramp

    if c.max() > 0:
        c = c * (peak / c.max())
    return c


# =============================================================================
# Forward simulation
# =============================================================================

def simulate_curve(
    theta:     np.ndarray,
    aif:       np.ndarray,
    time_var:  np.ndarray,
    model:     KineticModelSpec,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    cfg:       FittingConfig,
    wx_model:  Optional[WaterExchangeModelSpec] = None,
    aif_cfg:   Optional[AifConfig] = None,
    relax_cfg: Optional[RelaxationConfig] = None,
    S0:        float = 1000.0,
) -> np.ndarray:
    """
    Noiseless forward simulation from a known parameter vector.

    `theta` follows the SAME augmented layout the fitter uses:
        [*theta_kin, (tau_delay?), (tau_b?), tau_i?]
    Pass a delay-augmented `model` (build_delay_augmented_spec) to
    include tau_delay.

    Deliberately does NOT go through build_residual: that helper needs
    a measured curve to subtract from, and here there is no measured
    curve yet -- this function is what creates it.
    """
    from dce_mri.signal_models import concentration_to_signal

    theta = np.asarray(theta, dtype=float)
    n_kin = len(model.p0)
    theta_kin = theta[:n_kin]

    if wx_model is None:
        Ct = model.fn(theta_kin, time_var, aif)
        if cfg.fit_space == "concentration":
            return Ct
        return concentration_to_signal(Ct, acq, subject.T10, subject.B1, S0)

    if wx_model.n_params == 0:
        theta_wx = ()                        # NXL: theta is theta_kin only
    elif relax_cfg.fit_tau_b:
        theta_wx = (theta[n_kin], theta[n_kin + 1])
    else:
        theta_wx = (relax_cfg.tau_b_fixed, theta[n_kin])

    ve_idx = model.param_names.index("ve")
    vp_idx = model.param_names.index("vp")
    Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)
    return wx_model.to_signal(
        Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], theta_wx,
        acq, subject, aif_cfg, relax_cfg, S0)


def add_noise(
    S_clean: np.ndarray,
    sigma:   float,
    kind:    str = "gaussian",
    rng:     Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    kind : 'gaussian'      -- constant sigma (matches GAUSS_* likelihoods)
           'signal_scaled' -- sd proportional to sqrt(|S|), matching
                              GAUSS_SIGNAL_SCALED  [Sitek2016, Zhou2013]
           'rician'        -- magnitude MRI noise  [Gudbjartsson & Patz 1995]
           'student_t'     -- heavy-tailed, df=3   [Zhou2013]
    """
    rng = np.random.default_rng() if rng is None else rng
    S_clean = np.asarray(S_clean, dtype=float)

    if kind == "gaussian":
        return S_clean + rng.normal(0.0, sigma, S_clean.shape)

    if kind == "signal_scaled":
        scale = np.sqrt(np.abs(S_clean) / max(np.mean(np.abs(S_clean)), 1e-12))
        return S_clean + rng.normal(0.0, sigma, S_clean.shape) * scale

    if kind == "rician":
        re = S_clean + rng.normal(0.0, sigma, S_clean.shape)
        im = rng.normal(0.0, sigma, S_clean.shape)
        return np.sqrt(re ** 2 + im ** 2)

    if kind == "student_t":
        return S_clean + sigma * rng.standard_t(df=3, size=S_clean.shape)

    raise ValueError(f"unknown noise kind {kind!r}")
    
def simulate_curve_mfa(
    theta:     np.ndarray,
    aif:       np.ndarray,
    time_var:  np.ndarray,
    model:     KineticModelSpec,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    cfg:       FittingConfig,
    wx_model:  Optional[WaterExchangeModelSpec] = None,
    aif_cfg:   Optional[AifConfig] = None,
    relax_cfg: Optional[RelaxationConfig] = None,
    M0:        float = 1000.0,
    T10_ms:    Optional[float] = None,
    B1:        Optional[float] = None,
    engine:    str = "steady",
) -> np.ndarray:
    """
    Noiseless forward simulation on a multi-flip-angle / ACE schedule.

    The multi-flip-angle sibling of simulate_curve. Same `theta` layout,
    same configs, same wx dispatch -- four deliberate differences:

      1. ABSOLUTE signal, M0 explicit. simulate_curve returns
         S0 * S(t)/S(baseline), which cancels M0/T10/B1 only when one
         flip angle is played for the whole curve. Here the jump in
         signal at each flip-angle switch IS the information, so the
         baseline-ratio convention would destroy what the protocol
         encodes. M0 is a nuisance parameter to be profiled or
         marginalised downstream, not a normalisation.
      2. Per-frame flip angle and TR come from acq.frame_arrays(time_var).
         With acq.protocol=None that is a constant one-segment schedule,
         so single-FA is the degenerate case of this function, not a
         separate path.
      3. T10_ms and B1 are optional overrides of subject.T10 / subject.B1.
         Models M1 and M4 treat them as FREE parameters, so recovery
         tests need to generate data at a B1/T10 the SubjectConfig does
         not hold.
      4. `engine` selects the spin-dynamics model. Only "steady" exists
         today (steady-state SPGR, as in both Zhang papers); "transient"
         is Phase 6. The argument exists now so adding it later does not
         touch every call site.

    Frames are ALL returned, including the ones immediately after a
    flip-angle switch, where the steady-state assumption does not hold.
    Dropping those is the fitter's job -- see FrameArrays.keep and
    ACEProtocol.n_settle.

    Parameters
    ----------
    theta     : [*theta_kin, (tau_b?), tau_i?] -- exactly simulate_curve's
                layout. tau_b is present only when relax_cfg.fit_tau_b.
    M0        : total equilibrium magnetization (a.u.)
    T10_ms    : tissue T1 override (ms); defaults to subject.T10
    B1        : transmit scale override; defaults to subject.B1
    engine    : "steady" (only value implemented)

    Returns
    -------
    S_t : (T,) absolute signal, same units as M0 -- or, when
          cfg.fit_space == "concentration" and no protocol is set,
          the (T,) concentration curve, matching simulate_curve.
    """
    from dce_mri.signal_models import concentration_to_signal_mfa
    from dce_mri.water_exchange import _ssm3_to_signal_mfa

    if engine != "steady":
        raise NotImplementedError(
            f"engine={engine!r}: only 'steady' is implemented "
            "(per-excitation transients are Phase 6)"
        )

    theta = np.asarray(theta, dtype=float)
    n_kin = len(model.p0)
    theta_kin = theta[:n_kin]

    T10_ms = subject.T10 if T10_ms is None else float(T10_ms)
    B1     = subject.B1  if B1     is None else float(B1)

    if cfg.fit_space == "concentration":
        if acq.protocol is not None:
            raise ValueError(
                "fit_space='concentration' is incompatible with a "
                "multi-flip-angle protocol: the concentration curve is "
                "identical in every segment, so it carries none of the "
                "flip-angle encoding the protocol exists to provide."
            )
        return model.fn(theta_kin, time_var, aif)

    fa = acq.frame_arrays(time_var)

    # --- no water exchange (FXL) -----------------------------------------
    if wx_model is None or wx_model.n_pools == 1:
        Ct = model.fn(theta_kin, time_var, aif)
        return concentration_to_signal_mfa(
            Ct, fa.alpha_deg, fa.TR_ms, T10_ms, B1, M0, acq.r1)

    # --- three-pool shutter speed ----------------------------------------
    if wx_model.name != "SSM3":
        raise NotImplementedError(
            f"no multi-flip-angle forward model for wx_model "
            f"{wx_model.name!r} (have: FXL, SSM3)"
        )
    if relax_cfg is None or aif_cfg is None:
        raise ValueError(
            "SSM3 needs relax_cfg (tau_b handling) and aif_cfg "
            "(blood T10, hematocrit); pass them explicitly rather than "
            "letting defaults decide the blood baseline"
        )

    if relax_cfg.fit_tau_b:
        tau_b, tau_i = theta[n_kin], theta[n_kin + 1]
    else:
        tau_b, tau_i = relax_cfg.tau_b_fixed, theta[n_kin]

    ve_idx = model.param_names.index("ve")
    vp_idx = model.param_names.index("vp")
    Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)

    return _ssm3_to_signal_mfa(
        Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx], tau_b, tau_i,
        fa.alpha_deg, fa.TR_ms, T10_ms, aif_cfg.T10, B1, M0,
        acq.r1, aif_cfg.hematocrit,
    )

# =============================================================================
# Convenience setup
# =============================================================================

def default_test_setup(
    n_frames:  int   = 50,
    dt:        float = 3.0,
    S0:        float = 1000.0,
    aif_kind:  str   = "parker",
    t_arrival: float = 25.0,
):
    """
    A self-consistent set of configs + AIF + time grid.

    Defaults give a 0-150 s acquisition (50 frames at dt=3 s) with a
    Parker population AIF arriving at 25 s. That leaves ~8 pre-arrival
    frames, so acq.nbase (default slice(0,7)) sees a genuinely flat
    baseline and compute_S0 recovers S0 exactly -- verify this with
    `compute_S0(S_clean[None,:], acq)` if you change dt or t_arrival.

    aif_kind : 'parker' (population AIF, default) | 'biexp' (the older
               Weinmann/Orton biexponential form)
    """
    time_var = np.arange(n_frames, dtype=float) * dt
    acq      = AcquisitionConfig()
    subject  = SubjectConfig(subject_id="SYNTH")
    aif_cfg  = AifConfig()
    cfg      = FittingConfig(n_lhs=200, n_top=5, fit_space="signal")
    relax    = RelaxationConfig(fit_tau_b=False)

    if aif_kind == "parker":
        aif = make_parker_aif(time_var, t_arrival=t_arrival,
                              hematocrit=aif_cfg.hematocrit)
    elif aif_kind == "biexp":
        aif = make_synthetic_aif(time_var, t_arrival=t_arrival)
    else:
        raise ValueError(f"unknown aif_kind {aif_kind!r}")

    return dict(time_var=time_var, aif=aif, acq=acq, subject=subject,
                aif_cfg=aif_cfg, cfg=cfg, relax_cfg=relax, S0=S0)
