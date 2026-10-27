# -*- coding: utf-8 -*-
"""
test_synthetic_mfa.py
=====================
Verification tests for synthetic.simulate_curve_mfa (refactor Phase 2).

References are independent of the function under test:
  * simulate_curve            -- the legacy single-FA generator
  * concentration_to_signal_mfa / _ssm3_to_signal_mfa composed by hand
  * the qualitative result of Zhang et al. NMR Biomed 2019 Fig 1
"""

import numpy as np
import matplotlib.pyplot as plt

from dce_mri.ace_protocol import ACEProtocol
from dce_mri.config import (AcquisitionConfig, SubjectConfig, AifConfig,
                            FittingConfig, RelaxationConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.signal_models import spgr_signal_abs, concentration_to_signal_mfa
from dce_mri.synthetic import make_parker_aif, simulate_curve, simulate_curve_mfa
from dce_mri.water_exchange import SSM3, FXL




# Zhang 2019 Table 1 baseline kinetics, on a 540 s / 5.4 s grid.
T    = np.arange(0, 540, 5.4)
AIF  = make_parker_aif(T, t_arrival=25.0)
KIN  = np.array([0.45, 0.06, 0.8 / 60, 0.12 / 60])     # ve, vp, fp, ps
TAU_I = 0.2
T10, B1_REF, M0_REF = 1530.0, 1.0, 1000.0

CFG   = FittingConfig()
SUBJ  = SubjectConfig(subject_id="x", T10=T10, B1=B1_REF)
AIFC  = AifConfig()
RELAX = RelaxationConfig()          # fit_tau_b=False -> tau_b fixed


def _acq(protocol=None, FA=10.0, TR=3.64):
    return AcquisitionConfig(TR=TR, FA=FA, r1=3.8, protocol=protocol)


def _baseline_ratio(acq, T10_ms=T10, B1=B1_REF, M0=M0_REF):
    """S0-equivalent of M0 for this acquisition: the pre-contrast signal."""
    return spgr_signal_abs(np.array([1000.0 / T10_ms]), np.array([acq.FA]),
                           np.array([acq.TR]), B1, M0)[0]


# =============================================================================
# 1. Reduction to the legacy single-FA generator
# =============================================================================

def test_reduces_to_simulate_curve_fxl():
    """protocol=None, no water exchange: must equal simulate_curve once
    M0 is set so the pre-contrast signals agree."""
    for b1 in (0.8, 1.0, 1.2):
        acq  = _acq()
        subj = SubjectConfig(subject_id="x", T10=T10, B1=b1)
        S0   = _baseline_ratio(acq, B1=b1, M0=M0_REF)

        legacy = simulate_curve(KIN, AIF, T, TCXM, acq, subj, CFG, S0=S0)
        mfa    = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, subj, CFG,
                                    M0=M0_REF)
        rel = np.max(np.abs(mfa - legacy)) / np.max(np.abs(legacy))
        # 1e-9, not 1e-15: concentration_to_signal computes
        # R10 = 1/(T10_s + _EPS) with _EPS=1e-11, a 6.5e-12 relative
        # perturbation of R10 that spgr_signal_abs does not make.
        # Emulating that epsilon drops the difference to 2.3e-16, so the
        # residual below is the legacy guard, not a disagreement.
        assert rel < 1e-9, f"B1={b1}: relative error {rel:.3e}"


def test_reduces_to_simulate_curve_ssm3():
    """Same reduction with the three-pool model in place."""
    theta = np.r_[KIN, TAU_I]
    for b1 in (0.8, 1.0, 1.2):
        acq  = _acq()
        subj = SubjectConfig(subject_id="x", T10=T10, B1=b1)

        # legacy S0 convention: S0 * Eh/Eh_baseline, so the matching M0 is
        # the one whose own pre-contrast SSM3 signal equals S0.
        mfa_base = simulate_curve_mfa(theta, np.zeros_like(AIF), T, TCXM, acq,
                                      subj, CFG, wx_model=SSM3, aif_cfg=AIFC,
                                      relax_cfg=RELAX, M0=M0_REF)[0]
        legacy = simulate_curve(theta, AIF, T, TCXM, acq, subj, CFG,
                                wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX,
                                S0=mfa_base)
        mfa = simulate_curve_mfa(theta, AIF, T, TCXM, acq, subj, CFG,
                                 wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX,
                                 M0=M0_REF)
        rel = np.max(np.abs(mfa - legacy)) / np.max(np.abs(legacy))
        assert rel < 1e-10, f"B1={b1}: relative error {rel:.3e}"


def test_fxl_wx_model_is_the_same_as_none():
    """Passing the FXL spec must not change anything -- it is one pool."""
    acq = _acq(ACEProtocol.zhang2019_dfa(t_max=T[-1] + 1.0))
    a = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, SUBJ, CFG)
    b = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, SUBJ, CFG, wx_model=FXL)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


# =============================================================================
# 2. Wiring: hand-assembled composition
# =============================================================================

def test_matches_hand_assembled_forward_on_ace_schedule():
    """On the 7-segment ACE schedule, the FXL path must equal
    concentration_to_signal_mfa(model.fn(...)) composed by hand."""
    proto = ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0)
    acq   = _acq(proto)
    fa    = proto.frame_arrays(T)

    Ct  = TCXM.fn(KIN, T, AIF)
    ref = concentration_to_signal_mfa(Ct, fa.alpha_deg, fa.TR_ms,
                                      T10, B1_REF, M0_REF, acq.r1)
    got = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, SUBJ, CFG, M0=M0_REF)
    np.testing.assert_allclose(got, ref, rtol=1e-14)


# =============================================================================
# 3. Nuisance-parameter contracts
# =============================================================================

def test_M0_scales_linearly():
    acq = _acq(ACEProtocol.zhang2019_dfa(t_max=T[-1] + 1.0))
    theta = np.r_[KIN, TAU_I]
    kw = dict(wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX)
    s1 = simulate_curve_mfa(theta, AIF, T, TCXM, acq, SUBJ, CFG, M0=1.0, **kw)
    s2 = simulate_curve_mfa(theta, AIF, T, TCXM, acq, SUBJ, CFG, M0=2500.0, **kw)
    np.testing.assert_allclose(s2, 2500.0 * s1, rtol=1e-12)


def test_B1_override_acts_only_as_flip_angle_scale():
    """B1=b on schedule alpha == B1=1 on schedule b*alpha."""
    b = 0.85
    p1 = ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0)
    p2 = ACEProtocol(p1.t_edges, b * p1.alphas, p1.TRs, n_settle=p1.n_settle)
    theta = np.r_[KIN, TAU_I]
    kw = dict(wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX, M0=M0_REF)
    s_a = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(p1), SUBJ, CFG, B1=b, **kw)
    s_b = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(p2), SUBJ, CFG, B1=1.0, **kw)
    np.testing.assert_allclose(s_a, s_b, rtol=1e-12)


def test_T10_and_B1_overrides_beat_the_subject_config():
    """The overrides must fully replace subject.T10 / subject.B1 -- this is
    what lets M1/M4 recovery tests generate data off-config.

    Uses the 2017 schedule so TR varies frame to frame here too."""
    acq = _acq(ACEProtocol.zhang2017_ace(t_max=T[-1] + 1.0))
    subj_alt = SubjectConfig(subject_id="y", T10=1200.0, B1=0.75)
    a = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, SUBJ, CFG,
                           T10_ms=1200.0, B1=0.75, M0=M0_REF)
    b = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, subj_alt, CFG, M0=M0_REF)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # and they genuinely change the curve
    c = simulate_curve_mfa(KIN, AIF, T, TCXM, acq, SUBJ, CFG, M0=M0_REF)
    assert np.max(np.abs(a - c)) / np.max(c) > 0.05


def test_all_frames_returned_including_settling_ones():
    """Masking is the fitter's job; the generator returns every frame."""
    proto = ACEProtocol.zhang2019_dfa(t_max=T[-1] + 1.0)
    s = simulate_curve_mfa(KIN, AIF, T, TCXM, _acq(proto), SUBJ, CFG)
    assert s.shape == T.shape
    assert (~proto.frame_arrays(T).keep).sum() == 4     # mask exists, unused


# =============================================================================
# 4. Guardrails
# =============================================================================

def test_concentration_space_refused_with_a_protocol():
    cfg_c = FittingConfig(fit_space="concentration")
    acq_p = _acq(ACEProtocol.zhang2019_dfa(t_max=T[-1] + 1.0))
    try:
        simulate_curve_mfa(KIN, AIF, T, TCXM, acq_p, SUBJ, cfg_c)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for concentration + protocol")
    # without a protocol it still behaves like simulate_curve
    out = simulate_curve_mfa(KIN, AIF, T, TCXM, _acq(), SUBJ, cfg_c)
    np.testing.assert_allclose(out, TCXM.fn(KIN, T, AIF), rtol=1e-14)


def test_unknown_engine_raises():
    try:
        simulate_curve_mfa(KIN, AIF, T, TCXM, _acq(), SUBJ, CFG,
                           engine="transient")
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for engine='transient'")


def test_ssm3_without_configs_raises():
    theta = np.r_[KIN, TAU_I]
    try:
        simulate_curve_mfa(theta, AIF, T, TCXM, _acq(), SUBJ, CFG,
                           wx_model=SSM3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when SSM3 has no aif/relax cfg")


# =============================================================================
# 5. Zhang 2019 Fig 1: DFA separates tau_i, SFA does not
# =============================================================================

def _frac_response(proto, idx, frames, scale=1.2, tau0=0.5):
    """Mean signed fractional change in the curve over `frames` when
    parameter `idx` is scaled by `scale`. Same frames, same underlying
    concentration curve -- only the flip angle played there differs
    between the protocols passed in."""
    base = np.r_[KIN, tau0]
    pert = base.copy()
    pert[idx] *= scale
    kw = dict(wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX, M0=M0_REF)
    c0 = simulate_curve_mfa(base, AIF, T, TCXM, _acq(proto), SUBJ, CFG, **kw)[frames]
    c1 = simulate_curve_mfa(pert, AIF, T, TCXM, _acq(proto), SUBJ, CFG, **kw)[frames]
    return float(np.mean((c1 - c0) / c0))


def test_flip_angle_reweights_tau_i_against_the_kinetics():
    """
    The premise of Zhang 2019: flip angle changes how strongly tau_i is
    encoded, and does so DIFFERENTLY from how it changes the kinetic
    parameters -- which is what a single set of kinetics cannot fake.

    Measured on identical frames (the middle third), so the underlying
    concentration curve is the same and only the played angle differs:
    8 deg (SFA protocol) versus 25 deg (DFA protocol).
    """
    t_max = T[-1] + 1.0
    dfa = ACEProtocol.zhang2019_dfa(t_max=t_max, a1=8.0, a2=25.0, TR_ms=3.64)
    sfa = ACEProtocol.single_fa(alpha=8.0, TR_ms=3.64, t_max=t_max)
    mid = np.flatnonzero(dfa.frame_arrays(T).seg_id == 1)[2:]   # skip settling

    ratios = {}
    for name, idx in [("ve", 0), ("vp", 1), ("fp", 2), ("ps", 3), ("tau_i", 4)]:
        r8  = _frac_response(sfa, idx, mid)
        r25 = _frac_response(dfa, idx, mid)
        ratios[name] = r25 / r8
        print(f"    {name:6s} response  8deg {100*r8:+7.3f}%   "
              f"25deg {100*r25:+7.3f}%   ratio {r25/r8:6.3f}")

    kin_ratios = [ratios[n] for n in ("ve", "vp", "fp", "ps")]
    assert min(kin_ratios) > 1.2, (
        f"kinetic parameters should all become MORE visible at 25 deg: {ratios}")
    assert ratios["tau_i"] < 0.5, (
        f"tau_i should become LESS visible at 25 deg: {ratios}")
    assert ratios["tau_i"] / min(kin_ratios) < 0.2, (
        "the two angles must reweight tau_i against the kinetics by a large "
        f"factor, else the second angle adds nothing: {ratios}")


def test_flip_angle_switch_is_visible_in_the_curve():
    """A DFA curve must jump at the switch; an SFA curve must not."""
    t_max = T[-1] + 1.0
    dfa = ACEProtocol.zhang2019_dfa(t_max=t_max, a1=8.0, a2=25.0, TR_ms=3.64)
    sfa = ACEProtocol.single_fa(alpha=8.0, TR_ms=3.64, t_max=t_max)
    theta = np.r_[KIN, TAU_I]
    kw = dict(wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX, M0=M0_REF)

    sw = np.flatnonzero(np.diff(dfa.frame_arrays(T).seg_id) != 0)[0]
    s_dfa = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(dfa), SUBJ, CFG, **kw)
    s_sfa = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(sfa), SUBJ, CFG, **kw)

    jump_dfa = abs(s_dfa[sw + 1] - s_dfa[sw]) / s_dfa[sw]
    jump_sfa = abs(s_sfa[sw + 1] - s_sfa[sw]) / s_sfa[sw]
    assert jump_dfa > 0.2, f"expected a large jump at the switch, got {jump_dfa:.3f}"
    assert jump_sfa < 0.02, f"SFA curve should be smooth there, got {jump_sfa:.3f}"


# =============================================================================
# 6. Visual verification (not asserted -- for eyeballing the curves)
# =============================================================================

def plot_tau_i_sensitivity(taus=(1e-3, 0.5, 1.0, 2.0, 4.0)):
    """
    Plots simulate_curve_mfa output directly, SFA (8 deg) vs DFA (8/25/8),
    sweeping tau_i with the kinetics held fixed at KIN. This is the same
    generator the tests above call -- no separate plotting path -- so
    what you see here is exactly what test_flip_angle_reweights_tau_i_
    against_the_kinetics and test_flip_angle_switch_is_visible_in_the_
    curve are checking numerically.

    Top panel (8 deg only): the tau_i curves should sit close together --
    this is the SFA problem the DFA protocol exists to fix.
    Bottom panel (8/25/8): the same curves should visibly separate in the
    8 deg segments and nearly coincide in the 25 deg segment -- that
    segment pins the kinetics down almost independent of tau_i, which is
    what lets the 8 deg segments' tau_i signature stand out.
    """
    t_max = T[-1] + 1.0
    sfa = ACEProtocol.single_fa(alpha=8.0, TR_ms=3.64, t_max=t_max)
    dfa = ACEProtocol.zhang2019_dfa(t_max=t_max, a1=8.0, a2=25.0, TR_ms=3.64)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(taus)))
    kw = dict(wx_model=SSM3, aif_cfg=AIFC, relax_cfg=RELAX, M0=M0_REF)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    for tau, c in zip(taus, colors):
        theta = np.r_[KIN, tau]
        s = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(sfa), SUBJ, CFG, **kw)
        ax.plot(T, s, color=c, lw=1.6, label=fr"$\tau_i$ = {tau} s")
    ax.set_title(r"Single flip angle (8$\degree$): curves nearly overlap across all $\tau_i$")
    ax.set_ylabel("Signal (a.u.)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    seg = dfa.frame_arrays(T).seg_id
    switches_t = T[np.flatnonzero(np.diff(seg) != 0) + 1]
    for sw in switches_t:
        ax.axvline(sw, color="gray", lw=0.8, ls="--", alpha=0.6)
    for tau, c in zip(taus, colors):
        theta = np.r_[KIN, tau]
        s = simulate_curve_mfa(theta, AIF, T, TCXM, _acq(dfa), SUBJ, CFG, **kw)
        ax.plot(T, s, color=c, lw=1.6, label=fr"$\tau_i$ = {tau} s")
    ax.set_title(r"Dual flip angle (8$\degree$/25$\degree$/8$\degree$): 25$\degree$ segment "
                r"collapses onto one curve $\to$ $\tau_i$ barely visible there")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (a.u.)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("simulate_curve_mfa: Zhang 2019 Fig. 1 reproduction\n"
                r"(TCXM + SSM3; $v_e$=0.45, $v_p$=0.06, $F_p$=0.8/min, PS=0.12/min)",
                fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")

    plot_tau_i_sensitivity()
    plt.show()