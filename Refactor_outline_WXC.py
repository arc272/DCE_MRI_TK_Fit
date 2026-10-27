#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:57:32 2026

@author: allisonchen
"""

# DCE-MRI Pipeline — Water Exchange Integration Outline
# =======================================================
# Companion to REFACTOR_OUTLINE.py. Source material: bmri_dce_model.m,
# bmri_dce_model_fit.m, bmri_dce_model_calc.m (S. Gene Kim, NYU).
#
# Goal, as stated: add water exchange (FXL vs Shutter-Speed Model) as a
# PLUGGABLE axis alongside kinetic model choice, prioritizing edits to
# existing functions/configs/result types over new standalone code,
# with a "plug in models" architecture matching how KineticModelSpec
# already lets TCXM/GKM/DISPERSED_GKM be swapped freely.
#
# STATUS markers throughout: every phase here is NOT STARTED. This is
# a planning document, not a status report on completed work.
#
# Last updated: added the signal-space-only constraint (RESOLVED
# CONSTRAINT section below), a dedicated fitting_roi.py integration
# phase (WX-5, new), and a shared validation-helper design so the
# signal-space rule lives in one place instead of being hand-copied
# across fit_single_voxel and every fitting_roi.py function. Phases
# after the old WX-4 are renumbered accordingly (WX-5 through WX-9).
# Joint global fitting (ADMM) is explicitly set aside for now -- see
# the DEFERRED section near the end.
#
# Second update: added a third guardrail to _validate_wx_compatibility
# -- pool-count compatibility between the kinetic model and wx_model,
# keyed structurally off fn_compartments' return shape rather than a
# hardcoded model-name list (confirmed via the 3-pool prototype,
# step4_three_pool_*.py, that TCXM/GKM/DISPERSED_GKM all imply 3 pools).
# Also narrowed Phase WX-3's scope: only the 3-pool SSM3 model ships
# near-term, not a parallel 2-pool model, since no current kinetic
# model implies a 2-pool system and step4's vb->0 check confirmed the
# 3-pool implementation already subsumes the 2-pool case exactly.
# -----------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
# OPEN DESIGN QUESTIONS — resolve these before writing code
# ══════════════════════════════════════════════════════════════════════
#
# These are genuine forks with real tradeoffs, not things I've decided
# for you. Flagging them up front rather than burying the choice inside
# a phase.
#
# Q1. How do kinetic models expose per-compartment output?
#     Option A (additive, zero risk): add an optional companion
#       function per model, e.g. tcxm_ct_compartments(theta, t, aif)
#       -> (Cv, Ce), living alongside the existing tcxm_ct_analytic
#       (which keeps returning the sum, unchanged). Every existing
#       caller of .fn is untouched. Water-exchange-aware fitting calls
#       the new companion instead.
#     Option B (invasive): change what .fn itself returns everywhere
#       (e.g. always return (Cv, Ce), push summing to callers). Touches
#       every existing call site of model.fn across fitting.py,
#       fitting_roi.py, volume.py, vascular_aif.py, dispersed_gkm_ct.
#     Recommendation given your stated priority (edit what exists,
#     don't break it): Option A. See Phase WX-2.
#
# Q2. Where do compartment volume fractions for the exchange matrix
#     come from?
#     ve, vp are already fitted kinetic parameters (TCXM/GKM). vb (for
#     3-pool models) and vi = 1 - ve - vb fall out algebraically. No
#     new fitted parameters needed here -- only tau_i (and tau_b for
#     3-pool) are genuinely new free parameters. Confirm this matches
#     your intent before Phase WX-1.
#
# Q3. Bundle wx params into the existing params array, or give them
#     their own field?
#     Precedent exists for both, though this was clarified in a later
#     discussion: fit_with_delay_continuous BUNDLES tau into
#     theta_aug = [*theta_kin, tau] only as an INTERNAL mechanic for
#     the optimizer call -- the moment fitting finishes, DelayedFitResult
#     immediately unbundles it back into separate best_params
#     (kinetic-only) and best_tau fields before returning anything to
#     a caller. There is no example anywhere in this codebase of a
#     returned RESULT object that keeps tau smuggled inside params.
#     Recommendation: follow that same pattern exactly -- bundle only
#     transiently for the optimizer, unbundle into a separate wx_params
#     field before constructing any result object. See Phase WX-7.
#
# Q4. Which relaxation config gets extended -- SubjectConfig, or a new
#     dedicated config?
#     SubjectConfig currently carries T10 (one tissue T1) and B1.
#     FXL only ever needed one baseline R1. SSM needs baseline R1 for
#     EACH compartment (blood, tissue) plus hematocrit (already exists
#     as AifConfig.hematocrit -- reuse it, don't duplicate it).
#     Recommendation: new small config (RelaxationConfig, matching the
#     AifConfig/FitModeConfig pattern of focused, addable configs)
#     rather than growing SubjectConfig with SSM-only fields that are
#     meaningless under FXL. See Phase WX-1.
#
# Q5. Nelder-Mead or TRF for tau_i/tau_b?
#     The MATLAB code's own comment ("other methods not working, poor
#     fit") plus this session's TRF-vs-kink findings for a structurally
#     similar delay/dispersion parameter both point toward caution with
#     TRF here. Don't assume TRF will handle tau_i/tau_b well just
#     because it's bundled with otherwise-smooth kinetic params --
#     verify empirically (Phase WX-8) before trusting it, the same way
#     _fit_single_level's design was only settled after finding TRF's
#     local-optima trapping empirically for the analogous vascular
#     dispersion parameter. A separate, related finding from building
#     the exchange equation directly (see RESOLVED CONSTRAINT below):
#     the signal's sensitivity to tau_i is itself non-uniform -- nearly
#     flat at very fast or very slow exchange, steepest in an
#     intermediate range (roughly tau_i in [0.1, 3]s for the operating
#     points tested). A flat region in the cost landscape is exactly
#     where an optimizer is least reliable -- expect tau_i estimates
#     near either extreme to be less precise than ones in the sensitive
#     middle range, independent of which optimizer is ultimately used.


# ══════════════════════════════════════════════════════════════════════
# RESOLVED CONSTRAINT — water exchange is SIGNAL-SPACE ONLY
# ══════════════════════════════════════════════════════════════════════
#
# Confirmed while building the exchange equation from scratch (Steps
# 1-3 of the standalone prototyping work): the matrix-exponential SPGR
# machinery (expm(TR*X), the steady-state M formula) produces a SIGNAL
# prediction directly -- there is no equivalent "concentration-space"
# version of it, and there structurally cannot be one. Concentration is
# only a well-defined quantity UPSTREAM of the exchange step (it's the
# Ce(t)/Cv(t) INPUT the kinetic model hands to water_exchange.py, not
# something the exchange machinery itself outputs). There is no
# sensible way to define "the water-exchange model's predicted
# concentration" to diff against a measured Ct curve.
#
# Practical consequence: wx_model can ONLY be used when fitting in
# signal space. This is stricter than it might first appear, because
# it has to hold at TWO levels simultaneously, not one:
#
#   1. fitting_cfg.fit_space must be 'signal'
#      (already covered by Phase WX-4's design)
#
#   2. For ROI-level fitting specifically, roi_cfg.average_space must
#      ALSO be 'signal' -- NOT just fitting_cfg.fit_space. This is a
#      gap the current fit_roi validation does not close: today's
#      check only enforces "average_space='signal' implies
#      fit_space='signal'" (the direction that matters for avoiding
#      SPGR-conversion bias). It says nothing about someone setting
#      average_space='concentration' while ALSO passing a wx_model --
#      that combination would pass every existing check and then
#      silently do nothing, since the 'concentration' branch calls
#      _fit_concentration_curve, which has and can have no wx_model
#      wiring at all (same reasoning as the earlier finding that
#      _fit_concentration_curve can't be replaced by fit_single_voxel
#      for the 'concentration' aggregation branch -- Ct_mean there is
#      not reconstructable from any single signal curve, so there's
#      no signal for a wx_model to even operate on in that branch).
#      A silently-ignored wx_model is a worse failure mode than an
#      error -- it produces a confidently-wrong result with no warning.
#
#   3. wx_model's pool count must match what the kinetic model
#      structurally implies. Confirmed while building the 3-pool
#      prototype (step4_three_pool_*.py): TCXM, GKM, and
#      DISPERSED_GKM ALL fit vp -- meaning all three imply a 3-pool
#      system (vascular + extracellular + intracellular), never a
#      2-pool one. Pairing any of them with a 2-pool wx_model would
#      silently discard the fitted vascular volume fraction's
#      water-exchange implications entirely, with no error. This
#      generalizes cleanly rather than needing to hardcode those three
#      model names: fn_compartments' return signature already reveals
#      pool count structurally -- it returns (Cv, Ce), i.e. 2 curves,
#      for any model with a vascular term, implying 3 total pools once
#      the always-implicit intracellular pool is counted. A
#      hypothetical vascular-free model (MATLAB's plain 'GKM') would
#      return just (Ce,) -- 1 curve, implying 2 pools. Checking pool
#      count via this structural signal, rather than a hardcoded name
#      list, means a future 4th/5th kinetic model routes to the
#      correct wx_model automatically, no validation-logic changes
#      needed.
#
# Recommendation: a single shared validation helper, called from every
# entry point rather than reimplemented per-function (the fit_roi_with_
# delay_grid/_continuous print-statement copy-paste mismatch from
# earlier this session is exactly the kind of drift hand-copying this
# rule three times risks):
#
#     def _validate_wx_compatibility(wx_model, fit_space, model,
#                                     time_var, aif, average_space=None):
#         """
#         Call this before doing any water-exchange-aware fitting.
#         average_space is None for single-voxel calls (fit_single_voxel,
#         fit_with_delay_grid/continuous) where the concept doesn't
#         apply; pass roi_cfg.average_space for any fitting_roi.py call.
#         time_var/aif only need to be real arrays if model.fn_compartments
#         is not None -- used solely to probe its output shape, no fitting
#         happens here.
#         """
#         if wx_model is None:
#             return
#         if fit_space != "signal":
#             raise ValueError(
#                 f"wx_model requires fit_space='signal'; got "
#                 f"fit_space={fit_space!r}. Water exchange has no "
#                 f"concentration-space equivalent -- see the "
#                 f"RESOLVED CONSTRAINT note in the water-exchange outline."
#             )
#         if average_space is not None and average_space != "signal":
#             raise ValueError(
#                 f"wx_model requires roi_cfg.average_space='signal'; got "
#                 f"average_space={average_space!r}. The 'concentration' "
#                 f"aggregation branch has no signal curve for a wx_model "
#                 f"to fit against."
#             )
#         if model.fn_compartments is None:
#             raise ValueError(
#                 f"{model.name} does not support water exchange fitting "
#                 f"yet (no fn_compartments) -- see Phase WX-2."
#             )
#         n_conc_curves = len(model.fn_compartments(model.p0, time_var, aif))
#         n_pools_implied = n_conc_curves + 1   # + the always-implicit
#                                                # intracellular pool
#         if wx_model.n_pools != n_pools_implied:
#             raise ValueError(
#                 f"{model.name} implies a {n_pools_implied}-pool system "
#                 f"(fn_compartments returns {n_conc_curves} concentration "
#                 f"curve(s)), but {wx_model.name} is a "
#                 f"{wx_model.n_pools}-pool model. These are not compatible."
#             )
#
# Where this lives: probably fitting.py (alongside the other internal
# helpers), imported into fitting_roi.py the same way compute_aic/
# akaike_weights are already imported from vascular_aif.py into
# fitting.py -- precedent for a small cross-module utility already
# exists in this codebase, this isn't a new pattern.


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-1 — Relaxation config
# STATUS: Completed 8/6/2026
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/config.py
#
# New dataclass (see Q4 above):
#
# RelaxationConfig
#   Fields:
#     R1_blood   : baseline blood R1 (1/s) -- MATLAB's relax.blood
#     R1_tissue  : baseline tissue R1 (1/s) -- MATLAB's relax.tumor;
#                  note this is 1/T10, not T10 itself -- SubjectConfig
#                  already has T10, so this can likely be DERIVED
#                  (R1_tissue = 1/subject.T10) rather than duplicated
#                  as a separate user-supplied field. Decide whether to
#                  store it explicitly or compute it on the fly.
#     r1         : contrast relaxivity -- already exists as acq.r1,
#                  reuse directly, do not duplicate
#     hematocrit : already exists as AifConfig.hematocrit, reuse
#                  directly for the vascular-compartment correction,
#                  do not duplicate. Note this gets used for TWO
#                  physically distinct corrections in the pipeline
#                  (AIF whole-blood-to-plasma conversion, AND diluting
#                  vascular-pool plasma concentration by the fraction
#                  of vascular water that's actually plasma) -- worth a
#                  code comment at each use site so it isn't mistaken
#                  for redundant application later.
#     tau_i_lb/ub, tau_b_lb/ub : bounds for the new fitted exchange
#                  parameters, matching the tau_lb/tau_ub pattern
#                  already used for AIF delay fitting -- pass explicitly
#                  from the notebook, don't hardcode module constants
#                  inside fitting functions (same lesson as the earlier
#                  vascular_aif.py T0_LB/T0_UB discussion)
#
# Verification:
#   Confirm R1_tissue = 1/subject.T10 matches MATLAB's relax.tumor
#   convention on a known T10 value before trusting downstream results

from dce_mri.config import RelaxationConfig, AifConfig, SubjectConfig
relax_cfg = RelaxationConfig()
print(relax_cfg)  # should print cleanly, frozen dataclass repr

# ══════════════════════════════════════════════════════════════════════
# PHASE WX-2 — Compartment-resolved kinetic model output
# STATUS: Completed 8/6/2026
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/kinetic_models.py
#
# Per Q1 above (Option A -- additive, non-invasive):
#
# Add companion functions alongside the existing ones, NOT replacing
# them:
#
#   tcxm_ct_compartments(theta_kin, time_var, aif, upsample=50)
#     -> (Cv, Ce)
#     Copy tcxm_ct_analytic's body verbatim up through the point where
#     cp_fine/ce_fine are computed (confirmed to already exist
#     separately -- see kinetic_models.py line 163: currently
#     "Ct_fine = vp * cp_fine + ve * ce_fine"). Return
#     (vp*cp_fine[::upsample], ve*ce_fine[::upsample]) instead of their
#     sum. tcxm_ct_analytic itself stays completely unchanged -- ideally
#     it becomes a one-line wrapper that calls the new function and
#     sums, to avoid maintaining the convolution math twice:
#         def tcxm_ct_analytic(theta_kin, time_var, aif, upsample=50):
#             Cv, Ce = tcxm_ct_compartments(theta_kin, time_var, aif, upsample)
#             return Cv + Ce
#
#   gkm_ct_compartments(theta_kin, time_var, aif, upsample=50)
#     -> (Cv, Ce)
#     Same pattern (confirmed line 217: currently
#     "Ct_fine = vp*Cp_fine + Ktrans*conv(...)"). gkm_ct becomes a
#     wrapper the same way.
#
# KineticModelSpec gets one new OPTIONAL field:
#
#   fn_compartments: Optional[Callable] = None
#     Signature: (Cv, Ce) = fn_compartments(theta_kin, time_var, aif)
#     None for any model that doesn't support water-exchange fitting
#     yet (e.g. DISPERSED_GKM, until/unless extended). Every existing
#     KineticModelSpec construction (TCXM, GKM, DISPERSED_GKM) stays
#     valid without modification since this field has a default.
#
# This phase touches ZERO existing call sites. fit_single_voxel,
# fit_roi, volume.py, vascular_aif.py -- none of them need to change
# for this phase alone, since fn's behavior and signature are unchanged.
#
# Verification:
#   assert np.allclose(tcxm_ct_analytic(p, t, aif),
#                       sum(tcxm_ct_compartments(p, t, aif)))
#   -- confirms the refactor didn't change the existing, already-tested
#   behavior, before building anything new on top of it.

# edit verification check: 
import numpy as np
from dce_mri.kinetic_models import (
    TCXM, GKM, tcxm_ct_analytic, tcxm_ct_compartments,
    gkm_ct, gkm_ct_compartments,
)

t = np.linspace(0, 300, 60)
aif = np.clip(((t-20-15)/15)**4 * np.exp(-4*((t-20)/15-1)), 0, None)  # any reasonable AIF shape

# check 1: main branch
Ct = tcxm_ct_analytic(TCXM.p0, t, aif)
Cv, Ce = tcxm_ct_compartments(TCXM.p0, t, aif)
assert np.allclose(Ct, Cv + Ce)

# check 2: the fp<=EPS branch specifically
Ct2 = tcxm_ct_analytic([0.2, 0.1, 0.0, 0.005], t, aif)
Cv2, Ce2 = tcxm_ct_compartments([0.2, 0.1, 0.0, 0.005], t, aif)
assert np.allclose(Ct2, Cv2 + Ce2) and np.allclose(Cv2, 0) and np.allclose(Ce2, 0)

# check 3: the ps<=EPS branch specifically
Ct3 = tcxm_ct_analytic([0.2, 0.1, 0.005, 0.0], t, aif)
Cv3, Ce3 = tcxm_ct_compartments([0.2, 0.1, 0.005, 0.0], t, aif)
assert np.allclose(Ct3, Cv3 + Ce3) and np.allclose(Ce3, 0)

# check 4: GKM
Ct4 = gkm_ct(GKM.p0, t, aif)
Cv4, Ce4 = gkm_ct_compartments(GKM.p0, t, aif)
assert np.allclose(Ct4, Cv4 + Ce4)

print("all checks passed")


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-3 — Water exchange model module (new file, unavoidably new)
# STATUS: Completed 8/7/2026
# (the core equations here have already been prototyped and verified
#  standalone, step by step, outside the package -- see step1/2/3_*.py
#  for the 2-pool build-up, step4_three_pool_*.py for the real 3-pool
#  model. This phase is "move that verified logic into the real
#  module," not "derive it from scratch.")
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/water_exchange.py  (new, parallel to vascular_aif.py
# in the sense of being a new focused module -- this is the one piece
# that genuinely can't be an edit to something existing, since dce_mri
# has no water-exchange abstraction to extend at all yet)
#
# SCOPE DECISION: only the 3-pool model ships in the real catalog for
# now, not both 2-pool and 3-pool. TCXM, GKM, and DISPERSED_GKM all fit
# vp -- none of them imply a 2-pool system, so there is no current
# caller for a 2-pool WaterExchangeModelSpec (see RESOLVED CONSTRAINT
# guardrail 3, above). The 2-pool prototype (Steps 1-3) did its real
# job already: building and verifying the matrix machinery
# incrementally (expm, pinv, the steady-state derivation, the
# Ce/ve density correction) before the harder 3-pool chain topology.
# It does not need a parallel life as a second shipped model. Confirmed
# during prototyping (step4's Check 3) that this loses nothing even in
# principle: setting vb->0 in the 3-pool implementation reproduces the
# 2-pool result exactly (to 1e-5) -- so a voxel whose fitted vp happens
# to land near zero is already handled correctly by the 3-pool code
# alone, no separate 2-pool code path required. Revisit only if a
# vascular-free kinetic model (MATLAB's plain 'GKM', or NPL) is ever
# added to the catalog -- guardrail 3's structural check (keyed off
# fn_compartments' output count, not a hardcoded model list) will
# route such a model to a 2-pool wx_model automatically once one exists,
# with no changes needed to the validation logic itself.
#
# WaterExchangeModelSpec (dataclass, mirrors KineticModelSpec's role)
#   Fields:
#     name        : 'FXL' | 'SSM3' (only these two ship now, per the
#                   scope decision above -- 'SSM3' = 3-pool Shutter
#                   Speed Model. Avoid colliding with MATLAB's SSM/SSM2
#                   naming, which conflates pool-count with something
#                   else worth double-checking against the source paper
#                   this MATLAB code was implementing, not just the code)
#     n_pools     : 1 (FXL) | 3 (SSM3) -- used directly by guardrail 3's
#                   structural compatibility check
#     n_params    : 0 (FXL) | 2 (SSM3: tau_b, tau_i)
#     param_names : e.g. () | ('tau_b', 'tau_i')
#     to_signal   : callable, see below
#

#
# to_signal(Cv, Ce, ve, vb, theta_wx, acq, subject, relax_cfg, S0) -> S_t
#   ALWAYS returns a signal-space prediction -- see RESOLVED CONSTRAINT
#   above. There is no to_concentration counterpart and there should
#   not be one.
#
#   NOTE: ve/vb are REQUIRED as explicit arguments, separate from
#   Cv/Ce -- confirmed during prototyping that they cannot be derived
#   from the concentration curves alone (Cv(t)=vp*Cp(t) already has vp
#   baked in as a time-varying curve; there's no clean way to recover
#   the scalar vp back out of it). Callers (fit_single_voxel, once
#   Phase WX-4 wires this in) already know model.param_names and must
#   extract ve/vb from theta_kin themselves before calling to_signal --
#   this keeps water_exchange.py correctly kinetic-model-agnostic, per
#   the WX-9 extensibility goal, rather than needing to know any given
#   model's theta_kin layout.
#
#   FXL implementation:
#     R1 = 1/subject.T10 + acq.r1 * (Cv + Ce)
#     return concentration_to_signal-equivalent SPGR conversion --
#     this should literally CALL your existing concentration_to_signal
#     on (Cv+Ce) rather than reimplementing the SPGR equation a second
#     time. FXL is not new physics, it's the physics you already have,
#     wrapped to match the new pluggable interface. (Verified directly
#     during prototyping: the matrix formula collapsed to a single pool
#     reproduces the scalar SPGR equation to machine precision -- FXL
#     really is the n_pools=1 special case of the same underlying
#     equation, not a separately-derived approximation that merely
#     agrees with it.)
#
#   SSM implementation -- NOT built as a shipped model per the scope
#   decision above (no current kinetic model implies a 2-pool system).
#   Kept here purely as reference, since its derivation notes below
#   (Ce_density, the mass-balance tau_e relation) carry over directly
#   into the real SSM3 implementation just after it -- read this first
#   if SSM3's version below is unclear, then skip straight to SSM3 for
#   what actually gets built:
#     vi = 1 - ve  (ve comes from the kinetic params already fitted)
#     Ce_density = Ce / ve   -- CONFIRMED, not assumed: the whole-voxel-
#         averaged Ce from the kinetic model must be divided by ve to
#         get the LOCAL concentration density the compartment's own
#         water actually experiences. Verified concretely during
#         prototyping: using Ce directly (without this division) broke
#         the fast-exchange-limit consistency check by ~65%; dividing
#         by ve fixed it to exact agreement. This is not a minor detail,
#         it is load-bearing -- get it wrong and every downstream tau_i
#         estimate is silently biased with no error thrown anywhere.
#     R1e = 1/T10 + r1 * Ce_density
#     R1i = 1/T10  (unchanged by contrast -- no contrast reaches the
#           intracellular compartment)
        #     R1_b baseline uses AifConfig.T10 (genuinely distinct from
        #     subject.T10, NOT forced equal to it) -- confirmed against the
        #     MATLAB reference: FXL's own branch
        #     (R1 = relax.tumor + relax.ca*Cts) uses ONE baseline for the whole
        #     voxel, while SSM's branch
        #     (R1b = ...+relax.blood; R1e/R1i = ...+relax.tumor) uses TWO
        #     separate ones. This means SSM3(tau_b->0, tau_i->0) will NOT
        #     exactly equal plain FXL's output when vb>0 -- that's expected,
        #     matches the reference exactly, and is not a bug to reconcile.
        #     FXL and SSM3 are genuinely different models, not one derived as
        #     a strict limit of the other.
#     tau_e = tau_i * ve/vi  -- NOT independently free. Fixed by mass
#         balance (equal-and-opposite flux at equilibrium: ve/tau_e =
#         vi/tau_i). Confirmed during prototyping that this is the
#         correct, and only, relation -- tau_e is a derived quantity,
#         not a second free parameter.
#     X = [[-(R1e + 1/tau_e), 1/tau_i],
#          [1/tau_e,          -(R1i + 1/tau_i)]]
#     M0 = [ve, vi]   -- pure volume-fraction weights, sum to 1. S0 is
#         applied SEPARATELY, multiplicatively, AFTER this function
#         returns -- do not bake S0 into M0. Confirmed mathematically
#         equivalent either way, but keeping M0 as pure fractions keeps
#         this function's signature/contract identical in shape to
#         concentration_to_signal's existing S0-external convention, so
#         fit_single_voxel's existing S0 handling (computed once,
#         outside the optimization loop, from the measured baseline)
#         can be reused without modification.
#     eX = expm(TR*X)
#     S_t = sum( pinv(I - eX*cosFA) @ (I - eX) @ (sinFA*M0) )
#
#   SSM3 implementation (3-pool, vb/ve/vi -- THIS is what actually
#   ships. Matches MATLAB's GKMv/ATH+SSM case, and applies equally to
#   TCXM/GKM/DISPERSED_GKM per the RESOLVED CONSTRAINT guardrail 3
#   discussion -- all of them fit vp, all of them route here):
#     Same pattern, 3x3 rate matrix, needs vb (from vp, or a dedicated
#     fitted vb depending on which kinetic model -- GKMv's vb is
#     literally its own vp-equivalent parameter, confirm the mapping
#     per kinetic model rather than assuming vb=vp uniformly). Vascular
#     pool needs the SAME Ce_density-style local-concentration
#     correction (Cv_density), PLUS a separate, physically distinct use
#     of hematocrit: Cv_density = Cp(t) * (1 - Hct) -- diluting plasma
#     concentration by the fraction of vascular water that is actually
#     plasma (vs. sequestered inside red blood cells, which contrast
#     never reaches). This is NOT the same hematocrit correction
#     extract_aif already applies to the AIF -- that one converts
#     whole-blood-measured signal to plasma concentration. Both are
#     legitimate, both use the same AifConfig.hematocrit value, for two
#     different physical reasons -- comment both use sites clearly so a
#     future reader doesn't "simplify" one away as a duplicate.
#
# scipy equivalent of MATLAB's expm: scipy.linalg.expm (not
# scipy.sparse.linalg.expm -- these matrices are tiny, 2x2 or 3x3,
# dense is correct)
#
# Verification (already done for both the 2-pool prototype AND the real
# 3-pool SSM3 model during standalone prototyping -- Phase WX-8 covers
# porting these checks into the real module):
#   - FXL-limit consistency: tau_b, tau_i -> 0 (SSM3) matches plain FXL
#     to ~1e-6 (confirmed in step4)
#   - SXL-limit consistency: tau_b, tau_i -> infinity matches the fully-
#     decoupled three-independent-pools calculation to ~1e-8 (step4)
#   - Degenerate-case consistency: vb -> 0 in SSM3 reproduces the 2-pool
#     prototype's result to ~1e-5 (step4 Check 3) -- this is what
#     justifies the scope decision above (not shipping a separate
#     2-pool model)
#   - Smooth, monotonic transition between limits across the full
#     (tau_b, tau_i) grid, no jumps or sign flips (step4's 15-point sweep)
#   - Structural sanity check, confirmed empirically for realistic
#     TR/FA/R1 ranges: because the SPGR signal equation is CONCAVE in
#     R1 over the physiological range, FXL >= any intermediate-exchange
#     signal >= SXL, always, for any set of pool R1/volume values. If a
#     real fit ever produces SXL > FXL for the same tissue, that's not
#     an unusual tissue -- it's a bug. Worth keeping as a standing
#     assertion in the real test suite, not just a one-off prototyping
#     check.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
water_exchange_test.py

Comprehensive verification for water_exchange.py (Phase WX-3), run
against the real placed module -- not a reimplementation.

Covers:
  1. Rigorous fast-exchange-limit check (matrix-consistent reference,
     not a hand-derived formula) for BOTH TCXM and GKM
  2. vb -> 0 degenerate case stays finite, no crash
  3. Behavior across the ACTUAL RelaxationConfig bounds grid (not
     extreme synthetic tau values)
  4. S0 scales output linearly (pure amplitude, no shape distortion)
  5. Pool-count consistency between kinetic_models.py's
     fn_compartments and water_exchange.py's n_pools -- guardrail-3
     precursor check
  6. Real multiprocessing.Pool functional test (not just pickle.dumps)
"""

import multiprocessing

import numpy as np
from scipy.linalg import expm, pinv

from dce_mri.config import AcquisitionConfig, SubjectConfig, AifConfig, RelaxationConfig
from dce_mri.kinetic_models import TCXM, GKM, tcxm_ct_compartments, gkm_ct_compartments
from dce_mri.water_exchange import FXL, SSM3


def _M_sum_1pool(R1, TR_s, FA):
    """Matrix-consistent single-pool reference (same skeleton as
    everything else in this module, not a hand-derived formula)."""
    X = np.array([[-R1]])
    M0 = np.array([1.0])
    I = np.eye(1)
    eX = expm(TR_s * X)
    return (pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)).sum()


def _worker(args):
    Cv, Ce, ve, vp, theta_wx, acq, subject, aif_cfg, relax_cfg, S0 = args
    return SSM3.to_signal(Cv, Ce, ve, vp, theta_wx, acq, subject, aif_cfg, relax_cfg, S0)


if __name__ == "__main__":
    acq = AcquisitionConfig()
    subject = SubjectConfig("test")
    aif_cfg = AifConfig()
    relax_cfg = RelaxationConfig()
    t = np.linspace(0, 300, 60)
    t_b = t - 20
    aif = np.clip(((t_b / 15) ** 4) * np.exp(-4 * (t_b / 15 - 1)), 0, None)
    aif[:9] = 0
    aif /= aif.max()
    S0 = 1000.0

    print("=" * 60)
    print("Test 1: fast-exchange-limit check, TCXM and GKM, matrix-consistent reference")
    for name, model, compartments_fn in [("GKM", GKM, gkm_ct_compartments), ("TCXM", TCXM, tcxm_ct_compartments)]:
        Cv, Ce = compartments_fn(model.p0, t, aif)
        ve = model.p0[model.param_names.index("ve")]
        vp = model.p0[model.param_names.index("vp")]

        S_ssm3_fast = SSM3.to_signal(Cv, Ce, ve, vp, (1e-6, 1e-6), acq, subject, aif_cfg, relax_cfg, S0)

        TR_s = acq.TR / 1000.0
        FA = np.deg2rad(acq.FA * subject.B1)
        R1_tissue = 1000.0 / subject.T10
        R1_blood = 1000.0 / aif_cfg.T10
        vi = 1 - ve - vp
        Ce_density = Ce / ve
        Cv_density = Cv / vp
        R1_e_arr = R1_tissue + acq.r1 * Ce_density
        R1_b_arr = R1_blood + acq.r1 * Cv_density
        R1_avg_arr = vp * R1_b_arr + ve * R1_e_arr + vi * R1_tissue
        R1_avg_baseline = vp * R1_blood + ve * R1_tissue + vi * R1_tissue
        Eh_baseline_1p = _M_sum_1pool(R1_avg_baseline, TR_s, FA)
        S_ref = np.array([S0 * (_M_sum_1pool(R1_avg_arr[k], TR_s, FA) / Eh_baseline_1p) for k in range(len(t))])

        max_diff = np.max(np.abs(S_ssm3_fast - S_ref))
        print(f"  {name}: max diff = {max_diff:.6f}  (relative: {max_diff/S0:.2e})")
        assert max_diff / S0 < 1e-4, f"{name} failed fast-exchange-limit check"
    print("  PASS")

    print()
    print("Test 2: vb -> 0 degenerate case")
    Cv, Ce = gkm_ct_compartments(GKM.p0, t, aif)
    ve = GKM.p0[1]
    S_tiny_vb = SSM3.to_signal(Cv, Ce, ve, 1e-8, (0.005, 0.3), acq, subject, aif_cfg, relax_cfg, S0)
    assert np.all(np.isfinite(S_tiny_vb))
    print("  PASS")

    print()
    print("Test 3: RelaxationConfig bounds grid (5x5)")
    Cv, Ce = gkm_ct_compartments(GKM.p0, t, aif)
    ve, vp = GKM.p0[1], GKM.p0[2]
    n_bad = 0
    for tau_b in np.linspace(relax_cfg.tau_b_lb, relax_cfg.tau_b_ub, 5):
        for tau_i in np.linspace(relax_cfg.tau_i_lb, relax_cfg.tau_i_ub, 5):
            S = SSM3.to_signal(Cv, Ce, ve, vp, (tau_b, tau_i), acq, subject, aif_cfg, relax_cfg, S0)
            if not np.all(np.isfinite(S)):
                n_bad += 1
    assert n_bad == 0
    print("  PASS -- 25/25 grid points well-behaved")

    print()
    print("Test 4: S0 linear scaling")
    S_100 = SSM3.to_signal(Cv, Ce, ve, vp, (0.005, 0.3), acq, subject, aif_cfg, relax_cfg, 100.0)
    S_1000 = SSM3.to_signal(Cv, Ce, ve, vp, (0.005, 0.3), acq, subject, aif_cfg, relax_cfg, 1000.0)
    assert np.allclose(S_1000, S_100 * 10, rtol=1e-9)
    print("  PASS")

    print()
    print("Test 5: pool-count consistency (guardrail-3 precursor)")
    for name, model, compartments_fn in [("GKM", GKM, gkm_ct_compartments), ("TCXM", TCXM, tcxm_ct_compartments)]:
        n_curves = len(compartments_fn(model.p0, t, aif))
        assert n_curves + 1 == SSM3.n_pools
    print("  PASS")

    print()
    print("Test 6: real multiprocessing.Pool functional test")
    Cv, Ce = gkm_ct_compartments(GKM.p0, t, aif)
    tasks = [(Cv, Ce, ve, vp, (0.005, 0.3), acq, subject, aif_cfg, relax_cfg, S0) for _ in range(6)]
    with multiprocessing.Pool(3) as pool:
        results = pool.map(_worker, tasks)
    assert len(results) == 6
    assert all(np.all(np.isfinite(r)) for r in results)
    assert all(np.allclose(r, results[0]) for r in results)
    print("  PASS")

    print()
    print("ALL VERIFICATION TESTS PASSED")


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-4 — Wire into fit_single_voxel (single-voxel / volume.py)
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# PREREQUISITE (do this first, small edit, pulled forward from WX-7):
# VoxelFitResult (config.py) needs two new optional fields before
# fit_single_voxel can return wx fitting results at all:
#     wx_params:     Optional[np.ndarray] = None
#     wx_model_name: Optional[str] = None
# None default keeps every existing VoxelFitResult construction valid.
# Do this edit before touching fitting.py below -- fit_single_voxel's
# water-exchange return logic has nowhere to put its results otherwise.
#
# File: src/dce_mri/fitting.py
#
# Add THREE optional new parameters (not two -- aif_cfg is required
# too, since SSM3.to_signal needs it for the blood-T1 baseline),
# defaulting to None so every existing call site (fit_roi,
# fit_with_delay_grid/continuous, fit_dispersed_gkm, volume.py, all of
# it) continues to work completely unchanged:
#
#     def fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
#                           seed=None, wx_model=None, aif_cfg=None,
#                           relax_cfg=None):
#         _validate_wx_compatibility(wx_model, cfg.fit_space, model,
#                                     time_var, aif)   # NEW -- see
#             RESOLVED CONSTRAINT above. Call this FIRST, before
#             anything else, fail fast. Note this ALREADY includes the
#             fn_compartments-is-None check (guardrail's model check),
#             so no separate manual check is needed below -- the old
#             draft's standalone "if wx_model is not None and
#             model.fn_compartments is None" block is now redundant
#             with what the shared helper already does; don't duplicate it.
#
# When wx_model is None (the default): behave EXACTLY as today, zero
# behavior change, zero risk to existing code.
#
# When wx_model is provided: theta_kin and theta_wx get fit JOINTLY,
# matching fit_with_delay_continuous's augmented-parameter pattern.
# Two corrections vs. the original sketch:
#
#   1. Bounds/p0 order MUST match wx_model.param_names exactly --
#      SSM3.param_names = ("tau_b", "tau_i"), so tau_b's bounds go
#      FIRST in the augmented vector, tau_i's second. Getting this
#      order wrong silently swaps which fitted number means what --
#      no error, just wrong physics, so verify this carefully once
#      built (e.g. assert the augmented fit recovers a KNOWN tau_b 
#      tau_i test case in the right slots, not just "some two numbers").
#      Pull bounds from relax_cfg directly:
#          theta_aug_p0 = [*model.p0,
#                          (relax_cfg.tau_b_lb+relax_cfg.tau_b_ub)/2,
#                          (relax_cfg.tau_i_lb+relax_cfg.tau_i_ub)/2]
#          lb_aug = [*model.lb, relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]
#          ub_aug = [*model.ub, relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]
#
#   2. ve/vb must be extracted from theta_kin INSIDE residual_fn, fresh
#      on every call -- NOT precomputed once before the optimization
#      loop starts. ve/vb ARE part of theta_kin, so they change with
#      every candidate parameter set the optimizer tries:
#          ve_idx = model.param_names.index('ve')
#          vp_idx = model.param_names.index('vp')
#          def residual_fn(p):
#              theta_kin = p[:len(model.p0)]
#              theta_wx  = p[len(model.p0):]
#              ve_cand = theta_kin[ve_idx]
#              vb_cand = theta_kin[vp_idx]
#              Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)
#              S_pred = wx_model.to_signal(Cv, Ce, ve_cand, vb_cand,
#                                          theta_wx, acq, subject,
#                                          aif_cfg, relax_cfg, S0)
#              return sig - S_pred
#
# After fitting, unbundle (per Q3 -- transient bundling only, never
# left in a returned result) AND reconstruct S_pred for the final
# result, mirroring fit_single_voxel's existing reconstruction block
# but routed through wx_model.to_signal instead of
# concentration_to_signal:
#     best_params_kin = best_params[:len(model.p0)]
#     best_params_wx  = best_params[len(model.p0):]
#     Cv_final, Ce_final = model.fn_compartments(best_params_kin, time_var, aif)
#     ve_final = best_params_kin[ve_idx]
#     vb_final = best_params_kin[vp_idx]
#     S_pred = wx_model.to_signal(Cv_final, Ce_final, ve_final, vb_final,
#                                 best_params_wx, acq, subject, aif_cfg,
#                                 relax_cfg, S0_out)
#     return VoxelFitResult(
#         params=best_params_kin, cost=best_cost, success=best_status>0,
#         S_meas=sig, S_pred=S_pred,
#         wx_params=best_params_wx, wx_model_name=wx_model.name,
#     )
#
# Per Q5, decide TRF vs Nelder-Mead for this augmented fit only after
# Phase WX-8's empirical test -- don't assume TRF is fine just because
# it's already used for the kinetic-only case.

# ══════════════════════════════════════════════════════════════════════
# PHASE WX-5 — Wire into fitting_roi.py (NEW PHASE)
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/fitting_roi.py
#
# Applies to fit_roi and, once WX-6 covers the delay variants, to
# fit_roi_with_delay_grid/fit_roi_with_delay_continuous as well.
#
# The existing fit_roi 'signal' branch already calls fit_single_voxel
# directly on St_mean:
#     result = fit_single_voxel(St_mean, aif, time_var, model, acq,
#                                subject, fitting_cfg, seed)
#
# Extending this is mostly free-riding on Phase WX-4 -- add the same
# wx_model/relax_cfg passthrough parameters to fit_roi's own signature,
# and forward them into this exact call, unchanged otherwise:
#     result = fit_single_voxel(St_mean, aif, time_var, model, acq,
#                                subject, fitting_cfg, seed,
#                                wx_model=wx_model, relax_cfg=relax_cfg)
#
# The validation is where this phase earns being separate from WX-4,
# not just a copy of it. Per the RESOLVED CONSTRAINT section above,
# fit_roi needs the validation helper called with BOTH fit_space AND
# average_space, not just fit_space:
#     _validate_wx_compatibility(wx_model, fitting_cfg.fit_space,
#                                 average_space=roi_cfg.average_space)
#
# This should run BEFORE the existing average_space=='signal' + fit_
# space mismatch check that's already there (or the two checks can be
# combined into one call to the shared helper, replacing the current
# hand-written raise block -- worth doing regardless of wx_model, since
# it removes duplicated logic and reduces the chance of the kind of
# copy-paste drift already caught once in this file, between the grid
# and continuous variants' print statements).
#
# The 'concentration' branch requires NO changes and should not be
# touched -- it structurally cannot support wx_model (see RESOLVED
# CONSTRAINT above), and the validation helper's average_space check
# already rejects that combination cleanly with a clear error rather
# than needing special-case handling inside the branch itself.
#
# Verification:
#   - wx_model=None: fit_roi's behavior is completely unchanged (same
#     kind of before/after regression check already used for the
#     fit_space merge/guardrail change earlier this session)
#   - wx_model provided + average_space='signal' + fit_space='signal':
#     fits successfully, wx_params populated
#   - wx_model provided + average_space='concentration' (regardless of
#     fit_space): raises clearly, does NOT silently ignore wx_model --
#     this is the specific gap this phase exists to close, so it's
#     worth a dedicated test case confirming the fix, not just
#     confirming the happy path works


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-6 — fit_with_delay_grid / fit_with_delay_continuous, and
#              fit_roi_with_delay_grid / fit_roi_with_delay_continuous
# STATUS: NOT STARTED (only if you want delay + water-exchange combined)
# ══════════════════════════════════════════════════════════════════════
#
# Same optional wx_model/relax_cfg parameters, same default-None
# backward-compatibility pattern as Phase WX-4/WX-5, same shared
# _validate_wx_compatibility call (with average_space passed for the
# ROI variants, omitted for the single-voxel ones). Lower priority --
# confirm Phase WX-4/WX-5 work and are actually useful before extending
# the delay-fitting variants too. Combining AIF delay + water exchange
# in one joint fit is a THIRD parameter group stacked onto theta --
# worth checking empirically whether that's over-parameterized for
# realistic temporal resolution before building it (same class of
# concern as the n_lhs-budget finding for dispersed GKM earlier this
# session -- more free parameters generally need more coarse-search
# budget to remain reliable, this may be no exception).


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-7 — Result types
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/config.py
#
# Per Q3 (recommended: separate fields, following DelayedFitResult's
# actual precedent -- bundling is transient/internal-only, never left
# in a returned result object):
#
# VoxelFitResult (used by fit_single_voxel, and therefore by fit_roi's
# 'signal' branch too) gets two new OPTIONAL fields:
#   wx_params: Optional[np.ndarray] = None   (e.g. [tau_i] or [tau_i, tau_b])
#   wx_model_name: Optional[str] = None
#   None default keeps every existing VoxelFitResult construction valid.
#
# If Phase WX-6 is built, DelayedFitResult (and, per the RoiDelayedFitResult
# precedent from earlier this session, RoiDelayedFitResult too) would need
# the same two optional fields, for the same reason.
#
# Alternative: new WxVoxelFitResult wrapping VoxelFitResult the same
# way RoiDelayedFitResult wraps DelayedFitResult -- keeps
# VoxelFitResult itself completely untouched, at the cost of one more
# type in the system. Given this session already added
# RoiDelayedFitResult via exactly this wrapping pattern, this is the
# more consistent choice if you want to keep to precedent -- flagging
# both as legitimate rather than picking for you.


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-8 — Verification plan
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# In order, cheapest/most-informative first. Several of these were
# already run successfully during standalone prototyping (outside the
# package, in step1/2/3_*.py) -- this phase is largely "port those
# checks into the real module and its test suite," not "invent new
# checks from nothing."
#
#   1. FXL-limit and SXL-limit consistency -- but NOT against plain
#      FXL (concentration_to_signal) directly. Confirmed during
#      prototyping that SSM3(tau->0) and FXL genuinely differ by a
#      small, real, PREDICTABLE amount whenever vb>0 -- both this
#      codebase's concentration_to_signal AND the MATLAB reference use
#      ONE baseline (tissue only) for FXL, while SSM's fast-exchange
#      limit correctly uses a volume-weighted baseline
#      (vb*R1_blood + (ve+vi)*R1_tissue). Check SSM3(tau->0) against
#      THIS computed reference instead -- a real formula, not a loose
#      tolerance. SXL-limit check (tau->infinity, matches the fully
#      decoupled independent-pools calculation) is unaffected by this
#      and still checks to ~1e-8 as originally verified.
#
#   2. The FXL >= signal >= SXL structural check (concavity-derived,
#      confirmed empirically across representative TR/FA/R1 ranges) --
#      worth keeping as a standing assertion for ANY real fitted
#      voxel's wx result, not just synthetic test cases, since a
#      violation is a bug regardless of what tissue produced it.
#
#   3. Compartment-sum regression (Phase WX-2's own verification) --
#      confirms tcxm_ct_analytic/gkm_ct's existing behavior is
#      unchanged after refactoring them into wrappers.
#
#   4. Ce_density = Ce/ve convention check -- this was the single most
#      consequential bug found during prototyping (a silent ~65% error
#      in the fast-exchange-limit check). Worth a dedicated,
#      permanently-kept regression test specifically for this, not
#      just folded into the general FXL-limit check, given how easy it
#      was to get wrong and how silent the failure was.
#
#   5. Synthetic SSM recovery test: generate data with dispersed_gkm_ct-
#      style known ground truth (Ktrans, ve, vp, tau_i), run it through
#      the new to_signal path, fit it back, confirm recovery -- same
#      pattern as every synthetic-recovery test this session
#      (fit_single_level_test.py, aic_weights_test.py, etc.)
#
#   6. wx-compatibility guardrail tests -- all THREE checks in
#      _validate_wx_compatibility, not just the fit_space one:
#        - average_space='concentration' + wx_model provided raises
#          (Phase WX-5's own verification target -- the exact gap this
#          whole update exists to close)
#        - fit_space='concentration' + wx_model provided raises
#        - pool-count mismatch raises -- e.g. constructing a
#          hypothetical 2-pool wx_model and pairing it with TCXM/GKM/
#          DISPERSED_GKM should raise, confirming the structural
#          fn_compartments-based check actually catches the mismatch
#          it's designed for, not just the two happy-path model/wx_model
#          pairings that exist today
#
#   7. TRF vs Nelder-Mead comparison for tau_i specifically (per Q5),
#      the same kind of empirical comparison run for the vascular
#      dispersion t0 parameter earlier this session -- don't assume,
#      test. Given the confirmed non-uniform sensitivity of signal to
#      tau_i (flat at extremes, steep in an intermediate range), also
#      worth checking whether recovery reliability itself varies
#      depending on where the TRUE tau_i sits relative to that sensitive
#      range -- a fit-quality effect distinct from, but related to,
#      the optimizer-choice question.
#
#   8. n_lhs sensitivity for the augmented (kinetic + wx) parameter
#      vector, the same kind of test that revealed fit_dispersed_gkm
#      needed more LHS budget than its default -- an extra 1-2 free
#      parameters may need a larger n_lhs than what's currently
#      sufficient for TCXM/GKM alone.
#
#   9. Real-data sanity check against a known literature tau_i range
#      for the tissue type you're studying (breast, per your fastMRI
#      breast cohort) -- confirm fitted values land in a physiologically
#      plausible range before trusting them on real patient data.


# ══════════════════════════════════════════════════════════════════════
# PHASE WX-9 — "Plug in models" extensibility check
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# Once WX-1 through WX-8 are done, sanity-check the architecture
# actually achieves the stated "plug in models" goal:
#
#   - Could a THIRD water-exchange model (not FXL, not SSM) be added
#     by writing one new WaterExchangeModelSpec instance, with zero
#     changes to fit_single_voxel or fitting_roi.py? Should be yes, if
#     Phase WX-4/WX-5's wx_model dispatch is written generically.
#
#   - Could a new kinetic model (e.g. NPL or EPL from the MATLAB code,
#     if ever wanted) support water exchange by adding one
#     fn_compartments companion function, with zero changes to
#     water_exchange.py? Should be yes, if Phase WX-2/WX-3's interface
#     boundary is clean (water_exchange.py only ever consumes (Cv, Ce),
#     never anything kinetic-model-specific).
#
# If either check fails, that's a sign the interface boundary between
# "kinetics" and "water exchange" leaked something model-specific
# across it somewhere in WX-2 through WX-5 -- worth revisiting before
# calling this phase done.


# ══════════════════════════════════════════════════════════════════════
# DEFERRED — Joint global fitting (ADMM)
# ══════════════════════════════════════════════════════════════════════
#
# Explicitly set aside for now, per discussion. Not designed here since
# fitting_admm.py is currently entirely stubs (Phase 7 of the original
# REFACTOR_OUTLINE.py) -- there's nothing concrete yet for water
# exchange to integrate with.
#
# One thing worth keeping in mind once ADMM exists, not acting on now:
# if ADMM's eventual per-voxel x-update step reuses fit_single_voxel
# internally as its inner-loop local fit (plausible, since ADMM needs
# a per-voxel local fit regularized toward a spatial prior, and
# fit_single_voxel is the existing per-voxel fitter), then Phase WX-4's
# wx_model/relax_cfg extension may carry through to ADMM "for free" --
# i.e. worth checking, once ADMM is real, whether it needs its own
# water-exchange wiring at all, or whether it already gets it by
# construction. Revisit this specifically when Phase 7 moves off
# NOT STARTED, rather than guessing now.


# ══════════════════════════════════════════════════════════════════════
# RECOMMENDED BUILD ORDER
# ══════════════════════════════════════════════════════════════════════
#
#   1. Resolve Q1, Q2, Q4 (or as many as you're ready to commit to --
#      Q3 is now effectively resolved per the RESOLVED CONSTRAINT
#      section's precedent-following recommendation)
#   2. Phase WX-2 (compartment-resolved kinetics) -- lowest risk,
#      additive only, immediately verifiable in isolation
#   3. Phase WX-1 (RelaxationConfig) -- small, unblocks WX-3
#   4. Phase WX-3 (water_exchange.py) -- port the already-verified
#      standalone prototype (step1/2/3_*.py) into the real module,
#      keeping the Ce_density=Ce/ve convention and the FXL/SXL
#      consistency checks intact exactly as verified
#   5. Phase WX-4 (wire into fit_single_voxel) -- the first real
#      integration point, including the shared validation helper
#   6. Phase WX-5 (wire into fitting_roi.py) -- the second integration
#      point, closes the average_space gap the validation helper exists
#      to catch
#   7. Phase WX-7 (result types) -- needed to actually get wx_params
#      back out of a fit, from either entry point
#   8. Phase WX-8 (full verification suite) -- don't skip the
#      guardrail tests (item 6), the Ce_density regression test (item
#      4), or the TRF-vs-Nelder-Mead and n_lhs-sensitivity checks
#      (items 7-8); all four have precedent this session of catching
#      real, non-obvious problems
#   9. Phase WX-6 (delay + wx combined) and WX-9 (extensibility check)
#      -- only if steps 1-8 are solid and you still want them
#  10. Joint global fitting integration -- revisit once Phase 7 of the
#      original REFACTOR_OUTLINE.py is no longer NOT STARTED