"""
check_17_stage2_5d.py
=====================
check_14 in FIVE dimensions, for the two 5-parameter vectors this project
actually fits:

  variant "tau_i"      (ve, vp, fp, ps, tau_i)      -- the WX FIXED-delay run
                       (fit_lesions_bayes_wx_fixedtau.py: TCXM + fixed delay
                        + SSM3, tau_b held fixed)
  variant "tau_delay"  (ve, vp, fp, ps, tau_delay)  -- the FXL JOINT-delay run
                       (the run that supplies the fixed delays)

No single run samples all six parameters, so these are the only two 5-D
models the existing stage-1 draws support. check_14's 4-D result is the
baseline; T6 below compares against it on the same cohort.

WHY THE POPULATION SCALE DIFFERS BETWEEN THE VARIANTS (not a style choice)
  tau_i     -> log tau_i. Its stage-1 prior ("tau": "flat") is LOG-uniform, so
               p1_x(tau) proportional to 1/tau exactly cancels the Jacobian
               dtau/dlog tau = tau. log p1 on that coordinate is therefore
               CONSTANT, and the p1 correction has NO POWER on it. That is the
               theory, not a defect: it is the same thing check_14's run C
               showed for log_fp / log_ps (paired shifts 0.03 and 0.06 SD).
  tau_delay -> logit(tau / delay_ub). log is impossible: the delay's lower
               bound is 0. Its prior is uniform on [0, 25] (verified here in
               T0: the block's log_pdf is constant at -log 25), so on the logit
               scale log p1 keeps log w + log(1-w) with w = tau/25 -- NOT flat,
               so for this variant the correction IS load-bearing and T4 has
               real power on the 5th coordinate.
               DELAY_SCALE = "linear" keeps tau_delay in seconds instead. Then
               log p1 is flat there (no correction, no T4 power) and the
               Gaussian population can put mass below 0, which the logit scale
               cannot. Both are defensible; the logit is the default because it
               respects the support.

A BUG THIS SCRIPT WORKS AROUND
  hierarchical_lunn.fit_stage2 never passes `names`, so Stage2Result.names
  keeps the 4-element THETA_NAMES default. At D = 5, Stage2Result.summary()
  and check_14's report() both iterate enumerate(res.names) and SILENTLY DROP
  the 5th dimension -- no error, just a missing row. The Gibbs updates
  themselves are dimension-generic (kron(XtX, Sinv), eye(P*D), nu0 = D+2,
  Psi0 = eye(D)), so the fit is correct; only the reporting truncates.
  Fixed here by passing names explicitly. The library fix is one line in
  fit_stage2's return:
        names=tuple(THETA_NAMES) if D == 4 else tuple(f"theta{j}" for j in range(D)),
  or better, add a `names=None` argument and use it.

TESTS (same structure as check_14, plus T6)
  T0  the 5-D log_p1 against the PRIOR OBJECTS assemble_prior builds for each
      run -- including the tau / delay block. This is the test that caught a
      reversed rate family last time; it is the only thing standing between a
      wrong Jacobian and a plausible-looking population.
  T1  flat p1: recover MU and the population SDs.      |z| < 3 expected.
  T2  coverage of the 90% intervals over R cohorts.
  T3  non-flat p1 WITH the correction.
  T4  NEGATIVE CONTROL: same draws, correct_p1=False. Reported PER COORDINATE,
      because for the tau_i variant the 5th coordinate is expected to show no
      shift at all while ve/vp do.
  T5  group difference, with a real effect in the 5th coordinate for tau_i and
      a NULL for tau_delay (the AIF delay should not differ by diagnosis, so
      recovering ~0 there is the check).
  T6  DEGENERACY vs 4-D: the same cohort fitted with the first four
      coordinates only, so the acceptance rate and the distinct-draw count can
      be compared 4-D vs 5-D. This is the question check_14 could not answer:
      does a fifth dimension starve the independence sampler?

Run:
    PYTHONPATH=src python check_17_stage2_5d.py            # both variants
    PYTHONPATH=src python check_17_stage2_5d.py --quick
    PYTHONPATH=src python check_17_stage2_5d.py --only tau_i
"""
import os
import sys
import time

import numpy as np
from scipy import stats

from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.config import AcquisitionConfig, RelaxationConfig
from dce_mri import hierarchical_lunn as hl

QUICK = "--quick" in sys.argv
ONLY = None
if "--only" in sys.argv:
    ONLY = sys.argv[sys.argv.index("--only") + 1]

L, K = (40, 1500) if QUICK else (120, 3000)
N_DRAWS, N_TUNE = (1500, 500) if QUICK else (4000, 1000)
R_COHORTS = 15 if QUICK else 40
D = 5

DELAY_SCALE = "logit"          # "logit" | "linear"  (see the header)
DELAY_UB = 25.0                # must match the fit run's delay_bounds[1]
THETA_SWEEPS_HI = 8            # T6b: independence-MH passes per Gibbs sweep.
                               # The only tuning knob on the theta side -- the
                               # proposal is the saved stage-1 draws, so it
                               # cannot be widened or rescaled.

relax_cfg = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005,
                             tau_i_lb=0.01, tau_i_ub=1.5)
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))

# =============================================================================
# the two variants
# =============================================================================
# x5_truth   : the 5th parameter's population centre, in its natural unit
# sd         : population SDs on the POPULATION scale, 5th entry included
# corr_row   : correlation of the 5th coordinate with (ve, vp, fp, ps)
# v_range    : per-lesion stage-1 width for the 5th coordinate. tau_i is
#              DELIBERATELY wide: it is the weakly identified parameter, and a
#              wide per-lesion posterior is what stresses the sampler.
# delta5     : T5's group difference in the 5th coordinate
# choices_*  : the prior families of the corresponding real run
VARIANTS = {
    "tau_i": dict(
        name5="log_tau_i", x5_truth=0.20,
        sd=np.array([0.55, 0.70, 0.60, 0.65, 0.80]),
        corr_row=np.array([0.15, -0.10, 0.00, 0.20]),
        v_range=(0.60, 1.50),
        delta5=-0.40,
        choices_flatish={"rates": "loguniform", "volfrac": "flat_masked",
                         "tau": "flat"},
        choices_nonflat={"rates": "uniform", "volfrac": "flat_masked",
                         "tau": "flat"},
    ),
    "tau_delay": dict(
        name5=("logit_tau_delay" if DELAY_SCALE == "logit" else "tau_delay_s"),
        x5_truth=5.0,
        sd=np.array([0.55, 0.70, 0.60, 0.65,
                     0.50 if DELAY_SCALE == "logit" else 2.0]),
        corr_row=np.array([0.05, 0.10, 0.00, 0.00]),
        v_range=((0.20, 0.60) if DELAY_SCALE == "logit" else (0.8, 2.5)),
        delta5=0.00,
        choices_flatish={"rates": "loguniform", "volfrac": "flat_masked"},
        choices_nonflat={"rates": "uniform", "volfrac": "beta"},
    ),
}
CORR4 = np.array([[1.0, 0.30, 0.00, 0.20],
                  [0.30, 1.0, 0.35, 0.00],
                  [0.00, 0.35, 1.0, 0.30],
                  [0.20, 0.00, 0.30, 1.0]])
V_SD_RANGE4 = (0.25, 0.75)
DELTA4 = np.array([0.45, -0.60, 0.50, 0.00])


def bounds(variant):
    """(lb, ub) of the 5-vector, in the fit's own order and units."""
    lb = np.r_[np.asarray(TCXM.lb, float)[:4],
               relax_cfg.tau_i_lb if variant == "tau_i" else 0.0]
    ub = np.r_[np.asarray(TCXM.ub, float)[:4],
               relax_cfg.tau_i_ub if variant == "tau_i" else DELAY_UB]
    return lb, ub


def to_theta5(X, variant):
    """(..., 5) natural units -> population scale."""
    X = np.asarray(X, float)
    th4 = hl.to_theta(X[..., :4])
    x5 = X[..., 4]
    if variant == "tau_i":
        t5 = np.log(x5)
    elif DELAY_SCALE == "logit":
        w = np.clip(x5 / DELAY_UB, 1e-12, 1 - 1e-12)
        t5 = np.log(w) - np.log1p(-w)
    else:
        t5 = x5
    return np.concatenate([th4, t5[..., None]], axis=-1)


def from_theta5(TH, variant):
    """Inverse of to_theta5."""
    TH = np.asarray(TH, float)
    x4 = hl.from_theta(TH[..., :4])
    t5 = TH[..., 4]
    if variant == "tau_i":
        x5 = np.exp(t5)
    elif DELAY_SCALE == "logit":
        x5 = DELAY_UB / (1.0 + np.exp(-t5))
    else:
        x5 = t5
    return np.concatenate([x4, x5[..., None]], axis=-1)


def log_p1_theta5(choices, lb, ub, variant):
    """
    The stage-1 prior on the 5-D population scale, from CONSTRAINED draws.

    The first four coordinates are hierarchical_lunn.log_p1_theta exactly. The
    fifth adds its own prior and its own Jacobian:

      tau_i, "tau": "flat"  ->  log(1/tau) + log(tau) = 0   (flat: no correction)
      tau_i, "tau": "uniform" -> log(tau)                   (Jacobian only)
      tau_delay, logit scale  -> log(w) + log(1-w),  w = tau/DELAY_UB
      tau_delay, linear scale -> 0
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    base4 = hl.log_p1_theta(choices, lb[:4], ub[:4])
    tau_choice = choices.get("tau", "flat")
    if variant == "tau_i" and tau_choice not in ("flat", "uniform"):
        raise ValueError(f"tau prior {tau_choice!r}: expected 'flat' or 'uniform'")

    def log_p1(X):
        X = np.asarray(X, float)
        out = base4(X[..., :4])
        x5 = X[..., 4]
        inside = (x5 >= lb[4]) & (x5 <= ub[4])
        with np.errstate(divide="ignore", invalid="ignore"):
            if variant == "tau_i":
                # Jacobian dtau/dlog(tau) = tau
                add = np.log(x5)
                if tau_choice == "flat":        # p1_x propto 1/tau cancels it
                    add = add - np.log(x5)
            elif DELAY_SCALE == "logit":
                w = x5 / DELAY_UB
                add = np.log(w) + np.log1p(-w)   # + log(DELAY_UB), a constant
            else:
                add = np.zeros_like(x5)
        return np.where(inside, out + add, -np.inf)

    return log_p1


# =============================================================================
# T0: the 5-D log_p1 against the prior objects a real run builds
# =============================================================================
def real_blocks(variant, choices):
    from dce_mri.bayes import BayesConfig, build_param_layout, NOISE_OPTIONS
    from dce_mri.bayes_priors import assemble_prior
    from dce_mri.synthetic import make_parker_aif
    tt = np.arange(36) * 150 * 8 / 288
    aif = make_parker_aif(tt, t_arrival=35.0, hematocrit=0.4)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    if variant == "tau_i":
        cb = BayesConfig(prior_choices=choices, noise="gauss_marginalized_sigma",
                         delay_mode="fixed", delay_fixed=5.0,
                         delay_bounds=(0.0, DELAY_UB))
        lay = build_param_layout(TCXM, cb, aif=aif, time_var=tt, acq=acq,
                                 wx_model=SSM3, relax_cfg=relax_cfg)
        pr = assemble_prior(lay.model, relax_cfg, wx_model=SSM3,
                            delay_mode="fixed", n_noise=noise.n_params,
                            noise_names=noise.param_names, choices=choices,
                            delay_bounds=(0.0, DELAY_UB))
    else:
        cb = BayesConfig(prior_choices=choices, noise="gauss_marginalized_sigma",
                         delay_mode="joint", delay_bounds=(0.0, DELAY_UB))
        lay = build_param_layout(TCXM, cb, aif=aif, time_var=tt, acq=acq)
        pr = assemble_prior(lay.model, None, delay_mode="joint",
                            n_noise=noise.n_params,
                            noise_names=noise.param_names, choices=choices,
                            delay_bounds=(0.0, DELAY_UB))
    keep = ("volfrac", "rates", "tau" if variant == "tau_i" else "delay")
    return [b for b in pr.blocks if b.name in keep], lay


def test_T0(variant):
    lb, ub = bounds(variant)
    worst = 0.0
    V = VARIANTS[variant]
    for tag, ch in (("flat-ish", V["choices_flatish"]),
                    ("non-flat", V["choices_nonflat"])):
        bl, lay = real_blocks(variant, ch)
        names = list(lay.param_names)
        if names[:5] != ["ve", "vp", "fp", "ps",
                         "tau_i" if variant == "tau_i" else "tau_delay"]:
            raise AssertionError(f"unexpected sampled vector {names}")
        rng = np.random.default_rng(99)
        Xs = np.zeros((4000, 5))
        for b in bl:
            Xs[:, list(b.indices)] = b.sample(rng, len(Xs))
        # the package's own prior density, plus the change-of-variables Jacobian
        ref = np.array([sum(b.log_pdf(x[list(b.indices)]) for b in bl) for x in Xs])
        jac = (np.log(Xs[:, 0]) + np.log1p(-Xs[:, 0])
               + np.log(Xs[:, 1]) + np.log1p(-Xs[:, 1])
               + np.log(Xs[:, 2]) + np.log(Xs[:, 3]))
        if variant == "tau_i":
            jac = jac + np.log(Xs[:, 4])
        elif DELAY_SCALE == "logit":
            w = Xs[:, 4] / DELAY_UB
            jac = jac + np.log(w) + np.log1p(-w)
        mine = log_p1_theta5(ch, lb, ub, variant)(Xs)
        d = mine - (ref + jac)
        ok = np.isfinite(d)
        spread = float(np.max(d[ok]) - np.min(d[ok]))
        worst = max(worst, spread)
        print(f"      {tag:9s} {str(ch):62s} spread {spread:.2e}  "
              f"{'OK' if spread < 1e-9 else 'DIFFERENT DENSITY'}")
    return worst


# =============================================================================
# synthetic stage 1
# =============================================================================
def build(variant):
    V = VARIANTS[variant]
    mu4 = hl.to_theta(np.array([0.30, 0.05, 0.010, 0.005]))
    x5 = np.array([0.30, 0.05, 0.010, 0.005, V["x5_truth"]])
    MU = to_theta5(x5, variant)
    assert np.allclose(MU[:4], mu4)
    corr = np.eye(5)
    corr[:4, :4] = CORR4
    corr[4, :4] = corr[:4, 4] = V["corr_row"]
    sd = V["sd"]
    SIGMA = corr * np.outer(sd, sd)
    ev = np.linalg.eigvalsh(corr)
    if ev.min() <= 1e-8:
        raise ValueError(f"{variant}: the 5x5 correlation matrix is not positive "
                         f"definite (min eigenvalue {ev.min():.2e}) -- fix corr_row")
    return MU, sd, SIGMA, float(ev.min())


def make_cohort(variant, MU, SIGMA, rng, n, choices=None, group_delta=None, k=K):
    """(Stage1Samples, groups or None, mean importance-weight ESS)."""
    V = VARIANTS[variant]
    lb, ub = bounds(variant)
    groups = None
    m = np.tile(MU, (n, 1))
    if group_delta is not None:
        groups = (np.arange(n) % 2).astype(float)
        m = m + groups[:, None] * np.asarray(group_delta, float)
    theta_true = m + rng.multivariate_normal(np.zeros(D), SIGMA, size=n)

    draws = np.empty((n, k, D))
    log_p1 = np.zeros((n, k))
    ess = []
    fn = log_p1_theta5(choices, lb, ub, variant) if choices is not None else None
    for i in range(n):
        v = np.r_[rng.uniform(*V_SD_RANGE4, size=4),
                  rng.uniform(*V["v_range"])]
        hat = theta_true[i] + rng.standard_normal(D) * v
        if fn is None:
            draws[i] = hat + rng.standard_normal((k, D)) * v
            continue
        M = 60 * k                      # more candidates than 4-D: 5-D weights
        cand = hat + rng.standard_normal((M, D)) * v     # vary more
        lw = fn(from_theta5(cand, variant))
        ok = np.isfinite(lw)
        cand, lw = cand[ok], lw[ok]
        if len(cand) < 10 * k:
            print(f"      !! lesion {i}: only {len(cand)} of {M} candidates are "
                  f"inside the bounds; the synthetic stage 1 is unreliable here")
        w = np.exp(lw - lw.max())
        ess.append(w.sum() ** 2 / np.sum(w ** 2))
        take = rng.choice(len(cand), size=k, replace=True, p=w / w.sum())
        draws[i] = cand[take]
        log_p1[i] = lw[take]
    return (hl.Stage1Samples(draws=draws, log_p1=log_p1), groups,
            (float(np.mean(ess)) if ess else np.nan))


# =============================================================================
def report(tag, res, names, truth_mu, truth_sd, row=0):
    res.names = tuple(names)            # fit_stage2 does not set these (see header)
    s = res.summary(row=row)
    zs, zsd = [], []
    print(f"\n{tag}")
    print(f"   acceptance {res.accept.min():.3f}-{res.accept.max():.3f} "
          f"(median {np.median(res.accept):.3f})   distinct draws visited "
          f"{res.n_distinct.min()}-{res.n_distinct.max()} of {res.meta['K']}")
    print(f"   {'param':16s} {'truth':>8s} {'median':>8s} {'90% interval':>20s} "
          f"{'z':>6s}    {'pop SD truth':>12s} {'median':>7s} {'z':>6s}")
    for j, nm in enumerate(names):
        d = s[nm]
        z = (d["median"] - truth_mu[j]) / d["sd"]
        sdd = res.sd()[:, j]
        zd = (np.median(sdd) - truth_sd[j]) / np.std(sdd)
        zs.append(abs(z)); zsd.append(abs(zd))
        print(f"   {nm:16s} {truth_mu[j]:8.3f} {d['median']:8.3f} "
              f"[{d['q05']:8.3f},{d['q95']:8.3f}] {z:6.2f}    "
              f"{truth_sd[j]:12.3f} {np.median(sdd):7.3f} {zd:6.2f}")
    return max(zs), max(zsd)


def run_variant(variant):
    V = VARIANTS[variant]
    names = list(hl.THETA_NAMES) + [V["name5"]]
    MU, SD5, SIGMA, ev = build(variant)
    lb, ub = bounds(variant)
    print("\n" + "#" * 78)
    print(f"# VARIANT {variant}   theta = {names}")
    print(f"#   the {'WX FIXED-delay' if variant == 'tau_i' else 'FXL JOINT-delay'} "
          f"run's sampled vector")
    print("#" * 78)
    print(f"truth MU            = {np.round(MU, 3)}")
    print(f"   (ve 0.30, vp 0.05, fp 0.010, ps 0.005, "
          f"{'tau_i' if variant == 'tau_i' else 'tau_delay'} {V['x5_truth']})")
    print(f"truth population SD = {SD5}   (5x5 corr min eigenvalue {ev:.3f})")
    print(f"5th-coordinate bounds {lb[4]} to {ub[4]}; per-lesion stage-1 width "
          f"{V['v_range']}")

    print("\nT0  5-D log_p1 vs the prior objects assemble_prior builds "
          "(spread must be ~0: only a constant offset is allowed)")
    t0 = test_T0(variant)

    # ---- T1 -----------------------------------------------------------------
    rng = np.random.default_rng(170)
    s1, _, _ = make_cohort(variant, MU, SIGMA, rng, L)
    t = time.time()
    r1 = hl.fit_stage2(s1, n_draws=N_DRAWS, n_tune=N_TUNE, seed=1)
    z1, zsd1 = report(f"T1  flat p1, single population  ({time.time() - t:.0f} s)",
                      r1, names, MU, SD5)
    r1b = hl.fit_stage2(s1, n_draws=N_DRAWS, n_tune=N_TUNE, seed=2)
    seed_gap = max(abs(np.median(r1.mu()[:, j]) - np.median(r1b.mu()[:, j]))
                   / np.std(r1.mu()[:, j]) for j in range(D))
    print(f"   seed 1 vs seed 2, worst |median difference| / posterior SD: "
          f"{seed_gap:.2f}")

    # ---- T6 degeneracy vs 4-D, SAME cohort ----------------------------------
    s4 = hl.Stage1Samples(draws=s1.draws[:, :, :4],
                          log_p1=np.zeros((L, s1.draws.shape[1])))
    r4d = hl.fit_stage2(s4, n_draws=N_DRAWS, n_tune=N_TUNE, seed=1)
    # the remedy, measured rather than assumed: more independence-MH passes per
    # Gibbs sweep is the only knob on the theta side (the proposal is fixed by
    # stage 1, so there is nothing else to tune)
    r5s = hl.fit_stage2(s1, n_draws=N_DRAWS, n_tune=N_TUNE, seed=1,
                        theta_sweeps=THETA_SWEEPS_HI)
    r5sb = hl.fit_stage2(s1, n_draws=N_DRAWS, n_tune=N_TUNE, seed=2,
                         theta_sweeps=THETA_SWEEPS_HI)
    gap_hi = max(abs(np.median(r5s.mu()[:, j]) - np.median(r5sb.mu()[:, j]))
                 / np.std(r5s.mu()[:, j]) for j in range(D))
    print(f"\nT6  degeneracy, same cohort: 4-D vs 5-D, and 5-D with more sweeps")
    print(f"   {'':22s} {'accept min':>11s} {'accept med':>11s} "
          f"{'distinct min':>13s} {'distinct med':>13s}")
    for tag, r in (("4-D, 1 sweep", r4d), ("5-D, 1 sweep", r1),
                   (f"5-D, {THETA_SWEEPS_HI} sweeps", r5s)):
        print(f"   {tag:22s} {r.accept.min():11.3f} {np.median(r.accept):11.3f} "
              f"{r.n_distinct.min():13d} {int(np.median(r.n_distinct)):13d}")
    print(f"   worst 5-D lesion holds {r1.n_distinct.min()} of {r1.meta['K']} "
          f"distinct draws ({100 * r1.n_distinct.min() / r1.meta['K']:.1f}%) at 1 "
          f"sweep, {r5s.n_distinct.min()} "
          f"({100 * r5s.n_distinct.min() / r1.meta['K']:.1f}%) at "
          f"{THETA_SWEEPS_HI}")
    print(f"   seed-to-seed worst |median difference| / posterior SD: "
          f"{seed_gap:.2f} at 1 sweep, {gap_hi:.2f} at {THETA_SWEEPS_HI}  "
          f"(small = stage 2 has mixed)")
    z_hi, zsd_hi = report(f"T6b 5-D with theta_sweeps={THETA_SWEEPS_HI}",
                          r5s, names, MU, SD5)

    # ---- T2 coverage --------------------------------------------------------
    print(f"\nT2  coverage of the 90% interval for MU, {R_COHORTS} cohorts",
          flush=True)
    hits = np.zeros(D)
    for r in range(R_COHORTS):
        rr = np.random.default_rng(3000 + r)
        s, _, _ = make_cohort(variant, MU, SIGMA, rr, L, k=K // 2)
        res = hl.fit_stage2(s, n_draws=N_DRAWS // 2, n_tune=N_TUNE, seed=r)
        res.names = tuple(names)
        su = res.summary()
        for j, nm in enumerate(names):
            hits[j] += su[nm]["q05"] <= MU[j] <= su[nm]["q95"]
    cov = hits / R_COHORTS
    print("   " + "  ".join(f"{nm} {c:.2f}" for nm, c in zip(names, cov)))
    lo, hi = stats.binom.interval(0.95, R_COHORTS, 0.90)
    print(f"   expected 0.90; with {R_COHORTS} cohorts a correct sampler lands "
          f"in [{lo / R_COHORTS:.2f}, {hi / R_COHORTS:.2f}] 95% of the time")
    ok_cov = bool(np.all((hits >= lo) & (hits <= hi)))

    # ---- T3 / T4 ------------------------------------------------------------
    t34 = {}
    for tag, ch in (("flat-ish", V["choices_flatish"]),
                    ("non-flat", V["choices_nonflat"])):
        rng = np.random.default_rng(171)
        s3, _, ess3 = make_cohort(variant, MU, SIGMA, rng, L, choices=ch)
        spread = float(np.mean(np.ptp(s3.log_p1, axis=1)))
        print(f"\n({tag} p1 = {ch};\n  importance-weight ESS while building the "
              f"stage-1 draws {ess3:.0f}; mean within-lesion range of log p1 "
              f"{spread:.2f})")
        r3 = hl.fit_stage2(s3, n_draws=N_DRAWS, n_tune=N_TUNE, seed=3)
        z3, zsd3 = report(f"T3  {tag}, correction ON", r3, names, MU, SD5)
        r4 = hl.fit_stage2(s3, n_draws=N_DRAWS, n_tune=N_TUNE, seed=3,
                           correct_p1=False)
        z4, zsd4 = report(f"T4  NEGATIVE CONTROL: {tag}, correction OFF",
                          r4, names, MU, SD5)
        paired = [abs(np.median(r3.mu()[:, j]) - np.median(r4.mu()[:, j]))
                  / np.std(r3.mu()[:, j]) for j in range(D)]
        nat = [np.median(r4.mu()[:, j]) - np.median(r3.mu()[:, j]) for j in range(D)]
        print("   paired ON vs OFF, |median shift| / posterior SD: "
              + "  ".join(f"{n} {p:.2f}" for n, p in zip(names, paired)))
        print("   same shift in natural (population-scale) units:            "
              + "  ".join(f"{n} {b:+.3f}" for n, b in zip(names, nat)))
        if variant == "tau_i":
            print(f"   NOTE: {names[4]} is EXPECTED to show ~0 shift -- its "
                  f"log-uniform stage-1 prior is\n         flat on the log scale, "
                  f"so there is no correction to apply on that coordinate.")
        t34[tag] = (max(z3, zsd3), max(z4, zsd4), spread, max(paired), paired[4])

    # ---- T5 group difference ------------------------------------------------
    DELTA = np.r_[DELTA4, V["delta5"]]
    rng = np.random.default_rng(172)
    s5, g5, _ = make_cohort(variant, MU, SIGMA, rng, L, group_delta=DELTA)
    X5 = np.column_stack([np.ones(L), g5])
    r5 = hl.fit_stage2(s5, design=X5,
                       coef_names=("benign", "malignant_minus_benign"),
                       n_draws=N_DRAWS, n_tune=N_TUNE, seed=5)
    z5a, _ = report("T5  two groups: benign mean", r5, names, MU, SD5, row=0)
    z5b, _ = report("T5  two groups: malignant - benign", r5, names, DELTA, SD5,
                    row=1)
    if V["delta5"] == 0.0:
        print(f"   {names[4]}'s true group difference is 0 -- a non-zero estimate "
              f"here would be a FALSE effect.")
    return dict(t0=t0, z1=max(z1, zsd1), seed_gap=seed_gap, cov=cov,
                ok_cov=ok_cov, t34=t34, z5=max(z5a, z5b),
                worst_distinct=int(r1.n_distinct.min()),
                worst_distinct_4d=int(r4d.n_distinct.min()),
                worst_distinct_hi=int(r5s.n_distinct.min()),
                z_hi=max(z_hi, zsd_hi), gap_hi=gap_hi,
                acc_min=float(r1.accept.min()),
                acc_min_hi=float(r5s.accept.min()), K=r1.meta["K"])


# =============================================================================
print(f"L={L} lesions, K={K} stage-1 draws each, {N_TUNE}+{N_DRAWS} stage-2 "
      f"sweeps, D={D}")
print(f"DELAY_SCALE={DELAY_SCALE!r} (tau_delay variant only)")
OUT = {}
for v in (["tau_i", "tau_delay"] if ONLY is None else [ONLY]):
    OUT[v] = run_variant(v)

print("\n" + "=" * 78)
for v, o in OUT.items():
    print(f"[{v}]")
    print(f"  T0  5-D log_p1 == prior objects      spread {o['t0']:.1e}   "
          f"{'PASS' if o['t0'] < 1e-9 else 'FAIL'}")
    print(f"  T1  flat p1, recovery                worst |z| {o['z1']:5.2f}   "
          f"{'PASS' if o['z1'] < 3 else 'FAIL'}   "
          f"(seed-to-seed {o['seed_gap']:.2f} SD)")
    print(f"  T2  coverage of 90% intervals        {np.round(o['cov'], 2)}   "
          f"{'PASS' if o['ok_cov'] else 'FAIL'}")
    for tag, (zc, zu, spread, paired, p5) in o["t34"].items():
        print(f"  T3  {tag}, correction ON          worst |z| {zc:5.2f}   "
              f"{'PASS' if zc < 3 else 'FAIL'}")
        if spread < 0.5:
            vd = f"NO POWER by construction (log p1 varies only {spread:.2f})"
        elif paired > 3:
            vd = f"PASS (paired shift {paired:.1f} SD, so T3 means something)"
        else:
            vd = (f"WEAK (paired shift only {paired:.1f} SD: T3 proves little; "
                  f"more lesions needed)")
        print(f"  T4  {tag}, correction OFF         vs truth |z| {zu:5.2f}   {vd}")
        print(f"      5th-coordinate shift {p5:.2f} SD"
              + ("  (expected ~0: flat on the log scale)" if v == "tau_i"
                 else "  (expected > 0: the logit Jacobian is not flat)"))
    print(f"  T5  group difference                 worst |z| {o['z5']:5.2f}   "
          f"{'PASS' if o['z5'] < 3 else 'FAIL'}")
    print(f"  T6  worst lesion distinct draws      4-D {o['worst_distinct_4d']} "
          f"-> 5-D {o['worst_distinct']} of {o['K']}   "
          f"(5-D acceptance min {o['acc_min']:.3f})")
    print(f"  T6b 5-D, theta_sweeps={THETA_SWEEPS_HI}             worst lesion "
          f"{o['worst_distinct_hi']} of {o['K']}, acceptance min "
          f"{o['acc_min_hi']:.3f}\n"
          f"      recovery worst |z| {o['z_hi']:5.2f}   "
          f"{'PASS' if o['z_hi'] < 3 else 'FAIL'}   "
          f"(seed-to-seed {o['gap_hi']:.2f} SD vs {o['seed_gap']:.2f} at 1 sweep)")
print("=" * 78)
print("T6 has no pass/fail: it is the measurement that says whether a 5th\n"
      "dimension starves the independence sampler. Compare the two numbers, and\n"
      "treat a large drop as the reason to raise K or theta_sweeps before using\n"
      "REAL draws -- where the posteriors are not Gaussian and will be worse.")