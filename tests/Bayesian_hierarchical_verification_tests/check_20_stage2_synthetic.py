
#!/usr/bin/env python3
from __future__ import annotations
#!/usr/bin/env python3
"""
check_20_stage1_budget.py
=========================

Decides whether a stage-1 run needs MORE draws before stage 2, by measuring
the two things that actually set the usable sample size per lesion:

  1. stage-1 ESS            -- how many of the saved draws are independent
  2. Kish weight ESS        -- how many of THOSE survive the stage-2
                               reweighting w = p(theta|mu,Sigma) / p_1(theta)

check_18 (V4) showed the second factor decays geometrically in dimension and
is invisible to R-hat and ESS.  Usable draws per lesion is roughly the
product of the two, so a lesion can pass every stage-1 diagnostic and still
starve stage 2.

The decisive question this answers is NOT "what is the Kish ESS" but
"DOES IT GROW WITH K".  The script recomputes the Kish ESS on nested
thinned subsets:

  * Kish ESS roughly proportional to K  -> the weights are well behaved,
    more stage-1 draws buy proportionally more usable draws.  Refit.
  * Kish ESS flattens as K grows        -> a few draws dominate the weights,
    and more draws of the same kind will NOT help.  Refitting is wasted GPU
    time; the stage-1 PROPOSAL has to change (narrower p_1, a different
    parameterisation, or a continuous surrogate).

Settings are at the top.  Point RUN_DIR at a completed run.  With no run
directory present it falls back to a synthetic self-test so the code path
can be checked anywhere.

Run:  PYTHONPATH=src python check_20_stage1_budget.py
"""



import glob
import os

import numpy as np

# ---------------------------------------------------------------- settings
RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
           "run_20261021_160427")
PARAMS = ("ve", "vp", "fp", "ps", "tau_i")     # population-scale coordinates
# how each constrained parameter maps to the population scale
SCALE = {"ve": "logit", "vp": "logit", "fp": "log", "ps": "log",
         "tau_i": "log", "tau_delay": "logit25"}
# Subset sizes to test the growth of the Kish ESS.  Thinning is nested.
K_GRID = (750, 1500, 3000, 6000, 12000, 24000)
MAX_LESIONS = None          # None = all; set a number for a quick look
WORST_N = 12                # how many worst lesions to list
QC_ESS_MIN = 400            # your stage-1 QC gate, for flagging
# A trial population (mu, Sigma).  None = estimate it from the lesion
# medians, which is the right first pass when no stage-2 fit exists yet.
TRIAL_MU = None
TRIAL_SD = None


# ---------------------------------------------------------------- helpers
def to_theta(x, name):
    """Constrained parameter -> population scale."""
    s = SCALE[name]
    if s == "logit":
        return np.log(x) - np.log1p(-x)
    if s == "log":
        return np.log(x)
    if s == "logit25":
        u = x / 25.0
        return np.log(u) - np.log1p(-u)
    raise ValueError(f"unknown scale {s!r} for {name}")


def ess_bulk_tail(chains):
    """(bulk, tail) ESS for one parameter, chains shape (n_chain, n_draw)."""
    try:
        import arviz as az
        d = az.convert_to_dataset(chains[:, :, None], coords={}, dims={})
        return (float(az.ess(chains, method="bulk")),
                float(az.ess(chains, method="tail")))
    except Exception:
        # Geyer initial-positive-sequence fallback, pooled over chains
        n_c, n_d = chains.shape
        taus = []
        for c in range(n_c):
            x = chains[c] - chains[c].mean()
            f = np.fft.rfft(x, 2 * n_d)
            ac = np.fft.irfft(f * np.conjugate(f))[:n_d].real
            if ac[0] <= 0:
                taus.append(1.0)
                continue
            ac /= ac[0]
            s, t = 0.0, 1
            while t + 1 < n_d:
                g = ac[t] + ac[t + 1]
                if g <= 0:
                    break
                s += g
                t += 2
            taus.append(1.0 + 2.0 * s)
        tau = float(np.mean(taus))
        return (n_c * n_d / max(tau, 1e-12),) * 2


def kish(logw):
    logw = logw - logw.max()
    w = np.exp(logw)
    return float(w.sum() ** 2 / (w ** 2).sum())


def weight_ess_vs_K(theta, mu, chol_inv, k_grid):
    """
    Kish ESS of w = N(theta; mu, Sigma) on nested thinned subsets.
    p_1 is flat on the population scale here (log p_1 = const), which is
    the production case for rates/volfrac/tau; pass log_p1 in if not.
    """
    n = len(theta)
    out = []
    for K in k_grid:
        if K > n:
            out.append((K, np.nan))
            continue
        idx = np.linspace(0, n - 1, K).astype(int)      # nested thinning
        z = (theta[idx] - mu) @ chol_inv.T
        out.append((K, kish(-0.5 * np.einsum("nd,nd->n", z, z))))
    return out


# ---------------------------------------------------------------- loading
def load_run(run_dir, params, max_lesions=None):
    """-> dict sid -> (n_chain, n_draw, D) constrained draws for `params`."""
    files = sorted(glob.glob(os.path.join(run_dir, "*_draws.npz")))
    if max_lesions:
        files = files[:max_lesions]
    out = {}
    for f in files:
        sid = os.path.basename(f).replace("_draws.npz", "")
        with np.load(f, allow_pickle=True) as z:
            names = [str(s) for s in z["param_names"]]
            ch = z["chains"]
            miss = [p for p in params if p not in names]
            if miss:
                print(f"   {sid}: missing {miss}; has {names} -- skipped")
                continue
            cols = [names.index(p) for p in params]
            out[sid] = ch[:, :, cols]
    return out


def synthetic_run(n_lesions=12, n_chain=16, n_draw=4000, seed=0):
    """Self-test data in the same shape, with a deliberately wide lesion."""
    rng = np.random.default_rng(seed)
    mu = np.array([-0.847, -2.944, -4.605, -5.298, -1.609])
    sd = np.array([0.55, 0.70, 0.60, 0.65, 0.80])
    out = {}
    for i in range(n_lesions):
        th = mu + sd * rng.standard_normal(5)
        # lesion 0 is the pathological case: stage-1 much wider than the
        # population, which is what starves stage 2
        v = sd * (4.0 if i == 0 else rng.uniform(0.3, 1.2))
        # correlated draws, so stage-1 ESS < raw count, as in a real chain
        d = np.empty((n_chain, n_draw, 5))
        x = th + v * rng.standard_normal((n_chain, 5))
        rho = 0.9
        for t in range(n_draw):
            x = th + rho * (x - th) + v * np.sqrt(1 - rho ** 2) * \
                rng.standard_normal((n_chain, 5))
            d[:, t, :] = x
        out[f"SYN{i:03d}"] = d
    return out, True


# ---------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("check_20  stage-1 budget: do you need more draws, and would they help?")
    print("=" * 78)

    synthetic = False
    if os.path.isdir(RUN_DIR):
        print(f"run: {RUN_DIR}")
        draws_c = load_run(RUN_DIR, PARAMS, MAX_LESIONS)
        if not draws_c:
            print("   no usable *_draws.npz found; falling back to self-test")
            draws_c, synthetic = synthetic_run()
    else:
        print(f"RUN_DIR not present here ({RUN_DIR})")
        print("running the SYNTHETIC SELF-TEST instead -- numbers below are "
              "not your data")
        draws_c, synthetic = synthetic_run()

    sids = list(draws_c)
    n_chain, n_draw, D = draws_c[sids[0]].shape
    print(f"{len(sids)} lesions, {n_chain} chains x {n_draw} draws = "
          f"{n_chain*n_draw} raw draws each, D = {D}")
    print()

    # ---- stage-1 ESS ------------------------------------------------------
    print("1.  STAGE-1 ESS (how many of the raw draws are independent)")
    print(f"    {'lesion':<10}" + "".join(f"{p:>12}" for p in PARAMS)
          + f"{'min':>9}{'thin to':>9}")
    theta_flat, ess_min_all, ess_per_param = {}, {}, {}
    for sid in sids:
        ch = draws_c[sid]
        row, th = [], []
        for j, p in enumerate(PARAMS):
            t = to_theta(ch[:, :, j], p) if not synthetic else ch[:, :, j]
            th.append(t.reshape(-1))
            row.append(ess_bulk_tail(t)[0])
        theta_flat[sid] = np.column_stack(th)
        ess_min_all[sid] = min(row)
        ess_per_param[sid] = row
    e = np.array([ess_min_all[s] for s in sids])
    pp = np.array([ess_per_param[s] for s in sids])          # (L, D)
    print(f"    {'':<10}" + "".join(f"{p:>11}" for p in PARAMS) + f"{'min':>10}")
    for lbl, q in (("min", 0), ("5th pct", 5), ("median", 50), ("max", 100)):
        print(f"    {lbl:<10}"
              + "".join(f"{np.percentile(pp[:, j], q):>11.0f}" for j in range(D))
              + f"{np.percentile(e, q):>10.0f}")
    bad = sorted(sids, key=lambda s: ess_min_all[s])[:WORST_N]
    print(f"    worst {len(bad)} lesions by min-ESS:")
    for s in bad:
        flag = "  << below ESS_MIN" if ess_min_all[s] < QC_ESS_MIN else ""
        print(f"      {s:<10} {ess_min_all[s]:>8.0f}"
              + "  " + " ".join(f"{v:.0f}" for v in ess_per_param[s]) + flag)
    n_bad = int((e < QC_ESS_MIN).sum())
    print(f"    {n_bad} of {len(sids)} lesions fall below your QC gate "
          f"(ESS_MIN = {QC_ESS_MIN}) on the population scale")
    print(f"    -> thin each lesion to about its own min-ESS; saving more than")
    print(f"       that many draws does not add independent information.")
    print()

    # ---- trial population -------------------------------------------------
    med = np.array([np.median(theta_flat[s], 0) for s in sids])
    mu = np.array(TRIAL_MU, float) if TRIAL_MU is not None else med.mean(0)
    if TRIAL_SD is not None:
        sd = np.array(TRIAL_SD, float)
    else:
        # lesion-median spread OVERSTATES the population SD (it includes
        # estimation noise), so this is a conservative, wide trial Sigma
        sd = med.std(0, ddof=1)
    Sigma = np.diag(sd ** 2)
    chol_inv = np.linalg.inv(np.linalg.cholesky(Sigma))
    print("2.  TRIAL POPULATION used for the weights "
          f"({'given' if TRIAL_MU is not None else 'from lesion medians'})")
    print("    mu " + "  ".join(f"{v:+.3f}" for v in mu))
    print("    sd " + "  ".join(f"{v:.3f}" for v in sd))
    print("    NOTE: a median-based SD is too WIDE, so the weights below are")
    print("    optimistic.  Re-run with TRIAL_SD from a real stage-2 fit.")
    print()

    # ---- weight ESS vs K --------------------------------------------------
    print("3.  KISH WEIGHT ESS vs K  (does more K buy usable draws?)")
    growths, kish_tab, usable = [], {}, {}
    for sid in sids:
        th = theta_flat[sid]
        rows = weight_ess_vs_K(th, mu, chol_inv, K_GRID)
        kish_tab[sid] = rows
        # the honest current number: Kish ESS at this lesion's OWN stage-1 ESS,
        # which is the largest K at which the draws are still ~independent
        k_hon = int(max(50, min(ess_min_all[sid], len(th))))
        usable[sid] = weight_ess_vs_K(th, mu, chol_inv, [k_hon])[0][1]
        # restrict the growth slope to K values at or below the stage-1 ESS
        ok = [(k, v) for k, v in rows
              if np.isfinite(v) and k <= ess_min_all[sid]]
        if len(ok) >= 2:
            (k0, v0), (k1, v1) = ok[0], ok[-1]
            g = np.log(max(v1, 1e-9) / max(v0, 1e-9)) / np.log(k1 / k0)
        else:
            g = np.nan
        growths.append(g)
    g = np.array(growths)

    print(f"    {'':<10}" + "".join(f"{'K='+str(k):>10}" for k in K_GRID))
    for lbl, q in (("min", 0), ("5th pct", 5), ("median", 50), ("max", 100)):
        cells = ""
        for j, k in enumerate(K_GRID):
            col = np.array([kish_tab[s][j][1] for s in sids], float)
            col = col[np.isfinite(col)]
            cells += f"{np.percentile(col, q):>10.1f}" if len(col) else f"{'-':>10}"
        print(f"    {lbl:<10}{cells}")

    u = np.array([usable[s] for s in sids], float)
    print()
    print("    USABLE DRAWS NOW, per lesion: Kish ESS evaluated at that")
    print("    lesion's own stage-1 ESS (the largest honest K today)")
    print(f"      min {np.nanmin(u):.1f}   5th pct {np.nanpercentile(u, 5):.1f}"
          f"   median {np.nanmedian(u):.1f}   max {np.nanmax(u):.1f}")
    worst_u = sorted(sids, key=lambda s: usable[s])[:WORST_N]
    print(f"    worst {len(worst_u)} lesions by usable draws:")
    print(f"      {'lesion':<10}{'stage-1 ESS':>12}{'usable':>9}{'growth':>8}")
    for s in worst_u:
        print(f"      {s:<10}{ess_min_all[s]:>12.0f}{usable[s]:>9.1f}"
              f"{growths[sids.index(s)]:>8.2f}")
    print()
    print(f"    growth exponent across lesions: min {np.nanmin(g):.2f}  "
          f"median {np.nanmedian(g):.2f}  max {np.nanmax(g):.2f}")
    print("    1.0 = Kish ESS proportional to K, so more draws help "
          "proportionally")
    print("    0.0 = saturated, so more draws of the same kind do NOT help")
    print("    the slope is fitted ONLY over K at or below each lesion's own")
    print("    stage-1 ESS, since Kish assumes independent draws; larger K in")
    print("    the table above is therefore optimistic.")
    print()

    # ---- verdict ----------------------------------------------------------
    print("=" * 78)
    worst = int(np.nanargmin(g))
    print(f"worst-growing lesion: {sids[worst]} (exponent {g[worst]:.2f})")
    if np.nanmedian(g) > 0.75:
        print("VERDICT: the weights are well behaved -- raising K buys usable")
        print("draws roughly in proportion.  A longer stage 1 is worth it.")
    elif np.nanmedian(g) > 0.4:
        print("VERDICT: mixed.  Raising K helps, but with diminishing returns;")
        print("check which lesions saturate and whether they drive your result.")
    else:
        print("VERDICT: saturated.  More draws of the same kind will NOT fix")
        print("this -- the stage-1 proposal itself has to change (narrower")
        print("p_1, different parameterisation, or a continuous surrogate).")
    print("=" * 78)


if __name__ == "__main__":
    main()