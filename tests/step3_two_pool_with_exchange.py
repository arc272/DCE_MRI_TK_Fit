#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:26:04 2026
@author: allisonchen

Goal: turn Step 2's off-diagonal zeros into real exchange terms
(1/tau_i), and verify the result against TWO known limits:
  - FAST exchange (tau_i -> 0): pools mix so quickly they should
    behave like ONE pool with a volume-weighted-average R1. Checked
    against the plain scalar SPGR equation from Step 1.
  - SLOW exchange (tau_i -> infinity): pools shouldn't have time to
    exchange at all -- should reduce EXACTLY to Step 2's
    no-exchange result.
If both limits check out, everything in between is trustworthy too,
since a smooth, correctly-built physical system can't suddenly do
something wrong only in the middle.

This version adds a `verbose` option to every function -- when True,
it prints X, eX, and every intermediate matrix on the way to M, so
you can watch each step happen rather than only seeing the final
number.
"""

import numpy as np
from scipy.linalg import expm, pinv


def spgr_single_pool(R1, TR, FA, M0_val, verbose=False):
    """
    Step 1: one pool, no exchange possible (there's only one pool).

    Parameters
    ----------
    R1     : relaxation rate of this pool, in 1/s (R1 = 1/T1). How
             fast this pool's magnetization would recover toward its
             own equilibrium if left completely alone.
    TR     : repetition time, in s. Time between successive RF pulses
             in the SPGR pulse sequence.
    FA     : flip angle, in RADIANS. How far each RF pulse tips the
             longitudinal magnetization toward the transverse plane.
    M0_val : this pool's equilibrium magnetization WEIGHT -- its
             share of the total observable magnetization. For a
             single, undivided pool this is 1.0 (100% -- the pool
             IS the whole voxel). For multiple pools it becomes each
             pool's volume fraction (see the 2-pool functions below).
    verbose: if True, print every intermediate matrix.

    Returns
    -------
    The steady-state SPGR signal (a single number) -- what the
    scanner would actually record for this pool, right after each
    RF pulse tips it into the transverse plane.
    """
    X = np.array([[-R1]])
    # X ("the rate matrix"): encodes how fast each pool LOSES
    # magnetization to relaxation. For one pool with no exchange,
    # it's just -R1 -- ordinary exponential decay toward equilibrium.
    # Once exchange is added (see spgr_two_pool_with_exchange), X also
    # carries off-diagonal GAIN terms for magnetization arriving from
    # a neighboring pool.

    M0 = np.array([M0_val])
    # M0 ("equilibrium magnetization vector"): how much magnetization
    # each pool would have at full equilibrium, weighted by that
    # pool's share of the total voxel. Sums to 1.0 across all pools.

    I = np.eye(1)
    # I ("identity matrix", same size as X): needed because the
    # steady-state SPGR formula involves terms like (I - eX*cos(FA)),
    # i.e. "1 minus something" generalized to matrix form -- you can't
    # subtract a matrix from a plain scalar 1, you need the identity
    # matrix to play the role "1" plays in the scalar version.

    eX = expm(TR * X)
    # eX ("evolution matrix for one TR"): the matrix generalization of
    # the scalar decay factor E1 = exp(-TR*R1) from the ordinary SPGR
    # equation. Describes how magnetization changes over one TR of
    # free relaxation (and, once exchange is added, exchange too).

    if verbose:
        print(f"    X  (rate matrix):\n{X}")
        print(f"    M0 (equilibrium weights): {M0}")
        print(f"    I  (identity):\n{I}")
        print(f"    eX = expm(TR*X):\n{eX}")

    term_inv = pinv(I - eX * np.cos(FA))
    # term_inv: the matrix version of "1/(1 - E1*cosFA)" from the
    # scalar SPGR equation's denominator. pinv (pseudo-inverse) is
    # used instead of ordinary inversion so this doesn't crash if the
    # matrix ever becomes singular/ill-conditioned at extreme
    # parameter values -- for a normal, well-behaved case, pinv gives
    # back exactly what ordinary matrix inversion would.

    term_num = I - eX
    # term_num: the matrix version of "(1 - E1)" from the scalar SPGR
    # equation's numerator -- how much magnetization is genuinely
    # "new" (recovered) since the last pulse, per pool.

    if verbose:
        print(f"    term_inv = pinv(I - eX*cos(FA)):\n{term_inv}")
        print(f"    term_num = (I - eX):\n{term_num}")

    M = term_inv @ term_num @ (np.sin(FA) * M0)
    # M: each pool's contribution to the OBSERVED (transverse, right
    # after the RF pulse tips it) signal, at steady state. For one
    # pool, M has just one entry. M.sum() below combines every pool's
    # contribution into the single number an MR scanner would record
    # (it can't tell which pool a given bit of signal came from).

    if verbose:
        print(f"    M (each pool's signal contribution): {M}")
        print(f"    M.sum() (total observed signal): {M.sum():.8f}")

    return M.sum()


def spgr_two_pool_no_exchange(R1_a, R1_b, M0_a, M0_b, TR, FA, verbose=False):
    """
    Step 2: two pools, but NO exchange between them (off-diagonal
    entries of X are exactly zero). This is the Slow eXchange Limit
    (SXL) -- water essentially never crosses between compartments.

    Parameters
    ----------
    R1_a, R1_b : relaxation rate of pool a / pool b, in 1/s. Each
                 pool relaxes completely independently since there's
                 no exchange linking them.
    M0_a, M0_b : equilibrium magnetization weight (volume fraction)
                 of pool a / pool b. Should sum to 1.0 together.
    TR, FA     : same meaning as in spgr_single_pool.
    verbose    : if True, print every intermediate matrix.
    """
    X = np.array([[-R1_a, 0.0],
                  [0.0,  -R1_b]])
    # X is now 2x2. The DIAGONAL entries (-R1_a, -R1_b) are each
    # pool's own relaxation rate, same role as the 1x1 case. The
    # OFF-DIAGONAL zeros are what make this "no exchange" -- there is
    # no term here for magnetization moving between pool a and pool
    # b, so (as verified) this mathematically decomposes into two
    # completely independent single-pool problems bundled together.

    M0 = np.array([M0_a, M0_b])
    I = np.eye(2)
    eX = expm(TR * X)

    if verbose:
        print(f"    X  (2x2 rate matrix, off-diagonal = 0 -> NO exchange):\n{X}")
        print(f"    M0 (volume fractions [pool_a, pool_b]): {M0}")
        print(f"    I  (2x2 identity):\n{I}")
        print(f"    eX = expm(TR*X):\n{eX}")
        print(f"      (note: eX is also diagonal -- pools stay fully separate)")

    term_inv = pinv(I - eX * np.cos(FA))
    term_num = I - eX

    if verbose:
        print(f"    term_inv = pinv(I - eX*cos(FA)):\n{term_inv}")
        print(f"    term_num = (I - eX):\n{term_num}")

    M = term_inv @ term_num @ (np.sin(FA) * M0)

    if verbose:
        print(f"    M (each pool's own signal contribution): {M}")
        print(f"      -> M[0] is pool a's contribution ALONE, M[1] is pool b's")
        print(f"    M.sum() (total observed signal): {M.sum():.8f}")

    return M.sum()


def spgr_two_pool_with_exchange(R1_e, R1_i, ve, tau_i, TR, FA, verbose=False):
    """
    Step 3 (the new physics): two pools WITH real water exchange
    between them -- extracellular (e) and intracellular (i) -- at
    rate 1/tau_i.

    Parameters
    ----------
    R1_e  : extracellular pool's relaxation rate, in 1/s. This is
            the pool contrast agent actually reaches, so R1_e rises
            above baseline as contrast concentration increases:
            R1_e = 1/T10 + r1*(Ce/ve)  [Ce/ve = LOCAL concentration
            density inside the compartment, not the whole-voxel
            average -- see the earlier compartment-density discussion].
    R1_i  : intracellular pool's relaxation rate, in 1/s. Contrast
            agent never crosses into cells, so this typically stays
            fixed at the tissue baseline (1/T10) throughout the scan.
    ve    : extracellular volume fraction (fraction of the voxel's
            water that lives in the extracellular compartment).
            Already a fitted kinetic parameter in TCXM/GKM -- not a
            new unknown.
    tau_i : water residence time in the INTRACELLULAR pool, in
            seconds. How long, on average, a water molecule stays in
            this pool before crossing into the extracellular pool.
            THIS is the new, genuinely free water-exchange parameter
            being fit. Small tau_i = fast exchange (approaches FXL);
            large tau_i = slow exchange (approaches SXL, Step 2).
    TR, FA : same meaning as before.
    verbose: if True, print every intermediate matrix.

    tau_e is deliberately NOT a parameter here -- see below.
    """
    vi = 1 - ve
    # vi: intracellular volume fraction. Not independently specified
    # -- it's whatever's left of the voxel once vascular/extracellular
    # fractions are accounted for (here just ve, since this is the
    # simpler 2-pool case with no separate vascular pool).

    tau_e = tau_i * ve / vi
    # tau_e: water residence time in the EXTRACELLULAR pool. NOT an
    # independently fitted parameter -- fixed by mass balance. At
    # equilibrium, the flux of water leaving e for i must exactly
    # balance the flux leaving i for e (otherwise water would pile up
    # in one pool forever), which requires ve/tau_e = vi/tau_i, i.e.
    # tau_e = tau_i*ve/vi. Once tau_i and the volume fractions are
    # fixed, tau_e is completely determined -- it is not a free knob.

    X = np.array([[-(R1_e + 1/tau_e), 1/tau_i],
                  [1/tau_e,           -(R1_i + 1/tau_i)]])
    # X, now with REAL exchange terms:
    #   diagonal:      -(R1_pool + 1/tau_pool) -- ordinary relaxation
    #                  PLUS a loss term for water leaving this pool
    #                  for its neighbor
    #   off-diagonal:   1/tau_neighboring_pool -- a gain term for
    #                  water arriving FROM that neighboring pool,
    #                  carrying its magnetization state along with it
    # This is the ONLY structural difference from Step 2's X -- the
    # zeros became real numbers, nothing else about the code changes.

    M0 = np.array([ve, vi])
    I = np.eye(2)
    eX = expm(TR * X)

    if verbose:
        print(f"    ve={ve}, vi={vi}")
        print(f"    tau_i={tau_i}  ->  tau_e = tau_i*ve/vi = {tau_e:.6g}")
        print(f"    X  (2x2 rate matrix, off-diagonal now NONZERO -> real exchange):\n{X}")
        print(f"    M0 (volume fractions [ve, vi]): {M0}")
        print(f"    I  (2x2 identity):\n{I}")
        print(f"    eX = expm(TR*X):\n{eX}")
        print(f"      (note: eX is now NOT diagonal -- off-diagonal entries mean")
        print(f"       some of pool e's signal genuinely comes from pool i, and")
        print(f"       vice versa)")

    term_inv = pinv(I - eX * np.cos(FA))
    term_num = I - eX

    if verbose:
        print(f"    term_inv = pinv(I - eX*cos(FA)):\n{term_inv}")
        print(f"    term_num = (I - eX):\n{term_num}")

    M = term_inv @ term_num @ (np.sin(FA) * M0)

    if verbose:
        print(f"    M (each pool's signal contribution, now genuinely coupled): {M}")
        print(f"    M.sum() (total observed signal): {M.sum():.8f}")

    return M.sum()


if __name__ == "__main__":
    R1_e, R1_i = 0.95, 0.7692   # extracellular has extra contrast-driven R1
    ve = 0.20
    vi = 1 - ve
    TR = 0.005
    FA = np.deg2rad(15)

    print("#" * 60)
    print("# WORKED EXAMPLE -- full intermediate matrices, one call each")
    print("#" * 60)

    print("\n--- spgr_single_pool (Step 1) ---")
    spgr_single_pool(R1_e, TR, FA, M0_val=1.0, verbose=True)

    print("\n--- spgr_two_pool_no_exchange (Step 2, SXL) ---")
    spgr_two_pool_no_exchange(R1_e, R1_i, ve, vi, TR, FA, verbose=True)

    print("\n--- spgr_two_pool_with_exchange (Step 3, tau_i=0.3s) ---")
    spgr_two_pool_with_exchange(R1_e, R1_i, ve, tau_i=0.3, TR=TR, FA=FA, verbose=True)

    print()
    print("#" * 60)
    print("# VERIFICATION CHECKS (verbose off -- just the numbers)")
    print("#" * 60)

    print("\nCheck 1: FAST exchange (tau_i tiny) should match a single")
    print("volume-weighted-average pool")
    R1_avg = ve * R1_e + vi * R1_i
    fast_target = spgr_single_pool(R1_avg, TR, FA, M0_val=1.0)
    fast_result = spgr_two_pool_with_exchange(R1_e, R1_i, ve, tau_i=1e-6, TR=TR, FA=FA)
    print(f"  volume-weighted-average single pool: {fast_target:.8f}")
    print(f"  2-pool exchange, tau_i=1e-6:         {fast_result:.8f}")
    print(f"  match: {np.isclose(fast_target, fast_result, atol=1e-6)}")

    print("\nCheck 2: SLOW exchange (tau_i huge) should match Step 2 exactly")
    slow_target = spgr_two_pool_no_exchange(R1_e, R1_i, ve, vi, TR, FA)
    slow_result = spgr_two_pool_with_exchange(R1_e, R1_i, ve, tau_i=1e6, TR=TR, FA=FA)
    print(f"  Step 2 (no exchange):        {slow_target:.8f}")
    print(f"  2-pool exchange, tau_i=1e6:   {slow_result:.8f}")
    print(f"  match: {np.isclose(slow_target, slow_result, atol=1e-8)}")

    print("\nSweep: everything in between should move smoothly from one")
    print("limit to the other")
    print(f"{'tau_i':>10} {'signal':>12} {'vs fast limit':>14} {'vs slow limit':>14}")
    for tau_i in [1e-6, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 1e6]:
        s = spgr_two_pool_with_exchange(R1_e, R1_i, ve, tau_i, TR, FA)
        print(f"{tau_i:>10.2g} {s:>12.8f} {s-fast_target:>+14.2e} {s-slow_target:>+14.2e}")