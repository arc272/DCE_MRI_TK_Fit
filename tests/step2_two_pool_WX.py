#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:15:15 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_two_pool_no_exchange.py

Goal: extend Step 1's machinery to TWO pools, but with NO coupling
between them yet (off-diagonal entries of X are zero). Confirm this
reduces exactly to running Step 1 independently for each pool and
combining the results -- i.e. confirm the matrix machinery doesn't
introduce anything new by itself; only real off-diagonal coupling
(Step 3) will.
"""

import numpy as np
from scipy.linalg import expm, pinv


# ── Step 1's function, unchanged, reused as the "ground truth" for
#    what one isolated pool's contribution should look like ──
def spgr_single_pool(R1, TR, FA, M0_val):
    X = np.array([[-R1]])
    M0 = np.array([M0_val])
    I = np.eye(1)
    eX = expm(TR * X)
    M = pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)
    return M.sum()


# ── the new 2-pool version, X diagonal (NO exchange terms yet) ──
def spgr_two_pool_no_exchange(R1_a, R1_b, M0_a, M0_b, TR, FA):
    X = np.array([[-R1_a, 0.0],      # <-- off-diagonal entries are 0:
                  [0.0,  -R1_b]])    #     pools cannot influence each other
    M0 = np.array([M0_a, M0_b])
    I = np.eye(2)
    eX = expm(TR * X)
    M = pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)
    return M.sum(), M   # return the individual pool contributions too


if __name__ == "__main__":
    TR = 0.005
    FA = np.deg2rad(15)

    cases = [
        dict(R1_a=0.9, R1_b=0.7, M0_a=0.4, M0_b=0.6),
        dict(R1_a=1.5, R1_b=0.5, M0_a=0.1, M0_b=0.9),
        dict(R1_a=0.6, R1_b=0.6, M0_a=0.5, M0_b=0.5),   # identical pools
    ]

    print(f"{'R1_a':>6} {'R1_b':>6} {'M0_a':>6} {'M0_b':>6}   "
          f"{'2-pool total':>13} {'sum of 2 Step-1 calls':>22} {'match?':>7}")
    for c in cases:
        total, M = spgr_two_pool_no_exchange(**c, TR=TR, FA=FA)

        # ground truth: run Step 1 SEPARATELY for each pool, add the results
        s_a = spgr_single_pool(c["R1_a"], TR, FA, c["M0_a"])
        s_b = spgr_single_pool(c["R1_b"], TR, FA, c["M0_b"])
        independent_sum = s_a + s_b

        match = np.isclose(total, independent_sum)
        print(f"{c['R1_a']:>6.2f} {c['R1_b']:>6.2f} {c['M0_a']:>6.2f} {c['M0_b']:>6.2f}   "
              f"{total:>13.8f} {independent_sum:>22.8f} {'YES' if match else 'NO':>7}")

        # bonus check: does M[0] (from the 2-pool call) equal s_a exactly,
        # and M[1] equal s_b exactly? Not just the SUM matching by luck.
        assert np.isclose(M[0], s_a), "pool A's own contribution should match Step 1 exactly"
        assert np.isclose(M[1], s_b), "pool B's own contribution should match Step 1 exactly"

    print()
    print("Every row should say YES, AND each pool's own contribution (not just")
    print("the total) should match its Step-1 equivalent exactly. If so, the")
    print("2x2 machinery with zero off-diagonal is doing nothing more than")
    print("running two independent Step-1 problems side by side.")
    print()
    print("Step 3 will change ONLY the off-diagonal entries -- everything else")
    print("in this code stays the same.")