#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:35:38 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_single_pool.py

Goal: write the SPGR signal equation using the "matrix" machinery
(expm, matrix inverse) for just ONE pool, and confirm it gives
EXACTLY the same answer as the ordinary scalar SPGR equation you
already know from concentration_to_signal.

Nothing new physically happens here -- this step is purely about
getting comfortable with the shape of the matrix code before any
real coupling (Step 3) is introduced.
"""

import numpy as np
from scipy.linalg import expm, pinv

# ── the version you already know ──────────────────────────────────
def spgr_scalar(R1, TR, FA):
    """Standard scalar SPGR signal equation."""
    E1 = np.exp(-TR * R1)
    return (1 - E1) * np.sin(FA) / (1 - E1 * np.cos(FA))


# ── the SAME equation, written using matrix machinery ─────────────
def spgr_matrix(R1, TR, FA):
    """
    Same physics as spgr_scalar, but written as a 1x1 'matrix' problem.
    This is the exact skeleton the multi-pool version will reuse --
    only the SIZE of X will change in later steps, not its structure.
    """
    X = np.array([[-R1]])              # 1x1 rate matrix -- just -R1
    M0 = np.array([1.0])               # equilibrium magnetization (one pool -> weight 1)
    I = np.eye(1)

    eX = expm(TR * X)                  # matrix generalization of exp(-TR*R1)
    M = pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)
    return M.sum()                     # sum across pools (just one, here)


if __name__ == "__main__":
    # a handful of realistic R1/TR/FA combinations
    cases = [
        dict(R1=0.7692, TR=0.005, FA=np.deg2rad(15)),   # T1~1.3s tissue, typical fast SPGR
        dict(R1=1.5,    TR=0.003, FA=np.deg2rad(10)),
        dict(R1=0.5,    TR=0.010, FA=np.deg2rad(25)),
    
    ]

    print(f"{'R1':>6} {'TR':>6} {'FA(deg)':>8}   {'scalar':>12} {'matrix':>12} {'match?':>7}")
    for c in cases:
        s_scalar = spgr_scalar(**c)
        s_matrix = spgr_matrix(**c)
        match = np.isclose(s_scalar, s_matrix)
        print(f"{c['R1']:>6.3f} {c['TR']:>6.3f} {np.rad2deg(c['FA']):>8.1f}   "
              f"{s_scalar:>12.8f} {s_matrix:>12.8f} {'YES' if match else 'NO':>7}")

    print()
    print("If every row says YES, the matrix skeleton (X, expm, pinv, the")
    print("[I - eX*cos][I - eX]*sin structure) is correctly wired for the")
    print("1-pool case. Step 2 reuses this exact skeleton, just with a")
    print("bigger X.")