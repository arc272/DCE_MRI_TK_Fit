#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_wx_verification.py
==========================
Master runner for the water-exchange verification suite.

Usage
-----
    python run_all_wx_verification.py               # everything
    python run_all_wx_verification.py --quick       # fewer reps, faster
    python run_all_wx_verification.py --tiers ab    # only tiers A and B
    python run_all_wx_verification.py --no-realdata # skip Tier D

Tiers
-----
  A  analytical limits, structural bounds, conventions, guardrails
     (WX-8 #1,2,3,4,6)  -- seconds, deterministic, must all pass
  B  synthetic recovery, optimizer choice, LHS budget
     (WX-8 #5,7,8)      -- minutes, statistical
  C  delay + exchange interaction (WX-6e)
                         -- minutes, statistical
  D  real-data sanity   (WX-8 #9)  -- needs FMB data on disk

Everything writes figures and a summary into
    ~/wx_verification_results/<timestamp>/

Interpreting failures
---------------------
Tier A failures are BUGS -- something structural is wrong.
Tier B/C failures are usually FINDINGS -- e.g. "tau_i is not
recoverable at this SNR" is a result worth reporting, not a defect.
Tier D has no ground truth at all; its checks are plausibility flags.
Read the printed report, not just the pass/fail counts.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from wx_verify_common import RESULTS_DIR, init_results_dir


def main():
    args = sys.argv[1:]
    quick = "--quick" in args
    no_real = "--no-realdata" in args

    tiers = "abcd"
    if "--tiers" in args:
        tiers = args[args.index("--tiers") + 1].lower()
    if no_real:
        tiers = tiers.replace("d", "")

    init_results_dir()
    print()
    print("#" * 72)
    print("#  WATER-EXCHANGE VERIFICATION SUITE")
    print(f"#  tiers: {', '.join(tiers.upper())}"
          + ("   [QUICK MODE]" if quick else ""))
    print(f"#  results -> {RESULTS_DIR}")
    print("#" * 72)

    outcomes = {}
    t_start = time.perf_counter()

    if "a" in tiers:
        try:
            from wx_verify_a_limits import run_tier_a
            t0 = time.perf_counter()
            outcomes["A"] = run_tier_a()
            print(f"  [tier A took {time.perf_counter()-t0:.1f}s]")
        except Exception:
            traceback.print_exc()
            outcomes["A"] = False

    if "b" in tiers:
        try:
            from wx_verify_b_recovery import run_tier_b
            t0 = time.perf_counter()
            outcomes["B"] = run_tier_b(quick=quick)
            print(f"  [tier B took {time.perf_counter()-t0:.1f}s]")
        except Exception:
            traceback.print_exc()
            outcomes["B"] = False

    if "c" in tiers:
        try:
            from wx_verify_c_delay import run_tier_c
            t0 = time.perf_counter()
            outcomes["C"] = run_tier_c(quick=quick)
            print(f"  [tier C took {time.perf_counter()-t0:.1f}s]")
        except Exception:
            traceback.print_exc()
            outcomes["C"] = False

    if "d" in tiers:
        try:
            from wx_verify_d_realdata import run_tier_d
            t0 = time.perf_counter()
            outcomes["D"] = run_tier_d(quick=quick)
            print(f"  [tier D took {time.perf_counter()-t0:.1f}s]")
        except Exception:
            traceback.print_exc()
            outcomes["D"] = False

    elapsed = time.perf_counter() - t_start

    print()
    print("#" * 72)
    print("#  SUITE SUMMARY")
    print("#" * 72)
    for tier in "ABCD":
        if tier in outcomes:
            status = "ALL PASSED" if outcomes[tier] else "SOME FAILED"
            print(f"  Tier {tier}: {status}")
    print(f"\n  total time: {elapsed/60:.1f} min")
    print(f"  figures and output: {RESULTS_DIR}")

    figs = sorted(Path(RESULTS_DIR).rglob("*.png"))
    if figs:
        print(f"\n  {len(figs)} figures written:")
        for f in figs:
            print(f"    {f.relative_to(RESULTS_DIR)}")

    print()
    print("  Reminder on interpretation:")
    print("    Tier A failure  -> structural bug, fix before trusting anything")
    print("    Tier B/C failure-> often a FINDING about identifiability,")
    print("                       not a defect; read the printed detail")
    print("    Tier D          -> no ground truth; plausibility flags only")
    print("#" * 72)

    # exit non-zero only if Tier A failed -- B/C/D "failures" are frequently
    # legitimate scientific findings and should not break a CI-style run
    return 0 if outcomes.get("A", True) else 1


if __name__ == "__main__":
    sys.exit(main())
