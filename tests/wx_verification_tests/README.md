# Water-exchange verification suite

Verification for the SSM3 water-exchange integration (WX phases 4-8).
Covers every item in the WX-8 verification plan plus WX-6e's
delay/exchange questions, and re-runs the earlier ad-hoc checks from
this work as permanent regression tests.

## Files

```
wx_verify_common.py        shared infra: results dir, figures, configs,
                           INDEPENDENT reference implementations
wx_verify_a_limits.py      Tier A -- limits, bounds, conventions, guardrails
wx_verify_b_recovery.py    Tier B -- synthetic recovery, optimizer, LHS budget
wx_verify_c_delay.py       Tier C -- delay + exchange (WX-6e)
wx_verify_d_realdata.py    Tier D -- real-data sanity
run_all_wx_verification.py master runner
```

Put all six in the same directory and run from there.

## Usage

```bash
python run_all_wx_verification.py                # everything
python run_all_wx_verification.py --quick        # fewer reps, faster
python run_all_wx_verification.py --tiers ab     # only A and B
python run_all_wx_verification.py --no-realdata  # skip D
```

Individual tiers also run standalone:

```bash
python wx_verify_a_limits.py
python wx_verify_b_recovery.py --quick
```

Output goes to `~/wx_verification_results/<timestamp>/`, timestamped so
reruns don't clobber earlier figures.

## Coverage map

| Check | WX-8 item | Where |
|---|---|---|
| FXL-limit consistency | #1 | A1 |
| SXL-limit consistency | #1 | A2 |
| FXL >= signal >= SXL, across TR/FA/R1 grid | #2 | A3 |
| Compartment-sum regression | #3 | A5 |
| Ce/ve density + hematocrit convention | #4 | A4 |
| Synthetic SSM recovery | #5 | B1, B2 |
| Guardrail rules (all five) | #6 | A6 |
| TRF vs Nelder-Mead for tau_i | #7 | B3 |
| Recovery vs where true tau_i sits | #7 | B2 |
| n_lhs budget for augmented vector | #8 | B4 |
| Real-data / literature sanity | #9 | D1, D3 |
| tau_b separability from delay | WX-6e | C1 |
| Joint (tau_delay, tau_i) recovery | WX-6e | C2 |
| Sequential vs joint fitting | WX-6e | C3 |
| Joint-vector LHS coverage | WX-6e | C4 |
| ROI vs voxelwise tau_i | — | D2 |
| FXL vs SSM3 on real curves | — | D4 |

## Runtime

Tier A is seconds. Tiers B and C are minutes each — they run many
noise realisations through the full LHS+TRF fitter, and each SSM3
forward evaluation loops `expm` over timepoints. `--quick` cuts
repetitions roughly in half. Tier D depends on how many voxels you
screen.

If Tiers B/C are too slow to iterate on, the stacked-`expm`
vectorisation (build `X` as `(T,3,3)`, one `expm` call, `np.linalg.solve`
instead of `pinv`) gives roughly 10x on the forward model and was
verified to agree with the loop to ~1e-16. That optimisation is not
applied in `water_exchange.py` as of this suite being written.

## How to read the results

**Tier A failures are bugs.** These are analytical identities and
structural guarantees. If A1-A6 don't all pass, something in
`water_exchange.py` or `fitting.py` is structurally wrong and nothing
downstream is meaningful.

**Tier B/C failures are usually findings, not defects.** A wide tau_i
recovery distribution means "tau_i is not well recoverable at this
SNR" — that is a result to report, not a bug to fix. The scripts print
detailed distributions alongside pass/fail for exactly this reason;
read the numbers, not just the verdict.

**Tier D has no ground truth.** Its checks are plausibility flags
against a literature orientation range. A tau_i outside that range is
something to investigate, not proof of error.

The runner exits non-zero only if Tier A fails, for this reason.

## Known context from earlier work

Things already established before this suite existed, which inform
what the results should look like:

- The `Ce/ve`, `Cv/vb` density division is load-bearing — removing it
  breaks the fast-exchange limit by a large margin (A4 asserts this
  explicitly rather than trusting the docstring).
- `tau_b` failed identifiability screening in every configuration
  tested (ratios 0.06x-0.76x against the noise floor), which is why
  `fit_tau_b` defaults to `False`.
- `tau_i` recovers perfectly at zero noise but scatters across an
  order of magnitude at realistic SNR — B1 quantifies this properly.
- AIF delay is an exact rigid time shift; `tau_i`'s effect is
  amplitude, not timing. `tau_b` was never tested this way — C1 is
  the check that closes that gap.
- `ps` sits on a flat `fp`-`ps` ridge, so its LHS coverage collapses
  in the top-n_top cut. Expect C4 to show this; it is a known
  property of TCXM at coarse temporal resolution, not a sampling bug.

## Figures

Every test writes a PNG. Suitable for a progress update as-is —
each figure is self-titled with what it shows and what the reference
or threshold is.
