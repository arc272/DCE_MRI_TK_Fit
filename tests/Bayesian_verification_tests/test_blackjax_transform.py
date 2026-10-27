#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_blackjax_transform.py
==========================
Part A verifies the bayes_blackjax transform layer against the EXISTING
numpy prior in bayes_priors.py -- isolating the change-of-variables maths
from the sampler, so a failure there points at the transform, not BlackJAX.

Part B verifies make_log_lik_jax against the REAL jax_forward.py directly:
confirms it reproduces make_log_posterior's likelihood term exactly when
that function's prior argument is zeroed out. This is the numeric check
behind the fix for the prior-double-counting issue (make_log_posterior
fuses prior + likelihood; sample_blackjax needs likelihood only).

Checks
------
1. bound recovery      -- probed (lb, ub) match the model/relax bounds
2. round trip          -- unconstrain(constrain(z)) == z
3. support             -- constrain(z) is inside the box for extreme z
4. log-det Jacobian    -- vs jax.jacobian of constrain(), for both scales
5. change of variables -- integral of exp(log_prior + log_det) over z
                          equals 1 (i.e. the transformed prior is still
                          a normalised density)
6. density agreement   -- log_prior(z) equals prior.log_pdf(x(z)) from
                          bayes_priors, at random interior points
7. flat-prior identity -- log_prior + log_det == sum log w(1-w),
                          the cancellation noted in the module docstring
8. make_log_lik_jax     -- matches jax_forward.make_log_posterior(residual,
                          log_prior=lambda p: 0.0, noise=...) exactly, for
                          both noise='marginalized' and noise='free'

Run:
    JAX_ENABLE_X64=1 python tests/Bayesian_verification_tests/test_blackjax_transform.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.config import RelaxationConfig
from dce_mri.bayes_priors import assemble_prior
from dce_mri.bayes_blackjax import build_transform, make_log_lik_jax
from dce_mri.jax_forward import make_log_posterior

TOL = 1e-9
_fails = []


def check(name, err, tol=TOL):
    ok = bool(np.all(np.isfinite(err))) and float(np.max(np.abs(err))) < tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<46s} max err = {float(np.max(np.abs(err))):.3e}"
          f"   (tol {tol:g})")
    if not ok:
        _fails.append(name)


def main():
    relax = RelaxationConfig(fit_tau_b=False)

    for tag, choices in [("loguniform rates (default)", {}),
                         ("uniform rates", {"rates": "uniform"})]:
        print(f"\n{'='*72}\nprior: {tag}")
        prior = assemble_prior(TCXM, relax, wx_model=SSM3,
                               delay_mode="none", n_noise=0,
                               choices=choices)
        D = prior.n_params
        names = prior.param_names
        print(f"  D={D}  params={names}")
        for b in prior.blocks:
            print(f"    block {b.name:<9s} source={b.source!r:<28s} "
                  f"idx={b.indices}")

        T = build_transform(prior, D)

        # ---- 1. bound recovery ---------------------------------------
        print("\n  1. probed bounds vs model bounds")
        for b, bt in zip(sorted(prior.blocks, key=lambda b: min(b.indices)),
                         sorted(T.blocks, key=lambda b: min(b.indices))):
            lb = np.exp(np.asarray(bt.s_lb)) if bt.log_scale else np.asarray(bt.s_lb)
            ub = np.exp(np.asarray(bt.s_ub)) if bt.log_scale else np.asarray(bt.s_ub)
            print(f"     {b.name:<9s} log_scale={bt.log_scale!s:<5s} "
                  f"lb={np.array2string(lb, precision=6)} "
                  f"ub={np.array2string(ub, precision=6)}")

        rng = np.random.default_rng(0)

        # ---- 2. round trip -------------------------------------------
        z = jnp.asarray(rng.normal(size=(200, D)) * 2.5)
        x = jax.vmap(T.constrain)(z)
        z_rt = jax.vmap(T.unconstrain)(x)
        print()
        check("round trip  unconstrain(constrain(z)) == z",
              np.asarray(z_rt - z))

        # ---- 3. support ----------------------------------------------
        # |z| <= 12 is the realistic sampler range: sigmoid(12) = 1-6e-6,
        # i.e. already within 6e-6 of the bound. NUTS never needs to go
        # further, because the transformed density is finite everywhere
        # and has no reason to push into saturation.
        z_ext = jnp.asarray(rng.normal(size=(2000, D)) * 4.0)
        z_ext = jnp.clip(z_ext, -12.0, 12.0)
        x_ext = np.asarray(jax.vmap(T.constrain)(z_ext))
        lp = np.array([prior.log_pdf(xi) for xi in x_ext])
        check("constrain(z) in support for |z| <= 12",
              np.where(np.isfinite(lp), 0.0, 1.0), tol=0.5)

        # Documented edge, not a failure: at |z| >~ 37 the sigmoid
        # saturates in float64 and exp(log(ub)) can land 1-2 ulp above
        # ub, so prior.log_pdf -- which tests x > ub strictly -- reports
        # -inf. This does NOT affect sampling: the BlackJAX target is
        # log_lik + log_prior(z) + log|det J|, none of which call
        # prior.log_pdf. It only matters if constrained draws are later
        # fed back through prior.log_pdf.
        z_sat = jnp.full((1, D), 40.0)
        x_sat = np.asarray(jax.vmap(T.constrain)(z_sat))[0]
        n_out = int(np.sum(~np.isfinite([prior.log_pdf(x_sat)])))
        print(f"       (note: at z=+40, saturated draw "
              f"{'IS' if n_out else 'is not'} flagged out-of-support "
              f"by prior.log_pdf -- expected, see comment)")

        # ---- 4. log-det Jacobian vs autodiff -------------------------
        def logdet_ad(zi):
            J = jax.jacobian(T.constrain)(zi)
            return jnp.linalg.slogdet(J)[1]

        z_s = jnp.asarray(rng.normal(size=(50, D)) * 1.5)
        ld_impl = jax.vmap(T.log_det_jacobian)(z_s)
        ld_ad = jax.vmap(logdet_ad)(z_s)
        check("log|det J| vs jax.jacobian(constrain)",
              np.asarray(ld_impl - ld_ad))

        # ---- 5. change of variables integrates to 1 ------------------
        # 1-D marginal per coordinate, quadrature on z
        print()
        zz = np.linspace(-45, 45, 400001)
        for bt in T.blocks:
            for jj in range(len(bt.indices)):
                lb1 = jnp.asarray([bt.s_lb[jj]])
                ub1 = jnp.asarray([bt.s_ub[jj]])
                sub = type(bt)((0,), bt.log_scale, lb1, ub1)
                dens = np.exp(np.asarray(
                    jax.vmap(lambda t: sub.log_prior(jnp.asarray([t]))
                             + sub.log_det_jacobian(jnp.asarray([t])))(
                        jnp.asarray(zz))))
                integral = np.trapezoid(dens, zz)
                check(f"integral of transformed prior == 1  "
                      f"[{bt.log_scale and 'log' or 'lin'} coord {jj}]",
                      integral - 1.0, tol=1e-6)

        # ---- 6. density agreement with bayes_priors ------------------
        print()
        x_in = prior.sample(rng, 300)
        z_in = jnp.stack([T.unconstrain(jnp.asarray(xi)) for xi in x_in])
        lp_jax = np.asarray(jax.vmap(T.log_prior)(z_in))
        lp_np = np.array([prior.log_pdf(xi) for xi in x_in])

        # Compared UP TO AN ADDITIVE CONSTANT, which is the only thing
        # MCMC can see. An exact match is deliberately not required here
        # because bayes_priors._loguniform_funcs has a normalisation bug
        # (-(lub-llb) where it should be -log(lub-llb)); this transform
        # uses the correctly normalised form. Once that is fixed, the
        # offset below goes to zero and this check can be tightened to
        # an exact comparison.
        d = lp_jax - lp_np
        check("log_prior(z) == bayes_priors.log_pdf(x(z))  [up to const]",
              d - np.mean(d))
        if abs(float(np.mean(d))) > 1e-9:
            print(f"       (constant offset = {float(np.mean(d)):+.6f} nats "
                  f"-- bayes_priors log-uniform normalisation, "
                  f"sampling-irrelevant)")

        # ---- 7. flat-prior cancellation identity ---------------------
        def lhs(zi):
            return T.log_prior(zi) + T.log_det_jacobian(zi)

        def rhs(zi):
            return jnp.sum(jax.nn.log_sigmoid(zi) + jax.nn.log_sigmoid(-zi))

        z_c = jnp.asarray(rng.normal(size=(200, D)) * 3.0)
        check("log_prior + log|det J| == sum log w(1-w)",
              np.asarray(jax.vmap(lhs)(z_c) - jax.vmap(rhs)(z_c)))

    # ---- Part B: make_log_lik_jax vs. the real make_log_posterior -----
    print(f"\n{'='*72}\nmake_log_lik_jax vs. jax_forward.make_log_posterior "
         "(prior zeroed)")
    rng = np.random.default_rng(1)
    zero_prior = lambda p: 0.0

    def fake_residual(p_aug):
        # a real (T,)-shaped, nonlinear stand-in residual -- not a
        # trivial check, but not tied to the actual forward model either
        w = jnp.arange(1, 37, dtype=jnp.float64)
        return jnp.sin(w * jnp.sum(p_aug ** 2)) * 0.1 + jnp.sum(p_aug)

    for noise, D in [("marginalized", 5), ("free", 6)]:
        log_post_ref = make_log_posterior(fake_residual, zero_prior, noise=noise)
        log_lik_new = make_log_lik_jax(fake_residual, noise=noise)
        pts = rng.normal(scale=0.5, size=(500, D))
        if noise == "free":
            pts[:, -1] = np.abs(pts[:, -1]) + 0.05
        ref = np.array([float(log_post_ref(jnp.asarray(p))) for p in pts])
        new = np.array([float(log_lik_new(jnp.asarray(p))) for p in pts])
        check(f"make_log_lik_jax == make_log_posterior(prior=0)  "
              f"[noise={noise}]", ref - new)

    print(f"\n{'='*72}")
    if _fails:
        print(f"FAILED ({len(_fails)}): " + ", ".join(_fails))
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
