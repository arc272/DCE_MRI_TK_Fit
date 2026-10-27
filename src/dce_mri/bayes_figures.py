# -*- coding: utf-8 -*-
"""
bayes_figures.py
================
Figures for Bayesian posterior estimation.

The headline figure is plot_posterior_vs_prior(): for each parameter it
overlays the PRIOR and the POSTERIOR on the same axes, both normalised
so the area under each curve is 1. That normalisation is what makes the
comparison meaningful -- prior and posterior are different distributions
over the same axis, so only densities (not raw counts) are comparable.

Reading the figure
------------------
    posterior much narrower than prior  -> data constrained it
    posterior sitting on top of prior   -> data learned ~nothing
                                           (honest non-identifiability)
    posterior narrow but far from truth -> NOT identifiability; check
                                           R-hat first, this is the
                                           signature of a stuck sampler

All functions accept `save_path`; when given, the figure is written and
the Figure object still returned so it can be further customised.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# House style
# =============================================================================

C_PRIOR  = "#9E9E9E"
C_POST   = "#B22234"
C_TRUTH  = "#1B4F72"
C_MEDIAN = "#E07A2F"
C_CHAIN  = ["#B22234", "#1B4F72", "#2E7D32", "#8E44AD",
            "#E07A2F", "#00838F", "#6D4C41", "#AD1457"]


def _finish(fig, save_path, dpi=150):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  saved: {save_path}")
    return fig


def _grid(n, ncols=3, panel=(3.5, 2.7)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes, nrows


# =============================================================================
# THE headline figure: posterior vs prior
# =============================================================================
def plot_posterior_vs_prior(
    result,
    prior,
    truth:      Optional[dict] = None,
    params:     Optional[Sequence[str]] = None,
    n_prior:    int = 40000,
    bins:       int = 60,
    log_params: Sequence[str] = ("fp", "ps", "tau_i", "tau_b", "sigma", "Ktrans") , #"fp", "ps", "tau_i", "tau_b", "sigma", "Ktrans"),
    title:      Optional[str] = None,
    save_path:  Optional[str] = None,
    seed:       int = 0,
):
    """
    Overlay prior and posterior densities, per parameter.

    For parameters in `log_params`, both the prior and posterior are
    transformed to log10(parameter) before histogramming.

    This means the plotted density is:

        p(log10(parameter))

    rather than:

        p(parameter)

    Consequently, a log-uniform prior appears flat, as expected.

    IMPORTANT:
    The underlying x-coordinate is log10(parameter), but the x-axis
    tick labels are formatted back into the ORIGINAL parameter units.
    For example, the underlying coordinates

        -4, -3, -2, -1

    are displayed as

        0.0001, 0.001, 0.01, 0.1

    so the plot retains logarithmic spacing while remaining
    scientifically interpretable.

    For parameters not in `log_params`, the original parameter values
    are plotted on a linear axis.

    truth      : optional {param_name: true_value}; drawn as a vertical line
    log_params : parameters plotted in log10(parameter) coordinate space
    """

    from matplotlib.ticker import FuncFormatter, FixedLocator

    # ------------------------------------------------------------------
    # Helper: format a log10 coordinate using the ORIGINAL parameter
    # value rather than displaying the log10 value itself.
    # ------------------------------------------------------------------
    def format_log_tick(x, pos=None):
        value = 10.0 ** x

        if not np.isfinite(value):
            return ""

        # Use compact decimal notation for values that are easy to read.
        if 0.001 <= value < 1000:
            return f"{value:g}"

        # Use scientific notation for very small / very large values.
        return f"{value:.1e}"

    # ------------------------------------------------------------------
    # Sample prior
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    prior_draws = prior.sample(rng, n_prior)

    names = (
        list(params)
        if params is not None
        else list(result.param_names)
    )

    fig, axes, _ = _grid(len(names))

    # ------------------------------------------------------------------
    # Loop over parameters
    # ------------------------------------------------------------------
    for k, nm in enumerate(names):

        ax = axes[k]

        # ==============================================================
        # Posterior samples
        # ==============================================================
        j_post = result.param_names.index(nm)
        post = np.asarray(result.flat()[:, j_post], dtype=float)

        # ==============================================================
        # Prior samples
        # ==============================================================
        pri = None

        if nm in prior.param_names:
            pri = np.asarray(
                prior_draws[:, prior.param_names.index(nm)],
                dtype=float,
            )

        # ==============================================================
        # Decide whether to use log10(parameter) space
        # ==============================================================
        use_log = (
            nm in log_params
            and np.all(post > 0)
            and (
                pri is None
                or np.all(pri > 0)
            )
        )

        # ==============================================================
        # LOG PARAMETERS
        # ==============================================================
        if use_log:

            # ----------------------------------------------------------
            # Transform BOTH distributions to log10 space.
            # ----------------------------------------------------------
            post_plot = np.log10(post)

            pri_plot = None
            if pri is not None:
                pri_plot = np.log10(pri)

            # ----------------------------------------------------------
            # Determine plotting window in LOG10 space.
            #
            # This means the window expands symmetrically in decades,
            # rather than symmetrically in the original parameter.
            # ----------------------------------------------------------
            lo_p = np.min(post_plot)
            hi_p = np.max(post_plot)

            span = (hi_p - lo_p) + 1e-9

            margin = max(1.0, 1.5 * span)

            lo = lo_p - margin
            hi = hi_p + margin

            # Never extend outside the sampled prior support.
            if pri_plot is not None:
                lo = max(lo, np.min(pri_plot))
                hi = min(hi, np.max(pri_plot))

            # ----------------------------------------------------------
            # Linear bins in LOG10 space.
            #
            # This is what makes a log-uniform prior flat.
            # ----------------------------------------------------------
            edges = np.linspace(lo, hi, bins)

            # ----------------------------------------------------------
            # Prior
            # ----------------------------------------------------------
            if pri_plot is not None:

                pri_win = pri_plot[
                    (pri_plot >= edges[0])
                    & (pri_plot <= edges[-1])
                ]

                if pri_win.size:

                    ax.hist(
                        pri_win,
                        bins=edges,
                        density=True,
                        color=C_PRIOR,
                        alpha=0.55,
                        label="prior (windowed)",
                        edgecolor="none",
                    )

            # ----------------------------------------------------------
            # Posterior
            # ----------------------------------------------------------
            ax.hist(
                post_plot,
                bins=edges,
                density=True,
                color=C_POST,
                alpha=0.75,
                label="posterior",
                edgecolor="none",
            )

            # ----------------------------------------------------------
            # Posterior median
            #
            # The median is calculated in ORIGINAL parameter space,
            # then transformed to the plotting coordinate.
            # ----------------------------------------------------------
            med = float(np.median(post))
            med_plot = np.log10(med)

            ax.axvline(
                med_plot,
                color=C_MEDIAN,
                lw=1.8,
                ls="-",
                label=f"post. median\n{med:.4g}",
            )

            # ----------------------------------------------------------
            # Truth
            # ----------------------------------------------------------
            if (
                truth is not None
                and nm in truth
                and truth[nm] is not None
            ):

                truth_val = float(truth[nm])

                if truth_val > 0:

                    truth_plot = np.log10(truth_val)

                    ax.axvline(
                        truth_plot,
                        color=C_TRUTH,
                        lw=2.0,
                        ls="--",
                        label=f"truth {truth_val:.4g}",
                    )

            # ----------------------------------------------------------
            # Format x-axis.
            #
            # Underlying coordinate = log10(parameter)
            # Displayed coordinate   = original parameter value
            # ----------------------------------------------------------
            ax.xaxis.set_major_formatter(
                FuncFormatter(format_log_tick)
            )

            ax.set_xlabel(nm)

        # ==============================================================
        # LINEAR PARAMETERS
        # ==============================================================
        else:

            post_plot = post
            pri_plot = pri

            # ----------------------------------------------------------
            # Determine plotting window
            # ----------------------------------------------------------
            lo_p = np.min(post_plot)
            hi_p = np.max(post_plot)

            span = (hi_p - lo_p) + 1e-12

            lo = lo_p - 1.5 * span
            hi = hi_p + 1.5 * span

            # Do not extend beyond prior support.
            if pri_plot is not None:
                lo = max(lo, np.min(pri_plot))
                hi = min(hi, np.max(pri_plot))

            edges = np.linspace(lo, hi, bins)

            # ----------------------------------------------------------
            # Prior
            # ----------------------------------------------------------
            if pri_plot is not None:

                pri_win = pri_plot[
                    (pri_plot >= edges[0])
                    & (pri_plot <= edges[-1])
                ]

                if pri_win.size:

                    ax.hist(
                        pri_win,
                        bins=edges,
                        density=True,
                        color=C_PRIOR,
                        alpha=0.55,
                        label="prior (windowed)",
                        edgecolor="none",
                    )

            # ----------------------------------------------------------
            # Posterior
            # ----------------------------------------------------------
            ax.hist(
                post_plot,
                bins=edges,
                density=True,
                color=C_POST,
                alpha=0.75,
                label="posterior",
                edgecolor="none",
            )

            # ----------------------------------------------------------
            # Posterior median
            # ----------------------------------------------------------
            med = float(np.median(post))

            ax.axvline(
                med,
                color=C_MEDIAN,
                lw=1.8,
                ls="-",
                label=f"post. median\n{med:.4g}",
            )

            # ----------------------------------------------------------
            # Truth
            # ----------------------------------------------------------
            if (
                truth is not None
                and nm in truth
                and truth[nm] is not None
            ):

                truth_val = float(truth[nm])

                ax.axvline(
                    truth_val,
                    color=C_TRUTH,
                    lw=2.0,
                    ls="--",
                    label=f"truth {truth_val:.4g}",
                )

            ax.set_xlabel(nm)

        # ==============================================================
        # Common formatting
        # ==============================================================
        ax.set_ylabel("density")

        ax.legend(
            fontsize=6.5,
            framealpha=0.9,
        )

        # --------------------------------------------------------------
        # MCMC diagnostics
        # --------------------------------------------------------------
        rhat = result.rhat.get(nm, np.nan)
        ess = result.ess.get(nm, np.nan)

        flag = (
            ""
            if (
                np.isfinite(rhat)
                and rhat <= 1.01
            )
            else "  [!]"
        )

        ax.set_title(
            f"{nm}   R-hat={rhat:.3f}  ESS={ess:.0f}{flag}",
            fontsize=8.5,
        )

    # ------------------------------------------------------------------
    # Turn off unused axes
    # ------------------------------------------------------------------
    for k in range(len(names), len(axes)):
        axes[k].axis("off")

    # ------------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------------
    if title:
        fig.suptitle(
            title,
            fontsize=12,
            y=1.005,
        )

    # ------------------------------------------------------------------
    # Finish / save
    # ------------------------------------------------------------------
    return _finish(fig, save_path)

# =============================================================================
# Per-chain diagnostics
# =============================================================================

def per_chain_summary(result, truth: Optional[dict] = None,
                      verbose: bool = True) -> dict:
    """
    Where did each chain actually end up?

    Reports per-chain median (and its 2.5/97.5 percentiles) for every
    parameter. This is the concrete, readable version of what R-hat
    summarises in one number: if the four per-chain medians disagree
    with each other by more than their own within-chain spread, the
    chains have NOT converged to a common distribution and the pooled
    posterior is not yet meaningful.
    """
    ch = result.chains                      # (n_chains, n_draws, D)
    n_chains = ch.shape[0]
    out = {}

    if verbose:
        print(f"\n  per-chain medians ({n_chains} chains)")
        hdr = f"  {'param':<11}" + "".join(f"{'ch'+str(c):>11}" for c in range(n_chains))
        hdr += f"{'spread':>11}{'within':>11}  flag"
        print(hdr)

    for j, nm in enumerate(result.param_names):
        meds = np.array([np.median(ch[c, :, j]) for c in range(n_chains)])
        # within-chain scale: mean of per-chain IQRs
        iqrs = np.array([np.subtract(*np.percentile(ch[c, :, j], [75, 25]))
                         for c in range(n_chains)])
        spread = float(meds.max() - meds.min())     # between-chain disagreement
        within = float(np.mean(iqrs))               # typical within-chain width
        ratio  = spread / within if within > 0 else np.nan
        out[nm] = dict(chain_medians=meds, between=spread,
                       within=within, ratio=ratio)
        if verbose:
            flag = "OK" if (np.isfinite(ratio) and ratio < 0.5) else "DISAGREE"
            row = f"  {nm:<11}" + "".join(f"{m:11.5g}" for m in meds)
            row += f"{spread:11.4g}{within:11.4g}  {flag}"
            print(row)

    if verbose:
        print("    spread = max-min of chain medians;  within = mean per-chain IQR")
        print("    DISAGREE (spread > 0.5*within) => chains landed in different places")
    return out


def plot_trace(result, params: Optional[Sequence[str]] = None,
               title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Trace plot: every chain's path over draws, one panel per parameter.

    A healthy trace looks like a fuzzy horizontal band with all chains
    overlapping. Chains sitting in separate bands = not converged;
    slow drift = burn-in was too short.
    """
    names = list(params) if params is not None else list(result.param_names)
    fig, axes, _ = _grid(len(names), ncols=2, panel=(5.2, 2.2))
    ch = result.chains

    for k, nm in enumerate(names):
        ax = axes[k]
        j = result.param_names.index(nm)
        for c in range(ch.shape[0]):
            ax.plot(ch[c, :, j], lw=0.6, alpha=0.75,
                    color=C_CHAIN[c % len(C_CHAIN)], label=f"chain {c}")
        ax.set_xlabel("draw")
        ax.set_ylabel(nm)
        rhat = result.rhat.get(nm, np.nan)
        ax.set_title(f"{nm}   R-hat={rhat:.3f}", fontsize=8.5)
        if k == 0:
            ax.legend(fontsize=6, ncol=2)

    for k in range(len(names), len(axes)):
        axes[k].axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.005)
    return _finish(fig, save_path)


# =============================================================================
# Bayesian vs. residual bootstrap
# =============================================================================

def plot_bayes_vs_bootstrap(
    bayes_by_noise: dict,
    boot_by_noise:  dict,
    truth:          dict,
    params:         Sequence[str],
    title:          Optional[str] = None,
    save_path:      Optional[str] = None,
):
    """
    Density overlay of the Bayesian posterior and the residual-bootstrap
    replicate distribution, one row per noise level, one column per
    parameter. Both normalised to unit area (density=True).

    bayes_by_noise / boot_by_noise : {sigma_level: {param: 1-D samples}}
    """
    sigmas = sorted(bayes_by_noise.keys())
    nrows, ncols = len(sigmas), len(params)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.5 * nrows),
                             squeeze=False)

    for r, sg in enumerate(sigmas):
        for c, nm in enumerate(params):
            ax = axes[r][c]
            b = np.asarray(bayes_by_noise[sg][nm], float)
            f = np.asarray(boot_by_noise[sg][nm], float)
            b = b[np.isfinite(b)]
            f = f[np.isfinite(f)]
            if b.size == 0 and f.size == 0:
                ax.axis("off")
                continue

            allv = np.concatenate([b, f]) if f.size else b
            edges = np.linspace(np.percentile(allv, 0.5),
                                np.percentile(allv, 99.5), 45)

            ax.hist(b, bins=edges, density=True, color=C_POST, alpha=0.60,
                    label="Bayesian posterior", edgecolor="none")
            if f.size:
                ax.hist(f, bins=edges, density=True, color=C_TRUTH, alpha=0.45,
                        label="residual bootstrap", edgecolor="none")
            if nm in truth:
                ax.axvline(truth[nm], color="k", ls="--", lw=1.6, label="truth")

            ax.set_ylabel("density" if c == 0 else "")
            if r == 0:
                ax.set_title(nm, fontsize=10)
            if c == 0:
                ax.text(-0.32, 0.5, f"sigma = {sg:g}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9)
            if r == 0 and c == ncols - 1:
                ax.legend(fontsize=6.5)

    if title:
        fig.suptitle(title, fontsize=12, y=1.002)
    return _finish(fig, save_path)


def plot_width_vs_noise(summary_rows, params, title=None, save_path=None):
    """
    Interval width vs. noise level, Bayesian vs. bootstrap.

    summary_rows : list of dicts with keys
                   sigma, param, bayes_iqr, boot_iqr
    Both methods should widen with noise; systematic separation between
    the two curves is the interesting signal.
    """
    fig, axes, _ = _grid(len(params), ncols=len(params), panel=(3.6, 2.9))
    for k, nm in enumerate(params):
        ax = axes[k]
        rows = [r for r in summary_rows if r["param"] == nm]
        rows.sort(key=lambda r: r["sigma"])
        sg = [r["sigma"] for r in rows]
        ax.plot(sg, [r["bayes_iqr"] for r in rows], "o-", color=C_POST,
                label="Bayesian IQR")
        ax.plot(sg, [r["boot_iqr"] for r in rows], "s--", color=C_TRUTH,
                label="bootstrap IQR")
        ax.set_xlabel("simulated noise sigma")
        ax.set_ylabel(f"{nm} IQR")
        ax.set_title(nm, fontsize=9)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(params), len(axes)):
        axes[k].axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.005)
    return _finish(fig, save_path)


def plot_aif(time_var, aif, title="Parker AIF", save_path=None):
    """Sanity plot of the simulated AIF."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(time_var, aif, color=C_POST, lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("plasma [CA] (mM)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)
