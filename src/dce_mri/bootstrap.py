# -*- coding: utf-8 -*-
"""
bootstrap.py
============
Bootstrap uncertainty loops and sigma sweeps.

Public API
----------
    run_bootstrap(sig_orig, aif, time_var, model, acq, subject,
                  fitting_cfg, bootstrap_cfg)
        →  BootstrapResult
        Generate B noisy replicates of sig_orig and fit each one.

    run_sigma_sweep(S_est, aif, time_var, model, acq, subject,
                    fitting_cfg, bootstrap_cfg, sigmas, out_dir)
        →  dict[sigma: BootstrapResult]
        Sweep over noise levels, useful for evaluating parameter uncertainty
        as a function of SNR.

    run_ace_bootstrap(sig_orig, aif, time_var, protocol, model,
                      fit_mode, acq, subject, fitting_cfg, bootstrap_cfg,
                      B1_fixed, T10_fixed)
        →  BootstrapResult
        ACE-specific bootstrap loop.

Internal helpers
----------------
    _bootstrap_loop(fit_fn, sig_boot, aif, time_var, acq, subject,
                    fitting_cfg, verbose)
        →  BootstrapResult
        Shared loop used by both standard and ACE bootstrap.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    FittingConfig,
    BootstrapConfig,
    BootstrapResult,
)
from dce_mri.kinetic_models import KineticModelSpec


def run_bootstrap(
    sig_orig:     np.ndarray,
    aif:          np.ndarray,
    time_var:     np.ndarray,
    model:        KineticModelSpec,
    acq:          AcquisitionConfig,
    subject:      SubjectConfig,
    fitting_cfg:  FittingConfig,
    bootstrap_cfg: BootstrapConfig,
) -> BootstrapResult:
    """
    Generate B noisy replicates of sig_orig (Gaussian noise) and fit each.

    TODO:
    1. sig_boot = gaussian_noise_many(sig_orig, sigma, B, seed)
       NOTE: sigma is a free parameter here — caller sets it via the
             BootstrapConfig or passes it in. Consider adding sigma to
             BootstrapConfig or as a separate argument.
    2. return _bootstrap_loop(fit_single_voxel, sig_boot, aif, time_var,
                               model, acq, subject, fitting_cfg)
    """
    raise NotImplementedError


def run_sigma_sweep(
    S_est:        np.ndarray,
    aif:          np.ndarray,
    time_var:     np.ndarray,
    sigmas:       list,
    model:        KineticModelSpec,
    acq:          AcquisitionConfig,
    subject:      SubjectConfig,
    fitting_cfg:  FittingConfig,
    bootstrap_cfg: BootstrapConfig,
    out_dir:      str = ".",
) -> dict:
    """
    Run bootstrap uncertainty for each sigma in sigmas.

    Returns {sigma: BootstrapResult}

    TODO: loop over sigmas, call run_bootstrap for each, save/load npz
    """
    raise NotImplementedError


def _bootstrap_loop(
    fit_fn,
    sig_boot:    np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    verbose:     bool = True,
) -> BootstrapResult:
    """
    Inner loop: fit one replicate per row of sig_boot.

    TODO: copy from tcxm_bootstrap_pipeline._bootstrap_loop and update
          to call fit_fn(sig_boot[b], aif, time_var, model, acq, subject, cfg)
    """
    raise NotImplementedError


def gaussian_noise_many(
    y:     np.ndarray,
    sigma: float,
    B:     int           = 100,
    seed:  Optional[int] = None,
) -> np.ndarray:
    """
    Generate B Gaussian-noise replicates of y.

    TODO: copy from tcxm_bootstrap_pipeline.gaussian_noise_many
    """
    raise NotImplementedError
