# -*- coding: utf-8 -*-
"""
statistics.py
=============
Summary statistics for bootstrap and voxelwise results.

Public API
----------
    param_stats(params, p_ref)          →  dict of per-parameter metrics
    compute_over_sweep(results, ...)    →  dict of metrics vs sweep variable
    ace_param_stats(b1t10, kin, ok, ...) → dict
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_EPS = 1e-11


def param_stats(
    params: np.ndarray,
    p_ref:  np.ndarray,
) -> dict:
    """
    Per-parameter summary statistics for a (B, K) bootstrap array.

    Returns dict with keys:
        mean, median, bias_mean_pct, bias_med_pct, sd, sd_mad, cov_pct

    TODO: copy from tcxm_bootstrap_pipeline.param_stats — no changes needed
    """
    raise NotImplementedError


def compute_over_sweep(
    results:      dict,
    sweep_values: Sequence,
    method_key:   str,
    p_ref:        np.ndarray,
    success_only: bool = True,
) -> dict:
    """
    Aggregate param_stats across sweep values.

    TODO: copy from tcxm_bootstrap_pipeline.compute_over_sweep
    """
    raise NotImplementedError


def ace_param_stats(
    b1t10_boot: np.ndarray,
    kin_boot:   np.ndarray,
    ok_boot:    np.ndarray,
    b1_t10_ref: np.ndarray,
    kin_ref:    np.ndarray,
    kin_names:  list,
) -> dict:
    """
    Summary statistics for ACE bootstrap results (B1, T10, + kinetics).

    TODO: copy from ace_mri_model.ace_param_stats
    """
    raise NotImplementedError
