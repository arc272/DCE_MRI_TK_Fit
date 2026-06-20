# -*- coding: utf-8 -*-
"""
volume.py
=========
Chunked parallel voxelwise fitting over a full NIfTI volume.

Design: voxels are batched into fixed-size chunks before being dispatched
to worker processes.  This reduces multiprocessing overhead (fewer
process-creation and IPC events) compared to dispatching one voxel per task.

Public API
----------
    run_voxelwise_fitting(sig_4d, aif, time_var, model, acq, subject,
                          fitting_cfg, voxelwise_cfg, mask)
        →  VoxelwiseResult

    save_nifti_maps(result, ref_img, cfg)
        Saves each parameter map as a separate NIfTI file.

    load_nifti_canonical(path)   →  (np.ndarray, nib.Nifti1Image)
    create_binary_masks(seg, ...) →  list[np.ndarray]

Internal helpers
----------------
    _build_chunks(coords, chunk_size)   →  list[list[coord]]
    _process_chunk(chunk_args)          →  list[dict]  (worker function)
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import nibabel as nib
import numpy as np

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    FittingConfig,
    VoxelwiseConfig,
    VoxelwiseResult,
)
from dce_mri.kinetic_models import KineticModelSpec


# =============================================================================
# Public: chunked parallel voxelwise fitting
# =============================================================================

def run_voxelwise_fitting(
    sig_4d:       np.ndarray,
    aif:          np.ndarray,
    time_var:     np.ndarray,
    model:        KineticModelSpec,
    acq:          AcquisitionConfig,
    subject:      SubjectConfig,
    fitting_cfg:  FittingConfig,
    voxelwise_cfg: VoxelwiseConfig,
    mask:         Optional[np.ndarray] = None,
) -> VoxelwiseResult:
    """
    Fit kinetic model to every masked voxel in a 4D volume, in parallel.

    Chunking strategy
    -----------------
    All masked voxel coordinates are collected into a flat list, then
    split into chunks of size voxelwise_cfg.chunk_size.  Each chunk is
    sent to one worker as a single task.  The worker fits all voxels
    in its chunk sequentially and returns results as a list of dicts.

    This is more efficient than one-voxel-per-task because:
      - Process creation and IPC cost is amortised over chunk_size voxels
      - Workers stay busy for longer between result-collection pauses
      - Memory copies are reduced (one chunk array vs N_vox individual arrays)

    Parameters
    ----------
    sig_4d        : (X, Y, Z, T) signal volume
    aif           : (T,) global AIF
    time_var      : (T,) time grid  (s)
    model         : KineticModelSpec
    acq           : AcquisitionConfig
    subject       : SubjectConfig
    fitting_cfg   : FittingConfig
    voxelwise_cfg : VoxelwiseConfig  (n_workers, chunk_size, ...)
    mask          : (X, Y, Z) bool; if None, auto-generate from baseline SNR

    Returns
    -------
    VoxelwiseResult

    TODO:
    1. If mask is None: generate from baseline mean > 10th percentile
    2. coords = np.argwhere(mask)   # (N_vox, 3)
    3. n_workers = voxelwise_cfg.n_workers or cpu_count() - 1
       chunk_size = voxelwise_cfg.chunk_size or len(coords) // n_workers
    4. chunks = _build_chunks(coords, chunk_size)
    5. Build args list: each element is a dict/tuple containing the chunk
       coordinates + the full sig_4d array (workers index into it)
       NOTE: pass sig_4d once per chunk, not per voxel — let the worker
             index sig_4d[i, j, k, :] internally
    6. with Pool(n_workers) as pool:
           all_results = pool.map(_process_chunk, args_list)
           (or imap_unordered for progress logging)
    7. Flatten results and fill param_maps, cost_map, success_map
    8. Log progress, QC plots for first n_qc_plots successful voxels
    9. Return VoxelwiseResult
    """
    raise NotImplementedError


# =============================================================================
# Public: NIfTI I/O
# =============================================================================

def save_nifti_maps(
    result:  VoxelwiseResult,
    ref_img: nib.Nifti1Image,
    cfg:     VoxelwiseConfig,
) -> None:
    """
    Save each parameter map in result.param_maps as a NIfTI file.

    Output filenames: {cfg.save_dir}/{cfg.output_prefix}{param_name}.nii.gz
    Also saves cost_map and success_map.

    TODO: loop over result.param_maps items and save with nibabel,
          matching affine and header from ref_img
    """
    raise NotImplementedError


def load_nifti_canonical(
    path: str,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI, reorient to closest canonical orientation.

    Returns (data, img)

    TODO: nib.as_closest_canonical(nib.load(path)); img.get_fdata()
    """

    img  = nib.as_closest_canonical(nib.load(path))
    data = img.get_fdata()
    return data,img


def create_binary_masks(
    seg:         np.ndarray,
    labels:      Optional[np.ndarray] = None,
    return_list: bool                 = False,
) -> np.ndarray:
    """
    Convert a labelled segmentation into per-label binary masks.

    TODO: copy from tcxm_bootstrap_pipeline.create_binary_masks
    """
    raise NotImplementedError


# =============================================================================
# Internal: chunking and worker
# =============================================================================

def _build_chunks(
    coords:     np.ndarray,
    chunk_size: int,
) -> list:
    """
    Split coordinate array into chunks of size chunk_size.

    Returns list of np.ndarray, each of shape (chunk_size, 3) except
    possibly the last chunk which may be smaller.

    TODO: np.array_split(coords, max(1, len(coords) // chunk_size))
    """
    raise NotImplementedError


def _process_chunk(args: dict) -> list[dict]:
    """
    Worker function: fit all voxels in one chunk.

    Args dict contains:
        coords      : (chunk_size, 3) voxel coordinates
        sig_4d      : (X, Y, Z, T) full signal volume (read-only)
        aif         : (T,)
        time_var    : (T,)
        model       : KineticModelSpec
        acq         : AcquisitionConfig
        subject     : SubjectConfig
        fitting_cfg : FittingConfig

    Returns list of dicts, one per voxel:
        {'coord': (i,j,k), 'params': ..., 'cost': ..., 'success': ...,
         'S_meas': ..., 'S_pred': ...}

    This function is called by multiprocessing.Pool — it must be
    importable at the top level (no lambdas or nested functions as args).

    TODO:
    1. Unpack args
    2. from dce_mri.fitting import fit_single_voxel
    3. for coord in args['coords']:
           i, j, k = coord
           sig_v = args['sig_4d'][i, j, k, :]
           result = fit_single_voxel(sig_v, aif, time_var, model,
                                      acq, subject, fitting_cfg)
           results.append({'coord': coord, 'params': result.params, ...})
    4. return results
    """
    raise NotImplementedError
