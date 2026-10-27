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
    FitModeConfig,
    VoxelwiseConfig,
    VoxelwiseResult,
)
from dce_mri.kinetic_models import KineticModelSpec
from dce_mri.fitting import call_fitter


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
    fit_mode_cfg:  FitModeConfig        = FitModeConfig(),
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
    t_start = time.time()

    # ── Build mask ────────────────────────────────────────────────────
    if mask is None:
        baseline = sig_4d[..., acq.nbase].mean(axis=-1)
        mask = baseline > np.percentile(baseline, 10)
    
    coords    = np.argwhere(mask)
    n_voxels  = len(coords)
    n_workers = voxelwise_cfg.n_workers or max(1, cpu_count() - 1)
    chunk_size = voxelwise_cfg.chunk_size or max(1, n_voxels // n_workers)
    
    chunks = _build_chunks(coords, chunk_size)
    
    args_list = [
        {
            "coords":       chunk,
            "sig_4d":       sig_4d,
            "aif":          aif,
            "time_var":     time_var,
            "model":        model,
            "acq":          acq,
            "subject":      subject,
            "fitting_cfg":  fitting_cfg,
            "fit_mode_cfg": fit_mode_cfg,
        }
        for chunk in chunks
    ]
    
    print(f"[volume] fitting {n_voxels} voxels "
          f"({len(chunks)} chunks, {n_workers} workers, "
          f"chunk_size={chunk_size}) ...")
    
    with Pool(n_workers) as pool:
        chunk_results = pool.map(_process_chunk, args_list)
    
    all_results = [r for chunk in chunk_results for r in chunk]
    
    # ── Build output maps ─────────────────────────────────────────────
    shape_3d   = sig_4d.shape[:3]
    
    if fit_mode_cfg.mode == "dispersed":
        from dce_mri.kinetic_models import DISPERSED_GKM
        current_param_names = DISPERSED_GKM.param_names
    else:
        current_param_names = model.param_names

    param_maps = {name: np.zeros(shape_3d) for name in current_param_names}
    cost_map    = np.zeros(shape_3d)
    success_map = np.zeros(shape_3d, dtype=bool)
    
    if fit_mode_cfg.mode == "dispersed":
        param_maps["best_level"] = np.zeros(shape_3d)
    elif fit_mode_cfg.mode in ("delay_grid", "delay_continuous"):
        param_maps["best_tau"] = np.zeros(shape_3d)
    
    n_success = 0
    for r in all_results:
        i, j, k = r["coord"]
        for pi, name in enumerate(current_param_names):
            param_maps[name][i, j, k] = r["params"][pi]
        cost_map[i, j, k]    = r["cost"]
        success_map[i, j, k] = r["success"]
        if r["success"]:
            n_success += 1
        
        # extra keys
        if "best_level" in r:
            param_maps["best_level"][i, j, k] = r["best_level"]
        if "best_tau" in r:
            param_maps["best_tau"][i, j, k] = r["best_tau"]
    
    elapsed = time.time() - t_start
    print(f"[volume] done — {n_success}/{n_voxels} successful "
          f"in {elapsed:.1f}s")
    
    return VoxelwiseResult(
        param_maps  = param_maps,
        cost_map    = cost_map,
        success_map = success_map,
        n_voxels    = n_voxels,
        n_success   = n_success,
        elapsed_s   = elapsed,
    )
   


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
    Also saves cost_map and success_map.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    affine = ref_img.affine
    header = ref_img.header

    maps_to_save = dict(result.param_maps)
    maps_to_save["cost"]    = result.cost_map
    maps_to_save["success"] = result.success_map.astype(np.float32)

    for name, data in maps_to_save.items():
        fname = os.path.join(cfg.save_dir,
                             f"{cfg.output_prefix}{name}.nii.gz")
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header),
                 fname)
        print(f"[volume] saved {fname}")

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

    """
    n_chunks = max(1, len(coords) // chunk_size)
    return np.array_split(coords, n_chunks)

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

    """


    coords      = args["coords"]
    sig_4d      = args["sig_4d"]
    aif         = args["aif"]
    time_var    = args["time_var"]
    model       = args["model"]
    acq         = args["acq"]
    subject     = args["subject"]
    fitting_cfg = args["fitting_cfg"]
    fit_mode_cfg= args["fit_mode_cfg"]
    
    n_params = (len(model.param_names) if fit_mode_cfg.mode != "dispersed"
               else 4)
    
    results = []
    for coord in coords:
        i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
        sig_v = sig_4d[i, j, k, :]
        
        voxel_seed = hash((i, j, k)) % (2**31)   # deterministic per-voxel seed
        
        try:
            flat = call_fitter(sig_v, aif, time_var, model, acq,
                              subject, fitting_cfg, fit_mode_cfg, seed=voxel_seed)
        except Exception:
            flat = {"params": np.full(n_params, np.nan),
                   "cost": np.inf, "success": False}

        flat["coord"] = (i, j, k)
        results.append(flat)

    return results