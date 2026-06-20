# -*- coding: utf-8 -*-
"""
io.py
=====
All file I/O for the DCE-MRI pipeline.

Public API
----------
    load_aif(path, skip_rows)            →  np.ndarray
    load_bootstrap_npz(path)             →  dict
    save_results_npz(results, path, ...) →  None
    load_results_npz(path)               →  dict
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def load_aif(
    path:      str,
    skip_rows: int = 1,
) -> np.ndarray:
    """
    Load AIF from CSV file, skipping header rows.
    """
    return np.loadtxt(path, delimiter=",")[skip_rows:]


def load_bootstrap_npz(path: str) -> dict:
    """
    Load a compressed bootstrap NPZ archive.

    """
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}



#def save_results_npz(
#    results:      dict,
#    out_path:     str,
#    extra_arrays: Optional[dict] = None,
#) -> None:
#    """
    # Flatten and save results dict to compressed NPZ.

    # TODO: copy from tcxm_bootstrap_pipeline.save_results_npz
    # """
    # for sweep_val, method_dict in results.items():
    #     for method, arr_dict in method_dict.items():
    #         for key, arr in arr_dict.items():
    #             save_dict[f"{sweep_val:g}_{method}_{key}"] = arr

    # if extra_arrays:
    #     save_dict.update(extra_arrays)

    # os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # np.savez_compressed(out_path, **save_dict)
    # print(f"Saved: {out_path}")


def load_results_npz(path: str) -> dict:
    """
    Reconstruct nested results dict from a flattened NPZ file.

    TODO: copy from tcxm_bootstrap_pipeline.load_results_from_npz
    """
    raise NotImplementedError






