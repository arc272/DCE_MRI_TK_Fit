#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Sep 22 11:31:23 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_aifs.py
===============
Batch AIF extraction for a list of FMB subjects, run BEFORE the ROI fits.

For each subject:
  1. load the 4D DCE volume and the aorta ROI (canonical orientation)
  2. aif_idx = first labelled aorta slice
  3. erode that slice's 2-D mask (3x3 structure, EROSION_ITERS iterations)
     and place the eroded mask on aif_idx AND aif_idx+1
     (the eroded ROI is kept in memory only -- never written to disk)
  4. extract_aif(...)  -- the SAME function 06_roi_fitting.ipynb uses:
     mean signal in the mask -> SPGR inversion (blood T1) -> /(1 - Hct)
  5. save notebook-style diagnostic figures + one labelled CSV

All sequence / AIF parameters match 06_roi_fitting.ipynb cells [4] and [8].

Run:
    PYTHONPATH=src python scripts/extract_aifs.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from dce_mri.config import AcquisitionConfig, AifConfig
from dce_mri.signal_models import extract_aif
from dce_mri.volume import load_nifti_canonical

# =============================================================================
# settings
# =============================================================================

FMB_IDS = ["FMB001",
    "FMB003",
    "FMB005",
    "FMB010",
    "FMB011",
    "FMB013",
    "FMB014",
    "FMB016",
    "FMB020",
    "FMB024",
    "FMB027",
    "FMB032",
    "FMB036",
    "FMB038",
    "FMB040",
    "FMB041",
    "FMB044",
    "FMB045",
    "FMB046",
    "FMB048",
    "FMB050",
    "FMB052",
    "FMB055",
    "FMB056",
    "FMB060",
    "FMB070",
    "FMB071",
    "FMB072",
    "FMB074",
    "FMB079",
    "FMB080",
    "FMB081",
    "FMB083",
    "FMB084",
    "FMB085",
    "FMB093",
    "FMB095",
    "FMB096",
    "FMB097",
    "FMB099",
    "FMB100",
    "FMB104",
    "FMB107",
    "FMB108",
    "FMB109",
    "FMB111",
    "FMB112",
    "FMB115",
    "FMB118",
    "FMB122",
    "FMB124",
    "FMB126",
    "FMB128",
    "FMB129",
    "FMB132",
    "FMB136",
    "FMB140",
    "FMB141",
    "FMB143",
    "FMB146",
    "FMB147",
    "FMB148",
    "FMB149",
    "FMB152",
    "FMB158",
    "FMB162",
    "FMB164",
    "FMB165",
    "FMB167",
    "FMB170",
    "FMB172",
    "FMB176",
    "FMB177",
    "FMB183",
    "FMB184",
    "FMB185",
    "FMB186",
    "FMB187",
    "FMB189",
    "FMB192",
    "FMB194",
    "FMB196",
    "FMB198",
    "FMB200",
    "FMB202",
    "FMB203",
    "FMB207",
    "FMB212",
    "FMB213",
    "FMB215",
    "FMB216",
    "FMB217",
    "FMB219",
    "FMB225",
    "FMB226",
    "FMB228",
    "FMB231",
    "FMB232",
    "FMB233",
    "FMB234",
    "FMB238",
    "FMB239",
    "FMB244",
    "FMB245",
    "FMB246",
    "FMB248",
    "FMB249",
    "FMB259",
    "FMB264",
    "FMB265",
    "FMB271",
    "FMB276",
    "FMB283",
    "FMB286",
    "FMB288",
    "FMB289",
    "FMB290",
    "FMB291",
    "FMB293",
    "FMB294",
    "FMB295",
    "FMB296",
    "FMB297",
    "FMB298",
    "FMB299",]          
# e.g. ["FMB013", "FMB038", "FMB056", "FMB081"]

DATA_ROOT   = "/mnt/labspace/users/allison/fastMRI_data/tight_masks/canonical"
RESULTS_DIR = "/mnt/labspace/users/allison/fastMRI_data/aif_results"

# --- acquisition: 06_roi_fitting.ipynb cell [4] -----------------------------
SPOKES_PER_FRAME = 8
NBASE = slice(0, 7)          # "nbase will change with spokes_per_frame"
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)

NSPOKE   = 288                # total spokes across the whole scan
SCANTIME = 150                # total scan duration, s

# --- AIF: 06_roi_fitting.ipynb cell [8] -------------------------------------
aif_cfg = AifConfig(
    T10        = 1779.0,
    hematocrit = 0.4,
    top_frac   = 1.0,
    nbase      = acq.nbase,
)
B1 = 1.0                      # SubjectConfig(B1=1.0) in the notebook

# --- ROI erosion ------------------------------------------------------------
EROSION_ITERS = 2                     # default for every subject
EROSION_ITERS_OVERRIDE = {            # per-subject exceptions, e.g.
     "FMB192": 1,
     "FMB046": 1,
     "FMB128": 1,
     "FMB149": 1,
     "FMB170": 1,
     "FMB176": 1,
     "FMB185": 1,
     "FMB186": 1,
     "FMB189": 1,
     "FMB216": 1,
     "FMB238": 1,
     "FMB239": 1,
     "FMB276": 1,
     "FMB283": 1,
     "FMB294": 1,

}
KEEP_OTHER_LABELLED_SLICES = False    # False: ONLY aif_idx and aif_idx+1
                                      # contribute to the AIF

TIME_IDX_FOR_ROI_FIG = 15     # frame shown in the ROI inspection figure
SHOW_FIGS = False             # True to also plt.show() interactively

if not SHOW_FIGS:
    matplotlib.use("Agg")


# =============================================================================
# per-subject paths  (same pattern as 06_roi_fitting.ipynb cell [4])
# =============================================================================

class Args:
    def __init__(self, fmb_id):
        subj_dir = os.path.join(DATA_ROOT, fmb_id)
        self.fmb_id = fmb_id

        # DCE filename varies (e.g. "FMB001_tv1em05c_ttgv3em04z.nii.gz"),
        # so glob for it and refuse to guess if missing or ambiguous
        dce_hits = sorted(glob.glob(os.path.join(subj_dir, f"{fmb_id}_tv*.nii.gz")))
        if len(dce_hits) != 1:
            raise FileNotFoundError(
                f"{fmb_id}: expected exactly one '{fmb_id}_tv*.nii.gz' in "
                f"{subj_dir}, found {len(dce_hits)}: {dce_hits}")
        self.dce_file = dce_hits[0]

        # aorta ROI name varies (e.g. "56_Aorta_RotOrtho_6.nii"), so glob
        # for it and refuse to guess if it is missing or ambiguous
        hits = sorted(glob.glob(os.path.join(subj_dir, "*orta*.nii*")))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{fmb_id}: expected exactly one '*Aorta*.nii*' in "
                f"{subj_dir}, found {len(hits)}: {hits}")
        self.aif_mask_file = hits[0]

# =============================================================================
# helpers
# =============================================================================

def save_fig(fig, name, fmb_id):
    """Same naming as the notebook's save_fig: {subject}_{name}.png"""
    path = os.path.join(RESULTS_DIR, f"{fmb_id}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def build_eroded_aif_mask(aif_roi, iters):
    """
    Erode the first labelled aorta slice and apply the eroded 2-D mask to
    aif_idx and aif_idx+1. Returns (eroded 3-D mask, aif_idx, labelled slices).
    Nothing is written to disk.
    """
    labelled = np.where(aif_roi.sum(axis=(0, 1)) > 0)[0]
    if labelled.size == 0:
        raise ValueError("aorta ROI is empty")
    aif_idx = int(labelled[0])

    eroded_2d = binary_erosion(aif_roi[:, :, aif_idx],
                               structure=np.ones((3, 3)), iterations=iters)
    if not eroded_2d.any():
        raise ValueError(
            f"erosion with {iters} iteration(s) removed every voxel on slice "
            f"{aif_idx} -- lower EROSION_ITERS for this subject")

    if aif_idx + 1 >= aif_roi.shape[2]:
        raise ValueError(f"aif_idx+1 = {aif_idx + 1} is outside the volume")

    if KEEP_OTHER_LABELLED_SLICES:
        mask = aif_roi.copy()
    else:
        mask = np.zeros_like(aif_roi, dtype=bool)
    mask[:, :, aif_idx]     = eroded_2d
    mask[:, :, aif_idx + 1] = eroded_2d
    return mask, aif_idx, labelled


def plot_roi_inspection(sig_4d, aif_roi, aif_mask, aif_idx, time_idx, fmb_id):
    """
    Notebook 'roi_inspection' AIF panel, extended to the two slices that
    contribute: original ROI (outline) vs eroded ROI (fill).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, z in zip(axes, (aif_idx, aif_idx + 1)):
        ax.imshow(sig_4d[:, :, z, time_idx].T, cmap="gray", origin="lower")
        ax.imshow(aif_mask[:, :, z].T, cmap="Reds",
                  alpha=0.5 * aif_mask[:, :, z].T, origin="lower")
        if aif_roi[:, :, z].any():
            ax.contour(aif_roi[:, :, z].T, levels=[0.5], colors="yellow",
                       linewidths=0.8, origin="lower")
        ax.set_title(f"Aorta ROI z={z}  (eroded fill, original outline)")
        ax.axis("off")
    fig.suptitle(f"{fmb_id}, t={time_idx}")
    fig.tight_layout()
    save_fig(fig, "aif_roi_inspection", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_aif_inspection(t, aif_signal, aif, fmb_id):
    """Notebook 'aif_inspection' figure (cell [8]), unchanged."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, aif_signal, "o-")
    axes[0].set_title("Mean aorta signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Signal Intensity(a.u)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, aif, "o-", color="crimson")
    axes[1].set_title("AIF (plasma, hematocrit-corrected)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(fmb_id)
    fig.tight_layout()
    save_fig(fig, "aif_inspection", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# =============================================================================
# main loop
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    summary = []

    for fmb_id in FMB_IDS:
        print(f'\n{"=" * 80}\nProcessing {fmb_id}\n{"=" * 80}')
        try:
            args = Args(fmb_id)

            sig_4d, _ = load_nifti_canonical(args.dce_file)
            aif_roi   = load_nifti_canonical(args.aif_mask_file)[0].astype(bool)
            t = np.arange(sig_4d.shape[-1]) * dt
            print(f"Loaded: {sig_4d.shape} volume, {aif_roi.sum()} aorta voxels")

            iters = EROSION_ITERS_OVERRIDE.get(fmb_id, EROSION_ITERS)
            aif_mask, aif_idx, labelled = build_eroded_aif_mask(aif_roi, iters)
            print(f"Labelled aorta slices : {labelled.tolist()}")
            print(f"Aorta slice (aif_idx) : {aif_idx}  -> eroded ROI on "
                  f"{aif_idx}, {aif_idx + 1}  ({iters} erosion iteration(s))")
            print(f"Voxels contributing   : {int(aif_mask.sum())} "
                  f"(of {int(aif_roi.sum())} in the original ROI)")

            aif, aif_signal = extract_aif(sig_4d, aif_mask, acq, aif_cfg, B1=B1)

            print(f"AIF signal range : [{aif_signal.min():.1f}, {aif_signal.max():.1f}]")
            print(f"AIF plasma range : [{aif.min():.4f}, {aif.max():.4f}] mM")
            print(f"AIF peak time    : {t[np.argmax(aif)]:.1f} s")
            print(f"AIF peak value   : {aif.max():.4f} mM")
            print(f"AIF baseline     : {aif[acq.nbase].mean():.6f} mM")

            plot_roi_inspection(sig_4d, aif_roi, aif_mask, aif_idx,
                                TIME_IDX_FOR_ROI_FIG, fmb_id)
            plot_aif_inspection(t, aif_signal, aif, fmb_id)

            csv_path = os.path.join(RESULTS_DIR, f"{fmb_id}_AIF_plasma.csv")
            pd.DataFrame({
                "frame":           np.arange(len(t)),
                "time_s":          t,
                "aif_signal_au":   aif_signal,
                "aif_plasma_mM":   aif,
            }).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            summary.append(dict(
                subject_id=fmb_id, aif_mask_file=os.path.basename(args.aif_mask_file),
                aif_idx=aif_idx, slices_used=f"{aif_idx},{aif_idx + 1}",
                erosion_iters=iters, n_voxels=int(aif_mask.sum()),
                n_voxels_original=int(aif_roi.sum()),
                peak_time_s=float(t[np.argmax(aif)]), peak_mM=float(aif.max()),
                baseline_mM=float(aif[acq.nbase].mean()),
                T10_blood=aif_cfg.T10, hematocrit=aif_cfg.hematocrit,
                TR=acq.TR, FA=acq.FA, r1=acq.r1,
                nbase=f"{acq.nbase.start}:{acq.nbase.stop}",
                spokes_per_frame=SPOKES_PER_FRAME, dt_s=dt, error=""))

        except Exception as e:        # one bad subject must not stop the batch
            print(f"!! {fmb_id} FAILED: {e!r}")
            summary.append(dict(subject_id=fmb_id, error=repr(e)))

    summ_path = os.path.join(RESULTS_DIR, "AIF_extraction_summary.csv")
    pd.DataFrame(summary).to_csv(summ_path, index=False)
    print(f"\nSummary: {summ_path}")


if __name__ == "__main__":
    main()