#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:40:54 2026

@author: allisonchen
"""
"""
tighten_lesion_rois.py

Per-patient pipeline: locate each patient's ROI folder, split into
AIF mask + lesion mask(s), compute a voxelwise enhancement-ratio map
from the dynamic series, and intersect each lesion mask with the
enhancement map to drop non-enhancing voxels (necrosis, fluid, mask
overdraw, etc). Mirrors the logic in 20260706_2.ipynb's
make_tight_masks(), generalized to NIfTI + a shared multi-patient
ROI root with non-standardized sub-folder naming.

find_roi_dir / classify_roi_files / _has_nifti / AIF_ROI_KEYWORDS are
copied inline below from run_basis_batch.py (no shared roi_utils module
exists yet -- these currently only live in that script on the other
server). If a shared module gets split out later, swap this block for
an import.
"""

import os
import glob
import json
import logging

import numpy as np
import nibabel as nib  # TODO: confirm this is the nifti lib in use (nibabel vs SimpleITK)
import matplotlib
matplotlib.use("Agg")  # headless -- batch job, no display
import matplotlib.pyplot as plt
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# ROI file-finding helpers -- copied from run_basis_batch.py.
# Keep these in sync by hand until they're split into a shared module.
# ---------------------------------------------------------------------
AIF_ROI_KEYWORDS = ("aif", "aorta")

def calc_enhancement_ratio(time_series, baseline_frame_idx, enhanced_frame_idx,
                             cutoff=0.1, show_histogram=False):
    """
    Adjusted for nibabel's axis convention: time is the LAST axis.

    time_series: (H, W, T) for a single slice, or (H, W, Z, T) for a
        full volume -- i.e. whatever nib.load(...).get_fdata() returns
        for a 4D dynamic series, with no axis reordering needed.
    baseline_frame_idx / enhanced_frame_idx: indices into the time axis
        (last axis), e.g. range(1, 4).
    show_histogram: if True, prints (does not save) a histogram of all
        enhancement_ratio voxel values, with a dotted line at `cutoff`
        and the percent of voxels surviving thresholding in the title.
    """
    eps = 1e-7

    if time_series.ndim not in (3, 4):
        raise ValueError(
            f"expected (H, W, T) or (H, W, Z, T), got ndim={time_series.ndim} "
            f"shape={time_series.shape}"
        )

    baseline = time_series[..., baseline_frame_idx].mean(axis=-1)
    enhanced = time_series[..., enhanced_frame_idx].mean(axis=-1)

    enhancement_ratio = (enhanced - baseline) / (baseline + eps)

    survives = baseline >= cutoff
    pct_survive = 100.0 * survives.sum() / survives.size

    enhancement_ratio[~survives] = 0

    if show_histogram:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(enhancement_ratio.ravel(), bins=100)
        ax.axvline(cutoff, color="red", linestyle="--",
                   label=f"cutoff={cutoff}")
        ax.set_xlabel("Enhancement ratio")
        ax.set_ylabel("Voxel count")
        ax.set_title(f"Enhancement ratio distribution\n"
                     f"{pct_survive:.1f}% of voxels survived baseline threshold")
        ax.legend()
        plt.show()

    return enhancement_ratio

def _has_nifti(folder: str) -> bool:
    return any(f.lower().endswith((".nii", ".nii.gz")) for f in os.listdir(folder))


def find_roi_dir(roi_base_dir: str | None, hop_id_csv: str) -> str | None:
    """
    Find this patient's ROI folder inside a shared root directory where
    every patient's sub-folder has a different name pattern but contains
    the hop_id (e.g. "FMB229ROI/ROI/"). Returns the folder that directly
    contains the .nii / .nii.gz mask files.
    """
    if not roi_base_dir or not os.path.isdir(roi_base_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(roi_base_dir, f"*{hop_id_csv}*")))
    for c in candidates:
        roi_sub = os.path.join(c, "ROI")
        if os.path.isdir(roi_sub) and _has_nifti(roi_sub):
            return roi_sub
        if os.path.isdir(c) and _has_nifti(c):
            return c
    return None


def classify_roi_files(roi_dir: str) -> tuple[str | None, list[str]]:
    """
    Split a patient's ROI folder into (aif_roi_path, [lesion_roi_paths]).
    The AIF/aorta mask is identified by filename keyword match
    (AIF_ROI_KEYWORDS). Every other NIfTI file in the folder is treated
    as an independent lesion ROI -- lesion filenames are not standardized
    across patients, so we deliberately do NOT match on lesion-specific
    naming; we just take "everything that isn't the AIF mask".
    """
    all_files = sorted(
        f for f in os.listdir(roi_dir) if f.lower().endswith((".nii", ".nii.gz"))
    )
    aif_files = [f for f in all_files if any(k in f.lower() for k in AIF_ROI_KEYWORDS)]
    lesion_files = [f for f in all_files if f not in aif_files]
    aif_path = os.path.join(roi_dir, aif_files[0]) if aif_files else None
    if len(aif_files) > 1:
        log.warning("Multiple AIF-like ROI files in %s, using first: %s", roi_dir, aif_files)
    lesion_paths = [os.path.join(roi_dir, f) for f in lesion_files]
    return aif_path, lesion_paths


# ---------------------------------------------------------------------
# Config -- flat, no argparse for now; promote to CLI/config file later
# ---------------------------------------------------------------------
ROI_BASE_DIR = "/mnt/winshare/Labspace/Projects/fastMRIbreast/Lesion_nii_Files/Lesions_nii/"       # TODO: real path
IMAGE_DIR = "/mnt/winshare/Labspace/Projects/fastMRIbreast/FMB_8spf_PCA_NII"
OUTPUT_DIR = "/mnt/winshare/Labspace/Projects/fastMRIbreast/Lesion_nii_Files/tight_masks"      # TODO: real path
QC_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "qc_summary.json")

# Series filename convention:
#   <IMAGE_DIR>/<hop_id>/NIfTI/MGRASP_temptv_<hop_id>_lambda_<SERIES_LAMBDA_STR>.nii.gz
SERIES_LAMBDA_STR = "0_003"  # TODO: confirm this is fixed across all patients

BASELINE_FRAME_IDX = range(1, 7)     # TODO: confirm per-protocol frame indices
ENHANCED_FRAME_IDX = range(28, 36)
ENHANCEMENT_CRITERIA = 0.2
CUTOFF = 0.1

SAVE_QC_FIGURES = True
QC_FIG_DIR = os.path.join(OUTPUT_DIR, "qc_figures")
QC_ONLY_IF_CHANGED = True   # skip saving a figure when tightening dropped 0 voxels

CANONICAL_DIR = os.path.join(OUTPUT_DIR, "canonical")  # resaved canonical copies, per patient

HOP_ID_LIST = ["FMB010",
    "FMB013",
    "FMB016",
    "FMB020",
    "FMB024",
    "FMB036",
    "FMB038",
    "FMB040",
    "FMB050",
    "FMB056",
    "FMB070",
    "FMB071",
    "FMB072",
    "FMB079",
    "FMB081",
    "FMB084",
    "FMB095"]  # TODO: populate from cohort CSV, e.g. pd.read_csv(...)['hop_id'].tolist()


# ---------------------------------------------------------------------
# I/O helpers -- fill in to match your actual on-disk conventions
# ---------------------------------------------------------------------
def load_nifti_canonical(path: str, save_canonical_to: str | None = None) -> tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load a NIfTI file and reorient it to closest canonical orientation.
    Used for both the dynamic series and every mask, so everything this
    script touches lives in the same canonical space -- avoids the kind
    of orientation mismatch seen previously in GRASP vs PCA overlays.

    If save_canonical_to is given, the canonical-oriented image is
    written to disk at that path (parent dirs created as needed), so
    the resaved file itself -- not just the in-memory array -- lives in
    canonical space for downstream/manual use.

    Returns (data, img) -- img is the reoriented Nifti1Image, so its
    .affine / .header can be reused directly when saving derived masks.
    """
    img = nib.as_closest_canonical(nib.load(path))
    data = img.get_fdata()
    if save_canonical_to is not None:
        os.makedirs(os.path.dirname(save_canonical_to), exist_ok=True)
        nib.save(img, save_canonical_to)
    return data, img


def load_dynamic_series(hop_id: str, save_canonical_to: str | None = None) -> tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load this patient's dynamic series and reorient to closest canonical.
    Returns (data, img) via load_nifti_canonical. If save_canonical_to is
    given, the canonical series is also persisted to disk.

    Path convention:
        <IMAGE_DIR>/<hop_id>/NIfTI/MGRASP_temptv_<hop_id>_lambda_<SERIES_LAMBDA_STR>.nii.gz
    e.g. /mnt/winshare/Labspace/Projects/fastMRIbreast/FMB_8spf_PCA_NII/FMB010/NIfTI/MGRASP_temptv_FMB010_lambda_0_003.nii.gz
    """
    series_path = os.path.join(
        IMAGE_DIR, hop_id, "NIfTI", f"MGRASP_temptv_{hop_id}_lambda_{SERIES_LAMBDA_STR}.nii.gz"
    )

    if not os.path.isfile(series_path):
        # Fallback in case the lambda string ever differs from SERIES_LAMBDA_STR --
        # glob for any lambda value under this patient's NIfTI folder rather than
        # failing outright.
        pattern = os.path.join(IMAGE_DIR, hop_id, "NIfTI", f"MGRASP_temptv_{hop_id}_lambda_*.nii.gz")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"[{hop_id}] no dynamic series found at {series_path} "
                                     f"(or matching {pattern})")
        if len(candidates) > 1:
            log.warning("[%s] multiple dynamic series matched %s, using first: %s",
                        hop_id, pattern, candidates)
        series_path = candidates[0]

    return load_nifti_canonical(series_path, save_canonical_to=save_canonical_to)


def load_mask(path: str, save_canonical_to: str | None = None) -> tuple[np.ndarray, "nib.Nifti1Image"]:
    """
    Load a NIfTI mask, reoriented to closest canonical, as a uint8 array.
    Returns (data, img) via load_nifti_canonical -- img is kept so its
    affine/header can be reused when saving the tight mask. If
    save_canonical_to is given, the canonical mask is also persisted
    to disk at that path.
    """
    data, img = load_nifti_canonical(path, save_canonical_to=save_canonical_to)
    return data.astype(np.uint8), img


def save_tight_mask(path: str, tight_mask: np.ndarray, reference_img: "nib.Nifti1Image") -> str:
    """
    Save tight_mask using the affine/header of the already-canonical
    reference image (i.e. the img returned from load_mask on the
    original ROI file) -- NOT the raw on-disk affine/header, so the
    saved mask matches the canonical orientation everything else in
    this script is working in. Returns the new path.
    """
    out_img = nib.Nifti1Image(tight_mask.astype(np.uint8), reference_img.affine, reference_img.header)
    new_path = path.replace(".nii.gz", "_tight.nii.gz").replace(".nii", "_tight.nii")
    nib.save(out_img, new_path)
    return new_path


# ---------------------------------------------------------------------
# Core per-lesion tightening step (equivalent to the notebook's inner loop)
# ---------------------------------------------------------------------
def tighten_lesion_mask(mask: np.ndarray, enhancement: np.ndarray, criteria: float) -> tuple[np.ndarray, dict]:
    """
    mask & enhancement should already be boolean-safe (0/1) before this call --
    if masks ever carry multi-class labels (e.g. benign=1, malig=2), binarize
    with `mask > 0` first. Bitwise AND on a raw label value >1 will silently
    misbehave (see note from prior review).
    """
    mask_bin = (mask > 0)
    tight_mask = mask_bin & (enhancement > criteria)

    dropped_voxels = int(mask_bin.sum() - tight_mask.sum())
    stats = {
        "orig_voxels": int(mask_bin.sum()),
        "tight_voxels": int(tight_mask.sum()),
        "dropped_voxels": dropped_voxels,
        "pct_dropped": dropped_voxels / mask_bin.sum() if mask_bin.sum() > 0 else None,
    }
    return tight_mask.astype(np.uint8), stats


# ---------------------------------------------------------------------
# QC visualization -- equivalent to the notebook's largest_diff_slice +
# contour overlay, but saved to disk instead of plt.show() for batch use.
# ---------------------------------------------------------------------
def save_qc_figure(
    series: np.ndarray,
    enhancement: np.ndarray,
    mask: np.ndarray,
    tight_mask: np.ndarray,
    baseline_frame_idx,
    out_path: str,
    hop_id: str,
    lesion_name: str,
) -> str | None:
    """
    series: (H, W, Z, T) canonical dynamic series.
    enhancement, mask, tight_mask: (H, W, Z), already shape-matched.

    NOTE ON AXIS ORDER: slice axis is LAST here (Z), not first -- this
    matches the output of calc_enhancement_ratio above, which reduces
    the trailing time axis, leaving (H, W, Z) for a 4D input. An earlier
    version of this function assumed slice-axis-first (indexing
    enhancement[largest_diff_slice] and summing axis=(1,2)), which was
    silently slicing into H instead of Z and summing over the wrong
    axes. Fixed here: index [:, :, z] and sum over axis=(0, 1).

    Picks the single slice with the largest voxel drop (same logic as
    the notebook: argmax over sum(mask - tight_mask) per slice) purely
    for the left-hand image panel. The right-hand time-intensity-curve
    panel is NOT restricted to that slice -- it uses the FULL 3D mask
    volume (every slice where the mask has voxels), since a lesion's
    overall enhancement behavior can span many slices and a single
    worst-drop slice may not represent it well.

    Saves a two-panel figure:
      left  - enhancement map in grayscale at the worst-drop slice,
              original mask contour in red, tight mask contour in blue
      right - mean signal-enhancement-ratio curve over time across the
              ENTIRE 3D ROI, computed separately for the original mask
              (red) and the tight mask (blue)

    Returns the saved path, or None if there was nothing to show
    (e.g. mask and tight_mask are identical everywhere).
    """
    mask_bin = (mask > 0).astype(np.uint8)
    diff_per_slice = np.sum(mask_bin - tight_mask, axis=(0, 1))  # sum over H, W -> (Z,)
    largest_diff_slice = int(np.argmax(diff_per_slice))

    if QC_ONLY_IF_CHANGED and diff_per_slice[largest_diff_slice] == 0:
        return None

    z = largest_diff_slice
    enh_slice = enhancement[:, :, z]
    mask_slice = mask_bin[:, :, z].astype(bool)
    tight_slice = tight_mask[:, :, z].astype(bool)

    eps = 1e-7

    def _mean_enhancement_curve_3d(mask3d: np.ndarray):
        """Mean signal within mask3d across the WHOLE volume, per timepoint."""
        mask3d = mask3d.astype(bool)
        if not mask3d.any():
            return None
        voxels = series[mask3d, :]          # (Nvox_total_across_all_slices, T)
        curve = voxels.mean(axis=0)          # (T,)
        baseline = curve[list(baseline_frame_idx)].mean()
        return (curve - baseline) / (baseline + eps)

    tic_before = _mean_enhancement_curve_3d(mask_bin)
    tic_after = _mean_enhancement_curve_3d(tight_mask)

    fig, (ax_img, ax_tic) = plt.subplots(1, 2, figsize=(14, 7))

    ax_img.imshow(enh_slice, cmap="gray", vmin=0, vmax=1)
    ax_img.contour(mask_slice, colors="red", linewidths=0.6)
    ax_img.contour(tight_slice, colors="blue", linewidths=0.6)
    ax_img.set_title(
        f"{hop_id} | {lesion_name} | slice {z} | dropped {int(diff_per_slice[z])} vox"
    )
    ax_img.axis("off")

    # TODO: consider a legend/annotation distinguishing red=original vs
    # blue=tight -- fine for now since it's fixed across every figure.

    if tic_before is not None:
        ax_tic.plot(tic_before, "r-o", markersize=3, label="original mask (3D mean)")
    if tic_after is not None:
        ax_tic.plot(tic_after, "b-o", markersize=3, label="tight mask (3D mean)")
    if tic_before is None and tic_after is None:
        ax_tic.text(0.5, 0.5, "no voxels in ROI", ha="center", va="center")
    ax_tic.set_xlabel("Time frame")
    ax_tic.set_ylabel("Enhancement ratio (baseline-normalized)")
    ax_tic.set_title("Mean ROI enhancement over time (entire 3D mask)")
    ax_tic.legend()
    ax_tic.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Per-patient driver
# ---------------------------------------------------------------------
def process_patient(hop_id: str, qc_records: list) -> None:
    roi_dir = find_roi_dir(ROI_BASE_DIR, hop_id)
    if roi_dir is None:
        log.warning("No ROI dir found for %s, skipping", hop_id)
        qc_records.append({"hop_id": hop_id, "status": "no_roi_dir"})
        return

    aif_path, lesion_paths = classify_roi_files(roi_dir)
    if aif_path is None:
        log.warning("No AIF-like ROI found for %s (dir=%s)", hop_id, roi_dir)
        # TODO: decide whether this is fatal for this patient or just noted --
        # tightening lesion masks doesn't strictly need the AIF, so probably
        # just log and continue.

    if not lesion_paths:
        log.warning("No lesion ROI files found for %s, skipping", hop_id)
        qc_records.append({"hop_id": hop_id, "status": "no_lesion_rois"})
        return

    canonical_dir = os.path.join(CANONICAL_DIR, hop_id)

    # Resave the AIF ROI as canonical too, even though it isn't used for
    # lesion tightening -- keeps every NIfTI touched for this patient
    # (series, AIF mask, lesion masks) consistently on disk in canonical
    # space, not just the ones actively used downstream.
    if aif_path is not None:
        try:
            load_mask(aif_path, save_canonical_to=os.path.join(canonical_dir, os.path.basename(aif_path)))
        except Exception as e:
            log.warning("Failed to resave AIF ROI as canonical for %s: %s", hop_id, e)

    try:
        series, series_img = load_dynamic_series(
            hop_id,
            save_canonical_to=os.path.join(
                canonical_dir, f"MGRASP_temptv_{hop_id}_lambda_{SERIES_LAMBDA_STR}.nii.gz"
            ),
        )
    except NotImplementedError:
        raise
    except Exception as e:
        log.error("Failed to load dynamic series for %s: %s", hop_id, e)
        qc_records.append({"hop_id": hop_id, "status": "series_load_failed", "error": str(e)})
        return

    enhancement = calc_enhancement_ratio(
        series,
        baseline_frame_idx=BASELINE_FRAME_IDX,
        enhanced_frame_idx=ENHANCED_FRAME_IDX,
        cutoff=CUTOFF,
    )

    patient_out_dir = os.path.join(CANONICAL_DIR, hop_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    for lesion_path in lesion_paths:
        mask, mask_img = load_mask(
            lesion_path,
            save_canonical_to=None
            )

        # TODO: sanity-check mask.shape vs enhancement.shape here --
        # ROI files from a shared drop folder are exactly where a silent
        # shape/orientation mismatch is most likely to slip in. Both are
        # now loaded canonical, so a mismatch here means an actual
        # spatial-dimension disagreement, not an orientation artifact.
        if mask.shape != enhancement.shape:
            log.warning(
                "Shape mismatch for %s / %s: mask=%s enhancement=%s -- skipping",
                hop_id, lesion_path, mask.shape, enhancement.shape,
            )
            qc_records.append({
                "hop_id": hop_id, "lesion_file": os.path.basename(lesion_path),
                "status": "shape_mismatch",
            })
            continue

        tight_mask, stats = tighten_lesion_mask(mask, enhancement, ENHANCEMENT_CRITERIA)

        out_path = os.path.join(patient_out_dir, os.path.basename(lesion_path))
        saved_path = save_tight_mask(out_path, tight_mask, reference_img=mask_img)

        qc_fig_path = None
        if SAVE_QC_FIGURES:
            lesion_stem = os.path.basename(lesion_path).replace(".nii.gz", "").replace(".nii", "")
            fig_out_path = os.path.join(QC_FIG_DIR, hop_id, f"{lesion_stem}_qc.png")
            qc_fig_path = save_qc_figure(
                series=series,
                enhancement=enhancement,
                mask=mask,
                tight_mask=tight_mask,
                baseline_frame_idx=BASELINE_FRAME_IDX,
                out_path=fig_out_path,
                hop_id=hop_id,
                lesion_name=lesion_stem,
            )

        qc_records.append({
            "hop_id": hop_id,
            "lesion_file": os.path.basename(lesion_path),
            "tight_mask_path": saved_path,
            "qc_figure_path": qc_fig_path,
            "aif_found": aif_path is not None,
            "status": "ok",
            **stats,
        })


# ---------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qc_records = []
    for hop_id in tqdm(HOP_ID_LIST, desc="Tightening lesion ROIs"):
        process_patient(hop_id, qc_records)

    with open(QC_SUMMARY_PATH, "w") as f:
        json.dump(qc_records, f, indent=2)

    n_ok = sum(1 for r in qc_records if r.get("status") == "ok")
    log.info("Done: %d/%d lesion masks tightened successfully", n_ok, len(qc_records))


if __name__ == "__main__":
    main()