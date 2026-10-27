#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
extract_aif_roi_idif.py
==================
Batch AIF extraction over a whole data root, run BEFORE the ROI fits.

Differences from extract_aifs_auto.py:
  * subjects are DISCOVERED by walking DATA_ROOT -- no FMB_IDS list
  * the AIF is taken from the FireVoxel IDIF ROI (idif_case###.nii) when one
    exists, using the whole ROI (all labelled slices, no erosion) with
    TOP_FRAC_IDIF voxel selection
  * when no usable IDIF ROI exists, it falls back to the hand-drawn aorta
    ROI (*orta*.nii*) using extract_aifs.py's construction: eroded first
    labelled slice, applied to aif_idx and aif_idx+1
  * when BOTH ROIs exist, both curves are extracted and the IDIF-vs-aorta
    comparison figures + per-frame CSV are written, but only the primary
    (IDIF) curve goes into the canonical AIF CSV

Per subject:
  1. locate the 4D DCE volume and whichever ROIs are present
  2. build the IDIF mask and/or the eroded aorta mask
  3. baseline window (priority: NBASE_OVERRIDE > auto > fallback).
     auto: bolus arrival = first frame of the mean ROI signal above
           base + ARRIVAL_FRAC*(peak - base) that stays above for N_CONFIRM
           frames; baseline ends BASE_MARGIN frames before it.
     With NBASE_MODE = "shared_primary" the primary source's window is used
     for BOTH methods, so the comparison reflects the mask, not the window.
  4. extract_aif(...) -- the SAME function 06_roi_fitting.ipynb uses:
     voxel selection by top_frac -> mean signal -> SPGR inversion (blood T1)
     -> /(1 - Hct)
  5. PEAK GUARD: if the resulting plasma peak exceeds PEAK_MAX_MM, dilute the
     ROI with lower-enhancing voxels and re-extract --
       idif  : raise top_frac in TOP_FRAC_STEP steps up to TOP_FRAC_MAX
       aorta : peel back the erosion one step at a time down to 0 iterations
     Both knobs add the least-enhancing voxels back in, so the mean peak
     normally falls, but neither is guaranteed monotone (extract_aif ranks by
     peak / mean(nbase), not by absolute peak). The loops therefore stop at
     the FIRST setting that clears the ceiling and record the whole path.
     A subject that trips this guard should be treated as FLAGGED, not fixed:
     a >10 mM plasma peak (~6 mM whole blood after the /(1 - Hct) step) is
     more often the SPGR inversion losing conditioning near saturation, a
     wrong T10/B1, or inflow/pulsatility artifact than a too-selective mask.
  6. write, in RESULTS_DIR:
       {id}_AIF_plasma.csv                canonical curve, SAME columns and
                                          filename whatever the source was
                                          (extra 'aif_source' column)
       {id}_aif_inspection.png            primary signal + plasma curves
       {id}_aif_roi_inspection.png        mask(s) on the anatomy
       {id}_aifcmp_curves.png             both methods           (if both)
       {id}_aifcmp_difference.png         diff/ratio/scatter/shape (if both)
       {id}_AIF_method_comparison.csv     both curves per frame  (if both)
       AIF_extraction_summary.csv         one row per subject

  NOTE on top_frac: extract_aif ranks voxels by peak / mean(nbase), so the
  baseline window must exist before selection. nbase is therefore detected
  on the WHOLE-mask mean signal and then handed to extract_aif for both the
  selection and the inversion -- it is not re-detected on the selected
  subset, and it is NOT re-detected inside the peak guard either (bolus
  arrival is a timing property; letting the window chase the mask would
  confound the IDIF-vs-aorta comparison).

Sequence / AIF parameters match 06_roi_fitting.ipynb cells [4] and [8].

Run:
    PYTHONPATH=src python scripts/extract_aif_roi.py
"""

import os
import re
from dataclasses import replace

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

DATA_ROOT   = "/mnt/labspace/users/allison/fastMRI_data/tight_masks_4/canonical"
RESULTS_DIR = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"

# --- subject discovery ------------------------------------------------------
SUBJECT_RE   = re.compile(r"^FMB\d+$")   # directory names to treat as subjects
SUBJECT_SKIP = set()                     # e.g. {"FMB001"} to exclude
SUBJECT_ONLY = set()                     # non-empty = restrict to these ids

# filename fragments used to locate each file inside the subject folder
# (matched case-insensitively; exactly one hit required, no guessing)
DCE_FRAGMENTS   = ("_tv",)     # FMB001_tv1em05c_ttgv3em04z.nii.gz
AORTA_FRAGMENTS = ("orta",)    # 56_Aorta_RotOrtho_6.nii, aorta_case288.nii
IDIF_FRAGMENTS  = ("idif",)    # idif_case001.nii

# --- acquisition: 06_roi_fitting.ipynb cell [4] -----------------------------
SPOKES_PER_FRAME = 8
NSPOKE   = 288                # total spokes across the whole scan
SCANTIME = 150                # total scan duration, s

# --- baseline window --------------------------------------------------------
AUTO_NBASE     = True             # detect per subject from the ROI curve
NBASE_FALLBACK = slice(0, 7)      # used only if AUTO_NBASE = False
NBASE_OVERRIDE = {                # per-subject manual windows, win over auto
    # "FMB194": slice(0, 9),
}
# "shared_primary" : detect on the PRIMARY source's curve, use that window for
#                    both methods (isolates the ROI effect -- recommended)
# "per_method"     : each method detects its own window
NBASE_MODE     = "shared_primary"
ARRIVAL_FRAC   = 0.10   # arrival = first frame > base + 10% of (peak - base)
N_MIN_BASE     = 5      # minimum baseline length (frames)
N_CONFIRM      = 2      # frames that must stay above the threshold
BASE_MARGIN    = 1      # frames dropped immediately before arrival
BASE_CV_FLAG   = 0.10   # flag baselines noisier than this CV

# --- baseline drift ---------------------------------------------------------
DRIFT_T_FLAG             = 3.0    # flag if |slope| > 3 x its standard error
APPLY_LATE_BASE_ON_DRIFT = False  # True: if flagged, S0 from the last
                                  # LATE_BASE_K baseline frames only
LATE_BASE_K              = 4

# --- masks ------------------------------------------------------------------
EROSION_ITERS = 2                     # aorta method only
EROSION_ITERS_OVERRIDE = {            # explicit per-subject exceptions
    # "FMB046": 1,
}
AUTO_ERODE_BACKOFF = True             # if N iterations empty the slice, retry
                                      # N-1, N-2, ... 0 and record what was used
KEEP_OTHER_LABELLED_SLICES = False    # False: ONLY aif_idx and aif_idx+1
                                      # contribute to the aorta AIF
ERODE_IDIF  = 0                       # 0 = use the IDIF ROI exactly as given
IDIF_SLICES = "all"                   # "all", or "first2" to mimic the aorta
                                      # method's two-slice restriction

# --- AIF: 06_roi_fitting.ipynb cell [8] -------------------------------------
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE_FALLBACK)

aif_cfg = AifConfig(
    T10        = 1779.0,
    hematocrit = 0.4,
    top_frac   = 1.0,     # overridden per method below
    nbase      = acq.nbase,
)
B1 = 1.0                  # SubjectConfig(B1=1.0) in the notebook

TOP_FRAC = {
    "idif":  0.85,         # keep the top 80% enhancing voxels of the IDIF ROI
    "aorta": 1.0,         # extract_aifs.py behaviour -- average every voxel
}

# --- peak-concentration guard -----------------------------------------------
PEAK_MAX_MM        = 10.0   # plasma-peak ceiling; above this, dilute the ROI
TOP_FRAC_STEP      = 0.05   # idif : enlarge the kept fraction in these steps
TOP_FRAC_MAX       = 1.0
EROSION_STEP       = 1      # aorta: peel back this many iterations per retry
PEAK_GUARD_METHODS = ("idif", "aorta")   # set to () to disable entirely

# --- comparison windows -----------------------------------------------------
AUC_EARLY_S  = 60.0     # early-AUC window, s from t=0
SHAPE_FROM_S = 0.0      # peak-normalised shape comparison starts here

TIME_IDX_FOR_ROI_FIG = 15     # frame shown in the ROI inspection figure
MAX_SLICES_IN_FIG    = 6
SHOW_FIGS = False             # True to also plt.show() interactively

if not SHOW_FIGS:
    matplotlib.use("Agg")

METHODS = ("aorta", "idif")
COLORS  = {"aorta": "crimson", "idif": "tab:blue"}


# =============================================================================
# discovery / path resolution
# =============================================================================

def discover_subjects(root):
    """Every immediate subdirectory of root whose name matches SUBJECT_RE."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    ids = sorted(d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d)) and SUBJECT_RE.match(d))
    if SUBJECT_ONLY:
        ids = [i for i in ids if i in SUBJECT_ONLY]
    return [i for i in ids if i not in SUBJECT_SKIP]


def find_files(subj_dir, fragments):
    """All .nii/.nii.gz in subj_dir whose name contains every fragment
    (case-insensitive), sorted."""
    return sorted(
        os.path.join(subj_dir, f) for f in os.listdir(subj_dir)
        if f.lower().endswith((".nii", ".nii.gz"))
        and all(fr.lower() in f.lower() for fr in fragments))


def find_one(subj_dir, fragments, what):
    """Exactly one match, or raise. Refuses to guess."""
    hits = find_files(subj_dir, fragments)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one {what} file matching {fragments} in "
            f"{subj_dir}, found {len(hits)}: {[os.path.basename(h) for h in hits]}")
    return hits[0]


def find_optional(subj_dir, fragments, what):
    """(path, note). None if absent; still raises if ambiguous, because an
    ambiguous ROI is a labelling problem, not a missing file."""
    hits = find_files(subj_dir, fragments)
    if not hits:
        return None, f"no {what} file matching {fragments}"
    if len(hits) > 1:
        raise FileNotFoundError(
            f"expected at most one {what} file matching {fragments} in "
            f"{subj_dir}, found {len(hits)}: {[os.path.basename(h) for h in hits]}")
    return hits[0], ""


# =============================================================================
# mask construction
# =============================================================================

def build_aorta_mask(roi, iters, backoff=AUTO_ERODE_BACKOFF):
    """
    extract_aifs.py's mask: eroded first labelled slice, placed on aif_idx and
    aif_idx+1. With backoff, an erosion that empties the slice is retried with
    fewer iterations instead of failing the subject.
    Returns (mask, aif_idx, labelled_slices, iters_used).
    """
    labelled = np.where(roi.sum(axis=(0, 1)) > 0)[0]
    if labelled.size == 0:
        raise ValueError("aorta ROI is empty")
    aif_idx = int(labelled[0])
    if aif_idx + 1 >= roi.shape[2]:
        raise ValueError(f"aif_idx+1 = {aif_idx + 1} is outside the volume")

    tried = range(iters, -1, -1) if backoff else (iters,)
    eroded_2d, used = None, None
    for k in tried:
        cand = (roi[:, :, aif_idx] if k == 0 else
                binary_erosion(roi[:, :, aif_idx],
                               structure=np.ones((3, 3)), iterations=k))
        if cand.any():
            eroded_2d, used = cand, k
            break
    if eroded_2d is None:
        raise ValueError(
            f"erosion with {iters} iteration(s) removed every voxel on slice "
            f"{aif_idx} -- lower EROSION_ITERS for this subject")
    if used != iters:
        print(f"  ! erosion backed off {iters} -> {used} iteration(s) on "
              f"slice {aif_idx}")

    mask = roi.copy().astype(bool) if KEEP_OTHER_LABELLED_SLICES \
        else np.zeros_like(roi, dtype=bool)
    mask[:, :, aif_idx]     = eroded_2d
    mask[:, :, aif_idx + 1] = eroded_2d
    return mask, aif_idx, labelled, used


def build_idif_mask(roi, erode_iters=ERODE_IDIF, slices=IDIF_SLICES):
    """The FireVoxel IDIF ROI, used whole by default."""
    labelled = np.where(roi.sum(axis=(0, 1)) > 0)[0]
    if labelled.size == 0:
        raise ValueError("IDIF ROI is empty")

    mask = roi.copy().astype(bool)

    if slices == "first2":
        keep = set(int(z) for z in labelled[:2])
        for z in labelled:
            if int(z) not in keep:
                mask[:, :, int(z)] = False
    elif slices != "all":
        raise ValueError(f"IDIF_SLICES must be 'all' or 'first2', got {slices!r}")

    if erode_iters:
        for z in np.where(mask.sum(axis=(0, 1)) > 0)[0]:
            mask[:, :, int(z)] = binary_erosion(
                mask[:, :, int(z)], structure=np.ones((3, 3)),
                iterations=erode_iters)
        if not mask.any():
            raise ValueError(
                f"erosion with {erode_iters} iteration(s) emptied the IDIF ROI")

    return mask, int(labelled[0]), labelled, erode_iters


def mask_overlap(a, b):
    """Voxel overlap between the two contributing masks."""
    inter = int(np.logical_and(a, b).sum())
    na, nb = int(a.sum()), int(b.sum())
    return dict(n_overlap=inter,
                dice=2.0 * inter / (na + nb) if (na + nb) else np.nan,
                frac_aorta_in_idif=inter / na if na else np.nan,
                frac_idif_in_aorta=inter / nb if nb else np.nan)


# =============================================================================
# baseline helpers (identical to extract_aifs.py)
# =============================================================================

def detect_nbase(aif_signal, frac=ARRIVAL_FRAC, n_min=N_MIN_BASE,
                 n_confirm=N_CONFIRM, margin=BASE_MARGIN, n_iter=3):
    """
    Baseline window = frames before bolus arrival in the mean ROI signal.
    Returns (slice(0, stop), arrival_frame).
    """
    s = np.asarray(aif_signal, float)
    peak_idx = int(np.argmax(s))
    if peak_idx < n_min + margin + 1:
        raise ValueError(f"peak at frame {peak_idx}: too early for a "
                         f"{n_min}-frame baseline")
    stop, arrival = n_min, peak_idx
    for _ in range(n_iter):
        base = float(np.median(s[:stop]))
        thresh = base + frac * (s[peak_idx] - base)
        above = s[:peak_idx + 1] > thresh
        arrival = next((i for i in range(n_min, peak_idx + 1)
                        if above[i:i + n_confirm].all()), peak_idx)
        new_stop = max(n_min, arrival - margin)
        if new_stop == stop:
            break
        stop = new_stop
    return slice(0, stop), arrival


def baseline_cv(sig, nbase):
    y = np.asarray(sig[nbase], float)
    return float(np.std(y) / np.mean(y))


def baseline_drift(sig, nbase):
    """
    Straight-line fit over the baseline window.
    Returns (slope as % of baseline per frame, slope / its standard error).
    NaN for windows shorter than 4 frames.
    """
    y = np.asarray(sig[nbase], float)
    if y.size < 4:
        return np.nan, np.nan
    x = np.arange(y.size, dtype=float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se_b = np.sqrt(np.sum(resid ** 2) / (y.size - 2) / np.sum((x - x.mean()) ** 2))
    t = b / se_b if se_b > 0 else np.inf
    return float(100.0 * b / y.mean()), float(t)


# =============================================================================
# metrics
# =============================================================================

def curve_metrics(t, aif, aif_signal, nbase):
    early = t <= AUC_EARLY_S
    return dict(
        peak_mM        = float(aif.max()),
        peak_time_s    = float(t[int(np.argmax(aif))]),
        baseline_mM    = float(aif[nbase].mean()),
        auc_full_mM_s  = float(np.trapezoid(aif, t)),
        auc_early_mM_s = float(np.trapezoid(aif[early], t[early])),
        signal_min     = float(aif_signal.min()),
        signal_max     = float(aif_signal.max()),
        baseline_cv    = baseline_cv(aif_signal, nbase),
    )


def difference_metrics(t, a_aorta, a_idif):
    """Scale-sensitive and scale-free comparisons, kept separate on purpose:
    an amplitude offset propagates ~inversely into Ktrans/vp, a shape change
    does not."""
    diff  = a_idif - a_aorta
    denom = np.maximum(np.abs(a_aorta), 1e-9)

    sel = t >= SHAPE_FROM_S
    n_aorta = a_aorta / a_aorta.max()
    n_idif  = a_idif / a_idif.max()
    shape_rmse = float(np.sqrt(np.mean((n_idif[sel] - n_aorta[sel]) ** 2)))

    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(a_aorta, a_idif)[0, 1])

    return dict(
        max_abs_diff_mM     = float(np.max(np.abs(diff))),
        rmse_mM             = float(np.sqrt(np.mean(diff ** 2))),
        mean_signed_diff_mM = float(np.mean(diff)),
        median_abs_pct_diff = float(np.median(100.0 * np.abs(diff) / denom)),
        peak_ratio_idif_over_aorta = float(a_idif.max() / a_aorta.max()),
        auc_ratio_idif_over_aorta  = float(np.trapezoid(a_idif, t)
                                           / np.trapezoid(a_aorta, t)),
        shape_rmse_peaknorm = shape_rmse,
        pearson_r           = corr,
    )


# =============================================================================
# figures
# =============================================================================

def save_fig(fig, name, fmb_id):
    path = os.path.join(RESULTS_DIR, f"{fmb_id}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def plot_roi_inspection(sig_4d, masks, rois, time_idx, fmb_id):
    """Every contributing mask on every slice any of them touches, with the
    original (un-eroded) ROI outlined."""
    zs = sorted(set().union(*[
        set(np.where(m.sum(axis=(0, 1)) > 0)[0].tolist()) for m in masks.values()]))
    if len(zs) > MAX_SLICES_IN_FIG:
        print(f"  (showing {MAX_SLICES_IN_FIG} of {len(zs)} labelled slices)")
        zs = zs[:MAX_SLICES_IN_FIG]

    n = len(zs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), squeeze=False)
    for ax, z in zip(axes[0], zs):
        ax.imshow(sig_4d[:, :, z, time_idx].T, cmap="gray", origin="lower")
        bits = []
        for m in METHODS:
            if m not in masks:
                continue
            sl = masks[m][:, :, z].T
            if sl.any():
                ax.imshow(sl, cmap="Reds" if m == "aorta" else "Blues",
                          alpha=0.45 * sl, origin="lower")
            if rois.get(m) is not None and rois[m][:, :, z].any():
                ax.contour(rois[m][:, :, z].T.astype(float), levels=[0.5],
                           colors="yellow" if m == "aorta" else "deepskyblue",
                           linewidths=0.9, origin="lower")
            bits.append(f"{m} n={int(masks[m][:, :, z].sum())}")
        ax.set_title(f"z={z}   " + ", ".join(bits), fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{fmb_id}, t={time_idx}   contributing mask (fill), "
                 f"original ROI (outline)")
    fig.tight_layout()
    save_fig(fig, "aif_roi_inspection", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_aif_inspection(t, r, fmb_id, source):
    """Notebook 'aif_inspection' figure for the primary curve."""
    nb, arrival = r["nbase"], r["arrival"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, r["aif_signal"], "o-")
    axes[0].set_title(f"Mean ROI signal ({source}, "
                      f"{r['n_voxels_used']}/{r['n_voxels']} voxels)")
    axes[0].set_ylabel("Signal Intensity (a.u.)")

    axes[1].plot(t, r["aif"], "o-", color=COLORS[source])
    axes[1].set_title("AIF (plasma, hematocrit-corrected)")
    axes[1].set_ylabel("Concentration (mM)")

    for ax in axes:
        ax.axvspan(t[nb.start], t[nb.stop - 1], color="0.6", alpha=0.25,
                   label=f"baseline frames {nb.start}-{nb.stop - 1}")
        if arrival is not None:
            ax.axvline(t[arrival], ls="--", color="0.3", lw=1,
                       label=f"arrival frame {arrival}")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{fmb_id}  (AIF source: {source})")
    fig.tight_layout()
    save_fig(fig, "aif_inspection", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_curves(t, res, fmb_id, primary):
    """Signal + plasma curves, both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in METHODS:
        axes[0].plot(t, res[m]["aif_signal"], "o-", ms=3.5, color=COLORS[m],
                     label=f"{m} ({res[m]['n_voxels_used']}/{res[m]['n_voxels']} "
                           f"voxels, top_frac={res[m]['top_frac']})")
        axes[1].plot(t, res[m]["aif"], "o-", ms=3.5, color=COLORS[m],
                     label=f"{m}: peak {res[m]['aif'].max():.3f} mM")
    axes[0].set_title("Mean ROI signal")
    axes[0].set_ylabel("Signal intensity (a.u.)")
    axes[1].set_title("AIF (plasma, hematocrit-corrected)")
    axes[1].set_ylabel("Concentration (mM)")
    nb = res[primary]["nbase"]
    for ax in axes:
        ax.axvspan(t[nb.start], t[nb.stop - 1], color="0.6", alpha=0.2,
                   label=f"baseline {nb.start}-{nb.stop - 1}")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{fmb_id}: IDIF vs aorta ROI  (primary: {primary})")
    fig.tight_layout()
    save_fig(fig, "aifcmp_curves", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


def plot_difference(t, res, dm, fmb_id):
    a_a, a_i = res["aorta"]["aif"], res["idif"]["aif"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    ax = axes[0, 0]
    ax.plot(t, a_i - a_a, "o-", ms=3.5, color="0.25")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Difference (IDIF - aorta)")
    ax.set_ylabel("Concentration (mM)")

    ax = axes[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(a_a) > 1e-6, a_i / a_a, np.nan)
    ax.plot(t, ratio, "o-", ms=3.5, color="0.25")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylim(0, 2)
    ax.set_title("Ratio (IDIF / aorta), frames with |aorta| > 1e-6 mM")
    ax.set_ylabel("Ratio")

    ax = axes[1, 0]
    ax.plot(a_a, a_i, "o", ms=4, color="0.25")
    lim = [0, max(a_a.max(), a_i.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("aorta (mM)")
    ax.set_ylabel("IDIF (mM)")
    ax.set_title(f"Frame-by-frame, r = {dm['pearson_r']:.4f}, "
                 f"peak ratio = {dm['peak_ratio_idif_over_aorta']:.3f}")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for m in METHODS:
        a = res[m]["aif"]
        ax.plot(t, a / a.max(), "o-", ms=3.5, color=COLORS[m], label=m)
    ax.set_title(f"Peak-normalised shape, RMSE = {dm['shape_rmse_peaknorm']:.4f}")
    ax.set_ylabel("Concentration / peak")
    ax.legend(fontsize=8)

    for ax in axes.ravel()[:2].tolist() + [axes[1, 1]]:
        ax.set_xlabel("Time (s)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{fmb_id}: IDIF vs aorta ROI")
    fig.tight_layout()
    save_fig(fig, "aifcmp_difference", fmb_id)
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)


# =============================================================================
# per-subject work
# =============================================================================

def load_roi(path, shape3):
    roi = load_nifti_canonical(path)[0].astype(bool)
    if roi.shape != shape3:
        raise ValueError(f"ROI shape {roi.shape} != DCE spatial shape {shape3}")
    return roi


def choose_nbase(fmb_id, mean_sig):
    """(nbase, arrival, source) for one method's whole-mask mean signal."""
    if fmb_id in NBASE_OVERRIDE:
        return NBASE_OVERRIDE[fmb_id], None, "override"
    if AUTO_NBASE:
        nb, arr = detect_nbase(mean_sig)
        return nb, arr, "auto"
    return NBASE_FALLBACK, None, "fallback"


def extract_one(sig_4d, mask, nbase, top_frac, t):
    """extract_aif + QC metrics for one mask / one baseline window."""
    acq_s     = replace(acq, nbase=nbase)
    aif_cfg_s = replace(aif_cfg, nbase=nbase, top_frac=top_frac)
    aif, aif_signal = extract_aif(sig_4d, mask, acq_s, aif_cfg_s, B1=B1)
    drift_pct, drift_t = baseline_drift(aif_signal, nbase)
    n_mask = int(mask.sum())
    # mirrors extract_aif's own selection rule, for the record only
    n_used = n_mask if top_frac >= 1.0 else max(1, int(np.ceil(top_frac * n_mask)))
    return dict(aif=aif, aif_signal=aif_signal, nbase=nbase,
                top_frac=top_frac, n_voxels=n_mask, n_voxels_used=n_used,
                drift_pct=drift_pct, drift_t=drift_t,
                drift_flag=bool(np.isfinite(drift_t) and abs(drift_t) > DRIFT_T_FLAG),
                **curve_metrics(t, aif, aif_signal, nbase))


def guard_idif(sig_4d, mask, nbase, t, r):
    """
    Enlarge top_frac in TOP_FRAC_STEP steps until the plasma peak is at or
    below PEAK_MAX_MM, or until top_frac reaches TOP_FRAC_MAX.

    The mask is untouched -- top_frac is applied inside extract_aif -- so each
    retry is just a re-selection, not a re-eroded ROI.
    Returns (result, [(top_frac, peak_mM), ...]).
    """
    hist = [(r["top_frac"], r["peak_mM"])]
    tf = r["top_frac"]
    while r["peak_mM"] > PEAK_MAX_MM and tf < TOP_FRAC_MAX - 1e-12:
        tf = min(TOP_FRAC_MAX, round(tf + TOP_FRAC_STEP, 6))
        r = extract_one(sig_4d, mask, nbase, tf, t)
        hist.append((tf, r["peak_mM"]))
    return r, hist


def guard_aorta(sig_4d, roi, nbase, t, r, iters_used):
    """
    Peel back the erosion one step at a time until the plasma peak is at or
    below PEAK_MAX_MM, or until 0 iterations (the ROI exactly as drawn).

    backoff=False on purpose: iterations only ever DECREASE here, so a slice
    that was non-empty at iters_used cannot be emptied by fewer iterations,
    and an implicit extra back-off would desync the recorded value.
    Returns (result, history, mask_or_None, iters). mask is None if no retry
    ran (already at 0 iterations).
    """
    hist = [(iters_used, r["peak_mM"])]
    k, mask = iters_used, None
    while r["peak_mM"] > PEAK_MAX_MM and k > 0:
        k = max(0, k - EROSION_STEP)
        mask, _, _, _ = build_aorta_mask(roi, k, backoff=False)
        r = extract_one(sig_4d, mask, nbase, TOP_FRAC["aorta"], t)
        hist.append((k, r["peak_mM"]))
    return r, hist, mask, k


def process_subject(fmb_id, subj_dir, dt):
    """Returns the summary row for one subject. Raises on unrecoverable input."""
    dce_file = find_one(subj_dir, DCE_FRAGMENTS, "DCE")
    sig_4d, _ = load_nifti_canonical(dce_file)
    if sig_4d.ndim != 4:
        raise ValueError(f"DCE volume is {sig_4d.ndim}-D, expected 4-D")
    t = np.arange(sig_4d.shape[-1]) * dt
    shape3 = sig_4d.shape[:3]
    print(f"DCE   : {os.path.basename(dce_file)}  {sig_4d.shape}")

    idif_file,  idif_note  = find_optional(subj_dir, IDIF_FRAGMENTS, "IDIF ROI")
    aorta_file, aorta_note = find_optional(subj_dir, AORTA_FRAGMENTS, "aorta ROI")

    # ---- masks: each source is optional and independently recoverable ------
    masks, rois, info, notes = {}, {}, {}, {}
    iters_req = EROSION_ITERS_OVERRIDE.get(fmb_id, EROSION_ITERS)

    for m, path, builder, note in (
            ("idif",  idif_file,  build_idif_mask,  idif_note),
            ("aorta", aorta_file, build_aorta_mask, aorta_note)):
        if path is None:
            notes[m] = note
            print(f"{m:5s} : -- {note}")
            continue
        try:
            roi = load_roi(path, shape3)
            if m == "aorta":
                mask, idx, lab, used = builder(roi, iters_req)
                slices_used = f"{idx},{idx + 1}"
            else:
                mask, idx, lab, used = builder(roi)
                slices_used = ",".join(str(int(z)) for z in
                                       np.where(mask.sum(axis=(0, 1)) > 0)[0])
            masks[m], rois[m] = mask, roi
            info[m] = dict(file=os.path.basename(path), aif_idx=idx,
                           labelled=lab, slices_used=slices_used,
                           erosion_iters=used, n_roi=int(roi.sum()))
            notes[m] = ""
            print(f"{m:5s} : {os.path.basename(path)}  labelled "
                  f"{lab.tolist()} -> contributing {slices_used}  "
                  f"({int(mask.sum())} of {int(roi.sum())} voxels, "
                  f"{used} erosion iter)")
        except Exception as e:
            notes[m] = f"{m} ROI unusable: {e!r}"
            print(f"{m:5s} : !! {notes[m]} -- skipping this source")

    if not masks:
        raise FileNotFoundError(
            "no usable AIF ROI: " + "; ".join(v for v in notes.values() if v))

    primary = "idif" if "idif" in masks else "aorta"
    print(f"Primary AIF source : {primary}")

    # ---- baseline windows --------------------------------------------------
    mean_sig = {m: sig_4d[masks[m]].mean(axis=0) for m in masks}
    detected = {}
    for m in masks:
        nb, arr, src = choose_nbase(fmb_id, mean_sig[m])
        detected[m] = (nb, arr, src)
        arr_txt = (f", arrival frame {arr} (t={t[arr]:.1f}s)"
                   if arr is not None else "")
        print(f"{m:5s} : baseline ({src}) frames {nb.start}-{nb.stop - 1}{arr_txt}")

    if NBASE_MODE == "shared_primary":
        nbase_used = {m: detected[primary][0] for m in masks}
    elif NBASE_MODE == "per_method":
        nbase_used = {m: detected[m][0] for m in masks}
    else:
        raise ValueError(f"unknown NBASE_MODE {NBASE_MODE!r}")

    # optional late-baseline switch, driven by the PRIMARY source's drift
    nbase_src = detected[primary][2]
    dp, dtt = baseline_drift(mean_sig[primary], nbase_used[primary])
    if (APPLY_LATE_BASE_ON_DRIFT and np.isfinite(dtt) and abs(dtt) > DRIFT_T_FLAG
            and nbase_src == "auto"):
        nb = nbase_used[primary]
        if nb.stop - LATE_BASE_K >= 0:
            late = slice(nb.stop - LATE_BASE_K, nb.stop)
            nbase_used = {m: (late if (m == primary or
                                       NBASE_MODE == "shared_primary")
                              else nbase_used[m]) for m in masks}
            nbase_src = "auto-late (drift)"
            print(f"  ! drift slope/SE = {dtt:+.1f}: S0 from frames "
                  f"{late.start}-{late.stop - 1}")
    print(f"NBASE_MODE = {NBASE_MODE} ({nbase_src}): " + ", ".join(
        f"{m} {nbase_used[m].start}-{nbase_used[m].stop - 1}" for m in masks))

    # ---- extraction (+ peak guard) ----------------------------------------
    res = {}
    for m in masks:
        r = extract_one(sig_4d, masks[m], nbase_used[m], TOP_FRAC[m], t)
        hist = [(TOP_FRAC[m] if m == "idif" else info[m]["erosion_iters"],
                 r["peak_mM"])]

        if m in PEAK_GUARD_METHODS and r["peak_mM"] > PEAK_MAX_MM:
            if m == "idif":
                r, hist = guard_idif(sig_4d, masks[m], nbase_used[m], t, r)
            else:
                r, hist, new_mask, k = guard_aorta(
                    sig_4d, rois[m], nbase_used[m], t, r,
                    info[m]["erosion_iters"])
                if new_mask is not None:
                    masks[m] = new_mask               # figures/overlap follow
                    info[m]["erosion_iters"] = k
            knob = "top_frac" if m == "idif" else "erosion_iters"
            print(f"  ! {m} peak > {PEAK_MAX_MM:g} mM, {knob}: "
                  + " -> ".join(f"{v:g} ({p:.2f} mM)" for v, p in hist)
                  + ("" if r["peak_mM"] <= PEAK_MAX_MM else
                     f"   <-- STILL ABOVE {PEAK_MAX_MM:g} mM at the limit"))

        r["peak_guard_steps"] = len(hist) - 1
        r["peak_guard_ok"]    = bool(r["peak_mM"] <= PEAK_MAX_MM)
        r["arrival"]          = detected[m][1]
        r["nbase_source"]     = nbase_src if m == primary else detected[m][2]
        res[m] = r
        print(f"{m:5s} : peak {r['peak_mM']:.4f} mM at {r['peak_time_s']:.1f}s, "
              f"baseline {r['baseline_mM']:.6f} mM, AUC "
              f"{r['auc_full_mM_s']:.2f} mM.s, baseline CV {r['baseline_cv']:.3f}"
              f"{'  <-- NOISY' if r['baseline_cv'] > BASE_CV_FLAG else ''}"
              f"{'  <-- DRIFT' if r['drift_flag'] else ''}"
              f"{'' if r['peak_guard_ok'] else '  <-- PEAK CAP NOT MET'}")

    both = len(res) == 2
    dm = difference_metrics(t, res["aorta"]["aif"], res["idif"]["aif"]) if both else {}
    ov = mask_overlap(masks["aorta"], masks["idif"]) if both else {}
    if both:
        print(f"Mask overlap  : {ov['n_overlap']} voxels, Dice {ov['dice']:.3f}, "
              f"{100 * ov['frac_aorta_in_idif']:.0f}% of aorta inside IDIF")
        print(f"IDIF vs aorta : peak ratio "
              f"{dm['peak_ratio_idif_over_aorta']:.3f}, AUC ratio "
              f"{dm['auc_ratio_idif_over_aorta']:.3f}, median |diff| "
              f"{dm['median_abs_pct_diff']:.1f}%, shape RMSE "
              f"{dm['shape_rmse_peaknorm']:.4f}")

    # ---- figures -----------------------------------------------------------
    t_idx = min(TIME_IDX_FOR_ROI_FIG, sig_4d.shape[-1] - 1)
    plot_roi_inspection(sig_4d, masks, rois, t_idx, fmb_id)
    plot_aif_inspection(t, res[primary], fmb_id, primary)
    if both:
        plot_curves(t, res, fmb_id, primary)
        plot_difference(t, res, dm, fmb_id)

    # ---- CSVs --------------------------------------------------------------
    # canonical curve -- SAME name and columns whatever the source was
    p = res[primary]
    csv_path = os.path.join(RESULTS_DIR, f"{fmb_id}_AIF_plasma.csv")
    pd.DataFrame({
        "frame":         np.arange(len(t)),
        "time_s":        t,
        "aif_signal_au": p["aif_signal"],
        "aif_plasma_mM": p["aif"],
        "aif_source":    primary,
    }).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if both:
        cmp_path = os.path.join(RESULTS_DIR, f"{fmb_id}_AIF_method_comparison.csv")
        pd.DataFrame({
            "frame":            np.arange(len(t)),
            "time_s":           t,
            "aorta_signal_au":  res["aorta"]["aif_signal"],
            "idif_signal_au":   res["idif"]["aif_signal"],
            "aorta_plasma_mM":  res["aorta"]["aif"],
            "idif_plasma_mM":   res["idif"]["aif"],
            "diff_mM":          res["idif"]["aif"] - res["aorta"]["aif"],
        }).to_csv(cmp_path, index=False)
        print(f"Saved: {cmp_path}")

    # ---- summary row -------------------------------------------------------
    # top-level columns describe the PRIMARY curve, so downstream code that
    # reads nbase_start / arrival_frame keeps working unchanged
    row = dict(
        subject_id=fmb_id, aif_source=primary,
        has_idif=("idif" in res), has_aorta=("aorta" in res),
        compared=both,
        dce_file=os.path.basename(dce_file),
        idif_file=info.get("idif", {}).get("file", ""),
        aif_mask_file=info.get("aorta", {}).get("file", ""),
        aif_idx=info[primary]["aif_idx"],
        slices_used=info[primary]["slices_used"],
        erosion_iters=info[primary]["erosion_iters"],
        erosion_iters_requested=(iters_req if primary == "aorta" else ERODE_IDIF),
        n_voxels=p["n_voxels"], n_voxels_used=p["n_voxels_used"],
        n_voxels_original=info[primary]["n_roi"], top_frac=p["top_frac"],
        nbase_mode=NBASE_MODE, nbase_source=p["nbase_source"],
        nbase_start=p["nbase"].start, nbase_stop=p["nbase"].stop,
        arrival_frame=(p["arrival"] if p["arrival"] is not None else np.nan),
        arrival_time_s=(float(t[p["arrival"]]) if p["arrival"] is not None else np.nan),
        baseline_cv=p["baseline_cv"], baseline_cv_flag=bool(p["baseline_cv"] > BASE_CV_FLAG),
        drift_pct_per_frame=p["drift_pct"], drift_t=p["drift_t"],
        drift_flag=p["drift_flag"],
        peak_time_s=p["peak_time_s"], peak_mM=p["peak_mM"],
        baseline_mM=p["baseline_mM"],
        auc_full_mM_s=p["auc_full_mM_s"], auc_early_mM_s=p["auc_early_mM_s"],
        peak_max_mM=PEAK_MAX_MM,
        peak_guard_steps=p["peak_guard_steps"],
        peak_guard_ok=p["peak_guard_ok"],
        T10_blood=aif_cfg.T10, hematocrit=aif_cfg.hematocrit,
        TR=acq.TR, FA=acq.FA, r1=acq.r1,
        spokes_per_frame=SPOKES_PER_FRAME, dt_s=dt, n_frames=len(t),
        idif_note=notes.get("idif", ""), aorta_note=notes.get("aorta", ""),
        error="")

    # per-method detail, for the comparison
    for m in res:
        r = res[m]
        row.update({
            f"{m}_n_voxels":      r["n_voxels"],
            f"{m}_n_voxels_used": r["n_voxels_used"],
            f"{m}_slices_used":   info[m]["slices_used"],
            f"{m}_erosion_iters": info[m]["erosion_iters"],
            f"{m}_top_frac":      r["top_frac"],
            f"{m}_nbase_start":   r["nbase"].start,
            f"{m}_nbase_stop":    r["nbase"].stop,
            f"{m}_arrival_frame": r["arrival"],
            f"{m}_peak_mM":       r["peak_mM"],
            f"{m}_peak_time_s":   r["peak_time_s"],
            f"{m}_baseline_mM":   r["baseline_mM"],
            f"{m}_auc_full_mM_s": r["auc_full_mM_s"],
            f"{m}_auc_early_mM_s": r["auc_early_mM_s"],
            f"{m}_baseline_cv":   r["baseline_cv"],
            f"{m}_drift_pct_per_frame": r["drift_pct"],
            f"{m}_drift_t":       r["drift_t"],
            f"{m}_peak_guard_steps": r["peak_guard_steps"],
            f"{m}_peak_guard_ok":    r["peak_guard_ok"],
        })
    row.update(ov)
    row.update(dm)
    return row


# =============================================================================
# main loop
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    subjects = discover_subjects(DATA_ROOT)
    print(f"{len(subjects)} subject folder(s) under {DATA_ROOT}")
    summary = []

    for fmb_id in subjects:
        print(f'\n{"=" * 80}\nProcessing {fmb_id}\n{"=" * 80}')
        try:
            summary.append(process_subject(
                fmb_id, os.path.join(DATA_ROOT, fmb_id), dt))
        except Exception as e:      # one bad subject must not stop the batch
            print(f"!! {fmb_id} FAILED: {e!r}")
            summary.append(dict(subject_id=fmb_id, error=repr(e)))

    summ_path = os.path.join(RESULTS_DIR, "AIF_extraction_summary.csv")
    df = pd.DataFrame(summary)
    df.to_csv(summ_path, index=False)

    ok = df["error"].fillna("") == "" if "error" in df else pd.Series(dtype=bool)
    n_ok = int(ok.sum())
    print(f'\n{"=" * 80}')
    print(f"{n_ok}/{len(df)} subjects extracted")
    if n_ok and "aif_source" in df:
        counts = df.loc[ok, "aif_source"].value_counts().to_dict()
        print(f"AIF source : {counts}")
        if "compared" in df:
            print(f"both ROIs  : {int(df.loc[ok, 'compared'].sum())} subject(s) "
                  f"with an IDIF-vs-aorta comparison")
    if n_ok and "peak_guard_steps" in df:
        fired = df.loc[ok, "peak_guard_steps"].fillna(0) > 0
        unmet = ~df.loc[ok, "peak_guard_ok"].fillna(True).astype(bool)
        print(f"peak guard : fired on {int(fired.sum())} subject(s), "
              f"still above {PEAK_MAX_MM:g} mM on {int(unmet.sum())}")
        for _, r in df.loc[ok][unmet].iterrows():
            print(f"  CAP NOT MET {r['subject_id']}: peak {r['peak_mM']:.2f} mM "
                  f"({r['aif_source']}) -- inspect, do not trust")
    if len(df) - n_ok:
        for _, r in df[~ok].iterrows():
            print(f"  FAILED {r['subject_id']}: {r['error']}")
    print(f"Summary: {summ_path}")


if __name__ == "__main__":
    main()