#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Sep 22 14:07:42 2026

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
  4. baseline window, per subject (priority: NBASE_OVERRIDE > auto > fallback)
       auto: bolus arrival = first frame of the mean aorta signal above
             base + ARRIVAL_FRAC*(peak - base) that stays above for
             N_CONFIRM frames; baseline ends BASE_MARGIN frames before it
       drift: straight-line fit over the baseline window; slope reported
             as % of baseline per frame and as slope/SE. If
             APPLY_LATE_BASE_ON_DRIFT is True and |slope/SE| > DRIFT_T_FLAG,
             S0 is taken from the last LATE_BASE_K baseline frames only
  5. extract_aif(...)  -- the SAME function 06_roi_fitting.ipynb uses:
     mean signal in the mask -> SPGR inversion (blood T1) -> /(1 - Hct)
  6. save notebook-style diagnostic figures + one labelled CSV per subject,
     plus AIF_extraction_summary.csv (nbase_start/nbase_stop there are
     the baseline window to reuse for the tumor fits)

All sequence / AIF parameters match 06_roi_fitting.ipynb cells [4] and [8].

Run:
    PYTHONPATH=src python scripts/extract_aifs.py
"""

import glob
import os
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

FMB_IDS = [
    "FMB046",
    "FMB128",
    "FMB149",
    "FMB170",
    "FMB176",
    "FMB186",
    "FMB189",
    "FMB216",
    "FMB238",
    "FMB239",
    "FMB276",
    "FMB283",
    "FMB294",
    # "FMB001", "FMB003", "FMB005", "FMB010", "FMB011", "FMB013", "FMB014",
    # "FMB016", "FMB020", "FMB024", "FMB027", "FMB032", "FMB036", "FMB038",
    # "FMB040", "FMB041", "FMB044", "FMB045", "FMB046", "FMB048", "FMB050",
    # "FMB052", "FMB055", "FMB056", "FMB060", "FMB070", "FMB071", "FMB072",
    # "FMB074", "FMB079", "FMB080", "FMB081", "FMB083", "FMB084", "FMB085",
    # "FMB093", "FMB095", "FMB096", "FMB097", "FMB099", "FMB100", "FMB104",
    # "FMB107", "FMB108", "FMB109", "FMB111", "FMB112", "FMB115", "FMB118",
    # "FMB122", "FMB124", "FMB126", "FMB128", "FMB129", "FMB132", "FMB136",
    # "FMB140", "FMB141", "FMB143", "FMB146", "FMB147", "FMB148", "FMB149",
    # "FMB152", "FMB158", "FMB162", "FMB164", "FMB165", "FMB167", "FMB170",
    # "FMB172", "FMB176", "FMB177", "FMB183", "FMB184", "FMB185", "FMB186",
    # "FMB187", "FMB189", "FMB192", "FMB194", "FMB196", "FMB198", "FMB200",
    # "FMB202", "FMB203", "FMB207", "FMB212", "FMB213", "FMB215", "FMB216",
    # "FMB217", "FMB219", "FMB225", "FMB226", "FMB228", "FMB231", "FMB232",
    # "FMB233", "FMB234", "FMB238", "FMB239", "FMB244", "FMB245", "FMB246",
    # "FMB248", "FMB249", "FMB259", "FMB264", "FMB265", "FMB271", "FMB276",
    # "FMB283", "FMB286", "FMB288", "FMB289", "FMB290", "FMB291", "FMB293",
    # "FMB294", "FMB295", "FMB296", "FMB297", "FMB298", "FMB299",
]

DATA_ROOT   = "/mnt/labspace/users/allison/fastMRI_data/tight_masks/canonical"
RESULTS_DIR = "/mnt/labspace/users/allison/fastMRI_data/aif_results"

# --- acquisition: 06_roi_fitting.ipynb cell [4] -----------------------------
SPOKES_PER_FRAME = 8
NSPOKE   = 288                # total spokes across the whole scan
SCANTIME = 150                # total scan duration, s

# --- baseline window --------------------------------------------------------
AUTO_NBASE     = True             # detect per subject from the aorta curve
NBASE_FALLBACK = slice(0, 7)      # used only if AUTO_NBASE = False
NBASE_OVERRIDE = {                # per-subject manual windows, win over auto
    # "FMB194": slice(0, 9),
}
ARRIVAL_FRAC   = 0.10   # arrival = first frame > base + 10% of (peak - base)
N_MIN_BASE     = 5      # minimum baseline length (frames)
N_CONFIRM      = 2      # frames that must stay above the threshold
BASE_MARGIN    = 1      # frames dropped immediately before arrival
BASE_CV_FLAG   = 0.10   # flag baselines noisier than this CV (tune on cohort)

# --- baseline drift ---------------------------------------------------------
DRIFT_T_FLAG             = 3.0    # flag if |slope| > 3 x its standard error
APPLY_LATE_BASE_ON_DRIFT = False  # False: report drift only (first pass).
                                  # True: if flagged, S0 from the last
                                  # LATE_BASE_K baseline frames only
LATE_BASE_K              = 4

# module-level templates; each subject gets its own nbase via replace()
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE_FALLBACK)

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
EROSION_ITERS_OVERRIDE = {            # per-subject exceptions
"FMB046": 1, "FMB128": 1, "FMB149": 1, "FMB170": 1, "FMB176": 1, "FMB186": 1, "FMB189": 1, "FMB216": 1, "FMB238": 1, "FMB239": 1, "FMB276": 1, "FMB283": 1, "FMB294": 1,
}
KEEP_OTHER_LABELLED_SLICES = False    # False: ONLY aif_idx and aif_idx+1
                                      # contribute to the AIF

TIME_IDX_FOR_ROI_FIG = 15     # frame shown in the ROI inspection figure
SHOW_FIGS = False             # True to also plt.show() interactively

if not SHOW_FIGS:
    matplotlib.use("Agg")


# =============================================================================
# per-subject paths
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

        # aorta ROI name varies ("56_Aorta_RotOrtho_6.nii", "aorta_case288.nii");
        # "*orta*" matches both capitalisations. Refuse to guess if missing
        # or ambiguous.
        hits = sorted(glob.glob(os.path.join(subj_dir, "*orta*.nii*")))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{fmb_id}: expected exactly one '*orta*.nii*' in "
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


def detect_nbase(aif_signal, frac=ARRIVAL_FRAC, n_min=N_MIN_BASE,
                 n_confirm=N_CONFIRM, margin=BASE_MARGIN, n_iter=3):
    """
    Baseline window = frames before bolus arrival in the mean aorta signal.

    Arrival: first frame above base + frac*(peak - base) that stays above for
    n_confirm frames. The baseline ends `margin` frames before arrival; base
    is re-estimated (median) from the growing window until it stops changing.

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


def plot_aif_inspection(t, aif_signal, aif, fmb_id, nbase=None, arrival=None):
    """Notebook 'aif_inspection' figure (cell [8]) + baseline window / arrival."""
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

    if nbase is not None:
        for ax in axes:
            ax.axvspan(t[nbase.start], t[nbase.stop - 1], color="0.6", alpha=0.25,
                       label=f"baseline frames {nbase.start}-{nbase.stop - 1}")
            if arrival is not None:
                ax.axvline(t[arrival], ls="--", color="0.3", lw=1,
                           label=f"arrival frame {arrival}")
        axes[0].legend(fontsize=8, loc="lower right")

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

            # --- baseline window: override > auto > fallback ---------------
            mean_sig = sig_4d[aif_mask].mean(axis=0)   # == extract_aif's aif_signal
            arrival = None
            if fmb_id in NBASE_OVERRIDE:
                nbase_s, nbase_src = NBASE_OVERRIDE[fmb_id], "override"
            elif AUTO_NBASE:
                nbase_s, arrival = detect_nbase(mean_sig)
                nbase_src = "auto"
            else:
                nbase_s, nbase_src = NBASE_FALLBACK, "fallback"

            # drift over the FULL detected window (before any late-window switch)
            drift_pct, drift_t = baseline_drift(mean_sig, nbase_s)
            drift_flag = bool(np.isfinite(drift_t) and abs(drift_t) > DRIFT_T_FLAG)
            if (drift_flag and APPLY_LATE_BASE_ON_DRIFT and nbase_src == "auto"
                    and nbase_s.stop - LATE_BASE_K >= 0):
                nbase_s = slice(nbase_s.stop - LATE_BASE_K, nbase_s.stop)
                nbase_src = "auto-late (drift)"

            base_cv = baseline_cv(mean_sig, nbase_s)
            cv_flag = base_cv > BASE_CV_FLAG

            acq_s     = replace(acq, nbase=nbase_s)
            aif_cfg_s = replace(aif_cfg, nbase=nbase_s)

            arr_txt = (f", arrival frame {arrival} (t={t[arrival]:.1f}s)"
                       if arrival is not None else "")
            print(f"Baseline ({nbase_src}) : frames {nbase_s.start}-{nbase_s.stop - 1}"
                  f"{arr_txt}")
            print(f"Baseline CV      : {base_cv:.3f}"
                  f"{'  <-- NOISY BASELINE' if cv_flag else ''}")
            print(f"Baseline drift   : {drift_pct:+.2f} %/frame, slope/SE = {drift_t:+.1f}"
                  f"{'  <-- DRIFT' if drift_flag else ''}")

            aif, aif_signal = extract_aif(sig_4d, aif_mask, acq_s, aif_cfg_s, B1=B1)

            print(f"AIF signal range : [{aif_signal.min():.1f}, {aif_signal.max():.1f}]")
            print(f"AIF plasma range : [{aif.min():.4f}, {aif.max():.4f}] mM")
            print(f"AIF peak time    : {t[np.argmax(aif)]:.1f} s")
            print(f"AIF peak value   : {aif.max():.4f} mM")
            print(f"AIF baseline     : {aif[acq_s.nbase].mean():.6f} mM")

            plot_roi_inspection(sig_4d, aif_roi, aif_mask, aif_idx,
                                TIME_IDX_FOR_ROI_FIG, fmb_id)
            plot_aif_inspection(t, aif_signal, aif, fmb_id,
                                nbase=nbase_s, arrival=arrival)

            csv_path = os.path.join(RESULTS_DIR, f"{fmb_id}_AIF_plasma.csv")
            pd.DataFrame({
                "frame":           np.arange(len(t)),
                "time_s":          t,
                "aif_signal_au":   aif_signal,
                "aif_plasma_mM":   aif,
            }).to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            summary.append(dict(
                subject_id=fmb_id,
                aif_mask_file=os.path.basename(args.aif_mask_file),
                dce_file=os.path.basename(args.dce_file),
                aif_idx=aif_idx, slices_used=f"{aif_idx},{aif_idx + 1}",
                erosion_iters=iters, n_voxels=int(aif_mask.sum()),
                n_voxels_original=int(aif_roi.sum()),
                nbase_source=nbase_src,
                nbase_start=nbase_s.start, nbase_stop=nbase_s.stop,
                arrival_frame=(arrival if arrival is not None else np.nan),
                arrival_time_s=(float(t[arrival]) if arrival is not None else np.nan),
                baseline_cv=base_cv, baseline_cv_flag=bool(cv_flag),
                drift_pct_per_frame=drift_pct, drift_t=drift_t,
                drift_flag=drift_flag,
                peak_time_s=float(t[np.argmax(aif)]), peak_mM=float(aif.max()),
                baseline_mM=float(aif[acq_s.nbase].mean()),
                T10_blood=aif_cfg.T10, hematocrit=aif_cfg.hematocrit,
                TR=acq.TR, FA=acq.FA, r1=acq.r1,
                spokes_per_frame=SPOKES_PER_FRAME, dt_s=dt, error=""))

        except Exception as e:        # one bad subject must not stop the batch
            print(f"!! {fmb_id} FAILED: {e!r}")
            summary.append(dict(subject_id=fmb_id, error=repr(e)))

    summ_path = os.path.join(RESULTS_DIR, "AIF_extraction_summary.csv")
    pd.DataFrame(summary).to_csv(summ_path, index=False)
    print(f"\nSummary: {summ_path}")


if __name__ == "__main__":
    main()