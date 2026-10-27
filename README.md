# dce_mri

DCE-MRI pharmacokinetic modelling pipeline.

## Install

```bash
pip install -e ".[dev]"
```

## Structure

```
src/dce_mri/
    config.py           — all frozen dataclasses (configs + result types)
    signal_models.py    — SPGR signal ↔ concentration conversion
    kinetic_models.py   — TCXM, GKM, DispersedGKM forward models + KineticModelSpec
    vascular_aif.py     — Nejad-Davarani vascular tree dispersion model
    ace_protocol.py     — ACEProtocol, FitMode, ACE forward model
    fitting.py          — LHS + TRF single-voxel fitters (concentration & signal space)
    fitting_admm.py     — Joint TV-regularised voxelwise fitting via ADMM solver
    fitting_roi.py      — Whole-ROI fitting (mean concentration curve)
    bootstrap.py        — Bootstrap uncertainty loops and sigma sweeps
    volume.py           — Chunked parallel voxelwise processing, NIfTI I/O
    statistics.py       — param_stats, compute_over_sweep, ace_param_stats
    io.py               — load_aif, load_nifti_canonical, load/save npz
    visualization.py    — all plot_* functions

scripts/
    run_voxelwise.py            — standard per-voxel fitting over a volume
    run_voxelwise_admm.py       — TV-regularised joint fitting over a volume
    run_roi_fitting.py          — whole-ROI mean curve fitting
    run_bootstrap_sweep.py      — bootstrap uncertainty sigma sweep

notebooks/
    01_signal_to_concentration.ipynb
    02_single_voxel_fitting.ipynb
    03_dispersed_gkm.ipynb
    04_ace_protocol.ipynb
    05_bootstrap_uncertainty.ipynb
    06_voxelwise_maps.ipynb
```

## Usage

```python
from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig
from dce_mri.kinetic_models import TCXM, GKM, DISPERSED_GKM
from dce_mri.fitting import fit_single_voxel
from dce_mri.volume import run_voxelwise_fitting
```
