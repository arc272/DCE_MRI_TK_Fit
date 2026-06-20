# dce_mri

Academic code repository for DCE-MRI pharmacokinetic modelling pipeline.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd dce_mri
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n dce_mri python=3.11
conda activate dce_mri
```

Alternatively, create an environment directly from the provided specification file:

```bash
conda create --name dce_mri --file spec-file.txt
conda activate dce_mri
```

### 3. Install the package

Install the package in editable mode together with development dependencies:

```bash
pip install -e ".[dev]"
```

### 4. Verify installation

Start Python and confirm imports work:

```python
import dce_mri

from dce_mri.config import AcquisitionConfig
from dce_mri.kinetic_models import TCXM
```

No errors should be produced.


## Quick Start
```
Create an environment and install dependencies from spec-file.txt.
Launch Jupyter from the repository root.
Open a notebook under notebooks/.
```

## Structure

```
src/dce_mri/: core Python modules for models, optimizer
notebooks/: exploratory and experiment notebooks

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

notebooks/
    06_roi_fitting.ipynb
```
## Usage

```python
from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig
from dce_mri.kinetic_models import TCXM, GKM, DISPERSED_GKM
from dce_mri.fitting import fit_single_voxel
from dce_mri.volume import run_voxelwise_fitting
```
