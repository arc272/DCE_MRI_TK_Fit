#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:58:25 2026

@author: allisonchen
"""

# %% Setup
import numpy as np
from dce_mri.synthetic import default_test_setup, simulate_curve
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.fitting import build_residual, build_delay_augmented_spec

setup    = default_test_setup(n_frames=48)
aif      = setup["aif"]
time_var = setup["time_var"]
acq      = setup["acq"]
subject  = setup["subject"]
cfg      = setup["cfg"]
aif_cfg  = setup["aif_cfg"]
relax_cfg = setup["relax_cfg"]

# %% 1. FXL path — your exact line, extended
theta_true = TCXM.p0.copy()
S_clean = simulate_curve(theta_true, aif, time_var, TCXM, acq, subject, cfg)

b = build_residual(S_clean, aif, time_var, TCXM, acq, subject, cfg, on_failure="nan")
assert np.abs(b.residual_fn(theta_true)).max() < 1e-9
print("FXL residual OK  |  param_names:", b.param_names)

# %% 2. SSM3 water-exchange path (adds tau_i)
theta_wx = np.concatenate([TCXM.p0, [0.5]])          # + tau_i = 0.5
Sw_clean = simulate_curve(theta_wx, aif, time_var, TCXM, acq, subject, cfg,
                          SSM3, aif_cfg, relax_cfg)

bw = build_residual(Sw_clean, aif, time_var, TCXM, acq, subject, cfg,
                    SSM3, aif_cfg, relax_cfg, on_failure="nan")
assert np.abs(bw.residual_fn(theta_wx)).max() < 1e-9
print("SSM3 residual OK  |  param_names:", bw.param_names)

# %% 3. Delay (joint) + SSM3 path (adds tau_delay, then tau_i)
aug_spec = build_delay_augmented_spec(TCXM, aif, time_var, acq, wx_model=SSM3)
theta_delay_wx = np.concatenate([TCXM.p0, [8.0], [0.5]])   # + tau_delay=8.0, tau_i=0.5
Sd_clean = simulate_curve(theta_delay_wx, aif, time_var, aug_spec, acq, subject, cfg,
                          SSM3, aif_cfg, relax_cfg)

bd = build_residual(Sd_clean, aif, time_var, aug_spec, acq, subject, cfg,
                   SSM3, aif_cfg, relax_cfg, on_failure="nan")
assert np.abs(bd.residual_fn(theta_delay_wx)).max() < 1e-9
print("delay+SSM3 residual OK  |  param_names:", bd.param_names)

# %% 4. Confirm on_failure="nan" actually converts failures, not just NLLS-style penalty
bad_theta = np.array([0.9, 0.9, 0.005, 0.005, 8.0, 0.5])   # ve+vp > 1, should violate constraint
assert np.all(np.isnan(bd.residual_fn(bad_theta)))
print("on_failure='nan' behaves correctly on a constraint violation")

print("\nAll build_residual checks passed.")