import numpy as np
import sys
from pathlib import Path

import torch
from glob import glob

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from VAE import VAE
from unet import UNet_FS
import ants

def load_fs_unet(path, device=None, in_channels=12, channels=(128, 256, 512), n_steps=1000, strict=True):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("unet", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    model = UNet_FS(in_channels=in_channels, channels=list(channels), n_steps=n_steps)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    print("unet loaded with eval mode.")
    return model


def load_VAE(path, device=None, in_channels=1, base_channels=64, latent_channels=12, strict=True):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "vae" in checkpoint:
        state_dict = checkpoint["vae"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    vae = VAE(in_channels=in_channels, base_channels=base_channels, latent_channels=latent_channels).to(device)
    vae.load_state_dict(state_dict, strict=strict)
    vae.eval()
    print("VAE loaded with eval mode.")
    return vae


def create_diffusion_alphas_bar(n_steps=1000, beta_start=0.0001, beta_end=0.01, device=None):
    """Return the cumulative alpha schedule required for diffusion generation."""
    scale = 1000 / n_steps
    betas = torch.linspace(beta_start * scale, beta_end * scale, n_steps, device=device)
    return torch.cumprod(1 - betas, dim=0)


def get_mask_files(fastMRI_path, idx):
    mask_files = glob(fastMRI_path + f'/fastMRI_breast_{idx:03d}_*.npy')
    mask_files = [f for f in mask_files if ('benign' in f) or ('malig' in f)]
    mask_files = [f for f in mask_files if 'tight' not in f]
    return sorted(mask_files)

def get_tight_mask_files(fastMRI_path, idx):
    tight_mask_files = glob(fastMRI_path + f'/fastMRI_breast_{idx:03d}_*tight.npy')
    return sorted(tight_mask_files)

def get_img_file(fastMRI_path, idx):
    if fastMRI_path[-1] == '/':
        fastMRI_path = fastMRI_path[:-1]
    return  fastMRI_path + f"/fastMRI_breast_{idx:03d}_2.npy"

def calc_roi_curve(time_series, roi_mask):
    # time_series: (N, T, H, W) or (T, H, W)
    # roi_mask: (N, H, W) or (H, W)

    if isinstance(time_series, torch.Tensor):
        time_series = time_series.detach().cpu().numpy()
    if isinstance(roi_mask, torch.Tensor):
        roi_mask = roi_mask.detach().cpu().numpy()

    if time_series.ndim == 3:
        if (np.sum(roi_mask) == 0):
            return np.zeros(time_series.shape[0])
        
        roi_curve = np.sum(time_series * roi_mask[None, :, :], axis=(-2, -1)) / np.sum(roi_mask)
    elif time_series.ndim == 4:
        roi_curve = []

        for i in range(time_series.shape[0]):
            roi_curve.append(calc_roi_curve(time_series[i], roi_mask[i]))

        roi_curve = np.stack(roi_curve, axis=0)

    return roi_curve

def calc_roi_slope(roi_curve, time_window):
    # roi_curve: (N, T) or (T,)
    # time_window: int
    if isinstance(roi_curve, torch.Tensor):
        roi_curve = roi_curve.detach().cpu().numpy()

    if roi_curve.ndim == 1:
        roi_curve = roi_curve.reshape(1, -1)

    window_diff = roi_curve[:, time_window:] - roi_curve[:, :roi_curve.shape[1]-time_window]
    slopes = np.max(window_diff, axis=1) / time_window

    return slopes if slopes.shape[0] > 1 else slopes[0]

def calc_enhancement_ratio(time_series, baseline_frame_idx: list, enhanced_frame_idx: list, cutoff=0.1):
    # time_series: (N, T, H, W) or (T, H, W)
    # baseline_frame_idx: list of indices for baseline frames
    # enhanced_frame_idx: list of indices for enhanced frames
    if isinstance(time_series, torch.Tensor):
        time_series = time_series.detach().cpu().numpy()

    eps = 1e-7

    if time_series.ndim == 3:
        # (T, H, W)
        baseline = time_series[baseline_frame_idx].mean(axis=0)   # (H, W)
        enhanced = time_series[enhanced_frame_idx].mean(axis=0)   # (H, W)

        enhancement_ratio = (enhanced - baseline) / (baseline + eps)

        enhancement_ratio[baseline < cutoff] = 0
        
    elif time_series.ndim == 4:
        # (N, T, H, W)
        baseline = time_series[:, baseline_frame_idx].mean(axis=1)  # (N, H, W)
        enhanced = time_series[:, enhanced_frame_idx].mean(axis=1)  # (N, H, W)

        enhancement_ratio = (enhanced - baseline) / (baseline + eps)

        enhancement_ratio[baseline < cutoff] = 0

    return enhancement_ratio


def register_to_frame(series, fixed_frame_idx):
    assert series.ndim == 3

    series = series.astype(np.float32)
    fixed = ants.from_numpy(series[fixed_frame_idx])

    fixed_time_series = np.zeros_like(series)
    translations = []

    for t in range(0, series.shape[0]):
        if t == fixed_frame_idx:
            fixed_time_series[t] = series[t]
            continue

        moving = ants.from_numpy(series[t])

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            initial_transform="Identity",
            aff_metric="mattes",
            aff_sampling=32,
            aff_random_sampling_rate=0.2,
            singleprecision=True,
            verbose=False,
            random_seed=0
        )

        fixed_time_series[t] = reg["warpedmovout"].numpy()
        tx = ants.read_transform(reg["fwdtransforms"][0])
        translations.append(tx)

    translations = np.array(translations)

    return fixed_time_series, translations

def masked_mean(time_series, mask):
    # time_series: (Z, H, W)
    # mask: (H, W)
    mask = np.asarray(mask).astype(bool)

    if mask.sum() == 0:
        assert False

    return time_series[:, mask].mean(axis=1)

@torch.inference_mode()
def sample_ddim_FS(
    num_samples,
    diffusion_steps,
    ddim_steps,
    unet: UNet_FS,
    reference_frames,
    reference_times,
    gen_times,
    alphas_bar,
    eta=0.0,
    gamma=1.0,
):
    if reference_frames.shape[0] != num_samples:
        raise ValueError("batch size of reference_frames must match num_samples")

    if reference_times.shape[0] != num_samples:
        raise ValueError("batch size of reference_times must match num_samples")

    if len(gen_times) != num_samples:
        raise ValueError("batch size of gen_times must match num_samples")

    reference_frames = reference_frames.clone()
    reference_times = reference_times.clone()
    gen_times = gen_times.clone()

    unet.eval()
    device = next(unet.parameters()).device
    with torch.no_grad():
        spatial_shape = reference_frames.shape[2:]
        data = torch.randn((num_samples, *spatial_shape), device=device, dtype=reference_frames.dtype)
        t_linspace = torch.linspace(0, 1, steps=ddim_steps)
        timesteps = (t_linspace**gamma * (diffusion_steps - 1)).round().long().unique().flip(0)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.int)
            pred_noise = unet(reference_points=reference_frames, reference_t=reference_times, noise_x=data, noise_t=gen_times, diffusion_t=t_tensor)
            x0_pred = (data - torch.sqrt(1 - alphas_bar[t]) * pred_noise) / torch.sqrt(alphas_bar[t])

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                sigma = eta * torch.sqrt((1 - alphas_bar[t_prev]) / (1 - alphas_bar[t]) * (1 - alphas_bar[t] / alphas_bar[t_prev]))
                data = torch.sqrt(alphas_bar[t_prev]) * x0_pred + torch.sqrt(1 - alphas_bar[t_prev] - sigma**2) * pred_noise
                if eta != 0.0:
                    data += sigma * torch.randn_like(data)
            else:
                data = x0_pred

    return data


@torch.inference_mode()
def create_time_series_FS(
    time_series: torch.Tensor,
    current_time,
    target_time_list: np.ndarray,
    unet: UNet_FS,
    alphas_bar,
    diffusion_forward_steps=1000,
    ddim_steps=100,
    refine_weight=1.0,
    refine_steps=2,
    refine_gamma=0.7,
    ddim_gamma=1.0,
    alter_references=False,
    generation_batch_size=None,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """Generate target frames from the supplied FS reference frames."""
    if any(t < -1 or t > 1 for t in target_time_list):
        raise ValueError("return_time values must be between -1 and 1")
    if generation_batch_size is not None and generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be a positive integer or None")

    orig_reference_times = current_time.copy()
    current_time_series = time_series.clone()

    while True:
        next_times_to_generate = []

        for t0, t1 in zip(current_time[:-1], current_time[1:]):
            eps = 1e-5
            times_in_interval = sorted(x for x in target_time_list if t0 + eps < x < t1 - eps)

            if times_in_interval:
                next_times_to_generate.append(times_in_interval[(len(times_in_interval) - 1) // 2])

        if not next_times_to_generate:
            break

        target_batch_size = generation_batch_size or len(next_times_to_generate)
        generated_batches = []
        for batch_start in range(0, len(next_times_to_generate), target_batch_size):
            batch_times = next_times_to_generate[
                batch_start : batch_start + target_batch_size
            ]
            batch_size = len(batch_times)
            reference_frames = current_time_series.unsqueeze(0).repeat_interleave(
                batch_size, 0
            )
            reference_times = torch.tensor(
                current_time,
                dtype=torch.float,
                device=reference_frames.device,
            ).unsqueeze(0).repeat_interleave(batch_size, 0)

            generated_batches.append(
                sample_ddim_FS(
                    num_samples=batch_size,
                    diffusion_steps=diffusion_forward_steps,
                    ddim_steps=ddim_steps,
                    unet=unet,
                    reference_frames=reference_frames,
                    reference_times=reference_times,
                    gen_times=torch.tensor(
                        batch_times,
                        dtype=torch.float,
                        device=reference_frames.device,
                    ),
                    alphas_bar=alphas_bar,
                    eta=0.0,
                    gamma=ddim_gamma,
                )
            )
        generated_frames = torch.cat(generated_batches, dim=0)

        combined_series = torch.cat([current_time_series, generated_frames], dim=0)
        combined_times = list(current_time) + list(next_times_to_generate)
        order = sorted(range(len(combined_times)), key=lambda index: float(combined_times[index]))
        order_idx = torch.tensor(order, dtype=torch.long, device=combined_series.device)
        current_time_series = combined_series[order_idx]
        current_time = [float(combined_times[index]) for index in order]

    before_refine_series = current_time_series.clone()
    spatial_shape = current_time_series.shape[1:]

    for _ in range(refine_steps):
        if not alter_references:
            generated_idx = np.array([index for index, time in enumerate(current_time) if time not in orig_reference_times], dtype=np.int64)
        else:
            # The endpoint frames at t=-1 and t=1 are required references for
            # every UNet call, so only refine the interior frames.
            generated_idx = np.arange(1, len(current_time_series) - 1, dtype=np.int64)

        if generated_idx.size == 0:
            break

        noise_frames = current_time_series[generated_idx]
        noise_times = torch.tensor([current_time[index] for index in generated_idx], dtype=torch.float, device=noise_frames.device)

        # Each refinement sample uses every other frame as a reference. This
        # always removes exactly one frame, including when original reference
        # frames are also eligible for refinement.
        num_references = current_time_series.shape[0] - 1
        reference_frames = torch.empty((noise_frames.shape[0], num_references, *spatial_shape), dtype=noise_frames.dtype, device=noise_frames.device)
        reference_times = torch.empty((noise_frames.shape[0], num_references), dtype=torch.float, device=noise_frames.device)

        for row, generated_index in enumerate(generated_idx):
            ref_indices = [index for index in range(current_time_series.shape[0]) if index != generated_index]
            reference_frames[row] = current_time_series[ref_indices].clone()
            reference_times[row] = torch.tensor([current_time[index] for index in ref_indices], dtype=torch.float, device=noise_frames.device)

        refined_frames = sample_ddim_FS(
            num_samples=noise_frames.shape[0],
            diffusion_steps=diffusion_forward_steps,
            ddim_steps=ddim_steps,
            unet=unet,
            reference_frames=reference_frames,
            reference_times=reference_times,
            gen_times=noise_times,
            alphas_bar=alphas_bar,
            eta=0.0,
            gamma=ddim_gamma,
        )

        current_time_series[generated_idx] = (current_time_series[generated_idx] + refine_weight * refined_frames) / (1 + refine_weight)
        refine_weight *= refine_gamma

    return current_time_series, current_time, before_refine_series
