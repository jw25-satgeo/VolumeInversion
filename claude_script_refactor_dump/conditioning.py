"""Pure conditioning rules for the simulated-core experiment."""
import torch


def conditioning_mask(shape, covariance_size, n_angles=6, angle=0,
                      mode='single_angle', device='cuda'):
    """Construct the boolean conditioning mask for the covariance features."""
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    n_cov, D = covariance_size, shape[-1]
    per_angle = n_cov // n_angles
    if mode == 'single_angle':
        mask[..., D - 2*n_cov + angle*per_angle:D - 2*n_cov + (angle+1)*per_angle] = True
        mask[..., D - n_cov + angle*per_angle:D - n_cov + (angle+1)*per_angle] = True
    elif mode == 'all_angles':
        mask[..., -2*n_cov:] = True
    elif mode == 'notebook':
        mask[..., -n_cov:] = True
    else:
        raise ValueError(f'invalid mode {mode}')
    return mask
