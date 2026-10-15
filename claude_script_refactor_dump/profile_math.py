"""Physical-profile calculations that require only arrays and explicit parameters."""
import numpy as np
import torch


def geometric_thickness(profiles, salinity_length, temperature_length, ice_fraction,
                        cells_per_m=30, device='cuda'):
    """Return equivalent and geometric thickness for profiles using a supplied ice-fraction map.

    ``ice_fraction`` is a callable taking ``(temperature, salinity)`` tensors and returning
    the corresponding fraction. The solver/model itself is therefore not imported here.
    """
    profiles = torch.as_tensor(profiles, dtype=torch.float32, device=device)
    S = profiles[:, :salinity_length]
    T = profiles[:, salinity_length:salinity_length + temperature_length]
    with torch.no_grad():
        frac = ice_fraction(T, S)
    keep = ((frac == 0).cumsum(dim=-1) == 0)
    dz = 1.0 / cells_per_m
    equivalent = (frac * keep).sum(-1) * dz
    geometric = keep.sum(-1) * dz
    return equivalent.cpu().numpy(), geometric.to(torch.float32).cpu().numpy()


def descale_profile(profiles, mean, scale, lengths):
    """Undo a profile-block affine normalization without depending on a model class."""
    p = np.asarray(profiles).reshape(-1)
    mean = np.asarray(mean).reshape(-1)[:sum(lengths[:3])]
    scale = np.asarray(scale).reshape(-1)[:sum(lengths[:3])]
    return (p * scale + mean).reshape(3, -1)


def looks_whitened(profile, threshold=4.0):
    """Simple diagnostic for standardized rather than physical profile values."""
    p = np.asarray(profile)
    return bool(np.nanmax(np.abs(p)) < threshold)
