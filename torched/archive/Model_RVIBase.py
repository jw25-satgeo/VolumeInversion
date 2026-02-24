import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy

class TorchRVI(nn.Module):
    """
    PyTorch adaptation of the RV (Random Volume over Water/Ice) inversion model.
    
    This attempts to physically invert the following:
      - Ice thickness (H)
      - Age (A)
      - Surface–volume ratio (SVR)
    implementing finite differences with autograd, allowing GPU acceleration.
    
    Args:
        coherences (np.ndarray): complex input coherences, shape (kz, H, W)
        incidence_angle (float or np.ndarray): radar incidence angle(s)
        k_z (np.ndarray): vertical wavenumber grid, same length as kz dimension
        wavelength (float): radar wavelength, default 0.12 m
    """

    def __init__(self, coherences, incidence_angle, k_z, wavelength, vct_required=True):
        super().__init__()
        # --- Convert input data to torch tensors ---
        self.register_buffer("coherences", torch.tensor(coherences, dtype=torch.complex64))
        self.register_buffer("k_z", torch.tensor(k_z, dtype=torch.float32))
        self.register_buffer("inc_angle", torch.tensor(incidence_angle, dtype=torch.float32))
        self.wavelength = wavelength

        # --- Parameter initialization ---
        B, H, W = self.coherences.shape
        self.thickness = nn.Parameter(torch.full((H, W), 0.1))   # ice thickness (m)
        self.age = nn.Parameter(torch.full((H, W), 1.0))        # surface age (years)
        self.sv_ratio = nn.Parameter(torch.full((B, H, W), 0.5))  # surf-vol ratio

        # --- Optional static precomputation ---
        self.icesurf_phase = self._estimate_icesurf_phase()  # simple initial guess
        if (vct_required):
            print("Remember to initialize the volume coherence table! - x.sample_volume_coherence(...)")

    # --------------------------------------------------------------------------
    # Core Methods
    # --------------------------------------------------------------------------
    def _estimate_icesurf_phase(self):
        """
        Estimate initial ice-surface phase from maximum coherence.
        Rough equivalent of your `find_icesurf_phase()` method.
        """
        abs_coh = torch.abs(self.coherences)
        idx = torch.argmax(abs_coh, dim=0)
        phase = torch.angle(self.coherences[idx, torch.arange(abs_coh.shape[1])[:, None], torch.arange(abs_coh.shape[2])])
        return phase  # shape (H, W)

    def sample_volume_coherence(self, extinction_func,
                          hgt_min=0.1, hgt_max=1.0, hgt_bins=20,
                          age_min=10, age_max=200, age_bins=20,
                          k_z_bins=20):
        # integration axes
        z = np.linspace(0, hgt_max, int(hgt_bins + hgt_min*(hgt_max-hgt_min)/hgt_bins))
        hgts = np.linspace(hgt_min, hgt_max, hgt_bins)
        ages = np.linspace(age_min, age_max, age_bins)
        mesh = np.meshgrid(z, hgts, ages, indexing='ij')
        k_z_r = np.linspace(np.nanmin(self.k_z), np.nanmax(self.k_z), k_z_bins)
        # extinction coefficient A(z)
        ext = extinction_func(*mesh)
        # compute coherence integral per k_z
        vol_coh = np.stack([
            np.trapezoid(np.exp(-ext + 1j*KZ*mesh[0]), z, axis=0) /
            np.trapezoid(np.exp(-ext), z, axis=0)
            for KZ in k_z_r
        ], axis=-1)
        self.register_buffer("vol_coh_tensor", torch.tensor(vol_coh, dtype=torch.complex64))

        # basic array expanded by 2 singlet spatial dimensions after conversion to torch tensor
        bounds = np.array([[hgts[0],hgts[-1]], [ages[0],ages[-1]], [k_z_r[0], k_z_r[-1]]])
        self.register_buffer("vcs_bounds", torch.tensor(bounds[:,:,None,None], dtype=torch.float32))
    
    def forward(self):
        vcs_params = torch.stack((self.thickness, self.age, self.k_z), dim=0) #shape (3, azi, rng)
        # map to fractional indices for simple interpolation
        vcs_idx = (vcs_params-self.vcs_bounds[:,0])/(self.vcs_bounds[:,1]-self.vcs_bounds[:,0]).to(self.vol_coh_tensor.device)
        vcs_idx = (vcs_idx.to(self.vol_coh_tensor.device).clamp(0.0,1.0) * 2 - 1).permute(1,2,0)[None,None,...]

        vol_coh = F.grid_sample(self.vol_coh_tensor.real[None,None,...], vcs_idx, mode='bilinear',align_corners=True)[0,0]
        vol_coh += F.grid_sample(self.vol_coh_tensor.imag[None,None,...], vcs_idx, mode='bilinear',align_corners=True)[0,0]
        cohs = torch.exp(1j * self.icesurf_phase) * (vol_coh + self.sv_ratio) / (1 + self.sv_ratio)
        return cohs
    
    def data_captured(self):
        """Return observed coherences (complex tensor)."""
        return self.coherences

    def loss_fn(self, modelled, captured):
        """Mean square difference (L1 loss on magnitude)."""
        return torch.nanmean(torch.abs(modelled - captured)**2)