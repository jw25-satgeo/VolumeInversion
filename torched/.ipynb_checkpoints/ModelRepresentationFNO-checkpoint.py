import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy

import Model_RVIBase


# --- NN utility classes --- #
class Squeeze(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim;
    def forward(self, x): return x.squeeze(self.dim)
class Unsqueeze(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim;
    def forward(self, x): return x.unsqueeze(self.dim)

class HelperRFFT(nn.Module):
    def forward(self, x):
        tmp = torch.fft.rfft(x, dim=-1)
        return torch.cat((tmp.real, tmp.imag[..., 1:-1]), dim=-1)
class HelperIRFFT(nn.Module):
    def forward(self, x):
        cut = x.size(-1)//2 + 1
        tmp = torch.complex(x[..., :cut], F.pad(x[..., cut:], (1,1), value=0))
        #return torch.fft.irfft(tmp, n=x.size(-1)*2, dim=-1)[..., :x.size(-1)]
        return torch.fft.irfft(tmp, dim=-1)
class FNOConv(nn.Module):
    def __init__(self):
        super().__init__();
        self.ffn = nn.Sequential(HelperRFFT(), nn.Conv1d(1,1,5,padding=2), HelperIRFFT())
        self.rfn = nn.Conv1d(1,1,5,padding=2)
    def forward(self, x):
        return self.rfn.forward(x) + self.ffn.forward(x)

class SeasonalRepresentation(nn.Module):
    def __init__(self, time_bins, season=0.2):
        super().__init__()
        self.register_buffer("season", torch.tensor([[season]]))
        self.register_buffer("a_unscaled", torch.linspace(0,1,time_bins)[None, :])
    def forward(self, x):
        # param 0 : thickness | param 1 : age
        return (x[...,0,None] + torch.sin(6.2831853*(x[...,1,None]*self.a_unscaled-self.season)))*torch.exp(-x[...,1,None]*self.a_unscaled)

class Yeller(nn.Module):
    def __init__(self, name="Yeller"): super().__init__(); print(f'{name} iz born.'); self.name = name
    def forward(self, x): print(f'{name} iz c al l ed.'); return x

class VXTnn(nn.Module):
    def __init__(self, vxt_i_n, extra_layers=2, depth_bins=64, season=0.4):
        super().__init__()
        self.register_buffer("d_unscaled", torch.linspace(0,1,depth_bins)[None,:])
        #inputs of extFunc NN: thickness, age - float value
        vxt_seq = [SeasonalRepresentation(depth_bins, season)]
        vxt_seq += [Unsqueeze(1)]
        for i in range(extra_layers): vxt_seq += [FNOConv(), nn.GELU()]
        vxt_seq += [Squeeze(1)]
        self.vxt_fnn = nn.Sequential(*vxt_seq)
        
    def forward(self, x):
        #x shape : {batch_n, param_n}; params: (thickness, age, k_z)
        k_z = x[:, -1].unsqueeze(-1)
        
        ext = self.vxt_fnn.forward(x[:, :-1]) #real-valued extinction
        depths = self.d_unscaled * x[:,0,None] #scale depth bins by thickness
        mag, phs = torch.exp(ext*depths), torch.exp(1j*k_z*depths)
        vol_coh = (mag * phs).sum(-1) / mag.sum(-1)
        return vol_coh

# complex-valued PReLU for well-conditioned surface phase estimation
class cPReLU(nn.Module):
    def __init__(self): super().__init__(); self.rU = nn.PReLU(); self.iU = nn.PReLU()
    def forward(self, x): return torch.complex(self.rU.forward(x.real), self.iU.forward(x.imag))

class SFPnn(nn.Module): #surface phase extractor (experimental linear)
    def __init__(self, sfp_i_n, sfp_h_n):
        super().__init__()
        #inputs of spe NN: coherence, complex value
        nnl = [nn.Linear(sfp_i_n, sfp_h_n, dtype=torch.complex64), cPReLU(),
               nn.Linear(sfp_h_n, sfp_h_n, dtype=torch.complex64), cPReLU(),
               nn.Linear(sfp_h_n, 1, dtype=torch.complex64)]
        self.icesurf = nn.Sequential(*nnl)

    def forward(self, x):
        #x shape : {batch_n, param_n (complex)}
        return torch.angle(self.icesurf.forward(x))
        
class TorchRVI2(nn.Module):
    def __init__(self, coherences, incidence_angle, k_z, wavelength,
                 vxt_layers=1, depth_bins=64, season=0.4, pinit_means=(1.0,1.0), pinit_stds=(0.5,0.5)):
        super().__init__()
        # --- Convert input data to torch tensors ---
        self.register_buffer("coherences", torch.tensor(coherences, dtype=torch.complex64))
        self.register_buffer("k_z", torch.tensor(k_z, dtype=torch.float32))
        self.register_buffer("inc_angle", torch.tensor(incidence_angle, dtype=torch.float32))
        self.wavelength = wavelength

        # --- Parameter initialization ---
        B, H, W = self.coherences.shape
        #self.thickness = nn.Parameter(torch.full((H, W), 0.1))   # ice thickness (m)
        #self.age = nn.Parameter(torch.full((H, W), 1.0))        # surface age (years)
        tmp = torch.zeros(H,W).log_normal_(pinit_means[0], pinit_stds[0]).clamp(0,5)
        self.thickness = nn.Parameter(tmp)
        tmp = torch.zeros(H,W).log_normal_(pinit_means[1], pinit_stds[1]).clamp(0,10)
        self.age = nn.Parameter(tmp)
        
        self.sv_ratio = nn.Parameter(torch.full((B, H, W), 0.5))  # surf-vol ratio

        # VXT SETUP:
        # 2 input parameters
        # 3.0 m maximum ice depth (thickness)
        # 64 depth bins sampled from extinction function
        self.vxt = VXTnn(2, vxt_layers, depth_bins, season=season)
        self.steps = 0
        
        self.lcps0 = 16 #loss calculation patch size

        #self.register_buffer("icesurf", torch.tensor(self.icesurf_phase))
        self.sfp = SFPnn(coherences.shape[0], int(coherences.shape[0]//1.5))
        
    def forward(self):
        x1 = torch.stack((self.thickness, self.age, self.k_z), dim=-1).view(-1, 3)
        vch = self.vxt.forward(x1).view(self.age.shape)

        x2 = self.coherences.view(self.coherences.size(0), -1).transpose(0,1)
        sfp = self.sfp.forward(x2).view(self.age.shape)
        #sfp = self.icesurf
        
        cohs = torch.exp(1j * sfp) * (vch + self.sv_ratio) / (1 + self.sv_ratio)
        return cohs
    
    def loss_fn(self, predicted, target):
        #ph_ta_lin = torch.cov(torch.stack((self.thickness, self.age),dim=0).view(2, -1))[0,1] * 1e-5
        loss_lvar = predicted.unfold(1,self.lcps0,self.lcps0//2).unfold(2,self.lcps0,self.lcps0//2)
        loss_lvar = torch.nanmean(torch.std(loss_lvar, dim=(-1, -2)))
        loss_mse = torch.nanmean(torch.abs(predicted - target)**2)
        loss = loss_mse + loss_mse * (loss_lvar)# - ph_ta_lin)
        return loss

    def data_captured(self):
        """Return observed coherences (complex tensor)."""
        return self.coherences
        