import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy

import Model_RVIBase


# --- NN utility classes --- #

class TayFouSeries(nn.Module):
    def __init__(self, cta_n, cfa_n, cfb_n, bins):
        super().__init__()
        self.register_buffer("pos", torch.linspace(0,1,bins).unsqueeze(-1))
        self.cn = (cta_n, cta_n + cfa_n, cta_n + cfa_n + cfb_n)
        
    def forward(self, x):
        # shape: {batch_n, param_n}
        ct_a, cf_a, cf_b = x[:,:self.cn[0]], x[:,self.cn[0]:self.cn[1]], x[:,self.cn[1]:self.cn[2]]

        #taylor component
        degree = torch.arange(ct_a.size(-1), device=x.device).reshape(1,1,-1)
        rt = (ct_a.unsqueeze(-2) * self.pos**degree).sum(-1)

        #fourier component
        degree = torch.arange(1, cf_a.size(-1)+1, device=x.device).reshape(1,1,-1)
        rf = (cf_a.unsqueeze(-2) * torch.cos(6.2831853 * degree * (self.pos + cf_b.unsqueeze(-2)))).sum(-1)
        return rt + rf #dim: batch, bins

class Squeeze(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim;
    def forward(self, x): return x.squeeze(self.dim)
class Unsqueeze(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim;
    def forward(self, x): return x.unsqueeze(self.dim)
class PlusOne(nn.Module):
    def forward(self, x): return x + 1
#def nanscream(x):
#    if torch.isnan(x).any():
#        raise ValueError("aaa")

class VXTnn(nn.Module):
    def __init__(self, vxt_i_n, vxt_tf_n=(4,12), extra_layers=2, max_depth=5.0, depth_bins=64):
        super().__init__()
        #register depths
        self.register_buffer("depths", torch.linspace(0,max_depth,depth_bins)[None,:])
        #inputs of extFunc NN: thickness, age - float value
        vxt_h_n = int((vxt_i_n + vxt_tf_n[0] + vxt_tf_n[1]) // 1.5)
        vxt_seq = [nn.Linear(vxt_i_n, vxt_h_n, dtype=torch.float32),nn.ELU()]
        for i in range(extra_layers):
            vxt_seq.append(nn.Linear(vxt_h_n, vxt_h_n))
            vxt_seq.append(nn.ELU())
        vxt_seq += [nn.Linear(vxt_h_n, vxt_tf_n[0]+vxt_tf_n[1]*2),
                    nn.Tanh(),
                    TayFouSeries(vxt_tf_n[0], vxt_tf_n[1], vxt_tf_n[1], depth_bins),
                    nn.ELU(), PlusOne()]
                    #Unsqueeze(1),nn.Conv1d(1, 1, depth_bins//8, padding='same'),Squeeze(1),]
        self.vxt_tayfou = nn.Sequential(*vxt_seq)
        self.vxt_tayfou.register_forward_hook(self.accumulate_phys_loss)

        self.phys_loss

    def accumulate_phys_loss(self, module, input, output):
        # step 1 : convert extinction to ice-brine mixture
        # step 2 : calculate T-S boundary of ice-brine mixture
        # step 3 : assign probability function to T-S boundary
        # step 4 : calculate uniformity of heat flux through ice column
    
    def forward(self, x):
        #x shape : {batch_n, param_n}; params: thickness, age, k_z
        #cut = torch.fmax((x[:,0,None]-self.depths)/self.depths[0,1],  x[:,0,None]<self.depths)
        cut = torch.sigmoid((x[:,0,None] - self.depths) / self.depths[0,1])
        k_z = x[:, -1].unsqueeze(-1)
        ext = self.vxt_tayfou.forward(x[:, :-1])
        mag, phs = torch.exp(-ext), torch.exp(1j*k_z*self.depths)
        vol_coh = (mag * phs * cut).sum(-1) / mag.sum(-1)
        return vol_coh

class SFPnn(nn.Module):
    def __init__(self, sfp_i_n, sfp_h_n):
        #surface phase extractor (experimental linear)
        #inputs of spe NN: coherence amp, coherence phase, sv_ratio parameters (12 minimum) - float value
        sfp_i_n, sfp_h_n = coherences.shape[0]*3, int(coherences.shape[0]*1.5)
        nnl = [nn.Linear(sfp_i_n, sfp_h_n), nn.PReLU(),
               nn.Linear(sfp_h_n, 1)]
        self.icesurf = nn.Sequential(*nnl)

    def forward(self, x):
        #x shape : {batch_n, param_n}
        x = self.icesurf.forward().unsqueeze(-1)
        return torch.angle(x)
        
class TorchRVI2(Model_RVIBase.TorchRVI):
    def __init__(self, coherences, incidence_angle, k_z, wavelength, vxt_layers=1):
        super().__init__(coherences, incidence_angle, k_z, wavelength, False)
        # VXT SETUP:
        # 2 input parameters
        # 4 taylor components, 32 fourier components
        # 3.0 m maximum ice depth (thickness)
        # 64 depth bins sampled from extinction function
        self.vxt = VXTnn(2, (3,32), vxt_layers, 3.0, 64)
        #self.sfp = SFPnn()
        #self.register_buffer("t_measured", torch.tensor(params_measured[0]))
        #self.register_buffer("a_measured", torch.tensor(params_measured[1]))
        #self.t_measured_max = torch.max(self.t_measured)
        #self.a_measured_max = torch.max(self.a_measured)
        #self.punt_steps = punt_steps
        self.steps = 0
        self.lcps0 = 16 #loss calculation patch size
        
    def forward(self):
        x = torch.stack((self.thickness, self.age, self.k_z), dim=-1).view(-1, 3)
        vch = self.vxt.forward(x).view(self.age.shape)
        cohs = torch.exp(1j * self.icesurf_phase) * (vch + self.sv_ratio) / (1 + self.sv_ratio)
        return cohs

    def loss_fn(self, predicted, target):
        p_var = predicted.unfold(1,self.lcps0,self.lcps0//2).unfold(2,self.lcps0,self.lcps0//2)
        p_var = torch.nanmean(torch.std(p_var, dim=(-1, -2)))
        pt_mse = torch.nanmean(torch.abs(predicted - target)**2)
        loss = pt_mse + pt_mse * p_var
                
        #if (self.steps < self.punt_steps):
        #    sim_loss = torch.nanmean(((self.thickness - self.t_measured)/self.t_measured_max)**2)
        #    sim_loss += torch.nanmean(((self.age - self.a_measured)/self.a_measured_max)**2)
        #    return loss + (sim_loss) * loss * 0.1
        return loss
        