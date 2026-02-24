import torch
import torch.nn.functional as F
import numpy as np
import scipy
import math

class IceVVHHr(torch.nn.Module):
    '''Discrimination of open water from sea ice in the Labrador Sea using quad-polarized synthetic aperture radar
    (src: https://www.sciencedirect.com/science/article/pii/S0034425720303187)
    
    Models the VV/HH backscatter ratio (vvhhr) at a known incidence angle using relative dielectric constant of the surface (rdie, param[0]) and standard deviation of surface orientation angle shift (stdv, param[1]), representing surface roughness, as parameters.'''
    def __init__(self, shape, rdie_0=3.175, stdv_0=0.2, device='cpu', dtype=torch.float32):
        super().__init__()
        rdie = torch.ones(shape,device=device,dtype=dtype)*rdie_0#(torch.rand(shape, device=device, dtype=dtype) + 1) * rdie_0
        stdv = torch.ones(shape,device=device,dtype=dtype)*stdv_0#(torch.rand(shape, device=device, dtype=dtype) + 1e-6) * stdv_0
        self.rdie = torch.nn.Parameter(rdie)
        self.stdv = torch.nn.Parameter(stdv)
        #self.leak = torch.nn.Parameter(torch.
        

    def forward(self, ia):
        F_VV = (self.rdie - 1) * (torch.sin(ia)**2 - self.rdie*(1 + torch.sin(ia)**2)) / (self.rdie * torch.cos(ia) + torch.sqrt(self.rdie - torch.sin(ia)**2))
        F_HH = (torch.cos(ia) - torch.sqrt(self.rdie - torch.sin(ia)**2)) / (torch.cos(ia) + torch.sqrt(self.rdie - torch.sin(ia)**2))
        rough = math.sqrt(3) * self.stdv / torch.sin(ia)
        A = 0.5*(1+torch.sin(2*rough)/2*rough)*F_VV**2 + 0.5*(1-torch.sin(2*rough)/2*rough)*F_HH**2 + 0.125*(1-torch.sin(4*rough)/4*rough)*(F_VV-F_HH)**2
        B = 0.5*(1+torch.sin(2*rough)/2*rough)*F_HH**2 + 0.5*(1-torch.sin(2*rough)/2*rough)*F_VV**2 + 0.125*(1-torch.sin(4*rough)/4*rough)*(F_VV-F_HH)**2
        return A/B

    def loss_fn(self, modelled, captured):
        """Mean square difference (L1 loss on magnitude)."""
        return torch.nanmean(torch.abs(modelled - captured)**2)