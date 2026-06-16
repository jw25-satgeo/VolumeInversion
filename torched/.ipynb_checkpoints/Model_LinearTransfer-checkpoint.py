
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Simulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

    def loss(self, x, lossfunc):
        return

class RESICE_Diffusion(nn.Module):

    def __init__(self, dtype=torch.complex64, device='cpu'):
        super().__init__()
        self.layer = nn.Sequential([
            nn.Linear(1,1,dtype=dtype, device=device),
            nn.PReLU()  # hello i am a gnome and i 
        ])

    def forward(self, x):
        return


class PSF():
    
    def __init__(self, size=16, dtype=torch.float32, device='cpu'):
        self.size = size

    def estimation(self, image, rg, ia):
        psf = torch.randn((size,size), dtype=self.dtype, device=self.device)
        

    def apply(self, data):
        '''Warps the data before convolving to simulate adaptive kernel.
        src : https://github.com/sheliak/varconvolve/blob/master/varconvolve.py'''
        result = data
        return result

class RadarGeom_L1():

    def __init__(self, wv):
        self.wv = wv

    def solve_jones(sl_range, inc_angle, ref_coeff):
        '''use the four reflection coefficients to simulate an HH,HV,VH,VV image.
        sl_range  : B,C,H,W
        inc_angle : B,C,H,W
        ref_coeff : B,C,H,W,4 , provided in order (ss, sp, ps, pp)
        '''
        phase = torch.exp(1j * 2 * 2 * torch.pi * sl_range % self.wv)

    def speckle():
        