import numpy as np
import math
import pickle

import torch
import torch.nn.functional as F

import LBM4S_v0 as LBM4S
from torch_tps import ThinPlateSpline

class SimpleReflector():

    def __init__(self, er, T, cells_per_m, wavelength_m, batch_size=4):
        '''
        initialize from lattice simulation parameters:
        -- inputs
        er   : (B,C,Nx,Ny,Nz) complex tensor
        T    : (B,C,Nz) float tensor
        cells_per_m : scalar
        wavelength_m : scalar
        '''
        self.er = er
        self.T = T
        self.dtype = T.dtype
        self.device= T.device
        self.cells_per_m = cells_per_m
        self.wavelength_m = wavelength_m
        self.batch_size = batch_size

        # incident wavevectors must be consistent w. lattice periodic boundary
        #  λ_i*l_i = n_i; (λ_i^2 + ...) = λ^2; λ_i >= 0; λ_2 = 0;
        (B,C,Nx,Ny,Nz), periodic = er.shape, (0,1)
        v = 1/(wavelength_m * cells_per_m);  N = 4 * max(Nx,Ny,Nz) * v
        grid = np.stack(np.meshgrid(*[np.arange(0,N) for el in periodic],indexing='ij'),axis=-1)
        grid = (grid / np.array([er.shape[el+2] for el in periodic])[None,None,None,:])**2
        v_rem = v**2 - grid.sum(axis=-1)
        mask = (grid >= 0).all() * (v_rem > 0)
        if 1 in periodic: mask *= (grid[...,periodic.index(1)] == 0) #only xz-plane rotation
        v_set = np.zeros((*v_rem.shape, 3)); v_set[..., periodic] = grid; v_set[..., [el for el in (0,1,2) if el not in periodic]] = v_rem[...,None]
        v_set = v_set[mask]**0.5 #flattens to N,3
        self.ia_input = np.array([(np.acos(vi[-1]/v), np.atan2(vi[-2],vi[-3])) for vi in v_set])

    def save(self, fn='SimpleReflector_latest.pickle'):
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fn='SimpleReflector_latest.pickle'):
        with open(fn, 'rb') as f:
            return pickle.load(f)
    
    def slc_simulate(self, sit, rg, ia, T=0, rg_comp=None):
        '''
        simulates the propagation and return from scattering surface.
        -- inputs
        sit   : (az, rg) tensor  - sea ice thickness
        rg    : (az, rg) tensor  - SLC slant range
        ia    : (az, rg) tensor  - SLC incidence angle
        T     : (az, rg) tensor or scalar  - ice surface temperature
        '''
        k0 = 2*torch.pi/self.wavelength_m
        # get amplitude of propagation to surface
        amp = torch.exp(-1j*k0*rg)
        if (rg_comp is not None): amp = amp * rg_comp
        #if (rg_comp is not None): amp = amp * rg_comp# / rg
        
        # retrieve complex reflection coefficient
        if isinstance(T, torch.Tensor): T = T.flatten()
        crefl = self.tps_refl(sit.flatten(), T, ia.flatten()).view(list(amp.shape)+[4])

        # get reflection complex amplitude
        amp = amp[...,None] * crefl  # expand to 4-polarization
        
        # propagate back
        amp = amp * torch.exp(-1j*k0*rg)[...,None]
        
        return amp

    
    def generate(self):
        self.gen_refl_sample()
        self.gen_refl_spline()

    def tps_refl(self, sit, T, ia):
        '''
        Sample the generated spline at specified ice thicknesses, temperatures, and incidence angles.
        -- inputs
        sit: N,
        T  : N, (alt: scalar)
        ia : N,
        -- return
        refl: N,4 (complex) ~ 4-polarization return
        '''
        if (np.isscalar(T)): T = T*torch.ones_like(sit)
        samp = torch.stack((sit, T, ia), dim=-1)
        refl = self.tps.transform(samp).view(sit.shape[0], 4, 2)
        refl = torch.complex(refl[...,0],refl[...,1])
        return refl
        
    def gen_refl_sample(self, n_periods=4, ia_b=(0,55)):
        '''
        er (B,C,Nx,Ny,Nz) -> refl (B,C,pol,ia)
        '''
        B,C,Nx,Ny,Nz = self.er.shape
        dtype_c = (torch.ones((1), dtype=self.dtype) + 1j).dtype
        result = torch.zeros((B,C,4,len(self.ia_input)), device=self.device, dtype=dtype_c)
        batch_iters = math.ceil(self.er.shape[0] / self.batch_size)
        for i in range(len(self.ia_input)):
            if (np.rad2deg(self.ia_input[i,0]) > ia_b[1] or np.rad2deg(self.ia_input[i,0]) < ia_b[0]): continue
            print (f'processing incidence angle {np.rad2deg(self.ia_input[i])}, stage {i+1} of {len(self.ia_input)}')
            print('simulating H-transmit..')
            for j in range(batch_iters):
                j0, j1 = j*self.batch_size, (j+1)*self.batch_size
                ia_i = torch.as_tensor(self.ia_input[i]).to(self.device,self.dtype)[None,None,:].expand(j1-j0, C, -1)
                read, driver, kI = LBM4S.simulate( # output hv, hh
                    self.er[j0:j1], pol='H', wavelength_m=self.wavelength_m, cells_per_m=self.cells_per_m, n_periods=n_periods, inc=ia_i)
                result[j0:j1,...,(1,0),i] = LBM4S.get_scat_coeff(read, driver, kI, single_pixel=True, ignore_H=True); del(read, driver)
            print('simulating V-transmit..')
            for j in range(batch_iters):
                j0, j1 = j*self.batch_size, (j+1)*self.batch_size
                ia_i = torch.as_tensor(self.ia_input[i]).to(self.device,self.dtype)[None,None,:].expand(j1-j0, C, -1)
                read, driver, kI = LBM4S.simulate( # output vv, vh
                    self.er[j0:j1], pol='V', wavelength_m=self.wavelength_m, cells_per_m=self.cells_per_m, n_periods=n_periods, inc=ia_i)
                result[j0:j1,...,(3,2),i] = LBM4S.get_scat_coeff(read, driver, kI, single_pixel=True, ignore_H=True); del(read, driver)

        self.refl = result #B,C,4,i

    def gen_refl_spline(self, ia_b=(0,55)):
        '''
        refl (B,C,pol,ia) -> spline { (N,(sit,T,ia)) -> (N,(pol)) }
        '''
        thickness = (torch.abs(self.er).mean(dim=(-2,-3)) < 50).to(self.dtype).mean(dim=-1) * self.er.shape[-1]/self.cells_per_m
        ia_bounded = np.rad2deg(self.ia_input[:,0]); ia_bounded = ia_bounded[((ia_bounded > ia_b[0]) & (ia_bounded < ia_b[1]))]
        ia_bounded = np.deg2rad(ia_bounded)
        
        para_fit = torch.meshgrid(thickness.flatten(), torch.tensor(ia_bounded).to(thickness.device, thickness.dtype), indexing='ij') # (sample, inc)
        para_fit = (para_fit[0], self.T.max(dim=-1)[0][:,None].repeat(1,para_fit[0].shape[1]), para_fit[1])
        para_fit = torch.stack(para_fit, dim=-1).flatten(0,1) # (sample*inc,)
        
        refl_fit = self.refl.flatten(0,1).moveaxis(-1,1).flatten(0,1) # (sample, inc, ...)->(sample*inc,)
        refl_fit = torch.stack((refl_fit.real, refl_fit.imag),dim=-1).flatten(1) # ...,4,2 -> ...,8

        tps = ThinPlateSpline(alpha=0.005, device=self.device)
        tps.fit(para_fit, refl_fit)

        self.tps = tps





'''
Gaussian Random Field - to be used in surface roughness perturbation / random realization of sea ice cells
(implementation pending)
READ : https://www.sciencedirect.com/science/article/pii/0043164878901576
misc : https://github.com/fjprichard/PyAFBF
'''
def gaussian_field(alpha = 3.0, l_sc=(-444.0,114.0), size=(128,128), flag_normalize = True):
    """
    Returns a 2D tensor of shifted Fourier coordinates.
    -- input
    alpha (double, default = 3.0): The power of the power-law momentum distribution
    -- return
    gfield (2D tensor) : the gaussian field.
    """
    l_sc = torch.tensor(l_sc)[:,None,None]
    k_idx = torch.tensor(np.mgrid[:size[0], :size[1]] - (np.array([size[0], size[1]])[:,None,None]+1)/2)
    k_idx = torch.fft.fftshift(k_idx, dim=(-2,-1))

    # Defines the amplitude as a power law 1/(k^(alpha/2))
    amplitude = 1 / ((((l_sc + k_idx)**2).sum(dim=0) + 1e-10)**(-alpha/4.0))
    amplitude[0,0] = 0
    
    # Draws a complex gaussian random noise with normal (circular) distribution
    noise = torch.randn(size) + 1j * torch.randn(size)
    
    # To real space
    gfield = torch.fft.ifft2(noise * amplitude).real
    if flag_normalize:
        gfield = gfield - gfield.mean()
        gfield = gfield/gfield.std()
        
    return gfield







