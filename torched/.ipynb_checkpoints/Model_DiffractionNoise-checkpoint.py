import torch
import torch.nn.functional as F
import numpy as np
import scipy
import math
import PyPreprocess
import subaperture
import tqdm.auto as tqdm

# NOTE - refractive index and depth are free parameters! Constrain one as best as you can.
class SimpleLayer(torch.nn.Module):
    '''Autofocus-Based Estimation of Penetration Depth and Permittivity of Ice Volumes and Snow Using Single SAR Images
    src - (Benedikter et al.,2022) @ https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9648209)

    ia: incidence angle of SAR image
    rg: slant range
    '''
    def __init__(self, ia, rg, num_bands=4, device='cpu', dtype=torch.float32):
        super().__init__()
        self.register_buffer('ia', ia)
        self.register_buffer('H', rg * torch.cos(ia))
        
        self.n_ice_input = torch.nn.Parameter(torch.randn(1,num_bands,ia.shape[-2],ia.shape[-1], device=device,dtype=dtype))
        self.depth_input = torch.nn.Parameter(torch.randn(1,num_bands,ia.shape[-2],ia.shape[-1], device=device,dtype=dtype))

    def forward(self):
        self.n_ice = 1 + F.softplus(self.n_ice_input)
        self.depth = F.softplus(self.depth_input)
        #tmp = torch.cos(self.ia)/torch.cos(torch.asin(torch.sin(self.ia)/self.n_ice)) -- NaN instability
        tmp = torch.cos(self.ia)/((1 - ((torch.sin(self.ia)**2)/(self.n_ice**2)))**0.5) # analytically rewritten version of above
        zeta = (self.H*self.n_ice + tmp*self.n_ice**2) / (self.H*self.n_ice + tmp*self.depth)
        return zeta

def process_zeta_from_dataset(dataset,slc=None,size=(96,24),step=(48,12),rg_cut=1.0,interp_factor=16,device='cpu',dtype=torch.complex64,tqdm_disable=False):
    assert isinstance(dataset, PyPreprocess.Single_Dataset)
    az_len, rg_len = [(dataset.slc.shape[i+1]-size[i])//step[i] + 1 for i in range(len(size))]
    obv, wv, dt0, dt1, prf, azsp, azbw = (dataset.metadata['orbit_vector'], dataset.metadata['wv'][0], dataset.metadata['0dopp_t0'][0], dataset.metadata['0dopp_t1'][0], dataset.metadata['prf'][0], dataset.metadata['az_sp'][0], dataset.metadata['az_bw'][0])
    fptype = torch.zeros(1,dtype=dtype).real.dtype
    grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,az_len),torch.linspace(-1,1,rg_len),indexing='ij'),dim=-1).flip(-1).to(device).to(fptype)[None,]
    ia = F.grid_sample(torch.as_tensor(dataset.ia).to(device).to(fptype)[None,None,], grid, align_corners=True)[0,0]
    rg = F.grid_sample(torch.as_tensor(dataset.rg).to(device).to(fptype)[None,None,], grid, align_corners=True)[0,0]
    if (obv.size > 0): #Assumes ECEF (Earth-Centered Earth-Fixed) coordinate system : no surface rotation correction necessary
        tmp = scipy.interpolate.CubicHermiteSpline(obv[:,0],obv[:,1:4],obv[:,4:7],axis=0).derivative()(np.linspace(dt0,dt1,az_len))
        v0 = torch.as_tensor((tmp**2).sum(axis=1)**0.5, device=device, dtype=fptype)[:,None]
    else: v0 = prf * azsp
    az_t = torch.linspace(dt0,dt1,az_len, device=device, dtype=fptype)[:,None]
    if (slc is None): slc = torch.as_tensor(dataset.slc).to(device).to(dtype)
    return process_zeta(slc,(ia,rg,wv,v0,dt0,dt1,azbw),size,step,rg_cut,interp_factor,16,4,device,dtype,tqdm_disable)

def process_zeta(slc, phys, size=(96,24), step=(48,12), rg_cut=1.0, interp_factor=16, proc_bins=16, proc_iters=4, device='cpu',dtype=torch.complex64, tqdm_disable=False):
    '''Autofocus-Based Estimation of Penetration Depth and Permittivity of Ice Volumes and Snow Using Single SAR Images (Benedikter et al.,2021)
    slc: (B,az,rg) object
    phys: (ia, rg, wv, v0, azt0, azt1, azbw) : either scalar or 2D matching output spatial dimension
    '''
    slc = torch.as_tensor(slc).to(device).to(dtype)
    ia, rg, wv, v0, azt0, azt1, azbw = phys
    subaperture_azbw_factor = 0.333333333

    slc = torch.fft.ifft(subaperture.slc_fft_regularized(slc,device=device,dtype=dtype),dim=-2).unfold(-2,size[0],step[0]).unfold(-2,size[1],step[1])
    blk_dimn = len(slc.shape[:-4])
    block_az_t = torch.linspace((azt0-azt1)/2,(azt1-azt0)/2,size[0],dtype=slc.real.dtype, device=device)[blk_dimn*(None,)+(...,None,)]
    
    dopp_r_fs = torch.as_tensor(2*v0**2 / (rg * wv)).to(device).to(slc.real.dtype)[blk_dimn*(None,)+(...,)]
    dopp_r_err_cum = torch.ones(slc.shape[:-2], dtype=slc.real.dtype, device=device) * dopp_r_fs
    #for n in range(0, slc.shape[-4]-1, proc_bins):
    for n in tqdm.trange(0, slc.shape[-4]-1, proc_bins, desc='azimuth slices', disable=tqdm_disable):
        slc_proc = slc[...,n:n+proc_bins,:,:,:].detach().clone()
        for i in range(proc_iters):
            azi_drift = _mda(slc_proc, rg_cut=rg_cut, interp_factor=interp_factor, device=device)
            dopp_r_err = (2 * azi_drift / ((azbw*subaperture_azbw_factor)**2)) * dopp_r_err_cum[...,n:n+proc_bins,:]**2
            slc_proc = slc_proc.clone() * torch.exp(-1j*torch.pi*dopp_r_err[...,None,None]*block_az_t**2)
            dopp_r_err_cum[...,n:n+proc_bins,:] += dopp_r_err

    zeta = 1 + dopp_r_err_cum / dopp_r_fs #ratio between observed Doppler rate and free-space Doppler rate

    return zeta, ia, rg, dopp_r_err_cum, dopp_r_fs

def _sublook_pca(slc_batch, sub_n=16, sub_overlap=0.3, device='cpu'):
    
    # input: ...,az,rg
    sub_len, sub_step = slc_batch.shape[-2]//sub_n,(slc_batch.shape[-2]*(1 - sub_overlap))//sub_n
    slc_sub = subaperture.fft2sub_u(torch.fft.fft(slc_batch, dim=-2), sub_len, sub_step, device=device, dtype=slc_fft.dtype) #N,...,az,rg2

    # sub-aperture covariance; largest-eigenvalue maps to 
    V = torch.linalg.svd(slc_sub, full_matrices=False, dim=0)[0]
    # range covariance + eigenvalue, extract highest-significance eigenvector : N,...,az,rg2 -> N,...,az
    V = torch.linalg.svd(slc_sub, full_matrices=False)[0][..., 0] # SVD interpretation, still memory hungry
    #V = torch.randn(*slc_sub.shape[:-1], dtype=slc_sub.dtype, device=slc_sub.device)[...,None] # Power iteration method
    #for _ in range(pca_iters): V = slc_sub @ (slc_sub.adjoint() @ V); V /= V.norm(dim=-1, keepdim=True) + 1e-12
    #V = V.squeeze(-1).real

    return shift


def unwrap_phase(phi, dim=-1):
    """
    GPT-SOURCED : Torch-native phase unwrap along dimension `dim`.
    Matches numpy.unwrap behavior for 1D unwrapping.
    """
    dphi = torch.diff(phi, dim=dim); dphi_mod = (dphi + math.pi) % (2 * math.pi) - math.pi
    dphi_mod = torch.where((dphi_mod == -math.pi) & (dphi > 0), math.pi, dphi_mod)
    correction = torch.cumsum(dphi_mod - dphi, dim=dim)
    pad_shape = list(phi.shape); pad_shape[dim] = 1
    return phi + torch.cat([torch.zeros(pad_shape, device=phi.device, dtype=phi.dtype), correction], dim=dim)


    
'''def slc_mda_blocks(slc, size=(96,24),step=(48,12), rg_cut=1.0, pca_iters=50, interp_factor=16, device='cpu', dtype=torch.complex64):
    #x = torch.as_tensor(slc).unfold(1,size[0],step[0]).unfold(2,size[1],step[1]); x[torch.isnan(x)] = 1e-8
    fft_reg = subaperture.slc_fft_regularized(slc, device=device, dtype=dtype) #B,#,#,az,rg
    #fft_reg = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(slc).to(dtype).to(device), dim=-2), dim=-2) #...,az(fft),rg
    fft_reg = torch.fft.fft(torch.fft.ifft(fft_reg, dim=-2).unfold(1,size[0],step[0]).unfold(2,size[1],step[1]), dim=-2)
    if (rg_cut < 1):
        cutoff_idx = torch.topk(torch.sum(torch.abs(fft_reg)**2,dim=-2), int(rg_cut * fft_reg.shape[-1]), -1, True, False).indices[...,None,:]
        fft_reg = fft_reg.gather(dim=-1, index=cutoff_idx.expand(-1,-1,-1,fft_reg.shape[-2],-1)) #B,#,#,az,rg2
    slc_sub = subaperture.fft2sub2(fft_reg, device=device, dtype=dtype) #N,B,#,#,az,rg2
    
    #1. covariance matrix formation + eigenvalue decomposition, extract highest-significance eigenvector : N,B,#,#,az,az -> N,B,#,#,az
    #V = torch.linalg.eigh((slc_sub @ slc_sub.adjoint()) / slc_sub.shape[-1])[1][..., -1] # eigh explodes memory
    #V = torch.linalg.svd(slc_sub, full_matrices=False)[0][..., 0] # SVD interpretation, still memory hungry
    V = torch.randn(*slc_sub.shape[:-1], dtype=dtype, device=slc_sub.device)[...,None]   # Power iteration method for PCA
    for _ in range(pca_iters): V = slc_sub @ (slc_sub.adjoint() @ V); V /= V.norm(dim=-1, keepdim=True) + 1e-12
    V = V.squeeze(-1).real

    #2. circular convolution to calculate azimuthal shift
    n = V.shape[-1] * interp_factor
    cc = torch.fft.fft(V[0],n=n,dim=-1) * torch.fft.fft(V[1],n=n,dim=-1).conj() #B,#,#,az*interp_factor
    shift = torch.argmax(torch.abs(torch.fft.ifft(cc)), dim=-1).to(fft_reg.real.dtype) / interp_factor
    shift = torch.where(shift > V.shape[-1]/2, shift - V.shape[-1], shift) #B,#,#

    return shift'''
