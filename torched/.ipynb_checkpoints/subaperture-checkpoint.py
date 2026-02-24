import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def slc_fft_regularized(slc, device='cpu', dtype=torch.complex64):
    slc_fft = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(slc).to(dtype).to(device), dim=-2), dim=-2) #...,az(fft),rg
    fptype = slc_fft.real.dtype
    
    # step 1 - get smooth amplitude distribution
    std = slc_fft.shape[-2]//5
    g_kern = torch.exp(-torch.linspace(-3*std,3*std,slc_fft.shape[-2],device=device,dtype=fptype)**2/(2*std**2))/(6*std); g_kern /= g_kern.sum()
    azi_bin_mag = torch.abs(slc_fft).sum(dim=-1)  #...,az
    x1 = azi_bin_mag.mean(dim=-1, keepdim=True); x2 = azi_bin_mag < x1
    azi_bin_mag[x2] = x1.expand((-1,)*len(x1.shape[:-1])+(azi_bin_mag.shape[-1],))[x2]
    azi_bin_mag = torch.fft.irfft(torch.fft.rfft(azi_bin_mag, dim=-1) * torch.fft.rfft(g_kern[(None,)*len(azi_bin_mag.shape[:-1])], dim=-1))
    azi_bin_mag /= azi_bin_mag.max(dim=-1,keepdim=True)[0]
    
    # step 2 - fit hamming window
    fp = torch.nn.Parameter(torch.tensor([0.5,0,-0.8,0.8], requires_grad=True, dtype=fptype, device=device))
    lspc = torch.linspace(0,1,slc_fft.shape[-2], dtype=fptype, device=device)
    optimizer = torch.optim.Adam([fp], lr=0.05);
    for i in range(50):
        optimizer.zero_grad()
        hamc = torch.stack((torch.zeros(slc_fft.shape[-2],dtype=fptype, device=device), fp[2] + (fp[3] - fp[2]) * lspc), dim=-1)
        ham = (fp[0] - (1 - fp[0]) * torch.cos(2 * torch.pi * torch.arange(1000,device=device) / 999)).to(fptype)
        ham = F.grid_sample(ham[None,None,:,None], hamc[None,:,None,:], mode='bilinear', align_corners=True)[:,0,:,0] + fp[1]
        loss = ((azi_bin_mag**4 - ham**4)**2).mean()
        loss.backward(); optimizer.step()
    # step 3 - regularize slc_fft
    slc_fft = slc_fft / ham[...,None].clamp(1e-8).detach()
    return slc_fft

# Subaperture decomposition with 2 windows
def slc2sub2(slc, cut_factor=3, device='cpu', dtype=torch.complex64):
    return fft2sub2(slc_fft_regularized(slc, device=device, dtype=dtype), cut_factor=cut_factor, device=device, dtype=dtype)
def fft2sub2(slc, cut_factor=3, device='cpu', dtype=torch.complex64):
    sub_n = slc.shape[-2]//cut_factor
    hamming = torch.hamming_window(sub_n, False, alpha=0.75, device=device, dtype=slc.real.dtype)[(None,)*len(slc.shape[:-2])][...,None]
    slc_sub = torch.stack((slc[...,slc.shape[-2]//2-sub_n:slc.shape[-2]//2,:], slc[...,slc.shape[-2]//2:slc.shape[-2]//2+sub_n,:]), dim=0)
    slc_sub = torch.fft.ifft(slc_sub * hamming, dim=-2)
    return slc_sub

# Subaperture decomposition with arbitrary windowing
def slc2sub_u(slc,  sub_len=None, sub_step=None, cut_factor=3, device='cpu', dtype=torch.complex64):
    return fft2sub_u(slc_fft_regularized(slc, sub_len, sub_step, device=device, dtype=dtype), cut_factor=cut_factor, device=device, dtype=dtype)
def fft2sub_u(slc, sub_len=None, sub_step=None, cut_factor=3,  device='cpu', dtype=torch.complex64):
    end_cut = max(slc.shape[-2]//cut_factor//2, 1)
    if (sub_len is None): sub_len = (slc.shape[-2]-end_cut*2)//2
    if (sub_step is None): sub_step = sub_len
    hamming = torch.hamming_window(sub_len, False, alpha=0.75, device=device, dtype=slc.real.dtype)[(None,)*len(slc.shape[:-2])][...,None]
    slc_sub = slc[...,end_cut:-end_cut,:].unfold(dim=-2, size=sub_len, step=sub_step).moveaxis(-3,0).moveaxis(-1,-2)
    return torch.fft.ifft(slc_sub * hamming, dim=-2)


#def slc_mda_drift_single(slc, rg_cut=1.0, interp_factor=16, device='cpu', dtype=torch.complex64):
#    '''A Noniterative Map-Drift Autofocus Algorithm Using PCA for RDA-Based Stripmap SAR Imaging
#    Imanadar & Samadi (2024) - src: https://ieeexplore.ieee.org/document/10576625 '''
#    slc_reg = slc_fft_regularized(slc, device=device, dtype=dtype)
#    if (rg_cut < 1): # mask range bins of highest energy to reduce computation burden
#        cutoff_idx = torch.topk(torch.sum(torch.abs(slc_reg)**2,dim=1), int(rg_cut * slc_reg.shape[2]), 1, True, False).indices[:,None,:]
#        slc_reg = slc_reg.gather(dim=2, index=cutoff_idx.expand(-1,slc_reg.shape[1],-1))
#    slc_sub = slc2sub2_(slc_reg, device=device, dtype=dtype)
#
#    #1. covariance matrix formation + eigenvalue decomposition, extract highest-significance eigenvector
#    V = torch.linalg.eigh((slc_sub @ slc_sub.adjoint()) / slc_sub.shape[3])[1][..., -1] # N,B,az,az -> N,B,az
#
#    #2. circular convolution to calculate azimuthal shift
#    n = V.shape[2] * interp_factor
#    cc = torch.fft.fft(V[0],n=n) * torch.fft.fft(V[1],n=n).conj() #B,az*interp_factor
#    shift = torch.argmax(torch.torch.fft.ifft(cc), dim=-1).to(slc_reg.real.dtype) / interp_factor
#    shift = torch.where(shift > V.shape[2]/2, shift - V.shape[2], shift)
#
#    return shift
    










    