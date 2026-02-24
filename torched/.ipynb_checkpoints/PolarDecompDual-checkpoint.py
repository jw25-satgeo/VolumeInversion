import numpy as np
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm


def CoherenceBuilder(dpols, window=3, types=['dpol'], device='cuda', dtype=torch.complex64, tensor=True):
    '''returns a stack of complex coherences from dual-pol input signal using selected polarimetric basis transforms / coherence estimations.
    types: 'all', list() containing any of ['dpol', 'pauli', 'cmopt', 'pdopt']
    device: 'cuda', 'cpu'
    '''
    if (types=='all'): types = ['dpol', 'pauli','pdpco']
    b2c_stack = [] #list of (N,H,W,B) bases
    if ('dpol' in types):  #stack along basis axis (N, B,H,W -> N, H,W,B)
        b2c_stack += [torch.as_tensor(dpol).to(dtype).to(device).moveaxis(0,-1) for dpol in dpols]
    if ('pauli' in types):  #stack along basis axis (N, H,W,B -> N, H,W,B)
        b2c_stack += [DPauli(dpol, device=device, dtype=dtype) for dpol in dpols]
    if (len(b2c_stack) > 0):
        b2c_stack = torch.stack(b2c_stack, dim=0)
        b2c_stack = _complex_coh_perband(b2c_stack, window=window, device=device)
        c2c_stack = [b2c_stack]
    else: c2c_stack = [] #list of (P,H,W,B) coherences
        
    if ('pdpco' in types):
        c2c_stack.append(CMopt(qpol1, qpol2, window=window, device=device, dtype=dtype))
    if (len(c2c_stack) == 0):
        raise ValueError('No coherences in built list: ou must select at least one valid coherence set!')
    c2c_stack = torch.concat(c2c_stack, axis=-1) #LAST AXIS SHOULD BE BAND AXIS

    if (not tensor): return c2c_stack.detach().cpu().numpy()
    return c2c_stack
    
    
def _complex_coh_perband(basis, window=3, full=False, device='cuda'):
    '''tensor2tensor: (N,H,W,B) -> (P,H,W,B) or full (N,N,H,W,B)'''
    N, H, W, B = basis.shape
    omask = torch.ones((N,N),dtype=torch.bool,device=device).tril(-1) #NxN lower triangle mask excluding diagonal

    basis_p = basis.permute(1,2,3,0) #H,W,B,N form
    TN = basis_p[...,:,None] * basis_p[...,None,:].conj() #H,W,B,N,N via 1 transpose (T3 not needed, no basis transform)
    TN = TN.permute(2,3,4,0,1).view(B,N*N,H,W)
    TN = torch.complex(
        F.avg_pool2d(TN.real, window, 1, window//2),
        F.avg_pool2d(TN.imag, window, 1, window//2)
    ).view(B,N,N,H,W).permute(3,4,0,1,2)
    Tl = TN.diagonal(0,-2,-1).moveaxis(-1,0) #N,H,W,B
    Om = TN[..., omask].moveaxis(-1,0) #P,H,W,B : Oij per basis permutation P
    
    i, j = _k2ij_inverse_permutation_index(N, Om.shape[0], device=device)
    
    cohs = Om / torch.sqrt((Tl[i]) * (Tl[j]))
    cohs[torch.abs(cohs) > 1] = torch.nan

    if (full):
        coh_full = torch.ones(H,W,B,N,N, dtype=cohs.dtype, device=device)
        coh_full[...,i,j] = cohs.moveaxis(0,-1)
        coh_full[...,j,i] = cohs.moveaxis(0,-1).conj()
        return coh_full
    return cohs

def _k2ij_inverse_permutation_index(N, P, device='cuda'):
    k = torch.arange(P, device=device, dtype=torch.long)
    i = ((2*N - 1 - torch.sqrt((2*N-1)**2 - 8*k)) // 2).to(torch.long)
    j = i + 1 + (k - i*(2*N  - i - 1)//2)
    return i, j

def DPauli(dpol, transmit='v', device='cuda', dtype=torch.complex64, tensor=True):
    '''Co-pol & cross-pol scattering vector from Sentinel-1A type scattering vectors.
    src: Persistent Scatterer Analysis Using Dual-Polarization Sentinel-1 Data: Contribution From VH Channel
    (Shamshiri et al. 2018) - https://ieeexplore.ieee.org/document/8403894

    return (HH, 2HV) or (VV, 2VH) from (HH, HV) and (VH, VV).
    
    device: 'cuda', 'cpu'
    transmit: 'v', 'h'
    tensor: returns 'hwb', otherwise 'bhw' array '''

    A, B = torch.as_tensor(dpol).to(dtype).to(device)
    k = torch.zeros(dpol.shape, dtype=dtype, device=device)

    if (transmit == 'h'):
        k[0] = A
        k[1] = 2*B
    elif (transmit == 'v'):
        k[0] = B
        k[1] = 2*B
    else:
        raise ValueError("transmit mode must be 'h' or 'v'")
    
    if (tensor): return k.moveaxis(0, -1)
    return k.cpu().detach().numpy()

    
def PD_PCO(dpols, transmit='v', n=20, window=3, full=False, device='cuda', dtype=torch.complex64):
    '''returns the optimal complex intensity projection for maximum phase consistency across images.
    src : A phase-decomposition-based polarimetric coherence optimization method
    (Zhang et al. 2022) - https://www.sciencedirect.com/science/article/pii/S0303243422000976#s0010'''
    k = torch.stack([DPauli(dpol, transmit=transmit, device=device, dtype=dtype) for dpol in dpols], dim=-1)
    H, W, B, N = k.shape
    omask = torch.ones((N,N),dtype=torch.bool,device=device).tril(-1) #NxN lower triangle mask excluding diagonal

    p1, p2 = torch.meshgrid(torch.linspace(0, 2*torch.pi, n), torch.linspace(0, 2*torch.pi, n), indexing='ij')
    p1, p2 = p1.flatten()[:,None,None,None,None], p2.flatten()[:,None,None,None,None]
    w = torch.zeros((n*n,1,1,B,1), device=device, dtype=dtype)
    w[...,0,:] = torch.cos(p1)
    w[...,1,:] = torch.sin(p1) * torch.exp(1j * p2.T)  # band axis projection
    del(p1, p2)
    
    u = (k[None,...] * w.conj()).sum(dim=-2); del(w, k) #n*n,H,W,N
    u = u / torch.sqrt(F.avg_pool2d(torch.abs(u)**2, window, 1, window//2)) #normalize power
    C = u[...,:,None].conj() * u[...,None,:]; del(u) #n*n,H,W,N,N  ..  calculate coherence
    C = C.permute(3,4,0,1,2).view(N*N,n*n,H,W) #4d operation
    C = torch.complex(F.avg_pool2d(C.real, window, 1, window//2),
                      F.avg_pool2d(C.imag, window, 1, window//2)
        ).view(N,N,n*n,H,W).permute(2,3,4,0,1) #n*n,H,W,N,N
    idx_argm = torch.argmax(torch.linalg.eigvalsh(C)[...,-1], dim=0)[None,...,None,None].expand(1,H,W,N,N) #1,H,W,N,N
    C = torch.gather(C, 0, idx_argm)[0]; del(idx_argm) #H,W,N,N
    
    # stable complex coherence history extraction
    Po, Pp = torch.linalg.eigh(C)
    if (full): return Pp, Po #H,W,N,N | H,W,N
    return Pp[...,-1,None].moveaxis(-2,0), Po[...,-1,None]  #N,H,W,1 | H,W,1


    
