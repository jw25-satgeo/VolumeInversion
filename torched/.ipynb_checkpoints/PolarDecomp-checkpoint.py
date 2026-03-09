import numpy as np
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm


def CoherenceBuilder(qpol1, qpol2, window=3, types=['qpol'], device='cuda', dtype=torch.complex64, tensor=True):
    '''returns a stack of complex coherences from quad-pol input signal using selected polarimetric basis transforms / coherence estimations.
    types: 'all', list() containing any of ['qpol', 'pauli', 'circular', 'cmopt', 'pdopt']
    device: 'cuda', 'cpu'
    '''
    if (types=='all'): types = ['qpol', 'pauli', 'circular', 'cmopt', 'pdopt']
    b2c_stack = [] #list of (N,H,W,B) bases
    if ('qpol' in types):  #stack along basis axis (B,H,W -> N,H,W,B)
        b2c_stack.append(torch.stack((torch.as_tensor(qpol1).to(dtype).to(device),
                                      torch.as_tensor(qpol2).to(dtype).to(device)), dim=-1).permute(3,1,2,0))
    if ('pauli' in types):  #stack along basis axis (H,W,B -> N,H,W,B)
        b2c_stack.append(torch.stack((Pauli(qpol1, device=device, dtype=dtype),
                                      Pauli(qpol2, device=device, dtype=dtype)), dim=-1).moveaxis(-1,0))
    if ('circular' in types):
        b2c_stack.append(torch.stack((Circu(qpol1, device=device, dtype=dtype),
                                      Circu(qpol2, device=device, dtype=dtype)), dim=-1).moveaxis(-1,0))
    if (len(b2c_stack) > 0):
        b2c_stack = torch.concat(b2c_stack, axis=-1) #LAST AXIS SHOULD BE BAND AXIS
        b2c_stack = _complex_coh_perband(b2c_stack, window=window, device=device)
        c2c_stack = [b2c_stack]
    else: c2c_stack = [] #list of (P,H,W,B) coherences
        
    if ('cmopt' in types):
        c2c_stack.append(CMopt(qpol1, qpol2, window=window, device=device, dtype=dtype))
    if ('pdopt' in types):
        c2c_stack.append(PDopt(qpol1, qpol2, window=window, device=device, dtype=dtype))
    if (len(c2c_stack) == 0):
        raise ValueError('No coherences in built list: ou must select at least one valid coherence set!')
    c2c_stack = torch.concat(c2c_stack, axis=-1) #LAST AXIS SHOULD BE BAND AXIS

    if (not tensor): return c2c_stack.detach().cpu().numpy()
    return c2c_stack
    
    
def _complex_coh_perband(basis, window=3, device='cuda'):
    '''tensor2tensor: (N,H,W,B) -> (P,H,W,B)'''
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
    
    return cohs

def _k2ij_inverse_permutation_index(N, P, device='cuda'):
    k = torch.arange(P, device=device, dtype=torch.long)
    i = ((2*N - 1 - torch.sqrt((2*N-1)**2 - 8*k)) // 2).to(torch.long)
    j = i + 1 + (k - i*(2*N  - i - 1)//2)
    return i, j

def Pauli(qpol, device='cuda', dtype=torch.complex64, tensor=True):
    '''returns HHpVV, HHmVV, HVpVH, HHVV* from quad-pol input signal.
    device: 'cuda', 'cpu'
    ax_order: 'bhw'(default),'hwb' '''
    hh, hv, vh, vv = torch.as_tensor(qpol).to(dtype).to(device) #dim 0 is length 4
    pauli = torch.zeros(qpol.shape, dtype=dtype).to(device)

    pauli[0] = hh + vv
    pauli[1] = hh - vv
    pauli[2] = hv + vh
    pauli[3] = 1j*(hv - vh)
    
    pauli /= np.sqrt(2)

    if (tensor): return pauli.moveaxis(0, -1)
    return pauli
    

def Circu(qpol, device='cuda', dtype=torch.complex64, tensor=True):
    'returns LL, LR, RL, RR* from quad-pol input signal.'
    hh, hv, vh, vv = torch.as_tensor(qpol).to(dtype).to(device)
    hvr, vhr = 1j*hv, 1j*vh
    circu = torch.zeros(qpol.shape, dtype=dtype, device=device)
    circu[0] = hh + hvr - vhr + vv
    circu[1] = hh - hvr - vhr - vv
    circu[2] = hh + hvr + vhr - vv
    circu[3] = hh - hvr + vhr + vv

    circu /= 2
    
    if (tensor): return circu.moveaxis(0,-1)
    return circu

def T3_Rotated(qpol, device='cuda', dtype=torch.complex64):
    C, H, W = qpol.shape
    pauli = Pauli(qpol, device=device, dtype=dtype)
    T3 = pauli[...,:3,None] @ pauli[...,:3,None].adjoint() # Hermite matrix 3x3 T3 coherence
    #rotation 1
    tr = 2 * 0.25 * (torch.atan(-2*T3[...,1,2].real/(T3[...,1,1] - T3[...,2,2])).real % np.pi)
    rotary = torch.zeros((H,W,3,3), dtype=T3.dtype, device=device)
    rotary[..., 0, 0] = 1
    rotary[..., 1, 1] = torch.cos(tr)
    rotary[..., 1, 2] = torch.sin(tr)
    rotary[..., 2, 1] = -torch.sin(tr)
    rotary[..., 2, 2] = torch.cos(tr)
    T3 = rotary @ T3 @ rotary.adjoint()
    #rotation 2
    ti = 2 * 0.25 * (torch.atan(-2*T3[...,1,2].imag/(T3[...,1,1] - T3[...,2,2])).real % np.pi).to(device)
    rotary[..., 1, 1] = torch.cos(ti)
    rotary[..., 1, 2] = torch.sin(ti)
    rotary[..., 2, 1] = -torch.sin(ti)
    rotary[..., 2, 2] = torch.cos(ti)
    T3 = rotary @ T3 @ rotary.adjoint()
    return T3

# " An Optimal Polarization SAR Three-Component Target Decomposition Based on Semi-Definite Programming"
# src: Wang et al. (2023) - https://www.mdpi.com/2072-4292/15/22/5292?utm_source=chatgpt.com
# "Modified version of three-component model-based decomposition for polarimetric SAR data" 
# src : Zhang, Yu, & Wang (2019) - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8710375
def FreemanPlus(qpol, iterations=20, lr=0.05, device='cuda', dtype=torch.complex64, tensor=True, print_loss=False):
    
    C, H, W = qpol.shape
    T3 = T3_Rotated(qpol, device=device, dtype=dtype)

    model = torch.zeros((6,3,3), dtype=T3.dtype, device=device)
    model[0] = torch.tensor([[1,1,0],[1,1,0],[0,0,0]]) #bragg
    model[1] = torch.tensor([[1,1,0],[1,1,0],[0,0,0]]) #dihedral
    model[2] = torch.tensor([[2,0,0],[0,1,0],[0,0,1]])/4 #vol_uniform
    model[3] = torch.tensor([[15,-5,0],[-5,7,0],[0,0,8]])/30 #vol_horiz_thin
    model[4] = torch.tensor([[15,5,0],[5,7,0],[0,0,8]])/30 #vol_verti_thin
    model[5] = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])/3 #vol_entropic
    #model = torch.as_tensor(model).to(device)

    model = model.repeat((H, W, 1, 1, 1))#[None, None, :, :, :]

    B = torch.as_tensor((qpol[0] - qpol[3]) / qpol[0] + qpol[3]).to(T3.dtype).to(device); Bab = torch.abs(B)**2 
    A = 1 / B; Aab = torch.abs(A)**2
    #start filling in B for Bragg
    model[...,0,0,1] = model[...,0,0,1] * B.conj() / (Bab+1)
    model[...,0,1,0] = model[...,0,1,0] * B / (Bab+1)
    model[...,0,1,1] = model[...,0,1,1] * Bab / (Bab+1)
    #start filling in A for dihedral
    model[...,1,0,0] = model[...,1,0,0] * Aab / (Aab+1)
    model[...,1,0,1] = model[...,1,0,1] * A / (Bab+1)
    model[...,1,1,0] = model[...,1,1,0] * A.conj() / (Bab+1)

    p6 = torch.nn.parameter.Parameter(torch.zeros((H, W, 6), dtype=T3.real.dtype).to(device))
    optimizer = torch.optim.Adam([p6], lr=lr)
    loss_hist = torch.zeros(iterations)
    for i in range(iterations):
        optimizer.zero_grad()
        powers = torch.softmax(p6, dim=-1)
        loss = torch.nanmean(torch.abs(torch.sum(powers[...,None,None] * model, axis=2) - T3)**2)
        loss_hist[i] = loss
        loss.backward()
        optimizer.step()
    if(print_loss): print(loss_hist)
    if (tensor): return powers # H, W, band
    return np.moveaxis(powers.detach().cpu().numpy(), -1, 0)  # band, H, W

# based on Kapok implementation: https://github.com/simard-landscape-lab/kapok/blob/main/kapok/cohoptp.py
def PDopt(qpol1, qpol2, reg=0.5, window=3, iterations=30, device='cuda', dtype=torch.complex64, tqdm_disable=True):
    pauli = torch.stack((Pauli(qpol1, device=device, dtype=dtype), Pauli(qpol2, device=device, dtype=dtype)), dim=-1)
    
    H, W, B, N = pauli.shape

    omask = torch.ones((N,N),dtype=torch.bool,device=device).tril(-1) #NxN lower triangle mask excluding diagonal
    
    TN = pauli[...,:,None,:,None] * pauli[...,None,:,None,:].conj() #H,W,B,B,N,N (e.g. C11,C12,C21,C22) via 2 transposes
    TN = torch.nan_to_num(TN, nan=1).permute(2,3,4,5,0,1).view(B*B,N*N,H,W)
    TN = torch.complex(
        F.avg_pool2d(TN.real, window, 1, window//2),
        F.avg_pool2d(TN.imag, window, 1, window//2)
    ).view(B,B,N,N,H,W).permute(4,5,0,1,2,3)
    Tm = TN.diagonal(0,-2,-1).mean(dim=-1)[None,...] #1,H,W,B,B : polarimetric covariance, averaged over looks
    Tm = TN[..., 0, 0] #1,H,W,B,B : just get the T11 element as done in the Kapok implementation
    Om = TN[..., omask].moveaxis(-1,0) #P,H,W,B,B : interferometric covariance for each basis permutation P

    if reg > 0:
        mask = torch.eye(B, dtype=torch.bool, device=device)
        Tm[...,mask] = Tm[...,mask] + reg*Tm[...,mask].sum(axis=-1)[...,None]
        Om[...,mask] = Om[...,mask] + reg*Om[...,mask].sum(axis=-1)[...,None]
    
    pbase = torch.ones((1,H,W,1,1), dtype=Tm.real.dtype, device=device) * torch.pi
    nanmask = torch.isnan(Tm)

    dist = torch.zeros((Om.shape[0],H,W), dtype=Tm.real.dtype, device=device)
    pd_cohs = torch.zeros(Om.shape[0],H,W,2, dtype=Tm.dtype, device=device)
    for i in tqdm.trange(iterations, desc='rotations', disable=tqdm_disable):
        phase = pbase * ((i+1)/iterations)
        z12 = Om * torch.exp(1j * phase)
        z12 = (z12 + z12.adjoint())/2

        tmp = torch.linalg.solve_ex(Tm,z12)[0] #(torch.linalg.inv(Tm) @ z12)
        L, Q = torch.linalg.eig(tmp.masked_fill(torch.isnan(tmp),0))

        r = torch.argsort(L.real, dim=-1, descending=True)
        cohs = (Q.adjoint() @ Om @ Q).diagonal(0,-2,-1) / torch.abs(Q.adjoint() @ Tm @ Q).diagonal(0,-2,-1)
        cohs = cohs.take_along_dim(r, -1) #P,H,W,B

        n_dist = torch.abs(cohs[...,0] - cohs[...,-1]) #P,H,W
        mask = (n_dist > dist)[...,None].repeat(1,1,1,2)
        pd_cohs[mask] = cohs[...,[0,-1]][mask]
    
    return pd_cohs #P,H,W,2
    
def CMopt(qpol1, qpol2, window=3, device='cuda', dtype=torch.complex64):
    ''' Coherence magnitude optimization
    (Cloude & Papathassaniou, 1998) @ https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=718859'''
    pauli = torch.stack((Pauli(qpol1, device=device, dtype=dtype), Pauli(qpol2, device=device, dtype=dtype)), dim=-1)[...,:4,:]
    #pauli.real = F.avg_pool2d(pauli.real.moveaxis(-1,0), window, 1, window//2).moveaxis(0,-1)
    #pauli.imag = F.avg_pool2d(pauli.imag.moveaxis(-1,0), window, 1, window//2).moveaxis(0,-1)
    H, W, B, N = pauli.shape

    omask = torch.ones((N,N),dtype=torch.bool,device=device).tril(-1) #NxN lower triangle mask excluding diagonal
    
    TN = pauli[...,:,None,:,None] * pauli[...,None,:,None,:].conj() #H,W,B,B,N,N (e.g. C11,C12,C21,C22) via 2 transposes
    TN = torch.nan_to_num(TN, nan=1).permute(2,3,4,5,0,1).view(B*B,N*N,H,W)
    TN = torch.complex(
        F.avg_pool2d(TN.real, window, 1, window//2),
        F.avg_pool2d(TN.imag, window, 1, window//2)
    ).view(B,B,N,N,H,W).permute(4,5,0,1,2,3)
    Tl = TN.diagonal(0,-2,-1).moveaxis(-1,0) #N,H,W,B,B
    Tli = torch.linalg.inv_ex(Tl)[0] #N,H,W,B,B : inv of Tl per look n in N
    Om = TN[..., omask].moveaxis(-1,0) #P,H,W,B,B : Oij per basis permutation P
    
    i, j = _k2ij_inverse_permutation_index(N, Om.shape[0], device=device)
    
    eig_input = torch.linalg.solve_ex(Tli[i],Om)[0] @ torch.linalg.solve_ex(Tli[j],Om.adjoint())[0]
    L1, Q1 = torch.linalg.eig(torch.nan_to_num(eig_input, 0))
    eig_input = torch.linalg.solve_ex(Tli[j],Om.adjoint())[0] @ torch.linalg.solve_ex(Tli[i],Om)[0]
    L2, Q2 = torch.linalg.eig(torch.nan_to_num(eig_input, 0))
    #if (torch.abs(L2 - L1).sum() > 1e-2): raise RuntimeError("accumulated difference of L1 and L2 eigenvalues above 1e-2: enforce Hermitian symmetry before optimizing")
    r = torch.argsort(torch.abs(L1), dim=-1, descending=True)
    cohs = (Q1.adjoint() @ Om @ Q2).diagonal(0,-2,-1) / torch.sqrt((Q1.adjoint() @ Tl[i] @ Q1) * (Q2.adjoint() @ Tl[j] @ Q2)).diagonal(0,-2,-1)
    cohs[torch.abs(cohs) > 1] = torch.nan
    
    return cohs.take_along_dim(r,-1) #P,H,W,B
        
# "A phase-decomposition-based polarimetric coherence optimization method"
# src: Zhang et al. (2022) https://www.sciencedirect.com/science/article/pii/S0303243422000976
def PD_PCO(qpol1, qpol2, iterations=20, lr=0.05, window=3, device='cuda', dtype=torch.complex64):
    'returns the optimal intensity projection for maximum phase consistency across images. (WARNING - NOT CORRECTLY IMPLEMENTED [go to PolarDecompDual for correct version])'
    pauli = torch.stack((Pauli(qpol1, device=device, dtype=dtype), Pauli(qpol2, device=device, dtype=dtype)), dim=-1)
    H, W, B, N = pauli.shape

    # complex projective space coordinate
    w = torch.nn.parameter.Parameter(torch.randn((H, W, B, 1), dtype=pauli.dtype, device=device)/100)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()
        w1 = w / torch.linalg.norm(w, dim=-2, keepdim=True) #constrain to unit manifold
        u = w1.adjoint() @ pauli #H,W,1,B @ H,W,B,N -> H,W,1,N
        u = u / torch.abs(u) #remove magnitude info
        #C = F.avg_pool2d((u.adjoint() @ u).permute(2,3,0,1), window, 1, window//2).permute(2,3,0,1) #H,W,2,2
        C = u.adjoint() @ u
        C = C.masked_fill(torch.isnan(C),0)
        lam = torch.linalg.eigvals(C).real[-1] #largest eigenvalue (func returns in ascending order)
        loss = (-lam / abs(C.diagonal(0,-2,-1)).sum()).sum()
        loss.backward()
        optimizer.step()
    return w.adjoint() @ pauli  #H,W,B - largest eigenvector

# "PolSAR Image Registration Using Orientated Gradients of Polarimetric Features"
# src: https://ieeexplore.ieee.org/document/10758293
# "Fast and Robust Matching for Multimodal Remote Sensing Image Registration
# src: https://ieeexplore.ieee.org/document/8766118 
# "CW-SSIM BASED IMAGE CLASSIFICATION"
# src: Gao, Rehman, Wang (2011) https://ece.uwaterloo.ca/~z70wang/publications/icip11b.pdf
#from CW_SSIM import CW_SSIM
from CSIM import copula_sim
def OGPF(qpol1, qpol2, channels=8, chunk=(256,64), g_kern_std=(1,5,5), device='cuda', tensor=True, p_disp=True, fptype=torch.float16):
    '''
    Adapted OGPF : for each patch, determine the most similar scattering metric.
    '''
    B, H, W = qpol1.shape
    if (type(chunk) is int): chunk = (chunk, chunk)

    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=device,dtype=fptype)[None,None,...]
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],device=device,dtype=fptype)[None,None,...]
    theta = torch.linspace(0,np.pi,channels,device=device,dtype=fptype)[None,:,None,None]
    #Gaussian filter over channel, H, W
    g_std = torch.tensor(g_kern_std, device=device, dtype=fptype)
    k = torch.ceil(6 * g_std).int() #length 6 std, or radius 3 std
    tmp = torch.tensor((channels,chunk[0],chunk[1]), device=device).to(k.dtype)//3; k = torch.minimum(k, tmp)
    k = k - (k%2 == 0).long() #ensures odd kernel size.
    k = torch.stack(torch.meshgrid(*[torch.arange(k[i],device=device)-k[i]//2 for i in [0,1,2]],indexing='ij'),dim=0)
    k = torch.exp(-0.5*(k**2 / g_std[:,None,None,None]**2)) / (((8*torch.pi**3)**0.5)*torch.prod(g_std))
    k = k.sum(dim=0)[None,None,...] / k.sum() #1,1,z,x,y, normalized
    #pad = np.array([(k.shape[2 + i//2])//2 - i%2 for i in reversed(range(6))])
    #pz = pad.copy(); pz[:-2] = 0; pz = tuple(pz)
    #pf = pad.copy(); pf[-2:] = 0; pf = tuple(pf)
    kz, ky, kx = k.shape[-3:]
    pz = (0,0,0,0,kz//2,kz//2)
    pf = (kx//2,kx//2,ky//2,ky//2,0,0)
    
    OGPF = torch.zeros((2,channels,H,W), device=device, dtype=fptype)
    for i in tqdm.trange(H//chunk[0] + 1 if(H%chunk[0] > 0) else H//chunk[0], desc='FMP decomp & selection chunks', disable=not p_disp):
        for j in range(W//chunk[1] + 1 if (W%chunk[1] > 0) else W//chunk[1]):
            i1, i2 = chunk[0]*i, chunk[0]*(i+1)
            j1, j2 = chunk[1]*j, chunk[1]*(j+1)
            x = torch.stack((FreemanPlus(qpol1[:,i1:i2,j1:j2], device=device),
                             FreemanPlus(qpol2[:,i1:i2,j1:j2], device=device)), dim=-1).to(fptype)
            #H,W,C,2 to 2,H,W,C to Hc,Wc to Hc,Wc,band
            scs = torch.stack([copula_sim(x.permute(3,0,1,2)[...,i,None]) for i in range(x.shape[2])], dim=-1)
            scs = torch.argmin(scs, dim=2)[None,None,...].to(fptype) #Hc,Wc,band to 1,1,Hc,Wc
            scs = F.interpolate(scs,size=x.shape[:2],mode='nearest').permute(2,3,0,1).expand(-1,-1,-1,2).int() #1,1,Hc,Wc to H,W,1,2
            x = x.gather(2, scs).permute(3,2,0,1) # H,W,1,2 to 2,1,H,W (B,C,H,W) - so dim0 spans both im1 and im2
            del scs
            
            gx = F.conv2d(x, sobel_x, padding=1, groups=1)
            gy = F.conv2d(x, sobel_y, padding=1, groups=1)
            x = torch.abs(torch.cos(theta) * gx + torch.sin(theta) * gy) #2,C,H,W from 1,C,1,1 x 2,1,H,W
            x = F.pad(F.pad(x.unsqueeze(1), pz, mode='circular'), pf, mode='reflect')
            x = F.conv3d(x, k).squeeze(1) #2,1,C,H,W circular conv 1,1,x,y,z to 2,C,H,W
            OGPF[:,:,i1:i2,j1:j2] = x

    OGPF = OGPF.permute(0,2,3,1) #2,C,H,W to 2,H,W,C
    if (tensor): return OGPF
    return OGPF[0].moveaxis(-1,0).detach().cpu().numpy(), OGPF[1].moveaxis(-1,0).detach.cpu().numpy()
    
    

    