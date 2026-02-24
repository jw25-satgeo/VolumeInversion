import torch
import torch.nn.functional as F
import PolarDecomp
from thin_plate_spline import ThinPlateSpline

def KeypointFlow_2024Cheng(qpol1, qpol2, avg_window=3, p_size=(48,48), radius=(8,2), alpha=5, iterations=20, coarseness=4, k=0.5, device=device, dtype=torch.complex64):
    '''
    PolSAR Image Registration Using Orientated Gradients of Polarimetric Features
    (Cheng et al. 2024 - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10758293)
    Given a small OGPF descriptor cube extracted at a keypoint in the reference image, find the most similar OGPF descriptor cube in a small search window of the sensed image.
    '''
    # ensure window fold/unfold doesn't alter image dimension
    assert (qpol1.shape[0] % p_size[0] == 0) and (qpol1.shape[1] % p_size[1] == 0)
    assert (p_size[0] % 2 == 0) and (p_size[1] % 2 == 0)
    fptype = torch.empty(1,dtype=dtype).real.dtype

    # STAGE 1 : keypoint detection (only on reference image)
    with torch.no_grad():
        qpr = torch.as_tensor(qpol1, device=device, dtype=dtype).moveaxis(0,-1) # B,H,W to H,W,B
        H,W,B = qpr.shape
        C3 = (qpr[...,:,None] * qpr[...,None,:].conj())# H,W,B,B
        C3 = C3.permute(2,3,0,1)
        C3 = torch.complex(
            F.avg_pool2d(C3.real, avg_window, 1, avg_window//2),
            F.avg_pool2d(C3.imag, avg_window, 1, avg_window//2),
        ).permute(2,3,0,1)
    
        intensity = torch.abs(-qpr[...,None].adjoint() @ torch.linalg.solve_ex(C3, qpr[...,None])[0]); del(C3)
        pwf_inv = (1 / intensity).squeeze(-1).squeeze(-1); del(intensity) # polarimetrically whitened reciprocal (see p.6 of paper)
        pwf_inv = pwf_inv.unfold(0,p_size[0]//2,p_size[0]//2).unfold(1,p_size[1]//2,p_size[1]//2) #Hc,Wc,h,w
        
        _, key = torch.max(pwf_inv.reshape(pwf_inv.shape[0], pwf_inv.shape[1], -1), dim=-1, keepdim=False) # Hc,Wc
        key = torch.stack(
             ((key // pwf_inv.shape[-1]) + (torch.arange(key.shape[0], dtype=torch.long, device=device)[:,None] - 1) * (p_size[0]//2),
              (key %  pwf_inv.shape[-1]) + (torch.arange(key.shape[1], dtype=torch.long, device=device)[None,:] - 1) * (p_size[1]//2)), dim=0
        ).view(2,-1); key = key[:, (key >= search_pad).all(dim=0)] - search_pad # 2,N : eliminate all keys without complete search windows
    
    # STAGE 2 : template matching
    ogpf = PolarDecomp.OGPF(qpr.moveaxis(-1,0), qpol2, device=device, p_disp=False, fptype=fptype).detach() # 2,H,W,C
    
    with torch.no_grad():
        ogpf = F.pad(ogpf, (0,0,search_pad+p_size[0]//2, search_pad+p_size[0]//2, search_pad+p_size[1]//2, search_pad+p_size[1]//2))
        ogpf = ogpf.unfold(1,p_size[0]+search_pad*2,1).unfold(2,p_size[1]+search_pad*2,1)[:,:-1,:-1] # 2,H,W,C,h,w
        
        cosim = torch.empty(key.shape[1], search_pad*2+1, search_pad*2+1, device=device, dtype=fptype)
        for i in range(key.shape[1]):
            temp = ogpf[0,key[0,i],key[1,i],:,search_pad:-search_pad,search_pad:-search_pad][None,None,:,:,:].flatten(-3) #1,1,C,h,w -> 1,1,D
            temp = temp / torch.linalg.norm(temp,dim=-1)[...,None]
            targ = ogpf.moveaxis(3,-1)[1,key[0,i],key[1,i]].unfold(0,p_size[0],1).unfold(1,p_size[1],1).flatten(-3) #ih,iw,C,h,w -> ih,iw,D
            targ = targ / torch.linalg.norm(targ,dim=-1)[...,None]
            cosim[i] = torch.nansum(temp*targ, dim=-1) #ih,wh 
        del(ogpf)
        # confidence masking
        tmp = cosim.sum(dim=(1,2)); tmp /= tmp.max()
        mask = tmp > 0.5
        cosim, key = cosim[mask], key[:,mask]
        '''
        A Subpixel Registration Algorithm for Low PSNR Images
        (Feng et al. 2018 - https://arxiv.org/abs/1804.00174)
        '''
        #wgt = torch.exp(alpha * (cosim - cosim.max()))[...,None] # N,ih,iw,1
        #dij = torch.arange(-search_pad, search_pad+1, dtype=fptype, device=device)
        #dij = torch.stack(torch.meshgrid(dij, dij, indexing='ij'),dim=-1)[None,...] # 1,ih,iw,2
        #offset = (wgt * dij).nansum(dim=(1,2)) / wgt.sum(dim=(1,2)) #N,2
        #del(wgt, dij, cosim)
        
        '''
        Thin plate spline - keypoint offset to global mapping
        '''
        tps = ThinPlateSpline(alpha=0.0, order=2, enforce_tps_kernel=False, device=device)
        tps.fit((key.moveaxis(0,1) + offset)[:,[1,0]], key.moveaxis(0,1)[:,[1,0]])
        
        coreg_map = torch.stack(torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij'),dim=-1).view(H*W,2).to(fptype).to('cpu') #N,2
        coreg_map = tps.transform(coreg_map).detach().view(H,W,2)[None,...].to(device)
        coreg_map[...,0] = coreg_map[...,0] / (0.5*coreg_map.shape[1]) - 1
        coreg_map[...,1] = coreg_map[...,1] / (0.5*coreg_map.shape[2]) - 1
        coreg_map = coreg_map[...,[1,0]]
    
        qp2_coreg = torch.complex(F.grid_sample(torch.as_tensor(qpol2.real, device=device, dtype=fptype)[None,...], coreg_map, mode='bicubic'),
                                  F.grid_sample(torch.as_tensor(qpol2.imag, device=device, dtype=fptype)[None,...], coreg_map, mode='bicubic'))[0]

    return qpr.moveaxis(-1,0), qp2_coreg


    
    