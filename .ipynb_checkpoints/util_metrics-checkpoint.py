import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import math
import matplotlib.transforms as mtransforms


# matplotlib display

class RadialLogTransform(mtransforms.Transform):
    input_dims = output_dims = 2
    is_separable = False

    def __init__(self, scale=1.0, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def transform_non_affine(self, xy):
        r = np.linalg.norm(xy, axis=-1, keepdims=True)
        r_new = np.log1p(self.scale * r) / np.log1p(self.scale) # signed/log-like radial scaling
        return xy * r_new / np.maximum(r, self.eps)

    def inverted(self):
        return InvertedRadialLogTransform(self.scale, self.eps)


class InvertedRadialLogTransform(mtransforms.Transform):
    input_dims = output_dims = 2
    is_separable = False

    def __init__(self, scale=1.0, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def transform_non_affine(self, xy):
        r = np.linalg.norm(xy, axis=-1, keepdims=True)
        r_new = np.expm1(r * np.log1p(self.scale)) / self.scale
        return xy * r_new / np.maximum(r, self.eps)

    def inverted(self): return RadialLogTransform(self.scale, self.eps)




def copula_sim(x, patch_size=64, eps=1e-6):
    """
    COPULA SIMILARITY - converted to pytorch by GPT
    -- --
    CSIM: A Copula-based similarity index sensitive to local changes for Image quality assessment
        Safouane El Ghazouali, Umberto michelucci, Yassin El Hillali, Hichem Nouira (2024)
        [src : https://arxiv.org/abs/2410.01411] & https://github.com/safouaneelg/copulasimilarity/blob/main/CopulaSimilarity/CSM.py
    -- --
    x: (2, H, W, C)
    returns: (H/P, W/P)
    """
    _, H, W, _ = x.shape
    # (2, H, W, C) → (2, C, H, W)
    x = x.permute(0, 3, 1, 2)
    # (2, C, H, W) → (2, C*P*P, B)
    x = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    # (2, C*P*P, B) → (2, B, P*P, C)
    x = x.transpose(1, 2).view(2, x.shape[-1], patch_size * patch_size, x.shape[1] // (patch_size**2))

    # ranks: (2, B, P*P, C) → (2, B, P*P) - converts mag. distribution to indices of mag. order, e.g. flattened relational
    x = (torch.argsort(torch.argsort(x, dim=2), dim=2) + 1).float().mean(dim=3)

    # normalize to (0,1)
    x = (x / (x.shape[-1] + 1.0)).clamp(eps, 1 - eps)

    # Gaussian copula: (2, B, P*P) - uses inverse G. error function [erfinv] to remap normed mag-relational to gaussian coords
    x = torch.nan_to_num( math.sqrt(2.0)*torch.erfinv(2*x - 1) , nan=0.0, posinf=0.0, neginf=0.0)
    
    # distance (B, P*P) → similarity: (B,)
    x = (1.0 - torch.norm(x[0] - x[1], dim=1) / math.sqrt(H)).clamp(0.0, None)
    
    # (B,) → (H/P, W/P)
    return x.view((H - patch_size) // patch_size + 1,
                  (W - patch_size) // patch_size + 1)



# CW_SSIM
'''
Implementation from - https://github.com/dingkeyan93/IQA-optimization
"Comparison of Image Quality Models for Optimization of Image Processing Systems"
src : Ding et al. (2020) https://arxiv.org/abs/2005.01338

Necessity inspired by : "CW-SSIM BASED IMAGE CLASSIFICATION"
 src: Gao, Rehman, Wang (2011) https://ece.uwaterloo.ca/~z70wang/publications/icip11b.pdf
 Algorithm from wikipedia.
'''

class CW_SSIM(nn.Module):
    '''
    This is an pytorch implementation of Complex-Wavelet 
    Structural SIMilarity (CW-SSIM) index. 

    M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey. 
    "Complex Wavelet Structural Similarity: A New Image Similarity Index", 
    IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.
    
    Matlab version:
    https://www.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim
    '''
    def __init__(self, imgSize=[256,256], channels=3, level=4, ori=8, device = torch.device("cuda")):
        assert imgSize[0]==imgSize[1]
        super().__init__()
        self.ori = ori
        self.level = level
        self.channels = channels
        self.win7 = (torch.ones(channels,1,7,7)/(7*7)).to(device)
        s = imgSize[0]/2**(level-1)
        self.w = fspecial_gauss(s-7+1, s/4, 1, device=device)
        self.SP = SteerablePyramid(imgSize=imgSize, K=ori, N=level, hilb=True,device=device)

    def abs(self, x):
        return torch.sqrt(x[:,0,...]**2+x[:,1,...]**2+1e-12)

    def conj(self, x, y):
        a = x[:,0,...]
        b = x[:,1,...]
        c = y[:,0,...]
        d = -y[:,1,...]
        return torch.stack((a*c-b*d,b*c+a*d),dim=1)

    def conv2d_complex(self, x, win, groups = 1): 
        real = F.conv2d(x[:,0,...], win, groups = groups)# - F.conv2d(x[:,1], win, groups = groups)
        imaginary = F.conv2d(x[:,1,...], win, groups = groups)# + F.conv2d(x[:,0], win, groups = groups)
        return torch.stack((real,imaginary),dim=1)

    def cw_ssim(self, x, y):
        cw_x = self.SP(x)
        cw_y = self.SP(y)
        bandind = self.level
        band_cssim = []
        for i in range(self.ori):
            
            band1 = cw_x[bandind][:,:,:,i,:,:]
            band2 = cw_y[bandind][:,:,:,i,:,:]
            corr = self.conj(band1,band2)
            corr_band = self.conv2d_complex(corr, self.win7, groups = self.channels)
            varr = (self.abs(band1))**2+(self.abs(band2))**2
            varr_band = F.conv2d(varr, self.win7, stride=1, padding=0, groups = self.channels)
            cssim_map = (2*self.abs(corr_band) + 1e-12)/(varr_band + 1e-12)
            band_cssim.append((cssim_map*self.w.repeat(cssim_map.shape[0],1,1,1)).sum([2,3]).mean(1)) 

        return torch.stack(band_cssim,dim=1).mean(1)

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score = self.cw_ssim(x, y)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = self.cw_ssim(x, y)
            return score

def fspecial_gauss(size, sigma, channels, device='cuda'):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0).to(device)
    return g.repeat(channels,1,1,1)

class SteerablePyramid(nn.Module):
    # refer to https://github.com/LabForComputationalVision/pyrtools
    # https://github.com/olivierhenaff/steerablePyramid
    def __init__(self, imgSize=[256,256], K=4, N=4, hilb=True, includeHF=True, device=torch.device("cuda")):
        super(SteerablePyramid, self).__init__()
        assert imgSize[0]==imgSize[1]
        size = [ imgSize[0], imgSize[1]//2 + 1 ]
        # self.imgSize = imgSize
        self.hl0 = HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ freq_shift( size[0], True, device  ) ] 
        self.indB = [ freq_shift( size[0], False, device ) ] 


        for n in range( self.N ):

            l = L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            b = B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            s = S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( freq_shift( size[0], True, device ) )
            self.indB.append( freq_shift( size[0], False, device ) )


    def forward(self, x):

        x = torch.fft.rfft2(x, dim=(-2,-1))
        x = torch.stack((x.real, x.imag), dim=1 ).unsqueeze( -3 )
        x = torch.index_select( x, -2, self.indF[0] )

        x   = self.hl0 * x 
        h0f = x.select( -3, 0 ).unsqueeze( -3 )
        l0f = x.select( -3, 1 ).unsqueeze( -3 )
        lf  = l0f 

        output = []

        for n in range( self.N ):

            bf = self.b[n] * lf 
            lf = self.l[n] * central_crop( lf ) 
            if self.hilb:
                hbf = self.s[n] * torch.cat( (bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1 )
                bf  = torch.cat( ( bf , hbf ), -3 )
            if self.includeHF and n == 0:
                bf  = torch.cat( ( h0f,  bf ), -3 )

            output.append( bf )

        output.append( lf  ) 

        for n in range( len( output ) ):
            output[n] = torch.index_select( output[n], -2, self.indB[n] )
            sig_size = (output[n].shape[-2], (output[n].shape[-1]-1)*2)
            #output[n] = torch.stack((output[n].select(1,0), output[n].select(1,1)),-1)
            output[n] = torch.complex(output[n].select(1,0), output[n].select(1,1))
            output[n] = torch.fft.irfft2( output[n], s = sig_size)

        if self.includeHF:
            output.insert( 0, output[0].narrow( -3, 0, 1                    ) )
            output[1]       = output[1].narrow( -3, 1, output[1].size(-3)-1 )

        for n in range( len( output ) ):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 = output[n].narrow( -3,     0, nfeat ).unsqueeze(1)
                    o2 = -output[n].narrow( -3, nfeat, nfeat ).unsqueeze(1)
                    output[n] = torch.cat( (o2, o1), 1 ) 
                else:
                    output[n] = output[n].unsqueeze(1)

        for n in range( len( output ) ):
            if n>0:
                output[n] = output[n]*(2**(n-1))
                
        return output

### TAKEN FROM SteerPyrUtils.py

from torch.autograd import Variable

def L( r ):
    return torch.where(r <= torch.pi / 4, 2,
                       torch.where(r >= torch.pi / 2, 0,
                                2 * torch.cos( torch.pi / 2 * torch.log( 4 * r / torch.pi ) / np.log( 2 ) ) ) )   

def H( r ):
    return torch.where(r <= torch.pi / 4, 0,
                       torch.where(r >= torch.pi / 2, 1,
                                  torch.cos( torch.pi / 2 * torch.log( 2 * r / torch.pi ) / np.log( 2 ) ) ) )

def G( t, k, K ):

    t0 = torch.pi * k / K 
    aK = 2**(K-1) * math.factorial(K-1) / np.sqrt( K * math.factorial( 2 * (K-1) ) )
    
    dt = ((t - t0) + torch.pi/2) % torch.pi - torch.pi/2 
    
    return aK * (torch.cos( dt ))**(K-1)

def S( t, k, K ):

    t0 = torch.pi * k / K 
    dt = torch.abs(t-t0)

    return torch.where(dt < torch.pi/2, 1,
                       torch.where(dt == torch.pi/2, 0, -1) )
    
def L0( r ):
    return L( r/2 ) / 2 
    
def H0( r ):
    return H( r/2 )

def polar_map( s , device='cuda', dtype=torch.float16):

    x = torch.linspace(0, torch.pi, s[1], device=device, dtype=dtype).view( 1, s[1] ).expand( s )
    if s[0] % 2 == 0 :
        y = torch.linspace(-torch.pi, torch.pi, s[0]+1, device=device, dtype=dtype).narrow(0,1,s[0])
    else: y = torch.linspace(-torch.pi, torch.pi, s[0], device=device, dtype=dtype)
    y = y.view( s[0], 1 ).expand( s ).mul( -1 )

    r = ( x**2 + y**2 ).sqrt()
    t = torch.atan2( y, x )

    return r, t 

def S_matrix( K, s , device='cuda'):

    _, t = polar_map( s , device=device)
    sm = torch.Tensor( K, s[0], s[1] ).to(device)

    for k in range( K ):
        sm[k] = S(t, k, K)
    #    for i in range( s[0] ):
    #        for j in range( s[1] ):
    #            sm[k][i][j] = S( t[i][j], k, K )

    return sm 

def G_matrix( K, s , device='cuda'):

    _, t = polar_map( s , device=device) 
    g = torch.Tensor( K, s[0], s[1] ).to(device)

    for k in range( K ):
        g[k] = G(t, k, K)

    return g 

def B_matrix( K, s , device='cuda'):

    g = G_matrix( K, s , device=device)

    r, _ = polar_map( s , device=device)
    h = H(r).unsqueeze(0)

    return h * g 
    
def L_matrix( s , device='cuda'):

    r, _ = polar_map( s , device=device)

    return L(r)

def HL0_matrix( s , device='cuda'):

    r, _ = polar_map( s , device=device)
    h = H0(r.clone()).view( 1, s[0], s[1] )
    l = L0(r.clone()).view( 1, s[0], s[1] )

    return torch.cat( ( h, l ), 0 )

def central_crop( x ):

    ns = [ x.size(-2)//2 , x.size(-1)//2 + 1 ]

    return x.narrow( -2, ns[1]-1, ns[0] ).narrow( -1, 0, ns[1] )

def cropped_size( s ):

    return [ s[0]//2 , s[1]//2 + 1 ]

def L_matrix_cropped( s ):

    l = L_matrix( s ) 

    ns = cropped_size( s ) 

    return l.narrow( 0, ns[1]-1, ns[0] ).narrow( 1, 0, ns[1] ) 

def freq_shift( imgSize, fwd, device ):
    ind = torch.LongTensor( imgSize ).to(device)
    sgn = 1 if fwd else -1
    for i in range( imgSize ):
        ind[i] = (i + sgn*((imgSize-1)//2) ) % imgSize

    return Variable( ind ) 
