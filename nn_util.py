import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as nnpm

from nn_special_entmax import Entmax15

def get_act(name='sigmoid', dim=-1):
     match name:
         case 'sigmoid': return nn.Sigmoid()
         case 'softmax': return nn.Softmax(dim=dim)
         case 'entmax':  return Entmax15(dim=dim)
         case 'relu': return nn.ReLU()
         case 'silu': return nn.SiLU() # also known as swish @ beta=1
         #case 'swiglu': return ...
         case _: raise ValueError('invalid activation')

#class SinkhornDist(nn.Module):
#    def forward(self, x, y):
#        return x

class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(-1,-2)
        
class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm()

class Clamp(nn.Module):
    def __init__(self, minc=None, maxc=None):
        super().__init__()
        self.minc = minc
        self.maxc = maxc
    def forward(self, x):
        return torch.clamp(x, self.minc, self.maxc)

# Online Normalization for Training Neural Networks (Chiley et al. 2019) @ https://arxiv.org/abs/1905.05894v3
# use of Welford's algorithm .. Streaming Deep Reinforcement Learning Finally Works (Elsayed et al. 2026) @ https://arxiv.org/html/2410.14606v3
# Note on a method for calculating corrected sums of squares and products. (Welford, 1962) @ https://doi.org/10.1080/00401706.1962.10490022
# Updating Formulae and a Pairwise Algorithm for Computing Sample Variances (Chan et al. 1979) @ https://doi.org/10.1007/978-3-642-51461-6_3
class RunningStat(nn.Module):
    '''An online statistics descriptor adapting Chan's variant of Welford's algorithm. Accepts (B, d...) format.'''
    def __init__(self, n_vars):
        super().__init__()
        self.register_buffer('n', torch.zeros(1))
        self.register_buffer('tse', torch.zeros(n_vars))
        self.register_buffer('mean', torch.zeros(n_vars))

    def var(self): return self.tse / self.n
    def bessel_var(self): return self.tse / (self.n - 1)

    def update(self, x):
        x1 = x.flatten(1) # (B, d...) tensor
        bm, bv, bn = x1.mean(0), x1.var(0), x1.shape[0]
        delta , _n = bm - self.mean, self.n
        self.mean += delta * bn / (_n+bn)
        self.tse  += (bv * (bn - 1)) + delta**2 * _n*bn/(_n+bn)
        self.n    += bn
    
    def single_update(self, x):
        assert len(x.shape) == 0
        mean0 = self.mean
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.tse  += (x - mean0) * (x - self.mean)
# ..........

class Linear_T(nn.Module):
    def __init__(self, linearLayer, fslice=None):
        super().__init__()
        self.fslice = fslice
        self.bias = None
        self.linear = linearLayer
        if linearLayer.bias is not None:
            bias_init_stdv = linearLayer.weight.shape[0]**(-0.5)
            self.bias = nn.Parameter(torch.zeros(linearLayer.weight.shape[1], requires_grad=True))
            self.bias.data.uniform_(-bias_init_stdv, bias_init_stdv)
    def forward(self, x):
        if self.fslice is None:
              w = self.linear.weight.T
        else: w = self.linear.weight.T[..., self.fslice]
        return F.linear(x, w, bias=self.bias)

class SymSkewLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))
        self.W = nn.Parameter(torch.randn((dim,dim), requires_grad=True))
    def forward(self, x):
        sym = (self.W + self.W.transpose(-1,-2))/2
        skw = (self.W - self.W.transpose(-1,-2))/2
        return F.linear(x,sym) + self.alpha*F.linear(x,skw)

class SymSkewLinear_MH(nn.Module):
    def __init__(self, dim, h=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand((1,h,1,1), requires_grad=True))
        self.W = nn.Parameter(torch.randn((1,h,dim,dim), requires_grad=True))
    def forward(self, x):
        sym = (self.W + self.W.transpose(-1,-2))/2
        skw = (self.W - self.W.transpose(-1,-2))/2
        return x[...,None,:] @ sym + self.alpha*(x[...,None,:] @ skw)

# Mutual Regression Distance @ https://arxiv.org/html/2501.10617v1
class MRD_Dist(nn.Module):
    def forward(self, x, y):
        return x # INCOMPLETE

# Maximum Mean Discrepancy (MMD) @ https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bw_mult = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth
    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth
    def forward(self, X):
        L2_distances, bw_mult = torch.cdist(X, X) ** 2, self.bw_mult.to(X.device, X.dtype)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * bw_mult)[:, None, None]).sum(dim=0)

class MMD_Loss(nn.Module):
    def __init__(self, kernel=None):
        super().__init__()
        if kernel is None:
            kernel = RBF()
        self.kernel = kernel
    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

# Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention @ https://arxiv.org/html/2604.00754v2
# +:  The Information Pathways Hypothesis: Transformers are Dynamic Self-Ensembles @ https://arxiv.org/pdf/2306.01705
# +:  Always-Sparse Training by Growing Connections with Guided Stochastic Exploration @ https://arxiv.org/html/2401.06898v2
class StScPick(torch.autograd.Function):
    '''
    NOT CURRENTLY WELL-STUDIED : DO NOT USE FOR PRODUCTION
    : This appears to do little more than ruin performance at the moment.
    
    '''
    @staticmethod
    def forward(ctx, x, w, sigma, k:int, permute:bool=False):
        p = w[None,None,:] + (1/sigma)*torch.randn_like(x)
        ii = torch.topk(p, k, dim=-1, sorted=permute)[1]
        ctx.x_shape = x.shape
        ctx.save_for_backward(w, sigma, ii)
        return x.gather(-1,ii), ii
    @staticmethod
    def backward(ctx, grad, grad_ii):
        w, sigma, ii = ctx.saved_tensors
        sdiff = (w[ii][...,None] - w[None,None,None,])**2
        gkern_prod = grad[...,None] * torch.exp(-sigma*sdiff)
        
        gradw = gkern_prod.sum(dim=(0,1,2))
        grads = (gkern_prod * -sdiff).sum().reshape_as(sigma)
        gradx = torch.zeros(ctx.x_shape).to(grad.device, grad.dtype).scatter_add_(-1, ii, grad)
        return (gradx, gradw, grads, None, None)
class StochasticSlots(nn.Module):
    def __init__(self, dim=16, hdim=4, permute=False):
        super().__init__()
        self.hdim = hdim
        self.permute = permute
        self.w = nn.Parameter(torch.rand(dim, requires_grad=True))
        self.sigma = nn.Parameter(torch.ones(1, requires_grad=True))
    def forward(self, x):
        return StScPick.apply(x, torch.softmax(self.w,-1), F.softplus(self.sigma), self.hdim, self.permute)
    def restore(self, x, ii, shape):
        return torch.zeros(shape, device=x.device, dtype=x.dtype).scatter_add(-1, ii, x)

# for funsies
def visualize_scoremodel_density(model, feature_length, num_samples=200):
    import matplotlib.pyplot as plt
    x = torch.randn(num_samples, 1, feature_length).to('cuda')
    runs = [x[:,0].detach().cpu()]
    for j in range(1):
        for i in range(len(model.net)):
            x = (model.net[i](x))
            runs.append(x[:,0].detach().cpu())
    x = torch.stack(runs, dim=0)[-20:]
    
    #  visualize in PCA projected plane of time-weighted maximum score variance
    scores = x[1:] - x[:-1]
    score_covs = scores.transpose(-1,-2) @ scores
    _, q = torch.linalg.eigh((score_covs * torch.linspace(0,1,scores.shape[0])[:,None,None]**2).mean(dim=0))
    loc_ = x[1:] @ q[:,:2]
    sco_ = scores @ q[:,:2]
    
    grid = torch.stack(torch.meshgrid(torch.linspace(-2,2,200), torch.linspace(-2,2,200), indexing='ij'), dim=-1)  # (H,W,2)
    d2 = ((grid[:,:,None,None,:] - loc_.view(1,1,x.shape[0]-1,x.shape[1],2))**2).sum(-1)  # (H,W,t,N)
    density = torch.exp(-d2 / (2*(0.05)**2)).sum(-1)
    density = (density[...,-1] - density[...,0])
    plt.imshow(density)



