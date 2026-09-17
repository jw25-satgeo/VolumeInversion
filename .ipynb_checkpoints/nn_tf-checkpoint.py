import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as nnpm

import nn_util

# paper queue:
# Inverse distance weighting attention @ https://arxiv.org/abs/2310.18805
# ... possibility of applying EBAM epanechnikov-like euclidean kernels to attention
# Sparse Attention as Compact Kernel Regression
# ... equivalence of sparsemax attention and Epanechnikov kernel, prev. investigated in epanechnikov energy
# Hyper-SET: Designing Transformers via Hyperspherical Energy Minimization
# ...  ??
# MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens (Chen et al. 2026) @ https://arxiv.org/html/2603.23516v1
# ... use of auxiliary layer-wise contrastive loss to boost transformer performance using Hopfield energy perspective
# Residual Connections Harm Self-Supervised Abstract Feature Learning @ https://arxiv.org/html/2404.10947v1#S3
# Echo State Transformer: Attention over Finite Memories @ https://arxiv.org/pdf/2507.02917
# Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth (Dong et al. 2021)

# Gated Attention :  https://github.com/kalyani-25/reimplementation-gated-attention
#    self.P_g = nn.Linear(dim*channels, hdim*channels) # head-shared elementwise gate projection
#    with torch.no_grad(): self.P_g.weight.normal_(mean=0.0, std=0.02) # 
#    X1 = X1 * torch.sigmoid(self.P_g(X0))[:,None,:,:] # apply attention & input-dependent gating

# Critical attention scaling in long-context transformers @ https://openreview.net/forum?id=7SLtElfqCW
# Two failure modes of deep transformers and how to avoid them: a unified theory of signal propagation at initialisation @ https://arxiv.org/html/2505.24333v3

# Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers @ https://arxiv.org/html/2502.00213v2

# Attention-Only Transformers via Unrolled Subspace Denoising (Wang et al. 2025) @ https://arxiv.org/abs/2506.03790v1
# + : Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators? @ https://arxiv.org/abs/2307.14023
# + :  A Simplified Query-Only Attention for Encoder-Based Transformer Models (Yeom & An, 2024) @ https://www.mdpi.com/2076-3417/14/19/8646
#
# desc  -  A transformer stack performs token denoising operations in each optimally projected subspace given to its attention heads.
class r1PA_block(nn.Module): # residual single projection multi-head subspace attention : MHSA block
    
    def __init__(self, h=4, dim=16, channels=1, activation='sigmoid', bias=True):
        super().__init__()
        self.h = h
        self.act = nn_util.get_act(activation)
        self.beta = nn.Parameter(torch.ones(1, requires_grad=True))
        self.P_q = nn.Linear(dim*channels, h*dim*channels, bias=bias)
        
    def forward(self, x): # accepts (B,C,N) where (B,N) should be expanded to (B,1,N).
        B,C,N = x.shape
        X = F.layer_norm(x, (C,N))
        X1 = self.P_q(X.view(B,-1)).reshape(B,self.h,C,N) #B,h,C,N
        X1 = self.act(self.beta * X1 @ X1.transpose(-1,-2)) @ X1
        return x + X1.sum(dim=1)#self.P_o(X1.view(B,-1)).reshape(B,C,N)

# linear bottlenecked attention
class r2PA_block(nn.Module):
    '''
    Train with AdamW! SGD really doesn't seem to like this one.
    '''
    def __init__(self, h=4, dim=16, hdim=4, channels=1, activation='sigmoid', bias=True):
        super().__init__()
        self.h = h
        self.hd = hdim
        self.act = nn_util.get_act(activation)
        self.P_q = nn.Linear(dim*channels, h*hdim*channels, bias=bias) # query diverging projection
        self.P_o = nn.Linear(h*hdim*channels, dim*channels, bias=bias) # merging dense projection
        self.beta = nn.Parameter(torch.ones((1,h,1,1), requires_grad=True))
        
    def forward(self, x):
        B,C,N = x.shape
        X0 = F.layer_norm(x, (C,N))
        X1 = self.P_q(X0.view(B,-1)).reshape(B,self.h,C,self.hd) # project query & split heads
        X1 = self.act(self.beta * X1 @ X1.transpose(-1,-2)) @ X1
        X1 = self.P_o(X1.view(B,-1)).view(B,C,N)
        return x + X1

class r2PA_Transformer(nn.Module):
    def __init__(self, h=4, c=1, dim=16, hdim=4, activation='sigmoid', layers=6):
        super().__init__()
        #self.net = nn.ModuleList([r1PA_block(h,dim,c,activation) for _ in range(layers)])
        self.net = nn.Sequential(*[r2PA_block(h,dim,hdim,c,activation) for _ in range(layers)])
    def forward(self, x):
        return self.net(x)

class r2PA_block_CSDE(r2PA_block):
    # Reverse-time conditional SDE block in residual two-projection form
    def __init__(self, h=4, dim=16, hdim=4, channels=1, activation='sigmoid', bias=True, sigma=0):
        super().__init__(h,dim,hdim,channels,activation,bias)
        self.sigma = sigma
    def forward(self, x, x_c=None, cond_mask=None):
        B,C,N = x.shape
        z = self.P_q(F.layer_norm(x, (C,N)).view(B,-1)).reshape(B,self.h,C,self.hd)
        z = self.act(self.beta * z @ z.transpose(-1,-2)) @ z
        z = self.P_o(z.view(B,-1)).view(B,C,N)
        X2 = (x + z) + self.sigma*torch.randn_like(x)
        if cond_mask is not None:
            if cond_mask.dtype is torch.bool:
                X2[cond_mask] = x[cond_mask]
            elif cond_mask.dtype.is_floating_point:
                X2 = cond_mask*x + (1 - cond_mask)*X2
        return X2

class r2PA_CSDE_Transformer(nn.Module):
    def __init__(self, h=4, c=1, dim=16, hdim=4, activation='sigmoid', sigmas=None):
        super().__init__()
        self.net = nn.ModuleList([r2PA_block_CSDE(h,dim,hdim,c,activation,sigma=sigma) for sigma in sigmas])
    def forward(self, x, cond_mask=None):
        x0 = x.clone()
        for i in range(len(self.net)):
            x = self.net[i](x, x0, cond_mask)
        return x

class r2PA_Transformer_Recurrent(nn.Module):
    def __init__(self, h=4, c=1, dim=16, hdim=4, activation='sigmoid', layers=6, iters=10):
        super().__init__()        # ref layer multiple times
        self.net = nn.Sequential(*sum([[r2PA_block(h,dim,hdim,c,activation)] * iters for _ in range(layers)], []))
    def forward(self, x):
        return self.net(x)


class VAE_reparam(nn.Module):
    def forward(self, x):
        mean, logv = x[...,:x.shape[-1]//2], x[...,x.shape[-1]//2:]
        return mean + torch.exp(0.5 * logv) * torch.randn_like(mean)
def VAE_loss(x, y, id_lossfunc, vae_model):
    x1, mean, logv = vae_model.forward(x, get_emb=True)
    KLD = -0.5 * (1 + logv - mean.pow(2) - logv.exp()).sum(dim=-1).mean()
    return id_lossfunc(x1, y) + KLD

class r2PA_SDE_VAE(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, depth=(10,5), act='silu'):
        super().__init__()
        if emb_dim % 2 != 0 : raise ValueError('latent dimension of VAE must be even')
        self.emb_dim = emb_dim
        enc, dec = [], []
        r1, r2 = torch.linspace(in_dim, emb_dim*2, depth[0]).to(torch.int32), torch.linspace(emb_dim, out_dim, depth[1])
        #for i in range(depth[0]):
            #enc += [nn.Linear(r1[i], r1[i+1]), nn_util.get_act(act), r2PA_block_(h=)]
            #enc += [
        enc += [nn.LayerNorm(emb_dim*2), VAE_reparam(), nn.LayerNorm(emb_dim)]
        for i in range(depth[1]):
            dec ++ [nn.Linear(r2[i], r2[i+1]), nn_util.get_act(act)]
        self.enc = nn.Sequential(*(enc + [nn.LayerNorm(emb_dim*2), VAE_reparam(), nn.LayerNorm(emb_dim)]))
        self.dec = nn.Sequential(*([] + self.dec))
    def forward(self, x, get_emb=False):
        if get_emb:
            x1 = self.enc_(x); x2 = self.dec(self.rpm(x1))
            return x2, x1[...,:x1.shape[-1]//2], x1[...,x1.shape[-1]//2:]
        return self.dec(self.enc(x))
    def get_loss(self, x, y, id_lossfunc):
        return id_lossfunc(self.forward(x, get_emb=False), y)


# Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy @ https://arxiv.org/html/2603.06875v1

class S1PA_block(nn.Module):
    
    def __init__(self, h=4, dim=16, c=1, activation='sigmoid', bias=True):
        super().__init__()
        self.h = h
        self.act = nn_util.get_act(activation)
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.ones(1, requires_grad=True))
        nnpm.register_parametrization(self, "alpha", nn.Softplus())
        nnpm.register_parametrization(self, "beta", nn.Softplus())
        self.P_q = nn.Linear(dim*c, h*dim*c, bias=bias)

    def forward(self, x):
        B,C,N = x.shape
        X1 = F.layer_norm(x, (C,N))
        X1 = self.P_q(X1.view(B,-1)).reshape(B,self.h,C,N)
        noise = (2*self.alpha/self.beta)**0.5 * torch.randn_like(X1)
        X1 = self.act(self.beta * (X1) @ X1.transpose(-1,-2)) @ X1 + noise
        return x + X1.sum(dim=1)

class S1PA_narrow_block(nn.Module):

    def __init__(self, h=4, dim=16, hdim=4, c=1, activation='sigmoid', bias=True):
        super().__init__()
        self.h = h
        self.hd = hdim
        self.act = nn_util.get_act(activation)
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.ones(1, requires_grad=True))
        nnpm.register_parametrization(self, "alpha", nn.Softplus())
        nnpm.register_parametrization(self, "beta", nn.Softplus())

        self.P_q = nn.Linear(dim*c, h*hdim*c, bias=bias)
        self.P_q1 = nn.Linear(hdim*c, hdim*c, bias=bias)
        self.P_o = nn.Linear(h*hdim*c, dim*c, bias=bias)

    def forward(self, x):
        B,C,N = x.shape
        X1 = F.layer_norm(x, (C,N))
        X1 = self.P_q(X1.view(B,-1)).reshape(B,self.h,C,self.hd)
        X1 = self.P_q1(X1.view(B*self.h,-1)).reshape(B,self.h,C,self.hd)
        noise = (2*self.alpha/self.beta)**0.5 * torch.randn_like(X1)
        
        X1 = self.act(self.beta * (X1) @ X1.transpose(-1,-2)) @ X1 + noise
        
        # -- self-attention direct translation of paper
        #X1 = self.alpha * self.act(self.beta * X1 @ X1.transpose(-1,-2)) @ X1 + noise
        
        X1 = self.P_o(X1.view(B,-1)).view(B,C,N)
        
        return self.alpha * x + X1


# pure MLP for comparative testing
class MLP_narrow_block(nn.Module):

    def __init__(self, h=4, dim=16, hdim=4, channels=1, activation='sigmoid'):
        super().__init__()
        self.h = h
        self.hd = hdim
        self.act = nn_util.get_act(activation)
        self.P_q = nn.Linear(dim*channels, h*hdim*channels, bias=True)
        self.P_o = nn.Linear(h*hdim*channels, dim*channels, bias=True)
        
    def forward(self, x):
        B,C,N = x.shape
        x = F.layer_norm(x, (C,N))
        X1 = self.P_q(x.view(B,-1)).reshape(B,self.h,C,self.hd)
        X1 = self.P_o(X1.view(B,-1)).view(B,C,N)
        return x + X1








# ------   DUMP   ------ #

# StochasticSlots trial
'''
NONFUNCTIONAL (only makes whiteout)
class r1PA_randomsparse_block(nn.Module):

    def __init__(self, h=4, dim=16, hdim=4, channels=1, activation='sigmoid', bias=True):
        super().__init__()
        self.h = h
        self.hd = hdim
        self.act = nn_util.get_act(activation)
        self.stsl = nn_util.StochasticSlots(dim, hdim, permute=True)
        self.P_q = nn.Linear(hdim*channels, h*hdim*channels, bias=bias)
        
    def forward(self, x):
        B,C,N = x.shape
        T = 1 / self.hd**0.5  #  Mind the Gap .. (Saada et al. 2025)
        x = F.layer_norm(x, (C,N))
        X1, ii = self.stsl(x)
        X1 = self.P_q(X1.view(B,-1)).reshape(B,self.h,C,self.hd)
        X1 = (self.act( (X1 @ X1.transpose(-1,-2)) / (self.hd**0.5) ) / T - T) @ (T * X1)
        X1 = self.stsl.restore(X1.sum(dim=1), ii, (B,C,N))
        return x + X1
'''

