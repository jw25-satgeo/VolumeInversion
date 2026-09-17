import sys, os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_util

# -- training wrappers

class AE_SparseWrapper(nn.Module):
    '''
    A means for training an autoencoder on simultaneous denoising and masked recovery.
    >> e.g. an additive & multiplicative noise conditioned autoencoder training pipeline
    '''
    # Masked Autoencoders Are Effective Tokenizers for Diffusion Models @ https://icml.cc/virtual/2025/poster/44606
    # Denoising Masked AutoEncoders Help Robust Classification @ https://arxiv.org/abs/2210.06983
    #
    # note : VAE seems to not handle the Bernoulli (masking) noise well w/o separate no-corruption reconstruction loss.
    #
    def __init__(self, ae_model, r=8, sigma=0.2, sparse=0.5):
        super().__init__()
        self.ae = ae_model
        self.r = r
        self.sigma = sigma
        self.sparse = sparse
        
    def get_loss(self, x, id_lossfunc):
        loss = 0
        for i in range(self.r):
            xgp = self.ae.enc(x)    #gaussian latent noise
            xgp = self.ae.dec(xgp + self.sigma * torch.randn_like(xgp))
            x0 = F.dropout(xgp, p=self.sparse) # bernoulli feature noise
            loss = loss + self.ae.get_loss(x0, x, id_lossfunc)
        return loss

class VAE_SparseWrapper(nn.Module):
    # Trial wrapper to condition VAE on stochastically treating noisy inputs without semantic drift
    def __init__(self, vae_model, r=4, sigma0=1e-4, sigma1=0.3):
        super().__init__()
        self.ae = vae_model
        self.sigma = torch.linspace(sigma0, sigma1, r)
        
    def get_loss(self, x, y, id_lossfunc):
        bank = torch.zeros_like(x)
        bnp  = (self.sigma // self.sigma[-1])
        for i in range(len(self.sigma)):
            x_gp = self.ae.enc(F.dropout(x, p=bnp[i]))
            bank = bank + self.ae.dec(x_gp + self.sigma[i] * torch.randn_like(x_gp))
        loss = id_lossfunc(y, bank / len(self.sigma)) # mean value preservation
        return loss
            

# -- deterministic autoencoders

class AE(nn.Module):

    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        return self.dec(self.enc(x))

class AE_MLP_Tied(nn.Module):
    # A tied-weight autoencoder for the linear dimensionality reduction of sample data @ https://www.nature.com/articles/s41598-024-77080-8
    # + to read : From Variational to Deterministic Autoencoders @ https://arxiv.org/abs/1903.12436
    # + to read : Weight Tying Biases Token Embeddings Towards the Output Space @ https://arxiv.org/html/2603.26663v1
    def __init__(self, in_dim, hidden_dim, emb_dim, sigma0=-1):
        super().__init__()
        a1 = [
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        ]
        self.enc = nn.Sequential(*a1)
        self.dec = nn.Sequential(*[
            nn_util.Linear_T(a1[2]),
            nn.Softplus(),
            nn_util.Linear_T(a1[0]),
        ])
        self.emb_dim = emb_dim
        self.sigma0 = sigma0

    def forward(self, x):
        x1 = self.enc(x)
        if self.sigma0 > 0:
            x1 = x1 + self.sigma0 * torch.randn_like(x1)
        return self.dec(x1)

    def get_loss(self, x, y, id_lossfunc):
        return id_lossfunc(x, y)

# -- VAE utils

class VAE_reparam(nn.Module):
    def forward(self, x):
        mean, logv = x[...,:x.shape[-1]//2], x[...,x.shape[-1]//2:]
        return mean + torch.exp(0.5 * logv) * torch.randn_like(mean)

def VAE_loss(x, y, id_lossfunc, vae_model):
    x1, mean, logv = vae_model.forward(x, get_emb=True)
    KLD = -0.5 * (1 + logv - mean.pow(2) - logv.exp()).sum(dim=-1).mean()
    return id_lossfunc(x1, y) + KLD

# note : Diffusion Autoencoders are Scalable Image Tokenizers
# >> LayerNorm on latent z to give final z removes need for KL loss! (author cites Ba et al. 2016)

# -- VAE models

class VAE_T(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, depth=(5,3), act='silu'):
        super().__init__()
        if emb_dim % 2 != 0 : raise ValueError('latent dimension of VAE must be even')
        
        

class VAE_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, depth=3, act='relu'):
        super().__init__()
        if emb_dim % 2 != 0 : raise ValueError('latent dimension of VAE must be even')
        self.emb_dim = emb_dim
        ramp = torch.linspace(hidden_dim, emb_dim*2, depth).to(torch.int32)
        self.enc = [nn.Linear(in_dim, hidden_dim)] + sum([[nn_util.get_act(act), nn.Linear(ramp[i], ramp[i+1])] for i in range(len(ramp)-1)], [])
        self.enc = nn.Sequential(*(self.enc + [nn.LayerNorm(emb_dim*2), VAE_reparam(), nn.LayerNorm(emb_dim)]))

        ramp = torch.linspace(emb_dim, hidden_dim, depth).to(torch.int32)
        self.dec = sum([[nn.Linear(ramp[i], ramp[i+1]), nn_util.get_act(act)] for i in range(len(ramp)-1)], []) + [nn.Linear(hidden_dim, in_dim)]
        self.dec = nn.Sequential(*([] + self.dec)) # resacaling before reparameterization
        
    def forward(self, x, get_emb=False):
        if get_emb:
            x1 = self.enc_(x); x2 = self.dec(self.rpm(x1))
            return x2, x1[...,:x1.shape[-1]//2], x1[...,x1.shape[-1]//2:]
        return self.dec(self.enc(x))
    def get_loss(self, x, y, id_lossfunc):
        return id_lossfunc(self.forward(x, get_emb=False), y)
        #return VAE_loss(x, y, id_lossfunc, self)

class VAE_MLP_Tied(nn.Module):
    # Bidirectional Variational Autoencoders (Kosko & Adigun 2025) @ https://arxiv.org/html/2505.16074v1
    def __init__(self, in_dim, hidden_dim, emb_dim):
        super().__init__()
        if emb_dim % 2 != 0 : raise ValueError('latent dimension of VAE must be even')
        self.emb_dim = emb_dim
        a1 = [
            nn.Linear(in_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        ]
        self.enc = nn.Sequential(*a1)
        self.dec = nn.Sequential(*[
            VAE_reparam(),  # emb_dim is concatenated mean & variance
            nn_util.Linear_T(a1[2], fslice=slice(0, emb_dim//2)),
            nn.Softplus(),
            nn_util.Linear_T(a1[0])
        ])

    def forward(self, x, get_emb=False):
        x1 = self.enc(x) # concatenated mean & logvar
        if get_emb: #   output            mean                   log-variance
            return self.dec(x1), x1[...,:x1.shape[-1]//2], x1[...,x1.shape[-1]//2:]
        return self.dec(x1)
    def get_loss(self, x, y, id_lossfunc):
        return VAE_loss(x, y, id_lossfunc, self)













