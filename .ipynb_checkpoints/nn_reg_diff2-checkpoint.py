import sys, os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# STUFF TO READ ..
# GUD: Generation with Unified Diffusion (Gerdes & Welling, 2024) @ https://arxiv.org/html/2410.02667v1
# 

def _bnoise(x, p): return torch.bernoulli(torch.full_like(x, p))
def _aniso_gnoise(x, p): return x.var(0, keepdim=True) * torch.randn_like(x) * p

class ScoreMatching_fBNrGN_3L(nn.Module):
    ''' train ResNet model blocks on a monotonic Bernoulli corruption schedule with adaptive Gaussian contrastive learning.'''
    def __init__(self, resnet, lossf, msched='cosine', mfrac=0.95):
        super().__init__()
        self.resnet = resnet
        self.lossf = lossf
        self.sched = torch.linspace(0, mfrac, len(resnet.net))
        match msched: # stochastic masking schedule (0-1)
            case 'linear': pass     #   cosine schedule
            case 'cosine': self.sched = torch.cos(torch.pi/2 * self.sched)

    def train(self, x, do_backprop=True):
        n, loss, resid = len(self.resnet.net), torch.zeros(1, device=x.device, dtype=x.dtype), None
        bnoise = _bnoise(x[None,].expand(n, *(x.shape)), self.sched)
        for i in range(n):
            x1 = x * (1 - mask*bnoise[i  ])
            


def EqualDensity_Noising_Sigmas(N, sigma0=1e-3, sigma1=10, p_mean=-1.2, p_std=1.2, ascending=False):
    norm = torch.distributions.normal.Normal(0, 1)
    cdf0 = norm.cdf((math.log(sigma0) - p_mean) / torch.Tensor([p_std]))
    cdf1 = norm.cdf((math.log(sigma1) - p_mean) / torch.Tensor([p_std]))
    p = cdf0 + (cdf1 - cdf0) * torch.linspace(0, 1, N)
    sigmas = torch.exp(p_mean + p_std * norm.icdf(p)).flip(dims=[0])
    if (ascending): sigmas = sigmas.flip(dims=[0])
    return sigmas
class ScoreMatching_fBNfGN_3L(nn.Module):
    ''' train ResNet model blocks on a monotonic Gaussian/Bernoulli corruption schedule.'''
    def __init__(self, resnet, lossfunc, sigmas):
        super().__init__()
        self.resnet = resnet
        self.lossfunc = lossfunc
        self.sigma = sigmas
        
    def train(self, x, mask, do_backprop=True):
        # noise init
        n, sigmas = len(self.resnet.net), self.sigma[:, None, None, None]
        gnoise = _aniso_gnoise(x[None,].expand(n, *(x.shape)), sigmas )
        bnoise =       _bnoise(x[None,].expand(n, *(x.shape)), sigmas / sigmas.amax(dim=0) )
        loss   = torch.zeros(1, device=x.device, dtype=x.dtype)
        for i in range(n):
            x1 = ( x + mask*bnoise[i  ]*gnoise[i  ] ) * (1 - mask*bnoise[i  ])
            x2 = ( x + mask*bnoise[i+1]*gnoise[i+1] ) * (1 - mask*bnoise[i+1])
            x2i, lossi = self.resnet.net[i](x1, ~mask), 0
            lossi = lossi + self.lossfunc(x2i, x2) / (2 * self.sigma[i]**2) # score matching loss
            lossi = lossi + 0.5 * self.lossfunc(self.resnet.net[i](x, ~mask), x ) / (2 * self.sigma[i]**2) # identity loss
            lossi = lossi + 0.5 * self.lossfunc(x2i, self.resnet.net[i](x2i, ~mask)) / self.sigma[i] # idempotence loss
            if do_backprop: lossi.backward()
            loss += lossi
        return loss / len(self.resnet.net)