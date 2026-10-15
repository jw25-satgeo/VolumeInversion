import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_util

def WienerNoiser(x, t=1.0, L=1.0, n=16):
    '''
    Admit a 1D vector (B,C,N) and add scaled noise of a Wiener process of Gaussian window-correlated perturbations with correlation length L.
    '''
    (B, C, N), K = x.shape, torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
    K = torch.exp(-0.5 * (K[:,None] - K[None,:]).pow(2) / L**2) # gaussian spatial kernel
    K /= K.square().sum(dim=-1, keepdim=True).sqrt() # spatial kernel L2 normalization (preserve variance)
    z = torch.randn(B,C,n,N, device=x.device, dtype=x.dtype) # random perturbations
    return x + (t / n)**0.5 * (z @ K.T).sum(dim=-2)

def RandWalk(x, sigma=0.0, L=0, steps=1):
    '''
    Admit a 1D vector (B,C,N) and apply a random perturbation of Gaussian-kernel correlation length L.
    '''
    (B, C, N), y = x.shape, x.clone()
    inds = torch.arange(N)[None,None,:]
    for i in range(steps):
        z = sigma * torch.randn_like(x[...,None,0]) # amplitude
        z_p = torch.rand_like(x[...,None,0]) * (N - 1) # position
        if L == 0     : L     = torch.randn_like(z)    # correlation length
        z = z * torch.exp(-0.5 * (inds - z_p)**2 / L**2)
        y += z
    return y


def Tilt(x, rad:tuple):
    '''
    Admit an N-D lattice and warp its contents using displacement and rotation.
    '''
    return

# - - - - - -
#   A Kernel Theory of Modern Data Augmentation @ https://pmc.ncbi.nlm.nih.gov/articles/PMC6879382/
#  ..   A procedural data augmentation method which enables evaluation of
#  ..   a Markov augmentation process model using a kernel evaluation.
# - - - - - -

class Augment_MarkovChain(nn.Module):
    
    def __init__(self, data, *transforms):
        super().__init__()
        self.transforms = list(transforms)
        self.register_buffer('obs', data)
        # logits for probability of transforms & retractions
        self.wTR = nn.Parameter(torch.ones(len(self.transforms) + self.obs.shape[0]))

    def forward(self, n=10):
        '''
        evolves a sample set through a series of probabilistic augmentation transformations.
        '''
        dTR = torch.distributions.categorical.Categorical(logits=self.wTR)
        x, cut = self.obs.clone(), len(self.transforms)
        log_probs = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device) #per-sample outcome
        for i in range(n):
            iTR = dTR.sample(x.shape[0]); lpTR = dTR.log_prob(iTR)
            for j in range(cut): x[j == iTR] = self.transforms[j](x[j == iTR]) # transformation
            for j in range(x.shape[0]): x[j == (iTR - cut)] = self.obs[j]      # retraction
            log_probs += lpTR
        return x, log_probs

    def kernel_metric_loss(self, n=10, targ_var=1.0):
        x1, log_probs = self.forward(n=n)
        # .. REWARD CALCULATION : MMD loss (distribution matching) & sample divergence
        reward = nn_util.MMD_Loss(x1, self.obs) - F.mse_loss(x1, x)
        loss = -log_probs * reward
        return loss

















    