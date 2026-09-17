import random
import numpy as np
import numbers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# (func distance:) Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models
# ... src @ (Millidge et al. 2022) https://arxiv.org/abs/2202.04557
def dist_dp(x, y): return - x @ y.transpose(-1,-2)                     # dotprod
def dist_ed(x, y): return ((x[...,None,:]-y[...,None,:,:])**2).sum(-1)  # euclidian
def dist_mh(x, y): return (x[...,None,:]-y[...,None,:,:]).abs().sum(-1) # manhattan
def get_comb_dist(*P):
    return lambda x, y: (torch.stack([f(x, y) for f in P], dim=-1)).sum(dim=-1)

# Should EBMs model the energy or the score?  @ https://openreview.net/forum?id=9AS-TF2jRNb
# .. i.e. energy based models, can perform just as well as unconstrained (score-based) models when using a comparable model structure.
# Learning Energy-Based Models by Diffusion Recovery Likelihood @ https://github.com/ruiqigao/recovery_likelihood
# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation @ https://arxiv.org/abs/2506.14202
# .. also read @ https://intuitivepapers.ai/diffusionblocks/#results

# // ------- ENERGY BASED ASSOCIATIVE MEMORY ------- //
#
#  A study of storage and retrieval dynamics in the 1-layer kernel-support neural network.
#    
# Dense Associative Memory with Epanechnikov Energy @ https://arxiv.org/abs/2506.10801
def lsr_energy(x, w, P, beta=1):
    return -(1/beta) * torch.log(1e-8 + F.relu(1 - 0.5*beta*P(x,w)).sum(dim=-1, keepdim=True))
def lss_energy(x, w, P, beta=1): # softplus replacement for stability
    return -(1/beta) * torch.log(1e-8 + F.softplus(1 - 0.5*beta*P(x,w)).sum(dim=-1, keepdim=True))
# Hopfield Networks is All You Need @ https://arxiv.org/abs/2008.02217
def lse_energy(x, w, P, beta=1):
    a = 0.5*(x**2).sum(dim=-1, keepdim=True)
    b = -(1/beta)*torch.logsumexp(-0.5*beta*P(x,w),dim=-1,keepdim=True) + math.log(w.shape[-2])
    return a + b

# Theory, Analysis, and Best Practices for Sigmoid Self-Attention @ https://arxiv.org/abs/2409.04431
# Sigmoid Gating is More Sample Efficient than Softmax Gating in Mixture of Experts @ https://arxiv.org/html/2405.13997v3
# ... possible relation to softplus adaptation of Epanechnikov energy of Hoover et al. 2025
def update_lss(x, w, P, beta=1):
    z = 1 - 0.5 * beta * P(x, w)
    return (torch.sigmoid(z)/F.softplus(z).sum(dim=-1, keepdim=True)) @ w
def update_lss_unnormalized(x, w, P, beta=1):
    return torch.sigmoid(1 - 0.5*beta*P(x,w)) @ w
# .. softmax attention equivalent hopfield update
def update_lse(x, w, P, beta=1):
    return torch.softmax(-0.5*beta*P(x,w),dim=-1) @ w

class EBAM(nn.Module):

    def __init__(self, inp_size=16, mem_size=2, lr=0.05, iters=20, dist=dist_mh, **pparams):
        super().__init__()
        self.default_iters = iters
        self.register_buffer('w', torch.randn(mem_size, inp_size))
        self.w_optim = torch.optim.SGD([torch.zeros(1)], lr=lr, momentum=0.8, dampening=0.1, weight_decay=0.2, fused=True)
        self.rebuf = (torch.ones(1), torch.zeros(1, inp_size))
        self.post_init(lr=lr, iters=iters, **pparams)
        self.P = dist
    def post_init(self, **params):
        pass
    # terminate recursive access to internal networks during global optimization
    #def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
    #    return super().named_parameters(prefix, False, remove_duplicate)

    # generative recall for continuoal learning
    def gen_rebuf(self, n, beta0, iters):
        x = torch.randn((n, self.w.shape[-1]), device=self.w.device, dtype=self.w.dtype)
        x = self.forward(x, beta0=beta0, iters=iters).detach()
        weight = torch.softmax(self.energy(x, self.w, beta0).flatten(), dim=0).detach()
        self.rebuf = weight, x
    def rebuf_mix(self, x, frac=0.5): # .. random weight retrieve + append
        rn = max(int(x.shape[0] * frac), 1)
        ri = torch.multinomial(self.rebuf[0], num_samples=rn, replacement=True)
        return torch.cat((x, self.rebuf[1][ri]), dim=0)

    def learn(self, x, beta0=1, iters=None, c=8, loss_hist=None, rebuf=-1):
        if (rebuf >= 0):
            self.learn_(self.rebuf_mix(x, rebuf), beta0, iters, c, loss_hist)
            self.gen_rebuf(self.w.shape[0]*2, beta0, iters=100)
        else:
            self.learn_(x, beta0, iters, c, loss_hist)

    def energy(self, x, w, beta=1):
        return lse_energy(x, w, self.P, beta)
    def forward(self, x, beta0=1, iters=None, loss_hist=None, wb=None):
        if (iters is None): iters = self.default_iters
        if (wb is None): wb = self.w
        for i in range(iters):
            x = update_lse(x, wb, self.P, beta0)
            if loss_hist is not None: loss_hist.append(lse_energy(x, wb, self.P, beta0).sum().detach().item())
        return x

    # ENERGY MINIMIZATION FOR TRAINING DENSE ASSOCIATIVE MEMORY @ https://openreview.net/pdf/5dfb5eb06ae1fbf962e6657abdd43155e817f78f.pdf
    def learn_(self, x, beta0=1, iters=None, c=8, loss_hist=None):
        if (iters is None): iters = self.default_iters
        x1 = x[None,:,:].repeat(c,1,1) # minibatch forward
        res = torch.zeros_like(self.w).requires_grad_()
        self.w_optim.add_param_group({'params':[res]})
        ns = torch.linspace(0,1,c)[:,None,None].to(x.device,x.dtype) # noise schedule
        for i in range(iters):
            t_ = float(i+1)/iters
            wb = self.w + res
            self.w_optim.zero_grad()
            noise = ((1-t_)*ns)**2 * torch.randn_like(x1) # quadratic cooling schedule
            loss = self.energy(x1 + noise, wb, beta0).mean(dim=0).sum()
            loss.backward()
            self.w_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.w_optim.param_groups.clear()
        self.w_optim.state.clear()
        self.w += res

# tried a thing.
class dynEBAM(EBAM):
    '''
    Q : Why does this work?
    We first assume a langevin sampler with balanced exploration and convergence behaviour.
    Each sampled trajectory must terminate in a state with a high correspondence to the label, via the energy function.
    dynEBAM may be a type of finite-step denoising flow model.
    '''
    def post_init(self, **params):
        self.r_sigma = params['r_sigma'] # retrieval noise
        self.r_optim = torch.optim.SGD([torch.zeros(1)], lr=params['lr']*10, momentum=0.2, weight_decay=0.2, fused=True)
    
    def opt_forward(self, x, beta0=1, iters=None, loss_hist=None, wb=None):
        if (iters is None): iters = self.default_iters
        if (wb is None): wb = self.w
        res = torch.zeros_like(x, requires_grad=True)
        self.r_optim.add_param_group({'params':[res]})
        for i in range(iters):
            self.r_optim.zero_grad()
            noise = self.r_sigma * (1 - float(i+1)/iters) * torch.randn_like(x)
            loss = lss_energy(x + res + noise, wb, self.P, beta0).sum()
            loss.backward()
            self.r_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.r_optim.param_groups.clear()
        self.r_optim.state.clear()
        return  update_lse(x + res, wb, self.P, beta0) # stabilize w. detEBAM sharp final step
    
    def learn_(self, x, beta0=1, iters=None, c=8, loss_hist=None):
        if (iters is None): iters = self.default_iters
        #xr = self.recall_append(x, beta0, iters=iters, frac=0.2)
        x1 = x[None,:,:].repeat(c,1,1).view(-1,x.shape[-1]) # minibatch forward
        res = torch.zeros_like(self.w).requires_grad_()
        self.w_optim.add_param_group({'params':[res]})
        for i in range(iters):
            wb = self.w + res
            self.w_optim.zero_grad()
            loss = self.energy(wb, self.opt_forward(x1, beta0, iters=5, wb=wb), beta0).sum()
            loss.backward()
            self.w_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.w_optim.param_groups.clear()
        self.w_optim.state.clear()
        self.w += res

# an attempt to understand if dynEBAM is noise-contrastive.
class cl_dynEBAM(dynEBAM):

    def learn_(self, x, beta0=1, iters=None, c=8, loss_hist=None):
        if (iters is None): iters = self.default_iters
        #xr = self.recall_append(x, beta0, iters=iters, frac=0.2)
        x1 = x[None,:,:].repeat(c,1,1).view(-1,x.shape[-1]) # minibatch forward
        res = torch.zeros_like(self.w).requires_grad_()
        self.w_optim.add_param_group({'params':[res]})
        for i in range(iters):
            t_ = float(i+1)/iters
            wb = self.w + res
            self.w_optim.zero_grad()
            r1 = self.opt_forward(x1, beta0, iters=5, wb=wb)
            loss = (self.energy(r1, wb, beta0) - self.energy(r1 + t_*torch.randn_like(r1), wb, beta0)).sum()
            loss.backward()
            self.w_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.w_optim.param_groups.clear()
        self.w_optim.state.clear()
        self.w += res

'''

ramblings : what is up with dynEBAM?

The LSR/LSS energy performed much better than LSE for dynEBAM, but worse for original EBAM, and terribly for det(deterministic)EBAM. LSR/LSS is naturally characterized by having the property of 'simultaneous generative state and exact state storage', which if you follow one of the papers stating the vulnerability of high-order kernel energies to 'dead neurons' and flat spaces (which we interpret as spurious memories), then we consider that through the lens of a multi-transport process, LSR/LSS is prone to stable intermediate states that lose their stability upon entering the kernel span of stable states closer to the objective.

inspiration from : Associative memory and dead neurons @ https://arxiv.org/html/2410.13866v2

By having a domain-limited descent, the LSR/LSS kernel reinforces the performance of multi-step retrievals which are able to 'hopscotch' their way to the objective; in the deterministic model, however, minimizing only the terminal state energy during learning means that these 'hopscotch' states are not optimized. It should also be noted that momentum-exploiting stochastic samplers can vastly improve the trajectory conditioning and discovery of such 'hopscotch' states, while first-order deterministic methods may fail to detect such transitions.

'''