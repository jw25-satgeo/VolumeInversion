import sys, os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# TO STUDY:

# Explaining Grokking and Information Bottleneck through Neural Collapse Emergence @ https://openreview.net/forum?id=sLX5P7FTfT
# Optimizer Choice Matters For The Emergence of Neural Collapse @  https://openreview.net/forum?id=9EPYWJrib1
# Avoid Catastrophic Forgetting with Rank-1 Fisher from Diffusion Models @ https://openreview.net/forum?id=zCZcbRsc4g

# -- --


# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation @ https://arxiv.org/html/2506.14202v4
# .. also read @ https://intuitivepapers.ai/diffusionblocks
# + : tutorial @ https://jmtomczak.github.io/blog/16/16_score_matching.html
# + : discussion @ https://sander.ai/2024/06/14/noise-schedules.html : the harm of the 'noise schedule' term
# + : Langevin Dynamics and Score Matching @ https://github.com/enhuiz/langevin-dynamics-and-score-matching
def EqualDensity_Noising_Sigmas(N, sigma0=1e-3, sigma1=10, p_mean=-1.2, p_std=1.2, ascending=False):
    norm = torch.distributions.normal.Normal(0, 1)
    cdf0 = norm.cdf((math.log(sigma0) - p_mean) / torch.Tensor([p_std]))
    cdf1 = norm.cdf((math.log(sigma1) - p_mean) / torch.Tensor([p_std]))
    p = cdf0 + (cdf1 - cdf0) * torch.linspace(0, 1, N)
    sigmas = torch.exp(p_mean + p_std * norm.icdf(p)).flip(dims=[0])
    if (ascending): sigmas = sigmas.flip(dims=[0])
    return sigmas
class DiffusionBlocksWrapper(nn.Module):
    '''
    A simple score matching wrapper for ResNet architectures interpreted as denoising diffusion models.
    '''
    def __init__(self, resnet, lossfunc, sigmas):
        super().__init__()
        self.resnet = resnet
        self.lossfunc = lossfunc
        self.sigma = sigmas
        # convenience references
        self.whitener = nn.Identity()
        self.unwhitener = nn.Identity()
    
    def train(self, x, y=None):
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        if y is None: y = x
        for i in range(len(self.resnet.net)):
            # single block operates within prescribed noise range (DiffusionBlocks)
            noise = torch.randn_like(x)
            x1 = x + self.sigma[i  ] * noise  # high noise
            x2 = y + self.sigma[i+1] * noise  # low  noise
            x1 = self.resnet.net[i](x1)
            # score matching uses sigma scaled loss (jmtomczak's tutorial)
            loss1 = self.lossfunc(x1, x2)
            # identity constraint / zero-score case for target centred score matching
            loss2 = self.lossfunc(self.resnet.net[i](x), y)
            # backprop
            lossi = (loss1 + loss2) / (2 * self.sigma[i])**2
            lossi.backward()
            loss += lossi.detach()
        return loss

# Improved Techniques for Training Score-Based Generative Models (Song & Ermon 2020) @ https://proceedings.neurips.cc/paper_files/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf
# GUD: Generation with Unified Diffusion (Gerdes & Welling, 2024) @ https://arxiv.org/html/2410.02667v1
def _bnoise(x, p): return torch.bernoulli(torch.ones_like(x)*p)
class DiffusionBlocksWrapper_Conditional(DiffusionBlocksWrapper):
    ''' '''
    def train(self, x, mask, do_backprop=True):
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        b_prob = (self.sigma / self.sigma.amax(dim=0)) # from sigma->0 noise to 1->0 dropout
        bnoise = (1, _bnoise(x, b_prob[0]))
        for i in range(len(self.resnet.net)):
            gnoise = torch.randn_like(x)
            bnoise = ( bnoise[1], bnoise[1] * _bnoise(x, b_prob[i+1]/b_prob[i]) ) # gradual corruption
            x1 = ( x + mask*gnoise*self.sigma[i  ]*(1-bnoise[0]) ) * (~mask)*bnoise[0]
            x2 = ( x + mask*gnoise*self.sigma[i+1]*(1-bnoise[1]) ) * (~mask)*bnoise[1]
            x2i, lossi = self.resnet.net[i](x1, ~mask), 0
            lossi = lossi + self.lossfunc(x2i, x) / (2 * self.sigma[i]**2) # SM loss
            lossi = lossi + 0.25 * self.lossfunc(x2i, x2) / (2 * self.sigma[i]**2) # consistency loss
            lossi = lossi + 0.25 * self.lossfunc(self.resnet.net[i](x, ~mask), x ) / (2 * self.sigma[i]**2) # identity loss
            lossi = lossi + 0.25 * self.lossfunc(x2i, self.resnet.net[i](x2i, ~mask)) / self.sigma[i] # idempotence loss
            if do_backprop: lossi.backward()
            loss += lossi
        return loss / len(self.resnet.net)

# Mean Velocity Matching: Rethinking Generative Dynamics in Diffusion Models (Zhang et al. 2026) @ https://arxiv.org/html/2609.25444v1
class VMSampler_Cond(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.net = wrapper.net
        self.schedf = wrapper.schedf
        self.T = wrapper.T
        self.n = wrapper.n

    def forward(self, x, cond_mask=None):
        sched = torch.cat((torch.zeros(1), torch.linspace(0,1,self.n)))
        sched = self.schedf(sched.to(x.device,x.dtype), invert=True).detach()
        sched = (sched[1:] - sched[:-1])[:,None,None,:]
        for dt in sched.flip(0):
            x = self.step(x, dt, cond_mask)
        return x
    
    def step(self, x, dt, cond_mask=None):
        eta = dt**0.5 * (1 / self.T)**0.5
        x1 = x + self.net(x) * dt + eta * torch.randn_like(x)
        if cond_mask is not None:
            if cond_mask.dtype is torch.bool: x1[cond_mask] = x[cond_mask]
            elif cond_mask.dtype.is_floating_point: x1 = cond_mask*x + (1 - cond_mask)*x1
        return x1

from timewarp import Timewarp_Logistic
class VMWrapper(nn.Module):
    def __init__(self, net, n_features, n=32, T=10.0):
        super().__init__()
        self.net = net    # velocity network
        self.T = T
        self.n = n # number of timesteps
        self.schedf = Timewarp_Logistic(n_features, torch.zeros(1), torch.full((1,),T))

    def get_sampler(self):
        return VMSampler_Cond(self)
        
    def train(self, x, do_backprop=True):
        loss, T = torch.empty(self.n, x.shape[-1]).to(x.device, x.dtype), self.T
        self.sched = torch.linspace(0,1,self.n).to(x.device, x.dtype)
        self.sched = self.schedf(self.sched, invert=True)
        xv = x.var(dim=0) + 1e-5
        for i in range(self.n):
            t, gnoise = self.sched[i][None,None,:].detach(), torch.randn_like(x)
            x1 = x * (1 - t/T) + gnoise * (t / T)**0.5
            v = self.net(x1)
            # reparameterized regression loss
            u_r = (t**0.5 / T) * x - gnoise / T**0.5
            u_i = (t**0.5    ) * v
            lossi = ( (u_r - u_i).pow(2).mean(dim=0) / xv ).flatten() # velocity matching loss
            loss[i] = lossi.detach()
            if do_backprop: lossi.mean().backward() # block-wise only backpropagation
        if do_backprop: self.schedf.loss_fn(self.sched, loss).mean().backward()
        return loss.mean()
        

class VelMatchWrapper_Cond_DiffusionBlocks(nn.Module):
    def __init__(self, resnet, T=10):
        raise ValueError('This module may not be functional! Reformulation may be necessary.')
        super().__init__()
        self.resnet = resnet
        self.T = T        # for forward drift
        for i in range(len(self.resnet.net)):
            self.resnet.net[i].eta = (1/T)**0.5
            self.resnet.net[i].dt  = (1/T)
        
    def train(self, x, mask, do_backprop=True):
        n, loss, T = len(self.resnet.net), 0, self.T
        for i in range(n):
            t, gnoise = T * (i/n) , torch.randn_like(x)
            x1 = x * (1 - t/T) + gnoise * (t/T)**0.5
            xi = self.resnet.net[i](x1, ~mask)
            # reparameterized regression loss
            u_ref = (t**0.5 / T) * x - gnoise / T**0.5
            u_inf = t**0.5 * (xi - x1)
            lossi = ( (u_ref - u_inf).pow(2) ).mean() # VM loss, normalized
            if do_backprop: lossi.backward() # block-wise only backpropagation
            loss = loss + lossi.detach() / n
        return loss

# latent space conditional denoiser
class ScoreMatchingWrapper_LatentConditional(DiffusionBlocksWrapper):

    def __init__(self, resnet, lossfunc, ae, sigma0=1e-3, sigma1=10, p_mean=1.2, p_std=1.2):
        super().__init__(resnet, lossfunc, sigma0, sigma1, p_mean, p_std)
        # deep copy via disk
        torch.save(ae, '_tmp.pth')
        self.ae = torch.load('_tmp.pth', weights_only=False).to(self.resnet.parameters().__next__().device)
        os.remove('_tmp.pth')
        for param in self.ae.parameters():
            param.requires_grad = False
    
    def train(self, x, mask, do_backprop=True):
        '''
        x : raw input before encoding
        mask : 1D feature dimension mask for unknown variable locations
        '''
        z = self.ae.enc(x)
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        b_prob = torch.linspace(1,0,len(self.sigma))
        for i in range(len(self.resnet.net)):
            # missingness scheduling with feature-space dropout w. masked conditionality
            #x_b1 = x * (1 - mask * torch.bernoulli(torch.full_like(x, b_prob[i  ])))
            #x_b2 = x * (1 - mask * torch.bernoulli(torch.full_like(x, b_prob[i+1])))

            self.ae.enc(x * (1 - mask * torch.bernoulli(torch.full_like(x, b_prob[i  ]))))
            
            z0 = F.dropout(z, b_prob[i  ]) + self.sigma[i+1] * torch.randn_like(z)
            z1 = F.dropout(z, b_prob[i+1]) + self.sigma[i  ] * torch.randn_like(z)
            
            z1i = self.resnet.net[i](z0)
            
            loss1 = self.lossfunc(z1, z1i) / (2 * self.sigma[i]**2) # x1 = f(x0)
            #loss2 = self.lossfunc(self.resnet.net[i](z), z) # z = f(z)
            #loss3 = (self.ae.dec(z1i)[...,~mask[0,0]] - x[...,~mask[0,0]]).abs().sum() # f(c) = c
            # backprop
            lossi = loss1# + loss2# + loss3
            if do_backprop: lossi.backward()
            loss += lossi
        return loss


# WARNING - USE APPEARS TO CAUSE GENERALIZATION PROBLEMS (but allows faster convergence)
# .. Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization @ https://proceedings.mlr.press/v139/wadia21a/wadia21a.pdf
class FixedRescaler(nn.Module):
    # A very basic feature normalizer that uses statistics of the first input batch.
    def __init__(self, data=None, eps=1e-5):
        super().__init__(); self.initialized = False; self.eps = eps
        if data is not None: self.initialize(data)
    def initialize(self, x):
        self.register_buffer('mean', x.mean(0, keepdims=True).detach().clone())
        self.register_buffer('sq_v', (x.var(0, keepdims=True) + self.eps).pow(0.5).detach().clone())
        self.initialized = True
    def forward(self, x):
        if not self.initialized: self.initialize(x)
        return (x - self.mean) / self.sq_v
class FixedDescaler(nn.Module):
    def __init__(self, fixedrescaler):
        super().__init__()
        self.fr = fixedrescaler
    def forward(self, x):
        return (x * self.fr.sq_v) + self.fr.mean

class FixedRescaler_mm(nn.Module):
    def __init__(self, data=None, eps=1e-5):
        super().__init__(); self.initialized = False; self.eps = eps
        if data is not None: self.initialize(data)
    def initialize(self, x):
        self.register_buffer('min', x.amin(0, keepdims=True).detach().clone())
        self.register_buffer('max', x.amax(0, keepdims=True).detach().clone())
        self.initialized = True
    def forward(self, x):
        if not self.initialized: self.initialize(x)
        return 2 * (x - self.min)/(self.max - self.min) - 1
class FixedDescaler_mm(nn.Module):
    def __init__(self, fixedrescaler):
        super().__init__(); self.fr = fixedrescaler
    def forward(self, x): return 0.5*(x + 1)*(self.fr.max - self.fr.min) + self.fr.min
        
        

        
# proportional noisy sensitizer
# The Dormant Neuron Phenomenon in Deep Reinforcement Learning @ https://arxiv.org/abs/2302.12902
# Adding Gradient Noise Improves Learning for Very Deep Networks @ https://arxiv.org/abs/1511.06807
class Sensitizer():
    '''
    A scaled gradient noise injection method targeting first-order loss sensitivity of weights.
    1. Use with L2 normalization!
    2. Use between loss.backwards() and optimizer.step().
    Note - Seems to do well on SGD, but little point in it on AdamW, which appears to address its issues. In fact, it seems to break it.
    '''
    def __init__(self, params, beta=100, sigma0=1e-3):
        # should be retrieved via nn.Module.parameters()
        self.params = params
        self.beta = beta
        self.sigma0 = sigma0

    def step(self):
        for param in self.params:   # restoration_noise = σ*exp(-B|∇L_W||W|)
            sigma = self.sigma0 * torch.exp(-self.beta * param.grad.abs() * param.data.abs())
            param.grad = param.grad + sigma * torch.randn_like(param.grad)


# Generative Cross-Entropy: A Strictly Proper Loss for Data-Efficient Classification (Zhan, Zhou, Shen 2026) @ https://arxiv.org/html/2604.06689v2
#  ... why is this so similar to ebm stuff? Huh. I guess it's a bit like energy contrastive learning.
# ++ WARNING ++
#   Improving Discriminator Guidance in Diffusion Models @ https://arxiv.org/abs/2503.16117
#    "Specifically, we show that training the discriminator using Cross-Entropy loss, as commonly done, can in fact
#     increase the Kullback-Leibler divergence between the model and target distributions, particularly when the discriminator overfits."
# My own experience: For an iterative retrieval task, overfitting on GCE may cause rank collapse via overconfident or premature separation.
# ++         ++
def batch_gce_loss(x, y, beta=1):
    '''
    A batchwise cross-entropy loss incorporating generative probability theory.
    1. Use with small batches / minibatching!
    2. Use with MAE/MSE - GCE is a contrastive loss only, and may cause manifold drift.
    '''
    B, N = x.shape
    score = (1/beta) * (x @ y.T) # scaled similarity
    loss = -(torch.logsumexp(score, dim=-1) - score.diag()).mean()
    return loss







# --
#  ..  BELOW  :  NOT STUDIED / NEEDS INVESTIGATION TO IMPLEMENT
# --


# One-Step Generative Modeling via Wasserstein Gradient Flows @ https://arxiv.org/html/2605.11755v2
#  ..  Calculates Sinkhorn divergence (entropy regularized Wasserstein dist metric) as loss function.
# Wasserstein Gradients & Langevin Diffusions @ https://alexxthiery.github.io/notes/wasserstein_langevin/wasserstein_langevin.html
# Rethinking the Diffusion Model from a Langevin Perspective @ https://iclr-blogposts.github.io/2026/blog/2026/rethinking-diffusion-langevin/
# Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models @ https://proceedings.mlr.press/v235/wu24i.html
#  ..  Extends memory retrieval to arbitrary separation losses in the kernelized energy framework.
# A kernel method for the learning of Wasserstein geometric flows @ https://arxiv.org/html/2511.06655v1
class Training_WGFMWrapper(nn.Module):
    '''
    WORK IN PROGRESS
    '''
    def __init__(self, resnet, lossfunc, sigma0=1.0, T0=100.0, generator=None):
        super().__init__()
        self.rnet = resnet
        self.lossfunc = lossfunc
        self.d1_gen = generator
        self.sigma0 = sigma0
        self.T0 = T0
        if generator is None: # diffusion task default : staged Gaussian
            self.d1_gen = lambda x, y : x + y*torch.randn_like(x)

    def forward(self, x):
        L, (B,C,N) = len(self.rnet.net), x.shape
        T = torch.linspace(T0, 1.0).to(x.device, x.dtype)
        grads = torch.zeros(L,B,C,N).to(x.device, x.dtype)
        
        xi = self.generator(x, self.sigma0)
        for i in range(N):
            grads[i] = self.rnet.net[i](xi) - xi
            gnorm = grads[i].flatten(1).norm(dim=1, keepdim=True) # stepsize = diff magnitude
            xi = xi + grads[i] + (2*gnorm*T[i]).pow(0.5) * torch.randn_like(xi) # Langevin descent
        
        loss = self.lossfunc(x, xi)
        return loss


# Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels (Zhang & Sabuncu, 2018) https://arxiv.org/abs/1805.07836
# + : Generalized Kullback-Leibler Divergence Loss (Cui et al. 2025) @ https://arxiv.org/html/2503.08038v1   ...    https://github.com/jiequancui/DKL
# https://github.com/AlanChou/Truncated-Loss/tree/master (implementation of Zhang & Sabuncu)
class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q # scalar
        self.k = k # scalar
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


