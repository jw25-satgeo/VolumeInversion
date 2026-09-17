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
class ScoreMatchingWrapper(nn.Module):
    '''
    A simple score matching wrapper for ResNet architectures interpreted as denoising diffusion models.
    '''
    def __init__(self, resnet, lossfunc, sigmas, whitener=None):
        super().__init__()
        self.resnet = resnet
        self.lossfunc = lossfunc
        self.sigma = sigmas
        # convenience references
        self.whitener = whitener
    
    def train(self, x, y=None, iters=1):
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        if y is None: y = x
        for i in range(len(self.resnet.net)):
            # single block operates within prescribed noise range (DiffusionBlocks)
            x1 = x + self.sigma[i+1] * torch.randn_like(x)
            x2 = y + self.sigma[ i ] * torch.randn_like(x)
            for j in range(iters):
                x1 = self.resnet.net[i](x1)
            # score matching uses sigma scaled loss (jmtomczak's tutorial)
            loss1 = self.lossfunc(x1, x2)
            # identity constraint / zero-score case for target score matching
            loss2 = self.lossfunc(self.resnet.net[i](x), y)
            # backprop
            lossi = (loss1 + loss2) / (2 * self.sigma[i])**2
            lossi.backward()
            loss += lossi.detach()
        return loss

    def forward(self, x, iters=1): # Langevin sampling
        N = len(self.resnet.net)
        for i in range(N):
            eta = self.sigma[i+1] - self.sigma[i]
            for j in range(iters):
                x = self.resnet.net[i](x) + eta * torch.randn_like(x)
        return x

def _bnoise(x, p):
    return torch.bernoulli(torch.full_like(x, p))
class ScoreMatchingWrapper_Conditional_Bernoulli(ScoreMatchingWrapper):
    def train(self, x, mask, do_backprop=True):
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        b_prob = self.sigma / (self.sigma.min() + self.sigma.max())
        for i in range(len(self.resnet.net)):
            x1 = ( x + mask*self.sigma[i+1]*_bnoise(x,b_prob[i  ])*torch.randn_like(x) ) * (1 - mask*_bnoise(x,b_prob[i  ]))
            x2 = ( x + mask*self.sigma[i  ]*_bnoise(x,b_prob[i+1])*torch.randn_like(x) ) * (1 - mask*_bnoise(x,b_prob[i+1]))
            x2i = self.resnet.net[i](x1)
            loss1 = self.lossfunc(x2i, x2) / (2 * self.sigma[i]**2)
            loss2 = self.lossfunc(x2i, x ) / (2 * self.sigma[i]**2)
            lossi = loss1 + 0.1 * loss2
            if do_backprop: lossi.backward()
            loss += lossi
        return loss / len(self.resnet.net)

class ScoreMatchingWrapper_Conditional(ScoreMatchingWrapper):

    def train(self, x, mask, do_backprop=True):
        loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        b_prob = self.sigma / self.sigma.max()
        for i in range(len(self.resnet.net)):
            #x1 = x + mask*self.sigma[i+1]*torch.randn_like(x)
            #x2 = x + mask*self.sigma[i  ]*torch.randn_like(x)
            x1 = ( x + mask*self.sigma[i+1]*_bnoise(x,b_prob[i  ])*torch.randn_like(x) ) * (1 - mask*_bnoise(x,b_prob[i  ]))
            x2 = ( x + mask*self.sigma[i  ]*_bnoise(x,b_prob[i+1])*torch.randn_like(x) ) * (1 - mask*_bnoise(x,b_prob[i+1]))
            x2i = self.resnet.net[i](x1, ~mask)
            losses = []
            losses.append( self.lossfunc(x2i, x2) / (2 * self.sigma[i]**2) ) # score matching loss
            losses.append( 0.5 * self.lossfunc(self.resnet.net[i](x, ~mask), x ) / (2 * self.sigma[i]**2) ) # identity loss
            losses.append( 0.5 * self.lossfunc(x2i, self.resnet.net[i](x2i, ~mask)) / self.sigma[i] ) # idempotence loss
            lossi = sum(losses)
            if do_backprop: lossi.backward()
            loss += lossi
        return loss / len(self.resnet.net)

# latent space conditional denoiser
class ScoreMatchingWrapper_LatentConditional(ScoreMatchingWrapper):

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
        super().__init__()
        self.initialized = False
        self.eps = eps
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


