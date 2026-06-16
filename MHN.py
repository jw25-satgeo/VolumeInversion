import random
import numpy as np
import numbers
import math

import torch
import torch.nn.functional as F

# look : https://github.com/MichaelArbel/GeneralizedEBM ~ underdamped langevin (similar regime as momentum SGD) drives full transport
# look : https://arxiv.org/html/2505.18069v1 ~ SGD with regularization can converge to Hebbian-like learning

# look : Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy @ https://arxiv.org/html/2603.06875v3 ~ Temperature boundaries for SLGD MHN
# ++ : Modern Hopfield Energy does not require a separate partition function ; Hopfield is analytic-intrinsic normalized , other EBMs are not.
# Energy minimization for trianing dense associative memory @ https://openreview.net/pdf?id=d7PjRsguov
class EBAM(torch.nn.Module):

    def __init__(self, key_size=16, emb_size=32, mem_size=64, e_beta=1, iters=10, i_lr=0.05, r_beta=0, r_sigma=0):
        super().__init__()
        self.e_beta = e_beta
        self.default_iters = iters
        self.i_lr = i_lr
        if emb_size != key_size and (emb_size % 2 != 0 or emb_size//2 < key_size):
            raise ValueError('embedding length must be even & at least double input length')
        self.key_size = key_size
        self.emb_size = emb_size
        self.mem_size = mem_size
        # internal RNN memory should be isolated from Module parameter graph
        self.register_buffer('w', torch.randn(emb_size, mem_size, requires_grad=True))

        # retrieval dynamic parameters
        self.r_beta  = r_beta  # momentum
        self.r_sigma = r_sigma # langevin noise injection

    def io_convert(self, x):
        if (self.emb_size == self.key_size): return x
        if (x.shape[-1] == self.key_size):
            x1 = F.pad(x, (0, self.emb_size//2 - self.key_size))
            x1 = torch.view_as_real(torch.fft.fft(x1, dim=-1)).reshape(x.shape[0],-1)
        elif (x.shape[-1] == self.emb_size):
            x1 = torch.fft.ifft(torch.view_as_complex(x.reshape(x.shape[0],-1,2)), dim=-1).real
            x1 = x1[..., :self.key_size]
        else: raise ValueError('input is neither key nor embedding shape')
        return x1

    #  A Convergent Method for Energy Optimization in Modern Hopfield Networks (Bao et al. 2026 @ https://www.mdpi.com/2504-2289/10/3/71)
    def energy(self, x): # MHN energy
        a = -(1/self.e_beta) * (torch.logsumexp(-self.e_beta * (x @ self.w), dim=-1, keepdim=True) + math.log(self.w.shape[0]))
        b = 0.5 * (x**2).sum(dim=-1,keepdim=True)
        return a + b
    
    def forward(self, x, iters=None):
        if (iters is None): iters = self.default_iters
        x1 = self.io_convert(x)
        res_lr = (1+torch.amax(x1**2))/iters
        res = torch.zeros_like(x1, requires_grad=True) #  key residual
        vel = torch.zeros_like(x1) # momentum,         annealing sched
        #sched = torch.cos(torch.linspace(0,torch.pi/2 - 1/iters,iters))
        for i in range(iters):
            loss = self.energy(x1 + res).sum()
            print(loss)
            grad = torch.autograd.grad(loss, res, create_graph=True)[0]
            noise = self.r_sigma*sched[i]*torch.randn_like(res)
            vel = self.r_beta*vel - sched[i]*res_lr*grad + noise
            res = res + vel
        y = x1 + res
        return self.io_convert(y) # storage dim to key dim
        
    def learn(self, x, level=None, iters=None):
        if (iters is None): iters = self.default_iters
        x1 = self.io_convert(x)
        for i in range(iters):
            loss = self.energy(x1).sum() # 'level' drives weighted update
            grad = torch.autograd.grad(loss, self.w)[0]
            with torch.no_grad():
                self.w -= self.i_lr * grad
        
# ref : https://github.com/santiag0m/hopfield-networks/tree/main
# ref : https://ml-jku.github.io/hopfield-layers/#energy
class ContinuousHopfield_Query:
    def __init__(self, beta):
        self.beta = beta

    # softmax is equivalent to update rule operant on MHN scalar energy
    def forward(self, query, patterns):
        state = torch.softmax(self.beta * query @ patterns.T, dim=-1) @ patterns
        return query

    # scalar energy
    def energy(self, query, patterns):
        lse = torch.logsumexp(self.beta * (query @ patterns.T), dim=-1) / self.beta # S x F @ F x N = S x N
        norm = (query**2).sum(dim=-1, keepdim=True) * 0.5
        energy = -1 * lse + norm + np.log(patterns.shape[0])/self.beta + 0.5*(patterns**2).sum(axis=-1).max()
        return energy

# ref : https://github.com/MAGICS-LAB/UHop/tree/main
class UMHN(torch.nn.Module):

    def __init__(self, beta=1, kernel=None):
        super().__init__()
        self.beta = beta
        self.activation = torch.softmax
        self.overlap = kernel.kernel_fn # weighted matrix multiplication effector
        self.Xi = torch.zeros(33)       # stored trainable patterns (backprop-updated)

    def energy(self, x):
        # x: D, Xi: (D, M)
        return -torch.logsumexp(self.beta*(self.Xi.T @ x),dim=0) + 0.5*torch.dot(x,x) + torch.log(torch.tensor(self.Xi.size(-1))) + 0.5

    def update_rule(self, x, steps=100):
        for _ in range(steps):
            score = self.beta * self.activation(self.overlap(self.Xi, x), dim=-1)
            x = self.Xi @ score
        return x
        


#def memory_retrieval(Xi, update_rule, activation=torch.softmax, overlap=dot_product, steps=1, beta=1, noise_level=1):
#    dist = []
#    Xi = Xi.T
#    for m in range(Xi.size(-1)):
#        x = Xi[:,m].clone()
#        p = torch.normal(0, noise_level, size=x.size())
#        q = torch.clamp(torch.abs(x.clone() + p), 0, 1)
#
#        x_new = update_rule(Xi, q, beta, steps, overlap=overlap, activation=activation)
#        dist.append(sqdiff(x, x_new))
#    return torch.tensor(dist).mean()