import random
import numpy as np
import numbers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adaptive Hopfield Network: Rethinking Similarities in Associative Memory
# @ : https://arxiv.org/html/2511.20609v1 , https://github.com/shurongwang/Adaptive-Hopfield-Network
# ..
# adaptive kernel projected dot product similarity
class AdKernel(nn.Module):
    def __init__(self, inp_size=16, num_layers=3):
        super().__init__()
        self.net = nn.Sequential(*sum([[
            nn.LayerNorm(inp_size),
            nn.Linear(inp_size, inp_size*2),
            nn.GELU(),
            nn.GLU(dim=-1)
        ] for i in range(num_layers)], []))
    def forward(self, x, y):
        return self.embed(x) @ self.embed(y).transpose(-1,-2)
    def embed(self, x): #...,B,N, ...,L,N
        return self.net(x.view(-1,x.shape[-1])).view_as(x)

def f_dp_sim(x, y): return x @ y.transpose(-1,-2)
def f_ed_sim(x, y): return -((x[...,None,:]-y[...,None,:,:])**2).sum(-1)

# (func distance:) Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models
# ... src @ (Millidge et al. 2022) https://arxiv.org/abs/2202.04557
# Dense Associative Memory with Epanechnikov Energy @ https://arxiv.org/abs/2506.10801
def lsr_energy(x, w, P, beta=1): # note : relu replaced with softplus for stability
    return -(1/beta) * torch.log1p(1 + F.softplus(0.5*beta*P(x,w))).sum(dim=-1, keepdim=True)
# Hopfield Networks is All You Need @ https://arxiv.org/abs/2008.02217
def lse_energy(x, w, P, beta=1):
    a = 0.5*(x**2).sum(dim=-1, keepdim=True)
    b = -(1/beta)*torch.logsumexp(-0.5*beta*P(x,w),dim=-1,keepdim=True) + math.log(w.shape[-2])
    return a + b

class EBAM(nn.Module):

    def __init__(self, inp_size=16, mem_size=2, lr=0.05, iters=20, r_sigma=0):
        super().__init__()
        self.default_iters = iters
        self.register_buffer('w', torch.randn(mem_size, inp_size))
        self.w_optim = torch.optim.SGD([torch.zeros(1)], lr=lr, momentum=0.8, dampening=0.1, weight_decay=0.2, fused=True)

        # retrieval hyperparameters
        self.r_sigma = r_sigma # retrieval noise
        self.r_optim = torch.optim.SGD([torch.zeros(1)], lr=lr, momentum=0.8, weight_decay=0.2, fused=True)
        
        # overlap adaptive distance
        self.adk = AdKernel(inp_size, num_layers=2)
        self.adk_optim = torch.optim.SGD(self.adk.parameters(), lr=lr, momentum=0.4, weight_decay=0.5, fused=True)

    # terminate recursive access to internal networks during global optimization
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return super().named_parameters(prefix, False, remove_duplicate)

    def energy(self, x, w, dist_f, beta=1):
        return lse_energy(x, w, dist_f, beta)

    def forward(self, x, beta0=1, iters=None, loss_hist=None, kernel=None):
        if (iters is None): iters = self.default_iters
        if (kernel is None): kernel = self.adk
        res = torch.zeros_like(x, requires_grad=True)
        self.r_optim.add_param_group({'params':[res]})
        for i in range(iters):
            self.r_optim.zero_grad()
            noise = self.r_sigma * (1 - float(i+1)/iters) * torch.randn_like(x)
            loss = self.energy(x + res + noise, self.w, kernel, beta0).sum()
            loss.backward()
            self.r_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.r_optim.param_groups.clear()
        self.r_optim.state.clear()
        return x + res

    # ENERGY MINIMIZATION FOR TRAINING DENSE ASSOCIATIVE MEMORY @ https://openreview.net/pdf/5dfb5eb06ae1fbf962e6657abdd43155e817f78f.pdf
    def learn(self, x, beta0=1, iters=None, c=8, loss_hist=None, kern_iters=20):
        if (iters is None): iters = self.default_iters
        x1 = x[None,:,:].repeat(c,1,1) # minibatch forward
        res = torch.zeros_like(self.w).requires_grad_()
        self.w_optim.add_param_group({'params':[res]})
        ns = torch.linspace(0,1,c)[:,None,None].to(x.device,x.dtype) # noise schedule
        # 1 :: weight optimization
        for i in range(iters):
            t_ = float(i+1)/iters
            wb = self.w + res
            self.w_optim.zero_grad()
            noise = ((1-t_)*ns)**2 * torch.randn_like(x1) # quadratic cooling schedule
            loss = self.energy(x1 + noise, wb, f_dp_sim, beta0).mean(dim=0).sum()
            loss.backward()
            self.w_optim.step()
            if loss_hist is not None: loss_hist.append(loss.detach().item())
        self.w_optim.param_groups.clear()
        self.w_optim.state.clear()
        self.w += res
        # 2 :: kernel optimization while dreaming
        for i in range(kern_iters):
            self.adk_optim.zero_grad()
            wb = self.adk.embed(self.forward(torch.randn_like(x), beta0, iters//2, kernel=f_ed_sim))
            loss = (-2 * torch.pdist(F.normalize(wb, dim=-1), p=2)).exp().mean(dim=-1).log().sum()
            loss.backward()
            self.adk_optim.step()
            if i % 5 == 0 :print('kern_loss :', loss.item())


# look : https://github.com/MichaelArbel/GeneralizedEBM ~ underdamped langevin (similar regime as momentum SGD) drives full transport
# look : https://arxiv.org/html/2505.18069v1 ~ SGD with regularization can converge to Hebbian-like learning

# look : Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy @ https://arxiv.org/html/2603.06875v3 ~ Temperature boundaries for SLGD MHN
# ++ : Modern Hopfield Energy does not require a separate partition function ; Hopfield is analytic-intrinsic normalized , other EBMs are not.
# Energy minimization for trianing dense associative memory @ https://openreview.net/pdf?id=d7PjRsguov
class EBAM_noCD_nonfunctional(torch.nn.Module):

    def __init__(self, key_size=16, emb_size=32, mem_size=64, iters=10, i_lr=0.05, r_beta=0, r_sigma=0):
        super().__init__()
        self.default_iters = iters
        self.i_lr = i_lr
        if emb_size != key_size and (emb_size % 2 != 0 or emb_size//2 < key_size):
            raise ValueError('embedding length must be even & at least double input length')
        self.key_size = key_size
        self.emb_size = emb_size
        self.mem_size = mem_size
        # internal RNN memory should be isolated from Module parameter graph
        self.register_buffer('w', torch.randn(emb_size, mem_size, requires_grad=True))
        self.w_optim = torch.optim.SGD([self.w], lr=i_lr, momentum=0.99, weight_decay=0, fused=True)

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
    def energy(self, x, ibeta=1): # MHN energy, with inverse beta parameter
        x1, w1 = F.normalize(x, dim=0), F.normalize(self.w, dim=0)
        #a = -ibeta * (torch.logsumexp(-(1/ibeta) * (x @ self.w), dim=-1, keepdim=True) + math.log(self.w.shape[0]))
        #b = 0.5 * (x**2).sum(dim=-1,keepdim=True)
        a = -ibeta * (torch.logsumexp(-(1/ibeta) * (x1 @ w1), dim=-1, keepdim=True) + math.log(w1.shape[0]))
        b = 0.5 * (x1**2).sum(dim=-1,keepdim=True)
        return a + b
    
    def forward(self, x, iters=None, ibeta0=1):
        if (iters is None): iters = self.default_iters
        x1 = self.io_convert(x)
        res = torch.zeros_like(x1, requires_grad=True) #  key residual
        vel = torch.zeros_like(x1) # momentum,         annealing sched
        sched = torch.cos(torch.linspace(0, torch.pi/2, iters)).to(x1.device, x1.dtype)
        #sched = torch.exp(torch.linspace(0, -8, iters)).to(x1.device, x1.dtype)
        grad = torch.ones(1).to(x1.device, x1.dtype) # init grad for adaptive energy annealing
        for i in range(iters):
            ibeta = ibeta0
            #ibeta = (ibeta0 * torch.exp(-grad.norm()**0.05)).clamp(min=1e-5)
            loss = self.energy(x1 + res, ibeta).sum()
            #loss = loss + (F.softplus(loss) * (res**2).sum())
            if i%20==0: print(loss, ibeta)
            grad = torch.autograd.grad(loss, res, create_graph=True)[0]
            vel = self.r_beta*vel - sched[i]*(self.i_lr*grad + self.r_sigma*torch.randn_like(res))
            res = res + vel
        y = x1 + res
        return self.io_convert(y) # storage dim to key dim

    #def learn(self, x, ibeta=None, iters=None):
        
        
    def learn(self, x, ibeta0=None, iters=None):
        if (iters is None): iters = self.default_iters
        ibeta = ibeta0
        x1 = self.io_convert(x)
        for i in range(iters):
            self.w_optim.zero_grad()
            loss = self.energy(x1, ibeta).sum() # 'level' drives weighted update
            if i == 0: loss0 = loss.detach()
            #ibeta = (ibeta0 * torch.exp(-((loss0 - loss.detach()).abs()**0.05))).clamp(min=1e-5)
            ibeta = ibeta0
            loss.backward()
            self.w_optim.step()
            print(loss, ibeta)
            #grad = torch.autograd.grad(loss, self.w)[0]
            #with torch.no_grad():
            #    self.w -= self.i_lr * grad
        
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