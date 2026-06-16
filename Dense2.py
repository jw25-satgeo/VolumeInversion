import os, sys
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# classic residual MLP block
class MLPBlock(nn.Module):
    def __init__(self, size:int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),     # normalization
            nn.Linear(size, size),  # feedforward
            nn.GELU()               # activation
        )
    def forward(self, x):
        return x + self.block(x) # residual + forward

## https://arxiv.org/html/2510.01796v1 - Rethinking the shape convention of a residual MLP : Hourglass-shaped, or bulging MLP?
class hdrMLPBlock(nn.Module): #high-dimensional residual MLP
    def __init__(self, size:int, narrow:int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, size))
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, narrow),
            nn.GELU(),
            nn.Linear(narrow, size)
        )
    def forward(self, x):
        return x + torch.sigmoid(self.a) * self.block(x) # residual + weighted forward
        
class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int=None, hidden_size:int = 128, hidden_layers:int = 5):
        super().__init__()
        # set input/output sizes
        self.input_size = input_size
        if (output_size is None): self.output_size = hidden_size
        else : self.output_size = output_size
        # init layers
        layers = [nn.Linear(self.input_size, hidden_size), nn.GELU()]
        for i in range(hidden_layers):
            narrow = int(hidden_size - ((i+0.5)/hidden_layers)*(hidden_size - input_size))
            layers.append(hdrMLPBlock(hidden_size, narrow)) # forward-shrinking bottlenecks
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.Linear(hidden_size, self.output_size))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

# straight-through estimator for nondifferentiable modules
class Oper_STE_Wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        raise ValueError("THIS FUNCTION IS NOT YET COMPLETE : invalid for idim != odim")
        return module.o_wrap(module.run_model(*module.i_unwrap(x.detach())))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# operator
class SimOp(nn.Module):
    def __init__(self, ishape:list, oshape:list, diff:bool=True):
        '''
        A basic function interface for neural surrogate-compatible models.
        ishape : a list of shape tuples that the (flat) input vector should expand to.
        oshape : a list of shape tuples representing the output tensors before flattening.
        '''
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape

        self.idim = int(sum(np.prod(s) for s in ishape))
        self.odim = int(sum(np.prod(s) for s in oshape))
        
        self.can_diff = diff
        
        tmp = np.cumsum([0] + [np.prod(s) for s in ishape])
        self.i_inds = [(tmp[i], tmp[i+1]) for i in range(len(ishape))]

    def i_unwrap(self, x):
        variables = []
        for i in range(len(self.ishape)):
            j0,j1 = self.i_inds[i]           #B,          (...shape)
            variables.append(x[:, j0:j1].view(x.shape[0], *self.ishape[i]))
        return variables

    def o_wrap(self, x):
        return torch.cat([x[i].view(x[i].shape[0], -1) for i in range(len(x))], dim=1)

    def forward(self, x):
        if (self.can_diff):
            return self.o_wrap(self.run_model(*self.i_unwrap(x)))
        else:
            return Oper_STE_Wrapper.apply(x, self)

    def run_model(self, x):
        return x

# simulator
class SimProc(nn.Module):
    def __init__(self, operations=[]):
        super().__init__()
        self.ops = nn.ModuleList()
        for op in operations:
            if not isinstance(oper, SimOp):
                raise ValueError('all operations must inherit SimOp class.')
            if len(self.ops) > 0 and (op.idim != self.ops[-1].odim):
                raise ValueError('simulator stages are non-contiguous: I/O shape discrepancy')
            self.ops.append(op)
            
    def run_stage(self, x, index:int):
        return self.ops[index](x)

    def forward(self, x):
        for i in range(len(self.ops)):
            x = self.run_stage(x, i)
        return x

# research
# https://github.com/opendilab/awesome-exploration-rl

# ImpRAG: Retrieval-Augmented Generation with Implicit Queries
# src : (Zhang et al. 2025) @ https://arxiv.org/abs/2506.02279
#
# Associative Memory Based Experience Replay for Deep Reinforcement Learning
# src : (Li et al. 2022) @ https://arxiv.org/abs/2207.07791
#
# Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models
# src : (Millidge et al. 2022) @ https://arxiv.org/abs/2202.04557
#
# Prototypical Networks for Few-shot Learning
# src : (Snell et al. 2017) @ https://arxiv.org/abs/1703.05175

class Memory_Assoc(nn.Module):
    '''
    An associative memory object for storing simulation outcomes and rapidly retrieving them.
    '''
    def __init__(self, stage_lengths, beta=10.0):
        super().__init__()
        self.beta = beta
        #self.register_buffer("mem", torch.empty(0, sum(stage_lengths)))
        self.mem = torch.zeros((sum(stage_lengths),))
        self.mem_s_i = [sum(stage_lengths[:i] for i in range(len(stage_lengths)+1)]

    def forward(self, x):
        # x : (Bx, N), mem: (By, N)
        

class Memory_Simple(nn.Module):
    '''
    A simple memory object for storing outcomes and retrieving them with a content-addressable similarity metric.
    '''
    def __init__(self, mem_samples, stage_lengths):
        super().__init__()
        self.stage_lengths = stage_lengths
        self.mem = torch.zeros([mem_samples, sum(stage_lengths)])
        #1. start index of memory for each process stage
        #2. start index of memory for new samples to save
        self.mem_s_i = [sum(stage_lengths[:i]) for i in range(len(stage_lengths)+1)]
        self.mem_b_i = 0
        # .. use variance to normalize similarity search
        self.mem_var = torch.zeros_like(self.mem)[0] #(N,)

    def io2v(self, x=None, y=None): #change input and/or output to full memory embedding
        if (x is None) and (y is None): raise ValueError('at least one i/o should be given')
        membed = torch.zeros((x.shape[0], sum(self.stage_lengths)), device=x.device, dtype=x.dtype)
        if (x is not None): membed[:, self.mem_s_i[0]:self.mem_s_i[1]] = x
        if (y is not None): membed[:, self.mem_s_i[-2]:self.mem_s_i[-1]] = y
        return membed

    def forward(self, x): #(Bx,N) input
        '''
        compares input (Bx, N) values to stored memory (By, N) values.
        returns (score, val):
        score : (Bx, B) proximity weighted memory-normalized square distance vector between x and memories.
        val   : (Bx, N) tensor of memories closest to input x.
        '''
        if (self.mem_b_i == 0): raise ValueError('memory is empty')
        mem_ = self.mem[:self.mem_b_i] # use only filled memory cells
        L = ((x[:,None,:] - mem_[None,:,:])/self.mem_var[None,None,:])**2 # (Bx,By,N) variance normalized square distance
        val = mem_[torch.min(L.sum(dim=-1), dim=1, keepdim=False)[1]]     # (Bx,By,N) -argmin-> (Bx->iy,) -> (Bx, N)
        score = (F.softmax(-L.sum(dim=-1), dim=1)[...,None]**2 * L).mean(dim=1) # (Bx,N) # weight by softmax(-score)^2 in memory dimension
        return score, val

    def save(self, x, error=True):
        n = min(x.shape[0], self.mem.shape[0] - self.mem_b_i)
        if n <= 0 and error: raise ValueError('could not write: no memory slots remaining')
        self.mem[self.mem_b_i:self.mem_b_i+n, :] = x[:n]
        # update sample write location & variance
        self.mem_b_i += n
        self.mem_var = torch.var(self.mem[:self.mem_b_i], dim=0, keepdim=False) + 1e-6
        memory_full = self.mem.shape[0] <= self.mem_b_i
        return ~memory_full

    def save_buffer(self):
        self.save(self.mem_buf)
        
    def buffer_stage(self, x, stage:int=0, nb_flag=False):
        n = min(x.shape[0], self.mem.shape[0] - self.mem_b_i)
        if (nb_flag): self.mem_buf = torch.zeros((n, sum(stage_lengths)), self.device, self.dtype)
        s0, s1 = self.mem_s_i[stage], self.mem_s_i[stage+1]
        if (x.shape[1] != s1 - s0):
            raise ValueError('staged entry must be of size (B,N_stage)')
        self.mem_buf[:, s0:s1] = x[:n]

# Hyperbolic Chamfer Distance for Point Cloud Completion and Beyond
# Lin et al. 2024 @ https://arxiv.org/html/2412.17951v1
def hypercd(x, y, alpha=1.0, eps=1e-6):
    # x: (B,N,D)  y: (B,M,D)
    D = torch.cdist(x, y, p=2)**2
    dxy = torch.acosh(1 + alpha * D.min(dim=-1)[0] + eps)
    dyx = torch.acosh(1 + alpha * D.min(dim=-2)[0] + eps)
    return dxy.mean() + dyx.mean()

# Kernel MMD Two-Sample Tests for Manifold Data
# Cheng & Xie 2021 @ https://arxiv.org/abs/2105.03425
def kmmd(x, y, sigma=1.0, k=16):
    # x: (B,N,D)  y: (B,M,D)
    psqd = lambda fx : (fx[0]**2).sum(-1, keepdim=True) + (fx[1]**2).sum(-1, keepdim=True).transpose(1,2) - 2*torch.bmm(fx[0],fx[1].transpose(1,2))
    rbfk = lambda fx : torch.exp(-torch.topk(fx, k=k, dim=-1, largest=False)[0] / (2*sigma**2)).mean()
    return rbfk(psqd((x,x))) + rbfk(psqd((y,y))) - 2*rbfk(psqd((x,y)))

# inspired by : Entropic Open-Set Active Learning
# Safaei te al. 2023 @ https://arxiv.org/abs/2312.14126
#def hou_score(mem_loss, y1, y2):
#    return torch.softmax(mem_loss, dim=0) * kmmd(y1, y2)
# 
# hm. There has to be something here...
# .. hmhmhm. Hm. Homhumhoom. Maybe kmmd is enough? I should try and weight it by mem_loss somehow.
#
#
#

# forward-inverse system
class SimCycler(nn.Module):
    def __init__(self, sim_proc:SimProc, mem_type:str='direct', mem_size:int=1024, stage_depth:int=5):
        super().__init__()
        self.fsim = sim_proc
        self.fnet = nn.ModuleList() # forward surrogate network
        for i in range(len(sim_proc.opers)):
            ixs, oxs = sim_proc.ops[i].idim, sim_proc.ops[i].odim
            self.fnet.append(MLP(ixs, oxs, 2*max(ixs, oxs), stage_depth))
        iys, oys = sim_proc.ops[-1].odim, sim_proc.ops[0].idim
        self.inet = MLP(iys, oys, 2*max(iys,oys), len(sim_proc.opers)*stage_depth//2) #reverse network
        match mem_type: # memory for variably valid samples
            case 'direct': self.mem = Memory_Simple(mem_size, [op.odim for op in sim_proc.ops])
            case 'mhn': raise ValueError('not implemented.')
            case 'mhn.langevin': raise ValueError('not implemented.')
            case _: raise ValueError("memory type must be of 'direct' or 'hopfield'.")

    def simproc(self, x, mem_toggle=True):
        for i in range(len(self.fsim.ops)):
            x = self.fsim.run_stage(x, i)
            if (mem_toggle): self.mem.buffer_stage(x, i, nb_flag=(i == 0))
        if (mem_toggle): self.mem.save_buffer()
        return x
    
    def forward(self, x, mem_toggle=False):
        for i in range(len(self.fnet)):
            x = self.fnet[i](x)
            if (mem_toggle): self.mem.buffer_stage(x, i, nb_flag=(i == 0))
        if (mem_toggle): self.mem.save_buffer()
        return x

    def inverse(self, x):
        x = self.inet(x)
        return x

    def get_unknowns(self, y, x1=None, k=20, pf=0.05):
        if (x1 is None): x1 = self.inverse(y)
        k = min(k, y.shape[0])
        d1, z1 = self.mem(self.mem.io2v(x=x1, y=y)) # change input & output to embedding before use as recall key
        d2, _ = self.mem(z1) # get 2nd traversal energy ~ stable state density on stochastic memory manifold
        d1, d2 = (d1**2).mean(dim=-1), (d2**2).mean(dim=-1) # reduce vector score to total score
        
        score = d1 - d2   # difference of novelty and recall instability
        idx = torch.topk(score, k=k).indices # top-k indexes
        unknowns = x1[idx] + pf/(1+torch.softmax(score[idx], dim=0)) * torch.randn_like(x1[idx]) * x1[idx].std(dim=0, keepdim=True)
        return unknowns
    
    def run_bidirectional_loss(self, x=None, y=None, get_unknowns=True, unkn_k=20):
        loss = torch.zeros(1).to(self.device, self.dtype)
        if (y is None):
            y1 = self.forward(x)
            x1 = self.inverse(y1)
            loss += F.mse_loss(x1, x) # cycle consistency
            loss += self.mem(self.mem.io2v(x=x1,y=y1))[0] # memory loss
            loss += self.mem(self.mem.io2v(x=x,y=y1))[0]
        elif (x is None):
            x1 = self.inverse(y)
            y1 = self.forward(x1)
            loss += F.mse_loss(y1, y) # cycle consistency
            loss += self.mem(self.mem.io2v(x=x1,y=y1))[0].mean() # memory loss
            loss += self.mem(self.mem.io2v(x=x1,y=y))[0].mean()
        else:
            y1 = self.forward(x)
            x1 = self.inverse(y)
            y2 = self.inverse(x1)
            x2 = self.forward(y1)
            loss += F.mse_loss(x, x2) # cycle consistency
            loss += F.mse_loss(y, y2)
            loss += kmmd(y2[None,...], y[None,...]) # cycle consistent kernel supported loss
            loss += kmmd(x2[None,...], x[None,...]) 
            loss += self.mem(self.mem.io2v(x=torch.cat(x1,x,dim=0), y=torch.cat(y1,y,dim=0)))[0]
        if (get_unknowns):
            if (y is not None): unknowns = self.get_unknowns(y, x1, unkn_k)
            else : unknowns = self.get_unknowns(y1, x1, unkn_k)
            return loss, unknowns
        else:
            return loss
        












