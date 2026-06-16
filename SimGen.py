import os
import torch
import torch.nn.functional as F

class ReflectorSim(Dense2.SimProc):
    '''
    A simulation process stack for reflectivity value generation.
    input (as flattened tensor):
    T  : N,(D)
    S  : N,(D)
    rho: N,(D)
    ia - N,(C,2)
    output (tuple):
    refl_coeff N,(C,4)
    refl_coeff_v2 N,(C,4)
    '''
    def __init__(self, L, W, D, C, cells_per_m, wavelength, anneal_iters=0, anneal_window=(11,11,3), batch_size=4):
        operations = [
            LatticeSimOp(L, W, D, C, wavelength, anneal_iters, anneal_window),
            ReflSimOp(L, W, D, C, cells_per_m, wavelength, batch_size),
            ReflCorrectorOp(C)
        ]
        super().__init__(operations)

class ReflCorrectorOp(Dense2.SimOp):
    '''
    A tail-end complex-valued multiplicative correction module using a simple neural network.
    input:
    refl_coeff : N,(C,4)
    floquet_invalidity : N,(C)
    ia : N,(C,2)
    output:
    refl_coeff : N,(C,4)
    refl_coeff_v2 : N,(C,4)
    '''
    def __init__(self, C, eps=0.001):
        ishape = [(C,4), (C,), (C,2)]
        oshape = [(C,4), (C,4)]
        super().__init__(ishape, oshape)
        self.combinator = Dense2.MLP(3, 8, 12, 5)
        self.eps = eps

    def run_model(self, refl_coeff, floq_err, ia):
        N,C,_ = ia.shape
        x = self.combinator(torch.cat((ia.view(-1,2), floq_err.view(-1,1)),dim=-1)).view(N,C,4,2)
        y = torch.sigmoid(torch.abs(refl_coeff) + (floq_err[...,None] + self.eps)*x[...,0])
        y *= torch.exp(1j * (torch.angle(refl_coeff) + (floq_err[...,None] + self.eps)*x[...,1]))
        return refl_coeff, y
        

import LBM4S_v0 as LBM4S
class ReflSimOp(Dense2.SimOp):
    '''
    A compiled module of the reflectivity lattice simulator.
    input:
    er : N,(C,L,W,D)
    ia : N,(C,2)
    output:
    refl_coeff : N,(C,4)
    floquet_invalidity : N,(C)
    ia : N,(C,2)
    '''
    def __init__(self, L, W, D, C, cells_per_m, wavelength, batch_size=4, periodic=(0,1)):
        self.batch_size = batch_size
        self.cells_per_m = cells_per_m
        self.wavelength_m = wavelength
        self.n_periods = 2
        
        ishape = [(C,L,W,D), (C,2)]
        oshape = [(C,4), (C,), (C,2)]
        super().__init__(ishape, oshape)
        
        # calculate wavevector incidences matched to periodic Floquet-Bloch cell boudnary
        v = 1 / (self.wavelength_m * self.cells_per_m)
        N = int(4 * max(L,W,D) * v)                       # integer Floquet harmonics over periodic axes
        harmonics = np.stack(np.meshgrid(*[np.arange(-N, N+1) for _ in periodic],indexing='ij'),axis=-1)
        v_set = np.zeros((*harmonics.shape[:-1], 3))
        for i, ax in enumerate(periodic): # assign periodic components
            v_set[..., ax] = harmonics[..., i] / (L,W,D)[ax]
        v_rem = v**2 - np.sum(v_set[..., periodic]**2, axis=-1)
        v_set[..., [ax for ax in (0,1,2) if ax not in periodic][0]] = np.sqrt(np.maximum(v_rem, 0)) # set propagation direction behavior
        v_set = v_set[v_rem > 0] # flatten propagating modes  ..  # incidence convention [kz, ky, kx]:(theta,phi)
        ia_periodic = np.stack([np.arctan2(v_set[:,0], v_set[:,2]), np.arcsin(v_set[:,1] / v)], axis=-1).swapaxes(0,1) # 2,n
        self.register_buffer('ia_periodic', torch.as_tensor(ia_periodic)[None,None,:,:])

    def run_model(self, er, ia):
        '''
        er N,(C,L,W,D), ia -> refl N,(C,4), floquet_validity N,(1,), ia N,(1,),
        '''
        (N,C,L,W,D), batch_iters = er.shape, math.ceil(er.shape[0] / self.batch_size)
        result = torch.zeros((N,C,4), device=self.device, dtype=er.dtype)
        for j in range(batch_iters):
            j0, j1 = j*self.batch_size, (j+1)*self.batch_size
            result[j0:j1,...,(1,0)] = LBM4S.get_scat_coeff(*LBM4S.simulate(
                er[j0:j1], pol='H', wavelength_m=self.wavelength_m, cells_per_m=self.cells_per_m, n_periods=self.n_periods, inc=ia[j0:j1]
            ), single_pixel=True, ignore_H=True)
        for j in range(batch_iters):
            j0, j1 = j*self.batch_size, (j+1)*self.batch_size
            result[j0:j1,...,(3,2)] = LBM4S.get_scat_coeff(*LBM4S.simulate( 
                er[j0:j1], pol='V', wavelength_m=self.wavelength_m, cells_per_m=self.cells_per_m, n_periods=self.n_periods, inc=ia[j0:j1]
            ), single_pixel=True, ignore_H=True)
        floquet_invalidity = ((self.ia_periodic - ia[...,None])**2).sum(dim=-2).min(dim=-1, keepdim=True)[0]
        return result, floquet_invalidity, ia

import brack2bing
class LatticeSimOp(Dense2.SimOp):
    '''
    A compiled module form of the icepack lattice forming simulator.
    input:
    T  : N,(D)
    S  : N,(D)
    rho: N,(D)
    ia - N,(C,2)
    output:
    er - N,(C,L,W,D)
    ia - N,(C,2)
    '''
    def __init__(self, L, W, D, C, wavelength, anneal_iters=0, anneal_window=(15,15,3), solver_cache='brack2bing_solver_cache'):
        ishape = [(D,),(D,),(D,),(C,2)]
        oshape = [(C,L,W,D), (C,2)]
        super().__init__(ishape, oshape)
        if (solver_cache is not None):
            if os.path.exists(os.path.join(solver_cache, 'init.pickle')):
                self.solver = brack2bing.LatticeSolver.load(cache_dir=solver_cache)
            else:
                self.solver = brack2bing.LatticeSolver()
                self.solver.save(cache_dir=solver_cache)
        else:
            self.solver = brack2bing.LatticeSolver()
        self.solve_iters = anneal_iters
        self.solve_window = anneal_window
        self.wavelength = wavelength

    def run_model(self, T, S, rho, ia):
        x = self.solver.gen_lattice(T, S, rho)
        for i in range(self.solve_iters):
            x = self.solver.anneal_step(
                x,
                T[:,None,None,None,:],
                S[:,None,None,None,:],
                self.solve_window
            )
        x = solver.get_dielectric(x, T, self.wavelength)
        return x, ia
















