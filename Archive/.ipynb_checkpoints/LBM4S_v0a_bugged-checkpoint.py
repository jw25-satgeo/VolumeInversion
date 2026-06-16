import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F


def get_scat_coeff(plane_reader, driver, kI, single_pixel=True, ignore_H=True):
    '''
    Retrieve 2D plane of complex scattering coefficients from LBM simulation results.
    -- inputs
    plane_reader : (B,C,T,Nz,Ny,6:[Ez,Ey,Ex,Hz,Hy,Hx])  -  recorded electric field at read plane during simulation.
    driver       : (1,1,T,Nz,Ny)  -  source wave should contain phase offsets caused by nonzero incidence angle.
    kI           : (3:[kz,ky,kx])  -  3D lattice wavenumbers of incident wave.
    -- returns
    scatter      : (B,C,Nz,Ny,2:[V,H],2:[E,H]) or (B,C,2[V,H]) if single_pixel and ignore_H are true
    '''
    device, dtype = kI.device, kI.dtype
    kcond, vref = torch.tensor([1e-12,0,0]).to(device,dtype), torch.tensor([0,0,1]).to(device,dtype)
    T, kI_n = plane_reader.shape[2], ((kI+kcond)/torch.norm(kI+kcond)) # kcond ensures 0 inc is usable as hpol plane reference
    # get normalized polarization projection vectors
    hpol = torch.cross(vref, kI_n, dim=-1)
    vpol = torch.cross(kI_n, hpol, dim=-1)
    vpol = (vpol/torch.clamp(torch.norm(vpol),min=1e-12))[None,None,None,None,None,:]
    hpol = (hpol/torch.clamp(torch.norm(hpol),min=1e-12))[None,None,None,None,None,:]
    scatter = torch.stack([ # E component dot product (scattering vector projection)
        (vpol * plane_reader[...,:3]).sum(dim=-1),
        (hpol * plane_reader[...,:3]).sum(dim=-1)
    ], dim=-1)
    if not ignore_H:  # add H component dot product (scattering vector projection)
        scatter = torch.stack((scatter, torch.stack([
            (vpol * plane_reader[...,3:]).sum(dim=-1),
            (hpol * plane_reader[...,3:]).sum(dim=-1)
        ], dim=-1)), dim=-1)
    driver_c = s2c(driver)[...,None] if ignore_H else s2c(driver)[...,None,None]
    coeff = s2c(scatter) / driver_c
    coeff = coeff.mean(dim=2) # collapses time dimension
    if (single_pixel): return coeff.mean(dim=(2,3))
    return coeff

def s2c(x): # signal amplitude to complex via hilbert transform
    y = -1j * torch.fft.rfft(x, dim=2); y[:,:,0]=0
    y = torch.fft.irfft(y, n=x.shape[2], dim=2) 
    return torch.complex(x, y)

def plane_wave_LP(EHF, t, omega, zmin, zmax, ymin, ymax, x, k_i, polarization='H'):
    '''
    In-place writes an incident planewave field in forcing plane defined by coordinates.
    EHF  : (B,C,Nz,Ny,Nx,6[Ez,Ey,Ex,Hz,Hy,Hx])
    mask : 
    '''
    device, dtype = EHF.device, EHF.dtype
    xp, zp, k_n = torch.tensor([0,0,1]).to(device,dtype), torch.tensor([1,0,0]).to(device,dtype), k_i / torch.norm(k_i)

    # find E-plane for incident planewave
    E = torch.cross(xp, k_n, dim=-1)
    if (torch.norm(E) < 1e-8): E = torch.cross(zp, k_n, dim=-1)
    if (polarization == 'V'): E = torch.cross(k_n, E, dim=-1) # get orthogonal
    elif (polarization != 'H'): raise ValueError('incident polarization must be H or V')
    
    # sine amplitude w. tilted planewave phase offsets .. k_i[2]*val[2] not included : operation does not span x dimension
    val = torch.meshgrid(torch.arange(zmax-zmin).to(device,dtype), torch.arange(ymax-ymin).to(device,dtype), indexing='ij')
    val = torch.sin(k_i[0]*val[0] + k_i[1]*val[1] + omega * t)[None,None,:,:,None]

    E = val * (E / torch.clamp(torch.norm(E), min=1e-12))[None,None,None,None,:]
    H = torch.cross(k_n[None,None,None,None,:], E, dim=-1)

    EHF[...,zmin:zmax,ymin:ymax,x,:3] = E
    EHF[...,zmin:zmax,ymin:ymax,x,3:] = H
    
def simulate(erc, mur=None, wavelength_m=1, cells_per_m=20, pol='H', inc=(0,0),
             n_periods=5, periodic=(0,1), periodic_err=0.05, return_field=False, verbose=True, dtype=torch.float32, device='cuda'):
    '''
    Simulate wave propagation through complex dielectric media using Lattice Boltzmann.
    -- inputs
    er : dielectric permittivity 3D tensor (B, C, Nz, Ny, Nx)
    mur : magnetic permeability (1.0 by default)
    wavelength_m : wavelength in metres
    cells_per_m  : simulation resolution parameter
    pol : polarization of incident LP wave ('TE' (xy-plane), 'TM' (xz-plane))
    inc : source incidence angle in radians (theta : xz-plane , phi : xy-plane) .. Do not use angles greater than π/2.
    n_periods : number of wave periods to simulate over
    n_reflections : number of maximum-thickness traversal times to simulate over
    periodic : dimensions over which to define Floquet boundary condition (unlisted: Dirichlet)
    '''
    
    wavelength = wavelength_m * cells_per_m
    k      = 2 * np.pi / wavelength
    omega  = velocity * k
    period = int(np.round(wavelength / velocity))
    # incident wavenumbers
    kI = (2*torch.pi/wavelength) * torch.tensor([
        np.sin(inc[0])*np.cos(inc[1]), # kz
        np.sin(inc[1]),                # ky
        np.cos(inc[0])*np.cos(inc[1])  # kx
    ]).to(dtype).to(device)
    erI  = torch.ones((1,1,1,1,1), dtype=dtype, device=device)
    murI = torch.ones((1,1,1,1,1), dtype=dtype, device=device)
    
    # heuristically check floquet-bloch validity of incident wave: kI[i] * er.shape[i] % (2*pi) == 0
    for i in range(2,5):
        if (i-2) in periodic and kI[i-2] != 0:
            err = 0.25*(kI[i-2]*erc.shape[i])*torch.abs(torch.sin(kI[i-2]*erc.shape[i]/2))
            if (err > periodic_err): print (f'axis {i-2} cell periodicity error above minimum at {err}')

    erc = F.pad(erc.to(device), (2,0), value=erI.mean()) #empty space for incident wave write & scattering wave read
    if (erc.dtype.is_complex): er, sigma = erc.real.to(dtype), -erc.imag.to(dtype) #separate e_r (note: dielectric complex convention is e'-1j*e")
    else: er, sigma = erc.to(dtype), torch.zeros_like(erc, dtype=dtype)
    sigma  = sigma * omega #convert dielectric loss factor to conductivity via lattice scale angular frequency
    
    B, C, Nz, Ny, Nx = er.shape
    if (mur == None): # note : most dielectric materials have magnetic permeability near free-space value
        mur = torch.full((B, C, Nz, Ny, Nx), 1.0,  dtype=dtype, device=device)
    if (cells_per_m * wavelength_m < 10):
        raise ValueError("Cancelled run : sub-resolution wavelength will cause dispersion error")

    Time = int(3 * (Nx * erI + n_periods * period))
    if (dtype == torch.float16):
        Time = 2**Time.bit_length() # cufft can only handle length powers of 2 when fp16
    
    # Source & read plane location
    xloc_src = 0
    xloc_read = 1
    ymin, ymax = 0, Ny
    zmin, zmax = 0, Nz
    
     #Initialize read tensor : T,Nz,Ny,Nx,(v[x,y,z]),(v[E,H])
    plane_reader = torch.zeros((B, C, Time, Nz, Ny, 6), dtype=dtype, device=device)
    
    # DISTRIBUTION FUNCTIONS  (total field and incident field)
    ehfI = torch.zeros((B, C, Nz, Ny, Nx, 6, Q), dtype=dtype, device=device) # incident field
    ehf  = torch.zeros((B, C, Nz, Ny, Nx, 6, Q), dtype=dtype, device=device) # full field
    # MACROSCOPIC FIELDS
    EHFI = torch.zeros((B, C, Nz, Ny, Nx, 6), dtype=dtype, device=device)
    EHF  = torch.zeros((B, C, Nz, Ny, Nx, 6), dtype=dtype, device=device)

    # reusable solver arrays
    solver_reuse = torch.zeros((B,C,Nz,Ny,Nx,6,Q), dtype=dtype, device=device)
    # forcing mask for collision step
    forcing_mask = coll_forcing_mask(ehf, xloc_src, ymin, ymax, zmin, zmax)
    
    # ===========================================================================
    # MAIN TIME LOOP
    # ===========================================================================
    t0 = time.time()

    
    for t in range(Time):
        # 1.  Macro fields from distribution functions
        EHFI = macro_field(ehfI, erI, murI)
        EHF  = macro_field(ehf, er, mur)
        
        # 2.  Inject plane-wave source
        plane_wave_LP(EHFI, t, omega, zmin, zmax, ymin, ymax, xloc_src, kI, pol)
        plane_wave_LP(EHF,  t, omega, zmin, zmax, ymin, ymax, xloc_src, kI, pol)

        # read scattered fields
        plane_reader[:,:,t,:,:,:] = EHF[...,xloc_read,:] - EHFI[...,xloc_read,:]
        
        # 3.  Damping    (apply dielectric loss term for real media)
        #   Stable lattice Boltzmann model for Maxwell equations in media (Hauser & Verhey 2017)
        #   src @ https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.063306
        decay = torch.exp(-sigma/er)
        EHF[...,:3] *= decay[...,None]
        
        # 4.  Collision  (forcing-node variant for both incident & total)
        ehfI = coll_node(ehfI, EHFI, erI, murI, solver_reuse, forcing_mask)
        ehf  = coll_node(ehf,  EHF,  er,  mur,  solver_reuse, forcing_mask)
        
        # 5.  Streaming
        ehfI = streaming(ehfI, solver_reuse, periodic=periodic)
        ehf  = streaming(ehf , solver_reuse, periodic=periodic)

        #print(solver_reuse.sum(), 'bbb')
        
        # 8.  Progress report
        t2 = time.time()
        if verbose and t > 0 and t % 100 == 0:
            remaining = (t2 - t0) * (int(Time) - t) / (t * 60)
            print(f"Approx. time left: {remaining:.2f} min", end="\r")
    if verbose:
        print((time.time() - t0) / 60)

    # copied from planewave implementation
    driver = torch.meshgrid(torch.arange(zmax-zmin,dtype=dtype,device=device),torch.arange(ymax-ymin,dtype=dtype,device=device),indexing='ij');
    driver = torch.sin((kI[0]*driver[0] + kI[1]*driver[1])[None,...] + omega*torch.arange(Time,dtype=dtype,device=device)[:,None,None])
    driver = driver[None,None,...] #extend to B,C dimensions
    
    if (return_field): return EHF
    return plane_reader, driver, kI


# ---------------------------------------------------------------------------
# LBM velocity table  V[q, 3]  (vx, vy, vz)
# ---------------------------------------------------------------------------
velocity = 1.0 / 3  # EM wave speed in vacuum (LBM units)
Q = 7               # D3Q7 velocity set
W_lbm = 1.0 / 6.0


def macro_field(dis_func, er_, mur_) -> torch.Tensor:
    """
    Args:
        dis_func  : (B, C, z, y, x, 6, q)  distribution functions
        er        : (B, C, z, y, x)     material property
        mur       : (B, C, z, y, x)     material property
    Returns:
        field     : (B, C, z, y, x, 6)
    """
    tmp_EHF = dis_func.sum(dim=-1)
    tmp_EHF[...,:3] /= er_[...,None]
    tmp_EHF[...,3:] /= mur_[...,None]
    return tmp_EHF

def _equilibrium(EHFi, ehf_eq, er_, mur_, device):
    '''
    EHF: (B,C,z,y,x,6[Ez,Ey,Ex,Hz,Hy,Hx]) -> ehf:(B,C,z,y,x,6[Ez,Ey,Ex,Hz,Hy,Hx],q)
    via ehf_eq = W(F - (v x G) in z,y,x ordering
    '''

    ehf_eq[:] = 0
    
    ehf_eq[...,:3, 0] = ((er_  - 1)[...,None] * EHFi[...,:3])
    ehf_eq[...,3:, 0] = ((mur_ - 1)[...,None] * EHFi[...,3:])
    
    ehf_eq[...,0,1] = EHFi[...,0] - EHFi[...,3]
    ehf_eq[...,0,2] = EHFi[...,0] + EHFi[...,3]
    ehf_eq[...,0,3] = EHFi[...,0] + EHFi[...,4]
    ehf_eq[...,0,4] = EHFi[...,0] - EHFi[...,4]
    
    ehf_eq[...,1,1] = EHFi[...,1] - EHFi[...,5]
    ehf_eq[...,1,2] = EHFi[...,1] + EHFi[...,5]
    ehf_eq[...,1,5] = EHFi[...,1] + EHFi[...,3]
    ehf_eq[...,1,6] = EHFi[...,1] - EHFi[...,3]

    ehf_eq[...,2,5] = EHFi[...,2] - EHFi[...,4]
    ehf_eq[...,2,6] = EHFi[...,2] + EHFi[...,4]
    ehf_eq[...,2,3] = EHFi[...,2] + EHFi[...,5]
    ehf_eq[...,2,4] = EHFi[...,2] - EHFi[...,5]

    ehf_eq[...,3,1] = EHFi[...,3] - EHFi[...,1]
    ehf_eq[...,3,2] = EHFi[...,3] + EHFi[...,1]
    ehf_eq[...,3,3] = EHFi[...,3] + EHFi[...,2]
    ehf_eq[...,3,4] = EHFi[...,3] - EHFi[...,2]

    ehf_eq[...,4,5] = EHFi[...,4] - EHFi[...,0]
    ehf_eq[...,4,6] = EHFi[...,4] + EHFi[...,0]
    ehf_eq[...,4,1] = EHFi[...,4] + EHFi[...,2]
    ehf_eq[...,4,2] = EHFi[...,4] - EHFi[...,2]

    ehf_eq[...,5,3] = EHFi[...,5] - EHFi[...,2]
    ehf_eq[...,5,4] = EHFi[...,5] + EHFi[...,2]
    ehf_eq[...,5,5] = EHFi[...,5] + EHFi[...,1]
    ehf_eq[...,5,6] = EHFi[...,5] - EHFi[...,1]

    ehf_eq[...,:,1:] *= W_lbm

    return ehf_eq
    #ehf_eq[...,:3,1:] = W_lbm * (EHF[...,:3,None] - V[...,:,1:]*EHF[...,3:,None].permute(0,1,2,3,4,1,5))
    #ehf_eq[...,3:,1:] = W_lbm * (EHF[...,3:,None] + V[...,:,1:]*EHF[...,:3,None].permute(0,1,2,3,4,1,5))

def _equilibrium(F, C1, Vb, C2, Vc, mat):
    """
    Vectorised equilibrium for one field component.
      feq[..., 0] = (mat-1)*F
      feq[..., l] = W*(F - Vb[l]*C1 + Vc[l]*C2)  for l=1..6
    All spatial tensors: (B, C, Z, Y, X).  Vb, Vc: (Q,) velocity component vectors.
    Returns feq: (B, C, Z, Y, X, Q).
    """
    rest = ((mat - 1.0) * F).unsqueeze(-1)                        # (B,C,Z,Y,X,1)
    Vb_  = Vb.view(1,1,1,1,1,Q)
    Vc_  = Vc.view(1,1,1,1,1,Q)
    prop = W_lbm * (F.unsqueeze(-1)
                    - Vb_ * C1.unsqueeze(-1)
                    + Vc_ * C2.unsqueeze(-1))                      # (B,C,Z,Y,X,Q)
    return torch.cat([rest, prop[..., 1:]], dim=-1)
def _all_equilibria(EHFi, er, mur, device):
    Ez,Ey,Ex,Hz,Hy,Hx = [EHFi[...,i] for i in range(6)]
    V_np = np.array([
        [ 0,  0,  0],
        [ 0,  0,  1], #1 - x+ shift
        [ 0,  0, -1], #2 - x- shift
        [ 0,  1,  0], #3 - y+ shift
        [ 0, -1,  0], #4 - y- shift
        [ 1,  0,  0], #5 - z+ shift
        [-1,  0,  0], #6 - z- shift
    ], dtype=np.float32)
    V = torch.as_tensor(V_np, device=device, dtype=Ez.dtype)
    vx, vy, vz = V[:, 0], V[:, 1], V[:, 2]
    ex_eq = _equilibrium(Ex, Hz, vy, Hy, vz, er)
    ey_eq = _equilibrium(Ey, Hx, vz, Hz, vx, er)
    ez_eq = _equilibrium(Ez, Hy, vx, Hx, vy, er)
    hx_eq = _equilibrium(Hx, Ey, vz, Ez, vy, mur)
    hy_eq = _equilibrium(Hy, Ez, vx, Ex, vz, mur)
    hz_eq = _equilibrium(Hz, Ex, vy, Ey, vx, mur)
    ehfo = torch.stack((ez_eq,ey_eq,ex_eq,hz_eq,hy_eq,hx_eq),dim=-2)
    return ehfo
    

def coll_forcing_mask(ehf, xloc, ymin, ymax, zmin, zmax):
    """
    Mask for collision step with a soft-source forcing plane.
    : Forcing region  (k==xloc, j∈[ymin,ymax), i∈[zmin,zmax])
    """
    device = ehf.device
    B, C, Z, Y, X, F, _ = ehf.shape
    # Build mask (1, 1, Z, Y, X, 1, 1)
    iz = torch.arange(Z, device=device)
    jy = torch.arange(Y, device=device)
    kx = torch.arange(X, device=device)
    mask = ((iz >= zmin) & (iz < zmax))[:, None, None] & ((jy >= ymin) & (jy < ymax))[None, :, None] & (kx == xloc)[None, None, :]
    mask = mask[None,None,...,None,None]
    return mask

def coll_node(ehfi, EHFi, er_, mur_, ehf_eq, mask=None):
    """
    BGK collision step with an optional soft-source forcing region defined by a mask.
      - Forcing region :  fb = feq
      - Everywhere else:  fb = 2*feq - f
    """
    ehf_eq = _all_equilibria(EHFi, er_, mur_, ehfi.device)
    ehfb = 2.0 * ehf_eq - ehfi
    #print((ehf_eq - ehfi).sum())
    if (mask is not None): ehfb = torch.where(mask, ehf_eq, ehfb)
    return ehfb

def streaming(ehfi, ehfb, periodic=(0,1)):
    '''
    Shifts each Q-component along its lattice velocity.
    input : ehf (B,C,z,y,x,6[ez,ey,ex,hz,hy,hx],q)
    '''
    # velocity vector: (0, +1z,-1z,+1y,-1y,+1x,-1x)
    ehfb[...,0] = ehfi[...,0]
    for i in range(1,Q): # each direction Q must be streamed according to velocity
        dim, shift = (i - 1)//2, ((i%2)*2 - 1)
        i0, L = (shift + 1)//2, ehfi.size(2 + dim)
        if dim in periodic: saved_edge = ehfi[...,i].select(2+dim, -i0).clone()  #periodic : wrap missing edge
        #else:               saved_edge = 0                                      #dirichlet : zero-fill missing edge
        else:               saved_edge = ehfi[...,i].select(2+dim,i0-1).clone()  #clamped  : zerograd missing edge
        ehfb[...,i].narrow(2+dim, i0, L-1).copy_(ehfi[...,i].narrow(2+dim, 1-i0, L-1)) #1-shift
        ehfb[...,i].select(2+dim, i0-1).copy_(saved_edge)  # apply boundary condition
    return ehfb.clone()

'''
def _clamped_roll(f: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    out = torch.roll(f, shifts=shift, dims=dim)
    idx = [slice(None)] * f.ndim
    idx[dim] = 0 if shift > 0 else -1
    out[tuple(idx)] = f[tuple(idx)]
    return out
def streaming(ehf, ehfb, periodic=(0, 1)):
    def _stream(fb):
        f = torch.empty_like(fb)
        f[..., 0] = fb[..., 0]
        for i in range(1, Q):
            if (2 - int((i - 1) / 2)) in periodic:
                f[..., i] = torch.roll(fb[..., i], dims=-3+(Q-i-1)//2, shifts=+(i%2 * 2 - 1)) #standard circular
            else:                                   #2,2,1,1,0,0     #1,-1,1,-1,1,-1
                f[..., i] = _clamped_roll(fb[..., i], dim=-3+(Q-i-1)//2, shift=+(i%2 * 2 - 1)) #zero-boundary
                                                    #-2,-2,-3,-3,-4,-4 (-1 is Q-dimension)
        return f
    for i in range(ehfb.shape[-2]):
        ehfb[...,i,:] = _stream(ehf[...,i,:])
    return ehfb
'''






