import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

def get_scat_coeff(plane_reader, driver, kI, single_pixel=True, ignore_H=True):
    '''
    Retrieve 2D plane of complex scattering coefficients from LBM simulation results.
    -- inputs
    plane_reader : (B,C,T,Nz,Ny,3:[Fz,Fy,Fx],2:[E,H])  -  recorded electric field at read plane during simulation.
    driver       : (1,1,T,Nz,Ny)  -  source wave should contain phase offsets caused by nonzero incidence angle.
    kI           : (3:[kz,ky,kx])  -  3D lattice wavenumbers of incident wave.
    -- returns
    scatter      : (B,C,Nz,Ny,2:[V,H],2:[E,H]) or (B,C,2[V,H]) if single_pixel and ignore_H are true
    '''
    kcond, vref = torch.tensor([1e-12,0,0], device=kI.device, dtype=kI.dtype), torch.tensor([0,0,1], device=kI.device, dtype=kI.dtype)
    T, kI_n = plane_reader.shape[2], ((kI+kcond)/torch.norm(kI+kcond)) # kcond ensures 0 inc can still be used as hpol plane reference
    # get normalized polarization projection vectors
    hpol = torch.cross(vref, kI_n, dim=-1)
    vpol = torch.cross(kI_n, hpol, dim=-1)
    vpol = (vpol/torch.clamp(torch.norm(vpol),min=1e-12))[None,None,None,None,None,:,None]
    hpol = (hpol/torch.clamp(torch.norm(hpol),min=1e-12))[None,None,None,None,None,:,None]
    scatter = torch.stack([(vpol * plane_reader).sum(dim=-2), (hpol * plane_reader).sum(dim=-2)], dim=-2)
    def s2c(x): # signal amplitude to complex via hilbert transform
        y = -1j * torch.fft.rfft(x, dim=2); y[:,:,0]=0
        y = torch.fft.irfft(y, n=x.shape[2], dim=2) 
        return torch.complex(x, y)
    coeff = s2c(scatter) / s2c(driver)[...,None,None]
    if (ignore_H): coeff = coeff[...,0]
    coeff = coeff.mean(dim=2)
    if (single_pixel): return coeff.mean(dim=(2,3))
    return coeff

def plane_wave_LP(Ez, Ey, Ex, Hz, Hy, Hx, t: int, omega: float, x: int, ymin: int, ymax: int, zmin: int, zmax: int, k_i: torch.Tensor, polarization='H'):
    # V-pol wave - normalized incidence wavenumber set k_i: torch.Tensor([kz, ky, kx])
    val = torch.meshgrid(torch.arange(zmax-zmin, dtype=Ez.dtype, device=Ez.device), torch.arange(ymax-ymin, dtype=Ez.dtype, device=Ez.device), indexing='ij');
    val = torch.sin(k_i[0]*val[0] + k_i[1]*val[1] + omega * t) # point phase offsets .. k_i[2]*val[2] unnecessary : operation does not span x dimension

    up, k_n = torch.tensor([0,0,1], dtype=Ez.dtype, device=Ez.device), k_i / torch.norm(k_i)
    E = torch.cross(up, k_n, dim=-1)
    if (torch.norm(E) < 1e-8): # alternative axis assignment to handle normal incidence degenerate case
        E = torch.cross(torch.tensor([1,0,0],dtype=Ez.dtype,device=Ez.device), k_n)
    if (polarization == 'V'): E = torch.cross(k_n, E, dim=-1)
    elif (polarization == 'H'): pass
    else: raise ValueError('incident polarization must be H or V')
    
    E = val[:,:,None] * (E / torch.clamp(torch.norm(E), min=1e-12))[None,None,None,None,:]
    H = torch.cross((k_i/torch.norm(k_i))[None,None,None,None,:], E, dim=-1)
    
    Ez[...,zmin:zmax, ymin:ymax, x] = E[...,0]
    Ey[...,zmin:zmax, ymin:ymax, x] = E[...,1]
    Ex[...,zmin:zmax, ymin:ymax, x] = E[...,2]
    Hz[...,zmin:zmax, ymin:ymax, x] = H[...,0]
    Hy[...,zmin:zmax, ymin:ymax, x] = H[...,1]
    Hx[...,zmin:zmax, ymin:ymax, x] = H[...,2]

erI, murI = 1.0, 1.0
def simulate(erc, mur=None, wavelength_m=1, cells_per_m=20, pol='H', inc=(0,0),
             n_periods=5, floquet=(0,1), floquet_err=0.05, return_field=False, verbose=True, dtype=torch.float32, device='cuda'):
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
    floquet : dimensions over which to define Floquet boundary condition (unlisted: Dirichlet)
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
    
    # satisfy floquet-bloch condition of forcing wave: kI[i] * er.shape[i] % (2*pi) == 0
    pads = []
    for i in range(2,5):
        Nb, errb = erc.shape[i], 2
        if i-2 in floquet and kI[i-2] != 0:
            for N in range(erc.shape[i], 4*erc.shape[i]):
                err = 0.25*(kI[i-2]*N)*torch.abs(torch.sin(kI[i-2]*N/2))
                if (err < errb): Nb, errb = N, err
                if (err < floquet_err): break
        pads += [0, Nb - erc.shape[i]]
    er = F.pad(erc, tuple(pads[::-1]), mode='circular')

    er = F.pad(er.to(device), (2,0), value=erI) #empty space for incident wave write & scattering wave read
    if (er.dtype.is_complex): er, sigma = er.real.to(dtype), -er.imag.to(dtype) #separate e_r (note: dielectric complex convention is e'-1j*e")
    else: er, sigma = er.to(dtype), torch.zeros_like(er, dtype=dtype)
    sigma  = sigma * omega #convert dielectric loss factor to conductivity via lattice scale angular frequency
    
    B, C, Nz, Ny, Nx = er.shape
    if (mur == None): # note : most dielectric materials have magnetic permeability near free-space value
        mur = torch.full((B, C, Nz, Ny, Nx), 1.0,  dtype=dtype, device=device)
    if (cells_per_m * wavelength_m < 10):
        raise ValueError("Cancelled run : sub-resolution wavelength will cause dispersion error")

    #Time = int(3 * (Nx * torch.min(torch.sqrt(er.real))) + n_periods * period)
    Time = int(3 * (Nx * erI + n_periods * period))
    if (dtype == torch.float16):
        Time = 2**Time.bit_length() # cufft can only handle length powers of 2 when fp16
    
    # Source & read plane location
    xloc_src = 0
    xloc_read = 1
    ymin, ymax = 0, Ny
    zmin, zmax = 0, Nz
    
     #Initialize read tensor : T,Nz,Ny,Nx,(v[x,y,z]),(v[E,H])
    plane_reader = torch.zeros((B, C, Time, Nz, Ny, 3, 2), dtype=dtype, device=device)
    
    # DISTRIBUTION FUNCTIONS  (total field and incident field)
    def _df(Q=7):  return torch.zeros((B, C, Nz, Ny, Nx, Q), dtype=dtype, device=device)
    exI, eyI, ezI, hxI, hyI, hzI   = _df(), _df(), _df(), _df(), _df(), _df()  # incident field
    ex,  ey,  ez,  hx,  hy,  hz    = _df(), _df(), _df(), _df(), _df(), _df()  # full field
    # MACROSCOPIC FIELDS
    def _mf(): return torch.zeros((B, C, Nz, Ny, Nx), dtype=dtype, device=device)
    ExI, EyI, EzI, HxI, HyI, HzI  = _mf(), _mf(), _mf(), _mf(), _mf(), _mf()
    Ex,  Ey,  Ez,  Hx,  Hy,  Hz   = _mf(), _mf(), _mf(), _mf(), _mf(), _mf()
    
    # ===========================================================================
    # MAIN TIME LOOP
    # ===========================================================================
    t0 = time.time()
    
    for t in range(Time):
        # 1.  Macro fields from distribution functions
        ExI = macro_field(exI, erI);  EyI = macro_field(eyI, erI);  EzI = macro_field(ezI, erI)  # free-space propagation
        HxI = macro_field(hxI, murI); HyI = macro_field(hyI, murI); HzI = macro_field(hzI, murI)
        Ex  = macro_field(ex,  er);   Ey  = macro_field(ey,  er);   Ez  = macro_field(ez,  er)   # dense propagation
        Hx  = macro_field(hx,  mur);  Hy  = macro_field(hy,  mur);  Hz  = macro_field(hz,  mur)
        
        # 2.  Inject plane-wave source
        plane_wave_LP(EzI, EyI, ExI, HzI, HyI, HxI, t, omega, xloc_src, ymin, ymax, zmin, zmax, kI, pol)
        plane_wave_LP(Ez,  Ey,  Ex,  Hz,  Hy,  Hx,  t, omega, xloc_src, ymin, ymax, zmin, zmax, kI, pol)

        # read scattered fields
        plane_reader[:,:,t,:,:,0,0] = Ez[...,xloc_read] - EzI[...,xloc_read]
        plane_reader[:,:,t,:,:,0,1] = Hz[...,xloc_read] - HzI[...,xloc_read]
        plane_reader[:,:,t,:,:,1,0] = Ey[...,xloc_read] - EyI[...,xloc_read]
        plane_reader[:,:,t,:,:,1,1] = Hy[...,xloc_read] - HyI[...,xloc_read]
        plane_reader[:,:,t,:,:,2,0] = Ex[...,xloc_read] - ExI[...,xloc_read]
        plane_reader[:,:,t,:,:,2,1] = Hx[...,xloc_read] - HxI[...,xloc_read]
        
        # 3.  Damping    (apply dielectric loss term for real media)
        #   Stable lattice Boltzmann model for Maxwell equations in media (Hauser & Verhey 2017)
        #   src @ https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.063306
        decay = torch.exp(-sigma/er)
        Ex *= decay
        Ey *= decay
        Ez *= decay
        
        # 4.  Collision  (forcing-node variant for both incident & total)
        exI, eyI, ezI, hxI, hyI, hzI = coll_forcing_node(
            exI, eyI, ezI, hxI, hyI, hzI,
            ExI, EyI, EzI, HxI, HyI, HzI,
            erI, murI, xloc_src, ymin, ymax, zmin, zmax
        )
        ex, ey, ez, hx, hy, hz = coll_forcing_node(
            ex,  ey,  ez,  hx,  hy,  hz,
            Ex,  Ey,  Ez,  Hx,  Hy,  Hz,
            er,  mur,  xloc_src, ymin, ymax, zmin, zmax
        )
        
        # 5.  Streaming
        exI, eyI, ezI, hxI, hyI, hzI = streaming(
            exI, eyI, ezI, hxI, hyI, hzI,
            exI, eyI, ezI, hxI, hyI, hzI,
            floquet=floquet
        ) 
        ex, ey, ez, hx, hy, hz = streaming(
            ex,  ey,  ez,  hx,  hy,  hz,
            ex,  ey,  ez,  hx,  hy,  hz,
            floquet=floquet
        )
        
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
    
    if (return_field): return Ex,Ey,Ez,Hx,Hy,Hz
    return plane_reader, driver, kI


# ---------------------------------------------------------------------------
# LBM velocity table  V[q, 3]  (vx, vy, vz)
# ---------------------------------------------------------------------------
velocity = 1.0 / 3  # EM wave speed in vacuum (LBM units)
erI, murI = 1.0, 1.0
Q = 7               # D3Q7 velocity set
V_np = np.array([
    [ 0,  0,  0],
    [ 0,  0,  1], #1 - x+ shift
    [ 0,  0, -1], #2 - x- shift
    [ 0,  1,  0], #3 - y+ shift
    [ 0, -1,  0], #4 - y- shift
    [ 1,  0,  0], #5 - z+ shift
    [-1,  0,  0], #6 - z- shift
], dtype=np.float32)
V_torch = torch.tensor(V_np)

W_lbm = 1.0 / 6.0


def macro_field(dis_func: torch.Tensor, mat_prop: torch.Tensor) -> torch.Tensor:
    """
    Args:
        dis_func  : (B, C, z, y, x, q)  distribution functions
        mat_prop  : (B, C, z, y, x)     material property (epsilon_r or mu_r)
    Returns:
        field     : (B, C, z, y, x)
    """
    return dis_func.sum(dim=-1) / mat_prop


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


def _all_equilibria(Ex, Ey, Ez, Hx, Hy, Hz, er, mur, device):
    V = V_torch.to(device).to(Ex.dtype)
    vx, vy, vz = V[:, 0], V[:, 1], V[:, 2]
    ex_eq = _equilibrium(Ex, Hz, vy, Hy, vz, er)
    ey_eq = _equilibrium(Ey, Hx, vz, Hz, vx, er)
    ez_eq = _equilibrium(Ez, Hy, vx, Hx, vy, er)
    hx_eq = _equilibrium(Hx, Ey, vz, Ez, vy, mur)
    hy_eq = _equilibrium(Hy, Ez, vx, Ex, vz, mur)
    hz_eq = _equilibrium(Hz, Ex, vy, Ey, vx, mur)
    return ex_eq, ey_eq, ez_eq, hx_eq, hy_eq, hz_eq


def coll_forcing_node(ex, ey, ez, hx, hy, hz,
                      Ex, Ey, Ez, Hx, Hy, Hz,
                      er, mur,
                      xloc, ymin, ymax, zmin, zmax):
    """
    Collision step with a soft-source forcing plane.
      - Forcing region  (k==xloc, j∈[ymin,ymax), i∈[zmin,zmax]):  fb = feq
      - Everywhere else:                                            fb = 2*feq - f
    """
    device = ex.device
    B, C, Z, Y, X, _ = ex.shape

    ex_eq, ey_eq, ez_eq, hx_eq, hy_eq, hz_eq = \
        _all_equilibria(Ex, Ey, Ez, Hx, Hy, Hz, er, mur, device)

    # BGK everywhere first
    exb = 2.0 * ex_eq - ex
    eyb = 2.0 * ey_eq - ey
    ezb = 2.0 * ez_eq - ez
    hxb = 2.0 * hx_eq - hx
    hyb = 2.0 * hy_eq - hy
    hzb = 2.0 * hz_eq - hz

    # Build mask (1, 1, Z, Y, X, 1)
    iz = torch.arange(Z, device=device)
    jy = torch.arange(Y, device=device)
    kx = torch.arange(X, device=device)
    mask = ((iz >= zmin) & (iz < zmax))[:, None, None] \
         & ((jy >= ymin) & (jy < ymax))[None, :, None] \
         & (kx == xloc)[None, None, :]
    mask = mask[None,None,...,None]

    exb = torch.where(mask, ex_eq, exb)
    eyb = torch.where(mask, ey_eq, eyb)
    ezb = torch.where(mask, ez_eq, ezb)
    hxb = torch.where(mask, hx_eq, hxb)
    hyb = torch.where(mask, hy_eq, hyb)
    hzb = torch.where(mask, hz_eq, hzb)

    return exb, eyb, ezb, hxb, hyb, hzb


def coll_not_forcing_node(ex, ey, ez, hx, hy, hz,
                          Ex, Ey, Ez, Hx, Hy, Hz,
                          er, mur):
    """Standard BGK collision: fb = 2*feq - f."""
    device = ex.device
    ex_eq, ey_eq, ez_eq, hx_eq, hy_eq, hz_eq = \
        _all_equilibria(Ex, Ey, Ez, Hx, Hy, Hz, er, mur, device)
    return (2.0*ex_eq - ex, 2.0*ey_eq - ey, 2.0*ez_eq - ez,
            2.0*hx_eq - hx, 2.0*hy_eq - hy, 2.0*hz_eq - hz)


def _clamped_roll(f: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    """
    Roll f by `shift` along `dim`, then restore the edge that wrapped around
    so the boundary condition is zero-gradient (clamped / Neumann).
    """
    out = torch.roll(f, shifts=shift, dims=dim)
    idx = [slice(None)] * f.ndim
    idx[dim] = 0 if shift > 0 else -1
    out[tuple(idx)] = f[tuple(idx)]
    return out


def streaming(ex, ey, ez, hx, hy, hz,
              exb, eyb, ezb, hxb, hyb, hzb, floquet=(0, 1)):
    """
    Streaming step: each population l is shifted along its lattice velocity.
    Boundary: clamped (zero-gradient).
    Overwrites ex..hz in-place and returns them.
    """
    def _stream(fb):
        f = torch.empty_like(fb)
        f[..., 0] = fb[..., 0]
        for i in range(1, Q):
            if (2 - int((i - 1) / 2)) in floquet:
                f[..., i] = torch.roll(fb[..., i], dims=-3+(Q-i-1)//2, shifts=+(i%2 * 2 - 1)) #standard circular
            else:                                   #2,2,1,1,0,0     #1,-1,1,-1,1,-1
                f[..., i] = _clamped_roll(fb[..., i], dim=-3+(Q-i-1)//2, shift=+(i%2 * 2 - 1)) #zero-boundary
                                                    #-2,-2,-3,-3,-4,-4 (-1 is Q-dimension)
        return f
    return (_stream(exb), _stream(eyb), _stream(ezb),
            _stream(hxb), _stream(hyb), _stream(hzb))
