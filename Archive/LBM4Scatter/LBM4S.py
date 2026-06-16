import numpy as np
import torch
import torch.nn.functional as F
import sys, os, time

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
