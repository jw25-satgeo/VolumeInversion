import numpy as np
import torch
import torch.nn.functional as F
import os, pickle

from PhreeqcSolver import frezchem_seawater

c = 299792458 #m/s (SI)
pe_0 = 8.8541878188e-12 #vacuum permittivity F/m

class LatticeSolver(torch.nn.Module):
    default_label = "brack2bing_solver"
            
    def __init__(self, dtype=torch.float32, device='cpu'):
        super().__init__()
        #self.percol = torch.nn.parameter.Parameter(torch.zeros((3,3,3,channels),dtype=dtype,device=device))
        self.dtype = dtype
        self.device = device
        Tmin, Tmax, Tc, Smin, Smax, Sc = -30, 20, 50, 0.1, 300.1, 100
        self.register_buffer('ts_grid', torch.as_tensor(
            frezchem_seawater().create_table( #  Tlen, Slen, 2 (iceFrac, brinSal)
                TminC = Tmin,   TmaxC = Tmax,   Tc = Tc,
                Smin  = Smin,   Smax  = Smax,   Sc = Sc,
                logError=False
            ), dtype=dtype, device=device
        ).moveaxis(-1, 0))  # (iceFrac, brinSal) 2, Tlen, Slen
        self.ts_grid_d3base = self.ts_grid[:,None,None,:,:,None] #2,1(N),1(C),Tlen,Slen,1(none)
        self.register_buffer('t_idx', torch.linspace(Tmin,Tmax,Tc,dtype=dtype,device=device))
        self.register_buffer('s_idx', torch.linspace(Smin,Smax,Sc,dtype=dtype,device=device))
        self.register_buffer('shift_ops', torch.as_tensor( # for rapid coordinate shift setting
            [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]
            ], dtype=torch.int16, device=device
        )) #7, 3
        
    def TSnorm(self, T, S): # rescale to [-1,1]
        Tn = ((T - self.t_idx[0]) / (self.t_idx[-1] - self.t_idx[0])) * 2 - 1
        Sn = ((S - self.s_idx[0]) / (self.s_idx[-1] - self.s_idx[0])) * 2 - 1
        return Tn, Sn

    @staticmethod
    def load(cache_dir=None):
        if (cache_dir is None): cache_dir = f'{LatticeSolver.default_label}_cache'
        fn = os.path.join(cache_dir, 'init.pickle')
        with open(fn, 'rb') as f: return pickle.load(f)

    def save(self, cache_dir=None):
        if (cache_dir is None): cache_dir = f'{LatticeSolver.default_label}_cache'
        with open(os.path.join(cache_dir, 'init.pickle'), 'wb') as f:
            pickle.dump(self, f)

    def gen_lattice(self, T, S, rho=None, hshape=(50,50), num_channels=1):
        '''
        Initializes lattice grids for scaled sets of temperature and salinity depth profiles.
        
        inputs (Tensor)
        shape : (length, width, depth)
        T, S : (N,D) depth profile
        rho : (N,D) depth profile or (N,1) bulk density

        output : (N,C,L+2,W+2,D+2)

        positive values : brine salinity
        negative values : air pressure
        zero values : ice

        boundary values:
        -999 : top boundary (zero depth)
        -998 : bottom boundary
        -997 : horizontal boundary
        '''
        N,C,L,W,D = T.shape[0], num_channels, hshape[0], hshape[1], T.shape[1]
        
        rho = 0.9 * torch.ones((N,1,1,1,D),dtype=self.dtype,device=self.device) if (rho is None) else rho
        rho = rho[:,None,None,None,:] if (rho.size(-1)==1) else rho
        Tn, Sn = self.TSnorm(T,S)
        
        grid = torch.zeros((N,C,L,W,D), dtype=self.dtype, device=self.device)

        ts_sample = torch.stack((Sn[:,:,None], Tn[:,:,None]), dim=-1) #N,D,1,2  note : grid_sample coordinates are xyz ordered, reverse of ijk
        ts_output = F.grid_sample(self.ts_grid[None,...].repeat(N,1,1,1), ts_sample, padding_mode='border', align_corners=False) #N,2,?,? to N,2,D,1

        if_, bs_ = ts_output[:,0,None,None,None,:,0], ts_output[:,1,None,None,None,:,0] # N,2,D,1 to N,1,1,1,D
        
        grid[torch.rand_like(grid) > if_] = 1 #random value above ice fraction is brine
        grid = grid * bs_  # multiply all values by brine salinity (doesn't affect zero values)

        # air fraction : 1 - actual density/(density of pure ice + density of brine w. salinity in g/kg)
        af_ = 1 - rho / (0.9167 * (grid==0).to(self.dtype).mean(dim=[2,3],keepdim=True) + ((grid > 0) + F.relu(grid)/1000).mean(dim=[2,3],keepdim=True))
        af_ *= (if_ > 0.3)  # only regions with decent ice fraction will have air bubbles
        grid[torch.rand_like(grid) < af_] = -1 #random value below air fraction is air

        # pad grid in all directions for boundary access
        grid = F.pad(F.pad(F.pad(grid, (0,0,0,0,1,0), mode='constant', value=-999),
                           (0,0,0,0,0,1), mode='constant', value=-998),
                     (1,1,1,1,0,0), mode='constant', value=-997)
        return grid

    def get_dielectric(self, x, T, wavelength):
        '''
        x: (N,C,L+2,W+2,D+2)
        T: (N,D)
        wavelength: scalar
        '''
        ox = x[:,:,1:-1,1:-1,1:-1]
        er = torch.zeros_like(ox).to(torch.complex128)
        freq_GHz = 1e-9*c/wavelength
        S = torch.clamp(ox.clone(), min=1, max=200)
        T = T[:,None,None,None,:]
        
        # A model for the complex permittivity of ice at frequencies below 1 THz 
        #  (Hufford, 1991) - https://link.springer.com/article/10.1007/bf01008898
        a = 300/(273.15+T) - 1;  a = (50.4 + 62*a)*1e-4*torch.exp(-22.1*a)
        b = (1e-4*(0.445+0.00211*T) + 0.585e-4/torch.clamp((1 - T/29.1)**2, min=1e-8))
        er += (ox == 0) * (3.15 - 1j * (a/freq_GHz + b*freq_GHz).to(torch.complex128))
        
        # High Salinity Permittivity Models for Water Cut Sensing
        # (Black et al. 2013) - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6553153
        T_eff = T - 0.06*S
        # The Complex Dielectric Constant of Pure and Sea Water From Microwave Satellite Observations
        # (Meissner & Wentz 2004) - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1333169
        pe_0_GHz = 17.97510  #vacuum permittivity GHz*m/S
        sw_cond_s35 = 2.903602 + 8.607e-2*T_eff+ 4.738817e-4*T_eff**2 - 2.991e-6*T_eff**3 + 4.3047e-9*T_eff**4
        R15 = S * (37.5109 + 5.45216*S + 1.4409e-2*S**2)/(1004.75 + 182.283*S + S**2)
        RTdR15 = 1 + ((6.9431 + 3.2841*S - 9.9486e-2*S**2)/(84.850 + 69.024*S + S**2))*(T_eff-15) / ((49.843 - 0.2276*S + 0.198e-2*S**2) + T_eff)
        sw_cond = sw_cond_s35 * R15 * RTdR15
        a = [5.7230e+0, 2.2379e-2, -7.1237e-4, 5.0478e+1, -7.0315e-2, 6.0059e-4, 3.6143e+0, 2.8841e-2, 1.3652e-1, 1.4825e-3, 2.4166e-4]
        b = [-3.56417e-3, 4.74868e-6, 1.15574e-5, 2.39357e-3, -3.1353e-5, 2.52477e-7, -6.28908e-3, 1.76032e-4, -9.22144e-5, -1.99723e-2, 1.81176e-4, -2.04265e-3, 1.57883e-4]
        # pure water parameters
        eS = (3.70886e+4 - 8.2168e+1*T_eff)/(4.21854e+2 + T_eff)
        e1 = a[0] + a[1]*T_eff + a[2]*T_eff**2
        v1 = (45 + T_eff) / (a[3] + a[4]*T_eff + a[5]*T_eff**2)
        einf = a[6] + a[7]*T_eff
        v2 = (45 + T_eff) / (a[8] + a[9]*T_eff + a[10]*T_eff**2)
        # convert to brine parameters
        eS *= torch.exp(b[0]*S + b[1]*S**2 + b[2]*T_eff*S)
        v1 *= 1 + S * (b[3] + b[4]*T_eff + b[5]*T_eff**2)
        e1 *= torch.exp(b[6]*S + b[7]*S**2 + b[8]*T_eff*S)
        v2 *= 1 + S * (b[9] + b[10]*T_eff)
        einf *= 1 + S * (b[11] + b[12]*T_eff)
        # apply domain stabilization (bugfix)
        e1 = torch.minimum(e1, eS)
        einf = torch.minimum(einf, e1)
        v1 = torch.clamp(v1, min=1e-3)
        v2 = torch.clamp(v2, min=1e-3)
        # implement double Debye relaxation model
        er += (ox > 0) * ((eS - e1)/(1 + 1j*freq_GHz/v1) + (e1 - einf)/(1 + 1j*freq_GHz/v2) + einf - 1j*sw_cond/(2*torch.pi*pe_0_GHz)*freq_GHz)
        if not torch.all(er.imag <= 0): raise RuntimeError("Nonphysical value: positive dielectric loss")
        er[er.real < 1] = 1
        
        return er.to((torch.zeros(1,dtype=self.dtype)+0j).dtype)

    def TSF(self, ox, T, S, ts_samp=(15,15,3)):
        '''
        1. Calculates difference between current salinity and reference salinity.
        2. Calculates difference between brine fraction of current T-S equilibrium and current brine fraction. 
        input
        ox : (N,C,Li,Wi,Di) float tensor
        T  : (N,1,1,1,Di) float tensor
        S  : (N,1,1,1,Di) float tensor
        output
        tuple( (N,C,Li,Wi,Di), (N,C,Li,Wi,Di) ) of 0~1 float tensors
        '''
        N,C,Li,Wi,Di = ox.shape
        dtype,device = ox.dtype, ox.device
        ts_pad = tuple([ts_samp[i//2]//2 for i in range(5,-1,-1)])
        
        xS = ox.clone(); xS[xS < 0] = 0
        xS = F.avg_pool3d(F.pad(xS, ts_pad, mode='reflect'), ts_samp) # mean downsampled salinity
        yS = F.avg_pool3d(F.pad(S, ts_pad[:2], mode='reflect'), (1,1,ts_samp[-1]))
        yT = F.avg_pool3d(F.pad(T, ts_pad[:2], mode='reflect'), (1,1,ts_samp[-1]))
        Lj,Wj,Dj = xS.shape[2:]
        Tn, Sn = self.TSnorm(yT, xS)
        
        ts_input = torch.stack((torch.zeros_like(Sn), Sn, Tn.expand(-1,C,Lj,Wj,-1)), dim=-1).view(N*C,Lj,Wj,Dj,3) # 0,S,T in samp xyz -> T,S,0 in ref ijk
        ts_grid = self.ts_grid_d3base[0].expand(N*C,-1,-1,-1,-1) #N*C,1,tc,sc,1  .. to preserve ts_input C dim, collapse to batch dim
        # COMMENT - ts_grid_d3base[1] brine salinity balance is ignored : salinity transport should occur dynamically
        ybf = 1 - F.grid_sample(ts_grid, ts_input).reshape(N,C,Lj,Wj,Dj) #N*C,1,Lj,Wj,Dj to N,C,Lj,Wj,Dj
        xbf = F.avg_pool3d(F.pad(ox > 0, ts_pad, mode='reflect').to(dtype), ts_samp)
        del(Tn, Sn, ts_grid, ts_input)
        
        dS = F.interpolate(yS - xS, size=(Li,Wi,Di), mode='trilinear')
        dbf = F.interpolate(ybf - xbf, size=(Li,Wi,Di), mode='trilinear')
        return dS, dbf

    def PCF(self, ox, px, T, ts_samp=(15,15,3), max_percolations=200):
        '''
        Calculates the discrepancy between lattice cluster and empirical cluster properties in current T-S conditions.
        input
        ox : (N,C,Li,Wi,Di) float tensor
        output
        tuple(  ) of 0~1 float tensors
        
        ref - Growth rate dependence of the permeability and percolation threshold of young sea ice
        (Maus, 2025) @ src: https://pubs.rsc.org/en/content/articlehtml/2025/fd/d4fd00172a
        ref - Seasonal evolution of granular and columnar sea ice pore microstructure and pore network connectivity
        (Oggier & Eicken, 2022) @ src: https://doi.org/10.1017/jog.2022.1
        '''
        N,C,Li,Wi,Di = ox.shape
        dtype,device = ox.dtype, ox.device
        ts_pad = tuple([ts_samp[i//2]//2 for i in range(5,-1,-1)])

        # porosity (inclusive of brine & gas inclusions)
        xP = F.avg_pool3d(F.pad(ox != 0, ts_pad, mode='reflect').to(dtype), ts_samp)
        Lj,Wj,Dj = xP.shape[2:]

        # melt-growth-transition mapping
        yT = F.avg_pool3d(F.pad(T, ts_pad[:2], mode='reflect'), (1,1,ts_samp[-1]))
         # T_eq assumption : all points as linear (2-point mean value) between window extrema
        yT_eq = F.pad((yT[...,ts_samp[2]//2:] + yT[...,:-ts_samp[2]//2]) / (ts_samp[2] - 1), ts_pad[:2], mode='replicate')
        NES = yT - yT_eq; NES /= F.pad(torch.var(NES.unfold(dim=-1,size=ts_samp[2],step=1), dim=-1)**0.5, ts_pad[:2], mode='replicate')
        # positive/negative NES : melting/growing
        # is melt : melt boundary @ positive NES, source @ top and bottom
        p_m = torch.cumsum(F.softmax(NES, dim=-1), dim=-1); p_m /= p_m[..., -1]
        p_m = torch.mean(p_m, dim=-1, keep_dim=True) * (1 - p_m * (1 - p_m))**p_m.sum(dim=-1, keep_dim=True)
        # is growth : melt boundary @ negative NES, source @ bottom
        p_g = torch.cumsum(F.softmax(-NES, dim=-1), dim=-1); p_g /= p_g[..., -1]
        p_g = torch.mean(p_g, dim=-1, keep_dim=True) * p_g
        
        # using Chebyshev connectivity, explore connectivity of a random brine cell.
        poro = (ox != 0)
        percol = ((torch.rand((1,1,Li,Wi,Di),device=device) < 0.1) * poro).to(torch.int16) # select with 10% chance a gas or brine cell
        ids = torch.cumsum(percol.view(N,C,-1), dim=-1).view(N,C,Li,Wi,Di); percol *= ids # assign dense unique ID within each (N,D)
        for i in range(max_percolations): percol = poro * F.max_pool3d(percol, 3, 1, 1)

        percos = percol > 0; dss = percos.to(torch.int16); dss[torch.cumsum(dss, axis=-1)==1] = 2; dss -= max_percolations;
        for i in range(max_percolations): tmp = F.max_pool3d(-dss, 3, 1, 1); dss[tmp > dss] = tmp + 1

        # clusters are given an energetic favorability proportional to their size relative to the local cluster strength.
        dCS = torch.zeros_like(percol); poroc = poro.sum(dim=(-3,-2,-1), keepdim=True)
        for i in range(ids[...,-1].max()):
            cluster = (percol == i); cl_n = cluster.sum(dim=(-3,-2,-1), keepdim=True)
            gs_p = ((px * cluster).sum(dim=(-2,-3,-4), keepdim=True) / cl_n).view(N*C,1,1,1,3).reverse(dim=-1) #ijk to xyz
            cs_xP = F.grid_sample(xP.view(N*C,1,Lj,Wj,Dj), gs_p, mode='trilinear').view(N,C,1,1,1) # retrieve local porosity
            targ_cl_s = (cs_xP - 0.002)**0.82 # Maus et al. empirical relation for isotropic percolation

            # tortuosity approximation
            #tort = cluster.any(dim=(-3,-2), keepdim=True)
            
            dCS += cluster * torch.exp(-((cl_n / poroc) - targ_cl_s)**2)  # migration probability is high when cluster strength is not at expectation

        # diffuse cluster potential via multiple gaussian convolutions
        wind = torch.meshgrid(*[torch.arange(e) - e//2 for e in ts_samp], indexing='ij')
        wind = torch.exp(-0.125*(wind[0]**2 + wind[1]**2 + wind[2]**2))[None,None,...]
        for i in range(3): dCS = F.conv3d(dCS.view(N*C,1,Li,Wi,Di), wind, padding=ts_pad)
        
        return dCS#, tort
    
    def anneal_step(self, x, T, S, ts_samp=(15,15,3)):
        '''
        Perform inplace annealing optimization on a D3Q7 lattice.
        -- input
        x : (N, C, L, W, D) float tensor with +2 boundary pad in 3 dimensions
        T : (N, 1, 1, 1, Di) float tensor - reference temperature (variable)
        S : (N, 1, 1, 1, Di) float tensor - reference salinity
        '''
        N,C,L,W,D = x.shape
        Li,Wi,Di = L-2,W-2,D-2 # boundary-excluding sizes
        dtype, device = x.dtype, x.device
        if ((np.array(window)/2 - np.array(window)//2) != 0).any():
            raise ValueError("window must be odd length in all dimensions.")
            
        ox = x[...,1:-1,1:-1,1:-1] # slice only : is direct reference
        # coordinate grid for shift-based access
        px = torch.stack(torch.meshgrid(    # N,C,Li,Wi,Di,3
            torch.arange(1, L-1, dtype=torch.int32, device=device),
            torch.arange(1, W-1, dtype=torch.int32, device=device),
            torch.arange(1, D-1, dtype=torch.int32, device=device),
            indexing='ij'
        ),dim=-1)[None,None,...].expand(N,C,-1,-1,-1,-1)
        
        # calculate thermophysical potential
        dS, dbf = TSF(ox, T, S, ts_samp)
        pbf = torch.tanh(dbf); del(dbf)
        # calculate network potential
        dCS = PCF(ox, px, T, ts_samp, max_percolations = ox.shape[-1]*2)
        pCS = torch.sigmoid(dCS); del(dCS)
        # interaction kinetic
        bar = torch.exp(-torch.softplus(-T))
        bar = 1 - bar*(torch.randn_like(ox) + 1)
        del(px)
        
        for i in torch.randperm(self.shift_ops.shape[0]):
            opd = self.shift_ops[i]
            oy = x.narrow(2, opd[0]+1, Li).narrow(3, opd[1]+1,Wi).narrow(4, opd[2]+1,Di)
            
            # interaction masks
            m_tp, m_bt, m_hz = (oy == -999), (oy == -998), (oy == -997)
            #ymnb = ~(m_tp | m_bt | m_hz) # interior / not-boundary mask
            xmi, xmb, xmg = (ox == 0), (ox >  0), (ox <  0)
            ymi, ymb, ymg = (oy == 0), (oy >  0), (oy <  0)
            
            # ---------- ice-ice ---------- #
            fm = xmi & ymi
            # .. spontaneous melting
            fmm = fm & (pbf**3 > bar)
            ox[fmm] += 1e-8
            
            # ---------- brine-air ---------- #
            fm = xmb & ymg
            # .. air rising
            if opd[-1] > 0: # y.z > x.z
                tx = ox[fm].clone()
                ox[fm], oy[fm] = oy[fm], tx
            # .. horizontal brine travel
            if opd[-1] == 0:
                fmm = fm & (pCS > bar)
                tx = ox[fmm].clone()
                ox[fmm], oy[fmm] = oy[fmm], tx
                
            # ---------- ice-brine ---------- #
            fm = xmi & ymb
            # .. ice melting
            fmm = fm & (pbf > bar)
            oy[fmm] = oy[fmm] / 2
            ox[fmm] = oy[fmm]
            # .. brine migration
            fmm = fm & (pCS > bar)
            tx = ox[fmm].clone()
            ox[fmm], oy[fmm] = oy[fmm], tx

            # ---------- brine-brine ---------- #
            fm = xmb & ymb
            # .. brine freezing
            fmm = fm & (pbf < -bar)
            oy[fmm] = ox[fmm] + oy[fmm]
            ox[fmm] = 0
            # .. displacement
            if opd[-1] > 0:
                fmm = fm & (oy < ox) # y below fresher than x above
                tx = ox[fmm].clone()
                ox[fmm], oy[fmm] = oy[fmm], tx
            # .. equalization
            fmm = fm & (bar < 0.25)
            ox[fmm] = (ox[fmm] + oy[fmm]) / 2
            oy[fmm] =  ox[fmm]

            # ---------- brine-edge ---------- #
            fm = xmb & (m_bt | m_hz)
            # .. edge salt flux
            ox[fm] += dS[fm] * (1 - bar[fm]).clamp(0, 1)




    #def roughen(self, x, b):
    #    '''
    #    Deform the icepack to apply mesoscale surface roughness properties.
    #    -- input
    #    x : (N, C, L, W, D) float tensor with +2 boundary pad in 3 dimensions
    #    b : (N, P)            float tensor - damage field parameters.
    #   .. damage field parameters:
    #    0, 1, 2 : L, W correlation lengths
    #    '''
    #    return x