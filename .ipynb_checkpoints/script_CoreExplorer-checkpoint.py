import sys, os, datetime, warnings
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn.functional as F

import data_resice
import brack2bing
import crefl2slc
import util_collector
import util_filters

# --- SAR IMAGE ANALYZER  --- #
import data_cryos
import PyPreprocess_Single
from PIL import Image
def analyze_image(cache_dir, model, vlens, floqs, slc_ds=None, rmask_fn='mask_bisplit.png', sr=1, box_filter=(16,4), valids=['AMSR_KH', 'CS2', 'SMOS', 'neXtSIM', 'IC1'], hist_bins=20, aspect=4, device='cuda'):
    if slc_ds is None: slc_ds = PyPreprocess_Single.Dataset_SNAPbase(cache_dir=cache_dir, read_nc=True)
    slc, ia = torch.as_tensor(slc_ds.slc[:, ::sr, ::sr]).to(device), torch.as_tensor(slc_ds.ia[::sr,::sr]).to(device)
    icesim = brack2bing.LatticeSolver.load()
    slc[slc == 0] = torch.nan # write null value
    slc = util_filters.nanpute_n1mean(slc, 1000, (1,2))
    B,H,W = slc.shape

    # load region mask
    rmask = torch.as_tensor(np.array(Image.open(os.path.join(cache_dir, rmask_fn)), dtype=np.uint8)[::sr,::sr,0]).to(device)
    rM = [rmask == val for val in rmask.unique()]
    rX, rXn = [None] * len(rM), torch.zeros(len(rM), device=device)
    
    vends = np.array(vlens).cumsum()

    torch.set_float32_matmul_precision('high')
    with torch.no_grad():
        metric = torch.empty((H,W), device=device)
        stride = (200, 200)
        patches = ((i, j) for i in range(0, H, stride[0]) for j in range(0, W, stride[1]))
        for i, j in tqdm(patches, total=((H+stride[0]-1)//stride[0] * (W+stride[1]-1)//stride[1])):
            slc_ = torch.as_tensor(slc[:, i:i+stride[0], j:j+stride[1]]).moveaxis(0,-1).cuda()
            ia_  = torch.as_tensor(ia[i:i+stride[0], j:j+stride[1]]).cuda()
            h, w = slc_.shape[0], slc_.shape[1]
            corevec, cond_mask = slc2core(slc_, ia_, vends[-3], model.whitener, sim_inc=floqs)
            corevec = model(corevec, cond_mask)
            # accumulate region aggregate output corevec
            for k in range(len(rM)):
                _rM = rM[k][i:i+stride[0], j:j+stride[1]].cuda().flatten()
                if not _rM.any(): continue
                vals = corevec[_rM, ...]
                if rX[k] is None: rX[k]  = vals.sum(dim=0, keepdim=True)
                else:             rX[k] += vals.sum(dim=0, keepdim=True)
                rXn[k] += vals.shape[0]
            STr = model.unwhitener(corevec)[:,0,:vends[2]]
            S, T, rho = STr[:,:vends[0]], STr[:,vends[0]:vends[1]], STr[:,vends[1]:vends[2]]
            core = icesim.get_icefrac(T, S).reshape(h,w,-1)
            ccore = core * ((core < 0.1).cumsum(dim=-1) == 0) # zero all entries after first 90% melt layer in depth dim
            ccore[rho.reshape(h,w,-1) < 0.25] = 0 # density below 0.25 g/cm3 is free space, do not count towards thickness
            metric[i:i+h, j:j+w] = (ccore > 0).sum(-1) # get thickness as all remaining bins
        metric /= 30;
        # smoothing filter
        _p = np.array(box_filter)-1; box_filter_pad = tuple(np.array([_p//2, _p-_p//2]).T[::-1].flatten())
        metric = F.avg_pool2d(F.pad(metric[None,None,], box_filter_pad, mode='replicate'), box_filter, stride=1)[0,0]
        # region aggregate output whitening
        rX = [model.unwhitener(rX[i] / rXn[i]).detach().cpu() for i in range(len(rM))]
    metric = metric.detach().cpu()
    # region aggregate histograms
    #plt.hist([metric[_rM] for _rM in rM], bins=hist_bins); plt.legend([f'Region {i}' for i in range(len(rM))])
    #plt.title('IceMatcher'); plt.xlabel('Sea ice thickness (m)'); plt.ylabel('Pixels'); plt.yscale('log')
    #plt.savefig(os.path.join(cache_dir, 'SIT_histograms.png')); plt.close()
    
    # validation data
    image_date = int(datetime.datetime.fromtimestamp(slc_ds.metadata['0dopp_t0'][0]).strftime('%Y%m%d'))
    dlat, dlon = slc_ds.lat[::sr,::sr], slc_ds.lon[::sr,::sr]
    d_valids = {}; print(f'retrieving validation data for date {image_date}')
    for valid in valids:
        try:
            print(f'retrieving {valid} data...')
            d_valids[data_cryos.TITLES[valid]] = data_cryos.get(valid, image_date, dlat, dlon, mode='nearest')
        except Exception as e:
            print(f'could not access dataset (error : {e})')
    print('data retrieved.')
    # ...
    #plt.figure(figsize=(16, 4*(1+len(d_valids))))
    #for i, (title, vdat) in enumerate(d_valids.items()):
    #    plt.subplot(1 + len(d_valids), 1, i+1); plt.imshow(vdat.T, aspect=aspect, cmap='plasma'); plt.title(title); plt.colorbar()
    #plt.subplot(1 + len(d_valids),1,1 + len(d_valids)); plt.imshow(metric.T, aspect=aspect, cmap='plasma'); plt.title('IceMatcher (m)'); plt.colorbar()
    #plt.savefig(os.path.join(cache_dir, 'SIT_plots.png')); plt.close()    
    
    return metric, d_valids, rX

# --- BACKSCATTER + INC TO CORE VECTOR --- #

def get_core(f, count=100, whitener=None):
    if isinstance(f, str): collector = util_collector.Dataset(f, mode='r')
    else: collector = f
    feed = get_randoms(count, reshape=False, collector=collector)
    # form covariance matrix
    rf = feed['rf'].reshape([-1] + collector.h5f['rf'].attrs['fshape'].tolist()) #B,1,4,6
    rf, (B,C,P,I) = torch.as_tensor(rf).transpose(-1, -2), rf.shape
    rf_C4 = (rf[...,None] * rf.conj()[...,None,:]) # polarimetric covariance   ..  B,1,6,16
    rf_C4 = ( rf_C4 / torch.diagonal(rf_C4, 0, -2, -1).sum(dim=-1)[...,None,None] ).reshape(B,C,I*P*P) # normalized C4, B,1,6*4*4
    # create flattened vector
    feed = [feed['S'].real, feed['T'].real, feed['rho'].real, rf_C4.real, rf_C4.imag]
    feed_sp = [el.shape[1:] for el in feed]; feed_vl = [torch.as_tensor(el).prod() for el in feed_sp]
    feed = torch.cat([torch.as_tensor(el).flatten(1) for el in feed], dim=1)[:,None,:].to('cuda', torch.float32)
    if whitener is not None: feed = whitener(feed)
    # ...
    floqs = collector.h5f.attrs['floq']
    if isinstance(f, str): collector.close()
    return feed, floqs, feed_vl, feed_sp
    
def slc2core(slc, inc, struct_vlen, whitener, sim_inc=[21.5, 27.3, 33.3, 39.9, 47.1, 55.5], C4_smooth=(11,11)):
    assert (slc.shape[:2] == inc.shape)
    (H,W,B), I, ref_inc = slc.shape, len(sim_inc), torch.deg2rad(torch.tensor(sim_inc).to(inc.device, inc.dtype))
    wind_pad = tuple([el // 2 for el in C4_smooth])
    C4 = slc[...,:,None] * slc[...,None,:].conj() # construct normalized C4
    C4 = torch.complex(
        F.avg_pool2d(C4.real.permute(2,3,0,1), C4_smooth, stride=1, padding=wind_pad, count_include_pad=False).permute(2,3,0,1),
        F.avg_pool2d(C4.imag.permute(2,3,0,1), C4_smooth, stride=1, padding=wind_pad, count_include_pad=False).permute(2,3,0,1)
    )
    C4 = C4 / torch.diagonal(C4, 0, -2, -1).sum(dim=-1)[...,None,None] # H,W,B,B
    inc_p = (inc[...,None] - ref_inc[None,None,:]) / abs(ref_inc[-1] - ref_inc[0])
    inc_p = 0.5 * torch.exp(-2*(ref_inc.numel() * inc_p).pow(2))[...,None,None] # (H,W,I,1,1) Gaussian splat
    # .. fill empty core vector
    C4 = C4.reshape(H*W, 1, B*B).expand(-1, I, -1).reshape(H*W,I*B*B)
    core_vec = torch.zeros((H*W, 1, struct_vlen + 2*I*B*B), device=slc.device, dtype=inc.dtype)    
    core_vec[:, 0, -2*I*B*B:-I*B*B] = C4.real ; core_vec[:, 0, -I*B*B:] = C4.imag
    core_vec = whitener(core_vec)# / 1000 # move to diffusion state space
    # .. build conditional mask from gaussian splat
    cond_mask = torch.zeros_like(core_vec)
    cond_mask[:, 0, -2*I*B*B:] = inc_p[:,:,None,...].expand(-1,-1,2,-1,B,B).reshape(H*W,2*I*B*B)
    # .. destroy information according to conditional mask
    #core_vec *= cond_mask
    return core_vec, cond_mask

# --- [RAND] & [REAL] CORE SIMULATION/ACCUMULATION SCRIPT --- #

def _get_ltsolvr():
    if (os.path.exists("brack2bing_solver_cache/init.pickle")):
        return brack2bing.LatticeSolver.load()
    icesim = brack2bing.LatticeSolver(device='cuda')
    icesim.save()
    return icesim

def generate_(cores, fn='em_funcgen.hdf5', batch_n=12, bins=None, hshape=(50,20), wavelength_m=0.236, cells_per_m=20, ia_b=(20,60), device='cuda', verbose=False, collector=None, attrs={}):
    (T_, S_, rho_), icesim = cores, _get_ltsolvr()
    if collector is None : collector = util_collector.Dataset(fn)
    batches = ((T_[i:i+batch_n], S_[i:i+batch_n], rho_[i:i+batch_n]) for i in range(0, T_.shape[0], batch_n))
    for i, (T, S, rho) in enumerate(tqdm(batches)):
        lattice = icesim.gen_lattice(T, S, rho, hshape=hshape)
        er = icesim.get_dielectric(lattice, T, wavelength_m)
        sref = crefl2slc.SimpleReflector(er, T, cells_per_m=cells_per_m,wavelength_m=wavelength_m, batch_size=4)
        floq = np.round(np.rad2deg(sref.ia_input), decimals=2)[:,0]; floq = floq[(floq >= ia_b[0]) & (floq < ia_b[1])]
        if i == 0: print('floquet incidence modes:', floq)
        sref.gen_refl_sample(ia_b=ia_b, verbose=verbose)
        collector.add(T=T, S=S, rho=rho, rf=sref.refl)
    collector.set_attrs(dbins=bins or T.shape[-1], hshape=hshape, wavelength_m=wavelength_m, cells_per_m=cells_per_m, ia_b=ia_b, floq=floq, **attrs)
    return collector

def generate_randoms(iters, fn='em_randoms.hdf5', batch_n=12, bins=100, hshape=(50,20), wavelength_m=0.236, cells_per_m=20, ia_b=(20,60), device='cuda'):
    collector = None
    for i in range(iters):
        batches = gen_rand_cores(batch_n, bins=bins, device=device) # could be simplified
        collector = generate_(batches, fn=fn, bins=bins, hshape=hshape, wavelength_m=wavelength_m,
                              cells_per_m=cells_per_m, ia_b=ia_b, device=device, collector=collector)
    return collector
    
def generate_resice(fn='em_resice.hdf5', iters=1, bins=100, hshape=(50,20), wavelength_m=0.236, cells_per_m=20, ia_b=(20,60), lattice_freeboard_m=0.25, min_samples=3, get_oversize=False, parts=['A'], augment_function=None, device='cuda'):
    collector, column_thick = None, bins/cells_per_m
    for i in range(iters):
        for part in parts:   # T, S, rho
            T, S, rho = data_resice.get_scaled_cores( f'RESICE/RESICE_Part{part}.csv', bins=bins, max_depth=column_thick - lattice_freeboard_m,
                                                      max_height=lattice_freeboard_m, min_samples=min_samples, get_oversize=get_oversize, device=device )
            if rho.shape[-1] == 1: rho = rho.expand(-1, T.shape[-1]);
            if augment_function is not None: T, S, rho = augment_function(T, S, rho)
            print(f'simulating {T.shape[0]} RESICE retrievals')
            collector = generate_((T,S,rho), fn=fn, bins=bins, hshape=hshape, wavelength_m=wavelength_m, cells_per_m=cells_per_m, ia_b=ia_b,
                                  device=device, verbose=True, collector=collector, attrs={'get_oversize': get_oversize})
    return collector
    
# --- UTIL_COLLECTOR based 
    
def get_randoms(count, collector=None, fn='em_randoms.hdf5', reshape=False):
    if collector is None:
        flag = True
        collector = util_collector.Dataset(fn, mode='r')
    else: flag = False
    w_ind = collector.h5f.attrs['w_ind']
    if (count > w_ind): warnings.warn(f'requested count {count} exceeds dataset sample count {w_ind}; retrieving {w_ind} samples.')
    xi = np.random.choice(range(0, w_ind), size=min(count, w_ind), replace=False)
    results = collector.read(xi,reshape=reshape,clamp=False)
    if flag:
        collector.close()
    return results

def view_random(collector, wavelength_m=0.229, cells_per_m=20):    # plot simulation results
    tmp = collector.read([np.random.randint(0, collector.h5f.attrs['w_ind'])], reshape=True, clamp=False)
    print(tmp.keys())

    icesim = brack2bing.LatticeSolver.load()
    die = icesim.get_dielectric(torch.as_tensor(tmp['lt']), torch.as_tensor(tmp['T']), wavelength_m).numpy()
    depth = np.linspace(0,tmp['T'].shape[-1]/cells_per_m,tmp['T'].shape[-1])
    plt.figure(figsize=(11.83,5.8))
    plt.subplot(1,6,1); plt.plot(tmp['T'][0], depth); plt.xlabel('T(C)'); plt.ylabel('depth(m)'); plt.title('temperature')
    plt.subplot(1,6,2); plt.plot(tmp['S'][0], depth); plt.xlabel('S(g/kg)'); plt.yticks([]); plt.title('salinity')
    plt.subplot(1,6,3); plt.plot(tmp['rho'][0], depth); plt.xlabel('rho (g/cm3)'); plt.yticks([]); plt.title('density')
    plt.subplot(1,6,4); plt.imshow(tmp['lt'][0,0,2,1:-1,1:-1][...,::-1].T, aspect=0.7); plt.xticks([]); plt.yticks([]); plt.xlabel('lattice'); plt.title('ice')
    plt.subplot(1,6,5); plt.imshow(die[0,0,26][...,::-1].T.real, aspect=0.7); plt.xticks([]); plt.yticks([]); plt.xlabel('e_r'); plt.title('refraction')
    plt.subplot(1,6,6); plt.imshow(die[0,0,26][...,::-1].T.imag, aspect=0.7); plt.xticks([]); plt.yticks([]); plt.xlabel('e_i'); plt.title('absorption')
    plt.suptitle('parameters')
    plt.show()
    
    plt.figure(figsize=(12,3))
    angles = [21.5, 27.3, 33.3, 39.9, 47.1, 55.5]
    polars = ['HH','HV','VH','VV']
    for i in range(len(polars)):
        plt.subplot(1,len(polars),i+1)
        radius = np.abs(tmp['rf'][0,0,i]).max()
        plt.plot(tmp['rf'][0,0,i,:].real, tmp['rf'][0,0,i,:].imag)
        for j in range(len(angles)):
            plt.text(tmp['rf'][0,0,i,j].real, tmp['rf'][0,0,i,j].imag, f'{angles[j]}', fontsize=7)
        plt.xlim(-radius,radius); plt.ylim(-radius,radius)
        
        thetas = np.linspace(0,2*np.pi, 16, endpoint=True)
        [plt.plot([0, np.cos(theta)],[0, np.sin(theta)],color="lightgray",linewidth=0.5) for theta in np.linspace(0,2*np.pi, 16, endpoint=False)]
        [plt.plot(R*np.cos(thetas),R*np.sin(thetas),color='lightgray',linewidth=0.5) for R in np.linspace(0,radius,5)]
        [plt.text(r*0.7071, -r*0.7071, f'{r:.1e}') for r in np.linspace(0,radius,5)[1:]]
        plt.xticks([])
        plt.yticks([])
        plt.title(polars[i])
    plt.show()

# --- [RAND] CORE AUGMENTATION SCRIPT --- #

def WienerNoiser(x, t=1.0, L=1.0, n=16):
    '''
    Admit a 1D vector (B,C,N) and add scaled noise of a Wiener process of Gaussian window-correlated perturbations with correlation length L.
    '''
    (B, C, N), K = x.shape, torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
    K = torch.exp(-0.5 * (K[:,None] - K[None,:]).pow(2) / L**2) # gaussian spatial kernel
    K /= K.square().sum(dim=-1, keepdim=True).sqrt() # spatial kernel L2 normalization (preserve variance)
    z = torch.randn(B,C,n,N, device=x.device, dtype=x.dtype) # random perturbations
    return x + (t / n)**0.5 * (z @ K.T).sum(dim=-2)

def RandWalk(x, sigma=0.0, L=0, steps=1):
    '''
    Admit a 1D vector (B,C,N) and apply a random perturbation of Gaussian-kernel correlation length L.
    '''
    (B, C, N), y = x.shape, x.clone()
    inds = torch.arange(N)[None,None,:]
    for i in range(steps):
        z = sigma * torch.randn_like(x[...,None,0]) # amplitude
        z_p = torch.rand_like(x[...,None,0]) * (N - 1) # position
        if L == 0     : L     = torch.randn_like(z)    # correlation length
        z = z * torch.exp(-0.5 * (inds - z_p)**2 / L**2)
        y += z
    return y
    
# --- [RAND] CORE GENERATION SCRIPT --- #

def capped_sumnoise(v0=-2, scale=30, bins=10, device='cpu', dtype=torch.float32):
    tmp = torch.linspace(0, 1, bins, device=device, dtype=dtype)
    b = torch.randn_like(tmp)*(1-tmp)**2
    b = (b / (1+b.pow(2))).cumsum(dim=0)
    return (v0 - scale * F.gelu(scale**0.5 * -F.gelu(b)))#.flip(dims=(0,))
def gdiffuse_bias_weighted(x, steps, k=5, alpha=0.2):
    gauss = torch.exp(-0.5*torch.linspace(-4,4,k)**2).to(x.device)[None,None,:]; gauss /= gauss.sum()
    wgt = torch.linspace(0,1,x.shape[-1]).pow(1.2).to(x.device) # more diffusion closer to surface (@ index 0)
    for i in range(steps):
        y = F.conv1d(F.pad(x[None,],(k//2,k//2),mode='replicate')[None,], gauss)[0,0]
        x += alpha * wgt * (y - x)
    return x
def gen_rand_cores(n_samples, bins=32, dtype=torch.float32, device='cuda'):
    # 1. initialize open water condition
    T = torch.full((n_samples,bins), -1.8, dtype=dtype, device=device)
    S = torch.full((n_samples,bins), 35.0, dtype=dtype, device=device)
    rho = torch.full((n_samples,bins), 1.028, dtype=dtype, device=device)
    # 2. generate fill range
    d_ice = torch.randint(1, bins+1, (n_samples,), dtype=torch.int32)
    for i in range(n_samples):
        T[i,:d_ice[i]] = capped_sumnoise(-2, 30, d_ice[i], device, dtype)
        T[i] = gdiffuse_bias_weighted(T[i], d_ice[i])
        S[i,:d_ice[i]] = 34 - capped_sumnoise(33, 28, d_ice[i], device, dtype).abs()
        S[i] = gdiffuse_bias_weighted(S[i].flip((0,)), d_ice[i]//8).flip((0,))
        rho[i,:d_ice[i]] = capped_sumnoise(0.917, 2, d_ice[i], device, dtype)
        rho[i] = gdiffuse_bias_weighted(rho[i], d_ice[i]//2).clamp(max=1.034)
    return T,S,rho


