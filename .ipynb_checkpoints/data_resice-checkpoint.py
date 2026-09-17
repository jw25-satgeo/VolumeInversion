import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy

import pandas as pd

'''
This data access module processes the RESICE dataset.
Reference : Simson, A., Yildiz, A. & Kowalski, J. Reusability-targeted enrichment of sea ice core data. Sci Data 12, 465 (2025). https://doi.org/10.1038/s41597-025-04665-x
'''

# note - 'max_depth' creates a cutoff for icepack access
def get_cores(fn = "RESICE/RESICE_PartA.csv", min_samples=10, max_depth=2.5):
    core_data = pd.read_csv(fn)
    core_data = core_data[
        ~np.isnan(core_data['depth [m]']) *
        ~np.isnan(core_data['thickness_sea_ice [m]']) *
        ~np.isnan(core_data['salinity_sea_ice [ppt]']) *
        #~np.isnan(core_data['freeboard_sea_ice [m]']) * #note : freeboard measurements only present for 25% of data
        ~np.isnan(core_data['temperature_sea_ice [K]'])]
    core_data = core_data[['id','thickness_sea_ice [m]','depth [m]','temperature_sea_ice [K]',
                           'salinity_sea_ice [ppt]','freeboard_sea_ice [m]','thickness_snow [m]']]
    cores = [ el for el in [core_data[core_data['id']==core_id] for core_id in set(core_data['id'])]
                       if len(el) >= min_samples and el['thickness_sea_ice [m]'].iloc[0] < max_depth ]
    return cores

def get_scaled_cores(fn="RESICE/RESICE_PartA.csv", bins=32, min_samples=10, max_depth=2.5, max_height=0.5, get_oversize=False, dtype=torch.float32, device='cuda'):
    if get_oversize : cores = get_cores(fn, min_samples, 20.0)
    else            : cores = get_cores(fn, min_samples, max_depth)
    cores = list(zip(*[core2scaledTS(core, bins, max_depth) for core in cores]))
    T = torch.as_tensor(np.array(cores[0]), dtype=dtype, device=device)
    S = torch.as_tensor(np.array(cores[1]), dtype=dtype, device=device)
    rho = torch.as_tensor(np.array(cores[2]), dtype=dtype, device=device)
    # -- basic prototype sample additions
    # 1 : open water
    T = torch.cat((T, torch.zeros_like(T[0])[None,]), dim=0)
    S = torch.cat((S, torch.ones_like(S[0])[None,]*S.max()), dim=0)
    rho = torch.cat((rho, torch.ones_like(rho[0])[None,]*(1+S[-1,0]/1000)), dim=0)
    # 2 : pure ice
    T = torch.cat((T, -torch.ones_like(T[0])[None,]), dim=0)
    S = torch.cat((S, torch.zeros_like(S[0])[None,]), dim=0)
    rho = torch.cat((rho, torch.ones_like(rho[0])[None,]*0.917), dim=0)
    
    return T,S,rho


def core2scaledTS(core, bins=32, max_depth=2.5, max_height=0.5):
    '''
    returns T, S, rho ((N), (N), float) tuple.
    -- units
    T: Celsius
    S: parts per thousand (eq: g/kg)
    rho: g/cm3
     ~ NOTE ~ because this uses a PchipInterpolator, oversized and undersized cores are resampled without issue.
    '''
    T,S,d = [np.array(el) for el in (core['temperature_sea_ice [K]'], core['salinity_sea_ice [ppt]'], core['depth [m]'])]
    rho = np.ones_like(T) # RESICE does not contain density data
    freeb = np.nan_to_num(np.array(core['freeboard_sea_ice [m]']), nan=0.0).max()
    snowd = np.nan_to_num(np.array(core['thickness_snow [m]']), nan=0.0).max()
    thick = np.array(core['thickness_sea_ice [m]']).max()
    if ((T > 100).any()): T -= 273.15 # Kelvin to Celsius
    d = d - min(freeb, max_height) # offset by freeboard if space available
    rho_ice = (1.028*thick - 0.35*snowd - 1.028*freeb) / thick # buoyancy calculation
    if freeb == 0 : rho_ice = 0.917 # pure ice density prior
    
    scale = np.linspace(-max_height, max_depth, bins)
    scaled = np.nan_to_num(scipy.interpolate.PchipInterpolator(d, np.stack((T,S,rho), axis=-1),extrapolate=False)(scale), nan=0.0)
    
    snowmask = (scale < -freeb) & (scale >= -(freeb + snowd))
    scaled[:,0][snowmask & (scaled[:,0] == 0)] = T.min() # infill temperature
    scaled[:,1][snowmask & (scaled[:,0] == 0)] = S.min() # infill salinity
    # fill in icepack density
    icemask = (scale >= 0) & (scale <= thick)
    scaled[icemask, 2] = rho_ice
    # fill in bottom w. seawater at freezing point (calculated using TEOS-10)
    watermask = (scale > thick)
    scaled[watermask, 0] = -1.8 # Celsius
    scaled[watermask, 1] = 35   # g/kg ~ ppt
    scaled[watermask, 2] = 1.028      # g/cm3
    # space between ice surface & first measurement
    surfmask = (d.min() > scale) & (-freeb <= scale)
    scaled[surfmask] = scaled[surfmask.cumsum().max()]
    # ..
    return scaled[:,0], scaled[:,1], scaled[:,2]