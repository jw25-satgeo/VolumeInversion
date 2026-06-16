import os
import numpy as np
import torch
import scipy

import pandas as pd

def get_cores(fn = "RESICE/RESICE_PartA.csv", min_samples=10, max_depth=2.5):
    core_data = pd.read_csv(fn)
    core_data = core_data[
        ~np.isnan(core_data['depth [m]']) *
        ~np.isnan(core_data['thickness_sea_ice [m]']) *
        ~np.isnan(core_data['salinity_sea_ice [ppt]']) *
        ~np.isnan(core_data['freeboard_sea_ice [m]']) * #note : freeboard measurements only present for 25% of data
        ~np.isnan(core_data['temperature_sea_ice [K]'])]
    core_data = core_data[['id','thickness_sea_ice [m]','depth [m]','temperature_sea_ice [K]',
                           'salinity_sea_ice [ppt]','freeboard_sea_ice [m]','thickness_snow [m]']]
    cores = [el for el in [core_data[core_data['id']==core_id] for core_id in set(core_data['id'])]
             if len(el) >= min_samples and el['thickness_sea_ice [m]'].iloc[0] < max_depth]
    return cores

def get_scaled_cores(fn="RESICE/RESICE_PartA.csv", bins=32, min_samples=10, max_depth=2.5, dtype=torch.float32, device='cuda'):
    cores = get_cores(fn, min_samples, max_depth)
    cores = list(zip(*[core2scaledTS(core, bins, max_depth) for core in cores]))
    T = torch.as_tensor(np.array(cores[0]), dtype=dtype, device=device)
    S = torch.as_tensor(np.array(cores[1]), dtype=dtype, device=device)
    rho = torch.as_tensor(np.array(cores[2]), dtype=dtype, device=device)[:,None]
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

def core2scaledTS(core, bins=32, max_depth=2.5):
    '''
    returns T, S, rho ((N), (N), float) tuple.
    -- units
    T: Celsius
    S: parts per thousand (eq: g/kg)
    rho: g/cm3
    '''
    T,S,d = [np.array(el) for el in (core['temperature_sea_ice [K]'], core['salinity_sea_ice [ppt]'], core['depth [m]'])]
    if ((T > 100).any()): T -= 273.15 # Kelvin to Celsius
    scale = np.linspace(0, max_depth, bins)
    scaled = scipy.interpolate.PchipInterpolator(d, np.stack((T,S), axis=-1),extrapolate=False)(scale)
    mask = d.min() > scale  # fill in top via extend
    scaled[mask] = scaled[mask.cumsum().max()]
    scaled[np.isnan(scaled)] = 0
    scaled += (scaled==0) * np.array([[-1.8,35]]) #fill in bottom with seawater

    snow_depth = np.nan_to_num(np.array(core['thickness_snow [m]']), nan=0.0).max()
    freeboard = np.nan_to_num(np.array(core['freeboard_sea_ice [m]']), nan=0.0).max()
    thickness = np.array(core['thickness_sea_ice [m]']).max()
    rho = (1.024*thickness - 0.35*snow_depth - 1.024*freeboard) / thickness
    
    return scaled[:,0], scaled[:,1], rho