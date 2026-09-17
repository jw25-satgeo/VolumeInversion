import os, pathlib, datetime, h5py, netCDF4
import scipy
import numpy as np
import torch
import torch.nn.functional as F

import copernicusmarine

from torch_tps import ThinPlateSpline

'''
This data access module collects data from the Arctic Ocean Sea Ice (Reanalysis) and (Analysis and Forecast) Dataset(s).
.. DOI
Analysis and Forecast : https://doi.org/10.48670/moi-00004
Reanalysis            : https://doi.org/10.48670/mds-00336
-- Reference
Williams, T., Korosov, A., Rampal, P., and Ólason, E.: Presentation and evaluation of the Arctic sea ice forecasting system neXtSIM-F, The Cryosphere, 15, 3207–3227, https://doi.org/10.5194/tc-15-3207-2021, 2021.
'''

def scipy_interp(data, data_ll, mode='nearest', rbf_neighbors=100):
    ll2data = None
    match mode:
        case 'nearest': ll2data = scipy.interpolate.NearestNDInterpolator(data_ll, data)
        case 'linear' : ll2data = scipy.interpolate.LinearNDInterpolator(data_ll, data)
        case 'rbf'    : ll2data = scipy.interpolate.RBFInterpolator(data_ll, data, neighbors=rbf_neighbors)
    return ll2data

def wgs84_llh2xyz(lat, lon, h=0): # h : radial height
    a, f = 6378137.0, 1.0 / 298.257223563  # ellipsoid parameters
    e2 = f*(2 - f) # eccentricity squared
    N = (a / np.sqrt(1 - e2*np.sin(np.deg2rad(lat))**2))
    x = (N + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y = (N + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z = (N*(1 - e2) + h)*np.sin(np.deg2rad(lat))
    return x, y, z

def wgs84_xyz2llh(x, y, z):
    a, f = 6378137.0, 1/298.257223563
    b, p, e2 = a * (1 - f),  np.sqrt(x**2 + y**2), f*(2 - f)
    theta = np.arctan2(a*z, b*p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ((a**2 - b**2)/b)*np.sin(theta)**3, p - a*e2*np.cos(theta)**3)
    h = p / np.cos(lat) - (a / np.sqrt(1 - e2*np.sin(lat)**2))
    return np.rad2deg(lat), np.rad2deg(lon), h

class Dataset():
    '''
    Arctic Ocean Sea Ice Reanalysis
    TemporalRes : Monthly
    Dates       : 1 Jan 1993 to 31 Dec 2024
    SpatialRes  : 3x3 km
    Latitude    : 49.98 to 90
    Longitude   : -180 to 179.97
    
    Arctic Ocean Sea Ice Analysis and Forecast Dataset
    TemporalRes : Hourly
    Dates       : 1 Aug 2019 to 8 Jun 2026
    SpatialRes  : 3x3 km
    Latitude    : 52.6 to 90
    Longitude   : -180 to 179.98
    '''

    def __init__(self, fn_id=None, variables=['sithick'], latb=(60, 90), lonb=(-180, 179.97), cache_dir=None, cred_file=None, compression='lzf', device='cpu'):
        if cache_dir is None:
            cache_dir = "working_cache_aosi"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        if cred_file is None:
            cred_file = '/mnt/c/Users/user/Desktop/rnd/AISat/VolumeInversion/torched/.copernicusmarine/.copernicusmarine-credentials'

        self.cache_dir = pathlib.Path(cache_dir)
        self.cred_file = pathlib.Path(cred_file)

        if fn_id is None: self.uniqueID = f'{datetime.datetime.utcnow().timestamp()}_{latb[0]}t{latb[1]}_{lonb[0]}t{lonb[1]}'
        else : self.uniqueID = fn_id
        self.h5f = h5py.File(os.path.join(cache_dir, self.uniqueID+'.hdf5'), 'a')
        self.compression = compression
        self.device = device
        
        self.variables = variables
        self.latb = np.clip(np.array(latb), 60.0, 90.0)
        self.lonb = np.clip(np.array(lonb), -180, 179.97)
        
        self.h5_dtstring_fm = '%Y%m%dT%H%M%S%f'
        self.cp_dtstring_fm = '%Y-%m-%dT%H:%M:%S'

    def read_at(self, xyz, datetimes:list, mode='nearest'):
        buffer, xyz = [], torch.as_tensor(xyz).cpu()
        if xyz.shape[-1] == 2: # convert lat-lon to cartesian coordinates
            xyz = torch.stack(wgs84_llh2xyz(xyz[...,0], xyz[...,1]), dim=-1)
        if isinstance(datetimes[0], str): # check if datetime is string list
            datetimes = [datetime.datetime.strptime(el, self.h5_dtstring_fm) for el in datetimes]
        xyzi = self.h5f['xyz'][()]
        for dt in datetimes:
            dtn = dt.strftime(self.h5_dtstring_fm)
            ddata = torch.stack([torch.as_tensor(self.h5f['scenes'][dtn][f'var_{var}'][()]) for var in self.variables], dim=-1).to(xyz.device, xyz.dtype)
            ddata = scipy_interp(ddata.reshape(-1,len(self.variables)), xyzi.reshape(-1,3), mode=mode)
            buffer.append(torch.as_tensor(ddata(xyz)).reshape(xyz.shape[0],xyz.shape[1],-1).permute(2,0,1))
        return torch.stack(buffer, dim=0)
        
    def read_at_torched(self, xyz, datetimes:list, gsi=8, mode='nearest'):
        '''
        xyz : (H,W,3) geolocation grid .. ALT : input (H,W,2(lat,lon)) geolocation grid
        '''
        buffer, xyz = [], torch.as_tensor(xyz, device=self.device)
        if xyz.shape[-1] == 2: # convert lat-lon to cartesian coordinates
            xyz = torch.stack(wgs84_llh2xyz(xyz[...,0], xyz[...,1]), dim=-1)
        if isinstance(datetimes[0], str): # check if datetime is string list
            datetimes = [datetime.datetime.strptime(el, self.h5_dtstring_fm) for el in datetimes]
        
        # generate spline for conversion of xyz to dataset pixel coordinates
        xyzi = torch.as_tensor(self.h5f['xyz'][()], device=self.device).to(xyz.dtype)
        igrid = torch.as_tensor(np.mgrid[:xyzi.shape[0],:xyzi.shape[1]], device=self.device, dtype=xyzi.dtype).view(2,-1).T
        #tps = ThinPlateSpline(device=self.device, order=order); tps.fit(xyzi.reshape(-1,3), igrid); del(xyzi, igrid)
        tps = scipy_interp(igrid.cpu(), xyzi.reshape(-1,3).cpu(), mode=mode); del(xyzi, igrid)

        # populate dense target grid for sampling op (note : interpolation has high memory overhead)
        grid = torch.empty((xyz.shape[0], xyz.shape[1], 2), device=self.device, dtype=xyz.dtype)
        di, dj = max(1, xyz.shape[0]//gsi), max(1, xyz.shape[1]//gsi)
        for bi in range(0, xyz.shape[0]+gsi, di):
            if bi >= xyz.shape[0] : continue
            for bj in range(0, xyz.shape[1]+gsi, dj):
                if bj >= xyz.shape[1] : continue
                xyz_v = xyz[bi:bi+di, bj:bj+dj]   #  write xyz2pix retrieval result to destination array
                #grid[bi:bi+di, bj:bj+dj] = tps.transform(xyz_v.reshape(-1,3)).reshape(xyz_v.shape[0], xyz_v.shape[1], 2)
                grid[bi:bi+di, bj:bj+dj] = torch.as_tensor(tps(xyz_v.reshape(-1,3).cpu()), device=grid.device).reshape(xyz_v.shape[0], xyz_v.shape[1], 2)
        # get data
        match mode:
            case 'nearest': gs_mode = 'nearest'
            case 'linear': gs_mode='bilinear'
            case 'rbf': gs_mode='bilinear'
            case None: raise ValueError('invalid mode: must be in [nearest, linear, rbf]')
        for dt in datetimes:
            dtn = dt.strftime(self.h5_dtstring_fm)
            data = torch.stack([torch.as_tensor(self.h5f['scenes'][dtn][f'var_{var}'][()]) for var in self.variables], dim=0).to(self.device,xyz.dtype)
            grid_i = (2*(grid+0.5) / torch.as_tensor(data.shape).to(self.device,grid.dtype)[None,None,-2:] - 1).flip(dims=(-1,))# j,i for grid_sample
            buffer.append(F.grid_sample(data[None,], grid_i[None,], mode='bilinear', align_corners=False)[0]) # var,H,W
        # ..
        return torch.stack(buffer, dim=0)

    def download_data(self, datetimes:list, delta=datetime.timedelta(hours=0.5), overwrite=False):
        if isinstance(datetimes[0], str): # check if datetime is string list
            datetimes = [datetime.datetime.strptime(el, self.h5_dtstring_fm) for el in datetimes]
        
        no_xyz_flag = True
        for dt in datetimes:
            dtn = dt.strftime(self.h5_dtstring_fm)
            if not overwrite and f'scenes/{dtn}' in self.h5f :
                continue
            dt1, dt2 = dt - delta, dt + delta
            dt1n, dt2n = dt1.strftime(self.cp_dtstring_fm), dt2.strftime(self.cp_dtstring_fm)
            if dt1 < datetime.datetime(year=2019, month=8, day=1):
                ds_id   = 'cmems_mod_arc_phy_my_nextsim_P1M-m'
            else: ds_id = 'cmems_mod_arc_phy_anfc_nextsim_hm'
            if os.path.exists(os.path.join(self.cache_dir, 'tmp.nc')):
                os.remove(os.path.join(self.cache_dir, 'tmp.nc'))
            copernicusmarine.subset(
                dataset_id=ds_id,
                variables=self.variables,
                minimum_longitude=self.lonb[0],
                maximum_longitude=self.lonb[1],
                minimum_latitude=self.latb[0],
                maximum_latitude=self.latb[1],
                start_datetime=dt1n,  # example format : "2009-01-01T00:00:00",
                end_datetime=dt2n,    #                  "2009-12-31T00:00:00",
                output_filename='tmp.nc',
                output_directory=self.cache_dir,
                credentials_file=self.cred_file
            )
            if not os.path.isfile(os.path.join(self.cache_dir, 'tmp.nc')):
                raise ValueError('failed download: dataset file could not be found')
            ds = netCDF4.Dataset(os.path.join(self.cache_dir, 'tmp.nc'))
    
            lat, lon = ds.variables['latitude'][:].squeeze(), ds.variables['longitude'][:].squeeze()
            if len(lat.shape) < 2 or len(lon.shape) < 2 :
                lat, lon = np.meshgrid(lat, lon, indexing='ij')
            xyz = np.stack(wgs84_llh2xyz(lat, lon), axis=-1); del(lat, lon)
            
            if overwrite and f'scenes/{dtn}' in self.h5f : del(self.h5f[f'scenes/{dtn}'])
            scene_ = self.h5f.create_group(f'scenes/{dtn}', track_order=True)
            #scene_['_datetime'] = write actual datetime of received scene for bookkeeping purposes
            if no_xyz_flag:
                if 'xyz' in self.h5f: del(self.h5f['xyz'])
                self.h5f.create_dataset('xyz', data=xyz, compression=self.compression)
                no_xyz_flag = False
            for var in self.variables:
                v_n = f'var_{var}'
                if v_n in scene_: del(scene_[v_n])
                scene_.create_dataset(v_n, data=ds.variables[var][:].squeeze(), compression=self.compression)
            ds.close()
            # clean up temp file
            if os.path.exists(os.path.join(self.cache_dir, 'tmp.nc')):
                os.remove(os.path.join(self.cache_dir, 'tmp.nc'))
        # ..




# ---- OLD METHODS


def get(dt, llgrid, cache_dir="/mnt/d/nextsim_cache", variables=['vxsi','vysi','siconc','sithick'], idt=-1, verbose=False, device='cpu'):
    '''
    returns a (B,H,W) data tensor from an (H,W,2) latitude/longitude mapping tensor.
    '''
    dtype = llgrid.dtype; llgrid = llgrid.to(device)
    output = torch.zeros((len(variables),llgrid.shape[0],llgrid.shape[1]), dtype=dtype, device=device)
    tps = ThinPlateSpline(device=device)
    for i in range(len(variables)):
        fn_l = [el for el in os.listdir(cache_dir) if variables[i] in el]
        ddt_l = np.array([dt - datetime.datetime.strptime(os.path.splitext(el)[0].split('_')[idt], '%Y-%m-%d') for el in fn_l])
        nearest_fn = fn_l[np.argmin(np.abs(ddt_l))]
        if (verbose): print(f'retrieving file : {nearest_fn} closest to target date {dt}')
    
        nc_ds = netCDF4.Dataset(os.path.join(cache_dir, nearest_fn), 'r')
        nxt_ll = torch.as_tensor(np.stack(np.meshgrid(nc_ds.variables['latitude'][:],nc_ds.variables['longitude'][:],indexing='ij'),axis=-1)).to(device,dtype)
        nxt_dat = torch.as_tensor(np.array(nc_ds.variables[variables[i]][:]), device=device, dtype=dtype)
        nc_ds.close()
        print (nxt_ll.max(), nxt_ll.min(), nxt_ll.shape)
        
        if (nxt_ll[...,1].max() > llgrid[...,1].max() + 179): nxt_ll[...,1] -= 180
        nxt_llb = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,nxt_ll.shape[0],dtype=dtype,device=device),
            torch.linspace(-1,1,nxt_ll.shape[1],dtype=dtype,device=device),indexing='ij'
        ),dim=-1)
        mask = (nxt_ll < llgrid.view(-1,2).max(dim=0)[0]).all(dim=-1) + (nxt_ll > llgrid.view(-1,2).min(dim=0)[0]).all(dim=-1)
        mask = mask * (torch.rand(mask.shape) < 2000/mask.sum()) # limit to 2000 randomly sampled GCPs for sampling
        nxt_ll, nxt_llb = nxt_ll[mask], nxt_llb[mask];
        tps.fit(nxt_ll, nxt_llb);
        crgrid = tps.transform(llgrid.view(-1,2)).view(output.shape[1],output.shape[2],2)
        output[i] = F.grid_sample(nxt_dat[None,...], crgrid[None,...], mode='bilinear', align_corners=False)[0,0]
    
    return output

def download(cache_dir="/mnt/d/nextsim_cache", start=2012, end=2020, variables=['vxsi','vysi','siconc','sithick'], cred_file=None, overwrite=False):
    if (cred_file is None):
        cred_file = '/mnt/c/Users/user/Desktop/rnd/AISat/VolumeInversion/torched/.copernicusmarine/.copernicusmarine-credentials'
    dates = []
    for y in range(start, end):
        for m in range(1,13):
            tmp = str(datetime.datetime(y, m, 1))
            dates.append(tmp[:10]+'T'+tmp[11:])
    for date in dates:
        copernicusmarine.subset(
            dataset_id="cmems_mod_arc_phy_my_nextsim_P1M-m",
            variables=variables, #siconc
            minimum_longitude=-180,
            maximum_longitude=179.97000000001367,
            minimum_latitude=49.98,
            maximum_latitude=90.03000000000151,
            start_datetime=date,
            end_datetime=date,
            output_directory=cache_dir,
            credentials_file=pathlib.Path(cred_file),
            overwrite = overwrite,
            skip_existing = ~overwrite
        )