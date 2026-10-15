import sys, os, datetime

import numpy as np
import scipy
import pandas as pd
import data_aosi

import netCDF4
import h5py
import util_collector

'''
This data access module retrieves sea ice property data from a variety of sources for validation purposes, interpolating as needed for the input coordinates.
'''

def get(name, YYYYMMDD, lat, lon, variable='sit', mode='nearest', hemisphere='NH'):
    match name:
        case 'AMSR_KH': return AMSR_KimuraN_HasumiH(YYYYMMDD, lat, lon, mode)
        case 'AMSR'   : return AMSR_NSDIC(YYYYMMDD, lat, lon, variable, mode, hemisphere)
        case 'SMOS'   : return UBremen_SMOS(YYYYMMDD, lat, lon, mode)
        case 'neXtSIM': return AOSI(YYYYMMDD, lat, lon, mode)
        case 'IC1'    : return ICESat_Gridded(YYYYMMDD, lat, lon, variable, mode)
        case 'CS2'    : return CryoSat2_Data(YYYYMMDD, lat, lon, variable, mode)
    print('data retrieved.')

TITLES = {
    'AMSR_KH'  : 'AMSR Kimura N. and Hasumi H. 10 x 10 km gridded',
    'AMSR'     : 'AMSR-E/AMSR2 Unified L3 Daily 12.5 km gridded',
    'SMOS'     : 'Combined SMAP/SMOS Thin Sea Ice 12.5 x 12.5 km gridded',
    'neXtSIM'  : 'neXtSIM-F 3 x 3 km gridded',
    'IC1'      : 'ICESat/GLAS L3A Seasonal 25 x 25 km gridded',
    'CS2'      : 'CryoSat-2 25 x 25 km gridded'
}

def scipy_interp(data, data_ll, mode='nearest', rbf_neighbors=100):
    ll2data = None
    match mode:
        case 'nearest': ll2data = scipy.interpolate.NearestNDInterpolator(data_ll, data)
        case 'linear' : ll2data = scipy.interpolate.LinearNDInterpolator(data_ll, data)
        case 'rbf'    : ll2data = scipy.interpolate.RBFInterpolator(data_ll, data, neighbors=rbf_neighbors)
    return ll2data

def pytorch_interp(data):
    pass

def wgs84_llh2xyz(lat, lon, h=0): # h : radial height
    a, f = 6378137.0, 1.0 / 298.257223563  # ellipsoid parameters
    e2 = f*(2 - f) # eccentricity squared
    N = (a / np.sqrt(1 - e2*np.sin(np.deg2rad(lat))**2))
    x = (N + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y = (N + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z = (N*(1 - e2) + h)*np.sin(np.deg2rad(lat))
    return x, y, z
    
def lon_wrap(longitudes): # for -180~180 range convention
    if longitudes.max() > 180: longitudes[longitudes > 180] -= 360
    return longitudes

import data_aosi
def AOSI(YYYYMMDD, lat, lon, mode):
    dt_targ = [datetime.datetime.strptime(str(YYYYMMDD), '%Y%m%d')]
    aosi_ds = data_aosi.Dataset('CoreExplorer', variables=['sithick'])
    aosi_ds.download_data(dt_targ)
    return aosi_ds.read_at(np.stack([lat, lon], axis=-1), dt_targ, mode=mode)[0,0]
    
# Kimura, N. and Hasumi, H.: Estimating Arctic sea ice thickness from satellite-based ice history, The Cryosphere, 20, 2331–2349, https://doi.org/10.5194/tc-20-2331-2026, 2026. 
def AMSR_KimuraN_HasumiH(YYYYMMDD, lat, lon, mode='nearest'):
    amsr_kh_fn = f'/mnt/d/AMSR_prod_SIT_07XXXX_11XXXX_KimuraN_HasumiH/sit{str(YYYYMMDD)[:4]}/sit{str(YYYYMMDD)[2:]}nh.txt'
    amsr_kh_sit = np.array([[float(el) for el in line.split()] for line in open(amsr_kh_fn, 'r')]) #N, d(lat, lon, sit)
    amsr_kh_sit = np.stack(list(wgs84_llh2xyz(amsr_kh_sit[:,0], amsr_kh_sit[:,1])) + [amsr_kh_sit[:,-1]], axis=-1) # N,d(x,y,z,sit)
    amsr_kh_sit = scipy_interp(amsr_kh_sit[:,-1], amsr_kh_sit[:,:-1], mode=mode)
    amsr_kh_sit = amsr_kh_sit(np.stack(wgs84_llh2xyz(lat.flatten(), lon.flatten()), axis=-1)).reshape(lat.shape)
    return amsr_kh_sit

# Meier, W. N., Markus, T. & Comiso, J. C. (2018). AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion & Snow Depth Polar Grids. (AU_SI12, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/RA1MIJOYPK3P. [describe subset used if applicable]. Date Accessed 08-13-2026.
def AMSR_NSDIC(YYYYMMDD, lat, lon, variable='sic', mode='nearest', hemisphere='NH'):
    amsr_h5f = h5py.File(f'/mnt/d/AMSR_data_060516_110512/AMSR_UE_L3_SeaIce12km_B04_{YYYYMMDD}.he5')
    match variable:
        case 'sic':  key = f'SI_12km_{hemisphere}_ICECON_DAY'
        case 'sdp': key = f'SI_12km_{hemisphere}_SNOWDEPTH_5DAY'
        case 'diff': key = f'SI_12km_{hemisphere}_ICEDIFF_DAY'
        case None: raise ValueError('invalid variable')
    gridn = f'{hemisphere[0]}pPolarGrid12km'
    dlat, dlon = amsr_h5f['HDFEOS']['GRIDS'][gridn]['lat'][:], amsr_h5f['HDFEOS']['GRIDS'][gridn]['lon'][:]
    ddat = amsr_h5f['HDFEOS']['GRIDS'][gridn]['Data Fields'][key][:]
    amsr_h5f.close()
    dat = scipy_interp(ddat.flatten(), np.stack(wgs84_llh2xyz(dlat.flatten(), dlon.flatten()), axis=-1), mode=mode)
    dat = dat(np.stack(wgs84_llh2xyz(lat.flatten(), lon.flatten()), axis=-1)).reshape(lat.shape)
    return dat

# Patilea, C., G. Heygster, M. Huntemann, & G. Spreen (2019). Combined SMAP/SMOS Thin Sea Ice Thickness Retrieval. The Cryosphere, 13, 675–691. doi:10.5194/tc-13-675-2019
# .. Access @ https://seaice.uni-bremen.de/thin-ice-thickness/
def UBremen_SMOS(YYYYMMDD, lat, lon, mode='nearest'):
    smos_prod = netCDF4.Dataset('/mnt/d/UBremen_SeaIce_Data/SMOS_SIT/LongitudeLatitudeGrid-n12500-Arctic.hdf')
    dlat, dlon = smos_prod['Latitudes'][:], smos_prod['Longitudes'][:]
    smos_prod.close();
    smos_prod = netCDF4.Dataset(f'/mnt/d/UBremen_SeaIce_Data/SMOS_SIT/{YYYYMMDD}_hvnorth_rfi_l1c.nc')
    ddat = smos_prod.variables['thickness'][:] * 0.01 # convert cm to m
    smos_prod.close();
    dat = scipy_interp(ddat.flatten(), np.stack(wgs84_llh2xyz(dlat.flatten(), dlon.flatten()), axis=-1), mode=mode)
    dat = dat(np.stack(wgs84_llh2xyz(lat.flatten(), lon.flatten()), axis=-1)).reshape(lat.shape)
    return dat

# Spreen, G., L. Kaleschke, and G.Heygster (2008), Sea ice remote sensing using AMSR-E 89 GHz channels J. Geophys. Res.,vol. 113, C02S03, doi:10.1029/2005JC003384. 
# .. Access @ https://seaice.uni-bremen.de/sea-ice-concentration/amsre-amsr2/
def UBremen_SIC(YYYYMMDD):
    pass

#Kurtz, N. & Harbeck, J. (2017). CryoSat-2 Level-4 Sea Ice Elevation, Freeboard, and Thickness. (RDEFT4, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/96JO0KIFDAS8. [describe subset used if applicable]. Date Accessed 08-19-2026.
def CryoSat2_Data(YYYYMMDD, lat, lon, variable='sit', mode='nearest'):
    cs2_prod = netCDF4.Dataset(f'/mnt/d/CryoSat2_data_100920_110512/RDEFT4_{YYYYMMDD}.nc')
    dlat, dlon = cs2_prod['lat'][:], cs2_prod['lon'][:]
    #dlon[dlon > 180] = dlon[dlon > 180] - 360;  # rewrap 0~360 to -180~180
    match variable:
        case 'sit': ddat = cs2_prod['sea_ice_thickness'][:]
        case 'sic': ddat = cs2_prod['ice_con'][:]
        case 'sdp': ddat = cs2_prod['snow_depth'][:]
        case 'sdn': ddat = cs2_prod['snow_density'][:]
        case 'frb': ddat = cs2_prod['freeboard'][:]
        case 'rgh': ddat = cs2_prod['roughness'][:]
    cs2_prod.close()
    dmask = ddat != -9999.0  # ignore fill values
    if mode == 'nearest': ddat[~dmask] = 0
    else: dlat, dlon, ddat = dlat[dmask], dlon[dmask], ddat[dmask]
    dat = scipy_interp(ddat.flatten(), np.stack(wgs84_llh2xyz(dlat.flatten(), dlon.flatten()), axis=-1), mode=mode)
    dat = dat(np.stack(wgs84_llh2xyz(lat.flatten(), lon.flatten()), axis=-1)).reshape(lat.shape)
    return dat

# Yi, D. & Zwally, H. J. (2009). Arctic Sea Ice Freeboard and Thickness. (NSIDC-0393, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/SXJVJ3A2XIZT. Date Accessed 08-19-2026.
def Compile_ICESat_Tracks():
    dn = "/mnt/d/ICESat1_data_060222_080321/NSIDC0393_GLAS_SI_Freeboard_v01/"
    for dn1 in os.listdir(os.path.join(dn, 'glas_seaice_vectors')):
        for fn in os.listdir(os.path.join(dn, 'glas_seaice_vectors', dn1)):
            fn_ = os.path.join(dn, 'glas_seaice_vectors', dn1, fn)
            icesat_df, mdt = pd.read_csv(fn_, sep=r"\s+", skiprows=22), {}
            for line in open(fn_, 'r'):
                if line.startswith("Latitude"): break
                if ": " not in line: continue
                key, value = (x.strip() for x in line.split(":", 1)); mdt[key] = value
            dt = datetime.datetime(int(mdt['Year']), int(mdt['Month']), int(mdt['Day']), int(mdt['Hour']), int(mdt['Minute']))
            icesat_df['timestamp'] = dt.timestamp()
            print(f'writing to {dt.strftime('%Y%m%d')}')
            wf = util_collector.Dataset(os.path.join(dn, f'daily/{dt.strftime('%Y%m%d')}.hdf5'), increment=1)
            if not (dt.timestamp() == wf.h5f['timestamp']['d'][:]).any():
                wf.add(**{column:icesat_df[column].values for column in icesat_df.columns})
            wf.close()
    print('completed')
def ICESat_Tracks_loc(YYYYMMDD, sr=1):
    prod = h5py.File(f'/mnt/d/ICESat1_data_060222_080321/NSIDC0393_GLAS_SI_Freeboard_v01/daily/{YYYYMMDD}.hdf5')
    lat, lon, ts = prod['Latitude']['d'][::sr], prod['Longitude']['d'][::sr], prod['timestamp']['d'][::sr]
    tracks = []; prod.close()
    return pd.DataFrame(np.stack([lat, lon, ts], axis=-1)[:,0,:], columns=['lat', 'lon', 'timestamp'])
def ICESat_Tracks(YYYYMMDD, lat, lon, mode='nearest'):
    prod = h5py.File(f'/mnt/d/ICESat1_data_060222_080321/NSIDC0393_GLAS_SI_Freeboard_v01/daily/{YYYYMMDD}.hdf5')
    dlat, dlon, dtime, ddat = prod['Latitude']['d'][:], prod['Longitude']['d'][:], prod['timestamp']['d'][:], prod['Thickness']['d'][:]
    prod.close()
    time = datetime.datetime.strptime('%Y%m%d').timestamp()
    dat = scipy_interp(ddat, np.stack(list(wgs84_llh2xyz(dlat, dlon)) + [dtime], axis=-1), mode=mode)
    dat = dat(np.stack(list(wgs84_llh2xyz(lat.flatten(), lon.flatten())) + [np.full_like(lon.flatten(), time)], axis=-1)).reshape(lat.shape)
    return dat

# Petty, A. A. & Kurtz, N. (2023). ICESat L4 Seasonal Gridded Sea Ice Thickness. (ISSITGR4, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/1S5M59IQ0OK3. [describe subset used if applicable]. Date Accessed 08-19-2026.
def ICESat_Gridded(YYYYMMDD, lat, lon, variable='sit', mode='nearest'):
    dn, dt, flag = '/mnt/d/ICESat1_data_060222_080321/SEASONAL_GRIDDED', datetime.datetime.strptime(str(YYYYMMDD), '%Y%m%d'), False
    variable = {
        'sit':'ice_thickness_int', 'sit_r':'ice_thickness', 'sit_u': 'ice_thickness_unc',
        'frb':'total_freeboard_int', 'frd_r':'total_freeboard', 'frd_u':'total_freeboard_unc',
        'sdp':'snow_depth_int', 'sdp_r':'snow_depth', 'sdn':'snow_density',
        'sic':'sea_ice_conc',
        'type':'ice_type'
    }[variable]
    for fn in os.listdir(dn):
        if fn[-3:] != '.nc': continue
        prod = netCDF4.Dataset(os.path.join(dn, fn))
        if dt < datetime.datetime.strptime(prod['campaign_dates'].getncattr('first_day'), '%Y-%m-%d'): continue
        if dt > datetime.datetime.strptime(prod['campaign_dates'].getncattr('last_day' ), '%Y-%m-%d'): continue
        flag = True
    if not flag : prod.close(); raise ValueError('input date not within dataset bounds')
    dlat, dlon, ddat = prod['latitude'], prod['longitude'], prod[variable]
    prod.close()
    dat = scipy_interp(ddat.flatten(), np.stack(wgs84_llh2xyz(dlat.flatten(), dlon.flatten()), axis=-1), mode=mode)
    dat = dat(np.stack(wgs84_llh2xyz(lat.flatten(), lon.flatten()), axis=-1)).reshape(lat.shape)
    return dat


# DAY - Hall, D. K. & Riggs, G. A. (2021). MODIS/Terra Sea Ice Extent Daily L3 Global 1km EASE-Grid Night. (MOD29P1N, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD29P1N.061. [describe subset used if applicable]. Date Accessed 08-18-2026.
# NIGHT - Hall, D. K. & Riggs, G. A. (2021). MODIS/Terra Sea Ice Extent Daily L3 Global 1km EASE-Grid Day. (MOD29P1D, Version 61). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD29P1D.061. [describe subset used if applicable]. Date Accessed 08-18-2026.
def MODIS_1km(YYYYMMDD, lat, lon, mode='nearest'):
    pass


# Olsen, Ida Lundtorp; Skourup, Henriette (2026). Climate Change Initiative sea ice thickness Round Robin Data Package (CCI SIT RRDP). Technical University of Denmark. Dataset. https://doi.org/10.11583/DTU.24787341.






'''
a download script scrap for UBremen_SMOS:
import subprocess, datetime
grabdate = datetime.datetime(2010, 5, 1)
while grabdate < datetime.datetime(2011, 5, 12):
    #print(int())
    if f'{grabdate.strftime('%Y%m%d')}_hvnorth_rfi_l1c.nc' in os.listdir('/mnt/d/UBremen_SeaIce_Data/SMOS_SIT'):
        print('already here!')
        grabdate += datetime.timedelta(days=1)
        continue
    subprocess.run([
        "wget", '-P', '/mnt/d/UBremen_SeaIce_Data/SMOS_SIT',
        f'https://data.seaice.uni-bremen.de/smos/ncs/{grabdate.strftime('%Y%m%d')}_hvnorth_rfi_l1c.nc',
    ], check=False)
    grabdate += datetime.timedelta(days=1)
'''
    