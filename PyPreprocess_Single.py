import numpy as np
import datetime
import scipy
import h5py
import os
import pickle
import netCDF4
import math

# utility funcs
def meansample2d(signal, step_azi=20, step_rng=10):
    tmp = signal
    nb_azi = tmp.shape[0] // step_azi
    nb_rng = tmp.shape[1] // step_rng
    output = np.zeros((nb_azi, nb_rng)).astype(tmp.dtype)
    for i in range(nb_azi):
        for j in range(nb_rng):
            mi,mj = min((i+1)*step_azi, tmp.shape[0]), min((j+1)*step_rng, tmp.shape[1])
            output[i,j] = np.nanmean(tmp[i*step_azi:mi,j*step_rng:mj])
    return output
def import_datasets_h5(fn, str_filter=None, getDict=False):
    f = h5py.File(fn, 'r'); keys = list(f.keys()); nkeys = []; flag = True
    while (flag):
        keys = [[k for k in keys if hasattr(f[k], 'keys')], [k for k in keys if not hasattr(f[k], 'keys')]]
        nkeys = sum([[k1 + '/' + k2 for k2 in f[k1].keys()] for k1 in keys[0]], [])
        flag = len(keys[0]) > 0; keys = nkeys + keys[1]
    if (isinstance(str_filter, list)): keys = [el for el in keys if any(ft in el for ft in str_filter)]
    elif (isinstance(str_filter, str)): keys = [el for el in keys if str_filter in el]
    if (getDict): return {keys[i] : f[keys[i]] for i in range(len(keys))}
    return [f[keys[i]] for i in range(len(keys))]
def iq2calib_alos1(image, incidence_angle, cf_offset=0, output_type='beta'):
    # src https://earth.esa.int/eogateway/documents/20142/37627/ALOS-PALSAR-calibration-products-ADEN.pdf
    dn_sqr = image.imag**2 + image.real**2
    cf = -83.4 #QUADRATURE MODE ONLY
    k = cf - cf_offset
    σ0 = k * dn_sqr
    if (output_type=='s0'): return σ0
    β0 = σ0/np.sin(incidence_angle)
    if (output_type=='beta'): return β0
    γ0 = σ0/np.cos(incidence_angle)
    if (output_type=='gamma'): return γ0
    return
def coh(s1, s2, window_size=1): 
    wind = np.ones((window_size, window_size))/window_size**2
    edge = int(window_size / 2)
    val = np.sqrt(scipy.signal.convolve2d(np.abs(s1)**2, wind, mode='same') * scipy.signal.convolve2d(np.abs(s2)**2, wind, mode='same'))
    #val[np.abs(val) < 1e-12] = 1e-12
    val = scipy.signal.convolve2d(s1 * np.conj(s2), wind, mode='same') / (val + 1e-12)
    val[:edge] = val[-edge:] = 0#np.nan#0
    val[:,:edge] = val[:,-edge:] = 0#np.nan#0
    return val
def stackcoh(s1, s2, window_size=1):
    return np.stack([coh(s1[i], s2[i], window_size) for i in range(s1.shape[0])])


#default single dataset
class Dataset:
    default_label = "Dataset_latest"

    @staticmethod
    def load_cache(cache_dir=None, cache_fn=None):
        if (cache_dir is None): cache_dir = f'{default_label}_cache'
        if (cache_fn is None): cache_fn = 'pdset.pickle'
        fn = os.path.join(cache_dir, cache_fn)
        with open(fn, 'rb') as f: return pickle.load(f)

    def cache(self, cache_fn=None):
        if (cache_fn is None): cache_fn = f'{self.cache_dir}/pdset.pickle'
        with open(cache_fn, 'wb') as f:
            pickle.dump(self, f)

#SNAP
def SNAP_date2ts(datestring):
    try: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
    except: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
def SNAP_unitFixer(meta_tuple):
    data, unit, desc = meta_tuple
    if unit in metric2base.keys(): unit, factor = metric2base[unit]; data *= factor
    elif unit == 'utc': data, unit = SNAP_date2ts(data), 's'
    return data, unit, desc

# base file import & preprocessing w. SNAPISTA
class Dataset_SNAPbase(Dataset):
    default_label="SNAPbase_latest"
    def __init__(self, src_fn=None, cache_dir=None, nc_fn=None, read_nc=True, alos1_deskew=False):
        self.cache_dir = cache_dir
        if (nc_fn is None): nc_fn = 'BEAM.nc'
        self.nc_fn = nc_fn
        if (cache_dir is None): self.cache_dir = f'{self.default_label}_cache'
        if((cache_dir is None) or (not read_nc)):
            self.SnapistaProc(src_fn, nc_fn, alos1_deskew=alos1_deskew)
        elif (nc_fn in os.listdir(self.cache_dir)):
            print(f'located cache at {self.cache_dir}')
        else: raise ValueError(f'could not find cache at {self.cache_dir}.')
        self.ncGet()
        print('dataset loaded.')
        
    def SnapistaProc(self, src_fn, nc_fn, alos1_deskew=False):
        from esa_snappy import snapista as snp
        print ('beginning ESA SNAPPY preprocessing..')
        graph = snp.Graph()
        bnd_used = ['i_HH','q_HH,i_HV,q_HV,i_VH,q_VH,i_VV,q_VV']
        dem_used = 'Copernicus 30m Global DEM'
        graph.add_node(operator=snp.Operator('Read', file=src_fn), node_id = 'read') # Read
        calib = snp.Operator('Calibration', sourceBandNames=bnd_used, outputImageInComplex='true', outputSigmaBand='false', selectedPolarisations='HH,HV,VH,VV')
        if (alos1_deskew):
            deskew = snp.Operator('ALOS-Deskewing', sourceBandNames=bnd_used, demName=dem_used) # Deskew
            graph.add_node(operator = deskew, node_id = 'deskew', source = 'read')
            graph.add_node(operator = calib, node_id = 'calib', source = 'deskew')
        else: graph.add_node(operator = calib, node_id = 'calib', source = 'read')
        write = snp.Operator('Write', file=f'{self.cache_dir}/{nc_fn}', formatName='NETCDF4-BEAM')
        graph.add_node(operator = write, node_id = 'write', source = 'calib')
        try: os.mkdir(self.cache_dir)
        except FileExistsError: print(f'{self.cache_dir} already exists - data will be overwritten.')
        graph.run()

    def ncGet(self, ia=True, rg=False, lat=False, lon=False):
        ncd = netCDF4.Dataset(os.path.join(self.cache_dir, self.nc_fn))
        print('extracting parameters...')
        # Metadata retrieval
        def get_attr(filt, fix_unit=False): 
            attrns = [el for el in ncd.variables['metadata'].ncattrs() if filt in el]
            data, desc, unit = None, None, None
            for i in range(len(attrns)):
                if ('_unit' == attrns[i][-5:]): unit = ncd.variables['metadata'].getncattr(attrns[i])
                elif ('_descr' == attrns[i][-6:]): desc = ncd.variables['metadata'].getncattr(attrns[i])
                else: data = ncd.variables['metadata'].getncattr(attrns[i])
            if (fix_unit): data, unit, desc = SNAP_unitFixer((data,unit,desc))
            return data, unit, desc
        self.metadata = {}

        #radar processing properties
        self.metadata['wv'] = get_attr('wavelength', True)
        self.metadata['freq'] = get_attr('radar_frequency', True)
        self.metadata['rg_bw'] = get_attr('range_bandwidth', True)
        self.metadata['rg_sr'] = get_attr('range_sampling_rate', True)
        self.metadata['rg_sp'] = get_attr('range_spacing', True)
        self.metadata['az_bw'] = get_attr('azimuth_bandwidth', True)
        self.metadata['az_sr'] = get_attr('azimuth_sampling_rate', True)
        self.metadata['az_sp'] = get_attr('azimuth_spacing', True)
        self.metadata['prf'] = get_attr('pulse_repetition_frequency', True)
        self.metadata['0dopp_t0'] = get_attr('first_line_time', True)
        self.metadata['0dopp_t1'] =  get_attr('last_line_time', True)
        self.metadata['inc_near'] = get_attr('incidence_near', True)
        self.metadata['inc_far'] = get_attr('incidence_far', True)
        
        #orbit state vectors
        buffer, labels = [], ['Abstracted_Metadata:Orbit_State_Vectors:orbit_vector','time', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel','z_vel']
        for i in sorted(set([int(el.split('orbit_vector')[1].split(':')[0]) for el in ncd.variables['metadata'].ncattrs() if labels[0] in el])):
            tmp = [ncd.variables['metadata'].getncattr(labels[0] + f'{i}:{el}') for el in labels[1:]]; tmp[0] = SNAP_date2ts(tmp[0])
            tmp = [float(el) for el in tmp]; buffer.append(tmp)
        self.metadata['orbit_vector'] = np.array(buffer); buffer = None
            
        #latitude & longitude corners
        self.metadata['lat'] = [get_attr('first_near_lat')[0], get_attr('first_far_lat')[0], get_attr('last_near_lat')[0], get_attr('last_far_lat')[0]]
        self.metadata['lon'] = [get_attr('first_near_long')[0], get_attr('first_far_long')[0], get_attr('last_near_long')[0], get_attr('last_far_long')[0]]
        #match label of range and azimuth dimensions
        rg = ncd.variables['slant_range_time'][:]
        self.dim_desc = ('band', 'azi', 'rng') if (np.abs(rg[-1,0]-rg[0,0]) < np.abs(rg[0,-1]-rg[0,0])) else ('band','rng','azi')
        
        print('retrieved metadata.')

        slckey = [el for el in ncd.variables.keys() if 'i_' in el or 'q_' in el]
        slckey = [(slckey[i], slckey[i+1]) for i in np.arange(0,len(slckey),2)]
        self.nc_slckey = slckey
        self.slc = np.array([ncd.variables[el[0]][:] + ncd.variables[el[1]][:] * 1j for el in slckey])
        print('retrieved SLC complex (B,H,W).')

        print('rescaling auxiliary data...')
        ia = ncd.variables['incident_angle'][:]
        refmap = np.meshgrid(np.linspace(0,ia.shape[0]-1,self.slc.shape[1]),
                             np.linspace(0,ia.shape[1]-1,self.slc.shape[2]), indexing='ij')
        def rescale_tp2ref(tp): return scipy.ndimage.map_coordinates(tp, refmap)
        if (ia > np.pi/2).any(): ia = np.radians(ia); print('converted incidence angle from degrees to radians.')
        self.ia = rescale_tp2ref(ia)
        self.rg = rescale_tp2ref(ncd.variables['slant_range_time'][:])
        self.lat = rescale_tp2ref(ncd.variables['latitude'][:])
        self.lon = rescale_tp2ref(ncd.variables['longitude'][:])
        print('all parameters extracted.')

    def ncWrite(self, arr, name='newBand'):
        '''
        NOTE - expects SLC band resolution / data size! Also enforces SLC template real-valued datatype.
        '''
        ncd = netCDF4.Dataset(os.path.join(self.cache_dir, self.nc_fn), 'r+')
        if name in ncd.variables.keys():
            ncd.variables[name][:] = arr
        else:
            template_band = ncd.variables[[el for el in ncd.variables.keys() if 'i_' in el or 'q_' in el][0]]
            band = ncd.createVariable(name, template_band.datatype, template_band.dimensions)
            band.setncatts(template_band.__dict__)
            band[:] = arr.astype(template_band.dtype)
        #ncd[''][:] = array

metric2base = {
    'W':    ('W', 1.0),
    'kW':   ('W', 1e3),
    'dB':   ('dB', 1.0),     # keep dimensionless
    'dBHz': ('dBHz', 1.0),   # also dimensionless (log domain)
    
    # Doppler rate, chirp rate, FM rate
    'Hz/s':    ('Hz/s', 1.0),
    'kHz/s':   ('Hz/s', 1e3),
    'Hz/ms':   ('Hz/s', 1e3),
    # PRF sometimes shows up as 1/s
    '1/s':     ('Hz', 1.0),

    'rad':   ('rad', 1.0),
    'deg':   ('rad', math.pi / 180.0),
    'mrad':  ('rad', 1e-3),
    
     # Length
    'm':    ('m', 1.0),
    'km':   ('m', 1e3),
    'cm':   ('m', 1e-2),
    'mm':   ('m', 1e-3),
    'µm':   ('m', 1e-6),
    'nm':   ('m', 1e-9),
    'm/s':   ('m/s', 1.0),
    'km/s':  ('m/s', 1e3),
    'km/h':  ('m/s', 1e3 / 3600),

    # Frequency
    'Hz':   ('Hz', 1.0),
    'kHz':  ('Hz', 1e3),
    'MHz':  ('Hz', 1e6),
    'GHz':  ('Hz', 1e9),

    # Time
    's':    ('s', 1.0),
    'ms':   ('s', 1e-3),
    'us':   ('s', 1e-6),
    'µs':   ('s', 1e-6),
    'ns':   ('s', 1e-9),
}