import numpy as np
import datetime
import scipy
import skimage
import h5py
import os
import pickle
import netCDF4
import PolarDecomp

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
class Single_Dataset:
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
import unit_lookup
def SNAP_date2ts(datestring):
    try: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
    except: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
def SNAP_unitFixer(meta_tuple):
    data, unit, desc = meta_tuple
    if unit in unit_lookup.metric2base.keys(): unit, factor = unit_lookup.metric2base[unit]; data *= factor
    elif unit == 'utc': data, unit = SNAP_date2ts(data), 's'
    return data, unit, desc

# base file import & preprocessing w. SNAPISTA
class Single_Dataset_SNAPbase(Single_Dataset):
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

    def ncGet(self):
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
        print('retrieved SLC complex.')

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

# default dataset
class Pair_Dataset:
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

    def gen_cohs(self, window_size=3):
        '''Calculates the interferometric complex coherence of a pair dataset and assigns it to the variable 'cohs'.'''
        self.cohs = np.zeros(self.s1.shape, dtype=np.complex128)
        for i in range(self.cohs.shape[0]):
            self.cohs[i] = coh(self.s1[i], self.s2[i], window_size)

    def synthesize_bands(self, pauli=True, circu=True, pd_opt=True):
        '''Available bands: Pauli 4-basis, Rotational 4-basis, Phase Difference (PD) Optimized 2-basis coherence'''
        return


# SNAP
# base file import & preprocessing w. SNAPISTA
class Pair_Dataset_SNAPbase(Pair_Dataset):
    default_label="SNAPbase_latest"
    
    def __init__(self, s1_fn=None, s2_fn=None, cache_dir=None, nc_fn=None, read_nc=True, is_alos1=True):
        self.cache_dir = cache_dir
        self.is_alos1 = is_alos1
        if (nc_fn is None): nc_fn = 'BEAM.nc'
        self.nc_fn = nc_fn
        if (cache_dir is None): self.cache_dir = f'{self.default_label}_cache'
        if((cache_dir is None) or (not read_nc)):
            self.SnapistaProc(s1_fn, s2_fn, nc_fn)
        elif (nc_fn in os.listdir(self.cache_dir)):
            print(f'located cache at {self.cache_dir}')
        else: raise ValueError(f'could not find cache at {self.cache_dir}.')
        self.ncGet()
        print('dataset loaded.')

    def _get_snapista_calib_graph(self, s1_fn, s2_fn):
        from esa_snappy import snapista as snp
        graph = snp.Graph()
        bnd_used = ['i_HH','q_HH,i_HV,q_HV,i_VH,q_VH,i_VV,q_VV']
        dem_used = 'Copernicus 30m Global DEM'
        graph.add_node(operator=snp.Operator('Read', file=s1_fn), node_id = 'read_s1') # Read
        graph.add_node(operator=snp.Operator('Read', file=s2_fn), node_id = 'read_s2')
        calib = snp.Operator('Calibration', sourceBandNames=bnd_used, outputImageInComplex='true', outputSigmaBand='false', selectedPolarisations='HH,HV,VH,VV')
        if (self.is_alos1):
            deskew = snp.Operator('ALOS-Deskewing', sourceBandNames=bnd_used, demName=dem_used) # Deskew
            graph.add_node(operator = deskew, node_id = 'deskew_s1', source = 'read_s1')
            graph.add_node(operator = deskew, node_id = 'deskew_s2', source = 'read_s2')
            graph.add_node(operator = calib, node_id = 'calib_s1', source = 'deskew_s1')
            graph.add_node(operator = calib, node_id = 'calib_s2', source = 'deskew_s2')
        else:
            graph.add_node(operator = calib, node_id = 'calib_s1', source = 'read_s1')
            graph.add_node(operator = calib, node_id = 'calib_s2', source = 'read_s2')
        return graph
    
    def SnapistaProc(self, s1_fn, s2_fn, nc_fn):
        from esa_snappy import snapista as snp
        print ('beginning ESA SNAPPY preprocessing..')
        graph = self._get_snapista_calib_graph(s1_fn, s2_fn)
        oper1 = snp.Operator('CreateStack', resamplingType='NEAREST_NEIGHBOUR', initialOffsetMethod='Product Geolocation')
        oper2 = snp.Operator('BandSelect', bandNamePattern='^._([HVRL]{2})_.*') # follows pattern i_??_...
        oper3 = snp.Operator('Write', file=f'{self.cache_dir}/{nc_fn}', formatName='NETCDF4-BEAM')
        graph.add_node(operator = oper1, node_id = 'stk_op1', source = ['calib_s1','calib_s2'])
        graph.add_node(operator = oper2, node_id = 'stk_op2', source = 'stk_op1')
        graph.add_node(operator = oper3, node_id = 'stk_write', source = 'stk_op2')
        try: os.mkdir(self.cache_dir)
        except FileExistsError: print(f'{self.cache_dir} already exists - data will be overwritten.')
        graph.run()

    def ncGet(self):
        ncd = netCDF4.Dataset(os.path.join(self.cache_dir, self.nc_fn))
        print('extracting parameters...')
        # Metadata retrieval
        baseline_attr = [el for el in ncd.variables['metadata'].ncattrs() if 'Perp_Baseline' in el][1]
        self.baseline = ncd.variables['metadata'].getncattr(baseline_attr)
        wavelength_attr = [el for el in ncd.variables['metadata'].ncattrs() if 'wavelength' in el][0]
        self.wavelength = ncd.variables['metadata'].getncattr(wavelength_attr)
        print('retrieved metadata.')
        kz = 4*np.pi*self.baseline / (self.wavelength*np.sin(np.radians(ncd['incident_angle'][:]))*ncd['slant_range_time'][:])
        print('calculated vertical wavenumber.')

        s1key = [el for el in ncd.variables.keys() if '_mst' in el and 'Intensity' not in el]
        s1key = [(s1key[i],s1key[i+1]) for i in np.arange(0, len(s1key), 2)]
        s2key = [el for el in ncd.variables.keys() if '_slv' in el and 'Intensity' not in el]
        s2key = [(s2key[i],s2key[i+1]) for i in np.arange(0, len(s2key), 2)]
        self.nc_s1key = s1key
        self.nc_s2key = s2key
        self.s1 = np.array([ncd.variables[el[0]][:] + ncd.variables[el[1]][:] * 1j for el in s1key])
        self.s2 = np.array([ncd.variables[el[0]][:] + ncd.variables[el[1]][:] * 1j for el in s2key])
        print('retrieved SLC complex.')

        print('rescaling auxiliary data...')
        refmap = np.meshgrid(np.linspace(0,kz.shape[0]-1,self.s1.shape[1]),
                             np.linspace(0,kz.shape[1]-1,self.s1.shape[2]), indexing='ij')
        def rescale_tp2ref(tp): return scipy.ndimage.map_coordinates(tp, refmap)
        self.kz = rescale_tp2ref(kz)
        self.ia = rescale_tp2ref(ncd.variables['incident_angle'][:])
        self.rg = rescale_tp2ref(ncd.variables['slant_range_time'][:])
        self.lat = rescale_tp2ref(ncd.variables['latitude'][:])
        self.lon = rescale_tp2ref(ncd.variables['longitude'][:])
        print('all parameters extracted.')

# pythonic coregistration pipeline
class Pair_DatasetSNAPPyCorgi(Pair_Dataset_SNAPbase):
    default_label="SNAPPyCorgi"

    # src - https://ieeexplore.ieee.org/document/8766118 "Fast and Robust Matching for Multimodal Remote Sensing Image Registration"
    # src - https://ieeexplore.ieee.org/document/10758293 "PolSAR Image Registration Using Orientated Gradients of Polarimetric Features"
    def ncGet(self):
        super().ncGet()
        try:
            if(self._corgi_beamerized):
                print('retrieved .nc stack is coregistered - skipping python coregistration.')
                return
        except NameError:
            self._corgi_beamerized = False
        print('beginning coregistration...')
        #self.
        
        
        
# graph processing incorporates coregistration
class Pair_Dataset_SNAPNative(Pair_Dataset_SNAPbase):
    default_label="SNAPNative_latest"

    def SnapistaProc(self, s1_fn, s2_fn, nc_fn):
        from esa_snappy import snapista as snp
        print ('beginning ESA SNAPPY preprocessing..')
        graph = self._get_snapista_calib_graph(s1_fn, s2_fn)
        oper1 = snp.Operator('CreateStack', resamplingType='NEAREST_NEIGHBOUR', initialOffsetMethod='Product Geolocation') #NEAREST_NEIGHBOUR, Orbit
        oper2 = snp.Operator('Cross-Correlation');  oper3 = snp.Operator('Warp')
        graph.add_node(operator = oper1, node_id = 'stk_op1', source = ['calib_s1','calib_s2'])
        graph.add_node(operator = oper2, node_id = 'stk_op2', source = 'stk_op1')
        graph.add_node(operator = oper3, node_id = 'stk_op3', source = 'stk_op2')
        oper1 = snp.Operator('BandSelect', bandNamePattern='^._([HVRL]{2})_.*') # follows pattern i_??_...
        oper2 = snp.Operator('Write', file=f'{self.cache_dir}/{nc_fn}', formatName='NETCDF4-BEAM')
        graph.add_node(operator = oper1, node_id = 'stk_op4', source = 'stk_op3')
        graph.add_node(operator = oper2, node_id = 'stk_write', source = 'stk_op4')
        try: os.mkdir(self.cache_dir)
        except FileExistsError: print(f'{self.cache_dir} already exists - data will be overwritten.')
        graph.run()



#MISC        
# Conversion of SNAP-exported data in .h5 form - physical defaults set to ALOS-1 PALSAR               
class Pair_Dataset_SNAPh5(Pair_Dataset):
    default_label="SNAPh5_latest"
    
    def __init__(self, fn_1, fn_2, step_azi=20, step_rng=10, wavelength=0.235, drift=None, rubbersheet=True):
        s1_src_i = import_datasets_h5(fn_1, ["bands/i_"])
        s1_src_q = import_datasets_h5(fn_1, ["bands/q_"])
        s2_src_i = import_datasets_h5(fn_2, ["bands/i_"])
        s2_src_q = import_datasets_h5(fn_2, ["bands/q_"])
        
        ia1_src = import_datasets_h5(fn_1, "incident_angle")[0]
        ia2_src = import_datasets_h5(fn_2, "incident_angle")[0]

        rt1_src = import_datasets_h5(fn_1, "slant_range_time")[0]
        rt2_src = import_datasets_h5(fn_2, "slant_range_time")[0]
        
        ll1_src = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_1, "itude")}
        ll2_src = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_2, "itude")}
    
        lat1, lon1 = np.array(ll1_src['latitude']), np.array(ll1_src['longitude'])
        lat2, lon2 = np.array(ll2_src['latitude']), np.array(ll2_src['longitude'])

        # yey, signals!
        s1 = np.array([meansample2d(el, step_azi, step_rng) for el in np.array(s1_src_i) + 1j*np.array(s1_src_q)])
        s2 = np.array([meansample2d(el, step_azi, step_rng) for el in np.array(s2_src_i) + 1j*np.array(s2_src_q)])
                
        # ------ Prepare coregistration ------ #
        
        lat1, lon1 = np.array(ll1_src['latitude']), np.array(ll1_src['longitude'])
        lat2, lon2 = np.array(ll2_src['latitude']), np.array(ll2_src['longitude'])
        #map s1 pixel to s1 lat/lon
        tmp1, tmp2 = np.meshgrid(np.linspace(0,s1.shape[1]-1,lat1.shape[0]), np.linspace(0,s1.shape[2]-1,lat1.shape[1]), indexing='ij')
        conv1 = (scipy.interpolate.LinearNDInterpolator(np.stack((tmp2.ravel(),tmp1.ravel()),axis=1),lat1.ravel()),
                 scipy.interpolate.LinearNDInterpolator(np.stack((tmp2.ravel(),tmp1.ravel()),axis=1),lon1.ravel()))
        # map s2 lat/lon to s2 pixel
        tmp1, tmp2 = np.meshgrid(np.linspace(0,s2.shape[1]-1,lat2.shape[0]), np.linspace(0,s2.shape[2]-1,lat2.shape[1]), indexing='ij')
        conv2 = (scipy.interpolate.LinearNDInterpolator(np.stack([lat2.ravel(),lon2.ravel()],axis=1), tmp1.ravel()),
                 scipy.interpolate.LinearNDInterpolator(np.stack([lat2.ravel(),lon2.ravel()],axis=1), tmp2.ravel()))
        # s1 pixel -> s1 lat/lon -> s2 pixel
        map122 = np.meshgrid(np.arange(s1.shape[1]), np.arange(s1.shape[2]), indexing='ij')
        map122 = (conv1[0](np.stack((map122[1].ravel(), map122[0].ravel()),axis=1)),  # lat
                  conv1[1](np.stack((map122[1].ravel(), map122[0].ravel()),axis=1)))  # lon
        map122 = (conv2[0](np.stack([map122[0].ravel(),map122[1].ravel()],axis=1)).reshape(s1.shape[1:]),
                  conv2[1](np.stack([map122[0].ravel(),map122[1].ravel()],axis=1)).reshape(s1.shape[1:]))
        self.map122 = map122

        # ------ Perform coregistration (geolocate, rubbersheet) ------ #

        s2o = np.array([scipy.ndimage.map_coordinates(s2[i],np.array(map122)) for i in range(s2.shape[0])])
        s1o = s1.copy(); s1o[:,s2o[0]==0] = 0
        
        # get cropping slicer in s1 space
        tmp = s2o[0] != 0
        trim11, trim12 = scipy.stats.mode(tmp.argmax(axis=0))[0], scipy.stats.mode(tmp.shape[0] - tmp[::-1,:].argmax(axis=0))[0]
        trim21, trim22 = scipy.stats.mode(tmp.argmax(axis=1))[0], scipy.stats.mode(tmp.shape[1] - tmp[:,::-1].argmax(axis=1))[0]
        self.overlap_slice = (trim11,trim12,trim21,trim22)
        
        s1o, s2o = s1o[:, trim11:trim12,trim21:trim22], s2o[:, trim11:trim12,trim21:trim22]

        if (drift is not None):
            print('sorry, nothing here yet...')
            #drift = 
        
        if (rubbersheet):
            flow = skimage.registration.optical_flow_ilk(np.abs(s2o[0]), np.abs(s1o[0]), radius=40, num_warp=20)
            tmp = np.meshgrid(np.arange(s1o.shape[1]), np.arange(s1o.shape[2]), indexing='ij');
            tmp = np.array([tmp[0]+flow[0], tmp[1]+flow[1]])
            tmp2 = np.array([scipy.ndimage.map_coordinates(s2o[i].real, tmp).astype(complex) +
                            scipy.ndimage.map_coordinates(s2o[i].imag, tmp) * 1j for i in range(s1.shape[0])])
            s2o = tmp2

        self.s1 = s1o
        self.s2 = s2o

        # ------ Coregister & scale auxiliary variables ------ #

        c = 299792458
        def rescale_tp2ref(ref, tp):
            tmp = np.meshgrid(np.linspace(0,tp.shape[0]-1,ref.shape[0]), np.linspace(0,tp.shape[1]-1,ref.shape[1]), indexing='ij')
            return scipy.ndimage.map_coordinates(tp, tmp)
        
        tmp = np.array(np.meshgrid(np.arange(s1o.shape[1]), np.arange(s1o.shape[2]), indexing='ij')).astype(float)
        if (rubbersheet): tmp += np.array(flow)[:2]
        map122_wobbled = np.array([scipy.ndimage.map_coordinates(el[trim11:trim12,trim21:trim22], tmp) for el in map122])
        self.map122 = map122_wobbled #save coordinate transform for accessibility
        
        self.ia1 = rescale_tp2ref(s1[0], np.radians(np.array(ia1_src)))[trim11:trim12,trim21:trim22]
        self.ia2 = scipy.ndimage.map_coordinates(rescale_tp2ref(s2[0], np.radians(np.array(ia2_src))), map122_wobbled)

        #note - check rng correctness as needed : ALOS-1 PALSAR orbital height is 691.65 km
        self.rg1 = rescale_tp2ref(s1[0], np.array(rt1_src) * 1e-9/2 * c)[trim11:trim12,trim21:trim22]
        self.rg2 = scipy.ndimage.map_coordinates(rescale_tp2ref(s2[0], np.array(rt2_src) * 1e-9/2 * c), map122_wobbled)
        
        self.lat = rescale_tp2ref(s1[0], lat1)[trim11:trim12,trim21:trim22]
        self.lon = rescale_tp2ref(s1[0], lon1)[trim11:trim12,trim21:trim22]

        #-- assuming no squint and same incidence plane (e.g. flat earth incidence angle)
        self.k_z = np.abs((4*np.pi / wavelength) * (self.rg1 * np.cos(self.ia1) - self.rg2 * np.cos(self.ia2)) / self.rg1) 

# Conversion of NISAR .h5 data
class Pair_Dataset_NISAR(Pair_Dataset):
    default_label="NISAR_latest"

    def __init__(self, fn_1, fn_2, step_azi=20, step_rng=10):
        self.ds_fill = -1e12

        s1_src = import_datasets_h5(fn_1, ["swaths/frequencyB/H", "swaths/frequencyB/V"])
        s2_src = import_datasets_h5(fn_2, ["swaths/frequencyB/H", "swaths/frequencyB/V"])
        
        gl1 = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_1, "metadata/geolocationGrid/")}
        gl2 = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_2, "metadata/geolocationGrid/")}
        
        ob1 = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_1, "metadata/orbit/")}
        ob2 = {el.name[el.name.rfind('/')+1:] : el for el in import_datasets_h5(fn_2, "metadata/orbit/")}

        # ---- orbital height bin interpolation
        
        t1t, t1s, tmp, hae_r1 = np.array(gl1['zeroDopplerTime']), np.array(ob1['time']), np.array(ob1['position']), np.array(gl1['heightAboveEllipsoid'])
        sp1 = np.array([np.interp(t1t, t1s, tmp[:,i]) for i in range(tmp.shape[1])])
        # WGS84 src : https://topex.ucsd.edu/geodynamics/14gravity1_2.pdf
        shae_r1 = np.sum(sp1**2,axis=0)**0.5; shae_r1 = (shae_r1**2 / (np.sum(sp1[(0,1)]**2, axis=0) / 6378137**2 + sp1[2]**2 / 6356752.3**2))**0.5
        
        t2t, t2s, tmp, hae_r2 = np.array(gl2['zeroDopplerTime']), np.array(ob2['time']), np.array(ob2['position']), np.array(gl2['heightAboveEllipsoid'])
        sp2 = np.array([np.interp(t2t, t2s, tmp[:,i]) for i in range(tmp.shape[1])])
        shae_r2 = np.sum(sp2**2,axis=0)**0.5; shae_r2 = (shae_r2**2 / (np.sum(sp2[(0,1)]**2, axis=0) / 6378137**2 + sp2[2]**2 / 6356752.3**2))**0.5

        def hgt_reshape(value, hgt_bins, hgt_ref, interp_sample_step=10):
            ia = [el.ravel() for el in np.meshgrid(hgt_bins, np.arange(value.shape[1]), np.arange(value.shape[2]), indexing='ij')]
            vval = value.copy(); vval[vval==ds_fill] = 0
            ia = np.stack(ia + [vval.ravel()], axis=1)
            ia = ia[np.arange(0, ia.shape[0], interp_sample_step)];
            interp = scipy.interpolate.NearestNDInterpolator(ia[:, :-1], ia[:, -1])
        
            ib = [el.ravel() for el in np.meshgrid(np.arange(value.shape[1]), np.arange(value.shape[2]), indexing='ij')]
            ib = np.stack([np.fromiter(map(lambda x: hgt_ref[x], ib[0]), dtype=float)] + ib, axis=1)
            ib = interp(ib).reshape(value.shape[1:]); x,y,mask = np.arange(ib.shape[0]), np.arange(ib.shape[1]), ib==0
            ib = scipy.interpolate.RectBivariateSpline(x, y, ib, s=1)(x, y); ib[mask] = 0
            return ib
        def hgtfix_g1(value, interp_sample_step=10): return hgt_reshape(value, hae_r1, shae_r1, interp_sample_step)
        def hgtfix_g2(value, interp_sample_step=10): return hgt_reshape(value, hae_r2, shae_r2, interp_sample_step)

        # ---- swath azimuth/range remapping
        
        s1dt = np.array(import_datasets_h5(fn_1, "swaths/zeroDopplerTime")[0])
        s1sr = np.array(import_datasets_h5(fn_1, "swaths/frequencyB/slantRange")[0])
        s2dt = np.array(import_datasets_h5(fn_2, "swaths/zeroDopplerTime")[0])
        s2sr = np.array(import_datasets_h5(fn_2, "swaths/frequencyB/slantRange")[0])
        g1sr, g2sr = np.array(gl1['slantRange']), np.array(gl2['slantRange'])
        def scale_g2s(gval, g_dt, g_sr, s_dt, s_sr): return scipy.interpolate.RectBivariateSpline(g_dt, g_sr, gval, s=1)(s_dt, s_sr)
        def scale_g2s1(gval): return scale_g2s(gval, t1t, g1sr, s1dt, s1sr)
        def scale_g2s2(gval): return scale_g2s(gval, t2t, g2sr, s2dt, s2sr)

        # --      incidence angle calculation
        
        lk1 = np.array([hgtfix_g1(np.array(gl1['losUnitVectorX'])), hgtfix_g1(np.array(gl1['losUnitVectorY'])), hgtfix_g1(np.array(gl1['elevationAngle']))])
        lk1[2] = np.sin(np.radians(lk1[2])) * (lk1[0]**2 + lk1[1]**2)**0.5; tmp = np.sum(lk1**2, axis=0)**0.5; tmp[tmp==0]=1; lk1 /= tmp
        tp1 = np.array([hgtfix_g1(np.array(gl1['coordinateX'])), hgtfix_g1(np.array(gl1['coordinateY']))])
        
        lk2 = np.array([hgtfix_g2(np.array(gl2['losUnitVectorX'])), hgtfix_g2(np.array(gl2['losUnitVectorY'])), hgtfix_g2(np.array(gl2['elevationAngle']))]);
        lk2[2] = np.sin(np.radians(lk2[2])) * (lk2[0]**2 + lk2[1]**2)**0.5; tmp = np.sum(lk2**2, axis=0)**0.5; tmp[tmp==0]=1; lk2 /= tmp
        tp2 = np.array([hgtfix_g2(np.array(gl2['coordinateX'])), hgtfix_g2(np.array(gl2['coordinateY']))])
        
        ## tp1 and tp2 are in the form (i, j, dimensions)
        interps = [np.concatenate((tp2, lk2[i][np.newaxis])).reshape((tp2.shape[0]+1, tp2.shape[1]*tp2.shape[2])) for i in range(lk2.shape[0])]
        interps = [scipy.interpolate.LinearNDInterpolator(el[:2].T, el[2]) for el in interps]
        lk2 = np.array([el(tp1.reshape((tp1.shape[0], tp1.shape[1]*tp1.shape[2])).T).reshape(lk1[0].shape) for el in interps]); lk2[lk2==0] = np.nan
        
        incidence_angle_delta, tmp = np.sum(lk1*lk2, axis=0), (np.sum(lk1**2,axis=0)**0.5 * np.sum(lk2**2,axis=0)**0.5)
        tmp[tmp==0] = 1; tmp[tmp>0] = 1; incidence_angle_delta[incidence_angle_delta==0] = 1;
        incidence_angle_delta = np.arccos(incidence_angle_delta / tmp)
        incidence_angle_delta[np.isnan(incidence_angle_delta)] = 0
        
        incidence_angle = hgtfix_g1(np.array(gl1['incidenceAngle']));
        incidence_angle[incidence_angle==ds_fill] = 0
        incidence_angle = np.radians(incidence_angle)
        
        x, y = np.arange(incidence_angle.shape[0]), np.arange(incidence_angle.shape[1])
        interp = scipy.interpolate.RectBivariateSpline(x, y, incidence_angle, s=8)
        incidence_angle = interp(x, y)

        # ---- retrieve swath SLC 
        
        s1 = np.array([meansample2d(el,step_azi,step_rng) for el in s1_src])
        s2 = np.array([meansample2d(el,step_azi,step_rng) for el in s2_src])

        # --      height-adjusted geogrid coordinate correction
        tp1a, tp2a = tp1.copy(), tp2.copy()
        tp1a = np.array((scale_g2s1(tp1a[0]), scale_g2s1(tp1a[1])))
        tp2a = np.array((scale_g2s2(tp2a[0]), scale_g2s2(tp2a[1])))
        tp1b = tp1a.reshape((tp1a.shape[0], np.prod(np.array(tp1a.shape[1:])))).T
        tp2b = tp2a.reshape((tp2a.shape[0], np.prod(np.array(tp2a.shape[1:])))).T
        s2_interp = (np.concatenate((tp2b, s2.real.reshape((s2.shape[0], np.prod(np.array(s2.shape[1:])))).T), axis=1),
                     np.concatenate((tp2b, s2.imag.reshape((s2.shape[0], np.prod(np.array(s2.shape[1:])))).T), axis=1))
        topt = {"balanced_tree":False, "compact_nodes":False}
        s2_interp = ([scipy.interpolate.NearestNDInterpolator(s2_interp[0][:,:2], s2_interp[0][:,i], tree_options=topt) for i in range(2, s2_interp[0].shape[1])],
                     [scipy.interpolate.NearestNDInterpolator(s2_interp[1][:,:2], s2_interp[1][:,i], tree_options=topt) for i in range(2, s2_interp[1].shape[1])])
        s2o = np.array([s2_interp[0][i](tp1b) + s2_interp[1][i](tp1b) * 1j for i in range(s2.shape[0])]).reshape(s1.shape)

        # --      optical flow-based final correction
        flow = skimage.registration.optical_flow_ilk(np.abs(s2o[0]), np.abs(s1[0]), radius=30, num_warp=50)
        tmp = np.meshgrid(np.arange(s2o.shape[1]), np.arange(s2o.shape[2]), indexing='ij');
        tmp = np.array([tmp[0]-flow[0], tmp[1]-flow[1]])
        s2o = np.array([scipy.ndimage.map_coordinates(s2o[i].real, tmp).astype(complex) +
                        scipy.ndimage.map_coordinates(s2o[i].imag, tmp) * 1j for i in range(s2o.shape[0])])
        self.s1 = s1o
        self.s2 = s2o
        self.ia1 = scipy.ndimage.map_coordinates(scale_g2s1(incidence_angle), tmp)
        self.iad = scipy.ndimage.map_coordinates(scale_g2s1(incidence_angle_delta), tmp)
        