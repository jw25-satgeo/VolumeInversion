import numpy as np
import datetime
import scipy
import skimage
import h5py
import os
import pickle
import netCDF4


# SNAP
# base file import & preprocessing w. SNAPISTA
class Pair_Dataset_SNAPbase():
    default_label="SNAPbase_latest"
    
    def __init__(self, slc_dir=None, cache_dir=None, nc_fn=None, read_nc=True):
        self.cache_dir = cache_dir
        self.is_alos1 = is_alos1
        if (nc_fn is None): nc_fn = 'BEAM.nc'
        self.nc_fn = nc_fn
        if (cache_dir is None): self.cache_dir = f'{self.default_label}_cache'
        if((cache_dir is None) or (not read_nc)): self.SnapistaProc(s1_fn, slc_dir)
        elif (nc_fn in os.listdir(self.cache_dir)): print(f'located cache at {self.cache_dir}')
        else: raise ValueError(f'could not find cache at {self.cache_dir}.')
        self.ncGet()
        print('dataset loaded.')

    def _get_snapista_calib_graph(self, slc_dir):
        from esa_snappy import snapista as snp
        graph = snp.Graph()
        bnd_used = ['i_HH','q_HH,i_HV,q_HV,i_VH,q_VH,i_VV,q_VV']
        dem_used = 'Copernicus 30m Global DEM'
        graph.add_node(operator=snp.Operator('Read', file=s1_fn), node_id = 'read_s1') # Read
        graph.add_node(operator=snp.Operator('Read', file=s2_fn), node_id = 'read_s2')
        calib = snp.Operator('Calibration', sourceBandNames=bnd_used, outputImageInComplex='true', outputSigmaBand='false')
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