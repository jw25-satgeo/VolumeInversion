import numpy as np
import scipy
import skimage
import h5py
import os
import pickle

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

# src https://earth.esa.int/eogateway/documents/20142/37627/ALOS-PALSAR-calibration-products-ADEN.pdf
def iq2calib_alos1(image, incidence_angle, cf_offset=0, output_type='beta'):
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

# Conversion of SNAP-exported data in .h5 form - physical defaults set to ALOS-1 PALSAR
class Pair_Dataset_SNAPh5:

    @staticmethod
    def load_cache(fn="Pair_Dataset_SNAPh5_cache.pickle"):
        with open(fn, 'rb') as f: return pickle.load(f)

    def cache(self, cache_fn="Pair_Dataset_SNAPh5_cache.pickle"):
        with open(cache_fn, 'wb') as f:
            pickle.dump(self, f)
    
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

class Pair_Dataset_NISAR:
    
    @staticmethod
    def load_cache(fn="Pair_Dataset_NISAR_cache.pickle"):
        with open(fn, 'rb') as f: return pickle.load(f)

    def cache(self, cache_fn="Pair_Dataset_NISAR_cache.pickle"):
        with open(cache_fn, 'wb') as f:
            pickle.dump(self, f)

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
        