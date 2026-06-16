import os, pathlib, shutil, warnings, getpass, warnings
import numpy as np
import torch
import torch.nn.functional as F

import datetime
import pandas as pd
import xml.etree.ElementTree as ET

from torch_tps import ThinPlateSpline, PolynomialFeatures
import h5py

from osgeo import gdal, osr, gdal_array
#GDAL ref : https://gdal.org/en/stable/doxygen/classGDALDataset.html

def srs_llh2xyz(srs, lat, lon, h=0): # h : radial height
    a, f = srs.GetSemiMajor(), 1/srs.GetInvFlattening()  # ellipsoid parameters
    e2 = f*(2 - f) # eccentricity squared
    N = (a / np.sqrt(1 - e2*np.sin(np.deg2rad(lat))**2))
    x = (N + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y = (N + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z = (N*(1 - e2) + h)*np.sin(np.deg2rad(lat))
    return x, y, z

def srs_xyz2llh(srs, x, y, z):
    a, f = srs.GetSemiMajor(), 1/srs.GetInvFlattening()
    b, p, e2 = a * (1 - f),  np.sqrt(x**2 + y**2), f*(2 - f)
    theta = np.arctan2(a*z, b*p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ((a**2 - b**2)/b)*np.sin(theta)**3, p - a*e2*np.cos(theta)**3)
    h = p / np.cos(lat) - (a / np.sqrt(1 - e2*np.sin(lat)**2))
    return np.rad2deg(lat), np.rad2deg(lon), h

# utility for dictionary / h5 object save/load
def d2g_h5(group, d):
    for key, value in d.get('attr',{}).items():
        group.attrs[key] = value
    for key, value in d.get('data',{}).items():
        if key in group: del(group[key])
        group.create_dataset(key, data=value)
    for key, value in d.get('c',{}).items():
        sub = group.create_group(key)
        d2g_h5(sub, value)
        
def g2d_h5(group):
    out = {'attr': {},'data': {},'c': {}}
    for key, value in group.attrs.items():
        out['attr'][key] = value
    for key, item in group.items():
        if isinstance(item, h5py.Group): out['c'][key] = g2d_h5(item)
        else: out['data'][key] = item[()]
    return out

class DataStack():

    def __init__(self, data_fn='default.hdf5', si=1, gsi=4, img_tags=['HH','HV','VH','VV'], compression='lzf', device='cpu'):
        self.img_tags = img_tags
        self.si = si
        self.gsi = gsi
        self.device = device
        self.h5f = h5py.File(data_fn, 'a')
        self.compression = compression

    def build(self, data_dir='/mnt/d/S1Burst_CAVancouver_2020', corner=(0,0), width=(99999,99999), idate=3, bf=8):
        gdal.UseExceptions()
        all_fn_look = os.listdir(data_dir)
        fnl = [el for el in all_fn_look if (len(el.split('.'))==2 and os.path.splitext(el)[1] in ('.xml', '.tiff'))] # exclude directories & .pickle files
        dtl = list(set([el.split('_')[idate] for el in fnl]))
        dtl = sorted(dtl, key=lambda x: datetime.datetime.strptime(x, '%Y%m%dT%H%M%S%f').timestamp()) #ascending
        
        for dtn in dtl:
            fn_i = [el for el in os.listdir(data_dir) if os.path.splitext(el)[1]=='.tiff' and dtn in el]
            fn_i = sorted(fn_i, key=lambda x: ([i for i, tag in enumerate(self.img_tags) if (f"_{tag}_" in x)] + [999])[0] )
            
            date_entry = []
            
            for i in range(len(fn_i)):
                ds_i = gdal.Open(os.path.join(data_dir, fn_i[i]))
                ds_i_srs = osr.SpatialReference(); ds_i_srs.ImportFromWkt(ds_i.GetGCPProjection())
                if ds_i_srs.GetAuthorityCode(None) != '4326': raise ValueError('current program assumes EPSG:4326; input CRS differs.')
                self.default_srs = ds_i_srs

                # select tags for DataStack filename->bandname case
                img_tag = ([tag for tag in self.img_tags if tag in fn_i[i]]+['UNKNOWN'])[0]
                
                # truncate metadata
                xml_trunc_fn = os.path.splitext(fn_i[i])[0]+'_TRUNC.xml'
                if (not os.path.exists(xml_trunc_fn)):
                    tree = ET.parse(os.path.join(data_dir, os.path.splitext(fn_i[i])[0]+'.xml'))
                    meta = tree.getroot().find('metadata')
                    token_l = fn_i[i].split('_')
                    i = 0
                    while i < len(meta.findall('product')):
                        tmp = meta.findall('product')[i]
                        swath, polar = tmp.find('swath').text, tmp.find('polarisation').text
                        if not (swath in token_l and polar in token_l):
                            meta.remove(meta.findall('product')[i])
                            meta.remove(meta.findall('noise')[i])
                            meta.remove(meta.findall('calibration')[i])
                        else : i += 1
                    tree.write(os.path.join(data_dir, xml_trunc_fn))
                    tree = None
        
                grid = ET.parse(os.path.join(data_dir, xml_trunc_fn)).getroot()
                grid = grid.find('metadata').find('product').find('content').find('geolocationGrid').find('geolocationGridPointList')
                grid = pd.DataFrame([{e.tag : e.text for e in el} for el in grid])
                iw, ih = ds_i.RasterXSize, ds_i.RasterYSize

                gt = [grid.line, grid.pixel, grid.latitude, grid.longitude, grid.height,
                      grid.azimuthTime, grid.slantRangeTime, grid.incidenceAngle, grid.elevationAngle]
                gt[5] = np.array([datetime.datetime.strptime(el, '%Y-%m-%dT%H:%M:%S.%f').timestamp() for el in gt[5]])
                gt = np.stack(gt, axis=-1).astype(np.float32)
                gt[:, :2] = (gt[:, :2] - np.array(corner)[None,:]) / self.si # adjust line/pixel coordinate to sampling rule
                gt[:, 2:5] = np.stack(srs_llh2xyz(ds_i_srs, *(gt[:,2:5].T)), axis=-1) # lat-lon-h to x-y-z (euclidian geolocation)
                gt_ = torch.as_tensor(gt).to(self.device)
                
                tps = StorableTPS(enforce_tps_kernel=True, device=self.device); tps.fit(gt_[:,:2], gt_[:,2:5]) # ij to xyz
                gc_ = torch.as_tensor(np.mgrid[0:ih+self.gsi:self.gsi, 0:iw+self.gsi:self.gsi]).to(self.device, gt_.dtype)
                gc_ = tps.transform(gc_.view(2, -1).T).view(gc_.shape[1],gc_.shape[2],3) # Hc, Wc, 3

                gt = StorableTPS(enforce_tps_kernel=True, device=self.device); gt.fit(gt_[:,2:5], gt_[:,5:]) #xyz to geoparam
                
                for j in range(1, ds_i.RasterCount + 1):                    
                    b = ds_i.GetRasterBand(j)
                    b_n = img_tag + f'_rast{j}' + b.GetDescription()
                    b_img = b.ReadAsArray()[corner[0]:corner[0]+width[0]:self.si, corner[1]:corner[1]+width[1]:self.si]
                    b_img = torch.as_tensor(b_img).to(self.device)
                    date_entry.append((b_img, b_n, gc_, gt))
                ds_i = None

            if f'scenes/{dtn}' in self.h5f : del(self.h5f[f'scenes/{dtn}'])
            scene_ = self.h5f.create_group(f'scenes/{dtn}', track_order=True)
            bands_ = scene_.create_group('b', track_order=True)
            
            # coregister date entry (prevent misregistered files within date)
            img1, b_n1, gc1, gt1 = date_entry[0]
            b = bands_.create_group(b_n1, track_order=True)
            b.create_dataset('data', data=img1.detach().cpu().numpy(), compression=self.compression)
            b_tps = b.create_group('tps')
            d2g_h5(b_tps, gt1.compress2dict())
            # common scene position set to gc1
            if 'xyz_coarse' in scene_: del(scene_['xyz_coarse'])
            scene_.create_dataset('xyz_coarse', data=gc1.detach().cpu().numpy(), compression=self.compression, track_order=True)
            del(img1, b_n1, gt1)
            for i in range(1, len(date_entry)):
                img, b_n, gc, gt = date_entry[i]
                img = self.coreg(img, gc1, gc, bf).detach().cpu().numpy()
                b = bands_.require_group(b_n)
                if 'data' in b: del(band['data'])
                b.create_dataset('data', data=img, compression=self.compression, track_order=True)
                b_tps = b.require_group('tps')
                d2g_h5(b_tps, gt.compress2dict())

    def coreg_stack(self, primary_index=0, bf=8):
        s1_key = list(self.h5f['scenes'].keys())[primary_index]
        s1_gc = self.h5f['scenes'][s1_key]['xyz_coarse'][()]
        for ikey, dtx in self.h5f['scenes'].items():
            if ikey == s1_key: continue
            if not isinstance(dtx, h5py.Group): continue
            s2_gc = dtx['xyz_coarse'][()]
            for jkey, band in dtx['b'].items():
                img = torch.as_tensor(band['data'][()]).to(self.device)
                img = self.coreg(img, s1_gc, s2_gc, bf).detach().cpu().numpy()
                del(band['data'])
                band.create_dataset('data', data=img, compression=self.compression, track_order=True)
            del(dtx['xyz_coarse'])
            dtx.create_dataset('xyz_coarse', data=s1_gc, compression=self.compression, track_order=True)
    
    def coreg(self, data, gc1, gc2, bf=4):
        # gc : (Hc, Wc, 3) format
        Hc, Wc = gc1.shape[:2]
        gc1t = torch.as_tensor(gc1, device=self.device).reshape(-1,3)
        gc2t = torch.as_tensor(gc2, device=self.device).reshape(-1,3)
        ig2 = torch.as_tensor(np.mgrid[:Hc, :Wc], dtype=gc1t.dtype, device=self.device).moveaxis(0,-1).reshape(-1,2)
        ig2 *= self.gsi # multiply by geolocation sample interval for coarse grid -> fine grid coordinates
        tps = ThinPlateSpline(device=self.device); tps.fit(gc2t, ig2)
        igT = tps.transform(gc1t).reshape(Hc,Wc,2); del(ig2, tps)
        
        dst = torch.zeros_like(data)
        for pi in range(0, Hc, bf):
            for pj in range(0, Wc, bf):
                _batched_coord_warp_torched(data, dst, pi, pj, igT, self.gsi, bf)
        return dst

    def read(self, index, get_aux_p=False):
        key = list(self.h5f['scenes'].keys())[index]
        gc = torch.as_tensor(self.h5f['scenes'][key]['xyz_coarse'][()], device=self.device)
        labels, img, plabels, p = [], [], ['az_t','sr_t','ia','ea'], []
        tps_flag = True
        for ikey, band in self.h5f['scenes'][key]['b'].items():
            labels.append(ikey)
            immg = band['data'][()]; img.append(immg)
            if (tps_flag or get_aux_p):
                tps = StorableTPS.decompress4dict(g2d_h5(band['tps']), device=self.device)
                pt = tps.transform(gc.reshape(-1,3)).reshape(gc.shape[0],gc.shape[1],-1)
                pt = F.interpolate(pt.moveaxis(-1,0)[None], scale_factor=self.gsi, mode='bilinear', align_corners=False)[0,:,1:,1:]
                pt = pt.moveaxis(0,-1)[:immg.shape[0], :immg.shape[1], :] # crop expanded coarse bounds to fine bounds
                p.append(pt); tps_flag = False
        gc = F.interpolate(gc.moveaxis(-1,0)[None,], scale_factor=self.gsi, mode='bilinear', align_corners=False)[0,:,1:,1:]
        gc = gc.moveaxis(0, -1)[:immg.shape[0], :immg.shape[1], :].clone()
        # collate & give data as torch tensor
        if (get_aux_p):
            data = torch.as_tensor(np.stack(img, axis=-1), device=self.device)[...,None,:]
            data = torch.cat((data, torch.stack(p, dim=-1).to(dtype=data.dtype)), dim=-2).swapaxes(-2,-1)
            return data, labels, plabels
        else:
            data = torch.as_tensor(np.stack(img, axis=-1), device=self.device)
            data = torch.cat((data, p[0].to(dtype=data.dtype)), dim=-1)
            labels += plabels    
        return data, labels, gc

def _batched_coord_warp_torched(src, dst, pi, pj, crds, strd, bf):
    """
    src   : source tensor (Hs, Ws)
    dst   : destination tensor (Hd, Wd)
    pi,pj : coarse index location
    crds  : coarse sampling index tensor (Hc, Wc, 2)
    strd  : coarseness of coordinate field (scalar)
    bf    : coarse-grid batch size (scalar)
    """
    if (pi * strd >= dst.shape[0]) or (pj * strd >= dst.shape[1]): return
    dvc, cdtp = dst.device, crds.dtype
    pc = crds[pi:pi+bf, pj:pj+bf]
    pco = torch.as_tensor(np.mgrid[pi:pi+bf, pj:pj+bf], device=dvc, dtype=cdtp).moveaxis(0,-1)[:pc.shape[0], :pc.shape[1]] #h,w,2
    mdyc, mdxc = torch.ceil(torch.amax(torch.abs(pc/strd - pco), dim=(0,1))).int().tolist()

    # read from coarse grid to halo'd tensor & expand
    y0c, x0c = max(0, pi-mdyc), max(0, pj-mdxc)
    y1c, x1c = min(pi+bf+mdyc, crds.shape[0]), min(pj+bf+mdxc, crds.shape[1])
    pc = torch.full((bf+2*mdyc, bf+2*mdxc, 2), -1.0, device=dvc, dtype=cdtp)
    pc[y0c-(pi-mdyc):y1c-(pi-mdyc), x0c-(pj-mdxc):x1c-(pj-mdxc), :] = crds[y0c:y1c, x0c:x1c, :]
    pc = F.interpolate(pc.moveaxis(-1,0)[None], scale_factor=strd, mode='bilinear', align_corners=False)[0,:,1:,1:].moveaxis(0,-1)

    # get reduced source tensor view to reduce sampling overhead
    n0, n1 = torch.clamp(torch.floor(pc.amin(dim=(0,1))), min=0).to(torch.int32),  torch.ceil(pc.amax(dim=(0,1))).to(torch.int32)
    if (n1 - n0).amin() == 0: return # abort tile - degenerate source coordinate access
    src_crop = src[n0[0]:n1[0], n0[1]:n1[1]]

    # build normalized grid &then map coordinates
    grid = (2*(pc - n0[None,None,:])/torch.clamp(torch.as_tensor(src_crop.shape, device=dvc, dtype=cdtp) - 1, min=1)[None,None,:] - 1).flip(dims=(-1,))
    if torch.is_complex(src):
        mapped = torch.complex(
            F.grid_sample(src_crop.real[None,None],grid[None],mode='nearest',padding_mode='zeros',align_corners=True)[0,0],
            F.grid_sample(src_crop.imag[None,None],grid[None],mode='nearest',padding_mode='zeros',align_corners=True)[0,0])
    else: mapped = F.grid_sample(src_crop[None,None],grid[None],mode='nearest',padding_mode='zeros',align_corners=True)[0,0]
    mapped = mapped[mdyc*strd:(bf+mdyc)*strd, mdxc*strd:(bf+mdxc)*strd] # cut away halo

    # write to within destination bounds
    yd1, xd1 = min((pi+bf)*strd, dst.shape[0]), min((pj+bf)*strd, dst.shape[1])
    yp1, xp1 = yd1 - pi*strd, xd1 - pj*strd
    dst[pi*strd:yd1, pj*strd:xd1] = mapped[:yp1, :xp1]


class StorableTPS(ThinPlateSpline):
    '''
    A ThinPlateSpline with an additional protocol for dictionary conversion, in hdf5-like convention.
    '''
    def compress2dict(self):
        package = {}
        package['attr'] = {
            'fitted': self._fitted,
            'alpha' : self.alpha,
            'order' : self.order,
            'enforce_tps': self.enforce_tps_kernel,
            'device': str(self.device)
        }
        package['data'] = {
            'pm' : self.parameters.detach().cpu().numpy(),
            'cp' : self.control_points.detach().cpu().numpy()
        }
        pn = self._polynomial_features
        package['c'] = {'pf': {'data': {'exp' : pn.exponents.detach().cpu().numpy()}}}
        return package
    
    @classmethod
    def decompress4dict(cls, package:dict, device:str=None):
        tps = cls()
        tps._fitted = package['attr']['fitted']
        tps.alpha = package['attr']['alpha']
        tps.order = package['attr']['order']
        tps.enforce_tps_kernel = package['attr']['enforce_tps']
        if device is None: tps.device = package['attr']['device']
        else : tps.device = device
        
        tps.parameters = torch.as_tensor(package['data']['pm']).to(tps.device)
        tps.control_points = torch.as_tensor(package['data']['cp']).to(tps.device)

        pf = PolynomialFeatures(tps.order - 1)
        pf.exponents = torch.as_tensor(package['c']['pf']['data']['exp']).to(tps.device)
        pf._fitted = tps._fitted
        tps._polynomial_features = pf
        
        return tps
