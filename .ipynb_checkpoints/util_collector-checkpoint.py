import sys, os, h5py
import numpy as np

import torch

class Dataset():

    def __init__(self, fn, compression='lzf', increment=100, mode='a'):
        if (os.path.splitext(fn)[1] != '.hdf5'): raise ValueError('extension must be .hdf5')
        self.fn = fn
        self.mode = mode
        self.h5f = h5py.File(fn, mode) # note : Don't implement SWMR - once activated, group creation is disabled!
        self.cmp = compression
        self.inc = increment
        if 'w_ind' not in self.h5f.attrs:
            self.h5f.attrs['w_ind'] = 0
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()
    def close(self):
        self.h5f.close()

    def set_attrs(self, **ix):
        if not ix: return
        for key, x in ix.items():
            self.h5f.attrs[key] = x
        self.h5f.flush()
    
    def add(self, **ix):
        # note : dataset size incrementation works for arbitrarily large batches.
        if not ix: return
        if self.mode != 'a': raise ValueError('mode must be "a" for collector increment')
        shapes = [x.shape[0] for key, x in ix.items()]
        if len(set(shapes)) > 1: raise ValueError('unequal batch size: cannot write to collector.')
        di, B = self.h5f.attrs['w_ind'], shapes[0]
        for key, x in ix.items():
            grp = self.h5f.require_group(key)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if 'fshape' not in grp.attrs:
                grp.attrs['fshape'] = x.shape[1:]
                N = np.array(x.shape[1:]).prod()
                grp.create_dataset('d', (self.inc,N), maxshape=(None,N), chunks=(min(self.inc, 64),N), dtype=x.dtype, compression=self.cmp)
            elif tuple(grp.attrs['fshape']) != x.shape[1:]: raise ValueError(f'input was not of previously stored shape{grp.attrs['fshape']}.')
            spill = di + B - len(grp['d'])
            if spill > 0:
                spill = ((spill + self.inc - 1) // self.inc) * self.inc
                grp['d'].resize(len(grp['d']) + spill, axis=0)
            grp['d'][di:di+B, :] = x.reshape(B,-1)
        self.h5f.attrs['w_ind'] = di + B
        self.h5f.flush()

    def read(self, mask, reshape=False, clamp=False):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.sort(mask)
        items = {}
        for var in self.h5f:
            grp = self.h5f[var] # check : does this access groups without issue?
            if 'd' not in grp: continue
            if clamp: x = grp['d'][:self.h5f.attrs['w_ind']][mask]
            else:     x = grp['d'][mask]
            if reshape:  # reform x into initially stored shape
                x = x.reshape([x.shape[0]] + list(grp.attrs['fshape']))
            items[var] = x
        return items




# scraps
# ....
#elif self.h5f['d'].shape[1] != emb_len:
#    confirm = input(f'Dataset shape is {self.h5f['d'].shape}, but specified embedding length is {emb_len}. Replace data? [Y/N]')
#    if (confirm == 'Y' or confirm == 'y'): del(self.h5f['d']); make_flag=True; print('Replaced data.')
#    else: self.h5f.close(); print('Cancelled operation - dataset is inoperable.')