import sys, os
import numpy as np
import pandas as pd


import getpass
import ASF

def download(overlap, data_dir='data', delta=86400 * 10, offset = 86400*0.02, cred_user=None, cred_pass=None, pickles_only=False):
    '''
    Search and download ALOS-1 SLC files overlapping with time-space (d, lat, lon) data from ASF Vertex.
    
    data_dir  : downloads directory
    delta  : minimum time sampling density (seconds)
    offset : maximum time offset of image from a buoy measurement (seconds)
    cred_user : EarthData or ASF credentials
    cred_pass : EarthData or ASF credentials
    '''
    buoy_ids = set(buoy_data.BuoyID)
    if (cred_user is None or cred_pass is None):
        cred_user = input('Username:')
        cred_pass = getpass.getpass('Password:')
    for buoy_id in buoy_ids:
        buoy_path = os.path.join(data_dir,str(buoy_id))
        if not os.path.exists(buoy_path): os.mkdir(buoy_path)
        scroller_fn = os.path.join(buoy_path, f'{buoy_id}_BuoyScroller.pickle')
        scroller = None
        if (os.path.exists(scroller_fn)):
            print (f'downloading files for ')
            scroller = ASF.BuoyScroller.from_cache(scroller_fn)
        elif (not pickles_only):
            scroller = ASF.BuoyScroller(buoy_id, buoy_data)
            scroller.search(delta=delta, offset=offset)
            scroller.save_cache(scroller_fn)
        if (scroller is not None):
            scroller.download_results(cache_dir=buoy_path,cred_user=cred_user,cred_pass=cred_pass)