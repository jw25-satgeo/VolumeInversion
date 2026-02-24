import numpy as np
import scipy
import datetime, calendar
import netCDF4
import shapely
import pickle

import asf_search

import time, random
import multiprocessing
import concurrent.futures

class SICEScroller:
    
    def __init__(self, start=20070101, end=20120101, season=None, query_retries=5):
        self.query_retries = query_retries

        self.label = f'{start}-{end}-Scroller'
        
        # dates initializiation
        date0 = datetime.datetime(start//10000, start//100%100, start%100)
        date1 = datetime.datetime(end//10000, end//100%100, end%100)
        self.dates = []
        while date0 <= date1:
            season_check = (date0 - datetime.datetime(date0.year, 1, 1)).days
            if (season is None or len(season)<1 or (season_check > season[0] and season_check < season[1])):
                self.dates.append(date0)
            date0 += datetime.timedelta(days=1)

        # coordinate grid initialization
        llgrid_fn = f"neXtSIM_cache/cmems_mod_arc_phy_my_nextsim_P1M-m_siconc_180.00W-179.97E_49.98N-90.03N_{str(self.dates[0])[:7]}-01.nc"
        tmp = netCDF4.Dataset(llgrid_fn, 'r')
        self.llgrid = np.array(np.meshgrid(tmp.variables['latitude'], tmp.variables['longitude'], indexing='ij'))
        
        # for result list initialization
        self.results = {}

    def save_cache(self, file_name=None):
        if (file_name is None):
            file_name = f'{self.label}.pickle'
        with open(file_name, mode='wb') as f:
            return pickle.dump(self, f)
    
    @classmethod
    def from_cache(self, file_name):
        with open(file_name, mode='rb') as f:
            return pickle.load(f)
    
    def search_geo(self, c_min=0.2, c_max = 0.8, v_min = 0, v_max = 1e-4, ll_sample_len=(1, 2), is_coarse_monthly=True):
        #CALCULATE VALID WKT
        self.monthly_wktgeo_dict = {}
        d2m = {date : datetime.datetime(date.year,date.month,1) for date in self.dates}
        
        args = [(date, c_min, c_max, v_min, v_max, ll_sample_len) for date in set(d2m.values())]
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
            futures = [executor.submit(self.monthly_wktgeo, *arg) for arg in args]
            for future in concurrent.futures.as_completed(futures):
                month_key, wktgeo = future.result()
                if (len(wktgeo) > 0): # @date tmp[0], set list of WKT objects tmp[1]
                    self.monthly_wktgeo_dict[month_key] = wktgeo

        #BEGIN SEARCH QUERIES
        self.results = {}
        rate_limiter = RateLimiter(rate_per_sec=4.0)

        # monthly coarse search
        if (is_coarse_monthly):
            tasks = [(dt, datetime.datetime(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1]), wkt)
                     for dt in self.monthly_wktgeo_dict.keys() for wkt in self.monthly_wktgeo_dict[dt]]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.single_search_throttled, rate_limiter, wkt, dt1, dt2) for dt1, dt2, wkt in tasks]
                for future in concurrent.futures.as_completed(futures):
                    date_key, result = future.result()
                    if (len(result) > 0):
                        print(f'{len(result)} results found')
                        self.results.setdefault(date_key, []).extend(result)
            return

        # day-by-day search
        tasks = [(date, wkt) for date in self.dates for wkt in self.monthly_wktgeo_dict[d2m[date]]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.single_search_geo, rate_limiter, wkt, date) for date, wkt in tasks]
            for future in concurrent.futures.as_completed(futures):
                date_key, result = future.result()
                if (len(result) > 0):
                    print(f'{len(result)} results found!')
                    self.results.setdefault(date_key, []).extend(result)

    #def single_search_sbas(self, limiter, ref): #ref should be ASFProduct
    #    print("Querying:", ref.meta['native-id'])
    #    single_results = None
    #    for i in range(self.query_retries):
    #        limiter.acquire()
    #        try:
    #            single_results = 
    
    def single_search_geo(self, limiter, wkt, start_dt, end_dt=None):
        print("Querying:", start_dt, wkt)
        single_results = None
        if (end_dt is None): end_dt = start_dt
        for i in range(self.query_retries):
            limiter.acquire()
            try:
                single_results = asf_search.geo_search( intersectsWith = wkt,
                    dataset = asf_search.constants.DATASET.ALOS_PALSAR, polarization = 'quadrature', flightDirection='ASCENDING',
                    start=str(start_dt)[:10]+'T00:00:00Z', end=str(end_dt)[:10]+'T23:59:59Z', maxResults=25 )
            except Exception as e:
                print(f"Error on {start_dt}-{end_dt}:{e} \n ... retrying {self.query_retries - i} more times.")
                single_results = None
                time.sleep((2 ** i) + random.random())
        if (single_results is None):
            single_results = []
        return str(start_dt)[:10], single_results

    def monthly_wktgeo(self, date, c_min, c_max, v_min, v_max, ll_sample_len):
        print ("generating valid WKT points for: ",str(date))
        mask = self.monthly_mask(date, c_min, c_max, v_min, v_max)
        
        llg_valids = self.llgrid[:, mask].transpose()
        v_inds = np.arange(llg_valids.shape[0])
        v_mask = np.ones(v_inds.size, dtype=bool)
        
        llsep = np.array(ll_sample_len)[None, :]
        points = []
        while(v_mask.any() > 0):
            pt = np.random.choice(v_inds[v_mask])
            v_mask[(np.abs(llg_valids - llg_valids[pt]) < llsep).all(axis=-1)] = 0
            points.append(llg_valids[pt])
        points = [f'POINT({pt[1]} {pt[0]})' for pt in np.round(points,4)] # WKT LON-LAT format
        return date, points
        
    def monthly_mask(self, date, c_min, c_max, v_min, v_max):
        print("generating filter mask for: ",str(date))
        ymseg = str(date)[:7]
        siconc_fn = f"neXtSIM_cache/cmems_mod_arc_phy_my_nextsim_P1M-m_siconc_180.00W-179.97E_49.98N-90.03N_{ymseg}-01.nc"
        sivxvy_fn = f"neXtSIM_cache/cmems_mod_arc_phy_my_nextsim_P1M-m_vxsi-vysi_180.00W-179.97E_49.98N-90.03N_{ymseg}-01.nc"
        
        sic = netCDF4.Dataset(siconc_fn, 'r')
        sic = sic.variables['siconc'][0]
        siv = netCDF4.Dataset(sivxvy_fn, 'r')
        siv = (siv.variables['vxsi'][0]**2 + siv.variables['vysi'][0]**2)**0.5

        mask = (siv < v_max) & (siv > v_min) & (sic > c_min) & (sic < c_max)
        mask[np.isnan(mask)] = False
        return mask.data


import threading, time
class RateLimiter:
    def __init__(self, rate_per_sec, capacity=None):
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self._tokens = self.capacity
        self._lock = threading.Lock()
        self._last = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                # refill tokens
                delta = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + delta * self.rate)

                if self._tokens >= 1:
                    self._tokens -= 1
                    return  # success — caller may proceed

            # no token available: sleep a bit and retry
            time.sleep(1.0 / self.rate)




    