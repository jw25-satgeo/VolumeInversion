import numpy as np
import scipy
import datetime
import netCDF4
import shapely

import asf_search

import time, random
import multiprocessing
import concurrent.futures

class SICEBaselineScroller:
    
    def __init__(self, start=20070101, end=20120101, season=[30,150], query_retries=5):
        self.query_retries = query_retries
        
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
    
    def search(self, c_min=0.2, c_max = 0.8, v_min = 0, v_max = 1e-4, ll_sample_len=200, d0=2):
        #CALCULATE VALID POLYGONS
        self.monthly_polys_dict = {}
        d2m = {date : datetime.datetime(date.year,date.month,1) for date in self.dates}

        
        
        args = [(date, c_min, c_max, v_min, v_max, ll_sample_len, d0) for date in set(d2m.values())]
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
            futures = [executor.submit(self.monthly_polys, *arg) for arg in args]
            for future in concurrent.futures.as_completed(futures):
                month_key, polys = future.result()
                self.monthly_polys_dict[month_key] = polys # @date tmp[0], set list of WKT polygons tmp[2]

        #BEGIN SEARCH QUERIES
        self.results = {}
        tasks = [(date, poly) for date in self.dates for poly in self.monthly_polys_dict[d2m[date]]]
        rate_limiter = RateLimiter(rate_per_sec=4.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.single_search_throttled, rate_limiter, date, poly) for date, poly in tasks]
            for future in concurrent.futures.as_completed(futures):
                date_key, result = future.result()
                self.results.setdefault(date_key, []).extend(result)

    def single_search_throttled(self, limiter, date, poly):
        print("Querying:", date)
        single_results = None
        for i in range(self.query_retries):
            limiter.acquire()
            try:
                single_results = asf_search.geo_search( intersectsWith = poly,
                    dataset = 'ALOS-2', polarization = 'quadrature', flightDirection='ASCENDING',
                    start=str(date)[:10]+'T00:00:00Z', end=str(date)[:10]+'T23:59:59Z', maxResults=100 )
            except Exception as e:
                print(f"Error on {date}:{e} \n ... retrying {self.query_retries - i} more times.")
                single_results = None
                time.sleep((2 ** i) + random.random())
        if (single_results is None):
            single_results = []
        return str(date)[:10], single_results

    def monthly_polys(self, date, c_min, c_max, v_min, v_max, ll_sample_len, d0):
        print("generating valid WKT polys for: ",str(date))
        mask = self.monthly_mask(date, c_min, c_max, v_min, v_max)
        polys = []
        for i in range(self.llgrid.shape[1] // ll_sample_len):
            c1 = i * ll_sample_len; c2 = min(c1 + ll_sample_len, self.llgrid.shape[1])
            for j in range(self.llgrid.shape[2] // ll_sample_len):
                c3 = j * ll_sample_len; c4 = min(c3 + ll_sample_len,self.llgrid.shape[2])
                #print(c1,c2,c3,c4)
                sample_polys = MaskPolys(mask[c1:c2,c3:c4],
                                         self.llgrid[0,c1,0], self.llgrid[0,c2-1,0],
                                         self.llgrid[1,0,c3], self.llgrid[1,0,c4-1],
                                         d0=d0 )
                polys += sample_polys
        return date, polys
        
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
    
def MaskPolys(mask, lat_min, lat_max, lon_min, lon_max, d0=2):
    '''A simple function for converting a mask into a list of WKT polygons.
    '''
    mask = mask.copy(); mask[0, :] = mask[-1, :] = False; mask[:, 0] = mask[:, -1] = False;
    distance = scipy.ndimage.distance_transform_edt(mask)
    seed_mask = (1 / (distance**2 + 1e-3)) > (np.random.random(distance.shape))
    seed_mask[distance < d0] = False
    
    di0 = int(d0)
    minimask = np.zeros((di0*2, di0*2)); minimask[d0,d0] = 1
    for i, j in np.array(np.where(seed_mask)).T:
        if (np.sum(seed_mask[i-di0:i+di0, j-di0:j+di0]) > 1):
            seed_mask[i-di0:i+di0, j-di0:j+di0] = minimask

    tmp = np.array(np.where(seed_mask)).T
    seeds = [(shapely.geometry.Point(i,j), distance[i,j]) for i,j in tmp]
    polys = []
    for s, g in seeds: poly = s.buffer(g); polys.append(poly)
    multipoly = shapely.unary_union(polys).simplify(tolerance=d0, preserve_topology=True)
    def i2lonlat(coord): return np.stack((coord[:,1]/mask.shape[1]*(lon_max - lon_min)+lon_min,
                                          coord[:,0]/mask.shape[0]*(lat_max - lat_min)+lat_min), axis=-1)
    multipoly = shapely.transform(multipoly, i2lonlat)
    if multipoly.geom_type == 'Polygon':
        return [multipoly.wkt]
    if multipoly.geom_type == 'MultiPolygon':
        return [geom.wkt for geom in multipoly.geoms]
    return []


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




    