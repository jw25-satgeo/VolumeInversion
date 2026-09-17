import datetime, time, calendar
import pathlib, itertools, random, threading, getpass, warnings
import concurrent.futures

import numpy as np

import pickle

import asf_search
from asf_search.download.file_download_type import FileDownloadType

class BaseScroller:

    def save_cache(self, fn=None):
        if (fn is None):
            fn = f'{self.label}.pickle'
        with open(fn, mode='wb') as f:
            return pickle.dump(self, f)
    
    @classmethod
    def from_cache(self, file_name):
        with open(file_name, mode='rb') as f:
            return pickle.load(f)
    
    def download_results(self, cache_dir=None, cred_user=None, cred_pass=None, max_workers=10, fileType=FileDownloadType.ALL_FILES):
        if (cache_dir is None): cache_dir = f'{self.label}_downloads'
        pathlib.Path(cache_dir).mkdir(exist_ok=True)

        if (cred_user is None or cred_pass is None):
            cred_user = input('Username:')
            cred_pass = getpass.getpass('Password:')
            
        session = asf_search.ASFSession().auth_with_creds(cred_user, cred_pass)
        rate_limiter = RateLimiter(rate_per_sec=4.0)
        asf_results = list(itertools.chain.from_iterable(self.results.values()))
        downloader = self.single_download
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(downloader,
                product=result,
                path=cache_dir,
                session=session,
                fileType=fileType,
                limiter=rate_limiter
            ) for result in asf_results]
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
        
    def single_download(self, product, path, session, limiter, fileType):
        for url in product.get_urls(fileType):
            filename = product.properties['fileID'] + '.' + url.split('.')[-1]
            if (pathlib.Path(path+'/'+filename).exists()):
                print(f'{filename} already exists - skipping download.')
                return
            for i in range(self.query_retries):
                limiter.acquire()
                try:
                    asf_search.download.download_url(url, path=path, filename=filename, session=session)
                    break
                except Exception as e:
                    print(f"Error on {filename}:{e} \n ... retrying {self.query_retries - i} more times.")
                    single_results = None
                    time.sleep((2 ** i) + random.random())
        
    def single_search_geo(self, limiter, dataset, polarization, beamMode, wkt, start_dt, end_dt, max_results=10):
        ellipsis = "...)" if len(wkt) > 30 else ""
        print("Querying:", start_dt, wkt[:5]+ellipsis)
        single_results = None
        for i in range(self.query_retries):
            limiter.acquire()
            try:
                single_results = asf_search.geo_search(
                    dataset = dataset,
                    polarization=polarization,
                    beamMode=beamMode,
                    intersectsWith = wkt,
                    start=start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end=end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    maxResults=max_results )
                break
            except Exception as e:
                print(f"Error on {start_dt}-{end_dt}:{e} \n ... retrying {self.query_retries - i} more times.")
                single_results = None
                time.sleep((2 ** i) + random.random())
        if (single_results is None):
            single_results = []
        return int(start_dt.strftime('%Y%m%d%H%M%S')), single_results



class TrackScroller(BaseScroller):
    
    def __init__(self, track_id, track, query_retries=5):
        self.query_retries = query_retries
        self.label = f'{track_id}_TrackScroller'
        # extract time & position
        track = track.sort_values('timestamp')
        self.dates = [datetime.datetime.fromtimestamp(t) for t in track['timestamp'].to_numpy()]
        lons = track['lon'].to_numpy(dtype=float); lons[lons > 180] -= 360;
        lats = track['lat'].to_numpy(dtype=float)
        self.dtlls = list( zip(self.dates, zip(lons, lats)) )
        # initialize result list
        self.results = {}

    def search(self, delta=21600, offset=3600, max_results=10, dataset=asf_search.constants.DATASET.ALOS_PALSAR, beamMode=None, polarization='quadrature'):
        
        warnings.filterwarnings('ignore', module='asf_search')
        
        # temporal batching 
        key, wktgeos, access_dates = None, {}, {}
        for date, ll in self.dtlls:
            if key is None or key + datetime.timedelta(seconds=delta) <= date:
                key, wktgeos[key], access_dates[key] = date, [], []
            access_dates[key].append(date)
            wktgeos[key].append(ll)
        wktgeos = {k:list(dict.fromkeys(v)) for k, v in wktgeos.items()}
        #wktgeos = {k:f'LINESTRING({", ".join([f'{v1[0]} {v1[1]}' for v1 in v])})' for k, v in wktgeos.items()}
        wktgeos = {
            k: (f'POINT({v[0][0]} {v[0][1]})' if len(v) == 1 else f'LINESTRING({", ".join(f"{v1[0]} {v1[1]}" for v1 in v)})')
            for k, v in wktgeos.items()
        }
        
        #BEGIN SEARCH QUERIES
        self.results = {}
        rate_limiter = RateLimiter(rate_per_sec=4.0)
        margin = datetime.timedelta(seconds=offset)
        tasks = [(d1 - margin, d2[-1] + margin, wktgeos[d1]) for d1, d2 in access_dates.items()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(
                self.single_search_geo,
                rate_limiter,
                dataset,
                polarization,
                beamMode, 
                wkt,
                d1,
                df,
                max_results
            ) for d1, df, wkt in tasks]
            for future in concurrent.futures.as_completed(futures):
                date_int, result = future.result()
                lindates = access_dates[datetime.datetime.strptime(str(date_int), '%Y%m%d%H%M%S')]
                result = [
                    res for res in result if
                    min( abs((datetime.datetime.strptime(
                        res.properties['startTime'],
                        '%Y-%m-%dT%H:%M:%SZ') - lindate).total_seconds())
                         for lindate in lindates
                    ) <= offset
                ]
                if (len(result) > 0):
                    print(f'{len(result)} results found!')
                    self.results.setdefault(date_int, []).extend(result)
        
        warnings.filterwarnings("default", module='asf_search')

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
