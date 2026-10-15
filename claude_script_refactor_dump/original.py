import os, datetime, csv

import numpy as np
import scipy

import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

import brack2bing
import data_cryos
import PyPreprocess_Single
import script_CoreExplorer
import util_collector
import util_filters

'''
This module produces the quantitative comparison results for the manuscript:
Table 4 (retrieval on held-out simulated cores, with a regression baseline) and
Section 3.4 / Figure 10 (aggregate comparison against reference thickness products).

Results are cached per image, in the scene's own *_cache folder alongside SIT_plots.png,
as 'icematcher_result.npz'. Aggregate statistics are then read back from those caches,
so the seventeen-scene loop is resumable and figures can be rebuilt without a GPU.

Typical use:
    model, vl, floqs = load_model('260815_DiffusionBlock_langevin0.05_epoch500_t128h8d16_RESICE.pth')
    results = run_scenes(targets, model, vl, floqs)          # skips cached scenes
    rows = comparison_table(results)                          # Table 5
    fig  = plot_comparison(results)                           # Figure 10
    rows4, arrays = evaluate_heldout(model, vl, floqs)        # Table 4
'''

DEFAULT_VALIDS = ['AMSR_KH', 'CS2', 'SMOS', 'IC1', 'neXtSIM']
TITLES = {
    'AMSR_KH': ('AMSR-E ice history', '10 km'),
    'CS2'    : ('CryoSat-2 RDEFT4', '25 km'),
    'SMOS'   : ('SMAP/SMOS thin ice', '12.5 km'),
    'IC1'    : ('ICESat ISSITGR4', '25 km'),
    'neXtSIM': ('neXtSIM-F', '3 km'),
}
RESULT_FN = 'icematcher_result.npz'


# --- MODEL & CORPUS LOADING --- #

def load_model(fn, corpus_fn='em_resice.hdf5', device='cuda'):
    '''
    returns (model, vl, floqs).
    vl : feature segment lengths [S, T, rho, C4 real, C4 imag] = [90, 90, 90, 96, 96]
    floqs : the six admissible Floquet incidence angles, degrees
    '''
    model = torch.load(fn, weights_only=False).to(device)
    model.eval()

    sample = script_CoreExplorer.get_randoms(1, fn=corpus_fn, reshape=True)
    shapes = [sample['S'].real.shape[1:], sample['T'].real.shape[1:], sample['rho'].real.shape[1:]]

    collector = util_collector.Dataset(corpus_fn)
    C, P, I = collector.h5f['rf'].attrs['fshape'].tolist()   # (channels, polarizations, incidences)
    floqs = np.asarray(collector.h5f.attrs['floq'])
    collector.close()

    shapes += [(C, I, P, P), (C, I, P, P)]                   # covariance real, covariance imag
    vl = [int(np.prod(s)) for s in shapes]
    return model, vl, floqs


# --- WHITENING CHECKS --- #
# FixedRescaler initialises lazily on its first forward and FixedDescaler reads that same
# instance's buffers. slc2core calls the whitener on a vector that is zero except for the
# covariance block, so if a checkpoint arrives with an uninitialised whitener the statistics
# are taken from that near-zero batch: mean ~ 0 and sq_v ~ sqrt(eps) ~ 3e-3. Unwhitening
# then returns values in +/-1 instead of physical units, and the damage is silent because
# initialize() only ever runs once.

def corpus_statistics(vl, corpus_fn='em_resice.hdf5', count=None, device='cuda'):
    '''Mean and standard deviation of the raw training feed - the statistics whitening
    should be using. Returns (mean, sq_v) shaped (1, 1, 462).'''
    collector = util_collector.Dataset(corpus_fn)
    n = int(collector.h5f.attrs['w_ind']) if count is None else count
    feed, _ = gen_feed(n, collector, device)
    collector.close()
    return feed.mean(0, keepdim=True), (feed.var(0, keepdim=True) + 1e-5).pow(0.5)


def check_whitening(model, vl, corpus_fn='em_resice.hdf5', tol=0.25, verbose=True):
    '''
    Confirm the whitener carries usable statistics before a long run.

    Returns True if the profile-block statistics look like physical units. A near-zero sq_v
    is the signature of a whitener initialised on the zero-padded SAR vector.
    '''
    w = getattr(model, 'whitener', None)
    if w is None: print('model has no whitener'); return False
    if not getattr(w, 'initialized', False):
        print('WHITENER NOT INITIALISED - it will take its statistics from the first patch, '
              'which is zero outside the covariance block. Call fit_whitener() first.')
        return False

    mean, sq_v = w.mean.flatten()[:sum(vl[:3])], w.sq_v.flatten()[:sum(vl[:3])]
    ok = True
    if float(sq_v.median()) < 1e-2:
        print(f'WHITENER SUSPECT: median sq_v over the profile block is {float(sq_v.median()):.2e}; '
              'unwhitened output will collapse towards zero.')
        ok = False
    if verbose:
        S, T, R = mean[:vl[0]], mean[vl[0]:vl[0]+vl[1]], mean[vl[0]+vl[1]:]
        print(f'whitener means  S {float(S.mean()):+.2f} g/kg  T {float(T.mean()):+.2f} C  '
              f'rho {float(R.mean()):+.3f} g/cm3')
        print(f'whitener sq_v   median {float(sq_v.median()):.3f}  min {float(sq_v.min()):.2e}')

    if corpus_fn:
        try:
            cm, cs = corpus_statistics(vl, corpus_fn, device=w.mean.device)
            rel = ((sq_v - cs.flatten()[:sum(vl[:3])]).abs() / cs.flatten()[:sum(vl[:3])].clamp(min=1e-9))
            if float(rel.median()) > tol:
                print(f'WHITENER DISAGREES WITH CORPUS: median relative sq_v difference '
                      f'{float(rel.median()):.2f}. The checkpoint was not whitened on this corpus.')
                ok = False
            elif verbose:
                print(f'whitener matches corpus statistics (median relative difference '
                      f'{float(rel.median()):.3f})')
        except Exception as e:
            print(f'could not compare against the corpus ({type(e).__name__}: {e})')
    return ok


def fit_whitener(model, vl, corpus_fn='em_resice.hdf5', device='cuda'):
    '''
    Force the whitener onto the raw-corpus statistics, the same ones training used. Use when
    check_whitening reports an uninitialised or collapsed rescaler. Mutates the model in
    memory only; the checkpoint on disk is untouched.
    '''
    import torch
    mean, sq_v = corpus_statistics(vl, corpus_fn, device=device)
    w = model.whitener
    for name, val in (('mean', mean), ('sq_v', sq_v)):
        if hasattr(w, name): setattr(w, name, val.detach().clone())
        else: w.register_buffer(name, val.detach().clone())
    w.initialized = True
    print('whitener refitted from the corpus; descaling now returns physical units')
    return model


def descale_profile(model, prof, vl):
    '''Apply the profile-block descaling to an array already sliced to the first 270 entries.
    Used to repair region profiles cached while the whitener was collapsed.'''
    w = model.whitener
    mean = w.mean.flatten()[:sum(vl[:3])].detach().cpu().numpy()
    sq_v = w.sq_v.flatten()[:sum(vl[:3])].detach().cpu().numpy()
    return (np.asarray(prof).reshape(-1) * sq_v + mean).reshape(3, -1)


def looks_whitened(prof, vl=None):
    '''Heuristic: physical profiles have salinity to tens of g/kg and temperature well below
    zero. Everything inside +/-4 is standardised output.'''
    p = np.asarray(prof)
    return bool(np.nanmax(np.abs(p)) < 4.0)


def repair_region_profiles(cache_dir, model, vl, write=True):
    '''
    Descale region profiles that were cached in whitened form, without re-running inference.
    Returns the corrected array, or None if nothing needed fixing.
    '''
    fields = load_fields(cache_dir)
    if fields is None or 'region_profiles' not in fields: return None
    prof = np.asarray(fields['region_profiles'])
    if prof.size == 0 or not looks_whitened(prof):
        print(f'{os.path.basename(cache_dir)}: region profiles already physical')
        return None
    fixed = np.stack([descale_profile(model, p, vl) for p in prof]).astype(np.float32)
    if write:
        out = dict(fields); out['region_profiles'] = fixed
        np.savez_compressed(os.path.join(cache_dir, FIELDS_FN), **out)
    print(f'{os.path.basename(cache_dir)}: repaired {len(prof)} region profiles '
          f'(S now {fixed[:,0].min():.1f}..{fixed[:,0].max():.1f} g/kg)')
    return fixed


# --- PER-IMAGE RETRIEVAL --- #

def retrieve_scene(cache_dir, model, vl, floqs, sr=8, box_filter=(16, 4), cells_per_m=30,
                   region_fn='mask_bisplit.png', slc_ds=None, hist_bins=20):
    '''
    Retrieve one scene by calling script_CoreExplorer.analyze_image directly, so there is a
    single implementation of the retrieval and this module only handles caching, statistics
    and figures.

    valids is passed empty, so analyze_image performs no validation-data fetching: the
    reference products are attached afterwards by run_scene, which can take them from the
    fields cache instead of re-interpolating. analyze_image still writes its own
    SIT_histograms.png into cache_dir.

    Returned fields:
      h_geometric     the smoothed thickness map, in metres, exactly as analyze_image
                      computes it (retained bins / 30, then the box_filter average)
      region_profiles the region-aggregate (S, T, rho) profiles from the rX path
      lat, lon        the decimated geolocation grid, for reference interpolation

    Note that analyze_image applies its own box_filter internally, so no unsmoothed map is
    available here; h_geometric_raw is absent and the summaries drop it automatically.
    '''
    if slc_ds is None:
        slc_ds = PyPreprocess_Single.Dataset_SNAPbase(cache_dir=cache_dir, read_nc=True)

    metric, _, rX = script_CoreExplorer.analyze_image(
        cache_dir, model, list(vl), floqs, slc_ds=slc_ds, rmask_fn=region_fn,
        sr=sr, box_filter=box_filter, valids=[], hist_bins=hist_bins)

    metric = np.asarray(metric.detach().cpu() if hasattr(metric, 'detach') else metric,
                        dtype=np.float32)

    # rX holds one unwhitened vector per region; the first 270 entries are [S | T | rho]
    vends = np.array(vl).cumsum()
    profiles = []
    for v in rX:
        v = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
        profiles.append(v.reshape(-1)[:vends[2]].reshape(3, -1))

    fields = {
        'h_geometric': metric,
        'region_profiles': np.asarray(profiles, dtype=np.float32),
        'lat': np.asarray(slc_ds.lat[::sr, ::sr], dtype=np.float32),
        'lon': np.asarray(slc_ds.lon[::sr, ::sr], dtype=np.float32),
        'vh_intensity': np.abs(np.asarray(slc_ds.slc[2, ::sr, ::sr])).astype(np.float32),
        'date': datetime.datetime.utcfromtimestamp(slc_ds.metadata['0dopp_t0'][0]).strftime('%Y%m%d'),
        'subsample': sr,
        'c4_smooth': np.asarray((11, 11)),
        'post_boxcar': np.asarray(box_filter),
        'saturated_fraction': np.float32(np.nan),
        'cells_per_m': np.int64(cells_per_m),
    }
    return fields


def region_mask(cache_dir, shape, sr=1, fn='mask_bisplit.png'):
    '''
    Load the hand-drawn bisplit mask, decimated by the same sr as the SLC and cropped
    to `shape`. The mask is a visual reference for inspecting model behaviour and does
    not reliably delineate ice regimes, so it is used for figures only and never enters
    a reported statistic.
    '''
    path = os.path.join(cache_dir, fn)
    if not os.path.exists(path): return None
    m = np.asarray(Image.open(path), dtype=np.uint8)
    m = m[..., 0] if m.ndim == 3 else m
    m = m[::sr, ::sr]
    h, w = min(m.shape[0], shape[0]), min(m.shape[1], shape[1])
    out = np.full(shape, 255, dtype=np.uint8)
    out[:h, :w] = m[:h, :w]
    return out


# --- REFERENCE PRODUCTS --- #
# data_cryos.get has no 'CS2' case (so CryoSat-2 silently returned None through analyze_image),
# passes a bare `nearest` in its AMSR branch, and its ICESat_Gridded closes the netCDF handle
# before reading. These are routed around here rather than edited in place.

def icesat_gridded(YYYYMMDD, lat, lon, variable='sit', mode='nearest',
                   dn='/mnt/d/ICESat1_data_060222_080321/SEASONAL_GRIDDED'):
    import netCDF4
    key = {'sit':'ice_thickness_int', 'sit_r':'ice_thickness', 'sit_u':'ice_thickness_unc',
           'frb':'total_freeboard_int', 'sdp':'snow_depth_int', 'sdn':'snow_density',
           'sic':'sea_ice_conc', 'type':'ice_type'}[variable]
    dt, match_fn = datetime.datetime.strptime(str(YYYYMMDD), '%Y%m%d'), None
    for fn in sorted(os.listdir(dn)):
        if not fn.endswith('.nc'): continue
        with netCDF4.Dataset(os.path.join(dn, fn)) as prod:
            first = datetime.datetime.strptime(prod['campaign_dates'].getncattr('first_day'), '%Y-%m-%d')
            last  = datetime.datetime.strptime(prod['campaign_dates'].getncattr('last_day' ), '%Y-%m-%d')
        if first <= dt <= last: match_fn = fn; break
    if match_fn is None: raise ValueError(f'{YYYYMMDD} falls between ICESat campaigns')

    with netCDF4.Dataset(os.path.join(dn, match_fn)) as prod:
        dlat, dlon, ddat = [np.ma.filled(np.asarray(prod[k][:], dtype=np.float64), np.nan)
                            for k in ('latitude', 'longitude', key)]
    good = np.isfinite(dlat) & np.isfinite(dlon) & np.isfinite(ddat)
    interp = data_cryos.scipy_interp(ddat[good], np.stack(data_cryos.wgs84_llh2xyz(dlat[good], dlon[good]), axis=-1), mode=mode)
    return np.asarray(interp(np.stack(data_cryos.wgs84_llh2xyz(lat.ravel(), lon.ravel()), axis=-1))).reshape(lat.shape)


def get_reference(name, YYYYMMDD, lat, lon, mode='nearest', valid_range=(0.0, 6.0)):
    '''Returns (array, None) on success or (None, reason) on failure; never raises.'''
    fetchers = {
        'AMSR_KH': lambda: data_cryos.AMSR_KimuraN_HasumiH(YYYYMMDD, lat, lon, mode=mode),
        'CS2'    : lambda: data_cryos.CryoSat2_Data(YYYYMMDD, lat, lon, variable='sit', mode=mode),
        'SMOS'   : lambda: data_cryos.UBremen_SMOS(YYYYMMDD, lat, lon, mode=mode),
        'neXtSIM': lambda: data_cryos.AOSI(YYYYMMDD, lat, lon, mode),
        'IC1'    : lambda: icesat_gridded(YYYYMMDD, lat, lon, 'sit', mode),
    }
    if name not in fetchers: return None, f'no fetcher for {name}'
    try: dat = fetchers[name]()
    except Exception as e: return None, f'{type(e).__name__}: {e}'
    if dat is None: return None, 'fetcher returned None'

    dat = np.asarray(dat.detach().cpu() if hasattr(dat, 'detach') else dat, dtype=np.float64).squeeze()
    if dat.shape != lat.shape: return None, f'shape {dat.shape} != SAR grid {lat.shape}'
    dat[(dat < valid_range[0]) | (dat > valid_range[1])] = np.nan
    if not np.isfinite(dat).any(): return None, 'no valid values over footprint'
    return dat, None


def attach_references(fields, valids=None, valid_range=(0.0, 6.0), verbose=True):
    valids = DEFAULT_VALIDS if valids is None else valids
    lat, lon = fields['lat'].astype(np.float64), fields['lon'].astype(np.float64)
    misses = {}
    for name in valids:
        dat, reason = get_reference(name, int(fields['date']), lat, lon, valid_range=valid_range)
        if dat is None:
            misses[name] = reason
            if verbose: print(f'    {name:8s} unavailable - {reason}')
        else:
            fields[f'ref_{name}'] = dat.astype(np.float32)
            if verbose: print(f'    {name:8s} ok ({100*np.isfinite(dat).mean():.1f}% valid)')
    fields['reference_misses'] = misses
    return fields


# --- CACHING (per image, in the scene's own folder) --- #
# Two files are written into each scene's *_cache folder:
#   icematcher_summary.npz : small reduction - sums, per-cell tables, 2D histograms.
#                            Aggregate statistics are computed from these alone.
#   icematcher_fields.npz  : the full-resolution retrieved fields, for re-plotting.
# Only one image's fields are ever held in memory: run_scene writes its figures and its
# summary, then releases the arrays before moving on. Nothing in the aggregate path
# opens a fields file.

SUMMARY_FN = 'icematcher_summary.npz'
FIELDS_FN  = 'icematcher_fields.npz'
FIELD_KEYS = ['h_geometric', 'lat', 'lon']   # whichever are present
HIST_BINS = 100
HIST_RANGE = (0.0, 6.0)
CELL_TAIL_THRESHOLDS = (0.5, 1.0, 2.0)   # metres; per-cell counts above each, for diagnostics
BAND_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 6.0)


def pair_sums(reference, retrieval):
    '''
    Sufficient statistics for bias, RMSE, MAE, correlation and the OLS fit.
    These add across scenes, so the pooled statistic is exact without pooling pixels.
    '''
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x, y = np.asarray(reference)[m].astype(np.float64), np.asarray(retrieval)[m].astype(np.float64)
    d = y - x
    return {'n': float(x.size), 'sx': x.sum(), 'sy': y.sum(), 'sxx': (x*x).sum(),
            'syy': (y*y).sum(), 'sxy': (x*y).sum(), 'sad': np.abs(d).sum()}


def stats_from_sums(S):
    '''
    Bias, RMSE, MAE, error standard deviation, r and slope from accumulated sums. All are
    exact functions of the sums, so pooling scenes is identical to pooling pixels.

    Median bias is the one statistic lost by the streaming design: order statistics cannot
    be recombined from sums. If it is wanted, take it per scene from that scene's fields.
    '''
    n = S['n']
    if n < 2: return None
    sd, sdd = S['sy'] - S['sx'], S['syy'] - 2*S['sxy'] + S['sxx']   # sums of d and d^2
    out = {'n_pixels': int(n),
           'bias': sd / n,
           'rmse': float(np.sqrt(max(sdd / n, 0.0))),
           'mae': S['sad'] / n,
           'std_error': float(np.sqrt(max((sdd - sd*sd/n) / (n - 1), 0.0))) if n > 1 else np.nan,
           'ref_mean': S['sx'] / n, 'ret_mean': S['sy'] / n}
    vx, vy = n*S['sxx'] - S['sx']**2, n*S['syy'] - S['sy']**2
    if vx > 1e-9 and vy > 1e-9:
        cov = n*S['sxy'] - S['sx']*S['sy']
        out['pearson_r'] = cov / np.sqrt(vx*vy)
        out['slope'] = cov / vx
        out['intercept'] = (S['sy'] - out['slope']*S['sx']) / n
    else:
        out['pearson_r'] = out['slope'] = out['intercept'] = np.nan
    return out


def describe_retrieval(fields_or_dir, quantities=('h_geometric',)):
    '''
    Percentiles and clipping diagnostics for the retrieved fields of one scene. Use this to
    check a plot against the field it came from: if the field reaches 2 m but the cell-mean
    panel tops out at 0.5, the compression is in the aggregation, not in the retrieval.
    '''
    fields = load_fields(fields_or_dir) if isinstance(fields_or_dir, str) else fields_or_dir
    if fields is None: print('no cached fields'); return None
    dz = 1.0 / 30
    rows = []
    for q in quantities:
        if q not in fields: continue
        v = np.asarray(fields[q], np.float64).ravel()
        finite = np.isfinite(v)
        d = v[finite]
        if d.size == 0: continue
        row = {'field': q, 'n_finite': int(d.size), 'nan_fraction': float(1 - finite.mean()),
               'min': float(d.min()), 'max': float(d.max()),
               'zero_fraction': float((d <= dz/2).mean())}
        for p in (5, 25, 50, 75, 95, 99):
            row[f'p{p}'] = float(np.percentile(d, p))
        for thr in CELL_TAIL_THRESHOLDS:
            row[f'frac_above_{thr}'] = float((d > thr).mean())
        rows.append(row)
        print(f'{q:20s} max {row["max"]:.2f} m  median {row["p50"]:.2f}  p99 {row["p99"]:.2f}  '
              f'zeros {100*row["zero_fraction"]:.1f}%  >1 m {100*row["frac_above_1.0"]:.1f}%  '
              f'NaN {100*row["nan_fraction"]:.1f}%')
    return rows


def describe_cells(summaries, product, quantity='h_geometric', basis='all'):
    '''
    How the cell aggregation relates to the pixels behind it. If max_y greatly exceeds the
    cell means, the cells are internally bimodal and a mean-only panel understates the
    retrieval.
    '''
    cells = pooled_cells(summaries, product, quantity, basis)
    if cells is None: print(f'{product}: no cells'); return None
    n = cells['count'].astype(np.float64)
    mean_y = cells['sum_y'] / n
    out = {'n_cells': int(n.size), 'total_pixels': int(n.sum()),
           'cell_mean_max': float(mean_y.max()), 'cell_mean_median': float(np.median(mean_y)),
           'pixel_max': float(cells['max_y'].max()) if 'max_y' in cells else np.nan,
           'reference_max': float(cells['value'].max())}
    for thr in CELL_TAIL_THRESHOLDS:
        k = f'n_above_{thr}'
        if k in cells: out[f'pixels_above_{thr}'] = int(cells[k].sum())
    print(f'{product}: {out["n_cells"]} cells, {out["total_pixels"]:,} px | '
          f'cell means reach {out["cell_mean_max"]:.2f} m, individual pixels reach '
          f'{out["pixel_max"]:.2f} m | reference reaches {out["reference_max"]:.2f} m')
    for thr in CELL_TAIL_THRESHOLDS:
        if f'pixels_above_{thr}' in out:
            print(f'   pixels above {thr} m: {out[f"pixels_above_{thr}"]:,}')
    return out


def cell_statistics(cells, min_count=1):
    '''
    Unweighted statistics over cell means: every reference cell contributes equally,
    whatever its pixel count.

    This is the local view. The pixel-level statistics from stats_from_sums weight each cell
    by its area, so a single large cell can dominate; here a 25 km cell and a 3 km cell count
    the same. Comparing the two slopes separates agreement with the reference field as a
    whole from agreement cell by cell.
    '''
    if cells is None or cells['value'].size == 0: return None
    keep = cells['count'] >= min_count
    x = cells['value'][keep]
    y = cells['sum_y'][keep] / cells['count'][keep].astype(np.float64)
    if x.size < 2: return None
    d = y - x
    out = {'n_cells': int(x.size), 'bias_cell': float(d.mean()),
           'rmse_cell': float(np.sqrt((d**2).mean())), 'mae_cell': float(np.abs(d).mean())}
    if x.std() > 1e-9 and y.std() > 1e-9:
        out['pearson_r_cell'] = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        out['slope_cell'], out['intercept_cell'] = float(slope), float(intercept)
    else:
        out['pearson_r_cell'] = out['slope_cell'] = out['intercept_cell'] = np.nan
    return out


def scene_statistics(summaries, product, quantity='h_geometric', basis='all'):
    '''Per-image statistics: one stats dict per scene that carries this product.'''
    out = []
    for s in summaries:
        key = f'sums|{_tag(product, quantity, basis)}'
        if key not in s: continue
        st = stats_from_sums(_unpack_sums(s[key]))
        if st is None: continue
        st['scene'] = str(np.asarray(s['scene']).item() if hasattr(s['scene'], 'item') else s['scene'])
        out.append(st)
    return out


def image_mean_statistics(summaries, product, quantity='h_geometric', basis='all'):
    '''
    Per-image statistics averaged over scenes: every image counts once, whatever its size or
    how many reference cells it covers.

    This is the third view. The pixel statistics weight by area, the cell statistics weight
    every reference cell equally within the pooled set, and this weights every acquisition
    equally - so a scene that happens to span more reference cells does not dominate. The
    spread across scenes is reported alongside, and is usually the more informative number:
    a large slope_image_sd means the retrieval behaves differently from image to image.
    '''
    per = scene_statistics(summaries, product, quantity, basis)
    if not per: return None
    out = {'n_images': len(per)}
    for src, dst in (('bias', 'bias_image'), ('rmse', 'rmse_image'), ('mae', 'mae_image'),
                     ('pearson_r', 'pearson_r_image'), ('slope', 'slope_image'),
                     ('intercept', 'intercept_image')):
        vals = np.array([p[src] for p in per], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[dst] = float(vals.mean()) if vals.size else np.nan
        out[f'{dst}_sd'] = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
    return out


def point_statistics(retrieval, reference):
    '''Statistics directly from two pixel arrays. Equivalent to stats_from_sums(pair_sums(..)),
    kept for one-off use on a single scene held in memory; the aggregate path uses the sums.'''
    return stats_from_sums(pair_sums(reference, retrieval))


def add_sums(a, b):
    return {k: a[k] + b[k] for k in a}


def cell_table(reference, retrieval, decimals=6):
    '''
    One row per distinct reference value: its value, the pixel count, and the sums of the
    retrieval over it. Nearest-neighbour interpolation makes the reference piecewise
    constant, so a distinct value stands for a distinct source cell. This is what the
    bootstrap resamples, and it is a few thousand rows rather than millions of pixels.
    '''
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x, y = np.asarray(reference)[m].astype(np.float64), np.asarray(retrieval)[m].astype(np.float64)
    _, inv = np.unique(np.round(x, decimals), return_inverse=True)
    count = np.bincount(inv).astype(np.int64)
    n_cells = count.size
    # group on the rounded value, but carry the exact per-cell mean, so statistics
    # rebuilt from the table match a direct pixel computation to machine precision
    out = {'value': np.bincount(inv, weights=x) / count,
           'count': count,
           'sum_y': np.bincount(inv, weights=y),
           'sum_yy': np.bincount(inv, weights=y*y)}
    # within-cell range and upper tail: a cell mean alone cannot show that a bimodal cell
    # contains thick pixels, so the extremes are carried explicitly
    hi = np.full(n_cells, -np.inf); lo = np.full(n_cells, np.inf)
    np.maximum.at(hi, inv, y); np.minimum.at(lo, inv, y)
    out['max_y'], out['min_y'] = hi, lo
    for thr in CELL_TAIL_THRESHOLDS:
        out[f'n_above_{thr}'] = np.bincount(inv, weights=(y > thr).astype(np.float64))
    return out


def band_sums(reference, retrieval, edges=BAND_EDGES):
    '''Per thickness band: count and error sums, additive across scenes. Stratified by the
    reference product, never by the hand-drawn region mask.'''
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x, y = np.asarray(reference)[m].astype(np.float64), np.asarray(retrieval)[m].astype(np.float64)
    d = y - x
    n, sd, sdd = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        n.append(float(sel.sum())); sd.append(d[sel].sum()); sdd.append((d[sel]**2).sum())
    return np.array([n, sd, sdd], dtype=np.float64)


def joint_hist(reference, retrieval, bins=HIST_BINS, rng=HIST_RANGE):
    '''2D histogram on a fixed grid, so scene histograms add to give the pooled density
    that Figure 10 draws.'''
    m = np.isfinite(reference) & np.isfinite(retrieval)
    H, _, _ = np.histogram2d(np.asarray(reference)[m], np.asarray(retrieval)[m],
                             bins=bins, range=[rng, rng])
    return H.astype(np.float32)


def _misses_dict(x):
    '''Accept the miss record either as the live dict or as the array of 'key: reason'
    strings it becomes after a round trip through the npz.'''
    if x is None: return {}
    if isinstance(x, dict): return x
    out = {}
    for el in np.atleast_1d(np.asarray(x)).ravel():
        t = str(el)
        if not t: continue
        k, _, v = t.partition(':')
        out[k.strip()] = v.strip()
    return out


def summarize_scene(fields, cache_dir, quantities=('h_geometric',),
                    valids=None, edges=BAND_EDGES):
    '''
    Reduce one image to everything the aggregates need. Returns a flat dict of small
    arrays; the pixel fields are not referenced again after this.
    '''
    valids = DEFAULT_VALIDS if valids is None else valids
    misses = _misses_dict(fields.get('reference_misses'))
    out = {'scene': os.path.basename(cache_dir), 'date': str(fields['date']),
           'missing_products': np.array(sorted(misses.keys())),
           'miss_reasons': np.array([f'{k}: {v}' for k, v in sorted(misses.items())]),
           'subsample': fields.get('subsample', np.int64(-1)),
           'c4_smooth': np.asarray(fields.get('c4_smooth', (0, 0))),
           'post_boxcar': np.asarray(fields.get('post_boxcar', (0, 0))),
           'saturated_fraction': fields.get('saturated_fraction', np.float32(np.nan)),
           'products': np.array([p for p in valids if f'ref_{p}' in fields]),
           'quantities': np.array(list(quantities)), 'band_edges': np.asarray(edges)}

    # 'all'    : each product over its own valid pixels - maximum coverage, unbalanced
    # 'common' : every product over the pixels valid in ALL products present in this scene,
    #            so the products are compared on identical support within the scene
    common = None
    for p in out['products']:
        v = np.isfinite(np.asarray(fields[f'ref_{p}'], np.float64))
        common = v if common is None else (common & v)
    out['n_common_pixels'] = float(common.sum()) if common is not None else 0.0

    quantities = [q for q in quantities if q in fields]
    out['quantities'] = np.array(list(quantities))
    for q in quantities:
        for p in out['products']:
            ref, ret = np.asarray(fields[f'ref_{p}'], np.float64), np.asarray(fields[q], np.float64)
            for basis, sel in (('all', None), ('common', common)):
                r, t = (ref, ret) if sel is None else (np.where(sel, ref, np.nan), np.where(sel, ret, np.nan))
                S = pair_sums(r, t)
                tag = f'{q}|{p}' if basis == 'all' else f'common|{q}|{p}'
                out[f'sums|{tag}'] = np.array([S[k] for k in ('n','sx','sy','sxx','syy','sxy','sad')])
                out[f'bands|{tag}'] = band_sums(r, t, edges)
                out[f'hist|{tag}'] = joint_hist(r, t)
                for k, v in cell_table(r, t).items(): out[f'cells|{tag}|{k}'] = v

    # reference against reference, for the inter-product disagreement panel
    for i, a in enumerate(out['products']):
        for b in out['products'][i+1:]:
            xa, xb = np.asarray(fields[f'ref_{a}'], np.float64), np.asarray(fields[f'ref_{b}'], np.float64)
            S = pair_sums(xb, xa)
            out[f'sums|inter|{a}~{b}'] = np.array([S[k] for k in ('n','sx','sy','sxx','syy','sxy','sad')])
            out[f'hist|inter|{a}~{b}'] = joint_hist(xb, xa)

    # distribution of the unsmoothed retrieval, which is where the bimodality shows
    dist = np.asarray(fields.get('h_geometric_raw', fields['h_geometric']), np.float64)
    out['thickness_hist'] = np.histogram(dist[np.isfinite(dist)], bins=HIST_BINS, range=HIST_RANGE)[0].astype(np.int64)
    out['thickness_bins'] = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    return out


def _unpack_sums(arr):
    return dict(zip(('n','sx','sy','sxx','syy','sxy','sad'), [float(v) for v in arr]))


def save_summary(cache_dir, summary, fn=SUMMARY_FN):
    np.savez_compressed(os.path.join(cache_dir, fn), **summary)
    return os.path.join(cache_dir, fn)


def load_summary(cache_dir, fn=SUMMARY_FN):
    '''Returns the small per-scene reduction, or None if the scene has not been run.'''
    path = os.path.join(cache_dir, fn)
    if not os.path.exists(path): return None
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    out['cache_dir'] = cache_dir
    return out


def load_cached_references(cache_dir, shape=None, fn=FIELDS_FN):
    '''
    Read the ref_* arrays back from a scene's fields cache, so a re-retrieval does not have
    to re-interpolate five products or re-download neXtSIM.

    Returns (refs, misses, reason): refs is {} when the cache is unusable and reason says
    why, or the reference arrays with reason None. misses is always a dict.
    A shape mismatch - a different sr, say - counts as unusable.
    '''
    path = os.path.join(cache_dir, fn)
    if not os.path.exists(path): return {}, {}, 'no fields cache'
    out = {}
    with np.load(path, allow_pickle=True) as z:
        keys = [k for k in z.files if str(k).startswith('ref_')]
        if not keys: return {}, {}, 'fields cache holds no reference fields'
        for k in keys:
            v = z[k]
            if shape is not None and v.shape != shape:
                return {}, {}, f'cached reference {v.shape} does not match retrieval {shape}'
            out[k] = v
        misses = _misses_dict(z['reference_misses']) if 'reference_misses' in z.files else {}
    return out, misses, None


def save_fields(cache_dir, fields, keys=None, fn=FIELDS_FN):
    '''Cache the full-resolution fields for re-plotting. Pass save=False to run_scene if
    disk is tight - the summaries alone are sufficient for every aggregate.'''
    keys = FIELD_KEYS if keys is None else keys
    out = {k: np.asarray(fields[k], np.float32) for k in keys if k in fields}
    out.update({k: np.asarray(v, np.float32) for k, v in fields.items() if str(k).startswith('ref_')})
    # carry the scalars and the region cores too, so a summary can be rebuilt from this file
    # alone - see resummarize() - without re-running inference
    for k in ('date', 'subsample', 'c4_smooth', 'post_boxcar', 'cells_per_m',
              'region_profiles', 'region_counts'):
        if k in fields: out[k] = np.asarray(fields[k]) if not isinstance(fields[k], str) else str(fields[k])
    out['reference_misses'] = np.array([f'{k}: {v}' for k, v in (fields.get('reference_misses') or {}).items()])
    np.savez_compressed(os.path.join(cache_dir, fn), **out)
    return os.path.join(cache_dir, fn)


def load_fields(cache_dir, fn=FIELDS_FN):
    path = os.path.join(cache_dir, fn)
    if not os.path.exists(path): return None
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def resummarize(cache_dir, quantities=('h_geometric',), valids=None,
                edges=BAND_EDGES, write=True):
    '''
    Rebuild a scene's summary from its cached fields, with no inference. Use this after
    changing the thickness bands, the histogram range, or which quantity is compared -
    none of which need the model re-run.
    '''
    fields = load_fields(cache_dir)
    if fields is None: return None
    fields = {k: (str(v) if v.dtype.kind in 'US' and v.ndim == 0 else v) for k, v in fields.items()}
    fields.setdefault('saturated_fraction', np.float32(np.nan))
    summary = summarize_scene(fields, cache_dir, quantities=quantities, valids=valids, edges=edges)
    if write: save_summary(cache_dir, summary)
    summary['cache_dir'] = cache_dir
    return summary


def run_scene(cache_dir, model, vl, floqs, skip_existing=True, plot=True, save=True,
              valids=None, reuse_references=True, **kw):
    '''
    Retrieve one image, write its figures and caches, and return only the small summary.
    The pixel fields are released before this returns, so memory never exceeds one image.

    reuse_references takes the ref_* arrays from the existing fields cache rather than
    re-fetching them, which is the expensive part of a re-run: five nearest-neighbour
    interpolations onto the full SAR grid plus a CMEMS download. The reference products do
    not change, so this is safe whenever the geolocation grid is unchanged; the shape is
    checked and a mismatch falls back to fetching.
    '''
    import gc
    cached = load_summary(cache_dir)
    if cached is not None and skip_existing:
        print(f'{os.path.basename(cache_dir)}: cached, skipping')
        return cached

    print(f'{os.path.basename(cache_dir)}: retrieving')
    fields = retrieve_scene(cache_dir, model, vl, floqs, **kw)
    print(f'  retrieved {fields["date"]}')

    refs = {}
    if reuse_references:
        refs, misses, reason = load_cached_references(cache_dir, fields['h_geometric'].shape)
        if refs:
            fields.update(refs)
            fields['reference_misses'] = misses
            print(f'  reusing {len(refs)} cached reference fields')
        else:
            print(f'  cached references unusable ({reason}); fetching')
    if not refs:
        fields = attach_references(fields, valids=valids)

    if plot:
        for p in plot_scene_figures(cache_dir, fields, sr=kw.get('sr', 1)):
            print(f'  wrote {os.path.basename(p)}')
    summary = summarize_scene(fields, cache_dir, valids=valids)
    save_summary(cache_dir, summary)
    if save: save_fields(cache_dir, fields)

    del fields
    gc.collect()
    summary['cache_dir'] = cache_dir
    return summary


def run_scenes(cache_dirs, model, vl, floqs, skip_existing=True, plot=True, save=True,
               valids=None, reuse_references=True, **kw):
    '''Loop over scenes, holding one image at a time. Returns the list of summaries.

    reuse_references=True takes the validation products from each scene's fields cache, so a
    re-retrieval costs only inference. Set it False to force a fresh fetch.'''
    summaries = []
    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            print(f'{cache_dir}: missing, skipping'); continue
        try:
            summaries.append(run_scene(cache_dir, model, vl, floqs, skip_existing, plot, save,
                                       valids, reuse_references, **kw))
        except Exception as e:
            print(f'{os.path.basename(cache_dir)}: failed - {type(e).__name__}: {e}')
    return summaries


def collect_summaries(cache_dirs):
    '''Read back whatever has been cached so far, ignoring scenes not yet run.'''
    out = [load_summary(d) for d in cache_dirs]
    return [s for s in out if s is not None]


# --- AGGREGATE STATISTICS --- #
# Everything below consumes summaries, never pixels. Comparison is on the SAR grid, with
# each product nearest-neighbour interpolated onto it, so neighbouring pixels within one
# reference cell are perfectly correlated. Alongside the pixel count each table reports
# n_cells - the number of distinct reference values, i.e. independent reference
# observations - and intervals resample cells rather than pixels.

def _tag(product, quantity, basis):
    return f'{quantity}|{product}' if basis == 'all' else f'common|{quantity}|{product}'


def select_scenes(summaries, valids=None, require_all=False):
    '''Scenes to aggregate over. require_all keeps only scenes carrying every product,
    which makes the product rows directly comparable at the cost of coverage.'''
    valids = DEFAULT_VALIDS if valids is None else valids
    if not require_all: return summaries
    return [s for s in summaries if set(valids).issubset(set(s['products'].tolist()))]


def pooled_sums(summaries, product, quantity='h_geometric', basis='all'):
    total, n_scenes = None, 0
    for s in summaries:
        key = f'sums|{_tag(product, quantity, basis)}'
        if key not in s: continue
        S = _unpack_sums(s[key])
        total = S if total is None else add_sums(total, S)
        n_scenes += 1
    return total, n_scenes


def pooled_cells(summaries, product, quantity='h_geometric', basis='all'):
    '''Concatenated cell tables across scenes. Cells are kept per scene, never merged.'''
    keys = ['value', 'count', 'sum_y', 'sum_yy', 'max_y', 'min_y'] + \
           [f'n_above_{t}' for t in CELL_TAIL_THRESHOLDS]
    parts = {k: [] for k in keys}
    idx, names = [], []
    for s in summaries:
        base = f'cells|{_tag(product, quantity, basis)}'
        if f'{base}|value' not in s: continue
        for k in list(parts):
            if f'{base}|{k}' in s: parts[k].append(s[f'{base}|{k}'])
        idx.append(np.full(s[f'{base}|value'].size, len(names), dtype=np.int32))
        names.append(str(np.asarray(s['scene']).item() if hasattr(s['scene'], 'item') else s['scene']))
    if not parts['value']: return None
    out = {k: np.concatenate(v) for k, v in parts.items() if v}
    out['scene_index'] = np.concatenate(idx)      # which image each cell came from
    out['scene_names'] = np.array(names)
    return out


def bootstrap_by_cell(cells, statistic='bias', n_boot=1000, seed=0):
    '''
    Percentile CI, resampling reference cells. Exact from the cell sums: for a cell with
    reference value x, count n and retrieval sums Sy, Syy,
        sum of errors    = Sy - n x
        sum of squares   = Syy - 2 x Sy + n x^2
    '''
    if cells is None or cells['value'].size < 3: return (np.nan, np.nan), 0
    x, n, sy, syy = cells['value'].astype(np.float64), cells['count'].astype(np.float64), cells['sum_y'], cells['sum_yy']
    err = sy - n*x
    sq = syy - 2*x*sy + n*x*x
    n_cells = x.size

    rng, vals = np.random.default_rng(seed), np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        tot = n[pick].sum()
        if statistic == 'bias': vals[i] = err[pick].sum() / tot
        elif statistic == 'rmse': vals[i] = np.sqrt(max(sq[pick].sum() / tot, 0.0))
        else: raise ValueError(f'invalid statistic {statistic}')
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), n_cells


def stratified_statistics(summaries, product, quantity='h_geometric', basis='all', min_pixels=100):
    '''Bias and RMSE within reference-thickness bands, pooled from the band sums.'''
    total, edges = None, None
    for s in summaries:
        key = f'bands|{_tag(product, quantity, basis)}'
        if key not in s: continue
        total = s[key].copy() if total is None else total + s[key]
        edges = s['band_edges']
    if total is None: return []
    rows = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        n, sd, sdd = total[0, k], total[1, k], total[2, k]
        if n < min_pixels: continue
        rows.append({'band': f'{lo:.1f}-{hi:.1f} m', 'n_pixels': int(n),
                     'bias': float(sd/n), 'rmse': float(np.sqrt(max(sdd/n, 0.0)))})
    return rows


# --- SEASONAL GROUPING --- #
# Scenes are binned by the acquisition date each summary already carries, taken from the
# first-line zero-Doppler time. Growth covers October to April, melt May to September;
# seasons pool across years, so all Marches from every year fall in the same bin.

SEASONS = {'growth': (10, 11, 12, 1, 2, 3, 4), 'melt': (5, 6, 7, 8, 9)}


def season_of(date, seasons=None):
    '''Season label for a YYYYMMDD date, or None if it falls in no defined bin.'''
    seasons = SEASONS if seasons is None else seasons
    month = int(str(np.asarray(date).item() if hasattr(date, 'item') else date)[4:6])
    for name, months in seasons.items():
        if month in months: return name
    return None


def split_by_season(summaries, seasons=None):
    '''Group summaries into {season: [summary, ...]}, preserving the season order.'''
    seasons = SEASONS if seasons is None else seasons
    out = {k: [] for k in seasons}
    for s in summaries:
        k = season_of(s['date'], seasons)
        if k is not None: out[k].append(s)
    return {k: v for k, v in out.items() if v}


def seasonal_comparison_table(summaries, quantity='h_geometric', valids=None, basis='all',
                              require_all=False, n_boot=1000, seasons=None,
                              include_total=True, verbose=True):
    '''
    comparison_table split by season, with a 'season' column on every row.

    Each season is an independent pooling: its rows use only the scenes acquired in that
    season, so n_pixels, n_cells and n_scenes are per season and the seasons never mix.
    include_total appends the all-season rows under the label 'all' for reference.

    Product coverage is very uneven across seasons - CryoSat-2 is October to April only and
    ICESat exists only within its campaigns - so read these rows alongside
    availability_table rather than comparing seasons at face value.
    '''
    groups = split_by_season(summaries, seasons)
    rows = []
    for name, subset in groups.items():
        if verbose: print(f'{name} ({len(subset)} scenes):')
        for r in comparison_table(subset, quantity, valids, basis, require_all, n_boot, verbose):
            r['season'] = name
            rows.append(r)
    if include_total:
        if verbose: print(f'all seasons ({len(summaries)} scenes):')
        for r in comparison_table(summaries, quantity, valids, basis, require_all, n_boot, verbose):
            r['season'] = 'all'
            rows.append(r)
    return rows


def seasonal_availability(summaries, valids=None, seasons=None):
    '''Scene and product counts per season - the context the seasonal table needs.'''
    valids = DEFAULT_VALIDS if valids is None else valids
    groups = split_by_season(summaries, seasons)
    rows = []
    for name, subset in groups.items():
        row = {'season': name, 'n_scenes': len(subset)}
        for p in valids:
            row[p] = sum(1 for s in subset if p in set(s['products'].tolist()))
        rows.append(row)
        print(f'  {name:8s} {len(subset):>3} scenes | ' +
              '  '.join(f'{p} {row[p]}' for p in valids))
    return rows


def plot_seasonal_comparison(summaries, quantity='h_geometric', valids=None, basis='all',
                             seasons=None, figsize=(7.09, None), min_count=1):
    '''
    One row of cell-mean panels per season, one column per reference product. Empty cells
    are drawn as blanks so the grid stays aligned when a product is absent in a season.
    Returns the figure.
    '''
    import matplotlib.pyplot as plt
    valids = DEFAULT_VALIDS if valids is None else valids
    paper_style()
    groups = split_by_season(summaries, seasons)
    if not groups:
        print('no scene falls in a defined season'); return None

    present = [p for p in valids
               if any(pooled_sums(sub, p, quantity, basis)[0] is not None for sub in groups.values())]
    if not present:
        print('no product overlaps the retrieval'); return None

    nrow, ncol = len(groups), len(present)
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] or 3.0*nrow),
                             squeeze=False, layout='constrained')
    for r, (name, subset) in enumerate(groups.items()):
        for c, product in enumerate(present):
            ax = axes[r][c]
            total, n_scenes = pooled_sums(subset, product, quantity, basis)
            if total is None or total['n'] < 2:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f'{TITLES[product][0]}\nnot available in {name}',
                        ha='center', va='center', fontsize=6.5, transform=ax.transAxes)
                continue
            _cell_panel(ax, pooled_cells(subset, product, quantity, basis),
                        stats_from_sums(total), TITLES[product][0],
                        f'{name}, {n_scenes} scenes', min_count=min_count,
                        image_stats=image_mean_statistics(subset, product, quantity, basis))
            ax.set_xlabel('Reference thickness (m)')
            ax.set_ylabel('Retrieved thickness (m)' if c == 0 else '')
    return fig


def comparison_table(summaries, quantity='h_geometric', valids=None, basis='all',
                     require_all=False, n_boot=1000, verbose=True):
    '''
    One row per reference product.

    basis='all' pools each product over its own valid pixels: maximum coverage, but the
    rows rest on different scenes and different footprints, so a difference in bias
    between two rows is confounded with which data each one saw.
    basis='common' restricts every product to the pixels valid in all products present in
    that scene. Combined with require_all=True, which keeps only scenes carrying the full
    product set, this gives a balanced comparison on identical support - fewer pixels, but
    the rows become directly comparable. Report the balanced version alongside the
    coverage-maximising one rather than choosing between them.
    '''
    valids = DEFAULT_VALIDS if valids is None else valids
    summaries = select_scenes(summaries, valids, require_all)
    if not summaries:
        print('no scene carries the full product set'); return []
    rows = []
    for product in valids:
        total, n_scenes = pooled_sums(summaries, product, quantity, basis)
        if total is None or total['n'] < 2:
            if verbose: print(f'{product}: no overlapping valid pixels')
            continue
        stats = stats_from_sums(total)
        cells = pooled_cells(summaries, product, quantity, basis)
        (blo, bhi), n_cells = bootstrap_by_cell(cells, 'bias', n_boot)
        (rlo, rhi), _       = bootstrap_by_cell(cells, 'rmse', n_boot)
        cstats = cell_statistics(cells)
        if cstats: stats.update(cstats)
        istats = image_mean_statistics(summaries, product, quantity, basis)
        if istats: stats.update(istats)
        stats.update({'product': TITLES[product][0], 'grid': TITLES[product][1], 'key': product,
                      'n_cells': n_cells, 'n_scenes': n_scenes, 'basis': basis,
                      'bias_ci_lo': blo, 'bias_ci_hi': bhi, 'rmse_ci_lo': rlo, 'rmse_ci_hi': rhi})
        rows.append(stats)
        if verbose:
            #print(f'  {stats["product"]:24s} pixel: bias {stats["bias"]:+.3f} RMSE {stats["rmse"]:.3f} '
            #      f'slope {stats["slope"]:.2f} | cell: bias {stats.get("bias_cell", float("nan")):+.3f} '
            #      f'RMSE {stats.get("rmse_cell", float("nan")):.3f} slope {stats.get("slope_cell", float("nan")):.2f} '
            #      f'| {stats["n_pixels"]:,} px / {n_cells:,} cells / {n_scenes} scenes')
            print(
                f'  {stats["product"]:24s} pixel: bias {stats["bias"]:+.3f} RMSE {stats["rmse"]:.3f} '
                f'r {stats["pearson_r"]:.3f} slope {stats["slope"]:.2f} | '
                f'cell: bias {stats.get("bias_cell", float("nan")):+.3f} '
                f'RMSE {stats.get("rmse_cell", float("nan")):.3f} '
                f'r {stats.get("pearson_r_cell", float("nan")):.3f} '
                f'slope {stats.get("slope_cell", float("nan")):.2f} | '
                f'{stats["n_pixels"]:,} px / {n_cells:,} cells / {n_scenes} scenes'
            )
    return rows


def inter_reference_table(summaries, valids=None):
    '''Pairwise disagreement between reference products, which bounds measurable agreement.'''
    valids = DEFAULT_VALIDS if valids is None else valids
    rows = []
    for i, a in enumerate(valids):
        for b in valids[i+1:]:
            total = None
            for s in summaries:
                key = f'sums|inter|{a}~{b}'
                if key not in s: continue
                S = _unpack_sums(s[key])
                total = S if total is None else add_sums(total, S)
            if total is None or total['n'] < 2: continue
            stats = stats_from_sums(total)
            stats.update({'pair': f'{TITLES[a][0]} vs {TITLES[b][0]}', 'a': a, 'b': b})
            rows.append(stats)
    return rows


def availability_table(summaries, valids=None):
    '''
    Which products were retrieved for which scene, and why the misses happened. Report this
    alongside Table 5: without it the differing n_scenes per product is invisible, and the
    seasonal restrictions on ICESat and CryoSat-2 make the missingness systematic rather
    than incidental.
    '''
    valids = DEFAULT_VALIDS if valids is None else valids
    rows = []
    for s in summaries:
        have = set(s['products'].tolist())
        row = {'scene': str(s['scene']), 'date': str(s['date']),
               'n_products': len(have), 'n_common_pixels': int(s.get('n_common_pixels', 0))}
        row.update({p: ('yes' if p in have else 'no') for p in valids})
        row['misses'] = '; '.join(str(m) for m in s.get('miss_reasons', []))
        rows.append(row)
    counts = {p: sum(r[p] == 'yes' for r in rows) for p in valids}
    print('product availability over', len(rows), 'scenes:')
    for p, c in counts.items(): print(f'  {TITLES[p][0]:24s} {c}/{len(rows)} scenes')
    n_full = sum(r['n_products'] == len(valids) for r in rows)
    print(f'  scenes carrying all {len(valids)} products: {n_full}')
    return rows


def pooled_hist(summaries, product, quantity='h_geometric', kind='product', basis='all'):
    '''Sum the fixed-grid 2D histograms across scenes, for the Figure 10 panels.'''
    key = f'hist|{_tag(product, quantity, basis)}' if kind == 'product' else f'hist|inter|{product}'
    total = None
    for s in summaries:
        if key not in s: continue
        total = s[key].astype(np.float64).copy() if total is None else total + s[key]
    return total
# --- HELD-OUT SIMULATED CORES (Table 4) --- #

def gen_feed(count=100, collector=None, device='cuda'):
    '''gen_feed from the notebooks, unchanged in behaviour.'''
    train = script_CoreExplorer.get_randoms(count, reshape=False, collector=collector)
    rf = train['rf'].reshape([-1] + collector.h5f['rf'].attrs['fshape'].tolist())
    rf, (B, C, P, I) = torch.as_tensor(rf).transpose(-1, -2), rf.shape
    rf_c = rf[..., None] * rf.conj()[..., None, :]
    rf_c = rf_c / torch.diagonal(rf_c, 0, -2, -1).sum(dim=-1)[..., None, None]
    rf_c = rf_c.reshape(B, C, I*P*P)
    parts = [train['S'].real, train['T'].real, train['rho'].real, rf_c.real, rf_c.imag]
    feed = torch.cat([torch.as_tensor(el).flatten(1) for el in parts], dim=1)[:, None, :]
    return feed.to(device, torch.float32), (C, I, P, P)


def conditioning_mask(shape, vl, n_angles=6, angle=0, mode='single_angle', device='cuda'):
    '''
    Boolean mask, True where an entry is observed and clamped. Feature layout is
    [S | T | rho | C4 real | C4 imag]; angle a occupies entries [16a, 16a+16) of each block.

    'single_angle' reproduces training, which masks in one angle's sixteen entries, real
    and imaginary. 'notebook' reproduces Inference_CoreExplorer cell 6, which does
    test[..., :-96] = 0 and so keeps only the imaginary half across all six angles -
    a different conditioning set, probably unintended. 'all_angles' is the upper bound.
    '''
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    n_cov, D = vl[-1], shape[-1]
    per_angle = n_cov // n_angles
    if mode == 'single_angle':
        mask[..., D - 2*n_cov + angle*per_angle : D - 2*n_cov + (angle+1)*per_angle] = True
        mask[..., D -   n_cov + angle*per_angle : D -   n_cov + (angle+1)*per_angle] = True
    elif mode == 'all_angles': mask[..., -2*n_cov:] = True
    elif mode == 'notebook'  : mask[..., -n_cov:] = True
    else: raise ValueError(f'invalid mode {mode}')
    return mask


def ridge_fit(X, Y, lambdas=None):
    '''
    Closed-form multi-output ridge with a GCV-selected penalty; the non-generative
    baseline Table 4 needs. Returns a predict function carrying .lam.
    '''
    lambdas = np.logspace(-6, 3, 40) if lambdas is None else lambdas
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    xm, ym, xs = X.mean(0), Y.mean(0), X.std(0)
    xs[xs < 1e-12] = 1.0
    Xc, Yc = (X - xm) / xs, Y - ym

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY, n, best = U.T @ Yc, Xc.shape[0], (np.inf, None, None)
    for lam in lambdas:
        W = Vt.T @ ((s / (s**2 + lam))[:, None] * UtY)
        dof = float((s**2 / (s**2 + lam)).sum())
        gcv = ((Yc - Xc @ W)**2).sum() / max(n - dof, 1e-6)**2
        if gcv < best[0]: best = (gcv, lam, W)
    _, lam, W = best

    def predict(Xn): return ((np.asarray(Xn, np.float64) - xm) / xs) @ W + ym
    predict.lam = lam
    return predict


def thickness_from_profiles(profiles, vl, icesim, cells_per_m=30, device='cuda'):
    '''(h_equivalent, h_geometric) in metres, by the same rule as the SAR retrieval.'''
    dz = 1.0 / cells_per_m
    P = torch.as_tensor(profiles, dtype=torch.float32, device=device)
    S, T = P[:, :vl[0]], P[:, vl[0]:vl[0]+vl[1]]
    with torch.no_grad(): frac = icesim.get_icefrac(T, S)
    keep = ((frac == 0).cumsum(dim=-1) == 0)
    return ((frac * keep).sum(-1) * dz).cpu().numpy(), (keep.sum(-1) * dz).to(torch.float32).cpu().numpy()


def profile_errors(pred, truth, vl):
    '''Per-variable bias and RMSE in physical units, plus RMSE against depth.'''
    out = {}
    for name, a, b in [('salinity_ppt', 0, vl[0]),
                       ('temperature_C', vl[0], vl[0]+vl[1]),
                       ('density_gcm3', vl[0]+vl[1], sum(vl[:3]))]:
        d = pred[:, a:b] - truth[:, a:b]
        out[f'{name}_bias'] = float(d.mean())
        out[f'{name}_rmse'] = float(np.sqrt((d**2).mean()))
        out[f'{name}_rmse_by_depth'] = np.sqrt((d**2).mean(axis=0))
    return out


def evaluate_heldout(model, vl, floqs, corpus_fn='em_resice.hdf5', count=100,
                     angle=0, mode='single_angle', n_draws=1, icesim=None, device='cuda'):
    '''
    Table 4. Returns (rows, arrays) where rows holds one dict per model and arrays holds
    the truth and predicted profiles for plotting. The 80/20 split is by core, matching
    the notebook, so no incidence angle leaks across it.
    '''
    if icesim is None: icesim = brack2bing.LatticeSolver.load()
    collector = util_collector.Dataset(corpus_fn)
    feed, _ = gen_feed(min(count, int(collector.h5f.attrs['w_ind'])), collector, device)
    collector.close()

    feed = model.whitener(feed)
    n_valid, n_profile = feed.shape[0] // 5, sum(vl[:3])
    train_raw = model.unwhitener(feed[:-n_valid]).detach()[:, 0, :].cpu().numpy()
    valid_raw = model.unwhitener(feed[-n_valid:]).detach()[:, 0, :].cpu().numpy()
    print(f'{feed.shape[0]} cores, {feed.shape[0]-n_valid} train / {n_valid} held out; conditioning: {mode}'
          + (f', angle {angle} ({floqs[angle]:.2f} deg)' if mode == 'single_angle' else ''))

    mask = conditioning_mask(feed[-n_valid:].shape, vl, len(floqs), angle, mode, device)
    x = feed[-n_valid:].clone()
    x[~mask] = 0.0
    with torch.no_grad():
        gen = torch.stack([model.unwhitener(model(x.clone(), mask))[:, 0, :] for _ in range(n_draws)])
    gen_med = gen.median(dim=0).values.cpu().numpy()

    cols = mask[0, 0].detach().cpu().numpy()
    predict = ridge_fit(train_raw[:, cols], train_raw[:, :n_profile])
    print(f'ridge penalty selected by GCV: {predict.lam:.3g}')

    truth, rows = valid_raw[:, :n_profile], []
    h_eq_t, h_ge_t = thickness_from_profiles(truth, vl, icesim, device=device)
    for name, pred in [('generative', gen_med[:, :n_profile]), ('ridge', predict(valid_raw[:, cols]))]:
        stats = profile_errors(pred, truth, vl)
        h_eq_p, h_ge_p = thickness_from_profiles(pred, vl, icesim, device=device)
        stats.update({'model': name, 'n_held_out': int(truth.shape[0]),
                      'thickness_equivalent_bias': float((h_eq_p - h_eq_t).mean()),
                      'thickness_equivalent_rmse': float(np.sqrt(((h_eq_p - h_eq_t)**2).mean())),
                      'thickness_geometric_bias': float((h_ge_p - h_ge_t).mean()),
                      'thickness_geometric_rmse': float(np.sqrt(((h_ge_p - h_ge_t)**2).mean()))})
        rows.append(stats)
        print(f'  {name:11s} thickness bias {stats["thickness_geometric_bias"]:+.3f} m, '
              f'RMSE {stats["thickness_geometric_rmse"]:.3f} m')

    arrays = {'truth': truth, 'generative': gen_med[:, :n_profile],
              'ridge': predict(valid_raw[:, cols]),
              'generative_std': gen.std(dim=0).cpu().numpy()[:, :n_profile] if n_draws > 1 else None}
    return rows, arrays


# --- FIGURES --- #

def paper_style(font=8):
    import matplotlib
    matplotlib.rcParams.update({'font.size': font, 'axes.titlesize': font, 'axes.labelsize': font,
                                'xtick.labelsize': font-1, 'ytick.labelsize': font-1,
                                'legend.fontsize': font-1, 'axes.linewidth': 0.6,
                                'lines.linewidth': 0.9, 'savefig.dpi': 400,
                                'pdf.fonttype': 42, 'figure.constrained_layout.h_pad': 0.06,
                                'figure.constrained_layout.w_pad': 0.06,
                                'figure.constrained_layout.hspace': 0.06,
                                'figure.constrained_layout.wspace': 0.04})


def plot_scene_figures(cache_dir, fields, sr=1, display_stride=1, aspect=4, valids=None,
                       icesim=None, im_figsize_base=(16,4)):
    '''
    Write every per-image figure into the scene's own cache folder, and return their paths.
    Called from run_scene while the fields are still in memory, so nothing needs reloading.

      SIT_comparison.png : retrieval, VH log-intensity, each reference
      SIT_histograms.png : unsmoothed and smoothed thickness distributions, and the
                           hand-drawn region split (qualitative, see region_mask)
      SIT_scatter.png    : this scene against each available reference

    display_stride=1 draws every pixel, so the panel axes carry the true SAR image extent.
    Raise it only for a quick look; the statistics are always computed on the full field
    regardless.
    '''
    import matplotlib.pyplot as plt
    valids = DEFAULT_VALIDS if valids is None else valids
    d = display_stride
    paths = []

    refs = {k[4:]: v for k, v in fields.items() if str(k).startswith('ref_') and k[4:] in TITLES}
    has_vh = 'vh_intensity' in fields

    # -- maps
    n = (2 if has_vh else 1) + len(refs)
    fig = plt.figure(figsize=(im_figsize_base[0], im_figsize_base[1]*n), layout='constrained', dpi=300)
    if has_vh:
        plt.subplot(n, 1, 1); plt.imshow(np.log1p(fields['vh_intensity'][::d, ::d]).T, aspect=aspect, cmap='gray')
        plt.title('VH log-intensity');# plt.colorbar()
    plt.subplot(n, 1, (2 if has_vh else 1)); plt.imshow(fields['h_geometric'][::d, ::d].T, aspect=aspect, cmap='plasma')
    plt.title('Retrieved ice column thickness (m)'); plt.colorbar()
    for i, (name, dat) in enumerate(refs.items()):
        plt.subplot(n, 1, (3 if has_vh else 2)+i); plt.imshow(np.asarray(dat)[::d, ::d].T, aspect=aspect, cmap='plasma')
        plt.title(f'{TITLES[name][0]} {TITLES[name][1]}'); plt.colorbar()
    p = os.path.join(cache_dir, 'SIT_comparison.png'); plt.savefig(p); plt.close(fig); paths.append(p)

    # -- distributions
    mask = region_mask(cache_dir, fields['h_geometric'].shape, sr=sr)
    has_raw = 'h_geometric_raw' in fields
    ncol = 3 if has_raw else 2
    fig = plt.figure(figsize=(im_figsize_base[0]*ncol, im_figsize_base[1]), layout='constrained', dpi=300)
    if has_raw:
        plt.subplot(1, ncol, 1)
        raw = fields['h_geometric_raw']; plt.hist(raw[np.isfinite(raw)], bins=60); plt.yscale('log')
        plt.title('unsmoothed, per pixel'); plt.xlabel('Sea ice thickness (m)'); plt.ylabel('Pixels')
    plt.subplot(1, ncol, ncol - 1)
    sm = fields['h_geometric']; plt.hist(sm[np.isfinite(sm)], bins=60); plt.yscale('log')
    box = tuple(np.atleast_1d(fields.get('post_boxcar', ())).tolist())
    plt.title(f'smoothed, {box} boxcar' if box else 'smoothed')
    plt.xlabel('Sea ice thickness (m)')
    if not has_raw: plt.ylabel('Pixels')
    plt.subplot(1, ncol, ncol)
    if mask is not None:
        groups = [sm[(mask == v) & np.isfinite(sm)] for v in np.unique(mask)]
        plt.hist(groups, bins=40); plt.yscale('log')
        plt.legend([f'region {i}' for i in range(len(groups))])
        plt.title('hand-drawn region split (qualitative)')
    else:
        plt.text(0.5, 0.5, 'no region mask', ha='center'); plt.title('region split')
    plt.xlabel('Sea ice thickness (m)')
    p = os.path.join(cache_dir, 'SIT_histograms.png'); plt.savefig(p); plt.close(fig); paths.append(p)

    # -- per-scene scatter against each reference
    if refs:
        fig, axes = plt.subplots(1, len(refs), figsize=(im_figsize_base[0]*len(refs), im_figsize_base[1]), squeeze=False,
                                 layout='constrained', dpi=300)
        for ax, (name, dat) in zip(axes[0], refs.items()):
            ref = np.asarray(dat, np.float64)
            st = stats_from_sums(pair_sums(ref, fields['h_geometric']))
            _cell_panel(ax, cell_table(ref, fields['h_geometric']), st, TITLES[name][0], TITLES[name][1])
            ax.set_xlabel('Reference thickness (m)')
        axes[0][0].set_ylabel('Retrieved thickness (m)')
        p = os.path.join(cache_dir, 'SIT_scatter.png'); plt.savefig(p); plt.close(fig); paths.append(p)

    p = plot_region_cores(cache_dir, fields, icesim=icesim)
    if p is not None: paths.append(p)

    return paths


def region_lattice(profile, icesim, hshape=(50, 20), num_channels=1):
    '''
    Realize the lattice for one stored (S, T, rho) profile, borders removed.

    gen_lattice returns (N, C, L+2, W+2, D+2) with flagged pads on every axis: -999 and
    -998 on the length axis, -997 on the width and depth axes. These are boundary markers,
    not medium, so they are trimmed here exactly as get_dielectric does. The returned array
    is (L, W, D) with the sign convention of the solver: positive is brine salinity,
    negative is air, zero is ice.

    The stored profiles are read only; nothing is written back.
    '''
    '''
    Realize the lattice for one stored (S, T, rho) profile, borders removed.

    gen_lattice returns (N, C, L+2, W+2, D+2) with flagged pads on every axis: -999 and
    -998 on the length axis, -997 on the width and depth axes. These are boundary markers,
    not medium, so they are trimmed here exactly as get_dielectric does. The returned array
    is (L, W, D) with the sign convention of the solver: positive is brine salinity,
    negative is air, zero is ice.

    The stored profiles are read only; nothing is written back.
    '''
    import io, contextlib, torch
    dev = getattr(icesim, 'device', 'cuda')
    dt = getattr(icesim, 'dtype', torch.float32)
    S, T, rho = [torch.as_tensor(np.asarray(profile[k], dtype=np.float64))[None, :].to(dev, dt)
                 for k in range(3)]                      # stored order is [S | T | rho]
    with contextlib.redirect_stdout(io.StringIO()):      # gen_lattice prints a shape line
        lat = icesim.gen_lattice(T, S, rho, hshape=hshape, num_channels=num_channels)
    return lat[0, 0, 1:-1, 1:-1, 1:-1].detach().cpu().numpy()


def cross_section(lat):
    '''
    Slice the lattice along its longest horizontal axis, at the midpoint of the other.
    Returns (section, axis_name) with section shaped (horizontal, depth).
    '''
    L, W, D = lat.shape
    if L >= W: return lat[:, W // 2, :], 'length'
    return lat[L // 2, :, :], 'width'


def plot_cross_section(ax, lat, cells_per_m=30, mode='class'):
    '''Draw a lattice cross-section, depth downward, axes in metres.'''
    from matplotlib.colors import ListedColormap, BoundaryNorm
    sec, axis_name = cross_section(lat)
    extent = (0, sec.shape[0]/cells_per_m, sec.shape[1]/cells_per_m, 0)
    if mode == 'class':
        cls = np.select([sec < 0, sec > 0], [0, 2], default=1)   # air, ice, brine
        cmap = ListedColormap(['#dfe6ee', '#f7f7f7', '#2c5f8a'])
        im = ax.imshow(cls.T, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                       extent=extent, aspect='auto', interpolation='nearest')
    else:
        im = ax.imshow(sec.T, cmap='viridis', extent=extent, aspect='auto', interpolation='nearest')
    ax.set_xlabel(f'{axis_name} (m)'); ax.set_ylabel('depth (m)')
    return im, axis_name


def plot_region_cross_sections(cache_dir, fields=None, icesim=None, hshape=(50, 20),
                               cells_per_m=30, mode='class', fn='SIT_region_lattice.png'):
    '''
    Regenerate the lattice from each cached region profile and draw its cross-section along
    the longest horizontal axis. Reads region_profiles from the fields cache; requires a
    LatticeSolver because gen_lattice runs on the GPU.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if fields is None: fields = load_fields(cache_dir)
    if fields is None: print('no cached fields'); return None
    prof = np.asarray(fields.get('region_profiles', np.empty((0, 3, 0))))
    if prof.size == 0: print('no region profiles cached'); return None
    if icesim is None: icesim = brack2bing.LatticeSolver.load()

    fig, axes = plt.subplots(len(prof), 1, figsize=(7.09, 2.6*len(prof)), squeeze=False,
                             layout='constrained')
    for r in range(len(prof)):
        lat = region_lattice(prof[r], icesim, hshape=hshape)
        _, axis_name = plot_cross_section(axes[r][0], lat, cells_per_m, mode)
        axes[r][0].set_title(f'region {r} - lattice cross-section along {axis_name}, '
                             f'{lat.shape[0]}x{lat.shape[1]}x{lat.shape[2]} cells', fontsize=8)
    if mode == 'class':
        axes[0][0].legend(handles=[Patch(fc='#f7f7f7', ec='0.6', label='ice'),
                                   Patch(fc='#2c5f8a', label='brine'),
                                   Patch(fc='#dfe6ee', ec='0.6', label='air')],
                          loc='upper right', fontsize=6.5, frameon=True)
    path = os.path.join(cache_dir, fn)
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    return path


def plot_region_cores(cache_dir, fields, icesim=None, cells_per_m=30, hshape=(50, 20),
                      fn='SIT_region_cores.png'):
    '''
    The region-aggregate retrieved profiles, and the lattice cross-section regenerated from
    each, as in Inference_ALOSRunner cell 6. Written into the scene's own folder.

    The regions come from the hand-drawn mask, so this is interpretability output: it shows
    what the model believes a column looks like over a delineated area, not a property of
    an ice regime.
    '''
    import matplotlib.pyplot as plt
    prof = np.asarray(fields.get('region_profiles', np.empty((0, 3, 0))))
    if prof.size == 0: return None
    counts = np.asarray(fields.get('region_counts', np.zeros(len(prof))))
    depth = np.arange(prof.shape[-1]) / cells_per_m
    labels = ['Salinity (g kg$^{-1}$)', 'Temperature ($^\\circ$C)', 'Density (g cm$^{-3}$)']

    ncol = 4 if icesim is not None else 3
    fig, axes = plt.subplots(len(prof), ncol, figsize=(3.2*ncol, 3.2*len(prof)), squeeze=False,
                             layout='constrained')
    for r in range(len(prof)):
        for c in range(3):
            axes[r][c].plot(prof[r, c], depth)
            axes[r][c].invert_yaxis(); axes[r][c].set_xlabel(labels[c])
            if c == 0: axes[r][c].set_ylabel(f'region {r} (n={counts[r]:,})\nDepth (m)')
        if icesim is not None:
            lat = region_lattice(prof[r], icesim, hshape=hshape)
            plot_cross_section(axes[r][3], lat, cells_per_m)
            axes[r][3].set_title('lattice cross-section', fontsize=8)
    path = os.path.join(cache_dir, fn)
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    return path


def replot_scene(cache_dir, sr=1, with_intensity=False, icesim=None, **kw):
    '''
    Redraw a scene's figures from its cached fields, with no inference and no model.

    The VH intensity panel is omitted by default because the SLC is not cached - it is
    large and recoverable. Pass with_intensity=True to read the band back from the SNAP
    cache for the figure. Pass icesim to also draw the region lattice cross-sections.
    '''
    fields = load_fields(cache_dir)
    if fields is None:
        print(f'{os.path.basename(cache_dir)}: no cached fields'); return []
    fields = dict(fields)
    if with_intensity:
        slc_ds = PyPreprocess_Single.Dataset_SNAPbase(cache_dir=cache_dir, read_nc=True)
        fields['vh_intensity'] = np.abs(np.asarray(slc_ds.slc[2, ::sr, ::sr])).astype(np.float32)
        del slc_ds
    return plot_scene_figures(cache_dir, fields, sr=sr, icesim=icesim, **kw)


def replot_scenes(cache_dirs, **kw):
    '''Redraw every cached scene, one image in memory at a time.'''
    import gc
    out = []
    for d in cache_dirs:
        out.append(replot_scene(d, **kw)); gc.collect()
    return out


def _cell_panel(ax, cells, stats, title, subtitle, min_count=1, max_points=4000,
                show_range=True, color_by_scene=True, legend_max=8, image_stats=None):
    '''
    One point per reference cell: its value against the mean retrieval over it, sized by
    pixel count, with a bar of one within-cell standard deviation.

    A pixel-density scatter is the wrong display here. The reference is nearest-neighbour
    interpolated, so it is piecewise constant and a scene contributes only a handful of
    distinct reference values; plotting millions of pixels against them collapses to a few
    vertical stripes, and a real but modest trend disappears into the within-cell scatter.
    Because the reference is constant inside a cell, the count-weighted fit through these
    cell means is algebraically identical to the pixel-level OLS, so nothing is lost by
    plotting the aggregate instead of the pixels.
    '''
    if cells is None or cells['value'].size == 0:
        ax.text(0.5, 0.5, 'no overlap', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title}\n{subtitle}', fontsize=8); return

    keep = cells['count'] >= min_count
    x, n = cells['value'][keep], cells['count'][keep].astype(np.float64)
    mean_y = cells['sum_y'][keep] / n
    var_y = np.maximum(cells['sum_yy'][keep] / n - mean_y**2, 0.0)
    sd_y = np.sqrt(var_y)

    idx = None
    if x.size > max_points:                       # keep the largest cells if very many
        idx = np.argsort(n)[-max_points:]
        x, n, mean_y, sd_y = x[idx], n[idx], mean_y[idx], sd_y[idx]

    cstats = cell_statistics({k: v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape
                              else v for k, v in cells.items()}, min_count=1)

    top = np.nanmax(mean_y + sd_y)
    if show_range and 'max_y' in cells:
        mx = cells['max_y'][keep]
        top = max(top, np.nanpercentile(mx[idx] if idx is not None else mx, 99))
    hi = float(np.ceil(max(np.nanmax(x), top) * 2) / 2)
    lims = (0.0, max(hi, 0.5))

    if show_range and 'max_y' in cells and 'min_y' in cells:
        hi_c, lo_c = cells['max_y'][keep], cells['min_y'][keep]
        if idx is not None: hi_c, lo_c = hi_c[idx], lo_c[idx]   # x is already subsampled
        ax.vlines(x, lo_c, hi_c, color='0.88', lw=0.5, zorder=0)   # full within-cell range
    ax.errorbar(x, mean_y, yerr=sd_y, fmt='none', ecolor='0.72', elinewidth=0.5, zorder=1)
    sizes = 4 + 30 * (n / n.max())**0.5

    scene_idx = cells.get('scene_index') if color_by_scene else None
    if scene_idx is not None:
        scene_idx = scene_idx[keep]
        if idx is not None: scene_idx = scene_idx[idx]
    if scene_idx is not None and np.unique(scene_idx).size > 1:
        import matplotlib.cm as cm
        names = cells.get('scene_names', np.array([]))
        uniq = np.unique(scene_idx)
        cmap = cm.get_cmap('tab20' if uniq.size > 10 else 'tab10')
        for k, u in enumerate(uniq):
            m_ = scene_idx == u
            label = str(names[u]) if u < len(names) else f'scene {u}'
            ax.scatter(x[m_], mean_y[m_], s=sizes[m_], color=cmap(k % cmap.N),
                       alpha=0.8, linewidths=0, zorder=3, label=label)
        if uniq.size <= legend_max:
            ax.legend(fontsize=5.5, frameon=False, loc='lower right', handletextpad=0.2,
                      borderpad=0.2, labelspacing=0.2, markerscale=0.8)
        else:
            ax.text(0.97, 0.03, f'{uniq.size} scenes, coloured by image',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5, color='0.35')
    else:
        ax.scatter(x, mean_y, s=sizes, c='#1f77b4', alpha=0.7, linewidths=0, zorder=3)
    ax.plot(lims, lims, color='0.2', lw=0.8, ls='--', zorder=2)
    xs = np.array(lims)
    if stats is not None and np.isfinite(stats.get('slope', np.nan)):
        ax.plot(xs, stats['slope']*xs + stats['intercept'], color='crimson', lw=1.0, zorder=4,
                label='pixel fit')
    if cstats is not None and np.isfinite(cstats.get('slope_cell', np.nan)):
        ax.plot(xs, cstats['slope_cell']*xs + cstats['intercept_cell'], color='#1a7f5a',
                lw=1.0, ls=(0, (4, 1.5)), zorder=4, label='cell fit')
    if image_stats is not None and np.isfinite(image_stats.get('slope_image', np.nan)):
        ax.plot(xs, image_stats['slope_image']*xs + image_stats['intercept_image'],
                color='#6a3d9a', lw=1.0, ls=(0, (1.5, 1.5)), zorder=4, label='image-mean fit')

    if stats is not None:
        rows = [('pixel', stats['bias'], stats['rmse'], stats['pearson_r'], stats['slope'])]
        if cstats is not None:
            rows.append(('cell', cstats['bias_cell'], cstats['rmse_cell'],
                         cstats['pearson_r_cell'], cstats['slope_cell']))
        if image_stats is not None:
            rows.append(('image', image_stats['bias_image'], image_stats['rmse_image'],
                         image_stats['pearson_r_image'], image_stats['slope_image']))
        txt = '       bias  RMSE     r slope'
        for name, b, rm, r_, sl in rows:
            txt += f'\n{name:<6}{b:+5.2f} {rm:5.2f} {r_:5.2f} {sl:5.2f}'
        txt += f'\nn = {x.size:,} cells'
        if image_stats is not None: txt += f', {image_stats["n_images"]} images'
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=5.2,
                linespacing=1.25, family='DejaVu Sans Mono',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', lw=0.5, alpha=0.9))
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_title(f'{title}\n{subtitle}', fontsize=8)


def _density_panel(ax, H, stats, title, subtitle, bins=None):
    '''Pixel-density view, kept for scenes where the reference resolves many cells.'''
    from matplotlib.colors import LogNorm
    edges = np.linspace(*HIST_RANGE, HIST_BINS + 1) if bins is None else bins
    centres = 0.5*(edges[:-1] + edges[1:])
    occ_x, occ_y = H.sum(axis=1) > 0, H.sum(axis=0) > 0
    hi = max(centres[occ_x].max() if occ_x.any() else 1.0, centres[occ_y].max() if occ_y.any() else 1.0)
    lims = (0.0, float(np.ceil(hi*2)/2))
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_where(H.T == 0, H.T),
                         norm=LogNorm(), cmap='viridis', shading='auto')
    ax.plot(lims, lims, color='0.2', lw=0.8, ls='--')
    if stats is not None and np.isfinite(stats.get('slope', np.nan)):
        xs = np.array(lims)
        ax.plot(xs, stats['slope']*xs + stats['intercept'], color='crimson', lw=1.0)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_title(f'{title}\n{subtitle}', fontsize=8)
    return mesh


def plot_comparison(summaries, quantity='h_geometric', valids=None, basis='all',
                    require_all=False, style='cells', figsize=(7.09, None), min_count=1):
    '''
    Figure 10, drawn from the summaries - no pixel data is loaded.

    style='cells' (default) plots one point per reference cell: the cell's value against the
    mean retrieval over it, sized by pixel count, with a within-cell spread bar. This is the
    display that shows whether a trend exists, because the reference is piecewise constant
    and a pixel scatter degenerates into vertical stripes.
    style='density' gives the pooled pixel histogram instead, which is honest about how the
    pixels are distributed but hides modest trends.

    The fitted line is the pixel-level OLS, exact from the accumulated sums, so it is the
    same line under either style.
    '''
    import matplotlib.pyplot as plt
    valids = DEFAULT_VALIDS if valids is None else valids
    paper_style()
    summaries = select_scenes(summaries, valids, require_all)
    if not summaries:
        print('no scene carries the full product set'); return None

    present = []
    for p in valids:
        total, n_scenes = pooled_sums(summaries, p, quantity, basis)
        if total is None or total['n'] < 2: continue
        present.append((p, stats_from_sums(total), n_scenes))
    if not present:
        print('no product overlaps the retrieval'); return None

    inter = inter_reference_table(summaries, valids)
    n_panels = len(present) + (1 if inter else 0)
    ncol = min(3, n_panels); nrow = int(np.ceil(n_panels/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] or 3.0*nrow),
                             layout='constrained', dpi=300)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for k, (product, st, n_scenes) in enumerate(present):
        sub = f'{TITLES[product][1]}, {n_scenes} scenes'
        if style == 'cells':
            _cell_panel(axes[k], pooled_cells(summaries, product, quantity, basis), st,
                        TITLES[product][0], sub, min_count=min_count,
                        image_stats=image_mean_statistics(summaries, product, quantity, basis))
        else:
            H = pooled_hist(summaries, product, quantity, 'product', basis)
            mesh = _density_panel(axes[k], H, st, TITLES[product][0], sub)
        axes[k].set_xlabel('Reference thickness (m)')
        axes[k].set_ylabel('Retrieved thickness (m)' if k % ncol == 0 else '')

    if inter:
        best = max(inter, key=lambda r: r['n_pixels'])
        H = pooled_hist(summaries, f'{best["a"]}~{best["b"]}', kind='inter')
        k = len(present)
        if H is not None:
            _density_panel(axes[k], H, best, 'Inter-reference disagreement',
                           f'{TITLES[best["a"]][0]} vs {TITLES[best["b"]][0]}')
            axes[k].set_xlabel('Reference thickness (m)')
            axes[k].set_ylabel('Reference thickness (m)' if k % ncol == 0 else '')

    for ax in axes[n_panels:]: ax.set_visible(False)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.02).set_label('Pixel count')
    return fig


def plot_thickness_distribution(summaries, figsize=(7.09, 2.4)):
    '''
    Pooled distribution of the unsmoothed per-pixel retrieval across all cached scenes,
    from the 1D histograms in the summaries. This is where bimodality in the retrieval
    shows, and it is the evidence for whether the post-model boxcar is averaging across
    two genuine modes or across a collapse.
    '''
    import matplotlib.pyplot as plt
    paper_style()
    total, bins = None, None
    for s in summaries:
        if 'thickness_hist' not in s: continue
        total = s['thickness_hist'].astype(np.float64).copy() if total is None else total + s['thickness_hist']
        bins = s['thickness_bins']
    if total is None: return None
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.stairs(total, bins, fill=True)
    ax.set_yscale('log'); ax.set_xlabel('Unsmoothed retrieved thickness (m)'); ax.set_ylabel('Pixels')
    return fig


def plot_depth_rmse(rows, vl, cells_per_m=30, figsize=(7.09, 2.2)):
    '''RMSE against depth per variable and model; the companion to Figure 7.'''
    import matplotlib.pyplot as plt
    paper_style()
    depth = np.arange(vl[0]) / cells_per_m
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    for ax, (key, label) in zip(axes, [('salinity_ppt', 'Salinity (g kg$^{-1}$)'),
                                       ('temperature_C', 'Temperature ($^\\circ$C)'),
                                       ('density_gcm3', 'Density (g cm$^{-3}$)')]):
        for r in rows: ax.plot(r[f'{key}_rmse_by_depth'], depth, label=r['model'])
        ax.invert_yaxis(); ax.set_xlabel(f'RMSE, {label}')
    axes[0].set_ylabel('Depth (m)'); axes[0].legend(frameon=False)
    return fig


# --- TABLE OUTPUT --- #

def write_csv(rows, fn):
    if not rows: return None
    flat = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
    with open(fn, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()), extrasaction='ignore')
        w.writeheader(); w.writerows(flat)
    return fn


def write_latex_comparison(rows, fn='table5_reference_comparison.tex', quantity='geometric'):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Retrieved sea ice thickness against reference products over the Beaufort Sea '
             r'ALOS-1 PALSAR scenes. Comparison is on the SAR grid, with each product nearest-neighbour '
             r'interpolated to it; $n_{\mathrm{pix}}$ is the number of valid pixel pairs and '
             r'$n_{\mathrm{cell}}$ the number of distinct reference cells contributing. Intervals are '
             r'95\% percentile bootstrap over reference cells. Retrieval is reported as ' + quantity +
             r' thickness.}', r'\label{tab:reference_comparison}',
             r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Product & Grid & $n_{\mathrm{pix}}$ & $n_{\mathrm{cell}}$ & Bias (m) & RMSE (m) & MAE (m) & $r$ \\',
             r'\midrule']
    for r in rows:
        lines.append(f'{r["product"]} & {r["grid"]} & {r["n_pixels"]:,} & {r["n_cells"]:,} & '
                     f'{r["bias"]:+.3f} [{r["bias_ci_lo"]:+.3f}, {r["bias_ci_hi"]:+.3f}] & '
                     f'{r["rmse"]:.3f} & {r["mae"]:.3f} & {r["pearson_r"]:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(fn, 'w') as f: f.write('\n'.join(lines) + '\n')
    return fn


def write_latex_heldout(rows, fn='table4_simulated_retrieval.tex'):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Retrieval accuracy on the held-out simulated cores, in physical units. '
             r'The ridge baseline maps the same conditioning entries directly to the profile. '
             r'Thickness is geometric thickness, derived from each profile by the rule of Section 2.6.}',
             r'\label{tab:simulated_retrieval}', r'\begin{tabular}{lcccccccc}', r'\toprule',
             r'& \multicolumn{2}{c}{Salinity (g kg$^{-1}$)} & \multicolumn{2}{c}{Temperature ($^\circ$C)} '
             r'& \multicolumn{2}{c}{Density (g cm$^{-3}$)} & \multicolumn{2}{c}{Thickness (m)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'Model & Bias & RMSE & Bias & RMSE & Bias & RMSE & Bias & RMSE \\', r'\midrule']
    for r in rows:
        lines.append(f'{r["model"]} & {r["salinity_ppt_bias"]:+.2f} & {r["salinity_ppt_rmse"]:.2f} & '
                     f'{r["temperature_C_bias"]:+.2f} & {r["temperature_C_rmse"]:.2f} & '
                     f'{r["density_gcm3_bias"]:+.3f} & {r["density_gcm3_rmse"]:.3f} & '
                     f'{r["thickness_geometric_bias"]:+.3f} & {r["thickness_geometric_rmse"]:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(fn, 'w') as f: f.write('\n'.join(lines) + '\n')
    return fn


def write_latex_heldout(rows, fn='table4_simulated_retrieval.tex'):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Retrieval accuracy on the held-out simulated cores, in physical units. '
             r'The ridge baseline maps the same conditioning entries directly to the profile. '
             r'Thickness is geometric thickness, derived from each profile by the rule of Section 2.6.}',
             r'\label{tab:simulated_retrieval}', r'\begin{tabular}{lcccccccc}', r'\toprule',
             r'& \multicolumn{2}{c}{Salinity (g kg$^{-1}$)} & \multicolumn{2}{c}{Temperature ($^\circ$C)} '
             r'& \multicolumn{2}{c}{Density (g cm$^{-3}$)} & \multicolumn{2}{c}{Thickness (m)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'Model & Bias & RMSE & Bias & RMSE & Bias & RMSE & Bias & RMSE \\', r'\midrule']
    for r in rows:
        lines.append(f'{r["model"]} & {r["salinity_ppt_bias"]:+.2f} & {r["salinity_ppt_rmse"]:.2f} & '
                     f'{r["temperature_C_bias"]:+.2f} & {r["temperature_C_rmse"]:.2f} & '
                     f'{r["density_gcm3_bias"]:+.3f} & {r["density_gcm3_rmse"]:.3f} & '
                     f'{r["thickness_geometric_bias"]:+.3f} & {r["thickness_geometric_rmse"]:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(fn, 'w') as f: f.write('\n'.join(lines) + '\n')
    return fn