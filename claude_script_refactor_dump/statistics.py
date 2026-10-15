"""Interface-agnostic numerical reductions and statistics.

No file I/O, model objects, SAR classes, or project-specific data structures live here.
"""
import numpy as np


def pair_sums(reference, retrieval):
    """Return additive sufficient statistics for paired observations."""
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x = np.asarray(reference)[m].astype(np.float64)
    y = np.asarray(retrieval)[m].astype(np.float64)
    d = y - x
    return {
        'n': float(x.size), 'sx': x.sum(), 'sy': y.sum(),
        'sxx': (x * x).sum(), 'syy': (y * y).sum(),
        'sxy': (x * y).sum(), 'sad': np.abs(d).sum(),
    }


def add_sums(a, b):
    """Add two compatible sufficient-statistic dictionaries."""
    return {k: a[k] + b[k] for k in a}


def stats_from_sums(s):
    """Recover bias, RMSE, MAE, error SD, correlation and OLS fit from pair sums."""
    n = s['n']
    if n < 2:
        return None
    sd = s['sy'] - s['sx']
    sdd = s['syy'] - 2 * s['sxy'] + s['sxx']
    out = {
        'n_pixels': int(n),
        'bias': sd / n,
        'rmse': float(np.sqrt(max(sdd / n, 0.0))),
        'mae': s['sad'] / n,
        'std_error': float(np.sqrt(max((sdd - sd * sd / n) / (n - 1), 0.0))),
        'ref_mean': s['sx'] / n,
        'ret_mean': s['sy'] / n,
    }
    vx = n * s['sxx'] - s['sx'] ** 2
    vy = n * s['syy'] - s['sy'] ** 2
    if vx > 1e-9 and vy > 1e-9:
        cov = n * s['sxy'] - s['sx'] * s['sy']
        out['pearson_r'] = cov / np.sqrt(vx * vy)
        out['slope'] = cov / vx
        out['intercept'] = (s['sy'] - out['slope'] * s['sx']) / n
    else:
        out['pearson_r'] = out['slope'] = out['intercept'] = np.nan
    return out


def cell_table(reference, retrieval, decimals=6, tail_thresholds=(0.5, 1.0, 2.0)):
    """Reduce a piecewise-constant reference field to one record per reference cell."""
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x = np.asarray(reference)[m].astype(np.float64)
    y = np.asarray(retrieval)[m].astype(np.float64)
    _, inv = np.unique(np.round(x, decimals), return_inverse=True)
    count = np.bincount(inv).astype(np.int64)
    n_cells = count.size
    out = {
        'value': np.bincount(inv, weights=x) / count,
        'count': count,
        'sum_y': np.bincount(inv, weights=y),
        'sum_yy': np.bincount(inv, weights=y * y),
    }
    hi = np.full(n_cells, -np.inf)
    lo = np.full(n_cells, np.inf)
    np.maximum.at(hi, inv, y)
    np.minimum.at(lo, inv, y)
    out['max_y'], out['min_y'] = hi, lo
    for threshold in tail_thresholds:
        out[f'n_above_{threshold}'] = np.bincount(
            inv, weights=(y > threshold).astype(np.float64))
    return out


def cell_statistics(cells, min_count=1):
    """Calculate unweighted statistics over reference-cell means."""
    if cells is None or cells['value'].size == 0:
        return None
    keep = cells['count'] >= min_count
    x = cells['value'][keep]
    y = cells['sum_y'][keep] / cells['count'][keep].astype(np.float64)
    if x.size < 2:
        return None
    d = y - x
    out = {
        'n_cells': int(x.size),
        'bias_cell': float(d.mean()),
        'rmse_cell': float(np.sqrt((d ** 2).mean())),
        'mae_cell': float(np.abs(d).mean()),
    }
    if x.std() > 1e-9 and y.std() > 1e-9:
        out['pearson_r_cell'] = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        out['slope_cell'] = float(slope)
        out['intercept_cell'] = float(intercept)
    else:
        out['pearson_r_cell'] = out['slope_cell'] = out['intercept_cell'] = np.nan
    return out


def band_sums(reference, retrieval, edges):
    """Return additive count, error-sum and squared-error-sum per reference band."""
    m = np.isfinite(reference) & np.isfinite(retrieval)
    x = np.asarray(reference)[m].astype(np.float64)
    d = np.asarray(retrieval)[m].astype(np.float64) - x
    rows = [[], [], []]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        rows[0].append(float(sel.sum()))
        rows[1].append(d[sel].sum())
        rows[2].append((d[sel] ** 2).sum())
    return np.asarray(rows, dtype=np.float64)


def joint_hist(reference, retrieval, bins=100, rng=(0.0, 6.0)):
    """Return a fixed-grid 2-D histogram suitable for additive pooling."""
    m = np.isfinite(reference) & np.isfinite(retrieval)
    h, _, _ = np.histogram2d(
        np.asarray(reference)[m], np.asarray(retrieval)[m],
        bins=bins, range=[rng, rng])
    return h.astype(np.float32)


def bootstrap_by_cell(cells, statistic='bias', n_boot=1000, seed=0):
    """Bootstrap reference cells, preserving the pixel weighting within each cell."""
    if cells is None or cells['value'].size < 3:
        return (np.nan, np.nan), 0
    x = cells['value'].astype(np.float64)
    n = cells['count'].astype(np.float64)
    sy, syy = cells['sum_y'], cells['sum_yy']
    err = sy - n * x
    sq = syy - 2 * x * sy + n * x * x
    n_cells = x.size
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        total = n[pick].sum()
        if statistic == 'bias':
            values[i] = err[pick].sum() / total
        elif statistic == 'rmse':
            values[i] = np.sqrt(max(sq[pick].sum() / total, 0.0))
        else:
            raise ValueError(f'invalid statistic {statistic}')
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))), n_cells


def profile_errors(pred, truth, lengths):
    """Per-variable bias/RMSE and depthwise RMSE for [S|T|rho] profiles."""
    out = {}
    for name, a, b in [
        ('salinity_ppt', 0, lengths[0]),
        ('temperature_C', lengths[0], lengths[0] + lengths[1]),
        ('density_gcm3', lengths[0] + lengths[1], sum(lengths[:3])),
    ]:
        d = pred[:, a:b] - truth[:, a:b]
        out[f'{name}_bias'] = float(d.mean())
        out[f'{name}_rmse'] = float(np.sqrt((d ** 2).mean()))
        out[f'{name}_rmse_by_depth'] = np.sqrt((d ** 2).mean(axis=0))
    return out
