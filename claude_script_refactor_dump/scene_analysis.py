"""Scene-level reductions and aggregate analysis.

This module consumes ordinary in-memory field dictionaries. It does not know where fields came
from, how they were cached, or how the source data are laid out on disk.
"""
import numpy as np
from .statistics import (
    pair_sums, add_sums, stats_from_sums, cell_table, band_sums, joint_hist,
    cell_statistics, bootstrap_by_cell,
)

DEFAULT_VALIDS = ['AMSR_KH', 'CS2', 'SMOS', 'IC1', 'neXtSIM']
BAND_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 6.0)
HIST_BINS = 100
HIST_RANGE = (0.0, 6.0)
CELL_TAIL_THRESHOLDS = (0.5, 1.0, 2.0)


def summarize_fields(fields, scene=None, quantities=('h_geometric',), valids=None,
                     edges=BAND_EDGES):
    """Reduce one in-memory field set to a small, recomputable analysis record.

    Nothing is written and no assumptions are made about the field's provenance.
    """
    valids = DEFAULT_VALIDS if valids is None else list(valids)
    products = [p for p in valids if f'ref_{p}' in fields]
    common = None
    for p in products:
        valid = np.isfinite(np.asarray(fields[f'ref_{p}'], np.float64))
        common = valid if common is None else common & valid

    quantities = [q for q in quantities if q in fields]
    comparisons = {}
    for q in quantities:
        comparisons[q] = {}
        for p in products:
            ref = np.asarray(fields[f'ref_{p}'], np.float64)
            ret = np.asarray(fields[q], np.float64)
            comparisons[q][p] = {}
            for basis, sel in [('all', None), ('common', common)]:
                r, t = (ref, ret) if sel is None else (
                    np.where(sel, ref, np.nan), np.where(sel, ret, np.nan))
                comparisons[q][p][basis] = {
                    'sums': pair_sums(r, t),
                    'bands': band_sums(r, t, edges),
                    'hist': joint_hist(r, t, HIST_BINS, HIST_RANGE),
                    'cells': cell_table(r, t, tail_thresholds=CELL_TAIL_THRESHOLDS),
                }

    inter = {}
    for i, a in enumerate(products):
        for b in products[i + 1:]:
            xa = np.asarray(fields[f'ref_{a}'], np.float64)
            xb = np.asarray(fields[f'ref_{b}'], np.float64)
            inter[(a, b)] = {
                'sums': pair_sums(xb, xa),
                'hist': joint_hist(xb, xa, HIST_BINS, HIST_RANGE),
            }

    raw = np.asarray(fields.get('h_geometric_raw', fields.get('h_geometric')), np.float64)
    finite = raw[np.isfinite(raw)]
    return {
        'scene': scene,
        'date': str(fields.get('date', '')),
        'products': products,
        'quantities': quantities,
        'n_common_pixels': int(common.sum()) if common is not None else 0,
        'missing_products': [p for p in valids if p not in products],
        'comparisons': comparisons,
        'inter_reference': inter,
        'thickness_hist': np.histogram(finite, bins=HIST_BINS, range=HIST_RANGE)[0].astype(np.int64),
        'thickness_bins': np.linspace(*HIST_RANGE, HIST_BINS + 1),
    }


def select_scenes(records, valids=None, require_all=False):
    valids = DEFAULT_VALIDS if valids is None else valids
    if not require_all:
        return list(records)
    return [r for r in records if set(valids).issubset(r['products'])]


def pooled_sums(records, product, quantity='h_geometric', basis='all'):
    total = None
    n_scenes = 0
    for record in records:
        try:
            current = record['comparisons'][quantity][product][basis]['sums']
        except KeyError:
            continue
        total = current if total is None else add_sums(total, current)
        n_scenes += 1
    return total, n_scenes


def pooled_cells(records, product, quantity='h_geometric', basis='all'):
    parts = []
    scene_index = []
    scene_names = []
    for index, record in enumerate(records):
        try:
            cells = record['comparisons'][quantity][product][basis]['cells']
        except KeyError:
            continue
        if cells is None or cells['value'].size == 0:
            continue
        parts.append(cells)
        scene_index.append(np.full(cells['value'].size, index, dtype=np.int32))
        scene_names.append(str(record.get('scene', index)))
    if not parts:
        return None
    keys = parts[0].keys()
    out = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    out['scene_index'] = np.concatenate(scene_index)
    out['scene_names'] = np.asarray(scene_names)
    return out


def scene_statistics(records, product, quantity='h_geometric', basis='all'):
    out = []
    for record in records:
        try:
            sums = record['comparisons'][quantity][product][basis]['sums']
        except KeyError:
            continue
        stats = stats_from_sums(sums)
        if stats is not None:
            stats['scene'] = record.get('scene')
            out.append(stats)
    return out


def image_mean_statistics(records, product, quantity='h_geometric', basis='all'):
    per = scene_statistics(records, product, quantity, basis)
    if not per:
        return None
    out = {'n_images': len(per)}
    for src, dst in [('bias', 'bias_image'), ('rmse', 'rmse_image'),
                     ('mae', 'mae_image'), ('pearson_r', 'pearson_r_image'),
                     ('slope', 'slope_image'), ('intercept', 'intercept_image')]:
        values = np.asarray([p[src] for p in per], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[dst] = float(values.mean()) if values.size else np.nan
        out[f'{dst}_sd'] = float(values.std(ddof=1)) if values.size > 1 else np.nan
    return out


def stratified_statistics(records, product, quantity='h_geometric', basis='all', min_pixels=100):
    total = None
    edges = None
    for record in records:
        try:
            current = record['comparisons'][quantity][product][basis]['bands']
        except KeyError:
            continue
        total = current.copy() if total is None else total + current
        edges = BAND_EDGES
    if total is None:
        return []
    return [
        {'band': f'{lo:.1f}-{hi:.1f} m', 'n_pixels': int(n),
         'bias': float(sd / n), 'rmse': float(np.sqrt(max(sdd / n, 0.0)))}
        for n, sd, sdd, lo, hi in zip(
            total[0], total[1], total[2], edges[:-1], edges[1:])
        if n >= min_pixels
    ]


def comparison_table(records, titles, quantity='h_geometric', valids=None,
                     basis='all', require_all=False, n_boot=1000):
    """Build comparison rows without reading or writing any external data."""
    valids = list(titles) if valids is None else list(valids)
    records = select_scenes(records, valids, require_all)
    rows = []
    for product in valids:
        total, n_scenes = pooled_sums(records, product, quantity, basis)
        if total is None or total['n'] < 2:
            continue
        stats = stats_from_sums(total)
        cells = pooled_cells(records, product, quantity, basis)
        (blo, bhi), n_cells = bootstrap_by_cell(cells, 'bias', n_boot)
        (rlo, rhi), _ = bootstrap_by_cell(cells, 'rmse', n_boot)
        cstats = cell_statistics(cells)
        if cstats:
            stats.update(cstats)
        istats = image_mean_statistics(records, product, quantity, basis)
        if istats:
            stats.update(istats)
        stats.update({
            'product': titles[product][0], 'grid': titles[product][1], 'key': product,
            'n_cells': n_cells, 'n_scenes': n_scenes, 'basis': basis,
            'bias_ci_lo': blo, 'bias_ci_hi': bhi,
            'rmse_ci_lo': rlo, 'rmse_ci_hi': rhi,
        })
        rows.append(stats)
    return rows


def inter_reference_table(records, titles, valids=None):
    valids = list(titles) if valids is None else list(valids)
    rows = []
    for i, a in enumerate(valids):
        for b in valids[i + 1:]:
            total = None
            for record in records:
                pair = record['inter_reference'].get((a, b))
                if pair is None:
                    continue
                total = pair['sums'] if total is None else add_sums(total, pair['sums'])
            if total is None or total['n'] < 2:
                continue
            stats = stats_from_sums(total)
            stats.update({'pair': f'{titles[a][0]} vs {titles[b][0]}', 'a': a, 'b': b})
            rows.append(stats)
    return rows


def pooled_hist(records, product, quantity='h_geometric', kind='product', basis='all'):
    total = None
    for record in records:
        if kind == 'product':
            try:
                current = record['comparisons'][quantity][product][basis]['hist']
            except KeyError:
                continue
        else:
            a, b = product.split('~', 1)
            try:
                current = record['inter_reference'][(a, b)]['hist']
            except KeyError:
                continue
        total = current.astype(np.float64).copy() if total is None else total + current
    return total
