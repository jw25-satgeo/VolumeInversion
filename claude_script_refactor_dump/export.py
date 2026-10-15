"""Output serialization only. No analysis or data loading occurs here."""
import csv
import numpy as np


def write_csv(rows, filename):
    if not rows:
        return None
    flat = [{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in rows]
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(flat)
    return filename


def latex_comparison(rows, quantity='geometric'):
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Product & Grid & $n_{\mathrm{pix}}$ & $n_{\mathrm{cell}}$ & Bias (m) & RMSE (m) & MAE (m) & $r$ \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f'{row["product"]} & {row["grid"]} & {row["n_pixels"]:,} & {row["n_cells"]:,} & '
            f'{row["bias"]:+.3f} [{row["bias_ci_lo"]:+.3f}, {row["bias_ci_hi"]:+.3f}] & '
            f'{row["rmse"]:.3f} & {row["mae"]:.3f} & {row["pearson_r"]:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def latex_heldout(rows):
    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\begin{tabular}{lcccccccc}', r'\toprule',
        r'Model & Bias & RMSE & Bias & RMSE & Bias & RMSE & Bias & RMSE \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f'{row["model"]} & {row["salinity_ppt_bias"]:+.2f} & {row["salinity_ppt_rmse"]:.2f} & '
            f'{row["temperature_C_bias"]:+.2f} & {row["temperature_C_rmse"]:.2f} & '
            f'{row["density_gcm3_bias"]:+.3f} & {row["density_gcm3_rmse"]:.3f} & '
            f'{row["thickness_geometric_bias"]:+.3f} & {row["thickness_geometric_rmse"]:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'
