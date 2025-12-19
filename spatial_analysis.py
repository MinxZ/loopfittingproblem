# -*- coding: utf-8 -*-
"""
Spatial Analysis of Hysteresis Loop Fitting Results

This script maps fitted parameters back to the 50x50 spatial grid,
creating heatmaps that reveal physical structures in the ferroelectric film.

Key Parameters Visualized:
- a2, a3: Coercive voltages (where polarization switching occurs)
- a1: Loop amplitude (related to polarization magnitude)
- a4: Linear slope (indicates leakage/drift)
- RMSE: Fit quality map

Physical Interpretation:
- Uniform regions = single domain areas
- Sharp boundaries = domain walls or grain boundaries
- High RMSE regions = measurement artifacts or complex switching
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.special import erf
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import multiprocessing


# =============================================================================
# Core fitting functions (copied from loopfittingproblem.py for standalone use)
# =============================================================================

def clean_by_large_diff(v, y, diff_thresh=None, k_mad=8.0, expand=1):
    """Flag points around unusually large |dy| jumps, then linearly interpolate."""
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return y.copy()

    dy = np.diff(y)
    ady = np.abs(dy)

    if diff_thresh is None:
        med = np.median(ady)
        mad = np.median(np.abs(ady - med)) + 1e-12
        diff_thresh = med + k_mad * (1.4826 * mad)

    jumps = ady > diff_thresh
    mask = np.zeros(n, dtype=bool)
    idx = np.where(jumps)[0]
    if idx.size > 0:
        mask[idx] = True
        mask[idx + 1] = True
        if expand > 0:
            base = np.where(mask)[0]
            for d in range(1, expand + 1):
                mask[np.clip(base - d, 0, n - 1)] = True
                mask[np.clip(base + d, 0, n - 1)] = True

    y_clean = y.copy()
    good = ~mask
    if good.sum() >= 2 and mask.any():
        x = np.arange(n)
        y_clean[mask] = np.interp(x[mask], x[good], y[good])

    return y_clean


def loop_fit_function(vdc, coef_vec):
    """9 parameter fit function for hysteresis loops."""
    a = coef_vec[:5]
    b = coef_vec[5:]
    d = 1000

    v1 = np.asarray(vdc[:int(len(vdc) / 2)])
    v2 = np.asarray(vdc[int(len(vdc) / 2):])

    g1 = (b[1] - b[0]) / 2 * (erf((v1 - a[2]) * d) + 1) + b[0]
    g2 = (b[3] - b[2]) / 2 * (erf((v2 - a[3]) * d) + 1) + b[2]

    y1 = (g1 * erf((v1 - a[2]) / g1) + b[0]) / (b[0] + b[1])
    y2 = (g2 * erf((v2 - a[3]) / g2) + b[2]) / (b[2] + b[3])

    f1 = a[0] + a[1] * y1 + a[4] * v1
    f2 = a[0] + a[1] * y2 + a[4] * v2

    return np.hstack((f1, f2))


def loop_model_coef(vdc, coef_vec, d=1000.0):
    """Model with adjustable d parameter for continuation."""
    coef_vec = np.asarray(coef_vec, float)
    a0, a1, a2, a3, a4, b0, b1, b2, b3 = coef_vec

    vdc = np.asarray(vdc, float)
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]

    g1 = (b1 - b0) / 2.0 * (erf((v1 - a2) * d) + 1.0) + b0
    g2 = (b3 - b2) / 2.0 * (erf((v2 - a3) * d) + 1.0) + b2

    y1 = (g1 * erf((v1 - a2) / g1) + b0) / (b0 + b1)
    y2 = (g2 * erf((v2 - a3) / g2) + b2) / (b2 + b3)

    f1 = a0 + a1 * y1 + a4 * v1
    f2 = a0 + a1 * y2 + a4 * v2
    return np.hstack([f1, f2])


def residuals_coef(coef_vec, vdc, y_obs, d=1000.0):
    return loop_model_coef(vdc, coef_vec, d=d) - np.asarray(y_obs, float)


def _switch_center_by_max_slope(v, y):
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    dy = np.gradient(y, v)
    return float(v[np.argmax(np.abs(dy))])


def _robust_slope_tail(v, y, frac=0.2):
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    n = len(v)
    k = max(3, int(frac * n))
    order = np.argsort(v)
    idx = np.r_[order[:k], order[-k:]]
    X = np.column_stack([np.ones(len(idx)), v[idx]])
    beta, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
    return float(beta[1])


def _width_10_90(v, y):
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymax - ymin < 1e-12:
        return 2.0
    yn = (y - ymin) / (ymax - ymin)
    order = np.argsort(v)
    v2, yn2 = v[order], yn[order]

    def interp_x(level):
        idx = np.where(yn2 >= level)[0]
        if len(idx) == 0:
            return float(v2[-1])
        i = idx[0]
        if i == 0:
            return float(v2[0])
        x0, x1 = v2[i - 1], v2[i]
        y0, y1 = yn2[i - 1], yn2[i]
        t = 0.0 if abs(y1 - y0) < 1e-12 else (level - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))

    return max(0.2, abs(interp_x(0.90) - interp_x(0.10)))


def make_x0_coef(vdc, y_obs):
    """Generate initial parameter guess from loop shape."""
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]
    y1, y2 = y_obs[:n2], y_obs[n2:]

    a2 = _switch_center_by_max_slope(v1, y1)
    a3 = _switch_center_by_max_slope(v2, y2)
    a4 = _robust_slope_tail(vdc, y_obs, frac=0.2)

    a0 = float(np.median(y_obs))
    a1 = float(np.max(y_obs) - np.min(y_obs))
    if abs(a1) < 1e-12:
        a1 = 1.0

    w1 = _width_10_90(v1, y1)
    w2 = _width_10_90(v2, y2)

    b0 = max(0.1, 0.5 * w1)
    b1 = max(0.2, 1.2 * w1)
    b2 = max(0.1, 0.5 * w2)
    b3 = max(0.2, 1.2 * w2)

    return np.array([a0, a1, a2, a3, a4, b0, b1, b2, b3], float)


def fit_optionA_strong_coef(vdc, y_obs, n_starts=30, d_schedule=(40, 120, 400, 1000), seed=0, b_max=20.0):
    """
    Multi-start optimization with continuation (gradually increasing d).

    This approach helps escape local minima by:
    1. Starting with smooth objective (low d)
    2. Gradually sharpening to match physics (high d)
    3. Trying multiple random initializations
    """
    rng = np.random.default_rng(seed)
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    vmin, vmax = float(np.min(vdc)), float(np.max(vdc))

    lb = np.array([-np.inf, -np.inf, vmin, vmin, -np.inf, 1e-3, 1e-3, 1e-3, 1e-3], float)
    ub = np.array([np.inf, np.inf, vmax, vmax, np.inf, b_max, b_max, b_max, b_max], float)

    base = make_x0_coef(vdc, y_obs)

    def jitter(coef):
        c = coef.copy()
        c[2] += rng.normal(0, 0.8)
        c[3] += rng.normal(0, 0.8)
        for j in [5, 6, 7, 8]:
            c[j] *= np.exp(rng.normal(0, 0.35))
        c[0] += rng.normal(0, 0.2 * np.std(y_obs))
        c[1] *= np.exp(rng.normal(0, 0.2))
        c[4] += rng.normal(0, 0.2 * abs(c[4]) + 1e-6)
        return c

    best = None
    best_cost = np.inf

    for s in range(n_starts):
        x0 = base if s == 0 else jitter(base)
        x0 = np.minimum(np.maximum(x0, lb), ub)

        x = x0
        for d in d_schedule:
            res = least_squares(
                residuals_coef, x,
                args=(vdc, y_obs, d),
                method="trf",
                loss="soft_l1",
                bounds=(lb, ub),
                x_scale="jac",
                max_nfev=4000
            )
            x = res.x

        if res.cost < best_cost:
            best_cost = res.cost
            best = x

    return best


# =============================================================================
# Spatial analysis functions
# =============================================================================


def fit_single_loop(idx, y, vdc_rolled, n_starts):
    """Fit a single loop - helper function for parallel processing."""
    p_best = fit_optionA_strong_coef(
        vdc_rolled, y,
        n_starts=n_starts,
        d_schedule=(30, 80, 200, 600, 1000),
        seed=42 + idx  # Different seed per loop for diversity
    )
    y_pred = loop_fit_function(vdc_rolled, p_best)
    rmse = np.sqrt(np.mean((y_pred - y)**2))

    return {
        'loop_id': idx,
        'rmse': rmse,
        'a0': p_best[0],
        'a1': p_best[1],
        'a2': p_best[2],
        'a3': p_best[3],
        'a4': p_best[4],
        'b0': p_best[5],
        'b1': p_best[6],
        'b2': p_best[7],
        'b3': p_best[8],
    }


def load_or_compute_results(
    loops_flat,
    vdc_rolled,
    result_path="result_spatial.csv",
    n_starts=8,
    force_recompute=False,
    n_jobs=-1
):
    """
    Load cached fitting results or compute them using parallel processing.

    Parameters
    ----------
    loops_flat : ndarray
        Flattened loop data (n_loops, n_points)
    vdc_rolled : ndarray
        Rolled voltage vector
    result_path : str
        Path to cache results
    n_starts : int
        Number of random starts for fitting
    force_recompute : bool
        If True, recompute even if cache exists
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)

    Returns
    -------
    DataFrame with columns: loop_id, rmse, a0-a4, b0-b3
    """
    result_path = Path(result_path)

    if result_path.exists() and not force_recompute:
        print(f"Loading cached results from {result_path}")
        return pd.read_csv(result_path, index_col=0)

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    print(f"Fitting {len(loops_flat)} loops (n_starts={n_starts}, n_jobs={n_jobs})...")

    # Parallel fitting with progress bar
    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_single_loop)(idx, loops_flat[idx], vdc_rolled, n_starts)
        for idx in range(len(loops_flat))
    )

    result = pd.DataFrame(rows)
    result = result.sort_values('loop_id').reset_index(drop=True)
    result.to_csv(result_path)
    print(f"Results saved to {result_path}")
    return result


def reshape_to_grid(values, grid_shape=(50, 50)):
    """Reshape 1D array of loop values to 2D spatial grid."""
    return np.array(values).reshape(grid_shape)


def plot_parameter_map(
    ax, data_2d, title, cmap='viridis',
    center_zero=False, vmin=None, vmax=None,
    scale_um=2.0
):
    """
    Plot a single parameter heatmap.

    Parameters
    ----------
    ax : matplotlib axis
    data_2d : 2D array
        Parameter values on spatial grid
    title : str
    cmap : str
        Colormap name
    center_zero : bool
        If True, center colormap at zero (for signed quantities)
    scale_um : float
        Physical scale in micrometers
    """
    extent = [0, scale_um, 0, scale_um]

    if center_zero:
        abs_max = max(abs(np.nanmin(data_2d)), abs(np.nanmax(data_2d)))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax.imshow(data_2d, cmap=cmap, norm=norm, extent=extent, origin='lower')
    else:
        im = ax.imshow(data_2d, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin='lower')

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def create_spatial_analysis_figure(result_df, grid_shape=(50, 50), save_path=None):
    """
    Create comprehensive spatial analysis figure.

    This figure shows:
    1. Coercive voltage maps (a2, a3) - where switching occurs
    2. Coercive voltage asymmetry - difference between switching voltages
    3. Amplitude map (a1) - strength of ferroelectric response
    4. Linear slope (a4) - leakage current indicator
    5. Fit quality (RMSE) - identifies problematic regions
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spatial Analysis of Ferroelectric Hysteresis Parameters\n(PbTiO₃ thin film, 2μm × 2μm)',
                 fontsize=14, fontweight='bold')

    # Reshape parameters to 2D grid
    a2_map = reshape_to_grid(result_df['a2'], grid_shape)
    a3_map = reshape_to_grid(result_df['a3'], grid_shape)
    a1_map = reshape_to_grid(result_df['a1'], grid_shape)
    a4_map = reshape_to_grid(result_df['a4'], grid_shape)
    rmse_map = reshape_to_grid(result_df['rmse'], grid_shape)

    # Coercive voltage asymmetry
    asymmetry_map = a2_map + a3_map  # Should be ~0 for symmetric loops

    # Plot 1: Coercive voltage a2 (negative branch)
    plot_parameter_map(
        axes[0, 0], a2_map,
        'Coercive Voltage a₂ (V)\n(Negative→Positive switching)',
        cmap='coolwarm', center_zero=True
    )

    # Plot 2: Coercive voltage a3 (positive branch)
    plot_parameter_map(
        axes[0, 1], a3_map,
        'Coercive Voltage a₃ (V)\n(Positive→Negative switching)',
        cmap='coolwarm', center_zero=True
    )

    # Plot 3: Asymmetry (a2 + a3)
    plot_parameter_map(
        axes[0, 2], asymmetry_map,
        'Switching Asymmetry (a₂ + a₃)\n(0 = symmetric loop)',
        cmap='PuOr', center_zero=True
    )

    # Plot 4: Amplitude
    plot_parameter_map(
        axes[1, 0], np.abs(a1_map),
        'Loop Amplitude |a₁|\n(Polarization magnitude)',
        cmap='plasma'
    )

    # Plot 5: Linear slope (leakage)
    plot_parameter_map(
        axes[1, 1], a4_map,
        'Linear Slope a₄\n(Leakage/drift indicator)',
        cmap='RdBu_r', center_zero=True
    )

    # Plot 6: RMSE (fit quality)
    plot_parameter_map(
        axes[1, 2], rmse_map,
        'Fit RMSE\n(Lower = better fit)',
        cmap='hot_r'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def create_domain_map(result_df, grid_shape=(50, 50), save_path=None):
    """
    Create a simplified domain map based on coercive voltage signs.

    This identifies regions with:
    - Normal polarity (majority)
    - Reversed polarity (flipped domains)
    - Asymmetric switching
    """
    a2_map = reshape_to_grid(result_df['a2'], grid_shape)
    a3_map = reshape_to_grid(result_df['a3'], grid_shape)

    # Domain classification based on coercive voltage signs
    # Normal: a2 < 0 and a3 > 0
    # Reversed: a2 > 0 and a3 < 0
    domain_map = np.zeros(grid_shape)
    domain_map[(a2_map < 0) & (a3_map > 0)] = 1   # Normal
    domain_map[(a2_map > 0) & (a3_map < 0)] = -1  # Reversed
    # 0 = unusual/mixed

    fig, ax = plt.subplots(figsize=(8, 7))

    extent = [0, 2.0, 0, 2.0]
    im = ax.imshow(domain_map, cmap='coolwarm', vmin=-1, vmax=1, extent=extent, origin='lower')

    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title('Ferroelectric Domain Map\n(Blue=Reversed, White=Unusual, Red=Normal)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Reversed', 'Unusual', 'Normal'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Domain map saved to {save_path}")

    plt.show()
    return fig


def create_histogram_figure(result_df, save_path=None):
    """
    Create histogram distributions for all fitted parameters.

    This helps identify:
    - Parameter distributions (normal, bimodal, skewed)
    - Outliers and anomalies
    - Physical parameter ranges
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('Parameter Distributions from Hysteresis Loop Fitting',
                 fontsize=14, fontweight='bold')

    params = [
        ('a0', 'Offset a₀', 'steelblue'),
        ('a1', 'Amplitude a₁', 'darkorange'),
        ('a2', 'Coercive Voltage a₂ (V)', 'crimson'),
        ('a3', 'Coercive Voltage a₃ (V)', 'forestgreen'),
        ('a4', 'Linear Slope a₄', 'purple'),
        ('b0', 'Width b₀', 'teal'),
        ('b1', 'Width b₁', 'brown'),
        ('b2', 'Width b₂', 'olive'),
        ('b3', 'Width b₃', 'navy'),
    ]

    for ax, (param, label, color) in zip(axes.flatten(), params):
        data = result_df[param].dropna()

        # Remove extreme outliers for better visualization
        q01, q99 = np.percentile(data, [1, 99])
        data_clipped = data[(data >= q01) & (data <= q99)]

        ax.hist(data_clipped, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2e}')
        ax.axvline(data.median(), color='black', linestyle='-', linewidth=2, label=f'Median: {data.median():.2e}')

        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram figure saved to {save_path}")

    plt.show()
    return fig


def create_correlation_figure(result_df, save_path=None):
    """
    Create correlation matrix and scatter plots for key parameters.

    Physical interpretations:
    - a2 vs a3: Should be anticorrelated (opposite switching directions)
    - a1 vs RMSE: Large amplitude loops may be harder to fit
    - a2 vs a3 asymmetry: Indicates imprint or built-in bias
    """
    # Select key parameters for correlation
    params = ['a0', 'a1', 'a2', 'a3', 'a4', 'rmse']
    param_labels = ['a₀ (offset)', 'a₁ (amplitude)', 'a₂ (V)', 'a₃ (V)', 'a₄ (slope)', 'RMSE']

    df_subset = result_df[params].copy()

    # Compute correlation matrix
    corr_matrix = df_subset.corr()

    fig = plt.figure(figsize=(16, 12))

    # Create grid: correlation matrix on left, scatter plots on right
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

    # Correlation heatmap
    ax_corr = fig.add_subplot(gs[:, 0])
    im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_corr.set_xticks(range(len(params)))
    ax_corr.set_yticks(range(len(params)))
    ax_corr.set_xticklabels(param_labels, rotation=45, ha='right')
    ax_corr.set_yticklabels(param_labels)
    ax_corr.set_title('Parameter Correlation Matrix', fontsize=12, fontweight='bold')

    # Add correlation values as text
    for i in range(len(params)):
        for j in range(len(params)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax_corr.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04, label='Correlation')

    # Scatter plot 1: a2 vs a3 (most important - coercive voltages)
    ax1 = fig.add_subplot(gs[0, 1])
    scatter1 = ax1.scatter(result_df['a2'], result_df['a3'],
                           c=result_df['rmse'], cmap='viridis',
                           alpha=0.5, s=10, edgecolors='none')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('a₂ - Coercive Voltage (V)')
    ax1.set_ylabel('a₃ - Coercive Voltage (V)')
    ax1.set_title(f'Coercive Voltages\n(r = {corr_matrix.loc["a2", "a3"]:.3f})', fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='RMSE')
    ax1.grid(True, alpha=0.3)

    # Add quadrant labels
    ax1.text(0.95, 0.95, 'Normal', transform=ax1.transAxes, ha='right', va='top',
             fontsize=10, color='green', fontweight='bold')
    ax1.text(0.05, 0.05, 'Reversed', transform=ax1.transAxes, ha='left', va='bottom',
             fontsize=10, color='blue', fontweight='bold')

    # Scatter plot 2: Amplitude vs RMSE
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(np.abs(result_df['a1']), result_df['rmse'],
                alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax2.set_xlabel('|a₁| - Loop Amplitude')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'Amplitude vs Fit Quality\n(r = {corr_matrix.loc["a1", "rmse"]:.3f})', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Scatter plot 3: Asymmetry distribution
    ax3 = fig.add_subplot(gs[1, 1])
    asymmetry = result_df['a2'] + result_df['a3']
    ax3.hist(asymmetry, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Symmetric (0)')
    ax3.axvline(asymmetry.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {asymmetry.mean():.2f}V')
    ax3.set_xlabel('Switching Asymmetry (a₂ + a₃) [V]')
    ax3.set_ylabel('Count')
    ax3.set_title('Loop Asymmetry Distribution\n(Imprint indicator)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Scatter plot 4: Coercive voltage spread (|a2| + |a3|) vs amplitude
    ax4 = fig.add_subplot(gs[1, 2])
    coercive_spread = np.abs(result_df['a2']) + np.abs(result_df['a3'])
    scatter4 = ax4.scatter(coercive_spread, np.abs(result_df['a1']),
                           c=result_df['rmse'], cmap='viridis',
                           alpha=0.5, s=10, edgecolors='none')
    ax4.set_xlabel('Coercive Spread |a₂| + |a₃| (V)')
    ax4.set_ylabel('|a₁| - Loop Amplitude')
    ax4.set_title('Coercive Spread vs Amplitude', fontweight='bold')
    plt.colorbar(scatter4, ax=ax4, label='RMSE')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Parameter Correlations and Relationships', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Correlation figure saved to {save_path}")

    plt.show()
    return fig


def create_pairplot(result_df, save_path=None):
    """
    Create a comprehensive pairplot of key parameters.
    Color-coded by domain type for physical insight.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Skipping pairplot. Install with: pip install seaborn")
        return None

    # Add domain classification
    df_plot = result_df[['a1', 'a2', 'a3', 'a4', 'rmse']].copy()

    # Classify domains
    conditions = [
        (result_df['a2'] < 0) & (result_df['a3'] > 0),  # Normal
        (result_df['a2'] > 0) & (result_df['a3'] < 0),  # Reversed
    ]
    choices = ['Normal', 'Reversed']
    df_plot['Domain'] = np.select(conditions, choices, default='Unusual')

    # Create pairplot
    g = sns.pairplot(df_plot, hue='Domain',
                     palette={'Normal': 'red', 'Reversed': 'blue', 'Unusual': 'gray'},
                     diag_kind='hist',
                     plot_kws={'alpha': 0.4, 's': 15},
                     diag_kws={'alpha': 0.6, 'bins': 30},
                     corner=True)

    g.fig.suptitle('Pairwise Parameter Relationships by Domain Type', y=1.02, fontsize=14, fontweight='bold')

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")

    plt.show()
    return g


def print_statistics(result_df):
    """Print summary statistics with physical interpretation."""
    print("\n" + "="*60)
    print("FITTING RESULTS SUMMARY")
    print("="*60)

    print("\n📊 Fit Quality:")
    print(f"   Mean RMSE: {result_df['rmse'].mean():.2e}")
    print(f"   Median RMSE: {result_df['rmse'].median():.2e}")
    print(f"   Max RMSE: {result_df['rmse'].max():.2e}")
    poor_fits = (result_df['rmse'] > result_df['rmse'].quantile(0.95)).sum()
    print(f"   Poor fits (>95th percentile): {poor_fits} loops")

    print("\n⚡ Coercive Voltages:")
    print(f"   a₂ (neg→pos): {result_df['a2'].mean():.2f} ± {result_df['a2'].std():.2f} V")
    print(f"   a₃ (pos→neg): {result_df['a3'].mean():.2f} ± {result_df['a3'].std():.2f} V")
    asymmetry = result_df['a2'] + result_df['a3']
    print(f"   Asymmetry (a₂+a₃): {asymmetry.mean():.2f} ± {asymmetry.std():.2f} V")

    print("\n📈 Loop Characteristics:")
    print(f"   Amplitude |a₁|: {np.abs(result_df['a1']).mean():.2e} ± {np.abs(result_df['a1']).std():.2e}")
    print(f"   Linear slope a₄: {result_df['a4'].mean():.2e} ± {result_df['a4'].std():.2e}")

    # Domain statistics
    normal = ((result_df['a2'] < 0) & (result_df['a3'] > 0)).sum()
    reversed_dom = ((result_df['a2'] > 0) & (result_df['a3'] < 0)).sum()
    unusual = len(result_df) - normal - reversed_dom

    print("\n🔬 Domain Distribution:")
    print(f"   Normal polarity: {normal} ({100*normal/len(result_df):.1f}%)")
    print(f"   Reversed polarity: {reversed_dom} ({100*reversed_dom/len(result_df):.1f}%)")
    print(f"   Unusual/mixed: {unusual} ({100*unusual/len(result_df):.1f}%)")
    print("="*60)


if __name__ == "__main__":
    # Load data
    print("Loading hysteresis loop data...")
    hysteresis_loops = np.load('hysteresis_loops.npy')
    dc_vec = np.load('dc_vec.npy')

    # Flatten if needed
    if hysteresis_loops.ndim == 3:
        loops_flat = hysteresis_loops.reshape(-1, hysteresis_loops.shape[-1])
    else:
        loops_flat = hysteresis_loops

    # Roll voltage vector
    imin = np.argmin(dc_vec)
    vdc_rolled = np.roll(dc_vec, -imin)

    # Clean loops (same preprocessing as main script)
    print("Preprocessing loops...")
    cleaned_loops = []
    for y in tqdm(loops_flat, desc="Cleaning"):
        y_rolled = np.roll(y, -imin)
        q10 = np.quantile(y_rolled, 0.10)
        q90 = np.quantile(y_rolled, 0.90)
        spread = q90 - q10
        y_clean = clean_by_large_diff(vdc_rolled, y_rolled, diff_thresh=spread, expand=1)
        cleaned_loops.append(y_clean)
    cleaned_loops = np.array(cleaned_loops)

    # Fit all loops (or load cached results)
    result_df = load_or_compute_results(
        cleaned_loops,
        vdc_rolled,
        result_path="result_spatial.csv",
        n_starts=8,  # Increase for better fits
        force_recompute=False
    )

    # Print statistics
    print_statistics(result_df)

    # Create visualizations
    print("\nGenerating spatial analysis plots...")
    create_spatial_analysis_figure(result_df, save_path="spatial_analysis.png")
    create_domain_map(result_df, save_path="domain_map.png")

    # Create histogram and correlation plots
    print("\nGenerating histogram distributions...")
    create_histogram_figure(result_df, save_path="histograms.png")

    print("\nGenerating correlation analysis...")
    create_correlation_figure(result_df, save_path="correlations.png")

    print("\nGenerating pairplot (by domain type)...")
    create_pairplot(result_df, save_path="pairplot.png")

    print("\n✅ Analysis complete!")
