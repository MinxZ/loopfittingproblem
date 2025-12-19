# -*- coding: utf-8 -*-
"""
Improved Preprocessing for Hysteresis Loop Fitting

Better spike/outlier detection methods:
1. Savitzky-Golay smoothing + residual detection
2. Median filter for robust spike removal
3. Iterative outlier detection
4. Comparison with original MAD-based method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import median_filter
from scipy.special import erf
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from pathlib import Path


# =============================================================================
# Original preprocessing (baseline)
# =============================================================================

def clean_by_large_diff_original(v, y, diff_thresh=None, k_mad=8.0, expand=1):
    """Original MAD-based spike detection."""
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


# =============================================================================
# IMPROVED METHOD 1: Savitzky-Golay based detection
# =============================================================================

def clean_savgol(y, window_length=11, polyorder=3, threshold_sigma=3.0):
    """
    Use Savitzky-Golay filter to detect outliers.

    Method:
    1. Apply SG filter to get smooth baseline
    2. Compute residuals
    3. Flag points with residuals > threshold_sigma * MAD
    4. Interpolate flagged points

    Advantages:
    - Preserves loop shape better than simple smoothing
    - Detects isolated spikes effectively
    """
    y = np.asarray(y, float)
    n = len(y)

    if n < window_length:
        return y.copy()

    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1

    # Apply Savitzky-Golay filter
    y_smooth = savgol_filter(y, window_length, polyorder)

    # Compute residuals
    residuals = y - y_smooth

    # Robust threshold using MAD
    med_res = np.median(residuals)
    mad = np.median(np.abs(residuals - med_res)) + 1e-12
    threshold = threshold_sigma * 1.4826 * mad

    # Flag outliers
    mask = np.abs(residuals - med_res) > threshold

    # Interpolate flagged points
    y_clean = y.copy()
    good = ~mask
    if good.sum() >= 2 and mask.any():
        x = np.arange(n)
        y_clean[mask] = np.interp(x[mask], x[good], y[good])

    return y_clean


# =============================================================================
# IMPROVED METHOD 2: Median filter based detection
# =============================================================================

def clean_median_filter(y, kernel_size=5, threshold_sigma=3.0):
    """
    Use median filter to detect outliers.

    Method:
    1. Apply median filter to get robust baseline
    2. Compute difference from median
    3. Flag points with large deviations
    4. Replace with median values

    Advantages:
    - Very robust to isolated spikes
    - Non-parametric, no assumptions about signal shape
    """
    y = np.asarray(y, float)
    n = len(y)

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply median filter
    y_median = medfilt(y, kernel_size)

    # Compute deviations
    deviations = np.abs(y - y_median)

    # Robust threshold
    med_dev = np.median(deviations)
    mad = np.median(np.abs(deviations - med_dev)) + 1e-12
    threshold = threshold_sigma * 1.4826 * mad

    # Replace outliers with median values
    mask = deviations > threshold
    y_clean = y.copy()
    y_clean[mask] = y_median[mask]

    return y_clean


# =============================================================================
# IMPROVED METHOD 3: Iterative outlier detection
# =============================================================================

def clean_iterative(y, max_iter=3, threshold_sigma=3.5):
    """
    Iteratively detect and remove outliers.

    Method:
    1. Fit a smooth curve (polynomial or spline)
    2. Detect outliers from residuals
    3. Interpolate outliers
    4. Repeat until convergence

    Advantages:
    - Can handle multiple nearby spikes
    - Adapts to local signal characteristics
    """
    y = np.asarray(y, float)
    n = len(y)
    y_clean = y.copy()
    x = np.arange(n)

    for iteration in range(max_iter):
        # Fit polynomial to current data
        # Use robust fitting by downweighting outliers
        good_mask = np.ones(n, dtype=bool)

        # Fit degree-5 polynomial
        coeffs = np.polyfit(x, y_clean, deg=5)
        y_fit = np.polyval(coeffs, x)

        # Compute residuals
        residuals = y_clean - y_fit

        # Robust threshold
        med_res = np.median(residuals)
        mad = np.median(np.abs(residuals - med_res)) + 1e-12
        threshold = threshold_sigma * 1.4826 * mad

        # Detect outliers
        outlier_mask = np.abs(residuals - med_res) > threshold

        if not outlier_mask.any():
            break

        # Interpolate outliers
        good = ~outlier_mask
        if good.sum() >= 2:
            y_clean[outlier_mask] = np.interp(x[outlier_mask], x[good], y_clean[good])

    return y_clean


# =============================================================================
# IMPROVED METHOD 4: Combined approach (best of all)
# =============================================================================

def clean_combined(y, v=None):
    """
    Combined approach using multiple methods.

    Strategy:
    1. First pass: Median filter for extreme spikes
    2. Second pass: Savitzky-Golay for subtler outliers
    3. Third pass: MAD-based diff check for remaining jumps

    This catches different types of artifacts.
    """
    y = np.asarray(y, float)

    # Pass 1: Median filter (catches extreme isolated spikes)
    y_clean = clean_median_filter(y, kernel_size=5, threshold_sigma=4.0)

    # Pass 2: Savitzky-Golay (catches subtler deviations)
    y_clean = clean_savgol(y_clean, window_length=9, polyorder=3, threshold_sigma=3.5)

    # Pass 3: Gradient-based check (catches remaining jumps)
    n = len(y_clean)
    dy = np.diff(y_clean)
    ady = np.abs(dy)
    med_dy = np.median(ady)
    mad_dy = np.median(np.abs(ady - med_dy)) + 1e-12
    threshold = med_dy + 5.0 * 1.4826 * mad_dy

    jumps = ady > threshold
    if jumps.any():
        mask = np.zeros(n, dtype=bool)
        idx = np.where(jumps)[0]
        mask[idx] = True
        mask[np.minimum(idx + 1, n - 1)] = True

        good = ~mask
        if good.sum() >= 2:
            x = np.arange(n)
            y_clean[mask] = np.interp(x[mask], x[good], y_clean[good])

    return y_clean


# =============================================================================
# Fitting functions (same as before)
# =============================================================================

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


def make_x0_coef(vdc, y_obs):
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]
    y1, y2 = y_obs[:n2], y_obs[n2:]

    def switch_center(v, y):
        dy = np.gradient(y, v)
        return float(v[np.argmax(np.abs(dy))])

    a2 = switch_center(v1, y1)
    a3 = switch_center(v2, y2)

    k = max(3, int(0.2 * len(vdc)))
    order = np.argsort(vdc)
    idx = np.r_[order[:k], order[-k:]]
    X = np.column_stack([np.ones(len(idx)), vdc[idx]])
    beta, *_ = np.linalg.lstsq(X, y_obs[idx], rcond=None)
    a4 = float(beta[1])

    a0 = float(np.median(y_obs))
    a1 = float(np.max(y_obs) - np.min(y_obs))
    if abs(a1) < 1e-12:
        a1 = 1.0

    def width_10_90(v, y):
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

    w1 = width_10_90(v1, y1)
    w2 = width_10_90(v2, y2)

    b0 = max(0.1, 0.5 * w1)
    b1 = max(0.2, 1.2 * w1)
    b2 = max(0.1, 0.5 * w2)
    b3 = max(0.2, 1.2 * w2)

    return np.array([a0, a1, a2, a3, a4, b0, b1, b2, b3], float)


def fit_single_loop(vdc, y, n_starts=8, seed=42):
    """Fit a single loop with multi-start optimization."""
    rng = np.random.default_rng(seed)
    vmin, vmax = float(np.min(vdc)), float(np.max(vdc))

    lb = np.array([-np.inf, -np.inf, vmin, vmin, -np.inf, 1e-3, 1e-3, 1e-3, 1e-3], float)
    ub = np.array([np.inf, np.inf, vmax, vmax, np.inf, 20.0, 20.0, 20.0, 20.0], float)

    base = make_x0_coef(vdc, y)

    def jitter(coef):
        c = coef.copy()
        c[2] += rng.normal(0, 0.8)
        c[3] += rng.normal(0, 0.8)
        for j in [5, 6, 7, 8]:
            c[j] *= np.exp(rng.normal(0, 0.35))
        c[0] += rng.normal(0, 0.2 * np.std(y))
        c[1] *= np.exp(rng.normal(0, 0.2))
        c[4] += rng.normal(0, 0.2 * abs(c[4]) + 1e-6)
        return c

    best = None
    best_cost = np.inf
    d_schedule = (30, 80, 200, 600, 1000)

    for s in range(n_starts):
        x0 = base if s == 0 else jitter(base)
        x0 = np.minimum(np.maximum(x0, lb), ub)

        x = x0
        for d in d_schedule:
            try:
                res = least_squares(
                    residuals_coef, x,
                    args=(vdc, y, d),
                    method="trf",
                    loss="soft_l1",
                    bounds=(lb, ub),
                    x_scale="jac",
                    max_nfev=4000
                )
                x = res.x
            except:
                continue

        if res.cost < best_cost:
            best_cost = res.cost
            best = x

    return best


def fit_and_score(idx, y, vdc, n_starts=8):
    """Fit a loop and return RMSE."""
    p = fit_single_loop(vdc, y, n_starts=n_starts, seed=42+idx)
    y_pred = loop_fit_function(vdc, p)
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    return {'loop_id': idx, 'rmse': rmse, **{f'p{i}': p[i] for i in range(9)}}


# =============================================================================
# Comparison functions
# =============================================================================

def preprocess_all_methods(loops_flat, vdc_rolled, imin):
    """Apply all preprocessing methods to all loops."""
    methods = {
        'original': lambda y: clean_by_large_diff_original(
            vdc_rolled, np.roll(y, -imin),
            diff_thresh=np.quantile(np.roll(y, -imin), 0.90) - np.quantile(np.roll(y, -imin), 0.10),
            expand=1
        ),
        'savgol': lambda y: clean_savgol(np.roll(y, -imin), window_length=11, threshold_sigma=3.0),
        'median': lambda y: clean_median_filter(np.roll(y, -imin), kernel_size=5, threshold_sigma=3.0),
        'iterative': lambda y: clean_iterative(np.roll(y, -imin), max_iter=3, threshold_sigma=3.5),
        'combined': lambda y: clean_combined(np.roll(y, -imin)),
    }

    results = {}
    for name, method in methods.items():
        print(f"Applying {name} preprocessing...")
        cleaned = []
        for y in tqdm(loops_flat, desc=name):
            cleaned.append(method(y))
        results[name] = np.array(cleaned)

    return results


def visualize_preprocessing_comparison(loops_flat, vdc_rolled, imin, sample_indices=None):
    """Visualize preprocessing methods on sample loops."""
    if sample_indices is None:
        # Find loops with potential spikes (high gradient variation)
        spike_scores = []
        for i, y in enumerate(loops_flat):
            y_rolled = np.roll(y, -imin)
            dy = np.diff(y_rolled)
            score = np.max(np.abs(dy)) / (np.median(np.abs(dy)) + 1e-12)
            spike_scores.append((i, score))
        spike_scores.sort(key=lambda x: -x[1])
        sample_indices = [s[0] for s in spike_scores[:6]]  # Top 6 spikiest

    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(20, 3*len(sample_indices)))

    methods = ['Original MAD', 'Savgol', 'Median', 'Iterative', 'Combined']

    for row, idx in enumerate(sample_indices):
        y_raw = np.roll(loops_flat[idx], -imin)

        # Apply each method
        spread = np.quantile(y_raw, 0.90) - np.quantile(y_raw, 0.10)
        y_original = clean_by_large_diff_original(vdc_rolled, y_raw, diff_thresh=spread, expand=1)
        y_savgol = clean_savgol(y_raw, window_length=11, threshold_sigma=3.0)
        y_median = clean_median_filter(y_raw, kernel_size=5, threshold_sigma=3.0)
        y_iterative = clean_iterative(y_raw, max_iter=3, threshold_sigma=3.5)
        y_combined = clean_combined(y_raw)

        cleaned_versions = [y_original, y_savgol, y_median, y_iterative, y_combined]

        for col, (y_clean, method_name) in enumerate(zip(cleaned_versions, methods)):
            ax = axes[row, col]
            ax.plot(vdc_rolled, y_raw, 'b-', alpha=0.3, label='Raw')
            ax.plot(vdc_rolled, y_clean, 'r-', linewidth=1.5, label='Cleaned')

            # Highlight removed points
            diff = np.abs(y_raw - y_clean)
            changed = diff > 1e-12
            if changed.any():
                ax.scatter(vdc_rolled[changed], y_raw[changed], c='blue', s=20, marker='x', zorder=5)

            if row == 0:
                ax.set_title(method_name, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Loop {idx}', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Preprocessing Methods Comparison on Spiky Loops', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved preprocessing_comparison.png")
    plt.show()

    return sample_indices


def run_fitting_comparison(preprocessed_data, vdc_rolled, n_starts=8, n_jobs=-1):
    """Run fitting on all preprocessing methods and compare."""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    results = {}

    for method_name, cleaned_loops in preprocessed_data.items():
        print(f"\nFitting with {method_name} preprocessing...")

        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(fit_and_score)(idx, cleaned_loops[idx], vdc_rolled, n_starts)
            for idx in range(len(cleaned_loops))
        )

        df = pd.DataFrame(rows)
        df = df.sort_values('loop_id').reset_index(drop=True)
        results[method_name] = df

        print(f"   {method_name}: Mean RMSE = {df['rmse'].mean():.2e}, Median = {df['rmse'].median():.2e}")

    return results


def create_comparison_summary(results, save_path="preprocessing_results.png"):
    """Create summary comparison of all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Plot 1: RMSE distributions
    ax1 = axes[0, 0]
    for method, color in zip(methods, colors):
        ax1.hist(results[method]['rmse'], bins=50, alpha=0.5, label=method, color=color)
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Count')
    ax1.set_title('RMSE Distributions by Preprocessing Method')
    ax1.legend()
    ax1.set_yscale('log')

    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    data_for_box = [results[m]['rmse'].values for m in methods]
    bp = ax2.boxplot(data_for_box, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Distribution Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')

    # Plot 3: Mean and Median comparison
    ax3 = axes[1, 0]
    means = [results[m]['rmse'].mean() for m in methods]
    medians = [results[m]['rmse'].median() for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax3.bar(x - width/2, means, width, label='Mean', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, medians, width, label='Median', color='darkorange', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Mean vs Median RMSE by Method')
    ax3.legend()
    ax3.set_yscale('log')

    # Plot 4: Poor fits (>95th percentile) count
    ax4 = axes[1, 1]
    # Use original as reference for threshold
    threshold = results['original']['rmse'].quantile(0.95)
    poor_counts = [((results[m]['rmse'] > threshold).sum()) for m in methods]
    bars = ax4.bar(methods, poor_counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Count of Poor Fits')
    ax4.set_title(f'Loops with RMSE > {threshold:.2e} (95th percentile of original)')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, count in zip(bars, poor_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontsize=10)

    plt.suptitle('Preprocessing Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.show()

    # Print summary table
    print("\n" + "="*70)
    print("PREPROCESSING COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<15} {'Mean RMSE':<15} {'Median RMSE':<15} {'Poor Fits':<12} {'Improvement':<12}")
    print("-"*70)

    baseline_mean = results['original']['rmse'].mean()
    for method in methods:
        mean_rmse = results[method]['rmse'].mean()
        median_rmse = results[method]['rmse'].median()
        poor = (results[method]['rmse'] > threshold).sum()
        improvement = (baseline_mean - mean_rmse) / baseline_mean * 100
        print(f"{method:<15} {mean_rmse:<15.2e} {median_rmse:<15.2e} {poor:<12} {improvement:>+.1f}%")

    print("="*70)

    return threshold


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    hysteresis_loops = np.load('hysteresis_loops.npy')
    dc_vec = np.load('dc_vec.npy')

    if hysteresis_loops.ndim == 3:
        loops_flat = hysteresis_loops.reshape(-1, hysteresis_loops.shape[-1])
    else:
        loops_flat = hysteresis_loops

    imin = np.argmin(dc_vec)
    vdc_rolled = np.roll(dc_vec, -imin)

    # Visualize preprocessing on sample spiky loops
    print("\nVisualizing preprocessing methods on spiky loops...")
    sample_indices = visualize_preprocessing_comparison(loops_flat, vdc_rolled, imin)

    # Apply all preprocessing methods
    print("\nApplying all preprocessing methods to all loops...")
    preprocessed = preprocess_all_methods(loops_flat, vdc_rolled, imin)

    # Run fitting comparison
    print("\nRunning fitting comparison (this may take a while)...")
    results = run_fitting_comparison(preprocessed, vdc_rolled, n_starts=8, n_jobs=-1)

    # Save results
    for method, df in results.items():
        df.to_csv(f'result_{method}.csv')
        print(f"Saved result_{method}.csv")

    # Create summary
    create_comparison_summary(results)

    print("\n✅ Preprocessing comparison complete!")
