# -*- coding: utf-8 -*-
"""
Improved Hysteresis Loop Fitting Pipeline

Improvements over baseline:
1. Domain-Aware Initialization - Pre-classify loops for better initial guesses
2. Robust Outlier Detection - Isolation Forest for anomaly identification
3. Adaptive Fitting Strategy - Retry failed fits with different approaches

Author: Analysis Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import least_squares
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from pathlib import Path


# =============================================================================
# Core fitting functions (same as before)
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


# =============================================================================
# IMPROVEMENT 1: Domain-Aware Initialization
# =============================================================================

def extract_loop_features(vdc, y):
    """
    Extract simple features from a loop for domain classification.

    Features:
    - slope_first_half: Average slope in first half (indicates switching direction)
    - slope_second_half: Average slope in second half
    - max_gradient_pos: Position of maximum gradient
    - loop_area: Approximate hysteresis area (indicates ferroelectric strength)
    """
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]
    y1, y2 = y[:n2], y[n2:]

    # Slopes
    slope1 = np.polyfit(v1, y1, 1)[0] if len(v1) > 1 else 0
    slope2 = np.polyfit(v2, y2, 1)[0] if len(v2) > 1 else 0

    # Maximum gradient position (where switching occurs)
    dy = np.gradient(y, vdc)
    max_grad_idx = np.argmax(np.abs(dy))
    max_grad_pos = vdc[max_grad_idx]

    # Loop area (crude estimate)
    area = np.abs(np.trapz(y1, v1) - np.trapz(y2, v2))

    # Asymmetry indicator
    y_mean_first = np.mean(y1)
    y_mean_second = np.mean(y2)
    asymmetry = y_mean_first - y_mean_second

    return {
        'slope1': slope1,
        'slope2': slope2,
        'max_grad_pos': max_grad_pos,
        'area': area,
        'asymmetry': asymmetry
    }


def classify_domain_type(vdc, y):
    """
    Pre-classify a loop as 'normal', 'reversed', or 'uncertain'.

    Normal: negative coercive voltage on ascending, positive on descending
    Reversed: opposite pattern
    """
    features = extract_loop_features(vdc, y)

    # Simple heuristic based on gradient positions and slopes
    n2 = len(vdc) // 2
    dy = np.gradient(y, vdc)

    # Find max gradient in each half
    max_grad_first = np.argmax(np.abs(dy[:n2]))
    max_grad_second = np.argmax(np.abs(dy[n2:])) + n2

    v_switch_first = vdc[max_grad_first]
    v_switch_second = vdc[max_grad_second]

    # Classification logic
    if v_switch_first > 0 and v_switch_second < 0:
        return 'reversed'
    elif v_switch_first < 0 and v_switch_second > 0:
        return 'normal'
    else:
        return 'uncertain'


def make_domain_aware_x0(vdc, y_obs, domain_type='auto'):
    """
    Generate initial parameters with domain-aware initialization.

    For 'reversed' domains: a2 > 0, a3 < 0 (typical for this dataset)
    For 'normal' domains: a2 < 0, a3 > 0
    """
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]
    y1, y2 = y_obs[:n2], y_obs[n2:]

    if domain_type == 'auto':
        domain_type = classify_domain_type(vdc, y_obs)

    # Basic parameters (same for all)
    a0 = float(np.median(y_obs))
    a1 = float(np.max(y_obs) - np.min(y_obs))
    if abs(a1) < 1e-12:
        a1 = 1.0

    # Linear slope from tails
    k = max(3, int(0.2 * len(vdc)))
    order = np.argsort(vdc)
    idx = np.r_[order[:k], order[-k:]]
    X = np.column_stack([np.ones(len(idx)), vdc[idx]])
    beta, *_ = np.linalg.lstsq(X, y_obs[idx], rcond=None)
    a4 = float(beta[1])

    # Domain-specific coercive voltage initialization
    if domain_type == 'reversed':
        # Typical for this dataset: a2 > 0, a3 < 0
        a2 = 3.0   # Positive coercive voltage
        a3 = -3.0  # Negative coercive voltage
    elif domain_type == 'normal':
        # Opposite pattern
        a2 = -3.0
        a3 = 3.0
    else:
        # Uncertain: use gradient-based detection
        dy1 = np.gradient(y1, v1)
        dy2 = np.gradient(y2, v2)
        a2 = float(v1[np.argmax(np.abs(dy1))])
        a3 = float(v2[np.argmax(np.abs(dy2))])

    # Width parameters (same heuristic)
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

    return np.array([a0, a1, a2, a3, a4, b0, b1, b2, b3], float), domain_type


# =============================================================================
# IMPROVEMENT 2: Robust Outlier Detection
# =============================================================================

def detect_outlier_loops(loops_flat, vdc, contamination=0.1):
    """
    Use Isolation Forest to detect anomalous loops.

    Features used:
    - Loop statistics (mean, std, range, skewness)
    - Gradient statistics
    - Area and asymmetry

    Returns:
    - outlier_mask: Boolean array (True = outlier)
    - outlier_scores: Anomaly scores (-1 to 0, lower = more anomalous)
    """
    print("Extracting features for outlier detection...")
    features = []

    for y in tqdm(loops_flat, desc="Feature extraction"):
        # Basic statistics
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_range = np.max(y) - np.min(y)
        y_skew = np.mean(((y - y_mean) / (y_std + 1e-12)) ** 3)

        # Gradient statistics
        dy = np.diff(y)
        dy_max = np.max(np.abs(dy))
        dy_std = np.std(dy)

        # Number of sign changes in gradient (smoothness indicator)
        sign_changes = np.sum(np.diff(np.sign(dy)) != 0)

        # Split statistics
        n2 = len(y) // 2
        diff_halves = np.abs(np.mean(y[:n2]) - np.mean(y[n2:]))

        features.append([
            y_mean, y_std, y_range, y_skew,
            dy_max, dy_std, sign_changes, diff_halves
        ])

    features = np.array(features)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Isolation Forest
    print("Running Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto'
    )

    predictions = iso_forest.fit_predict(features_scaled)
    scores = iso_forest.decision_function(features_scaled)

    outlier_mask = predictions == -1

    print(f"Detected {outlier_mask.sum()} outliers ({100*outlier_mask.mean():.1f}%)")

    return outlier_mask, scores


# =============================================================================
# IMPROVEMENT 3: Adaptive Fitting Strategy
# =============================================================================

def fit_with_strategy(vdc, y_obs, strategy='standard', seed=42, b_max=20.0):
    """
    Fit a single loop with a specified strategy.

    Strategies:
    - 'standard': Normal multi-start with domain-aware init
    - 'aggressive': More starts, finer d-schedule
    - 'flipped': Try opposite domain type
    - 'simple': Fewer parameters (fix some b's)
    """
    vmin, vmax = float(np.min(vdc)), float(np.max(vdc))
    lb = np.array([-np.inf, -np.inf, vmin, vmin, -np.inf, 1e-3, 1e-3, 1e-3, 1e-3], float)
    ub = np.array([np.inf, np.inf, vmax, vmax, np.inf, b_max, b_max, b_max, b_max], float)

    rng = np.random.default_rng(seed)

    if strategy == 'standard':
        base, domain_type = make_domain_aware_x0(vdc, y_obs, 'auto')
        n_starts = 10
        d_schedule = (30, 80, 200, 600, 1000)
    elif strategy == 'aggressive':
        base, domain_type = make_domain_aware_x0(vdc, y_obs, 'auto')
        n_starts = 30
        d_schedule = (20, 50, 100, 200, 400, 700, 1000)
    elif strategy == 'flipped':
        # Try opposite domain type
        _, auto_type = make_domain_aware_x0(vdc, y_obs, 'auto')
        flip_type = 'normal' if auto_type == 'reversed' else 'reversed'
        base, domain_type = make_domain_aware_x0(vdc, y_obs, flip_type)
        n_starts = 15
        d_schedule = (30, 80, 200, 600, 1000)
    elif strategy == 'simple':
        # Fix b parameters to reduce complexity
        base, domain_type = make_domain_aware_x0(vdc, y_obs, 'auto')
        base[5:9] = [1.0, 2.0, 1.0, 2.0]  # Fixed widths
        n_starts = 20
        d_schedule = (50, 150, 400, 1000)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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
            try:
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
            except Exception:
                continue

        if res.cost < best_cost:
            best_cost = res.cost
            best = x

    return best, best_cost, strategy


def adaptive_fit_single_loop(idx, y, vdc, outlier_score):
    """
    Adaptively fit a single loop with fallback strategies.

    Strategy order:
    1. Standard fit with domain-aware initialization
    2. If RMSE > threshold, try aggressive fit
    3. If still bad, try flipped domain type
    4. If still bad, try simplified model
    """
    # Threshold for "bad" fit (will be calibrated)
    rmse_threshold = np.std(y) * 0.1  # 10% of signal variation

    strategies_tried = []
    results = []

    # Strategy 1: Standard
    p_best, cost, strategy = fit_with_strategy(vdc, y, 'standard', seed=42+idx)
    y_pred = loop_fit_function(vdc, p_best)
    rmse = np.sqrt(np.mean((y_pred - y)**2))
    strategies_tried.append(strategy)
    results.append((p_best, rmse, strategy))

    # If outlier or bad fit, try more strategies
    if outlier_score < -0.1 or rmse > rmse_threshold:
        # Strategy 2: Aggressive
        p_best2, cost2, strategy2 = fit_with_strategy(vdc, y, 'aggressive', seed=42+idx+1000)
        y_pred2 = loop_fit_function(vdc, p_best2)
        rmse2 = np.sqrt(np.mean((y_pred2 - y)**2))
        strategies_tried.append(strategy2)
        results.append((p_best2, rmse2, strategy2))

        if rmse2 > rmse_threshold:
            # Strategy 3: Flipped
            p_best3, cost3, strategy3 = fit_with_strategy(vdc, y, 'flipped', seed=42+idx+2000)
            y_pred3 = loop_fit_function(vdc, p_best3)
            rmse3 = np.sqrt(np.mean((y_pred3 - y)**2))
            strategies_tried.append(strategy3)
            results.append((p_best3, rmse3, strategy3))

    # Select best result
    best_idx = np.argmin([r[1] for r in results])
    p_final, rmse_final, strategy_final = results[best_idx]

    return {
        'loop_id': idx,
        'rmse': rmse_final,
        'a0': p_final[0],
        'a1': p_final[1],
        'a2': p_final[2],
        'a3': p_final[3],
        'a4': p_final[4],
        'b0': p_final[5],
        'b1': p_final[6],
        'b2': p_final[7],
        'b3': p_final[8],
        'strategy_used': strategy_final,
        'strategies_tried': len(strategies_tried),
        'outlier_score': outlier_score
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_improved_fitting(
    loops_flat,
    vdc_rolled,
    result_path="result_improved.csv",
    n_jobs=-1,
    force_recompute=False,
    outlier_contamination=0.15
):
    """
    Run the improved fitting pipeline.

    Steps:
    1. Detect outliers using Isolation Forest
    2. Fit all loops with adaptive strategy
    3. Save results with metadata
    """
    result_path = Path(result_path)

    if result_path.exists() and not force_recompute:
        print(f"Loading cached results from {result_path}")
        return pd.read_csv(result_path, index_col=0)

    # Step 1: Outlier detection
    outlier_mask, outlier_scores = detect_outlier_loops(
        loops_flat, vdc_rolled, contamination=outlier_contamination
    )

    # Step 2: Parallel fitting with adaptive strategy
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    print(f"\nFitting {len(loops_flat)} loops with adaptive strategy (n_jobs={n_jobs})...")

    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(adaptive_fit_single_loop)(idx, loops_flat[idx], vdc_rolled, outlier_scores[idx])
        for idx in range(len(loops_flat))
    )

    result = pd.DataFrame(rows)
    result = result.sort_values('loop_id').reset_index(drop=True)
    result['is_outlier'] = outlier_mask

    result.to_csv(result_path)
    print(f"\nResults saved to {result_path}")

    return result


def compare_results(old_result_path, new_result_path):
    """Compare old and new fitting results."""
    old_df = pd.read_csv(old_result_path, index_col=0)
    new_df = pd.read_csv(new_result_path, index_col=0)

    print("\n" + "="*60)
    print("COMPARISON: Original vs Improved Fitting")
    print("="*60)

    print("\n📊 RMSE Statistics:")
    print(f"   Original - Mean: {old_df['rmse'].mean():.2e}, Median: {old_df['rmse'].median():.2e}")
    print(f"   Improved - Mean: {new_df['rmse'].mean():.2e}, Median: {new_df['rmse'].median():.2e}")

    improvement = (old_df['rmse'].mean() - new_df['rmse'].mean()) / old_df['rmse'].mean() * 100
    print(f"\n   Mean RMSE improvement: {improvement:.1f}%")

    # Per-loop improvement
    rmse_diff = old_df['rmse'] - new_df['rmse']
    improved_count = (rmse_diff > 0).sum()
    print(f"   Loops improved: {improved_count} ({100*improved_count/len(rmse_diff):.1f}%)")

    # Strategy usage
    if 'strategy_used' in new_df.columns:
        print("\n🎯 Strategy Usage:")
        strategy_counts = new_df['strategy_used'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"   {strategy}: {count} ({100*count/len(new_df):.1f}%)")

    # Poor fits comparison
    old_poor = (old_df['rmse'] > old_df['rmse'].quantile(0.95)).sum()
    new_poor = (new_df['rmse'] > new_df['rmse'].quantile(0.95)).sum()
    print(f"\n⚠️ Poor Fits (>95th percentile):")
    print(f"   Original: {old_poor}")
    print(f"   Improved: {new_poor}")

    print("="*60)

    return old_df, new_df


def plot_comparison(old_df, new_df, save_path="comparison.png"):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMSE histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(old_df['rmse'], bins=50, alpha=0.5, label='Original', color='blue')
    ax1.hist(new_df['rmse'], bins=50, alpha=0.5, label='Improved', color='green')
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Count')
    ax1.set_title('RMSE Distribution Comparison')
    ax1.legend()
    ax1.set_yscale('log')

    # RMSE improvement scatter
    ax2 = axes[0, 1]
    ax2.scatter(old_df['rmse'], new_df['rmse'], alpha=0.3, s=5)
    max_rmse = max(old_df['rmse'].max(), new_df['rmse'].max())
    ax2.plot([0, max_rmse], [0, max_rmse], 'r--', label='No change')
    ax2.set_xlabel('Original RMSE')
    ax2.set_ylabel('Improved RMSE')
    ax2.set_title('Per-Loop RMSE Comparison')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # RMSE improvement by loop
    ax3 = axes[1, 0]
    improvement = (old_df['rmse'] - new_df['rmse']) / old_df['rmse'] * 100
    ax3.hist(improvement, bins=50, color='purple', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', label='No improvement')
    ax3.axvline(improvement.mean(), color='orange', linestyle='-',
                label=f'Mean: {improvement.mean():.1f}%')
    ax3.set_xlabel('Improvement (%)')
    ax3.set_ylabel('Count')
    ax3.set_title('Per-Loop RMSE Improvement')
    ax3.legend()

    # Strategy usage (if available)
    ax4 = axes[1, 1]
    if 'strategy_used' in new_df.columns:
        strategy_counts = new_df['strategy_used'].value_counts()
        colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
        ax4.bar(strategy_counts.index, strategy_counts.values, color=colors[:len(strategy_counts)])
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Count')
        ax4.set_title('Fitting Strategy Usage')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Strategy data not available',
                 transform=ax4.transAxes, ha='center', va='center')

    plt.suptitle('Improved Fitting Pipeline Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")

    plt.show()
    return fig


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

    # Clean loops
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

    # Run improved fitting
    result_improved = run_improved_fitting(
        cleaned_loops,
        vdc_rolled,
        result_path="result_improved.csv",
        n_jobs=-1,
        force_recompute=True,  # Set to False to use cached results
        outlier_contamination=0.15
    )

    # Compare with original results
    if Path("result_spatial.csv").exists():
        old_df, new_df = compare_results("result_spatial.csv", "result_improved.csv")
        plot_comparison(old_df, new_df, save_path="comparison.png")

    print("\n✅ Improved fitting complete!")
