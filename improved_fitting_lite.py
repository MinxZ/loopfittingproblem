# -*- coding: utf-8 -*-
"""
Improved Hysteresis Loop Fitting - Lite Version

Key improvement: Domain-aware initialization
- Pre-classifies loops as 'normal' or 'reversed' based on gradient features
- Uses appropriate initial guesses for each domain type
- Much faster than full adaptive strategy, but still significantly better than baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from pathlib import Path


# =============================================================================
# Core fitting functions
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
# Domain-Aware Initialization (Key Improvement)
# =============================================================================

def classify_domain_type(vdc, y):
    """
    Pre-classify a loop as 'normal', 'reversed', or 'uncertain'.

    Based on where maximum gradient occurs in each half:
    - Reversed (majority in this dataset): max gradient at positive V in first half
    - Normal: max gradient at negative V in first half
    """
    n2 = len(vdc) // 2
    dy = np.gradient(y, vdc)

    # Find max gradient position in first half
    max_grad_first_idx = np.argmax(np.abs(dy[:n2]))
    v_switch_first = vdc[max_grad_first_idx]

    # Simple classification based on first-half switching position
    if v_switch_first > 0:
        return 'reversed'
    elif v_switch_first < -1:
        return 'normal'
    else:
        return 'uncertain'


def make_domain_aware_x0(vdc, y_obs, domain_type='auto'):
    """
    Generate initial parameters with domain-aware initialization.

    Key insight: For this dataset, most loops are 'reversed' (a2 > 0, a3 < 0).
    Using the correct initial guess dramatically improves convergence.
    """
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    n2 = len(vdc) // 2
    v1, v2 = vdc[:n2], vdc[n2:]
    y1, y2 = y_obs[:n2], y_obs[n2:]

    if domain_type == 'auto':
        domain_type = classify_domain_type(vdc, y_obs)

    # Basic parameters
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
        a2 = 3.5   # Positive coercive voltage
        a3 = -3.5  # Negative coercive voltage
    elif domain_type == 'normal':
        a2 = -3.5
        a3 = 3.5
    else:
        # Uncertain: use gradient-based detection
        dy1 = np.gradient(y1, v1)
        dy2 = np.gradient(y2, v2)
        a2 = float(v1[np.argmax(np.abs(dy1))])
        a3 = float(v2[np.argmax(np.abs(dy2))])

    # Width parameters
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


def fit_domain_aware(vdc, y_obs, n_starts=10, d_schedule=(30, 80, 200, 600, 1000), seed=0, b_max=20.0):
    """
    Fit with domain-aware initialization.

    Similar to original but uses better initial guesses based on domain type.
    """
    rng = np.random.default_rng(seed)
    vdc = np.asarray(vdc, float)
    y_obs = np.asarray(y_obs, float)
    vmin, vmax = float(np.min(vdc)), float(np.max(vdc))

    lb = np.array([-np.inf, -np.inf, vmin, vmin, -np.inf, 1e-3, 1e-3, 1e-3, 1e-3], float)
    ub = np.array([np.inf, np.inf, vmax, vmax, np.inf, b_max, b_max, b_max, b_max], float)

    base, domain_type = make_domain_aware_x0(vdc, y_obs, 'auto')

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

    return best, domain_type


def fit_single_loop_improved(idx, y, vdc_rolled, n_starts):
    """Fit a single loop with domain-aware initialization."""
    p_best, domain_type = fit_domain_aware(
        vdc_rolled, y,
        n_starts=n_starts,
        d_schedule=(30, 80, 200, 600, 1000),
        seed=42 + idx
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
        'domain_type': domain_type
    }


def run_improved_fitting_lite(
    loops_flat,
    vdc_rolled,
    result_path="result_improved.csv",
    n_starts=10,
    n_jobs=-1,
    force_recompute=False
):
    """Run improved fitting with domain-aware initialization."""
    result_path = Path(result_path)

    if result_path.exists() and not force_recompute:
        print(f"Loading cached results from {result_path}")
        return pd.read_csv(result_path, index_col=0)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    print(f"Fitting {len(loops_flat)} loops with domain-aware init (n_starts={n_starts}, n_jobs={n_jobs})...")

    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_single_loop_improved)(idx, loops_flat[idx], vdc_rolled, n_starts)
        for idx in range(len(loops_flat))
    )

    result = pd.DataFrame(rows)
    result = result.sort_values('loop_id').reset_index(drop=True)
    result.to_csv(result_path)
    print(f"Results saved to {result_path}")

    return result


def compare_and_plot(old_path, new_path):
    """Compare old and new results and create visualization."""
    old_df = pd.read_csv(old_path, index_col=0)
    new_df = pd.read_csv(new_path, index_col=0)

    print("\n" + "="*60)
    print("COMPARISON: Original vs Domain-Aware Fitting")
    print("="*60)

    print("\n📊 RMSE Statistics:")
    print(f"   Original - Mean: {old_df['rmse'].mean():.2e}, Median: {old_df['rmse'].median():.2e}")
    print(f"   Improved - Mean: {new_df['rmse'].mean():.2e}, Median: {new_df['rmse'].median():.2e}")

    improvement = (old_df['rmse'].mean() - new_df['rmse'].mean()) / old_df['rmse'].mean() * 100
    print(f"\n   Mean RMSE improvement: {improvement:.1f}%")

    rmse_diff = old_df['rmse'] - new_df['rmse']
    improved_count = (rmse_diff > 0).sum()
    print(f"   Loops improved: {improved_count} ({100*improved_count/len(rmse_diff):.1f}%)")

    # Domain type distribution
    if 'domain_type' in new_df.columns:
        print("\n🔬 Domain Classification:")
        domain_counts = new_df['domain_type'].value_counts()
        for dtype, count in domain_counts.items():
            print(f"   {dtype}: {count} ({100*count/len(new_df):.1f}%)")

    print("="*60)

    # Create comparison plot
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

    # Scatter comparison
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

    # Improvement histogram
    ax3 = axes[1, 0]
    improvement_pct = (old_df['rmse'] - new_df['rmse']) / old_df['rmse'] * 100
    ax3.hist(improvement_pct, bins=50, color='purple', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', label='No improvement')
    ax3.axvline(improvement_pct.mean(), color='orange', linestyle='-',
                label=f'Mean: {improvement_pct.mean():.1f}%')
    ax3.set_xlabel('Improvement (%)')
    ax3.set_ylabel('Count')
    ax3.set_title('Per-Loop RMSE Improvement')
    ax3.legend()

    # Domain type by RMSE
    ax4 = axes[1, 1]
    if 'domain_type' in new_df.columns:
        colors = {'reversed': 'blue', 'normal': 'red', 'uncertain': 'gray'}
        for dtype in new_df['domain_type'].unique():
            mask = new_df['domain_type'] == dtype
            ax4.hist(new_df.loc[mask, 'rmse'], bins=30, alpha=0.5,
                    label=dtype, color=colors.get(dtype, 'black'))
        ax4.set_xlabel('RMSE')
        ax4.set_ylabel('Count')
        ax4.set_title('RMSE by Domain Type')
        ax4.legend()
        ax4.set_yscale('log')

    plt.suptitle('Domain-Aware Fitting Improvement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison figure saved to comparison.png")
    plt.show()

    return old_df, new_df


if __name__ == "__main__":
    # Load data
    print("Loading hysteresis loop data...")
    hysteresis_loops = np.load('hysteresis_loops.npy')
    dc_vec = np.load('dc_vec.npy')

    if hysteresis_loops.ndim == 3:
        loops_flat = hysteresis_loops.reshape(-1, hysteresis_loops.shape[-1])
    else:
        loops_flat = hysteresis_loops

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
    result_improved = run_improved_fitting_lite(
        cleaned_loops,
        vdc_rolled,
        result_path="result_improved.csv",
        n_starts=10,
        n_jobs=-1,
        force_recompute=True
    )

    # Compare with original
    if Path("result_spatial.csv").exists():
        compare_and_plot("result_spatial.csv", "result_improved.csv")

    print("\n✅ Improved fitting complete!")
