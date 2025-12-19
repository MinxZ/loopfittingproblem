# Hysteresis Loop Fitting Problem

## Task

The task is to **fit hysteresis loop data** from piezoresponse force spectroscopy (PFS) measurements on a ferroelectric thin film. The goal is to extract meaningful physical parameters from 2,500 (50x50 grid) hysteresis loops by fitting them to a 9-parameter mathematical model.

## Background

### Dataset
- **Source**: Piezoresponse Force Microscopy (PFM) / Atomic Force Microscopy (AFM)
- **Sample**: PbTiO3 (Lead Titanate) thin film - a ferroelectric material
- **Dimensions**: 50x50 spatial grid = 2,500 hysteresis loops
- **Scale**: 2μm x 2μm scan area
- **Author**: R. Vasudevan (CNMS/ORNL)

### What is a Hysteresis Loop?
Hysteresis loops in ferroelectric materials show how the material's polarization responds to an applied electric field (voltage). The "loop" shape indicates:
- **Coercive fields** (where switching occurs)
- **Remnant polarization** (polarization at zero field)
- **Saturation behavior**

### The 9-Parameter Model
The fit function uses error functions (erf) to model the switching behavior:
```
Parameters: [a0, a1, a2, a3, a4, b0, b1, b2, b3]
- a0: vertical offset
- a1: amplitude scaling
- a2, a3: switching voltage centers (coercive voltages)
- a4: linear slope (drift/leakage)
- b0, b1, b2, b3: width parameters controlling transition sharpness
```

## Proposed Solution

### 1. Preprocessing - Spike Removal
**Problem**: Raw loops contain measurement spikes/artifacts.

**Solution**: `clean_by_large_diff()` function
- Detects anomalies via large |Δy| jumps using MAD-based threshold
- Expands detection to neighboring points
- Linearly interpolates flagged points

### 2. Data Alignment
**Problem**: Voltage vector needs proper ordering for the split-branch model.

**Solution**: Roll data so minimum voltage (-8.5V) starts at index 0, creating:
- First half: increasing voltage branch
- Second half: decreasing voltage branch

### 3. Clustering with PCA + HDBSCAN
**Problem**: Some loops are too corrupted to fit or have different characteristics.

**Solution**:
- **PCA**: Reduce dimensionality (keeping 90% variance)
- **HDBSCAN**: Density-based clustering to identify:
  - **Group 0**: Complete loops (horizontally flipped)
  - **Group 5**: Complete hysteresis loops (majority)
  - **Groups 1-4**: Loops requiring further cleaning
  - **Noise (-1)**: Outliers too different to cluster

### 4. Robust Fitting with Multi-Start Optimization
**Problem**: Optimization is highly sensitive to initialization; single-start often finds local minima.

**Solution**: `fit_optionA_strong_coef()`
- **Multi-start** (30-40 random initializations with jitter)
- **Continuation schedule**: Gradually increase `d` parameter (30→80→200→600→1000)
  - Low `d`: Smoother landscape, easier to find global region
  - High `d`: Sharp transitions matching true physics
- **Robust loss**: `soft_l1` loss function reduces outlier sensitivity
- **Heuristic initialization**: Uses gradient-based switch detection, tail slopes, and width estimates

## What Else Can Be Done

### Immediate Improvements

1. **Better Spike Detection**
   - Use Savitzky-Golay filtering instead of simple interpolation
   - Wavelet denoising for multi-scale artifact removal
   - Median filtering as a preprocessing step

2. **Adaptive Clustering**
   - Try different `min_cluster_size` values
   - Use UMAP instead of PCA for better nonlinear structure preservation
   - Implement iterative cleaning: cluster → clean → re-cluster

3. **Fitting Enhancements**
   - **Bayesian optimization** for hyperparameter tuning (n_starts, d_schedule)
   - **Differential evolution** or **basin-hopping** as alternative global optimizers
   - **Constrained optimization** with physical priors (e.g., a2 < a3 for certain polarization states)

### Advanced Extensions

4. **Uncertainty Quantification**
   - Bootstrap resampling for parameter confidence intervals
   - MCMC (Markov Chain Monte Carlo) for full posterior distributions
   - Report fit quality metrics beyond RMSE (R², AIC, BIC)

5. **Spatial Analysis**
   - Map fitted parameters (a2, a3 coercive voltages) back to 50x50 spatial grid
   - Identify domain walls, grain boundaries from parameter discontinuities
   - Compute spatial correlation functions

6. **Alternative Models**
   - Try simpler models (fewer parameters) for noisy loops
   - Implement model selection (AIC/BIC) to choose between models
   - Piecewise linear models for loops that don't match the erf-based form

7. **Machine Learning Approaches**
   - Train a neural network to predict initial parameters from loop shape
   - Use autoencoders for anomaly detection (replacing HDBSCAN)
   - Physics-informed neural networks (PINNs) for simultaneous fitting

8. **Automation & Scalability**
   - Parallel fitting using `joblib` or `multiprocessing`
   - GPU acceleration with JAX or PyTorch for large datasets
   - Adaptive stopping: fewer starts for "easy" loops

9. **Quality Control**
   - Automatic flagging of poor fits (RMSE threshold)
   - Visual inspection tool for flagged loops
   - Re-fitting pipeline for failed loops with different strategies

### Physical Interpretation

10. **Extract Physical Properties**
    - Map coercive voltages to coercive fields (V/thickness)
    - Calculate switching asymmetry: |a2| vs |a3|
    - Correlate loop shape parameters with material properties
