#!/bin/bash
# Hysteresis Loop Fitting Pipeline
# ================================
# This script runs the full analysis pipeline for PFM hysteresis loop fitting.

set -e  # Exit on error

echo "=========================================="
echo "Hysteresis Loop Fitting Pipeline"
echo "=========================================="

# ------------------------------------------------------------------------------
# Step 1: Baseline Fitting
# ------------------------------------------------------------------------------
# Original fitting algorithm with:
# - MAD-based spike removal
# - Alignment via cross-correlation
# - HDBSCAN clustering for initialization
# - Multi-start optimization with d-continuation
#
# Output: Fitted parameters for all 2500 loops
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1] Running baseline fitting..."
python loopfittingproblem.py

# ------------------------------------------------------------------------------
# Step 2: Spatial Analysis & Visualization
# ------------------------------------------------------------------------------
# Parallel fitting of all loops with analysis:
# - Spatial parameter heatmaps (50x50 grid)
# - Domain classification map
# - Parameter histograms
# - Correlation analysis
# - Pairwise scatter plots
#
# Output: result_spatial.csv, spatial_analysis.png, domain_map.png,
#         histograms.png, correlations.png, pairplot.png
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2] Running spatial analysis..."
python spatial_analysis.py

# ------------------------------------------------------------------------------
# Step 3: Improved Preprocessing
# ------------------------------------------------------------------------------
# Enhanced spike/outlier detection using combined approach:
# - Pass 1: Median filter (catches extreme isolated spikes)
# - Pass 2: Savitzky-Golay filter (catches subtle deviations)
# - Pass 3: Gradient check (catches remaining jumps)
#
# Result: 43.9% improvement in mean RMSE over baseline
#
# Output: result_combined_preproc.csv, cleaned_combined.npy,
#         preprocessing_improvement.png
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3] Running improved preprocessing..."
python improved_preprocessing.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - result_spatial.csv          : Baseline fitting results"
echo "  - result_combined_preproc.csv : Improved preprocessing results"
echo ""
echo "Visualizations:"
echo "  - spatial_analysis.png        : Parameter heatmaps"
echo "  - domain_map.png              : Domain classification"
echo "  - histograms.png              : Parameter distributions"
echo "  - correlations.png            : Parameter correlations"
echo "  - pairplot.png                : Pairwise relationships"
echo "  - preprocessing_improvement.png : Preprocessing comparison"
echo ""

# ------------------------------------------------------------------------------
# Optional: Additional experiments (not run by default)
# ------------------------------------------------------------------------------
# - improved_fitting.py      : Full adaptive strategy (memory intensive)
# - improved_fitting_lite.py : Domain-aware initialization only
# ------------------------------------------------------------------------------
