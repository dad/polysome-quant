# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative analysis of polysome profiles in yeast (Saccharomyces cerevisiae). Polysome profiling measures RNA distribution in lysed cells via ultracentrifugation through sucrose gradients. Since ribosomes contain most cellular RNA, the signal primarily reflects ribosome distribution. Data is 2D: distance (mm) along the gradient and absorbance (254/260 nm, arbitrary units).

## Data

- `data/merged_profiling_dataset_02052026.csv` - Processed combined dataset (125k+ rows)
- `data/raw_data/` - 39 raw CSV files from Teledyne Gradient Profiler system

**Raw data naming pattern:** `[number]-[date]_[strain]_[temp].csv`

**Strains:** BY4742, BY4741 (wild-type), MV_A, MV_I, yHG005, yHG008 (mutants)

**Temperatures:** 30C, 40C, 42C, 46C

## Analysis Goals

### Primary quantification
- Proportion of RNA in monosomes/free RNA vs. polysomes (2+ ribosomes)

### Normalization challenge
Profiles need consistent normalization due to:
- Variable sample loading amounts
- Gradient inconsistencies (too shallow/deep/uneven)

**Profile structure (distance in mm):**
- 0-15mm: Free fraction (variable, often dominant peak)
- ~20-25mm: 60S subunit
- ~25-35mm: 80S monosome
- 35-60mm: Polysomes (2-some, 3-some, etc.)
- >60mm: Poor peak resolution
- >85mm: Artifacts (ignore)

**Landmark peaks for alignment:** 60S, 80S, disome, trisome. Polysome peaks should be evenly spaced in a well-formed linear gradient. Do NOT normalize by 80S peak height - this reflects translational state (the biological signal of interest).

**Normalization transforms to explore:** affine transformations (x/y scaling, x/y offsets, tilt). Use replicate data to distinguish natural variation from technical artifacts.

### Peak analysis
- Peaks are skewed; fit skew-normal mixture models to deconvolve overlapping peaks
- Extract peak parameters: amplitude, location, scale, skewness, mode, height, FWHM
- Understand how peak resolution affects quantification accuracy

### Data pipeline
- Parse raw CSV files, extracting metadata from headers (date/time) and filenames (strain, temp, replicate)
- Output standardized R-readable format with columns for experiment type, strain, date, etc.

## Development

**Python** for data processing and analysis code. **R** for visualization.

New code goes in `scripts/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Scripts

### normalize_profiles_v2.py (recommended)

Normalize profiles using local peak detection and individual skew-normal fits. Handles gradient nonlinearity with spline-based alignment.

```bash
source venv/bin/activate
python scripts/normalize_profiles_v2.py data/raw_data/*.csv -o data/normalized_profiles.tsv
```

**Outputs:**
- `normalized_profiles.tsv`: Normalized profiles with columns: identifier, distance, distance.normalized, absorbance, absorbance.normalized, peak
- `normalized_profiles_fits.tsv`: Peak fit parameters with columns: identifier, peak, amplitude, location, scale, skewness, mode, height, fwhm

### normalize_profiles.py

Simpler normalization using affine (linear) x-axis alignment. Less accurate for nonlinear gradients.

### normalization_diagnostic.qmd

Quarto document for visualizing normalization results. Renders to HTML with raw/normalized profile plots and fit diagnostics.

```bash
quarto render scripts/normalization_diagnostic.qmd
```
