#!/usr/bin/env python3
"""
Normalize polysome profiles using mixture model fitting.

Fits skew-normal mixture models to detect peaks, then uses the fitted peak
positions for nonlinear (spline-based) distance alignment.

Usage:
    python normalize_profiles_mixture.py file1.csv file2.csv ... -o output.tsv

Outputs:
    - output.tsv: Normalized profiles with columns:
        identifier, distance, distance.normalized, absorbance, absorbance.normalized, peak
    - output_fits.tsv: Peak fit parameters with columns:
        identifier, peak, amplitude, location, scale, skewness, height, fwhm
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skewnorm
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid
import sys
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# Region boundaries (in mm)
FREE_FRACTION_END = 15.0
RIBOSOME_REGION_START = 15.0
POLYSOME_REGION_END = 60.0
ARTIFACT_START = 85.0

# Peak labels in order
PEAK_LABELS = ['free', '40S', '60S', '80S', '2-some', '3-some', '4-some',
               '5-some', '6-some', '7-some', '8-some']


def parse_raw_file(filepath):
    """Parse a Gradient Profiler raw data file."""
    metadata = {}
    data_lines = []
    in_data = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Data Columns:'):
                in_data = True
                continue

            if in_data:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        distance = float(parts[0].strip())
                        absorbance = float(parts[1].strip())
                        data_lines.append({'distance': distance, 'absorbance': absorbance})
                    except ValueError:
                        continue
            else:
                if ':' in line:
                    key, _, value = line.partition(':')
                    metadata[key.strip()] = value.strip()

    df = pd.DataFrame(data_lines)
    return metadata, df


def skewnorm_pdf(x, amplitude, loc, scale, skewness):
    """Skew-normal PDF scaled by amplitude."""
    return amplitude * skewnorm.pdf(x, skewness, loc=loc, scale=scale)


def mixture_model(x, *params):
    """
    Mixture of skew-normal distributions plus a baseline.

    params: [baseline, amp1, loc1, scale1, skew1, amp2, loc2, scale2, skew2, ...]
    """
    n_peaks = (len(params) - 1) // 4
    result = np.full_like(x, params[0], dtype=float)  # baseline

    for i in range(n_peaks):
        idx = 1 + i * 4
        amp, loc, scale, skew = params[idx:idx+4]
        result += skewnorm_pdf(x, amp, loc, scale, skew)

    return result


def detect_initial_peaks(distance, absorbance, min_dist, max_dist, prominence_threshold=0.005):
    """Detect initial peak positions for mixture model initialization."""
    mask = (distance >= min_dist) & (distance <= max_dist)
    if not np.any(mask):
        return []

    region_indices = np.where(mask)[0]

    if len(absorbance) > 11:
        smoothed = savgol_filter(absorbance, window_length=11, polyorder=3)
    else:
        smoothed = np.array(absorbance)

    region_smoothed = smoothed[mask]
    peak_indices_local, properties = find_peaks(
        region_smoothed,
        prominence=prominence_threshold,
        distance=5
    )

    peaks = []
    for i, local_idx in enumerate(peak_indices_local):
        global_idx = region_indices[local_idx]
        peaks.append({
            'index': global_idx,
            'position': distance[global_idx],
            'height': absorbance[global_idx],
            'prominence': properties['prominences'][i]
        })

    return peaks


def fit_mixture_model(distance, absorbance, initial_peaks, baseline_percentile=5):
    """
    Fit a skew-normal mixture model to the profile.

    Returns:
        list of dicts with fitted peak parameters
    """
    if not initial_peaks:
        return []

    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Estimate baseline
    baseline = np.percentile(absorbance, baseline_percentile)

    # Build initial parameters
    # params: [baseline, amp1, loc1, scale1, skew1, ...]
    p0 = [baseline]
    bounds_low = [0]
    bounds_high = [np.max(absorbance)]

    for peak in initial_peaks:
        # Initial guesses
        amp = peak['height'] - baseline
        loc = peak['position']
        scale = 2.0  # Initial width estimate
        skew = 0.0   # Start symmetric

        p0.extend([amp, loc, scale, skew])

        # Bounds
        bounds_low.extend([0, loc - 5, 0.5, -10])
        bounds_high.extend([amp * 3, loc + 5, 15, 10])

    # Fit the model
    try:
        popt, pcov = curve_fit(
            mixture_model,
            distance,
            absorbance,
            p0=p0,
            bounds=(bounds_low, bounds_high),
            maxfev=10000
        )
    except (RuntimeError, ValueError) as e:
        print(f"    Warning: Fit failed ({e}), using initial estimates", file=sys.stderr)
        popt = p0

    # Extract fitted parameters
    fitted_peaks = []
    n_peaks = (len(popt) - 1) // 4

    for i in range(n_peaks):
        idx = 1 + i * 4
        amp, loc, scale, skew = popt[idx:idx+4]

        # Compute peak properties
        # Mode of skew-normal (approximate)
        delta = skew / np.sqrt(1 + skew**2)
        mode_shift = delta * np.sqrt(2/np.pi) * scale
        mode = loc + mode_shift

        # Height at mode
        height = skewnorm_pdf(mode, amp, loc, scale, skew)

        # FWHM (approximate for skew-normal)
        # For symmetric Gaussian, FWHM = 2.355 * sigma
        # Adjust for skewness
        fwhm = 2.355 * scale * (1 + 0.1 * abs(skew))

        fitted_peaks.append({
            'amplitude': amp,
            'location': loc,
            'scale': scale,
            'skewness': skew,
            'mode': mode,
            'height': height,
            'fwhm': fwhm,
            'original_position': initial_peaks[i]['position'] if i < len(initial_peaks) else loc
        })

    return fitted_peaks, popt[0]  # Return peaks and baseline


def identify_and_label_peaks(fitted_peaks, distance, absorbance):
    """
    Identify and label peaks as free, 60S, 80S, polysomes, etc.

    Returns:
        dict mapping peak labels to fitted peak dicts
    """
    if not fitted_peaks:
        return {}

    # Sort by position
    sorted_peaks = sorted(fitted_peaks, key=lambda p: p['mode'])

    labeled = {}

    # Find peaks in each region
    free_peaks = [p for p in sorted_peaks if p['mode'] < FREE_FRACTION_END]
    ribosome_peaks = [p for p in sorted_peaks if FREE_FRACTION_END <= p['mode'] < POLYSOME_REGION_END]

    # Label free fraction
    if free_peaks:
        labeled['free'] = max(free_peaks, key=lambda p: p['amplitude'])

    if not ribosome_peaks:
        return labeled

    # Find 80S: most prominent peak in 20-40mm range
    peaks_80s_region = [p for p in ribosome_peaks if 20 <= p['mode'] <= 40]
    if peaks_80s_region:
        peak_80s = max(peaks_80s_region, key=lambda p: p['amplitude'])
        labeled['80S'] = peak_80s

        # Find 60S: peak just before 80S
        peaks_before_80s = [p for p in ribosome_peaks
                           if p['mode'] < peak_80s['mode']
                           and p['mode'] > peak_80s['mode'] - 15]
        if peaks_before_80s:
            labeled['60S'] = max(peaks_before_80s, key=lambda p: p['mode'])

        # Find 40S: peak before 60S
        if '60S' in labeled:
            peaks_before_60s = [p for p in ribosome_peaks
                               if p['mode'] < labeled['60S']['mode']
                               and p['mode'] > labeled['60S']['mode'] - 10]
            if peaks_before_60s:
                labeled['40S'] = max(peaks_before_60s, key=lambda p: p['mode'])

        # Polysomes: peaks after 80S
        peaks_after_80s = [p for p in ribosome_peaks if p['mode'] > peak_80s['mode']]
        peaks_after_80s = sorted(peaks_after_80s, key=lambda p: p['mode'])

        polysome_labels = ['2-some', '3-some', '4-some', '5-some', '6-some', '7-some', '8-some']
        for i, peak in enumerate(peaks_after_80s):
            if i < len(polysome_labels):
                labeled[polysome_labels[i]] = peak

    return labeled


def build_nonlinear_transform(source_peaks, ref_peaks):
    """
    Build a nonlinear distance transform using spline interpolation.

    Uses all matched peak positions to create a monotonic spline mapping.

    Returns:
        callable: function that maps source distance to reference distance
    """
    # Find common peaks
    common_labels = set(source_peaks.keys()) & set(ref_peaks.keys())

    # Prioritize certain peaks for alignment
    priority = ['60S', '80S', '2-some', '3-some', '4-some', '40S', '5-some', '6-some']
    alignment_labels = [l for l in priority if l in common_labels]

    if len(alignment_labels) < 2:
        # Fall back to identity transform
        return lambda x: x

    # Build control points
    src_positions = [source_peaks[l]['mode'] for l in alignment_labels]
    ref_positions = [ref_peaks[l]['mode'] for l in alignment_labels]

    # Add boundary points to extend the transform
    # Use linear extrapolation at boundaries
    min_src = min(src_positions)
    max_src = max(src_positions)
    min_ref = min(ref_positions)
    max_ref = max(ref_positions)

    # Compute slope at boundaries for extrapolation
    if len(src_positions) >= 2:
        # Sort by source position
        sorted_pairs = sorted(zip(src_positions, ref_positions))
        src_sorted = [p[0] for p in sorted_pairs]
        ref_sorted = [p[1] for p in sorted_pairs]

        # Add boundary points
        slope_left = (ref_sorted[1] - ref_sorted[0]) / (src_sorted[1] - src_sorted[0])
        slope_right = (ref_sorted[-1] - ref_sorted[-2]) / (src_sorted[-1] - src_sorted[-2])

        # Extend to 0 and beyond max
        src_extended = [0] + src_sorted + [100]
        ref_extended = [ref_sorted[0] - slope_left * (src_sorted[0] - 0)] + ref_sorted + \
                       [ref_sorted[-1] + slope_right * (100 - src_sorted[-1])]

        # Create monotonic interpolator (PCHIP preserves monotonicity)
        try:
            interpolator = PchipInterpolator(src_extended, ref_extended)
            return interpolator
        except ValueError:
            # Fall back to linear if PCHIP fails
            pass

    # Simple linear fallback using first two points
    if len(alignment_labels) >= 2:
        src1, src2 = src_positions[0], src_positions[1]
        ref1, ref2 = ref_positions[0], ref_positions[1]
        scale = (ref2 - ref1) / (src2 - src1) if src2 != src1 else 1.0
        offset = ref1 - src1 * scale
        return lambda x: x * scale + offset

    return lambda x: x


def compute_total_area(distance, absorbance, exclude_artifacts=True):
    """Compute total area under the curve."""
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    if exclude_artifacts:
        mask = distance < ARTIFACT_START
        distance = distance[mask]
        absorbance = absorbance[mask]

    return trapezoid(absorbance, distance)


def process_profile(distance, absorbance, ref_peaks, ref_area, identifier):
    """
    Process a single profile: fit mixture model, align, normalize.

    Returns:
        tuple: (result_df, fits_df)
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Detect initial peaks for mixture model
    initial_peaks = []

    # Free fraction
    free_peaks = detect_initial_peaks(distance, absorbance, 0, FREE_FRACTION_END, 0.01)
    initial_peaks.extend(free_peaks)

    # Ribosome/polysome region
    ribosome_peaks = detect_initial_peaks(distance, absorbance,
                                          RIBOSOME_REGION_START, POLYSOME_REGION_END, 0.005)
    initial_peaks.extend(ribosome_peaks)

    # Sort by position
    initial_peaks = sorted(initial_peaks, key=lambda p: p['position'])

    # Fit mixture model
    if initial_peaks:
        fitted_peaks, baseline = fit_mixture_model(distance, absorbance, initial_peaks)
    else:
        fitted_peaks = []
        baseline = np.percentile(absorbance, 5)

    # Label peaks
    labeled_peaks = identify_and_label_peaks(fitted_peaks, distance, absorbance)

    # Build nonlinear transform
    transform = build_nonlinear_transform(labeled_peaks, ref_peaks)

    # Apply transform
    norm_distance = transform(distance)

    # Normalize amplitude by area
    area = compute_total_area(distance, absorbance)
    y_scale = ref_area / area if area > 0 else 1.0
    norm_absorbance = absorbance * y_scale

    # Create peak label column
    peak_labels = [''] * len(distance)
    for label, peak in labeled_peaks.items():
        # Find closest point to peak mode
        idx = np.argmin(np.abs(distance - peak['mode']))
        peak_labels[idx] = label

    # Build result dataframe
    result_df = pd.DataFrame({
        'identifier': identifier,
        'distance': distance,
        'distance.normalized': norm_distance,
        'absorbance': absorbance,
        'absorbance.normalized': norm_absorbance,
        'peak': peak_labels
    })

    # Build fits dataframe
    fits_data = []
    for label, peak in labeled_peaks.items():
        fits_data.append({
            'identifier': identifier,
            'peak': label,
            'amplitude': peak['amplitude'],
            'location': peak['location'],
            'scale': peak['scale'],
            'skewness': peak['skewness'],
            'mode': peak['mode'],
            'height': peak['height'],
            'fwhm': peak['fwhm']
        })

    fits_df = pd.DataFrame(fits_data)

    return result_df, fits_df, labeled_peaks, y_scale


def normalize_profiles(file_list):
    """
    Normalize multiple polysome profiles using mixture model fitting.

    Returns:
        tuple: (profiles_df, fits_df)
    """
    # Load all profiles
    profiles = []
    for filepath in file_list:
        path = Path(filepath)
        metadata, df = parse_raw_file(filepath)
        profiles.append({
            'filepath': filepath,
            'identifier': path.name,
            'metadata': metadata,
            'data': df
        })

    if len(profiles) < 2:
        raise ValueError("Need at least 2 profiles for normalization")

    # Process reference profile first
    ref_data = profiles[0]['data']
    ref_distance = ref_data['distance'].values
    ref_absorbance = ref_data['absorbance'].values
    ref_identifier = profiles[0]['identifier']

    # Fit reference profile
    print(f"Fitting reference profile: {ref_identifier}", file=sys.stderr)

    initial_peaks = []
    free_peaks = detect_initial_peaks(ref_distance, ref_absorbance, 0, FREE_FRACTION_END, 0.01)
    initial_peaks.extend(free_peaks)
    ribosome_peaks = detect_initial_peaks(ref_distance, ref_absorbance,
                                          RIBOSOME_REGION_START, POLYSOME_REGION_END, 0.005)
    initial_peaks.extend(ribosome_peaks)
    initial_peaks = sorted(initial_peaks, key=lambda p: p['position'])

    if initial_peaks:
        ref_fitted_peaks, ref_baseline = fit_mixture_model(ref_distance, ref_absorbance, initial_peaks)
    else:
        raise ValueError(f"No peaks detected in reference file: {ref_identifier}")

    ref_peaks = identify_and_label_peaks(ref_fitted_peaks, ref_distance, ref_absorbance)
    ref_area = compute_total_area(ref_distance, ref_absorbance)

    if '80S' not in ref_peaks:
        raise ValueError(f"Could not identify 80S peak in reference file: {ref_identifier}")

    print(f"  Detected peaks: {', '.join(sorted(ref_peaks.keys(), key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99))}", file=sys.stderr)
    print(f"  80S mode: {ref_peaks['80S']['mode']:.2f} mm", file=sys.stderr)
    print(f"  Total area: {ref_area:.4f}", file=sys.stderr)

    # Process all profiles
    all_results = []
    all_fits = []

    for profile in profiles:
        df = profile['data']
        identifier = profile['identifier']

        print(f"Processing: {identifier}", file=sys.stderr)

        result_df, fits_df, labeled_peaks, y_scale = process_profile(
            df['distance'].values,
            df['absorbance'].values,
            ref_peaks,
            ref_area,
            identifier
        )

        peak_labels = sorted(labeled_peaks.keys(),
                            key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99)
        print(f"  Detected peaks: {', '.join(peak_labels)}", file=sys.stderr)
        print(f"  Y scale: {y_scale:.4f}", file=sys.stderr)

        all_results.append(result_df)
        all_fits.append(fits_df)

    return pd.concat(all_results, ignore_index=True), pd.concat(all_fits, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize polysome profiles using mixture model fitting.'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Input polysome profile files (Gradient Profiler CSV format)'
    )
    parser.add_argument(
        '-o', '--output',
        default='normalized_profiles.tsv',
        help='Output file (tab-delimited, default: normalized_profiles.tsv)'
    )

    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("At least 2 input files are required")

    for f in args.files:
        if not Path(f).exists():
            parser.error(f"File not found: {f}")

    # Process profiles
    profiles_df, fits_df = normalize_profiles(args.files)

    # Write outputs
    profiles_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nWrote {len(profiles_df)} rows to {args.output}", file=sys.stderr)

    # Write fits file
    fits_output = args.output.replace('.tsv', '_fits.tsv')
    fits_df.to_csv(fits_output, sep='\t', index=False)
    print(f"Wrote {len(fits_df)} peak fits to {fits_output}", file=sys.stderr)


if __name__ == '__main__':
    main()
