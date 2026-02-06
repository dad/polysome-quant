#!/usr/bin/env python3
"""
Normalize polysome profiles using local peak fitting.

Detects peaks as local maxima, fits each peak individually with a skew-normal,
then uses fitted peak positions for nonlinear (spline) distance alignment.

Usage:
    python normalize_profiles_v2.py file1.csv file2.csv ... -o output.tsv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, argrelmax
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
# from scipy.interpolate import PchipInterpolator  # Not needed for affine transform
from scipy.integrate import trapezoid
import sys
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# Region boundaries (in mm)
FREE_FRACTION_END = 12.0       # Free fraction ends here
RIBOSOME_REGION_START = 12.0   # 40S, 60S, 80S start here
POLYSOME_REGION_END = 60.0
ARTIFACT_START = 85.0

# Peak detection parameters
SMOOTHING_WINDOW = 11
LOCAL_MAX_ORDER = 5  # Compare to N neighbors on each side
FIT_WINDOW_MM = 4.0  # Fit window half-width in mm

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


def find_local_maxima(distance, absorbance, min_dist, max_dist, order=LOCAL_MAX_ORDER):
    """
    Find local maxima using simple comparison to neighbors.

    Args:
        order: number of neighbors on each side to compare

    Returns:
        list of dicts with 'index', 'position', 'height' for each peak
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Create mask for region
    mask = (distance >= min_dist) & (distance <= max_dist)
    if not np.any(mask):
        return []

    region_indices = np.where(mask)[0]
    region_abs = absorbance[mask]

    # Smooth before finding maxima
    if len(region_abs) > SMOOTHING_WINDOW:
        smoothed = savgol_filter(region_abs, window_length=SMOOTHING_WINDOW, polyorder=3)
    else:
        smoothed = region_abs

    # Find local maxima
    local_max_local = argrelmax(smoothed, order=order)[0]

    peaks = []
    for local_idx in local_max_local:
        global_idx = region_indices[local_idx]
        peaks.append({
            'index': global_idx,
            'position': distance[global_idx],
            'height': absorbance[global_idx]
        })

    return peaks


def skewnorm_with_baseline(x, amplitude, loc, scale, skewness, baseline):
    """Skew-normal PDF with baseline offset."""
    return baseline + amplitude * skewnorm.pdf(x, skewness, loc=loc, scale=scale)


def fit_single_peak(distance, absorbance, peak_position, window_mm=FIT_WINDOW_MM):
    """
    Fit a single skew-normal to a peak in a local window.

    Returns:
        dict with fitted parameters, or None if fit fails
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Extract window around peak
    mask = (distance >= peak_position - window_mm) & (distance <= peak_position + window_mm)
    if np.sum(mask) < 10:
        return None

    x = distance[mask]
    y = absorbance[mask]

    # Initial estimates
    baseline = np.min(y)
    amplitude = (np.max(y) - baseline) * 2  # Approximate for PDF scaling
    loc = peak_position
    scale = 1.5
    skewness = 0.0

    # Bounds
    bounds_low = [0, peak_position - window_mm, 0.3, -10, 0]
    bounds_high = [amplitude * 5, peak_position + window_mm, window_mm, 10, np.max(y)]

    try:
        popt, _ = curve_fit(
            skewnorm_with_baseline,
            x, y,
            p0=[amplitude, loc, scale, skewness, baseline],
            bounds=(bounds_low, bounds_high),
            maxfev=2000
        )
        amplitude, loc, scale, skewness, baseline = popt

        # Compute mode
        delta = skewness / np.sqrt(1 + skewness**2)
        mode_shift = delta * np.sqrt(2/np.pi) * scale
        mode = loc + mode_shift

        # Height at mode
        height = skewnorm_with_baseline(mode, amplitude, loc, scale, skewness, baseline)

        # FWHM approximation
        fwhm = 2.355 * scale * (1 + 0.1 * abs(skewness))

        return {
            'amplitude': amplitude,
            'location': loc,
            'scale': scale,
            'skewness': skewness,
            'baseline': baseline,
            'mode': mode,
            'height': height,
            'fwhm': fwhm
        }
    except (RuntimeError, ValueError):
        # Fit failed, return simple estimate
        return {
            'amplitude': np.max(y) - np.min(y),
            'location': peak_position,
            'scale': 1.5,
            'skewness': 0.0,
            'baseline': np.min(y),
            'mode': peak_position,
            'height': np.max(y),
            'fwhm': 3.0
        }


def identify_and_label_peaks(peaks, distance, min_separation=3.0):
    """
    Identify and label peaks as free, 60S, 80S, polysomes, etc.

    Args:
        min_separation: minimum distance (mm) between distinct peaks
    """
    if not peaks:
        return {}

    max_distance = np.max(distance)

    # First, deduplicate peaks that are too close together
    sorted_peaks = sorted(peaks, key=lambda p: p['mode'])
    deduplicated = []
    for peak in sorted_peaks:
        if not deduplicated:
            deduplicated.append(peak)
        else:
            # Check if this peak is too close to the previous one
            if peak['mode'] - deduplicated[-1]['mode'] < min_separation:
                # Keep the taller peak
                if peak['height'] > deduplicated[-1]['height']:
                    deduplicated[-1] = peak
            else:
                deduplicated.append(peak)

    sorted_peaks = deduplicated

    labeled = {}

    # Free fraction: largest peak before 15mm
    free_peaks = [p for p in sorted_peaks if p['mode'] < FREE_FRACTION_END]
    if free_peaks:
        labeled['free'] = max(free_peaks, key=lambda p: p['height'])

    # Ribosome region peaks
    ribosome_peaks = [p for p in sorted_peaks
                      if RIBOSOME_REGION_START <= p['mode'] < POLYSOME_REGION_END]

    if not ribosome_peaks:
        return labeled

    # Find 80S: tallest peak in 20-40mm range
    peaks_80s_region = [p for p in ribosome_peaks if 20 <= p['mode'] <= 40]
    if peaks_80s_region:
        peak_80s = max(peaks_80s_region, key=lambda p: p['height'])
        labeled['80S'] = peak_80s

        # 60S: peak before 80S
        peaks_before_80s = [p for p in ribosome_peaks
                           if p['mode'] < peak_80s['mode']
                           and p['mode'] > peak_80s['mode'] - 12]
        if peaks_before_80s:
            labeled['60S'] = max(peaks_before_80s, key=lambda p: p['mode'])

        # 40S: peak before 60S
        if '60S' in labeled:
            peaks_before_60s = [p for p in ribosome_peaks
                               if p['mode'] < labeled['60S']['mode']
                               and p['mode'] > labeled['60S']['mode'] - 8]
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


def build_affine_transform(source_peaks, ref_peaks):
    """
    Build an affine transform to align source peaks to reference peaks.

    Anchors on free and 80S peaks - scales so that the distance between
    free and 80S matches the reference, then shifts so both peaks align.
    Transform: x_new = x * scale + offset

    Returns:
        tuple: (transform_function, scale, offset)
    """
    # Check for free and 80S - these are the anchors
    has_free = 'free' in source_peaks and 'free' in ref_peaks
    has_80S = '80S' in source_peaks and '80S' in ref_peaks

    if has_free and has_80S:
        # Affine transform to align both free and 80S
        src_free = source_peaks['free']['mode']
        src_80s = source_peaks['80S']['mode']
        ref_free = ref_peaks['free']['mode']
        ref_80s = ref_peaks['80S']['mode']

        src_distance = src_80s - src_free
        ref_distance = ref_80s - ref_free

        if abs(src_distance) > 1.0:  # Sanity check
            scale = ref_distance / src_distance
        else:
            scale = 1.0

        # Bound scale
        scale = max(0.7, min(1.4, scale))

        # Compute offset to align free peak (80S will also align due to matching scale)
        offset = ref_free - src_free * scale

    elif has_80S:
        # Fallback: shift to align 80S only
        src_80s = source_peaks['80S']['mode']
        ref_80s = ref_peaks['80S']['mode']
        scale = 1.0
        offset = ref_80s - src_80s

    elif has_free:
        # Fallback: shift to align free only
        src_free = source_peaks['free']['mode']
        ref_free = ref_peaks['free']['mode']
        scale = 1.0
        offset = ref_free - src_free

    else:
        # No anchor peaks - identity transform
        return lambda x: np.asarray(x, dtype=float), 1.0, 0.0

    def affine_transform(x):
        return np.asarray(x, dtype=float) * scale + offset

    return affine_transform, scale, offset


def compute_total_area(distance, absorbance):
    """Compute total area under the curve, excluding artifacts."""
    mask = distance < ARTIFACT_START
    return trapezoid(absorbance[mask], distance[mask])


def process_profile(distance, absorbance, ref_peaks, ref_area, identifier):
    """
    Process a single profile: detect peaks, fit each, align, normalize.
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Find local maxima
    all_peaks = []

    # Free fraction
    free_maxima = find_local_maxima(distance, absorbance, 0, FREE_FRACTION_END, order=10)
    all_peaks.extend(free_maxima)

    # Ribosome/polysome region
    ribosome_maxima = find_local_maxima(distance, absorbance,
                                         RIBOSOME_REGION_START, POLYSOME_REGION_END, order=5)
    all_peaks.extend(ribosome_maxima)

    # Fit each peak individually
    fitted_peaks = []
    for peak in all_peaks:
        fit_result = fit_single_peak(distance, absorbance, peak['position'])
        if fit_result:
            fit_result['original_index'] = peak['index']
            fitted_peaks.append(fit_result)

    # Label peaks
    labeled_peaks = identify_and_label_peaks(fitted_peaks, distance)

    # Build affine transform using all labeled peaks
    transform, x_scale, x_offset = build_affine_transform(labeled_peaks, ref_peaks)

    # Apply transform
    norm_distance = transform(distance)

    # Normalize amplitude by area
    area = compute_total_area(distance, absorbance)
    y_scale = ref_area / area if area > 0 else 1.0
    norm_absorbance = absorbance * y_scale

    # Create peak label column
    peak_labels = [''] * len(distance)
    for label, peak in labeled_peaks.items():
        # Find closest data point to peak mode
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
            'baseline': peak['baseline'],
            'mode': peak['mode'],
            'height': peak['height'],
            'fwhm': peak['fwhm']
        })

    fits_df = pd.DataFrame(fits_data)

    return result_df, fits_df, labeled_peaks, x_scale, x_offset, y_scale


def normalize_profiles(file_list):
    """Normalize multiple polysome profiles."""
    # Load all profiles
    profiles = []
    for filepath in file_list:
        path = Path(filepath)
        metadata, df = parse_raw_file(filepath)
        profiles.append({
            'filepath': filepath,
            'identifier': path.name,
            'data': df
        })

    if len(profiles) < 2:
        raise ValueError("Need at least 2 profiles for normalization")

    # Process reference profile
    ref_data = profiles[0]['data']
    ref_distance = ref_data['distance'].values
    ref_absorbance = ref_data['absorbance'].values
    ref_identifier = profiles[0]['identifier']

    print(f"Processing reference: {ref_identifier}", file=sys.stderr)

    # Find and fit reference peaks
    ref_peaks_raw = []
    free_maxima = find_local_maxima(ref_distance, ref_absorbance, 0, FREE_FRACTION_END, order=10)
    ref_peaks_raw.extend(free_maxima)
    ribosome_maxima = find_local_maxima(ref_distance, ref_absorbance,
                                         RIBOSOME_REGION_START, POLYSOME_REGION_END, order=5)
    ref_peaks_raw.extend(ribosome_maxima)

    ref_fitted = []
    for peak in ref_peaks_raw:
        fit_result = fit_single_peak(ref_distance, ref_absorbance, peak['position'])
        if fit_result:
            ref_fitted.append(fit_result)

    ref_peaks = identify_and_label_peaks(ref_fitted, ref_distance)
    ref_area = compute_total_area(ref_distance, ref_absorbance)

    if '80S' not in ref_peaks:
        raise ValueError(f"Could not identify 80S peak in reference: {ref_identifier}")

    peak_list = sorted(ref_peaks.keys(),
                       key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99)
    print(f"  Detected peaks: {', '.join(peak_list)}", file=sys.stderr)
    print(f"  80S mode: {ref_peaks['80S']['mode']:.2f} mm", file=sys.stderr)

    # Process all profiles
    all_results = []
    all_fits = []

    for profile in profiles:
        df = profile['data']
        identifier = profile['identifier']

        result_df, fits_df, labeled_peaks, x_scale, x_offset, y_scale = process_profile(
            df['distance'].values,
            df['absorbance'].values,
            ref_peaks,
            ref_area,
            identifier
        )

        peak_list = sorted(labeled_peaks.keys(),
                          key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99)
        print(f"Processed: {identifier} - {len(labeled_peaks)} peaks, X: {x_scale:.3f}x + {x_offset:.2f}, Y: {y_scale:.4f}", file=sys.stderr)

        all_results.append(result_df)
        all_fits.append(fits_df)

    return pd.concat(all_results, ignore_index=True), pd.concat(all_fits, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize polysome profiles using local peak fitting.'
    )
    parser.add_argument('files', nargs='+', help='Input profile files')
    parser.add_argument('-o', '--output', default='normalized_profiles.tsv',
                        help='Output file (default: normalized_profiles.tsv)')

    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("At least 2 input files required")

    for f in args.files:
        if not Path(f).exists():
            parser.error(f"File not found: {f}")

    profiles_df, fits_df = normalize_profiles(args.files)

    profiles_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nWrote {len(profiles_df)} rows to {args.output}", file=sys.stderr)

    fits_output = args.output.replace('.tsv', '_fits.tsv')
    fits_df.to_csv(fits_output, sep='\t', index=False)
    print(f"Wrote {len(fits_df)} peak fits to {fits_output}", file=sys.stderr)


if __name__ == '__main__':
    main()
