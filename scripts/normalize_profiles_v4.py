#!/usr/bin/env python3
"""
Normalize polysome profiles using platosome-guided detection.

This script builds on v3's multi-layered approach but uses the Platosome
(Platonic polysome model) as a prior for peak identification and quantification.

KEY FEATURES:
- Uses platosome to define expected peak positions relative to ruler
- Searches for peaks within platosome-defined bounds
- Infers missing peaks (like 40S) using platosome priors
- Reports detection confidence (detected vs inferred)

DETECTION LAYERS:
1. Primary landmarks (free, 80S) - detect without priors
2. Compute ruler and load platosome priors
3. Guided detection - search for peaks in platosome-defined regions
4. Inference - place missing peaks at expected positions with reduced confidence
5. Validation - check consistency with platosome expectations

Usage:
    python normalize_profiles_v4.py file1.csv file2.csv ... -o output.tsv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, argrelmax, argrelmin
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import sys
import warnings

from platosome import Platosome, PEAK_ORDER

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

FREE_FRACTION_END = 12.0
MONOSOME_REGION_START = 20.0
MONOSOME_REGION_END = 40.0
POLYSOME_REGION_END = 65.0
ARTIFACT_START = 85.0

SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3
FIT_WINDOW_MM = 4.0

# Confidence levels for peak detection
CONFIDENCE_DETECTED = 1.0      # Peak clearly detected
CONFIDENCE_WEAK = 0.7          # Peak detected but weak
CONFIDENCE_INFERRED = 0.4      # Peak inferred from platosome prior


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def smooth_profile(absorbance, window=SMOOTHING_WINDOW):
    """Apply Savitzky-Golay smoothing to the profile."""
    if len(absorbance) > window:
        return savgol_filter(absorbance, window_length=window, polyorder=SMOOTHING_POLYORDER)
    return absorbance


def find_local_maxima(distance, absorbance, min_dist, max_dist, order=5):
    """Find local maxima in a region."""
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    mask = (distance >= min_dist) & (distance <= max_dist)
    if not np.any(mask):
        return []

    region_indices = np.where(mask)[0]
    region_abs = absorbance[mask]

    smoothed = smooth_profile(region_abs)
    local_max_indices = argrelmax(smoothed, order=order)[0]

    peaks = []
    for local_idx in local_max_indices:
        global_idx = region_indices[local_idx]
        peaks.append({
            'index': global_idx,
            'position': distance[global_idx],
            'height': absorbance[global_idx],
            'smoothed_height': smoothed[local_idx]
        })

    return peaks


def find_local_minima(distance, absorbance, min_dist, max_dist, order=5):
    """Find local minima (troughs) in a region."""
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    mask = (distance >= min_dist) & (distance <= max_dist)
    if not np.any(mask):
        return []

    region_indices = np.where(mask)[0]
    region_abs = absorbance[mask]

    smoothed = smooth_profile(region_abs)
    local_min_indices = argrelmin(smoothed, order=order)[0]

    troughs = []
    for local_idx in local_min_indices:
        global_idx = region_indices[local_idx]
        troughs.append({
            'index': global_idx,
            'position': distance[global_idx],
            'depth': absorbance[global_idx],
            'smoothed_depth': smoothed[local_idx]
        })

    return troughs


def get_value_at_position(distance, absorbance, position):
    """Get interpolated absorbance value at a given position."""
    idx = np.searchsorted(distance, position)
    if idx == 0:
        return absorbance[0]
    if idx >= len(distance):
        return absorbance[-1]
    # Linear interpolation
    x0, x1 = distance[idx-1], distance[idx]
    y0, y1 = absorbance[idx-1], absorbance[idx]
    return y0 + (y1 - y0) * (position - x0) / (x1 - x0)


# =============================================================================
# LAYER 1: PRIMARY LANDMARK DETECTION (no priors needed)
# =============================================================================

def detect_primary_landmarks(distance, absorbance):
    """Detect free and 80S peaks, compute ruler."""
    result = {'free': None, '80S': None, 'ruler': None}

    # Free peak: largest in free fraction region
    free_peaks = find_local_maxima(distance, absorbance, 0, FREE_FRACTION_END, order=10)
    if free_peaks:
        result['free'] = max(free_peaks, key=lambda p: p['height'])

    # 80S peak: tallest in monosome region
    mono_peaks = find_local_maxima(distance, absorbance,
                                    MONOSOME_REGION_START, MONOSOME_REGION_END, order=5)
    if mono_peaks:
        result['80S'] = max(mono_peaks, key=lambda p: p['height'])

    # Compute ruler
    if result['free'] and result['80S']:
        result['ruler'] = result['80S']['position'] - result['free']['position']

    return result


# =============================================================================
# LAYER 2: TROUGH DETECTION
# =============================================================================

def detect_post_80S_trough(distance, absorbance, pos_80S, ruler):
    """Find the trough after 80S that marks the polysome boundary."""
    search_start = pos_80S + 0.05 * ruler
    search_end = pos_80S + 0.4 * ruler

    troughs = find_local_minima(distance, absorbance, search_start, search_end, order=7)

    if troughs:
        return min(troughs, key=lambda t: t['smoothed_depth'])
    return None


# =============================================================================
# LAYER 3: PLATOSOME-GUIDED DETECTION
# =============================================================================

def detect_peak_with_prior(distance, absorbance, platosome, peak_name,
                           ruler, free_position, detected_peaks):
    """
    Detect a peak using platosome prior as guide.

    Returns:
        dict with 'position', 'height', 'confidence', 'method'
    """
    if peak_name not in platosome.peaks:
        return None

    prior = platosome.peaks[peak_name]

    # Get search bounds from platosome (±2 SD)
    search_min, search_max = prior.position_bounds(ruler, free_position, n_sd=2.5)

    # Constrain to valid region
    search_min = max(0, search_min)
    search_max = min(np.max(distance), search_max)

    # Find local maxima in search region
    peaks = find_local_maxima(distance, absorbance, search_min, search_max, order=5)

    if peaks:
        # Expected position
        expected_pos = prior.expected_position(ruler, free_position)

        # Score peaks by proximity to expected position and height
        def score_peak(p):
            pos_score = -abs(p['position'] - expected_pos) / ruler
            height_score = p['smoothed_height'] * 0.1
            return pos_score + height_score

        best_peak = max(peaks, key=score_peak)

        # Determine confidence based on peak prominence
        height_80S = detected_peaks.get('80S', {}).get('height', 1.0)
        relative_height = best_peak['height'] / height_80S if height_80S else 0

        if relative_height > prior.amplitude_relative * 0.5:
            confidence = CONFIDENCE_DETECTED
        else:
            confidence = CONFIDENCE_WEAK

        return {
            'position': best_peak['position'],
            'height': best_peak['height'],
            'smoothed_height': best_peak['smoothed_height'],
            'index': best_peak['index'],
            'confidence': confidence,
            'method': 'detected'
        }

    # No peak found - infer from platosome
    expected_pos = prior.expected_position(ruler, free_position)
    expected_height = get_value_at_position(distance, absorbance, expected_pos)

    return {
        'position': expected_pos,
        'height': expected_height,
        'smoothed_height': expected_height,
        'index': np.argmin(np.abs(distance - expected_pos)),
        'confidence': CONFIDENCE_INFERRED,
        'method': 'inferred'
    }


def detect_polysomes_with_prior(distance, absorbance, platosome, ruler, free_position,
                                 post_80S_trough, detected_peaks):
    """
    Detect polysome peaks using platosome priors.
    """
    polysomes = {}

    pos_80S = detected_peaks['80S']['position']

    # Determine search start
    if post_80S_trough:
        search_start = post_80S_trough['position']
    else:
        search_start = pos_80S + 0.15 * ruler

    polysome_names = ['2-some', '3-some', '4-some', '5-some', '6-some']

    for peak_name in polysome_names:
        if peak_name not in platosome.peaks:
            continue

        prior = platosome.peaks[peak_name]
        expected_pos = prior.expected_position(ruler, free_position)

        # Skip if expected position is before search start
        if expected_pos < search_start:
            continue

        # Skip if expected position is beyond data
        if expected_pos > POLYSOME_REGION_END:
            break

        # Get search bounds
        search_min, search_max = prior.position_bounds(ruler, free_position, n_sd=2.0)
        search_min = max(search_start, search_min)
        search_max = min(POLYSOME_REGION_END, search_max)

        # Find peaks in region
        peaks = find_local_maxima(distance, absorbance, search_min, search_max, order=5)

        if peaks:
            # Prefer peak closest to expected position
            best_peak = min(peaks, key=lambda p: abs(p['position'] - expected_pos))

            # Check if it's a real peak (has some prominence)
            height_80S = detected_peaks.get('80S', {}).get('height', 1.0)
            relative_height = best_peak['height'] / height_80S if height_80S else 0

            if relative_height > 0.02:  # At least 2% of 80S height
                polysomes[peak_name] = {
                    'position': best_peak['position'],
                    'height': best_peak['height'],
                    'smoothed_height': best_peak['smoothed_height'],
                    'index': best_peak['index'],
                    'confidence': CONFIDENCE_DETECTED if relative_height > 0.05 else CONFIDENCE_WEAK,
                    'method': 'detected'
                }
                search_start = best_peak['position'] + 0.1 * ruler
                continue

        # Infer from platosome if detection rate warrants it
        if prior.detection_rate > 0.5:
            polysomes[peak_name] = {
                'position': expected_pos,
                'height': get_value_at_position(distance, absorbance, expected_pos),
                'smoothed_height': get_value_at_position(
                    distance, smooth_profile(absorbance), expected_pos),
                'index': np.argmin(np.abs(distance - expected_pos)),
                'confidence': CONFIDENCE_INFERRED,
                'method': 'inferred'
            }
            search_start = expected_pos + 0.1 * ruler

    return polysomes


# =============================================================================
# PEAK FITTING
# =============================================================================

def skewnorm_with_baseline(x, amplitude, loc, scale, skewness, baseline):
    """Skew-normal PDF with baseline offset."""
    return baseline + amplitude * skewnorm.pdf(x, skewness, loc=loc, scale=scale)


def fit_single_peak(distance, absorbance, peak_position, window_mm=FIT_WINDOW_MM):
    """Fit a single skew-normal to a peak."""
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    mask = (distance >= peak_position - window_mm) & (distance <= peak_position + window_mm)
    if np.sum(mask) < 10:
        return None

    x = distance[mask]
    y = absorbance[mask]

    baseline = np.min(y)
    amplitude = (np.max(y) - baseline) * 2
    loc = peak_position
    scale = 1.5
    skewness = 0.0

    loc_tolerance = min(2.0, window_mm * 0.5)
    bounds_low = [0, peak_position - loc_tolerance, 0.3, -5, 0]
    bounds_high = [amplitude * 5, peak_position + loc_tolerance, window_mm, 5, np.max(y)]

    try:
        popt, _ = curve_fit(
            skewnorm_with_baseline, x, y,
            p0=[amplitude, loc, scale, skewness, baseline],
            bounds=(bounds_low, bounds_high),
            maxfev=2000
        )
        amplitude, loc, scale, skewness, baseline = popt

        delta = skewness / np.sqrt(1 + skewness**2)
        mode_shift = delta * np.sqrt(2/np.pi) * scale
        mode = loc + mode_shift

        if abs(mode - peak_position) > window_mm * 0.75:
            mode = peak_position

        height = skewnorm_with_baseline(mode, amplitude, loc, scale, skewness, baseline)
        fwhm = 2.355 * scale * (1 + 0.1 * abs(skewness))

        return {
            'amplitude': amplitude, 'location': loc, 'scale': scale,
            'skewness': skewness, 'baseline': baseline,
            'mode': mode, 'height': height, 'fwhm': fwhm
        }
    except (RuntimeError, ValueError):
        return {
            'amplitude': np.max(y) - np.min(y), 'location': peak_position,
            'scale': 1.5, 'skewness': 0.0, 'baseline': np.min(y),
            'mode': peak_position, 'height': np.max(y), 'fwhm': 3.0
        }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_profile(distance, absorbance, platosome, ref_peaks, ref_area,
                    identifier, verbose=False):
    """Process a single profile using platosome-guided detection."""
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # LAYER 1: Primary landmarks (no priors)
    landmarks = detect_primary_landmarks(distance, absorbance)

    if not landmarks['free'] or not landmarks['80S']:
        print(f"  WARNING: Could not detect primary landmarks for {identifier}",
              file=sys.stderr)
        return None, None, {}, 1.0, 0.0, 1.0

    ruler = landmarks['ruler']
    free_pos = landmarks['free']['position']

    if verbose:
        print(f"  Layer 1 - Free: {free_pos:.1f}mm, "
              f"80S: {landmarks['80S']['position']:.1f}mm, Ruler: {ruler:.1f}mm",
              file=sys.stderr)

    # Build detected peaks dict
    detected_peaks = {
        'free': landmarks['free'],
        '80S': landmarks['80S']
    }
    detected_peaks['free']['confidence'] = CONFIDENCE_DETECTED
    detected_peaks['free']['method'] = 'detected'
    detected_peaks['80S']['confidence'] = CONFIDENCE_DETECTED
    detected_peaks['80S']['method'] = 'detected'

    # LAYER 2: Post-80S trough
    post_80S_trough = detect_post_80S_trough(
        distance, absorbance, landmarks['80S']['position'], ruler)

    if verbose and post_80S_trough:
        print(f"  Layer 2 - Post-80S trough: {post_80S_trough['position']:.1f}mm",
              file=sys.stderr)

    # LAYER 3: Platosome-guided detection

    # Detect 40S
    peak_40S = detect_peak_with_prior(distance, absorbance, platosome, '40S',
                                       ruler, free_pos, detected_peaks)
    if peak_40S:
        detected_peaks['40S'] = peak_40S

    # Detect 60S
    peak_60S = detect_peak_with_prior(distance, absorbance, platosome, '60S',
                                       ruler, free_pos, detected_peaks)
    if peak_60S:
        detected_peaks['60S'] = peak_60S

    # Detect polysomes
    polysomes = detect_polysomes_with_prior(
        distance, absorbance, platosome, ruler, free_pos,
        post_80S_trough, detected_peaks)
    detected_peaks.update(polysomes)

    if verbose:
        detected_str = ', '.join([
            f"{k}:{v['position']:.1f}({'D' if v['method']=='detected' else 'I'})"
            for k, v in sorted(detected_peaks.items(),
                              key=lambda x: PEAK_ORDER.index(x[0]) if x[0] in PEAK_ORDER else 99)
        ])
        print(f"  Layer 3 - Peaks: {detected_str}", file=sys.stderr)

    # Fit each detected/inferred peak
    labeled_peaks = {}
    for peak_name, peak_info in detected_peaks.items():
        fit = fit_single_peak(distance, absorbance, peak_info['position'])
        if fit:
            fit['confidence'] = peak_info['confidence']
            fit['method'] = peak_info['method']
            labeled_peaks[peak_name] = fit

    # Build transform
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

    # Build fits dataframe (with confidence and method)
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
            'fwhm': peak['fwhm'],
            'confidence': peak['confidence'],
            'method': peak['method']
        })

    fits_df = pd.DataFrame(fits_data)

    return result_df, fits_df, labeled_peaks, x_scale, x_offset, y_scale


def build_affine_transform(source_peaks, ref_peaks):
    """Build affine transform anchored on free and 80S."""
    has_free = 'free' in source_peaks and 'free' in ref_peaks
    has_80S = '80S' in source_peaks and '80S' in ref_peaks

    if has_free and has_80S:
        src_free = source_peaks['free']['mode']
        src_80s = source_peaks['80S']['mode']
        ref_free = ref_peaks['free']['mode']
        ref_80s = ref_peaks['80S']['mode']

        src_distance = src_80s - src_free
        ref_distance = ref_80s - ref_free

        if abs(src_distance) > 1.0:
            scale = ref_distance / src_distance
        else:
            scale = 1.0

        scale = max(0.7, min(1.4, scale))
        offset = ref_free - src_free * scale
    elif has_80S:
        scale = 1.0
        offset = ref_peaks['80S']['mode'] - source_peaks['80S']['mode']
    elif has_free:
        scale = 1.0
        offset = ref_peaks['free']['mode'] - source_peaks['free']['mode']
    else:
        return lambda x: np.asarray(x, dtype=float), 1.0, 0.0

    def affine_transform(x):
        return np.asarray(x, dtype=float) * scale + offset

    return affine_transform, scale, offset


def compute_total_area(distance, absorbance):
    """Compute total area under curve, excluding artifacts."""
    mask = distance < ARTIFACT_START
    return trapezoid(absorbance[mask], distance[mask])


def parse_raw_file(filepath):
    """Parse a Gradient Profiler raw data file."""
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

    return pd.DataFrame(data_lines)


def normalize_profiles(file_list, platosome, verbose=False):
    """Normalize multiple polysome profiles using platosome priors."""
    profiles = []
    for filepath in file_list:
        path = Path(filepath)
        df = parse_raw_file(filepath)
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

    # Detect reference peaks
    landmarks = detect_primary_landmarks(ref_distance, ref_absorbance)
    ruler = landmarks['ruler']
    free_pos = landmarks['free']['position']

    ref_peaks = {}
    for peak_name in ['free', '80S']:
        if landmarks[peak_name]:
            fit = fit_single_peak(ref_distance, ref_absorbance,
                                  landmarks[peak_name]['position'])
            if fit:
                ref_peaks[peak_name] = fit

    # Detect other reference peaks with platosome
    for peak_name in ['40S', '60S']:
        peak_info = detect_peak_with_prior(ref_distance, ref_absorbance, platosome,
                                            peak_name, ruler, free_pos, landmarks)
        if peak_info:
            fit = fit_single_peak(ref_distance, ref_absorbance, peak_info['position'])
            if fit:
                ref_peaks[peak_name] = fit

    ref_area = compute_total_area(ref_distance, ref_absorbance)

    peak_list = sorted(ref_peaks.keys(),
                       key=lambda x: PEAK_ORDER.index(x) if x in PEAK_ORDER else 99)
    print(f"  Reference peaks: {', '.join(peak_list)}", file=sys.stderr)
    print(f"  Ruler: {ruler:.2f} mm", file=sys.stderr)

    # Process all profiles
    all_results = []
    all_fits = []

    for profile in profiles:
        df = profile['data']
        identifier = profile['identifier']

        result_df, fits_df, labeled_peaks, x_scale, x_offset, y_scale = process_profile(
            df['distance'].values,
            df['absorbance'].values,
            platosome,
            ref_peaks,
            ref_area,
            identifier,
            verbose=verbose
        )

        if result_df is None:
            continue

        n_detected = sum(1 for p in labeled_peaks.values() if p.get('method') == 'detected')
        n_inferred = sum(1 for p in labeled_peaks.values() if p.get('method') == 'inferred')

        print(f"Processed: {identifier} - {len(labeled_peaks)} peaks "
              f"({n_detected} detected, {n_inferred} inferred), "
              f"X: {x_scale:.3f}x + {x_offset:.2f}, Y: {y_scale:.4f}", file=sys.stderr)

        all_results.append(result_df)
        all_fits.append(fits_df)

    return pd.concat(all_results, ignore_index=True), pd.concat(all_fits, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize polysome profiles using platosome-guided detection.'
    )
    parser.add_argument('files', nargs='+', help='Input profile files')
    parser.add_argument('-o', '--output', default='normalized_profiles.tsv',
                        help='Output file')
    parser.add_argument('-p', '--platosome', default='data/platosome.json',
                        help='Platosome JSON file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed detection information')

    args = parser.parse_args()

    # Load platosome
    platosome_path = Path(args.platosome)
    if not platosome_path.exists():
        print(f"Platosome not found at {platosome_path}", file=sys.stderr)
        print("Run: python platosome.py --estimate data/normalized_profiles_fits.tsv",
              file=sys.stderr)
        sys.exit(1)

    platosome = Platosome.load(str(platosome_path))
    print(f"Loaded platosome v{platosome.version} ({platosome.n_samples} samples)",
          file=sys.stderr)

    for f in args.files:
        if not Path(f).exists():
            parser.error(f"File not found: {f}")

    profiles_df, fits_df = normalize_profiles(args.files, platosome, verbose=args.verbose)

    profiles_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nWrote {len(profiles_df)} rows to {args.output}", file=sys.stderr)

    fits_output = args.output.replace('.tsv', '_fits.tsv')
    fits_df.to_csv(fits_output, sep='\t', index=False)
    print(f"Wrote {len(fits_df)} peak fits to {fits_output}", file=sys.stderr)


if __name__ == '__main__':
    main()
