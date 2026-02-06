#!/usr/bin/env python3
"""
Normalize polysome profiles using multi-layered feature detection.

This script implements a hierarchical approach to peak identification:

LAYER 1: Primary Landmarks
    - Free peak: dominant peak in free fraction region (0-12mm)
    - 80S peak: tallest peak in monosome region (20-40mm)
    - Ruler: distance from free to 80S (used for all relative measurements)

LAYER 2: Trough Detection
    - Find local minima (troughs) which segment the profile
    - Post-80S trough marks the boundary between 80S shoulder and polysomes
    - Inter-polysome troughs help validate peak assignments

LAYER 3: Secondary Landmarks (relative to ruler)
    - 60S: ~0.25 ruler before 80S
    - 40S: ~0.25 ruler before 60S (often weak/absent)
    - Polysomes: spaced ~0.25-0.35 ruler apart, starting AFTER post-80S trough

LAYER 4: Validation & Consistency
    - Peaks must be in ascending distance order
    - Inter-peak spacing must be consistent with ruler-based expectations
    - Each peak should have troughs on either side

Usage:
    python normalize_profiles_v3.py file1.csv file2.csv ... -o output.tsv
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

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Region boundaries (in mm)
FREE_FRACTION_END = 12.0
MONOSOME_REGION_START = 20.0
MONOSOME_REGION_END = 40.0
POLYSOME_REGION_END = 60.0
ARTIFACT_START = 85.0

# Relative spacing constraints (as fractions of the ruler = free-to-80S distance)
# Typical ruler is ~24mm, so 0.25 * ruler = ~6mm
RELATIVE_60S_OFFSET = 0.25          # 60S is ~0.25 ruler before 80S
RELATIVE_40S_OFFSET = 0.25          # 40S is ~0.25 ruler before 60S
RELATIVE_POLYSOME_SPACING_MIN = 0.20  # Min spacing between polysomes
RELATIVE_POLYSOME_SPACING_MAX = 0.40  # Max spacing between polysomes
RELATIVE_POLYSOME_SPACING_NOMINAL = 0.28  # Expected spacing

# Peak detection parameters
SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3

# Peak fitting
FIT_WINDOW_MM = 4.0

PEAK_LABELS = ['free', '40S', '60S', '80S', '2-some', '3-some', '4-some',
               '5-some', '6-some', '7-some', '8-some']


# =============================================================================
# LAYER 1: PRIMARY LANDMARK DETECTION
# =============================================================================

def smooth_profile(absorbance, window=SMOOTHING_WINDOW):
    """Apply Savitzky-Golay smoothing to the profile."""
    if len(absorbance) > window:
        return savgol_filter(absorbance, window_length=window, polyorder=SMOOTHING_POLYORDER)
    return absorbance


def find_local_maxima(distance, absorbance, min_dist, max_dist, order=5):
    """
    Find local maxima in a region using comparison to neighbors.

    Returns list of dicts with 'index', 'position', 'height' for each peak.
    """
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
    """
    Find local minima (troughs) in a region.

    Returns list of dicts with 'index', 'position', 'depth' for each trough.
    """
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


def detect_primary_landmarks(distance, absorbance):
    """
    LAYER 1: Detect free and 80S peaks, compute the ruler.

    Returns:
        dict with 'free', '80S' peaks and 'ruler' distance
    """
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

def detect_troughs(distance, absorbance, landmarks):
    """
    LAYER 2: Detect troughs that segment the profile.

    Key troughs:
        - post_80S: first significant trough after 80S (marks polysome boundary)
        - inter_polysome: troughs between polysome peaks

    Returns:
        dict with trough information
    """
    result = {'post_80S': None, 'all_troughs': []}

    if not landmarks['80S'] or not landmarks['ruler']:
        return result

    pos_80S = landmarks['80S']['position']
    ruler = landmarks['ruler']

    # Look for post-80S trough: should be 0.1-0.3 ruler after 80S
    search_start = pos_80S + 0.05 * ruler
    search_end = pos_80S + 0.4 * ruler

    troughs = find_local_minima(distance, absorbance, search_start, search_end, order=7)

    if troughs:
        # Take the most prominent trough (lowest point)
        result['post_80S'] = min(troughs, key=lambda t: t['smoothed_depth'])

    # Find all troughs in polysome region for later validation
    polysome_start = pos_80S + 0.1 * ruler
    all_troughs = find_local_minima(distance, absorbance,
                                     polysome_start, POLYSOME_REGION_END, order=7)
    result['all_troughs'] = sorted(all_troughs, key=lambda t: t['position'])

    return result


# =============================================================================
# LAYER 3: SECONDARY LANDMARK DETECTION
# =============================================================================

def detect_60S_40S(distance, absorbance, landmarks):
    """
    Detect 60S and 40S peaks relative to 80S using the ruler.

    60S should be ~0.25 ruler before 80S
    40S should be ~0.25 ruler before 60S
    """
    result = {'60S': None, '40S': None}

    if not landmarks['80S'] or not landmarks['ruler']:
        return result

    pos_80S = landmarks['80S']['position']
    ruler = landmarks['ruler']

    # Expected 60S position
    expected_60S = pos_80S - RELATIVE_60S_OFFSET * ruler
    search_60S_min = expected_60S - 0.15 * ruler
    search_60S_max = expected_60S + 0.10 * ruler

    # Find peaks in 60S region
    peaks_60S = find_local_maxima(distance, absorbance,
                                   max(FREE_FRACTION_END, search_60S_min),
                                   search_60S_max, order=5)

    if peaks_60S:
        # Take peak closest to expected position
        result['60S'] = min(peaks_60S,
                           key=lambda p: abs(p['position'] - expected_60S))

    # Expected 40S position (relative to 60S if found, else relative to 80S)
    if result['60S']:
        expected_40S = result['60S']['position'] - RELATIVE_40S_OFFSET * ruler
    else:
        expected_40S = pos_80S - 2 * RELATIVE_60S_OFFSET * ruler

    search_40S_min = expected_40S - 0.10 * ruler
    search_40S_max = expected_40S + 0.10 * ruler

    peaks_40S = find_local_maxima(distance, absorbance,
                                   max(FREE_FRACTION_END, search_40S_min),
                                   search_40S_max, order=5)

    if peaks_40S:
        result['40S'] = min(peaks_40S,
                           key=lambda p: abs(p['position'] - expected_40S))

    return result


def detect_polysomes(distance, absorbance, landmarks, troughs):
    """
    LAYER 3: Detect polysome peaks using ruler-based spacing constraints.

    Key constraints:
        - Polysomes must start AFTER the post-80S trough
        - Inter-polysome spacing should be 0.20-0.40 * ruler
        - Each successive polysome should be further from 80S

    Strategy:
        - Find the first peak after the post-80S trough (this is 2-some)
        - Then find subsequent peaks with appropriate spacing
    """
    polysomes = []

    if not landmarks['80S'] or not landmarks['ruler']:
        return polysomes

    pos_80S = landmarks['80S']['position']
    ruler = landmarks['ruler']

    # Determine polysome search start (right after post-80S trough)
    if troughs['post_80S']:
        search_start = troughs['post_80S']['position']
    else:
        # Fallback: start 0.15 ruler after 80S (minimum clearance from 80S shoulder)
        search_start = pos_80S + 0.15 * ruler

    # Expected spacing between consecutive polysomes
    nominal_spacing = RELATIVE_POLYSOME_SPACING_NOMINAL * ruler
    min_spacing = RELATIVE_POLYSOME_SPACING_MIN * ruler
    max_spacing = RELATIVE_POLYSOME_SPACING_MAX * ruler

    # Find all peaks in polysome region (starting right after trough)
    all_peaks = find_local_maxima(distance, absorbance,
                                   search_start, POLYSOME_REGION_END, order=5)

    if not all_peaks:
        return polysomes

    # Sort by position
    all_peaks = sorted(all_peaks, key=lambda p: p['position'])

    # First polysome (2-some): take the first significant peak after the trough
    # It should be close to the trough (within 0.15 * ruler)
    first_candidates = [p for p in all_peaks
                        if p['position'] < search_start + 0.15 * ruler]

    if not first_candidates:
        # If nothing close to trough, take the first peak
        first_candidates = all_peaks[:3] if len(all_peaks) >= 3 else all_peaks

    if first_candidates:
        # Take the most prominent (tallest) among early candidates
        first_polysome = max(first_candidates, key=lambda p: p['smoothed_height'])
        first_polysome['label'] = '2-some'
        polysomes.append(first_polysome)
        all_peaks = [p for p in all_peaks if p['index'] != first_polysome['index']]
        current_pos = first_polysome['position']
    else:
        return polysomes

    # Subsequent polysomes: use spacing constraints
    polysome_labels = ['3-some', '4-some', '5-some', '6-some', '7-some', '8-some']

    for label in polysome_labels:
        expected_pos = current_pos + nominal_spacing

        if expected_pos > POLYSOME_REGION_END:
            break

        # Find candidates within acceptable spacing range from current position
        candidates = [p for p in all_peaks
                      if min_spacing * 0.8 <= (p['position'] - current_pos) <= max_spacing * 1.3
                      and p['position'] > current_pos]

        if not candidates:
            # Try with more relaxed constraints
            candidates = [p for p in all_peaks
                          if p['position'] > current_pos + min_spacing * 0.5
                          and p['position'] < current_pos + max_spacing * 1.8]

        if candidates:
            # Prefer peak closest to expected position, with height as tiebreaker
            best = min(candidates,
                      key=lambda p: abs(p['position'] - expected_pos) - 0.1 * p['smoothed_height'])
            best['label'] = label
            polysomes.append(best)
            current_pos = best['position']
            all_peaks = [p for p in all_peaks if p['index'] != best['index']]
        else:
            break

    return polysomes


# =============================================================================
# LAYER 4: VALIDATION & CONSISTENCY CHECKS
# =============================================================================

def validate_peaks(landmarks, subunits, polysomes, troughs, ruler):
    """
    LAYER 4: Validate peak assignments for consistency.

    Checks:
        1. Peaks are in ascending order
        2. Inter-peak spacing is reasonable (relative to ruler)
        3. Troughs exist between adjacent peaks

    Returns:
        dict with validation results and any corrections
    """
    validation = {
        'valid': True,
        'warnings': [],
        'corrections': []
    }

    if not ruler:
        validation['valid'] = False
        validation['warnings'].append("No ruler computed - cannot validate")
        return validation

    # Collect all peaks in order
    all_peaks = []

    if landmarks.get('free'):
        all_peaks.append(('free', landmarks['free']['position']))
    if subunits.get('40S'):
        all_peaks.append(('40S', subunits['40S']['position']))
    if subunits.get('60S'):
        all_peaks.append(('60S', subunits['60S']['position']))
    if landmarks.get('80S'):
        all_peaks.append(('80S', landmarks['80S']['position']))
    for p in polysomes:
        all_peaks.append((p['label'], p['position']))

    all_peaks = sorted(all_peaks, key=lambda x: x[1])

    # Check ordering
    for i in range(len(all_peaks) - 1):
        if all_peaks[i][1] >= all_peaks[i + 1][1]:
            validation['warnings'].append(
                f"Peak order violation: {all_peaks[i][0]} >= {all_peaks[i + 1][0]}")

    # Check polysome spacing
    polysome_peaks = [(name, pos) for name, pos in all_peaks if '-some' in name]
    for i in range(len(polysome_peaks) - 1):
        spacing = polysome_peaks[i + 1][1] - polysome_peaks[i][1]
        relative_spacing = spacing / ruler

        if relative_spacing < RELATIVE_POLYSOME_SPACING_MIN * 0.7:
            validation['warnings'].append(
                f"Polysome spacing too small: {polysome_peaks[i][0]} to {polysome_peaks[i + 1][0]} "
                f"= {spacing:.1f}mm ({relative_spacing:.2f} * ruler)")
        elif relative_spacing > RELATIVE_POLYSOME_SPACING_MAX * 1.5:
            validation['warnings'].append(
                f"Polysome spacing too large: {polysome_peaks[i][0]} to {polysome_peaks[i + 1][0]} "
                f"= {spacing:.1f}mm ({relative_spacing:.2f} * ruler)")

    # Check for troughs between polysomes
    if troughs['all_troughs'] and len(polysome_peaks) >= 2:
        trough_positions = [t['position'] for t in troughs['all_troughs']]
        for i in range(len(polysome_peaks) - 1):
            pos1, pos2 = polysome_peaks[i][1], polysome_peaks[i + 1][1]
            troughs_between = [t for t in trough_positions if pos1 < t < pos2]
            if not troughs_between:
                validation['warnings'].append(
                    f"No trough between {polysome_peaks[i][0]} and {polysome_peaks[i + 1][0]}")

    return validation


# =============================================================================
# PEAK FITTING
# =============================================================================

def skewnorm_with_baseline(x, amplitude, loc, scale, skewness, baseline):
    """Skew-normal PDF with baseline offset."""
    return baseline + amplitude * skewnorm.pdf(x, skewness, loc=loc, scale=scale)


def fit_single_peak(distance, absorbance, peak_position, window_mm=FIT_WINDOW_MM):
    """
    Fit a single skew-normal to a peak in a local window.

    The fit is constrained to stay close to the detected peak position.
    If the fitted mode wanders too far from the detected position,
    we fall back to the detected position.
    """
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

    # Tighter bounds on location - keep within 2mm of detected position
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

        # If mode wandered too far from detected position, use detected position
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
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_profile(distance, absorbance, ref_peaks, ref_area, identifier, verbose=False):
    """
    Process a single profile through all detection layers.
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # LAYER 1: Primary landmarks
    landmarks = detect_primary_landmarks(distance, absorbance)

    if verbose:
        print(f"  Layer 1 - Free: {landmarks['free']['position']:.1f}mm, "
              f"80S: {landmarks['80S']['position']:.1f}mm, "
              f"Ruler: {landmarks['ruler']:.1f}mm", file=sys.stderr)

    # LAYER 2: Trough detection
    troughs = detect_troughs(distance, absorbance, landmarks)

    if verbose and troughs['post_80S']:
        print(f"  Layer 2 - Post-80S trough: {troughs['post_80S']['position']:.1f}mm",
              file=sys.stderr)

    # LAYER 3: Secondary landmarks
    subunits = detect_60S_40S(distance, absorbance, landmarks)
    polysomes = detect_polysomes(distance, absorbance, landmarks, troughs)

    if verbose:
        poly_str = ', '.join([f"{p['label']}:{p['position']:.1f}" for p in polysomes])
        print(f"  Layer 3 - 60S: {subunits['60S']['position'] if subunits['60S'] else 'N/A':.1f}mm, "
              f"Polysomes: {poly_str}", file=sys.stderr)

    # LAYER 4: Validation
    validation = validate_peaks(landmarks, subunits, polysomes, troughs, landmarks['ruler'])

    if verbose and validation['warnings']:
        for w in validation['warnings']:
            print(f"  Layer 4 WARNING: {w}", file=sys.stderr)

    # Compile all labeled peaks
    labeled_peaks = {}

    if landmarks['free']:
        fit = fit_single_peak(distance, absorbance, landmarks['free']['position'])
        if fit:
            labeled_peaks['free'] = fit

    if subunits['40S']:
        fit = fit_single_peak(distance, absorbance, subunits['40S']['position'])
        if fit:
            labeled_peaks['40S'] = fit

    if subunits['60S']:
        fit = fit_single_peak(distance, absorbance, subunits['60S']['position'])
        if fit:
            labeled_peaks['60S'] = fit

    if landmarks['80S']:
        fit = fit_single_peak(distance, absorbance, landmarks['80S']['position'])
        if fit:
            labeled_peaks['80S'] = fit

    for p in polysomes:
        fit = fit_single_peak(distance, absorbance, p['position'])
        if fit:
            labeled_peaks[p['label']] = fit

    # Build affine transform
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


def build_affine_transform(source_peaks, ref_peaks):
    """
    Build an affine transform to align source peaks to reference peaks.
    Anchors on free and 80S peaks.
    """
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
        src_80s = source_peaks['80S']['mode']
        ref_80s = ref_peaks['80S']['mode']
        scale = 1.0
        offset = ref_80s - src_80s

    elif has_free:
        src_free = source_peaks['free']['mode']
        ref_free = ref_peaks['free']['mode']
        scale = 1.0
        offset = ref_free - src_free

    else:
        return lambda x: np.asarray(x, dtype=float), 1.0, 0.0

    def affine_transform(x):
        return np.asarray(x, dtype=float) * scale + offset

    return affine_transform, scale, offset


def compute_total_area(distance, absorbance):
    """Compute total area under the curve, excluding artifacts."""
    mask = distance < ARTIFACT_START
    return trapezoid(absorbance[mask], distance[mask])


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


def normalize_profiles(file_list, verbose=False):
    """Normalize multiple polysome profiles."""
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

    # Detect reference peaks
    landmarks = detect_primary_landmarks(ref_distance, ref_absorbance)
    troughs = detect_troughs(ref_distance, ref_absorbance, landmarks)
    subunits = detect_60S_40S(ref_distance, ref_absorbance, landmarks)
    polysomes = detect_polysomes(ref_distance, ref_absorbance, landmarks, troughs)

    # Fit reference peaks
    ref_peaks = {}
    if landmarks['free']:
        fit = fit_single_peak(ref_distance, ref_absorbance, landmarks['free']['position'])
        if fit:
            ref_peaks['free'] = fit
    if landmarks['80S']:
        fit = fit_single_peak(ref_distance, ref_absorbance, landmarks['80S']['position'])
        if fit:
            ref_peaks['80S'] = fit
    if subunits['60S']:
        fit = fit_single_peak(ref_distance, ref_absorbance, subunits['60S']['position'])
        if fit:
            ref_peaks['60S'] = fit
    for p in polysomes:
        fit = fit_single_peak(ref_distance, ref_absorbance, p['position'])
        if fit:
            ref_peaks[p['label']] = fit

    ref_area = compute_total_area(ref_distance, ref_absorbance)

    if '80S' not in ref_peaks:
        raise ValueError(f"Could not identify 80S peak in reference: {ref_identifier}")

    peak_list = sorted(ref_peaks.keys(),
                       key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99)
    print(f"  Detected peaks: {', '.join(peak_list)}", file=sys.stderr)
    print(f"  Ruler: {landmarks['ruler']:.2f} mm", file=sys.stderr)

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
            identifier,
            verbose=verbose
        )

        peak_list = sorted(labeled_peaks.keys(),
                          key=lambda x: PEAK_LABELS.index(x) if x in PEAK_LABELS else 99)
        print(f"Processed: {identifier} - {len(labeled_peaks)} peaks, "
              f"X: {x_scale:.3f}x + {x_offset:.2f}, Y: {y_scale:.4f}", file=sys.stderr)

        all_results.append(result_df)
        all_fits.append(fits_df)

    return pd.concat(all_results, ignore_index=True), pd.concat(all_fits, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize polysome profiles using multi-layered feature detection.'
    )
    parser.add_argument('files', nargs='+', help='Input profile files')
    parser.add_argument('-o', '--output', default='normalized_profiles.tsv',
                        help='Output file (default: normalized_profiles.tsv)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed detection information')

    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("At least 2 input files required")

    for f in args.files:
        if not Path(f).exists():
            parser.error(f"File not found: {f}")

    profiles_df, fits_df = normalize_profiles(args.files, verbose=args.verbose)

    profiles_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nWrote {len(profiles_df)} rows to {args.output}", file=sys.stderr)

    fits_output = args.output.replace('.tsv', '_fits.tsv')
    fits_df.to_csv(fits_output, sep='\t', index=False)
    print(f"Wrote {len(fits_df)} peak fits to {fits_output}", file=sys.stderr)


if __name__ == '__main__':
    main()
