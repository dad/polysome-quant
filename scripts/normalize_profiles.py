#!/usr/bin/env python3
"""
Normalize polysome profiles for consistent comparison.

Takes multiple polysome profile files as input, aligns them using landmark peaks
(60S, 80S, disome) for x-axis alignment, and normalizes by total area under curve.

Profile structure:
- 0-15mm: Free fraction (variable, often dominant)
- 15-35mm: 60S, 80S peaks
- 35-60mm: Polysome peaks (2-some, 3-some, etc.)
- >60mm: Poor resolution
- >85mm: Artifacts (ignored)

Usage:
    python normalize_profiles.py file1.csv file2.csv ... -o output.tsv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import sys


# Region boundaries (in mm)
FREE_FRACTION_END = 15.0      # Free fraction ends here
RIBOSOME_REGION_START = 15.0  # Look for 60S, 80S, polysomes starting here
POLYSOME_REGION_END = 60.0    # Poor resolution past this
ARTIFACT_START = 85.0         # Artifacts begin here, ignore


def parse_raw_file(filepath):
    """
    Parse a Gradient Profiler raw data file.

    Returns:
        tuple: (metadata_dict, dataframe with Distance and Absorbance columns)
    """
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
                # Parse data line: Distance, Absorbance, [Fraction Number], [Fraction Volume]
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        distance = float(parts[0].strip())
                        absorbance = float(parts[1].strip())
                        data_lines.append({'distance': distance, 'absorbance': absorbance})
                    except ValueError:
                        continue
            else:
                # Parse metadata
                if ':' in line:
                    key, _, value = line.partition(':')
                    metadata[key.strip()] = value.strip()

    df = pd.DataFrame(data_lines)
    return metadata, df


def detect_peaks_in_region(distance, absorbance, min_dist, max_dist, prominence_threshold=0.005):
    """
    Detect peaks in a specific region of the profile.

    Returns:
        list of dicts with 'index', 'position', 'height', 'prominence' for each peak
    """
    # Create mask for region
    mask = (distance >= min_dist) & (distance <= max_dist)
    if not np.any(mask):
        return []

    # Get indices where mask is True
    region_indices = np.where(mask)[0]

    # Smooth the signal
    if len(absorbance) > 11:
        smoothed = savgol_filter(absorbance, window_length=11, polyorder=3)
    else:
        smoothed = np.array(absorbance)

    # Find peaks in the masked region
    region_smoothed = smoothed[mask]

    peak_indices_local, properties = find_peaks(
        region_smoothed,
        prominence=prominence_threshold,
        distance=5
    )

    peaks = []
    for i, local_idx in enumerate(peak_indices_local):
        # Convert local index back to global index
        global_idx = region_indices[local_idx]
        peaks.append({
            'index': global_idx,
            'position': distance[global_idx],
            'height': absorbance[global_idx],
            'prominence': properties['prominences'][i]
        })

    return peaks


def identify_landmark_peaks(distance, absorbance):
    """
    Identify landmark peaks: free fraction, 60S, 80S, and polysomes.

    Profile structure:
    - Free fraction: 0-15mm (first major peak)
    - 60S: ~20-25mm (smaller peak before 80S)
    - 80S: ~25-35mm (prominent peak)
    - Polysomes: 35-60mm (2-some, 3-some, etc.)

    Returns:
        dict mapping peak labels to peak dicts
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    labeled = {}

    # Detect free fraction peak (0-15mm)
    free_peaks = detect_peaks_in_region(distance, absorbance, 0, FREE_FRACTION_END, prominence_threshold=0.01)
    if free_peaks:
        # Take the most prominent peak in free fraction
        labeled['free'] = max(free_peaks, key=lambda p: p['prominence'])

    # Detect peaks in ribosome region (15-60mm)
    ribosome_peaks = detect_peaks_in_region(
        distance, absorbance,
        RIBOSOME_REGION_START, POLYSOME_REGION_END,
        prominence_threshold=0.005
    )

    if not ribosome_peaks:
        return labeled

    # Sort by position
    ribosome_peaks = sorted(ribosome_peaks, key=lambda p: p['position'])

    # Find 80S: most prominent peak in 20-40mm range
    peaks_80s_region = [p for p in ribosome_peaks if 20 <= p['position'] <= 40]
    if peaks_80s_region:
        peak_80s = max(peaks_80s_region, key=lambda p: p['prominence'])
        labeled['80S'] = peak_80s

        # Find 60S: peak just before 80S (within ~10mm)
        peaks_before_80s = [p for p in ribosome_peaks
                           if p['position'] < peak_80s['position']
                           and p['position'] > peak_80s['position'] - 15]
        if peaks_before_80s:
            # Take the closest peak before 80S
            labeled['60S'] = max(peaks_before_80s, key=lambda p: p['position'])

        # Find 40S: peak before 60S if exists
        if '60S' in labeled:
            peaks_before_60s = [p for p in ribosome_peaks
                               if p['position'] < labeled['60S']['position']
                               and p['position'] > labeled['60S']['position'] - 10]
            if peaks_before_60s:
                labeled['40S'] = max(peaks_before_60s, key=lambda p: p['position'])

        # Polysomes: peaks after 80S
        peaks_after_80s = [p for p in ribosome_peaks if p['position'] > peak_80s['position']]
        polysome_labels = ['2-some', '3-some', '4-some', '5-some', '6-some', '7-some', '8-some']

        for i, peak in enumerate(peaks_after_80s):
            if i < len(polysome_labels):
                labeled[polysome_labels[i]] = peak

    return labeled


def compute_total_area(distance, absorbance, exclude_artifacts=True):
    """
    Compute total area under the curve using trapezoidal integration.

    Args:
        distance: array of distance values
        absorbance: array of absorbance values
        exclude_artifacts: if True, exclude region past ARTIFACT_START
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    if exclude_artifacts:
        mask = distance < ARTIFACT_START
        distance = distance[mask]
        absorbance = absorbance[mask]

    return trapezoid(absorbance, distance)


def fit_x_transform(source_peaks, ref_peaks):
    """
    Fit x-axis affine transform (offset + scale) to align source peaks to reference peaks.

    Uses 60S and 80S as primary landmarks for exact alignment.
    With exactly 2 points, the affine transform fits perfectly through both.

    Returns:
        tuple: (x_offset, x_scale) or (0, 1) if fitting fails
    """
    # Find common landmarks between source and reference
    common_labels = set(source_peaks.keys()) & set(ref_peaks.keys())

    # Use 60S and 80S as primary alignment landmarks
    # These are the most reliable and biologically meaningful reference points
    primary_labels = ['60S', '80S']
    alignment_labels = [l for l in primary_labels if l in common_labels]

    # Fall back to other peaks if 60S/80S not available
    if len(alignment_labels) < 2:
        fallback_labels = ['2-some', '3-some', '40S', '4-some']
        for label in fallback_labels:
            if label in common_labels and label not in alignment_labels:
                alignment_labels.append(label)
            if len(alignment_labels) >= 2:
                break

    if len(alignment_labels) < 2:
        return 0.0, 1.0

    # Build arrays of positions
    src_positions = np.array([source_peaks[l]['position'] for l in alignment_labels])
    ref_positions = np.array([ref_peaks[l]['position'] for l in alignment_labels])

    # For exactly 2 points, solve directly for perfect fit
    # ref = src * scale + offset
    if len(alignment_labels) == 2:
        # Solve: ref1 = src1 * scale + offset
        #        ref2 = src2 * scale + offset
        # scale = (ref2 - ref1) / (src2 - src1)
        # offset = ref1 - src1 * scale
        src_diff = src_positions[1] - src_positions[0]
        if abs(src_diff) < 1e-6:
            return 0.0, 1.0
        scale = (ref_positions[1] - ref_positions[0]) / src_diff
        offset = ref_positions[0] - src_positions[0] * scale
    else:
        # For 3+ points, use least squares
        A = np.column_stack([np.ones_like(src_positions), src_positions])
        result, residuals, rank, s = np.linalg.lstsq(A, ref_positions, rcond=None)
        offset, scale = result

    return offset, scale


def normalize_single_profile(distance, absorbance, ref_peaks, ref_area):
    """
    Normalize a single profile using:
    1. X-axis affine transform to align landmark peaks (60S, 80S, 2-some)
    2. Y-axis scaling by total area under curve

    Returns:
        tuple: (normalized_distance, normalized_absorbance, labeled_peaks,
                x_offset, x_scale, y_scale)
    """
    distance = np.array(distance)
    absorbance = np.array(absorbance)

    # Detect and label peaks
    labeled_peaks = identify_landmark_peaks(distance, absorbance)

    # Fit x transform using landmark peaks
    x_offset, x_scale = fit_x_transform(labeled_peaks, ref_peaks)

    # Apply x transform
    norm_distance = distance * x_scale + x_offset

    # Compute area and normalize y
    area = compute_total_area(distance, absorbance)
    if area > 0:
        y_scale = ref_area / area
    else:
        y_scale = 1.0

    norm_absorbance = absorbance * y_scale

    # Update peak positions for the transformed coordinates
    transformed_peaks = {}
    for label, peak in labeled_peaks.items():
        transformed_peaks[label] = {
            'index': peak['index'],
            'position': peak['position'] * x_scale + x_offset,
            'height': peak['height'] * y_scale,
            'prominence': peak['prominence'] * y_scale
        }

    return norm_distance, norm_absorbance, transformed_peaks, x_offset, x_scale, y_scale


def normalize_profiles(file_list):
    """
    Normalize multiple polysome profiles.

    Uses the first file as the reference for normalization.
    X-axis: aligned using 60S, 80S, disome peaks
    Y-axis: normalized by total area under curve

    Returns:
        DataFrame with columns: identifier, distance, absorbance, absorbance.normalized, peak
    """
    # Load all profiles
    profiles = []
    for filepath in file_list:
        path = Path(filepath)
        metadata, df = parse_raw_file(filepath)
        df['identifier'] = path.name
        profiles.append({
            'filepath': filepath,
            'identifier': path.name,
            'metadata': metadata,
            'data': df
        })

    if len(profiles) < 2:
        raise ValueError("Need at least 2 profiles for normalization")

    # Use first profile as reference
    ref_data = profiles[0]['data']
    ref_distance = ref_data['distance'].values
    ref_absorbance = ref_data['absorbance'].values

    ref_peaks = identify_landmark_peaks(ref_distance, ref_absorbance)
    ref_area = compute_total_area(ref_distance, ref_absorbance)

    if '80S' not in ref_peaks:
        raise ValueError(f"Could not identify 80S peak in reference file: {profiles[0]['identifier']}")

    print(f"Reference profile: {profiles[0]['identifier']}", file=sys.stderr)
    print(f"  Detected peaks: {', '.join(ref_peaks.keys())}", file=sys.stderr)
    print(f"  80S position: {ref_peaks['80S']['position']:.2f} mm", file=sys.stderr)
    print(f"  Total area: {ref_area:.4f}", file=sys.stderr)

    # Normalize all profiles
    results = []

    for profile in profiles:
        df = profile['data']
        identifier = profile['identifier']
        distance = df['distance'].values
        absorbance = df['absorbance'].values

        norm_distance, norm_absorbance, labeled_peaks, x_off, x_scale, y_scale = normalize_single_profile(
            distance, absorbance, ref_peaks, ref_area
        )

        # Create peak label column (empty except at peak positions)
        peak_labels = [''] * len(distance)
        for label, peak in labeled_peaks.items():
            idx = peak['index']
            if 0 <= idx < len(peak_labels):
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

        # Report normalization stats
        print(f"Normalized: {identifier}", file=sys.stderr)
        print(f"  Detected peaks: {', '.join(labeled_peaks.keys())}", file=sys.stderr)
        print(f"  X transform: distance * {x_scale:.4f} + {x_off:.4f}", file=sys.stderr)
        print(f"  Y scale: {y_scale:.4f}", file=sys.stderr)

        results.append(result_df)

    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize polysome profiles for consistent comparison.'
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

    # Check files exist
    for f in args.files:
        if not Path(f).exists():
            parser.error(f"File not found: {f}")

    # Normalize profiles
    result = normalize_profiles(args.files)

    # Write output
    result.to_csv(args.output, sep='\t', index=False)
    print(f"\nWrote {len(result)} rows to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
