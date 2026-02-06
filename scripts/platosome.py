#!/usr/bin/env python3
"""
Platosome: A Platonic (idealized) polysome profile model.

The platosome serves as a prior for peak identification and quantification.
It captures the expected structure of polysome profiles with:
- Relative positions anchored to the ruler (free=0, 80S=1)
- Shape parameters (skew-normal) with variation estimates
- Empirical peak shape templates from training data
- Local step ratios for gradient-nonlinearity-aware positioning
- Polysome region envelope template for high-polysome normalization
- Relative amplitudes with coefficients of variation

Usage:
    from platosome import Platosome

    # Load the standard platosome
    plato = Platosome.load_default()

    # Generate an idealized profile
    profile = plato.generate_profile(ruler=24.5, free_position=4.0)

    # Use as prior for peak detection
    expected_pos = plato.expected_position('40S', ruler=24.5, free_position=4.0)

    # Use local step ratios for polysome detection
    pos, unc = plato.predicted_position('3-some', ruler=24.5, free_position=4.0,
                                         detected_peaks={'80S': 28.5, '2-some': 37.0})
"""

import json
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from scipy.stats import skewnorm
from scipy.signal import savgol_filter


# =============================================================================
# CONSTANTS
# =============================================================================

# Peak order constant
PEAK_ORDER = [
    'free', '40S', '60S', '80S', '2-some', '3-some',
    '4-some', '5-some', '6-some', '7-some', '8-some'
]

# Template grid: ruler-relative units centered on peak mode
TEMPLATE_GRID = np.linspace(-0.20, 0.20, 81)

# Envelope grid: ruler-relative units from post-80S to high polysomes
ENVELOPE_GRID = np.linspace(1.0, 3.0, 201)

# Predecessor triplets for local step ratio computation
# Maps each polysome peak to (pre_predecessor, predecessor)
PREDECESSOR_TRIPLETS = {
    '2-some': ('60S', '80S'),
    '3-some': ('80S', '2-some'),
    '4-some': ('2-some', '3-some'),
    '5-some': ('3-some', '4-some'),
    '6-some': ('4-some', '5-some'),
    '7-some': ('5-some', '6-some'),
    '8-some': ('6-some', '7-some'),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PeakPrior:
    """Prior distribution for a single peak in the platosome."""

    # Position relative to ruler (free=0, 80S=1)
    position_relative: float
    position_sd: float  # Standard deviation in ruler units

    # Skew-normal shape parameters
    scale: float        # omega parameter
    scale_sd: float
    skewness: float     # alpha parameter
    skewness_sd: float

    # Amplitude relative to 80S peak height
    amplitude_relative: float
    amplitude_cv: float  # Coefficient of variation

    # Detection rate (fraction of samples where peak is detected)
    detection_rate: float = 1.0

    # Whether this is an anchor peak (used for ruler definition)
    is_anchor: bool = False

    # Empirical shape template (v2)
    template_x: Optional[List[float]] = None
    template_mean: Optional[List[float]] = None
    template_sd: Optional[List[float]] = None
    template_n: int = 0
    fit_r2_mean: Optional[float] = None
    fit_r2_sd: Optional[float] = None

    # Local relative positioning (v2)
    local_step_ratio: Optional[float] = None
    local_step_ratio_sd: Optional[float] = None
    local_step_n: int = 0
    local_predecessor: Optional[str] = None
    local_pre_predecessor: Optional[str] = None

    def expected_position(self, ruler: float, free_position: float) -> float:
        """Compute expected absolute position given ruler and free position."""
        return free_position + self.position_relative * ruler

    def position_bounds(self, ruler: float, free_position: float,
                        n_sd: float = 2.0) -> Tuple[float, float]:
        """Compute position bounds (mean +/- n_sd) in absolute units."""
        expected = self.expected_position(ruler, free_position)
        margin = n_sd * self.position_sd * ruler
        return (expected - margin, expected + margin)

    def generate_curve(self, x: np.ndarray, ruler: float, free_position: float,
                       amplitude_80S: float = 1.0) -> np.ndarray:
        """Generate the expected peak curve at given x positions."""
        center = self.expected_position(ruler, free_position)
        # Scale the skew-normal scale parameter by ruler
        abs_scale = self.scale * ruler / 24.5  # Normalize to typical ruler
        amplitude = self.amplitude_relative * amplitude_80S

        return amplitude * skewnorm.pdf(x, self.skewness, loc=center, scale=abs_scale)

    def predicted_position_local(self, detected_peaks: Dict[str, float]) -> Optional[Tuple[float, float]]:
        """
        Predict this peak's position from local step ratio and predecessor positions.

        Args:
            detected_peaks: dict of {peak_name: position_mm}

        Returns:
            (predicted_position, uncertainty_mm) or None if predecessors not available
        """
        if (self.local_step_ratio is None or
                self.local_predecessor is None or
                self.local_pre_predecessor is None):
            return None

        pre = self.local_predecessor
        pre_pre = self.local_pre_predecessor

        if pre not in detected_peaks or pre_pre not in detected_peaks:
            return None

        local_ruler = detected_peaks[pre] - detected_peaks[pre_pre]
        if local_ruler <= 0:
            return None

        predicted = detected_peaks[pre] + self.local_step_ratio * local_ruler
        uncertainty = (self.local_step_ratio_sd * local_ruler
                       if self.local_step_ratio_sd else local_ruler * 0.1)

        return (predicted, uncertainty)


@dataclass
class PolysomeEnvelope:
    """Empirical envelope template for the polysome region."""

    envelope_x: List[float]        # ruler-relative grid
    envelope_mean: List[float]     # mean normalized absorbance (fraction of 80S height)
    envelope_sd: List[float]       # SD at each point
    envelope_n: List[int]          # samples contributing at each point
    envelope_n_total: int = 0      # total samples used


# =============================================================================
# PLATOSOME CLASS
# =============================================================================

@dataclass
class Platosome:
    """
    The Platonic polysome profile - an idealized reference model.

    Positions are encoded relative to the ruler:
        - free = 0.0 (anchor)
        - 80S = 1.0 (anchor)
        - Polysomes > 1.0
    """

    peaks: Dict[str, PeakPrior] = field(default_factory=dict)

    # Ruler statistics
    ruler_mean: float = 24.5
    ruler_sd: float = 2.3

    # Polysome region envelope
    envelope: Optional[PolysomeEnvelope] = None

    # Metadata
    version: str = "1.0"
    n_samples: int = 0
    description: str = ""

    def __post_init__(self):
        """Validate the platosome after initialization."""
        # Ensure anchors are present
        if 'free' in self.peaks:
            self.peaks['free'].is_anchor = True
            self.peaks['free'].position_relative = 0.0
        if '80S' in self.peaks:
            self.peaks['80S'].is_anchor = True
            self.peaks['80S'].position_relative = 1.0

    @classmethod
    def from_data(cls, fits_df, ruler_stats: dict,
                  templates=None, step_ratios=None, envelope=None) -> 'Platosome':
        """
        Estimate platosome parameters from fitted peak data.

        Args:
            fits_df: DataFrame with columns: identifier, peak, mode, scale,
                     skewness, height, plus computed: pos_relative, height_relative
            ruler_stats: dict with ruler_mean, ruler_sd
            templates: optional dict from compute_peak_templates()
            step_ratios: optional dict from compute_local_step_ratios()
            envelope: optional PolysomeEnvelope from compute_polysome_envelope()
        """
        peaks = {}

        n_total = fits_df['identifier'].nunique()

        for peak_name in PEAK_ORDER:
            peak_data = fits_df[fits_df['peak'] == peak_name]
            if len(peak_data) == 0:
                continue

            detection_rate = len(peak_data) / n_total

            prior = PeakPrior(
                position_relative=peak_data['pos_relative'].mean(),
                position_sd=peak_data['pos_relative'].std() if len(peak_data) > 1 else 0.05,
                scale=peak_data['scale'].mean(),
                scale_sd=peak_data['scale'].std() if len(peak_data) > 1 else 0.5,
                skewness=peak_data['skewness'].mean(),
                skewness_sd=peak_data['skewness'].std() if len(peak_data) > 1 else 1.0,
                amplitude_relative=peak_data['height_relative'].mean(),
                amplitude_cv=peak_data['height_relative'].std() / peak_data['height_relative'].mean()
                            if peak_data['height_relative'].mean() > 0 else 0.5,
                detection_rate=detection_rate,
                is_anchor=(peak_name in ['free', '80S'])
            )

            # Apply empirical template data
            if templates and peak_name in templates and templates[peak_name] is not None:
                t = templates[peak_name]
                prior.template_x = t['template_x']
                prior.template_mean = t['template_mean']
                prior.template_sd = t['template_sd']
                prior.template_n = t['template_n']
                prior.fit_r2_mean = t['fit_r2_mean']
                prior.fit_r2_sd = t['fit_r2_sd']

            # Apply local step ratio data
            if step_ratios and peak_name in step_ratios:
                sr = step_ratios[peak_name]
                prior.local_step_ratio = sr['ratio_mean']
                prior.local_step_ratio_sd = sr['ratio_sd']
                prior.local_step_n = sr['ratio_n']
                prior.local_predecessor = sr['predecessor']
                prior.local_pre_predecessor = sr['pre_predecessor']

            peaks[peak_name] = prior

        return cls(
            peaks=peaks,
            ruler_mean=ruler_stats['ruler_mean'],
            ruler_sd=ruler_stats['ruler_sd'],
            envelope=envelope,
            n_samples=n_total,
            description=f"Estimated from {n_total} polysome profiles"
        )

    def expected_position(self, peak_name: str, ruler: float,
                          free_position: float) -> float:
        """Get expected absolute position for a peak."""
        if peak_name not in self.peaks:
            raise ValueError(f"Unknown peak: {peak_name}")
        return self.peaks[peak_name].expected_position(ruler, free_position)

    def predicted_position(self, peak_name: str, ruler: float, free_position: float,
                           detected_peaks: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Get best predicted position combining global and local estimates.

        Returns (position, uncertainty_mm).
        Local estimate is preferred when available.
        """
        if peak_name not in self.peaks:
            raise ValueError(f"Unknown peak: {peak_name}")

        prior = self.peaks[peak_name]

        # Try local prediction first
        if detected_peaks is not None:
            local_pred = prior.predicted_position_local(detected_peaks)
            if local_pred is not None:
                return local_pred

        # Fall back to global prediction
        global_pos = prior.expected_position(ruler, free_position)
        global_uncertainty = prior.position_sd * ruler
        return (global_pos, global_uncertainty)

    def position_bounds(self, peak_name: str, ruler: float, free_position: float,
                        n_sd: float = 2.0) -> Tuple[float, float]:
        """Get position search bounds for a peak."""
        if peak_name not in self.peaks:
            raise ValueError(f"Unknown peak: {peak_name}")
        return self.peaks[peak_name].position_bounds(ruler, free_position, n_sd)

    def generate_profile(self, x: np.ndarray, ruler: float, free_position: float,
                         amplitude_80S: float = 1.0,
                         include_peaks: Optional[List[str]] = None) -> np.ndarray:
        """Generate an idealized profile curve."""
        y = np.zeros_like(x, dtype=float)

        peaks_to_include = include_peaks or list(self.peaks.keys())

        for peak_name in peaks_to_include:
            if peak_name in self.peaks:
                y += self.peaks[peak_name].generate_curve(
                    x, ruler, free_position, amplitude_80S
                )

        return y

    def infer_missing_peaks(self, detected_peaks: Dict[str, float],
                            ruler: float, free_position: float) -> Dict[str, dict]:
        """Infer positions and expected properties of missing peaks."""
        inferred = {}

        for peak_name, prior in self.peaks.items():
            if peak_name in detected_peaks:
                continue

            expected_pos = prior.expected_position(ruler, free_position)
            bounds = prior.position_bounds(ruler, free_position, n_sd=2.0)

            inferred[peak_name] = {
                'expected_position': expected_pos,
                'search_bounds': bounds,
                'expected_amplitude': prior.amplitude_relative,
                'detection_rate': prior.detection_rate,
                'confidence': prior.detection_rate * 0.8
            }

        return inferred

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        def clean_value(v):
            """Convert numpy types and handle NaN for JSON."""
            if isinstance(v, np.ndarray):
                return [None if np.isnan(x) else float(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, list):
                return [
                    None if (isinstance(x, float) and np.isnan(x)) else x
                    for x in v
                ]
            return v

        result = {
            'version': self.version,
            'n_samples': self.n_samples,
            'description': self.description,
            'ruler_mean': self.ruler_mean,
            'ruler_sd': self.ruler_sd,
            'peaks': {
                name: {k: clean_value(v) for k, v in asdict(prior).items()}
                for name, prior in self.peaks.items()
            }
        }

        if self.envelope is not None:
            result['envelope'] = {
                'envelope_x': [float(x) for x in self.envelope.envelope_x],
                'envelope_mean': [
                    None if (isinstance(x, float) and np.isnan(x)) else float(x)
                    for x in self.envelope.envelope_mean
                ],
                'envelope_sd': [
                    None if (isinstance(x, float) and np.isnan(x)) else float(x)
                    for x in self.envelope.envelope_sd
                ],
                'envelope_n': [int(x) for x in self.envelope.envelope_n],
                'envelope_n_total': self.envelope.envelope_n_total,
            }

        return result

    def save(self, filepath: str):
        """Save platosome to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Platosome':
        """Load platosome from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        peaks = {}
        for name, prior_data in data['peaks'].items():
            # Filter to only fields PeakPrior accepts (backward compat)
            valid_fields = {f.name for f in PeakPrior.__dataclass_fields__.values()}
            filtered = {k: v for k, v in prior_data.items() if k in valid_fields}
            peaks[name] = PeakPrior(**filtered)

        envelope = None
        if 'envelope' in data:
            env_data = data['envelope']
            envelope = PolysomeEnvelope(
                envelope_x=env_data['envelope_x'],
                envelope_mean=env_data['envelope_mean'],
                envelope_sd=env_data['envelope_sd'],
                envelope_n=env_data['envelope_n'],
                envelope_n_total=env_data.get('envelope_n_total', 0),
            )

        return cls(
            peaks=peaks,
            ruler_mean=data.get('ruler_mean', 24.5),
            ruler_sd=data.get('ruler_sd', 2.3),
            envelope=envelope,
            version=data.get('version', '1.0'),
            n_samples=data.get('n_samples', 0),
            description=data.get('description', '')
        )

    @classmethod
    def load_default(cls) -> 'Platosome':
        """Load the default platosome from the standard location."""
        default_path = Path(__file__).parent.parent / 'data' / 'platosome.json'
        if default_path.exists():
            return cls.load(str(default_path))
        else:
            raise FileNotFoundError(
                f"Default platosome not found at {default_path}. "
                "Run estimate_platosome.py first."
            )

    def summary(self) -> str:
        """Return a formatted summary of the platosome."""
        lines = [
            f"Platosome v{self.version}",
            f"Estimated from {self.n_samples} samples",
            f"Ruler: {self.ruler_mean:.1f} +/- {self.ruler_sd:.1f} mm",
            "",
            "Peak priors:",
            f"{'Peak':<10} {'Pos (rel)':<12} {'Detect%':<10} {'Amp (rel)':<12} {'Scale':<10}",
            "-" * 60
        ]

        for peak_name in PEAK_ORDER:
            if peak_name not in self.peaks:
                continue
            p = self.peaks[peak_name]
            anchor = " *" if p.is_anchor else ""
            lines.append(
                f"{peak_name:<10} {p.position_relative:>5.3f}+/-{p.position_sd:.3f}  "
                f"{p.detection_rate*100:>5.0f}%     "
                f"{p.amplitude_relative:>5.2f}+/-{p.amplitude_cv*100:.0f}%   "
                f"{p.scale:>5.2f}+/-{p.scale_sd:.2f}{anchor}"
            )

        lines.append("")
        lines.append("* = anchor peak (defines ruler)")

        # Templates
        has_templates = any(
            p.template_mean is not None for p in self.peaks.values()
        )
        if has_templates:
            lines.append("")
            lines.append("Empirical templates:")
            for peak_name in PEAK_ORDER:
                if peak_name not in self.peaks:
                    continue
                p = self.peaks[peak_name]
                if p.template_mean is not None:
                    r2_str = (f"R2={p.fit_r2_mean:.3f}+/-{p.fit_r2_sd:.3f}"
                              if p.fit_r2_mean is not None and p.fit_r2_sd is not None
                              else "R2=N/A")
                    lines.append(f"  {peak_name}: {p.template_n} samples, {r2_str}")
                else:
                    lines.append(f"  {peak_name}: no template")

        # Step ratios
        has_ratios = any(
            p.local_step_ratio is not None for p in self.peaks.values()
        )
        if has_ratios:
            lines.append("")
            lines.append("Local step ratios:")
            for peak_name in PEAK_ORDER:
                if peak_name not in self.peaks:
                    continue
                p = self.peaks[peak_name]
                if p.local_step_ratio is not None:
                    lines.append(
                        f"  {peak_name}: ratio={p.local_step_ratio:.3f}"
                        f"+/-{p.local_step_ratio_sd:.3f} "
                        f"(n={p.local_step_n}, "
                        f"{p.local_pre_predecessor}->{p.local_predecessor}->{peak_name})"
                    )

        # Envelope
        if self.envelope is not None:
            lines.append("")
            n_arr = np.array(self.envelope.envelope_n)
            x_arr = np.array(self.envelope.envelope_x)
            coverage_mask = n_arr >= 3
            if np.any(coverage_mask):
                max_x = x_arr[coverage_mask].max()
                min_x = x_arr[coverage_mask].min()
                lines.append(
                    f"Polysome envelope: {self.envelope.envelope_n_total} samples, "
                    f"coverage {min_x:.2f}-{max_x:.2f} ruler units"
                )
            else:
                lines.append(f"Polysome envelope: {self.envelope.envelope_n_total} samples (sparse)")

        return "\n".join(lines)


# =============================================================================
# TEMPLATE EXTRACTION AND COMPUTATION
# =============================================================================

def extract_peak_template(window_x, window_y, fit_params, ruler, grid_x=TEMPLATE_GRID):
    """
    Normalize a single peak's raw window into template coordinates.

    Uses the raw window data for normalization rather than fitted parameters,
    which avoids blow-ups when peaks have small amplitude above local baseline
    (common for peaks on sloping backgrounds like polysomes).

    Args:
        window_x: raw x positions (mm)
        window_y: raw y values (absorbance)
        fit_params: dict with 'mode', 'baseline', 'height' from skew-normal fit
        ruler: ruler length for this sample (mm)
        grid_x: common template x grid in ruler-relative units

    Returns:
        template_y: interpolated normalized y values on grid_x, or None if invalid
    """
    mode = fit_params['mode']
    if ruler <= 0:
        return None

    window_x = np.asarray(window_x)
    window_y = np.asarray(window_y)

    # Use window edges as baseline estimate (more robust than fit baseline
    # for peaks sitting on sloping backgrounds)
    n_edge = max(3, len(window_y) // 10)
    edge_baseline = min(np.mean(window_y[:n_edge]), np.mean(window_y[-n_edge:]))

    # Peak height above this baseline at the mode
    mode_idx = np.argmin(np.abs(window_x - mode))
    peak_val = window_y[mode_idx]
    peak_height = peak_val - edge_baseline

    if peak_height <= 0:
        return None

    # Convert to ruler-relative x centered on mode
    x_rel = (window_x - mode) / ruler

    # Normalize y: subtract baseline, divide by peak height
    y_norm = (window_y - edge_baseline) / peak_height

    # Sort by x_rel (should already be sorted, but be safe)
    sort_idx = np.argsort(x_rel)
    x_rel = x_rel[sort_idx]
    y_norm = y_norm[sort_idx]

    # Interpolate onto common grid, NaN outside data range
    template_y = np.interp(grid_x, x_rel, y_norm, left=np.nan, right=np.nan)

    return template_y


def compute_peak_templates(peak_windows):
    """
    Compute mean empirical templates from collected peak window data.

    Args:
        peak_windows: dict of {peak_name: list of (window_x, window_y, fit_params, ruler)}

    Returns:
        dict of {peak_name: {template_x, template_mean, template_sd, template_n,
                             fit_r2_mean, fit_r2_sd}} or None for peaks with too few samples
    """
    templates = {}

    for peak_name, windows in peak_windows.items():
        all_templates = []
        r2_values = []

        for window_x, window_y, fit_params, ruler in windows:
            # Only use high-quality fits
            r2 = fit_params.get('r_squared')
            if r2 is not None:
                r2_values.append(r2)
                if r2 < 0.3:
                    continue

            # Only use detected (not inferred) peaks
            if fit_params.get('method') == 'inferred':
                continue

            template_y = extract_peak_template(window_x, window_y, fit_params, ruler)
            if template_y is not None:
                all_templates.append(template_y)

        if len(all_templates) < 3:
            templates[peak_name] = None
            continue

        # Stack and compute mean/SD, ignoring NaN at edges
        stacked = np.array(all_templates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            template_mean = np.nanmean(stacked, axis=0)
            template_sd = np.nanstd(stacked, axis=0)

        # Count non-NaN contributions at each grid point
        n_valid = np.sum(~np.isnan(stacked), axis=0)

        # Mask positions where fewer than 3 samples contribute
        sparse_mask = n_valid < 3
        template_mean[sparse_mask] = np.nan
        template_sd[sparse_mask] = np.nan

        templates[peak_name] = {
            'template_x': TEMPLATE_GRID.tolist(),
            'template_mean': template_mean.tolist(),
            'template_sd': template_sd.tolist(),
            'template_n': len(all_templates),
            'fit_r2_mean': float(np.mean(r2_values)) if r2_values else None,
            'fit_r2_sd': float(np.std(r2_values)) if len(r2_values) > 1 else None,
        }

    return templates


# =============================================================================
# LOCAL STEP RATIO COMPUTATION
# =============================================================================

def compute_local_step_ratios(fits_df):
    """
    Compute local step ratios from fitted peak data.

    Only uses high-confidence detected peaks. Applies IQR-based
    outlier filtering to get robust estimates.

    Args:
        fits_df: DataFrame with columns: identifier, peak, mode, confidence, method

    Returns:
        dict of {peak_name: {ratio_mean, ratio_sd, ratio_n,
                             predecessor, pre_predecessor}}
    """
    # Filter to high-confidence detected peaks
    detected = fits_df[
        (fits_df['method'] == 'detected') &
        (fits_df['confidence'] >= 0.9)
    ]

    pivoted = detected.pivot_table(index='identifier', columns='peak', values='mode')

    results = {}

    for peak_name, (pre_pre, pre) in PREDECESSOR_TRIPLETS.items():
        cols = [pre_pre, pre, peak_name]
        if not all(c in pivoted.columns for c in cols):
            continue

        mask = pivoted[cols].notna().all(axis=1)
        subset = pivoted[mask]

        if len(subset) < 3:
            continue

        local_ruler = subset[pre] - subset[pre_pre]
        step = subset[peak_name] - subset[pre]

        # Guard against zero/negative local rulers
        valid_mask = local_ruler > 0.5
        if valid_mask.sum() < 3:
            continue

        ratio = (step / local_ruler)[valid_mask]

        # IQR-based outlier removal
        q1, q3 = ratio.quantile(0.25), ratio.quantile(0.75)
        iqr = q3 - q1
        filtered = ratio[(ratio >= q1 - 1.5 * iqr) & (ratio <= q3 + 1.5 * iqr)]

        if len(filtered) < 3:
            filtered = ratio  # fall back to unfiltered if too few remain

        results[peak_name] = {
            'ratio_mean': float(filtered.mean()),
            'ratio_sd': float(filtered.std()) if len(filtered) > 1 else 0.1,
            'ratio_n': int(len(filtered)),
            'predecessor': pre,
            'pre_predecessor': pre_pre,
        }

    return results


# =============================================================================
# POLYSOME ENVELOPE COMPUTATION
# =============================================================================

def extract_polysome_envelope(distance, absorbance, labeled_peaks, ruler, free_pos,
                              grid_x=ENVELOPE_GRID):
    """
    Extract the polysome region envelope from a single profile.

    Args:
        distance: array of distance values (mm)
        absorbance: array of absorbance values
        labeled_peaks: dict of {peak_name: fit_dict} with 'mode', 'height' etc.
        ruler: ruler length (mm)
        free_pos: free peak position (mm)
        grid_x: envelope grid in ruler-relative units

    Returns:
        envelope_y: interpolated normalized envelope on grid_x, or None
    """
    if '80S' not in labeled_peaks or ruler <= 0:
        return None

    height_80S = labeled_peaks['80S'].get('height', 0)
    baseline_80S = labeled_peaks['80S'].get('baseline', 0)
    peak_height_80S = height_80S - baseline_80S
    if peak_height_80S <= 0:
        return None

    distance = np.asarray(distance)
    absorbance = np.asarray(absorbance)

    # Convert to ruler-relative coordinates
    x_rel = (distance - free_pos) / ruler

    # Normalize y by 80S height (using peak height above baseline)
    y_norm = (absorbance - baseline_80S) / peak_height_80S

    # Smooth to reduce noise
    if len(y_norm) > 15:
        y_smooth = savgol_filter(y_norm, window_length=11, polyorder=3)
    else:
        y_smooth = y_norm

    # Only include the region covered by the grid
    mask = (x_rel >= grid_x[0]) & (x_rel <= grid_x[-1])
    if np.sum(mask) < 10:
        return None

    # Determine signal end: where smoothed signal drops to noise level
    # Use the tail region (beyond 2.5 ruler units) to estimate noise floor
    noise_mask = x_rel > 2.5
    if np.sum(noise_mask) > 10:
        noise_floor = np.median(y_smooth[noise_mask])
        noise_sd = np.std(y_smooth[noise_mask])
        signal_threshold = noise_floor + 2 * noise_sd
    else:
        signal_threshold = 0.02  # fallback: 2% of 80S height

    # Find last x where signal is above threshold
    above_threshold = x_rel[y_smooth > signal_threshold]
    if len(above_threshold) == 0:
        return None
    signal_end = above_threshold.max()

    # Interpolate onto grid, NaN beyond signal end
    envelope_y = np.interp(grid_x, x_rel, y_smooth, left=np.nan, right=np.nan)
    envelope_y[grid_x > signal_end] = np.nan

    return envelope_y


def compute_polysome_envelope(profile_data, grid_x=ENVELOPE_GRID):
    """
    Compute mean polysome envelope from collected profile data.

    Args:
        profile_data: list of (distance, absorbance, labeled_peaks, ruler, free_pos) tuples

    Returns:
        PolysomeEnvelope or None
    """
    all_envelopes = []

    for distance, absorbance, labeled_peaks, ruler, free_pos in profile_data:
        envelope_y = extract_polysome_envelope(
            distance, absorbance, labeled_peaks, ruler, free_pos, grid_x
        )
        if envelope_y is not None:
            all_envelopes.append(envelope_y)

    if len(all_envelopes) < 3:
        return None

    stacked = np.array(all_envelopes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        envelope_mean = np.nanmean(stacked, axis=0)
        envelope_sd = np.nanstd(stacked, axis=0)

    n_valid = np.sum(~np.isnan(stacked), axis=0)

    # Mask sparse positions
    sparse_mask = n_valid < 3
    envelope_mean[sparse_mask] = np.nan
    envelope_sd[sparse_mask] = np.nan

    return PolysomeEnvelope(
        envelope_x=grid_x.tolist(),
        envelope_mean=envelope_mean.tolist(),
        envelope_sd=envelope_sd.tolist(),
        envelope_n=n_valid.tolist(),
        envelope_n_total=len(all_envelopes),
    )


# =============================================================================
# V2 ESTIMATION ENTRY POINT
# =============================================================================

def estimate_platosome_v2(fits_filepath, peak_windows=None, profile_data=None,
                          output_filepath='data/platosome.json'):
    """
    Estimate platosome v2 with templates, local step ratios, and envelope.

    Args:
        fits_filepath: Path to normalized_profiles_fits.tsv
        peak_windows: dict of {peak_name: list of (window_x, window_y, fit_params, ruler)}
        profile_data: list of (distance, absorbance, labeled_peaks, ruler, free_pos)
        output_filepath: Where to save
    """
    import pandas as pd

    fits = pd.read_csv(fits_filepath, sep='\t')

    # Compute ruler for each sample
    rulers = fits[fits['peak'].isin(['free', '80S'])].pivot_table(
        index='identifier', columns='peak', values='mode'
    ).reset_index()
    rulers['ruler'] = rulers['80S'] - rulers['free']
    rulers = rulers.rename(columns={'free': 'free_pos', '80S': 'pos_80S'})

    # Join to compute relative positions
    fits_with_ruler = fits.merge(
        rulers[['identifier', 'free_pos', 'pos_80S', 'ruler']],
        on='identifier'
    )
    fits_with_ruler['pos_relative'] = (
        (fits_with_ruler['mode'] - fits_with_ruler['free_pos']) /
        fits_with_ruler['ruler']
    )

    # Compute height relative to 80S
    height_80S = (fits_with_ruler[fits_with_ruler['peak'] == '80S']
                  .set_index('identifier')['height'])
    fits_with_ruler['height_relative'] = fits_with_ruler.apply(
        lambda row: row['height'] / height_80S.get(row['identifier'], 1.0), axis=1
    )

    ruler_stats = {
        'ruler_mean': rulers['ruler'].mean(),
        'ruler_sd': rulers['ruler'].std()
    }

    # Task 1: Compute templates
    templates = compute_peak_templates(peak_windows) if peak_windows else None

    # Task 2: Compute local step ratios
    step_ratios = compute_local_step_ratios(fits_with_ruler)

    # Task 3: Compute polysome envelope
    envelope = compute_polysome_envelope(profile_data) if profile_data else None

    # Build platosome
    platosome = Platosome.from_data(
        fits_with_ruler, ruler_stats,
        templates=templates, step_ratios=step_ratios, envelope=envelope
    )
    platosome.version = "2.0"
    platosome.description = (
        f"Estimated from {platosome.n_samples} profiles "
        f"with templates, local ratios, and envelope"
    )

    platosome.save(output_filepath)
    print(platosome.summary())
    print(f"\nSaved to {output_filepath}")

    return platosome


# =============================================================================
# V1 ESTIMATION (preserved for backward compatibility)
# =============================================================================

def estimate_platosome_from_fits(fits_filepath: str, output_filepath: str):
    """
    Estimate platosome parameters from a fits TSV file (v1, parametric only).

    Args:
        fits_filepath: Path to normalized_profiles_fits.tsv
        output_filepath: Path to save platosome.json
    """
    import pandas as pd

    fits = pd.read_csv(fits_filepath, sep='\t')

    # Compute ruler for each sample
    rulers = fits[fits['peak'].isin(['free', '80S'])].pivot_table(
        index='identifier', columns='peak', values='mode'
    ).reset_index()
    rulers['ruler'] = rulers['80S'] - rulers['free']
    rulers = rulers.rename(columns={'free': 'free_pos', '80S': 'pos_80S'})

    # Join to compute relative positions
    fits_with_ruler = fits.merge(rulers[['identifier', 'free_pos', 'pos_80S', 'ruler']],
                                  on='identifier')
    fits_with_ruler['pos_relative'] = (
        (fits_with_ruler['mode'] - fits_with_ruler['free_pos']) /
        fits_with_ruler['ruler']
    )

    # Compute height relative to 80S
    height_80S = fits_with_ruler[fits_with_ruler['peak'] == '80S'].set_index('identifier')['height']
    fits_with_ruler['height_relative'] = fits_with_ruler.apply(
        lambda row: row['height'] / height_80S.get(row['identifier'], 1.0), axis=1
    )

    ruler_stats = {
        'ruler_mean': rulers['ruler'].mean(),
        'ruler_sd': rulers['ruler'].std()
    }

    platosome = Platosome.from_data(fits_with_ruler, ruler_stats)
    platosome.save(output_filepath)

    print(platosome.summary())
    print(f"\nSaved to {output_filepath}")

    return platosome


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Estimate or display platosome')
    parser.add_argument('--estimate', '-e', metavar='FITS_FILE',
                        help='Estimate from fits file (v1, parametric only)')
    parser.add_argument('--output', '-o', default='data/platosome.json',
                        help='Output file for estimated platosome')
    parser.add_argument('--show', '-s', metavar='JSON_FILE',
                        help='Show summary of platosome file')

    args = parser.parse_args()

    if args.estimate:
        estimate_platosome_from_fits(args.estimate, args.output)
    elif args.show:
        plato = Platosome.load(args.show)
        print(plato.summary())
    else:
        parser.print_help()
