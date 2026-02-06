#!/usr/bin/env python3
"""
Platosome: A Platonic (idealized) polysome profile model.

The platosome serves as a prior for peak identification and quantification.
It captures the expected structure of polysome profiles with:
- Relative positions anchored to the ruler (free=0, 80S=1)
- Shape parameters (skew-normal) with variation estimates
- Relative amplitudes with coefficients of variation

Usage:
    from platosome import Platosome

    # Load the standard platosome
    plato = Platosome.load_default()

    # Generate an idealized profile
    profile = plato.generate_profile(ruler=24.5, free_position=4.0)

    # Use as prior for peak detection
    expected_pos = plato.expected_position('40S', ruler=24.5, free_position=4.0)
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from scipy.stats import skewnorm


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

    def expected_position(self, ruler: float, free_position: float) -> float:
        """Compute expected absolute position given ruler and free position."""
        return free_position + self.position_relative * ruler

    def position_bounds(self, ruler: float, free_position: float,
                        n_sd: float = 2.0) -> Tuple[float, float]:
        """Compute position bounds (mean ± n_sd) in absolute units."""
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


# Peak order constant (class-level)
PEAK_ORDER = [
    'free', '40S', '60S', '80S', '2-some', '3-some',
    '4-some', '5-some', '6-some', '7-some', '8-some'
]


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
    def from_data(cls, fits_df, ruler_stats: dict) -> 'Platosome':
        """
        Estimate platosome parameters from fitted peak data.

        Args:
            fits_df: DataFrame with columns: identifier, peak, mode, scale,
                     skewness, height, plus computed: pos_relative, height_relative
            ruler_stats: dict with ruler_mean, ruler_sd
        """
        peaks = {}

        for peak_name in PEAK_ORDER:
            peak_data = fits_df[fits_df['peak'] == peak_name]
            if len(peak_data) == 0:
                continue

            n_total = fits_df['identifier'].nunique()
            detection_rate = len(peak_data) / n_total

            peaks[peak_name] = PeakPrior(
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

        return cls(
            peaks=peaks,
            ruler_mean=ruler_stats['ruler_mean'],
            ruler_sd=ruler_stats['ruler_sd'],
            n_samples=n_total,
            description=f"Estimated from {n_total} polysome profiles"
        )

    def expected_position(self, peak_name: str, ruler: float,
                          free_position: float) -> float:
        """Get expected absolute position for a peak."""
        if peak_name not in self.peaks:
            raise ValueError(f"Unknown peak: {peak_name}")
        return self.peaks[peak_name].expected_position(ruler, free_position)

    def position_bounds(self, peak_name: str, ruler: float, free_position: float,
                        n_sd: float = 2.0) -> Tuple[float, float]:
        """Get position search bounds for a peak."""
        if peak_name not in self.peaks:
            raise ValueError(f"Unknown peak: {peak_name}")
        return self.peaks[peak_name].position_bounds(ruler, free_position, n_sd)

    def generate_profile(self, x: np.ndarray, ruler: float, free_position: float,
                         amplitude_80S: float = 1.0,
                         include_peaks: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate an idealized profile curve.

        Args:
            x: Distance values (mm)
            ruler: Ruler length (free to 80S distance)
            free_position: Absolute position of free peak
            amplitude_80S: Height of 80S peak (sets overall scale)
            include_peaks: List of peaks to include (default: all)

        Returns:
            Array of absorbance values
        """
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
        """
        Infer positions and expected properties of missing peaks.

        Args:
            detected_peaks: Dict of {peak_name: detected_position}
            ruler: Ruler length
            free_position: Free peak position

        Returns:
            Dict of inferred peaks with position, confidence, etc.
        """
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
                'confidence': prior.detection_rate * 0.8  # Discount for being missing
            }

        return inferred

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'n_samples': self.n_samples,
            'description': self.description,
            'ruler_mean': self.ruler_mean,
            'ruler_sd': self.ruler_sd,
            'peaks': {name: asdict(prior) for name, prior in self.peaks.items()}
        }

    def save(self, filepath: str):
        """Save platosome to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Platosome':
        """Load platosome from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        peaks = {
            name: PeakPrior(**prior_data)
            for name, prior_data in data['peaks'].items()
        }

        return cls(
            peaks=peaks,
            ruler_mean=data.get('ruler_mean', 24.5),
            ruler_sd=data.get('ruler_sd', 2.3),
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
            f"Ruler: {self.ruler_mean:.1f} ± {self.ruler_sd:.1f} mm",
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
                f"{peak_name:<10} {p.position_relative:>5.3f}±{p.position_sd:.3f}  "
                f"{p.detection_rate*100:>5.0f}%     "
                f"{p.amplitude_relative:>5.2f}±{p.amplitude_cv*100:.0f}%   "
                f"{p.scale:>5.2f}±{p.scale_sd:.2f}{anchor}"
            )

        lines.append("")
        lines.append("* = anchor peak (defines ruler)")

        return "\n".join(lines)


def estimate_platosome_from_fits(fits_filepath: str, output_filepath: str):
    """
    Estimate platosome parameters from a fits TSV file.

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
                        help='Estimate from fits file')
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
