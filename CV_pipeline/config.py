"""
Central config.

All parameters here correspond directly to:
- lid geometry
- Hough detection behaviour
- orientation sampling
- evaluation thresholds
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:

    # ======================== DETECTION PART =======================

    # Expected lid radius in pixels.
    min_radius: int = 14
    max_radius: int = 17

    # HoughCircles parameters
    hough_dp: float = 1.2
    hough_min_dist: int = 25

    # Canny high threshold internally used by HoughCircles
    hough_param1: int = 80

    # Circle accumulator threshold.
    hough_param2: int = 14

    # Threshold for rejecting dark rack holes
    min_mean_intensity: float = 50.0

    # ====================== OUTLIER FILTERING PART ====================

    # Adaptive centroid-distance filtering:
    distance_std_factor: float = 0.5

    # Tray prior: expected number of tubes.
    min_tubes: int = 3
    max_tubes: int = 6

    # ================= ORIENTATION ESTIMATION PART ======================

    # Angular sampling resolution.
    n_angle_bins: int = 72

    # Gaussian smoothing over angular profile.
    radial_smooth_sigma: float = 2.0

    # Sample slightly outside detected circle
    # to capture tab protrusion intensity.
    orientation_radius_offset: int = 4

    # Radial band sampled around circle.
    orientation_band_start: int = -2
    orientation_band_end: int = 5

    # Peak detection prominence.
    orientation_peak_prominence: float = 1.0

    # =================== EVALUATION/ MATCHING PART ===================

    # Detection counts as TP if center within this distance.
    center_dist_threshold: float = 25.0

    # ======================= VISUALIZATION PART =======================

    arrow_length: int = 35
    output_dir: Path = Path("outputs")

Config = Config()