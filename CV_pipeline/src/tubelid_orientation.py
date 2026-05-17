"""
Estimates tube lid orientation by analysing the angular
intensity profile around the detected circular lid region.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple
from scipy.signal import find_peaks
import cv2

def orientation_vector_estimation(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Tuple[float, float]:

    # Now since the contours are supposed to be circular, we can use a different approach.
    # The lid joint and lid tab are protrusions out of this circle
    # One can find centroid of the contour detections and find intensity profile around it
    # This will result in 2 significant maximas occuring acros the angular bins at joint and tab
   
    # ------------------ Angular Intensity Profile based Logic----------------------------------------------

    # ---------- Step 1: Intensity Profile Generation -----------------

    # Only brightness variation is to be considered henceforth, thus changed to grayscale to save computation time
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Divided complete circle from 0 to 360 degrees into 
    no_of_divisions = 72
    angular_bins = np.linspace(
        0,
        2 * np.pi,
        no_of_divisions,
        endpoint=False
    )

    intensity_profile = []

    # Sample radius taken to be slightly out of the actual circle since protrusions exist out of circular boundary
    sample_radius = radius + 4

    for theta in angular_bins:
       
        intensity_values = []

        # Sampled values along a radius band between -2 to 4
        for offset in range(-2, 5):
            r = sample_radius + offset
            # Got cartesian coordinates for each pixel
            x = int(cx + r * np.cos(theta))
            y = int(cy + r * np.sin(theta))

            if (
                x < 0 or x >= gray_image.shape[1]
                or
                y < 0 or y >= gray_image.shape[0]
            ):
                continue

            intensity_values.append(float(gray_image[y, x]))

        if len(intensity_values) == 0:
            intensity_profile.append(0)
        else:
            intensity_profile.append(np.mean(intensity_values))

    intensity_profile = np.array(intensity_profile)

    # Angular bins for finding the average of earlier bins to decide upon an angle for that bin range
    bin_edges = np.linspace(
        0,
        2* np.pi,
        no_of_divisions + 1
    )
    bin_width = bin_edges[1] - bin_edges[0]

    # Many maximas were observed earlier due to additional noise
    # ------------ Step 2: Smoothened the profile to remove it --------------
    smoothened_profile = gaussian_filter1d(
        np.tile(intensity_profile, 3), sigma=2.0
    )[no_of_divisions : 2 * no_of_divisions]

    # ------------ Step 3: Maximas for intensity found ----------------------
    # Found maximas for intensity around the centroid of the circular detected contour
    peaks, properties = find_peaks(
        smoothened_profile,
        prominence=1.0
    )
    if len(peaks) == 0:
        peak_bin = int(np.argmax(smoothened_profile))
    else:
        # Maximas priority is decided based on term taking into account both prominence and width
        max_acc_to_priority = []

        for i, peak in enumerate(peaks):

            prominence = properties["prominences"][i]

            # Created a window of peak+-2 around the peak to estimate its width
            left = max(0, peak - 2)
            right = min(len(smoothened_profile)-1, peak + 2)
            width = np.sum(
                smoothened_profile[left:right+1]
                > (smoothened_profile[peak] - prominence * 0.5)
            )

            # A score calculated for each peak
            score = prominence * width

            max_acc_to_priority.append(score)

        best_idx = int(np.argmax(max_acc_to_priority))

        peak_bin = peaks[best_idx]

    # -------------------- Step 4: Final Orientation -------------------------

    maxima_angle_rad = (bin_edges[peak_bin] + bin_width / 2)
    maxima_angle_deg = float(np.degrees(maxima_angle_rad) % 360)

    # ---------------- Confidence calculated for both maximas (lid and joint)
    mean_intensity = float(np.mean(smoothened_profile))

    confidence = float(
        np.clip(
            (smoothened_profile[peak_bin] - mean_intensity)
            / (mean_intensity + 1e-6),
            0,
            1
        )
    )

    # The strongest maxima with more confidence is assumed to be tab direction

    return maxima_angle_deg, confidence
