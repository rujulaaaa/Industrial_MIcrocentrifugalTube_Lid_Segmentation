"""
Detects tube lids using Hough Circle detection
This is followed by filtering and other spatial outlier removal
to reduce false detections.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List
from sklearn.cluster import DBSCAN

# ------------------------ Config --------------------------
class config:
    # hyperparameters for expected min and max radius of tube lids
    min_radius = 14
    max_radius = 17

# ------------ Data Structure for Detected Tube Lid ------------
@dataclass
class TubeLidDetection:
    cx: float
    cy: float
    radius: float
    contour: np.ndarray = field(repr=False)


# -------------------------- Primary Tube Detection pipeline------------------------------

def tube_contour_detection(image: np.ndarray):

    # ------------------------ Preprocessing ------------------------
    # ----------------------- Lab color space -----------------------
    # Used because L channel contains brightness information
    # Had to use this instead of normal grayscale preprocessing 
    # because the intensities of the circles formed and the background seemed the same visually
    lab_form_output = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_form_output)

    # ----------------------- CLAHE on L channel -----------------------
    # Used because using L channel alone was not sufficient
    # The local contrast between the circles and the background was insufficient
    clahe_output = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    L = clahe_output.apply(L)

    # ----------------------- Median Blur -----------------------
    # Blur used again, since CLAHE increased the local contrast specially on the texture of table
    # Noise addition occurred 
    blurred_output = cv2.medianBlur(L, 5)

    # ------------------- Hough circle detection --------------------
    circles = cv2.HoughCircles(
        blurred_output,
        cv2.HOUGH_GRADIENT, #as it uses gradient information which adds to circle detection

        dp=1.2, #controls resolution of voting accumulator relative to i/p image resolution
        minDist=25, # min distance between centres of 2 formed circles, tuned over 8-9 images

        param1=80, # kept high for lesser noise since the wooden table shown had lot of texture
        param2=14, # increased for increased confidence of circle detection

        minRadius=config.min_radius,
        maxRadius=config.max_radius
    )

    detections: List[TubeLidDetection] = []

    # ----------------------- Further processing of circles for better contours ------------------------
    if circles is not None:

        # List of detected circles with [centre's x coordinate, centre's y coordinate, radius, radius]
        circles = np.round(circles[0]).astype(int)

        for (x_coordinate, y_coordinate, radius) in circles:
            # Created a synthetic circular contour from detected params
            # Done by approximating circle using many small points around the boundary
            contour = cv2.ellipse2Poly(
                (x_coordinate, y_coordinate), #centre of circle
                (radius, radius),             #axes
                0,                            #rotational angle
                0,                            #start angle in degrees
                360,                          #end angle in degrees 
                10                            #angular step size
            )

            contour = contour.reshape((-1, 1, 2))

            # Even after estimation of Hough circles
            # The contours were only formed where edges of tube lids could be straight away told to be different than the background
            # Meaning correct circles formed only where the black tube holder was in the background, 
            # and a striking difference between the background and the tube intensity was present
            # The table surface and tubes had insignificant intensity fluctuations, given the lightening
            # The circles were only formed using edges, so tried looking at the region filling inside the detected circles
            # ----------------------- Appearance Filter -------------------------

            # Created black circular mask same size as image
            black_circular_mask = np.zeros(
                L.shape,
                dtype=np.uint8
            )

            # Filled white circle created over detected lid region
            cv2.circle(
                black_circular_mask,
                (x_coordinate, y_coordinate),
                int(radius * 0.7),
                255,
                -1
            )

            # Average intensity inside masked circular region calculated
            mean_intensity = cv2.mean(
                L,
                mask=black_circular_mask
            )[0]

            # Rejected dark circular regions 
            if mean_intensity < 50:
                continue


            detections.append(
                TubeLidDetection(
                    cx=float(x_coordinate),
                    cy=float(y_coordinate),
                    radius=float(radius),
                    contour=contour
                )
            )

    # Now fake detections were also present. 
    # But since the tubes are present in a holder, they are within a threshold distance of each other
    # Considered and tried using clustering methods like DBSCAN
    # But since no irregular spatial group with very sparse data and only 4-8 detected circles, it could not be used
    # So instead went with using distance from centroid of detections as threshold to filter fake detections

    # ----------------- Outlier Removal -------------------

    if len(detections) > 0:

        points = np.array([
            [d.cx, d.cy]
            for d in detections
        ])

        # centroid of detections
        centroid = np.mean(points, axis=0)

        # Eearlier for the distance threshold filter, had kept the thrshold as a fixed number
        # Did not work as expected with every image
        # Understood that it must be as per the sparsity of the distribution of the circles
        # So instead used the simple approach of modeling the distance distribution from centroid as a Gaussian one
        # And finally points too far from the normal spread are regarded as outliers
        distances = []

        for d in detections:

            dist = np.linalg.norm(
                np.array([d.cx, d.cy]) - centroid
            )

            distances.append(dist)

        distances = np.array(distances)

        # Threshold is based upon distribution of distances and is no longer taken to be constant
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        threshold = max(
            mean_dist + 0.8 * std_dist,
            80
            )
        # Changed -0.5 * std_dist to -0.8 * std_dist
        # Because having 0.5 made the removal too aggressive
        # So incase of smaller number of lids, the actual detected tubes were also not considered earlier

        filtered = []

        for d, dist in zip(detections, distances):

            if dist < threshold:
                filtered.append(d)

        detections = filtered

        # Also, found another filtering method
        # Mentioned in Problem Statement that each tube holder rack will hold 3 to 6 tubes
        # So if number of detected circles are less than 3, search for more circles
        # If they are more than 6, then they are likely to be false positives
        # Now the actual tubes can be 6, and they tend to form a spatially structured group since the tray has a structure
        # So calculated the centroid of all the detections, the centroid tends to lie near the centre of this spatially structured group
        # SO the first 6 tubes close to the centroid are considered as actual ones
        # ------ For enforcing tube number constraint for being lesser than 6 -------

        if len(detections) > 6:

            points = np.array([
                [d.cx, d.cy]
                for d in detections
            ])

            centroid = np.mean(points, axis=0)

            distances = []

            # Distance from centroid calculated
            for d in detections:
                dist = np.linalg.norm(
                    np.array([d.cx, d.cy]) - centroid
                )

                distances.append((dist, d))

            # Sorted acc to proximity from centroid
            distances.sort(key=lambda x: x[0])

            # Closest proximal 6 detections considered to be true ones
            detections = [d for _, d in distances[:6]]

    return detections


