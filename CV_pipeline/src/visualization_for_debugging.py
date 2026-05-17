"""
Visualisation utilities.

Provides:
- detection overlays
- orientation arrows
- GT comparison
- confidence visualization
"""

import cv2
import numpy as np
from typing import Tuple
import sys, os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(__file__))
)
from config import Config


PRED_COLOR = (0, 200, 255)
GT_COLOR = (0, 255, 80)

FP_COLOR = (0, 60, 255)
FN_COLOR = (200, 200, 0)

CONF_LOW_COLOR = (60, 60, 255)

def draw_detections(
    image: np.ndarray,
    detections,
    results=None,
    gt_rows=None,
    unmatched_pred=None,
    unmatched_gt=None,
) -> np.ndarray:

    """
    Draw:
    - predicted circles
    - orientation arrows
    - GT arrows
    - FP/FN markers
    """

    out = image.copy()

    if out.ndim == 2:
        out = cv2.cvtColor(
            out,
            cv2.COLOR_GRAY2BGR
        )

    if gt_rows is not None:

        for _, row in gt_rows.iterrows():

            cx = row["center_x"]
            cy = row["center_y"]
            ang = row["angle_deg"]

            cv2.circle(
                out,
                (int(cx), int(cy)),
                4,
                GT_COLOR,
                -1
            )

            _draw_orientation_arrow(
                out,
                cx,
                cy,
                ang,
                GT_COLOR,
                scale=1.2
            )

    if results is not None:

        for i, r in enumerate(results):

            is_fp = (
                unmatched_pred is not None
                and
                i in unmatched_pred
            )

            color = FP_COLOR if is_fp else PRED_COLOR

            if r.confidence < 0.15:
                color = CONF_LOW_COLOR

            # detected circle
            cv2.circle(
                out,
                (int(r.centre_x), int(r.centre_y)),
                int(r.radius),
                color,
                2
            )

            # center
            cv2.circle(
                out,
                (int(r.centre_x), int(r.centre_y)),
                3,
                color,
                -1
            )

            # orientation arrow
            _draw_orientation_arrow(
                out,
                r.centre_x,
                r.centre_y,
                r.angle_deg,
                color
            )

            # label
            label = (
                f"{r.angle_deg:.0f} deg "
                f"c={r.confidence:.2f}"
            )

            cv2.putText(
                out,
                label,
                (
                    int(r.centre_x) + 5,
                    int(r.centre_y) - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1
            )

    if (
        gt_rows is not None
        and
        unmatched_gt is not None
    ):

        for j in unmatched_gt:

            row = gt_rows.iloc[j]

            cx = int(row["center_x"])
            cy = int(row["center_y"])

            cv2.drawMarker(
                out,
                (cx, cy),
                FN_COLOR,
                cv2.MARKER_CROSS,
                20,
                2
            )

    return out

def draw_roi_debug(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    angle_deg: float,
) -> np.ndarray:

    """
    Local ROI visualization around one lid.
    """

    pad = int(radius * 1.8)

    x0 = max(0, int(cx) - pad)
    y0 = max(0, int(cy) - pad)

    x1 = min(image.shape[1], int(cx) + pad)
    y1 = min(image.shape[0], int(cy) + pad)

    roi = image[y0:y1, x0:x1].copy()

    if roi.ndim == 2:
        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_GRAY2BGR
        )

    local_cx = cx - x0
    local_cy = cy - y0

    # detected circle
    cv2.circle(
        roi,
        (int(local_cx), int(local_cy)),
        int(radius),
        PRED_COLOR,
        2
    )

    # orientation arrow
    _draw_orientation_arrow(
        roi,
        local_cx,
        local_cy,
        angle_deg,
        PRED_COLOR
    )

    return roi

def _draw_orientation_arrow(
    img: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    color: Tuple[int, int, int],
    scale: float = 1.0,
) -> None:

    """
    Draw orientation arrow.
    """

    length = Config.arrow_length * scale

    rad = np.radians(angle_deg)

    ex = int(
        cx + length * np.cos(rad)
    )

    ey = int(
        cy + length * np.sin(rad)
    )

    cv2.arrowedLine(
        img,
        (int(cx), int(cy)),
        (ex, ey),
        color,
        2,
        tipLength=0.3
    )