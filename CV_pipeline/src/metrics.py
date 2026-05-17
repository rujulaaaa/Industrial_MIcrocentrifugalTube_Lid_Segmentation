"""
Evaluated required metrics.
Computes:
- precision
- recall
- F1
- orientation error
- confidence statistics
"""

from typing import List, Tuple, Optional
from src.utils import angle_diff
import numpy as np

def compute_metrics(
    matches: List[Tuple[int, int]],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
    pred_angles: List[float],
    gt_angles: List[float],
    pred_confidences: Optional[List[float]] = None,
) -> dict:

    # -------------------------------- Metrics ---------------------------------

    tp = len(matches)

    fp = len(unmatched_pred)

    fn = len(unmatched_gt)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ---------------------------- Angle Errors --------------------------------

    angle_errors = [

        angle_diff(
            pred_angles[p],
            gt_angles[g]
        )

        for p, g in matches
    ]

    # ----------------------- Confidence Statistics ----------------------------

    if pred_confidences is not None:

        matched_confidences = [
            pred_confidences[p]
            for p, _ in matches
        ]
    else:
        matched_confidences = []

    return {

        # detection part
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        # orientation part
        "mean_angle_error_deg":
            float(np.mean(angle_errors))
            if len(angle_errors) > 0
            else None,

        "median_angle_error_deg":
            float(np.median(angle_errors))
            if len(angle_errors) > 0
            else None,

        "angle_errors":
            angle_errors,

        # confidence part
        "mean_confidence":
            float(np.mean(matched_confidences))
            if len(matched_confidences) > 0
            else None,
    }

def print_metrics(
    m: dict
) -> None:

    print(
        f"TP={m['tp']}  "
        f"FP={m['fp']}  "
        f"FN={m['fn']}"
    )

    print(
        f"Precision={m['precision']:.3f}  "
        f"Recall={m['recall']:.3f}  "
        f"F1={m['f1']:.3f}"
    )

    if m["mean_angle_error_deg"] is not None:

        print(
            f"Mean angle error: "
            f"{m['mean_angle_error_deg']:.1f} deg"
        )

        print(
            f"Median angle error: "
            f"{m['median_angle_error_deg']:.1f} deg"
        )

    else:

        print(
            "No matched pairs "
            "- angle error undefined."
        )

    if m["mean_confidence"] is not None:

        print(
            f"Mean confidence: "
            f"{m['mean_confidence']:.2f}"
        )