"""
Matches predicted tube detections with ground-truth tube locations
using optimal one-to-one assignment based on centre distance.

Predictions that do not match any ground-truth tube are treated as
false positives, while unmatched ground-truth tubes are treated as
false negatives.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

Centers = List[Tuple[float, float]]
Matches = List[Tuple[int, int]]   
Indices = List[int]


def match_detections(
    pred_centers: Centers,
    gt_centers: Centers,
    dist_threshold: float,
) -> Tuple[Matches, Indices, Indices]:
    """
    Returns
    -------
    matches        : list of (pred_idx, gt_idx) pairs within threshold
    unmatched_pred : pred indices with no valid GT match  → false positives
    unmatched_gt   : GT indices with no valid pred match  → false negatives
    """
    if not pred_centers:
        return [], [], list(range(len(gt_centers)))
    if not gt_centers:
        return [], list(range(len(pred_centers))), []

    pred_arr = np.array(pred_centers)          
    gt_arr = np.array(gt_centers)              

    cost = np.linalg.norm(
        pred_arr[:, None, :] - gt_arr[None, :, :], axis=2
    )  

    cost[cost > dist_threshold] = 1e8

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pred, matched_gt = set(), set()
    matches: Matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e9:
            matches.append((r, c))
            matched_pred.add(r)
            matched_gt.add(c)

    unmatched_pred = [i for i in range(len(pred_centers)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_centers)) if i not in matched_gt]

    return matches, unmatched_pred, unmatched_gt
