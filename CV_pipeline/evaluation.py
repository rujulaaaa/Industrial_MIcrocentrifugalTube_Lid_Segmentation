"""
Evaluate predictions against ground truth.

Supports:
- standard evaluation
- robustness evaluation via augmentation

Examples
--------
python evaluate.py --gt annotations.csv --images data/images/

python evaluate.py --gt annotations.csv --images data/images/ --save_viz

python evaluate.py --gt annotations.csv --images data/images/ --augment
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from src.matching import match_detections
from src.matching import match_detections
from src.metrics import (
    compute_metrics,
    print_metrics
)
from src.visualization_for_debugging import draw_detections
from src.utils import (
    load_image,
    load_gt
)
from inference_generation import run_on_image

# -------------------------- Args -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gt",
        required=True,
        help="Ground truth CSV"
    )
    p.add_argument(
        "--images",
        required=True,
        help="Image directory"
    )
    p.add_argument(
        "--dist_threshold",
        type=float,
        default=Config.center_dist_threshold
    )
    p.add_argument(
        "--save_viz",
        action="store_true"
    )
    return p.parse_args()

def main():

    args = parse_args()

    Config.output_dir.mkdir(
        exist_ok=True
    )

    gt_df = load_gt(args.gt)

    image_dir = Path(args.images)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_angle_errors = []

    per_image_results = []

    # --------------------- Process Images --------------------------

    for img_path in sorted(image_dir.glob("*.png")):

        gt_rows = gt_df[
            gt_df["image"] == img_path.name
        ]

        if gt_rows.empty:
            continue

        image = load_image(img_path)

        if image is None:
            continue

        # --------------------- Complete Pipeline ----------------------
        results = run_on_image(
            image,
            img_path.name
        )

        pred_centers = [
            (r.centre_x, r.centre_y)
            for r in results
        ]

        pred_angles = [
            r.angle_deg
            for r in results
        ]

        pred_confidences = [
            r.confidence
            for r in results
        ]

        gt_centers = list(
            zip(
                gt_rows["center_x"],
                gt_rows["center_y"]
            )
        )

        gt_angles = list(
            gt_rows["angle_deg"]
        )

        # ------------------------------- Matching --------------------------------

        matches, unmatched_pred, unmatched_gt = (
            match_detections(
                pred_centers,
                gt_centers,
                args.dist_threshold
            )
        )

        # ------------------------ Metrics Calculation per Image ------------------

        m = compute_metrics(
            matches,
            unmatched_pred,
            unmatched_gt,
            pred_angles,
            gt_angles,
            pred_confidences
        )

        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]

        all_angle_errors.extend(
            m["angle_errors"]
        )

        print(f"\n{img_path.name}")

        print_metrics(m)

        per_image_results.append({

            "image":
                img_path.name,

            **{
                k: v
                for k, v in m.items()
                if k != "angle_errors"
            }
        })

        # ------------------ Save Visualization ------------------

        if args.save_viz:
            viz = draw_detections(
                image,
                [],
                results=results,
                gt_rows=gt_rows,
                unmatched_pred=unmatched_pred,
                unmatched_gt=unmatched_gt,
            )
            out_path = (
                Config.output_dir
                /
                f"eval_{img_path.name}"
            )
            cv2.imwrite(
                str(out_path),
                viz
            )

    # ------------------------- Aggregate Results -------------------------------
    precision = (total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0)
    recall = (total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0)
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    print("\n" + "=" * 40)
    print("AGGREGATE RESULTS")
    print(
        f"TP={total_tp}  "
        f"FP={total_fp}  "
        f"FN={total_fn}"
    )
    print(
        f"Precision={precision:.3f}  "
        f"Recall={recall:.3f}  "
        f"F1={f1:.3f}"
    )
    if len(all_angle_errors) > 0:
        print(
            f"Mean angle error: "
            f"{np.mean(all_angle_errors):.1f} deg"
        )
        print(
            f"Median angle error: "
            f"{np.median(all_angle_errors):.1f} deg"
        )
        print(
            f"% within 15 deg: "
            f"{np.mean(np.array(all_angle_errors) < 15)*100:.1f}%"
        )
        _save_angle_error_plot(
            all_angle_errors
        )

    # ------------------------- Save as CSV ----------------------------------

    if len(per_image_results) > 0:
        pd.DataFrame(
            per_image_results
        ).to_csv(
            Config.output_dir
            /
            "evaluation_per_image.csv",
            index=False
        )


# -------------------------------- Evaluated histogram ------------------------------

def _save_angle_error_plot(
    errors
):
    fig, ax = plt.subplots(
        figsize=(6, 3)
    )
    ax.hist(
        errors,
        bins=18,
        range=(0, 180)
    )
    ax.axvline(
        np.mean(errors),
        linestyle="--",
        label=f"mean {np.mean(errors):.1f} deg"
    )
    ax.set_xlabel(
        "Angle error (deg)"
    )
    ax.set_ylabel(
        "Count"
    )
    ax.set_title(
        "Orientation Error Distribution"
    )
    ax.legend()
    fig.tight_layout()
    out = (
        Config.output_dir
        /
        "angle_error_hist.png"
    )
    fig.savefig(
        str(out),
        dpi=120
    )
    plt.close(fig)
    print(f"Histogram saved at {out}")

if __name__ == "__main__":

    main()