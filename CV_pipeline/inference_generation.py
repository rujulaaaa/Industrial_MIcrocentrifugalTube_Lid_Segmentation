"""
Purpose: To run detection of tubes and orientation vector calculation on one image or a directory.
-----
Usage (As per Linux)
  python3 run_pipeline.py 
  python3 run_pipeline.py data/ --save_viz          # write annotated images to outputs/
  python3 run_pipeline.py data/ --save_viz --debug  # also save per-lid ROI crops
"""

import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from config import Config
from typing import List
from dataclasses import dataclass
from src.utils import load_image
from src.visualization_for_debugging import draw_detections
from src.tubelid_detection import tube_contour_detection
from src.tubelid_orientation import orientation_vector_estimation
from src.tubelid_detection import tube_contour_detection, TubeLidDetection
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@dataclass
class TubeLidResult:
    image_name: str
    centre_x: float
    centre_y: float
    radius: float
    angle_deg: float
    confidence: float

# -------------------- Args for creating CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input",
        nargs="?",
        default="data/TubeDetectionDataset_ZeonSystems2026/images",
        help="Image file or directory"
    )
    p.add_argument(
        "--save_viz",
        action="store_true",
        help="Save annotated images"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save ROI debug crops"
    )
    return p.parse_args()


# --------- Image collection consistant across (directory/ image path) input --------

def collect_images(path: Path):
    if path.is_dir():
        return (
            sorted(path.glob("*.png"))
            +
            sorted(path.glob("*.jpg"))
            +
            sorted(path.glob("*.jpeg"))
        )
    return [path]

def run_on_image(image: np.ndarray, image_name: str = "") -> List[TubeLidResult]:
    """
    Full per-image pipeline.  
    Returns one LidResult per detected lid.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detections = tube_contour_detection(image)
    detections = _NMS(detections)

    results: List[TubeLidResult] = []
    for det in detections:
        angle_deg, confidence = orientation_vector_estimation(image, det.cx, det.cy, det.radius)
        results.append(TubeLidResult(
            image_name=image_name,
            centre_x=det.cx,
            centre_y=det.cy,
            radius=det.radius,
            angle_deg=angle_deg,
            confidence=confidence,
        ))

    return results


# ------------- Non-maximum suppression for duplicate detections ------------

def _NMS(detections: List[TubeLidDetection], overlap_ratio: float = 0.4) -> List[TubeLidDetection]:
    """
    Remove duplicate tubes detected whose centres are within the same overlap_ratio of 0.4.
    It eventually keeps the detection with a large contour area.
    """
    if len(detections) <= 1:
        return detections

    # Sorts detections from biggest to smallest based on radius
    detections = sorted(detections, key=lambda d: d.radius, reverse=True)
    kept_detections = []
    for det in detections:
        if_close = any(
            np.hypot(det.cx - k.cx, det.cy - k.cy) < overlap_ratio * (det.radius + k.radius)
            for k in kept_detections
        )
        if not if_close:
            kept_detections.append(det)
    return kept_detections

# ------------------------------- Main function -------------------------------

def main():

    args = parse_args()

    # Output directory created for saving files 
    Config.output_dir.mkdir(
        exist_ok=True
    )

    paths = collect_images(
        Path(args.input)
    )
    if len(paths) == 0:
        print("No images found.")
        sys.exit(1)

    all_rows = []

    # ---------- For every image run the complete CV pipeline ----------------------
    for img_path in paths:

        image = load_image(img_path)
        if image is None:
            continue

        # ----------- Run pipeline -------------
        results = run_on_image(
            image,
            img_path.name
        )

        print(
            f"\n{img_path.name}"
            f" -> {len(results)} lid(s)"
        )

        # ------ Print results on terminal --------
        for r in results:
            print(
                f"center=({r.centre_x:.1f}, {r.centre_y:.1f})  "
                f"angle={r.angle_deg:.1f} deg  "
                f"conf={r.confidence:.2f}"
            )
            all_rows.append({
                "image":
                    r.image_name,
                "center_x":
                    round(r.centre_x, 1),
                "center_y":
                    round(r.centre_y, 1),
                "angle_deg":
                    round(r.angle_deg, 1),
                "confidence":
                    round(r.confidence, 3),
            })

        # --------------- Visualization ----------------
        if args.save_viz:
            detections = tube_contour_detection(image)
            viz = draw_detections(
                image,
                detections,
                results=results
            )
            out_path = (
                Config.output_dir
                /
                f"viz_{img_path.name}"
            )
            cv2.imwrite(
                str(out_path),
                viz
            )
            print(
                f"Saved visualization at {out_path}"
            )

    # ----------- Save CSV with predictions -----------------
    if len(all_rows) > 0:
        out_csv = (
            Config.output_dir
            / "predictions.csv"
        )
        pd.DataFrame(all_rows).to_csv(
            out_csv,
            index=False
        )
        print(
            f"\nPredictions saved at {out_csv}"
        )

if __name__ == "__main__":

    main()