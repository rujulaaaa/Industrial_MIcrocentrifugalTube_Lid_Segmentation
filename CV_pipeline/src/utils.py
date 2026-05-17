"""
Lightweight utilities.  Kept genuinely minimal — only things reused
across multiple modules live here.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_image(path: str | Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[warn] Could not read image: {path}")
    return img


def load_gt(csv_path: str | Path) -> pd.DataFrame:
    """Load ground-truth CSV.  Normalises column names to lower-snake_case."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Drop header-duplicate rows that sometimes appear in exported CSVs.
    df = df[df["image"] != "image"].reset_index(drop=True)
    for col in ["center_x", "center_y", "angle_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_angle(a: float) -> float:
    """Normalise angle to [0, 360)."""
    return float(a % 360)

def angle_diff(a1: float, a2: float) -> float:
    """
    Smallest angular difference in degrees.

    Example:
    359 vs 1 -> 2
    """

    diff = abs(a1 - a2) % 360

    return min(diff, 360.0 - diff)