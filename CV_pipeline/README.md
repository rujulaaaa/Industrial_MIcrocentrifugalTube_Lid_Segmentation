# Tube Lid Detection and Orientation Estimation

A classical computer vision pipeline for detecting tube lids and estimating their orientation under varying lighting and background conditions.

The project focuses on robust geometric localization and appearance-based orientation estimation using lightweight and interpretable computer vision techniques instead of deep learning.

---

### Features

- Tube lid detection using Hough Circle Transform
- Orientation estimation using angular intensity profiling
- Appearance-based false positive rejection
- Spatial outlier filtering
- Robust preprocessing using LAB + CLAHE
- Evaluation and visualization utilities
- Fully modular pipeline structure

---


### Overall Pipeline Flow

```text
Input Image
    ↓
LAB Color Conversion
    ↓
CLAHE Contrast Enhancement
    ↓
Median Blur
    ↓
Hough Circle Detection
    ↓
Appearance-Based Filtering
    ↓
Spatial Outlier Rejection
    ↓
Angular Intensity Profile Generation
    ↓
Gaussian Smoothing
    ↓
Peak Detection and Peak Scoring
    ↓
Orientation Estimation
    ↓
Final Visualization + Evaluation
```

---

### Project Structure

```text
.
├── config.py
├── run_pipeline.py
├── evaluate.py
├── README.md
├── requirements.txt
├── data/
│   └── TubeDetectionDataset_ZeonSystems2026/
│       ├── annotations.csv
│       └── images/
└── src/
    ├── detector.py
    ├── orientation.py
    ├── pipeline.py
    ├── visualization.py
    ├── matching.py
    ├── metrics.py
    └── utils.py
```

---

### How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the perception pipeline on the dataset:

```bash
python3 run_pipeline.py \
data/TubeDetectionDataset_ZeonSystems2026/images/ \
--save_viz
```

This will:
- run lid detection,
- estimate orientations,
- and save visualization outputs.

Run evaluation against ground truth:

```bash
python3 evaluate.py \
--gt data/TubeDetectionDataset_ZeonSystems2026/annotations.csv \
--images data/TubeDetectionDataset_ZeonSystems2026/images/ \
--save_viz
```

This computes:
- Precision
- Recall
- F1 Score
- Orientation error statistics

and also saves evaluation overlays and debugging visualizations.