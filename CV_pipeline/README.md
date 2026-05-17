# Tube Lid CV Pipeline 

## 1. Problem Understanding

The goal of this project was to build a perception pipeline capable of detecting tube lids, estimating their orientation, and rejecting false detections under varying illumination and background conditions.

Although the lids were approximately circular, experimentation revealed that the real challenge came not from geometry alone, but from:

- inconsistent lighting,
- weak boundary contrast,
- textured backgrounds,
- and visually similar circular structures.

Since the dataset contained only around seventy RGB images, the project deliberately avoided deep learning approaches and instead focused on a classical computer vision pipeline.

Over time, the design evolved toward a robust perception system based on layered geometric and appearance-based reasoning rather than large-scale learned models.

---

### 1.1 Understanding the Two Fundamental Subproblems

While initially approaching the task, it became clear that the overall perception problem was to be separated into two distinct subproblems:
1. Detection 
2. Segmentation 

---

### Detection Problem

The first subproblem was: determining whether a lid exists at a particular spatial location.

This mainly required:
- stable center estimation,
- approximate radius estimation,
- robustness to partial edge visibility,
- tolerance to weak boundaries.

Hough-circle-based geometric voting methods were used for the same.

---

### Segmentation 

The second subproblem was fundamentally different.Even after localizing a lid, the system still needed to infer:
- where the tab exists,
- where the joint lies,
- and therefore the orientation of the lid.

This stage depended not just on localization, but on understanding local structural asymmetry around the lid.

Initially, contour geometry was expected to provide this information directly. However, once contour extraction became unreliable under difficult lighting and texture conditions, the orientation stage also became unstable.

A detector can successfully estimate the lid center while still failing to infer the orientation.

---

## Architectural Consequence

This distinction significantly influenced the final pipeline design.

Instead of attempting to solve both problems using a single representation, the final architecture separated them:

- detection became geometry-driven,
- orientation estimation became appearance-driven.

The final detector focused only on robust localization using geometric evidence, while the orientation module independently analysed angular appearance asymmetry around the detected lid region.

This separation made the system substantially more stable because each stage solved only the specific problem it was best suited for.

# 2. Initial Detection Approach — Contour Based Segmentation

## 2.1 Initial Assumption

The earliest assumption was that the lids could be segmented directly from the image using thresholding and contour extraction.

This appeared reasonable because:

- lids were roughly circular,
- contours naturally provide geometric information,
- orientation estimation could directly use contour protrusions,
- and contour statistics such as circularity could help reject noise.

The initial detection pipeline therefore consisted of:

1. grayscale conversion,
2. Gaussian smoothing,
3. adaptive thresholding,
4. morphological operations,
5. contour extraction,
6. contour filtering using:
   - area,
   - perimeter,
   - circularity.

The expectation was that adaptive thresholding would handle varying illumination while contours would provide stable geometric representations.

---

# 3. Failure Analysis of the Contour Pipeline

## 3.1 Sensitivity to Background Texture

The first major issue appeared when images contained textured surfaces such as:

- wooden tables,
- patterned backgrounds,
- fabric-like regions.

Although adaptive thresholding locally adjusted intensity thresholds, the background textures themselves generated strong local gradients.

This caused:

- fragmented contours,
- noisy connected components,
- merged regions,
- and unstable object boundaries.

As a result, the contours no longer represented clean lid shapes.

---

## 3.2 Weak Boundary Contrast

Another major issue appeared for lids placed on bright backgrounds.

Several lids had extremely weak contrast against the surrounding surface. In these regions:

- the lid edge gradient became weak,
- thresholding merged the lid into the background,
- contour closure failed.

This failure mode was especially severe for:

- corner lids,
- partially illuminated lids,
- and lids near bright textured regions.

The contour extraction stage fundamentally depended on clean binary segmentation. Once segmentation failed, every downstream stage also became unstable.

This became the dominant weakness of the contour-based architecture.

---

## 3.3 Orientation Instability

The original orientation strategy attempted to estimate the tab direction using contour protrusions.

The reasoning was geometrically intuitive:

- the lid is approximately circular,
- the tab protrudes slightly outward,
- therefore the farthest contour region should correspond to the tab.

A radial contour profile was constructed around the centroid and protrusion peaks were analysed.

However, this failed in practice because the contour quality itself was unstable.

Observed problems included:

- noisy protrusions caused by threshold artefacts,
- false radial peaks,
- fragmented boundaries,
- inconsistent contour geometry,
- unstable orientation vectors.

The orientation stage therefore inherited all the weaknesses of the segmentation stage.

This revealed an important insight:

> relying too heavily on exact contour geometry was making the entire system fragile.

---

# 4. Intermediate Experiments

Several alternative strategies were explored before arriving at the final pipeline.

---

# 5. Experiment — Ellipse Based Detection

## 5.1 Motivation

Because some lids appeared slightly distorted under perspective effects, ellipse fitting was explored.

The assumption was:

- perspective distortion may convert circles into ellipses,
- ellipse fitting could capture boundaries more accurately,
- ellipse geometry might improve orientation estimation.

Both OpenCV ellipse fitting and skimage ellipse Hough transforms were experimented with.

---

## 5.2 Problems Observed

### Computational Cost

Ellipse Hough transforms introduced significantly higher computational complexity than circle fitting.

The parameter space became much larger because ellipse detection required estimating:

- center,
- major axis,
- minor axis,
- rotation angle.

This made the detector slower and more unstable.

---

### Excessive False Positives

Textured backgrounds generated many edge fragments that accidentally voted for large ellipses.

This resulted in:

- giant false ellipses,
- unstable detections,
- excessive parameter sensitivity.

Small parameter changes caused drastic behavioural differences.

---

### Overly Flexible Geometry

Although the lids were not perfectly circular in every image, they were still fundamentally much closer to circles than arbitrary ellipses.

The ellipse parameter space therefore became unnecessarily flexible.

This reduced robustness rather than improving it.

The ellipse approach was eventually abandoned.

---

# 6. Shift in Design Philosophy

At this stage, an important conceptual shift occurred.

Initially, the project attempted to:

> recover exact object boundaries.

However, repeated failures showed that exact segmentation was not necessary for successful localization.

A key observation was:

> even when contours failed, circular edge evidence was still visible.

This suggested that a voting-based geometric approach could succeed even under partial boundary visibility.

The problem was therefore reframed as:

- robustly estimate object center and radius,
- rather than perfectly recover the boundary.

This was a major architectural turning point.

---

# 7. Final Detection Architecture — Hough Circle Detection

## 7.1 Why Hough Circles

The Hough Circle Transform became attractive because:

- it accumulates evidence from distributed edge gradients,
- it tolerates partial boundary visibility,
- it does not require fully connected contours,
- and it remains stable under fragmented edge conditions.

Unlike contour extraction, Hough voting does not collapse entirely when part of the boundary becomes weak.

This directly addressed the earlier failure modes.

---

# 8. Improved Preprocessing Pipeline

Even after adopting Hough circles, preprocessing remained critical.

The final preprocessing pipeline evolved substantially through experimentation.

---

## 8.1 LAB Color Space

Instead of directly using grayscale conversion, the image was converted into LAB color space.

The primary motivation was that:

- the L channel isolates brightness information,
- while separating it from color information.

This improved robustness because several lids had grayscale intensities extremely similar to the background.

Using LAB improved local brightness representation without strongly amplifying irrelevant texture color variations.

---

## 8.2 CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) was introduced on the L channel.

This became necessary because:

- weak lid boundaries still remained difficult to distinguish,
- especially near bright surfaces.

CLAHE improved:

- local contrast,
- weak edge visibility,
- boundary separation.

Importantly, CLAHE enhanced local contrast rather than globally stretching the histogram.

This made it significantly more stable under mixed illumination.

---

## 8.3 Median Blur

After CLAHE, texture noise became stronger.

Initially Gaussian blur was used, but it softened weak circular edges too aggressively.

Median blur produced better results because it:

- preserved edge structure,
- reduced salt-and-pepper-like texture artefacts,
- maintained circular boundary sharpness.

This combination of:

- LAB,
- CLAHE,
- median blur

became substantially more robust than the earlier grayscale-threshold pipeline.

---

# 9. Hough Circle Parameter Tuning

Several parameters required careful tuning:

- minimum radius,
- maximum radius,
- accumulator resolution,
- minimum center distance,
- edge thresholds,
- voting thresholds.

The radius range was intentionally constrained tightly because all lids occupied a narrow size range within the dataset.

This reduced:

- spurious circle detections,
- unstable scale estimation,
- accidental background matches.

The tuning process itself became highly empirical and was performed by observing failure cases image-by-image.

---

# 10. Appearance Based Filtering

## 10.1 New Problem Introduced

Although Hough circles improved localization significantly, another issue emerged:

- empty rack holes and background structures were also circular.

Pure geometry alone was insufficient.

This introduced an important realization:

> geometry can localize candidates, but appearance must validate them.

---

## 10.2 Mean Intensity Filtering

To address this, an appearance-based filtering stage was introduced.

For each detected circle:

1. a circular mask was created,
2. mean intensity inside the region was computed,
3. dark regions were rejected.

This worked because:

- actual tube lids had different brightness characteristics than empty rack holes.

This stage significantly reduced false positives.

---

# 11. Spatial Outlier Rejection

Even after appearance filtering, occasional false detections still remained.

These typically appeared:

- isolated from the actual tray,
- near unrelated circular structures,
- or around textured background regions.

---

# 12. Attempted DBSCAN Clustering

DBSCAN clustering was initially explored because the lids naturally formed groups.

However, DBSCAN performed poorly because:

- each image contained only around 3–6 detections,
- the spatial distribution was sparse,
- density assumptions became unstable,
- parameter tuning became image dependent.

Small changes in epsilon or minimum samples caused:

- valid lids to disappear,
- or false detections to remain.

The dataset was simply too sparse for reliable density-based clustering.

---

# 13. Final Outlier Rejection Strategy

Instead of clustering, a simpler statistical approach was adopted.

The final method computed:

1. centroid of all detections,
2. distance of each detection from centroid,
3. mean and standard deviation of distances.

Detections too far from the normal spread were rejected.

The threshold became:

mean distance + k × standard deviation

rather than a hardcoded constant.

This worked better because:

- real lids formed spatially compact groups,
- false detections were usually isolated,
- the threshold adapted automatically to tray spread.

This became much more stable than DBSCAN.

---

# 14. Additional Structural Prior

The problem statement specified that trays typically contain between:

- 3 and 6 lids.

This prior knowledge was incorporated into the pipeline.

When detections exceeded six:

- detections nearest to the centroid were retained,
- distant candidates were rejected.

This introduced lightweight structural reasoning into the detector.

---

# 15. Orientation Estimation — Original Design

Initially, orientation estimation relied on contour protrusions.

The logic was:

- the tab and joint protrude from the circular body,
- radial contour analysis should reveal protrusion peaks,
- the dominant peak should correspond to the tab direction.

A radial contour profile was constructed and smoothed to suppress noise.

This was conceptually elegant but heavily dependent on contour quality.

---

# 16. Why Contour Based Orientation Failed

Once the pipeline transitioned to Hough circles, contours were no longer meaningful.

Synthetic circular contours generated from Hough detections contained no actual protrusion information.

As a result:

- radial contour profiles became artificially uniform,
- orientation estimates collapsed,
- arrows became biased and unstable.

This forced a complete redesign of the orientation module.

---

# 17. Final Orientation Strategy — Intensity Based Angular Sampling

The final orientation approach abandoned contour geometry entirely.

Instead, orientation estimation became appearance driven.

---

## 17.1 Key Insight

Although the tab may not produce a strong geometric contour, it still produces:

- local brightness asymmetry,
- subtle appearance protrusions,
- directional intensity variation.

Therefore, the image itself contains orientation information even when the contour does not.

---

## 17.2 Angular Intensity Profile

The final method constructed an angular intensity profile around the detected circle.

Procedure:

1. Divide the circle into angular bins.
2. Sample pixels slightly outside the detected radius.
3. Measure local intensities.
4. Build an angular intensity profile.
5. Smooth the profile using Gaussian filtering.
6. Detect significant peaks.

Sampling slightly outside the circle was important because:

- the tab extends beyond the nominal circular boundary.

This was one of the most important design changes in the project.

---

# 18. Peak Selection Logic

Simple maximum selection proved unstable because textured backgrounds generated many local maxima.

To improve robustness:

- peak prominence was analysed,
- local peak width was estimated,
- a combined score was computed.

The strongest structurally meaningful peak was then selected.

This reduced sensitivity to:

- noise spikes,
- texture artefacts,
- minor local intensity fluctuations.

---

# 19. Pipeline Modularization

As the system evolved, the codebase was modularized into separate components.

| Module | Purpose |
|---|---|
| detector.py | lid localization |
| orientation.py | tab direction estimation |
| pipeline.py | orchestration |
| visualization.py | debugging overlays |
| matching.py | GT assignment |
| metrics.py | evaluation |
| evaluate.py | benchmarking |

This separation greatly improved:

- debugging speed,
- readability,
- experimentation,
- and maintainability.

---

# 20. Robustness Evaluation Philosophy

Because the dataset was small, the project deliberately avoided pretending that large-scale training was possible.

Instead, the focus shifted toward:

> robustness evaluation.

The evaluation stage introduced perturbations such as:

- brightness changes,
- blur,
- Gaussian noise.

The objective was not synthetic data expansion, but rather controlled stress testing.

This aligned more naturally with industrial deployment philosophy where robustness matters more than benchmark overfitting.

---

# 21. Final Design Philosophy

The final pipeline became fundamentally different from the initial contour-based system.

The architecture evolved from:

> exact boundary extraction

toward:

> robust geometric localization + appearance based reasoning.

The final system intentionally prioritizes:

- robustness over geometric perfection,
- stability over fragile precision,
- interpretability over black-box learning,
- modularity over monolithic design.

Most importantly, the final architecture was shaped not by a single algorithmic preference, but by repeated observation of real failure modes.

The project therefore became an exercise in:

- iterative debugging,
- empirical reasoning,
- systems engineering,
- and perception architecture refinement.

---

# 22. Key Lessons Learned

1. Perfect segmentation is often unnecessary for robust localization.

2. Contour extraction becomes fragile under weak contrast and textured backgrounds.

3. Voting-based geometry can outperform exact segmentation under imperfect conditions.

4. Appearance cues are essential when geometry becomes ambiguous.

5. Sparse industrial layouts do not always suit clustering approaches like DBSCAN.

6. Orientation estimation can rely on appearance asymmetry rather than exact geometry.

7. Robustness evaluation can be more meaningful than dataset expansion for classical CV systems.

8. Failure analysis is often more valuable than theoretical elegance.

---

# 23. Final Outcome

The final perception system became:

- lightweight,
- deterministic,
- modular,
- geometry-guided,
- appearance-aware,
- and robust to moderate perturbations.

Most importantly, the pipeline evolved through continuous observation, redesign, and refinement rather than fixed assumptions.

The final architecture reflects a practical engineering approach to perception system design rather than a purely textbook implementation.