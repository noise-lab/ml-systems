# 16-Clustering — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026:** Added a `.vignette` based on a verified March 2025 paper in *Frontiers in Artificial Intelligence* (doi:10.3389/frai.2025.1625891) describing a BERT + GMM intrusion detection system that achieves 95.6% accuracy on unknown attack types. The paper is a primary source, the doi is verifiable, and no numbers were embellished. This replaces the original slides' generic "other types of clustering" framing with a concrete, dated, networking-specific result.

## Suggested missing coverage on broad themes (point 3)

- **HDBSCAN:** The source slides mention OPTICS in passing; the book references HDBSCAN as addressing DBSCAN's varying-density limitation. A slide on HDBSCAN's hierarchical density approach would strengthen coverage.
- **Semi-supervised learning bridge:** The book's Chapter 6 continues into semi-supervised learning (cluster-then-label, label propagation). A one-slide bridge connecting clustering output to label propagation would set up a natural follow-on lecture or reading.
- **Evaluation metrics in depth:** Silhouette score and Adjusted Rand Index are mentioned but not derived; a worked numerical example on a small 2D dataset would help.
- **Standardization:** The book explicitly warns that failing to standardize features is the most common K-means mistake; this is noted in speaker notes but could be a dedicated bullet slide.
- **Mean-shift:** Covered in the source slides (slide 30) but omitted from this rebuild as the book does not include it. If students encounter mean-shift in the wild, add it as an optional slide.
- **Deep clustering / autoencoder + clustering pipeline:** The book's autoencoder section explicitly connects to clustering; a slide showing the autoencoder → embedding → cluster workflow would be useful alongside Lecture 15.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh" (web-verify; swap only for something fresher and confirmed). Items likely to age:

- **BERT + GMM vignette (doi:10.3389/frai.2025.1625891):** Check whether this paper has been superseded by a stronger result or extended by the same group. The 95.6% figure and "unknown attack" framing are the teaching points — swap for a fresher paper that makes the same point if one exists.
- **"2025" framing in vignette:** Update year literal and source date each term.
- **HDBSCAN reference:** HDBSCAN is mentioned by name in the book; by 2027 it may have displaced vanilla DBSCAN as the default recommendation. Track adoption.

## Curated images

- **`images/s22-i01.png`** — Used on the "K-Means Failure Modes" slide. Shows K-means applied to elongated, non-spherical clusters (four cluster output visible). Useful teaching image: illustrates the spherical assumption failure. Retained.
- **`images/s24-i02.png`** — Used on the "DBSCAN" slide. Shows a KDE density curve with three peaks (red, green, blue) and sparse outlier points (blue stars). Illustrates density-based cluster intuition directly. Retained.
- Both images come from the original PowerPoint extraction. Neither is a logo, headshot, or decorative chrome. No other images were present in `images/`.

## Source

- Rebuilt from `_source-extract.md` (31 slides) + book Chapter 6 ("Unsupervised Learning") of "Machine Learning for Networking" — specifically the Clustering section (K-Means, GMM, DBSCAN, Hierarchical) and the Applications subsection.
- Book and source slides are largely consistent in algorithm coverage; the book provides deeper discussion of evaluation, standardization, and the autoencoder-clustering connection. Where source slides included Mean-shift and Kernel Density Estimation as standalone topics, the book does not cover them in detail, so they were omitted from the rebuild.
- Original slide count: 31. Rebuilt deck: 20 slides (including section dividers).
