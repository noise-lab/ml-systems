# 08-Pipelines — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026**: Added a `::: {.vignette}` hook citing Kapoor & Narayanan, "Leakage and
  the Reproducibility Crisis in ML-based Science," *Patterns* (Cell Press), September
  2023. This is a verified, peer-reviewed, widely-cited study that audited 294 ML
  papers across 17 fields and found data leakage causing reproducibility failures.
  The civil-war-prediction result (correcting leakage erased the apparent ML advantage)
  is memorable and cross-domain. Also referenced arXiv:2603.25826 (March 2026),
  "Understanding AI Methods for Intrusion Detection and Cryptographic Leakage," which
  shows the near-perfect training accuracy / degraded real-world performance gap
  persists in network intrusion detection under distribution shift — a direct,
  current network-security angle on the same teaching point.

## Divergences between old slides and the book

- The original slides (25 slides) mixed pipeline overview, hyperparameter tuning, and
  evaluation metrics in a somewhat scattered order. The rebuilt deck follows the book's
  chapter structure: data engineering → training/splitting → cross-validation →
  evaluation metrics → leakage pitfalls.
- The original slides included a "machine learning framework" diagram (slides 2–5) that
  was rendered as a flow-chart of unlabeled icons. These are clip-art-level visuals and
  were dropped. The only images retained are the bias-variance tradeoff plot (s06-i06.png),
  the ROC curve (s20-i07.png), and the precision-recall curve (s21-i08.png) — all data
  plots that teach something directly.
- The book does not use the term "meta-learning" for hyperparameter tuning (the original
  slides did). The rebuilt deck uses the book's term "hyperparameter tuning" or
  "hyperparameter optimization" instead.

## Audit pass (2026-09)

- Verified all 25 source slides against the qmd. All substantive pptx content is present
  or documented above as an intentional restructuring. One genuine gap found and fixed:
  original slide 19 ("The Threshold Itself Can Vary") — the operating-point concept
  (threshold tuned per application to trade detection rate against false positives) was
  not stated anywhere. Restored as a bullet on the "ROC vs. Precision-Recall" slide,
  with expanded speaker notes.
- Re-verified the five dropped s02 images visually: all are generic icons (document
  stack, neural-net outline/filled, 3×3 grid, gear/cycle) — correctly dropped.
- All image references in slides.qmd (s06-i06.png, s20-i07.png, s21-i08.png) resolve on
  disk. No .wmf files in this deck.

## Suggested missing coverage on broad themes

- **Temporal/distributional shift in network ML:** the book mentions it, but a dedicated
  slide on concept drift (traffic patterns change over months/years) would reinforce
  why periodic re-evaluation and re-training are necessary for deployed network models.
- **Calibration:** precision/recall and AUC measure discrimination, but not whether a
  model's predicted probabilities are well-calibrated. A calibration curve (reliability
  diagram) is worth one slide for students who will use model scores as inputs to
  downstream systems.
- **Nested cross-validation:** for simultaneous hyperparameter search and generalization
  estimation; mentioned in the book but not in the slides.
- **Class-imbalance remedies:** SMOTE, cost-sensitive learning, and threshold tuning
  are closely related to the precision-recall discussion but not covered here.
- **Evaluation of unsupervised models:** the chapter focuses on supervised evaluation.
  A brief note on silhouette scores and within-cluster variance for clustering would
  round out the lecture for students who will do anomaly detection.

## Next-year refresh notes

Items placed in this build that will age and should be re-verified each year:

- **Kapoor & Narayanan (2023)** — this paper is stable (published), but watch for
  follow-up work or critiques. The "294 studies / 17 fields" count may grow as the
  community adds to their dataset. Check https://reproducible.cs.princeton.edu/ for
  updates.
- **arXiv:2603.25826 (March 2026)** — a preprint; check whether it has been published
  or superseded by the next annual refresh. If it has not been peer-reviewed by the
  time of the next term, consider replacing with a newer, published result on
  distribution shift in network intrusion detection.
- **AUC = 0.66 / AP = 0.39 figures** — these come from the course lab (images
  s20-i07.png and s21-i08.png) and will remain accurate as long as the lab dataset
  and code are unchanged. If the lab is updated, regenerate figures.
- **Spurious TTL correlation example** — drawn from Mahoney & Chan (2003) and Sommer &
  Paxson (2010, "Outside the Closed World"). These are stable classic references; the
  teaching example is unlikely to go stale, but a more recent replication would
  strengthen it.

## Curated images

| Image | Used? | Reason |
|---|---|---|
| s02-i01.png | Dropped | Generic document icon — clip-art, no teaching value |
| s02-i02.png | Dropped | Generic neural-net icon (outline circles) — clip-art |
| s02-i03.png | Dropped | Same icon as s02-i02 filled — clip-art |
| s02-i04.png | Dropped | Generic 3×3 grid icon — clip-art |
| s02-i05.png | Dropped | Generic gear/cycle icon — clip-art |
| s06-i06.png | **Used** | Bias-variance tradeoff diagram — clear, labeled, directly teaches the U-curve concept |
| s20-i07.png | **Used** | ROC curve with AUC = 0.66 from actual course lab — teaches realistic performance |
| s21-i08.png | **Used** | Precision-recall curve with AP = 0.39 from same lab — teaches imbalance problem |

## Source

- Rebuilt from `_source-extract.md` (25 slides) aligned to *Machine Learning for
  Networking* (Feamster et al.), Chapter 4: "Machine Learning Pipeline."
- Book sections covered: Data Engineering, Understanding the Data, Cleaning the Data,
  Irrelevant Features and Spurious Correlations, Labeling Data, Dividing Data (Training
  and Testing Sets, Validation Sets, Cross-Validation), Model Training (Overfitting and
  Bias-Variance Tradeoff), Model Evaluation (Performance Metrics: Accuracy,
  Precision/Recall, F1, ROC, Confusion Matrices).
