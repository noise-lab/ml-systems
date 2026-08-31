# 11-LogisticRegression — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026**: Added a `.vignette` box citing a January 2026 *Scientific Reports* paper
  ("A lightweight machine learning approach for DDoS detection and classification,"
  doi:10.1038/s41598-026-48535-x) that benchmarks logistic regression against Random
  Forest and Naïve Bayes for multiclass DDoS detection. Used to motivate the
  accuracy–interpretability–deployability tradeoff, not just raw accuracy comparison.
  The vignette is sourced from a primary publication (Nature/Scientific Reports);
  verified via web search June 2026.

## Suggested missing coverage on broad themes (point 3)

- **Evaluation metrics beyond accuracy**: Precision, recall, F1, ROC-AUC are essential
  for imbalanced network datasets (most traffic is benign). A one-slide treatment of
  the confusion matrix and AUC would strengthen the lecture.
- **Calibration**: Logistic regression outputs are well-calibrated probabilities; this
  property is useful in network anomaly scoring and worth a mention.
- **Feature scaling**: Logistic regression with gradient descent is sensitive to feature
  magnitude; standardization should be mentioned explicitly.
- **Log-odds / odds-ratio interpretation**: The coefficients of logistic regression have
  a direct log-odds interpretation, useful for explainability in networking contexts.
- **Hands-on activity pointer**: The book appendix has a DNS query vs. response activity;
  consider embedding the code output or a screenshot of the decision boundary.
- **Connection to neural networks**: The logistic regression unit is the basic building
  block of a neural network (a single-layer perceptron with sigmoid activation) — a
  one-slide bridge would connect to the deep-learning lectures later in the course.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed).

- **Vignette (2026):** The Scientific Reports DDoS paper (doi:10.1038/s41598-026-48535-x)
  may be superseded by more recent IoT/edge deployment studies. Re-search
  "logistic regression DDoS IoT classification 2027" to find a fresher hook if this
  paper feels dated. The *teaching point* (accuracy vs. deployability tradeoff) should
  stay constant; only the illustrating paper needs to change.
- **Accuracy figure (91.61% LR vs. 99.88% RF):** These numbers come from the cited
  paper. If a 2027 paper shows different baselines, update the vignette figures.
- The book reference ("Machine Learning for Networking," Chapter 5) should be checked
  each year in case chapter numbering or section titles change in a new edition.

## Curated images

- `images/s06-i02.png` — **Used.** Shows a sigmoid curve fitting clean binary data
  (survived=0 / died=1 vs. x). Good illustration of the logistic fit.
- `images/s06-i03.png` — **Used.** Shows a linear regression line applied to binary
  labels — the classic "why linear regression fails" contrast image.
- `images/s05-i01.tiff` — **Dropped.** TIFF format; content not confirmed useful
  (could not read as image). Dropped to avoid render issues; re-evaluate if the TIFF
  can be converted to PNG and its content identified as pedagogically valuable.

## Source

- Rebuilt from `_source-extract.md` (10 slides) + book chapter *Machine Learning for
  Networking*, Chapter 5: Supervised Learning, "Linear Models" and "Logistic Regression"
  subsections.
- Deck consolidates the 10 source slides into 14 content slides (including section
  dividers) with expanded networking examples from the book not present in the original
  PowerPoint.
- Original slides diverged from the book in two ways flagged here:
  1. The source extract does not mention the DNS query/response application; the book
     does — book preferred.
  2. The source extract uses a "Bernoulli distribution" framing early; the book leads
     with the sigmoid directly. Deck follows a hybrid order: MLE framing → Bernoulli
     → sigmoid, which is more pedagogically coherent.
