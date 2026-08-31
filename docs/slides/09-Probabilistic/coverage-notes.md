# 09-Probabilistic — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026:** Added a SpamAssassin vignette (still active; version 4.0.1 released
  January 2025) grounded in the book's historical spam-filtering case study.
  Verified SpamAssassin is still actively maintained at
  https://spamassassin.apache.org.
- **2026:** Added a 2025 research vignette: Bao & Gao, "Network Intrusion
  Detection Based on Improved KNN Algorithm," *Scientific Reports* 15, article
  29842 (March 2025). doi:10.1038/s41598-025-14199-2. Verified via direct
  search against nature.com.

## Suggested missing coverage on broad themes

- **Gaussian Naive Bayes worked example:** a slide showing actual Gaussian
  PDFs fitted to two traffic classes (e.g., benign vs. attack inter-arrival
  times) would make the likelihood estimation concrete for students.
- **k-NN with k-d tree visualization:** a diagram showing how a k-d tree
  partitions a 2-D feature space would help students understand the
  algorithmic side of k-NN beyond the conceptual description.
- **Comparison on a shared dataset:** using the same HTTP vs. Log4j dataset
  that appears in the book's k-NN and decision-tree examples (supervised.rst)
  to compare Naive Bayes and k-NN accuracy would create a natural bridge to
  the next lectures on linear models and trees.
- **Laplace smoothing worked example:** a concrete before/after on a small
  feature set would solidify the zero-probability fix.
- **Probabilistic output usage:** when downstream decisions (e.g., alerting
  thresholds in an IDS) depend on calibrated probabilities vs. just rank
  order, Naive Bayes calibration issues matter — worth a dedicated discussion.
- **Approximate nearest neighbor (ANN) methods:** the prediction latency
  problem for large training sets in networking is mentioned but not resolved;
  a brief mention of HNSW or FAISS would be useful for students who plan to
  deploy k-NN in production.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events
refresh" (web-verify; swap only for something fresher and confirmed).

- **SpamAssassin vignette (2025):** verify whether version 4.0.1 remains the
  latest release or whether 4.0.2+ has been released. The project was on a
  slow-release cadence as of 2025; check https://spamassassin.apache.org before
  the 2027 term.
- **Bao & Gao (2025) k-NN paper vignette:** check for follow-up work or more
  prominent 2026 results on k-NN for IDS on the NSL-KDD or CIC-IDS benchmarks.
  A stronger vignette would have a head-to-head comparison with deep models on
  the same dataset.
- If a major paper or system (e.g., a production network IDS) adopts Naive
  Bayes or k-NN in an explicitly stated architectural role in 2026–2027, that
  would be a stronger hook than the current research-paper vignette.

## Curated images

- **No images used.** The `images/` directory for this deck is empty —
  the original PowerPoint had no extractable diagrams that would illustrate the
  concepts better than the mathematical notation in the slides. The book's
  kNN-classifier figure (two-panel KernelPCA projection) would be excellent
  here if it can be exported from the book build artifacts.
- **Recommended addition (future):** export `fig-knn_classifier` from the
  book's supervised chapter (supervised.rst → inline/supervised_knn-classifier.rst)
  and add it to the k-NN applications slide.

## Source

- Rebuilt from `_source-extract.md` (12 slides, Naive Bayes-only content) +
  Supervised Learning chapter of "Machine Learning for Networking"
  (`text/supervised.rst`), sections: "Non-Parametric Models" (K-Nearest
  Neighbors), "Probabilistic Models" (Naive Bayes), including the spam
  filtering history, the paradox of Naive Bayes, and the zero-probability
  problem subsections.
- The source extract covered only Naive Bayes; k-NN coverage is added from
  the book, which places k-NN in the supervised learning chapter as a
  non-parametric model. This is a divergence from the original slide set:
  the book treats them in separate sections (non-parametric and probabilistic),
  while the original slides focused exclusively on probabilistic models. The
  deck follows the book's terminology and scope.
- Deck length: 18 slides (within the 15–30 target).
