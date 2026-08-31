# 12-Trees-Ensembles — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026**: Added a verified vignette based on a primary source published April 2026 in *Scientific Reports* (Nature Publishing Group): "An optimized gradient boosting framework for IoT intrusion detection: a comprehensive evaluation on the CICIoT2023 dataset" (doi: 10.1038/s41598-026-47399-5). XGBoost result: 98.54% accuracy and 93.06% AUC-ROC on multi-class IoT attack classification. This replaces the entirely dated source-extract content, which had no current-events hook at all.
- **2026**: Dropped the C4.5 slide (Slide 12 in source extract) — C4.5 is a historical algorithm, not a current teaching point. Coverage-notes flag this in "Suggested missing coverage" below.

## Suggested missing coverage on broad themes (point 3)

- **XGBoost / LightGBM implementation details**: The source extract and book cover the algorithm but not the systems innovations (histogram-based splitting, parallelism, regularization terms) that make XGBoost dominant in practice. A single slide on "why XGBoost is fast" would be valuable for students who will actually use it.
- **SHAP values in depth**: The summary slide mentions SHAP as the standard interpretability tool for ensemble models, but a worked example showing a SHAP waterfall plot for a network flow prediction would make this concrete. Consider pairing with the IoT privacy application.
- **Class imbalance and ensemble methods**: The 2026 CICIoT paper specifically highlights class imbalance as the key challenge. This deck notes it but does not address it — that topic belongs in the Data Preparation lecture (07-Preparation). Add a cross-reference note when teaching.
- **Historical context of boosting**: AdaBoost (Freund & Schapire, 1997) and its theoretical underpinnings (PAC learning, weak learnability) are mentioned but not developed. Relevant for graduate-level versions of this course.
- **C4.5 algorithm**: Dropped from the rebuild (was Slide 12 in the source extract). C4.5 is historically important but rarely used directly in practice; CART (scikit-learn's default) is the current standard. If instructors want to discuss the C4.5 vs. CART distinction (e.g., handling of categorical features, pruning strategy), add a supplementary slide or note.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh" (web-verify; swap only for something fresher and confirmed). Items placed in this refresh that will age:

- **2026 CICIoT vignette** (slide "Current Events: Gradient Boosting for IoT Intrusion Detection"): This result is from April 2026. Re-verify in 2027 that the doi resolves, check whether a larger benchmark or a follow-up study has superseded it, and update the accuracy figure if a newer result on the same benchmark is available. The CICIoT2023 dataset itself may be replaced by a 2024 or 2025 version.
- **"Competitive with deep learning on tabular data"** claim: Based on Grinsztajn et al. (2022). Track whether this finding holds up in 2027 benchmarks as tabular deep learning (e.g., TabNet, FT-Transformer) matures.
- **SHAP as "standard interpretability tool"**: True as of 2026. Alternative methods (e.g., LIME, integrated gradients) may gain traction; check the literature.
- **Apthorpe et al. 2017 IoT privacy study**: A classic, well-cited result unlikely to go stale conceptually, but worth flagging if newer follow-up work with more recent IoT devices is published.

Stronger alternative vignettes that were considered but not used:
- Alsadhan et al. (2025), kernel-based IDS for ICMPv6 DDoS (Results Engineering): This was used in the 11-SVM deck. Avoid reuse.
- Random Forest side-channel attack detection paper (ResearchGate, 2025): Could not verify journal/doi directly; excluded per no-fabrication rule.

## Curated images

- The `images/` directory in this folder is **empty** at time of authoring — no images were extracted from the source PPTX.
- The source extract (Slide 6) references a "Tree Map" figure; no image file was found. The deck does not reference any images.
- If images are later extracted from `12-Trees-Ensembles.pptx`, candidates for inclusion would be: a decision tree diagram showing a network traffic classification example, a random forest architecture diagram (parallel trees → vote), and a boosting diagram (sequential trees → residuals). These would be placed on slides "Core Idea", "Random Forests", and "Gradient Boosting" respectively.
- Do **not** include the title slide chrome, the "Trees and Ensembles" section divider slide (Slide 2), or any decorative logo images.

## Source

- Rebuilt from `_source-extract.md` (23 slides) — 15 content slides retained, 8 dropped (title, section divider, empty slide, C4.5 historical detail, redundant bagging slides, redundant advantages slide).
- Aligned to "Decision Trees" and "Ensemble Methods" sections of Chapter 5 (Supervised Learning) in *Machine Learning for Networking* (course textbook, `/text/supervised.rst`).
- Terminology and ordering follow the book: decision trees (CART, Gini, pruning, brittleness) → bagging → random forests (feature randomization, OOB, feature importance) → boosting (gradient boosting, AdaBoost, parameters).
- One divergence from source extract: source uses "RSS" throughout for regression split criterion; book and current scikit-learn documentation prefer "MSE" (mean squared error) — used MSE in the deck to match book and modern usage.
