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

- **2026-09 audit:** the source PPTX stores its figures as *hand-drawn ink*, with each pen stroke saved as a separate tiny PNG (300–600 of them, all under ~8 KB) — which is why the original extraction found "no images." The strokes were programmatically re-composited at their stored slide offsets into per-slide figures, cropped, and saved to `images/`:
  - `images/s06-i01.png` (source Slide 6): regression tree + feature-space partition (the classic *Hitters* example, regions R1/R2/R3). Placed on new slide "Representing the Feature Space".
  - `images/s11-i01.png` (source Slide 11 — listed as *empty* in `_source-extract.md` because it was all ink): a full worked example comparing two candidate splits by weighted Gini index (≈0.58 vs. ≈0.39). Restored as new slide "Worked Example: Choosing a Split". Note: the earlier "8 dropped" count treated this as an empty slide; it was actually substantive content.
  - `images/s16-i01.png` (source Slide 16): bagging diagram (D → bootstrap samples D1–D4 → bagged trees). Placed on "Bagging: Bootstrap Aggregation".
  - `images/s17-i01.png` (source Slide 17): random forest diagram (bagging + random feature subsets → trees T1–T4 → vote). Placed on "Random Forests: Decorrelating the Trees".
  - `images/s20-i01.png` (source Slide 20): income/credit scatter with two stump splits and circled misclassified points. Placed on "AdaBoost: Reweighting for Classification".
- Ink content judged redundant and **not** restored as images: Slide 5 note ("very intuitive: start at root, work down" — covered by Core Idea slide); Slide 7 handwritten RSS split objective (typeset in LaTeX on "Representing the Feature Space" instead); Slide 8 handwritten cost-complexity objective (already typeset on "Overfitting and Pruning"); Slide 10 handwritten Gini/entropy formulas (already typeset on "Splitting Criteria"); Slide 12 C4.5 sketch (slide intentionally dropped, see above).
- No `.wmf` or `.tiff` files exist in the source PPTX (all media are PNG stroke fragments).
- Do **not** include the title slide chrome, the "Trees and Ensembles" section divider slide (Slide 2), or any decorative logo images.

## 2026-09 content audit additions

- Added "Representing the Feature Space" slide: tree ↔ tree-map duality figure (source Slide 6) plus the recursive-binary-splitting RSS objective (source Slide 7), typeset in LaTeX.
- Added "Worked Example: Choosing a Split" slide: weighted-Gini split comparison (source Slide 11).
- Added the classification-error-rate bullet (third splitting criterion, too insensitive for growing; used in pruning) to "Splitting Criteria" — this was on source Slide 10 but missing from the rebuild.
- Added figures to the Bagging, Random Forests, and AdaBoost slides (see "Curated images" above); those slides were reflowed into two-column layouts.

## Source

- Rebuilt from `_source-extract.md` (23 slides) — 15 content slides retained, 8 dropped (title, section divider, empty slide, C4.5 historical detail, redundant bagging slides, redundant advantages slide).
- Aligned to "Decision Trees" and "Ensemble Methods" sections of Chapter 5 (Supervised Learning) in *Machine Learning for Networking* (course textbook, `/text/supervised.rst`).
- Terminology and ordering follow the book: decision trees (CART, Gini, pruning, brittleness) → bagging → random forests (feature randomization, OOB, feature importance) → boosting (gradient boosting, AdaBoost, parameters).
- One divergence from source extract: source uses "RSS" throughout for regression split criterion; book and current scikit-learn documentation prefer "MSE" (mean squared error) — used MSE in the deck to match book and modern usage.
