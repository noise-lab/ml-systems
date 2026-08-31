# 11-SVM — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026**: Added a verified `.vignette` hook citing Alsadhan et al. (2025), "Kernel-based machine
  learning intrusion detection systems for ICMPv6 DDoS detection," *Results Engineering* (Elsevier),
  published September 2025. Confirmed via ScienceDirect DOI and SSRN preprint listing. Key result:
  SVM with RBF kernel achieved 92.67% detection accuracy and 93% weighted precision/recall on
  ICMPv6 flood-attack data — directly instantiates the RBF kernel and C-tuning material in the deck.
  IPv6 / ICMPv6 context is topical given accelerating IPv6 deployment globally.

## Suggested missing coverage on broad themes (point 3)

- **SVM regression (SVR)**: the deck mentions SVMs work for regression but does not cover the
  epsilon-insensitive loss or SVR tuning; worth a slide or an aside for completeness.
- **Calibrated probabilities**: SVMs do not natively output probabilities; Platt scaling is the
  standard fix, but it is not covered. Practitioners using SVMs in pipelines that require probability
  scores (e.g., anomaly scoring with a threshold) need this.
- **Computational tricks for large-scale SVMs**: liblinear / SGD-based SVM approximations (sklearn's
  `LinearSVC`, `SGDClassifier` with hinge loss) are practically important for large traffic datasets.
  A brief comparison of exact quadratic programming vs. approximate methods would be useful.
- **Feature importance / interpretability**: SVMs have no direct feature importance score the way
  random forests do; SHAP values or permutation importance can be used, but this is not discussed.
- **Comparison to logistic regression**: a quantitative "when does SVM beat logistic regression (and
  vice versa)" slide would help students make practical model-selection decisions.
- **Multiclass SVM evaluation**: the one-vs-rest vs. one-vs-one tradeoff is mentioned but not
  illustrated with a worked networking example (e.g., application-type classification with 10 classes).

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items likely to age:

- **Alsadhan et al. vignette (2025)**: This is fresh as of the 2026 refresh. By 2027 there will
  likely be a more recent paper — search for "SVM kernel network intrusion detection 2026" or
  "SVM ICMPv6 IPv6 DDoS 2026." If a higher-accuracy result emerges (especially with a clear
  baseline comparison), prefer that. Keep the ICMPv6/IPv6 angle if IPv6 adoption continues to
  accelerate; replace with a different protocol context if a more prominent threat vector emerges.
- **IPv6 deployment statistics**: "IPv6 adoption past 45% of Google traffic by early 2025" was
  used in the speaker notes. Verify the current figure at https://www.google.com/intl/en/ipv6/
  statistics.html each year and update the speaker note if the framing changes.
- **CrowdStrike breakout-time stat**: not in this deck, but neighboring 02-Security uses it.
  Not an issue for this deck's refresh.

Stronger alternative vignettes flagged but not used:
- A 2025 paper on SVM-based encrypted traffic classification (application fingerprinting over
  TLS/QUIC) would be even more relevant to the networking-centric framing. Did not find a
  sufficiently specific, primary-source verified result in the 2025-2026 timeframe. Flag for next
  year.

## Curated images

Images used (all from `images/`, sourced from James et al. ISL figures):

| File | Content | Used on slide |
|---|---|---|
| `s17-i01.png` | Multiple valid separating hyperplanes (two-class scatter) | "Many Hyperplanes Exist" |
| `s18-i03.png` | Maximal margin hyperplane with support vectors and margin band | "Maximal Margin" |
| `s24-i04.png` | Four-panel comparison of different C values and their margin widths | "Tuning C" |
| `s26-i05.png` | Side-by-side polynomial kernel vs. RBF kernel non-linear boundaries | "Non-Linear Boundaries" |

Images dropped:
- `s17-i02.png`: shows a single separating hyperplane (one panel from ISL). Redundant given
  `s17-i01.png` which shows multiple hyperplanes and better motivates the "which one is best?"
  question. Dropped for concision.

## Source

- Rebuilt from `_source-extract.md` (18 slides from 11-LogisticRegression.pptx, slides 11–27)
  specifically the SVM portion (slides 11–27).
- Textbook alignment: "Support Vector Machines" section in Chapter 5 (Supervised Learning) of
  *Machine Learning for Networking* (course textbook, `/docs/slides/../ml/text/supervised.rst`),
  subsections: Max-Margin Classifiers, Training and Prediction, Kernel Methods, Networking
  Applications.
- Book notation followed throughout (C as violation budget / regularization parameter; $\gamma$
  as RBF bandwidth; one-vs-one preferred for multiclass).
- Divergence from source slides: source slides used $\rho$ for margin width; book uses $M$
  (consistent with ISL notation). Book notation ($M$) adopted here.
