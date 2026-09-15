---
description: Generate the Summer 2026 (Paris) combined final exam — or a practice version of it — from this term's agenda and the past exams
---

This is the term-specific recipe used to draft the **Summer 2026 combined final**
(Thu Sep 17, 2026). It instantiates the general prompt in
[`generate-exam.md`](generate-exam.md); read that first for the LaTeX
conventions, then apply the parameters below. Students: run this yourself to make
practice exams. You will not get the same questions the instructor got, but you
will get questions built from exactly the same inputs.

## Parameters

| Parameter | Value |
|---|---|
| Exam type | `combined` (single exam; no midterm this term) |
| Agenda | [`../agenda/2026-summer.md`](../agenda/2026-summer.md) — Meetings 1–8 as of Sep 14; add Meeting 9 (generative models) when it appears |
| Term page | [`../terms/2026-summer.md`](../terms/2026-summer.md) — lists what was skipped (Naive Bayes, SVMs, the nPrint hands-on, the standalone autoencoder lecture); those are **out of scope** |
| Assignments | Assignment 1 (Video Quality Inference) and the project proposal — at least one question on each |
| Hands-ons | 01 packet capture, 02 scanning, 03 QoE, 06 netml features, 08 pipeline (HTTP/log4j), 10 linear regression + basis expansion, 12 IoT trees/ensembles, 13 DDoS neural net, 15 PCA, 16 k-means |
| Past exams to imitate | `2025/` here and `../midterm/2025/` (one final, one midterm: the combined exam mixes both) |
| Length | 5 pages, **65 points** (the on-campus exams are 4 pages / 50 points and designed for 30–40 minutes; this one is ~30% longer, designed for 45–50 minutes with the full period available) |
| Question mix | roughly a third multiple choice (select all that apply), a third yes/no with "Why or why not?", a third short answer; ends with a 2-point feedback section |

## Coverage (proportional to time spent)

1. **Why ML for networks; the pipeline** (Meeting 1): Mitchell's definition, measure–model–control loop, where effort and errors actually go (data preparation), self-driving networks.
2. **Security** (Meeting 2): why security is hard for ML (class imbalance, concept drift, heterogeneous data, real time); content filters vs. network-level behavioral features (SNARE: ephemeral routes, geographic distance, sender neighbourhoods); predicting campaigns from domain registration.
3. **Performance and resource allocation** (Meeting 3 + Assignment 1): QoE inference from encrypted traffic, segment detection by inter-packet gaps, segment download rate; short- vs. long-term allocation; the what-if scenario evaluator as a simple regression whose inputs are correlated.
4. **Data representation** (Meeting 4): netml STATS/SAMP features; no single best representation; encryption limits what is visible (port 443, TLS 1.3, ECH); nPrint bit alignment and the three-valued encoding; spurious correlations (husky/wolf, source IP, TTL); aggregation timescale.
5. **Training and evaluation** (Meeting 5): lock the test set away before normalizing; temporal splits for forecasting; bias–variance; cross-validation and the curse of dimensionality; confusion matrix, the 99.9%-accurate "always no" classifier, precision/recall/F1, PR vs. ROC/AUC, operating points.
6. **Supervised models** (Meeting 6): linear regression and the acknowledgment-packet problem; basis expansion; ridge/lasso and the direction of λ; logistic regression and linear separability (DNS query vs. response); decision trees (splits, impurity, brittleness); bagging and random forests (two sources of randomness); RF as a baseline.
7. **Deep learning** (Meeting 7): neuron, activation, learning rate, validation loss as the overfitting signal; when DL earns its keep; the IoT DDoS result (k-NN and RF match the NN); the Nest Cam privacy inference and the four defenses.
8. **Unsupervised learning** (Meeting 8): PCA, scree plots, t-SNE for visualization only, autoencoders for anomaly detection; k-means and its failure modes; GMM; DBSCAN (ε, minPts, variable density); hierarchical clustering and when nesting holds.

## Questions the instructor flagged in class as "good exam questions"

Use these; they are in the agenda too.

- Give a scenario where you would accept a lower detection rate for a very low false-positive rate, and one where you want the reverse (Meeting 5).
- For a higher-complexity regularized model, do you turn λ up or down, and what does that do to bias and variance (Meeting 6)?
- Given a small decision tree, draw the partition of the feature space, or the reverse (Meeting 6).
- How would you handle clusters of very different density with a density-based method (Meeting 8)?
- The always-predict-negative classifier that is 99.9% accurate (Meeting 5).

## Output

- Instructor: `/Users/feamster/Dropbox/Tmp/exams/ml-systems/final/2026/` — never committed before the exam is given; afterwards it is published in `docs/final/2026/` with solutions.
- Students: any directory of your own, e.g. `~/ml-systems-practice/final/`. Copy `feamster.sty` and the `Makefile` from `2025/`, then `make all`.

## Validation

As in `generate-exam.md`: `make all`, then check `pdfinfo exam.pdf | grep Pages` gives 5 and the `\prob{}` points sum to 65; open the PDF and look for overflowing option text or answer boxes that broke across pages.
