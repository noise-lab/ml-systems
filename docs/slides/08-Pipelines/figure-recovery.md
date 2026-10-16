# Figure recovery: 08-Pipelines

Source: `docs/slides/08-Pipelines.pptx`, 25 slides, rendered 2026-10-16 (headless LibreOffice, 110 dpi contact sheet, 220 dpi captures).

Slides 3, 9, 10, 12, 17 and 18 are pen-ink annotations (`p:contentPart` with per-stroke fallback PNGs) that LibreOffice renders as blank; the figures below were recovered by compositing the fallback stroke images onto the 220 dpi render in slide coordinates.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | keep | new, "The Pipeline, End to End" after "The ML Pipeline: Three Stages" | ml-framework-pipeline.png | packets-to-model framework diagram; no builds; the five icons were previously extracted as s02-i01..i05 and dropped as clip art, but the assembled diagram with labels and arrows is the figure |
| 3 | keep | new, "Training vs. Testing: The Flow" before "Why Split at All?" | training-vs-testing-flow.png | ink sketch: fit vs. predict, test labels bypass the model |
| 4 | skip | | | bullets |
| 5 | skip | | | bullets |
| 6 | media | "The Bias-Variance Tradeoff" | s06-i06.png | already embedded |
| 7 | skip | | | bullets |
| 8 | skip | | | bullets |
| 9 | keep | new, "Random vs. Temporal Split, Drawn" after "How to Split: Random vs. Temporal" | random-vs-temporal-split.png | ink sketch; two click builds (Group 7, Group 6) both captured in final state |
| 10 | keep | new, "Train / Validation / Test, Drawn" after "Why Split at All?" | train-validation-test-split.png | ink sketch; two click builds (Group 2, Group 3) both captured in final state |
| 11 | skip | | | title only |
| 12 | keep | new, "K-Fold Cross-Validation, Drawn" after "K-Fold Cross-Validation" | k-fold-cross-validation-sketch.png | ink sketch beside the printed bullets: K = 4 fold rotation with the test slice set aside |
| 13 | skip | | | summary bullets |
| 14 | skip | | | section divider |
| 15 | skip | | | bullets |
| 16 | skip | | | confusion-matrix table; Quarto deck already has it as a Markdown table |
| 17 | keep | new, "The Four Metrics, Drawn" after "Core Metrics from the Confusion Matrix" | evaluation-metrics-sketch.png | ink: handwritten definitions of the four metrics plus a drawn positive/negative/detected box; printed body, table and its two labels stripped so only the ink remains |
| 18 | keep | new, "Precision-Recall: A Worked Example" after "Precision-Recall: The Imbalance Problem" | precision-recall-worked-example.png | ink: 1000-person screening example with the drawn confusion matrix (TP 50, FN 5, FP 5, TN 940) |
| 19 | skip | | | bullets |
| 20 | media (6 tiny ink strokes, skipped) | "ROC Curve and AUC" | s20-i07.png | already embedded |
| 21 | media (6 tiny ink strokes, skipped) | "Precision-Recall: The Imbalance Problem" | s21-i08.png | already embedded |
| 22 | skip | | | bullets |
| 23 | skip | | | bullets; 31 small ink strokes, a marginal annotation not worth a figure |
| 24 | skip | | | bullets |
| 25 | skip | | | summary bullets |

## Captured images

- `images/ml-framework-pipeline.png` (2090x445) — source: 08-Pipelines.pptx slide 2, 220 dpi render, box 70,255,2160,700, nothing stripped
- `images/training-vs-testing-flow.png` (2050x863) — source: 08-Pipelines.pptx slide 3, 132 ink stroke images composited onto the 220 dpi render, box 10,355,2060,1218
- `images/random-vs-temporal-split.png` (1920x825) — source: 08-Pipelines.pptx slide 9, 24 ink stroke images composited onto the 220 dpi render, box 170,270,2090,1095 (includes the two grey label boxes)
- `images/train-validation-test-split.png` (1760x595) — source: 08-Pipelines.pptx slide 10, 9 ink stroke images composited onto the 220 dpi render, box 250,470,2010,1065
- `images/k-fold-cross-validation-sketch.png` (985x558) — source: 08-Pipelines.pptx slide 12, 36 ink strokes composited by the tool onto the 220 dpi render, box 1165,680,2150,1238
- `images/evaluation-metrics-sketch.png` (2055x840) — source: 08-Pipelines.pptx slide 17, 143 ink strokes composited by the tool; `strip --drop "Content Placeholder 1,Group 77,TextBox 4,TextBox 5"` to remove the printed bullets, table and its labels; box 140,250,2195,1090 (excludes one stray dot above the title line)
- `images/precision-recall-worked-example.png` (2110x653) — source: 08-Pipelines.pptx slide 18, 160 ink strokes composited by the tool onto the 220 dpi render, box 25,585,2135,1238
