# Figure recovery: 08-Pipelines

Source: `docs/slides/08-Pipelines.pptx`, 25 slides, rendered 2026-10-16 (headless LibreOffice, 110 dpi contact sheet, 220 dpi captures).

Slides 3, 9 and 10 are pen-ink annotations (`p:contentPart` with per-stroke fallback PNGs) that LibreOffice renders as blank; the figures below were recovered by compositing the fallback stroke images onto the 220 dpi render in slide coordinates.

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
| 12 | skip | | | bullets |
| 13 | skip | | | summary bullets |
| 14 | skip | | | section divider |
| 15 | skip | | | bullets |
| 16 | skip | | | confusion-matrix table; Quarto deck already has it as a Markdown table |
| 17 | skip | | | metric formulas with a small confusion matrix; already in the deck |
| 18 | skip | | | bullets |
| 19 | skip | | | bullets |
| 20 | media | "ROC Curve and AUC" | s20-i07.png | already embedded |
| 21 | media | "Precision-Recall: The Imbalance Problem" | s21-i08.png | already embedded |
| 22 | skip | | | bullets |
| 23 | skip | | | bullets |
| 24 | skip | | | bullets |
| 25 | skip | | | summary bullets |

## Captured images

- `images/ml-framework-pipeline.png` (2090x445) — source: 08-Pipelines.pptx slide 2, 220 dpi render, box 70,255,2160,700, nothing stripped
- `images/training-vs-testing-flow.png` (2050x863) — source: 08-Pipelines.pptx slide 3, 132 ink stroke images composited onto the 220 dpi render, box 10,355,2060,1218
- `images/random-vs-temporal-split.png` (1920x825) — source: 08-Pipelines.pptx slide 9, 24 ink stroke images composited onto the 220 dpi render, box 170,270,2090,1095 (includes the two grey label boxes)
- `images/train-validation-test-split.png` (1760x595) — source: 08-Pipelines.pptx slide 10, 9 ink stroke images composited onto the 220 dpi render, box 250,470,2010,1065
