# Figure recovery: 13-Deep-Learning

Source: `docs/slides/13-Deep-Learning.pptx`, 41 slides, rendered 2026-10-16
(headless LibreOffice at 110 dpi for triage, 220 dpi for crops; pen-ink
annotations composited by `pptx-figures.py` on slides 3, 5, 6, 7, 10, 11, 12,
32, 35, 36, 37).

The deck has no `speaker-notes.md`, so no notes rows were added. The September
2026 rebuild already embeds `s25-i31` (MLP), `s18-i25` (nPrint bit vectors),
`s27-i38` (feature-importance heatmap), `s28-i39` (nPrintML results), and
`s39-i43` (VGG-16); none of those were duplicated.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–2 | skip | | | title, section divider |
| 3 | keep | new, "Where Deep Learning Sits" (after "The Limitations of Linear Models") | deep-learning-nested-circles.png | ink: ML ⊃ representation learning ⊃ deep learning |
| 4 | skip | | | bullets, covered by "Limitations of Linear Models" |
| 5 | keep | "The Artificial Neuron" (existing; text wrapped in columns, unchanged) | neuron-sum-activation.png | ink: weighted inputs, bias, Σ, activation box; notes said "draw this" |
| 6 | skip | | | ink MLP; deck already uses s25-i31 for the same picture |
| 7 | keep | new, "Activation Functions, Sketched" (after "Activation Functions") | activation-functions-sketch.png | ink: sigmoid / tanh / ReLU curves; deck had only the table |
| 8–9 | skip | | | bullets, covered by training and output-activation slides |
| 10 | skip | | | ink chain rule dL/dw = dL/dy · dy/dw; typeset in the appendix |
| 11 | keep | new, "Gradient Descent, Pictured" (after "Training: Forward and Backward Propagation") | gradient-descent-loss-bowl.png | ink loss bowl; "Want: …" chain-rule annotation whitened out (appendix typesets it) |
| 12 | skip | | | ink MSE backprop derivation; appendix typesets the general form |
| 13–14 | skip | | | summary, section divider |
| 15 | skip | | | base pipeline, superseded by 16 and 17 |
| 16 | keep | new, "Bespoke Solutions: One Pipeline per Problem" (after "The Classic ML Pipeline…") | bespoke-pipelines.png | two identical pipelines; no animation |
| 17 | keep | new, "Generalizable Solutions?" (after the above) | generalizable-pipeline.png | pipeline with the shareable span bracketed |
| 18–22 | skip | | | nPrint bit-vector builds; s18-i25 already embedded, the four highlight states are the bullets on that slide |
| 23–24 | skip | | | AutoML bullets |
| 25 | keep | new, "nPrintML in One Picture" (before "nPrintML: Combining nPrint with AutoML") | nprintml-pipeline.png | footer label whitened out |
| 26 | media | | | Nmap probe table; coverage-notes already excludes it |
| 27–30 | media | | | heatmap and results tables, already embedded (s27-i38, s28-i39) or duplicates |
| 31 | skip | | | section divider |
| 32 | keep | new, "Convolution as a Weighted Average" (before "How Convolutions Work") | convolution-as-smoothing.png | ink formula + three small plots; printed bullets excluded by crop |
| 33–34 | skip | | | bullets |
| 35 | skip | | | ink: image as 3D tensor; deck frames CNNs for 1D packet sequences |
| 36 | keep | new, "A Convolution Layer, by the Numbers" (after "How Convolutions Work") | conv-layer-dimensions.png | ink: 32×32×3 input, 5×5×3 filter → 28×28×1 map |
| 37 | keep | new, "Max Pooling, by Example" (after the above) | max-pooling-example.png | ink 4×4 → 2×2 grid; printed bullets excluded by crop |
| 38 | skip | | | dropout / flatten bullets, covered in "CNN Architecture" |
| 39 | media | | | VGG-16 diagram, s39-i43 already embedded |
| 40 | media | | | Zeiler & Fergus visualization, someone else's figure; excluded |
| 41 | skip | | | summary |

## Sources

- `images/deep-learning-nested-circles.png` — source: 13-Deep-Learning.pptx slide 3, ink only, crop 260,380–2640,1560 @220 dpi, auto-trimmed (2367×1168)
- `images/neuron-sum-activation.png` — source: slide 5, ink only, crop 500,500–2440,1300, auto-trimmed (1892×756)
- `images/activation-functions-sketch.png` — source: slide 7, ink only, crop 100,360–2820,1200, auto-trimmed (2672×808)
- `images/gradient-descent-loss-bowl.png` — source: slide 11, ink only, "Want:" annotation whitened (1800,250–2933,600) before crop 300,360–2000,1480 (1589×1083)
- `images/bespoke-pipelines.png` — source: slide 16, shapes 150–184, crop 120,420–2800,1460 (2590×1009)
- `images/generalizable-pipeline.png` — source: slide 17, shapes 188–204, crop 120,740–2820,1240 (2621×463)
- `images/nprintml-pipeline.png` — source: slide 25, shapes 622–646, footer shape 640 whitened before crop 100,510–2820,1560 (2631×1001)
- `images/convolution-as-smoothing.png` — source: slide 32, groups 16/46/62/26 plus ink, crop 180,940–2520,1650 (2244×686)
- `images/conv-layer-dimensions.png` — source: slide 36, groups 3–183 plus ink, crop 60,940–2520,1650 (2387×710)
- `images/max-pooling-example.png` — source: slide 37, groups 22–65 plus ink, crop 1520,770–2900,1460 (1318×643)
