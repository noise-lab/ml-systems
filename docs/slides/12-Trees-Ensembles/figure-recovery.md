# Figure recovery: 12-Trees-Ensembles

Source: `docs/slides/12-Trees-Ensembles.pptx`, 23 slides (11, 12, 22 hidden), rendered 2026-10-16 with `pptx-figures.py render` (110 dpi contact sheet, pen ink composited on slides 5, 6, 7, 8, 10, 11, 12, 16, 17, 20).

Result: **no new figures captured.** Every drawn figure in the pptx is pen ink, and the September 2026 audit (commit 6dca54b, "Audit 11-12 decks against source pptx; recover whiteboard figures") had already re-composited the five figures worth keeping into `images/` and embedded them. The remaining ink is handwritten equations that the Quarto appendix now typesets, a one-line note, and the dropped C4.5 slide. The fresh LibreOffice render was compared stroke for stroke against the five existing PNGs; they are complete and clean, so they were left untouched.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–4 | skip | | | title, section divider, bullet slides |
| 5 | skip | | | ink note "very intuitive: start at root, work down"; covered by "Decision Trees: Core Idea" |
| 6 | keep (done) | "Representing the Feature Space" | s06-i01.png | regression tree + Hitters feature-space partition; already embedded |
| 7 | skip | | | ink RSS split objective with "RSS in R1 / R2" braces; typeset in appendix "Regression-Tree Split Criterion" |
| 8 | skip | | | ink cost-complexity objective with "penalty for many terminal nodes" arrow; typeset in appendix "Cost-Complexity Pruning" |
| 9 | skip | | | bullets |
| 10 | skip | | | ink Gini and entropy formulas; typeset in appendix "Splitting Criteria" |
| 11 | keep (done) | "Worked Example: Choosing a Split" | s11-i01.png | hidden slide, all ink: weighted-Gini comparison of two root splits; already embedded |
| 12 | skip | | | hidden slide, ink C4.5 sketch; C4.5 slide deliberately dropped from the rebuild (see coverage-notes.md) |
| 13–15 | skip | | | bullets |
| 16 | keep (done) | "Bagging: Bootstrap Aggregation" | s16-i01.png | D → D1..D4 → bagged trees; already embedded |
| 17 | keep (done) | "Random Forests: Decorrelating the Trees" | s17-i01.png | bagging + random feature subsets → T1..T4 → vote; already embedded |
| 18–19 | skip | | | bullets |
| 20 | keep (done) | "AdaBoost: Reweighting for Classification" | s20-i01.png | income/credit scatter, two stumps, circled misclassified points; already embedded |
| 21–23 | skip | | | bullets (22 hidden, no ink); summary |

No animation builds affect any keep slide (the figures are ink-only, no printed overlays), so nothing was stripped.

## Sources of the embedded images (captured by the 2026-09 audit, verified against this render)

- `images/s06-i01.png` — source: 12-Trees-Ensembles.pptx slide 6, ink strokes only (title and axis-label text boxes excluded)
- `images/s11-i01.png` — source: 12-Trees-Ensembles.pptx slide 11 (hidden), ink strokes only
- `images/s16-i01.png` — source: 12-Trees-Ensembles.pptx slide 16, ink strokes only
- `images/s17-i01.png` — source: 12-Trees-Ensembles.pptx slide 17, ink strokes only
- `images/s20-i01.png` — source: 12-Trees-Ensembles.pptx slide 20, ink strokes only
