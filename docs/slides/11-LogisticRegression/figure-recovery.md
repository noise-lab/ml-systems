# Figure recovery: 11-LogisticRegression

Source: `docs/slides/11-LogisticRegression.pptx`, 27 slides, rendered 2026-10-16
(headless LibreOffice at 110 and 220 dpi, lecture pen-ink composited by the tool).
Slides 1–10 are the logistic regression lecture; slides 11–27 are the SVM
material that belongs to the `11-SVM` Quarto deck and were not triaged here.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | course title |
| 2 | skip | | | section divider |
| 3 | skip | | | bullets only (MLE framing, already in the appendix recap) |
| 4 | skip | | | Bernoulli PMF in ink; the deck typesets the same equation on "The Bernoulli Distribution for Binary Labels" |
| 5 | media | "The Sigmoid Function: Squashing to [0,1]" | `s05-i01.png` | already embedded; ink beside it only restates the sigmoid formula the deck typesets |
| 6 | keep | new, after "Decision Boundaries" | `decision-boundary-annotated.png` | the two embedded plots plus lecture ink: linear score vs. its sigmoid on the separable plot, "no single 1-D decision boundary" on the overlapping one; title placeholder stripped before cropping |
| 7 | keep | new, after "Relationship to Linear Regression" | `likelihood-gaussian-vs-bernoulli.png` | fully hand-drawn: Gaussian likelihood beside the Bernoulli likelihood with the sigmoid, then the y=1 case; the deck only states this in words and a table |
| 8 | skip | | | log-likelihood in ink; the deck typesets the cross-entropy loss on "Expressing the Loss: Cross-Entropy" |
| 9 | skip | | | bullets only (gradient descent, Newton, SGD) |
| 10 | skip | | | summary bullets |
| 11–27 | skip | | | SVM lecture, belongs to the 11-SVM deck |

## Captured images

- `images/decision-boundary-annotated.png` (2303×1326) — source: `docs/slides/11-LogisticRegression.pptx` slide 6, final build state, shape "Title 2" stripped, ink strokes composited, cropped from the 220 dpi render.
- `images/likelihood-gaussian-vs-bernoulli.png` (2856×1239) — source: `docs/slides/11-LogisticRegression.pptx` slide 7, final build state (all 13 build steps are ink groups that add to the board; nothing overlays the base), cropped from the 220 dpi render.
