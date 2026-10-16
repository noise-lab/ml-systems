# Figure recovery: 09-Probabilistic

Source: `docs/slides/09-Probabilistic.pptx`, 12 slides, rendered 2026-10-16.

The deck has no drawn diagrams. Its figures are the instructor's hand-written
(ink) derivations on slides 2, 3, 5, 6, 9 and 10. The ink is stored as
`mc:AlternateContent` with a PNG fallback per stroke, which headless
LibreOffice renders as blank; the fallback branch was substituted into the
slide XML in a scratch copy of the deck before rendering (153 blocks). Both
captures are the final animation state (all builds are entrance effects on
ink strokes, nothing is covered).

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | skip | | | handwritten optimal-Bayes argmax; typeset on "Bayes Classifier and the Naive Assumption" (appendix) |
| 3 | skip | | | Bayes' rule with numerator circled plus "Goal: find y for which P(y|x) is largest"; subsumed by slide 5 |
| 4 | skip | | | bullets |
| 5 | keep | new, "Bayes' Rule: Dropping the Denominator", after "The Optimal Bayes Classifier" | naive-bayes-drop-denominator.png | annotated chain: max posterior, Bayes rule, drop denominator |
| 6 | skip | | | handwritten factorization under independence; typeset on "Bayes Classifier and the Naive Assumption" (appendix) and stated in words on "The Naive Independence Assumption" |
| 7 | skip | | | bullets |
| 8 | skip | | | bullets (distribution table already in deck) |
| 9 | keep | new, "Which Term Is Hard to Estimate?", before "The Naive Independence Assumption" | naive-bayes-which-term-estimated.png | annotated argmax: P(x|y) requires estimation, P(y) prior is easy |
| 10 | skip | | | same factorization as slide 6 under a text bullet |
| 11 | skip | | | bullets |
| 12 | skip | | | summary bullets |

source: `docs/slides/09-Probabilistic.pptx` slide 5, ink fallback render at 220 dpi, box 170,340,2824,1650, downscaled to 1600x790 → `images/naive-bayes-drop-denominator.png`
source: `docs/slides/09-Probabilistic.pptx` slide 9, ink fallback render at 220 dpi, box 440,386,2836,1375, downscaled to 1600x660 → `images/naive-bayes-which-term-estimated.png`
