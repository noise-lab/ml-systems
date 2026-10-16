# Figure recovery: 11-SVM

Source: `docs/slides/11-LogisticRegression-SVM.pptx`, 27 slides (slides 11–27 are the SVM lecture; 1–10 belong to 11-LogisticRegression), rendered 2026-10-16 with `pptx-figures.py` at 110 and 220 dpi. Pen ink was composited on slides 14, 15, 16, 18, 19, 22, 24, 25, 26.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 11 | skip | | | section divider |
| 12 | skip | | | bullets (the three-classifier list; Quarto has it as a table) |
| 13 | skip | | | drawn scatter with one separating line; subset of slide 21 |
| 14 | skip | | | bullets plus a decorative hand-drawn curve |
| 15 | keep | new, "A Hyperplane in Two Dimensions" (after "Hyperplanes: The Decision Boundary") | hyperplane-example.png | all pen ink; worked example $1+2X_1+3X_2=0$ that the Quarto deck had only as text |
| 16 | skip | | | ink of the separating-hyperplane inequalities; the appendix typesets the same equations |
| 17 | media | "Many Hyperplanes Exist" | s17-i01.png | already embedded (ISL) |
| 18 | media | "Maximal Margin" | s18-i03.png | already embedded (ISL); the ink circles the support vectors, which the slide text already explains |
| 19 | skip | | | ink of the max-margin program with annotations; the appendix typesets it and its bullets say the same |
| 20 | skip | | | bullets |
| 21 | keep | new, "Margin Geometry: Distance and Width" (after "Maximal Margin: The Widest Gap") | margin-geometry.png | drawn diagram: $r$, $\rho$, circled support vectors; captured in final build state |
| 22 | skip | | | ink of the soft-margin program with slack annotation; the appendix typesets it |
| 23 | skip | | | bullets |
| 24 | media | "Tuning C" | s24-i04.png | already embedded (ISL); ink "large C / small C" labels duplicate the slide text |
| 25 | keep | new, "The Kernel Trick, Written Out" (after "Non-Linear Boundaries: Why Kernels?") | kernel-support-expansion.png | ink derivation: inner-product form, collapse to the support set, kernel substitution; printed text stripped; the kernel formulas at the bottom were left out because the next slide's table typesets them |
| 26 | media | "Non-Linear Boundaries" | s26-i05.png | already embedded (ISL); ink labels duplicate the caption |
| 27 | skip | | | summary bullets |

## Sources

- `images/hyperplane-example.png` (1556×1200): slide 15, strip `Title 1`, 220 dpi, crop box 30,420,1620,1620 then auto-trim. The red ink part `Ink 143` (an illegible thick-marker label under the $X_1$ axis, drawn as a separate build step) was erased from its fallback image in a scratch copy of the deck before rendering; the data points in that part were kept.
- `images/margin-geometry.png` (1014×851): slide 21, 220 dpi, crop box 940,690,1990,1570 then auto-trim; final build state (circles, distance lines, $\rho$ arrow all shown).
- `images/kernel-support-expansion.png` (1800×579): slide 25, strip `Title 1,Content Placeholder 2`, 220 dpi, crop box 600,160,2900,930 then auto-trim, downscaled from 2277 px wide.
