# Figure recovery: 10-LinearRegression

Source: `docs/slides/10-LinearRegression.pptx`, 29 slides, rendered 2026-10-16.

The deck is text, formula images, and matplotlib plots. The formula images were
already replaced by inline LaTeX and the plots are embedded as `s03-i01.png`,
`s03-i02.png`, `s14-i14.png`, `s16-i15.png`, `s23-i16.png`. The only drawn
content is pen ink on slides 13, 15 and 16. The ink `mc:Fallback` PNGs are
missing from the package (every image rel targets `NULL`), so `pptx-figures.py`
could not composite them; the strokes were rasterized from the InkML parts
(`ppt/ink/*.xml`) instead.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | skip | | | divider |
| 3 | media | "A Concrete Example: Scatter and Fit" | | s03-i01, s03-i02 already embedded; s03-i03 is a formula image, typeset |
| 4 | skip | | | bullets |
| 5 | skip | | | derivation laid out as formula images with callouts; appendix "Training: Minimizing RSS" typesets the same four steps in the same order |
| 6 | media | | | journal-article header, decorative |
| 7 | skip | | | bullets |
| 8 | media | | | Z-score and F-test formula images, typeset in appendix |
| 9 | skip | | | divider |
| 10 | skip | | | bullets |
| 11 | skip | | | SmartArt stock icons (stopwatch, banknote) with two captions; decorative, and the Quarto deck uses networking examples instead |
| 12 | media | | | basis-expansion formula image, typeset |
| 13 | skip | | | ink: polynomial expansion (already typeset on "Polynomial Regression in Practice") and one Fourier term; scribbly, low value |
| 14 | media | "Common Basis Functions" | | s14-i14 embedded |
| 15 | keep | new, after "Key Idea: Transform the Features" | basis-expansion-design-matrix.png | whole slide is ink: "Originally y = Xᵀβ" vs "Transform y = βᵀΦ(x)" with the two design matrices side by side |
| 16 | skip | "Common Basis Functions" | | ink Gaussian RBF formula next to the s16-i15 plot; clipped at the slide edge, illegible in places |
| 17 | skip | | | bullets |
| 18 | skip | | | summary |
| 19 | skip | | | divider |
| 20–22 | skip | | | bullets |
| 23 | media | "Underfitting vs. Overfitting" | | s23-i16 embedded |
| 24–25 | skip | | | bullets |
| 26–28 | media | | | ridge, lasso, elastic-net formula images, typeset in appendix and table |
| 29 | skip | | | summary |

## Captured

- `images/basis-expansion-design-matrix.png` (1600×712)
  source: `docs/slides/10-LinearRegression.pptx` slide 15, 36 ink parts (113 strokes) rasterized from `ppt/ink/ink28.xml`–`ink63.xml` onto the 220 dpi render, cropped below the title, downscaled from 2751 px.

## Left out

- Slide 16 ink (Gaussian RBF formula): runs off the right edge of the slide. If wanted, typeset
  $\phi_i(x) = \exp\{-\tfrac{1}{2\ell}\sum_j (x_j - \mu_{ij})^2\}$ next to the RBF plot instead.
- Slide 13 ink (Fourier term $\beta_0 + \beta_1\cos(\omega_1 x + \phi_1) + \cdots$): same, better typeset than captured.
