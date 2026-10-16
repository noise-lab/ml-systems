# Figure recovery: 15-Dimensionality-Reduction

Source: `docs/slides/15-Dimensionality-Reduction.pptx`, 10 slides, rendered 2026-10-16
(headless LibreOffice via `instructor-tools/src/pptx-figures.py`, 110 dpi contact sheet).

Result: **nothing to recover.** The deck contains no drawn diagrams, connectors,
groups, or pen-ink annotations. Every shape in `shapes.tsv` is a title/body
placeholder, a plain text box, or an embedded picture. All nine embedded
pictures were already extracted to `images/` by the original conversion, and
the September 2026 audit (see `coverage-notes.md`, "Curated images") embedded
eight of them in `slides.qmd` and deliberately dropped the ninth.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title slide |
| 2 | skip | | | section divider "Principal Component Analysis" |
| 3 | skip | | | bullets only ("What is PCA?"); content lives in "Two Views of PCA" |
| 4 | skip | | | bullets only ("Why PCA?"); content lives in "Why Dimensionality Reduction?" |
| 5 | media | "PCA in Pictures" | `s05-i01.png`, `s05-i02.png` | two matplotlib scatters (PC arrows; projection onto PC1), already embedded |
| 6 | media | "Input vs. Principal Component Space" | `s06-i03.png` | input-axes vs. component-axes panel, already embedded |
| 7 | media | "How to Compute Principal Components" (appendix) | `s07-i04.png`, `s07-i05.png` | typeset loading equation and objective, already embedded; no hand-drawn derivation |
| 8 | media | "Choosing the Number of Components: Scree Plot" | `s08-i06.png` | cumulative explained-variance curve, already embedded. `s08-i07.png` (left plot) is a two-point scree plot of a 2D dataset, uninformative; stays dropped |
| 9 | media | "T-SNE on Network Packet Data" | `s09-i08.png` | T-SNE scatter with DNS in red, already embedded |
| 10 | media | "Autoencoder Architecture" | `s10-i09.png` | encoder-bottleneck-decoder diagram, already embedded |

No `builds`, `strip`, or `crop` steps were needed; no new PNGs were written.
