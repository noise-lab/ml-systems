# Figure recovery: 16-Clustering

Source: `docs/slides/16-Clustering.pptx`, 31 slides (slide 16 hidden), rendered 2026-10-16 with `pptx-figures.py` at 110 and 220 dpi. The 1,090 media files in the archive are per-frame build animations and matplotlib exports, not distinct figures; triage was done from the contact sheet. Pen-ink annotations (slides 4, 14, 21, 23, 25–28, 30) were composited by the tool.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–3 | skip | | | title, divider, bullets |
| 4 | keep | new, "The Unsupervised Learning Landscape" (after "What Is Unsupervised Learning?") | unsupervised-learning-taxonomy.png | pen-ink taxonomy tree; the Quarto deck has only the roadmap table |
| 5 | skip | | | bullets |
| 6, 7, 8, 9, 11, 12 | keep | new, "K-Means, Step by Step" (after "How K-Means Works") | kmeans-steps.png | six drawn frames of the k-means walkthrough composited into a 2×3 grid; slide 10 is a near-duplicate of 9 and was dropped |
| 13 | skip | | | SSE equation, already typeset in the appendix |
| 14 | keep | new, "Two Shapes K-Means Cannot Split" (after "K-Means: Failure Modes") | kmeans-failure-sketch.png | pen-ink: concentric rings and a line with two outliers; complements the matplotlib s22-i01.png |
| 15 | skip | | | pros/cons bullets |
| 16 | skip | | | hidden EM bullets |
| 17 | keep | new, "Cutting the Dendrogram" (after "Hierarchical Clustering and Dendrograms") | dendrogram-cut.png | drawn dendrogram with cut line and four circled clusters; final build state kept |
| 18–20 | skip | | | bullets |
| 21 | skip | | | duplicate of slide 4 |
| 22 | media | "K-Means: Failure Modes" | s22-i01.png | already embedded |
| 23 | keep | new, "Two Families: Distribution-Based vs. Density-Based" (after "GMM vs. K-Means") | distribution-vs-density-clusters.png | pen-ink blobs vs. crescents, with the two column headings |
| 24 | media | "DBSCAN: Density-Based Clustering" | s24-i02.png | already embedded |
| 25 | keep | new, "GMM: A Mixture, Drawn" (after "GMM: Beyond Centroids") | gmm-mixture-sketch.png | pen-ink: observed density vs. three weighted Gaussians with priors; bullet text whited out along the top edge |
| 26 | keep | new, "EM: Known vs. Unknown Sources" (after "GMM: A Mixture, Drawn") | em-known-vs-unknown-sources.png | pen-ink: labeled points vs. gray points with soft responsibilities |
| 27 | skip | | | pen-ink crescents, superseded by slide 23 |
| 28 | keep | new, "DBSCAN, Drawn" (after "DBSCAN: Density-Based Clustering") | dbscan-sketch.png | pen-ink: two clusters with ε balls around seeds and three noise points |
| 29 | skip | | | bullets |
| 30 | skip | | | mean-shift sketch; the Quarto deck deliberately omits mean-shift (see coverage-notes.md) |
| 31 | skip | | | summary |

## Sources

- `images/unsupervised-learning-taxonomy.png` — source: 16-Clustering.pptx slide 4, full slide (ink only), auto-trimmed, downscaled to 1800 px wide.
- `images/kmeans-steps.png` — source: 16-Clustering.pptx slides 6, 7, 8, 9, 11, 12, plot region below the title from each, composited 2×3 with panel labels, 1800 px wide.
- `images/kmeans-failure-sketch.png` — source: 16-Clustering.pptx slide 14, region below the title, downscaled to 1800 px wide.
- `images/dendrogram-cut.png` — source: 16-Clustering.pptx slide 17, right-hand dendrogram group plus the "Dendrogram" label.
- `images/distribution-vs-density-clusters.png` — source: 16-Clustering.pptx slide 23, both columns below the title, downscaled to 1800 px wide.
- `images/gmm-mixture-sketch.png` — source: 16-Clustering.pptx slide 25, region below the bullets; a sliver of bullet text at the top edge painted white; downscaled to 1800 px wide.
- `images/em-known-vs-unknown-sources.png` — source: 16-Clustering.pptx slide 26, right-hand ink sketch.
- `images/dbscan-sketch.png` — source: 16-Clustering.pptx slide 28, right-hand ink sketch.
