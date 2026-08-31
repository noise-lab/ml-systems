# 15-Dimensionality-Reduction — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026-06**: Added a `.vignette` box on the February 2026 Scientific Reports paper
  "Feature importance-guided autoencoder for dimensionality reduction in intrusion
  detection systems" (FI-AE; DOI: 10.1038/s41598-026-36695-9). This is a peer-reviewed,
  dated, primary-source hook demonstrating that the design of dimensionality reduction
  directly affects IDS accuracy on high-dimensional, imbalanced network datasets.

## Suggested missing coverage on broad themes (point 3)

- **UMAP** (Uniform Manifold Approximation and Projection): The book covers PCA, T-SNE,
  and autoencoders but not UMAP. UMAP is increasingly used in network traffic analysis
  (see 5G traffic intrusion paper, arXiv 2312.04864) and often outperforms T-SNE on
  larger datasets. A one-slide comparison of T-SNE vs. UMAP tradeoffs would strengthen
  the deck.
- **Kernel PCA hands-on:** The book mentions kernel PCA but the deck only touches it
  briefly. A worked example showing when linear PCA fails (circular clusters) and kernel
  PCA succeeds would concretize the concept.
- **Variational Autoencoders (VAEs):** The book discusses VAEs as a variant. A slide on
  VAE latent space structure and its use for synthetic traffic generation (e.g., as a
  precursor to the diffusion/generative lectures) would provide better continuity.
- **Reconstruction error distributions:** A practical slide showing how to set an
  anomaly threshold on reconstruction error (e.g., ROC curve over a validation set) is
  missing and directly relevant to lab work.
- **Connection to nPrint (Meeting 14):** The deck notes that nPrint produces 100–1600
  bit features but does not show a worked example of PCA or autoencoder compression on
  an nPrint feature matrix. A concrete code/figure example would strengthen the
  cross-lecture connection.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items placed in this build
that will age:

- **FI-AE vignette (Feb 2026):** Verify whether the paper has been cited, extended, or
  superseded by competing feature-guided autoencoder approaches by 2027. Check for
  follow-up work by the same group or NDSS/IEEE S&P papers citing this work.
- **5G + PCA/T-SNE paper (arXiv 2312.04864):** Check whether a journal version has
  appeared and if 6G traffic analysis studies have superseded it.
- **nPrint reference:** nPrint has been actively developed; verify the current feature
  dimensionality (100–1600 claim) still holds with the latest release.
- Stronger alternative vignette considered but not used: the August 2025 IECSCIENCE
  journal paper on PCA efficiency for IDS (JOURNAL OF NETWORKING AND NETWORK
  APPLICATIONS, Vol. 5, Issue 1, Aug 2025). It was not used because the Feb 2026
  FI-AE paper is more current and makes a stronger teaching point about guided
  compression. Re-evaluate if FI-AE becomes stale.

## Curated images

Images used:

- `s05-i01.png`: PCA arrows overlaid on 2D scatter — illustrates principal component
  directions and relative magnitudes. Kept.
- `s05-i02.png`: Data projected onto PC1 (orange dots) — concretely shows the 1D
  reduction. Kept.
- `s06-i03.png`: Input vs. principal component coordinate axes — shows the change-of-
  basis interpretation. Kept.
- `s07-i04.png`: First PC loading equation — formula directly from source slides. Kept.
- `s07-i05.png`: Maximization objective with unit-norm constraint — supports the
  computation slide. Kept.
- `s08-i06.png`: Scree plot (cumulative explained variance) — the key practical guide
  for choosing number of components. Kept.
- `s09-i08.png`: T-SNE visualization of DNS packets (red) in a packet capture. Kept —
  directly illustrates that T-SNE reveals protocol-level cluster structure.
- `s10-i09.png`: Encoder-bottleneck-decoder architecture diagram. Kept — the clearest
  illustration of autoencoder structure.

Images dropped:

- `s08-i07.png`: A degenerate/near-flat scree plot (likely a placeholder with
  near-constant values 0.0–1.0 on a 0–1 x-axis). Dropped — uninformative and likely
  a rendering artifact from the source PowerPoint. Not referenced in the deck.

## Source

- Rebuilt from `_source-extract.md` (10 slides) — content-driven rebuild, not a
  slide-for-slide port.
- Primary book alignment: Chapter 6 "Unsupervised Learning" → "Dimensionality
  Reduction" section of *Machine Learning for Networking* (`text/unsupervised.rst`),
  covering PCA (Pearson 1901), T-SNE (van der Maaten & Hinton 2008), and Autoencoders
  (Hinton 2006).
- Book superseded the source slides on: (1) explicit coverage of kernel PCA, (2) the
  stochastic nature of T-SNE and guidance on reproducibility, (3) the autoencoder
  anomaly detection use case, (4) the autoencoder + clustering (deep clustering)
  workflow, and (5) autoencoder variants (VAE, sparse, denoising, convolutional).
  These are all covered or noted in the deck/coverage notes but were absent from the
  original 10-slide PowerPoint.
