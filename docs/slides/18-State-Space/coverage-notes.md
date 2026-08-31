# 18-State-Space — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026-06**: Added a `.vignette` box on the privacy implications of high-fidelity
  synthetic network traffic (arXiv:2511.20497, "Quantifying the Privacy Implications of
  High-Fidelity Synthetic Network Traffic"), which introduces membership-inference-based
  privacy metrics specific to network traces. This grounds the fidelity–diversity–privacy
  trilemma in a concrete, dated result and directly complements the NetSSM paper.
- **2026-06**: Framed the comparison table (GAN → Diffusion → SSM) from the book's
  chapter structure rather than the original Marp slide's flat list, aligning
  terminology (DoppelGANger, NetShare, NetDiffusion, NetSSM) with the book's notation.
- **2026-06**: Used the NetSSM ACM Networking publication (Proc. ACM Networking, Vol. 4,
  CoNEXT1, March 2025; arXiv:2503.22663) as the primary citation anchor rather than the
  earlier SIGCOMM workshop feasibility study (2024).

## Suggested missing coverage on broad themes

- **Formal SSM math**: The slides give the O(n) intuition but skip the state-transition
  equations (x_t = A x_{t-1} + B u_t; y_t = C x_t). A single equation slide would
  bridge to students who want the precise formulation.
- **Differential privacy for generative models**: The privacy vignette raises DP but
  the deck does not explain how DP is applied during training (e.g., DP-SGD). A follow-on
  slide or lab exercise on the noise-calibration tradeoff would close this gap.
- **NetFound / foundation models**: The book covers NetFound as a networking foundation
  model. A brief mention on the "Open Problems" slide is included but a dedicated slide
  comparing NetFound's generalist approach to NetSSM's specialist approach would deepen
  the discussion.
- **QUIC and UDP generation**: NetSSM is TCP-focused. As QUIC displaces TCP/TLS,
  extending the evaluation to QUIC flows is an important open problem worth a dedicated
  slide or homework problem.
- **Evaluation dataset details**: The deck cites 10 applications (video streaming,
  conferencing, social media) but does not name the specific datasets. Instructors should
  check whether UNIBS, CAIDA, or custom UChicago captures were used and add a dataset
  provenance note.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed):

- **Privacy vignette (arXiv:2511.20497)**: verify whether this preprint has been
  published in a venue by the next refresh cycle; if so, update to journal/conference
  citation. Also check whether follow-up papers on network-specific DP have appeared.
- **NetSSM citation status**: the paper was published in Proc. ACM Networking (March 2025,
  doi:10.1145/3786289). Verify at next refresh whether a longer journal version or
  follow-on (QUIC/UDP extension) has appeared.
- **Mamba architecture**: Mamba-2 (Dao & Gu, 2024) is the backbone. Check whether a
  Mamba-3 or successor architecture has displaced it by the next teaching cycle.
- **Comparison table numbers**: JSD 0.02, 0.97 accuracy, 8× context, 78× generation
  length are from the NetSSM paper. If a newer method surpasses these, update the table
  and reframe the "best in class" claim.
- **Foundation models (NetFound, NetLLM)**: both are mentioned as open directions;
  verify their publication and adoption status each year.

## Curated images

Images used from `figures/`:
- `figures/pipeline.png` — the NetSSM three-stage pipeline diagram; genuinely teaches
  the pre-processing → training → generation flow. **Kept.**
- `figures/mixing_rate_accuracy.png` — the synthetic data mix rate vs. accuracy plot;
  the flat NetSSM line vs. degrading competitors is the core empirical argument. **Kept.**
- `figures/avg_size_kde.png` — packet size KDE comparison; good visual confirmation of
  statistical fidelity. **Kept.**
- `figures/evaluation_simple.png` — summary bar chart; redundant with the inline tables
  in the results slides, so not referenced directly in the deck but available for
  instructors who prefer a visual. **Dropped from deck; kept in folder.**
- `figures/evaluation_detailed.png` — per-application breakdown; useful for a deeper
  seminar but too dense for a lecture slide. **Dropped from deck; kept in folder.**
- `figures/ks_sizes.png` — KS statistic plot; covers similar ground as the KDE; kept in
  folder for reference but not included in the deck to avoid redundancy. **Dropped.**

No clip-art, logos, or decorative images were included.

## Source

- Rebuilt from `slides.md` (Marp, 29 slides) + `speaker-notes.md`
- Book chapter: "Chapter 7: Generative Models" in *Machine Learning for Networking*
  (Feamster et al.); specifically the "State Space Models" and "NetSSM" subsections,
  and the "Privacy-Preserving Models" section for the vignette.
- Primary paper: Chu, Jiang, Liu, Bhagoji, Bronzino, Schmitt, Feamster,
  "NetSSM: Multi-Flow and State-Aware Network Trace Generation using State-Space Models,"
  *Proceedings of the ACM on Networking*, Vol. 4, CoNEXT1, March 2025.
  arXiv:2503.22663. DOI:10.1145/3786289.
- Current-events vignette source: arXiv:2511.20497,
  "Quantifying the Privacy Implications of High-Fidelity Synthetic Network Traffic" (2025).
