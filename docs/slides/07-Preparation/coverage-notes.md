# 07-Preparation — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026 (initial build):** Added a `.vignette` box on the Kapoor & Narayanan
  reproducibility-crisis paper (*Leakage and the Reproducibility Crisis in
  ML-based Science*, Patterns vol. 4 no. 9, September 2023; running leakage table
  updated May 2024 at reproducible.cs.princeton.edu). The paper catalogued 41 papers
  across 30 fields where data-leakage errors were found, collectively affecting 329+
  papers. Their follow-up book *AI Snake Oil* (Princeton University Press, September
  2024) was named one of Nature's 10 best books of 2024. This hook is verifiable via
  arxiv.org/abs/2207.07048 and the Princeton project website.
- **2026 (initial build):** The Finnish pronoun / Google Translate bias example
  (from `s17-i12.png`, tweet by @vuokko, March 2019) was retained as a memorable
  non-representative-data illustration. It is dated but widely recognized; may be
  swapped for a more current bias example in future years.
- **2026 (initial build):** The Banko & Brill (2001) data-quantity graph
  (`s16-i10.png`) was retained; it is a classic result unlikely to go stale but
  should be supplemented with a modern scaling-law example.

## Suggested missing coverage on broad themes (point 3)

- **Synthetic and augmented training data:** the deck mentions class imbalance and
  the need for representative data but does not cover SMOTE, GANs, or
  domain-randomization techniques for generating scarce attack traffic. This is
  increasingly important for network security ML.
- **Label quality and weak supervision:** labeling network traffic at scale requires
  heuristics, crowd-sourcing, or semi-supervised methods. The deck covers the
  mechanics of labeling but not the accuracy / noise properties of the labels
  themselves.
- **Cross-validation for time-series data:** standard k-fold cross-validation breaks
  temporal order; time-series cross-validation (walk-forward validation) is the
  correct approach for network data but is not covered.
- **Feature selection methods:** the deck describes feature types and the danger of
  irrelevant features but does not cover automated selection (mutual information,
  Lasso, recursive feature elimination).
- **Concept drift:** a trained model's distribution assumptions become stale as
  network behavior evolves (new applications, new attack patterns). This is a direct
  consequence of temporal distribution shift and is a gap in the current deck.
- **Data provenance and documentation:** datasheets for datasets, data cards, and
  the importance of recording *where* and *when* data were collected for
  reproducibility.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md → "Annual current-events refresh"`:

- **Kapoor & Narayanan vignette (2023/2024):** The running leakage table at
  `reproducible.cs.princeton.edu` is updated periodically — check for the latest
  figure (currently 41 papers / 329+ affected papers as of May 2024). If the count
  has grown substantially, update the vignette. Also check whether a journal
  replication or retraction study in network security / intrusion detection has
  since appeared — that would be a stronger domain-specific hook.
- **AI Snake Oil book (2024):** Named Nature's best books of 2024. If it has
  generated follow-up coverage, a replication study, or a notable critique by
  2026-2027, that would sharpen the vignette.
- **Finnish pronoun image (s17-i12.png, 2019):** This is increasingly dated. A
  stronger alternative would be a 2025-era bias/representation failure in a
  large language model or a network traffic classifier. Did not find a clean, dated,
  primary-source example in 2026 search — keep the Finnish example for now.
- **Banko & Brill (2001) scaling graph (s16-i10.png):** Classic result; unlikely to
  go stale, but consider adding a modern neural scaling law (Kaplan et al. 2020 or
  later) as a companion or replacement.
- **Google Translate pronoun bias:** Google has periodically updated its translation
  models; check whether the specific Finnish-to-English bias has been addressed and
  update the slide text accordingly if so.

## Curated images

Images used in the rebuilt deck:

- `s10-i06.png` — z-score standardization formula: used on the normalization slide.
  Clean, instructional, worth keeping.
- `s16-i10.png` — Banko & Brill (2001) accuracy-vs.-corpus-size plot: used on the
  "Insufficient Training Data" slide. Directly supports the lecture point.
- `s17-i12.png` — Finnish pronoun / Google Translate tweet: used on the
  "Non-Representative Training Data" slide. Memorable; dated but effective.
- `s19-i13.png` — LIME husky-vs.-wolf explanation figure (Ribeiro et al., KDD 2016):
  used on the spurious-correlations slide. Central to the irrelevant-features
  argument; keep.

Images dropped:

- `s02-i01.png` — generic document icon (decorative clip-art, no instructional value)
- `s02-i02.png` — generic neural-network diagram, hollow circles (decorative; no
  specific architecture depicted)
- `s02-i03.png` — same as s02-i02 but filled circles (decorative)
- `s02-i04.png` — empty 3×3 grid icon (decorative)
- `s02-i05.png` — gear/refresh icon (decorative)
- `s16-i11.png` — citation text image for Norvig et al. "The Unreasonable
  Effectiveness of Data": not referenced in the rebuilt deck directly, but the
  underlying idea is covered via the Banko & Brill plot.
- `s11-i07.tiff`, `s11-i08.tiff`, `s11-i09.tiff` — TIFF format; could not verify
  content. The LIME explanation image (s19-i13.png) already covers the husky-wolf
  example visually.

## Source

- Rebuilt from `_source-extract.md` (19 slides) + book chapter: *Machine Learning for
  Networking*, Chapter 4 "Machine Learning Pipeline" (`/docs/slides/../../../ml/text/pipeline.rst`).
- Book takes precedence over source slides where they diverge (e.g., the book provides
  the TTL spurious-correlation networking example which was not in the original slides).
- The source slides lacked explicit data-leakage / golden-rule content; this was added
  from the pipeline chapter, which covers it extensively under "Dividing Data."
