# 14-nPrint — Instructor Notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-Events Updates Made

- **2026**: Added a verified current-events vignette based on Akem et al., "Real-Time
  Encrypted Traffic Classification in Programmable Networks with P4 and Machine Learning,"
  *International Journal of Network Management*, 2025 (Wiley, DOI: 10.1002/nem.2320).
  The study deploys tree-based classifiers on P4 programmable switches at line rate
  with sub-microsecond latency and <10% hardware resource use. This ties the
  representation-learning theme directly to production deployment concerns and to the
  question of which nPrint columns survive encryption.

## Suggested Missing Coverage on Broad Themes

- **Temporal representations**: nPrint represents packets independently; the deck notes
  that RNNs or attention mechanisms are needed for temporal ordering, but does not show
  a concrete architecture. A follow-up slide or lab demonstrating a simple LSTM or
  Transformer over nPrint rows would strengthen this point.
- **Contrastive and self-supervised learning for traffic**: Recent work (2023–2025) applies
  self-supervised pre-training to traffic flows, analogous to BERT pre-training for text.
  This is a natural extension of the nPrint/representation-learning story.
- **Privacy implications of packet-level representation**: nPrint includes IP addresses,
  ports, and payload bytes — a privacy-sensitive representation. The deck does not
  discuss anonymization, differential privacy, or what practitioners strip before
  sharing datasets. Worth at least one slide.
- **Encrypted DNS / ECH and its impact on passive measurement**: The book covers DNS-based
  service identification, and the measurement.rst chapter notes that DoH makes
  DNS-based identification impossible. A concrete example of how this changes the
  feature space would complement the nPrint discussion.
- **Foundation models for traffic**: 2024–2025 saw the first large-scale pre-trained
  "traffic foundation models" (e.g., NetBench, arXiv 2403.10319). This is an emerging
  area directly downstream of the representation-learning argument in this deck.

## Next-Year Refresh Notes

Items with a shelf life — verify each at the start of each term:

- **Akem et al. 2025 vignette**: The *IJNM* paper was published in 2025. By 2027 there
  will likely be newer P4-based or SmartNIC deployment papers. Re-verify that this is
  still the freshest production example; replace if a 2026 paper demonstrates similar
  ideas with more real-world scale.
- **pcapML benchmarks leaderboard** (nprint.github.io/benchmarks): The leaderboard is
  live and actively maintained as of March 2026. Re-check annually; if it has been
  superseded by a new benchmark suite, update the reference on the "From Bespoke to
  Generalizable" slide.
- **nprintML / AutoML integration**: The nprintML GitHub repo was last updated in early
  2025. If the project is superseded or integrated into a larger framework, update the
  workflow slide.
- **CIC-IDS2017 TTL and Heartbleed artifacts**: These are pedagogically stable (the
  papers documenting them predate 2022). No shelf-life risk, but verify that the
  specific slide images (s08-i34, s09-i35) are still legible at the projected
  resolution before each term.

Stronger vignette considered but not used:
- NetBench (arXiv 2403.10319, 2024): a large-scale traffic benchmark for foundation
  models. Could replace the Akem vignette if the audience is more research-oriented
  and less operations-oriented.

## Curated Images

Images used:
- `s08-i34.png` — KDD'98 class-distribution table showing divergence across three
  independent analyses of the same dataset. Excellent reproducibility argument.
- `s09-i35.png` — CICIDS2017 class-distribution divergence table (five papers). Same
  point, stronger because the dataset is more recent.
- `s10-i36.jpg` — TCP vs. UDP header format comparison. Illustrates the alignment
  problem directly.
- `s13-i39.png` — The canonical nPrint bitmap diagram (IPv4/TCP/IP, IPv4/UDP/IP rows
  showing 1/0/−1 encoding). This is the single most important diagram in the deck.

Images dropped:
- `s03-i01.png` through `s06-i29.png` (various) — mostly clip-art icons (magnifying
  glass, document stack, database cylinder) used as decorative slide chrome in the
  original PowerPoint. No informational content.
- `s14-i40.png`, `s15-i41.png`, `s16-i42.png`, `s17-i43.png` — duplicate copies of
  the nPrint bitmap diagram; one (s13-i39.png) is sufficient.
- `s19-i44.png` — faded Pensieve/ABR agent diagram. Belongs to a different lecture
  (adaptive bitrate / RL). Out of scope for this deck.
- `s33-i57.png` — netUnicorn architecture diagram (client → Unicorn Installation →
  Infrastructure). The netUnicorn content was consolidated to one descriptive slide
  rather than a full sub-section; the diagram adds detail not needed at this level.
- All remaining images (s04 through s41 series): decorative arrows, generic flowchart
  boxes, or diagrams belonging to the netUnicorn/modular-experiment sub-topic that
  the rebuilt deck handles via prose rather than slide-for-slide.

## Source

- Rebuilt from `_source-extract.md` (43 slides) + book Chapter 3 "Network Data"
  (`measurement.rst`, section "Learning Traffic Representations with nPrint").
- The original deck split roughly 50/50 between nPrint/representation topics
  (slides 1–18) and netUnicorn/data-collection modular-experiment topics
  (slides 19–43). This rebuild focuses on the representation-learning theme
  (slides 1–18 of the original) and adds a short treatment of netUnicorn
  as context for the "data collection matters too" point. The full netUnicorn
  workflow and infrastructure abstraction material belongs in a dedicated
  data-collection lecture.
- Divergences from source: the original deck did not include pcapML; the book
  chapter discusses it explicitly and it was added here for completeness.
  The original slide 18 was blank; ignored.
