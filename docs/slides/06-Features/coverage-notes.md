# 06-Features — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026:** Added a `.vignette` based on Wickramasinghe et al., "SoK: Decoding the
  Enigma of Encrypted Network Traffic Classifiers" (arXiv:2503.20093, March 2025;
  accepted IEEE S&P 2025). This paper directly motivates why feature choice and
  dataset currency matter: the majority of published encrypted-traffic classifiers
  were found to have been trained on unencrypted legacy data, making their results
  not applicable to modern TLS 1.3 / QUIC deployments. The arXiv link is stable
  and the paper is freely available.

## Suggested missing coverage on broad themes

- **Dataset provenance and labeling:** The SoK paper raises this sharply. A full
  treatment of *how* to label network traffic (ground truth: process-level labels,
  DNS side-channel, application instrumentation) would fit naturally after the
  representation section.
- **Flow vs. packet vs. session granularity:** The slides touch on this but a
  worked example comparing classifier accuracy at each granularity on the same
  dataset would make the tradeoff concrete.
- **Privacy-preserving feature extraction:** GDPR, state-level privacy laws, and
  the shift to encrypted DNS are noted but not developed. A dedicated discussion
  of what features are legally/technically available in different regulatory
  jurisdictions would be valuable as the field matures.
- **netml vs. CICFlowMeter vs. tshark feature sets:** Students in the lab
  sometimes wonder why different tools produce slightly different feature counts
  for the same pcap. A comparison slide would preempt confusion.
- **Distribution shift:** Features extracted from 2020 training data may not match
  2026 traffic (application mix, TLS version, QUIC adoption). A brief discussion
  of concept drift in traffic features would connect to later pipeline lectures.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items likely to age:

- **SoK vignette (2025):** By 2027, this paper will be two years old. Check for
  follow-on work that either replicates/extends the findings or proposes solutions.
  The arXiv ID is 2503.20093; monitor for citation-based follow-ups.
- **TLS 1.3 / QUIC / ECH adoption stats:** The slide notes that ECH encrypts SNI
  in "modern TLS" — verify the actual browser/CDN adoption rate each year.
  Cloudflare, Mozilla, and Google publish ECH deployment statistics.
- **netml version:** The library was active as of late 2025; verify the GitHub
  (github.com/noise-lab/netml) is still maintained and the API has not changed
  before the lab assignment.
- **nPrint GitHub activity:** nPrint's GitHub (github.com/nprint) showed updates
  through December 2025. Verify it remains active; if abandoned, note alternatives.
- **"LLM-assisted feature engineering" note on slide 14:** This is a forward-
  looking comment; revisit whether it's still speculative or now well-supported
  by published results.

## Curated images

- **`images/s06-i01.png`** — USED. A clean table showing which feature types
  (Duration, IAT, SIZE, FFT, SAMP-NUM, SAMP-SIZE) are used across published
  anomaly detection papers, grouped by task (intrusion detection, IoT, DDoS).
  This is the most informative figure from the source pptx and directly supports
  the "which representation for which task" slide.
- **`images/s07-i02.png`** — USED (sparingly). A text-block extract from a
  research paper summarizing key findings about representation effectiveness.
  Placed in the two-column "What Representations Actually Capture" slide as a
  visual anchor. It is a dense text block, not a diagram, so it is secondary.
- No other images were extracted from the source pptx. The source had only 7
  slides with minimal graphics; the two images above represent the full
  extractable visual content.

## Source

- Rebuilt from `_source-extract.md` (7 slides) — the original pptx was sparse,
  providing a topic outline and feature taxonomy but little pedagogical structure.
- Primary alignment: *Machine Learning for Networking*, Chapter 3 "Network Data"
  (measurement.rst) — specifically the sections "Types of Network Data," "Data
  Collection Strategies," "Passive Measurement," "Data Structures to Features,"
  and "Learning Traffic Representations with nPrint."
- Current-events vignette: Wickramasinghe et al., arXiv:2503.20093 (S&P 2025).
