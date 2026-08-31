# 13-Deep-Learning — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026 (initial build):** Replaced the original source-extract's survey-style
  content with a verified 2025 primary-source vignette: "Deep learning for network
  security: an Attention-CNN-LSTM model for accurate intrusion detection,"
  *Scientific Reports* (Nature), July 1 2025,
  doi:10.1038/s41598-025-07706-y. Results cited: 97.5% accuracy on Bot-IoT,
  94.8% on NSL-KDD, sub-35 ms real-time inference. This is open-access and
  verifiable. The vignette illustrates the hybrid CNN+LSTM+attention pattern
  that has become standard in production intrusion detection systems.

## Suggested missing coverage on broad themes (point 3)

- **Transformer / attention for networking:** The Attention-CNN-LSTM vignette
  previews attention, but a dedicated slide on transformers applied to traffic
  (e.g., ET-BERT, NetBench, or foundation models for packets) would strengthen
  the bridge to the Chapter 7 generative models lecture.
- **Transfer learning and fine-tuning:** The book mentions "use existing
  architectures proven on similar tasks" but the deck does not cover
  fine-tuning a pre-trained network on network traffic. Relevant for practical
  labs where labeled network data is scarce.
- **Adversarial examples for network classifiers:** Deep learning models are
  known to be vulnerable to adversarial inputs; in the traffic classification
  context (evasion attacks on IDS), this is an important security caveat.
  Currently noted only in the nPrint interpretability slide.
- **Quantitative comparison across architectures:** A table showing MLP vs.
  CNN vs. LSTM vs. hybrid on the same traffic dataset would give students a
  concrete anchor. The nPrintML table (s28-i39) partially fills this but is
  not architecture-decomposed.

## Next-year refresh notes

Refresh per `../TEMPLATE.md` → "Annual current-events refresh" (web-verify;
swap only for something fresher and confirmed). Items likely to age:

- **Vignette (2025 Attention-CNN-LSTM paper):** Verify the paper remains
  state-of-the-art; if a 2026/2027 paper supersedes it with a clearly better
  or more compelling story, swap. The doi is stable so the link will not rot,
  but the accuracy figures (97.5%, 94.8%) may be eclipsed by newer work.
- **nPrint/nPrintML:** The CCS 2021 paper is the canonical citation; verify
  whether follow-on work (e.g., new AutoGluon versions, pcapML, or related
  tools) has superseded the pipeline description.
- **LSTM vs. transformers:** The book states LSTMs are "largely supplanted
  by foundation models." If by 2027 the community has mostly moved to
  transformer-based traffic classifiers as the production standard, the
  LSTM/GRU slide should be shortened and a new transformer-for-traffic slide
  promoted from "missing coverage" to a full slide.
- **Bot-IoT and NSL-KDD benchmarks:** These datasets are aging (NSL-KDD
  dates to 2009). Check whether fresher benchmarks (UNSW-NB15, CIC-IDS-2018,
  CAIDA) have become the standard by the next refresh.

## Curated images

Images used (all from `images/`):

| File | Slide | Why used |
|---|---|---|
| `s25-i31.png` | MLP slide | Simple feedforward network diagram — clean, protocol-agnostic |
| `s18-i25.png` | nPrint slide | nPrint bit-vector layout — essential for understanding the representation |
| `s27-i38.png` | nPrint interpretability slide | Feature importance heatmap over IP/TCP header fields — compelling result |
| `s28-i39.png` | nPrintML results slide | Breadth-of-task accuracy table — anchors the "universal pipeline" claim |
| `s39-i43.png` | CNN architecture slide | VGG-16 block diagram — illustrates conv+pool hierarchy pattern clearly |

Images dropped:

- `s15-i01` through `s17-i24` — icons and flowchart fragments from the
  classic-pipeline / bespoke-solutions animated sequence; the concept is
  conveyed in prose; individual icons add no information.
- `s25-i30, s25-i32–s25-i36` — decorative icons (document stack, gears,
  network node, scale, resize arrow); no instructional content.
- `s25-i34` — gear/refresh icon: decorative.
- `s26-i37` — Nmap packet transformation context; the Nmap vs. nPrint
  comparison belongs in the dedicated nPrint (deck 14) lecture, not here.
- `s29-i40` — duplicate of s28-i39 (same results table).
- `s30-i41` — unclear context (blank caption in source extract).
- `s39-i42` — VGG-16 feature visualization photo (Zeiler & Fergus); useful
  but requires careful attribution and the `s39-i43` block diagram is cleaner.
- `s40-i44` — low-resolution VGG-16 visualization; redundant with s39-i43.

## Source

- Rebuilt from `_source-extract.md` (41 slides, 13-Deep-Learning.pptx).
- Book alignment: *Machine Learning for Networking*, Chapter 5 ("Supervised
  Learning"), "Deep Learning" subsection — covering MLP, CNN, RNN/LSTM, and
  the nPrint/nPrintML case study.
- Deck was substantially restructured: the original split the content across
  "Deep Neural Nets," "Representation Learning in Network Traffic" (nPrint),
  and "Convolutional Neural Nets" as separate mini-units. The rebuild
  integrates them into a coherent narrative: motivation → neuron → MLP →
  training → nPrint case study → CNNs → RNNs → current events → summary.
- Slide count: 20 content slides (including 2 section dividers), well within
  the 15–30 target.
