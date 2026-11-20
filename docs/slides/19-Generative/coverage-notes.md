# 19-Generative — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026-06-03**: Added verified `.vignette` hook for **NetSSM** (Chu et al.,
  arXiv:2503.22663, submitted March 28 2025; published in *Proceedings of the ACM on
  Networking*, DOI: 10.1145/3786289). Verified via ACM DL and arXiv. NetSSM produces
  traces 8× and 78× longer than prior transformer-based approaches, with higher
  statistical fidelity and protocol compliance. This is the strongest current-events
  hook because it directly closes the sequence-length gap identified as NetDiffusion's
  main limitation, and was published within the 2025 calendar year.

- **2026-06-03**: Confirmed NetDiffusion publication venue (ACM SIGMETRICS 2024 /
  POMACS, DOI: 10.1145/3639037) via ACM DL. All paper titles, author surnames, and
  results figures match the source extract and the book chapter.

## Suggested missing coverage on broad themes

- **Federated generative models**: several 2025 papers (FGAN and others) generate
  synthetic traffic without centralizing real data — directly relevant to the
  privacy section but not yet in the book chapter. Worth one slide in a future revision.

- **Evaluation methodology depth**: the "train-on-synthetic / test-on-real"
  paradigm deserves a dedicated slide explaining why this is the hard evaluation and
  what pitfalls exist (distribution shift, metric mismatch between statistical
  similarity and ML accuracy).

- **Adversarial robustness of synthetic-data-trained models**: if a classifier is
  trained on NetDiffusion/NetSSM data, how robust is it to adversarially crafted
  real traffic? Not covered in the book chapter, but a natural extension.

- **Context transfer (zero-shot generation)**: the book chapter flags this as an
  open problem (can a model generalize to VPN-encrypted YouTube if trained on
  VPN-encrypted Netflix?). A worked example of this as a homework or lab would be
  high-value.

- **NetFound**: the foundation model for networking is mentioned briefly in the
  applications section. If it matures by the next revision, it warrants a full slide
  with evaluation results.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events
refresh" (web-verify; swap only for something fresher and confirmed). Items likely
to age:

- **NetSSM vignette (2025)**: if a NetSSM follow-on paper or a competing SSM-based
  traffic generator publishes results that supersede the "8×/78× longer traces"
  claim, swap the vignette for that paper. Check ACM SIGCOMM, SIGMETRICS, and
  IMC proceedings.

- **NetLLM (2024)**: the NetLLM slide references a 2024 paper. If a stronger
  LLM-for-networking result publishes by 2026–2027, consider swapping this example.
  The teaching point (adapting general LLMs via multimodal encoders) is stable; only
  the exemplar paper needs refreshing.

- **Statistical benchmarks in the evaluation slide**: the ">70% improvement" and
  ">60% accuracy improvement" numbers are from the 2024 NetDiffusion paper. If a
  2025/2026 evaluation updates these on a newer dataset, refresh the numbers.

- **Privacy-utility trade-off**: the differential privacy section cites Dwork (2014)
  and Shokri (2017). These foundations are stable; the current-events hook would be
  any concrete 2025–2026 deployment of DP for network data synthesis.

## Curated images

**Used:**

- `images/s23-i29.png` — GAN generator–discriminator diagram (clean, pedagogically
  clear, no branding; used on GAN slide)
- `images/s13-i22.png` — Transformer encoder–decoder architecture diagram (used on
  Transformer slide)
- `images/s35-i37.png` — nPrint-encoded TCP flow as pixel image (key visual for
  NetDiffusion; used on NetDiffusion slide)
- `images/s33-i35.png` — NetDiffusion full pipeline overview diagram (used on
  pipeline overview slide)
- `images/s46-i45.png` — Mixed synthetic+real training accuracy comparison plot
  (used on evaluation slide; shows NetDiffusion vs. NetShare as mixing rate varies)
- `images/s20-i27.png` — t-SNE plot of BERT embeddings for TLS misconfiguration
  (used on BERT/TLS slide)

**Dropped:**

- `images/s26-i30.png`, `s26-i31.png`, `s26-i32.jpg` — padlock/security clip-art;
  purely decorative, no informational content
- `images/s03-i01.png` through `s07-i18.png` (various TLS slides 3–7) — TCP/IP
  layer stack diagram and TLS protocol detail slides; the deck focuses on generative
  models, not TLS background
- `images/s30-i33.png`, `s30-i34.png` — NetFlow tabular data example; informational
  but not visually distinct enough to warrant a slide of its own
- `images/s38-i38.png`, `s38-i39.png`, `s38-i40.png` — ControlNet edge detection
  intermediate outputs; useful in a research talk but too detailed for a course
  lecture at this level
- `images/s40-i41.png`, `s41-i42.png` — post-processing dependency tree traversal
  outputs; same rationale as above
- `images/s43-i44.png` — evaluation dataset table; data is summarized in text
- `images/s47-i46.png`, `s47-i47.png` — class-balancing accuracy bar charts; the
  key result is summarized in the evaluation text
- `images/s48-i48.png`, `s49-i49.png` — Wireshark/TCPreplay compatibility table;
  adequate as a bullet point
- All title-slide chrome images (headshots, university logos)

## Source

- Rebuilt from `_source-extract.md` (53 slides, primarily NetDiffusion and BERT/TLS
  material) and aligned to *Machine Learning for Networking*, **Chapter 7: Generative
  Models** (generative.rst).
- Where the source extract focused heavily on TLS background (Slides 2–22), the deck
  consolidates to one slide (BERT for TLS Misconfiguration) following the book's
  ordering: GAN → Diffusion → Transformers → SSMs → Applications.
- The source extract contained no NetSSM content; NetSSM was added from the book
  chapter and verified against the published ACM paper (arXiv:2503.22663).
- State Space Models are treated briefly here as a bridge to Deck 18 (NetSSM), which
  covers them in full depth.

## Figures added (2026-11-20 graphics pass)

- `images/generative-landscape-timeline.svg` (generated, `figures.py`): GAN → diffusion → transformer → SSM, general-ML lane above, networking lane below; new slide "Four Generations of Generative Models".
- `images/attention-vs-ssm-cost.svg` (generated, `figures.py`): n² vs n on log axes with 1,000 / 10,000 / 100,000 packets marked; replaces the operations table on "The Scaling Problem".
- Graphviz diagrams in `slides.qmd`: forward/reverse diffusion ("How Diffusion Models Work"), four-step NetDiffusion strip above the paper's Fig. 1 ("NetDiffusion Pipeline Overview"), dependency-tree repair flow ("Protocol Compliance via Post-Processing"), words/packets → tokens → embeddings → attention ("Embeddings"), SSM recurrence ("State Space Models and Mamba").
- Legacy clippings kept under the landmark rule: GAN flow, Transformer encoder-decoder, nPrint image, NetDiffusion Fig. 1 and Fig. 5, NetShare tables, BERT/TLS t-SNE, pipeline and tokenization figures.
- Font gate: `.smaller` dropped from the privacy slide, the NetSSM dated note, and the summary table (trimmed to six columns); GAN-for-traffic, NetSSM and NetLLM bullets trimmed to stop footer overflow.
- Text: "Current Events: NetSSM in 2025" retitled "Where the Field Stood in 2025: NetSSM at CoNEXT" and written as a dated note; NetSSM venue is *Proc. ACM on Networking* (CoNEXT 2025); NetDiffusion author attribution corrected from "Liu et al." to Jiang et al.; Mamba dated 2023.
- `speaker-notes.md` written (one table, one row per slide).
