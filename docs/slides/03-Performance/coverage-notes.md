# 03-Performance — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- 2026: Added a `::: {.vignette}` box citing Berger et al., "Video QoE Metrics from
  Encrypted Traffic: Application-agnostic Methodology," arXiv:2504.14720, April 2025.
  This paper directly addresses the "excluded model" limitation shown in the
  Bronzino et al. deployment results (AP=0.29 for unseen services), proposing an
  approach that works across arbitrary proprietary video-call apps without per-service
  engineering. Source verified at https://arxiv.org/abs/2504.14720 on 2026-06-03.

## Suggested missing coverage on broad themes (point 3)

- **ABR and reinforcement learning:** The book's "Coding and Rate Control" section
  covers Pensieve-style RL for adaptive bitrate selection. This deck focuses on
  *inference* (observing quality passively) but the complement — *controlling* quality
  via learned ABR — deserves a slide or a pointer.
- **QUIC/HTTP3 impact on inference:** QUIC encrypts more of the transport header
  than TLS-over-TCP. Existing segment-boundary detection techniques may degrade.
  A slide on what changes under QUIC would be timely (HTTP3 adoption is now >30%
  of web traffic).
- **Adversarial robustness:** Services can add random padding to defeat size-based
  inference (e.g., Netflix's known anti-fingerprinting measures). A brief discussion
  of the cat-and-mouse dynamic would strengthen the lecture.
- **Performance prediction (what-if scenarios):** The book's "Performance Prediction"
  section (web service response time, WISE regression) is not covered here. It could
  be a short subsection or a separate deck.
- **Network provisioning and traffic forecasting:** Resource allocation and demand
  forecasting (cellular tower prediction, ISP node split) from the book's
  "Resource Allocation" section are likewise omitted. Could be added to this deck
  or given their own deck.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events
refresh" (web-verify; swap only for something fresher and confirmed):

- **arXiv:2504.14720 vignette (April 2025):** Re-check whether a peer-reviewed
  venue version has appeared (conference or journal publication). Replace the arXiv
  citation with the published venue if so. Also check for follow-up work on
  application-agnostic QoE inference that may supersede this result.
- **Cisco VNI traffic forecast figure (s04-i01.png):** The forecast runs through
  2022; it is already stale. Replace with a more recent traffic volume statistic
  (e.g., Sandvine Global Internet Phenomena report or Ericsson Mobility Report,
  whichever is most recent). The teaching point (video dominates) will remain true,
  but the absolute numbers and chart should be updated.
- **Page-load-time saturation plot (s06-i02.png):** This is a 2013 result. The
  general principle (throughput saturation) still holds, but new measurements for
  modern protocols (HTTP/2, HTTP/3, QUIC) on 4K video or interactive applications
  would be more compelling.
- **Deployment statistics ("~60 US homes, ~10 Paris, 210k sessions, 14 months"):**
  These are from Bronzino et al. (2019). If newer large-scale deployment papers have
  appeared, consider updating. The numbers themselves are not the main teaching point,
  but readers will notice they are dated.
- **TSKAN (arXiv:2509.20595, September 2025):** Found during research but not
  included because it focuses on DASH QoE modeling methodology rather than
  offering a concrete deployment result suitable for a vignette. Could replace the
  vignette if a stronger result is published with it, or if the Berger et al. paper
  is subsumed by newer work.

## Curated images

**Used:**
- `s04-i01.png` — Cisco VNI video traffic growth bar chart (2017–2022). Useful
  data plot; dated but the trend is pedagogically valid.
- `s06-i02.png` — Page load time vs. throughput (Krishnan 2013). Classic result,
  motivates why throughput alone doesn't explain experience.
- `s12-i17.png` — Netflix session scatter plot (Kbps vs. time, colored flows).
  Shows what encrypted traffic looks like to an operator.
- `s19-i45.png` — TCP segment boundary detection diagram. Key empirical insight
  for the application-layer feature engineering section.
- `s22-i56.png` — Startup delay vs. speed tier boxplots. Deployment result.
- `s23-i58.png` — Resolution distribution vs. speed tier stacked bars. Core
  "surprising result" finding.
- `s38-i92.png` — Precision-recall per service (Netflix/YouTube/Amazon/Twitch).
- `s39-i93.png` — Composite vs. Specific vs. Excluded P/R curves. Critical
  generality result.
- `s40-i94.png` — Feature importance (GINI, Netflix and YouTube). Validates
  segment-size dominance.
- `s41-i95.png` — Feature layer comparison P/R curves (Net, Net+Tran, Net+App, All).
- `s46-i96.png` — Packet size difference CDF for VCA frame-boundary detection.
- `s47-i97.png` — VCA feature table (flow, IP/UDP, RTP categories).
- `s48-i98.png` — FPS inference error boxplots (Meet, Teams, Webex).
- `s27-i64.png` — AC-DC ensemble framework diagram (fast model serving,
  source slide 27). Added on image-coverage pass: only graphic covering the
  model-serving challenge; pairs with the "real-time inference at line rate"
  open problem.
- `s29-i66.png` — WSJ headline "The Truth About Faster Internet: It's Not
  Worth It" (source slide 29). Added on image-coverage pass as the
  representative of the three WSJ clippings (`s29-i65` masthead and
  `s29-i67` caption text omitted as fragments of the same visual).

**Dropped:**
- `s11-i12.png` through `s11-i16.png` — loading spinner and decorative icons.
- `s15-i18.png` through `s15-i30.png` — clip-art cloud, laptop, and other
  decorative diagram fragments from PowerPoint animations.
- `s17-i31.png` through `s17-i44.png` — more clip-art clouds and arrows
  (PowerPoint animation layers).
- `s21-i46.png` through `s21-i55.png` — small clip-art laptop images.
- `s07-i03.png`, `s07-i04.png`, `s07-i05.png` — decorative images not
  identifiable as data plots.
- `s08-i06.png`, `s09-i07.png`, `s09-i08.jpg`, `s09-i09.png`, `s10-i10.png`,
  `s10-i11.png` — decorative or unidentifiable visuals.
- `s26-i62.png`, `s26-i63.png` — nPrint-encoded pcap pixel representation
  (belongs to Lecture 14 on nPrint, not this performance inference deck).
- `s22-i57.png`, `s23-i59.png` — "vs. measured speed" companions to the used
  "vs. nominal speed" plots (`s22-i56`, `s23-i58`); redundant on the slide.
- `s24-i60.png` — feature importance for the all-features model (throughput
  dominates); the used `s40-i94` (Net+App model, segment size dominates) makes
  the pedagogical point.
- `s25-i61.png` — earlier version of the VCA FPS-error boxplot; superseded by
  the used `s48-i98`.
- `s29-i65.png`, `s29-i67.png` — WSJ masthead and caption fragments; the
  headline `s29-i66` (now used) represents the visual.
- `s30-i68.png` through `s35-i91.png` — clip-art animation layers, exact
  duplicates of already-used plots (`s31-i80` = `s12-i17`, `s32-i81` =
  `s11-i12`), same-family variants of used deployment plots (`s33-i83/84`),
  and blank US/Europe map backgrounds (`s35-i85` through `s35-i91`).
- `s50-i99.png` — lab exercise screenshot, not a teaching diagram.

## Source

- Rebuilt from `_source-extract.md` (51 slides) + book chapter "Motivating
  Problems" (motivation.rst), "Performance Inference" and "Quality of Experience
  Inference" sections.
- Primary research reference: Bronzino et al. (2019), "Inferring Streaming Video
  Quality from Encrypted Traffic: Practical Models and Deployment Experience,"
  ACM POMACS 3(3). DOI: 10.1145/3366704.
- Current-events vignette: Berger et al. (April 2025), arXiv:2504.14720.
