# 04-Resource — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026:** Added a `.vignette` box on the **PLL-ABR** paper (Zhang, Han, Cai, Feng, Zhu;
  AIP Advances 15, 075042, published July 1 2025, DOI 10.1063/5.0276888), which extends
  Pensieve's RL-for-ABR framing with PPO + LSTM + real-time QoE feedback. Verified via
  the AIP Publishing page. This replaces the old slide's implicit assumption that Pensieve
  (2017) was the state of the art — the research line is clearly alive and well in 2025.

- **2026:** Retained the COVID-19 ISP traffic data (s20–s22 images) as the model-drift
  case study. These figures appear to be from Feamster-lab research (2018–2020 window).
  The teaching point (out-of-sample shocks expose model brittleness) remains entirely
  current. The specific COVID episode is now far enough in the past to be a stable
  pedagogical example rather than "news."

- **2026:** Added reference to APNIC Blog (January 15, 2025) on "Reliable IoT network
  traffic inference" and concept drift in residential ISP networks, as a note in the
  Model Drift slide's speaker notes. Not surfaced in a vignette (the PLL-ABR vignette
  is sufficient), but flagged for future use.

## Suggested missing coverage on broad themes (point 3)

- **Network scheduling / packet prioritization via RL:** The deck covers congestion
  control and ABR but does not discuss ML-based scheduling (e.g., DeepMind's work on
  data-center job scheduling, or Decima for DAG scheduling). A slide on this would
  complete the resource-allocation picture.

- **Multi-agent fairness:** The fairness trade-off is mentioned in the ABR section but
  not developed. A dedicated slide on how RL-based ABR handles multi-user scenarios
  (and the emergent tragedy-of-the-commons risk) would strengthen the module.

- **Sim-to-real transfer and Mahimahi:** The open challenges slide names the sim-to-real
  gap but does not explain Mahimahi or other network emulators that are commonly used
  to train RL policies before real-network deployment. A brief demo slide would help.

- **LLM-assisted network optimization:** The 2025 search results surfaced a framework
  (MDPI 2025) integrating GPT-2-based traffic prediction with multi-agent RL for
  self-optimizing networks. This is speculative but worth watching for next year.

- **Congestion control for AI workloads:** With massive GPU cluster interconnects (e.g.,
  RDMA networks at Meta and Google), ML-specific congestion control (DCQCN, TIMELY, and
  newer RL variants) is a growing topic that connects resource allocation directly to
  AI infrastructure. A slide here would be timely.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items likely to age:

- **PLL-ABR vignette (2025):** By 2027 this paper will be 2 years old. Search for a
  newer ABR or RL-for-networking paper from 2026–2027 to replace it. Candidate search
  terms: "adaptive bitrate reinforcement learning SIGCOMM/NSDI 2026."

- **COVID traffic data (2020):** The figures (s20–s22) show data through October 2020.
  These are now a stable historical example, but if a newer large-scale traffic-shock
  event occurs (another pandemic, a major outage, a AI-driven demand surge), swap the
  case study for that event.

- **Model drift / concept drift reference:** The APNIC blog (Jan 2025) and MDPI (2025)
  papers are recent. Re-verify these links each year; APNIC and MDPI URLs tend to be
  stable, but confirm the year of the associated study.

- **Remy (SIGCOMM 2013) and Pensieve (SIGCOMM 2017):** These are now ~8–12 years old.
  They remain the canonical citations for the teaching points, but check whether a
  newer "landmark" paper has displaced them in the community's consciousness.

- **WISE (SIGCOMM 2008):** 17+ years old. The what-if methodology is still relevant,
  but consider whether a more recent what-if / counterfactual prediction system for
  networks has been published at SIGCOMM, NSDI, or IMC that could replace or supplement
  it.

## Curated images

Images used:
- `s03-i01.png` — ABR streaming diagram (High/Mid/Low quality tiers, segment timeline).
  Excellent pedagogical figure; retained on the ABR intro slide.
- `s13-i04.png` — WISE causal discovery algorithm (three-step graph pruning).
  Good exposition of causal structure learning; retained.
- `s16-i07.png` — Learned causal DAG for Google web service response time.
  Retained on "Propagating a Change" slide alongside s17-i08.
- `s17-i08.png` — India FE drain scenario (July 2007 topology diagram).
  Retained on "Propagating a Change" slide.
- `s18-i09.png` — WISE prediction CDF vs. ground truth (KS statistic 9%).
  Retained as the accuracy evidence slide.
- `s20-i13.png` — ISP peak-hour traffic volume (Mbps) 2018–2020.
  Retained for COVID case study.
- `s20-i14.png` — ISP peak-hour utilization ratio 2018–2020.
  Retained for COVID case study (paired with s20-i13).
- `s21-i15.png` — Per-peer peak download rate scatter (Jan vs. Apr 2020, ρ=0.989).
  Retained; shows proportional download growth.
- `s21-i16.png` — Per-peer peak upload rate scatter (Jan vs. Apr 2020, ρ=0.983).
  Retained; shows systematic upload shift above diagonal.
- `s22-i17.png` — ISP A normalized capacity over time with slope annotations.
  Retained for ISP capacity-augmentation slide.
- `s22-i18.png` — ISP B normalized capacity over time.
  Retained for ISP capacity-augmentation slide (paired with s22-i17).

Images dropped:
- `s07-i02.png` — Caption-only text slide ("Figure 4: Results for each of the schemes
  over a 15 Mbps dumbbell topology…"). This is a figure caption extracted without the
  actual figure; no meaningful diagram to show students. Dropped.
- `s12-i03.png` — WISE Specification Language (WSL) syntax box. Useful but already
  reproduced as a code block in the slides; redundant image dropped.
- `s14-i05.png` — Causal DAG illustration showing "Modify x2 / Effect Cascades Down
  the DAG." Good conceptual diagram, but s13-i04.png and s16-i07.png are more
  application-specific and less abstract; dropped for concision.
- `s15-i06.png` — Google web-service architecture (Client → Front End → Back End with
  variable labels). Useful background, but s16-i07.png shows the same system in the
  more informative causal DAG form; dropped to avoid redundancy.
- `s19-i10.png` and `s19-i11.png` — WSJ article header "How Covid-19 Changed Americans'
  Internet Habits" and WSJ masthead logo. Decorative / newspaper chrome; no data;
  dropped per template guidelines.
- `s19-i12.png` — Not inspected (not listed in directory); if it exists, dropped by
  default unless it contains a data diagram.

## Source

- Rebuilt from `_source-extract.md` (23 slides, mostly sparse) + direct reading of
  the book chapter `/docs/ml/text/motivation.rst` (Resource Allocation section and
  Performance Prediction / "What-If" section).
- Book chapter: **Chapter 2: Motivating Problems** — "Resource Allocation" and
  "Performance Prediction" sections, _Machine Learning for Networking_ by Feamster.
- Key papers incorporated: Pensieve (SIGCOMM '17), Remy (SIGCOMM '13),
  WISE (SIGCOMM '08), PLL-ABR (AIP Advances, July 2025).
