# 02-Security — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026:** Replaced the old (undated, general) threat-landscape framing with the
  **CrowdStrike 2026 Global Threat Report** (released February 24, 2026) as the primary
  vignette. Key verified statistics used: AI-enabled adversaries up 89% year-over-year;
  average eCrime breakout time 29 minutes; fastest observed intrusion 27 seconds lateral
  movement. Source: crowdstrike.com/en-us/global-threat-report and the press release at
  ir.crowdstrike.com. These statistics motivate the real-time constraint on ML-based
  intrusion detection.
- **2026:** Dropped the 2012 Twitter propaganda detection slides (speaker notes in the
  source extract explicitly flag "Note: This study likely no longer works"). Replaced
  with the disinformation infrastructure detection case, which is more architecturally
  illustrative and does not depend on a specific platform's API.

## Suggested missing coverage on broad themes

- **Encrypted traffic classification:** The slides mention that detecting malicious
  content in encrypted traffic is a modern challenge, but do not cover the technical
  approaches (flow statistics, timing analysis, TLS fingerprinting). This is a natural
  extension of the behavioral-features slides and is directly relevant to the
  feature–problem space gap.
- **LLM-generated phishing and malware:** The KnowBe4 2025 Phishing Threat Trends
  Report found 82.6% of analyzed phishing emails between September 2024 and February
  2025 contained AI-generated content. This warrants a dedicated slide on how ML
  detectors must evolve when attacker-side content generation is also ML-powered.
- **Graph-based anomaly detection:** The source extract mentions "analysis of
  communication graphs" as a core challenge but the deck does not cover graph neural
  networks (GNNs) for intrusion detection. This connects to the GNN material elsewhere
  in the course.
- **Differential privacy for network ML:** The federated learning slide mentions
  differential privacy as a mechanism but does not explain the privacy-accuracy tradeoff
  concretely. A worked example (e.g., adding Gaussian noise to gradients) would
  strengthen this.
- **Evaluation benchmarks:** The deck does not mention standard IDS evaluation datasets
  (CICIDS, UNSW-NB15, KDD Cup). A slide on dataset limitations (label quality, class
  imbalance, train/test leakage) would be valuable, especially given the book's
  emphasis on data preparation.

## Divergences from the source extract

- The source extract (38 slides) is organized around two case studies: spam/DNS
  detection (Slides 7–25) and disinformation infrastructure (Slides 26–38). The rebuilt
  deck consolidates these into Part 1, adds Part 2 on adversarial ML (not in the
  original slides at all), and drops the 2012 Twitter propaganda detection slides
  (acknowledged as stale in the original speaker notes).
- The book chapter (security.rst) focuses on threats *to* ML systems; the original
  slides focus on ML *for* security. The rebuilt deck covers both, because the book
  represents the current canonical framing. Instructors who want to weight more heavily
  toward intrusion detection applications can expand Part 1; those who want to weight
  toward adversarial ML can expand Part 2.
- The book uses "Security and Privacy of ML in Networking" as the chapter framing. The
  deck title is "ML for Network Security" to be consistent with the course's
  application-oriented framing, but the content now includes the book's adversarial ML
  material.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items that will age:

- **CrowdStrike 2026 Global Threat Report vignette:** CrowdStrike releases a new Global
  Threat Report each February. By early 2027, the 2027 report will be available with
  updated breakout-time statistics. Replace the 29-minute figure and 89% figure with the
  next year's equivalents. Verify the specific numbers from the primary source
  (crowdstrike.com/en-us/global-threat-report) before updating.
- **AI-generated phishing statistics:** The KnowBe4 2025 Phishing Threat Trends Report
  figure (82.6% AI-generated phishing emails) is cited in speaker notes but not on
  slides. If added to slides in a future edit, verify against the updated annual report.
- **LLM/AI-powered malware examples:** The Google/Mandiant AI Risk and Resilience
  special report (2025) and the SentinelOne MalTerminal research (November 2025) are
  cited in background research but not in the deck. These may be worth adding as a
  vignette on LLM-powered adversaries if the "LLM-generated phishing" missing coverage
  suggestion is acted on.

## Curated images

**Used:**

- `s18-i14.png` — CDF of days from domain registration to first spam campaign use;
  clearly illustrates the detection window. Used on the "Predictive Analytics and DNS"
  slide.
- `s19-i15.png` — DNS query volume vs. domain age; malicious spike vs. flat legitimate
  baseline. Used on the same slide.
- `s29-i25.png` — Disinformation infrastructure timeline (Domain Registration →
  Certificate Issuance → Website Deployment → Content Publication → Distribution).
  Clean diagram, used on the "Disinformation Infrastructure Detection" slide.

**Dropped:**

- `s02-i01.png` — ML pipeline diagram (Data Ingestion → … → Deployment). Generic; better
  covered in the pipeline/preparation lectures, not specific to security.
- `s02-i02.png` — Data preparation text block. Redundant with pipeline lecture material.
- `s08-i03.png` — Spam/phishing trend chart (2010–2020). Data is 5+ years stale; trend
  is described textually instead.
- `s09-i04.png` — Screenshot of a spam email in a PDF reader. Low pedagogical value;
  decorative example.
- `s09-i05.png`, `s09-i06.png` — Additional spam message screenshots. Dropped.
- `s11-i07.png` — Clip-art envelope icons. Decorative; dropped.
- `s11-i09.png` — Screenshot of a pharmacy spam message. Old (2005); dropped.
- `s14-i10.wmf` — WMF vector file; cannot be rendered in Quarto/HTML without conversion.
- `s15-i11.wmf` — WMF vector file; same issue.
- `s15-i12.png` — MIT Technology Review article screenshot about SNARE (2009).
  Interesting provenance but not a teaching diagram; referenced in speaker notes only.
- `s15-i13.png` — Additional SNARE result figure; covered textually.
- `s20-i16.png` — DNS lookup clustering figure. Small and low-res; the key idea is
  explained textually.
- `s22-i17.png` — Browser address bar showing "cheaprx.com". Decorative example; dropped.
- `s22-i18.jpg` — Additional domain screenshot. Dropped.
- `s22-i19.png` — Red X stop-sign icon. Clip-art; dropped.
- `s22-i20.png` — Pill capsule clip-art. Decorative; dropped.
- `s26-i21.png` — Tweet volume time series (2010). Data 15+ years stale; removed with
  the Twitter propaganda section.
- `s26-i22.png`, `s27-i23.png`, `s28-i24.png` — Additional Twitter analysis figures.
  Dropped with that section.
- `s29-i26.png` — Alternate disinformation timeline diagram; superseded by s29-i25.png.
- `s32-i27.png` — Domain lifespan CDF (Disinfo vs. News vs. Other). Useful diagram; not
  included in the rebuilt deck to keep slide count lean, but worth re-adding if the
  disinformation section is expanded.
- `s33-i28.png` — Feature table (domain features for disinformation detection). Dense
  table; summarized textually on the Domain Features slide.
- `s34-i29.png` — CDF of SAN count by site category. Useful; summarized textually on the
  Disinformation Infrastructure slide.
- `s35-i30.png`, `s37-i31.png` — Certificate and hosting feature tables. Summarized
  textually; dropped to keep slides lean.

## Source

- Rebuilt from `_source-extract.md` (38 slides) + book chapter
  `text/security.rst` ("Security and Privacy of ML in Networking", Chapter 9).
- Where the original slides and the book diverge, the book's framing was adopted:
  the original slides contain no adversarial ML material; the book chapter organizes
  the topic around threat models, evasion, poisoning, and privacy attacks. Both
  perspectives are included in the rebuilt deck.
