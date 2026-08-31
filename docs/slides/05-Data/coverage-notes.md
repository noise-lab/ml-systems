# 05-Data — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026:** Added a `.vignette` hook on the "The New Era: eBPF for Passive Monitoring" slide
  documenting verified 2025 production deployments of eBPF-based measurement: Meta's
  NetEdit framework (ACM SIGCOMM 2024), Netflix's 90% CPU reduction claim (eBPF Foundation
  2025 Year in Review), and the SIGCOMM 2025 eBPF Workshop. All sourced from the eBPF
  Foundation's published 2025 Year in Review and ACM conference proceedings.
  Sources verified: https://ebpf.foundation/the-ebpf-foundations-2025-year-in-review/,
  https://conferences.sigcomm.org/sigcomm/2025/workshop/ebpf/,
  https://dl.acm.org/doi/10.1145/3651890.3672227

- **2026:** Updated speed-testing discussion to reference the FCC MBA 13th Report (released
  August 2024, data from September–October 2022). Note: the FCC has discontinued funding for
  the in-home contracted measurement program; the 13th report is the last issued under the
  original SamKnows methodology. Future speed-testing references should pivot to alternative
  active measurement programs such as Measurement Lab NDT or RIPE Atlas.

## Suggested missing coverage on broad themes (point 3)

- **Privacy and legal constraints on passive measurement:** Chapter 3 of the book mentions
  privacy briefly, but a dedicated slide on ECPA, GDPR/CPRA constraints on traffic capture
  (especially payload) would make the privacy tension concrete. Relevant for students who
  will deploy these systems in industry.
- **Measurement infrastructure at scale:** PlanetLab / RIPE Atlas / CAIDA are mentioned in
  the book but absent from these slides. A slide on *where* measurements are taken (vantage
  points) would round out the "context" pitfall discussion.
- **Label acquisition:** Getting ground-truth labels for supervised ML on network data is
  a separate challenge not covered here — covered partially in 07-Preparation but worth
  a brief mention here to motivate the problem.
- **Telemetry streaming (gRPC / OpenConfig):** Modern network device telemetry (as opposed
  to polling-based SNMP or periodic NetFlow export) is increasingly used in production but
  absent from the deck. A brief comparison would show how data acquisition is evolving.
- **Traffic Refinery / cost-aware representation:** The source extract and book both discuss
  the joint model-cost / representation-performance optimization problem. The slides mention
  this concept but could benefit from a dedicated slide with the Traffic Refinery diagram
  (images/s37-i13.png and s37-i14.png) showing the architecture more fully.

## Next-year refresh notes

Items to re-verify each year before teaching (in priority order):

1. **eBPF vignette (highest priority):** The eBPF production-deployment numbers (Netflix 90%
   CPU reduction, Meta fleet size) come from the eBPF Foundation 2025 Year in Review and
   should be checked annually — the Foundation publishes an annual review in Q1. Replace
   specific numbers with the freshest available.
2. **FCC MBA reference:** The 13th MBA report (August 2024) may be the last in this series
   (program funding discontinued). Check annually for a replacement FCC or NTIA active
   measurement program. If no new report exists, update the speaker note to reflect the
   program's end and redirect to Measurement Lab NDT.
3. **nPrint status:** The nPrint tool (Holland et al. 2021) has an active GitHub. Check
   annually for major updates or a follow-on tool that supersedes it.
4. **IMC 2025 / 2026 papers:** IMC holds annually in October–November. The "Data Sharing
   Requirements" and "Data Collection" focus of IMC 2025 (Madison, WI) provides a fresh
   hook; update the relevant speaker notes annually with the latest conference proceedings
   that are relevant to data acquisition or passive measurement.
5. **Speed testing bottleneck figure:** The Sundaresan et al. 2016 data (image s13-i04.png)
   is dated but the finding holds. If a more recent home-network bottleneck study is
   published (e.g., from RIPE Atlas or MBA), swap the figure. The 2015–2016 timeframe of
   the original data should be acknowledged in the speaker notes if the figure is retained.

Stronger alternative vignette not used: The ACM IMC 2025 proceedings (published Nov 2025,
Madison WI) may contain a directly relevant measurement paper on passive data collection
at scale or eBPF-based capture. Check proceedings at
https://dl.acm.org/doi/proceedings/10.1145/3730567 for a more specific teachable example
to replace or supplement the eBPF Foundation vignette.

## Curated images

Images used:
- `s11-i02.png` — Speedtest.net annotated screenshot showing the four performance metrics.
  Useful because it makes abstract metrics (throughput, latency, jitter, loss) concrete.
- `s13-i04.png` — Home network bottleneck scatter plot (Sundaresan et al. 2016).
  Genuine research figure illustrating where bottlenecks occur at different speed tiers.
- `s32-i08.png` — BPF architecture diagram (kernel filter, user-space buffer, network driver).
  Architecturally important for explaining why tcpdump works the way it does.
- `s33-i09.png` — DAG/Endace hardware capture card rack diagram.
  Shows what high-speed passive capture infrastructure looks like in practice.
- `s37-i13.png` — Traffic Refinery service specification JSON snippet (Netflix example).
  Concrete illustration of DNS-based service identification in a real system.

Images dropped:
- `s05-i01.png` — MRTG/RRD router traffic graph (Hurricane Electric). Not wrong, but
  SNMP polling is the oldest and simplest monitoring approach and not worth a dedicated
  slide in a lean deck.
- `s12-i03.png` — Download speed by device boxplot (2015–2016 data). Shows device
  variability; dropped because it is duplicative with the bottleneck message in s13-i04.png
  and the data is old.
- `s14-i05.png` — "Tracking Cable's Top Internet Speeds" 2007–2018 bar chart.
  This is a marketing/industry graphic, not a research figure. Dropped as decorative.
- `s33-i10.png` — Photo of Endace DAG PCIe card. Hardware product photo; dropped in favor
  of the architectural diagram s33-i09.png.
- `s37-i14.png` — AddPacket / CollectFeatures pseudocode snippet. The netml Python code
  block in the deck is more directly actionable for students.
- `s26-i06.wmf`, `s30-i07.wmf`, `s35-i11.wmf`, `s35-i12.wmf`, `s40-i15.wmf`,
  `s40-i16.wmf` — WMF format files; not renderable in Quarto without conversion and of
  unclear content (Traffic Refinery architecture, packet sampling diagrams). Described in
  speaker notes instead. If needed, convert to PNG for future use.

## Source

Rebuilt from `_source-extract.md` (40 slides, heavily redundant) + book Chapter 3
("Network Data" / `measurement.rst`) of *Machine Learning for Networking* by Feamster.
Deck order and terminology follow the book; the book is authoritative where slides and
book conflict. The source extract had three slides (36–39) on Traffic Refinery internals
that were collapsed into a single example slide (DNS-based service ID) because the full
Traffic Refinery architecture diagram is in .wmf format and not renderable. Flag for
future rebuild if PNG versions of those diagrams become available.

Book chapter section pointers:
- Slides 1–3 ("Why We Measure", "Two Approaches") → Chapter 3 introduction
- Slides 4–7 (Active Measurement) → "Active Measurement" section
- Slides 8–13 (Passive Measurement, flows, IPFIX) → "Passive Measurement" section
- Slides 14–16 (eBPF, features, DNS) → "Passive Measurement" / "Data Preparation" sections
- Slide 17 (netml) → "Packets to Data Structures" section
- Slide 18 (nPrint) → "Learning Traffic Representations with nPrint" section
- Slide 19 (sound measurement) → Chapter 3 intro + Paxson (2004) reference
