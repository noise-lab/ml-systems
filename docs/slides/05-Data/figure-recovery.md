# Figure recovery: 05-Data

Source: `docs/slides/05-Data.pptx`, 40 slides (slide 24 hidden), rendered 2026-10-16
with `instructor-tools/src/pptx-figures.py` (headless LibreOffice, 220 dpi). No pen-ink
annotations in this deck. The deck has no `speaker-notes.md`, so no notes rows were added;
each new slide carries its own `::: {.notes}` block.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–4, 6–10 | skip | | | title, section dividers, bullet slides already covered by the Quarto text |
| 5 | media | | | MRTG router graphs (`s05-i01.png`), dropped by the Sept 2026 audit |
| 11 | media | "The Many Faces of Speed" | | `s11-i02.png`, already embedded |
| 12 | media | | | device boxplot (`s12-i03.png`), dropped by the audit |
| 13 | media | "When Speed Testing Gets Hard" | | `s13-i04.png`, already embedded |
| 14 | media | | | cable marketing bar chart (`s14-i05.png`), dropped by the audit |
| 15 | skip | | | hands-on divider |
| 16 | keep | new, "How Traceroute Works" (before "Active Measurement Pitfall: Traceroute") | traceroute-ttl-probes.png | no builds; TTL 1/2/3 arcs plus ICMP return |
| 17 | skip | | | same diagram as 16 under the pitfall bullets; bullets already in Quarto |
| 18–21, 23, 24 | skip | | | divider, bullets, blank (23), hidden tasks list (24) |
| 22 | keep | new, "Traffic Refinery: Architecture" (before "DNS-Based Service Identification") | traffic-refinery-architecture.png | the diagram coverage-notes said was unrenderable (WMF); right half of slide |
| 25 | skip | | | We Are Legion film still, dated |
| 26 | keep | new, "Exporting Flow Records" (before "IPFIX Flow Records") | ipfix-export-collector.png | cloud, exporting router, collector; no builds |
| 27 | skip | | | bullets, covered by "IPFIX Flow Records" |
| 28 | keep | new, "Aggregating Packets into Flows" (after "What Is a Flow?") | packets-into-flows.png | wide banner above bullets; no builds |
| 29 | keep | new, "Packet Sampling Can Split a Flow" (after "Packet Sampling: Sampled NetFlow") | sampled-netflow-split-flow.png | builds are on the bullet placeholder only; diagram static |
| 30 | media | | | binomial-sampling plot, `s30-i07.wmf`; Quarto states the result in words |
| 31 | keep | new, "Flow-Level Sampling: Weight the Heavy Hitters" | stratified-flow-sampling.png | typeset flow list with sampling-probability arrows; no builds |
| 32 | media | "Packet Capture: tcpdump and BPF" | | `s32-i08.png`, already embedded |
| 33 | media | "Packet Capture on High-Speed Links" | | `s33-i09.png` embedded; DAG photo `s33-i10.png` dropped by the audit |
| 34 | skip | | | bullets, covered |
| 35 | media | | | two Traffic Refinery paper plots (`s35-i11/12.wmf`); left out, see below |
| 36 | keep | new, "Traffic Refinery: Searching the Cost–Performance Space" (after "The Accuracy–Cost Tradeoff") | traffic-refinery-cost-performance.png | stripped "Content Placeholder 2" (bullets that overlapped the crop); Venn group is build step 1 and is kept |
| 37–39 | skip | | | same architecture as 22 with one stage highlighted; the JSON on 37 is already embedded (`s37-i13.png`) |
| 40 | media | | | time-series plots (`s40-i15/16.wmf`), evaluation detail, left out |

Left out deliberately: slide 35's precision/recall and execution-time plots and slide
40's drop-rate plots are paper figures (Bronzino et al. 2022) whose findings the Quarto
deck already states in the cost table; embed later if a dedicated evaluation slide is
wanted. Slide 17's diagram duplicates slide 16.

## Sources

- `images/traceroute-ttl-probes.png` — source: `docs/slides/05-Data.pptx` slide 16, box 530,410,1610,850 at 220 dpi (1080×440)
- `images/traffic-refinery-architecture.png` — source: slide 22, box 1200,360,2200,1085 (1000×725); figure from Bronzino et al. (2022), Traffic Refinery
- `images/ipfix-export-collector.png` — source: slide 26, box 320,720,1840,1238 (1520×518)
- `images/packets-into-flows.png` — source: slide 28, box 320,310,1870,575 (1550×265)
- `images/sampled-netflow-split-flow.png` — source: slide 29, box 440,893,1660,1238 (1220×345)
- `images/stratified-flow-sampling.png` — source: slide 31, box 370,730,1810,1238 (1440×508)
- `images/traffic-refinery-cost-performance.png` — source: slide 36, "Content Placeholder 2" stripped, box 140,370,1630,1185 (1490×815)
