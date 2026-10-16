# Figure recovery: 03-Performance

Source: `docs/slides/03-performance.pptx`, 51 slides (slide 36 hidden), rendered 2026-10-16
with headless LibreOffice at 110 dpi (triage) and 220 dpi (crops). No pen-ink annotations.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–3 | skip | | | title, bullets, section title |
| 4 | media | "The Dominant Traffic Is Video" | | `s04-i01.png`, already embedded |
| 5 | skip | | | one-line framing |
| 6 | media | "More Speed Does Not Equal Better Experience" | | `s06-i02.png`, already embedded |
| 7 | media | "The Finding Made Headlines" | | WSJ clippings; `s29-i66.png` covers it |
| 8 | media | | | "Tracking cable's top speeds" bar chart, dated, not embedded |
| 9 | skip | | | Network Microscope hardware photos; device not discussed in the Quarto deck |
| 10 | skip | | | paper first page plus two bullets |
| 11 | skip | | | four QoE metric icons; the Quarto table covers it |
| 12 | keep | new, "Where the Passive Monitor Sits" | vantage-point-architecture.png | base state: Group 1 (home/ISP/IXP/servers/monitor); animation exits it and overlays the Netflix scatter (`s12-i17.png`, already embedded), so shapes 261–274 stripped |
| 13 | keep | new, "The Pipeline in One Picture" | inference-pipeline-stages.png | five-stage flowchart; dotted "Elbow Connector 15" stripped because it rendered across the labels |
| 14 | skip | | | bullets (design questions), covered by "Step 3" |
| 15 | skip | | | same vantage diagram as 12 plus one callout |
| 16 | skip | | | passive-DNS mapping drawn over a third-party DNS resolution picture (ccsf.edu example) with no attribution |
| 17 | keep | new, "What the Monitor Can Read in Each Packet" | feature-layers-at-monitor.png | final build state (header stack plus packets) |
| 18 | skip | | | full feature table from the paper; "Step 2" lists the features in text |
| 19 | media | "Segment Boundaries Are Visible…" | | `s19-i45.png` already embedded with its callouts; only a small "Get(Segment X)" cartoon is drawn |
| 20 | skip | | | bullets, covered by the QoE metrics table |
| 21 | keep | new, "Acquiring Labeled Data: The Lab Rig" | labeled-data-collection-rig.png | no animation; cropped below the bullets |
| 22–23 | media | "Surprising Result…" | | `s22-i56.png`, `s23-i58.png` already embedded |
| 24–25 | media | | | `s24-i60`, `s25-i61` deliberately dropped in the Sept 2026 audit |
| 26 | skip | | | nPrint diffusion pipeline; belongs to 14-nPrint |
| 27 | media | "Challenge: Fast Model Serving" | | `s27-i64.png` already embedded |
| 28 | skip | | | WSJ interactive screenshot, dated, not ours |
| 29 | media | "The Finding Made Headlines" | | `s29-i66.png` already embedded |
| 30 | skip | | | padlock overlay covers the "bitrate changes / transcoding" box; slide 12 figure plus "The Encryption Wall" bullets carry the point |
| 31–33 | media | | | duplicates of `s12-i17` and the speed-tier plots |
| 34, 36, 37 | skip | | | bullets (36 hidden) |
| 35 | skip | | | US/France maps with pins; decorative |
| 38–41 | media | generality, feature importance, layer comparison | | all four already embedded |
| 42 | keep | new, "Time-Bin Alignment: Lab Features vs. Field Features" | time-bins-aligned.png, time-bins-offset.png | 53-step build; captured two states: F bins (shapes 178–190, 213) and F' bins (191–204, 214); the F'' / F''' overlays and the callout were dropped and are narrated |
| 43–45, 49–51 | skip | | | bullets, section titles, lab handout |
| 46–48 | media | VCA slides | | `s46-i96`, `s47-i97`, `s48-i98` already embedded |

## Captured images

- `images/vantage-point-architecture.png` (2100×900) — source: `docs/slides/03-performance.pptx` slide 12, shapes 261–274 stripped, crop 470,540,2570,1440 at 220 dpi
- `images/inference-pipeline-stages.png` (2650×790) — source: slide 13, "Elbow Connector 15" stripped, crop 150,480,2800,1270
- `images/feature-layers-at-monitor.png` (2440×1240) — source: slide 17, final state, crop 360,320,2800,1560
- `images/labeled-data-collection-rig.png` (2430×760) — source: slide 21, crop 140,880,2570,1640
- `images/time-bins-aligned.png` (1550×818) — source: slide 42, shapes 191–212, 214, 215 stripped, crop 230,552,1780,1370
- `images/time-bins-offset.png` (1550×818) — source: slide 42, shapes 178–190, 205–213, 215 stripped, same crop

The deck has no `speaker-notes.md`; none was created.
