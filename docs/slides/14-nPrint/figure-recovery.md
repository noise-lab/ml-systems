# Figure recovery: 14-nPrint

Source: `docs/slides/14-nPrint.pptx`, 43 slides, rendered 2026-10-16 (headless
LibreOffice via `instructor-tools/src/pptx-figures.py`). No pen-ink annotations
on any slide. The September 2026 rebuild already embedded the extracted media
`s08-i34`, `s09-i35`, `s10-i36`, `s13-i39`; those are not duplicated here.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1–2 | skip | | | title, "Automation" divider |
| 3 | keep | new "Two Problems, Two Pipelines", after "The Classic ML Pipeline for Traffic" | bespoke-pipelines.png | two drawn pipelines with the Engineer Features ⇄ Train Models loop; no animation |
| 4–5 | skip | | | same pipeline as slide 3 with an underline moved; superseded |
| 6 | keep | new "Collapsing the Pipeline", after "The nPrint Insight" | automated-pipeline.png | four-stage pipeline with Automated Traffic Analysis; the deck's thesis in one row |
| 7 | keep | new "Diverging Branches", after "Feature Engineering Is Expensive" | diverging-branches.png | dataset → metadata attachment → feature extraction → models fan-out; eight `bentConnector3` connectors (rot 90°, adjust values >100%) render as vertical bars in LibreOffice, so they were rewritten as `straightConnector1` between the same endpoints in a scratch copy before rendering |
| 8–9 | media | already embedded (`s08-i34`, `s09-i35`) | | KDD'98 and CICIDS2017 divergence tables |
| 10 | media | already embedded (`s10-i36`) | | TCP/UDP header formats |
| 11–12 | skip | | | slide 10's image with a highlight box on Flags / on the bit-number column; the existing bullets state the same three obstacles |
| 13 | media | already embedded (`s13-i39`) | | nPrint bitmap |
| 14–16 | skip | | | nPrint bitmap with a highlight box (Complete / Constant Size / Normalized); near-duplicates of `s13-i39` |
| 17 | keep | new "Alignment, Illustrated", after "nPrint Properties: Complete and Aligned" | nprint-aligned.png | bitmap with the box spanning IPv4+TCP columns of both rows; the one overlay that adds intuition |
| 18 | skip | | | blank |
| 19 | skip | | | Pensieve/Genet screenshots, out of scope (per coverage-notes) |
| 20 | skip | | | Data → Production FAIL cartoon with a text list; the Quarto "What Can Go Wrong" slide carries it in words |
| 21–25, 27–29, 31–32 | skip | | | netUnicorn motivation text, bullets, and build-state variants |
| 26 | keep | new "The TTL Artifact, Seen in the Tree", after "What Can Go Wrong" | cicids-ttl-tree.png | decision tree (media `s26-i55`) plus the two red highlight boxes drawn on the slide; cropped from the render so the boxes survive. Figure attributed to Beltiukov et al., CCS 2023 in the caption |
| 30 | media | new "The Heartbleed Artifact, Seen in the Histogram", after the tree slide | `s30-i56.png` (existing) | backward-IAT histogram; embedded as extracted, attributed in the caption |
| 33 | media | skip | | netUnicorn architecture (`s33-i57`), deliberately dropped by the 2026 rebuild |
| 34–43 | skip | | | netUnicorn workflow, modularity icon builds (35–39), deployment diagram (41), platform list, dos and don'ts; the Quarto deck treats netUnicorn in one slide by design |

Captured (220 dpi):

- `images/bespoke-pipelines.png` (2640×1060) — source: 14-nPrint.pptx slide 3, box 140,400,2780,1460
- `images/automated-pipeline.png` (2060×460) — source: 14-nPrint.pptx slide 6, box 400,630,2460,1090
- `images/diverging-branches.png` (1910×770) — source: 14-nPrint.pptx slide 7, shapes 1148–1151 and 1163–1166 rewritten as straight connectors, box 170,450,2080,1220
- `images/nprint-aligned.png` (2250×710) — source: 14-nPrint.pptx slide 17, box 340,480,2590,1190
- `images/cicids-ttl-tree.png` (1560×815) — source: 14-nPrint.pptx slide 26, box 1120,730,2680,1545
- `images/s30-i56.png` (1759×706, pre-existing extracted media) — source: 14-nPrint.pptx slide 30

This deck has no `speaker-notes.md`; none was created.
