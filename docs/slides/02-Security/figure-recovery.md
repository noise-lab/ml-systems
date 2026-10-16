# Figure recovery: 02-Security

Source: `docs/slides/02-security.pptx`, 38 slides, rendered 2026-10-16 (headless LibreOffice, no pen ink found).

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | media | | | ML pipeline picture (`s02-i01.png`, `s02-i02.png`), dropped in the September audit as generic |
| 3, 38 | skip | | | "Security and Privacy Ideals" bullets (same slide twice) |
| 4 | keep | new, "The Loop in One Picture" after "The Measure–Model–Control Loop" | measure-model-control-loop.png | title placeholder "Rectangle 2" stripped; no builds |
| 5 | skip | | | simpler variant of slide 4 (ML / Measurement / Control with clouds); superseded by 4 |
| 6 | skip | | | third variant of the loop (Network Management "Cycle", SDN annotations); not security-specific |
| 7 | skip | | | bullets |
| 8 | media | | | spam trend chart `s08-i03.png`, dropped as stale |
| 9 | media | | | content-filter screenshots `s09-i0[4-6].png`, dropped |
| 10 | skip | | | mail-header text box and bullets |
| 11 | keep | new, "A Lot Happens Before Spam Is Sent" before "Going Back Earlier in Time" | spam-lifecycle-before-send.png | two build steps (Group 10, Group 8) are additive, so the final state is the full figure |
| 12, 13 | skip | | | bullets (agility / exposure / coordination), already on "Behavioral Features for Spam Detection" |
| 14 | media | | | kNN IP-distance plot, already embedded as `snare-knn-ip-distance.png` |
| 15 | media | | | RuleFit confusion matrix, already embedded as `snare-rulefit-confusion.png` |
| 16, 17 | skip | | | bullets, covered by "Going Back Earlier in Time" and "Predictive Analytics and DNS" |
| 18, 19 | media | | | `s18-i14.png`, `s19-i15.png`, already embedded |
| 20 | media | | | early-lookup table `s20-i16.png`, dropped as low-res |
| 21 | skip | | | one-line transition |
| 22 | skip | | | PREDATOR life cycle: three stock icons and a line; the "Going Back Earlier in Time" table already carries registration → use → detection |
| 23 | keep | new, "Registration-Time Features: Two Examples" after "Domain Features at Registration Time" | edit-distance-operations.png | lower half of the slide; no builds |
| 24 | skip | | | bullets |
| 25 | keep | same new slide as 23 | registration-batch-cohesion.png | table plus two labelled arrows; no builds |
| 26–28 | media | | | 2012 Twitter propaganda figures, deliberately dropped in the September audit |
| 29 | media | | | disinformation timeline, already embedded as `s29-i25.png` |
| 30, 31 | skip | | | bullets |
| 32, 34 | media | | | lifespan and SAN CDFs, already embedded as `s32-i27.png`, `s34-i29.png` |
| 33, 35, 37 | media | | | feature tables, dropped as dense; summarised in text |
| 36 | skip | | | bullets |

## Captured

- `images/measure-model-control-loop.png` (2140×1480) — source: `02-security.pptx` slide 4, 220 dpi, shape "Rectangle 2" (title) stripped, box 390,80,2530,1560.
- `images/spam-lifecycle-before-send.png` (2190×1165) — source: `02-security.pptx` slide 11, 220 dpi, final build state, box 395,335,2585,1500.
- `images/edit-distance-operations.png` (1495×505) — source: `02-security.pptx` slide 23, 220 dpi, box 720,995,2215,1500.
- `images/registration-batch-cohesion.png` (1590×815) — source: `02-security.pptx` slide 25, 220 dpi, box 580,835,2170,1650. LibreOffice renders the table's spacer row as a blank line under the "Uncohesive" arrow, as in the original.

No `speaker-notes.md` exists for this deck, so no notes rows were added.
