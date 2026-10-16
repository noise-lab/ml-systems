# Figure recovery: 01-Intro

Source: `docs/slides/01-intro.pptx`, 19 slides, rendered 2026-10-16 (headless LibreOffice, 220 dpi). No pen-ink annotations on any slide.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2–3 | skip | | | bullets (welcome, why this course) |
| 4 | media | "The ML Control Loop", "The Pipeline — 80% Is Not Modeling", "The Data-Preparation Reality" | s04-i01.png, s04-i02.png | already extracted and embedded by the Sep 2026 audit; red highlight box around "Modeling" is a drawn shape but the point is made in the slide text |
| 5 | skip | | | bullets (learning objectives) |
| 6 | keep | new, "How Networks Run: Measure, Model, Control", before "The ML Control Loop" | measure-model-control.png | no build; title placeholder stripped before crop |
| 7 | keep | new, "The Control Loop, Drawn", after "The ML Control Loop" | ml-control-loop.png | no build; the Quarto slide of the same name shows the pipeline picture, not this loop |
| 8 | skip | | | bullets (inference examples) |
| 9 | keep | new, "Closing the Cycle: What Programmability Enables", after "The Gap" | network-management-cycle.png | 11-step additive build, final state captured; an emphasis-only last step |
| 10–12 | skip | | | bullets (state of NetML, performance, security ideals); carried by "The Gap" slide |
| 13 | skip | | | section divider |
| 14–19 | skip | | | logistics bullets (readings, communication, assignments, project, late policy, looking ahead) |

## Captured

- `images/measure-model-control.png` (1400×971) — source: `docs/slides/01-intro.pptx` slide 6, shape "Rectangle 2" (title) stripped, auto-cropped, downscaled from 2122×1472.
- `images/ml-control-loop.png` (1400×963) — source: `docs/slides/01-intro.pptx` slide 7, box crop of the final render, downscaled from 1949×1340.
- `images/network-management-cycle.png` (1400×783) — source: `docs/slides/01-intro.pptx` slide 9, final animation state, box crop, downscaled from 2000×1118.

Three new slides added to `slides.qmd`; no existing slide text changed. This deck has no `speaker-notes.md`, so no notes rows were added.
