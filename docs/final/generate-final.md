---
description: Generate a final exam (nine-week offering) from the term's agenda and past finals
---

Instantiation of [`generate-exam.md`](../prompts/generate-exam.md) for the standard final.
Read that file for the LaTeX conventions and workflow; use these parameters.

| Parameter | Value |
|---|---|
| Exam type | `final` |
| Agenda | `docs/agenda/<term>.md`, Meetings 9 onward (supervised models, deep learning and nPrint, unsupervised learning, generative models), with light reference back to the pipeline material |
| Past exams | `docs/final/<YYYY>/` — the two or three most recent. From 2026 on, each also carries `rubric.yaml` (how it was scored), `grading-notes.md` (decisions made while giving it), and `analysis.md` (where the class split from the key): read those three first, and do not reuse a question they mark as ambiguous |
| Length | 4 pages, 50 points, designed for 30–40 minutes; ends with a 2-point feedback section |
| Must include | questions on the hands-ons the agenda records in the meeting range (typically regression, trees and ensembles, deep learning, PCA, clustering, generative models) and on the project deliverables; every final flag in the range |
| Instructor output | `/Users/feamster/Dropbox/Tmp/exams/ml-systems/final/<YYYY>/` (not committed until after the exam; when the instructor publishes it to `docs/final/<YYYY>/`, `rubric.yaml`, `grading-notes.md`, and `analysis.md` from the grading directory in instructor-data go with it, never the scores or review pages) |

For the three-week offering, which has a single combined exam, use
[`generate-combined.md`](../combined/generate-combined.md) instead.
