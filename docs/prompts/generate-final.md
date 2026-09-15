---
description: Generate a final exam (nine-week offering) from the term's agenda and past finals
---

Instantiation of [`generate-exam.md`](generate-exam.md) for the standard final.
Read that file for the LaTeX conventions and workflow; use these parameters.

| Parameter | Value |
|---|---|
| Exam type | `final` |
| Agenda | `docs/agenda/<term>.md`, Meetings 9 onward (supervised models, deep learning and nPrint, unsupervised learning, generative models), with light reference back to the pipeline material |
| Past exams | `docs/final/<YYYY>/` — the two or three most recent |
| Length | 4 pages, 50 points, designed for 30–40 minutes; ends with a 2-point feedback section |
| Must include | questions on the later hands-ons (linear/logistic regression, trees and ensembles, deep learning, PCA, clustering, diffusion) and on the project deliverables |
| Instructor output | `/Users/feamster/Dropbox/Tmp/exams/ml-systems/final/<YYYY>/` (not committed until after the exam) |

For a short term with a single combined exam, use the term-specific file instead
(e.g. [`generate-summer-final.md`](generate-summer-final.md)).
