---
description: Generate a midterm exam (nine-week offering) from the term's agenda and past midterms
---

Instantiation of [`generate-exam.md`](generate-exam.md) for the standard midterm.
Read that file for the LaTeX conventions and workflow; use these parameters.

| Parameter | Value |
|---|---|
| Exam type | `midterm` |
| Agenda | `docs/agenda/<term>.md`, Meetings 1–8 (through Model Training and Evaluation) |
| Past exams | `docs/midterm/<YYYY>/` — the two or three most recent |
| Length | 4 pages, 50 points, designed for 30–40 minutes; ends with a 3-point feedback section |
| Must include | one question on Assignment 1; questions on the hands-ons covered so far (packet capture, scanning, QoE, netml features, pipeline) |
| Instructor output | `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/<YYYY>/` (not committed until after the exam) |

Topics that recur on midterms: motivating applications (QoE, security, resource
allocation); passive vs. active measurement and flow records; feature engineering
and representation; data quality (missing, non-representative, irrelevant
features, outliers, the TTL example); the pipeline (split, cross-validation,
leakage); evaluation metrics; bias–variance; drift.
