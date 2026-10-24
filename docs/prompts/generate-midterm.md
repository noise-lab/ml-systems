---
description: Generate a midterm exam (standard Autumn offering) from the term's agenda and past midterms
---

Instantiation of [`generate-exam.md`](generate-exam.md) for the standard midterm.
Read that file for the LaTeX conventions and workflow; use these parameters.

| Parameter | Value |
|---|---|
| Exam type | `midterm` |
| Agenda | `docs/agenda/<term>.md`, Meetings 1–8 (through Model Training and Evaluation), or whatever range the agenda says the midterm covers |
| Past exams | `docs/midterm/<YYYY>/` — the two or three most recent |
| Length | 4 pages, 50 points, designed for 30–40 minutes; ends with a 3-point feedback section |
| Must include | one question on Assignment 1; questions on the hands-ons the agenda records in the meeting range; every midterm flag in the range |
| Instructor output | `/Users/feamster/Dropbox/Tmp/exams/ml-systems/midterm/<YYYY>/` (not committed until after the exam) |

Follow the agenda for what to cover. As orientation only, midterms usually draw
on: motivating applications (QoE, security, resource allocation); passive vs.
active measurement and flow records; feature engineering and representation; data
quality; the pipeline (splits, cross-validation, leakage); evaluation metrics; and
bias–variance. Include a topic only if that term's agenda covers it in the range
(drift, for example, is often taught later in the term) and never one the agenda
says will not be asked.
