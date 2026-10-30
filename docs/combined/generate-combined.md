---
description: Generate the single combined exam for the three-week offering (any year) from that term's agenda, term page, and past exams
---

Instantiation of [`generate-exam.md`](../prompts/generate-exam.md) for the **three-week
offering**, which has one combined exam and no midterm. It has typically run in
September (Paris), but the recipe depends on the offering's length, not the
season. Read `generate-exam.md` first for the LaTeX conventions and workflow, then
apply the parameters below. The nine-week offering uses
[`generate-midterm.md`](../midterm/generate-midterm.md) and [`generate-final.md`](../final/generate-final.md)
instead.

Students: run this yourself to make practice exams. You will not get the same
questions the instructor got, but you will get questions built from the same inputs.

## Parameters

| Parameter | Value |
|---|---|
| Exam type | `combined` (single exam; no midterm) |
| Agenda | `docs/agenda/<term>.md` for the term (e.g. `2026-summer.md`), every meeting up to the exam. Meeting numbers and content vary by year |
| Term page | `docs/terms/<term>.md`: exam date, lectures skipped or merged, and any bonus lectures given after the exam was set. Everything it lists as skipped or not on the exam is **out of scope** |
| Assignments | Every assignment and project milestone due before the exam (in 2026: Assignment 1 and the project proposal); at least one question on each |
| Hands-ons | The ones the agenda records for the term, not a fixed list |
| Past exams to imitate | The most recent combined exam(s) (`docs/combined/<YYYY>/`), plus the most recent nine-week midterm and final: a combined exam mixes both |
| Length | 6 pages (7 if the feedback item spills), **75 points**, designed for 45–50 minutes with the full class period available (the nine-week exams are 4 pages / 50 points) |
| Question mix | Mostly select-all / multiple choice and yes/no with "Why or why not?"; keep pure short answers to a handful (graded by keyword); where a question has two parts, make one part multiple choice and one a box; end with a 2-point feedback section |
| Instructor output | `/Users/feamster/Dropbox/Tmp/exams/ml-systems/combined/<YYYY>/` (not committed until after the exam) |

## Coverage

Proportional to time spent in that term's agenda, and nothing outside it. A
three-week term moves through the whole course, so expect roughly: why ML for
networks and the pipeline; the security, performance, and resource-allocation
applications; data acquisition and representation; training and evaluation;
supervised models; deep learning; unsupervised learning; and generative models.
Treat that list as orientation only. Drop whatever the term page says was
skipped, and anything the agenda says will not be asked.

## Flagged questions

Cover every question the agenda flags as a good exam question, including any
the instructor adds in an exam-review meeting near the end of the term. Collect
them before writing anything.

## Standing ground rules

Set in the 2026 review and kept unless that term's agenda says otherwise:

- No definition regurgitation (e.g. Mitchell's definition), no logistics or
  history, no attack trivia (e.g. "what is drop-catching").
- No computing PCA or t-SNE by hand; no asking for the difference between ridge
  and lasso.
- Throughput vs. latency may appear on every exam: a small trace or
  request/response timeline, compute both.

## Output

- Instructor: write to the Dropbox directory above. After the exam, publish it with
  solutions in `docs/combined/<YYYY>/` and add the term's prompt record (see
  [`generate-summer-final.md`](2026/generate-summer-final.md) for 2026) if the
  parameters were changed for that year.
- Students: any directory of your own, e.g. `~/ml-systems-practice/combined/`.
  Copy `feamster.sty` and the `Makefile` from the most recent past combined exam,
  then `make all`.

## Validation

As in `generate-exam.md`: `make all`, check that `pdfinfo exam.pdf | grep Pages`
gives 6 (or 7 if the feedback item spills) and that the `\prob{}` points sum to 75;
open the PDF and look for overflowing option text or answer boxes broken across pages.
