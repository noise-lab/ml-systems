# ML Systems Midterm, Autumn 2026

In-class midterm, Friday October 30, 2026. Covers Meetings 1-8 of
`docs/agenda/2026-autumn.md` (through Model Training and Evaluation).
Linear regression is excluded (Meeting 8 flag).

Build: `make all` (exam.pdf + exam-solution.pdf), `make clean`.
4 pages, 50 points. `feamster.sty` and the Makefile are copied from `docs/midterm/2025/`.

| Section | Questions | Points |
|---|---|---|
| Motivation and Applications | 1-4 (encryption/QoE, HO1 sanity checks, phishing features, HO2 time of day) | 11 |
| Data Acquisition | 5-7 (passive vs active, speed test after the fact, flow definition + HO5) | 9 |
| Features and Representation | 8-10 (netml vs nPrint, one-hot, imputation) | 9 |
| Training and Evaluation | 11-13 (temporal leakage, confusion matrix, bias-variance figure) | 11 |
| Assignment 1 | 14-15 (segment rate, held-out household) | 7 |
| Feedback | 16 | 3 |

Agenda "midterm flags" covered: attack -> features and data (M3), active vs
passive and trade-offs (M5), flow definition (M5), manual features vs
representation learning (M6), imputation appropriate vs not (M7),
bias-variance figure (M8), confusion-matrix arithmetic (M8).

## Grading

Graded from the scans with an automated grader, then reviewed by the
instructor. Everything used to grade is here, so you can check your own
exam (returned in `midterm/` in your course repo) against it:

- `exam-solution.pdf`: the answer key.
- `rubric.yaml`: the rubric applied to every exam, with the instructor's
  notes on Q8 and Q15.
- `grading-notes.md`: every grading decision and why. In short: unclear
  marks and disagreements between graders went to the student, written
  answers get partial credit, Q8 is one point per item with any reasonable
  answer credited, and the "which accuracy" part of Q15(b) carries no weight.
- `analysis.md`: where the class split from the key (it flagged no question).
- `scores.png`: score distribution and average by question.

Regrade requests: comment on the Gradebook issue in your repo.
