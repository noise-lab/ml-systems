---
description: Generate an exam (midterm, final, or single combined exam) from the term's agenda and past exams
---

You are helping create an exam for the course *Machine Learning for Computer
Systems*. Students may run this same prompt to generate practice exams; the
instructor runs it to draft the real one.

## Parameters (ask if not given)

1. **Exam type**: `midterm`, `final`, or `combined`. The nine-week offering has a
   midterm and a final; the three-week offering has one combined exam (see
   [`generate-combined.md`](../combined/generate-combined.md)). The nine-week offering
   has typically run in Autumn and the three-week one in September, but either may
   move to another part of the year; go by the offering's length, not the season.
2. **Term agenda file**: `docs/agenda/<term>.md` (e.g. `docs/agenda/2026-summer.md`).
   The agenda index is `docs/agenda.md`.
3. **Meeting range** to cover (e.g. Meetings 1–8 for a midterm, 1–12 for a combined exam).
   If the term has a page under `docs/terms/`, read it: it lists lectures that were
   skipped or merged and are therefore **out of scope**.
4. **Length**: pages (default 4 for a midterm, 6 for a final or combined exam) and
   total points (default 50; combined exam 75). Design for ~30–40 minutes of work
   per 50 points.
5. **Output directory**. Instructor: `/Users/feamster/Dropbox/Tmp/exams/ml-systems/<type>/<YYYY>/`
   (cloud-backed, outside the public repo — never commit exam files before the exam
   is given). Students: any directory of your own; nothing here is secret.
6. **Past exams to imitate** (read each one's `rubric.yaml`, `grading-notes.md`, and `analysis.md` first where present; skip questions they mark ambiguous): 2–3 most recent under `docs/midterm/<YYYY>/` and
   `docs/final/<YYYY>/` (each has `questions.tex`, `instructions.tex`, `exam.tex`,
   `feamster.sty`, and usually a Makefile). For a combined exam, read one midterm
   and one final. Imitate exams of the same kind: nine-week midterms and finals are
   consistent with each other; the three-week combined exam differs in length and layout.

## Task

Create an exam that:
1. Covers the material in the chosen meeting range of the agenda, in proportion to
   time spent, and nothing outside it. The agenda is the authority on scope, not
   the topic lists in these prompts or the past exams.
2. **Avoids everything the agenda rules out.** Anything the agenda marks "will not
   be asked", "not exam material", "not covered", or skipped, and anything a later
   meeting says will not be on this exam, is out of scope even if a past exam
   asked about it.
3. Fits on exactly the requested number of single-sided pages and totals exactly the
   requested points.
4. Mixes multiple choice ("select all that apply", 3–4 pts), yes/no with explanation
   (3–4 pts; ask "Why or why not?"), and short answer (2–5 pts, generous answer boxes).
5. Includes at least one question about each assignment due in the range and
   several about the hands-on activities the agenda records for the range (what the
   code did, why, what the result showed). Hands-ons vary by term; use the agenda,
   not a fixed list.
6. Uses specific examples from class: the agenda records the analogies and
   discussions the instructor used (e.g. husky/wolf spurious correlation, the
   acknowledgment-packet problem, Netflix segment download rate, the "always says no"
   99.9%-accurate classifier, ephemeral routes for spam). Where the agenda says
   "this would be a good exam question" or carries a **Midterm flag** / **Final
   flag**, it is: cover every such flag in the range.
7. Tests understanding, not memorization, and is not tricky.
8. Ends with a 3-point feedback section (interest, difficulty, one like, one suggestion).

## Steps

1. Read the agenda file for the meeting range and the term page (if any).
2. Read the past exams to learn the LaTeX conventions and the point/format mix.
3. List the key concepts and exam flags per meeting; drop anything the term page
   marks as skipped and anything the agenda says will not be asked.
4. Write `questions.tex` (with solutions), `instructions.tex`, `exam.tex`, a
   `Makefile`, and a `README.md`; copy `feamster.sty` and the `Makefile` from the
   most recent past exam of the same kind (e.g. the last nine-week offering's midterm).
5. Build both `exam.pdf` and `exam-solution.pdf`; verify page count, point total, and
   layout; iterate until they match.

## LaTeX conventions (from `feamster.sty`)

- `\prob{N}` … `\eprob` wraps a question worth N points. Do **not** use `\prob{}` for
  the acknowledgment box in `instructions.tex` — it would add a phantom point. Use:
  ```latex
  Write your full name in the box to acknowledge the instructions.

  \shortanswerbox{3.25}{Nick Feamster}
  ```
  The name box is not a numbered question; the first `\prob` is question 1.
- Multiple choice: `\correctanswercircle{}` for correct options, `\answercircle{}`
  otherwise. Keep option text short enough for the two-column layout.
- **Randomize where the correct answers sit.** The key must not be readable
  from the layout. In every multiple-choice question put the correct options at
  random positions: in a two-column grid spread them across both columns and
  different rows, never all in the left column or the top rows; in a
  single-column list, not always first. Across the exam, correct options should
  land about evenly in every position and column, consecutive single-answer
  questions should not share the key's position, and select-all questions should
  vary in how many options are correct. Balance yes/no keys too: use both
  `\yesnoyes` and `\yesnono`, in no fixed pattern. Before finishing, write out
  the whole key (for example "Q1: B, C, F; Q5: no; Q9: D") and fix any pattern
  you can see in it.
- **Show the answer key as a table before the exam is final**, for the
  instructor to look over, one row per question with an answer to check:

  | Question | Type | Correct answer(s) |
  |---|---|---|
  | Q3 | single answer | 2 of 5 |
  | Q5 | select all | 1, 2, 4 of 5 |
  | Q8 | yes/no | no |

  Give positions as they appear on the page (count across the row, then down,
  in a two-column grid). Note under the table any pattern still visible, such
  as two single-answer keys in the same position, so the instructor can decide.
- Yes/no: `\framebox{\yesnoyes}` or `\framebox{\yesnono}` (no braces around Yes/No).
- Short answer: `\answerbox{height-in-inches}{solution text}`; 1.0–3.0 in depending on
  expected length; solution text must be a single paragraph and must not contain `&`.
  Put a blank line before `\answerbox`. Very short answers: `\shortanswerbox{width}{text}`.
- Sections: `\section*{Name}` followed by `\vspace*{-0.1in}` to save space.
- `instructions.tex` must carry the permitted-materials statement verbatim:
  "This exam is closed-book and closed-notes except for {\bf one 8.5\,$\times$\,11-inch
  sheet of paper with handwritten notes (both sides permitted)}. No electronic devices,
  additional pages, or other materials are permitted."
- `exam.tex` uses `\usepackage[]{feamster}`; the Makefile produces the solution
  version by sed-inserting the `solution` option (`make exam`, `make solution`,
  `make all`, `make clean`).

## Validation

```bash
make all
pdfinfo exam.pdf | grep Pages          # must equal the requested page count
grep -o '\\prob{[0-9]*}' questions.tex | tr -dc '0-9\n' | awk '{s+=$1} END {print s}'   # must equal the point total
```
Open the PDF: no overlapping text, answer boxes start on a new line, multiple-choice
options don't spill across columns. Open `exam-solution.pdf` too and look at where
the filled bubbles and circled yes/no answers fall: no column, row or position
should stand out. Too long: remove page breaks between sections,
shrink boxes, merge questions. Too short: enlarge boxes.

## Instructor-only

- Exam files stay in Dropbox until the exam has been given; commit them to
  `docs/<type>/<YYYY>/` afterwards, with solutions, as practice material for the
  next offering.
