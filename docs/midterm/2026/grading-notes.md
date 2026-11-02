# Grading notes: ML for Systems midterm, Autumn 2026

Instructor decisions about this exam that grading must follow. The
`grade-exam` skill reads this file before grading and before entering or
returning any grade. Notes that change scoring are also mirrored as a `note:`
on the part in `rubric.yaml`, so the grader applies them; this file is the
source of truth, and the two must agree.

## All multiple-choice and yes/no questions

- **Unclear marks get the benefit of the doubt** (decided 2026-10-30). A
  bubble or yes/no circle the scan reads as ambiguous (part-filled, erased) or
  possibly crossed out is read the way that helps the student: as filled if it
  is a correct option, as empty if it is a wrong one. Rubric:
  `ambiguous_bubbles: favor_student`. These parts show a "(resolved) ...
  benefit of the doubt" note in review.html and do not block entry. Marks the
  scan reads clearly are scored as read.
- **Marks other than a filled bubble count** (decided 2026-11-01). An arrow,
  check, circle or X that clearly points at an option counts as marking it
  (exam-level overrides in `overrides.yaml`).

## All written answers

- **Independent second grade for uncertain answers** (decided 2026-10-30).
  Every written answer graded with confidence below 0.75 gets a second,
  independent grade from the scan image (the second grader is not shown the
  first). If the two agree within half a point, the answer keeps the higher
  score and leaves review. Rubric: `second_opinion: higher`.
- **Disagreements keep the higher grade** (decided 2026-11-01). When the two
  graders disagree by more than half a point, the student gets the higher of
  the two; nothing goes to review on that account. Always the benefit of the
  doubt.

## Partial credit everywhere

- **No written answer is all-or-nothing** (decided 2026-11-01). Any correct
  element earns partial credit; a zero is only for a blank or a wholly wrong
  answer. Every written zero was reread by hand on Nov 1 and the partial
  credit recorded as `points:` overrides in `overrides.yaml`, each with its
  reason. Multiple choice already scores right minus wrong per option.

## Q8 (netml STATS features vs. nPrint: one advantage and one disadvantage of each)

- **One point per item, any reasonable answer** (decided 2026-11-01). The
  key listed many acceptable answers under a single 4-point criterion, and
  the two graders disagreed on 20 of 57 exams. Each of the four items
  (netml advantage, netml disadvantage, nPrint advantage, nPrint
  disadvantage) is worth one point. Any correct, reasonable point earns it,
  not only those in the model answer (netml is cheap to compute or works
  with less data; nPrint can be less interpretable or needs more compute).
  An item stated loosely, or implied by a contrast with the other approach,
  is credited.
- Regrading under this note never lowered a score: each student keeps the
  higher of the grade before and after the note.

## Q15 (Assignment 1 classifier, 75% random split vs. ~30% household held out)

- **"Which accuracy is the better estimate" in (b) is a bad question.**
  Many students asked about its wording during the exam. It carries no
  weight: never deduct for which accuracy a student picked, or for leaving
  it out. Criterion `b` (2 points) is graded only on naming one reasonable
  change (remove household-specific proxy features, split by household,
  train on more diverse households, balance the classes, or similar).
- Part (a) is graded as written.
- With the full class in (Nov 1), the item analysis showed no other split
  on Q15 (class average 3.3 of 4); no further change.

Recorded 2026-10-30.
