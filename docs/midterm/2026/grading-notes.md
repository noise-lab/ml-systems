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

## Q4b, Q6b, Q11b ("Why or why not?"): rubric error, class-wide regrade

- **Status: applied 2026-11-04.** Announced in class and on Slack
  (2026-11-02). 57 of 59 students went up on at least one of the three
  parts; nobody went down. Full credit before and after: 4b 4 → 48,
  6b 3 → 50, 11b 15 → 37. Answers that still give no correct reason keep
  the partial credit they already had.
- **The error.** Each part asks only "Why or why not?", but the original key
  also required an alternative the question never asked for: better
  behavioral features (4b), passive measurement from the time of the problem
  (6b), and a time-ordered train/test split (11b). Many students lost points
  for leaving these out. That is an error in the original rubric, not in the
  answers.
- **The fix.** Full credit (2 points) for the correct answer (No) with a
  correct reason why. Never deduct for not proposing an alternative; an
  alternative that is offered may count as the reason. The model answers
  stay as examples, not requirements:
  - 4b: time of day is an artifact of how the data were collected (or:
    scans and browsing happen at any hour; the model would not generalize).
  - 6b: the test measures conditions now (the problem may have been
    transient), or measures the path to the speed-test server rather than
    to Zoom, or is limited by Wi-Fi or the device.
  - 11b: a random split trains on later hours and tests on earlier ones
    (temporal leakage), or hourly rows are not independent.
- **How.** Mirror this as a `note:` on parts 4b, 6b and 11b in rubric.yaml,
  regrade only those parts for every exam (`gradescope grade --parts
  4b,6b,11b`), and keep the higher of the old and new score per part. No
  score goes down.

## Q10 (filling missing values with the column average): key error, full credit for everyone

- **Decided 2026-11-04, applied the same day.** The original key was flawed
  in three ways, found while answering regrade requests:
  - its "reasonable" example (a timestamp bounded by its neighbors) is
    filling from the neighbors, not the column average the question asks
    about; a timestamp or other ordered series is in fact a fair *bad*
    example, because the column mean ignores order and hides bursts;
  - it missed categorical or discrete fields (protocol, port, TCP flags, a
    resolution label) as bad examples: their average is a value that does
    not exist;
  - it expected a reason for the reasonable example, but the question asks
    for a reason only for the bad one.
- **The fix.** Every answer receives full credit (3) on Q10. The model answer
  in the solutions is corrected.

## Q2, option E ("Timestamps that go backwards"): ambiguous clarification

- **Decided 2026-11-04.** A clarification given during the exam ("time flows
  forward") could be read two ways: packets out of order relative to each
  other (a data problem), or a trace file listed newest first (not a data
  problem). Leaving E blank is credited as correct; it applied to 10
  students.

## Q5, option E (inferring resolution from a capture of your Netflix session): stands

- **Considered 2026-11-04, no change.** A regrade request argued the option
  was ambiguous between passive and active measurement. 44 of 59 students
  marked it, so the class did not split on it; it stays as written.

## Answers added to the solutions on regrade (2026-11-04)

Reasonable, correct answers the original key did not list. Each is now in
the model answer, and the students who gave it received credit.

- **Q3:** email logs with the sending server's IP geolocation (distance from
  the recipients) and the sender's AS or BGP route.
- **Q14(b):** segments have roughly constant duration, so the bytes per
  segment reveal the bitrate and with it the resolution, even without the
  words "adaptive bitrate".
- **Q15(b):** "more data" counts as a fix when it means more, and more
  varied, households.

## Scan and reading corrections (2026-11-04)

Regrade requests also caught places where the scan, not the answer, was the
problem. Each was checked against the page image and corrected:

- writing that continued below or beside the answer box (Q7, Q10, Q12, Q14);
- a number read wrong or cut off at the box edge (980 read as 990; 75%
  read as 7%);
- a filled bubble the scanner missed (Q5), and feedback ratings read as
  blank or put in the wrong box (Q16).

Bigger answer boxes on the final.

## Regrade batch log

- **2026-11-04.** 18 comments on Gradebook issues: 16 regrade requests and
  2 notes about project proposals. All answered on the issue. With the
  class-wide fixes above, 58 of 59 students went up; nobody went down. Class
  mean 42.2 → 44.9 of 50.

