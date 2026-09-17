# ML Systems Final — Summer 2026 (Paris)

Given Thu Sep 17, 2026. Published here with solutions as practice material; build with `make all`.

Single combined exam for the three-week Paris offering (Thu Sep 17, 2026).
Designed for about 45–50 minutes; the full class period is available.
Generated from `docs/final/generate-summer-final.md` in the course repo
(agenda `docs/agenda/2026-summer.md`, Meetings 1–8; past 2025 midterm and final
as format models), then edited by hand.


## Files

- `exam.tex` — main file (`\usepackage[]{feamster}`; the Makefile flips on `solution`)
- `questions.tex` — questions with solutions (75 points incl. 2 feedback points; student version is 6 pages)
- `questions-v1.tex.bak` — the pre-review draft (65 points, 5 pages); `questions-v2.tex.bak` — the post-review draft before the format pass (more short answers)
- `instructions.tex` — standard instructions and permitted-materials statement
- `feamster.sty`, `Makefile` — copied from `docs/final/2025/`

## Build

```
make all        # exam.pdf and exam-solution.pdf
make clean
```

## Coverage note

Meetings 1–9, revised after the Sep 15 in-class review: adds throughput vs.
latency, the negative-RTT pipeline question, the block-list question, the
real-time select-all, a confusion-matrix computation, model-complexity knobs,
a dendrogram cut, and a Generative Models section; drops the tree region-map
and one short answer; more yes/no + why and select-all, fewer short answers. Format pass (Sep 15 evening): Q2 stage as MC, Q4 and Q7(a) and Q21(a) as yes/no + why, Q7(b), Q11, Q15(a), Q20, Q21(b) as MC, Q16 reworded, full-width boxes on Q19 and Q22. Only four pure short answers remain (Q6, Q13, Q14, Q16).
