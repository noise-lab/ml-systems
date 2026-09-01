# 01-Intro — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made

- **2026:** Added a `.vignette` hook using the HPE Juniper Mist AI "self-driving
  network" announcement (August 26, 2025). This is a concrete, dated, primary-source
  example of the measure → infer → control loop running in production on commercial
  enterprise networks. Source: HPE newsroom / Juniper Networks press release,
  2025-08-26. URL verified June 2026.
  (https://newsroom.juniper.net/news/news-details/2025/HPE-accelerates-self-driving-network-operations-with-new-Mist-agentic-AI-native-innovations-2025-Gw1M0nv_Oz/default.aspx)

## Suggested missing coverage on broad themes

- **Adversarial ML and feedback instability:** The vignette mentions human-in-the-loop
  trust but does not explain why closed-loop control can be dangerous (adversarial
  drift, feedback oscillation). A one-slide treatment would prepare students for the
  security and deployment chapters.
- **Privacy constraints on network data:** The intro motivates ML from a performance and
  security angle, but encryption as a *constraint* on what ML can observe is only
  touched briefly. A slide on "what the ML model can and cannot see" would set up the
  measurement chapter better.
- **Causality vs. correlation:** Many classical networking papers conflate the two. A
  brief note distinguishing predictive models from causal ones (and why it matters for
  control) would sharpen students' critical reading skills.
- **Ethical and legal framing:** The source extract (and book intro) defer this to later
  chapters. A one-sentence "this course addresses these" pointer on the course-arc slide
  would signal the topic's importance earlier.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items likely to age:

- **Vignette (HPE Juniper Mist AI, Aug 2025):** The "self-driving network" story will
  age quickly as competitors (Cisco, Nokia, Arista) make equivalent announcements. By
  2027 there will likely be a stronger, more quantified example (mean-time-to-remediate
  improvement, cost savings, etc.). Re-check the Mist AI blog and Juniper newsroom each
  August before the course starts.
- **History table row "2020s":** Currently vague ("Mist AI; LLM-assisted ops"). If a
  major paper or product deployment with measurable results ships before the next term,
  replace this row with something more concrete and cited.
- **Gartner Magic Quadrant reference:** Gartner releases a new quadrant annually;
  verify that Juniper (now HPE Juniper) retains Leader status for the current year
  before citing it.
- **nPrint/pcapML project:** Verify the leaderboard is still actively maintained and
  that benchmark tasks are current. If the project has been superseded or archived,
  update the project description slide.
- Stronger alternative vignettes not yet used: Google Cloud Network Intelligence Center
  (if they publish concrete ML-driven ops results with numbers); Cloudflare's published
  ML-for-DDoS-mitigation blog posts (2025).

## Curated images

- `images/s04-i01.png`: **Used.** The ML pipeline diagram (Data Ingestion → ... →
  Deployment) from the original slide 4. This is the single most important teaching
  figure in the deck; it recurs twice (pipeline overview and dedicated pipeline slide).
  Credit: Alex Gray (per source extract). Its caption on the "ML Control Loop" slide
  was corrected (2026-09) to describe the image as the pipeline, not the control loop.
- `images/s04-i02.png`: **Used.** Text slide emphasizing "80%+ of effort is in data
  preparation." Reinforces the pipeline slide and is readable as a figure.
- No other images in `images/` — only two were extracted from the source pptx.
  Verified 2026-09: every file in `images/` is referenced in `slides.qmd`, every
  reference resolves on disk, no `.wmf` files present, and the deck renders cleanly.
  The original pptx's diagrams on slides 6 ("Measure, Model, Control"), 7 ("ML
  Control Loop"), and 9 ("Network Management Cycle") were built from native
  PowerPoint shapes, not embedded images, so no image files exist for them; their
  content is carried textually by the "ML Control Loop" and "Gap" slides.
- Suggested addition for a future rebuild: the `ml-loop.png` figure from the book's
  `figures/` directory (referenced in `intro.rst` as `fig-ml-loop`) would replace the
  pipeline figure on the "ML Control Loop" slide with a cleaner, book-consistent
  diagram.

## Source

- Rebuilt from `_source-extract.md` (19 slides) — topics preserved: motivation,
  ML control loop, measure/infer/control for security and performance, course
  structure and logistics.
- Book chapter: **Introduction** (`text/intro.rst`) — terminology and framing follow
  the book throughout; Mitchell (1997) definition, network management taxonomy
  (short-term / long-term), "Why Now?" historical arc, and pipeline emphasis all come
  from the book.
- Source extract divergences: The source extract's "Structure and Syllabus" and
  logistics slides (slides 13–19) were consolidated into three slides ("Course Arc,"
  "Assignments and Project," "Course Mechanics") to stay within the 15–30 slide target.
  The source's "How Networks Run: Measure, Model, Control" diagram (slide 6) was folded
  into the ML control loop slide rather than reproduced literally, because the book's
  framing is cleaner.
