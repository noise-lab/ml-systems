# Summer 2026 — Paris: what's different this term

This offering runs for three weeks (Aug 31 – Sep 18, 2026) at the University of
Chicago Center in Paris, twelve meetings instead of the usual eighteen. The
[syllabus](../syllabus.md) describes the standard nine-week course; the items
below override it for this term. Everything else in the syllabus stands.

## Grading and deliverables

| Component | This term |
|---|---|
| Exam | **One combined exam** on **Thu Sep 17**, in class. No midterm. Designed for about 45–50 minutes; the full class period is available and there is no hard cutoff. Closed book except one 8.5×11 handwritten sheet, both sides. |
| Assignments | **Two**, not 3–5: [Assignment 1 (Video Quality Inference)](../assignments/1-Video-Quality-Inference.html), due Mon Sep 7, and the project proposal. No third assignment (class vote, Sep 10). |
| Project | Groups of **three** (four allowed with a written division of labor). One-page proposal due end of week 1 (Fri Sep 4), committed to the team's repo. **Show-and-tell Wed Sep 16**: informal, go around the room, slides optional. **Due Sun Sep 20, 11:59 pm Chicago time.** It's Git: you may keep pushing after the deadline and I will look at updates, up to about one day before the grade-submission deadline. See [project](../assignments/project.md). |
| Participation | Show up, be engaged. **No reading responses and no in-class quizzes** this term. |
| Late policy | Unchanged: 96 late hours for the term, tracked from commit timestamps, no need to ask. |

Submission is by Git: fill in the intake form (Canvas) with your GitHub username
and repo URL; commit assignment notebooks, the proposal, and the project to that
repo. Scores come back by Slack DM; written feedback arrives as GitHub issues on
your repo.

## Practice exams

The exam is drafted from the class agenda and past exams with the prompt in
[`prompts/generate-summer-final.md`](../prompts/generate-summer-final.md). You have all of the inputs, so you can
generate your own practice exams the same way: point the prompt at
[`agenda/2026-summer.md`](../agenda/2026-summer.md) and the [past midterms](../midterm/)
and [finals](../final/).

## What was covered, in order

The schedule table on the [course page](../index.md) lists the standard lecture
order. This term the meetings went:

| Meeting | Date | Lectures (schedule numbers) | Hands-on |
|---|---|---|---|
| 1 | Mon Aug 31 | 1 Introduction; syllabus | 01 Packet capture |
| 2 | Tue Sep 1 | 2 Security | 02 Scanning |
| 3 | Wed Sep 2 | 3 Performance, 4 Resource optimization | 03 QoE inference |
| 4 | Thu Sep 3 | 6 Feature extraction (incl. nPrint), start of 7 | 06 netml features |
| 5 | Mon Sep 7 | 8 Model training and evaluation | 08 Full pipeline (HTTP/log4j) |
| 6 | Wed Sep 9 | 10 Linear regression, 11 Logistic regression, 12 Trees and ensembles | 10 Linear regression + basis expansion |
| 7 | Thu Sep 10 | 13 Deep learning (with 14 nPrint recap) | 12 IoT trees/ensembles, 13 DDoS neural net |
| 8 | Mon Sep 14 | 15 Dimensionality reduction (incl. autoencoders), 16 Clustering | 15 PCA, 16 k-means |
| 9 | Tue Sep 15 | Exam review walk-through; 17 Diffusion / NetDiffusion, 19 Transformers and state-space models | (NetSSM Colab deferred) |
| 10–12 | Sep 16–18 | Project show-and-tell (Wed), exam (Thu, proctored by Samuel), wrap-up | |

Skipped or merged this term, and therefore **not on the exam**: 5 Data
acquisition (covered only in passing), 9 Naive Bayes (the spam example in
lecture 2 is fair game), the SVM half of 11, the separate nPrint hands-on (14),
and the standalone autoencoder lecture (22; the autoencoder material in lecture
15 is fair game). Lecture 18 (state-space models) was covered inside Meeting 9's
generative-models discussion. Bonus topics 20–23 were not covered.

## Excursions

Huawei Paris research lab (Tue Sep 8, morning) and Lyon with Francesco Bronzino,
co-author of the video QoE assignment (Fri Sep 11).
