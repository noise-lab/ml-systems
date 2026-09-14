# Class Project

**Project.** The course will have a small project. You will have the following options
for completing the project: 

1. **ML/Net Leaderboard.** nPrint/pcapML; your goal will be to re-produce or extend some of the
   best-known machine learning results for various applications of machine
   learning in computer networking. The nprint project has a
   [leaderboard](https://nprint.github.io/benchmarks/) with
   links to datasets and the best-known results for each of these problems.
   You are encouraged to follow the directions on that page.

2. **Extensions to Existing Libraries.** We have learned about various
   libraries in class, in particular the NetML and nPrint/pcapML libraries.
   These libraries, however, are works in progress, in terms of feature
   representations (how features are padded and interpolated, summary
   statistics, etc.). You could add different feature representations to these
   datasets and libraries and experiment with whether your representations
   result in greater model accuracy.

3. **Open Problem/Research.** You are welcome to work on an independent research project
   that involves machine learning and computer systems. This option is
   probably better suited for graduate students in computer science who are
   comfortable working on open-ended problems. Your project must be approved
   by the instructor, based on a concrete research proposal.
 
   Some ideas of open-ended problem areas:
   * the systems costs of different features
   * the transferability of a model trained on one dataset
     transferred to another domain (e.g., campus to ISP, IoT to research
     network, etc.)
   * how a model "drifts" over time (using packet captures captured
     at different times, or over time).
   * time-series anomaly detection methods for unlabeled data

## Parameters

* Work in groups of up to three. Groups of four are allowed if the project
  warrants it; in that case I expect a clear delineation of who is
  responsible for what in your project report.
  
* Let me know by the end of Week 5 what your project idea is. Turn in a **one
  page** PDF that outlines the following:
  * Project title
  * Project summary: What is the problem, and why is it important?
  * Data: What data will you be using for the project?
  * Machine learning: What models do you expect to try?
  * Evaluation: How will you evaluate your models?
  * Learning objective: What are you expecting to learn from your project?

## Deliverables

Turn in, committed to your course repository:

1. **Code that I can run end to end** to reproduce your results, with a clear
   entry point (a notebook with "restart kernel and run all" is the easiest
   way to guarantee this, but a script plus a README that says how to run it
   is fine too). Include, or point to, every data file it needs.
2. **A written report** in any reasonable format (Markdown, PDF, a rendered
   notebook) that I can read end to end, with the relevant parts of your code
   or results inline.

The code must run, and it needs to be clean. Presentation of your results is
as important as achieving them.

The writeup has no "expected length", but it needs to be clear. Imagine that
you want to use this as a portfolio for a job interview (something you did as
part of a class), or something I could show to future classes as a cool
project you did (or, that I could use for a future demonstration in the
class). 

## Grading Rubric

* If I can run everything above with one click, and everything is clearly
  documented in such a way that you (or I) could show it to someone else and
  they could reasonably understand it, that will get you an A.

Acknowledge collaborators, outside sources, and any AI tools you used, in the
report.
* If you clearly did the assignment and got some good results, but your code
  doesn't run properly, is messy, or there are other things that are
  incomplete or unclear that will get you a B.
* I don't expect to give many grades below B, but obviously if you don't
  follow the instructions above, you will receive a grade that corresponds to
  the quality of your final product.
  
