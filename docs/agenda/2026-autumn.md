## Agenda — Autumn 2026 (Chicago)

Nine-week term, meetings Mondays and Fridays, 80 minutes each. Notes are
reconstructed from the class recordings after each meeting; where a recording
was partly inaudible the entry says so rather than guessing.

### Meeting 1 (Mon Sep 28)

* **Course mechanics (syllabus walk-through)**
    * Everything hangs off the course web page and the GitHub repo: slides (now Quarto decks, kept current), hands-on notebooks, past midterms and finals (every exam question ever asked, including the Summer 2026 Paris final)
    * The book (`book.pdf`) is on Canvas Files, not on the web; a fresher copy will be uploaded. Canvas is otherwise unused
    * Agenda files: one per term, built from the class transcript after each meeting; an Autumn 2026 file (this one) is the answer to "what did we cover / what is on the exam"
    * Exam prompts (`docs/prompts/`) are public: past exams + agenda + topic list → generate your own practice midterms the same way the real one is drafted
    * Exams are in class but **not timed**; designed for a short sitting, you have the whole period and beyond. Accommodations page: `sds.html` on the course site (not linked publicly). Everyone gets untimed exams and deadline flexibility
    * Midterm: one 8.5×11 handwritten sheet, both sides. Final: two sheets (or reuse the midterm sheet plus one)
    * **To do tonight:** join Slack, fill out the intake Google Form with GitHub username and private repo URL, create the private repo. Roster will be checked against submissions; nudges, scores, and late-hour balances arrive by Slack DM; written feedback as GitHub issues
    * Repo setup clarified in Q&A: clone the course repo (notebooks, slides, web page; `git pull` occasionally for fixes) **and** make your own private repo for assignments. Ignore the template instructions for now, they are being simplified. The `#github` Slack channel shows every push to the course repo
    * Assignments: four, released Mondays, due the following Friday-ish (one to two weeks each), starting Oct 5. "Assignment 0" on the schedule is a placeholder. Topics subject to change (one will likely be generative models, after the popular September assignment). Each has a rubric; do the work and you get credit
    * Midterm has an in-class part plus a take-home part done alone (coding agents probably allowed; to be decided)
    * Project: groups of 3–4 (four encouraged given class size), proposal due week 5 (Oct 26), presentations in week 9, deliverable due the Sunday of finals week. Examples from the Paris term to come
    * Participation: no reading responses, no quizzes; show up, ask questions, engage
    * Late policy: 96 late hours for the term, tracked from Git commit timestamps, no need to ask; no additional extensions except for genuine extenuating circumstances (health, family), for which you should DM
    * Communication: post questions in public Slack first (classmates answering counts as participation); DMs for personal matters; email is the most reliable fallback. TAs Tajvir and Anagha; office hours by Calendly sign-up, posted in Slack
    * Academic honesty: collaborate freely, use coding agents freely, but **acknowledge collaborators and sources**; likely to be asked for prompts. Understand what you turned in: the midterm and final ask about the assignments. Copying without acknowledgment is the one thing that has gotten people in trouble
    * A dates sheet (assignment out/due dates, proposal, project) will be posted and pinned in Slack

* **Why this course exists**
    * Built five or six years ago as "everything I wish I knew as a practitioner in one place": how to apply ML to systems problems, hands-on, without a math-course treatment
    * Most students have seen linear regression already (mathematical foundations, adversarial ML) so the course spends its time on application, not derivations
    * Scope narrowed from "computer systems" to computer networks: "the internet is my favorite system." Few students have taken networking, so treat this as a networking course that happens to be applied ML

* **Motivation: why ML for networks (Introduction, Lecture 1, first half)**
    * The internet is critical infrastructure, prone to failure, misconfiguration, and attack, and it does not self-heal or self-manage (yet)
    * Troubleshooting example: a Zoom call or Netflix stream degrades. Detecting there is a problem before a trouble ticket, then localizing it (building Wi-Fi, upstream/CDN, the application, an old device) are all **inference problems**, hence ML
    * Attacks are continuous ("we are being attacked right now"); detection and mitigation are inference problems too
    * Capacity provisioning and forecasting (three devices per student now; how much more next year?)
    * Why not rules and closed-form equations: they used to work (compute loss rate and RTT, predict page load time); a modern page load opens many parallel connections to tens of servers, so no single equation applies. ML plus abundant data replaces the pencil-and-paper model
    * Pervasive encryption (past ~10 years): good for privacy, but resolution, bitrate, rebuffering can no longer be read off the traffic and must be inferred
    * Attacks evolve; conditions drift. **Model drift** anecdote: a Verizon model predicting call drops and data-rate degradation at cell towers stopped working when summer arrived (humidity and leaves on trees change radio propagation). Even models need retraining; covered around week 8–9

* **The ML pipeline picture**
    * A foundations course covers the "modeling" box; this course covers everything around it: getting data out of the system, representing it, presenting it to the model, and the efficiency consequences (training time, time to prediction)
    * First few weeks are mostly about getting network data into a form a model can consume

* **Three application areas (preview of Lectures 2–4)**
    * Security: detecting and mitigating attacks (Friday)
    * Performance: inferring video quality (resolution, rebuffering) from encrypted traffic; this is Assignment 1
    * Resource management: short- and long-term capacity decisions

* **Brief history of ML in networking**
    * ~30 years ago: email spam filtering with a **naive Bayes classifier** (asked in class; will be covered)
    * Mid-2000s: ML for botnet detection from network traffic (the instructor's startup era)
    * Mid-2000s to recent: network programmability. The vision: measure, infer (attack, performance degradation), then act through control software. Still the vision; increasing use of generative/agentic AI in production, but not there yet

* **Learning objectives**
    * Know when ML is the right tool at all; when a simple rule suffices; when regression, a tree ensemble, or a deep model fits. Deep learning gets compared against simpler models throughout; for many problems it is unnecessary
    * Apply an end-to-end pipeline: real data is dirty (errors, missing values, outliers). Look at the data before modeling
    * Understand data-representation trade-offs: the same phenomenon can be represented many ways
    * Practical pitfalls: overfitting, label scarcity, concept drift, privacy

* **Homework for Friday:** clone the course repo and make sure the Hands-On 1 (packet capture) notebook loads locally. No hands-on today; Friday will start with Hands-On 1 (possibly two hands-ons) and Lecture 2, Security

### Meeting 2 (Fri Oct 2)

*Plan for today. This entry is replaced with what was actually covered once the class transcript is in.*

* **Housekeeping**
    * Course staff: TAs are Anagha Tiwari and Taveesh Sharma (correcting what was said Monday); office hours to be announced
    * Repo setup check: private repo created, `feamster` invited, intake form filled out
    * Assignment 1 (Video Quality Inference) goes out Monday Oct 5; dates sheet is pinned in Slack
* **Finish the Introduction (Lecture 1)**
    * Why data representation matters: the same traffic can be represented many ways, with different costs and accuracy
    * The data-preparation reality: most of the effort and most of the errors are before the model
    * Tools and setup: course repo cloned, notebooks load locally
* **Hands-On 1: Packet Capture Basics**
    * Capture or load a trace, look at it in Wireshark
    * Packets to pandas: load a pcap into a dataframe and do first-pass analysis
* **Security (Lecture 2), Part 1: ML applied to network security**
    * Two directions: ML for security, and the security of ML systems
    * The measure, model, control loop
    * Spam and phishing: behavioral and network-level features (SNARE), and why they persist when content features do not
    * Moving earlier in time: predicting malicious domains from DNS registration features
* **Hands-On 2: Security (Scanning)**
    * Start in class; finish on your own before Monday
* **Preview of Monday**
    * Security, Part 2 (attacks on ML systems: evasion, poisoning, privacy), then Performance (Lecture 3)

