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

* **Hands-On 1: Packet Capture Basics** (instructor walk-through, then about 25 minutes in pairs)
    * Wireshark: capture on the Wi-Fi interface, with a capture filter restricting the trace to one host (the CS department web server). Find the server's address with `dig +short`, start the capture, reload the page, stop, and save
    * The first hands-ons use CSV exports; soon everything will use the raw packet-capture format (pcap)
    * The habit being built: do not hand a data set to a model before looking at it. Garbage in, garbage out. Ask whether the capture makes sense
    * Reading the trace
        * Each row is a packet. Length is in bytes. Sorting by length shows the large packets all flow from server to client, as expected for a download
        * Large packets top out near 1,400 to 1,500 bytes because packets have a maximum size. A negative length or a 10,000-byte packet would mean something is wrong with the data. (The exact numbers are not exam material; the sanity-checking habit is)
        * Opening sequence: the TCP three-way handshake, then the TLS client hello and server hello, then the data transfer, visible where the packets get large. TCP and TLS details are background, not exam material
        * A packet is layered. Link-layer headers (Wi-Fi, Ethernet) are not used in this course. The IP header carries source and destination addresses and length. The TCP header carries more fields
    * What becomes a feature: timing, packet length, direction, and header fields
        * Header fields are built differently by different operating systems, so they can identify the operating system or device (the idea behind `nmap`). Past exams ask when using such fields as model input is a good or a bad idea
    * The payload is encrypted, so the application data cannot be read from the trace. Good for privacy, and the reason inference is needed
        * This is Assignment 1: before encryption, video resolution, rebuffering, and startup delay could be read directly from traffic. Now a network operator who wants to know whether users are having a bad streaming experience has to infer it from data like this
    * Notebook steps 3 and 4: compute summary statistics (mean and median packet size) and visualize (histogram, cumulative distribution)
    * Using coding agents for syntax is encouraged. Hands-ons are not submitted, but exams can ask about them; solutions will be posted
    * Questions that came up during the hands-on
        * Estimating round-trip time: match a data packet's sequence number to the acknowledgment for it and subtract the timestamps. Assuming adjacent packets pair up is the sloppy version
        * Sanity-checking the result: about 41 ms was observed, high for a nearby server. A cross-country round trip is roughly 80 ms; a ping to a data center across the street was 3 to 4 ms
        * Latency varies with congestion, Wi-Fi, and server response time. Working out which is itself an inference problem. Plot the distribution rather than trusting one number
        * Two students doing the same task captured very different numbers of packets. Possible causes include retransmissions and different network conditions; left as an open question
    * Where this sits: the start of the pipeline, acquiring data, checking it, and cleaning it. Feature construction comes next
* **Security (Lecture 2), first part only**
    * The course framework applied to security: measure, model, then act (block, rate-limit)
    * Email spam was one of the first real-world applications of machine learning anywhere: naive Bayes on the words of the message (naive Bayes is covered around week 3)
    * Content-based filters can be evaded. Attackers hid innocuous text in the message (white-on-white text in HTML email) to confuse the classifier. A classic adversarial machine learning problem
    * Behavioral, network-level signals persist regardless of content: timing, sizes, and where the traffic goes
    * Predicting an attack before it happens: attacks need setup, and setup is observable. Example: many similar domain names registered at once is a feature that an attack is coming
    * The remaining slides in the deck were not covered
* **Preview of Hands-On 2 (Monday)**
    * Two traces: an ordinary page load and an attacker scanning a web server for a known vulnerability. Compare timing and size characteristics, as a step toward features and classification
* **Logistics**
    * Hands-On 2 was not reached today. All three use cases (security, performance, resource management) will be done by the end of next week
    * Assignment 1 goes out next week

### Meeting 3 (Mon Oct 5)

* **Housekeeping**
    * Assignment 1 (Video Quality Inference) is officially out: the notebook and a rubric are in the public template; copy them into your private repo
    * Office hours: a TA holds them Wednesdays at 3 pm on Zoom; the instructor's are Tuesday evenings by sign-up, details to be announced
    * Intake form and private repo: most are set; anyone with problems should say so
* **Security (Lecture 2), ML applied to network security**
    * Two directions: ML *for* security, and security *of* ML systems. This course does the first. The second has its own course (adversarial machine learning) and is not covered here, except that a detector's designer must expect attackers to try to evade it
    * Detecting attacks is a classic machine learning problem. Other examples: detecting malware infections from a device's abnormal behavior; attacks built on the domain name system (malicious domain registrations, phishing); anomaly detection in traffic in general (unusual volume, unusual time of day, unexpected destinations, devices that should not be on the network). Anomalies can be failures or misconfigurations as well as attacks
    * **Why security is an especially hard ML problem**
        * The things to detect are rare and often new, so there is little or no training data
        * Attacks are often one of a kind; the next one looks different
        * Class imbalance: unlimited normal traffic, very few examples of the attack. Generative models can help by synthesizing more attack examples (covered around week 8)
        * Concept drift: a deployed model stops working, because conditions change (the seasonal cellular example) or because adversaries adapt
        * Many data formats in practice; this term uses a small number on purpose
        * Real-time constraints: time to detection matters for closed-loop operation, not only accuracy on a test set
    * **Spam, revisited**
        * Email spam filtering was one of the first practical applications of machine learning; naive Bayes on message words (naive Bayes is covered in a couple of weeks)
        * Content filters are cheap to evade (stuffing innocuous text into the message). Behavioral and network-level signals are costlier to evade: sending volume (hundreds of messages in a short time), unusual hours, many recipients per message, connection rate, recipients per session, connection duration
        * The detailed slides on that work are not exam material; the idea that sending behavior is a signal independent of content is
    * **Predicting attacks before they happen**
        * A spam campaign needs a website, which needs a registered domain. With access to registration data, bursts of related registrations are a signal before any spam is sent
        * The same idea was applied to botnets and to disinformation campaigns: newly registered domains that imitate news sites, hosted somewhere that makes no sense for the intended audience
    * **Midterm flag:** given a kind of attack or unusual behavior (a botnet, a phishing campaign, a disinformation campaign), what features would you look for and what data would you need to detect it? Feature design and data collection, not a specific paper
* **Hands-On 2: Security (scanning)**
    * Two traces supplied in the course data: an ordinary web page fetch and a scan of a web server for the Log4j vulnerability. Same questions as Hands-On 1: packet count, duration, packet-length distribution, protocol types, inter-arrival times, number of unique destinations
    * What the traces show: scanning sessions are much shorter, there is little large server-to-client content, and the unique-destination counts and inter-arrival times differ. Charts to be posted
    * Clarified in class: you will not be asked to write Python functions; work cell by cell and use coding help for syntax. Only a few students have trained a model with scikit-learn before, which is fine
* **Performance (Lecture 3): inferring quality of experience**
    * Network operators have limited visibility into the user's experience on an application they do not control, yet their decisions affect it, and they need to know when a bad experience is their failure to fix
    * Video is the dominant traffic
    * **Speed is not experience.** A speed-test number does not map to what a user feels. A 2013 study for a regulator plotted page load time against throughput: beyond a point, more throughput does not load pages faster
        * Why: latency, not throughput, is the bottleneck. Server processing time, propagation delay (speed of light), queueing
        * Road analogy: lanes are throughput (cars per minute); trip time is latency. Adding lanes does not shorten the trip
    * **Video QoE metrics** (the subject of Assignment 1): startup delay, resolution, resolution switches, rebuffering
        * The same question, asked by journalists a few years later: does more speed mean better video? Same flattening. The finding ran on the front page of a national newspaper in August 2019
    * **The encryption wall.** As video took off around 2015 to 2016, traffic became encrypted. Operators cannot see resolution, startup, or rebuffering from the traffic. That turns QoE into an inference problem on timings, sizes, inter-arrival times, and the number of parallel connections (one player opens more connections under poor conditions, itself a possible signal)
    * **The pipeline used in that study**, broader than the assignment
        * Service identification: find the video traffic among everything else in a home, using DNS lookups for the video servers' domain names
        * Features: bytes per second, packets per second, packet-size statistics, retransmissions, latency
        * Segments: adaptive streaming (DASH) fetches video in segments whose size reflects the chosen resolution; segment boundaries are visible as gaps in the download, and segment sizes become features. Assignment 1 does exactly this
* **Hands-On 3: QoE inference** (started, about eight minutes)
    * Step 1: from the supplied capture, identify the video traffic by DNS lookups to the video server domains. Next: find segment downloads and their sizes from gaps in the traffic. Finish at the start of Friday's class
    * Demo: the browser's developer tools (Inspect, Network tab) show the same video server domains and the byte-range request for each segment. The session can be saved as an HTTP archive (HAR) file, another useful data source; not exam material
* **For Friday**
    * Finish Hands-On 3, then the third use case: resource optimization (how players and networks adapt)
