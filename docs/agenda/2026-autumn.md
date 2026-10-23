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

### Meeting 4 (Fri Oct 9)

* **Housekeeping**
    * Project proposal due Monday October 26. Talk to the TAs as you cook up ideas; they have announced office hours and are a good resource
    * In-class midterm Friday October 30, here
    * Monday's class is on Zoom (the instructor is in New York); the link is in Slack. Anyone who wants to sit in the classroom and join from there is welcome to
    * Office hours are by sign-up, Tuesday evenings, link in Slack. Sign up for two contiguous slots if you need more time. If the time does not work for many people, another evening will be added
    * Completed versions of the hands-ons will be released at some point
* **Hands-On 3, continued** (about 20 minutes): identify the video traffic from DNS, then find segment boundaries
    * Two tricks: use the DNS answers to pick out the video servers (the assignment will not need this step), and find segments, which are larger than packets, by the zero-byte packets that separate them. Segments per second then becomes a useful feature. Assignment 1 is the inference step that follows from this
    * **A student noticed** that the player opens several connections to Netflix but uses one far more than the others, and asked why the data is not simply split evenly across them. Nothing is sent without a request from the player: the player asks for segments one at a time, choosing a bitrate for each, so the connections are not a uniform split of one stream. The smaller connections are probably audio, and others may carry different resolutions or standby streams. Because the traffic is encrypted this is partly a guess, which is the whole point of the inference problem
* **Lecture 4: Resource allocation** (chapter 2.3)
    * How much of a finite resource, usually capacity (bitrate, throughput), to give to an application, flow, user, or device. Two timescales
    * **Short timescale:** packet-by-packet or segment-by-segment decisions: how fast to send, which video resolution to request, when and along which path to send inside a data center. Congestion control used to be heuristics in the operating system (watch loss and latency, speed up or slow down); increasingly these decisions are made by learned models in real time, a shift that began about ten years ago
    * **Long timescale:** forecasting. Campus traffic and device counts grow every year; IT has to decide on more access points, more internal capacity, more transit from commercial providers. Content delivery networks (a global network of web caches; students named Cloudflare, Google, AWS, Azure, Akamai, Fastly, and Netflix's own) face the same questions: where to put servers, what draining a data center for maintenance does to users
    * **What-if (counterfactual) evaluation:** is the system resilient to a Chicago power outage, and how bad would it be? You cannot decide whether a mitigation is worth its cost without predicting what happens. Twenty years ago these were closed-form equations (loss rate to TCP throughput); with parallel streams and adaptive players that math says nothing about user experience, which is where ML took over. Still a wide-open research area, including A/B testing and, in the instructor's group, whether generative models can produce traffic "in the style of" another player or another network type for counterfactuals. Good project territory
    * **A student asked** how resource allocation differs from the performance use case. The instructor's framing: performance (Lecture 3, Assignment 1) is *inference* from data; resource allocation is *deciding and acting*. The student connected this to the control loop from Lecture 1, inference on one side and action on the other, which the instructor adopted
    * **Adaptive bitrate streaming.** The video is encoded at several bitrates in segments, each with its own URL; the byte-range requests seen in the browser's developer tools on Monday were segment requests. The player continually estimates network conditions and the state of its playback buffer and requests the next segment at a chosen quality. Switching quality is just requesting a different object (DASH, dynamic adaptive streaming over HTTP). A Netflix paper used buffer occupancy as the main signal
        * Objectives in tension: high average quality; no quality switches (more annoying than a steady lower resolution); no rebuffering; fairness to other applications. Capacity estimates fluctuate because nobody has a reservation: run two speed tests back to back and you get two numbers
        * **Fairness is broken as a concept.** It used to be reasoned about per connection; now Netflix versus a browser is apples and oranges. Pie analogy: equal thirds is not fair if the kids only want a quarter each. A student suggested fairness in terms of each application's quality of experience rather than throughput; the instructor agreed nobody knows how to do that yet. Open research, good project territory
        * Rule-based heuristics do not generalize, so **reinforcement learning** has been applied: features (past segment throughput, download time, next-segment sizes per bitrate, buffer occupancy, segments remaining, last bitrate), an action (bitrate for the next segment), and a reward. Many students have seen RL before, so the RL chapter may get class time later in the term. Today was motivation only
    * **Learned congestion control.** One layer down, the operating system's transport protocol decides how fast to push packets, using signals such as acknowledgment timing and latency. Classic schemes (TCP Vegas, Reno) were pencil-and-paper heuristics; just over ten years ago a student of the instructor's own PhD advisor showed that a model could take those features and output the sending rate. **Will not be asked:** the names or workings of the classic congestion-control schemes
    * **WISE, the what-if scenario evaluator** (Google, about 2008; brought to the instructor by a former student). Search response time is a distribution and the tail matters; milliseconds are worth large sums. Web pages are served with a front-end/back-end split: the nearby front end immediately ships static content (logos, style sheets) while the search itself goes to a back end and comes back as structured data to be merged in. Question: if the India front end goes down for maintenance and Indian users are served from Taiwan, what is the new response-time distribution? Inputs: round-trip times browser-to-front-end and front-end-to-back-end, loss rate, and similar; target: response-time distribution. A regression with a basis expansion, trained on Google's abundant data, deployed at Google. The system also did **causal discovery**, because changing one feature (round-trip time) changes others (loss), so the change has to be propagated before the regression is applied. **Midterm flag:** you will not be asked how WISE works, but you will need to know linear regression and basis expansion, which are coming in the methods part of the course
    * **Provisioning and capacity forecasting.** How much capacity an ISP should buy toward its peers, decided months or years ahead because it means lighting or trenching fiber. **COVID-19** as the case: traffic volumes jumped and the upload/download ratio inverted as homes became mini campuses; ISPs had to augment capacity faster than their models predicted, using time-series forecasting. The plots were shown but not walked through
    * **Model drift / concept drift**, seeded for later (around week 8 or 9): a deployed model works until conditions change, as with seasonality in the cellular example and with COVID
    * Not covered: the Remy and Pensieve details, the "reality check" slide on learned congestion control, and the drift-mitigation list
* **For Monday (on Zoom):** a new module begins, the ML pipeline, starting with data acquisition (chapter 3.2 to 3.3). Class ended a few minutes early

### Meeting 5 (Mon Oct 12, on Zoom)

* **Housekeeping**
    * Class met on Zoom; back in person on Friday.
    * Assignment 1 due Friday October 23. Project proposal due Monday October 26. In-class midterm Friday October 30.
    * Office hours are by sign-up on Tuesday evenings; the first session is Tuesday October 13.
* **Hands-On 3 wrap-up**
    * Recap of what the hands-on did: take a traffic capture, use DNS to identify the service, then use properties of the traffic to find segment boundaries, so that features such as segments per second or segment size become possible.
    * Today's hands-on is probably the last one that does not use the netml library; later hands-ons build on it.
* **Lecture 5: Data acquisition** (chapter 3.2 to 3.3)
    * Data acquisition is the first stage of the ML pipeline: get data out of the network, turn it into features, feed those to models. In networking this is called network measurement or Internet measurement; it is a research area of its own (the Internet Measurement Conference) and the course spends one day on it.
    * Why measure, by use case:
        * Performance and operations: detect and diagnose outages and performance degradations (full outage versus intermittent degraded bandwidth or quality of experience); generate reports for customers and operators, including service level agreements (SLAs). An SLA such as "five nines" means 99.999% availability, about 5 minutes of downtime per year (correction: "maybe 30 seconds" was said in class). An enterprise like the University buying upstream service would expect an SLA from its provider.
        * Tune configuration to actual traffic: after a failure or an unexpected traffic increase, operators may want to adjust; the first step in any what-if evaluation (Meeting 4) is measuring that the scenario exists. Capacity planning, as discussed earlier.
        * Security: attack detection from logs or the traffic itself; characterizing anomalous behavior so it can be detected later; fingerprinting devices and services (example: finding legacy devices older than ten years that need upgrading); gathering normal and attack traffic to feed intrusion detection.
    * **Active versus passive measurement**
        * Active: inject synthetic traffic (probes) into the network and observe how endpoints respond. A speed test was run live in class: it picks a server, downloads as much as it can, then uploads, and reports download and upload throughput, idle latency, latency under download and upload, packet loss, and jitter. Ping to a well-known site and a DNS lookup (with its query time) were also shown as active measurements; the specific commands will not be asked.
        * Passive: capture existing traffic as it traverses the network and infer properties from it. Wireshark and Assignment 1 are both passive measurement.
        * **Midterm flag:** given a description of a measurement you have not seen before, say whether it is active or passive. Also a common question: the design trade-offs between the two.
        * Trade-offs. Active is easy to deploy (you need a server to measure against, though it can be someone else's), has no real privacy concern because the traffic is yours, but may not reflect real user experience and adds load to the network. Passive reflects actual user behavior at the time of the problem (if a user complained about Zoom half an hour ago, the log has it), but requires capture infrastructure and raises privacy concerns because all traffic is being captured. An active test run after the fact may not reproduce a transient problem.
        * A student asked what "capture infrastructure" means and whether active measurement does not also need infrastructure. Passive capture infrastructure is anything that captures bytes on the wire (tcpdump, Wireshark, libpcap, eBPF) plus whatever aggregates packets into flows; capturing someone else's traffic requires significant access. Active measurement needs a server to measure against, but ping, a DNS lookup, or a speed test need nothing deployed by the person measuring.
        * A student asked how, given full access to the network, you tell a network problem from a bad user device. Answer: with active measurement you often do not know what you are measuring: the access link, the home Wi-Fi, the user's device, or something else. A plot was shown in which circle size is the downstream speed measured by a speed test: with high ISP speeds the home Wi-Fi is usually the bottleneck, with low ISP speeds the access link usually is. Passive end-to-end measurement has the same limitation: you can see that a download was slow, not why.
        * Pitfall anecdote: speed-test data from phones was used to argue that an ISP never delivered more than 100 Mbps, but the phones' radios at the time could not exceed 100 Mbps, so the tests were measuring the phone, not the ISP. Lesson: be clear about the conditions under which data was gathered and what is actually being measured.
    * **Dimensions of an active measurement:** upload and download throughput ("speed"); idle latency (nothing else going on, as with ping); latency under load, also called responsiveness (latency while an upload or download is running); packet loss; jitter. Road analogy: idle latency is the drive downtown at 3 a.m., latency under load is rush hour, and road width only matters once there is load. Traceroute was skipped.
    * **Passive measurement: flow-level records**
        * Instead of a packet-by-packet log, summarize traffic per flow: packet and byte counts, start and end times (duration), whether any TCP flags were seen (a binary did-you-see-it), per direction, often in time windows.
        * A flow is the set of packets sharing a five-tuple key: source IP, destination IP, source port, destination port, protocol (commonly TCP). Packets are hashed into buckets by that key. Whether the reverse direction goes in the same bucket or a separate flow is a design choice. Conventionally a router ends a flow after a period of silence, about 15 to 30 seconds (30 is the usual default), and later packets start a new flow; post-process if that does not suit the problem.
        * **Midterm flag:** the precise definition of a flow is an easy question to ask and check.
        * Flow export is standardized (IPFIX, an IETF RFC), so any network hardware can export flow records; it is usually implemented on routers or line cards in the middle of the network where full packet capture is impractical. Standard records also carry the type-of-service byte, the input interface, and the next hop (not used in this course).
        * Aggregates are much lower volume than packet captures but lose most timing information; the netml library developed for the course keeps some timing and size information (for example inter-arrival times), which the course's own research found useful for security-motivated anomaly detection.
        * On very high-speed links flow records may be computed from sampled packets (for example every tenth packet); not something that arises with the end-host captures used in this class.
        * eBPF (extended Berkeley Packet Filter): custom, verified packet filters pushed safely into the kernel so feature-like statistics can be computed at high speed without passing every packet to user space. It replaced the old trade-off between flexible-but-slow software and fast-but-expensive capture hardware, and is standard in large industry networks. **Will not be asked:** how eBPF works or is implemented, or the capture hardware.
    * Preview: the two ways to represent a pcap for learning are aggregating into flow records (the netml approach) or feeding the whole capture, as bits, to a neural network. More on feature representation Friday.
    * A netml enhancement is listed as a possible course project.
* **Hands-On 5: Data acquisition** ([notebook](../notebooks/05-Data-Acquisition-Clean.html))
    * Load a packet trace and generate per-flow statistics by hand: bucket packets by source IP, destination IP, source port, and destination port (protocol can be omitted here because the trace is all UDP on port 53), then compute bytes, packets, and duration per flow, and bytes and packets per second.
    * This is the last time packets are manipulated without a library; netml does this automatically from Hands-On 6 and in Assignment 1. The point is to see what decisions aggregation involves.
    * Students worked in self-selected Zoom breakout rooms for about 15 minutes; the breakout work was not captured.
    * On return a student asked about flows with zero duration. These are one-packet flows; whether their duration is zero, the round-trip time, or something else is a design decision, and when someone hands you a flow dataset you do not know which decision they made.
* **Key takeaways named at the end:** the active versus passive trade-offs, what a flow is, and the design decisions involved in producing and consuming flow data.
* **For Friday (in person):** turning the extracted flow data into features (chapter 3.4); Hands-On 6 uses netml.

### Meeting 6 (Fri Oct 16)

*The recording begins a few minutes into housekeeping; the opening announcements are reconstructed from what was repeated on the recording and in Slack.*

* **Housekeeping**
    * Assignment 1 due Friday October 23. Project proposal due Monday October 26. In-class midterm Friday October 30; it covers everything through the end of week four, that is, through next week's classes.
    * TA office hours are posted in Slack; the instructor's office hours are Tuesday evenings by sign-up.
    * A student asked what a good project looks like. The instructor will post topics and examples from past terms (the summer-quarter projects were especially good); the project page on the site already lists possible directions.
    * Reminder: add the TAs as collaborators on your private repo along with the instructor; they will read assignments and proposals.
* **Lecture 6: Feature extraction and representation** (chapter 3.4)
    * Back to the pipeline: collect raw traffic, represent it as a feature matrix, train, deploy. Today is the second step, which is tricky and often matters as much as or more than the choice of model.
    * Options for a packet trace: throw the raw trace at a model (nPrint does this; trade-offs below) or compute features that separate the classes you care about. Example from the hands-ons: scan versus normal web traffic differ in typical packet sizes and flow lengths. Even one feature has many encodings: packet size as an integer, or binned as small, medium, large.
    * Why it is hard: raw captures are large and unstructured, many transformations are possible, and the best one depends on the task. Even packets per second has a hidden choice, the time bin (one second, ten, thirty, a minute). Encryption hides content, which is why Assignment 1 and the QoE work infer from metadata.
    * Building-block metrics: throughput, latency (the road analogy from Monday), jitter (variation in inter-packet delay), packet loss (itself representable many ways: count, rate, window, percentiles), and statistics over the five-tuple flow.
    * The common path from packets to features: parse into five-tuple flows, then aggregate per flow and, going beyond NetFlow and IPFIX, per time window within a flow to get a time series. Libraries: scapy for parsing, eBPF for in-kernel capture, netml for the course.
    * **netml** (links posted in Slack): one call turns a pcap into flows, another turns flows into a feature matrix that goes straight into scikit-learn. The weeks of hand-mangling captures in the hands-ons can now be two calls. The library is due for a refresh; improving it is a suggested project.
    * A student asked about packet length and payload. Captures commonly record only headers (the Wireshark columns), not the 1500-byte payload, which is encrypted anyway. netml can emit the per-packet size sequence (the SIZE representation).
    * **STATS features** per flow: duration, packets per second and bytes per second (over the flow, or per configurable time bin), and packet-size statistics (mean, standard deviation, interquartile range, min, max). Duration is a plausible attack-versus-legitimate discriminator. A student asked whether this is the only feature vector to know; no, it is a useful default, not gospel, and anything that helps training is fair game.
    * A student asked whether features should be normalized. It depends on the model: a linear model mixing 50-byte and 1,500-byte packets behaves badly without scaling, while tree-based models such as random forests split on thresholds and can usually skip it. netml does not normalize; another project opening.
    * A student asked about measuring how noisy or dirty data is. netml has no sanity checks (negative round-trip times, malformed addresses); adding them is a good project. Always look at the distributions of packet sizes and inter-arrival times before trusting a feature.
    * Other representations netml supports: SIZE (time series of packet sizes), IAT (inter-arrival times), and sampled versions that count packets or sum bytes per time bin, with the bin width configurable; windows can be equal time or equal bytes.
    * Survey of published anomaly-detection systems: no two papers represent traffic the same way, so there is no single right representation. Header fields such as TTL, TCP flags, and window size carry useful information. Frequency-domain (FFT) features, which netml can compute, did not help on the problems the group studied.
    * **Representation learning.** Deep learning is the main form: feed raw data with labels and let the model learn the features, as image classifiers work from pixels rather than hand-counted edges. The appeal is one representation applied once and reused across tasks.
    * **Midterm flag:** explain the trade-offs between manual feature engineering (netml-style) and representation learning (nPrint, deep learning).
    * Costs of manual engineering: it needs domain knowledge (you must know what a packet is to compute packets per second); reproducibility suffers (the DARPA 1998 intrusion dataset is described differently, down to packet and flow counts, across hundreds of papers); and features built for one task (intrusion detection, spam, scans) may not transfer, so the engineering is redone each time. A library like netml aims to cover most needs.
    * **nPrint:** if images classify from pixels, classify packets from bits. Each packet becomes a fixed-width bitmap aligned to protocol field boundaries; each bit is 0, 1, or -1 for a field that is absent. Alignment matters because the model is only seeing bit positions: if the source address sat at a different offset in different packets, position would mean nothing. Costs: about twice the storage, and no notion of time, so rates and timing are better computed directly. A link and the nPrint datasets and benchmarks were posted; running one of those benchmarks through your own pipeline is an acceptable if modest project.
    * **Spurious correlations.** The husky-versus-wolf classifier that turned out to detect snow in the background. In traffic: a model may latch onto the source IP address of the attack in the training set, or more subtly onto TTL, which encodes the number of hops and therefore the topology; the instructor's group once scrubbed addresses and still found the model had overfit to TTL. So representation learning does not remove the need for domain knowledge; you need it to check what the model learned. This is the same explainability concern that appears in fairness contexts (proxy features such as race in lending).
    * A student asked whether nPrint loses information. Not bit for bit, but it drops timing and the relationships between flows and directions. Whether the huge bitmap can be compressed without hurting accuracy is an open question and a good project.
    * A student asked whether nPrint lets you skip feature engineering. Not really: it lets a deep model find which header bits matter, but it cannot express rates, and the spurious-correlation trap remains. A student suggested using a deep model on raw data to discover representations, then encoding those features with netml for a lighter model; the instructor endorsed that as a general technique. Deep models also need more data.
    * Granularity choices: per packet, per flow, per time bin, per N packets; whether burstiness matters, since flow-level aggregation hides sudden rate changes unless bursts are computed first.
    * Privacy and availability constraints: full payloads carry user data even when they are not encrypted; DNS lookups are useful for application identification (the Netflix CDN example) but increasingly hidden by DNS over HTTPS and DNS over TLS, and the server name in the TLS handshake is being encrypted too; IP addresses are both an overfitting risk and, in some jurisdictions, personally identifiable information; TCP sequence numbers are privacy-safe but rarely useful.
    * Bottom line: representation is empirical. Set up the sweep (time bins, netml versus nPrint, dozens of variants) and let coding agents run it overnight; the instructor would like to see projects that automate representation search, since AutoML tools tune models but not representations.
    * Not covered: the encrypted-traffic classifier vignette, the feature-engineering comparison slide (discussed in Q&A instead), and the summary slide.
* **Hands-On 6: Feature extraction with netml** ([notebook](../notebooks/06-Feature-Extraction-Clean.html))
    * About 20 minutes with a neighbor: install netml, load a capture, pcap to flows, flows to features, inspect. Most of it can be copied from the netml documentation. Group work was not captured. A quick check at the end found students with netml installed, flows extracted, and features computed; the room stayed for the full period to finish.
* **Next (Mon Oct 19):** data preparation and representation (chapter 4.1), Hands-On 7.

### Meeting 7 (Mon Oct 19)

*The recording begins at the end of the housekeeping block.*

* **Housekeeping**
    * Assignment 1 is due Friday October 23; the project proposal Monday October 26; the in-class midterm Friday October 30. Add both TAs as collaborators on assignment repositories.
    * This is week four. Friday's meeting, on building the ML pipeline, ends the first half of the course; the second half is about models and their applications. The midterm covers the first half: data acquisition, representation, preparation, and the pipeline. The agenda file is the complete list of what it can draw on. The final is "cumulative only so far as life is cumulative" and will be mostly about the use of models.
    * Slides are in the course GitHub repository as Quarto source and are being revised continually; a legacy PowerPoint version with a few more figures is there too. The instructor will render the decks to PDF and post a zip in Slack rather than committing rendered PDFs.
    * Hands-On 6 was not revisited; nobody asked for more time on it.
* **Where we are in the pipeline**
    * Packets in, a transformation step, a representation step (netml, nPrint), an optional compression step, then model training. Today's first model is trained in the hands-on. Efficient representations and the compression step were flagged again as a project area.
    * Why representation matters: traffic is huge, so representations have to scale; sequential dependencies carry signal (a handshake followed by nothing may be a scan); inter-arrival times are one thing netml captures.
    * Cross-flow dependencies are an open research problem: a page load opens eight or more parallel connections, the TCP connection cannot happen before the DNS lookup, and object downloads cannot start before the index page arrives. Neither netml nor nPrint captures dependencies between flows. Students interested in this were invited to office hours.
    * Representation and model choice are coupled. A student described using an LSTM, which needs a representation with a time index; another reported that a time-series representation did not improve prediction in their application. The instructor's point: if the data has no real temporal structure, forcing it into a time-series representation and model may not help, a theme that returns in the deep-learning lectures.
    * Whether to normalize depends on the model: linear models need it, tree-based models such as random forests generally do not. The choice of model shapes how you represent the data and vice versa.
* **Features as domain knowledge**
    * The best engineered features encode domain knowledge a model on raw data might not discover. For the log4j scan: short connections, small packets, higher IP-address entropy. Verify the intuition in the data, then encode it; this tends to give simpler, more interpretable models.
    * Preview of the deep-learning reading: in its DDoS-detection comparison, nearest-neighbor and tree models perform as well as or better than deep models. Why is a second-half question.
    * Deciding what to encode is science; deciding how is more art. Example from the instructor's spam-detection research: hundreds of adjacent IP addresses all sending mail from one network is a sign of a bot farm, since a real network has one or a handful of mail servers. Encoding "density of mail senders" took five to ten attempts; what worked was the distance to the k nearest other senders. Exploring many encodings of one idea is a good fit for agentic coding tools.
* **Feature operations**
    * One-hot encoding turns a categorical feature into one binary column per value. Ordinal encoding (red, green, blue as 1, 2, 3) is fine for some models but wrong for linear regression, where it implies an order that does not exist.
    * Worked example: protocol as TCP, UDP, ICMP. Coding them 0, 1, 2 is fine for a random forest or other tree model, but a linear model would treat ICMP as twice UDP, so use one-hot. Neural networks also often do better with one-hot inputs.
    * Discretization: bin packets per second in tens, so 15 and 17 fall in the same bin. With domain knowledge, choose the bin edges: a 100-byte threshold separates acknowledgments from data packets; port numbers split into privileged server ports and unprivileged client ports, a natural step function. Binning also adds robustness: the slightly short last packet of a transfer is still "a 1500-byte packet" to the model.
    * Missing values: scikit-learn will impute missing values by default (for example with the column average), which you may not want. Options are to drop the row, impute, or encode "missing" as its own value, which is what nPrint does. A class discussion followed.
        * Students suggested cases: sequence numbers, where a missing value may mean a corrupted packet you want to keep; timestamps, where a missing ACK time is bounded by its neighbors and a negative or zero time is clearly wrong and safely corrected; packet sizes, probably not; using other correlated fields in the same row to fill the gap.
        * The networking twist: unlike a missing pixel in the middle of an ocean, a missing or strange value in traffic may be the anomaly you are trying to detect, so imputing it away can destroy the signal.
        * **Midterm flag:** the instructor said this may become a midterm question: give a case where imputing an average for a missing value is appropriate and a case where it is not.
    * Normalization: min-max scaling maps a feature to [0, 1] by subtracting the minimum and dividing by the range; standardization subtracts the mean and divides by the standard deviation. Needed for linear models for the same reason as one-hot ("blue is not twice green"); the bytes-to-packets regression in next week's linear-models lecture will show it.
    * Transformations: a log transform tames long-tailed network features, such as flow sizes where most flows are short and a few have ten thousand packets. Basis expansion comes with linear regression next week. None of these is a recipe; they are tools to try.
* **Data pitfalls** (covered in about ten minutes)
    * Five failure modes: insufficient data, non-representative data, poor-quality data, spurious correlations, overfitting.
    * Insufficient data: more data generally means better generalization, especially for deep learning (this answers last week's question about whether raw-bitmap representations need a lot of data: yes). In security the problem is rare attack classes; you may have to hunt for or generate examples of a specific scan or exploit. For Assignment 1, the dataset contains no 4K video, so no classifier trained on it can recognize 4K.
    * Non-representative data: the Finnish translation example. Finnish has no gendered pronouns; translating English to Finnish and back produced gender-stereotyped pronouns learned from the training text. Students observed that non-representative data is a special case of insufficient data, which the instructor accepted (either answer would get credit on a check-all-that-apply), and that a dataset can become non-representative as the world changes.
    * Spurious correlations: the husky-and-wolf slide, already covered twice, was not repeated.
    * Cleaning: this is why the hands-ons keep asking for max, min, and mean of every column. Not all outliers are errors; be intentional about removing or keeping them. If timestamps are definitely wrong you must decide whether to drop, fix, or keep the data, and when the data was expensive to collect, "no data, bad data, or patched-up data" is a real trade-off. Paxson's *Strategies for Sound Internet Measurement* was recommended as optional reading; it will not be examined. The instructor plans to give a deliberately messy dataset to clean in a future year.
* **Data leakage and splits** (covered briefly; more on Friday)
    * The golden rule: never train on the test set. It happens by accident: normalizing features before the train/test split leaks test-set statistics into training. The instructor will check an upcoming hands-on on linear models for exactly this mistake. Do the split first, before any other operation on the data.
    * Temporal leakage: shuffling all flows and then splitting puts future traffic in the training set, so the model predicts the past from the future.
    * Arvind Narayanan and colleagues at Princeton (the *AI Snake Oil* authors) went back through published research papers and found these mistakes to be common.
    * Validation set: the training set is further split to hold out a validation set used to tune model parameters. This will matter once models are being trained; details Friday.
    * Overfitting and the bias-variance trade-off, introduced only: a model can be made to fit its training set almost perfectly. A reported 99% accuracy should set off alarm bells. The warning sign is training error that keeps improving while validation error gets worse. The figure from Hastie et al. (test error forms a U against model complexity while training error only falls) was shown at the end of class and has been added to the slides. Much more on this in the coming weeks.
    * Not covered: the planned breakout on auditing a pipeline, the labeling slide, the data-engineering checklist, and the takeaways. The slides on training/validation/test sets and on overfitting were covered only as previews.
* **Hands-On 7: Data preparation** ([notebook](../notebooks/07-Data-Preparation-Clean.html))
    * About fifteen minutes: take the netml features from last time, normalize them, and make a first prediction with scikit-learn. If students need more time, it will be given at the start of Friday's class.
    * A student asked what to do when feature vectors come out with different lengths: truncate to the shorter or pad the longer. There is no single right answer; try both.
    * A student asked whether the assignments follow the hands-on philosophy of using an AI coding assistant for the code. Yes: the point is understanding the data and the models, not syntax, though the instructor noted that dimension-mismatch problems are the kind of thing it was once useful to struggle with.
    * Assignment 1 questions: some features are fingerprints of the particular household where a capture was taken rather than of video quality; a student found accuracy dropped from 70–80% to about 30% when one household's data was held out of training. Removing such proxy features, balancing the classes, and choosing how many examples of each class to train on all affect the errors a model makes, and some models tolerate class imbalance worse than others. The instructor flagged this as an extension to the assignment in a future year and encouraged exploring sampling strategies now.
    * Group work was not captured by the recording, and there was no report-back.
* **Friday:** the ML pipeline, training and evaluation (chapter 4.2–4.3), Hands-On 8. A student will post a question from the problem set in Slack for the instructor to address at the start of class.

### Meeting 8 (Fri Oct 23)

*Plan; replaced with the record after class.*

- Housekeeping: Assignment 1 due today; project proposal due Monday October 26; in-class midterm Friday October 30, covering everything through this meeting; add both TAs to assignment repositories
- Assignment 1 and Hands-On 7 questions
- Lecture 8: Model training and evaluation (chapter 4.2–4.3)
    - Training, validation, and test sets; random versus temporal splits of a traffic trace
    - Overfitting and the bias–variance trade-off; detection and remedies
    - K-fold cross-validation and overfitting to the validation set
    - Evaluation metrics: confusion matrix, accuracy, precision, recall, specificity, F1, ROC and AUC, precision–recall curves under class imbalance
    - Data leakage as an evaluation failure
- Hands-On 8: the ML pipeline, from two captures to a tuned and evaluated model ([notebook](../notebooks/08-ML-Pipeline-Clean.html))
- As time permits: whole-class audit of a plausible intrusion-detection pipeline (carried from Meeting 7)
- Next: probabilistic models (Naive Bayes), Hands-On 9
