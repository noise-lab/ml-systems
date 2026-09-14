## Agenda — Summer 2026 (Paris, September)

Twelve meetings, Aug 31 – Sep 18, 2026. Notes are reconstructed from the
class recordings after each meeting; where a recording was partly
inaudible the entry says so rather than guessing.

### Meeting 1 (Mon Aug 31)

* **Why Machine Learning for Networks? (Introduction, Lecture 1)**
  * (Recording begins partway into the motivation slides)
  * Four properties that make static rules/equations hard to write for networks:
    * System complexity
    * Traffic is (mostly) encrypted
    * Attack patterns constantly evolve
    * Traffic demand patterns shift
  * Definition of learning (Mitchell): a program learns from experience E with respect to task T and performance measure P if performance on T, measured by P, improves with E
    * Experience = training data; task = e.g., detect an attack, identify application traffic; performance = e.g., accuracy
  * Network data available for learning: raw packet traffic, flow records, counters/statistics/metrics, user feedback / QoE signals
  * ML sits between **measurement** (getting data out of the network) and **control** (acting on the inference, e.g., mitigating a detected attack)

* **The Machine Learning Pipeline**
  * Common misconception: "apply a model and you're done" - modeling is one step; most of the work is everything around it
  * Data ingestion: get data off web servers, network devices, firewalls, access points; where to store it, how to format it
  * Data fusion / aggregation / joining across data sets
  * Cleaning: missing values, lost data, corrections
  * Feature creation ("feature engineering") - often manual, benefits from domain expertise
    * Example: representing "volume" for attack detection - bytes/sec? packets/sec? connections/sec, /min, /hour? completed connections?
  * Feature representation: same feature (e.g., bytes/sec) can be continuous, rounded, or thresholded to a binary above/below value
  * Deep learning can learn representations from raw data - can surface feature combinations you wouldn't think of; covered later in the course
  * Training, then deployment: cloud vs. on-premises vs. distributed; send data to the model or push the model to the data
  * Maintenance: model accuracy drifts/degrades over time
  * Most effort and most errors in practice are in data preparation (dirty data, bad labels, missing data) - rarely covered in ML textbooks, but often determines model efficacy

* **Self-Driving Networks**
  * Analogy to self-driving cars; HPE and Juniper now ship products using GenAI to resolve misconfigurations, augment capacity, etc.
  * Research idea from ~10 years ago now coming to fruition; course focuses on the building blocks, not fully autonomous networks
  * Open question, as in any AI application: how much human in the loop? how much do you check the outputs? (same question applies to using agents on homework)

* **Three Application Areas (preview of Lectures 2-4)**
  * Security: DDoS detection, botnets, malware-infected devices (from their traffic), malicious email
  * Performance inference: quality of experience (QoE) for streaming (resolution, frame rate, stalls/rebuffering, startup delay), rebuffer prediction, application identification from encrypted traffic, congestion prediction
  * Resource management: traffic engineering (shifting traffic across links to relieve congestion), capacity provisioning (e.g., COVID-era demand growth), load balancing, cache placement

* **Brief History of ML in Networking**
  * Late 1990s: spam filters - first statistical classifiers on network data
  * 2000s: botnets emerge; ML for DDoS/botnet detection; mostly **offline** (capture trace to disk, then train/classify)
    * Most exercises in this class are offline classification on a captured trace
  * Late 2000s-2010s: software-defined networking (SDN) / OpenFlow let software programs control the network; fine-grained real-time measurement followed
    * Enabled closed-loop control using ML
  * Today: still mostly offline detection/prediction and low-level data collection; active/open areas: real-time coupled inference and control, inferring higher-level application properties from encrypted traffic, scalable distributed real-time control, QoE inference; early self-driving network deployments; agentic AI

* **Course Learning Objectives**
  * Know what ML/AI is and isn't good at - is this the right tool? often you don't need AI
  * Apply an end-to-end pipeline: data acquisition through training, evaluation, deployment (assignments provide traces; project may involve collecting data)
  * Select appropriate models and representations for network data (when linear vs. logistic regression vs. deep learning)
  * Practical pitfalls: overfitting, incomplete/inaccurate labels, concept/model drift, deployment cost (storage, traffic volume, system cost), privacy (balancing accuracy against privacy requirements)
  * Think of this as an applied ML course whose data happens to be networking; skills transfer to any domain

* **Course Structure**
  * Topic list: ~18 modules for a standard 9-week term; only 12 meetings here, so one to two modules per meeting; aim to reach at least module 16, hopefully generative models
  * Order: applications (security today; performance and resource management tomorrow), generative models / synthetic data, data collection and feature extraction, supervised and unsupervised learning (supervised = labeled data), deployment pitfalls (drift, adversarial evasion), privacy
  * Each module has a reading; do it before class. For tomorrow: through end of Chapter 2. Come with questions; each class will open with Q&A
  * In-class hands-on activities for each module - not graded but exam questions will draw on them; formerly hard to finish in class, now feasible with agentic coding tools
  * Class is recorded; instructor uses transcripts to update the agenda contemporaneously and to fold examples into the book
  * Exam prep: past exams (with solutions) and the instructor's exam-generation prompt are in the repo; students can generate their own practice exams from the agenda + past exams

* **Logistics and Tools**
  * Slack invite link on Canvas works (confirmed in class); instructor posted hello in #general; post questions publicly so others can answer - no TA for this class
  * Email (feamster@uchicago.edu) reaches instructor faster than Slack if urgent
  * Canvas form to submit GitHub ID; create your own GitHub repository for the course and fill out the form
  * Gradescope only for the exam
  * Clone the course repository: contains slides (some still older PowerPoint; being migrated), notebooks/hands-ons, past midterms and finals with solutions, exam prompts; ignore the stale assignments and old book copy in it
  * Meeting time: today started late (11:30) so ran to ~1:45 with a break; normal slot is 9:30-~11:50

* **Grading Components**
  * Exam: no midterm; one combined final on the last day; one 8.5x11 cheat sheet, both sides; time will be left essentially unbounded ("we have all day")
    * Student question about extended-time accommodation: applies to the in-class exam; take-home midterm from prior offering is dropped
  * Assignments: only two (not four or five) because of the short term
    * Assignment 1 posted today (link in Slack #general); due Monday Sep 7 at class time; clone the assignment repo and check work into your own repo
    * Assignment 2 out next Monday, due the following Monday
    * Hands-on 1 gives experience with the data set that Assignment 1 builds on
  * Project (third week): apply ML to a network problem on network data
    * Options: leaderboard-style tasks (link posted in Slack, includes data sets), extending traffic-processing libraries, or anything you want; instructor can point to data sets and past projects
    * Groups of three, maybe four; with four, clearly define who does what
    * One-pager due by end of this week (Friday), checked into your course repo (one per group, all names): title, problem and why it matters, data, candidate models, evaluation, why you picked it / what you expect to learn
    * Deliverables relaxed: runnable code that reproduces results plus a write-up; no required notebook or format
    * If it runs, reproduces, is understandable, and is a sound application of ML to network data, that's an A
  * Participation: show up, stay engaged; no reading responses, no in-class quizzes
  * Late policy: no extensions, but 96 late hours (4 days, rounded to the hour) to spend as you wish across the term; lateness measured from commit timestamps

* **Integrity and AI Policy**
  * Goal is learning, not ranking; skipping the work only robs you
  * Acknowledge collaborators and tools used; working together is fine if you personally understand the result - exams will ask about the assignments
  * Bright line: copying verbatim (from people or online) without understanding
  * "Could you explain what you did and why?" is the test; instructor is not interested in catching people, integrity is how you behave when no one is looking; come talk if grade stress is high
  * AI-assisted coding strongly encouraged for everything, especially hands-ons; writing code by hand is "nonsense" now
  * But: understand the output well enough to evaluate it (analogy: generating an essay on Eastern philosophy you couldn't judge); exam may ask what generated code does
  * Welcoming environment; no question is too basic

* **Hands-On 1: Packet Capture Basics**
  * Last ~45 minutes used for hands-on instead of the security lecture (security moves to tomorrow; modules 2-4 planned for Tue)
  * Notebook: `01-Packet-Capture-Basics` in the course repository (clone the whole repo); students located it and started working
  * Instructor circulated to help with setup (audio mostly inaudible)

### Meeting 2 (Tue Sep 1)

* **Logistics**
  * Going forward, instructor will post each day's agenda in Slack
  * All slides checked into the repo last night in both formats (Quarto and legacy PowerPoint)
  * Introductions done in class (instructor learning names)

* **Security Applications (Lecture 2)**
  * Topics for the module: spam and phishing detection, intrusion/malware detection, DNS abuse, anomaly detection
  * DNS primer (few students have taken a networking course; none assumed): maps names like uchicago.edu or google.com to IP addresses; can itself be attacked and abused
  * Anomaly detection: anything out of the ordinary - could be an attack or a system failure; feed traffic or system logs into a model and flag deviations
  * Two directions at the ML/security intersection:
    * ML *for* security (this course)
    * Security *of* ML - adversarial ML: evasion, poisoning, membership inference; covered in a separate UChicago course, not here

* **Why ML for Security Is Hard**
  * Class imbalance and label scarcity: need enough examples of each class (spam vs. ham, DDoS vs. normal web traffic); Google has Gmail-scale labeled data, we don't
    * Foreshadowing week 3: generative AI / LLM-based synthetic data is becoming important precisely because good labeled security data is scarce
  * Concept drift (a.k.a. model drift): adversaries adapt to the detector
    * Toy example: rule "more than 100 emails/minute = spammer" is defeated by spreading sends out; yesterday's labeled attacks may not look like tomorrow's
  * Heterogeneous data: packet captures (seen yesterday in Wireshark), flow aggregates (coming days), DNS-only subsets, who-talks-to-whom graph structure (e.g., sudden traffic from Eastern Europe); off-the-shelf Python models expect one flat table with labels, so data shape is a real problem
  * Offline vs. real time: 20 years ago detection was offline (capture, analyze, then respond); real-time decisions are increasingly necessary, and AI-enabled attackers move laterally fast (e.g., compromised Duo login leads into Canvas)

* **Pipeline Recap in the Security Context**
  * Gather data -> model performs prediction/inference -> take action (block, allow, alert)
  * Inputs: flow summaries, DNS lookups, BGP routing changes (how paths through the network change; to be covered later), packet payloads

* **Case Study: Email Spam Detection**
  * One of the earliest ML applications anywhere; first classifier was Naive Bayes on message content (a notebook in the repo does this; assigned)
  * Content filters look at words (Rolex, Viagra vs. homework, assignment) and were quickly evaded by obfuscation:
    * White-on-white text (Shakespeare excerpts padding an ad)
    * Content encoded as images or PDFs
    * Unicode look-alike characters
    * Today: AI-generated text
  * IP-address blacklists: defeated because spammers rotate IPs and compromise new machines, so senders are constantly unseen before
  * Behavioral / network-level features (instructor's early research, ~20 years ago): signals that persist even when content and sender addresses change
    * High volume, coordinated similar messages from many places, time of day, message size
    * Ephemeral BGP routes that appear and disappear quickly
    * Rapidly changing name-to-IP mappings ("fast flux"): uchicago.edu should resolve the same today as yesterday
    * Bursts of email, sends at unusual hours, many recipient domains
    * Connections per sender, recipients per connection, session duration
    * Large coordinated campaigns look unlike individual senders (newsletters are a gray area)
  * Feature engineering questions raised: how to represent "rapid route change" or "mapping change" - changes per hour? per minute? how far did it move?
  * System built: SNARE (Spatiotemporal Network-level Automatic Reputation Engine); deployed commercially
  * Model used: a modified decision tree, essentially random-forest-like; trees, ensembles, and random forests will be covered (Lecture 12); optional readings can be posted

* **Predicting Campaigns Before They Launch: Domain Registration Signals**
  * Campaigns need a link to click, which needs a domain, which needs a registration - registrars see the behavior
  * Bulk registration of many cheap, throwaway domains at once
  * Batches of related names for a specific campaign
  * Drop-catching: grabbing domains the instant they expire to inherit their good reputation
  * These features can detect/predict phishing and spam campaigns ahead of time (instructor's later work)
  * Student question: is drop-catching an expired domain illegal? Instructor: not technically, anyone can buy it
  * Student question: who sells domains / can the registrar see bulk purchases? Registrars (e.g., GoDaddy and similar) do see purchases; unknown whether they run detection on suspicious bulk buys; instructor will look into it (hard to imagine a legitimate use case for buying a thousand at once)

* **Hands-On**
  * Class introductions followed by continuing the notebook work (Hands-On 1: Packet Capture / Hands-On 2: Security), instructor walking through it in person; audio not captured

### Meeting 3 (Wed Sep 2)

*Recording note: audio is usable only for the first few minutes and roughly minutes 19–35 of the 105-minute session; the video bitrate adaptation discussion in between and the second half are not recoverable.*

* **Recap and Framing**
  * Covered so far: ML for security, then ML for performance (diagnostics, troubleshooting, inference)
  * Performance material leads into Assignment 1 (inference of application quality)
  * Today: third big application area, ML for resource allocation (Lecture 4, Resource Optimization)

* **What Is Resource Allocation?**
  * Networks have limited capacity; the resource is usually throughput the network can support
  * Can also mean deploying additional servers or caches to serve content
  * Two timescales to think about
    * Short-term, real-time: how much to send, where to send it, at what quality
      * Example: video player deciding whether to send a full HD/4K stream or a lower-quality stream the network can actually support
    * Long-term: whether to deploy a new data center
    * In between: add capacity, servers, or bandwidth to an existing data center

* **Short-Term Allocation: Early ML Approach**
  * Early paper applying ML to short-term resource allocation in the network
  * Mechanism: offline ML over observed features and decisions, compiled into a lookup table the system consults at runtime (like caching classifier outputs)
  * Suggested project area: survey what has happened since; likely feasible to do this kind of decision in real time now

* **Longer Timescales: What-If (Counterfactual) Analysis**
  * Question: if I change the configuration (add servers, increase capacity), what happens to user experience?
  * Instructor's early work with Google on search response-time distribution
    * Search is a complex multi-tier system: front end returns static content (logo, stylesheet); back end performs the lookup
    * Systems too complex to simulate or model analytically
    * Lots of existing data on searches and response times under different conditions
    * Learn response time as a function of features from that data, then change the features (e.g., deploying a server lowers round-trip time) and predict the new response
  * What-if scenario evaluator was a fairly simple regression model
  * Must learn the dependency structure between variables: changing one input (e.g., RTT) may change other correlated inputs; need to propagate changes through the model
  * Use case: planned maintenance on a front-end server; evaluator predicts whether redirecting users to the front-end server in Taiwan is acceptable (answer: yes)
  * Will return to how such models are evaluated when covering model evaluation

* **Capacity Provisioning**
  * How much capacity to deploy; very long timescale (months to a year, e.g., data center buildout)
  * ISPs decide how much capacity to purchase, where to install it, how to route traffic
  * Example of a period when ISPs had to act much faster than normal planning cycles; classic long-term resource allocation problem

* **Model Maintenance (Preview)**
  * Full module later in the course
  * Classic issue in networked systems: detect when a model is no longer accurate, then retrain

### Meeting 4 (Thu Sep 3)

*Recording note: audio is intact for the first ~38 minutes only; the netml hands-on, the Lecture 7 material, and later Q&A are not captured.*

* **ML Pipeline Recap**
  * Every ML system for this kind of data follows the same arc: collect raw traffic/measurements, represent as a feature matrix, train on features + labels (target variable), deploy, then keep re-measuring to evaluate the model over time
  * Today's focus: the representation step (Lecture 6, From Data to Analysis / Feature Extraction)
  * Representation is often skipped or treated casually, but the choice is very important and genuinely difficult

* **Why Representation Is Hard**
  * Raw packet captures are very large and unstructured (just bytes); most ML models want structured data
  * Many possible transformations: summarize, aggregate, sample, compress, extract subsets
  * No single best representation; instructor's earlier research on this question produced the netML paper
  * Best representation depends on the task (e.g., performance diagnosis vs. anomaly detection)
  * Encryption hides contents, so inference works on metadata: packet sizes, timings, arrival times; indirect features at best

* **Building-Block Metrics**
  * Throughput: data transferred per unit time; measured actively or passively
  * Latency: time for packets to reach the destination; typically active, but can be inferred passively from acknowledgment timing
  * Jitter: variation in inter-packet delay
  * Loss: fraction of packets dropped; active (ping, speed test) or passive (count retransmissions in a pcap)
  * Flow statistics: aggregates over packets in a flow

* **From Packets to Features**
  * Raw packets parsed into 5-tuple flows; netML recommended (other libraries exist)
  * Flows become feature vectors, i.e., rows of the X matrix
  * With labels: supervised learning; without labels: unsupervised learning / clustering; both covered later

* **Statistical Flow Features (netML STATS)**
  * Flow duration, packets per second, bytes per second
  * Packet-size statistics per flow: mean, standard deviation, interquartile range (25th/75th percentile), max
  * Simple, much smaller than a full trace, works well for many problems; evaluated for anomaly detection and found sufficient
  * Other netML representations
    * Inter-arrival time series between consecutive packets; recovers timing that industry-standard flow records throw away
    * Windows defined by byte count (e.g., every 100,000 bytes; 1,500 bytes is only one packet) or by packet count, not just by fixed duration
    * Byte/packet windows can be short or long in wall-clock time depending on sending rate

* **Which Representation for Which Task (survey from instructor's paper)**
  * Different tasks/papers use duration, inter-arrival time, packet-size vectors, frequency-domain features (FFT over the flow's time series; one DoS-detection paper), fixed-size windows
  * Takeaway: no single representation dominates; duration and inter-arrival time each appear in roughly half
  * Student question: why do papers pick just one representation? Often the paper's purpose is to evaluate a specific feature set (e.g., frequency domain); in practice consider more than one
  * Choosing a representation is a design decision that depends on where the discriminative information is for the task
  * Example: real-time apps (gaming, voice, video) send evenly spaced packets; high inter-arrival variation suggests degradation or anomaly, so inter-arrival time is a natural feature there
  * Domain knowledge about the task is one of the most important inputs to representation
  * Findings: frequency-domain representations typically do not beat simpler statistics (good news, since they are cheaper); payloads do not help much

* **Encryption and What Remains Visible**
  * Port numbers used to identify applications (OS uses them to demultiplex); now almost everything runs over port 443 (HTTPS), so ports and deep packet inspection are largely blind
  * Still visible: packet sizes, timings, IP header fields (source/destination address, TTL, DSCP prioritization bits), some connection-setup fields, flow duration
  * TLS 1.3 and Encrypted Client Hello are hiding handshake fields such as the server name indication that used to be visible
  * Trend: increasingly forced to rely on metadata for inference

* **Handcrafted Features vs. Learned Representations**
  * Handcrafted features need domain expertise (e.g., knowing voice traffic is evenly spaced)
  * Hard to reproduce from papers: "bytes per second" could be discretized/bucketed many ways
  * Changing the task may require an entirely different feature set; motivates a representation generated once and reused across tasks
  * Alternative: let ML learn the representation (deep learning, covered next week)

* **nPrint**
  * Motivation: if deep learning classifies images from pixels, why not packets from bits?
  * Each packet becomes a fixed-width bitmap aligned to protocol field boundaries
  * Problem: packets and headers vary in size, so bit offset 100 might be an IP address in one packet and TTL in another
  * Solution: canonical header layout padded to maximum size; absent fields (e.g., IP options) filled with -1
  * Cost: representation roughly doubles in size (each bit needs to encode three possible values)
  * Workflow: run nPrint on a pcap (e.g., IPv4 option), output CSV with one row per packet and hundreds of bit columns, feed directly to any classifier
  * Task-agnostic and deterministic: identical inputs give identical outputs
  * Feature-importance heatmap over the bitmap shows which header bits the model relies on (e.g., source address lighting up)
  * Limitations: twice the storage; no temporal relationship between packets encoded by default; can learn spurious correlations

* **Spurious Correlations**
  * Husky vs. wolf image classifier: perfect accuracy, but feature analysis showed it keyed on snow in the background; it was a snow detector
  * Network analog: attack traffic from one IP address; model learns the source-address bits, which is useless when the next attack comes from elsewhere
  * Source IP address is generally a poor feature; will revisit
  * Fundamental tradeoff: handcrafting requires domain knowledge up front; representation learning skips that step but still needs domain knowledge to validate what was learned

* **Tradeoff Summary**
  * Handcrafted: domain knowledge drives design; low dimensionality, faster training; interpretable by construction; you control what the model uses
    * Downsides: often task-specific; you can miss patterns you did not think to encode
    * Student question (from Wednesday's break): how do you know which features to feed the what-if evaluator? Enumerate as many as you can, keep the useful ones; risk is forgetting one
    * LLMs / generative AI increasingly used to assist feature engineering, sharply lowering the cost of handcrafting
  * Learned: model discovers structure from bits; high dimensionality, more compute; reusable across tasks; can find patterns and nonlinear combinations you would never find by hand
    * Still requires feature-importance analysis and validation

* **Aggregation Time Scale**
  * Statistics every 5 minutes vs. every 5 seconds: coarser is more compact and faster to process but misses short impulses and variations
  * Not just per-packet vs. aggregated; which time bin to use requires exploration; no one right answer
  * Considerations: what timescale the phenomenon operates on, acceptable storage and compute, whether the feature is visible at all in encrypted traffic, whether it will generalize

* **Informative but Often Inaccessible Features**
  * Full payloads: privacy and legal issues, plus encryption
  * DNS query names: very useful for malware detection (bulk-registered, non-human-readable domains) but increasingly encrypted
  * Server name indication and certificate names: increasingly encrypted
  * IP addresses: overfitting risk (behavior tied to an address in training may come from elsewhere next time); also considered personally identifiable information in many jurisdictions (ongoing debate)
  * Sequence numbers and similar fields: observable but rarely useful

* **Lab Preview**
  * Decide the unit of representation: per packet, per flow, or per time window
  * Choose the feature representation (netML options or a raw bitmap)
  * Validate that you can actually obtain the features, then iterate; no single right answer
  * Libraries have tunable parameters; project idea: use agentic AI to explore parameter and feature spaces (previously a manual, painstaking process; no project has done this yet)

* **Hands-On: netML, pcap to trained classifier**
  * Use netML to go from a pcap to a trained classifier
  * Documentation examples can nearly be copy-pasted; instructor curious whether Claude Code or similar can generate it
  * Important hands-on: the assignment, projects, and most later hands-ons build on this pipeline, so full time given to finish it
  * Observation: students now finish hands-ons thanks to AI coding agents; previously nobody finished because of grinding on Python syntax

* **Logistics / Plan for the Session**
  * Hands-on from about 10:20 until 11:00, then optional short break
  * Then Lecture 7, Data Preparation and Representation
  * Following hands-on uses netML on the Log4j data
  * May start Lecture 8 (Model Training and Evaluation) if time permits

### Meeting 5 (Mon Sep 7)

* **Logistics and Announcements**
  * Debrief on Friday's site visit: positive feedback; first time running the Paris version of the course, feedback welcome on excursions
  * Excursion to Huawei labs tomorrow (Tue) morning overlaps class time
    * Polled class on holding class in the afternoon instead; mixed feelings
    * Instructor inclination: don't cram more material, but only 12 meetings total; decision by tonight
  * Assignment 1 due tonight
    * Midnight Chicago time is acceptable ("it's the University of Chicago after all")
    * Next assignment will be released by the time this one is due
  * Assignment 1 Q&A
    * Student question: how high does accuracy need to be? Data split is in-distribution, so it should be achievable to get high accuracy; most important thing is running the whole pipeline end to end and evaluating
    * Broken data set link fixed Thursday; some students may have an older version of the assignment; link to be re-posted at the break
  * Course now at Lecture 8 (ML Pipeline); moving fast

* **ML Pipeline Recap: Where We Are**
  * Data engineering covered: understanding and cleaning a data set, labeling examples, train/validation/test splits (students doing this in Assignment 1)
  * Not yet covered: model training and model evaluation - focus of today

* **Understanding the Data (before training anything)**
  * How much data is there? What is the business need / what will you do with the output?
  * Summary statistics: min, max, mean, standard deviation; look for outliers
  * Sanity-check data against domain knowledge
    * Packet sizes should be in the valid range
    * Port numbers are 16 bits; a column outside that range means the data set is wrong
  * Look at feature distributions to guide feature design
    * Example: web/gaming traffic tends to have smaller, more regular packets; downloads have larger packets

* **Data Cleaning**
  * Look for impossible values: time going backwards, packet sizes larger than the maximum, values that can't occur
  * Outliers that are errors: the model may try to fit them, so it is no longer modeling the real phenomenon
  * Domain knowledge about the system that generated the data is the key to spotting errors
  * Principle applies beyond networking: understand the data before throwing it at a model

* **Labeling**
  * Supervised learning requires labeled examples: outcome vector Y
  * Regression labels are continuous: e.g., application latency, search response time (from earlier lecture)
  * Classification labels are categorical, e.g., binary

* **Train / Validation / Test Split**
  * Goal: know how the model performs on data it hasn't seen, when you have only one data set
  * **Lock the test set away first** - before normalization, before any training
    * Anything that touches model training must not use the test set; otherwise you are effectively training on the test set
  * Training set: learn model parameters
  * Validation set: tune hyperparameters (the "dials and knobs")
    * Examples: depth of a decision tree, degree of a polynomial in regression
    * Typically a smaller fraction carved out of the training set
  * Student question: how to split when data has structure (e.g., flows, temporal dependency)?
    * Not typically an issue for the data in this course, but matters when later data depends on earlier data
    * Random splits can leak future information into predictions of the past
    * Temporal split (train on earlier, test on later) is appropriate for forecasting problems
    * Example from earlier lecture: predicting traffic volume growth during COVID from the first months of the year

* **Overfitting and the Bias-Variance Tradeoff**
  * Board example: fitting points with a straight line (red) vs. a high-degree wiggly polynomial (green)
    * Higher model complexity -> lower error on the training set (can fit it perfectly)
    * On held-out test points, the straight line does better; the wiggle doesn't generalize
  * Definitions
    * Variance: how much the model changes with small changes in the training data
    * Bias: error on the training set
    * High-complexity model = high variance, low bias; simple model = low variance, high bias
    * Looking for the sweet spot in between
  * Finding it: plot training error and validation error vs. complexity; when validation error starts diverging from training error, stop
  * Remedies
    * More training data (not always available)
    * Stop training / stop refining the model early (common in deep learning)
    * Regularization: penalties on parameter sizes (not perfect)
    * Ensemble methods (e.g., Random Forests, later in course): combine many slightly different models

* **Cross-Validation**
  * Concern: tuning hyperparameters to one particular validation split
  * Solution: chop the training set into folds, rotate which fold is validation, average
  * scikit-learn has built-in cross-validation functions
  * Especially useful when the validation set is small
  * Student question -> curse of dimensionality
    * More features = higher-dimensional feature space; points become sparse
    * Need more data to cover the space
    * Motivates dimensionality reduction, but with a tradeoff in information lost

* **Model Evaluation**
  * Confusion matrix
    * Diagonal = correct predictions; off-diagonal = errors
    * Generalizes to multi-class
    * Two types of errors in binary classification (false positives, false negatives)
  * **Accuracy pitfall with rare positives**
    * Attacks are rare: a classifier that always says "no" needs no training and is 99.9% accurate
    * Never detects an attack; accuracy alone hides this
  * Precision: of the positive predictions, how many were correct? (TP / all predicted positive)
  * Recall (detection rate): of the actual positives, how many did we catch? (TP / (TP + FN))
  * Specificity: of the actual negatives, how many were correctly identified as negative
  * F1 score: harmonic mean of precision and recall; single number, higher is better; scikit-learn computes it
  * Precision-recall curve: want high precision and high recall (curve bowed toward top-right)
  * ROC curve: detection rate vs. false positive rate
    * Ideal is up and to the left
    * Area under the curve (AUC): close to 1 is good; a 45-degree diagonal is a random classifier
    * A classifier below the diagonal can be flipped to get one above it
  * When to use which: with imbalanced classes (spam, attacks), precision-recall is more informative
  * **Thresholds and operating points**
    * No single correct operating point; the classifier exercises a tradeoff
    * Tune threshold toward catching all attacks -> more false positives
    * Tune threshold toward avoiding false positives (spam filter: don't want job offers in the trash) -> more spam gets through
    * Where ML theory meets operations: pick the point acceptable for the application
    * Likely exam question: give an example where you'd accept lower detection for a very low false-positive rate, and one where the reverse holds
  * Data leakage: briefly noted; covered previously

* **Hands-On Activity #8: Complete ML Pipeline**
  * Data set: HTTP / Log4j traffic (used in earlier hands-on)
  * Reuse earlier parsing/assembly work; train and test a binary classifier
  * Any model is fine - specific models not covered yet
  * Compute evaluation metrics including ROC / AUC; precision-recall not required but encouraged
  * Work in existing partner groups; check-in at 11:00, then break

* **Preview**
  * Supervised learning models start next (after the break / next session)

### Meeting 6 (Wed Sep 9)

* **Linear Regression (Lecture 10)**
  * Most students have seen it (econ, data science); in ML it is used to predict, not to explain coefficients
  * Regression predicts a continuous value (not classification)
  * Setup: points (x, y) where x is typically a feature vector; learn a line/hyperplane
    * Slope plus intercept from grade school; multi-dimensional: learn a weight per feature plus intercept
  * Inputs: quantitative features such as packet counts, rates, round-trip times
    * Hands-on example: predict throughput from packet counts (packets have roughly fixed size)
    * Inputs can be transformed: log, square, square root, etc.
  * Board example: number of people in the park vs. temperature (Celsius, "we're in Europe")
    * Red dotted lines are residuals / prediction errors
    * Goal: learn the line that minimizes the residual sum of squares (least squares)
    * Closed-form solution; straightforward calculus - slides on the derivation skipped
  * **The acknowledgment-packet problem** (from the book)
    * Predicting bytes from packet count: data packets (~1500 bytes) and ACKs (small) form two clusters
    * Squared error amplifies large residuals, so the fit chases the data packets and ignores ACKs
    * Example of why models assuming a single mode can fail; worth thinking about in the hands-on
  * Reminder: split the data set before doing any normalization

* **Basis Expansion / Polynomial Regression**
  * Motivation: queue depth vs. delay - delay grows non-linearly as drops and retransmits kick in
  * Idea: add new features that are functions of the original ones (x, x^2, x^3, ...) and keep using a linear model
  * Common bases: polynomial; radial basis functions
    * Radial basis + linear regression was used in the search-response-time "what-if" work from an earlier lecture
  * Training is unchanged (still minimizing residual sum of squares)
  * Polynomial degree is a hyperparameter tuned on the validation set; higher degree fits training data better but risks overfitting

* **Regularization**
  * Add a penalty on the weights to the residual sum of squares, scaled by a hyperparameter (lambda; called alpha/C in scikit-learn)
    * Ridge regression: penalty is sum of squared weights
    * Lasso regression: penalty is sum of absolute values of weights
  * Lambda = 0: ordinary linear regression, all features considered, higher variance
  * Higher lambda: pushes weights toward zero, fewer non-zero coefficients, less complex model, lower variance, higher bias
  * Likely midterm question: "to get a higher-complexity model, do you turn lambda up or down?" (down)
  * No science to choosing ridge vs. lasso; try both

* **Linear Regression in Practice / Summary**
  * Paper example: predicting cellular throughput and latency from KPIs (key performance indicators from base stations: channel band, arrival time, deployment, signal strength, dropped calls); linear regression used as baseline
  * Closed-form optimum: one of the few ML models solved exactly rather than approximately
  * Easy to get feature importance from coefficients
  * Watch feature scaling: features on larger scales dominate squared error
  * Simple, often works well; good baseline for projects

* **Logistic Regression (Lecture 11)**
  * Classification predicts categories: malicious vs. benign flow, DNS query vs. response (hands-on example)
  * Why not linear regression? Output is unbounded; want a value in [0, 1]
  * Apply the sigmoid function to the linear combination of features: gives P(y = 1 | x)
  * Predict class 1 if output > 0.5, else 0
  * Works well when there is a clear, linearly separable decision boundary in feature space
    * DNS packet size: small responses vs. larger queries look like a sigmoid
    * Works in multiple dimensions too
  * Non-separable data needs basis expansion, a tree-based model, or a neural net
  * Suggested extension: classify data vs. acknowledgment packets with logistic regression using the linear-regression hands-on data
  * Paper example: logistic regression as a baseline for denial-of-service detection - less accurate than tree models but much faster
  * Advantages: very fast inference, interpretable coefficients, convex optimization converges quickly
  * Training details skipped (stepwise optimization)
  * Project advice: linear/logistic models are always a good baseline to compare a fancy model against

* **Hands-On: Linear Regression with netml**
  * Required: simple linear regression on netml features, and polynomial basis expansion
  * Part 3 and bonus optional
  * Logistic regression hands-on intended as a follow-on, but the notebook couldn't be located; skipped
  * Roughly half an hour, then 5 extra minutes, then break until 11:10

* **Decision Trees (Lecture 12)**
  * Most students have seen trees and Random Forests; treated as review
  * Board example: application classification
    * packets/sec > 50? -> destination port 443 (HTTPS)? -> duration < 1 second? -> leaf nodes with predictions
    * Every root-to-leaf path is a decision rule
  * Advantages
    * Simple and readable; each decision can be traced and explained
    * Handles numerical and categorical features; no feature scaling needed (unlike linear models - no distance computations)
    * Works for classification (majority class in a region) or regression (mean/median of values in the region)
  * **How splits are chosen**
    * Two questions at each node: which feature to split on, and what threshold
    * Board example with two features (x1, x2), red = attack, green = benign
    * Pick the feature and threshold that best separate the classes; quantify with impurity (entropy or Gini)
    * Pure region (all one class) has entropy 0; 50/50 has maximum entropy
    * Greedy, recursive: choose the split that most reduces impurity, repeat in each region
    * Possible midterm question: given a tree, partition the feature space, or vice versa
  * **Overfitting and tree depth**
    * Continuing to subdivide can classify every training point perfectly (memorizing the training data) but generalizes poorly
    * Hyperparameter: tree depth
    * Cost-complexity pruning (preferred): grow all the way down, then prune back using error plus a penalty on depth
    * Early stopping (stop when a split doesn't reduce impurity much) can miss a later split that helps a lot
  * **Brittleness / high variance**
    * Removing or adding a single training point can change the entire tree structure
    * Feature importance from impurity reduction is intuitive but can be unreliable for the same reason
    * Single trees often don't perform that well in practice

* **Ensembles: Bagging and Random Forests**
  * Two families: bootstrap aggregation (bagging) and boosting; only bagging covered today
  * Bootstrap aggregation
    * Sample the training set D with replacement (balls-in-a-bin analogy) to make B bootstrap samples, each with repeats and omissions
    * Train one tree per sample
    * Combine: majority vote for classification (use an odd number of trees), average for regression
  * Random Forest adds one more randomization: at each split, consider only a random subset of features
    * Seems suboptimal but decorrelates the trees so they aren't all making the same decisions
    * Student question: clarified that bootstrap sampling is on the data, feature subsetting happens within each tree's splits
  * Properties
    * Hundreds of trees; voting makes predictions stable and less sensitive to single training examples
    * Little tuning needed; more trees is almost always better; overfitting is less of a concern
    * Mixed feature types fine; training is highly parallelizable (trees are independent)
    * Very strong on tabular data like most networking data
  * Instructor anecdote: highly cited "deep learning" IoT anomaly-detection paper where k-nearest neighbors and Random Forest performed as well as deep learning
  * Project advice: Random Forest is a really good baseline - simple, efficient, interpretable, feature importance for free

* **Course Plan and Announcements**
  * Tomorrow: possibly two hands-ons, trees/ensembles and deep learning
  * Friday: LLMs
  * Next week: unsupervised learning and generative models
  * Social: climbing tomorrow night (details to be posted in the random channel, not official channels); instructor playing an open mic at a pub in the 6th tonight; music session with students sometime next week

### Meeting 7 (Thu Sep 10)

* **Logistics and Announcements**
  * Exam: single exam for the course, in class next Thursday (Sep 17)
    * Designed for ~45-50 minutes (about 30% longer than the on-campus 30-40 minute design, since there is only one exam)
    * Full class period (2h15) available; no hard cutoff if more time is needed
    * Past exams: min ~30 minutes, median ~45 minutes, long tail of a few students staying the whole time
  * Project deadline: Sunday Sep 20, 11:59 pm Chicago time
    * Class vote was roughly split between "finish before leaving Paris" and "more time"
    * Submissions are Git/Canvas and timestamped, so updates can be pushed after the deadline; instructor will look at updates
    * Hard limit is ~one day before the grade-submission deadline (instructor to confirm the date)
  * Project show-and-tell next Wednesday (Sep 16): informal go-around, no formal presentation required, optional slide or demo
  * No third assignment (class vote); course is compressed to three weeks
    * Optional unsupervised-learning assignments (clustering, spatial clustering, anomaly detection) to be posted in Slack #random for anyone who wants them
    * Generative AI material added last year; will be covered early next week
  * Agenda file of all topics covered to be generated from lecture transcripts and posted
  * Exam generation process: instructor drafts the exam with Claude Code from the agenda notes and past exams, then edits by hand
    * Prompt is (or will be) in the course repo; students can run the same process to generate unlimited practice questions
    * Idea credited to the dean of the Harris School (daughter used ChatGPT on course notes for practice questions)
  * Assignment 1 graded: rubric pushed back to each repo; nearly everyone received full credit
    * Written feedback to be pushed as GitHub issues on each repo; scores sent via Slack
    * Reminder: acknowledge any AI use in notebooks; several submissions appeared to use AI without acknowledging it
    * Voluntary request: share the prompts you used (e.g., as a separate file in the assignment repo); does not affect grade; helps instructor understand tool use (Claude Code vs. Cursor vs. IDE plugins) for future course design
    * Instructor note: Claude Code saves all prompts and responses, easy to export; IDE plugin history harder to export
  * Instructor's grading pipeline: replaced GitHub Classroom/Canvas with Git + spreadsheet + Claude; Slack MCP server for score notifications; project-partner form used to link repos to gradebook
  * Lyon trip tomorrow (Fri Sep 11): early pickup (~6:30), meet Francesco Bronzino, who co-wrote the video QoE assignment as a postdoc
  * Plan for today: trees/ensembles hands-on (skipped last time for lack of time), then deep learning lecture, break, deep learning hands-on
  * nPrint hands-on will not be done separately; nPrint already covered, so lectures 13 and 14 combined

* **Hands-On Activity #12: Trees and Ensembles (IoT Activity Classification)**
  * Data set: encrypted Nest Cam traffic captured in a home; classify device activity (e.g., motion detected) from traffic volume
  * Context paper: "A Smart Home is No Castle: Privacy Vulnerabilities of Encrypted IoT Traffic" (posted in Slack, optional reading)
    * Adversary identifies devices via DNS queries, then send/receive rates reveal user interactions
    * Figure 2C: motion detection events visible as spikes in Nest Cam traffic rate
    * Hands-on reproduces the traffic-rate part only (device already isolated)
  * Student question: is this attack still possible today?
    * Follow-up paper: "Closing the Blinds: Four Strategies for Protecting Smart Home Privacy" — traffic shaping / injecting traffic to limit an observer's confidence
    * Think about defenses: smooth out spikes, inject fake spikes
  * Example is contrived (could nearly draw a line by hand); point is to get facile with the data
  * Notebook formerly used k-nearest neighbors; changed to random forest; labels are provided
  * Data cleaning clarifications:
    * Filter to camera packets only; drop MAC addresses, IP addresses, and everything except time and volume
    * Normalize timestamps to a zero start time (data set has time zone issues)
  * Data cleaning was the grungy part; instructor will post a solution and clean up the notebook for next time

* **Deep Learning (Lecture 13)**
  * Motivation
    * All models so far rely on handcrafted features, which is painstaking
    * Linear models, polynomial basis expansion; kernels (not covered) do the expansion inside the model rather than on the features
    * Trees and ensembles hard to apply to sequences, images
    * Deep learning = representation learning: sequence of transformations on raw data so the final layer can do a linear separation
  * Caveats for networking: time-consuming to train, learned features may not transfer across tasks, patterns may be spurious or missed
    * Upshot repeated throughout: a random forest often works just as well as a deep neural net
  * The neuron
    * Weighted sum of feature values plus offset (bias), then an activation function (looks like logistic regression)
    * Activation functions: sigmoid, tanh (-1 to 1), rectified linear unit (ReLU, popular; zero then linear)
    * Output in a bounded range (e.g., 0 to 1); neuron "activates" or not depending on the weights
    * Deep neural net = many layers of neurons; with enough layers and tuned weights, can learn nonlinear relationships (classic example: XOR, which linear models cannot learn)
    * Architectures often prebuilt; you do not have to design your own
  * Network structure: input layer, hidden layers (transformations), output layer (binary, multi-class, or regression)
  * Poll: most students have built a neural net before (prior ML class); nearly everyone has seen backpropagation
  * Training and backpropagation (intuition only, no math)
    * Forward pass: push training data through the network, get a prediction (e.g., attack/no attack, or continuous video startup delay)
    * Compare prediction to label to compute the loss; one pass over the training set is an epoch
    * Initial weights are essentially random, so first predictions are poor
    * For each weight, compute the slope (derivative) of the loss with respect to that weight and adjust in the direction that reduces loss, working backward through the layers
    * Learning rate hyperparameter: how far to move the weight each step
      * Too small: training takes forever
      * Too large: overshoot the minimum and bounce around
  * Training pitfalls
    * Very small values through some activation functions cause layers to stop learning; very large values lead to undefined weights (vanishing/exploding gradients)
    * Overfitting: throwing raw bitmaps (e.g., nPrint packet traces) at a network risks memorizing IP addresses, TTL, etc. (the husky/snow problem)
    * Validation set is critical: rising validation loss signals overfitting
  * When deep learning makes sense
    * Lots of labeled data
    * Features hard to engineer, or you do not know what the features should be
  * Student question: how do you decide which model family to use?
    * Check whether the relationship looks linear or polynomial (linear regression may work)
    * Linearly separable in one dimension (e.g., the IoT volume threshold) is easy; range-based conditions suit a tree
    * Complex nonlinear interactions between features (dependencies not independent) argue for deep learning
    * If you already understand your features, start with an ensemble of trees; it is much easier

* **Learning Traffic Representations: nPrint (Lecture 14, brief)**
  * Representation: each packet aligned so a given bit position always means the same header field (e.g., bit 481 always first bit of TCP header), constant size, normalized
  * After training, apply feature-importance analysis to see which parts of the packet distinguish classes
  * Example: operating system fingerprinting
    * Different OSes have different default TTLs, window sizes, TCP options
    * Different client source-port ranges (server side is 443 for HTTPS; client ephemeral port range varies by OS)
    * Team had not thought to use source port; the model surfaced it (value of representation learning for insight even if a random forest is deployed)
    * Same approach also revealed the model relying on fields it should not have (TTL / topology detector)
  * Data hygiene note: ongoing work on machine-readable labels stored in packet capture metadata alongside nPrint (mentioned only)

* **IoT DDoS Detection Paper (lead-in to hands-on)**
  * Paper posted in Slack; notebook is the code that generated the paper's results
  * Pipeline: split packets by device, bin by time, extract source/destination IP, ports, protocol, packet size
  * Features: packet size, inter-packet arrival time, protocol, bandwidth, number of destination IPs, change in number of destination IPs
  * Classification: DDoS attack or not
  * Result: k-nearest neighbors and random forest work as well as the neural net; deep learning not needed (one of the instructor's most-cited papers anyway)

* **Hands-On Activity #13: Deep Learning (IoT DDoS)**
  * Data is clean (unlike hands-on 12); run steps 1-3 to load and featurize
  * Inspect the feature table, then look at how the (not very deep) neural net is built; experiment with the architecture
  * Reduce epochs from 100 to 10-20 to finish within class time
  * Compare random forest vs. neural net results
  * Instructor noted a few cells needing cleanup; will fix the notebook

* **Course Position**
  * End of week two; supervised learning essentially complete
  * Early next week: unsupervised learning and generative models

### Meeting 8 (Mon Sep 14)

* **Dimensionality Reduction (Lecture 15)**
  * (Recording begins mid-intro) Reasons to reduce dimensionality:
    * Visualization
    * Computational feasibility
    * Noise reduction: with hundreds of features, many carry little signal and do not contribute to prediction
    * Preprocessing for unsupervised learning: clustering in high dimensions is expensive and yields lower-quality clusters
  * Caveat: throwing out dimensions loses information; goal is "maximum juice for the squeeze" (cut dimensions while keeping most of the information)

* **Principal Component Analysis (PCA)**
  * Question PCA answers: can fewer derived features capture most of the variation in a high-dimensional data set?
  * Board illustration in 2D reduced to 1D (works the same in very high dimensions)
    * First principal component: direction along which projected data spreads out the most (maximum variance)
    * Equivalent view: the projection line that minimizes the sum of squared distances from each point to the line
    * Projection is perpendicular onto the line, unlike regression (not vertical residuals)
    * "Tilt your head": points projected onto the line are a 1D representation of the original 2D data
    * Second principal component: maximum variance orthogonal to the first
    * Any point can be expressed as some amount of PC1 plus some amount of PC2 (change of basis) instead of x1 and x2
  * Anomaly detection intuition: a point far from the main line has a large PC2 component and little PC1; PCA and clustering both useful for network security
  * Choosing the number of components
    * Fewer is better, but do not lose too much information
    * Scree plot: explained variance vs. number of components; all components explain all variance; look for the elbow (more art than science)
    * Domain knowledge (e.g., normal vs. attack traffic suggests trying two components)
    * Visualize results for different numbers of components and see what structure appears
  * Networking uses
    * Example: packet capture with 5 features (source/destination IP, source/destination port, length) reduced to 2
    * Cluster traffic by protocol or application; spot outliers
    * Common first step before clustering and also before supervised learning (too many dimensions = expensive training, overfitting)
  * Kernel PCA extends the idea to nonlinear relationships (mentioned only); plain PCA captures linear combinations only

* **t-SNE**
  * Useful for seeing cluster structure and visualization
  * Three-step sketch: compute pairwise similarities among all points, fit a Gaussian, find a 2D distribution that minimizes the divergence from that Gaussian
  * Nonlinear, so it preserves relationships PCA cannot
  * Stochastic: different runs may produce different outputs
  * Visualization of existing data only; cannot transform new data
  * Example: t-SNE of all packets in one of the course packet traces, DNS packets in red, showing visible structure

* **Autoencoders**
  * Deep learning analog of unsupervised dimensionality reduction
  * Architecture: encoder compresses input through progressively narrower layers to a bottleneck (latent representation), decoder reconstructs the input (e.g., 100 features to 10 and back to 100)
  * Training objective: minimize reconstruction loss (squared error between input and output)
  * For dimensionality reduction, use only the encoder after training
  * Advantages
    * Nonlinear representations: multiplicative/exponential feature interactions (e.g., packet size and inter-arrival time) that a linear model cannot capture
    * Scales to hundreds of features; can handle nPrint-sized inputs where PCA would struggle
    * Anomaly detection built in: train on normal traffic; attack traffic reconstructs poorly (high error)
  * Disadvantages
    * Needs much more training data than PCA
    * Must choose bottleneck size and network design
    * Parametric: if the underlying data distribution changes, must retrain (ties back to model maintenance discussion)

* **Comparison Summary**
  * PCA: linear change of basis, no learned parameters, deterministic, applies to new data, cheap, good preprocessing for clustering, interpretable via loadings (weights on each principal component)
  * t-SNE: nonlinear, non-parametric, stochastic, visualization only, hard to interpret beyond the picture
  * Autoencoders: nonlinear, parametric, stochastic, applies to new data, anomaly detection, high compute cost, hard to interpret
  * Common to try more than one and see what pops out
  * Unsupervised learning is more art than science: data exploration, revealing structure, no answer key (unlike labels in supervised learning)

* **Networking Applications of Dimensionality Reduction**
  * 2D scatter plots; color points by labels when available even if not training on them (as with the DNS plot)
  * Helps understand structure and may help decide how to label data
  * Feed reduced features to a supervised classifier or clustering algorithm
  * Intrusion / anomaly detection, including cellular networks
  * IoT: many device types and applications, so labels are incomplete; a labeled Nest Cam may behave like an unlabeled camera of another type
  * Reducing nPrint's huge dimensionality

* **Hands-On Activity #15: Dimensionality Reduction**
  * Notebook oddly contains a decision tree in the middle and a messy data-set section; instructor noted both for cleanup
  * Task: take the HTTP and log4j data, build a feature matrix, apply PCA as shown, produce a scree plot from the explained variance ratio, optionally color-code points by class
  * Bug noted by a student: scree plot x-axis starts at 0 components (should start at 1)
  * Student question: does each principal component add another projection dimension on top of the previous ones? (Yes; the 2D example projects onto PC1 only, but in high dimensions you keep as many as needed)
  * Student question: could PCA help with music recommendation where you want personal preference to pull certain songs closer? (Discussion: PCA can find which features matter; encoding a preference would require an embedding or distance metric that moves those points closer)
  * Instructor plans to add t-SNE and autoencoder exercises to this hands-on for the next offering

* **Clustering (Lecture 16)**
  * Unsupervised learning recap: no labels or target variable; algorithm sees only features; goal is to discover structure, groups, patterns
  * Clustering groups similar data points together
  * Motivating networking examples
    * Separate very large flows from tiny flows
    * Group flows by application type
    * Separate scanning traffic (e.g., log4j scans) from normal web traffic without labels
    * Anomaly detection without labeled attacks: new, previously unseen attacks cannot have labels in advance
    * Routing updates and other non-traffic network data
  * Four algorithms covered: k-means, Gaussian mixture models, DBSCAN, hierarchical

* **k-means**
  * Requires choosing k in advance
  * Algorithm: randomly place k centroids; assign each point to nearest centroid; move each centroid to the mean of its assigned points; repeat
  * Student question: can a point change clusters between iterations? (Yes: as centroids move, a point may become closer to a different centroid and is reassigned)
  * Stopping criteria: centroids stop moving; distances to centroids fall below a threshold; no more reassignments
  * Choosing k
    * Plot sum of squared errors vs. k; with n clusters SSE is zero, with one cluster it is large; find the sweet spot (elbow), similar in spirit to the scree plot
    * Domain knowledge (e.g., attack vs. benign suggests k=2)
  * Failure modes
    * Assumes roughly spherical, compact clusters; ring-shaped or irregular clusters are split badly
    * Sensitive to outliers: a far-away point pulls a centroid and produces odd partitions (rescaling can help)
    * Works best with uniform-density clusters that are denser within than between
    * Categorical features do not make sense (no Euclidean distance between TCP and UDP)
    * Highly sensitive to centroid initialization; k-means++ chooses initial centers more carefully
  * Student observation: best suited to compact, well-separated clusters (confirmed)

* **Gaussian Mixture Models (GMM)**
  * Same expectation/maximization structure as k-means, but clusters are Gaussians rather than centroids
  * 1D board example with k=2: a dense cluster and a more spread-out cluster, each fit by a Gaussian
  * New point assigned to the Gaussian it most likely came from; handles the overlap region probabilistically
  * Good for continuous-valued features, e.g., log4j scan packets (smaller) vs. legitimate web traffic (larger packet sizes)
  * Both k-means and GMM require specifying the number of clusters and are sensitive to outliers

* **DBSCAN (density-based clustering)**
  * Motivation: no pre-specified number of clusters, non-uniform density, irregular shapes, outlier robustness
  * Idea: dense regions belong to the same cluster; sparse points are noise ("eating breadcrumbs")
  * Algorithm: pick a random unvisited point; if it has at least minPts neighbors within radius epsilon, start a cluster; recursively visit each neighbor and grow the cluster; repeat from an unvisited point
  * Points that are too spread out never reach minPts within epsilon and are left unclustered (automatic outlier detection)
  * Advantages: no k, arbitrary cluster shapes, detects outliers, easy to interpret
  * Disadvantages
    * Non-deterministic (depends on starting point)
    * Single density threshold struggles with clusters of varying density (variants exist; possible exam question: how would you handle variable-density clusters?)
    * Slow in high dimensions
    * Must tune epsilon and minPts

* **Hierarchical (Agglomerative) Clustering**
  * Start with each point as its own cluster; repeatedly merge the closest pair based on pairwise distance; builds a tree (dendrogram)
  * Board example with points a-g merging step by step
  * No k in advance: cut the tree at any height to get the desired number of clusters (cut low = many clusters, cut at root = one)
  * Linkage options for distance between merged groups: min, max, average (average is common)
  * When to use: data with a strict hierarchy (room in building in campus in network; application protocol inside transport inside network layer)
  * When not to use: cross-cutting categories that do not nest (source IP prefix vs. protocol; not all TCP or video traffic comes from one part of the topology)

* **Choosing a Clustering Method**
  * Known k and compact clusters: k-means
  * Variable size/shape clusters with Gaussian structure: GMM
  * Unknown k and need outlier detection: DBSCAN
  * Hierarchical structure in the data: hierarchical clustering
  * If labeled data exists and test data is in-distribution, supervised learning may be better
  * Very common security use: detecting unknown attacks, where supervised approaches lack examples

* **Hands-On Activity #16: Clustering**
  * Only ~12 minutes left; try k-means on the data set; DBSCAN and GMM at the end of the notebook if time permits
  * Autoencoder lecture skipped for time
  * Instructor staying after class for questions
