## Agenda

### Meeting 1

* **Basics**
  * Slack
  * Github Classroom
  * Canvas/Github
  * Logistics 
* **Coverage of syllabus**
  * Course objectives
  * Topic outline
  * Due dates
  * What's new this year
  * Project
* **Lecture on course material**
  * Introduction to Computer Networks
  * Packet Capture
* **Getting Started with Hands-On 1 (Notebook Setup)**

### Meeting 2

* **Hands-On 1: Packet Capture**
  * Wireshark basics and installation
  * Getting started with Jupyter, etc.
* **Introduction (Slides)**
  * Motivating Applications
  * Security
* **Hands-On Activities**
  * Packet Capture
    * Learning Objectives
       * Wireshark Setup
       * Notebook Setup
       * Packet Capture
       * Packet Capture to Pandas
       * Analysis
* **Security Applications (Slides, Discussion)**

### Meeting 3

* **Security Hands-On**
* **More Motivation**
  * Application Quality of Experience
  * Overview of Assignment 1
  * Application quality hands-on (?)
* **Resource Provisioning Motivation (no hands-on)**
* **Project Team Formation Time (if needed)**

### Meeting 4

* **Application Performance Inference - Video Quality of Experience**
  * Relationship between throughput and application performance
  * Video quality metrics: startup delay, resolution, resolution switches, rebuffering
  * Challenge: Inference without direct endpoint access (ISP/infrastructure provider perspective)
  * Encrypted traffic complicates inference - can't see frames, resolutions directly

* **What Can Be Observed from Encrypted Traffic**
  * Metadata available: packets/second, bits/second, packet sizes, timing
  * Segment boundaries (gaps in packet sequences where bitrate switches occur)
  * Segment download rates and counts
  * Machine learning problem: infer quality metrics from observable metadata

* **Service Identification via DNS**
  * Domain Name System (DNS) maps domain names to IP addresses
  * Passive observation: watch DNS query/response to identify service traffic
  * Example: netflix.com lookup returns 3 IP addresses for redundancy
  * Client typically tries first address, falls back to others if needed
  * IP addresses vary by location (different answers in Chicago vs NYC)

* **DNS-Based Traffic Filtering**
  * Step 1: Observe DNS query for service domain (e.g., netflix.com)
  * Step 2: Extract IP addresses from DNS response
  * Step 3: Filter subsequent traffic to/from those IPs to isolate service traffic
  * Challenge: Multiple services can share same IP (cloud hosting platforms like AWS)
  * Challenge: IP addresses can change during session (Netflix manifest provides new URLs)

* **Encrypted DNS and Future Challenges**
  * DNS over HTTPS (DoH) encrypts DNS queries via browsers
  * Queries go to CloudFlare, etc. instead of visible on network
  * Can no longer use DNS for service identification
  * Next wave: service identification becomes its own ML problem

* **Netflix Streaming Internals (Live Demo)**
  * Right-click → Inspect → Network tab shows browser requests
  * First request: license manifest (now encrypted via POST requests)
  * Manifest contains URLs for video segments
  * Video fetched in ~4000 byte chunks via range requests
  * Content delivery network domains: nflxvideo.net, ipv4-c###-ord###-ix.deploy.nflxvideo.net
  * Multiple servers used simultaneously during playback
  * Preview hovering triggers segment downloads (not actual playback)

* **Hands-On Activity #4: Service Identification**
  * Use DNS lookups to identify Netflix traffic in mixed packet capture
  * Filter DNS traffic for Netflix domains
  * Extract IP addresses from DNS responses
  * Filter packet capture for traffic to/from Netflix IPs
  * Count packets, bytes, and analyze traffic patterns
  * Prepares for Assignment 1: video QoE inference

* **Practical Complications**
  * Packet captures contain mix of traffic (Netflix, web browsing, IoT devices)
  * Not all Netflix traffic is playback (catalog browsing, previews)
  * Identifying actual playback sessions requires additional inference
  * These corner cases make real-world deployment challenging

* **Course Administrative Notes**
  * Canvas used for optional question submissions
  * Allows instructor to tailor lecture emphasis to student questions
  * Participation grade component
  * Solutions to hands-on activities provided after class
  * Agenda file updated with detailed notes from each session

### Meeting 5

* **Follow-up from Previous Sessions**
  * Q&A: Packet capture and network monitoring
    * Does packet capture affect the network itself? (Generally no - passive observation)
    * Can you see all network traffic from your laptop? (No - only your device's traffic in modern networks)
    * Historical note: Older switches/wireless APs broadcast all traffic, but modern networks don't
    * Visibility requires special access (e.g., running packet capture at the wireless access point)
    * Privacy implications of network monitoring at different vantage points
  * Assignment 1 updates
    * Fixed broken data set link
    * Data files now available on Canvas (pickle file and traffic file)
    * Column labels still need cleanup
  * Request for more time on hands-on code walkthroughs

* **Resource Allocation Applications**
  * Definition and Context
    * Resource allocation: Managing finite resources (bandwidth, storage, compute) among competing demands
    * Short-term: Allocating fixed capacity among current users
    * Long-term: Capacity planning, adding servers, predicting future needs
    * Optimization goal: Maximize utility function (e.g., good experience for all users)

  * Video Bitrate Adaptation (Adaptive Bit Rate - ABR)
    * Problem: Network conditions change (e.g., surge in demand during live events like Super Bowl)
    * Videos encoded at multiple quality levels (3-5 levels: low, medium, high, 4K, etc.)
    * Video segments: Chunks of ~4000 bytes downloaded sequentially
    * Player decision problem: Which quality level to request for next segment?
    * Tradeoffs:
      * Higher quality = better experience but more bandwidth
      * Lower quality = less bandwidth but potentially degraded experience
      * Avoiding dramatic quality changes (e.g., 4K to low quality)
    * Buffer management:
      * Monitor buffer occupancy (how much video is buffered)
      * Adjust bitrate based on buffer state to prevent rebuffering
      * ML approach: Learn when to switch quality levels based on network conditions
    * Professor Junshan Zhang's work on this problem (UC faculty)
      * Used neural networks for bitrate adaptation
      * Had direct access to player metrics (frame rate, resolution)
      * Feedback loop: network conditions + quality metrics → bitrate adjustment

  * Multi-User Resource Allocation
    * Local vs. Global optimization
      * Players make decisions based on local observations
      * Must consider competition with other traffic on the network
      * Design goal: Get good experience without "steamrolling" other users
      * Classic distributed systems problem: local observation, global optimization
    * Fairness considerations
      * Pareto frontier: Efficiency vs. Fairness
      * Efficient resource utilization (use all available capacity)
      * Fair sharing among competing users
    * Decision-making process
      * Observations: buffering rates, packet drops, network congestion signals
      * Traditional approach: Rule-based (e.g., if packets dropped, reduce quality)
      * Modern approach: ML-based inference of network conditions
      * Progressive quality adjustments (high → medium → low) rather than dramatic drops

  * Other Resource Allocation Examples
    * Congestion control: Pacing traffic from laptop/OS
    * Web server deployment (Google search example)
      * Multi-tiered applications: static content (images, stylesheets) + dynamic content (search results)
      * "What-if" scenarios: Impact of deploying new servers in different locations
      * Example: Deploy front-end server in Greece - what's the impact on search response time?
      * Predicting entire distribution of response times (median, 90th percentile, etc.)
      * Planned maintenance scenarios: Redirecting traffic (e.g., India users to Taiwan during maintenance)
      * Solution: Regression-based prediction (not necessarily deep learning)
    * Note: Simple models (like regression) can be very effective
      * Don't always need the most complicated approach
      * Sometimes simple rules or linear models work well
      * Will start with linear regression, then polynomial basis expansion

* **Introduction to Data Acquisition**
  * Passive Measurement
    * Definition: Observing the network without affecting it
    * Key characteristic: Measurement does not change network conditions
    * Packet capture is an example of passive measurement
    * Vantage points: laptop, access point, middle of network, server

    * Types of Passive Data
      * Complete packet traces
        * Every byte sent/received on network interface
        * Advantages: Complete information, detailed timing
        * Disadvantages:
          * Very large storage requirements
          * Much of it is encrypted anyway
          * Privacy implications (similar to wiretapping)
          * Compute/memory intensive

      * Header-only traces
        * Capture only packet headers (not payload)
        * Information available: flags, TTL, header length, port numbers, timing
        * Example use case: DNS traffic analysis for malware detection
        * Company example: Dembala (2006-2007) - ML on DNS lookups to detect malware/botnets
        * Consideration: Be careful about features that could lead to overfitting

      * Flow records (5-tuple aggregation)
        * Key = (source IP, destination IP, source port, destination port, protocol)
        * Summary statistics per flow:
          * Start and end times (flow duration)
          * Packet count
          * Byte count
          * Sometimes: flags seen
        * Derived metrics:
          * Average bit rate (bytes / duration)
          * Average packet rate (packets / duration)
        * Missing information:
          * Per-packet timing (inter-arrival times)
          * Cannot detect rebuffering events or bursty traffic
        * Analogy: Like a phone bill showing who you called, when, and duration (not conversation content)
        * Tradeoffs vs. packet traces:
          * Much smaller storage footprint
          * Less privacy-invasive (more like call data records)
          * Missing fine-grained timing information

      * Other passive data sources
        * System logs (web server logs, failure logs, error rates)
        * DHCP logs: IP address assignments, device locations, device types, connection times
          * Use case: Network quality analysis by location/time
          * Example: "WiFi is bad in Rosenwald at 3pm" - check DHCP logs for device count
        * DNS-based blocklists (RBLs): Track spammers, botnets, compromised hosts
        * Routing data (BGP): Internet path information
          * Use case: Debugging why traffic takes unexpected paths
          * Example: Apple FaceTime notifications routed through multiple ISPs instead of nearby datacenter

    * Advantages of Passive Measurement
      * Does not disrupt network conditions
      * Captures actual user experience
      * Universal applicability (same packet trace can answer multiple questions)
      * No need to design application-specific tests

    * Disadvantages of Passive Measurement
      * Privacy concerns (GDPR compliance, data storage regulations)
      * Larger data sizes (especially complete traces)
      * Requires right vantage point (access issues)
      * May not have access to needed monitoring location

  * Active Measurement
    * Definition: Sending test traffic to measure network properties
    * Key characteristic: Measurement can affect what you're trying to measure

    * Examples
      * Speed tests (most common example)
        * User clicks "go" button
        * Sends traffic as fast as possible to test server
        * Measures download speed (receiving) and upload speed (sending)
        * Returns metrics: latency, download/upload speeds
        * Note: Results vary by time and location
      * Traceroute: Measure network paths and hop-by-hop delays

    * Advantages of Active Measurement
      * No privacy concerns (you control the test traffic)
      * Smaller data footprint (only test results)
      * Can perform measurements when and where you want
      * Don't need special network access/vantage points

    * Disadvantages of Active Measurement
      * Disrupts network conditions (adds load to network)
      * May not reflect actual user experience
      * Requires designing application-specific tests
        * Web page load time test ≠ video streaming test ≠ speed test
        * Different services require different test designs (Netflix vs. Amazon)
        * Tests break when services change their implementation
      * Need to probe service provider infrastructure (may be disruptive)
      * Limited insight into actual application QoE
        * Speed test doesn't tell you about frame rate or resolution
        * Need specialized tests for each metric of interest

  * Passive vs. Active Summary
    * Passive: Observe real traffic, privacy concerns, universal but requires access
    * Active: Generate test traffic, no privacy issues, flexible but application-specific
    * Classic tradeoff: Accuracy vs. Privacy vs. Flexibility vs. Access

* **Preview of Next Session (Friday)**
  * Hands-On: Data Acquisition and Flow Statistics
    * Load packet traces
    * Create flow records from packet traces
    * Compute flow statistics (byte counts, packet counts, durations)
    * Calculate derived metrics (bit rates, packet rates)
    * After this exercise: Will use Python library (netML) for automated extraction
  * Feature Extraction
    * How to extract features from gathered data
    * Representing data for ML models
  * Moving toward ML model training

* **Netflix Player "Nerd Stats" Note**
  * Can view detailed playback metrics with special keystroke (Ctrl+Alt+T or similar)
  * Provides overlay with frame rate, resolution, bitrate, etc.
  * This is how training data was collected for QoE inference models
  * (Note: May have been removed or changed by Netflix)

### Meeting 6

* **Active and passive measurement**
   * Advantages and disadvantages of active and passive measurement
     * Infrastructure considerations
     * Measurements when you want them
     * Systems costs considerations
     * Privacy considerations
   * Feature extraction from packet captures
   * What is a flow? (5-tuple)
* **Hands-On Activity**
   * Packet Statistics Extraction - Flow Statistics (Manual)

### Meeting 7

* **Course Transition: From Data Acquisition to Data Preparation**
  * Completed: How to get data out of networks
  * Moving to: How to prepare data for ML models
  * Looking ahead: Deep learning (weeks 6-7) - can throw raw data at models
    * Tools: nPrint (bit-level representation)
    * But first: Learn traditional feature extraction approaches

* **ML Pipeline Overview**
  * Input → Transformation → Dimensionality Reduction → Training → Output → Evaluation
  * Each step has associated costs (systems consideration)
  * Not just about accuracy - consider:
    * Amount of training data required
    * Data acquisition cost
    * Ability to move data to model
    * Time to detection (inference speed)
    * Storage and transmission costs
    * Compute requirements for transformation

* **Systems Considerations in ML for Networks**
  * **Key principle**: It's not always about 99% accuracy
  * Important factors beyond accuracy:
    * **Time to detection**: How quickly can model provide answer for system to act on it
    * **Willingness to tolerate wrong answers**: Sometimes 90% accuracy quickly is better than 99% slowly
    * **Training data requirements**: How much data needed
    * **Data access**: Can we actually get the data we need
    * **Model serving**: How to deploy model to where data is
  * Example use case: QoE inference for network re-provisioning
    * Need timely answers to add servers or adjust configuration
    * Speed of inference matters as much as accuracy

* **Supervised vs. Unsupervised Learning Review**
  * **Supervised Learning**: Training with labels
    * Mnemonic: "Supervising" the model with examples
    * Can evaluate using labels as "answer key"
  * **Unsupervised Learning**: Training without labels
    * Common tasks: Clustering, pattern detection
    * Can still do classification (e.g., anomaly detection with two clusters)
    * Evaluation is harder (no answer key)

* **Features and Labels**
  * **Features**: Inputs to the model (covariates in statistics)
  * **Labels**: What the model is trying to predict/classify (supervised learning only)
  * Feature selection remains one of most important parts of modeling process
  * How you represent data affects how model learns (or doesn't learn)

* **Data Representation Challenges in Network Systems**

  * **Scale**
    * Example: UChicago campus traffic = 5-10 Gbps
    * Generates ~1 GB per second of data
    * Too much to process all at once for training
    * Need strategies for managing volume (sampling, filtering, aggregation)

  * **Sequential Dependencies**
    * Network traffic has temporal ordering constraints
    * Example: Can't exchange data before connection handshake completes
    * No traffic should occur after connection ends
    * Need to encode these dependencies in data representation
    * Techniques: Positional encoding (like in transformers)
    * Research area: How to represent temporal dependencies in network traces

  * **Non-Uniform Time Series**
    * Unlike audio (smooth) or video (relatively smooth even with VBR)
    * Network traffic is very bursty
    * Challenge: How to represent this to a model
    * NetML solution: Compute statistics per time window (e.g., 5 seconds) instead of per packet
    * Trade-off: Lose some information but handle burstiness

  * **Multi-Flow Dependencies**
    * Web browsers open ~8 parallel connections to servers
    * How to represent relationships across multiple flows
    * Still an open research question

  * **Different Representation Types**
    * **Event time series**: Binary (did traffic exceed threshold?)
    * **Time-based**: Statistics computed per time bin
    * **Volume-based**: Statistics computed per data volume (e.g., per megabyte)
    * NetML library supports these different representations

* **Feature Engineering and Transformation**

  * **Why Feature Engineering Still Matters** (Despite Deep Learning)
    * Reduces complexity → faster training and inference
    * Easier to understand models
    * Easier to maintain
    * Some features easy to compute, others require state or deep packet inspection

  * **Types of Feature Transformations**
    * **Binary variables**: Threshold-based (e.g., traffic exceeded rate?)
    * **Categorical variables**: Time of day, day of week (weekday vs weekend)
    * **Scaling and normalization**: Common preprocessing
    * **Aggregation**: Average, median, percentiles
    * **Polynomial basis expansion**: Express non-linear relationships in linear models
      * Linear models are fast and have provable properties
      * Can capture non-linear patterns with feature expansion

  * **Dimensionality Reduction**
    * Reduces volume of data passed to model
    * Systems benefits: lower storage, memory, compute
    * Can use tree-based models (e.g., Random Forests) for feature importance
      * Helps identify which features matter most
      * Enables further dimensionality reduction

* **Data Quality Issues and Pitfalls**

  * **1. Erroneous/Missing Data**
    * Example: Negative round-trip time values (OS bug in measurement software)
    * **Critical first step**: Understand the process that led to erroneous data
    * Question: Why am I getting missing/incorrect values?
    * If you don't understand the cause, question the entire dataset
    * **Always visualize and explore data before training**
      * Look at distributions, outliers, unexpected values
      * Not "tangential" - this is central to ML workflow
    * Strategies for handling:
      * Fix if possible (e.g., recompute from other available data)
      * Remove erroneous data points
      * Remove entire dataset if systemic issues
    * **Beware of library defaults**: scikit-learn does automatic imputation
      * May mask data quality issues

  * **2. Insufficient Training Data**
    * Not enough to discover appropriate patterns
    * May not be able to discriminate classes or make accurate predictions

  * **3. Non-Representative Training Data**
    * **Challenge**: Often don't know if data is representative
    * Examples in networking:
      * Training on laptop Netflix data - will it work on Android phone?
        * Different players, different resolutions
        * May or may not generalize
      * Training on Netflix - will it work on Amazon Prime?
      * Training on campus network traffic vs. enterprise network
        * Different application mixes (Canvas usage vs. enterprise apps)
        * Could affect anomaly detection models
    * **Important consideration**: Where will model be deployed?
      * Training data should match deployment context

  * **4. Irrelevant Features** (Overfitting Risk)

    * **Classic Example: Husky vs. Wolf (Snow Detector)**
      * Model learned to detect snow in background, not animal features
      * Pixels indicating snow were most important, not animal characteristics

    * **Networking Example: TTL-based Attack Detection (nPrint paper)**
      * Model achieved high accuracy on attack detection
      * Post-hoc analysis revealed it learned Time-To-Live (TTL) field
      * **TTL explanation**:
        * Starts at high value (e.g., 255)
        * Decrements by 1 at each network hop
        * Indirectly encodes network distance
      * **Problem**: Model learned network distance, not attack characteristics
        * Attacker from different location → model fails
        * Legitimate traffic at same distance → false positive
        * Essentially built a "topology detector" not attack detector
      * Discovered after reviewer questioned what model was using

    * **Defense strategies**:
      * Use domain knowledge and intuition
      * **Examine trained models post-hoc** (explainable AI)
      * Look at feature importance and weights
      * Ask: "Does this make sense? Is this relevant?"
      * **Diversify training data** (important suggestion from class)
        * Add noisy versions with varied irrelevant features
        * Include data with different TTL values, timestamps, etc.
        * Forces model not to over-rely on spurious correlations

    * **Bias in ML** (Non-networking example)
      * Finnish→English translation with gendered pronouns
      * "He is a nurse" / "She is a doctor" reversed based on training data stereotypes
      * Model does pattern matching on training distribution
      * Root cause: Training data reflected societal biases
      * Entire research area: Fairness and algorithmic transparency
      * **Lesson for networking**: Be aware of biases in network datasets

  * **5. Outliers**
    * **Critical practice**: Always investigate outliers
    * Mentor's advice: "Either you have a bug or a really interesting research paper" (mostly bugs)
    * Examples:
      * 30-second page load time - why did this happen?
      * Negative RTT - measurement error
    * Decision framework:
      * Can it be explained by phenomenon you want to model? → Keep it
      * Is it a measurement artifact? → Remove it (but understand if it's systemic)
      * Is it a one-time anomaly? → Investigate thoroughly before deciding

* **Hands-On Activity: NetML Library**
  * **Purpose**: Feature extraction from network traffic
  * Available on PyPI: `pip install netml`
  * Compatible with Python 3.x
  * **Key capabilities**:
    * PCAP to flows conversion
    * Statistical feature generation
    * Multiple representation options (time-based, volume-based, event-based)
  * **Important note**: Assignments don't require NetML
    * Can complete assignments without it
    * Library provided as useful tool for understanding feature extraction
  * **Hands-on approach**:
    * Follow README documentation
    * Explore different feature extraction options
    * Understand what statistics are being computed
  * **Feedback welcome**: Open source project, contributions encouraged

* **Preview of Next Session (Friday)**
  * ML Pipelines and Evaluation Metrics
  * Topics:
    * False positives and true positives
    * Accuracy, Precision, Recall
    * F1 score
    * ROC curves and AUC
  * Will continue with NetML hands-on (more time allocated)

* **Key Midterm Concepts Highlighted**
  * Systems considerations beyond accuracy in ML for networks
  * Difference between passive and active measurement (from previous lectures)
  * Examples of non-representative training data in networking contexts
  * Understanding and addressing irrelevant features
  * Data quality issues and their impact

* **Technical Issues Noted**
  * Canvas outage (AWS issue) affecting file access
  * Files to be posted on Slack as backup
  * Some hands-on materials need cleanup/completion



### Meeting 8

* **Midterm Logistics and Study Resources**
  * **Past Midterms Available**
    * Previous exams with full solutions accessible on GitHub
    * Compile LaTeX files to get PDF with or without solutions (toggle switch in LaTeX)
    * Exam covers material through Meeting 8 only
  * **Exam Format**
    * In-class exam (next Friday)
    * Designed to be completable in 30 minutes
    * No strict time limit - take as much time as needed (typically 60-80 minutes max)
    * Fits on approximately one double-sided sheet (makes grading easier, reduces time pressure)
    * **Assignment-based question**: Always includes a question related to the assignment
      * Not asking to write code, but may show code excerpts
      * Understanding what you did in the assignment is critical
  * **Performance Distribution**
    * Median typically around 90%
    * Left-tailed distribution (most students do well)
    * Not designed to trick students
  * **Feedback Questions**
    * Three short questions at end: difficulty level, interest level, one suggestion
    * Suggestion should be about the course, not the midterm
    * Feedback reviewed before grading other questions
  * **Study Strategy Suggestion**
    * Can use LLMs to generate practice tests from notes, past exams, and agenda
    * Feed all materials into LLM and ask for practice questions
    * Professor experimenting with using LLMs to draft actual exam (will need editing)
  * **Solutions to Hands-On Activities**
    * All solutions through #8 will be provided by Monday
    * Solutions for incomplete/skipped activities also provided for reference

* **Course Progress and Transition**
  * **Ready for Machine Learning Models**
    * Framework now complete: data acquisition → pipeline → evaluation
    * Next phase (post-midterm): Learning about specific models
    * Starting Monday: Supervised learning models (Naive Bayes, Linear Regression, etc.)
  * **Assignment 1 Clarifications**
    * **NetML is NOT required** for completing the assignment
      * Some students chose to use it (that's fine)
      * Assignment designed to be completed without NetML knowledge
      * Assigned before NetML was introduced
    * **Multi-class ROC/AUC**
      * For resolution prediction: Can pick a threshold (high vs. low) to make it binary
      * Or use scikit-learn's `roc_auc_score` with `multiclass` parameter (one-vs-rest approach)
      * Graphing ROC for multiclass: Not required if using sklearn's built-in scoring
    * **PCAP vs. Pickle File**
      * Pickle file has all necessary features already processed
      * In practice, would start from PCAP, but pickle saves that step for this assignment
      * Use pickle file as validation/test set for real Netflix session prediction

* **Model Training and Evaluation Fundamentals**

  * **The Central Question: Does the Model Work?**
    * **What does "work" mean?**
      * High enough accuracy for prediction on the task
      * Generalizes beyond training data (doesn't overfit)
      * Can be applied to different datasets without retraining
    * **Context: Supervised Learning**
      * This discussion focuses on supervised learning (models with labels)
      * Unsupervised learning evaluation is different (deferred to later)
      * Goal: Can the model predict labels/outcomes given features?

  * **The Bias-Variance Tradeoff (Most Important Concept)**
    * **Understanding Model Complexity**
      * X-axis: Model complexity (more training data, more features, higher-order polynomials)
      * Y-axis: Prediction error (lower is better, but only to a point)
      * Cyan line: Training error (what model sees during training)
      * Red line: Test error (what model sees on unseen data - the real goal)
    * **Underfitting vs. Overfitting**
      * **Underfitting**: Model too simple, high error even on training set
      * **Overfitting**: Model too complex, memorizes training data but fails on test data
        * Example: Could fit perfect curve through all training points (zero training error)
        * But that complex function won't generalize
        * Network examples: Using TTL, IP addresses, timestamps as features
      * **Sweet Spot**: Just enough complexity to minimize test error
        * Balance between fitting training data and generalizing
        * Global/local minimum on test error curve
    * **The Challenge**: Can't see the red line during training
      * Test set must remain unseen (like not giving students the exam in advance)
      * Would defeat the purpose of evaluation
      * Must use other techniques to estimate what test performance will be

  * **Train-Test Split and Cross-Validation**
    * **Basic Train-Test Split**
      * **CRITICAL FIRST STEP**: Split data BEFORE any processing
      * Common split: 80% training, 20% testing
      * **Never touch test set during training** - lock it away immediately
      * Only use test set for final evaluation
    * **K-Fold Cross-Validation**
      * **Purpose**: Estimate test performance without seeing actual test set
      * **How it works**:
        * Take training data and hold out portion (e.g., 20%) for validation
        * Train on remaining 80%, test on held-out 20%
        * Repeat K times with different held-out portions (K=5 is very common for 80/20 split)
        * Average results across all folds
      * **Use for hyperparameter tuning**:
        * Try different model complexities (e.g., polynomial degrees, number of features)
        * Evaluate on held-out validation sets
        * Select parameters that minimize validation error
        * Hope that performs well on final test set
    * **When Cross-Validation May Fail: Model Drift**
      * Training and test distributions can differ due to real-world changes
      * Examples:
        * Denial of service attack
        * Flash crowd (Super Bowl traffic surge)
        * COVID-19 (no traffic on campus)
        * Seasonal changes (leaves on trees affecting radio propagation)
      * **Verizon Example**: Cellular network performance prediction
        * Summer vs. winter: Different radio propagation due to humidity, foliage
        * Models trained in summer may not work in winter
        * This is a model drift problem
      * Model drift detection is active research area

  * **Data Leakage - Critical Pitfall**
    * **The Problem**: Accidentally letting test set information leak into training
    * **Common Mistake Example**: Normalization/scaling
      * Wrong: Take average of entire dataset → normalize → split train/test
      * Why wrong: Test set statistics (mean, variance) influenced training
      * Right: Split first → normalize using only training set statistics → apply same normalization to test
    * **Best Practice**: FIRST thing in pipeline is train-test split
      * Before normalization, before feature engineering, before anything
      * Take 20% for test, put in a box, don't look at it
      * Do all processing on training set only
    * Very common, non-intuitive mistake even for experienced practitioners

* **Evaluation Metrics Beyond Accuracy**

  * **The Base Rate Fallacy**
    * **Pregnancy Test Example**
      * Test claims 99% accuracy
      * Population pregnancy rate: 1%
      * Can achieve 99% accuracy by always saying "not pregnant"
      * But this test is useless for detecting actual pregnancies
    * **Implication**: Accuracy alone is misleading when classes are imbalanced
    * Need metrics that capture detection performance on minority class

  * **Network Security Examples**
    * **Email Spam Filtering**
      * Spam rate historically ~90% of email
      * Could get 90% accuracy by marking everything as spam
      * **But**: Would miss job offers, important emails (high cost of false positives)
      * **Preference**: Lower detection rate (70%) with very low false positive rate (0.0001%)
      * Better to see some spam than to lose important legitimate email
    * **Cancer Screening (Prostate Cancer)**
      * Base rate very low (<1% of population)
      * **Detection rate (True Positive Rate)**: Don't want to miss actual cancer
      * **False Positive Rate**: Don't want unnecessary biopsies, surgeries
      * Both rates matter, different people weight them differently
      * Screening guidelines are controversial partly due to these tradeoffs

  * **Confusion Matrix (Key Tool for Understanding Performance)**
    * **Structure**
      * Rows: Actual labels (ground truth)
      * Columns: Predicted labels (model output)
      * **Diagonal**: Correct predictions (want high numbers here)
      * **Off-diagonal**: Errors (want low numbers here)
    * **Key Metrics from Confusion Matrix**
      * **True Positive (TP)**: Correctly predicted positive (diagonal, top-left for binary)
      * **True Negative (TN)**: Correctly predicted negative (diagonal, bottom-right for binary)
      * **False Positive (FP)**: Predicted positive, actually negative (off-diagonal)
      * **False Negative (FN)**: Predicted negative, actually positive (off-diagonal)
    * **Visual Reading**
      * Scikit-learn shades confusion matrices for easy reading
      * Darker diagonal = better performance
      * Can compute all metrics below from confusion matrix
    * **Multi-Class Extension**
      * Can be N×N for N classes (e.g., 10×10 for 10-way classification)
      * Shows which classes are confused with each other
      * Example: How often is "bear" classified as "horse" or "rabbit"?

  * **Core Evaluation Metrics Definitions**
    * **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
      * How many predictions were correct overall
      * Entire diagonal / entire matrix
      * Misleading when classes imbalanced
    * **Precision (Positive Predictive Value)**: TP / (TP + FP)
      * Of things predicted positive, how many were actually positive
      * 1 - false positive rate (roughly)
      * Visual: True positives / predicted positive column
    * **Recall (True Positive Rate, Sensitivity, Detection Rate)**: TP / (TP + FN)
      * Of actual positives, how many did we detect
      * How sensitive is the test to detecting positives
      * Visual: True positives / actual positive row
    * **Specificity (True Negative Rate)**: TN / (TN + FP)
      * Of actual negatives, how many did we correctly identify as negative
      * Visual: True negatives / actual negative row
    * **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
      * Harmonic mean of precision and recall
      * Single number capturing both metrics
      * Range: 0 to 1, higher is better
      * **Limitation**: Doesn't factor in true negative rate
      * Not a silver bullet - often need to look at multiple metrics

  * **Receiver Operating Characteristic (ROC) Curve**
    * **Purpose**: Visualize tradeoff between detection and false positives as threshold varies
    * **Axes**
      * X-axis: False Positive Rate
      * Y-axis: True Positive Rate (Detection Rate, Recall)
    * **Interpretation**
      * **Ideal curve**: Goes straight up (high detection) then right (low FPR)
      * **Good model**: Curve bows toward top-left corner
      * **Random classifier**: Diagonal line (worst case)
      * Can read off tradeoffs: "For 90% detection, I get 5% false positive rate"
    * **Area Under ROC Curve (AUC)**
      * Single number summarizing ROC curve
      * **Range**: 0.5 (random) to 1.0 (perfect)
      * 0.5 = straight diagonal line (worst)
      * 1.0 = perfect classifier
      * If curve sags below diagonal, just flip predictions
    * **Possible Midterm Question**: Read ROC curve, interpret tradeoffs, or sketch what good ROC should look like

  * **Precision-Recall Curve**
    * **Purpose**: Alternative visualization of model performance
    * **Axes**
      * X-axis: Recall (True Positive Rate)
      * Y-axis: Precision (Positive Predictive Value)
    * **Interpretation**
      * **Ideal curve**: Stays high (top-right) as recall increases
      * Want high precision even as we increase recall (say more "yes"es)
      * Mirror image of ROC curve conceptually
    * **Intuition**
      * As model says "yes" more often, recall goes up
      * But want precision to stay high (minimize noise in "yes" predictions)
      * Curve shows how precision degrades as we increase recall

* **Hands-On Activity: ML Pipeline (#8)**
  * **Dataset**: Web browsing vs. port scan detection
  * **Task**: Train-test split, model training, confusion matrix generation
  * **Model**: Random Forest (haven't covered yet, but can use any binary classifier)
  * **Key Steps**
    * Convert PCAPs to flows (two separate files: HTTP traffic and scan traffic)
    * Label data: One file is legitimate (0), other is attack (1)
    * Combine labeled datasets
    * Train-test split (use 80/20)
    * Train model
    * Generate predictions on test set
    * Create confusion matrix
    * Compute evaluation metrics (F1 score recommended over accuracy)
  * **Cross-Validation Notes**
    * K=5 is common (corresponds to 80/20 split done 5 times)
    * Use for hyperparameter tuning
    * Don't need to scale/normalize for Random Forest (one reason it's a favorite model)
  * **Important**: Do this hands-on - critical for understanding evaluation
    * Even if you've done train-test before, good review with network data
    * Understanding these concepts is key for everything after midterm

* **Key Concepts Students Should Understand** (Midterm Topics)
  * **Suggested Midterm Topics**:
    * Explain bias-variance tradeoff and the "sweet spot" in model complexity
    * Why can't we see test error during training, and how do we estimate it?
    * What is K-fold cross-validation and why do we use it?
    * Define and compute: Accuracy, Precision, Recall, Specificity, F1 Score
    * Read and interpret a confusion matrix (compute metrics from it)
    * Explain base rate fallacy with examples (spam, medical tests)
    * Why accuracy is insufficient for imbalanced datasets
    * Example: Network features that could cause overfitting (IP addresses, TTL, timestamps)
    * Read/interpret ROC curve or Precision-Recall curve
    * What is data leakage and how to prevent it (normalize AFTER split)
    * Tradeoffs between precision and recall in different applications (spam vs. cancer screening)
  * **Most Important Takeaway**: Understanding the bias-variance tradeoff figure
    * Relationship between model complexity, training error, and test error
    * Finding the sweet spot for generalization
    * This concept underpins everything in model training and evaluation

* **NOT Covered in Detail**
  * Mathematical derivations of metrics
  * Proof of why cross-validation works
  * Specific model algorithms (Random Forest details - coming later)
  * Evaluation of unsupervised learning (deferred)
  * How to compute Area Under Curve (AUC) by hand

* **Project Proposal**
  * **Due Date**: Originally Monday, extended to Friday (can use late hours)
  * **Format**: Very short proposal
  * **Required Information**:
    * What are you doing? (Brief description)
    * What data will you use? (MOST IMPORTANT - need data to do project)
    * Why are you doing this? What do you hope to learn?
  * **Data Resources**
    * Many network ML datasets available (links provided or will be updated)
    * Choose dataset first before finalizing project idea
    * Don't wait until week 9 to find data
  * **Purpose of Proposal**
    * Ensure you think about project early (not day 4 of week 9)
    * Verify you have access to necessary data
    * Pick something that teaches you what you're interested in learning
    * Get feedback/approval from instructor

* **Preview of Next Session (Monday)**
  * **Supervised Learning Models Begin**
    * Finally doing machine learning models!
    * Starting with Naive Bayes (non-parametric model)
    * Historical application: First ML use in network security (spam filtering)
    * Brush up on probability: Conditional probability, Bayes' rule
  * **Future Topics**
    * Meeting 10 (approx.): Linear Regression
    * Later: Logistic Regression, SVM, Decision Trees, Random Forests, Deep Learning
    * Keeping model coverage application-oriented (not proving least squares, etc.)
    * Most students have seen some of these - focus on network applications

**Midterm Coverage Stops Here** (Nothing below this point will be on the midterm)

### Meeting 9

* **Midterm Exam Format and Expectations - Detailed Walkthrough**

  * **Physical Format**
    * Now 4 pages instead of 2 (bigger answer boxes as requested)
    * Same amount of content, just more writing space
    * Double-sided pages (no backs available)
    * Answer boxes sized to indicate expected response length
    * Don't write outside boxes - may not be visible in GradeScope

  * **Question Types**
    * Mix of multiple choice and short answer
    * "Select all that apply" questions
    * Yes/No questions with explanation boxes
    * Explanation boxes worth real points (not just the checkbox)
    * Professor cares about rationale as much as correct answer

  * **Grading Philosophy**
    * If you get yes/no wrong but explanation is good → partial credit
    * Questions often ambiguous or underspecified
    * GradeScope allows clustering similar answers
    * If half the class gets question wrong → probably bad question
    * Explanation boxes are your opportunity to save yourself
    * Professor reads all explanations to understand reasoning

  * **Time Expectations**
    * Designed for ~30 minutes if fast
    * Nominally 80 minutes, but untimed (can stay as long as needed)
    * No after class, so could theoretically stay very long
    * Not testing time/speed - testing understanding
    * TAs take exam first; multiply their time by 2 for student estimate

  * **Study Resources Provided**
    * Exam format document (what to expect)
    * Three past exams with solutions available
    * Can use LLM to generate practice questions
    * Professor uses LLM to draft exam (then edits extensively)
    * Prompt file available showing how exam was generated
    * All topics listed in exam instructions

  * **What to Focus On**
    * Topics explicitly listed in exam document
    * Format matches past exams very closely
    * Can see point distributions from past exams
    * Professor edited draft, so fairly final (4 days before exam)
    * "Don't feel like you have to write three pages"

* **Take-Home Midterm (Assignment)**

  * **Format and Scope**
    * Similar to Assignment 1 (already completed)
    * Application: Video conferencing quality (not Netflix)
    * Public template (can find in advance if you look)
    * Same rules as Assignment 1 regarding tools

  * **Collaboration Policy - IMPORTANT**
    * NO collaboration on take-home midterm (unlike Assignment 1)
    * Do NOT work with others (only assessment with this requirement)
    * Can use any tools EXCEPT other humans
    * Can talk to TAs (they won't give answers, but can discuss)
    * Purpose: "Good to struggle by yourself a little bit"
    * Not about policing - about learning through struggle
    * "People have outed themselves in embarrassing ways"

  * **Stress Management**
    * "It's going to be okay"
    * "I hope everybody aces it"
    * "You have everything you need"
    * Goal: Pleasant experience (as pleasant as midterm can be)
    * Will be posted evening of class (mid-evening)

* **Project Proposal Timeline Clarification**

  * **Due Date Confusion Resolved**
    * Spreadsheet said "today" but syllabus unclear
    * Syllabus said "beginning of week 5"
    * **Clarified: End of Week 5** (not today)
    * Posted to Slack to memorialize decision
    * Will be added to agenda/class notes

* **Student Questions from Canvas**

  * **Three Comments Received**
    1. AWS outage discussion (covered below)
    2. Standard models for DDoS/malware detection (deferred to later)
    3. Naive Bayes assumption about underlying distribution (covered in lecture)

* **AWS Outage Discussion - Systems Perspective**

  * **Consolidation and Brittleness**
    * Increasing consolidation of services onto centrally managed infrastructure
    * Created brittleness in internet systems not seen in modern times
    * Research study 4 years ago (unpublished, on arXiv):
      * Measured top 10,000 websites
      * 26-27% hosted exclusively on Amazon
      * Reviewers thought it was "boring" (it wasn't)
      * Professor posted on LinkedIn: "I told you so"
    * Lesson: "Don't let anybody tell you what you're doing is bullshit or boring"

  * **Implications**
    * **Censorship**: Easier to pull content offline (everything on Amazon)
    * **Security and Availability**: Single points of failure
    * **Extreme vulnerability**: Points of brittleness in the system

  * **Relevance to Machine Learning**
    * What-if scenario evaluations in large systems
    * TA Chase has worked extensively on this
    * Student Luca worked on this summer

  * **Evolution of System Analysis**
    * **Era 1: Pencil and Paper (when dinosaurs roamed)**
      * Professor's advisor did everything on chalkboard
      * Used calculus and closed-form equations
      * Could model simple web servers mathematically
      * Example: Calculate throughput of web app with math
      * Professor "wasn't very good at calculus" and "doesn't like chalk"

    * **Era 2: Simulators**
      * Systems got too complicated for chalk
      * Built network simulators: NS2, then NS3
      * Joke about not getting to numbers higher than 3 (middle school reference)

    * **Era 3: Machine Learning (Current)**
      * Systems now so complex, closed-form equations intractable
      * Assignment 1 is example: can't model video QoE with equations
      * Front-end/back-end systems too complex
      * Need ML to predict: page response time, traffic patterns, failures
      * Questions now only answerable by machine learning:
        * Will this change cause too much traffic to one datacenter?
        * Will it cause routing loops?
        * Will part of system meltdown under load?

  * **Configuration Changes and ML**
    * Used to know what would happen when changing web server config
    * Now systems too complex to predict
    * ML already playing a role in what-if analysis
    * Will play even larger role in future
    * "Who knows, maybe the code that broke was AI generated" (probably was)

  * **Deferred Question**: Standard models for DDoS/malware
    * Will cover as specific models are introduced
    * Can't discuss SVM before learning SVM
    * Will mention applications throughout course

* **Naive Bayes Classifier - Comprehensive Coverage**

  * **Probability Foundations Review**
    * **Independence**
      * Coin flip: Next flip doesn't depend on previous (50/50 always)
      * Statistically independent events
    * **Conditional Probability (Non-Independence)**
      * Token prediction in LLMs
      * Example: Given word "machine", what's probability of next word?
      * "Learning" or "translation" more likely than "sandwich"
      * Probability conditioned on what's already observed
    * **Two key concepts**: Independence and conditional probability

  * **Bayes Rule as a Classifier**
    * **Formula**: P(Y|X) = [P(X|Y) × P(Y)] / P(X)
    * Poll: About half the class has seen Bayes rule before
    * **Reading it as a classifier**:
      * Y = target prediction (e.g., spam or not spam)
      * X = features (e.g., words in message)
      * P(Y|X) = Probability of class given features
      * P(X|Y) = Probability of features given class
      * P(Y) = Prior probability of class
      * P(X) = Prior probability of features (can drop this!)

  * **Maximum A Posteriori (MAP) Prediction**
    * **Ŷ = argmax_Y P(Y|X)**
    * Pick the Y that maximizes posterior probability
    * "Maximum a posteriori probability" (MAP)
    * In English: Pick outcome that's most likely given observations
    * Compare P(spam|X) vs. P(not spam|X), pick larger

  * **Simplification: Dropping the Denominator**
    * P(X) doesn't depend on Y (marginal probability)
    * Same across all classes when comparing
    * Can just compare numerators: P(X|Y) × P(Y)
    * **Big advantage**: Don't need to know P(X)!
      * Example: What's probability of word "machine" in all emails?
      * "I don't freaking know" - billions of emails
      * Thankfully, don't need to compute this

  * **Handling Multiple Features (The "Naive" Part)**
    * **Reality**: X is a vector of features (X₁, X₂, ..., Xₙ)
    * All words or word combinations
    * P(X|Y) = P(X₁, X₂, ..., Xₙ|Y) - joint distribution
    * **Problem**: Need to evaluate joint probability
      * Every email has different word combinations
      * Can't estimate probability of exact word combination

  * **Independence Assumption (Why "Naive")**
    * **Assumption**: Features are independent
    * P(X₁, X₂, ..., Xₙ|Y) = P(X₁|Y) × P(X₂|Y) × ... × P(Xₙ|Y)
    * Probability of joint event = product of individual probabilities
    * Undergraduate probability 101
    * **Final classifier**: Ŷ = argmax_Y [P(Y) × ∏P(Xᵢ|Y)]

  * **Reading the Classifier in Plain English**
    * Probability email is spam = P(spam) × ∏P(word|spam)
    * P(spam) = prior (e.g., 90% of email is spam)
    * Multiply by probability of each word appearing in spam
    * Words like "Viagra", "Rolex", "Canadian pharmacy" → high P(word|spam)
    * Those words rare in legitimate email → low P(word|not spam)
    * Compare spam score vs. not-spam score, pick higher

  * **Why It's Called "Naive"**
    * **Student question**: Are word combinations dependent?
    * **Answer**: YES! Features are NOT independent at all
    * Example: "machine" and "learning" are highly correlated
      * If you see "machine" in this class, ~90% chance next word is "learning"
      * Violates independence assumption completely
    * Called "naive" because makes unrealistic independence assumption
    * **Mystery**: It works anyway! Nobody fully understands why
      * "Welcome to the rest of the quarter"
      * Some theories exist, but not well understood
      * Common theme: Models work despite violated assumptions

  * **The Zero Probability Problem**
    * **Student insight**: What if word never appears in training set?
    * Example: "Viagra" never in legitimate email training set
    * P(Viagra|legitimate) = 0
    * Product becomes 0 → posterior probability = 0
    * **Real scenario**: Professor sends assignment about spam filtering
      * "We're going to do spam filtering, look for words like Viagra"
      * Legitimate instruction email marked as spam!
      * Zero probability because word not in legitimate training set

  * **Solution: Smoothing**
    * Never let probability be exactly zero
    * Add small value like 0.01 (nudge)
    * Scikit-learn has parameter to control smoothing amount
    * Should be "super small number" not zero

  * **Distribution Assumptions**
    * Need to know P(Xᵢ|Y) for each feature
    * **Categorical features**: Use Bernoulli distribution
    * **Numerical features**: Common to assume Gaussian distribution
    * **Does it hold in practice?** Sometimes better than others
    * "But independence doesn't hold either, so what are we worried about?"
    * Somehow it all works despite violated assumptions

  * **Advantages of Naive Bayes**
    * **Incredibly efficient**
      * Unlike deep learning and transformers
      * Basically multiplication and table lookup
      * No GPU required, no 3-week training time
    * **Small data requirements**
      * Don't need gigantic dataset
      * Just need good probability estimates
      * Can assume distribution and compute from few samples
    * **Works surprisingly well**
      * Despite independence assumption not holding
      * Despite distribution assumptions being approximate
      * Mystery why, but empirically effective

  * **Disadvantages**
    * **Not good at learning feature relationships**
      * Can't capture correlations between features
      * Assumes everything independent
    * **Sensitive to zero probabilities** (need smoothing)
    * **Requires distribution assumptions**

* **Historical Application: Spam Filtering**

  * **Spam Assassin - Early Days**
    * One of first ML applications in networking and security
    * Open source project (still exists!)
    * Originally 100% naive Bayes classifier
    * Ran locally on your machine
    * "Not a toy assignment" - this was real production spam filtering

  * **Highly Configurable**
    * Could tweak probability parameters
    * Example: "I really am interested in Viagra, I buy a lot of that"
    * Adjust P(word|class) values manually
    * Corresponds to tweaking the math we learned

  * **Arms Race: Attackers Adapt**
    * **Phase 1**: Naive Bayes spam filters work
    * **Phase 2**: Attackers figure out the algorithm
    * **Countermeasure 1**: Attach Shakespeare to bottom of spam emails
      * Adds legitimate words to shift probabilities
      * Makes spam look more like legitimate email
    * **Countermeasure 2**: White text on white background
      * Shakespeare invisible to humans
      * But parsed by naive Bayes filter
      * Defeats spam detection

  * **Exam Question Preview**
    * "Why would attaching Shakespeare mess up naive Bayes?"
    * Understand how it affects the math
    * Not on this midterm (Naive Bayes not covered)
    * But good question for final
    * Don't need to compute - understand intuitively

* **Hands-On Activity #9: SMS Spam Classification**

  * **Dataset**
    * SMS text message corpus
    * Labels: spam or ham (legitimate)
    * **Prior probabilities**:
      * 86% ham (legitimate)
      * 13% spam
    * Different from email (which was ~90% spam historically)

  * **Part 1: Data Preparation**
    * Load dataset into DataFrame
    * Columns: label and message text
    * Cleanup: Remove commas, punctuation, etc.
    * Train-test split (80/20)
    * **Verify priors same in both splits** (good practice)

  * **Building Vocabulary**
    * Create universe of all words in training set
    * Needed to compute P(word|class)
    * "Use your favorite AI assistant" for this
      * Would take long time to code manually in Python
      * Not the learning objective here
      * ~20 minutes even with assistance
    * Count word frequencies in each class

  * **Professor's Confession**
    * Found errors in own preprocessing
    * Some weird words in vocabulary
    * "I've got some work to do"
    * Demonstrates: Data cleaning is hard, iterative process

  * **Part 2: Scikit-Learn Implementation**
    * Used `MultinomialNB` classifier
    * Works well with categorical features
    * Had problems with Bernoulli version previously
    * Much easier than implementing from scratch

  * **Part 3: Implement from Math (Optional)**
    * Build naive Bayes from the mathematical formula
    * "Really don't expect you to do" but available
    * Part 1 sets you up for this if interested
    * Mostly for understanding, not practical

  * **Time Expectations**
    * Part 1: ~20 minutes (with AI assistance)
    * Full completion: Not expected in class
    * "Massive headache" without AI help
    * Focus: Understanding concepts, not Python string manipulation

* **Course Pacing and Next Steps**

  * **This Week**
    * Midterm on Friday (no lecture that day)
    * Won't finish hands-on today (15-20 minutes left in class)

  * **Next Week (Monday after midterm)**
    * Cover topics 10 and 11 together
    * Linear regression and logistic regression
    * "Shovel you all those vegetables" in one class
    * Both are linear models students have seen before
    * Moving quickly through familiar material

  * **Professor Available**
    * After every class
    * Can schedule evening Zoom calls
    * Open to suggestions, feedback, questions
    * "I hang out after class all the time"

* **Key Takeaways**
  * Naive Bayes: Simple, effective, historically important
  * Independence assumption violated but works anyway
  * First major ML application in network security
  * Understanding math helps understand adversarial attacks
  * Data preparation often hardest part (cleaning, preprocessing)

### Meeting 10 

* **In-Class Midterm**

### Meeting 11

* **Administrative Updates**

  * **Second Midterm Logistics**
    * Date: December 5th (Week 9, last Friday of term)
    * Not cumulative - covers topics 9-16 only (after first midterm)
    * Format similar to first midterm (in-class, untimed)
    * 2023 final/second midterm found and added to repo
    * All past exams now available: 2021, 2022, 2023
    * 2023 exam was ~4 pages (longer than first midterm's 2 pages)
    * Will cover supervised and unsupervised learning techniques
    * First time covering autoencoders on exam

  * **Cheat Sheet Policy - IMPORTANT CHANGE**
    * NOW OFFICIALLY ALLOWED for second midterm
    * Can bring TWO sheets (double-sided) to second midterm
    * Option 1: Bring sheet from first midterm + one new sheet
    * Option 2: Make two completely new sheets
    * Not previously announced in syllabus - will be added
    * Found on instructions of old practice exams, but not widely communicated
    * Professor's philosophy:
      * Making cheat sheet is excellent study exercise
      * Using it during exam actually slows you down
      * Main benefit is in the preparation process

  * **Grading Approach and Philosophy**
    * Uses GradeScope to cluster similar answers
    * Gives credit for reasonable variations when question was ambiguous
    * Example: "model drift" vs "concept drift" vs "data drift" all accepted
    * Not testing jargon knowledge - testing understanding
    * If 15 out of 40 students answer same way, likely question was unclear
    * Midterm solutions posted with explanations of partial credit

  * **Regrade Policy**
    * Clear mistakes (answer matches solution but not given credit): Fixed immediately
    * Subjective regrades (interpretation differences): Will review entire exam
    * Rationale: Discourage point-hunting; likely got credit elsewhere
    * Unlikely individual points will affect final grade
    * Open to discussions about any question

  * **Student Feedback Incorporated**
    * Positive response to hands-on activities - will continue
    * Solutions will be released faster - target: before next class
    * Slide coverage confusion addressed:
      * Students don't know which slides were actually covered vs. available
      * Short-term fix: Will call out slide numbers/titles during lecture
      * Long-term plan: Convert to markdown to auto-track slide usage
    * NetML confusion on Assignment 1:
      * Was NOT required for assignment (caused confusion for some)
      * Assignment predates NetML introduction in course
      * Can complete without NetML knowledge
    * Request: More candy (noted)

* **Linear Regression**

  * **Fundamental Concepts**
    * Poll: Everyone in class has seen linear regression before
    * General supervised learning framework
      * Goal: Learn function f(X) that predicts Y from input features X
      * Given: Training examples (X, Y) pairs
      * Find: Function that best predicts Y from X
    * Example: Temperature (X) vs. People in park (Y)
    * Fitting a line to data in simple case

  * **Error Measurement**
    * Prediction: Ŷ (Y-hat)
    * True value: Y
    * Error for single point: |Ŷᵢ - Yᵢ|
    * Total error: Sum over all data points
    * Two common approaches:
      * Absolute error: Σ|Ŷᵢ - Yᵢ|
      * Squared error: Σ(Ŷᵢ - Yᵢ)²
    * Squared error automatically produces positive values (no need for absolute value)
    * Further from true value = more amplification with squared error

  * **Least Squares Linear Regression**
    * Minimizes Residual Sum of Squares (RSS)
    * Visual interpretation: vertical distance from points to line
    * Red dotted lines in plot = (Ŷ - Y) for each point
    * Single input: Y = mX + b (choosing slope m and intercept b to minimize RSS)
    * Multiple inputs: Matrix/vector formulation (requires linear algebra)
    * Has closed-form solution (can solve with derivatives)
    * Scikit-learn typically works with multiple inputs (vector X)

  * **Linear Independence Assumption**
    * Features should be linearly independent
    * Often not true in practice (features correlate)
    * Many non-linear relationships in real data

  * **Application: Network Traffic Forecasting**
    * Using linear regression to predict future network traffic
    * Still used in practice for this application

  * **Hands-On Activity #10: Linear Regression on Network Traffic**
    * **Setup**
      * Use NetML for feature extraction from packet captures
      * Natural relationship: bytes vs. packets in flows
      * Task: Predict bytes from packet counts (columns 10 and 11 in NetML matrix)
      * "Doomed to succeed" - intentionally simple relationship

    * **Scikit-learn Workflow**
      * Select features (X) and target variable (Y)
      * Train model: `model.fit(X, Y)`
      * Generate predictions for plotting
      * Visualize: scatter plot of true values + regression line

    * **Key Observation: Acknowledgement Packet Artifact**
      * Plot shows two distinct clusters
      * Upper cluster: Data packets (high bytes, variable packets)
      * Lower cluster: Acknowledgement packets (low bytes, few packets)
      * Model fits poorly to acknowledgements (prediction error)

    * **Discussion: Why Does Model Fit to Upper Cluster?**
      * Squared error amplifies larger numbers
      * Higher byte counts have larger absolute errors
      * Minimizing RSS prioritizes fitting larger values
      * Trade-off: Good fit for data packets, poor fit for ACKs

    * **Design Question: How to Fix This?** (Potential exam question)
      * Approach 1: Rescale/normalize data
        * Would help model see differences between data points
        * Reduces impact of magnitude differences
      * Approach 2: Separate packet types (Professor's preference)
        * First classify: data packet vs. acknowledgement
        * Train two separate linear models (one per type)
        * Use appropriate model based on packet type
        * Better captures the two distinct distributions
      * Highlights: Sometimes one model isn't enough; pipeline may need classification → regression

* **Polynomial Basis Expansion**

  * **Motivation and Concept**
    * Real data often has non-linear relationships between features and target
    * Examples: squared relationships, cubed relationships, etc.
    * Want to capture non-linearity while keeping linear regression
    * Linear regression appeals:
      * Computationally efficient
      * Nice provable properties about error minimization
      * Closed-form solution

  * **How Basis Expansion Works**
    * Key insight: "Changing form of data, not form of function"
    * Instead of passing single feature X, create feature matrix
    * Columns: [X, X², X³, ...] (polynomial powers of X)
    * Example transformation:
      * Original: Single column X
      * Expanded: Matrix with X, X², X³ as separate columns
    * Model still uses linear regression
    * Exploring linear relationship between X² (as a feature) and Y
    * Not giving model more features - giving additional representations of same feature

  * **Implementation in Hands-On**
    * Build feature matrix with polynomial terms
    * Pass to standard linear regression model
    * Model finds linear combination of polynomial features
    * Effectively fitting high-degree polynomial to data

  * **Overfitting Risk**
    * Higher degree expansion → lower training error
    * But: Risk of overfitting to training set
    * Bias-variance tradeoff applies directly
    * Very high degree = perfect fit to training data = poor generalization

  * **Student Question: How to Choose Polynomial Degree?**
    * Critical hyperparameter tuning question
    * Answer (well explained by student Clarice):
      * More degrees = more complex model = better training fit
      * But comes at cost of overfitting
      * Classic model complexity tradeoff

  * **Hyperparameter Tuning Process**
    * **DON'T touch test set** (reinforce this concept)
    * Procedure:
      1. Split training set further (train/validation)
      2. Train models with different polynomial degrees (1, 2, 3, ...)
      3. Evaluate on validation set (held-out portion of training)
      4. Use K-fold cross-validation for robustness
      5. Monitor training error vs. validation error
      6. Watch for divergence (sign of overfitting)
    * Stop when validation error starts increasing
    * Best degree: Minimizes validation error
    * This is hyperparameter optimization (degree is parameter of the model)

  * **Connection to Previous Concepts**
    * Same principle as bias-variance tradeoff curve from Meeting 8
    * Training error decreases with complexity
    * Validation/test error has U-shape (sweet spot)
    * Find the minimum of validation error curve

  * **Professor's Note**
    * Could make great problem set question
    * "Build pipeline to test hyperparameter" (degree selection)
    * Good second midterm question candidate

* **Model Complexity Control - Regularization**

  * **Two Sources of Model Complexity in Linear Regression**
    1. Basis expansion (polynomial degree)
    2. Number of non-zero coefficients (how many features used)

  * **Ridge Regression (L2 Regularization)**
    * **The Problem**
      * Pass model 20 features
      * Model uses all 20 to minimize prediction error
      * May only need 3 features for good prediction
      * Want to reduce dimensionality and prevent overfitting

    * **The Solution: Penalty Term**
      * Modified error function: RSS + λ × Σ(coefficients²)
      * Standard part: Σ(Ŷᵢ - Yᵢ)² (prediction error)
      * Penalty part: λ × Σ(βⱼ²) (complexity penalty)
      * Sum over squares of all feature coefficients

    * **Penalty Parameter: λ (lambda)**
      * Scikit-learn calls it `alpha`
      * Controls trade-off between accuracy and complexity
      * λ = 0: No penalty, standard linear regression (may use all features)
      * High λ: Strong penalty, prefers fewer features (drives coefficients to zero)
      * "Turning the knob" on how much you care about model complexity

    * **How It Reduces Complexity**
      * Non-zero coefficient = feature is being used
      * Zero coefficient = feature is ignored
      * Penalty encourages coefficients to be zero
      * Reduces effective dimensionality of model
      * Fewer features = simpler model = less overfitting

    * **Hyperparameter Tuning for λ**
      * Same procedure as polynomial degree selection
      * Use cross-validation on training set
      * Don't touch test set
      * Try different λ values, monitor validation error
      * Select λ that minimizes validation error

  * **Student Question: Why Not Individual λ per Feature?**
    * **Question**: Why single global λ instead of λⱼ for each feature?
    * **Answer - Multiple Perspectives**:
      1. **Could do it** - technically possible, gives more control
         * Would allow weighting specific features differently
         * More flexibility in feature selection
      2. **Drawbacks**:
         * Hyperparameter space becomes very large
         * With N features, need to tune N different λ values
         * Becomes combinatorial search problem
         * No longer has closed-form solution
         * Computationally expensive
      3. **Standard approach benefits**:
         * Single parameter to tune (simpler)
         * Philosophy: "Don't care which features to zero out, just want fewer"
         * More features used = worse, regardless of which ones
      4. **What matters most**: Feature either in or out
         * Small non-zero coefficient still keeps feature in model
         * Still have high-dimensional model (doesn't help much)
         * Real goal: Drop features completely (zero them out)
      5. **Alternative approach**: Could search which features to drop
         * Treat as feature selection problem
         * But becomes combinatorial
         * Wouldn't be solvable in closed form

  * **Related Regularization Techniques**
    * **Lasso** (L1 regularization)
      * Uses absolute values instead of squares
      * More aggressive at zeroing coefficients
    * **Elastic Net**
      * Combination of L1 and L2
      * Gets benefits of both
    * Tradeoffs between techniques not critical for this class
    * All share same concept: penalize complexity

* **Real-World Application Example**
  * **Google Web Server Deployment Paper**
    * Reference: Section 5 of previously assigned paper
    * Application: Content Delivery Network (CDN) response time prediction
    * Question: How fast will web search results return?
    * Configuration changes: Adding/moving servers, redirecting traffic

  * **Technique Used: Kernel Regression**
    * Uses radial basis functions (type of basis expansion)
    * Basically linear regression with basis expansion
    * "Not much more complicated" than hands-on exercise
    * Standard regression approach, proven in production

* **Introduction to Logistic Regression (Preview for Next Session)**

  * **What Is Logistic Regression?**
    * Also a linear model (like linear regression)
    * Designed for classification instead of regression
    * Typically binary classification (two-class problems)
    * Examples: 0/1, yes/no, spam/not spam, query/response

  * **When It Works Well vs. Poorly**
    * **Works well**: Linearly separable data
      * Top plot example: clear decision boundary
      * Low X values → Class 0
      * High X values → Class 1
      * Can draw line to separate classes
    * **Works poorly**: Non-linearly separable data
      * Example: XOR pattern (can't separate with line)
      * Historically motivated neural networks
      * No linear model handles this well
    * Fortunately: Network data often has linear separability

  * **The Sigmoid Function (Key Component)**
    * **The Challenge**: Lines go to infinity in both directions
      * But we only care about predictions in range [0, 1]
      * Need to map linear function to probability range

    * **The Solution**: S-shaped sigmoid function
      * Maps any real number to range [0, 1]
      * Formula produces characteristic S-curve
      * Nice properties:
        * Maximum value = 1 (certain positive)
        * Minimum value = 0 (certain negative)
        * Middle = 0.5 (uncertain)
        * Smooth transition between classes

    * **Trade-off**: Math becomes non-convex
      * Can't take clean derivatives
      * No closed-form solution (unlike linear regression)
      * Must use iterative optimization
      * Slower training than linear regression

    * **Important Note**: Will see sigmoid again in deep learning
      * Fundamental building block of neural networks
      * Same S-shaped function

  * **Common Use Cases**
    * Binary classification (most common)
    * Can be extended to multi-class
    * Two-class is standard application

  * **Preview of Hands-On #11: DNS Classification**
    * Task: Classify DNS queries vs. responses
    * Feature: Packet size in bytes
    * Observation: Queries typically smaller than responses
    * Natural binary classification problem

    * **Alternative Application Discussed**:
      * Could use logistic regression on linear regression data
      * Classify: Data packet vs. Acknowledgement
      * Feature: Byte count
      * Larger bytes → Data packet
      * Smaller bytes → Acknowledgement
      * Would be more coherent approach than two linear models

  * **Next Session Plan**
    * Start with Hands-On #11 (Logistic Regression)
    * Finish logistic regression coverage
    * Support Vector Machines (no hands-on)
    * Decision Trees and Ensembles (professor's favorite)
    * Goal: End of Week 6 = through Topic 12

### Meeting 12

* **Decision Trees**
  * **Basic Concept and Representation**
    * Dividing feature space into regions through sequential splits
    * Tree structure: internal nodes (decisions), leaf nodes (predictions/classifications)
    * Splits based on feature thresholds (for numerical features)
    * Visual representation: decision boundaries in feature space
  * **Decision Trees for Regression vs. Classification**
    * Regression: Predict continuous values (e.g., baseball player salary)
      * Leaf nodes contain mean of training points in that region
      * Evaluate using Residual Sum of Squares (RSS)
    * Classification: Predict categorical outcomes (e.g., dog/cat/horse)
      * Leaf nodes contain class labels
      * Evaluate using classification error, Gini index, or entropy
  * **How Splits Are Determined**
    * For regression: Minimize RSS in resulting regions
    * For classification: Minimize Gini index or entropy (measure of class purity)
    * Exhaustive search over features and thresholds
    * Greedy approach: choose best split at each node
  * **Model Complexity and Tree Depth**
    * Tree depth = number of sequential splits
    * Deeper trees = more complex models
    * Maximum depth: each training point in its own region (perfect training accuracy, severe overfitting)
    * Growing strategy: grow large tree, then prune back
    * Can't just stop at local minimum (might miss important deeper splits)
  * **Advantages of Decision Trees**
    * Extremely easy to understand and interpret
    * Explainable: can trace path through tree to explain any prediction
    * Graphically intuitive
    * Handles mixed feature types (qualitative + quantitative) without preprocessing
    * No need to normalize/scale features
    * Mirrors human decision-making processes
  * **Disadvantages of Decision Trees**
    * Very sensitive to small changes in training data (brittle)
    * Initial splits cascade through entire tree structure
    * Generally poor performance without ensemble methods
    * High variance (small data changes cause big model changes)
    * Almost never used alone in modern ML applications

* **Ensemble Methods**
  * **General Concept**
    * Train multiple models instead of relying on single model
    * Combine predictions through voting (classification) or averaging (regression)
    * Key requirement: models must be different (can't just train same model 10 times)
    * Introduce variation through:
      * Different training data subsets (bagging)
      * Different feature subsets (random forests)
      * Sequential error correction (boosting)
  * **Bagging (Bootstrap Aggregation)**
    * Statistical technique: bootstrapped aggregation → "bagging"
    * **Process**:
      * Create multiple training sets through random sampling with replacement
      * Sample size = original training set size
      * Typically captures ~2/3 of unique data points (with repeats)
      * Train separate tree on each bootstrap sample
      * Aggregate: vote (classification) or average (regression)
    * Reduces variance compared to single decision tree
    * Still using same features at each split

* **Random Forests**
  * **Key Innovation Beyond Bagging**
    * Bagging PLUS random feature selection at each split
    * At each node: consider only random subset of features for splitting
    * In 2D example: flip coin to decide whether to split on X1 or X2
    * In high dimensions: randomly select subset of features per split
    * Further decorrelates trees (reduces correlation between ensemble members)
  * **Complete Random Forest Algorithm**
    * Step 1: Create bootstrap sample (random sampling with replacement)
    * Step 2: Build tree, but at each split:
      * Randomly select subset of features
      * Choose best split from only those features
      * Still optimize threshold within selected features
    * Step 3: Repeat for many trees (typical: 100-500 trees)
    * Step 4: Aggregate predictions across all trees
  * **Feature Importance**
    * Random forests provide feature importance scores
    * Based on average reduction in RSS (or Gini/entropy) per feature across all trees
    * Helps identify which features drive predictions
    * Useful even if not using random forest as final model (exploratory analysis)
  * **Advantages of Random Forests**
    * Much more accurate than single decision trees
    * Very robust to changes in training data
    * Efficient to train and extremely fast for inference
    * No need to normalize or scale features
    * Handles mixed feature types naturally
    * Provides feature importances
    * One of the most effective out-of-the-box classifiers
    * Professor's go-to first model to try on new problems
  * **Hyperparameters**
    * Number of trees in forest
    * Maximum tree depth (controls complexity)
    * More trees + deeper trees = more complex = more potential for overfitting
    * Same bias-variance tradeoff considerations as other models

* **Boosting (Brief Overview)**
  * **Different Ensemble Approach**
    * Sequential rather than parallel
    * Uses short "stubby" trees (depth 1-3)
    * Focus on correcting errors from previous trees
  * **Algorithm**
    * Train initial shallow tree on data
    * Identify misclassified examples
    * Increase weights on misclassified points
    * Train next tree on reweighted data (emphasizes previous errors)
    * Add to ensemble and repeat
  * **Hyperparameters**
    * Number of trees (more trees = more complexity)
    * Tree depth per boosted tree (typically very shallow)
    * Shrinkage parameter (learning rate / sensitivity to errors)
  * **Note**: Covered for completeness but not emphasized for exam

* **Hands-On Activity: IoT Privacy with Random Forests (#12)**
  * **Application Context**
    * Research on privacy risks in IoT network traffic
    * Question: Can network traffic reveal in-home activities?
    * Collaboration with PhD student Noah Apthorpe
    * Published research papers on this topic
  * **Specific Problem: Motion Detection via Network Traffic**
    * Device: Nest camera (sends video clips to cloud when motion detected)
    * Observable: Network traffic rates increase when motion occurs
    * Goal: Classify camera state (motion vs. idle) from traffic volume
    * Privacy implication: Encrypted traffic still reveals behavioral patterns
  * **Dataset**
    * Packet captures from Nest camera converted to traffic rates
    * Time series of upload/download rates
    * Labels: motion events vs. idle periods
  * **Note on Problem Complexity**
    * Intentionally simple ML problem (could solve with threshold)
    * Focus on learning concepts, not struggling with difficult modeling
    * Most notebooks use "easy" problems so students can focus on techniques
  * **Tasks in Notebook**
    * Data already formatted (provided to students to avoid timezone/formatting issues)
    * Train decision tree classifier
    * Train random forest classifier
    * Compare performance
    * Explore feature importance (if time permits)
    * Optional: Generate confusion matrix
  * **Discussion Topics**
    * Privacy risks from traffic analysis even with encryption
    * Countermeasures: traffic padding, rate smoothing
    * Future topic: Privacy and ML (potentially Week 9)

### Meeting 13

* **Introduction to Deep Learning and Neural Networks**

  * **Course Positioning and Context**
    * Transition from traditional ML (regression, trees, ensembles) to deep learning
    * Building on previous supervised learning concepts
    * Most complex topic of the course
    * Will continue through Meeting 14 (representation learning)
    * Focus: Understanding fundamentals, not implementing from scratch

  * **Why Deep Learning for Network Data?**
    * **Traditional approach limitations**
      * Manual feature engineering required for every problem
      * Domain expertise needed to design good features
      * Flow statistics, timing features, packet counts - all hand-crafted
      * Time-consuming and problem-specific
    * **Deep learning advantage**
      * Learns features automatically from raw data
      * Representation learning: discovers useful features during training
      * Can work with minimal preprocessing
      * Same architecture applies across many problems

  * **Key Tool: nPrint (Packet-as-Bitmap Representation)**
    * Converts raw packets into bitmap/image-like representation
    * Feed packets directly to neural network
    * No manual feature extraction needed
    * Will cover in more detail later (Meeting 17)
    * Enables end-to-end learning from raw network data

* **Neural Network Fundamentals**

  * **The Biological Neuron Analogy**
    * Brain neuron: dendrites (inputs) → soma (processing) → axon (output)
    * Weighted inputs combine at cell body
    * Fires signal when combined input exceeds threshold
    * Output travels to other neurons via axon
    * Note: Analogy breaks down quickly; artificial neurons quite different

  * **The Artificial Neuron (Perceptron)**
    * **Components**:
      * Inputs: x₁, x₂, ..., xₙ (features)
      * Weights: w₁, w₂, ..., wₙ (learned parameters)
      * Bias: b (learned parameter, like intercept in regression)
      * Weighted sum: z = Σ(wᵢxᵢ) + b
      * Activation function: f(z) → output
    * **Mathematical operation**: output = f(Σ(wᵢxᵢ) + b)
    * Single neuron is essentially weighted linear combination + non-linearity

  * **Activation Functions (Critical Component)**
    * **Why needed**: Without activation function, network is just linear regression
      * Stacking linear operations stays linear
      * Can't learn complex patterns
      * Need non-linearity to capture real-world relationships

    * **Sigmoid (σ)**
      * Formula: σ(z) = 1 / (1 + e⁻ᶻ)
      * S-shaped curve, output range [0, 1]
      * Smooth gradient everywhere
      * **Problems**:
        * Saturates at extremes (gradient → 0)
        * Causes vanishing gradient problem
        * Outputs not zero-centered
      * Historical importance: used in early neural networks
      * Still used in final layer for binary classification
      * Same function from logistic regression

    * **Tanh (Hyperbolic Tangent)**
      * Formula: tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
      * S-shaped like sigmoid
      * Output range [-1, 1]
      * Zero-centered (advantage over sigmoid)
      * Still saturates at extremes (vanishing gradient issue)
      * Better than sigmoid for hidden layers but still problematic

    * **ReLU (Rectified Linear Unit)** - Most Common Today
      * Formula: ReLU(z) = max(0, z)
      * Extremely simple: outputs input if positive, else zero
      * **Advantages**:
        * Very fast to compute
        * No saturation for positive values
        * Sparse activation (many neurons output zero)
        * Gradient is either 0 or 1 (simple backprop)
        * Works extremely well empirically
      * **Disadvantage**: "Dying ReLU" problem
        * Neurons can get stuck outputting zero
        * Gradient is zero for z < 0
        * Once dead, stays dead
      * Default choice for hidden layers in modern networks

    * **Leaky ReLU and Variants**
      * Formula: max(0.01z, z) or similar
      * Small negative slope for z < 0
      * Addresses dying ReLU problem
      * Variations: Parametric ReLU, ELU, etc.

    * **Key Takeaway**: Activation function choice matters
      * ReLU family most common in modern practice
      * Sigmoid only for output layer in binary classification
      * This is a design choice / hyperparameter

  * **Single Neuron Limitations**
    * Can only learn linear decision boundaries (even with activation function)
    * Example: Cannot solve XOR problem
    * Famous historical result that motivated multi-layer networks
    * Need multiple neurons in multiple layers for complex patterns

* **Feed-Forward Neural Networks (Multi-Layer Perceptrons)**

  * **Network Architecture**
    * **Input layer**: Raw features (not really a "layer" - just the data)
    * **Hidden layer(s)**: One or more layers of neurons
      * Each neuron receives all inputs from previous layer
      * Applies weights, bias, and activation function
      * "Hidden" because not directly observable (not input or output)
    * **Output layer**: Final predictions
      * Regression: typically one neuron, no activation (or linear)
      * Binary classification: one neuron with sigmoid activation
      * Multi-class: multiple neurons with softmax activation

  * **Fully Connected (Dense) Layers**
    * Each neuron connects to every neuron in previous layer
    * Most common layer type in basic neural networks
    * "Fully connected" and "dense" are synonymous
    * Number of weights = (neurons in layer i) × (neurons in layer i-1)

  * **Network Depth and Width**
    * **Width**: Number of neurons per layer
    * **Depth**: Number of layers
    * More depth/width = more parameters = more capacity = more complex patterns
    * But: more risk of overfitting, longer training time

  * **Why Multiple Layers Work**
    * First layer: learns simple features
    * Second layer: combines simple features into complex ones
    * Deep layers: hierarchical feature learning
    * Example in vision: edges → shapes → object parts → full objects
    * Network automatically learns feature hierarchy
    * This is "representation learning"

* **Training Neural Networks**

  * **Forward Propagation**
    * Input data flows through network layer by layer
    * Each layer computes: activation(weights × inputs + bias)
    * Final layer produces prediction
    * Deterministic process once weights are set

  * **Loss Functions**
    * **Regression**: Mean Squared Error (MSE)
      * L = (1/n) Σ(ŷᵢ - yᵢ)²
      * Same as linear regression
    * **Binary Classification**: Binary Cross-Entropy
      * Measures difference between predicted probability and true label
    * **Multi-class Classification**: Categorical Cross-Entropy
      * Extension to multiple classes
    * Loss function measures "how wrong" the predictions are

  * **Gradient Descent Optimization**
    * Goal: Minimize loss function by adjusting weights
    * **Algorithm**:
      1. Initialize weights randomly
      2. Forward pass: compute predictions and loss
      3. Compute gradient of loss with respect to each weight
      4. Update weights: w_new = w_old - learning_rate × gradient
      5. Repeat until convergence
    * **Learning rate**: Hyperparameter controlling step size
      * Too large: overshoots minimum, unstable
      * Too small: very slow convergence
      * Typical values: 0.001 to 0.01

  * **Backpropagation (The Key Algorithm)**
    * **What it does**: Efficiently computes gradients for all weights
    * **How it works**:
      * Start at output layer
      * Compute gradient of loss with respect to output
      * Use chain rule to propagate gradient backward through network
      * Each layer computes gradient with respect to its weights
      * Propagates gradient to previous layer
    * **Chain rule application**:
      * Gradient flows backward through activation functions
      * Through weight matrices
      * Through all layers back to input
    * **Efficiency**: Single backward pass computes all gradients
      * Much faster than computing each gradient independently
      * Makes training deep networks feasible
    * **Key innovation**: Made deep learning practical

  * **Training Epochs and Batches**
    * **Epoch**: One complete pass through entire training dataset
    * **Batch**: Subset of training data used for one gradient update
    * **Mini-batch gradient descent** (most common):
      * Split training data into batches (e.g., 32, 64, 128 examples)
      * Compute gradient on batch
      * Update weights
      * Move to next batch
    * **Why batching**:
      * More stable gradient estimates than single example
      * Faster than waiting for full dataset
      * Enables parallelization on GPU
    * **Batch size**: Another hyperparameter to tune

  * **Stochastic vs. Batch Gradient Descent**
    * **Batch (Full-batch)**: Use all training data per update
      * Most accurate gradient
      * Very slow for large datasets
    * **Stochastic (SGD)**: Use one example per update
      * Fast but noisy
      * Can escape local minima due to noise
    * **Mini-batch**: Middle ground (most common in practice)

* **Challenges in Training Deep Networks**

  * **Vanishing Gradient Problem**
    * **What happens**:
      * Gradients get smaller as they propagate backward
      * Early layers receive tiny gradients
      * Weights barely update
      * Network can't learn
    * **Causes**:
      * Sigmoid/tanh activation functions saturate
      * Gradient < 1 multiplied many times → approaches zero
      * Especially bad in deep networks (many layers)
    * **Impact**: Deep networks train very slowly or not at all
    * **Solutions**:
      * ReLU activation (gradient is 1 for positive values)
      * Batch normalization
      * Residual connections (skip connections)
      * Better weight initialization

  * **Exploding Gradient Problem**
    * Opposite problem: gradients become very large
    * Weight updates too big
    * Training becomes unstable
    * Weights can become NaN (not a number)
    * **Solutions**:
      * Gradient clipping (cap maximum gradient value)
      * Careful weight initialization
      * Lower learning rate

  * **Overfitting in Neural Networks**
    * Very high capacity models (millions of parameters)
    * Can memorize training data
    * Poor generalization to test data
    * **Solutions**:
      * Regularization (L1, L2 penalties on weights)
      * Dropout (randomly zero out neurons during training)
      * Early stopping (stop training when validation error increases)
      * Data augmentation
      * More training data

  * **Local Minima and Saddle Points**
    * Loss landscape is non-convex (unlike linear regression)
    * Many local minima exist
    * Gradient descent can get stuck
    * In practice: less of a problem than expected
      * High-dimensional spaces have fewer problematic local minima
      * SGD noise helps escape bad local minima

* **Practical Considerations and Hyperparameters**

  * **Network Architecture Choices**
    * Number of layers (depth)
    * Number of neurons per layer (width)
    * Activation functions
    * No universal formula - requires experimentation
    * Start simple, add complexity if needed

  * **Key Hyperparameters to Tune**
    * Learning rate (most important)
    * Batch size
    * Number of epochs
    * Network architecture (layers, neurons)
    * Regularization strength
    * Dropout rate
    * Choice of optimizer (Adam, SGD, RMSprop)

  * **Training Tips**
    * **Always start simple**: Small network, few layers
    * **Monitor training and validation loss**: Watch for overfitting
    * **Use validation set**: Don't touch test set during development
    * **Visualize training progress**: Plot loss curves
    * **Normalize inputs**: Scale features to similar ranges
    * **Initialize weights carefully**: Random but not too large/small
    * **Use Adam optimizer**: Good default choice (adaptive learning rate)

* **Why Deep Learning for Network Data?**

  * **Comparison with Traditional ML**
    * **Traditional (e.g., Random Forest)**:
      * Manually extract flow statistics
      * Engineer domain-specific features
      * Feed features to classifier
      * Interpretable and fast
    * **Deep Learning**:
      * Feed raw packets (via nPrint representation)
      * Network learns features automatically
      * End-to-end learning
      * More complex, less interpretable

  * **When to Use Deep Learning**
    * Large amounts of training data available
    * Complex patterns in data
    * Raw data is high-dimensional (images, packets, audio)
    * Manual feature engineering is difficult
    * State-of-the-art performance needed

  * **When Traditional ML May Be Better**
    * Small dataset
    * Need interpretability
    * Limited computational resources
    * Domain features are well understood
    * Fast inference required
    * Random forests often "good enough"

* **Deep Learning in Network Security and Management**

  * **Traffic Classification**
    * Identify application from encrypted traffic
    * Netflix vs. YouTube vs. web browsing
    * nPrint + CNN: learns packet-level patterns

  * **Intrusion Detection**
    * Detect malicious traffic patterns
    * Can learn complex attack signatures
    * Challenge: labeled attack data is scarce

  * **QoE Prediction**
    * Video quality inference from traffic
    * Assignment 1 used traditional features
    * Could use deep learning with more data

  * **Malware Detection**
    * Network behavior patterns
    * DNS query sequences
    * Packet timing and sizes

  * **Encrypted Traffic Analysis**
    * Can't inspect payload due to encryption
    * Deep learning on metadata: packet sizes, timing, counts
    * Privacy vs. security tradeoff

* **Preview of Next Session (Meeting 14)**
  * **Representation Learning** (deeper dive)
    * What do hidden layers actually learn?
    * Visualizing learned features
    * Transfer learning
  * **Convolutional Neural Networks (CNNs)**
    * Designed for spatial/structured data
    * How they work with network packets
    * nPrint application
  * **Recurrent Neural Networks (RNNs)**
    * Sequential data (time series, text)
    * Application to network time series
  * **Autoencoders**
    * Unsupervised representation learning
    * Dimensionality reduction
    * Anomaly detection in networks

* **Key Concepts to Remember**
  * **Neuron**: Weighted sum + bias + activation function
  * **Activation functions**: Introduce non-linearity (ReLU most common)
  * **Feed-forward network**: Input → hidden layers → output
  * **Forward propagation**: Data flows through network
  * **Backpropagation**: Gradients flow backward to update weights
  * **Gradient descent**: Iterative weight updates to minimize loss
  * **Epochs**: Full passes through training data
  * **Vanishing/exploding gradients**: Major training challenges
  * **Representation learning**: Networks learn features automatically
  * **Deep learning advantage**: No manual feature engineering needed


### Meeting 14

* **Learning Traffic Representations with nPrint**

  * **Motivation: The Traditional ML Pipeline Problem**
    * Every network traffic analysis task requires reinventing the entire pipeline
    * Example workflows that require separate pipelines:
      * Application identification (Netflix vs. YouTube vs. web browsing)
      * Attack detection (port scans, denial of service, exploits like Log4J)
    * Typical pipeline steps that must be repeated:
      1. Collect and label traffic data
      2. Design task-specific features (packet sizes, inter-arrival times, connection patterns, rates)
      3. Engineer features (write code to extract them from packet traces)
      4. Train models
      5. Start over for next problem

  * **Why Feature Engineering is Expensive**
    * Requires significant time investment
    * Demands domain expertise
    * Takes effort even with modern tools and libraries
    * Question posed: Could we generalize this pipeline?
      * Throw traffic at model with labels → get results
      * Let deep learning learn features automatically

  * **Reproducibility Challenges in Network ML Research**
    * **DARPA'98 Dataset Example**
      * Famous intrusion detection dataset from 1998
      * Two different research papers using same exact dataset
      * Both analyzing denial of service traffic
      * Reported packet counts differing by factor of 10
      * Same data, same stated approach, completely different numbers
    * **Root causes of inconsistency**:
      * Different feature extraction code
      * Different flow definitions
      * Different metadata interpretations
      * Differences not apparent from published descriptions
    * **Note**: AI agents may improve this (more consistent code generation from specs)

  * **nPrint: A Tool for Standardized Traffic Representation**
    * **Core idea**: Inspired by deep learning in computer vision
      * If models can learn from raw pixels in images
      * Why not throw raw packets at deep learning models?
    * **What nPrint does**:
      * Converts packet traces (pcap files) to standardized bitmap representations
      * Input: Packet trace
      * Output: Fixed-width bitmap where each bit position has consistent meaning

  * **nPrint Bitmap Encoding**
    * **Three-valued representation**:
      * **1**: Bit is set in packet header
      * **0**: Bit is not set in packet header
      * **-1**: Header field not present in this packet
    * **Why -1 is critical: The Alignment Problem**
      * Every bit position must mean the same thing across all packets
      * Example alignments:
        * Bits 0-3: Always IPv4 version field
        * Bits 96-127: Always source IP address
        * Bits 320-335: Always TCP options (even if not present)
      * Without alignment, models cannot learn consistent patterns
      * If bit 80 is source port in one packet but TTL in another → model fails
    * **Size considerations**:
      * Representation inflated by factor of ~2x
      * Includes all possible header fields even if not present in packet
      * Ensures alignment but increases data size

  * **Using nPrint: Workflow and Command Examples**
    * **Basic workflow**:
      1. Generate nPrints from pcap file
      2. Generate labels for classification task
      3. Train classifier (can use any ML model, not just deep learning)
      4. Analyze feature importance (which bits/header fields matter)
      5. Experiment with different nPrint configurations

    * **Command-line usage examples**:
      ```bash
      # Generate nPrint with first 30 bytes of payload
      nprint -P 30 input.pcap > output.csv

      # Include IPv4 and TCP headers with payload
      nprint -P 30 -4 -t input.pcap > output.csv
      ```

    * **nPrint flags**:
      * `-P N`: Include first N bytes of payload
      * `-4`: Include IPv4 header
      * `-6`: Include IPv6 header
      * `-t`: Include TCP header
      * Can mix and match to control which headers are included

    * **Output format**:
      * CSV file where each row = one packet
      * Each column = one bit position in standardized representation
      * Typically hundreds of features (bits) per packet
      * Can load directly into pandas DataFrame

  * **Example Application: Log4J Scan Detection**
    * **Task**: Classify network traffic as "benign" or "scan"
    * **Approach**:
      * Generate nPrint representations of traffic
      * Label scan traffic vs. legitimate web traffic
      * Train random forest classifier
      * Examine feature importance

    * **Feature importance analysis revealed**:
      * Destination port (bits corresponding to TCP destination port field)
      * Source port (bits corresponding to TCP source port field)
      * TCP flags (SYN, ACK, FIN bits)
      * TCP window size (bits in window size field)
      * Certain TCP options fields

    * **Critical questions to ask**:
      * Do these features make sense for distinguishing scans from legitimate traffic?
      * Are port numbers genuinely informative or spuriously correlated?
      * Are TCP options fields truly relevant or is model overfitting?
      * This requires domain expertise even with automated feature learning

  * **Benefits of nPrint Approach**
    * **Generalization across tasks**: Same representation works for multiple problems
      * Application identification
      * Attack detection
      * QoS inference
      * Don't need to redesign feature extraction
    * **Reproducibility**: Different researchers generate identical representations from same pcap
      * Eliminates major source of experimental variability
    * **Automation**: Deep learning models can identify important header fields automatically
      * Reduces need for manual feature engineering
    * **Feature importance analysis**: Models can reveal which specific bits drive decisions
      * Provides interpretability even with complex models

  * **Limitations and Considerations**
    * **Representation size**:
      * Very large (roughly 2x size of original packet headers)
      * For millions of packets, creates storage and memory challenges
      * Includes all possible headers even if not present

    * **Spurious correlations**:
      * nPrint includes all header fields indiscriminately
      * Models may latch onto patterns that don't generalize
      * Example: All training examples from one IP address range
        * Model might learn to recognize IP range instead of application behavior
      * Requires careful data collection, cross-validation, feature importance analysis

    * **Temporal relationships not automatically encoded**:
      * nPrint represents individual packets as independent bitmaps
      * Doesn't inherently encode packet ordering or causality
      * Example: Connection setup should come before connection teardown
      * Models might learn these patterns from data, but not guaranteed
      * Capturing temporal relationships requires:
        * More sophisticated architectures (recurrent networks, attention mechanisms)
        * Or explicit feature engineering to combine multiple packets

    * **Not "magic pixie dust"**:
      * Simply feeding nPrint to deep learning doesn't guarantee good results
      * Still need proper model architecture
      * Still need hyperparameter tuning
      * Still need quality training data
      * Deep learning is powerful but requires careful application

  * **The Role of AI-Assisted Feature Engineering**
    * **Historical context**: Manually writing feature extraction code was tedious
    * **Modern reality**: AI assistants can generate code from natural language in seconds
    * **Implications**:
      * Overhead of custom feature engineering has decreased
      * Still, nPrint offers advantages:
        * Reproducibility across research groups
        * Standardization for comparisons
        * Discovery of unexpected features not in human/AI design space
    * **Future likely involves hybrid approach**:
      * Use nPrint for baseline models and reproducible comparisons
      * Use AI-assisted feature engineering for task-specific optimization

  * **nPrint Website and Resources**
    * Website provides:
      * Tool installation instructions (must compile from source, C language)
      * Pre-formatted datasets for common classification tasks
      * Example applications and benchmarks
    * Can be used for course projects
    * Building nPrint:
      * Requires C compiler
      * Uses autoconf/configure build system
      * May need dependencies: `brew install autoconf automake pkgconfig pcap`
      * MacOS-specific issue: May need to specify paths to pcap libraries

  * **Hands-On Activity: nPrint for Scan Detection**
    * **Part 1: Generate nPrints**
      * Install nPrint tool (compile from source)
      * Process pcap files to generate bitmap representations
      * Takes ~5-10 minutes to get building properly

    * **Part 2: Train Classifier**
      * Load nPrint CSV into DataFrame
      * Generate labels (scan vs. benign)
      * Train random forest or other classifier
      * Typically random forest works well

    * **Part 3: Analyze Feature Importance**
      * Examine which bit positions (header fields) were most important
      * Interpret what those bits represent in protocol headers
      * Ask: Does this make sense for the task?

    * **Part 4: Experiment with Different Representations**
      * Try different nPrint flags (-4, -6, -t, etc.)
      * Compare performance with different header combinations
      * Understand which headers contain most informative features

    * **Part 5: Advanced (Optional)**
      * Could explore PcapML (mentioned but not required)
      * Integration with other tools

  * **Future Directions: Generative Models for Traffic**
    * **Text-to-image inspiration**: "Make me a painting in style of Picasso"
    * **Network traffic equivalent**: "Make me traffic in style of denial of service attack"
    * Graduate student research (Chase and others) on this topic
    * Using diffusion models to generate traffic from nPrint representations
    * **Challenges**:
      * Generated traffic may have invalid checksums
      * May lack proper connection establishment sequences
      * Similar to early AI image generation problems (people without thumbs)
      * Active research area
    * Will be covered more in future session on generative AI

  * **Key Takeaways**
    * nPrint enables representation learning for network traffic
    * Automates feature engineering but doesn't eliminate need for domain knowledge
    * Provides reproducibility and standardization benefits
    * Large representation size and potential spurious correlations are concerns
    * Feature importance analysis remains critical
    * Hybrid approaches with AI-assisted engineering may be future direction
    * Not a silver bullet, but valuable tool in network ML toolkit

### Meeting 15

* **Course Context**
  * Week 8 of course
  * Covering unsupervised learning techniques
  * May skip autoencoders to cover generative AI topics instead

* **Introduction to Unsupervised Learning**
  * Definition: Models that do not rely on labels to learn or analyze
  * Doesn't mean data doesn't have labels - just not used in training
  * Can still evaluate with labeled data if available
  * Focus on dimensionality reduction and clustering

* **Dimensionality Reduction Overview**
  * **Goal**: Represent high-dimensional data with smaller number of dimensions
  * **Applications**:
    * Understanding data better (which features are most important)
    * Visualization (reduce to 2D or 3D for plotting)
    * Reducing noise and compression (lossy but acceptable)
    * Preprocessing for clustering
    * Preprocessing for supervised learning (reduce complexity)

* **Principal Component Analysis (PCA)**
  * **Core Concept**:
    * Change of basis - representing data as linear combination of different features
    * Assumes linear relationships in data
    * Find most important features/directions for describing data points
  * **Mathematical Intuition**:
    * Two ways to view it:
      1. **Maximum variance**: Find direction that captures most variance in data
      2. **Minimum projection distance**: Minimize distance when projecting points onto lower-dimensional space
    * These are equivalent approaches
  * **Principal Components**:
    * First PC: Direction of maximum variance
    * Second PC: Typically orthogonal to first, captures remaining variance
    * Size of components (eigenvalues) indicates variance in each direction
  * **Choosing Number of Components**:
    * Scree plot: Plot variance explained vs. number of components
    * Look for "elbow" or "knee" in curve
    * Consider domain knowledge (e.g., if looking for 2 classes, try 2 components)
    * Explained variance: How much of total variance captured by N components
  * **Spectral Clustering with PCA**:
    * Can use PCA for clustering
    * Each point has components in PC1, PC2, etc.
    * Points with more of PC1 → cluster 1, more of PC2 → cluster 2
    * Number of PCs chosen = number of clusters
  * **Extensions**:
    * Kernel PCA: Handles non-linear relationships
    * Apply functions to PCA to express non-linear patterns

* **T-SNE (t-distributed Stochastic Neighbor Embedding)**
  * **Purpose**: Non-linear dimensionality reduction
  * **Key Features**:
    * Works on non-linear data sets
    * Commonly used for visualization
    * Math classes often stop before this (hard to prove properties)
  * **When to Use**:
    * Try alongside PCA to see which works better
    * Good for visualizing complex, non-linear patterns
  * **Evaluation Challenge**:
    * Hard to say if it "worked" without labels
    * Did it help you understand something?
    * Do clusters match to labels if you have them?
  * **Example**: DNS packet visualization
    * May or may not produce meaningful separation

* **Autoencoders**
  * **Architecture**:
    * Deep neural network with encoder and decoder
    * Encoder: Reduces dimensionality (drops coefficients)
    * Bottleneck: Compressed representation (reduced dimensionality)
    * Decoder: Attempts to reconstruct original input
  * **Training Process**:
    * Goal: Make output match input as closely as possible
    * If decoder can reconstruct from bottleneck, encoder did good job
    * Decoder is mainly for training the encoder
  * **Advantages over PCA/t-SNE**:
    * Don't need to think about feature engineering
    * Can work with raw data
    * Similar to deep learning vs. random forests in supervised learning
  * **Disadvantages**:
    * Much more expensive computationally
    * Requires more data and training time
  * **Applications**: Same as PCA/t-SNE (visualization, compression, preprocessing)

* **Hands-On Activity**
  * Notebook on dimensionality reduction
  * May focus on clustering notebook instead if covering both topics

### Meeting 16

* **Guest Lecture: Generative AI and Diffusion Models (Chase)**
  * **Speaker**: Chase (PhD student working on AI for networking)
  * **Topic**: Diffusion models for generating synthetic network traffic

* **Motivation and Challenges**
  * **Network Data Scarcity Problem**:
    * Network datasets are limited due to privacy concerns, maintenance costs, collection problems
    * Example: Sharing UChicago network traffic could expose network topology, router hierarchy
    * Need for synthetic data generation to address scarcity while preserving privacy
  * **Use Cases for Synthetic Data**:
    * Training machine learning models with insufficient real data
    * Augmenting datasets (e.g., scaling 100 traces to 10,000)
    * Testing hardware/software, protocol interoperability
    * Privacy-preserving data sharing

* **Early Methods for Synthetic Traffic Generation**
  * **Simulators**: NS3, GANS, MJANG
    * Focus on retransmitting existing traffic with modified metadata (IPs, timestamps, rates)
    * Limited utility for ML: copies don't add variation needed for model robustness
  * **GAN-Based Tools**: CoppoGAN, NetShare
    * Generate realistic variations of network traffic
    * Can simulate unseen events while maintaining statistical similarity
    * **Limitations**:
      * Low ML model accuracy when trained on synthetic data (e.g., NetShare)
      * Only generate aggregated flow statistics (NetFlow-like attributes)
      * Cannot generate packet-level details or full packet captures
      * Don't enforce protocol rules (e.g., TCP handshake requirements)

* **Diffusion Models Background**
  * **Core Concept**: Learn to reverse a noise-adding process
    * Forward process: Gradually add Gaussian noise to image until pure noise
    * Reverse process: Learn to denoise, recovering original image
  * **Inspiration**: Physical diffusion (e.g., dye drop diffusing in water)
  * **Training**: U-Net architecture predicts noise at each step
  * **Examples**: DALL-E, Stable Diffusion, Midjourney
  * **Key Advantage**: Gradual denoising is more stable than GAN's single-step generation

* **Conditional Generation**
  * **Text-to-Image Models**: Stable Diffusion, etc.
  * **Conditioning Mechanism**:
    * Encode text prompt into embedding vector (e.g., using CLIP)
    * Guide denoising process at each step based on prompt
    * Enable controlled generation ("generate a horse" vs random output)
  * **ControlNet**: Additional constraints for generation
    * Example: Sketch as input to guide spatial structure
    * For networks: Enforce protocol structure and field boundaries

* **NetDiffusion Framework**
  * **Step 1: Traffic to Image Conversion**
    * Convert raw packets to standardized format with bit-level encoding
    * Handle missing features (e.g., UDP traffic has no TCP fields → -1)
    * Create images where each row = one packet (up to 1024 packets)
    * Example: 1024×1088 pixel image containing up to 1024 packets
    * Visual structure preserves inter-packet relationships
  * **Step 2: Fine-Tuning with LoRA**
    * Use Stable Diffusion 1.5 as pre-trained base model
    * Apply Low Rank Adaptation (LoRA) for efficient fine-tuning
    * Pair traffic images with text prompts (e.g., "TCP Amazon traffic")
    * Fine-tuning vs training from scratch: much less data needed (few-shot learning)
  * **Step 3: Conditional Generation with ControlNet**
    * Problem: Diffusion models are "inherently creative" - may generate invalid traffic
    * Solution: Use ControlNet to enforce structural constraints
    * Edge detection on example traffic defines valid regions (e.g., TCP/IP layers only)
    * Ensures generated traffic follows protocol structure
  * **Step 4: Post-Processing with Dependency Trees**
    * **Intra-packet rules**: Checksum consistency, field relationships within packets
    * **Inter-packet rules**: TCP sequence numbers, handshake requirements, packet ordering
    * Traverse dependency tree bottom-up: fix intra-packet first, then inter-packet
    * Majority voting for fields like source/destination IP and ports
    * Goal: Minimize modifications while ensuring protocol compliance

* **Evaluation Results**
  * **Statistical Similarity**:
    * 30%+ improvement over baselines (NetShare) on aggregate statistics
    * 70%+ improvement on individual field accuracy
  * **Traffic Classification**:
    * Train ML model entirely on synthetic data, test on real data
    * 60%+ accuracy improvement vs baselines
    * Attribute: Higher granularity and more features than flow-level approaches
  * **Class Imbalance Handling**:
    * Pad under-represented classes (Facebook, Meet, Zoom) with synthetic data
    * Significant accuracy improvements for imbalanced classes
    * Example use case: Assignment 3 "play music" traffic (very small portion)
  * **Tool Compatibility**:
    * Wireshark: Successfully parses generated PCAPs
    * TCPreplay: No errors raised
    * Some protocol analysis tasks (e.g., TCP flags) not 100% accurate yet

* **Fidelity vs Diversity Trade-off**
  * **Core Challenge**: Balancing similarity to real data with useful variation
    * Zero distance: Perfect copy (useless - no new information)
    * Maximum distance: Complete noise (useless - not representative)
    * Need: Meaningful variation that preserves important characteristics
  * **Open Problem**: No consensus metric for optimal fidelity-diversity balance

* **Future Directions and Challenges**
  * **Text-to-Traffic Generation**: Using autoregressive models (transformer-based)
  * **Context Transfer**:
    * Example: "Take this normal traffic and encrypt it with VPN"
    * Example: "Make lab traffic look like it's from another lab"
    * Analogous to image style transfer ("make me Batman")
  * **Scalability**: Beyond 1024 packets
    * Current limitation: Image size constrains packet count
    * Potential solution: Autoregressive generation for longer traces

* **Practical Insights from Chase**
  * Knowledge of generative AI broadly applicable beyond networking
  * Applications in other domains:
    * Dating app: CLIP model for image-to-text conversion, latent space matching
    * Brain wave analysis: Data-oriented approach
    * Recommendation systems: Embedding-based similarity matching
  * Foundation in generative models transfers across data types and domains

* **Assignment Preview**
  * **Release**: Next week (week after Thanksgiving)
  * **No Hands-On Today**: Entire topic assigned as homework
  * **Tasks**:
    1. Explore converting M-Print to images (multiple approaches)
    2. Generate traffic using provided model
    3. Convert generated images back to M-Print
    4. Train ML model on synthetic data, test on real data
    5. Compare with NetDiffusion approach
  * **Tip**: First 3-10 packets contain most useful information (TCP handshake)
    * Later packets are mostly data transmission (less distinctive)

* **Course Logistics**
  * **Next Meeting**: Transformer architectures, state-space models (Mamba)
  * **Midterm 2**: Two weeks from this meeting
    * Coverage: Lectures 11-16 (inclusive)
    * Focus on ML modeling (not data preprocessing like Midterm 1)
    * Practice exam prompts to be provided
  * **Final Project**: Due during finals week (likely Thursday)

### Meeting 17

* **Generative AI**
   * GANs
   * Transformers
   * Stable Diffusion
* **Reasons and motivation to use generative AI for network data**
  * Data augmentation
  * Privacy constraints

