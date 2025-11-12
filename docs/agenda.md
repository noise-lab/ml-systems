## Agenda

### Meeting 1

* Basics
  * Slack
  * Github Classroom
  * Canvas/Github
  * Logistics 
* Coverage of syllabus
  * Course objectives
  * Topic outline
  * Due dates
  * What's new this year
  * Project
* Lecture on course material
  * Introduction to Computer Networks
  * Packet Capture
* Getting Started with Hands-On 1 (Notebook Setup)

### Meeting 2

* Hands-On 1: Packet Capture
  * Wireshark basics and installation
  * Getting started with Jupyter, etc.
* Introduction (Slides)
  * Motivating Applications
  * Security
* Hands-On Activities
  * Packet Capture
    * Learning Objectives
       * Wireshark Setup
       * Notebook Setup
       * Packet Capture
       * Packet Capture to Pandas
       * Analysis
* Security Applications (Slides, Discussion)

### Meeting 3

* Security Hands-On
* More Motivation
  * Application Quality of Experience
  * Overview of Assignment 1
  * Application quality hands-on (?)
* Resource Provisioning Motivation (no hands-on)
* Project Team Formation Time (if needed)

### Meeting 4

* Prof. Feamster out of town
* Project office hours
* Research in Networks/ML (Taveesh and Andrew)

### Meeting 5

* Follow-up from Previous Sessions
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

* Resource Allocation Applications
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

* Introduction to Data Acquisition
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

* Preview of Next Session (Friday)
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

* Netflix Player "Nerd Stats" Note
  * Can view detailed playback metrics with special keystroke (Ctrl+Alt+T or similar)
  * Provides overlay with frame rate, resolution, bitrate, etc.
  * This is how training data was collected for QoE inference models
  * (Note: May have been removed or changed by Netflix)

### Meeting 6

* Active and passive measurement
   * Advantages and disadvantages of active and passive measurement
     * Infrastructure considerations
     * Measurements when you want them
     * Systems costs considerations
     * Privacy considerations
   * Feature extraction from packet captures
   * What is a flow? (5-tuple)
* Hands-On Activity
   * Packet Statistics Extraction - Flow Statistics (Manual)

### Meeting 7

* Course Transition: From Data Acquisition to Data Preparation
  * Completed: How to get data out of networks
  * Moving to: How to prepare data for ML models
  * Looking ahead: Deep learning (weeks 6-7) - can throw raw data at models
    * Tools: nPrint (bit-level representation)
    * But first: Learn traditional feature extraction approaches

* ML Pipeline Overview
  * Input → Transformation → Dimensionality Reduction → Training → Output → Evaluation
  * Each step has associated costs (systems consideration)
  * Not just about accuracy - consider:
    * Amount of training data required
    * Data acquisition cost
    * Ability to move data to model
    * Time to detection (inference speed)
    * Storage and transmission costs
    * Compute requirements for transformation

* Systems Considerations in ML for Networks
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

* Supervised vs. Unsupervised Learning Review
  * **Supervised Learning**: Training with labels
    * Mnemonic: "Supervising" the model with examples
    * Can evaluate using labels as "answer key"
  * **Unsupervised Learning**: Training without labels
    * Common tasks: Clustering, pattern detection
    * Can still do classification (e.g., anomaly detection with two clusters)
    * Evaluation is harder (no answer key)

* Features and Labels
  * **Features**: Inputs to the model (covariates in statistics)
  * **Labels**: What the model is trying to predict/classify (supervised learning only)
  * Feature selection remains one of most important parts of modeling process
  * How you represent data affects how model learns (or doesn't learn)

* Data Representation Challenges in Network Systems

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

* Feature Engineering and Transformation

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

* Data Quality Issues and Pitfalls

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

* Hands-On Activity: NetML Library
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

* Preview of Next Session (Friday)
  * ML Pipelines and Evaluation Metrics
  * Topics:
    * False positives and true positives
    * Accuracy, Precision, Recall
    * F1 score
    * ROC curves and AUC
  * Will continue with NetML hands-on (more time allocated)

* Key Midterm Concepts Highlighted
  * Systems considerations beyond accuracy in ML for networks
  * Difference between passive and active measurement (from previous lectures)
  * Examples of non-representative training data in networking contexts
  * Understanding and addressing irrelevant features
  * Data quality issues and their impact

* Technical Issues Noted
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

* Hands-On Activity
   * ML-Pipeline (#7)
* Evaluation Metrics
   * Accuracy
   * Precision
   * Recall
   * ROC
   * AUC
* Supervised Learning Overview
* Naive Bayes

### Meeting 10 

* In-Class Midterm

### Meeting 11

* Linear Regression
* Hands-On Activity (#10 Linear Regression)

### Meeting 12

* Logistic Regression
* Hands-On Activity (#11 Logistic Regression)

### Meeting 13

* Decision Trees and Ensembles
* Advantages and disadvantages of decision trees
* Random Forests
  * Bagging / Design
  * Advantages of Random Forest over Decision Trees


### Meeting 14

* Bit-level representation of network data (nPrint)
  * Motivation
  * Applications
  * Challenges

### Meeting 15

* Dimensionality Reduction
* Motivation for Dimensionality Reduction
  * Visualization
  * Computation/Training Time
  * Interpretability
  * Noise Reduction/Model Robustness
* Example Dimensionality Reduction Techniques
   * PCA
   * t-SNE
   * PVA vs. t-SNE - when to use which?


### Meeting 16

* Clustering
   * K-means
   * GMM
   * Hierarchical Clustering
   * DBSCAN
* Hands-On Activity (#16 Clustering)

### Meeting 17

* Generative AI
   * GANs
   * Transformers
   * Stable Diffusion
* Reasons and motivation to use generative AI for network data
  * Data augmentation
  * Privacy constraints

