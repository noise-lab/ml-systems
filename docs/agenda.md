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

* In-Class Midterm

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

* TBD


### Meeting 14

* What is representation learning?
   * Deep Learning
   * Neural Networks
   * Backpropagation

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
* Hands-On Activity (#15 Clustering)

### Meeting 17

* Bit-level representation of network data (nPrint)
  * Motivation
  * Applications
  * Challenges
* Generative AI
   * GANs
   * Transformers
   * Stable Diffusion
* Reasons and motivation to use generative AI for network data
  * Data augmentation
  * Privacy constraints

