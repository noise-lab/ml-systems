# Lecture 1

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

# Lecture 2

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

# Lecture 3

* Security Hands-On
* More Motivation
  * Application Quality of Experience
  * Overview of Assignment 1
  * Application quality hands-on (?)
* Resource Provisioning Motivation (no hands-on)
* Project Team Formation Time (if needed)

# Lecture 4

* Prof. Feamster out of town
* Project office hours
* Research in Networks/ML (Taveesh and Andrew)

# Lecture 5

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

# Lecture 6

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

# Lecture 7

* Hands-On Activity
   * Packet Statistics Extraction - Flow Statistics (netML)
* Data Preparation and ML Pipelines
* Hands-On Activity
   * Data Preparation



# Lecture 8

* Hands-On Activity
  * Data Preparation and Model Training (#6)
* ML Pipelines
  * Training and testing
  * Train-test split
  * Cross-validation
  * Hyperparameter tuning
  * Evaluation metrics
* Hands-On Activity
   * ML-Pipeline (#7)
* Midterm Topics Stop **Here** (Nothing below here!)

# Lecture 9

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

# Lecture 10 

* In-Class Midterm

# Lecture 11

* Linear Regression
* Hands-On Activity (#10 Linear Regression)

# Lecture 12

* Logistic Regression
* Hands-On Activity (#11 Logistic Regression)

# Lecture 13

* Decision Trees and Ensembles
* Advantages and disadvantages of decision trees
* Random Forests
  * Bagging / Design
  * Advantages of Random Forest over Decision Trees


# Lecture 14

* What is representation learning?
   * Deep Learning
   * Neural Networks
   * Backpropagation

# Lecture 15

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


# Lecture 16

* Clustering
   * K-means
   * GMM
   * Hierarchical Clustering
   * DBSCAN
* Hands-On Activity (#15 Clustering)

# Lecture 17

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

