# Machine Learning for Computer Systems

## Course Description

This course will cover topics at the intersection of machine learning and
systems, with a focus on applications of machine learning to computer systems.
Topics covered will include applications of machine learning models to
security, performance analysis, and prediction problems in systems; data
preparation, feature selection, and feature extraction; design, development,
and evaluation of machine learning models and pipelines; fairness,
interpretability, and explainability of machine learning models; and testing
and debugging of machine learning models.

The topic of machine learning for computer systems is broad. Given the
expertise of the instructor, many of the examples this term will focus on
applications to computer networking. Yet, many of these principles apply
broadly, across computer systems.

You can and should think of this course as a practical hands-on introduction
to machine learning models and concepts that will allow you to apply these
models in practice. We'll focus on examples from networking, but you will walk
away from the course with a good understanding of how to apply machine
learning models to real-world datasets, how to use machine learning to help
computer systems operate better, and the practical challenges with deploying
machine learning models in practice.

## Syllabus

More details are in the [course syllabus](syllabus.md).

## Schedule 

| Lecture                            | Topic                                                                                                                   | Reading                                                                                                                                    | Assignment                                    |
| ---------------------------------- | -------------------------------------                                                                                   | -----------------------------                                                                                                              | ----------                                    |
| 1                                  | Introduction<br />([Packet Capture](notebooks/01-Packet-Capture-Basics-Clean.html))                                     | [Ch. 1](book/text/intro.html)                                                                                                              |                                               |
| **Use Cases**                      |                                                                                                                         |                                                                                                                                            |                                               |
| 2                                  | Security<br />([Scanning](notebooks/02-Motivation-Security-Clean.html))                                                 | [Ch. 2.1](book/text/motivation.html#applications-to-security)<br>[Self-Running Networks](https://arxiv.org/pdf/1710.11583)                 | Notebook Setup                                |
| 3                                  | Performance<br />([QoE Inference](notebooks/03-Performance-Service-Clean.html))                                         | [Ch. 2.2](book/text/motivation.html#applications-to-performance)<br>[Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8121867) |                                               |
| 4                                  | Resource Optimization                                                                                                   | [Ch. 2.3](https://noise-lab.github.io/ml-systems/book/text/motivation.html#application-service-and-device-identification)                  |                                               |
| **Data From Computer Systems**     |                                                                                                                         |                                                                                                                                            |                                               |
| 5                                  | Data Acquisition<br>([Data Acquisition](notebooks/05-Data-Acquisition-Clean.html))                                      | [Ch. 3.2-3.3](https://noise-lab.github.io/ml-systems/book/text/measurement.html#active-measurement)                                        |                                               |
| 6                                  | From Data to Analysis<br>([Feature Extraction](notebooks/06-Feature-Extraction-Clean.html))                             | [Ch. 3.4](https://noise-lab.github.io/ml-systems/book/text/measurement.html#from-data-to-analysis)                                         | Traffic Classification                        |
| **Machine Learning Pipeline**      |                                                                                                                         |                                                                                                                                            |                                               |
| 7                                  | Data Preparation and Representation<br>([Data Preparation](notebooks/07-Data-Preparation-Clean.html))                   | [Ch. 4.1 ](https://noise-lab.github.io/ml-systems/book/text/pipelines.html#data-preparation)                                               |                                               |
| 8                                  | Model Training and Evaluation<br>([Model Evaluation](notebooks/08-ML-Pipeline-Clean.html))                              | [Ch. 4.2-4.3](https://noise-lab.github.io/ml-systems/book/text/pipelines.html#model-training)                                              |                                               |
| **Supervised Learning**            |                                                                                                                         |                                                                                                                                            |                                               |
| 9                                  | Non-Parametric and Probabilistic Models<br>([Naive Bayes](notebooks/09-Naive-Bayes-Clean.html))                         | [Ch. 5](book/text/supervised.html)                                                                                                         |                                               |
| 10                                 | Linear and Polynomial Regression<br>([Linear Regression](notebooks/10-Linear-Regression-Clean.html))                    |                                                                                                                                            |                                               |
| 11                                 | Logistic Regression and Support Vector Machines<br>([Logistic Regression](notebooks/11-Logistic-Regression-Clean.html)) |                                                                                                                                            |                                               |
| 12                                 | Trees and Ensembles<br>([Trees and Ensembles](notebooks/12-Trees-Ensembles-Clean.html))                                 |                                                                                                                                            | [Midterm/Take-Home](assignments/midterm.html) |
| 13                                 | Deep Learning<br>([Deep Learning](notebooks/13-Deep-Learning-Clean.html))                                               |                                                                                                                                            |                                               |
| **Unsupervised Learning**          |                                                                                                                         | [Ch. 6](book/text/unsupervised.html)                                                                                                       |                                               |
| 14                                 | Dimensionality Reduction<br>([Dimensionality Reduction](notebooks/14-Dimensionality-Reduction-Clean.html))              |                                                                                                                                            |                                               |
| 15                                 | Clustering<br>([Clustering](notebooks/15-Clustering-Clean.html))                                                        |                                                                                                                                            | Anomaly Detection                             |
| **Deployment Challenges**          |                                                                                                                         |                                                                                                                                            |                                               |
| 16                                 | Programmability and Automation<br>([Automation](notebooks/16-Automation-Clean.html))                                    | Ch. 7.1                                                                                                                                    |                                               |
| 17                                 | Systems and Deployment Costs                                                                                            | Ch. 7.2                                                                                                                                    | Systems Costs                                 |
| 18                                 | Privacy, Ethics, and the Law                                                                                            | Ch. 7.3                                                                                                                                    |                                               |

Please come to class having done the reading. 


## Background Videos and Readings

The material below is strictly optional unless otherwise noted, although you
may find it useful.

* [Board Notes](https://www.dropbox.com/s/fef5y9enms6djlr/ML%20for%20Systems.pdf?dl=0)
* [Resource List](ml.md)



