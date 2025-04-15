<p text-align = "center" justify><h2>HYBRID MACHINE LEARNING MODELS FOR CYBER-ATTACK DFETECTION IN IOT WITH SHAP-DRIVEN INTERPRETABILITY AND PERFORMANCE ANALYSIS</h2></p> 

## Project Overview
The Internet of Things (IoT) has transformed industries by enabling interconnected devices to communicate seamlessly. However, this connectivity introduces significant security risks, as IoT devices are prime targets for cyberattacks. This project, titled Hybrid Machine Learning Models for Cyberattack Detection in IoT with SHAP-Driven Interpretability and Performance Analysis, aims to address these challenges by developing a robust detection system using hybrid machine learning models. The approach combines traditional machine learning (e.g., XGBoost) with deep learning techniques (e.g., CNNs, LSTMs) to detect cyberattacks effectively. Additionally, SHAP (SHapley Additive exPlanations) is employed to provide interpretability, ensuring transparency in model predictions—an essential feature for cybersecurity applications. The project also includes a comprehensive performance analysis to evaluate the models’ effectiveness.
## Objectives
* To develop hybrid machine learning models to detect cyberattacks in IoT environments.
* To leverage SHAP for interpretability to understand key features driving predictions.
* To conduct a performance analysis using metrics such as accuracy, precision, recall, and F1-score.
* To optimize the models through techniques like cross-validation and hyperparameter tuning.
## Data Collection and Structure
The CIC IoT Dataset 2023 is a comprehensive collection of data designed to enhance the development of security analytics in real IoT environments. Developed by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick, this dataset encompasses a wide array of attack scenarios executed within an extensive IoT topology.
### Key Features of the Dataset:
* Extensive IoT Topology: The dataset was generated using a network of 105 IoT devices, reflecting a realistic and diverse IoT environment.
* Diverse Attack Scenarios: A total of 33 distinct attacks were performed, categorized into seven major groups:
    * Distributed Denial of Service (DDoS): Including ACK fragmentation, UDP flood, SlowLoris, ICMP flood, RSTFIN flood, PSHACK flood, HTTP flood, UDP fragmentation, TCP flood, SYN     
      flood, and SynonymousIP flood.
    * Denial of Service (DoS): Comprising TCP flood, HTTP flood, SYN flood, and UDP flood.
    * Reconnaissance (Recon): Encompassing ping sweep, OS scan, vulnerability scan, port scan, and host discovery.
    * Web-based Attacks: Such as SQL injection, command injection, backdoor malware, uploading attack, XSS, and browser hijacking.
    * Brute Force: Specifically, dictionary brute force attacks.
    * Spoofing: Including ARP spoofing and DNS spoofing.
    * Mirai: Covering GRE IP flood, GRE Ethernet flood, and UDP Plain attacks.

All attacks were orchestrated by compromised IoT devices targeting other IoT devices, providing a realistic depiction of potential threat scenarios.
Source: https://www.unb.ca/cic/datasets/index.html, 24th March, 2025. 
## Data Structure
The dataset consist of 712,311 entries (indexed from 0 to 712,310) and 40 columns in total. It contain list of numerical and categorical columns:
* Numerical columns (39):
    * Float64 (30 columns): examples includes - Header_Length, Time_To_Live, Rate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC, AVG, Std, Tot size, IAT and Variance.
    * Int64 (9 columns): examples includes – Protocol_Type, ack_count, syn_count, fin_count, rst_count, Tot sum, Min, Max, and Number.
* Categorical columns (1):
    * Object: Label

Most columns have 712,311 entries non-null values, indicating no missing data in those columns. Std and variance are exceptions each having 11 missing values. The features can be categorize as:
* Network Traffic Attributes:
  * Flags (e.g., fin_flag_number, syn_flag_number, rst_flag_number), 
  * Counts (e.g., ack_count, syn_count, fin_count, rst_count), and 
  * Protocol indicators (e.g., HTTP, HTTPS, TCP, UDP).
* Statistical Summary
  * Min, Max, AVG, Std, Variance, likely representing packet sizes or other traffic metrics.
* Time-Based Features
  * IAT (Interval- Arrival Time), Time_To_Live.

The target variable: 
Label: original labels (e.g., indicating whether an entry is an attack or benign).

## Label Categorization
The raw label were categorized into attack classes and subtypes for easier analysis and modeling which changed its dimension to 712,311 entries (indexed from 0 to 712,310) and 42 columns in total. The categorization transforms a potentially messy label column into structured classes and subtypes, enabling multi-class or hierarchical classification. The class distribution after categorization of the label reveals that, the dataset contains a mix of malicious and benign network traffic, with a clear predominance of certain attack types. The distribution reveals a significant imbalance, where some categories, such as DDoS attacks, have a high number of instances, while others, like web-based attacks, are much less frequent. Here is a categorized breakdown of the dataset based on the Label class, and Attack subtype:

* **DDoS Attacks**: Distributed Denial of Service (DDoS) attacks are the most frequent in the dataset. They includes:
  * ICMP Flood: 108,662 instances
  * UDP Flood: 82,011 instances
  * TCP Flood: 68,289 instances
  * PSHACK Flood: 62,171 instances
  * RSTFIN Flood: 61,652 instances
  * SYN Flood: 61,460 instances
  * SynonymousIP Flood: 54,749 instances
  * Other DDoS (ICMP Fragmentation): 6,784 instances
  * ACK Fragmentation: 4,308 instances
  * UDP Fragmentation: 4,264 instances
  * HTTP Flood: 416 instances
  * SlowLoris: 354 instances
  
  **Total DDoS Instances**: Approximately 515,127

* **DoS Attacks**: Denial of Service (DDoS) attacks target system availability but are less distributed than DDoS attacks:
  * UDP Flood: 50,371 instances
  * TCP Flood: 40,391 instances
  * SYN Flood: 30,620 instances
  * HTTP Flood: 1,113 instances

**Total DoS Instances: 122,495**

* **Benign Traffic**: Non-malicious traffic:
  * Benign: 16,577 instances

**Total Benign Instances**: 16,577

* **Mirai Attacks**: Attack associated with the mirai botnet:
  * Greeth Flood: 15,135 instances
  * UDPPlain: 13,342 instances
  * GREIP Flood: 11,187 instances

**Total Mirai Instances**: 39,664

* **Reconnaissance (Recon)**: Activities aimed at gathering information about a target:
  * Vulnerability Scan: 5,805 instances
  * Host Discovery: 2,045 instances
  * OS Scan: 1,433 instances
  * Port Scan: 1,251 instances
  * Ping Sweep: 28 instances

**Total Recon Instances**: 10,562

* **Spoofing**: Attacks involving impersonation:
  * ARP Spoofing: 4,590 instances
  * DNS Spoofing: 2,738 instances

**Total Spoofing Instances**: 7,328

* **Brute Force**: Attempts to gain unauthorized access through repeated credential guesses:
  * Dictionary Brute Force: 204 instances

**Total Brute Force Instances**: 204

* **Web-based Attacks**: Attacks targeting web applications or services:
  * Browser Hijacking: 83 instances
  * SQL Injection: 77 instances
  * Command Injection: 73 instances
  * XSS (Cross-Site Scripting): 67 instances
  * Backdoor Malware: 41 instances
  * Uploading Attack: 20 instances

**Total Web-based Instances**: 361

The following observations were made from the class distribution:
Class Imbalance:
* The dataset is heavily skewed toward DDoS attacks, with ICMP Flood (108,662) being the most frequent subtype. In contrast, web-based attacks like Uploading Attack (20) and reconnaissance types like Ping Sweep (28) are extremely rare.
* Benign traffic (16,577) is significantly underrepresented compared to malicious traffic.

Diversity of Attack Types:
* DDoS and DoS categories show a wide variety of subtypes (e.g., floods targeting different protocols: ICMP, UDP, TCP).
* Other categories, such as web-based attacks, include diverse but infrequent subtypes (e.g., SQL Injection, XSS).

Rare Classes:
* Some attack types, such as Ping Sweep (28), Uploading Attack (20), and Dictionary Brute Force (204), have very low instance counts, which could make them challenging to detect or model effectively.

The following implications were deduced from the above observations:
* Modeling Challenges: The significant imbalance suggests that machine learning models may become biased toward frequent classes (e.g., DDoS ICMP Flood) unless techniques like oversampling (e.g., SMOTE), undersampling, or class weighting are applied.
* Target Selection: The dataset supports multi-class classification using either Label_class (broad categories like DDoS, DoS) or Attack_subtype (specific attack types like ICMP Flood). The choice depends on the desired granularity of prediction.
* Feature Analysis: Given the variety of attack types, understanding which features distinguish between classes (e.g., packet size, protocol type) will be critical for effective classification.

The distribution reveals a highly imbalanced dataset where DDoS attacks (515,127 instances) and DoS attacks (122,495 instances) dominate, while benign traffic (16,577 instances) and other attack types like Web-based (361 instances) and Brute Force (204 instances) are significantly underrepresented. This imbalance, combined with a wide variety of attack subtypes, highlights the need for careful data preprocessing and modeling strategies to ensure accurate detection of all traffic types, especially rare but potentially severe attacks. The distribution provides valuable insights into the dataset’s structure, guiding the development of robust cyberattack detection systems.

## Data Preprocessing
### Checking Missing Values:
* Identified which columns contain missing data.
* Quantified how many missing values are in each column.
* Informed decisions on handling missing data, such as imputing values, dropping rows or columns, or applying other cleaning strategies.

### Converting Categorical Features to Appropriate Type
The pandas category data type is designed for variables with a limited number of distinct values. Converting these columns offers several benefits:
* Memory Efficiency:
  * Numerical types like int64 or float64 use more memory than necessary if the column has few unique values. The category type stores the data as integer codes internally, with a mapping to the unique values, reducing memory usage.
  * For binary columns (e.g., 'HTTP' with values 0 and 1), the savings might be less pronounced compared to a Boolean type, but category is still efficient and more flexible if the number of unique values increases.
* Data Analysis and Visualization:
  * Categorical columns are treated as discrete entities rather than continuous numbers. This improves the interpretability of group-by operations, summary statistics, and plots (e.g., bar charts instead of treating 0 and 1 as a continuum).
* Preparation for Machine Learning:
  * Many machine learning models benefit from knowing which features are categorical. While some libraries (e.g., scikit-learn) require numerical input and thus need encoding (like one-hot or ordinal encoding), marking them as categorical now simplifies later steps. For instance, pandas’ get_dummies() can automatically one-hot encode categorical columns.
  * Advanced frameworks like XGBoost or TensorFlow can leverage categorical data directly or via embeddings, depending on how you preprocess them later.

### Label Encoding for the Target Feature (Label_class)
This tool from sklearn.preprocessing converts categorical labels (e.g., "DDoS", "Benign") into unique integers (e.g., 0, 1, 2 ...). Each distinct category in 'Label_class' is assigned a unique integer. This method first "fits" the encoder to the unique values in 'Label_class' (learning the mapping from categories to integers) and then "transforms" the column by replacing the original categorical values with their corresponding integers. 

### Checking for Outliers using the IQR Method
The IQR method is a standard statistical approach for outlier detection. It identifies values that fall significantly outside the middle 50% of the data, making it robust for many distributions, including those that are not perfectly normal. The median is a measure of central tendency that isn’t heavily influenced by extreme values, unlike the mean. Replacing outliers with the median reduces their impact without distorting the overall data distribution too much. Instead of removing rows with outliers, which reduces dataset size, and which in turn lead to substantial data loss and bias (especially if outliers correlate with attack types). This method keeps all data points, which is useful when sample size matters or when outliers might still hold some meaning.

### Checking for Non-Finite Values
Checking for non-finite values is important because of:
* Data Quality: Non-finite values can signal problems in your dataset that need fixing.
* Analysis Safety: Many calculations (e.g., averages, variances) break or give nonsense results with NaN or inf.
* Model Readiness: Machine learning models often choke on non-finite inputs, leading to errors or bad predictions.

### Data Scaling using RobustScaler
RobustScaler is a feature scaling method that’s particularly useful when your data contains outliers. Unlike StandardScaler (which uses the mean and standard deviation) or MinMaxScaler (which uses the minimum and maximum values), RobustScaler relies on robust statistics:
* Median: It subtracts the median of each feature to center the data.
* Interquartile Range (IQR): It divides by the IQR (the difference between the 75th and 25th percentiles) to scale the data.

This approach makes RobustScaler less sensitive to extreme values, which is ideal for datasets like network traffic data. Given the presence of outliers—common in network data due to bursts of activity or irregular events—RobustScaler is a solid choice. It ensures that the scaling process isn’t skewed by these extremes, providing a more stable transformation for machine learning models that rely on feature magnitudes (e.g., neural networks).

### Data Scaling using StandardScaler
StandardScaler is a feature scaling method widely used in machine learning to standardize data by transforming it to have a mean of zero and a standard deviation of one. Unlike RobustScaler (which uses the median and interquartile range) or MinMaxScaler (which uses the minimum and maximum values), StandardScaler relies on the following statistics:
* Mean: It subtracts the mean of each feature to center the data around zero.
* Standard Deviation: It divides by the standard deviation to scale the data to unit variance.

This approach makes StandardScaler effective for datasets where features follow a roughly Gaussian distribution, though it can be sensitive to outliers, which may skew the mean and standard deviation. It’s a good choice for datasets like financial or sensor data when outliers are minimal or managed separately, ensuring that features are on a comparable scale for models sensitive to magnitude, such as gradient-based algorithms (e.g., logistic regression, neural networks).

## Model Development, Shap Filtering and Evaluation
### XGBoost
XGBoost is a boosting algorithm that builds an ensemble of decision trees sequentially. Unlike traditional decision trees that operate independently, boosting algorithms combine multiple weak learners (trees that perform only slightly better than random guessing) into a single, robust predictive model. The idea is to sequentially add new models that correct the errors of the combined ensemble of previous models.
### Key features of XGBoost include:
* Gradient Boosting Framework: XGBoost optimizes a loss function by using gradient descent methods, where each new tree is constructed to reduce the residual errors made by the previous trees.
* Regularization: One of the significant improvements in XGBoost is the incorporation of regularization techniques (L1 and L2) which help in reducing overfitting, a common challenge in machine learning.
* Parallel Processing: XGBoost is designed to utilize multi-threaded processing, which makes it highly efficient and faster compared to other boosting implementations.
* Handling Missing Values: The algorithm is robust in handling missing values in the dataset, automatically learning the best direction to take when it encounters a missing value.
* Scalability: XGBoost is designed for distributed computing and can handle large datasets efficiently.
### XGBoost in Cyber Attack Detection
Cyber-attack detection involves identifying malicious activities in a network or system before they can cause harm. The task is challenging due to the high dimensionality of network data, the presence of noise, and the continuously evolving tactics of cyber attackers. XGBoost offers several advantages that make it highly suitable for this domain.
* Handling High-Dimensional Data
* Robustness and Flexibility
* Scalability and Speed
* Improved Detection Accuracy
* Interpretability and Decision Making

#### === XGBoost Evaluation ===
* Accuracy:   0.813927
* Precision:   0.787117
* Recall:   0.813927
* F1-score:   0.779458
* Training Time:  485.908224 seconds.

### SHAP Filtering for XGBoost
SHAP (SHapley Additive exPlanations) is an emerging framework for interpreting complex machine learning models through visualization and quantified feature contributions. 
It builds on cooperative game theory principles and offers a unified measure of feature importance for any model prediction, ensuring transparency even for black-box models.
SHAP quantifies the contribution of each feature to a specific prediction. Conceptually, it answers the question: “How would the prediction change if a feature were absent?” This idea is rooted in Shapley values, a concept from cooperative game theory. In such games, each player (feature) contributes to the overall “payout” (prediction), and the Shapley value represents the fair distribution of this payout among the players.

#### Key properties of SHAP include:
* Additivity: The prediction is decomposed additively into contributions from each feature. The sum of all SHAP values is equal to the difference between the model’s prediction for a given instance and the expected prediction (i.e., the average over the dataset).
* Consistency: When a feature’s contribution increases or stays the same regardless of other features, its SHAP value does not decrease.
* Local Interpretability: SHAP provides instance-level insights, meaning that one can understand why a specific prediction is made on a particular data sample, which is essential for personalized analysis in critical systems.

### SHAP in Cyber Attack Detection
Cyber-attack detection involves analyzing large volumes of network data, system logs, and other behavioral data to quickly identify threats. SHAP visualizations bring several benefits to cybersecurity applications:
* Enhancing Model Trust and Transparency
* Identifying Subtle Threats and Anomalies
* Prioritizing Security Responses
* Continuous Model Monitoring and Improvement

**Inferences** <br>
Based on the SHAP visualization output for the XGBoost model, the following features have been identified as valid due to their SHAP values exceeding a specified threshold, indicating their significant contribution to the model's predictions: ['Header_Length', 'Protocol Type', 'Rate', 'syn_flag_number', 'ack_flag_number', 'ack_count'
'syn_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP'
'IPv', 'Tot sum', 'Max', 'AVG', 'Std', 'IAT', 'Variance']

## CONVOLUTIONAL NEURAL NETWORKS (CNNs) 
Convolutional Neural Networks (CNNs) are a class of deep neural networks particularly well suited for processing data that have a grid-like topology, such as images. Unlike traditional fully connected networks, CNNs are designed to take advantage of the spatial structure in data through convolution operations. The main idea behind CNNs is to learn hierarchical patterns: low-level features (edges, textures) in the early layers and high-level features (shapes, objects) in deeper layers.
#### Key components of a CNN include:
* Convolutional Layers: These layers apply a set of learnable filters (or kernels) to input data. Each filter slides (convolves) across the input, performing element-wise multiplications and summing the results to create a feature map. This operation helps in detecting features like edges or corners.
* Activation Functions: Non-linear functions (such as ReLU, Sigmoid, or Tanh) introduce non-linearity into the model, allowing it to learn complex patterns.
* Pooling Layers: Pooling (often max pooling) reduces the spatial dimensions (width and height) of the input volume, which decreases the computational cost and controls overfitting by summarizing the responses in a local neighborhood.
* Fully Connected Layers: In the final stages of a CNN, fully connected layers combine features learned by convolutional layers to classify the input into various categories or perform regression tasks.
* Normalization Layers: Batch normalization or layer normalization is commonly used to accelerate training and improve stability.
### CNNs in Cyber Attack Detection
The development and implementation of CNNs in cyber attack detection represent a convergence of computer vision techniques and cybersecurity challenges. By transforming raw network data into image-like formats and applying state-of-the-art convolutional architectures, security systems can proactively detect anomalies and malicious behavior with enhanced accuracy and speed.
#### The Rationale for Using CNNs in Cyber Attack Detection
Cybersecurity involves monitoring complex and high-dimensional data. Traditionally, network intrusion detection systems (NIDS) relied on signature-based or statistical methods. However, as threats evolve — ranging from Distributed Denial of Service (DDoS) attacks to advanced persistent threats (APTs) — it becomes necessary to harness models capable of capturing subtle, intricate patterns. CNNs present several key advantages:
* Feature Extraction
* Handling Unstructured Data
* Robustness to Noise
* Scalability
#### Data Representation: From Raw Data to Visual Inputs
A common approach to applying CNNs in cyber-attack detection is to transform network traffic or log data into images. There are several methods to achieve this:
* Traffic as Images
* Log Visualization
* Spectrogram Representations
#### === CNN Evaluation ===
* Accuracy : 0.789892
* Precision : 0.659397
* Recall : 0.789892
* F1-score : 0.708950
* Training Time: 483.328848 seconds

### SHAP Filtering Implementation with CNN
**Inferences** <br> 
Based on the SHAP visualization output for the CNN model, the following features have been identified as valid due to their SHAP values exceeding a specified threshold, indicating their significant contribution to the model's predictions. Valid SHAP Features from CNN: ['Header_Length', 'Protocol Type', 'Rate', 'ack_flag_number', 'ack_count', 'HTTP', 'HTTPS', 'TCP', 'UDP', 'ICMP', 'Tot sum', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Variance']

## LONG SHORT-TERM MEMORY NETWORKS (LSTMs)
Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Network (RNN) designed to capture long-term dependencies and patterns in sequential data. Unlike standard RNNs that tend to suffer from vanishing and exploding gradients when processing long sequences, LSTMs incorporate memory cells and gating mechanisms that allow them to maintain and update information over extended time periods. This makes them particularly well suited for applications where temporal context and order are critical, such as natural language processing, speech recognition, and notably, cyber-attack detection in sequential or time series data.
#### Key Components of LSTMs Include: 
* Memory Cells: These units act as the network’s “memory” and are capable of retaining information over long sequences. The cell state, flowing through the network, is modulated by various gates.
* Input Gate: This gate controls how much new information flows into the cell from the current input. It determines the importance of the incoming data in relation to what is already stored.
* Forget Gate: This component decides which information in the cell state should be discarded. By filtering out irrelevant details, it helps the network focus on significant patterns over time.
* Output Gate: This gate manages the amount of information from the cell state that is output at each time step, contributing to the prediction or classification made by the network.
* Activation Functions: Common non-linear functions like Sigmoid and Hyperbolic Tangent (tanh) are used within the gates to normalize the flow of information, ensuring smooth updates and effective learning.

### LSTMs in Cyber Attack Detection
The dynamic and evolving nature of cyber threats necessitates advanced techniques capable of processing and analyzing sequential data streams. Cyber attack detection often involves examining log files, network traffic sequences, and system events that arrive in chronological order. LSTMs, with their built-in memory and gating structures, are particularly effective for these tasks as they can learn complex temporal dependencies that may indicate an attack. The application of LSTMs in cyber attack detection represents a strategic convergence between time-series analysis and cybersecurity defense mechanisms. By leveraging the sequential modeling capabilities of LSTMs, security systems can: 
* Recognize Evolving Threats: Detecting gradual changes in network behavior or user activities over time, LSTMs can pinpoint subtle patterns that indicate slow-developing attacks.
* Proactively Respond to Incidents: By forecasting potential security incidents based on historical trends, LSTMs enable proactive threat mitigation, reducing the likelihood of large-scale breaches.
* Enhance Anomaly Detection: With their ability to learn normal behavior over time, LSTMs can flag deviations with greater sensitivity, thereby improving the overall accuracy of intrusion detection systems.
  
#### The Rationale for Using LSTMs in Cyber Attack Detection 
Cybersecurity demands techniques that can not only identify instantaneous anomalies but also recognize patterns that develop slowly or evolve over time. LSTMs offer several advantages: 
* Temporal Pattern Recognition
* Handling Noisy, Time-Stamped Data
* Predictive and Proactive Analysis
* Adaptability to Evolving Threats
#### Data Representation: From Raw Logs to Sequential Inputs
Applying LSTMs in cyber-attack detection typically begins with transforming raw cybersecurity data into a format that captures temporal dependencies: 
* Sequence Generation
* Feature Engineering
* Normalization and Encoding

#### === LSTM Evaluation ===
* Accuracy   0.797389
* Precision   0.765481
* Recall   0.797389
* F1-score   0.733188
* Training Time 4190.108582 seconds

### SHAP Filtering Implementation with LSTM
**Inferences** 
Based on the SHAP visualization output for the LSTM model, the following features have been identified as valid due to their SHAP values exceeding a specified threshold, indicating their significant contribution to the model's predictions. Valid SHAP Features from LSTM: ['Header_Length', 'Protocol Type', 'Rate', 'ack_flag_number', 'ack_count', 'HTTP', 'HTTPS', 'TCP', 'UDP', 'ICMP', 'Tot sum', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Variance']

## Hybrid CNN-LSTM Model
Cyber attacks are growing in frequency, complexity, and severity. As modern networks expand in scale and interconnectivity — particularly with the rise of the Internet of Things (IoT) — traditional rule-based intrusion detection systems (IDS) are struggling to keep up. These systems rely on known signatures and fixed heuristics, making them vulnerable to new and unseen threats. To overcome these limitations, machine learning (ML) and deep learning (DL) techniques are being increasingly adopted for intelligent cyber attack detection. Among these, hybrid deep learning models that combine different neural network architectures have shown remarkable promise. One such powerful architecture is the **CNN-LSTM hybrid model**, which blends the feature extraction capabilities of Convolutional Neural Networks (CNN) with the sequential pattern modeling strengths of Long Short-Term Memory networks (LSTM). Together, they offer a robust mechanism for detecting complex, temporal, and multivariate patterns in network traffic data, improving both accuracy and adaptability in intrusion detection systems.

#### Motivation for a CNN-LSTM Hybrid Approach
Network traffic data, especially from IoT environments, is often:
* High-dimensional: Many features per data point (e.g., packet flags, size, time-to-live, protocol).
* Sequential: Temporal patterns (i.e., the order of packets and their timing) matter.
* Noisy and unstructured: Different attack types manifest in diverse, complex forms.

While CNNs are excellent at capturing spatial dependencies and identifying local feature patterns, they do not retain temporal memory — making them insufficient alone for fully understanding sequences of events. On the other hand, LSTMs are designed to capture long-range temporal dependencies but are often less efficient when processing raw or noisy input data directly. The CNN-LSTM hybrid addresses these shortcomings by first using CNN layers to extract higher-level abstract representations from input data, and then passing those representations into LSTM layers that model temporal relationships across these features.

#### === Hybrid CNN-LSTM Evaluation ===
Metric: Value
* Accuracy: 0.800512
* Precision: 0.766534
* Recall: 0.800512
* F1-score: 0.741459
* Training Time: 749.193997 seconds

#### SHAP Filtering Implementation with Hybrid CNN-LSTM Model
**Inferences**<br>
Based on the SHAP visualization output for the Hybrid CNN-LSTM model, the following features have been identified as valid due to their SHAP values exceeding a specified threshold, indicating their significant contribution to the model's predictions. Valid SHAP Features from Hybrid CNN-LSTM: ['ack_flag_number', 'AVG', 'Tot size', 'Rate', 'HTTPS', 'IAT', 'Protocol Type', 'UDP', 'TCP', 'Variance', 'ICMP', 'HTTP', 'Header_Length', 'ack_count', 'Tot sum', 'Max', 'Std']

##  Hybrid XGBoost-CNN Model
As cyber threats evolve in complexity and frequency, traditional static rule-based security solutions are proving insufficient. Intrusion Detection Systems (IDS) powered by machine learning and deep learning offer promising alternatives, enabling proactive, adaptive defense mechanisms. Among these, hybrid models combinations of different algorithms or architectures have shown significant advantages in detecting sophisticated cyber-attacks. One such innovative hybrid is the XGBoost-CNN model, which combines XGBoost (Extreme Gradient Boosting), a powerful ensemble machine learning algorithm, with Convolutional Neural Networks (CNN), a deep learning model renowned for its pattern recognition capabilities. This hybrid approach seeks to exploit the strengths of both models XGBoost’s structured feature learning and CNN’s spatial abstraction to improve the accuracy and robustness of cyber attack detection.
#### Motivation for a Hybrid XGBoost-CNN Approach
The foundation for creating hybrid models stems from the recognition that no single model is ideal for all types of data. Specifically:
* XGBoost excels at handling tabular data, learning complex non-linear interactions between features, and delivering high classification accuracy with less overfitting. It's known for its speed and scalability.
* CNNs shine in feature extraction, especially when dealing with spatial hierarchies, images, and transformed numerical data, such as time series or 2D representations of structured data.

**Cybersecurity data** such as network logs, packet information, and user activity can be viewed both as structured features (ideal for XGBoost) and transformed multidimensional arrays (ideal for CNNs). The hybrid XGB-CNN model leverages both perspectives, allowing:
* XGBoost to act as a feature selector or generator, transforming raw features into high-quality feature representations or class probability vectors.
* CNN to use these representations as input, enabling deeper pattern recognition across spatial and temporal structures.

#### === Hybrid XGBoost-CNN Evaluation ===
* Accuracy 0.816653
* Precision  0.790713
* Recall 0.816653
* F1-score  0.784540 
* Training Time 642.196714 seconds

#### Detailed Comparison
**Structured Evaluation Table**
| Model         | Accuracy      |Precision     |Recall        |F1-score      |Training Time (seconds) |
| ------------- | ------------- |------------- |------------- |------------- |----------------------- |
| XGBoost       | 0.8139        |0.7871        |0.8139        |0.7795        |483.33                  |
| CNN           | 0.7899        |0.6594        |0.7899        |0.7090        |483.33                  |
| LSTM          | 0.7974        |0.7655        |0.7974        |0.7332        |4190.11                 |
| Hybrid CNN-LSTM  | 0.8005     |0.7665        |0.8005        |0.7415        |749.19                  |
| Hybrid XGB-CNN  | 0.8167      |0.7907        |0.8167        |0.7845        |642.20                  |

The analysis compares five models XGBoost, CNN, LSTM, Hybrid CNN-LSTM, and Hybrid XGBoost-CNN based on their performance in detecting cyber-attacks. The metrics include Accuracy (overall correctness), Precision (correctness of positive predictions), Recall (ability to identify all positives), F1-score (balance of Precision and Recall), and Training Time (computational efficiency), with weighted averages for Precision, Recall, and F1-score to account for class imbalance.
* **Accuracy**: Accuracy measures the proportion of correctly classified instances out of the total, reflecting overall model performance.
  * Hybrid XGB-CNN: Leads with 81.67% accuracy, indicating it correctly identifies the most attack instances across all classes.
  * XGBoost: Close second at 81.39%, showing strong performance for a standalone model.
  * Hybrid CNN-LSTM: Achieves 80.05%, slightly better than the standalone LSTM (79.74%) and CNN (78.99%).
  * CNN: Lags at 78.99%, suggesting it struggles with the complexity of multi-class attack detection alone.
  * LSTM: At 79.74%, it performs better than CNN but falls short of hybrid models and XGBoost.

The Hybrid XGB-CNN’s superior accuracy suggests that combining XGBoost’s structured data modeling with CNN’s spatial feature extraction enhances overall detection capability. 
* **Precision**:Precision (weighted average) indicates the correctness of positive predictions per class, weighted by class frequency. High precision reduces false positives, crucial for minimizing unnecessary alerts in cybersecurity.
  * Hybrid XGB-CNN: Tops at 0.7907, reflecting high reliability in its attack classifications.
  * XGBoost: Strong at 0.7871, nearly matching the hybrid model.
  * Hybrid CNN-LSTM: Solid at 0.7665, slightly better than LSTM alone (0.7655).
  * LSTM: Respectable at 0.7655, but lower than hybrid models.
  * CNN: Significantly lower at 0.6594, indicating a higher rate of false positives.

The Hybrid XGB-CNN and XGBoost excel in precision, likely due to XGBoost’s ability to model feature interactions effectively.
* **Recall**:Recall (weighted average) measures the ability to identify all instances of each class, weighted by frequency. High recall ensures fewer attacks are missed.
  * Hybrid XGB-CNN: Highest at 0.8167, aligning with its top accuracy.
  * XGBoost: Close at 0.8139, showing robust detection across classes.
  * Hybrid CNN-LSTM: At 0.8005, slightly ahead of LSTM (0.7974).
  * LSTM: Decent at 0.7974, but outperformed by hybrids.
  * CNN: Lowest at 0.7899, indicating missed detections.

The Hybrid XGB-CNN’s high recall underscores its effectiveness in capturing diverse attack types, a key requirement for cybersecurity. 
* **F1-score**:F1-score (weighted average) balances Precision and Recall, providing a single metric for model performance across imbalanced classes.
  * Hybrid XGB-CNN: Leads with 0.7845, reflecting a strong balance.
  * XGBoost: Close at 0.7795, competitive with the hybrid model.
  * Hybrid CNN-LSTM: At 0.7415, outperforms standalone LSTM (0.7332).
  * LSTM: At 0.7332, better than CNN but below hybrids.
  * CNN: Lowest at 0.7090, due to poor precision.

The Hybrid XGB-CNN’s top F1-score indicates it achieves the best trade-off between identifying attacks and avoiding false positives. The CNN’s low F1-score highlights its limitations in multi-class scenarios, while hybrid models mitigate these weaknesses.

* **Training Time**:Training Time measures computational efficiency, impacting scalability and deployment feasibility.
  * XGBoost: Fastest at 483.33 seconds, typical for tree-based models.
  * CNN: Matches XGBoost at 483.33 seconds, efficient for a neural network with few epochs.
  * Hybrid XGB-CNN: Moderate at 642.20 seconds, reflecting the combined overhead of CNN and XGBoost integration.
  * Hybrid CNN-LSTM: Slower at 749.19 seconds, due to LSTM’s sequential processing.
  * LSTM: Outlier at 4190.11 seconds (nearly 70 minutes), highlighting its computational intensity.

XGBoost and CNN are the most efficient standalone models, while the Hybrid XGB-CNN offers a reasonable trade-off between performance and training time.


