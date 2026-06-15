# CatBoost Integration for Intrusion Detection in Evolving IoT Environments

An interactive Intrusion Detection System (IDS) research dashboard that evaluates CatBoost's native categorical handling against six baseline ML algorithms across traditional (UNSW-NB15) and modern IoT (CICIoT 2023) network traffic datasets.

## Overview

This project implements a binary network intrusion detection classifier that distinguishes benign traffic from attacks. It investigates how **CatBoost's Ordered Target Statistics** — a native categorical feature encoding method — handles the cross-dataset distribution shift problem that plagues IDS models when deployed across evolving network environments.

**Key research question:** Can CatBoost's native categorical handling enable more stable transfer of predictive logic between traditional university network traffic (UNSW-NB15, 2015) and modern IoT attack testbeds (CICIoT 2023)?

The answer is a qualified yes — but only when trained on a multi-domain "Master" dataset that includes both distributions. Single-domain training leads to catastrophic ranking collapse on cross-domain evaluation (AUC as low as 14.81%).

## Project Structure

```
ML-IDS-Streamlit/
  app.py                # Main Streamlit application (1886 lines)
  cb_master.joblib      # Trained CatBoost model (Master dataset)
  requirements.txt      # Python dependencies
  README.md             # This file
```

### Architecture

The application consists of a single-page Streamlit dashboard with six navigable panels:

| Page | Purpose |
|------|---------|
| **Simulation Lab** | Test pre-built traffic scenarios against the trained model in real time. Attack and benign scenarios with live prediction gauges, feature radar plots, and a "What-If" slider for exploring decision boundaries. |
| **Scenario Encyclopedia** | Comprehensive documentation of every built-in traffic scenario — attack technique descriptions, severity ratings, dataset provenance, and full feature breakdowns with visualisations. |
| **Model Performance** | Interactive cross-dataset evaluation matrix. Filter by algorithm, training set, and test set across 5 metrics (Accuracy, Precision, Recall, F1, AUC). Includes heatmaps, grouped bar charts, and radar comparisons. |
| **Manual Prediction** | Custom feature input with three synchronised modes: preset scenario, slider, or direct numeric input. Predictions update instantly on any change. |
| **Model Insights** | Empirical decision boundary analysis documenting a discovered service-label bias in the CatBoost model, safe benign parameter ranges, and the connection between training data composition and classification behaviour. |
| **About** | Full research context: background, related work comparison table, research objectives, dataset statistics, unified feature schema, CatBoost methodology, hyperparameters, key results across all test sets, and discussion of findings. |

## Datasets

### UNSW-NB15
- Captured by the Australian Centre for Cyber Security using IXIA PerfectStorm (2015)
- Contains 9 attack categories including Fuzzers, Backdoors, Shellcode, and Reconnaissance
- Represents traditional university network traffic with balanced benign/attack distribution
- Training split: 82,332 rows (44.9% benign, 55.1% attack)

### CICIoT 2023
- Created by the Canadian Institute for Cybersecurity (2023)
- Focuses on IoT network traffic including DDoS, DoS, reconnaissance, and data exfiltration
- Raw dataset is 97.64% attacks — undersampled to match UNSW class ratios
- Training split (balanced): 82,332 rows (44.9% benign, 55.1% attack)

### Master Dataset
- Concatenation of UNSW-NB15 and CICIoT 2023 training splits
- 164,664 rows total with balanced class distribution
- Designed to evaluate whether multi-domain training mitigates cross-dataset distribution shift

## Unified 10-Feature Schema

A semantically consistent feature set derived from both datasets:

| Feature | Type | Description |
|---------|------|-------------|
| flow_duration | Numerical | Total flow duration in seconds |
| rate | Numerical | Overall packet rate (packets per second) |
| srate | Numerical | Source-to-destination packet rate |
| drate | Numerical | Destination-to-source packet rate |
| tot_bytes | Numerical | Total bytes transferred (sbytes + dbytes) |
| avg_pkt_size | Numerical | Average bytes per packet |
| weight | Numerical | Flow weight (product of source/destination packet counts) |
| proto | Categorical | Transport protocol (tcp / udp / other) |
| service | Categorical | Application service (http / ssl / dns / ssh / smtp / other) |
| state | Categorical | Connection state (fin / rst / other) |

Categorical values are mapped to a common vocabulary across both datasets (e.g., non-overlapping services mapped to "other").

## Models Evaluated

Seven algorithms trained on three datasets each, evaluated on three test sets — 63 total configurations:

- **CatBoost** (proposed model) — gradient boosting with Ordered Target Statistics
- Random Forest
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Key Results

### CatBoost Master-Trained (best overall)
| Metric | Value |
|--------|-------|
| Accuracy | 94.38% |
| Precision | 98.86% |
| Recall | 93.49% |
| F1 Score | 96.10% |
| AUC | 99.25% |

### Cross-Dataset Degradation (UNSW-trained, tested on CICIoT)
| Model | AUC |
|-------|-----|
| CatBoost | 14.81% (worse than random) |
| Random Forest | 13.92% |
| Decision Tree | 48.72% |
| SVM | 19.14% |

This demonstrates that highly optimised ensemble methods leverage fine-grained feature interactions that do not transfer across heterogeneous network environments.

### In-Domain Performance (CICIoT-trained, CICIoT-tested)
| Model | Accuracy | AUC |
|-------|----------|-----|
| CatBoost | 98.85% | 99.75% |
| Random Forest | 98.50% | 99.73% |
| Decision Tree | 98.62% | 99.12% |

### Key Finding: Service-Label Bias

The CatBoost model exhibits a measurable **service-label bias**: the `service` categorical feature alone can shift prediction confidence from 25% (BENIGN) to 96% (ATTACK) while holding all numerical features constant. This is a direct empirical consequence of CatBoost's Ordered Target Statistics encoding the imbalanced class-ratio per service label from the CICIoT dataset, where HTTP, SMTP, SSH, and SSL flows are overwhelmingly attack-classified.

This finding is documented as a model insight in the dashboard and directly supports the paper's thesis on categorical distribution shift.

## Tech Stack

- **CatBoost** — gradient boosting with native categorical feature handling (Ordered Target Statistics, Symmetric Trees, Ordered Boosting)
- **scikit-learn** — baseline algorithms (Random Forest, SVM, Logistic Regression, Decision Tree, Gradient Boosting)
- **pandas / NumPy** — data processing and feature engineering
- **Streamlit** — interactive dashboard framework
- **Plotly** — interactive visualisations (gauges, radar charts, heatmaps, box plots)
- **joblib** — model serialisation

## Setup and Usage

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML-IDS-Streamlit

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Model File

The trained CatBoost model (`cb_master.joblib`) must be present in the working directory. This model was trained on the Master dataset (UNSW-NB15 + CICIoT 2023 combined) and achieves the best cross-domain performance.

## Research Context

This project was developed as part of a research study on **cross-dataset distribution shift in network intrusion detection systems**. The findings demonstrate that:

1. **CatBoost's Ordered Target Statistics** effectively encode categorical features without one-hot encoding, but faithfully represent the statistical biases present in training data
2. **Single-domain training is insufficient** for cross-domain deployment — models achieve near-perfect in-domain performance but collapse on out-of-domain evaluation
3. **Multi-domain (Master) training** restores robustness by providing balanced categorical coverage across both traditional and IoT network paradigms
4. **The service categorical feature dominates classification decisions** — a direct consequence of the statistical composition of training data rather than a model implementation flaw

### References

- Fathima et al. (2023) — Baseline ML evaluation on UNSW-NB15
- Hajjouz & Avksentieva (2024) — CatBoost on CICIoT 2023 with external feature selection
- Yan, Zhou & Chen (2025) — Contrastive learning for cross-dataset IDS
- Gulzar & Mustafa (2025) — DeepCLG hybrid ensemble on CICIoT 2023
- Al-Riyami et al. (2021) — Cross-dataset categorical mismatch in IDS

## License

This project is provided for research and educational purposes.

Link: https://cs20-ids.streamlit.app
