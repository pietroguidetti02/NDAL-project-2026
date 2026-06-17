# NDAL Project #2 - Packet Loss Event Classification

**Group Members:** (Add your names and student IDs here)

## 1. Project Overview & Architecture
This project aims to predict network packet loss events on a Metropolitan Milan SD-WAN testbed before they occur, based on historical latency (delay) measurements. We implemented Machine Learning algorithms (XGBoost, Neural Networks) and simulated a **Federated Learning** environment to compare decentralized training versus centralized data aggregation.

### Project Structure
```text
NDAL-project-2026/
├── config/                      # YAML configuration files for experiments (N, X, paths)
├── dataset/                     # Contains raw CSV data files for CPEs
├── src/                         # Source code modules
│   ├── data_loader.py           # Loading and chronological splitting
│   ├── preprocessor.py          # Time series cleaning, -1 imputation, sliding window
│   ├── features.py              # Statistical feature engineering
│   ├── models.py                # XGBoost and NN training, tuning, and evaluation
│   ├── federated.py             # Central controller and local client FedAvg logic
│   └── utils.py                 # Visualizations and helpers
├── results/                     # Saved outputs (models, metrics, plots)
├── 01_exploratory_analysis.ipynb
├── 02_main_comparison_LSTM.py
├── 03_main_sweep.py
├── 04_main_realtime_inference.py
└── 05_main_federated.py
```

## 2. Setup and Installation
Make sure you have a Python 3 environment active. The main dependencies required to run the scripts are:
- `pandas`, `numpy` (data manipulation)
- `scikit-learn`, `xgboost`, `torch` (machine learning models)
- `matplotlib`, `seaborn` (visualization)
- `shap` (feature importance)
- `pyyaml` (configuration parsing)

Installing via pip:
```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib seaborn shap pyyaml
```

## 3. Experiments (How to reproduce the results)

The project is structured into sequential experiments that cover all assignment requirements:

### Phase 1: Data Analysis (Requirement 1)
* **File:** `01_exploratory_analysis.ipynb`
* **Purpose:** Analyzes the correlation between delay and packet loss.
* **Output:** Exploratory plots showing data distribution and signal behavior before failure events.

### Phase 2: Model Training & Comparison (Requirement 2 & 3 - 12pt)
* **File:** `02_main_comparison_LSTM.py`
* **Purpose:** Trains XGBoost and a Neural Network (MLP), plus an LSTM baseline. Evaluates feature importance (SHAP/Gain) and handles class imbalance. It extracts features dynamically based on the Lookback window.
* **Output:** Trained models, confusion matrices, feature importance plots, and PR-Curves in the `results/` folder.

### Phase 3: Sliding Window N & X Tuning (Requirement 4 - 12pt)
* **File:** `03_main_sweep.py`
* **Purpose:** Sweeps across different values for the Lookback window ($N$) and Prediction window ($X$) to identify the optimal prediction horizon.
* **Output:** Comparative ROC/PR curves and tabular metrics saved across sweeps.

### Phase 4: Additional Profiling (Extra)
* **File:** `04_main_realtime_inference.py`
* **Purpose:** Tests inference latency to ensure predictions fit within strict real-time routing constraints.

### Phase 5: Federated Learning (Advanced Task - 3pt)
* **File:** `05_main_federated.py`
* **Purpose:** Simulates an SD-WAN Controller and 3 Local Clients. Performs Federated Averaging (FedAvg) and compares the performance of the Generalized Model versus the Local, CPE-specific models.
* **Output:** Federated vs Local performance charts and training time analysis (Straggler effect).

### Phase 6: Additional Profiling (Extra)
* **File:** `06_main_realtime_inference_fed.py`
* **Purpose:** Tests inference latency to ensure predictions fit within strict real-time routing constraints. Comparison between federated and centralized performances.


## 4. Key Findings & Conclusions
* **Feature Importance:** Delay metrics (mean, quantiles) in the lookback window are highly predictive on certain links (e.g., `cpe_a-cpe_c`), while other links exhibit catastrophic packet drops with near-zero prior warning.
In particular fiber has very poor performances wrt to our system.
* **Model Performance:** XGBoost significantly outperforms deep sequence models on tabular extracted features in terms of both inference speed and detection accuracy (optimized F1-Score).
* **Federated Learning:** While FedAvg successfully aggregates knowledge, the severe data imbalance (Non-IID) across clean vs noisy links introduces an anchor effect, slightly lowering the global F1-Score compared to a fully Centralized model. However, the Federated model inherently generalizes better to previously unseen routes.
