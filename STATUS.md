# Project 2: Packet Loss Event Classification with XGBoost & Federated Learning

This project aims to predict network packet loss events on a Metropolitan Milan SD-WAN testbed before they occur, based on historical latency (delay) measurements. The setup consists of three geographically distributed CPEs connected via fiber and 4G tunnels. 

## Project Status: 🟡 Execution & Refinement Phase

- **Latest Updates (June 2026):** 
  - **LSTM Integration & Homework Fusion:** Created `main_comparison_LSTM.py` to evaluate three completely different architectures side-by-side: XGBoost (tabular, from H1), MLP Neural Network (tabular, from H2), and LSTM (time-series, from H3). This unified pipeline standardizes evaluation across all course topics.
  - **Optimal Threshold Discovery (The PR-Curve Fix):** Discovered that a rigid `0.5` decision threshold masks the model's intelligence due to extreme class imbalance (giving empty confusion matrices). Modified `evaluate_model` to dynamically compute the **Optimal F1-Score Threshold** from the Precision-Recall curve (which often drops to ~0.08), drastically recovering True Positives.
  - **Data Physics & The "Invisible Fault":** Ran a direct Pearson correlation analysis on the raw signals. Discovered that on certain links (`cpe_a-cpe_c`), delay is highly predictive of future loss (delay doubles before failure). However, on other links (like `fiber` and `cpe_a-cpe_b`), packet loss is an instantaneous catastrophic event with **zero correlation** to historical delay. Implemented safe-guards in the pipeline to handle perfectly clean test sets (0 faults) where PR-optimization would otherwise divide by zero.
  - **Dynamic Configuration:** Extracted `tunnel_types` into the `config.yaml` to allow seamless toggling between evaluating solely on `mobile` or `fiber` without altering source code.
  - Addressed extreme class imbalance (packet loss < 0.5%) by implementing **SMOTE** and `scale_pos_weight` and custom `class_weights` with initial bias tuning for Deep Learning.
  - **4-Way Feature Evaluation Experiment (Statistical vs. Raw Sequence):** 
    - **XGBoost** performed best with **Statistical Features** (F1: 0.40, Recall: 77%). Decision trees struggle to extract temporal patterns from raw arrays, relying heavily on hand-crafted aggregate features like jitter.
    - **Neural Network (MLP)** performed significantly better with **Raw Sequence Features** (F1: 0.68, Recall: 62%, Precision: 75%).
- **Project Goal:** Train XGBoost, Neural Networks, and LSTMs to predict packet loss in a future window $X$ using a lookback window $N$, and simulate a Federated Learning setup to compare local vs. federated models.
- **Approach:** Modular Python scripts, organized for reproducibility, clean logging, live ROC/PR curve plotting, and configuration flexibility.

---

## Modular Architecture Design

```
NDAL-project-2026/
├── config               # Experiment and model configuration parameters (N, X, paths etc.)
│   └──exp1.yaml
│   └──exp2.yaml
│   └──exp3.yaml
│   └──...
├── exploratory_analysis.ipynb # Interactive EDA and data visualization
├── main.py                   # Orchestrator script to run data prep, training, FL, and evaluation
├── run_experiments.py        # Automation script to test different N (15s, 30s, 60s) and X (5s, 10s, 20s)
├── STATUS.md                 # Project tracking and status
├── dataset/                  # Contains raw CSV data files
│   ├── first_capture_window/
│   └── second_capture_window/
└── src/                      # Source code modules
    ├── __init__.py
    ├── data_loader.py        # Loading CSV files and chronological splitting
    ├── preprocessor.py       # Time series cleaning, -1 imputation, and sliding window generation
    ├── features.py           # Statistical feature engineering on lookback windows
    ├── models.py             # XGBoost and Neural Network training, tuning, and evaluation
    ├── federated.py          # Central controller and local client FL aggregation logic (FedAvg)
    └── utils.py              # Visualizations (feature importance, loss curves) and helper functions
```

---

## Implementation Plan & Checklist

### Phase 1: Exploration & Setup
- [x] Parse configuration settings from `*.yaml`.
- .yaml file structure example:
[N,X]: [15,5]
train/test:
	- A->B: [80, 20], [0, 0]
	- B->A: [80, 20], [0, 0]
	- B->C: none
	- C->B: none
	- A->C: [80, 20], [0, 0]
	- C->A: [80, 20], [0, 0]
  in this case it will use only, A to B, B to A with that percentage of triaing of first window and 0, of the second for both.
merging of dataset after parsing them from csv files.

- [x] Load and visualize raw time series data to analyze the delay vs. packet loss correlation.
- [x] Setup loggers and results directory structure.

### Phase 2: Data Preprocessing & Feature Engineering
- [x] Preprocess packet loss events by replacing `delay_ms = -1` with `NaN` so that subsequent window metrics are computed strictly on valid, successfully received packets (preserving the true delay distribution and avoiding bias from `-1`).
- [x] Implement the sliding window extractor:
  - Input: Statistical features computed over a lookback window of size $N$ (and/or the raw time-series window itself).
  - Target: Binary label (1 if packet loss occurs in prediction window $X$, else 0).
- [x] Engineer statistical features from the lookback window (skipping `NaN` values to represent the actual received traffic):
  - Base statistics: Mean, Jitter (standard deviation), Max, Min, Median
  - Tail spikes: Quantiles (90th, 95th, 99th)
  - Trend / Slope (`trend_slope` via rolling linear regression)
  - Rate of delay changes (`delay_change_rate`)
  - Recent delay delta (`recent_delta` over a short horizon $H$, e.g., 5s)
  - Historical packet loss counts (`hist_loss_count` — counting the number of original `-1` events in the window)
- [x] Handle edge cases for "Link Down" windows (where all values in the lookback window are `NaN` because of complete packet loss for $N$ seconds) by imputing default maximum delay/congestion values on the computed features.
- [x] Implement flexible dataset splitting and cross-window scenarios:
  - Support chronological splits (e.g., train on first 80%, test on remaining 20% of selected windows) to prevent temporal data leakage.
  - Support training on a subset of capture windows (e.g., only first_capture_window, only second_capture_window, or both) and testing on the remainder, customizable via `config.yaml`.

### Phase 3: Model Development (Local Models)
- [x] Build **XGBoost Classifier** model.
  - Implement hyperparameter tuning (max_depth, learning_rate, n_estimators) with cross-validation.
  - Evaluate feature importance using XGBoost's built-in importance and SHAP/permutation.
  - read homework1 files to get inspiration.
- [x] Build **Neural Network** (Multi-Layer Perceptron) model.
  - Scale features using standard scaling.
  - Implement hyperparameter tuning (layer sizes, learning rate, activation function). 
- read homework2 files to get inspiration. 
- [x] Evaluate performance using metrics: Accuracy, Precision, Recall, F1-score, MAE/MSE (if applicable), and Confusion Matrix.
- [x] Address Class Imbalance via `scale_pos_weight` and **SMOTE** (Synthetic Minority Over-sampling Technique).

### Phase 4: Federated Learning Simulation
- [ ] **Architecture Choice (Horizontal Federated Learning - HFL):** Our setup is a classic case of **Horizontal** (or sample-based) Federated Learning. 
  - *Reasoning for Slides:* All participating local clients (CPEs) share the exact same **Feature Space** (the columns: delay, jitter, packet loss), but they possess entirely different **Sample Spaces** (the rows: traffic events from distinct geographical routes like A->B vs C->A). Because the Neural Networks across all nodes have identical input architectures, we can directly average their mathematical weights. (In contrast, Vertical FL is used when clients share the same samples but hold different features, which doesn't apply here).
- [ ] **Decentralized Data Strategy:** Partition the dataset logically into completely isolated clients (e.g., Client A, Client B, Client C). Clients must *never* share raw CSV telemetry data to preserve bandwidth and privacy.
- [ ] **Local Model Training:** Each client initializes a local clone of the Neural Network (MLP/LSTM) and trains it strictly on its own local data for a small number of epochs.
- [ ] **Central Aggregation (FedAvg):** 
  - Instead of data, clients send only their learned network weights (`model.get_weights()`) to the Central Server.
  - The server averages the weights: $$\theta_{global} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_{local, k}$$
  - The server redistributes the Global Model back to all clients.
  - *Note: XGBoost is generally excluded from this federated averaging process due to the mathematical complexity of merging decision trees. Focus FL solely on MLP/LSTM.*
- [ ] **Performance Benchmarking:** Compare the **Federated Model** (which learns from all nodes without seeing their data) against the **Centralized Baseline** (our current script) and the **Strictly Local Models**.

- Bonus (angelo): include in the results an experimentation to give a sense of real difference between time of computation of big centralized model, against the time that u put in calculating the decentralized con cpes and merging all informations toghether, assuming a fixed delay and penalty of ip transport and physical transport.
so it is important always to measure the time needed to train a model. and we must redo simulations for big centralized model.
It is needed to decide which data (smaller part) to use. not whole dataset.

- Bonus2: ogni quanto è necessario mandare i pesi dalle macchine al controller centrale?


### Phase 5: Advanced Experimentation & Verification
- [ ] **Leave-One-Link-Out (LOLO) Spatial Test:** Train the model on 5 links (e.g., A->B, B->A, A->C, C->A, B->C) and test strictly on the 6th unseen link (e.g., C->B). Evaluates the model's zero-shot adaptability to newly installed routes.
- [x] **Window Size Physics (N, X Tuning):** Execute `run_experiments.py` over extreme combinations:
  - "Far-Sight" test: $N=60s$, $X=60s$. How far into the future can the model predict before the signal degrades?
  - "Short-Sight" test: $N=5s$, $X=5s$. Is a 5-second history enough to predict an immediate anomaly?
- [x] Produce comparative charts showing performance variation across different configurations.
- [ ] Write a final report / walkthrough summarizing results, best parameters, and insights.
