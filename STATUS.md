# Project 2: Packet Loss Event Classification with XGBoost & Federated Learning

This project aims to predict network packet loss events on a Metropolitan Milan SD-WAN testbed before they occur, based on historical latency (delay) measurements. The setup consists of three geographically distributed CPEs connected via fiber and 4G tunnels. 

## Project Status: 🟡 Execution & Refinement Phase

- **Latest Updates (June 2026):** 
  - **LSTM Integration & Homework Fusion:** Created `main_comparison_LSTM.py` to evaluate three completely different architectures side-by-side: XGBoost (tabular, from H1), MLP Neural Network (tabular, from H2), and LSTM (time-series, from H3). This unified pipeline standardizes evaluation across all course topics.
  - **Optimal Threshold Discovery (The PR-Curve Fix):** Discovered that a rigid `0.5` decision threshold masks the model's intelligence due to extreme class imbalance (giving empty confusion matrices). Modified `evaluate_model` to dynamically compute the **Optimal F1-Score Threshold** from the Precision-Recall curve (which often drops to ~0.08), drastically recovering True Positives.
  - **Data Physics & The "Invisible Fault":** Ran a direct Pearson correlation analysis on the raw signals. Discovered that on certain links (`cpe_a-cpe_c`), delay is highly predictive of future loss (delay doubles before failure). However, on other links (like `fiber` and `cpe_a-cpe_b`), packet loss is an instantaneous catastrophic event with **zero correlation** to historical delay. Implemented safe-guards in the pipeline to handle perfectly clean test sets (0 faults) where PR-optimization would otherwise divide by zero.
  - **Cross-Session Validation (OOD Testing):** To guarantee the models learn universal physics rather than memorizing daily traffic noise, we adopted a strict out-of-distribution split. We configure the YAML to train 100% on the *Second Capture Window* and test 100% on the *First Capture Window*.
  - **Federated Learning Sweeper:** Created `main_federated.py` that sweeps across different $N$ window sizes, executing parallel FedAvg for both MLP and LSTM, explicitly simulating network delays, and saving raw tensors (ROC/PR) and timing records incrementally into CSVs.

---

## Modular Architecture Design

```
NDAL-project-2026/
├── config               # Experiment and model configuration parameters (N, X, paths etc.)
│   ├── exp_federated.yaml
│   └── ...
├── main_comparison_LSTM.py   # Unified pipeline comparing XGB, MLP, and LSTM centrally
├── main_realtime_inference.py # Simulator testing prediction latency against strict X thresholds
├── main_federated.py         # Orchestrator simulating local clients, network delay, and FedAvg
├── STATUS.md                 # Project tracking and status
├── dataset/                  # Contains raw CSV data files
└── src/                      # Source code modules
    ├── __init__.py
    ├── data_loader.py        # Loading CSV files and chronological splitting
    ├── preprocessor.py       # Time series cleaning, -1 imputation, and sliding window generation
    ├── features.py           # Statistical feature engineering on lookback windows
    ├── models.py             # XGBoost and Neural Network training, tuning, and evaluation
    ├── federated.py          # Central controller and local client FL aggregation logic (FedAvg)
    └── utils.py              # Visualizations (feature importance, loss curves, Gantt charts)
```

---

## Implementation Plan & Checklist

### Phase 1: Exploration & Setup
- [x] Parse configuration settings from `*.yaml`.
- [x] Load and visualize raw time series data to analyze the delay vs. packet loss correlation.
- [x] Setup loggers and results directory structure.

### Phase 2: Data Preprocessing & Feature Engineering
- [x] Preprocess packet loss events by replacing `delay_ms = -1` with `NaN` so that subsequent window metrics are computed strictly on valid, successfully received packets.
- [x] Implement the sliding window extractor (Statistical + Raw Sequence).
- [x] Engineer statistical features from the lookback window (Mean, Jitter, Quantiles, Slopes).
- [x] Handle edge cases for "Link Down" windows.
- [x] Implement flexible dataset splitting: **Cross-Session Validation** proved to be the most rigorous approach (train on Window 2, test on Window 1) to avoid temporal data leakage.

### Phase 3: Model Development & Real-Time Profiling
- [x] Build **XGBoost Classifier** model (with hyperparameter tuning and SHAP).
- [x] Build **Neural Network** (Multi-Layer Perceptron) model.
- [x] Evaluate performance using metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- [x] Address Class Imbalance via `scale_pos_weight` and **SMOTE**.
- [x] **Real-Time Inference Profiling:** Created `main_realtime_inference.py` to act as a simulated router. Evaluated inference latencies via ECDF and Boxplots against hard prediction thresholds ($X=0.5s, 1s$). Confirmed that while tabular models (XGBoost, MLP) are lightning-fast, deep sequence models (LSTM) carry significant computational overhead that may violate microsecond real-time constraints.

### Phase 4: Federated Learning Simulation
- [x] **Architecture Choice (Horizontal Federated Learning - HFL):** Our setup uses **Horizontal FL**. All participating local clients (CPEs) share the exact same **Feature Space** (delay, jitter, packet loss metrics), but they possess entirely different **Sample Spaces** (traffic events from distinct geographical routes). Neural Network architectures are identical, enabling direct mathematical weight averaging.
- [x] **Decentralized Data Strategy:** Geographically partitioned the dataset into 3 isolated clients (CPE_A, CPE_B, CPE_C).
- [x] **Local Model Training & Parallelization:** Used `ThreadPool` in Python to simulate the *simultaneous* physical execution of local training rounds across the 3 routers. 
- [x] **Central Aggregation (FedAvg):** Implemented `src/federated.py` to extract weights, average them, and broadcast them back, applying a realistic **Network Latency Penalty** (e.g., 150ms) per round.
- [x] **Performance & Time Benchmarking (The Straggler Problem):**
  - Swept through lookback windows $N=[15, 30, 60]$.
  - **Time Findings:** For lightweight models (MLP), Centralized Training is faster because the math is trivial and FL suffers from the network delay penalty. However, for computationally heavy models (LSTM), the **parallelization of Federated Learning** offsets the network delay, making it competitive or faster than the bottlenecked Centralized server.
  - **The Straggler Effect:** Visualized via Stacked Bar Plots (`plot_fl_training_times`). Discovered that `CPE_B` takes vastly more time because it processes significantly more real-world traffic logs. Fast routers sit entirely "idle" waiting for CPE_B to finish, highlighting the primary physical bottleneck of FedAvg.
  - **The Non-IID Problem:** Acknowledged that since fault events are extremely rare (0.15%), a client with perfectly clean traffic (0 anomalies) will learn useless weights. During FedAvg, these inert weights act as an "anchor", dragging down the Global Model's F1-Score compared to a Centralized model that sees all data combined.

### Phase 5: Advanced Experimentation & Verification
- [ ] **Leave-One-Link-Out (LOLO) Spatial Test:** Train the model on 5 links (e.g., A->B, B->A, A->C, C->A, B->C) and test strictly on the 6th unseen link (e.g., C->B). Evaluates the model's zero-shot adaptability to newly installed routes.
- [x] **Window Size Physics (N, X Tuning):** Executed via dynamic sweeping over extreme combinations ($N=15, 30, 60$).
- [x] Produce comparative charts showing performance variation across different configurations (saved continuously as raw CSV tensors for ROC, PR, and Timing).
- [ ] Write a final report / walkthrough summarizing results, best parameters, and insights.
