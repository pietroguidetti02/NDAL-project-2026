# Project 2: Packet Loss Event Classification with XGBoost & Federated Learning

This project aims to predict network packet loss events on a Metropolitan Milan SD-WAN testbed before they occur, based on historical latency (delay) measurements. The setup consists of three geographically distributed CPEs connected via fiber and 4G tunnels. 

## Project Status: 🟢 Planning Phase

- **Project Goal:** Train XGBoost and Neural Network classifiers to predict packet loss in a future window $X$ using a lookback window $N$, and simulate a Federated Learning setup to compare local vs. federated models.
- **Approach:** Modular Python scripts, organized for reproducibility, clean logging, and configuration flexibility.

---

## Modular Architecture Design

```
NDAL-project-2026/
├── config.yaml               # Experiment and model configuration parameters (N, X, paths, etc.)
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
- [ ] Parse configuration settings from `config.yaml`.
- [ ] Load and visualize raw time series data to analyze the delay vs. packet loss correlation.
- [ ] Setup loggers and results directory structure.

### Phase 2: Data Preprocessing & Feature Engineering
- [ ] Preprocess packet loss events (imputing `delay_ms = -1` via forward fill or interpolation).
- [ ] Implement the sliding window extractor:
  - Input: Lookback window of size $N$ (e.g., 15s, 30s, 60s)
  - Target: Binary label (1 if packet loss occurs in prediction window $X$, else 0)
- [ ] Engineer statistical features from the lookback window:
  - Mean, Jitter (standard deviation), Max, Min, Median
  - Quantiles (90th, 95th, 99th) to capture tail spikes
  - Trend / Slope (using linear regression coefficients)
  - Rate of delay changes and recent delta
  - Historical packet loss counts
- [ ] Split dataset chronologically: keep the last few hours of each window as a hold-out test set to avoid temporal data leakage.

### Phase 3: Model Development (Local Models)
- [ ] Build **XGBoost Classifier** model.
  - Implement hyperparameter tuning (max_depth, learning_rate, n_estimators) with cross-validation.
  - Evaluate feature importance using XGBoost's built-in importance and SHAP/permutation.
- [ ] Build **Neural Network** (Multi-Layer Perceptron) model.
  - Scale features using standard scaling.
  - Implement hyperparameter tuning (layer sizes, learning rate, activation function).
- [ ] Evaluate performance using metrics: Accuracy, Precision, Recall, F1-score, MAE/MSE (if applicable), and Confusion Matrix.

### Phase 4: Federated Learning Simulation
- [ ] Define Local Client representing each directional CPE path/node.
- [ ] Implement Central Controller logic to aggregate model updates:
  - **Neural Network:** Perform Federated Averaging (FedAvg) on weights:
    $$\theta_{global} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_{local, k}$$
  - **XGBoost:** Implement model ensembling/bagging (collecting trees from all local clients) or run federated training on neural network only and compare.
- [ ] Train local models, perform central aggregation rounds, and redistribute weights.
- [ ] Compare performance: **Global Federated Model** vs. **Individual Local Models** tested on local hold-out sets.

### Phase 5: Experimentation & Verification
- [ ] Execute `run_experiments.py` over combinations of:
  - $N \in \{15, 30, 60\}$ seconds
  - $X \in \{5, 10, 20\}$ seconds
- [ ] Produce comparative charts showing performance variation across different $(N, X)$ parameters.
- [ ] Write a final report / walkthrough summarizing results, best parameters, and insights.
