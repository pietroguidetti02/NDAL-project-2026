# Project 2: Packet Loss Event Classification with XGBoost & Federated Learning

This project aims to predict network packet loss events on a Metropolitan Milan SD-WAN testbed before they occur, based on historical latency (delay) measurements. The setup consists of three geographically distributed CPEs connected via fiber and 4G tunnels. 

## Project Status: 🟡 Execution & Refinement Phase

- **Latest Updates (May 2026):** 
  - Addressed extreme class imbalance (packet loss < 0.5%) by implementing **SMOTE** and `scale_pos_weight`.
  - Expanded dataset configuration to include all CPE pairs across both capture windows, creating a massive global dataset.
  - Lowered XGBoost decision threshold to 10% to prioritize Recall over Precision.
  - **4-Way Feature Evaluation Experiment (Statistical vs. Raw Sequence):** Inspired by state-of-the-art literature on QoT Forecasting, we evaluated model performance using both aggregated statistical features (mean, jitter) and raw chronological sequences (15 exact delay values).
    - **XGBoost** performed best with **Statistical Features** (F1: 0.40, Recall: 77%). Decision trees struggle to extract temporal patterns from raw arrays, relying heavily on hand-crafted aggregate features like jitter.
    - **Neural Network (MLP)** performed significantly better with **Raw Sequence Features** (F1: 0.68, Recall: 62%, Precision: 75%). By feeding the raw chronological delays, the MLP successfully learned the exact temporal signature preceding a packet loss on the 4G network, drastically reducing false positives compared to XGBoost and proving the viability of sequence-based forecasting for anomaly detection.
  - **Key Insight on SD-WAN Predictive Routing (Fiber vs 4G):** 
    - **4G Mobile:** Statistical features (delay trends, jitter) show gradual degradation before a loss, allowing the models to successfully predict drops. The low precision (~15%) vs high recall is actually optimal for SD-WAN: a false alarm simply causes a safe, temporary traffic reroute to Fiber, whereas missing a drop (False Negative) degrades the user's VoIP/Video experience.
    - **Fiber:** Models fail to predict loss (0 True Positives) because fiber drops are exceedingly rare (e.g., 5 losses in over 104,000 test windows) and happen instantaneously. There is absolutely no preceding statistical degradation in the 15-second lookback window to warn the model.
  - Demonstrated via KDE feature distributions that the remaining false positives are due to the stochastic nature of network drops (features pre-loss heavily overlap with normal traffic).
- **Project Goal:** Train XGBoost and Neural Network classifiers to predict packet loss in a future window $X$ using a lookback window $N$, and simulate a Federated Learning setup to compare local vs. federated models.
- **Approach:** Modular Python scripts, organized for reproducibility, clean logging, and configuration flexibility.

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
