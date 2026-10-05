# NDAL Project #2 - Packet Loss Event Classification

**Group Members:** Angelo Ferrara, Pietro Guidetti, Rodrigo Barrero

## 1. Project Overview & Architecture
This project predicts packet loss events on a metropolitan (Milan) SD-WAN testbed before they occur, using historical delay measurements. Three CPEs are connected through fiber and 4G tunnels. We compare XGBoost, an MLP and an LSTM, simulate a **Federated Learning** (FedAvg) deployment against local and centralized training, and benchmark online learning latency on router-class hardware (Raspberry Pi 3 B+).

### Project Structure
```text
NDAL-project-2026/
├── config/                             # YAML configuration files for experiments (N, X, paths)
├── dataset/                            # Raw CSV data files for CPEs (not tracked in git, see below)
├── src/                                # Source code modules
│   ├── data_loader.py                  # Loading and chronological splitting
│   ├── preprocessor.py                 # Time series cleaning, -1 imputation, sliding window
│   ├── features.py                     # Statistical feature engineering
│   ├── models.py                       # XGBoost, MLP and LSTM training and evaluation
│   ├── federated.py                    # FedAvg client/server, update size and communication cost model
│   ├── plot_style.py                   # Shared paper figure style (fonts, colors, markers)
│   └── utils.py                        # Visualizations and helpers
├── results/                            # Saved outputs (models, metrics, plots) - not tracked in git
├── 01_exploratory_analysis.ipynb       # Data/correlation exploration
├── 02_a_main.py                        # XGBoost + MLP training and comparison
├── 02_b_main_comparison_LSTM.py        # XGBoost, MLP and LSTM comparison
├── 03_main_sweep.py                    # Lookback/prediction window (N, X) sweep
├── 04_main_realtime_inference.py       # Inference latency profiling (centralized, x86)
├── 05_main_federated.py                # Federated Learning simulation (FedAvg)
├── 06_main_realtime_inference_fed.py   # Inference latency profiling (federated, x86)
├── 07_main_hyperparameter_tuning.py    # Hyperparameter tuning
├── 08_main_realtime_inference_fed_rasp_constrained.py  # Online learning latency on constrained hardware
├── 09_paper_figures.py                 # Regenerates the manuscript figures from saved results
├── requirements.txt
└── LICENSE
```

### Data Availability
The raw CSV measurement files under `dataset/` are not tracked in this repository (see `.gitignore`) to keep the git history lightweight. Contact the authors or check the linked manuscript for access to the dataset.

## 2. Setup and Installation
Use Python 3.13 (other recent 3.x versions are likely to work). Install the dependencies with:
```bash
pip install -r requirements.txt
```
`requirements.txt` lists minimum versions and the exact versions used for the Raspberry Pi benchmark.

## 3. Experiments (How to reproduce the results)

### Phase 1: Data Analysis
* **File:** `01_exploratory_analysis.ipynb`
* **Purpose:** Analyzes the correlation between delay and packet loss on fiber and 4G links.

### Phase 2: Model Training & Comparison
* **Files:** `02_a_main.py`, `02_b_main_comparison_LSTM.py`
* **Purpose:** Trains XGBoost, an MLP and an LSTM on the same windows, handles class imbalance, evaluates feature importance (SHAP/gain) and picks the F1-optimal decision threshold from the PR curve.

### Phase 3: Sliding Window N & X Sweep
* **File:** `03_main_sweep.py`
* **Purpose:** Sweeps the lookback window $N$ and the prediction horizon $X$ under three data splits (S1 single dataset, S2 spatial split, S3 temporal split).

### Phase 4: Inference Latency (x86)
* **File:** `04_main_realtime_inference.py`
* **Purpose:** Measures inference latency of the three models on the x86 server.

### Phase 5: Federated Learning
* **File:** `05_main_federated.py`
* **Purpose:** Simulates an SD-WAN controller and 3 CPE clients. Compares Local, Federated (FedAvg) and Centralized training for MLP and LSTM, and records per-round compute, idle (straggler) and network time.
* **Reproducibility:** `--seed` sets the seed of model initialization, resampling and TensorFlow; run several seeds to report mean and spread, e.g. `python 05_main_federated.py --seed 1`.

### Phase 6: Federated Inference Latency (x86)
* **File:** `06_main_realtime_inference_fed.py`
* **Purpose:** Online federated inference/update latency on the x86 server.

### Phase 7: Hyperparameter Tuning
* **File:** `07_main_hyperparameter_tuning.py`
* **Purpose:** Randomized search for XGBoost and MLP against the default baseline.

### Phase 8: Online Learning Latency on Constrained Hardware
* **File:** `08_main_realtime_inference_fed_rasp_constrained.py`
* **Purpose:** Times one online step (inference + single-sample update) per CPE and one FedAvg round, pinning each worker process to a CPU core. Network cost per round is analytical: two transfers (update upload, model download), each charged one RTT plus the transmission time of the measured update size.
* **Main options:** `--num_simulations` steps timed per job; `--sampling spread` spreads them over the whole stream and reserves a share for positive windows (`--positive_share`); `--mlp_workers` / `--lstm_workers` concurrent workers (1 = no cross-process contention, required for TensorFlow on 1 GB boards); `--rtt_ms`, `--capacity_mbps` network assumptions.
* **Commands used for the paper (Raspberry Pi 3 B+):**
  ```bash
  python 08_main_realtime_inference_fed_rasp_constrained.py --model mlp  --n_sizes 15 60 10 30 --num_simulations 30000 --sampling spread --mlp_workers 1
  python 08_main_realtime_inference_fed_rasp_constrained.py --model lstm --n_sizes 15 60 10 30 --num_simulations 30000 --sampling spread --lstm_workers 1
  ```

### Phase 9: Paper Figures
* **File:** `09_paper_figures.py`
* **Purpose:** Regenerates every manuscript figure (vector PDF + PNG) in `results/paper_figures/` from the saved experiment outputs, without retraining. `--only <name ...>` regenerates a subset.

## 4. Key Findings
* **Links:** 4G delay is about five times higher and far more variable than fiber (median 71 ms vs 13 ms, 95th percentile 175 ms vs 21 ms). On some 4G links loss bursts follow delay surges; on fiber, losses are rare and show little prior delay signature, so they are hard to predict.
* **Model accuracy:** The LSTM reaches the highest PR-AUC in all three data splits (N=30, X=5: 0.69 / 0.53 / 0.54 for S1 / S2 / S3), ahead of the MLP and XGBoost. Accuracy drops as the prediction horizon grows (S3, N=30: PR-AUC from 0.53-0.75 at X=1 s to 0.37-0.41 at X=20 s).
* **Federated Learning:** FedAvg reaches F1-scores close to the average local model; the centralized model is usually the best, clearly so for the LSTM. Non-IID data across CPEs (very different loss rates) can drag the global model down, as for the MLP at N=60. CPE_B is the straggler of every round because its outgoing links carry two samples per second (two interleaved probe streams; the other links carry one), so it holds twice as many samples. These F1 values come from a single run; repeat `05` with several seeds before drawing fine-grained conclusions.
* **Constrained hardware (Raspberry Pi 3 B+, one process per core):** one online step (feature extraction + inference + update) takes about 31 ms for the MLP and 30-78 ms for the LSTM (N = 10-60), well within a 1 s prediction horizon. A federated round costs 32 ms (MLP) and 89-137 ms (LSTM) of compute plus about 330 ms of network time, so communication, not computation, dominates FedAvg on the edge.
* **Measurement note:** single-sample LSTM steps must use the compiled Keras APIs (`predict_on_batch`, `train_on_batch`); eager `model(x)` and per-sample `model.fit()` add 40-70x overhead on the Pi. Running several benchmark processes concurrently also inflates latency (about +50% for the MLP on the Pi), so the reported numbers use one process at a time.

## 5. License
This project is released under the [MIT License](LICENSE).
