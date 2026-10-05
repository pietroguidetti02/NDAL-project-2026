# Paper Notes: Assumptions, Methodology and Reviewer Risks

Working notes for the IEEE ICC manuscript. Every assumption the paper relies on, how to justify it,
and the points a reviewer is likely to challenge. Items marked **TODO** still need a decision or a source.

---

## 1. Dataset facts to state in the paper

| Fact | Value | Source in repo |
|---|---|---|
| Testbed | 3 CPEs (A, B, C) in Milan, full mesh, fiber and 4G (mobile) tunnels | `dataset/` |
| Capture windows | 1st: 5-6 Mar 2025, ~16 h; 2nd: 6-7 Mar 2025, ~32.6 h | CSV timestamps |
| Measurement | One `delay_ms` sample per probe; `-1` marks a lost probe | `src/preprocessor.py` |
| Sampling rate | **1 Hz on all links except those sourced at CPE_B (B->A, B->C), which carry two samples per second (two interleaved 1 Hz streams, gaps alternating 0.2 s / 0.8 s)** | CSV timestamps (see §4, risk R1) |
| Label | A window is positive if at least one probe is lost in the next X samples | `src/preprocessor.py` |
| Class imbalance | 0.2-0.4% positive windows per CPE (mobile, N=15, X=1) | computed from the dataset |
| Delay statistics | Fiber median 13.3 ms (p95 20.6 ms); 4G median 71.2 ms (p95 175.1 ms) | all links, both windows |
| Scope of ML experiments | Sweeps and federated experiments use the **mobile (4G) tunnel only** | `config/sweep*.yaml` |

**TODO:** confirm with the testbed owners whether `delay_ms` is one-way delay or RTT. The paper currently
must say only "delay", never "one-way delay" or "RTT".

## 2. Data splits

| Name | Train | Test | What it measures |
|---|---|---|---|
| S1 - single dataset | 2nd window, 80% (interleaved contiguous blocks) | 2nd window, 20% | In-distribution performance |
| S2 - spatial split | Links A->B, A->C, B->A (both windows) | Links B->C, C->A, C->B (both windows) | Generalization to unseen links |
| S3 - temporal split | 2nd window (100%, the longer one: 32.6 h) | 1st window (100%, 16 h) | Generalization across capture sessions |
| Federated (`05`) | Same as S3; clients = links grouped by source CPE | 1st window | FL vs local vs centralized |

Blocks are contiguous, so no sliding window crosses a train/test boundary.

## 3. Assumptions and how to justify them

### 3.1 Raspberry Pi 3 B+ as worst-case router-class hardware
- **Assumption:** online inference and learning are benchmarked on a Raspberry Pi 3 B+
  (4x ARM Cortex-A53 @ 1.4 GHz, 1 GB RAM), not on a commercial CPE.
- **Justification:** commercial branch SD-WAN CPEs use 2-8 low-power cores (Intel Atom or ARMv8) with
  2-4 GB RAM, e.g. VeloCloud Edge 510 (Atom C2358, 2 cores @ 1.7 GHz, 4 GB), VeloCloud Edge 600
  (Atom C3000), FortiGate 60F (8-core ARMv8, ~2 GB), Juniper SRX300 (4 GB). The Pi 3 B+ has older cores
  and less memory than all of them, so its latencies are a **pessimistic bound**: if the pipeline meets
  the deadline on the Pi, it meets it on a commercial branch CPE.
- **Sentence for the paper:** "We deliberately benchmark on a platform weaker than commercial branch CPEs,
  so that the reported latencies are an upper bound."
- **Sources:**
  [VeloCloud Edge 510](https://wikidevi.wi-cat.ru/VeloCloud_EDGE_510-AC),
  [Dell/VeloCloud Edge 600 FAQ](https://www.dell.com/support/kbdoc/en-us/000132845/dell-emc-sd-wan-edge-frequently-asked-questions),
  [FortiGate CPU/RAM table](https://yurisk.info/assets/2021-03-14-fortigate-hardware-cpu-memory-ram-per-model-table.pdf),
  [Juniper SRX300 specs](https://apps.juniper.net/hct/product/SRX300/hwspecs),
  [Cisco Catalyst 8200 uCPE](https://www.cisco.com/c/en/us/products/collateral/routers/catalyst-8200-series-edge-ucpe/nb-06-cat8200-series-edge-ucpe-ds-cte-en.html).

### 3.2 One core, one process, no GPU
- **Assumption:** each benchmark process is pinned to a single core (`sched_setaffinity`), numerical
  libraries and TensorFlow are limited to one thread, the GPU is disabled, and only one benchmark process
  runs at a time.
- **Justification:** on real CPEs packet forwarding is offloaded (ASIC/DPDK) and an ML agent would share
  the control-plane cores with other services, so one core is a realistic budget. Running processes
  concurrently was measured to inflate latency by ~50% (shared L2 cache and memory bus of the Cortex-A53),
  which cgroups cannot partition; TensorFlow processes also do not fit in 1 GB RAM concurrently.
- **Stated limitation:** the OS and remote-access daemons still share the board (load average ~0 during
  runs); CPU frequency (1.4 GHz) and SoC temperature (63-69 °C) were logged per batch and no thermal
  throttling occurred.

### 3.3 What one "online step" includes
- Feature extraction from the lookback window + scaling + inference + single-sample model update.
- Feature extraction is timed because in deployment features are computed on each new sample.
- Our feature code builds a pandas DataFrame per sample; a NumPy-only implementation would be faster, so
  the MLP numbers are an **upper bound**.
- The LSTM is invoked through Keras' compiled single-batch APIs (`predict_on_batch`, `train_on_batch`).
  Eager `model(x)` and per-sample `model.fit()` add 40-70x overhead on the Pi and would measure Keras
  overhead instead of the model.
- The warm-up training inside the benchmark uses synthetic labels: the benchmark measures latency only,
  never accuracy.

### 3.4 Which samples are timed
- Each job times ~30,000 online steps spread uniformly over the whole stream, plus all positive windows
  (~2,100 per N). Every CPE runs its own local job; the federated job runs the same steps for all CPEs.
- Latency does not depend on the class: medians of positive and negative steps differ by < 0.5%, and
  timing the first 5,000 steps or 5,000 spread steps gives the same percentiles (< 0.3 ms difference).

### 3.5 Federated round on constrained hardware
- The three CPEs are executed one after the other on the same core; the **round compute time is the
  slowest CPE plus the FedAvg aggregation**, emulating three devices working in parallel.
- Aggregation is per sample (one round per new sample), which is the most communication-intensive case.

### 3.6 Network cost model (analytical, not measured)
No measurement of the FedAvg traffic over the testbed links was available, so the network cost of a
round is modelled as

    T_net = 2 * (RTT + 8 * S / C)

| Symbol | Meaning | Value |
|---|---|---|
| S | Size of one model update (raw tensor bytes, measured by `payload_bytes()`) | MLP 16,008 B; LSTM 20,100 B |
| C | Link capacity | 10 Mbps (**TODO: cite a source**) |
| RTT | Round-trip time charged to each transfer | 150 ms |
| x 2 | Two transfers per round: client update upload + aggregated model download | |

Each transfer is charged a full RTT because it requires an acknowledgement, which is conservative.
With the values above: MLP 326 ms/round, LSTM 332 ms/round.

**Sensitivity (report it, it pre-empts the reviewer):**

| C | MLP | LSTM |
|---|---|---|
| 1 Mbps | 556 ms | 622 ms |
| 10 Mbps | 326 ms | 332 ms |
| 50 Mbps | 305 ms | 306 ms |

The RTT term dominates because updates are small; conclusions do not change over this range.
Protocol headers (TCP/IP) are not counted: ~11 MTU-sized packets per update, ~3% extra bytes.

**Capacity justification (C = 10 Mbps, uplink is the bottleneck):** real-user average mobile upload speeds
in Italy are 7.2-11.2 Mbps depending on the operator (Opensignal, Italy Mobile Network Experience Report,
Dec. 2025), while the regulator's drive tests report ~54-58 Mbps average upload in urban areas under
test conditions (AGCOM / Fondazione Ugo Bordoni, Misura Internet Mobile 2025). 10 Mbps therefore matches
typical user experience and is conservative w.r.t. the regulator's measurements. **TODO:** verify the
figures on the original pages before citing, and prefer the value of the operator of the testbed SIMs.
[Opensignal Italy Dec. 2025](https://insights.opensignal.com/reports/2025/12/italy/mobile-network-experience),
[AGCOM press release, Misura Internet Mobile](https://www.agcom.it/comunicazione/comunicati-stampa/comunicato-stampa-78),
[AGCOM network quality projects](https://www.agcom.it/competenze/consumatori/interventi-regolamentari-sulla-qualita-dei-servizi-attuazione-del-nuovo-0/progetti-qualit%C3%A0-reti).

**RTT justification:** 150 ms is conservative with respect to the 4G delays in our dataset
(median 71 ms, p95 175 ms). **TODO:** once the meaning of `delay_ms` is confirmed (§1), phrase this as
"about the median RTT" (if one-way) or "about the 85th-90th percentile" (if RTT).

### 3.7 Federated learning algorithm
- Synchronous FedAvg, 3 clients, 5 rounds, 3 local epochs per round.
- **Aggregation is an unweighted mean of client weights.** Standard FedAvg (McMahan et al., 2017) weights
  clients by their number of samples. State it explicitly, or switch to weighted averaging and rerun `05`.
- Training-time figures come from the x86 server: clients train in parallel threads and the round time is
  the slowest client plus a fixed 0.3 s network penalty (2 x 150 ms).

### 3.8 Evaluation metrics
- PR-AUC is computed as the trapezoidal area under the PR curve (`sklearn.metrics.auc(recall, precision)`).
  Average precision (`average_precision_score`) is the more common choice in the literature; state which
  one is used.
- Decision thresholds: for XGBoost and the MLP, the F1-optimal threshold is chosen **out-of-fold on the
  training set** (`optimize_threshold_cv`, 3-fold CV), never on the test set. The LSTM uses a fixed 0.5
  threshold, which disadvantages it in F1; PR-AUC is threshold-free and unaffected.
- The ROC curve is not reported as the main metric: with <0.5% positives, ROC-AUC (~0.96) overstates
  performance compared with PR-AUC.

## 4. Points a reviewer may not accept

| # | Issue | Risk | What to do / write |
|---|---|---|---|
| R1 | **CPE_B links carry two samples per second, the others one**, but N and X are counted in samples. For CPE_B, "N = 30, X = 5" means 15 s and 2.5 s. The inter-sample gaps alternate 0.2 s / 0.8 s and even/odd samples have slightly different medians (74.4 vs 76.8 ms), which points to **two interleaved 1 Hz probe streams** (e.g. a duplicated probe instance on CPE_B) rather than a uniform 2 Hz probe. | **High** | Fix implemented: `one_sample_per_second: true` (configs `config/*_1hz.yaml`) keeps one stream, so all links are 1 Hz; accuracy experiments are rerun on the x86 server with these configs. Still ask the testbed owners why CPE_B runs two probe streams. The latency benchmark (`08`) is unaffected: per-step cost is identical across CPEs. The straggler effect of CPE_B comes from having twice the samples, not from more traffic; the old STATUS wording must not appear in the paper. |
| R2 | S3 trains on the 2nd capture window and tests on the 1st (reverse chronology). | Medium | Justification: the 2nd window is twice as long (32.6 h vs 16 h), so training on it maximises the training data, and the two sessions are treated as exchangeable (cross-session validation, ~1 day apart). Optionally also report train-1st / test-2nd. |
| R3 | S1 interleaves train/test blocks from the same period, so neighbouring blocks are correlated. | Medium | Present S1 as the in-distribution upper bound; base claims on S2/S3. |
| R4 | Accuracy results come from a single run. | Medium | Run `05` (and ideally the sweeps) with several `--seed` values and report mean +/- std, or list it as a limitation. The MLP is visibly unstable (e.g. S3, N = 60, X = 1: PR-AUC 0.12). |
| R5 | Network cost is modelled, not measured. | Medium | Present the formula, parameters and sensitivity table (§3.6); cite a source for C. |
| R6 | Unweighted FedAvg. | Low-Medium | State it (§3.7) or switch to sample-weighted averaging. |
| R7 | `delay_ms` semantics (one-way vs RTT) not documented. | Medium | Confirm with the testbed owners before writing §1 and §3.6. |
| R8 | Benchmark on a Raspberry Pi, not on a real CPE. | Low | Worst-case argument with the hardware table (§3.1). |
| R9 | Per-sample federated aggregation is unrealistic. | Low | Present it as the communication-heaviest case; periodic aggregation every k samples divides the network cost by k. |
| R10 | x86 LSTM latencies from `04` / `06` use eager `model(x)` and per-sample `fit()`. | **Do not use** | Only report latencies from `08`. |
| R11 | Claim "federated generalizes better to unseen routes". | **Unsupported** | No leave-one-link-out experiment was run. Do not claim it. |
| R12 | Only the mobile tunnel is modelled; fiber losses are rare and show no delay precursor. | Low | State the scope explicitly and motivate it with the link characterization figure. |
| R13 | Feature extraction uses pandas per sample. | Low | Report MLP latency as an upper bound (§3.3). |

## 5. Numbers to report

Final Raspberry Pi latencies from `results/rasp/rasp_final_30000/` (28-29 Sep 2026; ~29,750 steps per
job, feature extraction included; 32/32 jobs OK; SoC 60-63 °C; all cores at 1.4 GHz throughout).
Positive vs negative windows: median latency differs by at most 0.26%.

| Model, N | Local step: features / inference / update | Local step p50 (p95) | FL round compute p50 (p95) | Network | FL round total |
|---|---|---|---|---|---|
| MLP, any N | 15.2 / 1.4 / 14.5 ms | 31.1 ms (31.8) | 31.8 ms (32.5-33.1) | 326 ms | ~358 ms |
| LSTM, N = 10 | 1.8 / 6.4 / 21.6 ms | 29.9 ms (30.5) | 88.6 ms (90.1) | 332 ms | ~421 ms |
| LSTM, N = 15 | 1.8 / 7.5 / 25.2 ms | 34.6 ms (35.5) | 92.7 ms (94.6) | 332 ms | ~425 ms |
| LSTM, N = 30 | 1.8 / 10.8 / 36.6 ms | 49.2 ms (50.2) | 107.7 ms (109.7) | 332 ms | ~440 ms |
| LSTM, N = 60 | 1.8 / 17.1 / 59.2 ms | 78.1 ms (79.4) | 136.9 ms (138.4) | 332 ms | ~469 ms |

MLP latency is independent of N (fixed-size feature vector) and about half of it is feature extraction;
LSTM latency grows linearly with N (sequence length).

Deadline to compare against: one new probe per second (X = 1 s) per stream; CPE_B ingests two streams,
i.e. one sample every 0.5 s on average.

## 6. Open decisions

- [ ] Verify and cite the C = 10 Mbps sources (§3.6); ideally the operator of the testbed SIMs
- [ ] Meaning of `delay_ms` (§1, R7)
- [ ] Why CPE_B runs two probe streams, and whether to keep only one (R1)
- [ ] Seeds for the accuracy results (R4)
- [ ] Weighted vs unweighted FedAvg (R6)
- [ ] Exact CPU model of the x86 server (for the setup section)
