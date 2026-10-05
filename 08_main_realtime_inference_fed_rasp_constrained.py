import os
# Thread limits must be set before importing any numerical library.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import datetime
import argparse
import subprocess
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.getcwd())
from src.preprocessor import clean_data
from src.features import engineer_features
from src.federated import payload_bytes, communication_time_s

CPE_FILES = {
    'CPE_A': ['cpe_a-cpe_b-mobile.csv', 'cpe_a-cpe_c-mobile.csv'],
    'CPE_B': ['cpe_b-cpe_a-mobile.csv', 'cpe_b-cpe_c-mobile.csv'],
    'CPE_C': ['cpe_c-cpe_a-mobile.csv', 'cpe_c-cpe_b-mobile.csv'],
}


def pin_process_to_core(core_id):
    """Pins the calling process to a single CPU core (Linux only). Returns True on success."""
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {core_id})
            return True
        except OSError as e:
            print(f"[!] Warning: could not pin process to core {core_id}: {e}")
    return False


def read_soc_temperature_c():
    """SoC temperature in Celsius from the Linux thermal zone, or NaN if unavailable."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return float('nan')


def read_throttled_flags():
    """Raspberry Pi throttling flags from `vcgencmd get_throttled` (e.g. '0x0' = never throttled)."""
    try:
        proc = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=5)
        out = proc.stdout.strip()
        return out.split('=')[-1] if proc.returncode == 0 and out.startswith('throttled=') else 'n/a'
    except (OSError, subprocess.SubprocessError):
        return 'n/a'


def read_cpu_freq_mhz():
    """Current frequency of every core in MHz (reveals thermal throttling when below the nominal maximum)."""
    freqs = []
    for core in range(os.cpu_count() or 1):
        try:
            with open(f'/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq') as f:
                freqs.append(str(int(f.read().strip()) // 1000))
        except (OSError, ValueError):
            freqs.append('n/a')
    return '/'.join(freqs)


def limit_worker_threads():
    """Reinforces single-threaded execution inside a worker process."""
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[var] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_pretraining_set(delays, packet_loss, N, global_max, train_samples):
    """Builds tabular and sequence windows for the warm-up training of a model."""
    from sklearn.preprocessing import StandardScaler

    X_tab, X_seq, y_train = [], [], []
    for j in range(train_samples):
        lookback_d = delays[j : j+N]
        lookback_l = packet_loss[j : j+N]
        X_tab.append(engineer_features(lookback_d, lookback_l, global_max_delay=global_max))
        X_seq.append(np.column_stack((np.nan_to_num(lookback_d, nan=global_max), lookback_l)))
        # Synthetic labels: this script only measures latency, so the warm-up just needs both classes.
        y_train.append(1 if j % 10 == 0 else 0)

    X_tab_df = pd.DataFrame(X_tab)
    X_seq_np = np.array(X_seq)
    y_train_np = np.array(y_train)

    tab_scaler = StandardScaler()
    X_tab_scaled = tab_scaler.fit_transform(X_tab_df)

    seq_scaler = StandardScaler()
    flat_seq = X_seq_np[:, :, 0].reshape(-1, 1)
    seq_scaler.fit(flat_seq)
    X_seq_scaled = np.copy(X_seq_np)
    X_seq_scaled[:, :, 0] = seq_scaler.transform(flat_seq).reshape(X_seq_np.shape[0], X_seq_np.shape[1])

    return {
        'X_tab_scaled': X_tab_scaled, 'X_seq_scaled': X_seq_scaled, 'y': y_train_np,
        'tab_scaler': tab_scaler, 'seq_scaler': seq_scaler, 'feature_cols': X_tab_df.columns,
    }


def build_model(base_model, prep):
    if base_model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=1, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prep['X_tab_scaled'], prep['y'])
        return model

    import tensorflow as tf
    from src.models import train_lstm
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    model = train_lstm(prep['X_seq_scaled'], prep['y'], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
    # Trace the compiled single-sample predict/train functions once, so tracing is not timed as latency.
    x0 = prep['X_seq_scaled'][:1].astype('float32')
    model.predict_on_batch(x0)
    model.train_on_batch(x0, prep['y'][:1])
    return model


def get_model_weights(base_model, model):
    if base_model == 'MLP':
        return {'coefs_': model.coefs_, 'intercepts_': model.intercepts_}
    return model.get_weights()


def window_label(packet_loss, i, N, X_size):
    return 1 if np.sum(packet_loss[i+N : i+N+X_size]) > 0 else 0


def select_eval_indices(cpe_data, N, X_size, start_idx, budget, positive_share, sampling, seed=42):
    """
    Picks the online steps to time. 'spread' covers the whole stream (evenly strided) and reserves a share
    of the budget for positive windows of any CPE, so both classes and all traffic regimes are exercised.
    'head' reproduces the plain first-`budget`-steps behaviour. The same indices are used by every job.
    """
    max_idx = min(len(pl) for _, pl in cpe_data.values()) - N - X_size + 1
    if sampling == 'head':
        return np.arange(start_idx, min(start_idx + budget, max_idx))

    rng = np.random.default_rng(seed)
    candidates = np.arange(start_idx, max_idx)
    is_positive = np.zeros(len(candidates), dtype=bool)
    for _, pl in cpe_data.values():
        cs = np.concatenate([[0], np.cumsum(pl > 0)])
        is_positive |= (cs[candidates + N + X_size] - cs[candidates + N]) > 0
    positives = candidates[is_positive]
    n_pos = min(len(positives), int(budget * positive_share))
    chosen_pos = rng.choice(positives, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
    strided = np.linspace(start_idx, max_idx - 1, budget - n_pos).astype(int)
    return np.unique(np.concatenate([strided, chosen_pos]).astype(int))


def online_step(base_model, model, prep, delays, packet_loss, i, N, X_size, global_max):
    """
    Runs one online step as a CPE would on each new sample: feature extraction from the lookback window,
    inference and single-sample update. Returns (features_s, inference_s, update_s, label, loss_in_lookback).
    """
    label = window_label(packet_loss, i, N, X_size)
    y_curr = np.array([label])

    if base_model == 'MLP':
        tf0 = time.perf_counter()
        lookback_delays = delays[i : i+N]
        lookback_losses = packet_loss[i : i+N]
        feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
        x = prep['tab_scaler'].transform(pd.DataFrame([feats])[prep['feature_cols']])

        t0 = time.perf_counter()
        _ = model.predict_proba(x)
        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.partial_fit(x, y_curr, classes=np.array([0, 1]))
        t2 = time.perf_counter()
    else:
        tf0 = time.perf_counter()
        lookback_delays = delays[i : i+N]
        lookback_losses = packet_loss[i : i+N]
        seq_d = prep['seq_scaler'].transform(np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1))
        x = np.column_stack((seq_d, lookback_losses)).reshape(1, N, 2).astype('float32')

        # Compiled single-batch APIs: eager model(x) and per-call model.fit() add 40-70x Keras overhead on batch=1.
        t0 = time.perf_counter()
        _ = model.predict_on_batch(x)
        t1 = time.perf_counter()
        model.train_on_batch(x, y_curr)
        t2 = time.perf_counter()

    return t0 - tf0, t1 - t0, t2 - t1, label, int(np.sum(lookback_losses) > 0)


def worker_local_simulation(cpe_name, base_model, N, X_size, delays, packet_loss, global_max,
                            train_samples, indices, core_id):
    """Simulates a single CPE learning purely locally on its own data stream (no server, no network)."""
    limit_worker_threads()
    pinned = pin_process_to_core(core_id)
    print(f"      [Worker] {base_model} Local {cpe_name} (N={N}) on core {core_id}"
          f"{'' if pinned else ' (NOT pinned)'} [PID {os.getpid()}]")

    prep = build_pretraining_set(delays, packet_loss, N, global_max, train_samples)
    model = build_model(base_model, prep)

    results = []
    for i in indices:
        feat_s, inf_s, upd_s, label, loss_lb = online_step(base_model, model, prep, delays, packet_loss, i, N, X_size, global_max)
        step_s = feat_s + inf_s + upd_s
        results.append({
            'Model': base_model, 'Scope': 'Local', 'CPE': cpe_name, 'N': N, 'Core': core_id, 'step_idx': int(i),
            'label': label, 'loss_in_lookback': loss_lb,
            'features_ms': feat_s * 1000.0, 'inference_ms': inf_s * 1000.0, 'update_ms': upd_s * 1000.0,
            'slowest_cpe_ms': np.nan, 'compute_ms': step_s * 1000.0,
            'server_ms': 0.0, 'payload_bytes': 0, 'network_ms': 0.0,
            'total_ms': step_s * 1000.0,
        })

    print(f"      [Worker] {base_model} Local {cpe_name} (N={N}) completed: {len(indices)} steps.")
    return results


def worker_federated_simulation(base_model, N, X_size, cpe_data, global_max, train_samples,
                                indices, core_id, rtt_ms, capacity_mbps):
    """
    Simulates one online FedAvg round per sample. The 3 CPEs are executed one after the other on
    the same core and the round compute time takes the slowest CPE, as they would run in parallel
    on separate devices. Network cost is analytical (RTT + payload transmission), not measured.
    """
    limit_worker_threads()
    pinned = pin_process_to_core(core_id)
    print(f"      [Worker] {base_model} Federated (N={N}) on core {core_id}"
          f"{'' if pinned else ' (NOT pinned)'} [PID {os.getpid()}]")

    cpe_names = list(cpe_data.keys())
    preps = {c: build_pretraining_set(*cpe_data[c], N, global_max, train_samples) for c in cpe_names}
    models = {c: build_model(base_model, preps[c]) for c in cpe_names}

    update_bytes = payload_bytes(get_model_weights(base_model, models[cpe_names[0]]))
    network_s = communication_time_s(update_bytes, rtt_ms, capacity_mbps)
    print(f"      [Worker] {base_model} model update size: {update_bytes} bytes "
          f"-> network cost {network_s * 1000:.2f} ms/round (RTT={rtt_ms} ms, C={capacity_mbps} Mbps)")

    results = []
    for processed, i in enumerate(indices, start=1):
        cpe_times, labels, losses_lb = [], [], []
        for c in cpe_names:
            feat_s, inf_s, upd_s, label, loss_lb = online_step(base_model, models[c], preps[c], *cpe_data[c], i, N, X_size, global_max)
            cpe_times.append(feat_s + inf_s + upd_s)
            labels.append(label)
            losses_lb.append(loss_lb)

        t0 = time.perf_counter()
        if base_model == 'MLP':
            n_layers = len(models[cpe_names[0]].coefs_)
            avg_coefs = [sum(models[c].coefs_[k] for c in cpe_names) / len(cpe_names) for k in range(n_layers)]
            avg_intercepts = [sum(models[c].intercepts_[k] for c in cpe_names) / len(cpe_names) for k in range(n_layers)]
            for c in cpe_names:
                models[c].coefs_ = [np.copy(w) for w in avg_coefs]
                models[c].intercepts_ = [np.copy(b) for b in avg_intercepts]
        else:
            all_w = [models[c].get_weights() for c in cpe_names]
            new_w = [sum(w[k] for w in all_w) / len(cpe_names) for k in range(len(all_w[0]))]
            for c in cpe_names:
                models[c].set_weights(new_w)
        server_s = time.perf_counter() - t0

        compute_s = max(cpe_times) + server_s
        results.append({
            'Model': base_model, 'Scope': 'Federated', 'CPE': 'ALL', 'N': N, 'Core': core_id, 'step_idx': int(i),
            'label': max(labels), 'loss_in_lookback': max(losses_lb),
            'features_ms': np.nan, 'inference_ms': np.nan, 'update_ms': np.nan, 'slowest_cpe_ms': max(cpe_times) * 1000.0,
            'compute_ms': compute_s * 1000.0,
            'server_ms': server_s * 1000.0, 'payload_bytes': update_bytes, 'network_ms': network_s * 1000.0,
            'total_ms': (compute_s + network_s) * 1000.0,
        })

        if processed % 1000 == 0:
            print(f"      [Worker] {base_model} Federated (N={N}): {processed}/{len(indices)} rounds...")

    print(f"      [Worker] {base_model} Federated (N={N}) completed: {len(indices)} rounds.")
    return results


def load_cpe_data(dir_path):
    cpe_data, global_max = {}, 0.0
    for cpe_name, filenames in CPE_FILES.items():
        dfs = [pd.read_csv(os.path.join(dir_path, f)) for f in filenames if os.path.exists(os.path.join(dir_path, f))]
        if not dfs:
            continue
        df_clean = clean_data(pd.concat(dfs, ignore_index=True))
        m = df_clean['delay_ms'].max()
        if pd.notna(m) and m > global_max:
            global_max = m
        cpe_data[cpe_name] = (df_clean['delay_ms'].values, df_clean['packet_loss'].values)
    return cpe_data, (global_max if global_max > 0 else 1000.0)


def run_constrained_simulation(dir_path, n_sizes, X, num_simulations, models_to_run,
                               rtt_ms, capacity_mbps, output_dir, lstm_workers=4,
                               sampling='spread', positive_share=0.2, mlp_workers=4):
    print(f"[*] Loading files for live simulation from: {dir_path}")
    cpe_data, global_max = load_cpe_data(dir_path)
    if not cpe_data:
        print("[!] No CPE data found.")
        return

    train_samples = 200
    all_results, run_log = [], []

    # One process per core, all four running concurrently: CPE_A/B/C on cores 0-2, the federated round on core 3.
    cpe_cores = {c: i for i, c in enumerate(sorted(cpe_data))}
    federated_core = len(cpe_cores)
    print(f"[*] Core map: {cpe_cores} | Federated -> core {federated_core}")
    print(f"[*] Models: {models_to_run} | N: {n_sizes} | X: {X} | steps per job: {num_simulations} ({sampling})")

    for N in n_sizes:
        indices = select_eval_indices(cpe_data, N, X, train_samples + 100, num_simulations, positive_share, sampling)
        n_pos = sum(any(window_label(pl, i, N, X) for _, pl in cpe_data.values()) for i in indices)
        print(f"[*] N={N}: timing {len(indices)} steps spanning indices {indices[0]}-{indices[-1]}, "
              f"{n_pos} with a loss in the next X on at least one CPE.")
        for base_model in models_to_run:
            # Each TensorFlow process needs a few hundred MB, so on small boards LSTM may need fewer concurrent workers.
            n_workers = min(mlp_workers if base_model == 'MLP' else lstm_workers, len(cpe_cores) + 1)
            print(f"\n>>> [{base_model}] N={N}: {n_workers} concurrent workers...")
            temp_start, t_start = read_soc_temperature_c(), time.perf_counter()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {f"Local {c}": executor.submit(
                               worker_local_simulation, c, base_model, N, X, *cpe_data[c], global_max,
                               train_samples, indices, core)
                           for c, core in cpe_cores.items()}
                futures["Federated"] = executor.submit(
                    worker_federated_simulation, base_model, N, X, cpe_data, global_max, train_samples,
                    indices, federated_core, rtt_ms, capacity_mbps)

                for job_name, future in futures.items():
                    try:
                        all_results.extend(future.result())
                        status = 'ok'
                    except Exception as e:
                        print(f"  [!] Worker failed ({base_model} {job_name}, N={N}): {e}")
                        status = f'error: {e}'
                    run_log.append({'Model': base_model, 'Job': job_name, 'N': N, 'status': status})

            batch_conditions = {
                'wall_time_s': time.perf_counter() - t_start,
                'temp_start_c': temp_start, 'temp_end_c': read_soc_temperature_c(),
                'throttled_flags': read_throttled_flags(), 'cpu_freq_mhz_end': read_cpu_freq_mhz(),
                'concurrent_workers': n_workers,
            }
            for row in run_log[-len(futures):]:
                row.update(batch_conditions)

            # Saved after every batch so partial results survive an interrupted run.
            pd.DataFrame(all_results).to_csv(os.path.join(output_dir, 'realtime_inference_results.csv'), index=False)

    pd.DataFrame(run_log).to_csv(os.path.join(output_dir, 'run_conditions.csv'), index=False)

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("[!] No results collected.")
        return

    results_df.to_csv(os.path.join(output_dir, 'realtime_inference_results.csv'), index=False)

    summary = results_df.groupby(['Model', 'Scope', 'CPE', 'N']).agg(
        compute_median_ms=('compute_ms', 'median'),
        compute_p99_ms=('compute_ms', lambda x: np.percentile(x, 99)),
        network_ms=('network_ms', 'mean'),
        payload_bytes=('payload_bytes', 'max'),
        total_median_ms=('total_ms', 'median'),
        total_p99_ms=('total_ms', lambda x: np.percentile(x, 99)),
    )
    summary.to_csv(os.path.join(output_dir, 'realtime_inference_summary.csv'))
    print(f"\n[*] Results saved in: {output_dir}")
    print(summary)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Constrained-hardware (e.g. Raspberry Pi) online learning latency benchmark.")
    parser.add_argument('--dataset_dir', type=str, default="dataset/second_capture_window", help="Dataset directory.")
    parser.add_argument('--n_sizes', nargs='+', type=int, default=[10, 15, 30, 60], help="Lookback window sizes N.")
    parser.add_argument('--x_size', type=int, default=1, help="Prediction horizon X.")
    parser.add_argument('--num_simulations', type=int, default=5000, help="Online steps (packets/rounds) timed per job.")
    parser.add_argument('--sampling', type=str, choices=['spread', 'head'], default='spread',
                        help="'spread': steps strided over the whole stream plus a share of positive windows; 'head': first steps only.")
    parser.add_argument('--positive_share', type=float, default=0.2, help="Share of the step budget reserved for positive windows ('spread' only).")
    parser.add_argument('--model', type=str, choices=['all', 'mlp', 'lstm'], default='all', help="Models to run.")
    parser.add_argument('--rtt_ms', type=float, default=150.0, help="Assumed client-server RTT in ms.")
    parser.add_argument('--capacity_mbps', type=float, default=10.0, help="Assumed link capacity in Mbps.")
    parser.add_argument('--lstm_workers', type=int, default=4, help="Max concurrent LSTM workers (lower it on low-RAM boards).")
    parser.add_argument('--mlp_workers', type=int, default=4, help="Max concurrent MLP workers (1 = no cross-process contention).")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory (optional).")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("results", f"exp_realtime_fed_rasp_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.dataset_dir):
        print(f"[!] Error: dataset folder '{args.dataset_dir}' not found.")
        sys.exit(1)

    run_constrained_simulation(
        dir_path=args.dataset_dir,
        n_sizes=args.n_sizes,
        X=args.x_size,
        num_simulations=args.num_simulations,
        models_to_run=['MLP', 'LSTM'] if args.model == 'all' else [args.model.upper()],
        rtt_ms=args.rtt_ms,
        capacity_mbps=args.capacity_mbps,
        output_dir=out_dir,
        lstm_workers=args.lstm_workers,
        mlp_workers=args.mlp_workers,
        sampling=args.sampling,
        positive_share=args.positive_share,
    )
