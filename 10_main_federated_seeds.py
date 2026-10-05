"""
Federated vs local vs centralized learning (MLP and LSTM), repeated over many seeds and an (N, X) grid.
All results go to one tidy CSV (one row per seed, model, N, X, scope and client), from which the paper
figures report mean and confidence intervals. Window extraction and scaling are done once per (N, X)
and shared by all seeds.

Run seeds in parallel processes with --seeds and --output part files, then join them with --merge.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (precision_recall_curve, auc, roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler

sys.path.append(os.getcwd())
from src.data_loader import load_config, load_and_split_data
from src.federated import FLServer, payload_bytes, communication_time_s

# Loaded via importlib because the filename starts with a digit and is not a valid module identifier.
_spec = importlib.util.spec_from_file_location("main_comp", "02_b_main_comparison_LSTM.py")
_main_comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_comp)
process_dataset_all = _main_comp.process_dataset_all

CLIENTS = ['CPE_A', 'CPE_B', 'CPE_C']
# Same feature set as the sweeps on the mobile tunnel (14 features).
DROPPED_FEATURES = ['recent_jitter', 'recent_slope', 'ratio_recent_mean_to_global', 'spikes_over_q95']
VAL_BLOCKS, VAL_EVERY = 20, 10   # every 10th of 20 contiguous blocks of each client is held out for validation


# ----------------------------------------------------------------------------- data

def block_validation_mask(n):
    """Contiguous blocks (not random rows), so overlapping sliding windows rarely straddle train and validation."""
    blocks = np.arange(n) * VAL_BLOCKS // max(n, 1)
    return (blocks % VAL_EVERY) == VAL_EVERY - 1


def prepare_grid_point(train_dfs, test_dfs, N, X):
    """Extracts, splits and scales the windows of every client for one (N, X). Shared by all seeds."""
    # Train windows are grouped by source CPE: the config lists A->B, A->C, B->A, B->C, C->A, C->B in order.
    groups = np.array_split(np.arange(len(train_dfs)), len(CLIENTS))
    raw = {}
    for cid, idx in zip(CLIENTS, groups):
        X_df, X_seq, y = process_dataset_all([train_dfs[i] for i in idx], N, X)
        X_df = X_df.drop(columns=[c for c in DROPPED_FEATURES if c in X_df.columns])
        val = block_validation_mask(len(y))
        raw[cid] = {'tab': X_df.values, 'seq': np.asarray(X_seq, dtype='float32'), 'y': np.asarray(y), 'val': val}
    X_test_df, X_test_seq, y_test = process_dataset_all(test_dfs, N, X)
    X_test_df = X_test_df.drop(columns=[c for c in DROPPED_FEATURES if c in X_test_df.columns])

    tab_scaler = StandardScaler().fit(np.concatenate([r['tab'][~r['val']] for r in raw.values()]))
    seq_scaler = StandardScaler().fit(np.concatenate([r['seq'][~r['val']][:, :, 0].reshape(-1, 1) for r in raw.values()]))

    def scale_seq(s):
        s = np.copy(s)
        s[:, :, 0] = seq_scaler.transform(s[:, :, 0].reshape(-1, 1)).reshape(s.shape[0], s.shape[1])
        return s

    clients = {}
    for cid, r in raw.items():
        tr, va = ~r['val'], r['val']
        clients[cid] = {
            'tab': tab_scaler.transform(r['tab'][tr]), 'seq': scale_seq(r['seq'][tr]), 'y': r['y'][tr],
            'tab_val': tab_scaler.transform(r['tab'][va]), 'seq_val': scale_seq(r['seq'][va]), 'y_val': r['y'][va],
        }
    test = {'tab': tab_scaler.transform(X_test_df.values), 'seq': scale_seq(np.asarray(X_test_seq, dtype='float32')),
            'y': np.asarray(y_test)}
    val = {k: np.concatenate([c[f'{k}_val'] for c in clients.values()]) for k in ('tab', 'seq', 'y')}
    return clients, val, test


def rebalance(X, y, seed):
    """SMOTE when there are enough positives, plain oversampling otherwise, nothing without positives."""
    n_pos = int(np.sum(y == 1))
    if n_pos > 5:
        return SMOTE(random_state=seed).fit_resample(X, y)
    if n_pos > 0:
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    return X, y


def class_weights(y):
    classes = np.unique(y)
    if len(classes) < 2:
        return {0: 1.0, 1: 1.0}
    return dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y)))


# ----------------------------------------------------------------------------- models

def build_mlp(seed):
    # Same architecture as the sweeps (sklearn default: one hidden layer of 100 units).
    return MLPClassifier(hidden_layer_sizes=(100,), batch_size=256, random_state=seed)


def build_lstm(N, positive_rate):
    """Same architecture as src/models.train_lstm (and the Raspberry Pi benchmark), without fitting."""
    import tensorflow as tf
    bias = tf.keras.initializers.Constant(np.log(positive_rate / (1 - positive_rate))) if 0 < positive_rate < 1 else 'zeros'
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(N, 2)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    return model


def predict(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X, batch_size=4096, verbose=0).ravel()


def mlp_epochs(model, X, y, epochs):
    for _ in range(epochs):
        model.partial_fit(X, y, classes=np.array([0, 1]))


def get_weights(model):
    if isinstance(model, MLPClassifier):
        return {'coefs_': [np.copy(w) for w in model.coefs_], 'intercepts_': [np.copy(b) for b in model.intercepts_]}
    return model.get_weights()


def set_weights(model, weights):
    if isinstance(model, MLPClassifier):
        model.coefs_ = [np.copy(w) for w in weights['coefs_']]
        model.intercepts_ = [np.copy(b) for b in weights['intercepts_']]
    else:
        model.set_weights(weights)


# ----------------------------------------------------------------------------- metrics

def evaluate(p_test, y_test, p_val, y_val):
    """Threshold-free metrics on test, plus thresholded metrics at the F1-optimal threshold chosen on validation."""
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    out = {'pr_auc': auc(recall, precision),
           'roc_auc': roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
           'f1_at_0.5': f1_score(y_test, p_test >= 0.5, zero_division=0)}
    threshold = 0.5
    if np.sum(y_val) > 0:
        pv, rv, tv = precision_recall_curve(y_val, p_val)
        f1v = 2 * pv[:-1] * rv[:-1] / np.clip(pv[:-1] + rv[:-1], 1e-12, None)
        threshold = float(tv[np.argmax(f1v)])
    pred = p_test >= threshold
    out.update({'threshold': threshold, 'f1': f1_score(y_test, pred, zero_division=0),
                'precision': precision_score(y_test, pred, zero_division=0),
                'recall': recall_score(y_test, pred, zero_division=0),
                'accuracy': accuracy_score(y_test, pred)})
    return out


# ----------------------------------------------------------------------------- one seed

def run_seed(seed, N, X, clients, val, test, args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    rows = []

    def record(model_name, scope, cpe, model, train_s, extra=None):
        mdl_X = 'tab' if model_name == 'MLP' else 'seq'
        row = {'seed': seed, 'model': model_name, 'N': N, 'X': X, 'scope': scope, 'cpe': cpe,
               'train_time_s': train_s, 'n_test': len(test['y']), 'n_test_pos': int(test['y'].sum())}
        row.update(evaluate(predict(model, test[mdl_X]), test['y'], predict(model, val[mdl_X]), val['y']))
        row.update(extra or {})
        rows.append(row)

    all_tab = np.concatenate([c['tab'] for c in clients.values()])
    all_seq = np.concatenate([c['seq'] for c in clients.values()])
    all_y = np.concatenate([c['y'] for c in clients.values()])
    pos_rate = float(all_y.mean())

    # ---- MLP (SMOTE-rebalanced tabular features) ----
    E, R = args.local_epochs, args.mlp_rounds
    Xc, yc = rebalance(all_tab, all_y, seed)
    m = build_mlp(seed); t0 = time.perf_counter(); mlp_epochs(m, Xc, yc, R * E)
    record('MLP', 'Centralized', 'ALL', m, time.perf_counter() - t0)
    local_bal = {cid: rebalance(c['tab'], c['y'], seed) for cid, c in clients.items()}
    for cid, (Xl, yl) in local_bal.items():
        m = build_mlp(seed); t0 = time.perf_counter(); mlp_epochs(m, Xl, yl, R * E)
        record('MLP', 'Local', cid, m, time.perf_counter() - t0)
    glob_m = build_mlp(seed); mlp_epochs(glob_m, Xc[:2], yc[:2], 1)
    fed_models = {cid: build_mlp(seed) for cid in clients}
    for cid, mm in fed_models.items():
        mlp_epochs(mm, Xc[:2], yc[:2], 1)
    update = payload_bytes(get_weights(glob_m))
    net_s = communication_time_s(update, args.rtt_ms, args.capacity_mbps)
    fed_time = 0.0
    for _ in range(R):
        times = []
        for cid, mm in fed_models.items():
            set_weights(mm, get_weights(glob_m))
            t0 = time.perf_counter(); mlp_epochs(mm, *local_bal[cid], E); times.append(time.perf_counter() - t0)
        # Clients run in parallel on separate devices: the round lasts as long as the slowest one.
        fed_time += max(times) + net_s
        set_weights(glob_m, FLServer().aggregate_weights([get_weights(mm) for mm in fed_models.values()], 'mlp'))
    record('MLP', 'Federated', 'ALL', glob_m, fed_time, {'update_bytes': update, 'rounds': R})

    # ---- LSTM (raw delay/loss sequences, class weights instead of resampling) ----
    E, R = args.local_epochs, args.lstm_rounds
    m = build_lstm(N, pos_rate); t0 = time.perf_counter()
    m.fit(all_seq, all_y, epochs=R * E, batch_size=256, class_weight=class_weights(all_y), verbose=0)
    record('LSTM', 'Centralized', 'ALL', m, time.perf_counter() - t0)
    for cid, c in clients.items():
        m = build_lstm(N, pos_rate); t0 = time.perf_counter()
        m.fit(c['seq'], c['y'], epochs=R * E, batch_size=256, class_weight=class_weights(c['y']), verbose=0)
        record('LSTM', 'Local', cid, m, time.perf_counter() - t0)
    glob_m = build_lstm(N, pos_rate)
    fed_models = {cid: build_lstm(N, pos_rate) for cid in clients}
    update = payload_bytes(glob_m.get_weights())
    net_s = communication_time_s(update, args.rtt_ms, args.capacity_mbps)
    fed_time = 0.0
    for _ in range(R):
        times = []
        for cid, mm in fed_models.items():
            mm.set_weights(glob_m.get_weights())
            t0 = time.perf_counter()
            mm.fit(clients[cid]['seq'], clients[cid]['y'], epochs=E, batch_size=256,
                   class_weight=class_weights(clients[cid]['y']), verbose=0)
            times.append(time.perf_counter() - t0)
        fed_time += max(times) + net_s
        glob_m.set_weights(FLServer().aggregate_weights([mm.get_weights() for mm in fed_models.values()], 'lstm'))
    record('LSTM', 'Federated', 'ALL', glob_m, fed_time, {'update_bytes': update, 'rounds': R})

    tf.keras.backend.clear_session()
    return rows


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Multi-seed federated learning experiment (single tidy CSV).")
    p.add_argument('--config', default='config/exp_federated_1hz.yaml')
    p.add_argument('--n_sizes', nargs='+', type=int, default=[10, 30, 60])
    p.add_argument('--x_values', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--local_epochs', type=int, default=3)
    p.add_argument('--lstm_rounds', type=int, default=5, help="FedAvg rounds for the LSTM (local/centralized train R*E epochs)")
    p.add_argument('--mlp_rounds', type=int, default=20, help="FedAvg rounds for the MLP (cheap; 15 epochs left it unconverged)")
    p.add_argument('--rtt_ms', type=float, default=150.0)
    p.add_argument('--capacity_mbps', type=float, default=10.0)
    p.add_argument('--output', default='results/fl_seeds/fl_results.csv')
    p.add_argument('--merge', default=None, help="Directory with part_*.csv files to join into fl_results.csv, then exit")
    args = p.parse_args()

    if args.merge:
        parts = sorted(f for f in os.listdir(args.merge) if f.startswith('part_') and f.endswith('.csv'))
        merged = pd.concat([pd.read_csv(os.path.join(args.merge, f)) for f in parts], ignore_index=True)
        merged = merged.sort_values(['model', 'N', 'X', 'scope', 'cpe', 'seed'])
        merged.to_csv(os.path.join(args.merge, 'fl_results.csv'), index=False)
        print(f"[*] Merged {len(parts)} parts, {len(merged)} rows -> {args.merge}/fl_results.csv")
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    config = load_config(args.config)
    train_dict, test_dict = load_and_split_data(config)
    train_dfs, test_dfs = train_dict['mobile'], test_dict['mobile']
    print(f"[*] Seeds {args.seeds} | N {args.n_sizes} | X {args.x_values} | output {args.output}", flush=True)

    for N in args.n_sizes:
        for X in args.x_values:
            t0 = time.perf_counter()
            clients, val, test = prepare_grid_point(train_dfs, test_dfs, N, X)
            print(f"[*] N={N} X={X}: data ready in {time.perf_counter() - t0:.0f} s "
                  f"(test positives {int(test['y'].sum())}/{len(test['y'])})", flush=True)
            for seed in args.seeds:
                t1 = time.perf_counter()
                rows = run_seed(seed, N, X, clients, val, test, args)
                # Appended after every seed so an interrupted run keeps everything already computed.
                pd.DataFrame(rows).to_csv(args.output, mode='a', index=False, header=not os.path.exists(args.output))
                lstm_fed = [r for r in rows if r['model'] == 'LSTM' and r['scope'] == 'Federated'][0]
                print(f"    seed {seed}: {time.perf_counter() - t1:.0f} s | LSTM fed PR-AUC {lstm_fed['pr_auc']:.3f}", flush=True)
    print("[*] Done.", flush=True)


if __name__ == '__main__':
    main()
