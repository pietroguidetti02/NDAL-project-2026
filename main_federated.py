import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from multiprocessing.pool import ThreadPool
import xgboost as xgb

sys.path.append(os.getcwd())
from src.data_loader import load_config, load_and_split_data
from src.models import evaluate_model, train_lstm
from src.utils import plot_roc_pr_curves_2, plot_fl_training_times
from src.federated import FLClient, FLServer
from main_comparison_LSTM import process_dataset_all

def create_lstm_model(input_shape):
    import tensorflow as tf
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def run_federated():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--config', type=str, default='config/exp_federated.yaml', help='Path to config file')
    parser.add_argument('--rounds', type=int, default=5, help='Number of Communication Rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local training epochs per round')
    parser.add_argument('--network_delay', type=int, default=150, help='Network latency (ms) for weight upload/download')
    parser.add_argument('--n_sizes', type=int, nargs='+', default=[15, 30, 60], help='List of Lookback Windows (N) to sweep')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_federated_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting FEDERATED LEARNING Simulation ===")
    config = load_config(args.config)
    X = config.get('X', 5)
    
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    tunnel = 'mobile'
    train_dfs = train_dfs_dict.get(tunnel, [])
    test_dfs = test_dfs_dict.get(tunnel, [])
    
    if not train_dfs or not test_dfs:
        print("[!] Mobile dataset not found. Exiting.")
        return

    print("\n[*] Partitioning Data across 3 Routers (Client A, B, C)...")
    indices = np.array_split(range(len(train_dfs)), 3)
    splits = [[train_dfs[i] for i in idx] for idx in indices]
    clients_dfs = {'CPE_A': splits[0], 'CPE_B': splits[1], 'CPE_C': splits[2]}
    
    summary_results = []
    
    for N in args.n_sizes:
        print(f"\n{'='*60}")
        print(f"=== SWEEPING LOOKBACK WINDOW N={N} (X={X}) ===")
        print(f"{'='*60}")
        
        clients_data = {}
        print("\n[*] Extracting Time Windows (Parallel)...")
        for client_id, dfs in clients_dfs.items():
            if len(dfs) > 0:
                print(f"  -> Processing local data for {client_id}")
                X_df, X_seq, y = process_dataset_all(list(dfs), N, X)
                clients_data[client_id] = {'X_df': X_df, 'X_seq': X_seq, 'y': y}

        print("\n[*] Processing Global Test Set (Server)...")
        X_test_df, X_test_seq, y_test = process_dataset_all(test_dfs, N, X)
        
        # SCALING
        all_X_df = pd.concat([d['X_df'] for d in clients_data.values()], ignore_index=True)
        all_X_seq = np.concatenate([d['X_seq'] for d in clients_data.values()], axis=0)
        all_y = pd.concat([d['y'] for d in clients_data.values()], ignore_index=True)
        
        tab_scaler = StandardScaler().fit(all_X_df)
        seq_scaler = StandardScaler()
        flat_seq = all_X_seq[:,:,0].reshape(-1, 1)
        seq_scaler.fit(flat_seq)
        
        X_test_scaled = pd.DataFrame(tab_scaler.transform(X_test_df), columns=X_test_df.columns)
        
        X_test_seq_scaled = np.copy(X_test_seq)
        flat_test = X_test_seq[:,:,0].reshape(-1, 1)
        X_test_seq_scaled[:,:,0] = seq_scaler.transform(flat_test).reshape(X_test_seq.shape[0], X_test_seq.shape[1])
        
        for cid in clients_data:
            d = clients_data[cid]
            d['X_scaled'] = pd.DataFrame(tab_scaler.transform(d['X_df']), columns=d['X_df'].columns)
            seq_sc = np.copy(d['X_seq'])
            flat_loc = d['X_seq'][:,:,0].reshape(-1, 1)
            seq_sc[:,:,0] = seq_scaler.transform(flat_loc).reshape(d['X_seq'].shape[0], d['X_seq'].shape[1])
            d['X_seq_scaled'] = seq_sc
            
        X_train_centr = pd.concat([d['X_scaled'] for d in clients_data.values()], ignore_index=True)
        y_train_centr = pd.concat([d['y'] for d in clients_data.values()], ignore_index=True)
        X_train_centr_seq = np.concatenate([d['X_seq_scaled'] for d in clients_data.values()], axis=0)
        
        server = FLServer()
        network_penalty = (args.network_delay * 2) / 1000.0

        # ========================================================================
        # --- EXPERIMENT 1: MLP (NEURAL NETWORK) ---
        # ========================================================================
        print(f"\n--- SCENARIO 1: CENTRALIZED TRAINING (MLP) [N={N}] ---")
        nn_base = MLPClassifier(max_iter=args.rounds * args.local_epochs, batch_size=256, random_state=42)
        start_centr_mlp = time.perf_counter()
        nn_base.fit(X_train_centr, y_train_centr)
        time_centr_mlp = time.perf_counter() - start_centr_mlp
        metrics_centr_mlp = evaluate_model(nn_base, X_test_scaled, y_test)
        f1_centr_mlp = metrics_centr_mlp.get('f1', 0)
        print(f"  -> Total Time: {time_centr_mlp:.2f} s | F1: {f1_centr_mlp:.4f}")
        
        summary_results.append({'N': N, 'Model': 'MLP', 'Type': 'Centralized', 'Time_s': time_centr_mlp, 'F1_Score': f1_centr_mlp})

        print(f"\n--- SCENARIO 2: FEDERATED LEARNING (MLP) [N={N}] ---")
        global_mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
        global_mlp.partial_fit(X_train_centr[:2], y_train_centr[:2], classes=np.array([0, 1]))
        
        mlp_clients = []
        for cid in clients_data:
            cm = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
            cm.partial_fit(X_train_centr[:2], y_train_centr[:2], classes=np.array([0, 1]))
            c = FLClient(cid, clients_data[cid]['X_scaled'], clients_data[cid]['y'])
            c.set_model(cm)
            mlp_clients.append(c)

        time_fed_mlp = 0.0
        timing_records_mlp = []
        
        for r in range(1, args.rounds + 1):
            glob_w = {'coefs_': global_mlp.coefs_, 'intercepts_': global_mlp.intercepts_}
            for c in mlp_clients: c.set_weights(glob_w)
                
            def train_c(client): return client.train(epochs=args.local_epochs, batch_size=256)
            with ThreadPool(len(mlp_clients)) as pool: times = pool.map(train_c, mlp_clients)
                
            r_time = max(times) + network_penalty
            time_fed_mlp += r_time
            print(f"  [Round {r}] MLP Local Times: {['%.2f'%t for t in times]} | Round Time: {r_time:.2f}s")
            
            rec = {'Round': r, 'Network': network_penalty}
            for i, c in enumerate(mlp_clients): rec[c.client_id] = times[i]
            timing_records_mlp.append(rec)
            
            c_weights = [c.get_weights() for c in mlp_clients]
            new_w = server.aggregate_weights(c_weights, model_type='mlp')
            global_mlp.coefs_ = new_w['coefs_']
            global_mlp.intercepts_ = new_w['intercepts_']

        metrics_fed_mlp = evaluate_model(global_mlp, X_test_scaled, y_test)
        f1_fed_mlp = metrics_fed_mlp.get('f1', 0)
        print(f"  -> Total FEDERATED Time: {time_fed_mlp:.2f} s | F1: {f1_fed_mlp:.4f}")
        summary_results.append({'N': N, 'Model': 'MLP', 'Type': 'Federated', 'Time_s': time_fed_mlp, 'F1_Score': f1_fed_mlp})
        
        plot_roc_pr_curves_2(metrics_centr_mlp, metrics_fed_mlp, 'MLP Centralized', 'MLP Federated', output_dir, f'FL_MLP_N{N}')
        plot_fl_training_times(timing_records_mlp, output_dir=output_dir, prefix=f'MLP_N{N}')

        # ========================================================================
        # --- EXPERIMENT 2: LSTM ---
        # ========================================================================
        print(f"\n--- SCENARIO 3: CENTRALIZED TRAINING (LSTM) [N={N}] ---")
        lstm_centr = create_lstm_model((N, 2))
        start_centr_lstm = time.perf_counter()
        lstm_centr.fit(X_train_centr_seq, y_train_centr, epochs=args.rounds * args.local_epochs, batch_size=256, verbose=0)
        time_centr_lstm = time.perf_counter() - start_centr_lstm
        metrics_centr_lstm = evaluate_model(lstm_centr, X_test_seq_scaled, y_test, threshold=0.5)
        f1_centr_lstm = metrics_centr_lstm.get('f1', 0)
        print(f"  -> Total Time: {time_centr_lstm:.2f} s | F1: {f1_centr_lstm:.4f}")
        
        summary_results.append({'N': N, 'Model': 'LSTM', 'Type': 'Centralized', 'Time_s': time_centr_lstm, 'F1_Score': f1_centr_lstm})

        print(f"\n--- SCENARIO 4: FEDERATED LEARNING (LSTM) [N={N}] ---")
        global_lstm = create_lstm_model((N, 2))
        lstm_clients = []
        for cid in clients_data:
            cm = create_lstm_model((N, 2))
            c = FLClient(cid, clients_data[cid]['X_seq_scaled'], clients_data[cid]['y'])
            c.set_model(cm)
            lstm_clients.append(c)

        time_fed_lstm = 0.0
        timing_records_lstm = []
        
        for r in range(1, args.rounds + 1):
            glob_w = global_lstm.get_weights()
            for c in lstm_clients: c.set_weights(glob_w)
                
            def train_l(client): return client.train(epochs=args.local_epochs, batch_size=256)
            with ThreadPool(len(lstm_clients)) as pool: times = pool.map(train_l, lstm_clients)
                
            r_time = max(times) + network_penalty
            time_fed_lstm += r_time
            print(f"  [Round {r}] LSTM Local Times: {['%.2f'%t for t in times]} | Round Time: {r_time:.2f}s")
            
            rec = {'Round': r, 'Network': network_penalty}
            for i, c in enumerate(lstm_clients): rec[c.client_id] = times[i]
            timing_records_lstm.append(rec)
            
            c_weights = [c.get_weights() for c in lstm_clients]
            new_w = server.aggregate_weights(c_weights, model_type='lstm')
            global_lstm.set_weights(new_w)

        metrics_fed_lstm = evaluate_model(global_lstm, X_test_seq_scaled, y_test, threshold=0.5)
        f1_fed_lstm = metrics_fed_lstm.get('f1', 0)
        print(f"  -> Total FEDERATED Time: {time_fed_lstm:.2f} s | F1: {f1_fed_lstm:.4f}")
        summary_results.append({'N': N, 'Model': 'LSTM', 'Type': 'Federated', 'Time_s': time_fed_lstm, 'F1_Score': f1_fed_lstm})
        
        plot_roc_pr_curves_2(metrics_centr_lstm, metrics_fed_lstm, 'LSTM Centralized', 'LSTM Federated', output_dir, f'FL_LSTM_N{N}')
        plot_fl_training_times(timing_records_lstm, output_dir=output_dir, prefix=f'LSTM_N{N}')

    # ========================================================================
    print("\n[*] Saving summary results to CSV...")
    df_summary = pd.DataFrame(summary_results)
    csv_path = os.path.join(output_dir, 'federated_sweeping_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print("\n", df_summary.to_string(index=False))
    
    print("\n[*] FL Sweeping completed! Check the results/ folder for CSV and comparative plots.")

if __name__ == '__main__':
    run_federated()
