import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing.pool import ThreadPool
import xgboost as xgb

sys.path.append(os.getcwd())
from src.data_loader import load_config, load_and_split_data
from src.models import evaluate_model, train_lstm
from src.utils import plot_roc_pr_curves_2, plot_fl_training_times, plot_roc_pr_curves_multi
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

def balance_tabular_data(X, y):
    """Applica SMOTE se ci sono abbastanza campioni, altrimenti RandomOverSampler, altrimenti nulla."""
    pos_count = np.sum(y == 1)
    if pos_count > 5:
        # Abbastanza per SMOTE (default k_neighbors=5)
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    elif pos_count > 0:
        # Troppo pochi per SMOTE, usiamo la duplicazione pura
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X, y)
    else:
        # Nessun guasto in questo router! Impossibile bilanciare.
        return X, y

def get_class_weights(y):
    """Calcola i pesi delle classi per LSTM."""
    classes = np.unique(y)
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    else:
        return {0: 1.0, 1: 1.0}

def save_roc_pr_csv_multi(metrics_dict, output_dir, prefix):
    rows = []
    for model_name, m in metrics_dict.items():
        if 'fpr' in m and 'tpr' in m:
            for i in range(len(m['fpr'])):
                rows.append({'Scenario': model_name, 'Curve': 'ROC', 'X': m['fpr'][i], 'Y': m['tpr'][i]})
        if 'precision_curve' in m and 'recall_curve' in m:
            for i in range(len(m['precision_curve'])):
                rows.append({'Scenario': model_name, 'Curve': 'PR', 'X': m['recall_curve'][i], 'Y': m['precision_curve'][i]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'{prefix}_roc_pr_data.csv'), index=False)

def append_summary(file_path, record):
    df = pd.DataFrame([record])
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

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
    
    summary_csv_path = os.path.join(output_dir, 'federated_sweeping_summary.csv')
    
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
            
            # BILANCIAMENTO LOCALE TABULARE PER MLP
            d['X_scaled_bal'], d['y_bal'] = balance_tabular_data(d['X_scaled'], d['y'])
            # PESI CLASSI LOCALI PER LSTM
            d['class_weights'] = get_class_weights(d['y'])
            
        X_train_centr = pd.concat([d['X_scaled'] for d in clients_data.values()], ignore_index=True)
        y_train_centr = pd.concat([d['y'] for d in clients_data.values()], ignore_index=True)
        X_train_centr_seq = np.concatenate([d['X_seq_scaled'] for d in clients_data.values()], axis=0)
        
        # BILANCIAMENTO GLOBALE CENTRALIZZATO
        X_train_centr_bal, y_train_centr_bal = balance_tabular_data(X_train_centr, y_train_centr)
        centr_class_weights = get_class_weights(y_train_centr)
        
        server = FLServer()
        network_penalty = (args.network_delay * 2) / 1000.0

        # ========================================================================
        # --- EXPERIMENT 1: MLP (NEURAL NETWORK) ---
        # ========================================================================
        multi_metrics_mlp = {}
        
        print(f"\n--- SCENARIO 1: CENTRALIZED TRAINING (MLP) [N={N}] ---")
        nn_base = MLPClassifier(max_iter=args.rounds * args.local_epochs, batch_size=256, random_state=42)
        start_centr_mlp = time.perf_counter()
        nn_base.fit(X_train_centr_bal, y_train_centr_bal)
        time_centr_mlp = time.perf_counter() - start_centr_mlp
        metrics_centr_mlp = evaluate_model(nn_base, X_test_scaled, y_test)
        f1_centr_mlp = metrics_centr_mlp.get('f1', 0)
        print(f"  -> Total Time: {time_centr_mlp:.2f} s | F1: {f1_centr_mlp:.4f}")
        append_summary(summary_csv_path, {'N': N, 'Model': 'MLP', 'Type': 'Centralized', 'Time_s': time_centr_mlp, 'F1_Score': f1_centr_mlp})
        multi_metrics_mlp['Centralized'] = metrics_centr_mlp

        print(f"\n--- SCENARIO 2: STRICTLY LOCAL MODELS (MLP) [N={N}] ---")
        for cid, d in clients_data.items():
            loc_model = MLPClassifier(max_iter=args.rounds * args.local_epochs, batch_size=256, random_state=42)
            start_loc = time.perf_counter()
            loc_model.fit(d['X_scaled_bal'], d['y_bal'])
            time_loc = time.perf_counter() - start_loc
            metrics_loc = evaluate_model(loc_model, X_test_scaled, y_test)
            f1_loc = metrics_loc.get('f1', 0)
            print(f"  -> {cid} Local Time: {time_loc:.2f} s | F1: {f1_loc:.4f} (Evaluated on Global Test Set)")
            append_summary(summary_csv_path, {'N': N, 'Model': 'MLP', 'Type': f'Local_{cid}', 'Time_s': time_loc, 'F1_Score': f1_loc})
            multi_metrics_mlp[f'Local {cid}'] = metrics_loc

        print(f"\n--- SCENARIO 3: FEDERATED LEARNING (MLP) [N={N}] ---")
        global_mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
        global_mlp.partial_fit(X_train_centr_bal[:2], y_train_centr_bal[:2], classes=np.array([0, 1]))
        
        mlp_clients = []
        for cid in clients_data:
            cm = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
            cm.partial_fit(X_train_centr_bal[:2], y_train_centr_bal[:2], classes=np.array([0, 1]))
            # Passiamo i dati BILANCIATI al client FL!
            c = FLClient(cid, clients_data[cid]['X_scaled_bal'], clients_data[cid]['y_bal'])
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
        append_summary(summary_csv_path, {'N': N, 'Model': 'MLP', 'Type': 'Federated', 'Time_s': time_fed_mlp, 'F1_Score': f1_fed_mlp})
        multi_metrics_mlp['Federated'] = metrics_fed_mlp
        
        prefix_mlp = f'FL_MLP_N{N}'
        plot_roc_pr_curves_multi(multi_metrics_mlp, output_dir, prefix_mlp)
        save_roc_pr_csv_multi(multi_metrics_mlp, output_dir, prefix_mlp)
        
        pd.DataFrame(timing_records_mlp).to_csv(os.path.join(output_dir, f'{prefix_mlp}_timing_records.csv'), index=False)
        plot_fl_training_times(timing_records_mlp, output_dir=output_dir, prefix=prefix_mlp)

        # ========================================================================
        # --- EXPERIMENT 2: LSTM ---
        # ========================================================================
        multi_metrics_lstm = {}
        
        print(f"\n--- SCENARIO 4: CENTRALIZED TRAINING (LSTM) [N={N}] ---")
        lstm_centr = create_lstm_model((N, 2))
        start_centr_lstm = time.perf_counter()
        # Per LSTM usiamo i pesi delle classi per non toccare le sequenze
        lstm_centr.fit(X_train_centr_seq, y_train_centr, epochs=args.rounds * args.local_epochs, batch_size=256, class_weight=centr_class_weights, verbose=0)
        time_centr_lstm = time.perf_counter() - start_centr_lstm
        metrics_centr_lstm = evaluate_model(lstm_centr, X_test_seq_scaled, y_test, threshold=0.5)
        f1_centr_lstm = metrics_centr_lstm.get('f1', 0)
        print(f"  -> Total Time: {time_centr_lstm:.2f} s | F1: {f1_centr_lstm:.4f}")
        append_summary(summary_csv_path, {'N': N, 'Model': 'LSTM', 'Type': 'Centralized', 'Time_s': time_centr_lstm, 'F1_Score': f1_centr_lstm})
        multi_metrics_lstm['Centralized'] = metrics_centr_lstm

        print(f"\n--- SCENARIO 5: STRICTLY LOCAL MODELS (LSTM) [N={N}] ---")
        for cid, d in clients_data.items():
            loc_model = create_lstm_model((N, 2))
            start_loc = time.perf_counter()
            loc_model.fit(d['X_seq_scaled'], d['y'], epochs=args.rounds * args.local_epochs, batch_size=256, class_weight=d['class_weights'], verbose=0)
            time_loc = time.perf_counter() - start_loc
            metrics_loc = evaluate_model(loc_model, X_test_seq_scaled, y_test, threshold=0.5)
            f1_loc = metrics_loc.get('f1', 0)
            print(f"  -> {cid} Local Time: {time_loc:.2f} s | F1: {f1_loc:.4f} (Evaluated on Global Test Set)")
            append_summary(summary_csv_path, {'N': N, 'Model': 'LSTM', 'Type': f'Local_{cid}', 'Time_s': time_loc, 'F1_Score': f1_loc})
            multi_metrics_lstm[f'Local {cid}'] = metrics_loc

        print(f"\n--- SCENARIO 6: FEDERATED LEARNING (LSTM) [N={N}] ---")
        global_lstm = create_lstm_model((N, 2))
        lstm_clients = []
        for cid in clients_data:
            cm = create_lstm_model((N, 2))
            # Aggiorniamo la classe FLClient se necessario, o aggiriamo passando class_weight al fit
            # Dato che FLClient definisce il suo metodo train(), per semplicità non glieli passiamo,
            # MA aspetta, il client LOCALE deve bilanciare altrimenti fa danni! 
            # Dobbiamo assicurarci che FLClient usi class_weight.
            c = FLClient(cid, clients_data[cid]['X_seq_scaled'], clients_data[cid]['y'])
            # Hack per passare class_weights dentro l'istanza client:
            c.class_weights = clients_data[cid]['class_weights']
            c.set_model(cm)
            lstm_clients.append(c)

        time_fed_lstm = 0.0
        timing_records_lstm = []
        
        for r in range(1, args.rounds + 1):
            glob_w = global_lstm.get_weights()
            for c in lstm_clients: c.set_weights(glob_w)
                
            def train_l(client): 
                # Chiamata manuale a keras fit nel client data la mancanza del parametro class_weight in FLClient.train()
                start_time = time.perf_counter()
                client.model.fit(client.X_train, client.y_train, epochs=args.local_epochs, batch_size=256, class_weight=client.class_weights, verbose=0)
                return time.perf_counter() - start_time
                
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
        append_summary(summary_csv_path, {'N': N, 'Model': 'LSTM', 'Type': 'Federated', 'Time_s': time_fed_lstm, 'F1_Score': f1_fed_lstm})
        multi_metrics_lstm['Federated'] = metrics_fed_lstm
        
        prefix_lstm = f'FL_LSTM_N{N}'
        plot_roc_pr_curves_multi(multi_metrics_lstm, output_dir, prefix_lstm)
        save_roc_pr_csv_multi(multi_metrics_lstm, output_dir, prefix_lstm)
        
        pd.DataFrame(timing_records_lstm).to_csv(os.path.join(output_dir, f'{prefix_lstm}_timing_records.csv'), index=False)
        plot_fl_training_times(timing_records_lstm, output_dir=output_dir, prefix=prefix_lstm)

    # ========================================================================
    print("\n[*] FL Sweeping completed! Check the results/ folder for CSVs and comparative plots.")

if __name__ == '__main__':
    run_federated()
