import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.getcwd())
from src.preprocessor import clean_data
from src.features import engineer_features
from src.utils import plot_inference_ecdf, plot_inference_boxplot

def worker_simulation(model_type, N, X_size, delays, packet_loss, global_max, train_samples, actual_sims, start_idx):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from src.features import engineer_features
    import warnings

    NETWORK_DELAY_MS = 150.0

    X_tab = []
    X_seq = []
    y_train = []
    
    for j in range(train_samples):
        lookback_d = delays[j : j+N]
        lookback_l = packet_loss[j : j+N]
        feats = engineer_features(lookback_d, lookback_l, global_max_delay=global_max)
        seq_d = np.nan_to_num(lookback_d, nan=global_max)
        seq = np.column_stack((seq_d, lookback_l))
        X_tab.append(feats)
        X_seq.append(seq)
        y_train.append(1 if j % 10 == 0 else 0)
        
    X_tab_df = pd.DataFrame(X_tab)
    X_seq_np = np.array(X_seq)
    y_train_np = np.array(y_train)
    
    tab_scaler = StandardScaler()
    X_tab_scaled = tab_scaler.fit_transform(X_tab_df)
    feature_cols = X_tab_df.columns
    
    seq_scaler = StandardScaler()
    flat_seq = X_seq_np[:,:,0].reshape(-1, 1)
    seq_scaler.fit(flat_seq)
    
    model = None
    if 'MLP' in model_type:
        from sklearn.neural_network import MLPClassifier
        if 'Federated' in model_type:
            m1 = MLPClassifier(max_iter=1, random_state=42).fit(X_tab_scaled[::3], y_train_np[::3])
            m2 = MLPClassifier(max_iter=1, random_state=43).fit(X_tab_scaled[1::3], y_train_np[1::3])
            m3 = MLPClassifier(max_iter=1, random_state=44).fit(X_tab_scaled[2::3], y_train_np[2::3])
            model = MLPClassifier(max_iter=1, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tab_scaled[:2], y_train_np[:2])
            model.coefs_ = [(c1+c2+c3)/3 for c1,c2,c3 in zip(m1.coefs_, m2.coefs_, m3.coefs_)]
            model.intercepts_ = [(i1+i2+i3)/3 for i1,i2,i3 in zip(m1.intercepts_, m2.intercepts_, m3.intercepts_)]
        else:
            model = MLPClassifier(max_iter=1, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tab_scaled, y_train_np)
    elif 'LSTM' in model_type:
        import tensorflow as tf
        from src.models import train_lstm
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.keras.backend.clear_session()
        
        X_seq_np_scaled = np.copy(X_seq_np)
        X_seq_np_scaled[:,:,0] = seq_scaler.transform(flat_seq).reshape(X_seq_np.shape[0], X_seq_np.shape[1])
        try:
            if 'Federated' in model_type:
                m1 = train_lstm(X_seq_np_scaled[::3], y_train_np[::3], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
                m2 = train_lstm(X_seq_np_scaled[1::3], y_train_np[1::3], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
                m3 = train_lstm(X_seq_np_scaled[2::3], y_train_np[2::3], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
                model = train_lstm(X_seq_np_scaled[:2], y_train_np[:2], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
                w1 = m1.get_weights()
                w2 = m2.get_weights()
                w3 = m3.get_weights()
                new_w = [(w1[i]+w2[i]+w3[i])/3 for i in range(len(w1))]
                model.set_weights(new_w)
            else:
                model = train_lstm(X_seq_np_scaled, y_train_np, params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
        except Exception as e:
            print(f"[!] Errore setup LSTM nel worker: {e}")
            return []

    results = []
    print(f"      [{model_type}] Inizio simulazione ONLINE TRAINING LIVE su {actual_sims} pacchetti...")
    
    for i in range(start_idx, start_idx + actual_sims):
        lookback_delays = delays[i : i+N]
        lookback_losses = packet_loss[i : i+N]
        
        future_losses = packet_loss[i+N : i+N+X_size]
        label = 1 if np.sum(future_losses) > 0 else 0
        y_curr = np.array([label])
        
        if 'MLP' in model_type:
            feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
            feats_df = pd.DataFrame([feats])[feature_cols]
            f_scaled = tab_scaler.transform(feats_df)
            
            if 'Federated' in model_type:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.partial_fit(f_scaled, y_curr, classes=np.array([0, 1]))
                t1 = time.perf_counter()
                network_overhead_s = (NETWORK_DELAY_MS * 2) / 1000.0
                t2 = time.perf_counter()
                _ = model.predict_proba(f_scaled)
                t3 = time.perf_counter()
                total_time_s = (t1 - t0) + network_overhead_s + (t3 - t2)
            else:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.partial_fit(f_scaled, y_curr, classes=np.array([0, 1]))
                t1 = time.perf_counter()
                t2 = time.perf_counter()
                _ = model.predict_proba(f_scaled)
                t3 = time.perf_counter()
                total_time_s = (t1 - t0) + (t3 - t2)
                
        elif 'LSTM' in model_type:
            seq_d = np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1)
            seq_d_scaled = seq_scaler.transform(seq_d)
            seq_final = np.column_stack((seq_d_scaled, lookback_losses)).reshape(1, N, 2)
            
            if 'Federated' in model_type:
                t0 = time.perf_counter()
                model.fit(seq_final, y_curr, epochs=1, verbose=0)
                t1 = time.perf_counter()
                network_overhead_s = (NETWORK_DELAY_MS * 2) / 1000.0
                t2 = time.perf_counter()
                _ = model(seq_final, training=False)
                t3 = time.perf_counter()
                total_time_s = (t1 - t0) + network_overhead_s + (t3 - t2)
            else:
                t0 = time.perf_counter()
                model.fit(seq_final, y_curr, epochs=1, verbose=0)
                t1 = time.perf_counter()
                t2 = time.perf_counter()
                _ = model(seq_final, training=False)
                t3 = time.perf_counter()
                total_time_s = (t1 - t0) + (t3 - t2)
            
        results.append({'Model': model_type, 'N': N, 'InferenceTime_ms': total_time_s * 1000.0})
        
        if i % 100 == 0 and i > start_idx:
            print(f"      [{model_type}] Elaborati {i - start_idx} pacchetti...")
            
    print(f"      [{model_type}] Simulazione completata.")
    return results

def run_realtime_simulation(dir_path, n_sizes=[10, 15, 30, 60], X=5, num_simulations=500, output_dir=None):
    print(f"[*] Caricamento file per simulazione live da: {dir_path}")
    
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith("mobile.csv")]
    print(f"[*] Trovati {len(files)} file mobile. Concatenazione in corso...")
    
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    df_clean = clean_data(df)
    
    global_max = df_clean['delay_ms'].max()
    if pd.isna(global_max): global_max = 1000.0
        
    delays = df_clean['delay_ms'].values
    packet_loss = df_clean['packet_loss'].values
    
    all_results = []
    train_samples = 200
    
    for N in n_sizes:
        print(f"\n{'='*50}\n[*] Avvio Pipeline Simulazione Multi-Processo per N={N}...\n{'='*50}")
        
        start_idx = train_samples + 100
        max_possible_sims = len(delays) - start_idx - N - X + 1
        
        if num_simulations == 'infinity' or num_simulations == float('inf'):
            actual_sims = max_possible_sims
        else:
            actual_sims = min(int(num_simulations), max_possible_sims)
            
        if actual_sims <= 0:
            print(f"  [!] Attenzione: dati insufficienti per avviare la simulazione con N={N}.")
            continue

        print(f"  -> Avvio dei 4 processi paralleli (MLP_Local, MLP_Fed, LSTM_Local, LSTM_Fed)...")
        print(f"  -> Ciascun processo effettuerà l'update ONLINE su {actual_sims} pacchetti sul proprio Core.")
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(worker_simulation, 'MLP_Local', N, X, delays, packet_loss, global_max, train_samples, actual_sims, start_idx),
                executor.submit(worker_simulation, 'MLP_Federated', N, X, delays, packet_loss, global_max, train_samples, actual_sims, start_idx),
                executor.submit(worker_simulation, 'LSTM_Local', N, X, delays, packet_loss, global_max, train_samples, actual_sims, start_idx),
                executor.submit(worker_simulation, 'LSTM_Federated', N, X, delays, packet_loss, global_max, train_samples, actual_sims, start_idx)
            ]
            
            for future in futures:
                try:
                    res = future.result()
                    all_results.extend(res)
                except Exception as e:
                    print(f"  [!] Errore critico nel processo worker: {e}")

    results_df = pd.DataFrame(all_results)
    
    if output_dir and not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'realtime_inference_results.csv'), index=False)
        
    print("\n" + "="*50)
    print("=== REPORT LATENZA ONLINE LEARNING (Ms) ===")
    if not results_df.empty:
        summary = results_df.groupby(['Model', 'N'])['InferenceTime_ms'].agg(['mean', 'max', lambda x: np.percentile(x, 99)])
        summary.columns = ['Mean (ms)', 'Max (ms)', '99th Pct (ms)']
        print(summary)
        
        print("\n[*] Generazione Grafici...")
        plot_inference_ecdf(results_df, x_thresholds=None, output_dir=output_dir, convert_to_seconds=False)
        plot_inference_boxplot(results_df, x_thresholds=None, output_dir=output_dir, convert_to_seconds=False)
        print(f"[*] Tutti i grafici salvati in: {output_dir}")
    else:
        print("[!] Nessun risultato raccolto.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_realtime_fed_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    dir_to_use = r"dataset/second_capture_window"
            
    if os.path.exists(dir_to_use):
        run_realtime_simulation(dir_to_use, n_sizes=[10, 15, 30, 60], X=1, num_simulations=500, output_dir=output_dir)
    else:
        print(f"[!] Errore: Cartella dataset '{dir_to_use}' non trovata per la simulazione.")
