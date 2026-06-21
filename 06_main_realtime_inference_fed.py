import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import copy
import warnings

sys.path.append(os.getcwd())
from src.preprocessor import clean_data
from src.features import engineer_features
from src.utils import plot_inference_ecdf, plot_inference_boxplot, plot_inference_barplot

def worker_local_simulation(cpe_name, base_model, N, X_size, delays, packet_loss, global_max, train_samples, num_simulations):
    """
    Simula una singola CPE puramente Locale sul suo streaming di dati (No server, No rete).
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import time
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from src.features import engineer_features
    import warnings
    
    X_tab, X_seq, y_train = [], [], []
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
    if base_model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=1, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tab_scaled, y_train_np)
    elif base_model == 'LSTM':
        import tensorflow as tf
        from src.models import train_lstm
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.keras.backend.clear_session()
        X_seq_np_scaled = np.copy(X_seq_np)
        X_seq_np_scaled[:,:,0] = seq_scaler.transform(flat_seq).reshape(X_seq_np.shape[0], X_seq_np.shape[1])
        model = train_lstm(X_seq_np_scaled, y_train_np, params={'epochs': 1, 'batch_size': 32, 'verbose': 0})

    results = []
    start_idx = train_samples + 100
    max_possible_sims = len(delays) - start_idx - N - X_size + 1
    actual_sims = num_simulations if num_simulations not in ['infinity', float('inf')] else max_possible_sims
    actual_sims = min(int(actual_sims), max_possible_sims)
    
    model_label = f"{base_model} Local - {cpe_name}"
    
    for i in range(start_idx, start_idx + actual_sims):
        lookback_delays = delays[i : i+N]
        lookback_losses = packet_loss[i : i+N]
        future_losses = packet_loss[i+N : i+N+X_size]
        label = 1 if np.sum(future_losses) > 0 else 0
        y_curr = np.array([label])
        
        if base_model == 'MLP':
            feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
            feats_df = pd.DataFrame([feats])[feature_cols]
            f_scaled = tab_scaler.transform(feats_df)
            
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.partial_fit(f_scaled, y_curr, classes=np.array([0, 1]))
            t1 = time.perf_counter()
            t2 = time.perf_counter()
            _ = model.predict_proba(f_scaled)
            t3 = time.perf_counter()
            total_time_s = (t1 - t0) + (t3 - t2)
            
        elif base_model == 'LSTM':
            seq_d = np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1)
            seq_d_scaled = seq_scaler.transform(seq_d)
            seq_final = np.column_stack((seq_d_scaled, lookback_losses)).reshape(1, N, 2)
            
            t0 = time.perf_counter()
            model.fit(seq_final, y_curr, epochs=1, verbose=0)
            t1 = time.perf_counter()
            t2 = time.perf_counter()
            _ = model(seq_final, training=False)
            t3 = time.perf_counter()
            total_time_s = (t1 - t0) + (t3 - t2)
            
        results.append({'Model': model_label, 'N': N, 'InferenceTime_ms': total_time_s * 1000.0})
        
        processed = i - start_idx + 1
        if processed % 1000 == 0 or processed == actual_sims:
            print(f"      [{model_label} | N={N}] - Processati {processed}/{actual_sims} pacchetti...")
        
    print(f"      [{model_label} | N={N}] - Simulazione locale completata.")
    return results

def worker_federated_simulation(base_model, N, X_size, cpe_data, global_max, train_samples, num_simulations):
    """
    Simula l'architettura Federated REALE:
    3 CPE leggono ognuna SOLO i propri file in parallelo (step-by-step).
    Il server attende che tutte completino il calcolo, riceve i pesi, 
    effettua il joint e weighting (FedAvg) e li rispedisce.
    Tempo Costo = Max(Tempi CPE) + Network Overhead + Tempo di Media del Server.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import time
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from src.features import engineer_features
    import warnings
    
    NETWORK_DELAY_MS = 150.0
    
    # 1. Preparazione scalers per ogni CPE (storico locale)
    cpe_names = list(cpe_data.keys())
    cpe_objects = {}
    
    for cpe in cpe_names:
        delays, packet_loss = cpe_data[cpe]
        X_tab, X_seq, y_train = [], [], []
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
        
        cpe_objects[cpe] = {
            'X_tab_scaled': X_tab_scaled, 'X_seq_np': X_seq_np, 'y_train_np': y_train_np,
            'tab_scaler': tab_scaler, 'seq_scaler': seq_scaler, 'feature_cols': feature_cols
        }

    # 2. Addestramento Base (Pre-training)
    cpe_models = {}
    if base_model == 'MLP':
        from sklearn.neural_network import MLPClassifier
        for cpe in cpe_names:
            m = MLPClassifier(max_iter=1, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(cpe_objects[cpe]['X_tab_scaled'], cpe_objects[cpe]['y_train_np'])
            cpe_models[cpe] = m
    elif base_model == 'LSTM':
        import tensorflow as tf
        from src.models import train_lstm
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.keras.backend.clear_session()
        for cpe in cpe_names:
            X_seq_np_scaled = np.copy(cpe_objects[cpe]['X_seq_np'])
            X_seq_np_scaled[:,:,0] = cpe_objects[cpe]['seq_scaler'].transform(cpe_objects[cpe]['X_seq_np'][:,:,0].reshape(-1,1)).reshape(cpe_objects[cpe]['X_seq_np'].shape[0], cpe_objects[cpe]['X_seq_np'].shape[1])
            cpe_objects[cpe]['X_seq_scaled'] = X_seq_np_scaled
            m = train_lstm(X_seq_np_scaled, cpe_objects[cpe]['y_train_np'], params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
            cpe_models[cpe] = m

    # 3. Simulazione Online Sincronizzata (Il vero Federated Server)
    start_idx = train_samples + 100
    # Ferma la simulazione quando il file più corto finisce, per mantenere il server sincronizzato
    max_possible_sims = min([len(cpe_data[c][0]) for c in cpe_names]) - start_idx - N - X_size + 1
    actual_sims = num_simulations if num_simulations not in ['infinity', float('inf')] else max_possible_sims
    actual_sims = min(int(actual_sims), max_possible_sims)
    
    results = []
    model_label = f"{base_model} Federated"
    
    for i in range(start_idx, start_idx + actual_sims):
        cpe_update_times = []
        cpe_inference_times = []
        
        # --- A. Fase LOCALE indipendente su ogni CPE (Processano SOLO i loro file) ---
        for cpe in cpe_names:
            delays, packet_loss = cpe_data[cpe]
            lookback_delays = delays[i : i+N]
            lookback_losses = packet_loss[i : i+N]
            future_losses = packet_loss[i+N : i+N+X_size]
            label = 1 if np.sum(future_losses) > 0 else 0
            y_curr = np.array([label])
            
            if base_model == 'MLP':
                feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
                feats_df = pd.DataFrame([feats])[cpe_objects[cpe]['feature_cols']]
                f_scaled = cpe_objects[cpe]['tab_scaler'].transform(feats_df)
                
                # Inferenza locale prima dell'update
                t2 = time.perf_counter()
                _ = cpe_models[cpe].predict_proba(f_scaled)
                t3 = time.perf_counter()
                cpe_inference_times.append(t3 - t2)
                
                # Update (Calcolo dei gradienti/pesi locali)
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cpe_models[cpe].partial_fit(f_scaled, y_curr, classes=np.array([0, 1]))
                t1 = time.perf_counter()
                cpe_update_times.append(t1 - t0)
                
            elif base_model == 'LSTM':
                seq_d = np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1)
                seq_d_scaled = cpe_objects[cpe]['seq_scaler'].transform(seq_d)
                seq_final = np.column_stack((seq_d_scaled, lookback_losses)).reshape(1, N, 2)
                
                t2 = time.perf_counter()
                _ = cpe_models[cpe](seq_final, training=False)
                t3 = time.perf_counter()
                cpe_inference_times.append(t3 - t2)
                
                t0 = time.perf_counter()
                cpe_models[cpe].fit(seq_final, y_curr, epochs=1, verbose=0)
                t1 = time.perf_counter()
                cpe_update_times.append(t1 - t0)

        # --- B. Fase SERVER (Joint and Weightening - FedAvg) ---
        t_server_0 = time.perf_counter()
        if base_model == 'MLP':
            avg_coefs = [sum(cpe_models[c].coefs_[k] for c in cpe_names) / len(cpe_names) for k in range(len(cpe_models[cpe_names[0]].coefs_))]
            avg_intercepts = [sum(cpe_models[c].intercepts_[k] for c in cpe_names) / len(cpe_names) for k in range(len(cpe_models[cpe_names[0]].intercepts_))]
            for c in cpe_names:
                cpe_models[c].coefs_ = avg_coefs
                cpe_models[c].intercepts_ = avg_intercepts
        elif base_model == 'LSTM':
            new_w = []
            for k in range(len(cpe_models[cpe_names[0]].get_weights())):
                new_w.append(sum(cpe_models[c].get_weights()[k] for c in cpe_names) / len(cpe_names))
            for c in cpe_names:
                cpe_models[c].set_weights(new_w)
        t_server_1 = time.perf_counter()
        server_time = t_server_1 - t_server_0
        
        # --- C. Costo del Federated (Tempo Totale Round) ---
        # Il server attende il worker più lento + la latenza di rete A/R + il suo tempo di averaging
        max_cpe_time = max([upd + inf for upd, inf in zip(cpe_update_times, cpe_inference_times)])
        network_overhead_s = (NETWORK_DELAY_MS * 2) / 1000.0
        
        total_federated_round_latency = max_cpe_time + network_overhead_s + server_time
        
        results.append({'Model': model_label, 'N': N, 'InferenceTime_ms': total_federated_round_latency * 1000.0})
        
        processed = i - start_idx + 1
        if processed % 1000 == 0 or processed == actual_sims:
            print(f"      [{model_label} | N={N}] - Processati {processed}/{actual_sims} round sincronizzati...")
            
    print(f"      [{model_label} | N={N}] - Simulazione federata completata.")
    return results

def run_realtime_simulation(dir_path, n_sizes=[10, 15, 30, 60], X=5, num_simulations=500, output_dir=None):
    print(f"[*] Caricamento file per simulazione live da: {dir_path}")
    
    # Rigoroso caricamento locale: A elabora solo se sorgente=A
    cpe_map = {
        'CPE_A': ['cpe_a-cpe_b-mobile.csv', 'cpe_a-cpe_c-mobile.csv'],
        'CPE_B': ['cpe_b-cpe_a-mobile.csv', 'cpe_b-cpe_c-mobile.csv'],
        'CPE_C': ['cpe_c-cpe_a-mobile.csv', 'cpe_c-cpe_b-mobile.csv']
    }
    
    cpe_data = {}
    global_max = 0
    
    for cpe_name, filenames in cpe_map.items():
        dfs = []
        for fname in filenames:
            fpath = os.path.join(dir_path, fname)
            if os.path.exists(fpath):
                dfs.append(pd.read_csv(fpath))
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df_clean = clean_data(df)
            m = df_clean['delay_ms'].max()
            if pd.notna(m) and m > global_max:
                global_max = m
            cpe_data[cpe_name] = (df_clean['delay_ms'].values, df_clean['packet_loss'].values)
            
    if global_max == 0: global_max = 1000.0
    
    all_results = []
    train_samples = 200
    
    for N in n_sizes:
        print(f"\n{'='*50}\n[*] Avvio Pipeline Simulazione Sincronizzata per N={N} (8 Processi)...\n{'='*50}")
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # --- MLP: 3 Locali indipendenti + 1 Sistema Federated Globale ---
            for cpe_name in cpe_data:
                delays, losses = cpe_data[cpe_name]
                futures.append(executor.submit(worker_local_simulation, cpe_name, 'MLP', N, X, delays, losses, global_max, train_samples, num_simulations))
            futures.append(executor.submit(worker_federated_simulation, 'MLP', N, X, cpe_data, global_max, train_samples, num_simulations))
            
            # --- LSTM: 3 Locali indipendenti + 1 Sistema Federated Globale ---
            for cpe_name in cpe_data:
                delays, losses = cpe_data[cpe_name]
                futures.append(executor.submit(worker_local_simulation, cpe_name, 'LSTM', N, X, delays, losses, global_max, train_samples, num_simulations))
            futures.append(executor.submit(worker_federated_simulation, 'LSTM', N, X, cpe_data, global_max, train_samples, num_simulations))
            
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
        plot_inference_barplot(results_df, x_thresholds=None, output_dir=output_dir, convert_to_seconds=False)
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
        run_realtime_simulation(dir_to_use, n_sizes=[10, 15, 30, 60], X=1, num_simulations="infinity", output_dir=output_dir)
    else:
        print(f"[!] Errore: Cartella dataset '{dir_to_use}' non trovata per la simulazione.")
