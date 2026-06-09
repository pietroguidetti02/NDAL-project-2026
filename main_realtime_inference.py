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

def worker_simulation(model_type, N, delays, packet_loss, global_max, train_samples, actual_sims, start_idx):
    import time
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from src.features import engineer_features
    import warnings

    # Generazione mini-dataset indipendente per ogni processo
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
    if model_type == 'XGBoost':
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
        model.fit(X_tab_scaled, y_train_np)
    elif model_type == 'MLP_NN':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(max_iter=1, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tab_scaled, y_train_np)
    elif model_type == 'LSTM':
        import tensorflow as tf
        from src.models import train_lstm
        # IMPORTANTE: Limitiamo i thread di TF dentro al processo per evitare collisioni/SIGSEGV
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.keras.backend.clear_session()
        
        X_seq_np_scaled = np.copy(X_seq_np)
        X_seq_np_scaled[:,:,0] = seq_scaler.transform(flat_seq).reshape(X_seq_np.shape[0], X_seq_np.shape[1])
        try:
            model = train_lstm(X_seq_np_scaled, y_train_np, params={'epochs': 1, 'batch_size': 32, 'verbose': 0})
        except Exception as e:
            print(f"[!] Errore setup LSTM nel worker: {e}")
            return []

    results = []
    print(f"      [{model_type}] Inizio simulazione LIVE su {actual_sims} pacchetti...")
    
    for i in range(start_idx, start_idx + actual_sims):
        lookback_delays = delays[i : i+N]
        lookback_losses = packet_loss[i : i+N]
        
        start_time = time.perf_counter()
        
        if model_type in ['XGBoost', 'MLP_NN']:
            feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
            feats_df = pd.DataFrame([feats])[feature_cols]
            f_scaled = tab_scaler.transform(feats_df)
            _ = model.predict_proba(f_scaled)
        elif model_type == 'LSTM':
            seq_d = np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1)
            seq_d_scaled = seq_scaler.transform(seq_d)
            seq_final = np.column_stack((seq_d_scaled, lookback_losses)).reshape(1, N, 2)
            # Uso chiamata diretta () invece di .predict() per mitigare overhead Keras su batch=1
            _ = model(seq_final, training=False)
            
        end_time = time.perf_counter()
        results.append({'Model': model_type, 'N': N, 'InferenceTime_ms': (end_time - start_time) * 1000.0})
        
        if i % 10000 == 0 and i > start_idx:
            print(f"      [{model_type}] Elaborati {i - start_idx} pacchetti...")
            
    print(f"      [{model_type}] Simulazione completata.")
    return results

def run_realtime_simulation(file_path, n_sizes=[10, 15, 30, 60], X=5, num_simulations=500, output_dir=None):
    print(f"[*] Caricamento file per simulazione live: {file_path}")
    df = pd.read_csv(file_path)
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
        max_possible_sims = len(delays) - start_idx - N + 1
        
        if num_simulations == 'infinity' or num_simulations == float('inf'):
            actual_sims = max_possible_sims
        else:
            actual_sims = min(int(num_simulations), max_possible_sims)
            
        if actual_sims <= 0:
            print(f"  [!] Attenzione: dati insufficienti per avviare la simulazione con N={N}.")
            continue

        print(f"  -> Avvio dei 3 processi paralleli (XGBoost, MLP, LSTM)...")
        print(f"  -> Ciascun processo effettuerà il setup e testerà {actual_sims} pacchetti sul proprio Core isolato.")
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(worker_simulation, 'XGBoost', N, delays, packet_loss, global_max, train_samples, actual_sims, start_idx),
                executor.submit(worker_simulation, 'MLP_NN', N, delays, packet_loss, global_max, train_samples, actual_sims, start_idx),
                executor.submit(worker_simulation, 'LSTM', N, delays, packet_loss, global_max, train_samples, actual_sims, start_idx)
            ]
            
            for future in futures:
                try:
                    res = future.result()
                    all_results.extend(res)
                except Exception as e:
                    print(f"  [!] Errore critico nel processo worker: {e}")

    results_df = pd.DataFrame(all_results)
    
    # Salvataggio Report
    if output_dir and not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'realtime_inference_results.csv'), index=False)
        
    print("\n" + "="*50)
    print("=== REPORT INFERENZA REAL-TIME (Ms) ===")
    if not results_df.empty:
        summary = results_df.groupby(['Model', 'N'])['InferenceTime_ms'].agg(['mean', 'max', lambda x: np.percentile(x, 99)])
        summary.columns = ['Mean (ms)', 'Max (ms)', '99th Pct (ms)']
        print(summary)
        
        print("\n[*] Generazione Grafici...")
        plot_inference_ecdf(results_df, x_thresholds=[0.5], output_dir=output_dir)
        plot_inference_boxplot(results_df, x_thresholds=[0.5], output_dir=output_dir)
        print(f"[*] Tutti i grafici salvati in: {output_dir}")
    else:
        print("[!] Nessun risultato raccolto.")

if __name__ == '__main__':
    # Fix per l'esecuzione del multiprocessing su sistemi Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_realtime_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # File per test
    test_file_to_use = r"dataset/first_capture_window/cpe_a-cpe_b-mobile.csv"
            
    if test_file_to_use and os.path.exists(test_file_to_use):
        run_realtime_simulation(test_file_to_use, n_sizes=[10, 15, 30, 60], X=1, num_simulations='infinity', output_dir=output_dir)
    else:
        print(f"[!] Errore: File dataset '{test_file_to_use}' non trovato per la simulazione.")
