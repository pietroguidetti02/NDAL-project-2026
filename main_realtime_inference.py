import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import threading

sys.path.append(os.getcwd())
from src.preprocessor import clean_data
from src.features import engineer_features
from src.models import train_xgboost, train_nn, train_lstm
from src.utils import plot_inference_ecdf, plot_inference_boxplot

def run_realtime_simulation(file_path, n_sizes=[10, 15, 30, 60], X=5, num_simulations=500, output_dir=None):
    print(f"[*] Caricamento file per simulazione live: {file_path}")
    df = pd.read_csv(file_path)
    df_clean = clean_data(df)
    
    global_max = df_clean['delay_ms'].max()
    if pd.isna(global_max): global_max = 1000.0
        
    delays = df_clean['delay_ms'].values
    packet_loss = df_clean['packet_loss'].values
    
    all_results = []
    
    for N in n_sizes:
        print(f"\n{'='*50}\n[*] Avvio Pipeline Simulazione per N={N}...\n{'='*50}")
        
        # 1. Preparazione rapida di un mini-dataset per inizializzare i modelli
        # (Ci servono modelli compilati per misurare il tempo di inferenza, l'accuracy qui non conta)
        print("  -> Generazione mini-dataset per setup modelli...")
        train_samples = 200 # Pochi campioni solo per fare la fit() veloce
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
            # Label random per far funzionare i classificatori se mancano positivi
            y_train.append(1 if j % 10 == 0 else 0)
            
        X_tab_df = pd.DataFrame(X_tab)
        X_seq_np = np.array(X_seq)
        y_train_np = np.array(y_train)
        
        tab_scaler = StandardScaler()
        X_tab_scaled = tab_scaler.fit_transform(X_tab_df)
        
        seq_scaler = StandardScaler()
        flat_seq = X_seq_np[:,:,0].reshape(-1, 1)
        seq_scaler.fit(flat_seq)
        X_seq_np_scaled = np.copy(X_seq_np)
        X_seq_np_scaled[:,:,0] = seq_scaler.transform(flat_seq).reshape(X_seq_np.shape[0], X_seq_np.shape[1])
        
        # 2. Addestramento "Fantasma" per inizializzare le reti
        print("  -> Inizializzazione modelli (XGBoost, MLP, LSTM)...")
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
        xgb_model.fit(X_tab_scaled, y_train_np)
        
        from sklearn.neural_network import MLPClassifier
        nn_model = MLPClassifier(max_iter=1, random_state=42)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn_model.fit(X_tab_scaled, y_train_np)
            
        try:
            # Pochi epoch per avere i pesi in memoria
            lstm_model = train_lstm(X_seq_np_scaled, y_train_np, params={'epochs': 1, 'batch_size': 32})
        except Exception as e:
            print(f"[!] Errore setup LSTM: {e}")
            lstm_model = None
            
        # 3. IL VERO TEST DI INFERENZA (Cronometrato)
        print(f"  -> Inizio simulazione LIVE su {num_simulations} pacchetti...")
        
        start_idx = train_samples + 100
        for i in range(start_idx, start_idx + num_simulations):
            # Simuliamo l'arrivo dei dati al router in questo esatto millisecondo
            lookback_delays = delays[i : i+N]
            lookback_losses = packet_loss[i : i+N]
            
            results_this_iter = []

            # --- MISURAZIONE XGBOOST ---
            def run_xgb():
                start_xgb = time.perf_counter()
                feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
                # Dobbiamo assicurarci che l'ordine delle colonne sia identico, passo il dataframe con le stesse colonne
                feats_df = pd.DataFrame([feats])[X_tab_df.columns]
                f_scaled = tab_scaler.transform(feats_df)
                _ = xgb_model.predict_proba(f_scaled)
                end_xgb = time.perf_counter()
                results_this_iter.append({'Model': 'XGBoost', 'N': N, 'InferenceTime_ms': (end_xgb - start_xgb) * 1000.0})
            
            # --- MISURAZIONE MLP ---
            def run_nn():
                start_nn = time.perf_counter()
                # MLP usa le stesse features, ma nella realtà il router avrebbe già fatto `engineer_features`. 
                # Per correttezza, se girano in parallelo sul router, ognuno fa la sua estrazione (o la condividono).
                # Assumiamo che la condivisione delle features avvenga e cronometriamo solo l'inferenza per MLP.
                # Se vogliamo essere cattivi cronometriamo anche l'estrazione:
                feats_nn = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
                f_nn_scaled = tab_scaler.transform(pd.DataFrame([feats_nn])[X_tab_df.columns])
                _ = nn_model.predict_proba(f_nn_scaled)
                end_nn = time.perf_counter()
                results_this_iter.append({'Model': 'MLP_NN', 'N': N, 'InferenceTime_ms': (end_nn - start_nn) * 1000.0})
            
            # --- MISURAZIONE LSTM ---
            def run_lstm():
                if lstm_model is not None:
                    start_lstm = time.perf_counter()
                    seq_d = np.nan_to_num(lookback_delays, nan=global_max).reshape(-1, 1)
                    seq_d_scaled = seq_scaler.transform(seq_d)
                    seq_final = np.column_stack((seq_d_scaled, lookback_losses)).reshape(1, N, 2)
                    _ = lstm_model.predict(seq_final, verbose=0)
                    end_lstm = time.perf_counter()
                    results_this_iter.append({'Model': 'LSTM', 'N': N, 'InferenceTime_ms': (end_lstm - start_lstm) * 1000.0})

            t_xgb = threading.Thread(target=run_xgb)
            t_nn = threading.Thread(target=run_nn)
            t_lstm = threading.Thread(target=run_lstm)

            t_xgb.start()
            t_nn.start()
            t_lstm.start()

            t_xgb.join()
            t_nn.join()
            t_lstm.join()

            all_results.extend(results_this_iter)


    results_df = pd.DataFrame(all_results)
    
    # Salvataggio Report
    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'realtime_inference_results.csv'), index=False)
        
    print("\n" + "="*50)
    print("=== REPORT INFERENZA REAL-TIME (Ms) ===")
    summary = results_df.groupby(['Model', 'N'])['InferenceTime_ms'].agg(['mean', 'max', lambda x: np.percentile(x, 99)])
    summary.columns = ['Mean (ms)', 'Max (ms)', '99th Pct (ms)']
    print(summary)
    
    print("\n[*] Generazione Grafici...")
    # Plot con linee a 0.5s, 1s, 5s (in base alle esigenze di routing)
    plot_inference_ecdf(results_df, x_thresholds=[0.5, 1.0, 5.0], output_dir=output_dir)
    plot_inference_boxplot(results_df, x_thresholds=[0.5, 1.0, 5.0], output_dir=output_dir)
    
    print(f"[*] Tutti i grafici salvati in: {output_dir}")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_realtime_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # CAMBIA QUESTO PERCORSO MANUALMENTE PER TESTARE ALTRI FILE
    # =========================================================================
    test_file_to_use = r"dataset/first_capture_window/cpe_a-cpe_b-mobile.csv"
            
    if test_file_to_use and os.path.exists(test_file_to_use):
        run_realtime_simulation(test_file_to_use, n_sizes=[10, 15, 30, 60], num_simulations=300, output_dir=output_dir)
    else:
        print(f"[!] Errore: File dataset '{test_file_to_use}' non trovato per la simulazione.")
