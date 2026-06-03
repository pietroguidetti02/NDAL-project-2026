import argparse
import pandas as pd
import numpy as np
import os
import sys
import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE

# To ensure the src module is found
sys.path.append(os.getcwd())

from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data
from src.features import engineer_features
from src.models import train_xgboost, train_nn, evaluate_model, train_lstm
from src.utils import plot_feature_importance, plot_metrics, plot_model_comparison_3, plot_roc_pr_curves_3

def extract_single_window_all(i, delays, packet_loss, N, X, global_max):
    # Extract traditional tabular features for XGBoost and NN
    lookback_delays = delays[i : i+N]
    lookback_losses = packet_loss[i : i+N]
    feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
    
    # Extract raw sequential time-series for LSTM
    seq = np.column_stack((lookback_delays, lookback_losses))
    
    # Label Extraction
    pred_losses = packet_loss[i+N : i+N+X]
    label = 1 if np.sum(pred_losses) > 0 else 0
    return feats, seq, label

def process_dataset_all(dfs, N, X):
    """
    Processes datasets and returns BOTH tabular features and sequential arrays.
    """
    X_features = []
    X_sequences = []
    y_labels = []
    
    for i, df in enumerate(dfs):
        print(f"    -> Processing file {i+1}/{len(dfs)} with {len(df)} original rows...")
        df_clean = clean_data(df)
        
        global_max = df_clean['delay_ms'].max()
        if pd.isna(global_max):
            global_max = 1000.0
            
        print(f"       Max delay calculated for Link Down penalty: {global_max:.2f} ms")
            
        delays = df_clean['delay_ms'].values
        packet_loss = df_clean['packet_loss'].values
        
        total_required = N + X
        windows_count = len(delays) - total_required + 1
        
        if windows_count <= 0:
            continue
            
        results = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(extract_single_window_all)(j, delays, packet_loss, N, X, global_max)
            for j in range(windows_count)
        )
        
        for feats, seq, label in results:
            X_features.append(feats)
            X_sequences.append(seq)
            y_labels.append(label)
        
    if len(X_features) == 0:
        return pd.DataFrame(), np.array([]), pd.Series()
        
    X_df = pd.DataFrame(X_features)
    X_seq = np.array(X_sequences)
    y_series = pd.Series(y_labels)
    return X_df, X_seq, y_series

def main():
    parser = argparse.ArgumentParser(description='NDAL Project - XGBoost vs NN vs LSTM')
    parser.add_argument('--config', type=str, default='config/exp1.yaml', help='Path to config file')
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_LSTM_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Created results directory: {output_dir} ===")
    
    print(f"[*] Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    N = config.get('N', 15)
    X = config.get('X', 5)
    print(f"[*] Experiment Parameters:")
    print(f"    - Lookback Window (N): {N} seconds")
    print(f"    - Prediction Window (X): {X} seconds")
    
    print("[*] Starting data loading and splitting...")
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    
    tunnel_types = config.get('tunnel_types', ['fiber', 'mobile'])
    
    for tunnel in tunnel_types:
        print(f"\n========================================================")
        print(f"==== Processing Domain: {tunnel.upper()} ====")
        print(f"========================================================")
        
        train_dfs = train_dfs_dict.get(tunnel, [])
        test_dfs = test_dfs_dict.get(tunnel, [])
        
        if not train_dfs or not test_dfs:
            print(f"[!] Warning: No data found for {tunnel}. Skipping...")
            continue
            
        print(f"\n  [*] PHASE 1: Processing TRAINING data for {tunnel}...")
        X_train_df, X_train_seq, y_train = process_dataset_all(train_dfs, N, X)
        
        print(f"\n  [*] PHASE 2: Processing TESTING data for {tunnel}...")
        X_test_df, X_test_seq, y_test = process_dataset_all(test_dfs, N, X)
        
        # --- FEATURE SELECTION for Tabular Models ---
        if tunnel == 'fiber':
            cols_to_drop = ['mean', 'jitter', 'max', 'q95', 'ratio_recent_mean_to_global', 'spikes_over_q95']
        elif tunnel == 'mobile':
            cols_to_drop = ['recent_jitter', 'recent_slope', 'ratio_recent_mean_to_global', 'spikes_over_q95']
        else:
            cols_to_drop = []
            
        if cols_to_drop:
            print(f"  [*] Dropping useless columns for {tunnel}: {cols_to_drop}")
            cols_to_drop_actual = [c for c in cols_to_drop if c in X_train_df.columns]
            X_train_df = X_train_df.drop(columns=cols_to_drop_actual)
            X_test_df = X_test_df.drop(columns=cols_to_drop_actual)
            
        print(f"\n  === Extracted Dataset Summary ({tunnel}) ===")
        print(f"  Samples in Training Set: {len(X_train_df)}")
        print(f"  Samples in Testing Set: {len(X_test_df)}")
        
        if len(X_train_df) == 0 or len(X_test_df) == 0:
            continue
            
        print(f"\n  [*] Applying StandardScaler for Tabular feature normalization...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
        
        print(f"  [*] Applying StandardScaler for Sequential data (LSTM)...")
        # We only scale the delay feature (index 0), packet_loss (index 1) remains binary.
        seq_scaler = StandardScaler()
        
        # Flatten the delays to fit the scaler
        train_delays_flat = X_train_seq[:,:,0].reshape(-1, 1)
        seq_scaler.fit(train_delays_flat)
        
        X_train_seq_scaled = np.copy(X_train_seq)
        X_train_seq_scaled[:,:,0] = seq_scaler.transform(train_delays_flat).reshape(X_train_seq.shape[0], X_train_seq.shape[1])
        
        test_delays_flat = X_test_seq[:,:,0].reshape(-1, 1)
        X_test_seq_scaled = np.copy(X_test_seq)
        X_test_seq_scaled[:,:,0] = seq_scaler.transform(test_delays_flat).reshape(X_test_seq.shape[0], X_test_seq.shape[1])
        
        # FIXING THE CLASS IMBALANCE
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"  [*] Dataset Imbalance -> Negatives: {num_neg}, Positives (Losses): {num_pos}")
        
        print(f"  [*] Applying SMOTE to balance classes for Neural Network...")
        if num_pos > 5:
            smote = SMOTE(random_state=42)
            X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_train_scaled_resampled, y_train_resampled = X_train_scaled, y_train
            
        # ================== TRAINING MODELS ==================
        print(f'\n  [*] --- Training 1/3: XGBoost Model ({tunnel}) ---')
        xgb_model = train_xgboost(X_train_df, y_train, params={
            'use_label_encoder': False, 
            'eval_metric': 'logloss', 
            'n_jobs': -1,
            'scale_pos_weight': scale_weight
        })
        xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test, threshold=0.05)
        plot_metrics(xgb_metrics, model_name=f'{tunnel}_XGBoost', output_dir=output_dir)
        
        print(f'\n  [*] --- Training 2/3: LSTM Model (Deep Learning sequence) ({tunnel}) ---')
        try:
            # We don't use SMOTE for LSTM because synthesizing sequences is hard,
            # we rely on class_weight inside train_lstm
            lstm_model = train_lstm(X_train_seq_scaled, y_train, params={'epochs': 15, 'batch_size': 256})
            lstm_metrics = evaluate_model(lstm_model, X_test_seq_scaled, y_test, threshold=0.5)
            plot_metrics(lstm_metrics, model_name=f'{tunnel}_LSTM', output_dir=output_dir)
        except Exception as e:
            print(f"[!] Error training LSTM: {e}")
            lstm_metrics = None
            
        print(f'\n  [*] --- Training 3/3: Neural Network Model (MLP) ({tunnel}) ---')
        nn_model = train_nn(X_train_scaled_resampled, y_train_resampled, params={'max_iter': 500, 'random_state': 42})
        nn_metrics = evaluate_model(nn_model, X_test_scaled, y_test)
        plot_metrics(nn_metrics, model_name=f'{tunnel}_NN', output_dir=output_dir)
        
        # ================== COMPARISON ==================
        if lstm_metrics is not None:
            print(f'\n  [*] Generating 3-Way Comparison Plots ({tunnel})...')
            plot_model_comparison_3(xgb_metrics, nn_metrics, lstm_metrics, 
                                    m1_name='XGBoost', m2_name='MLP_NN', m3_name='LSTM', 
                                    output_dir=output_dir, prefix=tunnel)
            plot_roc_pr_curves_3(xgb_metrics, nn_metrics, lstm_metrics, 
                                 m1_name='XGBoost', m2_name='MLP_NN', m3_name='LSTM', 
                                 output_dir=output_dir, prefix=tunnel)
        else:
            print(f'\n  [*] Skipping 3-Way plot due to LSTM error.')

    print(f'\n[*] Pipeline execution completed! All plots and metrics saved in: {output_dir}')

if __name__ == '__main__':
    main()
