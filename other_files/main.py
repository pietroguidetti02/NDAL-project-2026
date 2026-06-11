import argparse
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data
from src.features import engineer_features
from src.models import train_xgboost, train_nn, evaluate_model, optimize_threshold_cv
from src.utils import plot_feature_importance, plot_metrics, plot_model_comparison, plot_roc_pr_curves_2

def extract_single_window(i, delays, packet_loss, N, X, global_max):
    lookback_delays = delays[i : i+N]
    lookback_losses = packet_loss[i : i+N]
    pred_losses = packet_loss[i+N : i+N+X]
    label = 1 if np.sum(pred_losses) > 0 else 0
    feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
    return feats, label

def process_dataset(dfs, N, X):
    """
    Given a list of dataframes, clean, extract windows via NumPy, and build feature/label arrays.
    """
    X_features = []
    y_labels = []
    
    for i, df in enumerate(dfs):
        print(f"    -> Processing file {i+1}/{len(dfs)} with {len(df)} original rows...")
        # Clean data
        df_clean = clean_data(df)
        
        # Determine global max delay for this specific file to use as Link Down penalty
        global_max = df_clean['delay_ms'].max()
        if pd.isna(global_max):
            global_max = 1000.0
            
        print(f"       Max delay calculated for Link Down penalty: {global_max:.2f} ms")
            
        # Extract features and windows completely in RAM-safe NumPy
        delays = df_clean['delay_ms'].values
        packet_loss = df_clean['packet_loss'].values
        
        total_required = N + X
        windows_count = len(delays) - total_required + 1
        
        if windows_count <= 0:
            continue
            
        print(f"       Extracting and computing features for {windows_count} windows via Parallel NumPy...")
        
        results = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(extract_single_window)(j, delays, packet_loss, N, X, global_max)
            for j in range(windows_count)
        )
        
        for feats, label in results:
            X_features.append(feats)
            y_labels.append(label)
        
    if len(X_features) == 0:
        return pd.DataFrame(), pd.Series()
        
    X_df = pd.DataFrame(X_features)
    y_series = pd.Series(y_labels)
    return X_df, y_series

def main():
    parser = argparse.ArgumentParser(description='NDAL Project 2 - Main Orchestrator')
    parser.add_argument('--config', type=str, default='config/exp1.yaml', help='Path to config file')
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_{timestamp}")
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
    
    tunnel_types = config.get('tunnel_types', ['mobile', 'fiber'])
    
    for tunnel in tunnel_types:
        print(f"\n========================================================")
        print(f"==== Processing Domain: {tunnel.upper()} ====")
        print(f"========================================================")
        
        train_dfs = train_dfs_dict.get(tunnel, [])
        test_dfs = test_dfs_dict.get(tunnel, [])
        
        if not train_dfs or not test_dfs:
            print(f"[!] Warning: No data found for {tunnel}. Skipping...")
            continue
            
        print(f"  [*] Found {len(train_dfs)} TRAINING splits for {tunnel}.")
        print(f"  [*] Found {len(test_dfs)} TESTING splits for {tunnel}.")
            
        print(f"\n  [*] PHASE 1: Processing TRAINING data for {tunnel}...")
        X_train, y_train = process_dataset(train_dfs, N, X)
        
        print(f"\n  [*] PHASE 2: Processing TESTING data for {tunnel}...")
        X_test, y_test = process_dataset(test_dfs, N, X)
        
        # --- FEATURE SELECTION based on analysis reports ---
        if tunnel == 'fiber':
            cols_to_drop = ['mean', 'jitter', 
                            'max', 'q95', 
                            'ratio_recent_mean_to_global', 
                            'spikes_over_q95']
        elif tunnel == 'mobile':
            cols_to_drop = ['recent_jitter', 'recent_slope', 
                            'ratio_recent_mean_to_global', 
                            'spikes_over_q95']
        else:
            cols_to_drop = []
            
        if cols_to_drop:
            print(f"  [*] Dropping useless columns for {tunnel}: {cols_to_drop}")
            cols_to_drop_actual = [c for c in cols_to_drop if c in X_train.columns]
            X_train = X_train.drop(columns=cols_to_drop_actual)
            X_test = X_test.drop(columns=cols_to_drop_actual)
        # ---------------------------------------------------
        
        print(f"\n  === Extracted Dataset Summary ({tunnel}) ===")
        print(f"  Samples in Training Set: {len(X_train)}")
        print(f"  Samples in Testing Set: {len(X_test)}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  [!] Insufficient data for training {tunnel} models.")
            continue
            
        print(f"\n  [*] Applying StandardScaler for feature normalization...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # FIXING THE CLASS IMBALANCE (ZERO LOSS ISSUE)
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"  [*] Dataset Imbalance -> Negatives: {num_neg}, Positives (Losses): {num_pos}")
        print(f"      Calculated scale_pos_weight for XGBoost: {scale_weight:.2f}")
        
        print(f"  [*] Applying Hybrid Resampling (RandomUnderSampler + SMOTE)...")
        if num_pos > 5:
            from imblearn.under_sampling import RandomUnderSampler
            
            target_neg = max(10000, num_pos)
            target_neg = min(target_neg, num_neg)
            
            rus = RandomUnderSampler(sampling_strategy={0: target_neg, 1: num_pos}, random_state=42)
            smote = SMOTE(sampling_strategy={0: target_neg, 1: target_neg}, random_state=42)
            
            X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
            X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_rus, y_train_rus)
            print(f"      After Hybrid Resampling -> Negatives: {(y_train_resampled == 0).sum()}, Positives: {(y_train_resampled == 1).sum()}")
        else:
            X_train_scaled_resampled, y_train_resampled = X_train_scaled, y_train
            print("      Not enough positive samples for resampling. Using original data.")
            
        print(f'\n  [*] --- Training XGBoost Model ({tunnel}) ---')
        import xgboost as xgb
        xgb_params = {'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1}
        xgb_base = xgb.XGBClassifier(**xgb_params)
        xgb_thresh = optimize_threshold_cv(xgb_base, X_train_scaled_resampled, y_train_resampled, cv=3)
        xgb_model = train_xgboost(X_train_scaled_resampled, y_train_resampled, params=xgb_params)
        print('      Training completed.')
        
        print(f'\n  [*] Evaluating XGBoost Model ({tunnel})...')
        xgb_metrics = evaluate_model(xgb_model, X_test_scaled, y_test, threshold=xgb_thresh)
        plot_metrics(xgb_metrics, model_name=f'{tunnel}_XGBoost', output_dir=output_dir)
        plot_feature_importance(xgb_model, output_dir=output_dir)
        
        print(f'\n  [*] --- Training Neural Network Model (MLP) ({tunnel}) ---')
        from sklearn.neural_network import MLPClassifier
        nn_params = {'max_iter': 500, 'random_state': 42}
        nn_base = MLPClassifier(**nn_params)
        nn_thresh = optimize_threshold_cv(nn_base, X_train_scaled_resampled, y_train_resampled, cv=3)
        nn_model = train_nn(X_train_scaled_resampled, y_train_resampled, params=nn_params)
        print('      Training completed.')
        
        print(f'\n  [*] Evaluating Neural Network Model ({tunnel})...')
        nn_metrics = evaluate_model(nn_model, X_test_scaled, y_test, threshold=nn_thresh)
        plot_metrics(nn_metrics, model_name=f'{tunnel}_Neural_Network', output_dir=output_dir)
        
        print(f'\n  [*] Generating Comparison Plots ({tunnel})...')
        plot_model_comparison(xgb_metrics, nn_metrics, model1_name='XGBoost', model2_name='NN', output_dir=output_dir, prefix=tunnel)
        plot_roc_pr_curves_2(xgb_metrics, nn_metrics, m1_name='XGBoost', m2_name='NN', output_dir=output_dir, prefix=tunnel)

    print(f'\n[*] Pipeline execution completed! All plots and metrics saved in: {output_dir}')

if __name__ == '__main__':
    main()
