import argparse
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data
from src.features_advanced import engineer_statistical_features, engineer_raw_sequence_features
from src.models import train_xgboost, train_nn, evaluate_model
from src.utils import plot_feature_importance, plot_metrics

def extract_single_window_both(i, delays, packet_loss, N, X, global_max):
    lookback_delays = delays[i : i+N]
    lookback_losses = packet_loss[i : i+N]
    pred_losses = packet_loss[i+N : i+N+X]
    label = 1 if np.sum(pred_losses) > 0 else 0
    
    stats_feats = engineer_statistical_features(lookback_delays, lookback_losses, global_max_delay=global_max)
    raw_feats = engineer_raw_sequence_features(lookback_delays, global_max_delay=global_max)
    return stats_feats, raw_feats, label

def process_dataset_4comp(dfs, N, X):
    X_stats = []
    X_raw = []
    y_labels = []
    
    for i, df in enumerate(dfs):
        print(f"    -> Processing file {i+1}/{len(dfs)} with {len(df)} original rows...")
        df_clean = clean_data(df)
        
        global_max = df_clean['delay_ms'].max()
        if pd.isna(global_max):
            global_max = 1000.0
            
        delays = df_clean['delay_ms'].values
        packet_loss = df_clean['packet_loss'].values
        
        total_required = N + X
        windows_count = len(delays) - total_required + 1
        
        if windows_count <= 0:
            continue
            
        results = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(extract_single_window_both)(j, delays, packet_loss, N, X, global_max)
            for j in range(windows_count)
        )
        
        for sf, rf, label in results:
            X_stats.append(sf)
            X_raw.append(rf)
            y_labels.append(label)
        
    if len(X_stats) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
    return pd.DataFrame(X_stats), pd.DataFrame(X_raw), pd.Series(y_labels)

def main():
    parser = argparse.ArgumentParser(description='NDAL Project 2 - 4-Way Comparison')
    parser.add_argument('--config', type=str, default='config/exp1.yaml', help='Path to config file')
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"exp_4comp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Created results directory: {output_dir} ===")
    
    config = load_config(args.config)
    N = config.get('N', 15)
    X = config.get('X', 5)
    
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    tunnel_types = config.get('tunnel_types', ['mobile', 'fiber'])
    
    for tunnel in tunnel_types:
        print(f"\n========================================================")
        print(f"==== Processing Domain: {tunnel.upper()} ====")
        print(f"========================================================")
        
        train_dfs = train_dfs_dict.get(tunnel, [])
        test_dfs = test_dfs_dict.get(tunnel, [])
        
        if not train_dfs or not test_dfs:
            continue
            
        print(f"\n  [*] PHASE 1: Processing TRAINING data for {tunnel}...")
        X_train_stats, X_train_raw, y_train = process_dataset_4comp(train_dfs, N, X)
        
        print(f"\n  [*] PHASE 2: Processing TESTING data for {tunnel}...")
        X_test_stats, X_test_raw, y_test = process_dataset_4comp(test_dfs, N, X)
        
        if len(y_train) == 0 or len(y_test) == 0:
            continue
            
        # Scaling
        print("\n  [*] Scaling features...")
        scaler_stats = StandardScaler()
        X_train_stats_scaled = pd.DataFrame(scaler_stats.fit_transform(X_train_stats), columns=X_train_stats.columns)
        X_test_stats_scaled = pd.DataFrame(scaler_stats.transform(X_test_stats), columns=X_test_stats.columns)
        
        scaler_raw = StandardScaler()
        X_train_raw_scaled = pd.DataFrame(scaler_raw.fit_transform(X_train_raw), columns=X_train_raw.columns)
        X_test_raw_scaled = pd.DataFrame(scaler_raw.transform(X_test_raw), columns=X_test_raw.columns)
        
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        # 1. XGBoost Statistical
        print(f'\n  [1/4] Training XGBoost (Statistical) - {tunnel}')
        xgb_stats = train_xgboost(X_train_stats, y_train, params={'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1, 'scale_pos_weight': scale_weight})
        xgb_stats_metrics = evaluate_model(xgb_stats, X_test_stats, y_test, threshold=0.10)
        plot_metrics(xgb_stats_metrics, model_name=f'{tunnel}_XGBoost_Stats', output_dir=output_dir)
        plot_feature_importance(xgb_stats, output_dir=output_dir)
        
        # 2. XGBoost RawSequence
        print(f'\n  [2/4] Training XGBoost (Raw Sequence) - {tunnel}')
        xgb_raw = train_xgboost(X_train_raw, y_train, params={'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1, 'scale_pos_weight': scale_weight})
        xgb_raw_metrics = evaluate_model(xgb_raw, X_test_raw, y_test, threshold=0.10)
        plot_metrics(xgb_raw_metrics, model_name=f'{tunnel}_XGBoost_Raw', output_dir=output_dir)
        plot_feature_importance(xgb_raw, output_dir=output_dir)
        
        # 3. NN Statistical
        print(f'\n  [3/4] Training Neural Network (Statistical) - {tunnel}')
        nn_stats = train_nn(X_train_stats_scaled, y_train, params={'max_iter': 500, 'random_state': 42})
        nn_stats_metrics = evaluate_model(nn_stats, X_test_stats_scaled, y_test)
        plot_metrics(nn_stats_metrics, model_name=f'{tunnel}_NN_Stats', output_dir=output_dir)
        
        # 4. NN RawSequence
        print(f'\n  [4/4] Training Neural Network (Raw Sequence) - {tunnel}')
        nn_raw = train_nn(X_train_raw_scaled, y_train, params={'max_iter': 500, 'random_state': 42})
        nn_raw_metrics = evaluate_model(nn_raw, X_test_raw_scaled, y_test)
        plot_metrics(nn_raw_metrics, model_name=f'{tunnel}_NN_Raw', output_dir=output_dir)

if __name__ == '__main__':
    main()
