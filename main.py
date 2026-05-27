import argparse
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data, extract_sliding_windows
from src.features import engineer_features
from src.models import train_xgboost, train_nn, evaluate_model
from src.utils import plot_feature_importance, plot_metrics

def process_dataset(dfs, N, X):
    """
    Given a list of dataframes, clean, extract windows, and build feature/label arrays in parallel.
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
            
        # Extract windows
        windows, labels = extract_sliding_windows(df_clean, N, X)
        print(f"       Extracted {len(windows)} sliding windows. Engineering features in parallel...")
        
        # Engineer features for each window
        for window in windows:
            feats = engineer_features(window, global_max_delay=global_max)
            X_features.append(feats)
            
        y_labels.extend(labels)
        
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
    
    tunnel_types = ['fiber', 'mobile']
    
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
            
        print(f'\n  [*] --- Training XGBoost Model ({tunnel}) ---')
        # n_jobs=-1 enables parallel threading inside XGBoost
        xgb_model = train_xgboost(X_train, y_train, params={'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1})
        print('      Training completed.')
        
        print(f'\n  [*] Evaluating XGBoost Model ({tunnel})...')
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        plot_metrics(xgb_metrics, model_name=f'{tunnel}_XGBoost', output_dir=output_dir)
        plot_feature_importance(xgb_model, output_dir=output_dir)
        
        print(f'\n  [*] --- Training Neural Network Model (MLP) ({tunnel}) ---')
        nn_model = train_nn(X_train_scaled, y_train, params={'max_iter': 500, 'random_state': 42})
        print('      Training completed.')
        
        print(f'\n  [*] Evaluating Neural Network Model ({tunnel})...')
        nn_metrics = evaluate_model(nn_model, X_test_scaled, y_test)
        plot_metrics(nn_metrics, model_name=f'{tunnel}_Neural_Network', output_dir=output_dir)

    print(f'\n[*] Pipeline execution completed! All plots and metrics saved in: {output_dir}')

if __name__ == '__main__':
    main()
