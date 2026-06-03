import argparse
import pandas as pd
import numpy as np
import os
import sys
import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
import yaml

# To ensure the src module is found
sys.path.append(os.getcwd())

from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data
from src.features import engineer_features
from src.models import train_xgboost, train_nn, evaluate_model, train_lstm, optimize_threshold_cv
from src.utils import plot_feature_importance, plot_metrics, plot_model_comparison_3, plot_roc_pr_curves_3

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def extract_single_window_all(i, delays, packet_loss, N, X, global_max):
    # Extract traditional tabular features for XGBoost and NN
    lookback_delays = delays[i : i+N]
    lookback_losses = packet_loss[i : i+N]
    feats = engineer_features(lookback_delays, lookback_losses, global_max_delay=global_max)
    
    # Extract raw sequential time-series for LSTM
    seq_delays = np.nan_to_num(lookback_delays, nan=global_max)
    seq = np.column_stack((seq_delays, lookback_losses))
    
    # Label Extraction
    pred_losses = packet_loss[i+N : i+N+X]
    label = 1 if np.sum(pred_losses) > 0 else 0
    return feats, seq, label

def process_dataset_all(dfs, N, X, n_jobs=1):
    """
    Processes datasets and returns BOTH tabular features and sequential arrays.
    """
    X_features = []
    X_sequences = []
    y_labels = []
    
    for i, df in enumerate(dfs):
        # Reduced logging for sweep to avoid cluttered output
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
            
        results = Parallel(n_jobs=n_jobs)(
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

def run_single_experiment(N, X, config, base_output_dir, train_dfs_dict, test_dfs_dict):
    """
    Runs a single comparison experiment for a specific (N, X) pair.
    """
    exp_dir = os.path.join(base_output_dir, f"N_{N}_X_{X}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n[Run N={N}, X={X}] Starting. Results will be saved in: {exp_dir}")
    
    tunnel_types = config.get('tunnel_types', ['mobile', 'fiber'])
    
    for tunnel in tunnel_types:
        print(f"\n  [N={N}, X={X}] === Processing Domain: {tunnel.upper()} ===")
        train_dfs = train_dfs_dict.get(tunnel, [])
        test_dfs = test_dfs_dict.get(tunnel, [])
        
        if not train_dfs or not test_dfs:
            print(f"  [N={N}, X={X}] [!] Warning: No data found for {tunnel}. Skipping...")
            continue
            
        # PHASE 1 & 2: Data Processing
        print(f"  [N={N}, X={X}] [*] PHASE 1: Processing TRAINING data for {tunnel}...")
        X_train_df, X_train_seq, y_train = process_dataset_all(train_dfs, N, X, n_jobs=1)
        print(f"  [N={N}, X={X}] [*] PHASE 2: Processing TESTING data for {tunnel}...")
        X_test_df, X_test_seq, y_test = process_dataset_all(test_dfs, N, X, n_jobs=1)
        
        if len(X_train_df) == 0 or len(X_test_df) == 0:
            print(f"  [N={N}, X={X}] [!] Insufficient data for {tunnel}. Skipping...")
            continue
            
        # FEATURE SELECTION
        if tunnel == 'fiber':
            cols_to_drop = ['mean', 'jitter', 'max', 'q95', 'ratio_recent_mean_to_global', 'spikes_over_q95']
        elif tunnel == 'mobile':
            cols_to_drop = ['recent_jitter', 'recent_slope', 'ratio_recent_mean_to_global', 'spikes_over_q95']
        else:
            cols_to_drop = []
            
        if cols_to_drop:
            print(f"  [N={N}, X={X}] [*] Dropping useless columns for {tunnel}: {cols_to_drop}")
            cols_to_drop_actual = [c for c in cols_to_drop if c in X_train_df.columns]
            X_train_df = X_train_df.drop(columns=cols_to_drop_actual)
            X_test_df = X_test_df.drop(columns=cols_to_drop_actual)
            
        # SCALING
        print(f"  [N={N}, X={X}] [*] Applying StandardScalers...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
        
        seq_scaler = StandardScaler()
        train_delays_flat = X_train_seq[:,:,0].reshape(-1, 1)
        seq_scaler.fit(train_delays_flat)
        X_train_seq_scaled = np.copy(X_train_seq)
        X_train_seq_scaled[:,:,0] = seq_scaler.transform(train_delays_flat).reshape(X_train_seq.shape[0], X_train_seq.shape[1])
        
        test_delays_flat = X_test_seq[:,:,0].reshape(-1, 1)
        X_test_seq_scaled = np.copy(X_test_seq)
        X_test_seq_scaled[:,:,0] = seq_scaler.transform(test_delays_flat).reshape(X_test_seq.shape[0], X_test_seq.shape[1])
        
        # RESAMPLING
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        print(f"  [N={N}, X={X}] [*] Dataset Imbalance -> Negatives: {num_neg}, Positives (Losses): {num_pos}")
        
        if num_pos > 5:
            print(f"  [N={N}, X={X}] [*] Applying Hybrid Resampling (RandomUnderSampler + SMOTE)...")
            from imblearn.under_sampling import RandomUnderSampler
            target_neg = max(10000, num_pos)
            target_neg = min(target_neg, num_neg)
            rus = RandomUnderSampler(sampling_strategy={0: target_neg, 1: num_pos}, random_state=42)
            smote = SMOTE(sampling_strategy={0: target_neg, 1: target_neg}, random_state=42)
            X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
            X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_rus, y_train_rus)
        else:
            X_train_scaled_resampled, y_train_resampled = X_train_scaled, y_train

        # TRAINING MODELS
        # 1/3 XGBoost
        print(f"  [N={N}, X={X}] [*] --- Training 1/3: XGBoost Model ({tunnel}) ---")
        import xgboost as xgb
        xgb_params = {'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': 1}
        xgb_base = xgb.XGBClassifier(**xgb_params)
        xgb_thresh = optimize_threshold_cv(xgb_base, X_train_scaled_resampled, y_train_resampled, cv=3)
        xgb_model = train_xgboost(X_train_scaled_resampled, y_train_resampled, params=xgb_params)
        xgb_metrics = evaluate_model(xgb_model, X_test_scaled, y_test, threshold=xgb_thresh)
        plot_metrics(xgb_metrics, model_name=f'{tunnel}_XGBoost', output_dir=exp_dir)
        # Adding feature importance as it's often expected even if not in LSTM script
        plot_feature_importance(xgb_model, output_dir=exp_dir)
        
        # 2/3 LSTM
        print(f"  [N={N}, X={X}] [*] --- Training 2/3: LSTM Model ({tunnel}) ---")
        try:
            lstm_model = train_lstm(X_train_seq_scaled, y_train, params={'epochs': 15, 'batch_size': 256, 'verbose': 0})
            lstm_metrics = evaluate_model(lstm_model, X_test_seq_scaled, y_test, threshold=0.5)
            plot_metrics(lstm_metrics, model_name=f'{tunnel}_LSTM', output_dir=exp_dir)
        except Exception as e:
            print(f"  [N={N}, X={X}] [!] LSTM Error: {e}")
            lstm_metrics = None
            
        # 3/3 Neural Network (MLP)
        print(f"  [N={N}, X={X}] [*] --- Training 3/3: Neural Network Model ({tunnel}) ---")
        from sklearn.neural_network import MLPClassifier
        nn_params = {'max_iter': 500, 'random_state': 42}
        nn_base = MLPClassifier(**nn_params)
        nn_thresh = optimize_threshold_cv(nn_base, X_train_scaled_resampled, y_train_resampled, cv=3)
        nn_model = train_nn(X_train_scaled_resampled, y_train_resampled, params=nn_params)
        nn_metrics = evaluate_model(nn_model, X_test_scaled, y_test, threshold=nn_thresh)
        plot_metrics(nn_metrics, model_name=f'{tunnel}_NN', output_dir=exp_dir)
        
        # COMPARISON
        if lstm_metrics is not None:
            print(f"  [N={N}, X={X}] [*] Generating 3-Way Comparison Plots ({tunnel})...")
            plot_model_comparison_3(xgb_metrics, nn_metrics, lstm_metrics, 
                                    m1_name='XGBoost', m2_name='MLP_NN', m3_name='LSTM', 
                                    output_dir=exp_dir, prefix=tunnel)
            plot_roc_pr_curves_3(xgb_metrics, nn_metrics, lstm_metrics, 
                                 m1_name='XGBoost', m2_name='MLP_NN', m3_name='LSTM', 
                                 output_dir=exp_dir, prefix=tunnel)
            
    print(f"[Run N={N}, X={X}] Completed.")
    return True

def main():
    parser = argparse.ArgumentParser(description='NDAL Project - Parameter Sweep (N, X)')
    parser.add_argument('--config', type=str, default='config/sweep_time_split.yaml', help='Path to sweep config file')
    parser.add_argument('--max_cores', type=int, default=12, help='Maximum number of concurrent experiments')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    N_values = config.get('N_values', [10, 15, 20, 30, 60])
    X_values = config.get('X_values', [1, 5, 10, 15, 20])
    
    # Generate valid (N, X) pairs where X < N
    combinations = [(n, x) for n in N_values for x in X_values if x < n]
    
    print(f"[*] Starting Sweep with {len(combinations)} valid combinations.")
    print(f"[*] Max concurrent experiments: {args.max_cores}")
    
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("results", f"sweep_{config_name}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"[*] Results will be saved in: {base_output_dir}")
    
    # Setup logging to file
    log_file_path = os.path.join(base_output_dir, "sweep_log.txt")
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout
    
    print(f"[*] Logging started at {timestamp}")
    print(f"[*] Command: {' '.join(sys.argv)}")
    
    print("[*] Loading raw data once...")
    # We load raw data once and share it to save memory/time
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    
    # Run the sweep in parallel
    Parallel(n_jobs=args.max_cores)(
        delayed(run_single_experiment)(n, x, config, base_output_dir, train_dfs_dict, test_dfs_dict)
        for n, x in combinations
    )
    
    print(f"\n[*] All experiments in sweep completed! Total combinations: {len(combinations)}")
    print(f"[*] Check {base_output_dir} for results.")

if __name__ == '__main__':
    main()
