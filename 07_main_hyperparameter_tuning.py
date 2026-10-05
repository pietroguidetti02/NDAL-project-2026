import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())
from src.data_loader import load_config, load_and_split_data
from src.models import evaluate_model
from src.utils import plot_roc_pr_curves_2

import importlib.util
spec = importlib.util.spec_from_file_location("main_comp", "02_b_main_comparison_LSTM.py")
main_comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_comp)
process_dataset_all = main_comp.process_dataset_all

def plot_search_results(cv_results, base_f1, model_name, output_dir):
    """
    Plots the variance of the F1-Score across the different combinations tried
    by RandomizedSearchCV, and draws a red line for the Baseline model.
    """
    scores = cv_results['mean_test_score']
    scores = [s for s in scores if not np.isnan(s)]
    scores = sorted(scores)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='dodgerblue', linewidth=2, label='RandomizedSearch trials')
    plt.fill_between(range(1, len(scores) + 1), scores, min(scores)*0.95, alpha=0.2, color='dodgerblue')

    # Add the red line for the Baseline model
    plt.axhline(y=base_f1, color='crimson', linestyle='--', linewidth=2.5, label=f'Baseline')

    plt.title(f'{model_name}: F1-Score Variance vs Baseline', fontsize=14)
    plt.xlabel('Trial (sorted from worst to best)', fontsize=12)
    plt.ylabel('Mean Cross-Validated F1-Score', fontsize=12)
    plt.ylim([max(0, min(min(scores), base_f1)*0.9), min(1.0, max(max(scores), base_f1)*1.05)])
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_tuning_variance.png'))
    plt.close()

def tune_xgboost(X_train, y_train, X_test, y_test, output_dir):
    print("\n" + "="*50)
    print("=== XGBoost Tuning (Randomized Search) ===")
    print("="*50)
    
    print("[*] Training Baseline Model (Default)...")
    baseline_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    baseline_model.fit(X_train, y_train)
    metrics_base = evaluate_model(baseline_model, X_test, y_test, threshold=0.5)
    base_f1 = metrics_base.get('f1', 0)
    print(f"[*] BASELINE F1-Score: {base_f1:.4f}")
    
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.6, 0.8, 1.0]
    }
    xgb_cv = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb_cv, param_distributions=param_grid, 
        n_iter=15, scoring='f1', cv=3, verbose=3, random_state=42, n_jobs=-1
    )
    
    print("\n[*] Starting RandomizedSearchCV with 15 combinations x 3 Folds...")
    start_time = time.perf_counter()
    random_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time
    
    best_model = random_search.best_estimator_
    metrics_tuned = evaluate_model(best_model, X_test, y_test, threshold=0.5)
    tuned_f1 = metrics_tuned.get('f1', 0)
    
    print(f"\n[*] SEARCH COMPLETED IN {elapsed:.1f} seconds.")
    print(f"[*] TUNED F1-Score: {tuned_f1:.4f}")
    print(f"[*] Best Parameters Found: {random_search.best_params_}")
    print(f"[*] F1-Score Improvement vs Baseline: {tuned_f1 - base_f1:+.4f}")

    plot_roc_pr_curves_2(metrics_base, metrics_tuned, 'XGBoost Baseline', 'XGBoost Tuned', output_dir, 'Tuning_XGB')
    plot_search_results(random_search.cv_results_, base_f1, 'XGBoost', output_dir)

def tune_mlp(X_train, y_train, X_test, y_test, output_dir):
    print("\n" + "="*50)
    print("=== MLP Neural Network Tuning (Randomized Search) ===")
    print("="*50)

    print("[*] Training Baseline Model (Default)...")
    baseline_model = MLPClassifier(max_iter=500, random_state=42)
    baseline_model.fit(X_train, y_train)
    metrics_base = evaluate_model(baseline_model, X_test, y_test, threshold=0.5)
    base_f1 = metrics_base.get('f1', 0)
    print(f"[*] BASELINE F1-Score: {base_f1:.4f}")
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'activation': ['relu', 'tanh']
    }
    mlp_cv = MLPClassifier(max_iter=500, random_state=42)
    
    random_search = RandomizedSearchCV(
        mlp_cv, param_distributions=param_grid, 
        n_iter=10, scoring='f1', cv=3, verbose=3, random_state=42, n_jobs=-1
    )
    
    print("\n[*] Starting RandomizedSearchCV with 10 combinations x 3 Folds...")
    start_time = time.perf_counter()
    random_search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time
    
    best_model = random_search.best_estimator_
    metrics_tuned = evaluate_model(best_model, X_test, y_test, threshold=0.5)
    tuned_f1 = metrics_tuned.get('f1', 0)
    
    print(f"\n[*] SEARCH COMPLETED IN {elapsed:.1f} seconds.")
    print(f"[*] TUNED F1-Score: {tuned_f1:.4f}")
    print(f"[*] Best Parameters Found: {random_search.best_params_}")
    print(f"[*] F1-Score Improvement vs Baseline: {tuned_f1 - base_f1:+.4f}")

    plot_roc_pr_curves_2(metrics_base, metrics_tuned, 'MLP Baseline', 'MLP Tuned', output_dir, 'Tuning_MLP')
    plot_search_results(random_search.cv_results_, base_f1, 'MLP', output_dir)

def main():
    print("=== NDAL Hyperparameter Tuning Justification Script ===")
    
    output_dir = os.path.join("results", "hyperparam_tuning_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config('config/exp1.yaml')
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    
    tunnel = 'mobile'
    train_dfs = train_dfs_dict.get(tunnel, [])
    test_dfs = test_dfs_dict.get(tunnel, [])
    
    if not train_dfs or not test_dfs:
        print("[!] No data found.")
        return
        
    N, X = config.get('N', 15), config.get('X', 5)
    
    print(f"[*] Extracting data (N={N}, X={X}) for a quick single subset of the dataset...")
    X_train_df, _, y_train = process_dataset_all(train_dfs[:3], N, X)
    X_test_df, _, y_test = process_dataset_all(test_dfs[:1], N, X)
    
    if len(X_train_df) == 0: return
    
    cols_to_drop = ['recent_jitter', 'recent_slope', 'ratio_recent_mean_to_global', 'spikes_over_q95']
    cols_to_drop_actual = [c for c in cols_to_drop if c in X_train_df.columns]
    X_train_df = X_train_df.drop(columns=cols_to_drop_actual)
    X_test_df = X_test_df.drop(columns=cols_to_drop_actual)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    pos_count = (y_train == 1).sum()
    if pos_count > 5:
        target_neg = min(5000, (y_train == 0).sum())
        rus = RandomUnderSampler(sampling_strategy={0: target_neg, 1: pos_count}, random_state=42)
        X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
        
        smote = SMOTE(sampling_strategy={0: target_neg, 1: target_neg}, random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_rus, y_train_rus)
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    print(f"[*] Dataset prepared: {X_train_bal.shape[0]} training samples.")
    
    tune_xgboost(X_train_bal, y_train_bal, X_test_scaled, y_test, output_dir)
    tune_mlp(X_train_bal, y_train_bal, X_test_scaled, y_test, output_dir)
    
    print(f"\n=== SCRIPT COMPLETED. ROC/PR and Variance plots are saved in: {output_dir} ===")

if __name__ == "__main__":
    main()
