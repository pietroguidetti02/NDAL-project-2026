import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Append current path to ensure imports work
sys.path.append(os.getcwd())

from src.data_loader import load_config, load_and_split_data
from main import process_dataset

def plot_for_domain(tunnel, dfs, N=60, X=1):
    print(f"\n=========================================")
    print(f"Processing Domain: {tunnel.upper()}")
    print(f"=========================================")
    
    # Process all dataframes together using the function from main
    X_features, y_labels = process_dataset(dfs, N, X)
    
    if X_features.empty:
        print(f"No data extracted for {tunnel}.")
        return

    print(f"Target distribution:\n{y_labels.value_counts()}")
    
    X_features['Target'] = y_labels.map({0: 'No Loss', 1: 'Loss'})
    
    print("Generating distribution plots and separability report...")
    features_to_plot = [c for c in X_features.columns if c != 'Target']
    
    num_features = len(features_to_plot)
    cols = 4
    rows = (num_features + cols - 1) // cols
    
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import ks_2samp
    
    report_lines = [f"=== Feature Separability Report for {tunnel.upper()} ===\n"]
    report_lines.append(f"{'Feature':<28} | {'AUC':<6} | {'KS':<6} | {'MI':<6} | {'Med(0)':<8} | {'Med(1)':<8} | {'Rec.'}")
    report_lines.append("-" * 90)
    
    # Pre-split data for KS test
    df_0 = X_features[X_features['Target'] == 'No Loss']
    df_1 = X_features[X_features['Target'] == 'Loss']
    
    # Pre-compute Mutual Information for all features at once for efficiency
    X_filled = X_features.drop(columns=['Target']).fillna(0)
    mi_scores = mutual_info_classif(X_filled, y_labels, random_state=42)
    mi_dict = dict(zip(X_filled.columns, mi_scores))
    
    plt.figure(figsize=(24, 5 * rows))
    for i, feature in enumerate(features_to_plot, 1):
        # --- Compute Metrics ---
        # KS Test
        try:
            stat, pval = ks_2samp(df_0[feature].dropna(), df_1[feature].dropna())
        except:
            stat = 0.0
            
        # ROC-AUC
        try:
            auc = roc_auc_score(y_labels, X_features[feature].fillna(0))
            predictive_power = max(auc, 1 - auc)
        except:
            predictive_power = 0.5
            
        mi = mi_dict.get(feature, 0.0)
        
        med_0 = df_0[feature].median()
        med_1 = df_1[feature].median()
            
        # Recommendation Logic based strictly on AUC and KS, corroborated by MI
        if predictive_power > 0.65 or stat > 0.3:
            rec = "✅"
        elif predictive_power > 0.55 or stat > 0.15:
            rec = "⚠️"
        else:
            rec = "❌"
            
        report_lines.append(f"{feature:<28} | {predictive_power:<6.2f} | {stat:<6.2f} | {mi:<6.3f} | {med_0:<8.2f} | {med_1:<8.2f} | {rec}")
        
        # --- Plotting ---
        plt.subplot(rows, cols, i)
        try:
            # We use log_scale=False but add empirical CDFs alongside KDE
            sns.kdeplot(data=X_features, x=feature, hue='Target', common_norm=False, fill=True, alpha=0.5)
        except Exception as e:
            pass
        plt.title(f'{feature}\nAUC:{predictive_power:.2f} | MI:{mi:.3f}')
        
    plt.tight_layout()
    output_path_img = os.path.join(os.getcwd(), f'feature_distributions_{tunnel}.png')
    plt.savefig(output_path_img)
    plt.close()
    
    report_lines.append("\nMetrics explanation:")
    report_lines.append("- AUC: 0.5 is useless. > 0.65 is very good.")
    report_lines.append("- KS: Max distance between CDFs. > 0.3 is very good.")
    report_lines.append("- MI (Mutual Info): Higher is better (0 means independent).")
    report_lines.append("- Med(0) / Med(1): The median value for No-Loss vs Loss. If they are identical, the plot overlaps heavily.\n")
    
    output_path_txt = os.path.join(os.getcwd(), f'feature_report_{tunnel}.txt')
    with open(output_path_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"Plot saved to {output_path_img}")
    print(f"Report saved to {output_path_txt}")

def main():
    print("Loading full dataset from config/exp1.yaml...")
    config = load_config('config/exp1.yaml')
    
    # Load all data splits
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    
    # Combine train and test to have ALL CSVs of a given type
    dfs_dict = {
        'fiber': train_dfs_dict.get('fiber', []) + test_dfs_dict.get('fiber', []),
        'mobile': train_dfs_dict.get('mobile', []) + test_dfs_dict.get('mobile', [])
    }
    
    # Use N=60 and X=1 as per user's latest parameters
    N = config.get('N', 60)
    X = config.get('X', 1)
    
    for tunnel in ['fiber', 'mobile']:
        dfs = dfs_dict.get(tunnel, [])
        if dfs:
            print(f"Found {len(dfs)} total files for {tunnel}.")
            plot_for_domain(tunnel, dfs, N=N, X=X)
        else:
            print(f"No files found for {tunnel}.")

if __name__ == '__main__':
    main()
