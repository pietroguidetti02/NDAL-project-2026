import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np

def get_best_metrics(base_dir, model_name, tunnel):
    """
    Finds the maximum ROC AUC and maximum PR AUC for a given model across all combinations in a sweep directory.
    """
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('N_')]
    
    best_roc_auc = 0.0
    best_pr_auc = 0.0
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        csv_path = os.path.join(folder_path, f"{tunnel}_{model_name}_predictions.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            y_true = df['y_true']
            y_prob = df['y_prob']
            
            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                
            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    return best_roc_auc, best_pr_auc

def main():
    parser = argparse.ArgumentParser(description="Compare best ROC AUC and PR AUC across S1, S2, S3 sweeps")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to the results directory containing S1, S2, S3 folders")
    parser.add_argument("--tunnel", type=str, default="mobile", help="Tunnel type (e.g., mobile, fiber)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Directory {args.results_dir} does not exist.")
        return
        
    # Dynamically find S1, S2, S3 folders in the results directory
    s1_dir = next((f for f in os.listdir(args.results_dir) if f.startswith('S1')), None)
    s2_dir = next((f for f in os.listdir(args.results_dir) if f.startswith('S2')), None)
    s3_dir = next((f for f in os.listdir(args.results_dir) if f.startswith('S3')), None)
    
    experiments = [
        ('S1 (Single)', os.path.join(args.results_dir, s1_dir) if s1_dir else None),
        ('S2 (Spatial)', os.path.join(args.results_dir, s2_dir) if s2_dir else None),
        ('S3 (Time)', os.path.join(args.results_dir, s3_dir) if s3_dir else None)
    ]
    
    models = ["XGBoost", "NN", "LSTM"]
    colors = ['c', 'y', 'm'] # Colors matching utils.py (XGBoost=c, NN=y, LSTM=m)
    hatches = ['///', '---', '\\\\\\']
    
    results_roc = {model: [] for model in models}
    results_pr = {model: [] for model in models}
    valid_experiments = []
    
    for exp_name, exp_path in experiments:
        if not exp_path or not os.path.exists(exp_path):
            print(f"Warning: Directory for {exp_name} not found.")
            continue
            
        valid_experiments.append(exp_name)
        for model in models:
            best_roc, best_pr = get_best_metrics(exp_path, model, args.tunnel)
            results_roc[model].append(best_roc)
            results_pr[model].append(best_pr)
            print(f"{exp_name} - {model}: Best ROC AUC = {best_roc:.4f}, Best PR AUC = {best_pr:.4f}")
            
    if not valid_experiments:
        print("No valid experiments found.")
        return

    # --- Plotting ROC AUC ---
    x = np.arange(len(valid_experiments))
    w = 0.25
    
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        bars = plt.bar(x + (i - 1) * w, results_roc[model], width=w, edgecolor='black', 
                color=colors[i], align='center', hatch=hatches[i], label=model)
        plt.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    
    plt.xticks(x, valid_experiments, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Best Model Performance Comparison (ROC AUC) across Sweeps S1,S2,S3', fontsize=16)
    plt.ylabel('Max ROC AUC', fontsize=14)
    plt.ylim([0.0, 1.1])
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    output_file_roc = os.path.join(args.results_dir, f"{args.tunnel}_comparison_S123_roc_auc.png")
    plt.tight_layout()
    plt.savefig(output_file_roc, dpi=300, bbox_inches="tight")
    plt.close()
    
    # --- Plotting PR AUC ---
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        bars = plt.bar(x + (i - 1) * w, results_pr[model], width=w, edgecolor='black', 
                color=colors[i], align='center', hatch=hatches[i], label=model)
        plt.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    
    plt.xticks(x, valid_experiments, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Best Model Performance Comparison (PR AUC) across Sweeps S1,S2,S3', fontsize=16)
    plt.ylabel('Max PR AUC', fontsize=14)
    plt.ylim([0.0, 1.1])
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    output_file_pr = os.path.join(args.results_dir, f"{args.tunnel}_comparison_S123_pr_auc.png")
    plt.tight_layout()
    plt.savefig(output_file_pr, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved ROC comparison plot to {output_file_roc}")
    print(f"Saved PR comparison plot to {output_file_pr}")

if __name__ == "__main__":
    main()

