import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_for_model(base_dir, model_name, tunnel, output_path):
    # Find all folders corresponding to combinations
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('N_')]
    
    plt.figure(figsize=(22, 16))
    
    # Store all curve data to sort them by AUC later
    curve_data = []
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        csv_path = os.path.join(folder_path, f"{tunnel}_{model_name}_predictions.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            y_true = df['y_true']
            y_prob = df['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Format label for legend, e.g., "N=10, X=1"
            try:
                parts = folder.split('_')
                n_val = parts[1]
                x_val = parts[3]
                label_base = f"N={n_val}, X={x_val}"
            except:
                label_base = f"{folder}"
                
            curve_data.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'label_base': label_base
            })
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    # Sort curves by AUC descending so the best is at the top of the legend
    curve_data.sort(key=lambda x: x['auc'], reverse=True)
    
    # Use a colormap with enough colors for all combinations
    # 'tab20' has 20 distinct colors which is perfect for 19 combinations
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(curve_data))]
    
    for idx, data in enumerate(curve_data):
        label = f"{data['label_base']} (AUC = {data['auc']:.3f})"
        plt.plot(data['fpr'], data['tpr'], color=colors[idx % 20], lw=2, label=label)
            
    # Plot random guessing baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title(f'ROC Curve for {model_name} ({tunnel})', fontsize=24)
    
    # Place legend outside the plot or adjust it so it doesn't overlap the curves
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=30)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves for all combinations in a sweep directory")
    parser.add_argument("--dir", type=str, required=True, help="Path to the sweep results directory")
    parser.add_argument("--tunnel", type=str, default="mobile", help="Tunnel type (e.g., mobile, fiber)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} does not exist.")
        return
        
    models = ["LSTM", "NN", "XGBoost"]
    
    for model in models:
        output_file = os.path.join(args.dir, f"{args.tunnel}_all_combinations_{model}_roc.png")
        plot_roc_for_model(args.dir, model, args.tunnel, output_file)

if __name__ == "__main__":
    main()
