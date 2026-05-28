import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def get_latest_exp_dir(base_dir="results"):
    # Find the latest exp_4comp directory
    dirs = [d for d in glob.glob(os.path.join(base_dir, "exp_4comp_*")) if os.path.isdir(d)]
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

def plot_4way_comparison(exp_dir):
    models = ['XGBoost_Stats', 'XGBoost_Raw', 'NN_Stats', 'NN_Raw']
    metrics_to_plot = ['Precision', 'Recall', 'F1 Score']
    
    # We will only plot for Mobile since Fiber has all zeros.
    domain = "mobile"
    
    data = {m: [] for m in metrics_to_plot}
    
    for model in models:
        json_path = os.path.join(exp_dir, f"{domain}_{model}_metrics.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                data['Precision'].append(metrics.get('precision', 0) * 100)
                data['Recall'].append(metrics.get('recall', 0) * 100)
                data['F1 Score'].append(metrics.get('f1', 0) * 100)
        else:
            for m in metrics_to_plot:
                data[m].append(0)
                
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, data['Precision'], width, label='Precision', color='#3498db')
    rects2 = ax.bar(x, data['Recall'], width, label='Recall', color='#2ecc71')
    rects3 = ax.bar(x + width, data['F1 Score'], width, label='F1 Score', color='#e74c3c')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'4-Way Feature Representation Comparison ({domain.upper()})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=11)
    ax.legend(fontsize=11)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    output_path = os.path.join(exp_dir, f"{domain}_4way_comparison_barplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Barplot saved successfully at: {output_path}")

if __name__ == '__main__':
    latest_dir = get_latest_exp_dir()
    if latest_dir:
        print(f"Generating comparative barplot using data from: {latest_dir}")
        plot_4way_comparison(latest_dir)
    else:
        print("No 4-way comparison experiment directory found.")
