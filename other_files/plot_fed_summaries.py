import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import auc

def plot_fed_performance_summary(results_dir, model_name, output_dir):
    summary_path = os.path.join(results_dir, 'federated_sweeping_summary.csv')
    if not os.path.exists(summary_path):
        return
        
    df = pd.read_csv(summary_path)
    df = df[df['Model'] == model_name]
    if df.empty:
        return

    cpe_a = df[df['Type'] == 'Local_CPE_A'][['N', 'Type', 'F1_Score']]
    cpe_b = df[df['Type'] == 'Local_CPE_B'][['N', 'Type', 'F1_Score']]
    cpe_c = df[df['Type'] == 'Local_CPE_C'][['N', 'Type', 'F1_Score']]
    
    fed_df = df[df['Type'] == 'Federated'][['N', 'Type', 'F1_Score']]
    centr_df = df[df['Type'] == 'Centralized'][['N', 'Type', 'F1_Score']]
    
    plot_df = pd.concat([cpe_a, cpe_b, cpe_c, fed_df, centr_df])
    plot_df['Type'] = plot_df['Type'].replace({'Local_CPE_A': 'Local A', 'Local_CPE_B': 'Local B', 'Local_CPE_C': 'Local C'})
    
    pivot_df = plot_df.pivot(index='N', columns='Type', values='F1_Score')
    cols = ['Local A', 'Local B', 'Local C', 'Federated', 'Centralized']
    cols = [c for c in cols if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728']
    
    ax = pivot_df.plot(kind='bar', figsize=(12, 6), edgecolor='black', color=colors[:len(cols)], zorder=3)
                       
    plt.title(f'Performance Comparison (F1-Score) - {model_name}', fontsize=16)
    plt.xlabel('Lookback Window (N seconds)', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim([0.0, 1.1])
    plt.legend(title='Training Type', fontsize=12, loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.xticks(rotation=0)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'summary_performance_{model_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

def plot_compact_times(results_dir, model_name, output_dir):
    n_values = [15, 30, 60]
    data = []
    
    for n in n_values:
        file_path = os.path.join(results_dir, f'FL_{model_name}_N{n}_timing_records.csv')
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        clients = [c for c in df.columns if c not in ['Round', 'Network']]
        
        df['Max_Compute'] = df[clients].max(axis=1)
        round_totals = df['Max_Compute'] + df['Network']
        min_total = round_totals.min()
        max_total = round_totals.max()
        avg_total = round_totals.mean()
        
        avg_network = df['Network'].mean()
        
        for c in clients:
            avg_compute = df[c].mean()
            avg_idle = (df['Max_Compute'] - df[c]).mean()
            
            data.append({
                'N': n,
                'Client': c,
                'Avg_Compute': avg_compute,
                'Avg_Idle': avg_idle,
                'Avg_Network': avg_network,
                'Min_Total': min_total,
                'Max_Total': max_total,
                'Avg_Total': avg_total
            })
            
    if not data:
        return
        
    df_plot = pd.DataFrame(data)
    clients = df_plot['Client'].unique()
    valid_n = df_plot['N'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(valid_n))
    width = 0.8 / len(clients)
    
    colors_compute = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color_idle = 'lightgray'
    color_network = '#d62728'
    
    legend_handles = []
    max_y = df_plot['Max_Total'].max()
    
    for i, client in enumerate(clients):
        client_data = df_plot[df_plot['Client'] == client].sort_values('N')
        pos = x - 0.4 + (i + 0.5) * width
        
        computes = client_data['Avg_Compute'].values
        idles = client_data['Avg_Idle'].values
        networks = client_data['Avg_Network'].values
        
        yerr = [
            client_data['Avg_Total'].values - client_data['Min_Total'].values,
            client_data['Max_Total'].values - client_data['Avg_Total'].values
        ]
        
        ax.bar(pos, computes, width, color=colors_compute[i % len(colors_compute)], edgecolor='black')
        legend_handles.append(mpatches.Patch(facecolor=colors_compute[i % len(colors_compute)], edgecolor='black', label=f'{client.replace("_", " ")} Compute'))
        
        ax.bar(pos, idles, width, bottom=computes, color=color_idle, hatch='////', edgecolor='black')
        if i == 0:
            legend_handles.append(mpatches.Patch(facecolor=color_idle, hatch='////', edgecolor='black', label='Idle Time (Wait)'))
        
        # Aggiungiamo le error bars al top (Network delay)
        ax.bar(pos, networks, width, bottom=computes+idles, color=color_network, alpha=0.8, edgecolor='black', yerr=yerr, capsize=4)
        if i == 0:
            legend_handles.append(mpatches.Patch(facecolor=color_network, alpha=0.8, edgecolor='black', label='Network Delay'))
            
        # Aggiungiamo il testo del tempo totale sopra la barra centrale (i==1)
        if i == 1:
            for j, val in enumerate(client_data['Avg_Total'].values):
                m_tot = client_data['Max_Total'].values[j]
                ax.text(pos[j], m_tot + 0.05 * max_y, f"{val:.2f}s", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {n}' for n in valid_n], fontsize=12)
    ax.set_ylabel('Time (Seconds)', fontsize=12)
    ax.set_title(f'Compact Federated Training Times (Avg + Min/Max) - {model_name}', fontsize=16)
    
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    
    plt.ylim(0, max_y * 1.25)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'compact_times_{model_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

def plot_best_worst_roc(results_dir, model_name, n_val, output_dir):
    file_path = os.path.join(results_dir, f'FL_{model_name}_N{n_val}_roc_pr_data.csv')
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    df_roc = df[df['Curve'] == 'ROC']
    
    scenarios = df_roc['Scenario'].unique()
    local_scenarios = [s for s in scenarios if s.startswith('Local')]
    
    aucs = {}
    curves = {}
    for s in scenarios:
        s_data = df_roc[df_roc['Scenario'] == s]
        x = s_data['X'].values
        y = s_data['Y'].values
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        curves[s] = (x, y)
        aucs[s] = auc(x, y)
        
    if not local_scenarios:
        return
        
    best_local = max(local_scenarios, key=lambda s: aucs[s])
    worst_local = min(local_scenarios, key=lambda s: aucs[s])
    
    plt.figure(figsize=(9, 8))
    plt.plot(curves[worst_local][0], curves[worst_local][1], linestyle='-.', color='#ff7f0e', lw=2, 
             label=f'Worst Local ({worst_local.replace("Local ", "")}) - AUC: {aucs[worst_local]:.3f}')
    plt.plot(curves[best_local][0], curves[best_local][1], linestyle='--', color='#1f77b4', lw=2, 
             label=f'Best Local ({best_local.replace("Local ", "")}) - AUC: {aucs[best_local]:.3f}')
             
    if 'Federated' in curves:
        plt.plot(curves['Federated'][0], curves['Federated'][1], linestyle='-', color='#2ca02c', lw=3, 
                 label=f'Federated - AUC: {aucs["Federated"]:.3f}')
    if 'Centralized' in curves:
        plt.plot(curves['Centralized'][0], curves['Centralized'][1], linestyle=':', color='#d62728', lw=2, 
                 label=f'Centralized - AUC: {aucs["Centralized"]:.3f}')
                 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.5)
    plt.title(f'ROC Curves Comparison - {model_name} (N={n_val})', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f'best_worst_roc_{model_name}_N{n_val}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot all Federated Learning summaries")
    parser.add_argument("--dir", type=str, default="results/exp_federated_20260612_151021", help="Path to the federated results directory")
    args = parser.parse_args()
    
    # Crea cartella images
    images_dir = os.path.join(args.dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print("Generating Performance Summaries (F1-Score)...")
    plot_fed_performance_summary(args.dir, 'MLP', images_dir)
    plot_fed_performance_summary(args.dir, 'LSTM', images_dir)
    
    print("\nGenerating Compact Time Breakdowns (Compute vs Idle)...")
    plot_compact_times(args.dir, 'MLP', images_dir)
    plot_compact_times(args.dir, 'LSTM', images_dir)
    
    print("\nGenerating Best vs Worst ROC Curves...")
    for n in [15, 30, 60]:
        plot_best_worst_roc(args.dir, 'MLP', n, images_dir)
        plot_best_worst_roc(args.dir, 'LSTM', n, images_dir)
        
    print(f"\nAll federated plots generated successfully inside {images_dir}!")

if __name__ == '__main__':
    main()
